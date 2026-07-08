"""Media session continuation ranking: score 12 candidates per query so the item the user
actually continued with (relevance=3 next item, relevance=1 later follow-ups) ranks above
distractors.

Reads dataset/public/{train,test,sample_submission}.csv and writes one response_score per
test row to ./working/submission.csv. Scored by mean NDCG@3 per query_id.

Train and test users are disjoint (no shared user_id), and the task explicitly warns that
distractors are drawn to have comparable-or-higher item "volume" (popularity) than the true
continuation -- so ranking by global popularity alone is deliberately insufficient; the model
has to learn transferable continuation patterns instead.

Two EDA findings drive the feature design (see readme.txt for the full writeup):
  - Rewatch signal: a candidate already sitting in the query's own 6-item history is *never*
    a distractor (relevance=0 rate is exactly 0%), but is fairly often the true next item or a
    later follow-up (relevance=1/3 rates ~19%/29%). A clean, leak-free binary feature straight
    from history_1..6.
  - Cross-query "future match" signal (the dominant feature by a wide margin): queries are
    sliding windows over each user's real interaction stream -- most users (~75-83%, train and
    test alike) have more than one query, and consecutive queries for the same user show the
    6-item history window shifting forward in time. So checking whether a candidate appears in
    *another* query's history for the *same user*, restricted to that other query's
    anchor_week being >= the current query's week ("future" match), is close to a ground-truth
    signal (measured on train: relevance=0 rate 4.8% vs. relevance=1/3 rates ~60%/61%). This
    only uses the provided history_*/candidate_id sequence columns already in train.csv/
    test.csv (no labels, no external data, no reverse-mapping) -- it's the intended
    sequence-modeling signal, just applied across a user's multiple queries rather than within
    a single one.

On top of these two, a label-free item-item association signal (PMI + an SVD-embedding cosine
similarity over a small item-item co-occurrence matrix built from every history-slot ->
candidate pair across train+test) and plain popularity features round out the feature set, so
queries whose user has no other query to cross-reference still get a reasonable ranking.

A LightGBM lambdarank model is trained on the combined feature set, validated with GroupKFold
split on user_id (not query_id) to mirror the real disjoint-user gap, then refit on 100% of
train. Predictions are seed-bagged (averaged across SEED_BAG_SEEDS independently-seeded
models): a real graded submission scored notably below local CV, and stress-testing the CV
protocol with repeated GroupKFold showed the ~0.80 estimate is stable across many different
train-user partitions -- so the gap looks like imperfect transfer to the fully disjoint real
test users rather than ordinary CV noise. Of several remedies tried (monotonic constraints,
heavier regularization, feature pruning -- see readme.txt Results), only seed-bagging gave a
consistent win (higher mean, lower variance, better worst-case fold) under that same repeated
protocol, so it's the one adopted. Falls back to a pairwise-ranking scikit-learn classifier
(HistGradientBoostingClassifier trained on within-query pairwise feature differences, scored
by round-robin win-rate at inference) if LightGBM is unavailable, so a valid submission is
always produced either way.

Later rounds added exact sliding-window shift reconstruction, random-walk-with-restart (RWR)
transition features, a leak-safe exact_head stacked feature, and a temporal popularity trend
feature (all validated to help CV and, where tested, the real score -- see readme.txt for the
full round-by-round account, including several reranking/blending formulations that were tried
and rejected). The one change confirmed to help the *real* score, not just CV, was blending two
differently-configured model versions; that principle is now built directly into this script as
an internal ensemble of a "full" model (FEATURE_COLS + exact_head) and a deliberately simpler
"core" model (CORE_FEATURE_COLS), blended via per-query z-score at ENSEMBLE_WEIGHT_FULL, rather
than living as an external, manually-run blend script. The trend feature (comparing an item's
recent-window popularity against its all-time baseline) was the largest single CV lift found
since the original shift reconstruction, and the most robust: it beat the no-trend baseline in
100% of a 40-fold repeated-CV check. A later parallel search for a second independent signal
axis (mirroring the process that found trend) added seq_recency_skew, a population-level
positional signature (does an item typically sit in "just watched" vs. "aging out of the
window" slots) -- validated non-redundant (max correlation 0.25 with existing features) and a
consistent, if smaller, CV lift (19/20 repeated-CV folds improved). Three other candidate ideas
from that same search (a pair-level temporal-transition feature, user-level aggregate stats, a
session-depth/remaining-length signal) were tried and rejected -- see readme.txt for why.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
HIST_COLS = [f"history_{i}" for i in range(1, 7)]
SLOT_WEIGHT = np.array([1.0, 0.85, 0.72, 0.61, 0.52, 0.44])  # recency decay history_1..6
SVD_DIM = 24
SHIFT_MAX_K = 5
RWR_ALPHA = 0.15
N_WEEKS = 53
TREND_WINDOW = 4
# Continuous features whose within-query (group-relative) z-score/percentile rank is added
# as extra features -- raw values don't compare across queries with different item pools and
# different overall association/popularity scale, but relative-within-pool values do.
GROUP_RELATIVE_COLS = ["pmi_weighted_sum", "cnt_weighted_sum", "svd_cos_sim", "cand_volume",
                        "shift_overlap_len", "shift_vote_score", "rwr_weighted_sum", "trend",
                        "seq_recency_skew"]
# Chosen via a parallel ablation panel (LightGBM/XGBoost/CatBoost, several feature subsets,
# an ensemble, and a hand-tuned heuristic baseline -- see readme.txt Results): LightGBM
# lambdarank with num_leaves=31 was the single best config (mean CV NDCG@3 0.8029, vs.
# 0.7919 XGBoost, 0.7959 CatBoost, 0.8010 for a 3-model ensemble -- ensembling gave no real
# lift over a well-tuned single LightGBM, so it was rejected in favor of simplicity).
# NUM_BOOST_ROUND is the average best_iteration observed across CV folds with early stopping.
NUM_BOOST_ROUND = 180
# Seed-bagging: average predictions from SEED_BAG_SEEDS independently-seeded models trained
# on identical data/features. Under repeated GroupKFold-by-user_id (6 repeats x 5 folds, i.e.
# 30 different random partitions of the train users), this raised mean NDCG@3 from 0.7989 to
# 0.8028, *lowered* fold-to-fold std (0.0065 -> 0.0052), and raised the worst-fold score
# (0.7850 -> 0.7903) -- a consistent win on mean, variance, and worst case alike, unlike
# heavier regularization or monotonic constraints, which were also tried and made things worse
# (see readme.txt Results). This matters most for a single real held-out evaluation, where
# lower prediction variance directly reduces the risk of an unlucky single-model draw.
SEED_BAG_SEEDS = list(range(8))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "public")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_DIR = os.path.join(BASE_DIR, "working")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

LGB_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[3],
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    lambda_l2=1.0,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    seed=SEED,
    bagging_seed=SEED,
    feature_fraction_seed=SEED,
    data_random_seed=SEED,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)

FEATURE_COLS = [
    "in_own_hist", "own_hist_pos", "own_hist_ndistinct",
    "has_future_match", "n_future_match", "min_future_pos", "min_future_weekdiff",
    "has_sameweek_match", "n_sameweek_match", "min_sameweek_pos",
    "has_past_match", "n_past_match", "min_past_pos",
    "n_user_queries", "anchor_week",
    "is_shift_exact_next", "is_shift_later_followup", "unique_shift_exact",
    "shift_distance", "shift_overlap_len", "shift_vote_score",
    "pmi_slot1", "pmi_slot2", "pmi_slot3", "pmi_slot4", "pmi_slot5", "pmi_slot6",
    "pmi_weighted_sum", "cnt_weighted_sum", "svd_cos_sim",
    "rwr_slot1", "rwr_slot2", "rwr_slot3", "rwr_slot4", "rwr_slot5", "rwr_slot6",
    "rwr_weighted_sum", "rwr_max",
    "cand_volume", "cand_volume_log", "cand_vol_rank", "trend", "seq_recency_skew",
] + [f"{c}_gz" for c in GROUP_RELATIVE_COLS] + [f"{c}_gpct" for c in GROUP_RELATIVE_COLS]

# Leak-sensitive stacked feature (see add_exact_head_feature): appended to FEATURE_COLS only
# at model-training time, never precomputed globally like the rest of FEATURE_COLS above.
EXACT_HEAD_COL = "exact_head"

# CORE_FEATURE_COLS: a simpler, more conservative subset of FEATURE_COLS (drops
# unique_shift_exact, shift_overlap_len/shift_vote_score and their group-relative variants,
# and all RWR features), trained as a second model alongside the full one and blended in.
# The point isn't that this subset is individually better -- alone it scores clearly worse
# (0.8657 vs. 0.8709 CV) -- it's that it's different enough from the full model (Spearman
# correlation 0.77 between their predictions) to add real ensemble diversity: blending the two
# (roughly 0.8 full / 0.2 core, validated via CV) scored 0.8728, beating the full model alone.
# This mirrors the one confirmed *real-world* win in this project so far -- a z-score blend of
# two differently-configured models (round 2 + round 4) scored 0.5923, the best real score to
# date, measurably above either alone -- so it's built into the shipped model directly instead
# of staying an external, manually-run blend script.
CORE_FEATURE_COLS = [
    "in_own_hist", "own_hist_pos", "own_hist_ndistinct",
    "has_future_match", "n_future_match", "min_future_pos", "min_future_weekdiff",
    "has_sameweek_match", "n_sameweek_match", "min_sameweek_pos",
    "has_past_match", "n_past_match", "min_past_pos",
    "n_user_queries", "anchor_week",
    "is_shift_exact_next", "is_shift_later_followup", "shift_distance",
    "pmi_slot1", "pmi_slot2", "pmi_slot3", "pmi_slot4", "pmi_slot5", "pmi_slot6",
    "pmi_weighted_sum", "cnt_weighted_sum", "svd_cos_sim",
    "cand_volume", "cand_volume_log", "cand_vol_rank", "trend", "seq_recency_skew",
    "pmi_weighted_sum_gz", "cnt_weighted_sum_gz", "svd_cos_sim_gz", "cand_volume_gz", "trend_gz",
    "seq_recency_skew_gz",
    "pmi_weighted_sum_gpct", "cnt_weighted_sum_gpct", "svd_cos_sim_gpct", "cand_volume_gpct",
    "trend_gpct", "seq_recency_skew_gpct",
]
ENSEMBLE_WEIGHT_FULL = 0.75


# ---------------------------------------------------------------------------
# Item-item association (unsupervised, label-free): co-occurrence -> PMI -> SVD embedding.
# Built once from train+test combined -- this only uses history_*/candidate_id columns that
# are present (unlabeled) in both files, so it is not a labeling leak.
# ---------------------------------------------------------------------------
def _normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def build_item_association(train_df, test_df):
    all_items = sorted(set(pd.concat(
        [train_df[c] for c in HIST_COLS + ["candidate_id"]]
        + [test_df[c] for c in HIST_COLS + ["candidate_id"]]
    )))
    item2idx = {it: i for i, it in enumerate(all_items)}
    n_items = len(all_items)

    co_raw = np.zeros((n_items, n_items), dtype=np.float64)
    total_vol = np.zeros(n_items, dtype=np.float64)
    for df in (train_df, test_df):
        cand_idx = df["candidate_id"].map(item2idx).to_numpy()
        np.add.at(total_vol, cand_idx, 1.0)
        for col in HIST_COLS:
            hidx = df[col].map(item2idx).to_numpy()
            np.add.at(co_raw, (hidx, cand_idx), 1.0)
            np.add.at(total_vol, hidx, 1.0)

    n_pairs = co_raw.sum()
    row_sum = co_raw.sum(axis=1, keepdims=True)
    col_sum = co_raw.sum(axis=0, keepdims=True)
    eps = 1.0
    # PMI normalizes away raw popularity (log co-occurrence relative to what independence
    # would predict), which matters here since raw co-occurrence counts are dominated by a
    # handful of extremely popular items and would otherwise just re-derive popularity.
    pmi = np.log((co_raw + eps) * n_pairs / ((row_sum + eps) * (col_sum + eps)))

    # The catalog is tiny (258 items here), so a full numpy SVD is cheap and avoids
    # sklearn's randomized TruncatedSVD solver, which hits spurious divide-by-zero/overflow
    # warnings on this particular matrix.
    u, s, vt = np.linalg.svd(pmi, full_matrices=False)
    item_emb = _normalize_rows(u[:, :SVD_DIM] * s[:SVD_DIM])       # "as history context"
    item_emb_cand = _normalize_rows(vt[:SVD_DIM, :].T)             # "as candidate"

    # Random-walk-with-restart (personalized PageRank) over the item transition graph: a soft,
    # multi-step complement to PMI (single-step) and the exact shift reconstruction (requires
    # an exact structural sequence match). T[i,j] = P(next=j | current=i), row-normalized from
    # co_raw; RWR = alpha*(I - (1-alpha)*T)^-1 gives the stationary restart-distribution reached
    # by repeatedly following transitions with probability (1-alpha) and restarting with
    # probability alpha -- closed-form since the catalog is tiny (258x258, cheap to invert).
    # Validated: rwr_weighted_sum correlates monotonically with relevance (0.069/0.125/0.163 for
    # relevance 0/1/3) and adding it as a feature measurably improved CV (see readme.txt).
    row_sum_pos = co_raw.sum(axis=1, keepdims=True)
    row_sum_pos[row_sum_pos == 0] = 1.0
    trans = co_raw / row_sum_pos
    rwr = RWR_ALPHA * np.linalg.inv(np.eye(n_items) - (1 - RWR_ALPHA) * trans)

    # Temporal popularity trend: every other feature so far is an all-time aggregate, blind to
    # whether an item is currently "hot" vs. cooling off. weekly_count[w, i] = how often item i
    # appears (as candidate or in any history slot) in anchor_week w; trend[w, i] compares the
    # item's volume in a +/-TREND_WINDOW-week window around w against what a uniform spread of
    # its all-time volume across all N_WEEKS would predict (log1p ratio, so 0 = "on trend",
    # positive = trending up locally, negative = quiet locally relative to its own baseline).
    # Validated before adopting: near-zero correlation with cand_volume (0.08) and cand_vol_rank
    # (-0.03) -- genuinely new information, not a rehash of static popularity -- and a strong,
    # clean monotonic relationship with relevance (-0.033/0.204/0.395 for relevance 0/1/3 at
    # this window). Adding it lifted CV NDCG@3 by +0.0099, beating the baseline in all 40/40
    # folds of a repeated-CV check -- the most robust single improvement found in this project.
    weekly_count = np.zeros((N_WEEKS, n_items), dtype=np.float64)
    for df in (train_df, test_df):
        wk = df["anchor_week"].to_numpy()
        for col in HIST_COLS + ["candidate_id"]:
            idx = df[col].map(item2idx).to_numpy()
            np.add.at(weekly_count, (wk, idx), 1.0)
    windowed = np.zeros((N_WEEKS, n_items), dtype=np.float64)
    window_span = np.zeros(N_WEEKS, dtype=np.float64)
    for w in range(N_WEEKS):
        lo, hi = max(0, w - TREND_WINDOW), min(N_WEEKS - 1, w + TREND_WINDOW)
        windowed[w] = weekly_count[lo:hi + 1].sum(axis=0)
        window_span[w] = hi - lo + 1
    expected = total_vol[None, :] * (window_span[:, None] / N_WEEKS)
    trend = np.log1p(windowed) - np.log1p(expected)

    # Item sequence role: a population-level positional signature, distinct from popularity,
    # temporal trend, or pairwise association -- does this item typically sit in history_1
    # ("just watched") whenever it appears in a window, or history_6 ("about to age out")?
    # hist_slot_counts[i, k] = how often item i appears in history_(k+1) across train+test;
    # normalized to a per-item distribution over the 6 slots, then reduced to a single
    # recency-skew score via weights [6,5,4,3,2,1] (history_1..6) -- high = skews recent,
    # low = skews old. Items with zero history-slot appearances (candidate-only items) get
    # the neutral midpoint 3.5. Validated before adopting: passes the redundancy check cleanly
    # (max correlation 0.25, with RWR -- conceptually related since both derive from the same
    # co-occurrence structure, but far under the ~0.3 flag), and beat the no-feature baseline
    # in 19/20 folds of a repeated-CV check (+0.0039 average) -- weaker and noisier than trend's
    # signature but a genuinely new, non-redundant axis nonetheless.
    hist_slot_counts = np.zeros((n_items, 6), dtype=np.float64)
    for df in (train_df, test_df):
        for k, col in enumerate(HIST_COLS):
            idx = df[col].map(item2idx).to_numpy()
            np.add.at(hist_slot_counts[:, k], idx, 1.0)
    slot_row_sum = hist_slot_counts.sum(axis=1)
    seq_recency_skew = np.full(n_items, 3.5, dtype=np.float64)
    has_hist = slot_row_sum > 0
    slot_weight = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    norm_dist = hist_slot_counts[has_hist] / slot_row_sum[has_hist, None]
    seq_recency_skew[has_hist] = (norm_dist * slot_weight[None, :]).sum(axis=1)

    return dict(item2idx=item2idx, pmi=pmi, co_raw=co_raw, rwr=rwr, trend=trend,
                seq_recency_skew=seq_recency_skew,
                item_emb=item_emb, item_emb_cand=item_emb_cand, total_vol=total_vol)


# ---------------------------------------------------------------------------
# Cross-query structure: queries are sliding windows over each user's real interaction
# stream. Index each user's queries as (query_id, anchor_week, history-list) so every row
# can check whether its candidate shows up in another query's history for the same user.
# ---------------------------------------------------------------------------
def build_user_index(df):
    qtab = df.drop_duplicates("query_id")[["query_id", "user_id", "anchor_week"] + HIST_COLS]
    index = defaultdict(list)
    for qid, uid, wk, *hist in qtab.itertuples(index=False, name=None):
        index[uid].append((qid, wk, list(hist)))
    return index


# ---------------------------------------------------------------------------
# Exact sliding-window shift reconstruction: a much higher-precision alternative to the fuzzy
# "candidate appears somewhere in another query's history" check above. If another query B's
# history is this query A's history shifted forward by k newly-consumed items (1<=k<=
# SHIFT_MAX_K), i.e. hist_B[k:6] == hist_A[0:6-k], then those k new items are recoverable in
# exact chronological order: the earliest one (closest to A's snapshot) is precisely the true
# "next" item, and the rest are "later followups", in order.
#
# Refinements over a first version that only kept the single largest-overlap match:
#  - wk_b < wk_a matches (B chronologically before A despite the shift pattern implying B comes
#    after) are down-weighted rather than excluded: precision on just those contradictory cases
#    was 11.0% (exact_next) / 20.3% (followup) on train, vs. 87.9%/69.7% for wk_b >= wk_a --
#    real signal, just much weaker, so BACKWARD_MATCH_WEIGHT scales their contribution down
#    instead of discarding it outright (a hard exclude was tried and tested no better in
#    practice -- see readme.txt Results).
#  - Aggregate evidence across *all* qualifying other queries B for a given item, not just the
#    single best-overlap one: shift_vote_score (confidence-weighted corroboration -- sum of
#    weighted overlap_len across every B that identifies this item) and min_shift_distance (the
#    smallest implied distance across all of them) reward candidates multiple queries agree on.
#  - unique_shift_exact (in engineer_features below): different other-queries can occasionally
#    identify *different* items as the distance-1 "exact next" (conflicting evidence, e.g. from
#    a coincidental backward match). is_shift_exact_next fires for every item tied for the
#    minimum distance, so a separate flag marks whether this query's exact-next claim is
#    contested by another candidate or singular.
#  - 2-hop chaining: if A's window shifts to B by k1 items, and B's window separately shifts to
#    C by k2 items, then C's newly-revealed items are (k1+j) positions after A even when A and
#    C don't overlap directly at all (window fully rolled over, k1+k2 > SHIFT_MAX_K) -- chaining
#    through the bridging query B recovers them. Validated directly against train labels before
#    adopting: chaining surfaces new candidate identifications on 28% of queries, but only 16%
#    of those land on one of the query's actual 12 candidates, and precision there is 44.3% for
#    relevance>=1 (vs. 69.7%/77.3% for direct followups) and ~1% for relevance==3 (expected --
#    a chained item is by construction at distance >= 2, so it can only ever be a "later
#    followup" candidate, never the exact-next one). Weaker than direct evidence but a real,
#    non-trivial signal (roughly 2x the ~23% unconditional relevance>=1 base rate), so it's
#    folded into the same evidence pool with reduced weight (CHAIN_DECAY) rather than kept
#    separate.
# ---------------------------------------------------------------------------
# A sweep of BACKWARD_MATCH_WEIGHT in {0, 0.1, 0.2, 0.35, 0.5, 1.0} showed no consistent CV
# preference (all within 0.8673-0.8689, well inside fold-to-fold noise) -- consistent with this
# project's broader finding that changes at this scale aren't reliably distinguishable from
# noise (see readme.txt Results). 0.3 is kept as a principled middle value (roughly matching
# the ~11-20% vs. 88-93% precision ratio observed between backward and forward matches) rather
# than cherry-picking whichever value happened to score highest on a small, noisy CV sweep.
BACKWARD_MATCH_WEIGHT = 0.3
CHAIN_DECAY = 0.5


def _one_hop_matches(qlist, max_k=SHIFT_MAX_K):
    """qid_a -> list of (qid_b, wk_b, k, new_items, weight) for every valid 1-hop shift from
    qid_a to another of the same user's queries (both directions; backward down-weighted)."""
    pairwise = defaultdict(list)
    for qid_a, wk_a, hist_a in qlist:
        for qid_b, wk_b, hist_b in qlist:
            if qid_b == qid_a:
                continue
            weight = 1.0 if wk_b >= wk_a else BACKWARD_MATCH_WEIGHT
            for k in range(1, max_k + 1):
                if hist_b[k:6] == hist_a[0:6 - k]:
                    new_items = list(reversed(hist_b[0:k]))  # [x_1 next, ..., x_k]
                    pairwise[qid_a].append((qid_b, wk_b, k, new_items, weight))
                    break
    return pairwise


def build_shift_info(user_index, max_k=SHIFT_MAX_K):
    info = {}
    for uid, qlist in user_index.items():
        pairwise = _one_hop_matches(qlist, max_k)
        for qid_a, wk_a, hist_a in qlist:
            evidence = defaultdict(list)  # item -> [(distance, weighted_overlap), ...]
            one_hop = pairwise.get(qid_a, [])
            direct_items = set()
            for (qid_b, wk_b, k1, new_items, w1) in one_hop:
                overlap_len = (6 - k1) * w1
                for dist, item in enumerate(new_items, start=1):
                    evidence[item].append((dist, overlap_len))
                    direct_items.add(item)

            # 2-hop chain, forward direction only (chaining a backward hop would compound an
            # already-weak, direction-ambiguous signal on top of another uncertainty).
            for (qid_b, wk_b, k1, _new_items_ab, w1) in one_hop:
                if wk_b < wk_a:
                    continue
                for (qid_c, wk_c, k2, new_items_bc, w2) in pairwise.get(qid_b, []):
                    if qid_c == qid_a or wk_c < wk_b:
                        continue
                    for j, item in enumerate(new_items_bc, start=1):
                        if item in direct_items:
                            continue  # direct 1-hop evidence already covers this item
                        dist = k1 + j
                        overlap_len = min(6 - k1, 6 - k2) * w1 * w2 * CHAIN_DECAY
                        evidence[item].append((dist, overlap_len))

            if evidence:
                min_dist_overall = min(min(d for d, _ in obs) for obs in evidence.values())
                n_at_min = sum(1 for obs in evidence.values() if min(d for d, _ in obs) == min_dist_overall)
                per_item = {
                    item: dict(min_distance=min(d for d, _ in obs),
                               vote_score=sum(o for _, o in obs),
                               max_overlap=max(o for _, o in obs),
                               unique=(n_at_min == 1))
                    for item, obs in evidence.items()
                }
                info[qid_a] = per_item
    return info


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df, user_index, assoc):
    n = len(df)
    cand = df["candidate_id"].to_numpy()
    uids = df["user_id"].to_numpy()
    qids = df["query_id"].to_numpy()
    weeks = df["anchor_week"].to_numpy()
    hist_vals = [df[c].to_numpy() for c in HIST_COLS]
    shift_info = build_shift_info(user_index)

    in_own_hist = np.zeros(n)
    own_hist_pos = np.full(n, np.nan)
    own_hist_ndistinct = np.zeros(n)
    n_future = np.zeros(n)          # strictly future: other query's week > this week
    n_sameweek = np.zeros(n)        # same anchor_week bucket -- order is ambiguous from week alone
    n_past = np.zeros(n)
    min_future_pos = np.full(n, np.nan)
    min_future_weekdiff = np.full(n, np.nan)
    min_sameweek_pos = np.full(n, np.nan)
    min_past_pos = np.full(n, np.nan)
    n_user_queries = np.zeros(n)
    is_shift_exact_next = np.zeros(n)
    is_shift_later_followup = np.zeros(n)
    unique_shift_exact = np.zeros(n)
    shift_distance = np.full(n, np.nan)
    shift_overlap_len = np.full(n, np.nan)
    shift_vote_score = np.zeros(n)

    for i in range(n):
        hl = [hist_vals[k][i] for k in range(6)]
        own_hist_ndistinct[i] = len(set(hl))
        c = cand[i]
        if c in hl:
            in_own_hist[i] = 1.0
            own_hist_pos[i] = hl.index(c) + 1

        others = user_index[uids[i]]
        qid, wk = qids[i], weeks[i]
        n_user_queries[i] = len(others)
        best_fpos = best_fwd = best_spos = best_ppos = None
        nf = n_sw = n_pa = 0
        for oqid, owk, ohist in others:
            if oqid == qid:
                continue
            if c in ohist:
                pos = ohist.index(c) + 1
                wd = owk - wk
                if wd > 0:
                    nf += 1
                    if best_fpos is None or pos < best_fpos:
                        best_fpos, best_fwd = pos, wd
                elif wd == 0:
                    n_sw += 1
                    if best_spos is None or pos < best_spos:
                        best_spos = pos
                else:
                    n_pa += 1
                    if best_ppos is None or pos < best_ppos:
                        best_ppos = pos
        n_future[i] = nf
        n_sameweek[i] = n_sw
        n_past[i] = n_pa
        if best_fpos is not None:
            min_future_pos[i] = best_fpos
            min_future_weekdiff[i] = best_fwd
        if best_spos is not None:
            min_sameweek_pos[i] = best_spos
        if best_ppos is not None:
            min_past_pos[i] = best_ppos

        info = shift_info.get(qid)
        if info is not None and c in info:
            ev = info[c]
            shift_distance[i] = ev["min_distance"]
            shift_overlap_len[i] = ev["max_overlap"]
            shift_vote_score[i] = ev["vote_score"]
            if ev["min_distance"] == 1:
                is_shift_exact_next[i] = 1.0
                unique_shift_exact[i] = 1.0 if ev["unique"] else 0.0
            else:
                is_shift_later_followup[i] = 1.0

    out = pd.DataFrame({
        "id": df["id"].values,
        "query_id": qids,
        "in_own_hist": in_own_hist,
        "own_hist_pos": own_hist_pos,
        "own_hist_ndistinct": own_hist_ndistinct,
        "has_future_match": (n_future > 0).astype(float),
        "n_future_match": n_future,
        "min_future_pos": min_future_pos,
        "min_future_weekdiff": min_future_weekdiff,
        "has_sameweek_match": (n_sameweek > 0).astype(float),
        "n_sameweek_match": n_sameweek,
        "min_sameweek_pos": min_sameweek_pos,
        "has_past_match": (n_past > 0).astype(float),
        "n_past_match": n_past,
        "min_past_pos": min_past_pos,
        "n_user_queries": n_user_queries,
        "anchor_week": weeks.astype(float),
        "is_shift_exact_next": is_shift_exact_next,
        "is_shift_later_followup": is_shift_later_followup,
        "unique_shift_exact": unique_shift_exact,
        "shift_distance": shift_distance,
        "shift_overlap_len": shift_overlap_len,
        "shift_vote_score": shift_vote_score,
    })

    item2idx, pmi, co_raw = assoc["item2idx"], assoc["pmi"], assoc["co_raw"]
    cand_idx = df["candidate_id"].map(item2idx).to_numpy()
    pmi_slots = np.zeros((n, 6))
    cnt_slots = np.zeros((n, 6))
    for k, col in enumerate(HIST_COLS):
        hidx = df[col].map(item2idx).to_numpy()
        pmi_slots[:, k] = pmi[hidx, cand_idx]
        cnt_slots[:, k] = co_raw[hidx, cand_idx]
        out[f"pmi_slot{k + 1}"] = pmi_slots[:, k]
    out["pmi_weighted_sum"] = (pmi_slots * SLOT_WEIGHT).sum(axis=1)
    out["cnt_weighted_sum"] = (cnt_slots * SLOT_WEIGHT).sum(axis=1)

    # Recency-weighted average of the 6 history items' "context" embeddings, compared via
    # cosine similarity to the candidate's "as-candidate" embedding -- a smoothed association
    # signal that generalizes beyond exact pairs the raw PMI matrix has little data on.
    hist_idx_mat = np.stack([df[c].map(item2idx).to_numpy() for c in HIST_COLS], axis=1)
    hist_embs = assoc["item_emb"][hist_idx_mat]
    w = SLOT_WEIGHT / SLOT_WEIGHT.sum()
    user_ctx_emb = _normalize_rows((hist_embs * w[None, :, None]).sum(axis=1))
    cand_emb = assoc["item_emb_cand"][cand_idx]
    out["svd_cos_sim"] = (user_ctx_emb * cand_emb).sum(axis=1)

    rwr = assoc["rwr"]
    rwr_slots = np.zeros((n, 6))
    for k, col in enumerate(HIST_COLS):
        hidx = df[col].map(item2idx).to_numpy()
        rwr_slots[:, k] = rwr[hidx, cand_idx]
        out[f"rwr_slot{k + 1}"] = rwr_slots[:, k]
    out["rwr_weighted_sum"] = (rwr_slots * SLOT_WEIGHT).sum(axis=1)
    out["rwr_max"] = rwr_slots.max(axis=1)

    out["cand_volume"] = assoc["total_vol"][cand_idx]
    out["cand_volume_log"] = np.log1p(out["cand_volume"])
    out["cand_vol_rank"] = out.groupby("query_id")["cand_volume"].rank(ascending=False, method="average")

    out["trend"] = assoc["trend"][weeks.astype(int), cand_idx]
    out["seq_recency_skew"] = assoc["seq_recency_skew"][cand_idx]

    add_group_relative_features(out, GROUP_RELATIVE_COLS)

    return out


# ---------------------------------------------------------------------------
# Group-relative (within-query) normalization: raw association/popularity scores don't
# compare across queries with different item pools, but their z-score/percentile rank among
# the current query's own 12 candidates does. Mirrors the pattern used in the sibling
# "Mobile App Privacy Policy Evidence Routing" quest, where this was "the single biggest
# lift in the first round of tuning".
# ---------------------------------------------------------------------------
def add_group_relative_features(df, cols):
    grp = df.groupby("query_id")
    for col in cols:
        mean = grp[col].transform("mean")
        std = grp[col].transform("std").replace(0, np.nan)
        df[f"{col}_gz"] = ((df[col] - mean) / std).fillna(0.0)
        df[f"{col}_gpct"] = grp[col].rank(pct=True)


# ---------------------------------------------------------------------------
# Post-model reranking: tried and rejected, kept wired in (as a no-op -- see RERANK_MIN_OVERLAP
# below) so this remains a live hook rather than dead code. A first attempt blended the model's
# score with a fixed additive boost for is_shift_exact_next at *every* confidence level, and
# that made CV monotonically worse. The hypothesis was that gating the override on high
# confidence (shift_overlap_len, where raw precision reaches 92.1% at overlap=4 and 99.2% at
# overlap=5) would fix that. It didn't: CV was flat-to-slightly-worse even at overlap>=5 only
# (delta -0.0001 to -0.0023 across thresholds 3-5). Root cause, found by inspecting OOF
# predictions directly: among overlap>=5 exact-next candidates, the model *already* ranks them
# #1 in 1990/1997 cases (99.6%), with 99.5% precision there -- it has already learned to trust
# this signal almost perfectly. In the remaining 7 cases where the model does NOT rank the
# shift candidate #1, precision is only 14.3% (1/7) -- the model was usually *right* to
# disagree, using context (other features) a fixed rule can't see. Forcing an override in that
# tiny disagreement set actively hurts more than the (near-zero) agreement set could ever gain.
# RERANK_MIN_OVERLAP=6 is unreachable (SHIFT_MAX_K caps shift_overlap_len at 5), so
# apply_rerank_rules is an intentional no-op in the shipped model.
# ---------------------------------------------------------------------------
RERANK_MIN_OVERLAP = 6

# Soft rank blending: rather than forcing an override, nudge the model's own per-query rank
# percentile toward a shift-confidence-weighted percentile by a small weight -- tried as a
# gentler alternative to the hard override above (which was rejected), in case a soft nudge
# would avoid the disagreement-case damage a hard override causes. It didn't: tested at
# RERANK_BLEND_WEIGHT in {0.05, 0.1, 0.2, 0.3, 0.5}, results were flat at the smallest weights
# (0.05-0.1, where the blend barely moves anything) then monotonically worse from 0.2 upward
# (0.8684 -> 0.8646 at 0.5) -- the same pattern as the hard override, for the same underlying
# reason: the model already trusts these features about as well as any external rule can.
# This is the third distinct reranking formulation tried and rejected in this project (blanket
# additive boost, confidence-gated hard override, soft rank blend); 0.0 disables it, i.e. the
# model's own score is used unchanged.
RERANK_BLEND_WEIGHT = 0.0


def apply_rerank_rules(df, score_col):
    score = df[score_col].to_numpy(dtype=float).copy()

    if RERANK_BLEND_WEIGHT > 0:
        model_pct = df.groupby("query_id")[score_col].rank(pct=True)
        shift_raw = (df["is_shift_exact_next"] * 3.0 + df["is_shift_later_followup"] * 1.0) \
            * (1.0 + df["shift_overlap_len"].fillna(0.0))
        shift_pct = shift_raw.groupby(df["query_id"]).rank(pct=True)
        score = ((1.0 - RERANK_BLEND_WEIGHT) * model_pct + RERANK_BLEND_WEIGHT * shift_pct).to_numpy(dtype=float)

    overlap = df["shift_overlap_len"].to_numpy(dtype=float)
    is_exact = df["is_shift_exact_next"].to_numpy(dtype=float)
    override = (is_exact == 1.0) & (overlap >= RERANK_MIN_OVERLAP)
    if override.any():
        max_score = pd.Series(score).groupby(df["query_id"].values).transform("max").to_numpy(dtype=float)
        # + overlap as a tiebreaker if two candidates in the same query both qualify (rare --
        # only happens with conflicting shift evidence from different other-queries).
        score = np.where(override, max_score + 1.0 + overlap, score)
    return score


# ---------------------------------------------------------------------------
# Local metric implementation (NDCG@3) matching the grader's exact formula
# ---------------------------------------------------------------------------
def dcg(relevance):
    relevance = np.asarray(relevance, dtype=float)
    gains = np.power(2.0, relevance) - 1.0
    discounts = np.log2(np.arange(2, len(relevance) + 2, dtype=float))
    return float(np.sum(gains / discounts))


def query_ndcg_at_3(true_relevance, predicted_scores):
    true_relevance = np.asarray(true_relevance, dtype=float)
    predicted_scores = np.asarray(predicted_scores, dtype=float)
    order = np.argsort(-predicted_scores)
    ranked = true_relevance[order][:3]
    ideal = np.sort(true_relevance)[::-1][:3]
    ideal_dcg = dcg(ideal)
    return dcg(ranked) / ideal_dcg if ideal_dcg > 0 else 1.0


def mean_ndcg_at_3(df, score_col, rel_col="relevance", qid_col="query_id"):
    scores = [query_ndcg_at_3(g[rel_col].values, g[score_col].values)
              for _, g in df.groupby(qid_col, sort=False)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Model training (LightGBM lambdarank primary, pairwise-ranking fallback below).
# Both are seed-bagged: train several identically-configured models that only differ in
# random seed, then average their predictions -- see SEED_BAG_SEEDS above for why.
# ---------------------------------------------------------------------------
def train_lgb_bagged(X_tr, y_tr, group_tr, num_boost_round=NUM_BOOST_ROUND, seeds=SEED_BAG_SEEDS):
    models = []
    for sd in seeds:
        params = dict(LGB_PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
        train_set = lgb.Dataset(X_tr, label=y_tr, group=group_tr)
        models.append(lgb.train(params, train_set, num_boost_round=num_boost_round))
    return models


def predict_lgb_bagged(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def _pairwise_transform(X, y, group):
    """Within each query, build one training example per candidate pair of differing
    relevance: the feature difference (X_a - X_b), labeled 1 if a's relevance is higher."""
    Xi, labels = [], []
    start = 0
    for g in group:
        idx = np.arange(start, start + g)
        start += g
        Xg, yg = X[idx], y[idx]
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a == b or yg[a] == yg[b]:
                    continue
                Xi.append(Xg[a] - Xg[b])
                labels.append(1 if yg[a] > yg[b] else 0)
    return np.array(Xi), np.array(labels)


def train_pairwise_bagged(X_tr, y_tr, group_tr, seeds=SEED_BAG_SEEDS):
    """Pairwise-ranking fallback for when LightGBM is unavailable. A plain pointwise
    regressor on `relevance` ignores that only the *relative* order within a query matters
    (an absolute relevance value carries no meaning by itself); this instead trains a
    classifier on within-query pairwise feature differences (a simplified RankNet-style
    approach) and, at inference, scores each candidate by its round-robin win-rate against
    every other candidate in its own query -- see predict_pairwise_bagged."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xd, yd = _pairwise_transform(X_tr, y_tr, group_tr)
    models = []
    for sd in seeds:
        model = HistGradientBoostingClassifier(random_state=sd, max_iter=NUM_BOOST_ROUND)
        model.fit(Xd, yd)
        models.append(model)
    return models


def predict_pairwise_bagged(models, X, group):
    n = len(X)
    scores = np.zeros(n)
    start = 0
    for g in group:
        idx = np.arange(start, start + g)
        start += g
        sub = X[idx]
        m = len(idx)
        if m <= 1:
            continue
        pairs = [(a, b) for a in range(m) for b in range(m) if a != b]
        diffs = np.array([sub[a] - sub[b] for a, b in pairs])
        win_prob = np.mean([mdl.predict_proba(diffs)[:, 1] for mdl in models], axis=0)
        group_scores = np.zeros(m)
        for (a, _), p in zip(pairs, win_prob):
            group_scores[a] += p
        scores[idx] = group_scores
    return scores


# ---------------------------------------------------------------------------
# Exact-next binary head: a LightGBM binary classifier predicting P(relevance==3), stacked in
# as one extra feature for the main ranker rather than used as a standalone score or blended
# in directly. Both alternatives were tried and rejected first: the classifier alone scores far
# worse than the ranker (0.8413 vs. 0.8693 CV NDCG@3, since collapsing relevance to a binary
# "is it exactly this one" target throws away the relevance=1 information needed to fill
# ranking positions 2-3), and rank-blending it in at any real weight (tested 0.5-0.9) makes
# things monotonically worse, mirroring every other manual-reranking attempt in this project.
# Feeding its prediction in as a feature instead lets the main ranker learn how much to trust
# it -- validated via repeated CV: +0.0021 NDCG@3 on top of the RWR features (20 folds, 4
# repeats). Must be produced out-of-fold to avoid leaking relevance into its own feature: a
# 3-fold inner GroupKFold on the training portion supplies OOF predictions for those rows, and
# a model refit on the full training portion scores the validation/test rows.
# ---------------------------------------------------------------------------
EXACT_HEAD_INNER_FOLDS = 3
EXACT_HEAD_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    lambda_l2=1.0,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    seed=SEED,
    bagging_seed=SEED,
    feature_fraction_seed=SEED,
    data_random_seed=SEED,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)


class _NeutralExactHead:
    """Fallback used when LightGBM is unavailable: a constant, uninformative prediction, so
    the pairwise-ranking fallback path still runs without the exact_head feature crashing it
    (it just carries no signal in that case)."""
    def predict(self, X):
        return np.zeros(len(X))


def add_exact_head_oof(tr_df, feature_cols):
    """Returns (tr_df with an OOF-safe 'exact_head' column, a model fit on all of tr_df to
    score rows outside it, e.g. a validation fold or the test set)."""
    tr_df = tr_df.reset_index(drop=True).copy()
    if not HAS_LGB:
        tr_df[EXACT_HEAD_COL] = 0.0
        return tr_df, _NeutralExactHead()

    stack = np.zeros(len(tr_df))
    inner_gkf = GroupKFold(n_splits=EXACT_HEAD_INNER_FOLDS)
    for itr, iva in inner_gkf.split(tr_df, tr_df["relevance"].values, groups=tr_df["user_id"].values):
        y_inner = (tr_df.iloc[itr]["relevance"].values == 3).astype(int)
        d = lgb.Dataset(tr_df.iloc[itr][feature_cols], label=y_inner)
        m = lgb.train(EXACT_HEAD_PARAMS, d, num_boost_round=NUM_BOOST_ROUND)
        stack[iva] = m.predict(tr_df.iloc[iva][feature_cols])
    tr_df[EXACT_HEAD_COL] = stack

    y_full = (tr_df["relevance"].values == 3).astype(int)
    d_full = lgb.Dataset(tr_df[feature_cols], label=y_full)
    full_model = lgb.train(EXACT_HEAD_PARAMS, d_full, num_boost_round=NUM_BOOST_ROUND)
    return tr_df, full_model


def zscore_per_query(df, col, qid_col="query_id"):
    grp = df.groupby(qid_col)[col]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).fillna(0.0)


def _train_predict(feature_cols, X_tr, y_tr, group_tr, X_va, group_va, used_fallback):
    """Train+predict with LightGBM (falling back to pairwise-ranking if it fails/unavailable),
    returning (predictions, used_fallback)."""
    if not used_fallback:
        try:
            return predict_lgb_bagged(train_lgb_bagged(X_tr, y_tr, group_tr), X_va), False
        except Exception as e:
            print(f"[WARN] LightGBM failed ({e}); switching to pairwise-ranking fallback.")
    return predict_pairwise_bagged(train_pairwise_bagged(X_tr, y_tr, group_tr), X_va, group_va), True


def run_cv_and_final_fit(train_feat):
    """GroupKFold-by-user_id CV for diagnostics, then refit on 100% of train.

    Trains two models -- FEATURE_COLS+exact_head ("full") and the smaller CORE_FEATURE_COLS
    ("core") -- and blends their per-query z-scores (ENSEMBLE_WEIGHT_FULL toward full). The
    two are different enough (correlation ~0.77 between their predictions) that this measurably
    beats the full model alone in CV, mirroring the one confirmed real-world win in this
    project: a blend of two differently-configured models outscored either alone (see
    readme.txt Results)."""
    train_feat = train_feat.sort_values("query_id", kind="stable").reset_index(drop=True)
    y_all = train_feat["relevance"].values
    groups_user = train_feat["user_id"].values
    full_cols = FEATURE_COLS + [EXACT_HEAD_COL]

    used_fallback = not HAS_LGB
    oof_score = np.zeros(len(train_feat))

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_ndcgs = []
    for fold_num, (tr_idx, va_idx) in enumerate(gkf.split(train_feat, y_all, groups=groups_user), start=1):
        tr_df, exact_head_model = add_exact_head_oof(train_feat.iloc[tr_idx], FEATURE_COLS)
        va_df = train_feat.iloc[va_idx].copy()
        va_df[EXACT_HEAD_COL] = exact_head_model.predict(va_df[FEATURE_COLS])
        group_tr = tr_df.groupby("query_id", sort=False).size().values
        group_va = va_df.groupby("query_id", sort=False).size().values

        pred_full, used_fallback = _train_predict(
            full_cols, tr_df[full_cols].values, tr_df["relevance"].values, group_tr,
            va_df[full_cols].values, group_va, used_fallback)
        pred_core, used_fallback = _train_predict(
            CORE_FEATURE_COLS, tr_df[CORE_FEATURE_COLS].values, tr_df["relevance"].values, group_tr,
            va_df[CORE_FEATURE_COLS].values, group_va, used_fallback)

        va_df["pred_full"] = pred_full
        va_df["pred_core"] = pred_core
        va_df["pred"] = (ENSEMBLE_WEIGHT_FULL * zscore_per_query(va_df, "pred_full")
                          + (1 - ENSEMBLE_WEIGHT_FULL) * zscore_per_query(va_df, "pred_core"))
        va_df["final_score"] = apply_rerank_rules(va_df, "pred")
        ndcg = mean_ndcg_at_3(va_df, "final_score")
        fold_ndcgs.append(ndcg)
        print(f"[CV fold {fold_num}] NDCG@3={ndcg:.4f}")
        oof_score[va_idx] = va_df["final_score"].values

    train_feat["oof_pred"] = oof_score
    overall_ndcg = mean_ndcg_at_3(train_feat, "oof_pred")
    print(f"[CV overall] pooled OOF NDCG@3={overall_ndcg:.4f} "
          f"(fold mean={np.mean(fold_ndcgs):.4f} +/- {np.std(fold_ndcgs):.4f}, "
          f"range {min(fold_ndcgs):.4f}-{max(fold_ndcgs):.4f})")

    print(f"[INFO] using {'pairwise-ranking fallback' if used_fallback else 'LightGBM lambdarank'} "
          f"for the final full+core ensemble ({len(SEED_BAG_SEEDS)}-seed bagged each), "
          f"refit on 100% of train.")
    train_feat, exact_head_model_full = add_exact_head_oof(train_feat, FEATURE_COLS)
    y_all = train_feat["relevance"].values
    group_full = train_feat.groupby("query_id", sort=False).size().values

    if not used_fallback:
        try:
            full_models = train_lgb_bagged(train_feat[full_cols].values, y_all, group_full)
            core_models = train_lgb_bagged(train_feat[CORE_FEATURE_COLS].values, y_all, group_full)
        except Exception as e:
            print(f"[WARN] LightGBM failed on final fit ({e}); switching to pairwise-ranking fallback.")
            used_fallback = True
            full_models = train_pairwise_bagged(train_feat[full_cols].values, y_all, group_full)
            core_models = train_pairwise_bagged(train_feat[CORE_FEATURE_COLS].values, y_all, group_full)
    else:
        full_models = train_pairwise_bagged(train_feat[full_cols].values, y_all, group_full)
        core_models = train_pairwise_bagged(train_feat[CORE_FEATURE_COLS].values, y_all, group_full)

    return full_models, core_models, used_fallback, exact_head_model_full


def predict_test(full_models, core_models, used_fallback, test_feat, exact_head_model_full):
    # Sort by query_id so group sizes line up contiguously for the pairwise fallback path;
    # build_submission reindexes by "id" afterward, so this reordering is harmless either way.
    test_feat = test_feat.sort_values("query_id", kind="stable").reset_index(drop=True)
    test_feat[EXACT_HEAD_COL] = exact_head_model_full.predict(test_feat[FEATURE_COLS])
    full_cols = FEATURE_COLS + [EXACT_HEAD_COL]

    if used_fallback:
        group_test = test_feat.groupby("query_id", sort=False).size().values
        test_feat["pred_full"] = predict_pairwise_bagged(full_models, test_feat[full_cols].values, group_test)
        test_feat["pred_core"] = predict_pairwise_bagged(core_models, test_feat[CORE_FEATURE_COLS].values, group_test)
    else:
        test_feat["pred_full"] = predict_lgb_bagged(full_models, test_feat[full_cols].values)
        test_feat["pred_core"] = predict_lgb_bagged(core_models, test_feat[CORE_FEATURE_COLS].values)

    test_feat["pred"] = (ENSEMBLE_WEIGHT_FULL * zscore_per_query(test_feat, "pred_full")
                          + (1 - ENSEMBLE_WEIGHT_FULL) * zscore_per_query(test_feat, "pred_core"))
    test_feat["response_score"] = apply_rerank_rules(test_feat, "pred")
    return test_feat


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def build_submission(test_feat, test_ids):
    sub = test_feat[["id", "response_score"]].copy()
    sub = sub.set_index("id").loc[test_ids].reset_index()

    assert len(sub) == len(test_ids), f"expected {len(test_ids)} rows, got {len(sub)}"
    assert set(sub["id"]) == set(test_ids), "id set mismatch vs. test.csv"
    assert sub["id"].duplicated().sum() == 0, "duplicate id in submission"
    assert np.isfinite(sub["response_score"]).all(), "non-finite response_score values"
    assert list(sub.columns) == ["id", "response_score"]

    return sub


def main():
    t0 = time.time()
    np.random.seed(SEED)

    print(f"[INFO] HAS_LGB={HAS_LGB}" + ("" if HAS_LGB else " -- lightgbm not importable, "
          "will use the pairwise-ranking fallback for the whole run."))

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    print(f"[INFO] train: {train_df.shape}, test: {test_df.shape}")

    assoc = build_item_association(train_df, test_df)
    print(f"[INFO] item association built ({len(assoc['item2idx'])} unique items)")

    train_user_index = build_user_index(train_df)
    test_user_index = build_user_index(test_df)

    train_feat = engineer_features(train_df, train_user_index, assoc)
    train_feat["relevance"] = train_df["relevance"].values
    train_feat["user_id"] = train_df["user_id"].values
    test_feat = engineer_features(test_df, test_user_index, assoc)
    print(f"[INFO] engineered {len(FEATURE_COLS)} features "
          f"(train {train_feat.shape}, test {test_feat.shape}) in {time.time() - t0:.1f}s")

    full_models, core_models, used_fallback, exact_head_model_full = run_cv_and_final_fit(train_feat)
    test_feat = predict_test(full_models, core_models, used_fallback, test_feat, exact_head_model_full)

    sub = build_submission(test_feat, sample_sub["id"].tolist())

    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"[DONE] wrote {len(sub)} rows to {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
