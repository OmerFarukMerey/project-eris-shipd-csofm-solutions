"""Privacy-policy evidence routing: rank candidate policy segments per question.

Reads dataset/public/{train,test}.csv, trains a model from scratch, and writes the
top-5 ranked candidate_ids per test query_id to ./working/submission.csv.

Train and test are policy-disjoint (no shared policy_id), so the model cannot rely on
memorized policy text -- it has to learn transferable textual patterns instead:
  - keyword-category alignment between the question and the segment (does the question
    ask about "sharing" and does the segment talk about sharing/third parties, etc.)
  - lexical overlap (stopword-filtered token overlap / Jaccard)
  - TF-IDF (word + character n-gram) and LSA cosine similarity between question and segment
  - structural cues (segment length, section-header-style short segments)
  - group-relative (per-query) normalization of all the continuous signals above, since raw
    similarity scales differ a lot across policies with different vocabularies

A LightGBM lambdarank model is trained on these features with GroupKFold cross-validation
split on policy_id (not query_id), matching the train/test policy-disjoint structure. The
model's ranking is blended with the raw TF-IDF-char and LSA similarity signals (rank-averaged)
since that measurably improved held-out ranking quality over the model alone.

If lightgbm is unavailable, the script falls back to a scikit-learn
HistGradientBoostingClassifier trained pointwise on relevance, so a valid submission is
always produced.

Optionally, if torch/transformers and a local copy of microsoft/deberta-v3-base are
available, a cross-encoder (question [SEP] segment -> relevance) is fine-tuned with
hard-negative mining and its score is fed in as an extra feature (both into the LightGBM
model and the final blend). This step requires a pretrained model that may not be present
in every environment, so it is wrapped in a try/except and silently skipped if unavailable
-- the rest of the pipeline is unaffected either way.
"""

import math
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
TOP_K = 5
NUM_BOOST_ROUND = 150
ENSEMBLE_WEIGHTS = {"model": 0.75, "sim_c": 0.10, "sim_lsa": 0.15}
# Used instead of ENSEMBLE_WEIGHTS when the cross-encoder feature is available. A first
# attempt at 0.55/0.10/0.10/0.25 measurably *hurt* OOF score (80.48) vs the LightGBM model
# alone (77.16) -- the model already ingests ce_score as an input feature and is far
# stronger than before, so it needs much less rescuing from the raw similarity signals.
# These weights lean heavily on the model with only small hedges; not exhaustively
# grid-searched (that costs another full paired-holdout sweep on top of the ~40min
# cross-encoder training), but directly justified by that large, unambiguous gap.
ENSEMBLE_WEIGHTS_WITH_CE = {"model": 0.90, "sim_c": 0.03, "sim_lsa": 0.02, "ce": 0.05}

# ---------------------------------------------------------------------------
# Cross-encoder (optional): fine-tune a pretrained transformer as a
# question [SEP] segment -> relevance scorer. See maybe_add_cross_encoder_features.
# ---------------------------------------------------------------------------
CROSS_ENCODER_MODEL = "microsoft/deberta-v3-base"
CE_MAX_LENGTH = 96
CE_BATCH_SIZE = 32
CE_INFER_BATCH = 128
CE_EPOCHS = 3
CE_LR = 2e-5
CE_HARD_NEG_K = 20
CE_FOLDS = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "public")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_DIR = os.path.join(BASE_DIR, "working")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")
CE_CACHE_PATH = os.path.join(OUT_DIR, "ce_cache.npz")

LGB_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[5],
    learning_rate=0.05,
    num_leaves=15,
    min_data_in_leaf=40,
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

# ---------------------------------------------------------------------------
# Keyword categories: question/segment both mentioning the same privacy-practice
# topic is a much stronger signal than raw word overlap (which many irrelevant
# segments share via generic words like "data"/"information"/"collect").
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "collect": [r"collect", r"gather", r"obtain", r"acquir"],
    "share_disclose": [r"\bshare", r"disclos", r"\bsell", r"third[- ]part", r"third parties",
                        r"vendor", r"business partner"],
    "security": [r"secur", r"encrypt", r"safeguard", r"protect"],
    "retention_deletion": [r"retain", r"retention", r"delet", r"eras", r"remov", r"how long"],
    "access_rights_optout": [r"right to access", r"opt[- ]out", r"unsubscrib",
                              r"withdraw.*consent", r"rectif", r"correct your", r"portab",
                              r"your rights", r"do not sell"],
    "cookies": [r"cookie", r"web beacon", r"pixel", r"tracking technolog", r"\bsdk\b", r"analytics"],
    "location": [r"location", r"geolocat", r"\bgps\b", r"geograph"],
    "children": [r"\bchild", r"\bminor", r"under the age", r"coppa"],
    "account": [r"\baccount", r"sign[- ]up", r"registrat", r"log[- ]in", r"username", r"password"],
    "marketing": [r"market", r"advertis", r"promotion", r"newsletter"],
    "contact": [r"contact us", r"email us", r"customer support", r"reach us"],
    "changes_to_policy": [r"chang.*polic", r"updat.*polic", r"revis", r"amend", r"effective date"],
    "international_transfer": [r"international", r"cross[- ]border", r"outside (?:the|your) countr",
                                r"privacy shield", r"gdpr"],
    "payment": [r"payment", r"credit card", r"billing", r"purchase", r"transaction"],
    "social_login": [r"facebook", r"google", r"single sign[- ]on", r"\bsso\b", r"oauth",
                      r"social media"],
    "breach": [r"breach", r"incident", r"unauthorized access", r"compromis", r"\bhack"],
}
CATEGORY_PATTERNS = {
    name: re.compile("|".join(kws), re.IGNORECASE) for name, kws in CATEGORY_KEYWORDS.items()
}

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)

GROUP_RELATIVE_BASE_COLS = [
    "sim_w", "sim_c", "sim_lsa", "lex_jaccard", "lex_overlap", "lex_recall_q",
    "struct_seg_charlen", "struct_seg_wordcount",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def tokenize(text):
    return set(w for w in TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS and len(w) > 1)


def add_keyword_category_features(df):
    q_lower = df["question"].str.lower()
    s_lower = df["policy_segment"].str.lower()
    q_cat_count = np.zeros(len(df), dtype=np.int16)
    s_cat_count = np.zeros(len(df), dtype=np.int16)
    match_count = np.zeros(len(df), dtype=np.int16)
    for name, pattern in CATEGORY_PATTERNS.items():
        q_match = q_lower.str.contains(pattern, regex=True, na=False)
        s_match = s_lower.str.contains(pattern, regex=True, na=False)
        df[f"kw_{name}"] = (q_match & s_match).astype(np.int8)
        # segment raises a topic the question never asked about -- a distractor signal
        df[f"kw_{name}_mismatch"] = (s_match & ~q_match).astype(np.int8)
        q_cat_count += q_match.values
        s_cat_count += s_match.values
        match_count += (q_match & s_match).values
    df["kw_q_category_count"] = q_cat_count
    df["kw_s_category_count"] = s_cat_count
    # categories the segment raises that the question didn't ask about, at all
    df["kw_mismatch_count"] = s_cat_count - match_count


def add_lexical_overlap_features(df):
    q_tokens = df["question"].map(tokenize)
    s_tokens = df["policy_segment"].map(tokenize)
    overlap = np.array([len(a & b) for a, b in zip(q_tokens, s_tokens)])
    q_n = q_tokens.map(len).values
    s_n = s_tokens.map(len).values
    union = q_n + s_n - overlap
    df["lex_overlap"] = overlap
    df["lex_jaccard"] = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
    df["lex_recall_q"] = np.divide(overlap, q_n, out=np.zeros_like(overlap, dtype=float), where=q_n > 0)
    df["lex_qtok_n"] = q_n
    df["lex_segtok_n"] = s_n


def add_structural_features(df):
    seg = df["policy_segment"]
    word_count = seg.str.split().str.len()
    df["struct_seg_charlen"] = seg.str.len()
    df["struct_seg_wordcount"] = word_count
    df["struct_q_charlen"] = df["question"].str.len()
    is_short = word_count <= 5
    looks_like_header = seg.str.strip().str.match(r"^[A-Z][a-zA-Z ]+\.?$")
    df["struct_short_header"] = (is_short & looks_like_header).astype(np.int8)


class TextSimilarityFeaturizer:
    """TF-IDF (word + char) and LSA cosine similarity between question and segment."""

    def fit(self, *text_series):
        pool = pd.concat(text_series, ignore_index=True).drop_duplicates()
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", min_df=2, sublinear_tf=True
        ).fit(pool)
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000
        ).fit(pool)
        # algorithm="arpack" avoids a divide-by-zero/overflow instability that the default
        # randomized algorithm hits on this sparse matrix's shape/rank.
        self.svd = TruncatedSVD(n_components=100, random_state=SEED, algorithm="arpack").fit(
            self.word_vec.transform(pool)
        )
        return self

    @staticmethod
    def _row_cosine_sparse(A, B):
        # TfidfVectorizer rows are L2-normalized by default, so cosine == dot product.
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()

    @staticmethod
    def _row_cosine_dense(A, B):
        num = (A * B).sum(axis=1)
        denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9
        return num / denom

    def transform(self, question_series, segment_series):
        Qw = self.word_vec.transform(question_series)
        Sw = self.word_vec.transform(segment_series)
        sim_w = self._row_cosine_sparse(Qw, Sw)

        Qc = self.char_vec.transform(question_series)
        Sc = self.char_vec.transform(segment_series)
        sim_c = self._row_cosine_sparse(Qc, Sc)

        Ql = self.svd.transform(Qw)
        Sl = self.svd.transform(Sw)
        sim_lsa = self._row_cosine_dense(Ql, Sl)

        return pd.DataFrame({"sim_w": sim_w, "sim_c": sim_c, "sim_lsa": sim_lsa})


def add_group_relative_features(df):
    grp = df.groupby("query_id")
    for col in GROUP_RELATIVE_BASE_COLS:
        mean = grp[col].transform("mean")
        std = grp[col].transform("std").replace(0, np.nan)
        df[f"{col}_gz"] = ((df[col] - mean) / std).fillna(0.0)
        df[f"{col}_gpct"] = grp[col].rank(pct=True)


def build_features(train_df, test_df):
    """Compute the full feature pipeline for train and test, fit only on unsupervised text."""
    featurizer = TextSimilarityFeaturizer().fit(
        train_df["question"], train_df["policy_segment"],
        test_df["question"], test_df["policy_segment"],
    )

    for df in (train_df, test_df):
        add_keyword_category_features(df)
        add_lexical_overlap_features(df)
        add_structural_features(df)

    train_sim = featurizer.transform(train_df["question"], train_df["policy_segment"])
    test_sim = featurizer.transform(test_df["question"], test_df["policy_segment"])
    for col in train_sim.columns:
        train_df[col] = train_sim[col].values
        test_df[col] = test_sim[col].values

    add_group_relative_features(train_df)
    add_group_relative_features(test_df)

    feature_cols = (
        [f"kw_{name}" for name in CATEGORY_KEYWORDS]
        + [f"kw_{name}_mismatch" for name in CATEGORY_KEYWORDS]
        + ["kw_q_category_count", "kw_s_category_count", "kw_mismatch_count"]
        + ["lex_overlap", "lex_jaccard", "lex_recall_q", "lex_qtok_n", "lex_segtok_n"]
        + ["struct_seg_charlen", "struct_seg_wordcount", "struct_q_charlen", "struct_short_header"]
        + ["sim_w", "sim_c", "sim_lsa"]
        + [f"{col}_gz" for col in GROUP_RELATIVE_BASE_COLS]
        + [f"{col}_gpct" for col in GROUP_RELATIVE_BASE_COLS]
    )
    return feature_cols


# ---------------------------------------------------------------------------
# Local metric implementation (ndcg@5, AP@5) matching the grader's definition
# ---------------------------------------------------------------------------
def ndcg_at_5(rels_top5, r_total):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels_top5))
    k = min(5, r_total)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_5(rels_top5, r_total):
    hits, ap_sum = 0, 0.0
    for i, r in enumerate(rels_top5):
        if r:
            hits += 1
            ap_sum += hits / (i + 1)
    denom = min(r_total, 5)
    return ap_sum / denom if denom > 0 else 0.0


def evaluate(df, score_col):
    """Per-query ndcg@5 / AP@5 given a df with query_id, relevance, and a score column."""
    rows = []
    for _, g in df.groupby("query_id"):
        g_sorted = g.sort_values(score_col, ascending=False)
        rels_top5 = g_sorted["relevance"].values[:5]
        r_total = int(g["relevance"].sum())
        ndcg = ndcg_at_5(rels_top5, r_total)
        ap = ap_at_5(rels_top5, r_total)
        rows.append((ndcg, ap))
    arr = np.array(rows)
    ndcg_mean, ap_mean = arr[:, 0].mean(), arr[:, 1].mean()
    utility = 0.7 * ndcg_mean + 0.3 * ap_mean
    score = 100 * (1 - utility)
    return ndcg_mean, ap_mean, score


def rank_pct_within_query(df, score_col):
    return df.groupby("query_id")[score_col].rank(pct=True)


def ensemble_blend(df, model_score_col):
    rank_model = rank_pct_within_query(df, model_score_col)
    if "ce_score_gpct" in df.columns:
        w = ENSEMBLE_WEIGHTS_WITH_CE
        return (
            w["model"] * rank_model
            + w["sim_c"] * df["sim_c_gpct"]
            + w["sim_lsa"] * df["sim_lsa_gpct"]
            + w["ce"] * df["ce_score_gpct"]
        )
    w = ENSEMBLE_WEIGHTS
    return (
        w["model"] * rank_model
        + w["sim_c"] * df["sim_c_gpct"]
        + w["sim_lsa"] * df["sim_lsa_gpct"]
    )


# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional -- requires torch/transformers + a locally available
# pretrained model). Fine-tunes question [SEP] segment -> relevance, with hard-negative
# mining (most of a ~144-candidate pool is trivially irrelevant and teaches little; the
# near-miss distractors are what the model needs to learn to discriminate). OOF scores for
# train are produced via GroupKFold-by-policy_id (fewer folds than the LightGBM CV, since
# fine-tuning is far more expensive per fold) so the feature is leak-free regardless of how
# the LightGBM stage later folds the same data -- each row's score always comes from a
# cross-encoder that never saw that row's policy during its own training.
# ---------------------------------------------------------------------------
def _ce_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ce_hard_negative_subset(df, k=CE_HARD_NEG_K):
    parts = []
    for _, g in df.groupby("query_id", sort=False):
        parts.append(g[g["relevance"] == 1])
        parts.append(g[g["relevance"] == 0].nlargest(k, "sim_c"))
    return pd.concat(parts, ignore_index=True)


def _ce_encode(tokenizer, questions, segments, device):
    enc = tokenizer(
        list(questions), list(segments), padding="max_length", truncation=True,
        max_length=CE_MAX_LENGTH, return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _ce_train(train_subset, device):
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        CROSS_ENCODER_MODEL, num_labels=1
    ).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=CE_LR)

    questions = train_subset["question"].tolist()
    segments = train_subset["policy_segment"].tolist()
    labels = train_subset["relevance"].astype(float).tolist()
    n = len(questions)
    rng = np.random.RandomState(SEED)

    for _ in range(CE_EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, CE_BATCH_SIZE):
            idx = order[start:start + CE_BATCH_SIZE]
            b_q = [questions[i] for i in idx]
            b_s = [segments[i] for i in idx]
            b_y = torch.tensor([labels[i] for i in idx], dtype=torch.float32, device=device)
            enc = _ce_encode(tokenizer, b_q, b_s, device)
            logits = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, tokenizer


def _ce_predict(model, tokenizer, df, device):
    model.eval()
    questions = df["question"].tolist()
    segments = df["policy_segment"].tolist()
    scores = np.zeros(len(df))
    with torch.no_grad():
        for start in range(0, len(df), CE_INFER_BATCH):
            b_q = questions[start:start + CE_INFER_BATCH]
            b_s = segments[start:start + CE_INFER_BATCH]
            enc = _ce_encode(tokenizer, b_q, b_s, device)
            logits = model(**enc).logits.squeeze(-1)
            scores[start:start + len(b_q)] = torch.sigmoid(logits).cpu().numpy()
    return scores


def _ce_cache_signature(train_df, test_df):
    return "|".join(str(x) for x in [
        CROSS_ENCODER_MODEL, CE_MAX_LENGTH, CE_EPOCHS, CE_LR, CE_HARD_NEG_K, CE_FOLDS,
        SEED, len(train_df), len(test_df),
    ])


def maybe_add_cross_encoder_features(train_df, test_df):
    """Adds ce_score(_gz/_gpct) to train_df/test_df in place; returns True on success.
    Returns False (no columns added) if torch/the pretrained model is unavailable or
    training fails for any reason -- callers must treat this as a normal, expected path.

    Fine-tuning is expensive (tens of minutes), so results are cached to disk keyed by a
    signature of the config that affects them; reruns with unchanged CE_* settings and
    input sizes reuse the cache instead of retraining from scratch."""
    signature = _ce_cache_signature(train_df, test_df)
    if os.path.exists(CE_CACHE_PATH):
        try:
            cached = np.load(CE_CACHE_PATH, allow_pickle=True)
            if str(cached["signature"]) == signature:
                oof_ce = cached["oof_ce"]
                test_ce = cached["test_ce"]
                print(f"[CE] loaded cached scores from {CE_CACHE_PATH} (signature matched)")
                _ce_attach_features(train_df, test_df, oof_ce, test_ce)
                return True
            print("[CE] cache signature stale (config or data changed) -- retraining.")
        except Exception as e:
            print(f"[CE] cache unreadable ({e}) -- retraining.")

    if not HAS_TORCH:
        print("[CE] torch/transformers not available -- skipping cross-encoder.")
        return False

    device = _ce_device()
    try:
        AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
    except Exception as e:
        print(f"[CE] could not load {CROSS_ENCODER_MODEL} ({e}) -- skipping cross-encoder.")
        return False

    print(f"[CE] fine-tuning {CROSS_ENCODER_MODEL} on device={device} "
          f"({CE_FOLDS}-fold OOF + a final full-data model)")
    t0 = time.time()
    try:
        oof_ce = np.zeros(len(train_df))
        gkf = GroupKFold(n_splits=CE_FOLDS, shuffle=True, random_state=SEED)
        policies = train_df["policy_id"].values
        for fold_num, (tr_idx, va_idx) in enumerate(gkf.split(train_df, groups=policies), start=1):
            fold_train = _ce_hard_negative_subset(train_df.iloc[tr_idx])
            model, tokenizer = _ce_train(fold_train, device)
            oof_ce[va_idx] = _ce_predict(model, tokenizer, train_df.iloc[va_idx], device)
            del model
            print(f"[CE] fold {fold_num}/{CE_FOLDS} done ({time.time() - t0:.0f}s elapsed)")

        final_train = _ce_hard_negative_subset(train_df)
        final_model, final_tokenizer = _ce_train(final_train, device)
        test_ce = _ce_predict(final_model, final_tokenizer, test_df, device)
        print(f"[CE] final model trained, test scored ({time.time() - t0:.0f}s elapsed)")
    except Exception as e:
        print(f"[CE] training failed ({e}) -- skipping cross-encoder.")
        return False

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(CE_CACHE_PATH, oof_ce=oof_ce, test_ce=test_ce, signature=signature)
    print(f"[CE] cached scores to {CE_CACHE_PATH}")

    _ce_attach_features(train_df, test_df, oof_ce, test_ce)
    print(f"[CE] done in {time.time() - t0:.0f}s")
    return True


def _ce_attach_features(train_df, test_df, oof_ce, test_ce):
    train_df["ce_score"] = oof_ce
    test_df["ce_score"] = test_ce
    for df in (train_df, test_df):
        grp = df.groupby("query_id")["ce_score"]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        df["ce_score_gz"] = ((df["ce_score"] - mean) / std).fillna(0.0)
        df["ce_score_gpct"] = grp.rank(pct=True)


# ---------------------------------------------------------------------------
# Model training (LightGBM lambdarank primary, HistGradientBoosting fallback)
# ---------------------------------------------------------------------------
def train_lgb_fold(X_tr, y_tr, group_tr, X_va):
    train_set = lgb.Dataset(X_tr, label=y_tr, group=group_tr)
    model = lgb.train(LGB_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)
    return model.predict(X_va), model


def train_hgb_fold(X_tr, y_tr, X_va):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(random_state=SEED, max_iter=NUM_BOOST_ROUND)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1], model


def run_cv_and_final_fit(train_df, feature_cols):
    """GroupKFold-by-policy_id CV for diagnostics, then refit on 100% of train."""
    train_df = train_df.sort_values("query_id", kind="stable").reset_index(drop=True)
    X_all = train_df[feature_cols].values
    y_all = train_df["relevance"].values
    groups_policy = train_df["policy_id"].values

    used_fallback = not HAS_LGB
    oof_model_score = np.zeros(len(train_df))

    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_num = 0
    fold_scores = []
    for tr_idx, va_idx in gkf.split(X_all, y_all, groups=groups_policy):
        fold_num += 1
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr = y_all[tr_idx]

        if not used_fallback:
            try:
                group_tr = (
                    train_df.iloc[tr_idx].groupby("query_id", sort=False).size().values
                )
                pred, _ = train_lgb_fold(X_tr, y_tr, group_tr, X_va)
            except Exception as e:
                print(f"[WARN] LightGBM failed ({e}); switching to HistGradientBoosting fallback.")
                used_fallback = True
                pred, _ = train_hgb_fold(X_tr, y_tr, X_va)
        else:
            pred, _ = train_hgb_fold(X_tr, y_tr, X_va)

        oof_model_score[va_idx] = pred

        fold_df = train_df.iloc[va_idx].copy()
        fold_df["model_score"] = pred
        n5, a5, sc = evaluate(fold_df, "model_score")
        fold_scores.append(sc)
        print(f"[CV fold {fold_num}] model-only: ndcg@5={n5:.4f} ap@5={a5:.4f} score={sc:.2f}")

    train_df["model_score"] = oof_model_score
    train_df["blend_score"] = ensemble_blend(train_df, "model_score")

    n5, a5, sc = evaluate(train_df, "model_score")
    print(f"[CV overall] model-only OOF: ndcg@5={n5:.4f} ap@5={a5:.4f} score={sc:.2f}")
    n5, a5, sc = evaluate(train_df, "blend_score")
    print(f"[CV overall] ensemble blend OOF: ndcg@5={n5:.4f} ap@5={a5:.4f} score={sc:.2f}")
    print(
        f"[CV note] fold-to-fold variance across held-out policies can be large "
        f"(this run: {min(fold_scores):.0f}-{max(fold_scores):.0f} across {N_FOLDS} folds); "
        f"the pooled OOF score above is an optimistic estimate since the real test set "
        f"uses entirely unseen policies."
    )

    print(f"[INFO] using {'HistGradientBoosting fallback' if used_fallback else 'LightGBM lambdarank'} for the final model.")
    group_full = train_df.groupby("query_id", sort=False).size().values
    if not used_fallback:
        try:
            _, final_model = train_lgb_fold(X_all, y_all, group_full, X_all)
        except Exception as e:
            print(f"[WARN] LightGBM failed on final fit ({e}); switching to HistGradientBoosting fallback.")
            used_fallback = True
            _, final_model = train_hgb_fold(X_all, y_all, X_all)
    else:
        _, final_model = train_hgb_fold(X_all, y_all, X_all)

    return final_model, used_fallback


def predict_test(final_model, used_fallback, test_df, feature_cols):
    X_test = test_df[feature_cols].values
    if used_fallback:
        test_df["model_score"] = final_model.predict_proba(X_test)[:, 1]
    else:
        test_df["model_score"] = final_model.predict(X_test)
    test_df["blend_score"] = ensemble_blend(test_df, "model_score")
    return test_df


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def build_submission(test_df, sample_sub_order):
    query_to_candidates = test_df.groupby("query_id")["candidate_id"].apply(set).to_dict()

    rows = []
    for qid, g in test_df.groupby("query_id"):
        top5 = g.nlargest(TOP_K, "blend_score")["candidate_id"].tolist()
        rows.append((qid, "|".join(top5)))

    sub = pd.DataFrame(rows, columns=["query_id", "ranked_candidate_ids"])
    sub = sub.set_index("query_id").loc[sample_sub_order].reset_index()

    assert len(sub) == 215, f"expected 215 rows, got {len(sub)}"
    assert set(sub["query_id"]) == set(test_df["query_id"].unique()), "query_id set mismatch"
    assert sub["query_id"].duplicated().sum() == 0, "duplicate query_id in submission"
    for qid, ids_str in zip(sub["query_id"], sub["ranked_candidate_ids"]):
        ids = ids_str.split("|")
        assert len(ids) == TOP_K, f"query {qid} does not have exactly {TOP_K} candidate ids"
        assert len(set(ids)) == TOP_K, f"query {qid} has duplicate candidate ids"
        valid = query_to_candidates[qid]
        assert set(ids).issubset(valid), f"query {qid} has an id outside its candidate set"
    assert list(sub.columns) == ["query_id", "ranked_candidate_ids"]

    return sub


def main():
    t0 = time.time()
    np.random.seed(SEED)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print(f"[INFO] train: {train_df.shape}, test: {test_df.shape}")

    feature_cols = build_features(train_df, test_df)
    print(f"[INFO] built {len(feature_cols)} features")

    if maybe_add_cross_encoder_features(train_df, test_df):
        feature_cols = feature_cols + ["ce_score", "ce_score_gz", "ce_score_gpct"]
        print(f"[INFO] cross-encoder features added, now {len(feature_cols)} features")

    final_model, used_fallback = run_cv_and_final_fit(train_df, feature_cols)
    test_df = predict_test(final_model, used_fallback, test_df, feature_cols)

    sub = build_submission(test_df, sample_sub["query_id"].tolist())

    os.makedirs(OUT_DIR, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"[DONE] wrote {len(sub)} rows to {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
