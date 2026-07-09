"""
Coastal Sensor Signature Recommendation.

Recommends the top-5 candidate_patterns.csv pattern_id values for each test
query, by estimating the hidden 6-hour segment's summary statistics from the
observed before/after context and matching them against each candidate's
precomputed hidden-segment summary stats, plus:
  - two supervised regressors (LightGBM and Ridge) that learn to predict
    those summary statistics directly from training labels, fed as separate
    feature blocks so the ranker learns to combine their different biases
    (the single biggest lever in this project -- see readme),
  - a k-NN "neighbor vote" signal that exploits how heavily candidate
    patterns are reused across queries,
  - a light post-hoc frequency penalty aligned with the frequency-balanced
    evaluation metric.

Run: python solution.py   (produces ./working/submission.csv)
"""
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Apple Accelerate's BLAS backend spuriously flags "divide/invalid value encountered
# in matmul" on plain dot products with no actual NaN/Inf in the output (verified) -
# silence it so it doesn't drown out real warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "public")
WORKING_DIR = os.path.join(BASE_DIR, "working")

N_FOLDS = 5
ENSEMBLE_SEEDS = (42, 43, 44)
KNN_KS = (10, 30, 60, 120)
N_INNER_FOLDS = 6
FREQ_PENALTY_LAMBDA = 0.07

# slope weights for a 6-point evenly spaced window (relative times -2.5..2.5)
SLOPE_W = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64) / 17.5

# ---------------------------------------------------------------------------
# Feature engineering: query-side hidden-segment estimates
# ---------------------------------------------------------------------------

# (variable, has_delta, has_range) -- has_range only applies to wave height,
# the only candidate field with a *_range summary.
QUERY_VAR_SPECS = [
    ("wvht_anom", True, True),
    ("dpd", False, False),
    ("apd", False, False),
    ("wspd_anom", True, False),
    ("pres_anom", True, False),
    ("mwd_sin", False, False),
    ("mwd_cos", False, False),
]


def add_query_features(df):
    """Estimate hidden-segment summary stats from the boundary context (tm01/tp06),
    two ways per variable:
      - naive: simple average/difference of the immediate boundary values.
      - v2: one-step linear extrapolation using the trend over the last/first 6
        known hours on each side, projected 1 hour into the gap -- a tighter
        estimate of the values right at the edges of the true 6-hour hidden
        window (hours 0 and 5) than the raw boundary observations, which sit
        one hour outside it.
    Both are kept (not just v2) since the pair is a cheap ensemble-style signal
    for the ranker to combine, and CV confirmed v2 is only a mild improvement
    on its own -- combining both was never worse.
    """
    df = df.copy()

    def slope(cols):
        return df[cols].to_numpy(dtype=np.float64) @ SLOPE_W

    for var, has_delta, has_range in QUERY_VAR_SPECS:
        pre_win = [f"tm{i:02d}_{var}" for i in range(6, 0, -1)]
        post_win = [f"tp{i:02d}_{var}" for i in range(6, 12)]
        pre1 = df[f"tm01_{var}"]
        post1 = df[f"tp06_{var}"]

        df[f"q_{var}_naive_mean"] = (pre1 + post1) / 2.0
        if has_delta:
            df[f"q_{var}_naive_delta"] = post1 - pre1

        slope_pre = slope(pre_win)
        slope_post = slope(post_win)
        extrap_t0 = pre1.to_numpy(dtype=np.float64) + slope_pre
        extrap_t5 = post1.to_numpy(dtype=np.float64) - slope_post
        df[f"q_{var}_v2_mean"] = (extrap_t0 + extrap_t5) / 2.0
        if has_delta:
            df[f"q_{var}_v2_delta"] = extrap_t5 - extrap_t0
        df[f"q_{var}_slope_pre"] = slope_pre
        df[f"q_{var}_slope_post"] = slope_post

        if has_range:
            tight_win = [f"tm03_{var}", f"tm02_{var}", f"tm01_{var}",
                         f"tp06_{var}", f"tp07_{var}", f"tp08_{var}"]
            df[f"q_{var}_range_tight"] = df[tight_win].max(axis=1) - df[tight_win].min(axis=1)
            wide_win = pre_win + post_win
            df[f"q_{var}_range_wide"] = df[wide_win].max(axis=1) - df[wide_win].min(axis=1)
            df[f"q_{var}_std_wide"] = df[wide_win].std(axis=1)

    # No candidate counterpart for atmp / wtmp_anom: kept as auxiliary
    # (query-only) context so the model can still learn interactions with
    # candidate band features (e.g. water_temp_band).
    for var in ["atmp", "wtmp_anom"]:
        pre1 = df[f"tm01_{var}"]
        post1 = df[f"tp06_{var}"]
        df[f"q_{var}_pre"] = pre1
        df[f"q_{var}_post"] = post1
        df[f"q_{var}_mean"] = (pre1 + post1) / 2.0
        df[f"q_{var}_delta"] = post1 - pre1

    hour_angle = df["hour_band"].astype(float) * (2 * np.pi / 4.0)
    df["q_hour_sin"] = np.sin(hour_angle)
    df["q_hour_cos"] = np.cos(hour_angle)

    return df


def add_candidate_features(cand_df):
    """season_band is a clean 8-bin (45deg) circular bucket -> reconstruct a bin-center
    angle so query season_sin/season_cos can be compared via circular similarity
    instead of relying on exact season_band equality (candidates only realize bands 0-5)."""
    cand_df = cand_df.copy()
    season_center_rad = np.radians(cand_df["season_band"].astype(float) * 45.0 + 22.5)
    cand_df["c_season_sin"] = np.sin(season_center_rad)
    cand_df["c_season_cos"] = np.cos(season_center_rad)

    hour_angle = cand_df["hour_band"].astype(float) * (2 * np.pi / 4.0)
    cand_df["c_hour_sin"] = np.sin(hour_angle)
    cand_df["c_hour_cos"] = np.cos(hour_angle)
    return cand_df


# (query_col, cand_col) matched pairs -- diffed against the same candidate field for
# both the naive and v2 query-side estimates (see add_query_features docstring).
MATCHED_PAIRS = [
    ("q_wvht_anom_naive_mean", "cand_wvht_mean"),
    ("q_wvht_anom_v2_mean", "cand_wvht_mean"),
    ("q_wvht_anom_naive_delta", "cand_wvht_delta"),
    ("q_wvht_anom_v2_delta", "cand_wvht_delta"),
    ("q_wvht_anom_range_tight", "cand_wvht_range"),
    ("q_wvht_anom_range_wide", "cand_wvht_range"),
    ("q_wvht_anom_std_wide", "cand_wvht_range"),
    ("q_dpd_naive_mean", "cand_dpd_mean"),
    ("q_dpd_v2_mean", "cand_dpd_mean"),
    ("q_apd_naive_mean", "cand_apd_mean"),
    ("q_apd_v2_mean", "cand_apd_mean"),
    ("q_wspd_anom_naive_mean", "cand_wspd_mean"),
    ("q_wspd_anom_v2_mean", "cand_wspd_mean"),
    ("q_wspd_anom_naive_delta", "cand_wspd_delta"),
    ("q_wspd_anom_v2_delta", "cand_wspd_delta"),
    ("q_pres_anom_naive_mean", "cand_pres_mean"),
    ("q_pres_anom_v2_mean", "cand_pres_mean"),
    ("q_pres_anom_naive_delta", "cand_pres_delta"),
    ("q_pres_anom_v2_delta", "cand_pres_delta"),
    ("q_mwd_sin_naive_mean", "cand_mwd_sin_mean"),
    ("q_mwd_sin_v2_mean", "cand_mwd_sin_mean"),
    ("q_mwd_cos_naive_mean", "cand_mwd_cos_mean"),
    ("q_mwd_cos_v2_mean", "cand_mwd_cos_mean"),
]
QUERY_MATCHED_COLS = [p[0] for p in MATCHED_PAIRS]
CAND_MATCHED_COLS = [p[1] for p in MATCHED_PAIRS]

AUX_QUERY_COLS = [
    "season_sin", "season_cos", "hour_band", "swell_band", "wind_band", "water_temp_band",
    "q_atmp_pre", "q_atmp_post", "q_atmp_mean", "q_atmp_delta",
    "q_wtmp_anom_pre", "q_wtmp_anom_post", "q_wtmp_anom_mean", "q_wtmp_anom_delta",
    "q_wvht_anom_slope_pre", "q_wvht_anom_slope_post",
    "q_wspd_anom_slope_pre", "q_wspd_anom_slope_post",
    "q_pres_anom_slope_pre", "q_pres_anom_slope_post",
]

PAIR_FEATURE_NAMES = (
    [f"diff_{q}__{c}" for q, c in MATCHED_PAIRS]
    + [f"absdiff_{q}__{c}" for q, c in MATCHED_PAIRS]
    + [f"raw_{c}_{i}" for i, c in enumerate(CAND_MATCHED_COLS)]
    + ["season_sim", "hour_sim", "hour_eq",
       "swell_eq", "swell_absdiff", "wind_eq", "wind_absdiff", "wtemp_eq", "wtemp_absdiff"]
    + AUX_QUERY_COLS
)
KNN_FEATURE_NAMES = [f"knn_raw_k{k}" for k in KNN_KS] + [f"knn_summary_k{k}" for k in KNN_KS]
FEATURE_NAMES = PAIR_FEATURE_NAMES + KNN_FEATURE_NAMES


def build_pair_matrix(query_df, cand_df):
    """Vectorized (query x candidate) pairwise feature matrix via numpy broadcasting."""
    n_q = len(query_df)
    n_c = len(cand_df)

    Q = query_df[QUERY_MATCHED_COLS].to_numpy(dtype=np.float32)
    C = cand_df[CAND_MATCHED_COLS].to_numpy(dtype=np.float32)
    diff = Q[:, None, :] - C[None, :, :]
    absdiff = np.abs(diff)
    raw_cand = np.broadcast_to(C[None, :, :], (n_q, n_c, C.shape[1]))

    q_season_sin = query_df["season_sin"].to_numpy(dtype=np.float32)
    q_season_cos = query_df["season_cos"].to_numpy(dtype=np.float32)
    c_season_sin = cand_df["c_season_sin"].to_numpy(dtype=np.float32)
    c_season_cos = cand_df["c_season_cos"].to_numpy(dtype=np.float32)
    season_sim = (q_season_sin[:, None] * c_season_sin[None, :]
                  + q_season_cos[:, None] * c_season_cos[None, :])

    q_hour_sin = query_df["q_hour_sin"].to_numpy(dtype=np.float32)
    q_hour_cos = query_df["q_hour_cos"].to_numpy(dtype=np.float32)
    c_hour_sin = cand_df["c_hour_sin"].to_numpy(dtype=np.float32)
    c_hour_cos = cand_df["c_hour_cos"].to_numpy(dtype=np.float32)
    hour_sim = q_hour_sin[:, None] * c_hour_sin[None, :] + q_hour_cos[:, None] * c_hour_cos[None, :]

    q_hour_band = query_df["hour_band"].to_numpy(dtype=np.float32)
    c_hour_band = cand_df["hour_band"].to_numpy(dtype=np.float32)
    hour_eq = (q_hour_band[:, None] == c_hour_band[None, :]).astype(np.float32)

    def band_eq_absdiff(col):
        q = query_df[col].to_numpy(dtype=np.float32)
        c = cand_df[col].to_numpy(dtype=np.float32)
        eq = (q[:, None] == c[None, :]).astype(np.float32)
        ad = np.abs(q[:, None] - c[None, :])
        return eq, ad

    swell_eq, swell_ad = band_eq_absdiff("swell_band")
    wind_eq, wind_ad = band_eq_absdiff("wind_band")
    wtemp_eq, wtemp_ad = band_eq_absdiff("water_temp_band")

    aux = query_df[AUX_QUERY_COLS].to_numpy(dtype=np.float32)
    aux_full = np.repeat(aux, n_c, axis=0)

    X = np.concatenate(
        [
            diff.reshape(n_q * n_c, -1),
            absdiff.reshape(n_q * n_c, -1),
            raw_cand.reshape(n_q * n_c, -1),
            season_sim.reshape(-1, 1),
            hour_sim.reshape(-1, 1),
            hour_eq.reshape(-1, 1),
            swell_eq.reshape(-1, 1),
            swell_ad.reshape(-1, 1),
            wind_eq.reshape(-1, 1),
            wind_ad.reshape(-1, 1),
            wtemp_eq.reshape(-1, 1),
            wtemp_ad.reshape(-1, 1),
            aux_full,
        ],
        axis=1,
    ).astype(np.float32)

    return X


def build_labels(query_df, cand_df):
    cand_id_to_idx = {pid: i for i, pid in enumerate(cand_df["pattern_id"].to_numpy())}
    n_q = len(query_df)
    n_c = len(cand_df)
    labels = np.zeros((n_q, n_c), dtype=np.float32)
    for qi, rel_str in enumerate(query_df["relevant_pattern_ids"].to_numpy()):
        for pid in rel_str.split("|"):
            idx = cand_id_to_idx.get(pid)
            if idx is not None:
                labels[qi, idx] = 1.0
    return labels.reshape(-1)


# ---------------------------------------------------------------------------
# k-NN neighbor-vote features
#
# 636 of the 650 candidates are reused ~14x on average across the 1750 training
# queries, so "which candidates were relevant for similar training queries" is
# informative beyond raw aggregate-stat matching alone. For every query
# (train or test), find its K nearest OTHER training queries (leave-one-out
# when the query is itself a training row) and, per candidate, compute what
# fraction of those neighbors had that candidate in their relevant set. Two
# independent similarity spaces are used since they turned out complementary
# in CV: the full raw 216-column context (recency-weighted -- points nearer
# the hidden gap matter more) and the compact engineered summary-estimate
# space (feature-importance-weighted).
# ---------------------------------------------------------------------------

RAW_CONTEXT_COLS = (
    [f"tm{i:02d}_{v}" for i in range(12, 0, -1)
     for v in ["wvht_anom", "dpd", "apd", "wspd_anom", "pres_anom", "atmp", "wtmp_anom", "mwd_sin", "mwd_cos"]]
    + [f"tp{i:02d}_{v}" for i in range(6, 18)
       for v in ["wvht_anom", "dpd", "apd", "wspd_anom", "pres_anom", "atmp", "wtmp_anom", "mwd_sin", "mwd_cos"]]
)


def _raw_context_recency_weights():
    weights = []
    for col in RAW_CONTEXT_COLS:
        offset = int(col[2:4])
        dist_to_gap = offset if col.startswith("tm") else offset - 5
        weights.append(1.0 / dist_to_gap)
    return np.array(weights, dtype=np.float64)


RAW_CONTEXT_WEIGHTS = _raw_context_recency_weights()

SUMMARY_SIM_COLS_WEIGHTS = [
    ("q_wvht_anom_v2_mean", 1.5), ("q_wvht_anom_v2_delta", 1.2), ("q_wvht_anom_range_wide", 0.8),
    ("q_dpd_v2_mean", 0.8), ("q_apd_v2_mean", 1.0),
    ("q_wspd_anom_v2_mean", 1.8), ("q_wspd_anom_v2_delta", 1.4),
    ("q_pres_anom_v2_mean", 1.3), ("q_pres_anom_v2_delta", 1.0),
    ("q_mwd_sin_v2_mean", 1.0), ("q_mwd_cos_v2_mean", 1.0),
    ("season_sin", 1.0), ("season_cos", 1.0), ("q_hour_sin", 0.7), ("q_hour_cos", 0.7),
    ("swell_band", 1.3), ("wind_band", 0.5), ("water_temp_band", 1.0),
]
SUMMARY_SIM_COLS = [c for c, _ in SUMMARY_SIM_COLS_WEIGHTS]
SUMMARY_SIM_WEIGHTS = np.array([w for _, w in SUMMARY_SIM_COLS_WEIGHTS], dtype=np.float64)


def pairwise_sq_dist(A, B):
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    ab = A @ B.T
    return np.maximum(a2 + b2 - 2 * ab, 0.0)


def knn_adjacency(dist, k, exclude_self):
    dist = dist.copy()
    if exclude_self:
        np.fill_diagonal(dist, np.inf)
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    n, m = dist.shape
    adj = np.zeros((n, m), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    adj[rows, idx.reshape(-1)] = 1.0 / k
    return adj


def _knn_vote(tr_sim, va_sim, R_tr, ks):
    dist_tr_tr = pairwise_sq_dist(tr_sim, tr_sim)
    dist_va_tr = pairwise_sq_dist(va_sim, tr_sim)
    vote_tr, vote_va = [], []
    for k in ks:
        adj_tr = knn_adjacency(dist_tr_tr, k, exclude_self=True)
        vote_tr.append((adj_tr @ R_tr).reshape(-1, 1).astype(np.float32))
        adj_va = knn_adjacency(dist_va_tr, k, exclude_self=False)
        vote_va.append((adj_va @ R_tr).reshape(-1, 1).astype(np.float32))
    return np.concatenate(vote_tr, axis=1), np.concatenate(vote_va, axis=1)


def knn_vote_features(tr_q, va_q, cand_df, ks=KNN_KS):
    """Returns (vote_tr, vote_va): for tr_q, a leave-one-out vote among tr_q itself
    (safe to use as a training feature); for va_q, a vote against all of tr_q (safe
    whether va_q is a held-out CV fold or the real test set, since neither overlaps
    tr_q)."""
    n_c = len(cand_df)
    R_tr = build_labels(tr_q, cand_df).reshape(len(tr_q), n_c)

    raw_std = tr_q[RAW_CONTEXT_COLS].to_numpy(dtype=np.float64).std(axis=0)
    raw_std[raw_std < 1e-6] = 1.0
    raw_tr = (tr_q[RAW_CONTEXT_COLS].to_numpy(dtype=np.float64) / raw_std) * RAW_CONTEXT_WEIGHTS
    raw_va = (va_q[RAW_CONTEXT_COLS].to_numpy(dtype=np.float64) / raw_std) * RAW_CONTEXT_WEIGHTS
    raw_vote_tr, raw_vote_va = _knn_vote(raw_tr, raw_va, R_tr, ks)

    sum_std = tr_q[SUMMARY_SIM_COLS].to_numpy(dtype=np.float64).std(axis=0)
    sum_std[sum_std < 1e-6] = 1.0
    sum_tr = (tr_q[SUMMARY_SIM_COLS].to_numpy(dtype=np.float64) / sum_std) * SUMMARY_SIM_WEIGHTS
    sum_va = (va_q[SUMMARY_SIM_COLS].to_numpy(dtype=np.float64) / sum_std) * SUMMARY_SIM_WEIGHTS
    sum_vote_tr, sum_vote_va = _knn_vote(sum_tr, sum_va, R_tr, ks)

    vote_tr = np.concatenate([raw_vote_tr, sum_vote_tr], axis=1)
    vote_va = np.concatenate([raw_vote_va, sum_vote_va], axis=1)
    return vote_tr, vote_va


# ---------------------------------------------------------------------------
# Supervised pseudo-hidden-summary regressor
#
# The hand-crafted naive/v2 estimates above are physics-motivated heuristics
# with no calibration to the labels. But every TRAINING query already reveals
# its 5 true relevant candidates, so a much better estimate is available:
# average those 5 candidates' own cand_* fields into a "pseudo-target" for
# that query's true hidden segment, then train a regressor (raw 216-column
# context + the query's own season/hour/profile bands -> pseudo target) to
# learn the mapping directly from data. This was the single biggest lever
# found (CV MAP@5 0.373 -> 0.452 for the regressor itself, then a further
# 0.452 -> 0.485 just from adding the band/season inputs -- see readme).
# Nested out-of-fold predictions are used throughout so no row's regressor
# estimate is ever informed by its own labels.
# ---------------------------------------------------------------------------

REG_TARGET_COLS = [
    "cand_wvht_mean", "cand_wvht_delta", "cand_wvht_range",
    "cand_dpd_mean", "cand_apd_mean",
    "cand_wspd_mean", "cand_wspd_delta",
    "cand_pres_mean", "cand_pres_delta",
    "cand_mwd_sin_mean", "cand_mwd_cos_mean",
]

# The regressor's input space: the full raw context PLUS the query's own
# season/hour/profile bands. Adding the bands was the single biggest lever in
# round 5 (CV 0.373->0.452 in round 3 from the regressor itself, then a
# further 0.452->0.485 just from these 6 extra columns) -- the raw context
# alone tells the regressor the shape of the surrounding conditions, but not
# which physical regime (e.g. swell/wind/water-temp bucket) they sit in,
# which turns out to sharpen the summary-stat prediction substantially.
REG_EXTRA_COLS = ["season_sin", "season_cos", "hour_band", "swell_band", "wind_band", "water_temp_band"]
REG_INPUT_COLS = RAW_CONTEXT_COLS + REG_EXTRA_COLS

REGRESSOR_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbosity=-1,
)


def pseudo_targets(query_df, cand_indexed):
    """Per-query pseudo target for each candidate summary field: the average of
    that field across the query's 5 TRUE relevant candidates."""
    targets = {}
    for col in REG_TARGET_COLS:
        targets[col] = np.array([
            cand_indexed.loc[rel.split("|"), col].mean() for rel in query_df["relevant_pattern_ids"]
        ])
    return targets


def oof_regressor_estimates(sub_df, targets, n_inner=N_INNER_FOLDS):
    """Leave-one-inner-fold-out regressor predictions for every row of sub_df --
    safe to use as a training feature for rows in sub_df itself, since no row's
    prediction ever came from a model that saw that row's own pseudo target."""
    X = sub_df[REG_INPUT_COLS]
    n = len(sub_df)
    oof = {col: np.zeros(n) for col in REG_TARGET_COLS}
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=SEED)
    for inner_tr, inner_ho in kf.split(np.arange(n)):
        for col in REG_TARGET_COLS:
            model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
            model.fit(X.iloc[inner_tr], targets[col][inner_tr])
            oof[col][inner_ho] = model.predict(X.iloc[inner_ho])
    return oof


def regressor_estimates_for(tr_df, targets, va_df):
    """Fit on all of tr_df, predict va_df -- safe whether va_df is a held-out CV
    fold or the real test set, since neither overlaps tr_df."""
    X_tr = tr_df[REG_INPUT_COLS]
    X_va = va_df[REG_INPUT_COLS]
    preds = {}
    for col in REG_TARGET_COLS:
        model = lgb.LGBMRegressor(**REGRESSOR_PARAMS)
        model.fit(X_tr, targets[col])
        preds[col] = model.predict(X_va)
    return preds


# A second, differently-biased regressor (round 6): LGBM trees split on axis-aligned
# thresholds and can't extrapolate a smooth trend across the hidden gap, while a
# linear model can. Fed as a SEPARATE pair-feature block (not blended with the LGBM
# regressor's score) so the ranker learns how to weight/combine them itself -- this
# succeeds where every attempt to blend two FINAL ranking scores failed (see readme):
# here the ranker's own boosting does the combining, not a fixed external blend.
RIDGE_ALPHA = 3.0


def oof_ridge_estimates(sub_df, targets, n_inner=N_INNER_FOLDS):
    X_raw = sub_df[REG_INPUT_COLS].to_numpy(dtype=np.float64)
    n = len(sub_df)
    oof = {col: np.zeros(n) for col in REG_TARGET_COLS}
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=SEED)
    for inner_tr, inner_ho in kf.split(np.arange(n)):
        scaler = StandardScaler().fit(X_raw[inner_tr])
        X_tr_s = scaler.transform(X_raw[inner_tr])
        X_ho_s = scaler.transform(X_raw[inner_ho])
        for col in REG_TARGET_COLS:
            model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
            model.fit(X_tr_s, targets[col][inner_tr])
            oof[col][inner_ho] = model.predict(X_ho_s)
    return oof


def ridge_estimates_for(tr_df, targets, va_df):
    X_tr_raw = tr_df[REG_INPUT_COLS].to_numpy(dtype=np.float64)
    X_va_raw = va_df[REG_INPUT_COLS].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_s = scaler.transform(X_tr_raw)
    X_va_s = scaler.transform(X_va_raw)
    preds = {}
    for col in REG_TARGET_COLS:
        model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        model.fit(X_tr_s, targets[col])
        preds[col] = model.predict(X_va_s)
    return preds


def build_regressor_pair_features(reg_estimates, cand_df, n_q):
    """reg_estimates: dict[col] -> array of length n_q. Returns a
    (n_q * n_c, 2 * len(REG_TARGET_COLS)) diff/absdiff pair-feature block,
    built the same way as the hand-crafted MATCHED_PAIRS above."""
    n_c = len(cand_df)
    est = np.stack([reg_estimates[c] for c in REG_TARGET_COLS], axis=1)
    cand_vals = cand_df[REG_TARGET_COLS].to_numpy(dtype=np.float64)
    diff = est[:, None, :] - cand_vals[None, :, :]
    absdiff = np.abs(diff)
    return np.concatenate(
        [diff.reshape(n_q * n_c, -1), absdiff.reshape(n_q * n_c, -1)], axis=1
    ).astype(np.float32)


REG_FEATURE_NAMES = (
    [f"reg_diff_{c}" for c in REG_TARGET_COLS]
    + [f"reg_absdiff_{c}" for c in REG_TARGET_COLS]
)
RIDGE_FEATURE_NAMES = (
    [f"ridge_diff_{c}" for c in REG_TARGET_COLS]
    + [f"ridge_absdiff_{c}" for c in REG_TARGET_COLS]
)
FEATURE_NAMES = FEATURE_NAMES + REG_FEATURE_NAMES + RIDGE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Frequency penalty (post-processing)
#
# Frequency-balanced MAP@5 weights every distinct target pattern equally
# regardless of how often it's a training answer, but the ranker still sees
# popular candidates as positive labels more often during training. A small
# log-frequency penalty applied to scores (not seen during training, so the
# model itself can't learn to game it) nudges ties toward less-common
# candidates. Re-tuned after adding the regressor features: the optimum
# shifted down from ~0.2-0.25 (pre-regressor) to ~0.07 (post-regressor, CV
# grid over 0.0-0.4), since the regressor already fixes most of the
# popularity-bias problem this penalty was originally correcting for -- the
# remaining effect is small (~+0.001 CV) but real and never hurts nearby.
# ---------------------------------------------------------------------------

def compute_candidate_freq(query_df, cand_df):
    n_c = len(cand_df)
    R = build_labels(query_df, cand_df).reshape(len(query_df), n_c)
    return R.sum(axis=0)


def apply_frequency_penalty(scores, freq, n_q, n_c, lam=FREQ_PENALTY_LAMBDA):
    penalty = lam * np.log1p(freq)
    return (scores.reshape(n_q, n_c) - penalty[None, :]).reshape(-1)


# ---------------------------------------------------------------------------
# Scoring / ranking helpers
# ---------------------------------------------------------------------------

def to_feature_df(X):
    """Wrap the raw numpy pair matrix with named columns for LightGBM (avoids a
    spurious sklearn feature-name mismatch warning and gives readable importances)."""
    return pd.DataFrame(X, columns=FEATURE_NAMES, copy=False)


def rank_top5(scores, n_q, n_c, cand_ids):
    scores2d = scores.reshape(n_q, n_c)
    top_idx = np.argsort(-scores2d, axis=1)[:, :5]
    return cand_ids[top_idx]


def make_baseline_scorer(cand_df):
    """Fixed-weight nearest-neighbor baseline: standardized absolute differences on the
    matched physical features, minus small bonuses for band/circular similarity."""
    cand_std = {}
    for qcol, ccol in zip(QUERY_MATCHED_COLS, CAND_MATCHED_COLS):
        std = float(cand_df[ccol].std())
        cand_std[qcol] = std if std > 1e-6 else 1.0

    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def score(X):
        z = np.zeros(X.shape[0], dtype=np.float32)
        for qcol, ccol in MATCHED_PAIRS:
            z += X[:, idx[f"absdiff_{qcol}__{ccol}"]] / cand_std[qcol]
        bonus = (
            1.0 * X[:, idx["season_sim"]]
            + 1.0 * X[:, idx["hour_sim"]]
            + 0.5 * X[:, idx["swell_eq"]]
            + 0.5 * X[:, idx["wind_eq"]]
            + 0.5 * X[:, idx["wtemp_eq"]]
        )
        return -z + bonus

    return score


def make_lgbm_ranker(random_state=SEED):
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_ensemble(X_tr, y_tr, group_tr, X_va, seeds=ENSEMBLE_SEEDS):
    """Average predictions over a few LGBMRanker seeds -- cheap variance reduction."""
    X_tr_df = to_feature_df(X_tr)
    X_va_df = to_feature_df(X_va)
    scores = np.zeros(X_va.shape[0], dtype=np.float64)
    last_model = None
    for seed in seeds:
        model = make_lgbm_ranker(random_state=seed)
        model.fit(X_tr_df, y_tr, group=group_tr)
        scores += model.predict(X_va_df)
        last_model = model
    return scores / len(seeds), last_model


# ---------------------------------------------------------------------------
# Official metric: frequency-balanced MAP@5
# ---------------------------------------------------------------------------

def average_precision_at_5(rec_list, relevant_set):
    hits = 0
    total = 0.0
    for i, rec in enumerate(rec_list[:5]):
        if rec in relevant_set:
            hits += 1
            total += hits / (i + 1)
    return total / 5.0


def frequency_balanced_map5(rec_lists, relevant_lists):
    ap_per_row = [average_precision_at_5(rec_lists[i], relevant_lists[i]) for i in range(len(rec_lists))]
    pattern_scores = defaultdict(list)
    for i, relevant in enumerate(relevant_lists):
        for p in relevant:
            pattern_scores[p].append(ap_per_row[i])
    pattern_avgs = [np.mean(v) for v in pattern_scores.values()]
    return float(np.mean(pattern_avgs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_full_features(tr_q, va_q, cand_df, cand_indexed):
    """Everything the ranker sees for one (tr_q -> va_q) direction: hand-crafted
    pair features, k-NN votes, and OOF/full-fit regressor pair features. Used
    identically for CV folds (tr_q=fold train, va_q=fold val) and for the final
    model (tr_q=full train_df, va_q=test_df)."""
    n_c = len(cand_df)

    X_tr = build_pair_matrix(tr_q, cand_df)
    X_va = build_pair_matrix(va_q, cand_df)

    vote_tr, vote_va = knn_vote_features(tr_q, va_q, cand_df)

    targets_tr = pseudo_targets(tr_q, cand_indexed)
    oof_tr = oof_regressor_estimates(tr_q, targets_tr)
    reg_va = regressor_estimates_for(tr_q, targets_tr, va_q)
    reg_feats_tr = build_regressor_pair_features(oof_tr, cand_df, len(tr_q))
    reg_feats_va = build_regressor_pair_features(reg_va, cand_df, len(va_q))

    oof_ridge_tr = oof_ridge_estimates(tr_q, targets_tr)
    ridge_va = ridge_estimates_for(tr_q, targets_tr, va_q)
    ridge_feats_tr = build_regressor_pair_features(oof_ridge_tr, cand_df, len(tr_q))
    ridge_feats_va = build_regressor_pair_features(ridge_va, cand_df, len(va_q))

    X_tr = np.concatenate([X_tr, vote_tr, reg_feats_tr, ridge_feats_tr], axis=1)
    X_va = np.concatenate([X_va, vote_va, reg_feats_va, ridge_feats_va], axis=1)
    return X_tr, X_va


def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    cand_df = pd.read_csv(os.path.join(DATA_DIR, "candidate_patterns.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    train_df = add_query_features(train_df)
    test_df = add_query_features(test_df)
    cand_df = add_candidate_features(cand_df)
    cand_indexed = cand_df.set_index("pattern_id")

    n_c = len(cand_df)
    cand_ids = cand_df["pattern_id"].to_numpy()

    baseline_scorer = make_baseline_scorer(cand_df)

    # ---- Cross-validation: compare baseline vs. improved (regressor + kNN + LGBM) ----
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    query_index = np.arange(len(train_df))

    fold_scores = {"baseline": [], "improved": []}

    print(f"Running {N_FOLDS}-fold CV (frequency-balanced MAP@5)...")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(query_index)):
        tr_q = train_df.iloc[tr_idx].reset_index(drop=True)
        va_q = train_df.iloc[va_idx].reset_index(drop=True)

        X_tr, X_va = build_full_features(tr_q, va_q, cand_df, cand_indexed)
        y_tr = build_labels(tr_q, cand_df)
        group_tr = [n_c] * len(tr_q)
        freq = compute_candidate_freq(tr_q, cand_df)

        base_scores_va = baseline_scorer(X_va)
        base_top5 = rank_top5(base_scores_va, len(va_q), n_c, cand_ids)

        # single-seed for CV speed; the final model below uses a multi-seed ensemble
        improved_scores_va, _ = fit_predict_ensemble(X_tr, y_tr, group_tr, X_va, seeds=(SEED,))
        improved_scores_va = apply_frequency_penalty(improved_scores_va, freq, len(va_q), n_c)
        improved_top5 = rank_top5(improved_scores_va, len(va_q), n_c, cand_ids)

        relevant_lists = [set(s.split("|")) for s in va_q["relevant_pattern_ids"]]

        base_map5 = frequency_balanced_map5(base_top5.tolist(), relevant_lists)
        improved_map5 = frequency_balanced_map5(improved_top5.tolist(), relevant_lists)

        fold_scores["baseline"].append(base_map5)
        fold_scores["improved"].append(improved_map5)

        print(f"  fold {fold}: baseline={base_map5:.4f}  improved={improved_map5:.4f}")

        del X_tr, X_va, y_tr

    cv_summary = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    print("CV mean frequency-balanced MAP@5:")
    for k, v in cv_summary.items():
        print(f"  {k}: {v:.4f}")

    best_method = max(cv_summary, key=cv_summary.get)
    print(f"Best method by CV: {best_method} ({cv_summary[best_method]:.4f})")

    # ---- Retrain on full training data, predict on test ----
    X_train_full, X_test = build_full_features(train_df, test_df, cand_df, cand_indexed)
    y_train_full = build_labels(train_df, cand_df)
    freq_full = compute_candidate_freq(train_df, cand_df)

    base_scores_test = baseline_scorer(X_test)

    improved_scores_test, final_model = fit_predict_ensemble(
        X_train_full, y_train_full, [n_c] * len(train_df), X_test, seeds=ENSEMBLE_SEEDS
    )
    improved_scores_test = apply_frequency_penalty(improved_scores_test, freq_full, len(test_df), n_c)

    importances = pd.Series(final_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("Top 15 LightGBM feature importances (gain-based split count, last ensemble seed):")
    print(importances.head(15).to_string())

    final_scores_test = base_scores_test if best_method == "baseline" else improved_scores_test

    top5_test = rank_top5(final_scores_test, len(test_df), n_c, cand_ids)

    submission = pd.DataFrame({
        "query_id": test_df["query_id"].to_numpy(),
        "rec_1": top5_test[:, 0],
        "rec_2": top5_test[:, 1],
        "rec_3": top5_test[:, 2],
        "rec_4": top5_test[:, 3],
        "rec_5": top5_test[:, 4],
    })

    # Align exactly to sample_submission's query_id order.
    submission = sample_sub[["query_id"]].merge(submission, on="query_id", how="left")

    os.makedirs(WORKING_DIR, exist_ok=True)
    out_path = os.path.join(WORKING_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(submission)} rows) using method='{best_method}'")


if __name__ == "__main__":
    main()
