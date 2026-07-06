"""
Chess move-prefix -> outcome distribution prediction.

Trains on dataset/public/train.csv (visible SAN move-prefix fields + empirical
cohort outcome rates) and predicts a calibrated 3-class outcome distribution
(white win / draw / black win) plus a confidence score for every row of
dataset/public/test.csv. Writes working/submission.csv.

All modeling is fit from scratch on the public training file using only the
visible prefix fields (move_prefix, prefix_ply_count, side_to_move) as model
inputs. No external game/engine lookups, no id/row-order signal.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "public", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "public", "test.csv")
OUT_DIR = os.path.join(BASE_DIR, "working")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

CLASSES = ["white_win_rate", "draw_rate", "black_win_rate"]
EPS = 1e-4

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# --------------------------------------------------------------------------
# Tokenization / handcrafted features
# --------------------------------------------------------------------------
def tokenize(move_prefix):
    return move_prefix.split()


def piece_type(tok):
    if tok.startswith("O-O"):
        return "K"
    c = tok[0]
    if c in "NBRQK":
        return c
    return "P"


def handcrafted_features(move_prefix):
    toks = tokenize(move_prefix)
    n = len(toks)
    feat = {}

    w_caps = b_caps = 0
    w_checks = b_checks = 0
    w_castle_k = w_castle_q = b_castle_k = b_castle_q = 0
    w_piece_counts = {"N": 0, "B": 0, "R": 0, "Q": 0, "K": 0, "P": 0}
    b_piece_counts = {"N": 0, "B": 0, "R": 0, "Q": 0, "K": 0, "P": 0}

    for i, tok in enumerate(toks):
        is_white = (i % 2 == 0)
        is_cap = "x" in tok
        is_check = ("+" in tok) or ("#" in tok)
        pt = piece_type(tok)
        if is_white:
            w_piece_counts[pt] += 1
            if is_cap:
                w_caps += 1
            if is_check:
                w_checks += 1
            if tok.startswith("O-O-O"):
                w_castle_q += 1
            elif tok.startswith("O-O"):
                w_castle_k += 1
        else:
            b_piece_counts[pt] += 1
            if is_cap:
                b_caps += 1
            if is_check:
                b_checks += 1
            if tok.startswith("O-O-O"):
                b_castle_q += 1
            elif tok.startswith("O-O"):
                b_castle_k += 1

    feat["w_caps"] = w_caps
    feat["b_caps"] = b_caps
    feat["w_checks"] = w_checks
    feat["b_checks"] = b_checks
    feat["w_caps_rate"] = w_caps / n if n else 0.0
    feat["b_caps_rate"] = b_caps / n if n else 0.0
    feat["w_checks_rate"] = w_checks / n if n else 0.0
    feat["b_checks_rate"] = b_checks / n if n else 0.0
    feat["w_castle_k"] = w_castle_k
    feat["w_castle_q"] = w_castle_q
    feat["b_castle_k"] = b_castle_k
    feat["b_castle_q"] = b_castle_q
    for p in "NBRQK":
        feat[f"w_{p}_moves"] = w_piece_counts[p]
        feat[f"b_{p}_moves"] = b_piece_counts[p]
    feat["w_pawn_moves"] = w_piece_counts["P"]
    feat["b_pawn_moves"] = b_piece_counts["P"]

    last = toks[-1] if toks else ""
    last_pt = piece_type(last) if last else "P"
    for p in "NBRQKP":
        feat[f"last_is_{p}"] = int(last_pt == p)
    feat["last_is_capture"] = int("x" in last) if last else 0
    feat["last_is_check"] = int(("+" in last) or ("#" in last)) if last else 0

    feat["n_tokens"] = n
    return feat


def build_handcrafted_matrix(move_prefixes):
    rows = [handcrafted_features(mp) for mp in move_prefixes]
    return pd.DataFrame(rows)


def first_move_of(move_prefix):
    return move_prefix.split()[0] if move_prefix.strip() else ""


def first4_of(move_prefix):
    return " ".join(move_prefix.split()[:4])


# --------------------------------------------------------------------------
# Hierarchical empirical-Bayes backoff priors (global -> first-move -> first-4)
# --------------------------------------------------------------------------
def fit_hierarchical_priors(df, k_first_move=1000.0, k_first4=500.0):
    rates = df[CLASSES].values
    weights = df["cohort_game_count"].values.astype(float)

    global_prior = np.average(rates, axis=0, weights=weights)

    fm = df["move_prefix"].apply(first_move_of)
    first_move_stats = {}
    for move, idx in fm.groupby(fm).groups.items():
        w = weights[df.index.get_indexer(idx)]
        r = rates[df.index.get_indexer(idx)]
        n_b = w.sum()
        raw = np.average(r, axis=0, weights=w)
        shrunk = (raw * n_b + global_prior * k_first_move) / (n_b + k_first_move)
        first_move_stats[move] = shrunk

    f4 = df["move_prefix"].apply(first4_of)
    first4_stats = {}
    for key, idx in f4.groupby(f4).groups.items():
        w = weights[df.index.get_indexer(idx)]
        r = rates[df.index.get_indexer(idx)]
        n_b = w.sum()
        raw = np.average(r, axis=0, weights=w)
        parent_move = key.split()[0] if key else ""
        parent = first_move_stats.get(parent_move, global_prior)
        shrunk = (raw * n_b + parent * k_first4) / (n_b + k_first4)
        first4_stats[key] = shrunk

    def predict_fn(move_prefix_series):
        n = len(move_prefix_series)
        out_fm = np.zeros((n, 3))
        out_f4 = np.zeros((n, 3))
        for i, mp in enumerate(move_prefix_series):
            m = first_move_of(mp)
            fm_prior = first_move_stats.get(m, global_prior)
            out_fm[i] = fm_prior
            key = first4_of(mp)
            out_f4[i] = first4_stats.get(key, fm_prior)
        return out_fm, out_f4

    return {
        "global_prior": global_prior,
        "first_move_stats": first_move_stats,
        "first4_stats": first4_stats,
        "predict_fn": predict_fn,
    }


def baseline_a_predict(priors, move_prefix_series):
    _, f4 = priors["predict_fn"](move_prefix_series)
    p = np.clip(f4, EPS, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


# --------------------------------------------------------------------------
# Categorical opening buckets (top-K + other), one-hot
# --------------------------------------------------------------------------
def build_bucket_encoder(values, top_k):
    counts = pd.Series(values).value_counts()
    keep = set(counts.index[:top_k])

    def encode(vals):
        return pd.Series([v if v in keep else "__other__" for v in vals])

    categories = sorted(keep) + ["__other__"]
    return encode, categories


def one_hot(series, categories):
    cat = pd.Categorical(series, categories=categories)
    return pd.get_dummies(cat)


# --------------------------------------------------------------------------
# Feature matrix assembly
# --------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, min_df=5, top_first_move=6, top_first4=60):
        self.min_df = min_df
        self.top_first_move = top_first_move
        self.top_first4 = top_first4

    def fit(self, df, priors):
        self.priors = priors
        
        # CountVectorizer yerine TfidfVectorizer kullanıyoruz, n-gram aralığı (1, 5) yapıldı
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, 
            token_pattern=None, 
            lowercase=False,
            min_df=self.min_df, 
            ngram_range=(1, 5),
            sublinear_tf=True
        )
        self.vectorizer.fit(df["move_prefix"])

        first_moves = df["move_prefix"].apply(first_move_of)
        first4s = df["move_prefix"].apply(first4_of)
        self.fm_encode, self.fm_categories = build_bucket_encoder(first_moves, self.top_first_move)
        self.f4_encode, self.f4_categories = build_bucket_encoder(first4s, self.top_first4)

        hc = build_handcrafted_matrix(df["move_prefix"])
        hc["prefix_ply_count"] = df["prefix_ply_count"].values
        # side_to_move bilgisini sayısal (0 ve 1) olarak dahil ediyoruz
        hc["is_white_to_move"] = (df["side_to_move"].str.lower() == "white").astype(int)
        
        self.handcrafted_columns = list(hc.columns)

        fm_oh = one_hot(self.fm_encode(first_moves), self.fm_categories)
        f4_oh = one_hot(self.f4_encode(first4s), self.f4_categories)
        fm_prior, f4_prior = priors["predict_fn"](df["move_prefix"])

        dense = pd.concat(
            [hc.reset_index(drop=True), fm_oh.reset_index(drop=True), f4_oh.reset_index(drop=True)],
            axis=1,
        )
        dense["fm_prior_w"] = fm_prior[:, 0]
        dense["fm_prior_d"] = fm_prior[:, 1]
        dense["fm_prior_b"] = fm_prior[:, 2]
        dense["f4_prior_w"] = f4_prior[:, 0]
        dense["f4_prior_d"] = f4_prior[:, 1]
        dense["f4_prior_b"] = f4_prior[:, 2]

        self.dense_columns = list(dense.columns)
        self.scaler = StandardScaler()
        dense_scaled = self.scaler.fit_transform(dense.values.astype(float))

        ngram = self.vectorizer.transform(df["move_prefix"])
        X_linear = sparse.hstack([sparse.csr_matrix(dense_scaled), ngram], format="csr")
        X_tree = np.hstack([dense.values.astype(float), ngram.toarray().astype(float)])
        return X_linear, X_tree

    def transform(self, df):
        first_moves = df["move_prefix"].apply(first_move_of)
        first4s = df["move_prefix"].apply(first4_of)

        hc = build_handcrafted_matrix(df["move_prefix"])
        hc["prefix_ply_count"] = df["prefix_ply_count"].values
        hc["is_white_to_move"] = (df["side_to_move"].str.lower() == "white").astype(int)

        fm_oh = one_hot(self.fm_encode(first_moves), self.fm_categories)
        f4_oh = one_hot(self.f4_encode(first4s), self.f4_categories)
        fm_prior, f4_prior = self.priors["predict_fn"](df["move_prefix"])

        dense = pd.concat(
            [hc.reset_index(drop=True), fm_oh.reset_index(drop=True), f4_oh.reset_index(drop=True)],
            axis=1,
        )
        dense["fm_prior_w"] = fm_prior[:, 0]
        dense["fm_prior_d"] = fm_prior[:, 1]
        dense["fm_prior_b"] = fm_prior[:, 2]
        dense["f4_prior_w"] = f4_prior[:, 0]
        dense["f4_prior_d"] = f4_prior[:, 1]
        dense["f4_prior_b"] = f4_prior[:, 2]
        dense = dense.reindex(columns=self.dense_columns, fill_value=0)

        dense_scaled = self.scaler.transform(dense.values.astype(float))
        ngram = self.vectorizer.transform(df["move_prefix"])
        X_linear = sparse.hstack([sparse.csr_matrix(dense_scaled), ngram], format="csr")
        X_tree = np.hstack([dense.values.astype(float), ngram.toarray().astype(float)])
        return X_linear, X_tree


# --------------------------------------------------------------------------
# Model B: multinomial logistic regression via sample-expansion trick
# --------------------------------------------------------------------------
def fit_predict_logreg(X_train, y_train_rates, w_train, X_val, C=0.3):
    X_list, y_list, w_list = [], [], []
    for c in range(3):
        w_c = y_train_rates[:, c] * w_train
        mask = w_c > 1e-9
        if not mask.any():
            continue
        X_list.append(X_train[mask])
        y_list.append(np.full(int(mask.sum()), c))
        w_list.append(w_c[mask])

    X_exp = sparse.vstack(X_list, format="csr")
    y_exp = np.concatenate(y_list)
    w_exp = np.concatenate(w_list)

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", C=C, max_iter=3000, random_state=SEED,
    )
    clf.fit(X_exp, y_exp, sample_weight=w_exp)

    proba = np.zeros((X_val.shape[0], 3))
    proba[:, clf.classes_] = clf.predict_proba(X_val)
    return proba, clf


# --------------------------------------------------------------------------
# Model C: tree-based per-class regression (LightGBM or HGBR fallback)
# --------------------------------------------------------------------------
def fit_predict_gbm(X_train, y_train_rates, w_train, X_val):
    preds = np.zeros((X_val.shape[0], 3))
    models = []
    for c in range(3):
        target = y_train_rates[:, c]
        if HAS_LGB:
            # Model kapasitesi önemli ölçüde artırıldı (n_estimators, num_leaves, max_depth)
            model = lgb.LGBMRegressor(
                n_estimators=400,        
                num_leaves=63,           
                max_depth=7,             
                learning_rate=0.05,      
                min_child_samples=15,
                subsample=0.8, 
                colsample_bytree=0.8,
                reg_alpha=0.2, 
                reg_lambda=1.0, 
                random_state=SEED, 
                verbose=-1,
            )
            model.fit(X_train, target, sample_weight=w_train)
        else:
            model = HistGradientBoostingRegressor(
                max_depth=7, max_iter=400, learning_rate=0.05,
                min_samples_leaf=15, l2_regularization=1.0, random_state=SEED,
            )
            model.fit(X_train, target, sample_weight=w_train)
        preds[:, c] = model.predict(X_val)
        models.append(model)
    preds = np.clip(preds, EPS, None)
    preds = preds / preds.sum(axis=1, keepdims=True)
    return preds, models


# --------------------------------------------------------------------------
# Model D: CatBoost per-class regression (ensemble diversity; optional)
# --------------------------------------------------------------------------
def fit_predict_catboost(X_train, y_train_rates, w_train, X_val):
    preds = np.zeros((X_val.shape[0], 3))
    models = []
    for c in range(3):
        model = CatBoostRegressor(
            depth=8, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0,
            loss_function="RMSE", random_seed=SEED, verbose=False,
        )
        model.fit(X_train, y_train_rates[:, c], sample_weight=w_train)
        preds[:, c] = model.predict(X_val)
        models.append(model)
    preds = np.clip(preds, EPS, None)
    preds = preds / preds.sum(axis=1, keepdims=True)
    return preds, models


# --------------------------------------------------------------------------
# Scoring (mirrors the grader's per-row Brier / KL-log / confidence terms)
# --------------------------------------------------------------------------
def row_brier(y_true, p):
    return np.sum((p - y_true) ** 2, axis=1)


def row_kl(y_true, p):
    p_safe = np.clip(p, 1e-9, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(np.where(y_true > 0, y_true, 1.0) / p_safe)
    terms = np.where(y_true > 0, y_true * log_ratio, 0.0)
    return np.sum(terms, axis=1)


def skill(loss, ref_loss):
    ref_mean = ref_loss.mean()
    if ref_mean <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - loss.mean() / ref_mean, 0.0, 1.0))


def composite_proxy(y_true, p, conf, ply_groups, baseline_p):
    brier = row_brier(y_true, p)
    kl = row_kl(y_true, p)
    conf_err = np.abs(conf - y_true.max(axis=1))

    base_brier = row_brier(y_true, baseline_p)
    base_kl = row_kl(y_true, baseline_p)
    base_conf_err = np.abs(baseline_p.max(axis=1) - y_true.max(axis=1))

    s_brier = skill(brier, base_brier)
    s_log = skill(kl, base_kl)
    s_conf = skill(conf_err, base_conf_err)

    worst = 1.0
    for g in np.unique(ply_groups):
        mask = ply_groups == g
        if mask.sum() == 0:
            continue
        worst = min(worst, skill(brier[mask], base_brier[mask]))
    s_worst = worst

    composite = 0.55 * s_brier + 0.20 * s_log + 0.10 * s_conf + 0.15 * s_worst
    return {
        "s_brier": s_brier, "s_log": s_log, "s_conf": s_conf, "s_worst": s_worst,
        "composite": composite,
    }


def fit_perclass_blend_weights(y_true, model_preds):
    weights = []
    for c in range(3):
        M = np.column_stack([p[:, c] for p in model_preds])
        w, _ = nnls(M, y_true[:, c])
        weights.append(w)
    return weights


def apply_perclass_blend(model_preds, weights):
    n = model_preds[0].shape[0]
    out = np.zeros((n, 3))
    for c in range(3):
        M = np.column_stack([p[:, c] for p in model_preds])
        out[:, c] = M @ weights[c]
    return out


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y_all = train[CLASSES].values
    w_all_raw = train["cohort_game_count"].values.astype(float)
    ply_all = train["prefix_ply_count"].values

    weight_fns = {
        "sqrt": lambda n: np.sqrt(n),
        "log1p": lambda n: np.log1p(n),
    }

    use_D = HAS_CATBOOST

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(train, ply_all))

    oof_preds = {name: np.zeros((len(train), 3)) for name in ["A", "B", "C", "D"]}

    best_wfn_name = "sqrt"
    best_wfn_score = -np.inf

    for wfn_name, wfn in weight_fns.items():
        oof_B = np.zeros((len(train), 3))
        oof_C = np.zeros((len(train), 3))
        oof_A = np.zeros((len(train), 3))
        oof_D = np.zeros((len(train), 3))
        for tr_idx, va_idx in folds:
            df_tr = train.iloc[tr_idx].reset_index(drop=True)
            df_va = train.iloc[va_idx].reset_index(drop=True)

            priors = fit_hierarchical_priors(df_tr)
            fb = FeatureBuilder()
            X_lin_tr, X_tree_tr = fb.fit(df_tr, priors)
            X_lin_va, X_tree_va = fb.transform(df_va)

            w_tr = wfn(df_tr["cohort_game_count"].values.astype(float))

            oof_A[va_idx] = baseline_a_predict(priors, df_va["move_prefix"])
            p_B, _ = fit_predict_logreg(X_lin_tr, df_tr[CLASSES].values, w_tr, X_lin_va)
            oof_B[va_idx] = p_B
            p_C, _ = fit_predict_gbm(X_tree_tr, df_tr[CLASSES].values, w_tr, X_tree_va)
            oof_C[va_idx] = p_C
            if use_D:
                p_D, _ = fit_predict_catboost(X_tree_tr, df_tr[CLASSES].values, w_tr, X_tree_va)
                oof_D[va_idx] = p_D

        base_p = np.tile(np.average(y_all, axis=0, weights=w_all_raw), (len(train), 1))
        model_preds = [oof_A, oof_B, oof_C, oof_D] if use_D else [oof_A, oof_B, oof_C]
        perclass_w = fit_perclass_blend_weights(y_all, model_preds)
        blend = apply_perclass_blend(model_preds, perclass_w)
        blend = np.clip(blend, EPS, None)
        blend = blend / blend.sum(axis=1, keepdims=True)
        wfn_score = composite_proxy(y_all, blend, blend.max(axis=1), ply_all, base_p)["composite"]

        if wfn_score > best_wfn_score:
            best_wfn_score = wfn_score
            best_wfn_name = wfn_name
            oof_preds["A"], oof_preds["B"], oof_preds["C"], oof_preds["D"] = oof_A, oof_B, oof_C, oof_D

    oof_A, oof_B, oof_C, oof_D = oof_preds["A"], oof_preds["B"], oof_preds["C"], oof_preds["D"]
    base_p = np.tile(np.average(y_all, axis=0, weights=w_all_raw), (len(train), 1))

    standalone = [("A (empirical-Bayes prior)", oof_A),
                  ("B (logreg, sample-expansion)", oof_B),
                  ("C (GBM per-class)", oof_C)]
    if use_D:
        standalone.append(("D (CatBoost per-class)", oof_D))
    for name, p in standalone:
        p_c = np.clip(p, EPS, None)
        p_c = p_c / p_c.sum(axis=1, keepdims=True)
        r = composite_proxy(y_all, p_c, p_c.max(axis=1), ply_all, base_p)
        print(f"[CV] standalone {name}: composite={r['composite']:.4f} "
              f"s_brier={r['s_brier']:.4f} s_log={r['s_log']:.4f} s_worst={r['s_worst']:.4f}")

    model_preds = [oof_A, oof_B, oof_C, oof_D] if use_D else [oof_A, oof_B, oof_C]
    perclass_weights = fit_perclass_blend_weights(y_all, model_preds)
    oof_blend = apply_perclass_blend(model_preds, perclass_weights)
    oof_blend = np.clip(oof_blend, EPS, None)
    oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True)
    oof_conf_raw = oof_blend.max(axis=1)
    oof_conf_true = y_all.max(axis=1)
    best_detail = composite_proxy(y_all, oof_blend, oof_conf_raw, ply_all, base_p)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1 / 3, y_max=1.0)
    iso.fit(oof_conf_raw, oof_conf_true)
    calibrated = iso.predict(oof_conf_raw)

    mae_raw = np.abs(oof_conf_raw - oof_conf_true).mean()
    mae_cal = np.abs(calibrated - oof_conf_true).mean()
    use_isotonic = mae_cal < mae_raw

    model_names = "A=prior, B=logreg, C=gbm, D=catboost" if use_D else "A=prior, B=logreg, C=gbm"
    print(f"[CV] chosen cohort-weight transform: {best_wfn_name}")
    print(f"[CV] per-class NNLS blend weights ({model_names}):")
    for cls_name, w in zip(CLASSES, perclass_weights):
        print(f"       {cls_name}: {np.round(w, 3).tolist()}")
    print(f"[CV] proxy composite={best_detail['composite']:.4f} "
          f"s_brier={best_detail['s_brier']:.4f} s_log={best_detail['s_log']:.4f} "
          f"s_conf={best_detail['s_conf']:.4f} s_worst={best_detail['s_worst']:.4f}")
    print(f"[CV] confidence MAE raw={mae_raw:.4f} isotonic={mae_cal:.4f} "
          f"-> using {'isotonic' if use_isotonic else 'raw max(p)'}")

    # ---------------- Final fit on full training data ----------------
    wfn = weight_fns[best_wfn_name]
    priors_full = fit_hierarchical_priors(train)
    fb_full = FeatureBuilder()
    X_lin_full, X_tree_full = fb_full.fit(train, priors_full)
    w_full = wfn(train["cohort_game_count"].values.astype(float))

    X_lin_test, X_tree_test = fb_full.transform(test)

    p_A_test = baseline_a_predict(priors_full, test["move_prefix"])
    p_B_test, _ = fit_predict_logreg(X_lin_full, train[CLASSES].values, w_full, X_lin_test)
    p_C_test, _ = fit_predict_gbm(X_tree_full, train[CLASSES].values, w_full, X_tree_test)
    if use_D:
        p_D_test, _ = fit_predict_catboost(X_tree_full, train[CLASSES].values, w_full, X_tree_test)
        test_model_preds = [p_A_test, p_B_test, p_C_test, p_D_test]
    else:
        test_model_preds = [p_A_test, p_B_test, p_C_test]

    p_test = apply_perclass_blend(test_model_preds, perclass_weights)
    p_test = np.clip(p_test, EPS, None)
    p_test = p_test / p_test.sum(axis=1, keepdims=True)

    conf_test = p_test.max(axis=1)
    if use_isotonic:
        conf_test = iso.predict(conf_test)
    conf_test = np.clip(conf_test, 0.0, 1.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = pd.DataFrame({
        "id": test["id"].values,
        "white_win_prob": p_test[:, 0],
        "draw_prob": p_test[:, 1],
        "black_win_prob": p_test[:, 2],
        "confidence": conf_test,
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"[DONE] wrote {len(out)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()