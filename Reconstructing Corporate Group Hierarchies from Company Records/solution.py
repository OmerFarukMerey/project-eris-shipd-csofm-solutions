#!/usr/bin/env python3
"""Reconstructing corporate group hierarchies from company records.

Usage: python3 solution.py <public_dir> <submission_out>

Pipeline (everything is fit on training families only, test families are only transformed):
  1. Pairwise features for every (child i, candidate parent j) inside a family: name similarity
     (leading words, char 3-gram Jaccard, IDF-weighted token overlap, token containment), legal form /
     jurisdiction / city / country agreement, registration-date gaps, and family-relative ranks.
  2. Node-level text models (TF-IDF char n-grams + logistic regression) that learn from training names
     whether an entity has subsidiaries and whether it sits under an intermediate parent. Their
     out-of-fold scores become pair features.
  3. A LightGBM classifier (a bag of 3 seeds) scores every (child, candidate) pair.
  4. The scores are turned into edge marginals of a distribution over spanning trees rooted at the
     ultimate parent (Matrix-Tree theorem), and each child gets its best non-root candidate whenever
     that marginal clears a threshold.
  The number of boosting rounds, the decode temperature and the threshold are all searched in-script
  on out-of-fold predictions from a family-grouped 5-fold CV of the full pipeline (every transform
  re-fit inside each fold). Test pairs are scored by the average of the 5 fold models (3 seeds each),
  each applied to test features built with that fold's own transforms.
"""
import math
import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 0
N_THREADS = 4                       # performance knob only (LightGBM runs with deterministic=True)
N_FOLDS = 5                         # family-grouped CV folds for in-script model / decode selection
INNER_FOLDS = 5                     # out-of-fold node text features inside every fit
GBM_BASE = dict(objective="binary", num_leaves=63, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                cat_smooth=10, min_data_per_group=20, verbose=-1, deterministic=True,
                force_row_wise=True, seed=SEED, num_threads=N_THREADS)
GBM_GRID = [dict(learning_rate=0.025, feature_fraction=0.4, min_data_in_leaf=40)]
N_SEEDS = 3                         # seed-bagged LightGBM per fit (OOF and test use the same bag)
MAX_ROUNDS = 1000
ROUND_CHECKPOINTS = [400, 600, 800, 1000]
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]           # temperature of the edge potentials
T_GRID = np.round(np.arange(0.02, 0.61, 0.01), 3)     # marginal threshold for emitting a link
CAT_COLS = ["form_i", "form_j", "jur_i", "jur_j", "form_r", "jur_r"]
META_COLS = {"cid", "pid", "family_id", "y"}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# text normalisation and entity preparation
# ----------------------------------------------------------------------------------------------
def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = re.sub(r"[\W_]+", " ", s)
    return s.strip()


def char_ngrams(s, n=3):
    s = " " + s + " "
    return {s[k:k + n] for k in range(max(1, len(s) - n + 1))}


def prep_entities(df):
    df = df.copy()
    df["nname"] = df.legal_name.map(norm_text)
    df["toks"] = df.nname.str.split()
    df["lcity"] = df.legal_city.map(norm_text)
    df["hcity"] = df.hq_city.map(norm_text)
    d = pd.to_datetime(df.registered_on, errors="coerce")
    df["dord"] = (d - pd.Timestamp("2000-01-01")).dt.days.astype(float)
    df["is_root"] = (df.is_ultimate_parent.astype(str).str.strip() == "1").astype(int)
    return df


def fit_idf(ents):
    """Token inverse document frequency, documents = families (fit on training families only)."""
    nf = ents.family_id.nunique()
    cnt = {}
    for _, toks in ents.groupby("family_id").toks:
        s = set()
        for t in toks:
            s.update(t)
        for t in s:
            cnt[t] = cnt.get(t, 0) + 1
    idf = {t: math.log((nf + 1) / (c + 1)) + 1.0 for t, c in cnt.items()}
    return idf, math.log(nf + 1) + 1.0


def _lead(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


# ----------------------------------------------------------------------------------------------
# pairwise features inside one family
# ----------------------------------------------------------------------------------------------
def family_pairs(g, idf, idf_def):
    n = len(g)
    ids = g.entity_id.values
    toks = list(g.toks.values)
    tsets = [set(t) for t in toks]
    names = list(g.nname.values)
    c3 = [char_ngrams(s) for s in names]
    idfw = [{t: idf.get(t, idf_def) for t in ts} for ts in tsets]
    # math.fsum: exact, iteration-order independent sums (set order varies with PYTHONHASHSEED)
    idfsum = [math.fsum(w.values()) + 1e-9 for w in idfw]
    form = g.legal_form.values
    jur = g.jurisdiction.values
    lc = g.lcity.values
    hc = g.hcity.values
    lco = g.legal_country.values
    hco = g.hq_country.values
    dord = g.dord.values.astype(float)
    isroot = g.is_root.values
    r = int(np.argmax(isroot))

    lead = np.zeros((n, n)); lcp = np.zeros((n, n)); jac = np.zeros((n, n)); c3j = np.zeros((n, n))
    idfo_i = np.zeros((n, n)); idfo_j = np.zeros((n, n)); jin = np.zeros((n, n)); maxidf = np.zeros((n, n))
    sub = np.zeros((n, n), bool)  # sub[a, b]: every token of b appears in a
    for a in range(n):
        for b in range(a + 1, n):
            lead[a, b] = lead[b, a] = _lead(toks[a], toks[b])
            lcp[a, b] = lcp[b, a] = _lead(names[a], names[b])
            inter = tsets[a] & tsets[b]
            u = len(tsets[a] | tsets[b])
            jac[a, b] = jac[b, a] = len(inter) / u if u else 0.0
            cu = len(c3[a] | c3[b])
            c3j[a, b] = c3j[b, a] = len(c3[a] & c3[b]) / cu if cu else 0.0
            wi = math.fsum(idfw[a][t] for t in inter)
            idfo_i[a, b] = wi / idfsum[a]; idfo_i[b, a] = wi / idfsum[b]
            idfo_j[a, b] = wi / idfsum[b]; idfo_j[b, a] = wi / idfsum[a]
            jin[a, b] = len(inter) / max(1, len(tsets[b])); jin[b, a] = len(inter) / max(1, len(tsets[a]))
            maxidf[a, b] = maxidf[b, a] = max((idfw[a][t] for t in inter), default=0.0)
            sub[a, b] = len(tsets[b]) > 0 and tsets[b] <= tsets[a]
            sub[b, a] = len(tsets[a]) > 0 and tsets[a] <= tsets[b]
    eq = lambda u, v: (u[:, None] == v[None, :]).astype(float)
    same_j = eq(jur, jur); same_lc = eq(lc, lc); same_hc = eq(hc, hc); hc_lc = eq(hc, lc)
    same_lco = eq(lco, lco); same_hco = eq(hco, hco); same_form = eq(form, form)
    ddiff = dord[:, None] - dord[None, :]

    vc = lambda v: pd.Series(v).map(pd.Series(v).value_counts()).values
    cnt_jur, cnt_date, cnt_lc, cnt_form = vc(jur), vc(dord), vc(lc), vc(form)
    date_rank = pd.Series(dord).rank(pct=True).values
    clus = (lead >= 1).sum(1)
    clus2 = (idfo_i > 0.5).sum(1)
    cont = sub.sum(0); ext = sub.sum(1)

    I, J = [], []
    for a in range(n):
        if isroot[a]:
            continue
        for b in range(n):
            if b != a:
                I.append(a); J.append(b)
    I = np.array(I, int); J = np.array(J, int)
    F = {"n": np.full(len(I), n), "j_is_root": isroot[J]}
    for nm, M in [("lead", lead), ("lcp", lcp), ("jac", jac), ("c3", c3j), ("idfo_i", idfo_i),
                  ("idfo_j", idfo_j), ("jin", jin), ("maxidf", maxidf)]:
        v = M[I, J]
        Mm = M.copy(); np.fill_diagonal(Mm, -1)
        best = Mm.max(1)
        rk = (Mm[:, None, :] > Mm[:, :, None]).sum(2) + 1  # 1 + #candidates strictly better (ties share)
        F[nm] = v
        F[nm + "_gap"] = v - best[I]
        F[nm + "_rk"] = rk[I, J]
        F[nm + "_iroot"] = M[I, r]
        F[nm + "_jroot"] = M[J, r]
        F[nm + "_revgap"] = v - best[J]
    F["ntok_i"] = np.array([len(toks[a]) for a in I]); F["ntok_j"] = np.array([len(toks[b]) for b in J])
    F["nch_i"] = np.array([len(names[a]) for a in I]); F["nch_j"] = np.array([len(names[b]) for b in J])
    for nm, M in [("same_jur", same_j), ("same_lc", same_lc), ("same_hc", same_hc), ("hc_lc", hc_lc),
                  ("same_lco", same_lco), ("same_hco", same_hco), ("same_form", same_form)]:
        F[nm] = M[I, J]
        F[nm + "_iroot"] = M[I, r]
        F[nm + "_jroot"] = M[J, r]
    F["lc_hc_rev"] = hc_lc[J, I]
    F["ddiff"] = ddiff[I, J]; F["absdd"] = np.abs(ddiff[I, J])
    F["dd_iroot"] = ddiff[I, r]; F["dd_jroot"] = ddiff[J, r]
    F["date_i"] = dord[I]; F["date_j"] = dord[J]
    F["drank_i"] = date_rank[I]; F["drank_j"] = date_rank[J]
    F["cnt_jur_i"] = cnt_jur[I]; F["cnt_jur_j"] = cnt_jur[J]
    F["cnt_date_i"] = cnt_date[I]; F["cnt_date_j"] = cnt_date[J]
    F["cnt_lc_i"] = cnt_lc[I]; F["cnt_lc_j"] = cnt_lc[J]
    F["cnt_form_i"] = cnt_form[I]; F["cnt_form_j"] = cnt_form[J]
    F["clus_j"] = clus[J]; F["clus_i"] = clus[I]; F["clus2_j"] = clus2[J]
    F["n_jur"] = np.full(len(I), len(set(jur)))
    F["cont_j"] = cont[J]; F["cont_i"] = cont[I]; F["ext_j"] = ext[J]; F["ext_i"] = ext[I]
    F["cont_frac_j"] = cont[J] / n; F["cont_frac_i"] = cont[I] / n
    out = pd.DataFrame(F)
    out["cid"] = ids[I]; out["pid"] = ids[J]
    out["form_i"] = form[I]; out["form_j"] = form[J]
    out["jur_i"] = jur[I]; out["jur_j"] = jur[J]
    out["form_r"] = form[r]; out["jur_r"] = jur[r]
    out["family_id"] = g.family_id.values[0]
    return out


def build_pairs(ents, idf, idf_def):
    parts = []
    for _, g in ents.groupby("family_id", sort=True):
        g = g.sort_values("entity_id").reset_index(drop=True)
        if g.is_root.sum() != 1 or len(g) < 2:
            continue  # malformed family; handled by the per-family fallback at prediction time
        parts.append(family_pairs(g, idf, idf_def))
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# node-level text models: P(entity has subsidiaries), P(entity sits under an intermediate parent)
# ----------------------------------------------------------------------------------------------
NODE_TARGETS = ["y_hc", "y_inter"]


def _side_doc(df):
    return ("F_" + df.legal_form.astype(str).str.replace(" ", "_") + " J_" + df.jurisdiction.astype(str)).values


class NodeText:
    def fit(self, df, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float64)
        self.v2 = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, sublinear_tf=True, dtype=np.float64)
        X = self._stack(self.v1.fit_transform(df.nname.values), self.v2.fit_transform(_side_doc(df)))
        self.m = LogisticRegression(C=1.0, max_iter=3000)
        self.m.fit(X, y)
        return self

    @staticmethod
    def _stack(a, b):
        from scipy import sparse
        return sparse.hstack([a, b]).tocsr()

    def predict(self, df):
        X = self._stack(self.v1.transform(df.nname.values), self.v2.transform(_side_doc(df)))
        return self.m.predict_proba(X)[:, 1]


def family_folds(fams, k, seed):
    fams = np.array(sorted(set(fams)))
    perm = np.random.RandomState(seed).permutation(len(fams))
    return {fams[p]: i % k for i, p in enumerate(perm)}


def node_scores(fit_ents, labels, apply_list):
    """Out-of-fold node scores for fit_ents (inner family folds) and full-model scores for every
    frame in apply_list (pure per-row inference)."""
    fit_nr = fit_ents[fit_ents.is_root == 0]
    app_nrs = [a[a.is_root == 0] for a in apply_list]
    fm = family_folds(fit_nr.family_id.unique(), INNER_FOLDS, SEED + 17)
    fold = fit_nr.family_id.map(fm).values
    out_fit = pd.DataFrame(index=fit_nr.entity_id.values)
    out_apps = [pd.DataFrame(index=a.entity_id.values) for a in app_nrs]
    for tgt in NODE_TARGETS:
        y = fit_nr.entity_id.map(labels[tgt]).values.astype(int)
        oof = np.zeros(len(fit_nr))
        for f in range(INNER_FOLDS):
            m = fold != f
            oof[~m] = NodeText().fit(fit_nr[m], y[m]).predict(fit_nr[~m])
        out_fit["p_" + tgt] = oof
        full = NodeText().fit(fit_nr, y)
        for a, o in zip(app_nrs, out_apps):
            o["p_" + tgt] = full.predict(a) if len(a) else []
    return out_fit, out_apps


def add_node_feats(P, ents, ns):
    """Attach node scores of child and candidate plus within-family relative ranks."""
    P = P.copy()
    for k in ns.columns:
        P[k + "_j"] = P.pid.map(ns[k]).fillna(-1.0).values
        P[k + "_i"] = P.cid.map(ns[k]).values
    nodes = ents.loc[ents.is_root == 0, ["entity_id", "family_id"]].copy()
    nodes["hc"] = nodes.entity_id.map(ns["p_y_hc"]).values
    gb = nodes.groupby("family_id").hc
    nodes["hc_rank"] = gb.rank(ascending=False, method="average")
    nodes["hc_rel"] = nodes.hc / (gb.transform("max") + 1e-12)
    nodes = nodes.set_index("entity_id")
    P["hc_rank_j"] = P.pid.map(nodes.hc_rank).fillna(0.0).values
    P["hc_rel_j"] = P.pid.map(nodes.hc_rel).fillna(1.0).values
    P["hc_minus_i"] = P.p_y_hc_j - P.p_y_hc_i
    return P


def make_labels(ents, parent_of):
    root_of = dict(zip(ents.family_id[ents.is_root == 1], ents.entity_id[ents.is_root == 1]))
    fam_of = dict(zip(ents.entity_id, ents.family_id))
    has_child = set(parent_of.values())
    y_hc = {e: int(e in has_child) for e in ents.entity_id}
    y_inter = {e: int(e in parent_of and parent_of[e] != root_of.get(fam_of[e])) for e in ents.entity_id}
    return {"y_hc": y_hc, "y_inter": y_inter}


def make_features(fit_ents, parent_of, apply_list):
    """Fit every transform on fit_ents (training families) and return the pair frame for fit_ents plus
    one pair frame per entity frame in apply_list (transform only)."""
    idf, idf_def = fit_idf(fit_ents)
    P_fit = build_pairs(fit_ents, idf, idf_def)
    labels = make_labels(fit_ents, parent_of)
    ns_fit, ns_apps = node_scores(fit_ents, labels, apply_list)
    P_fit = add_node_feats(P_fit, fit_ents, ns_fit)
    P_fit["y"] = (P_fit.pid.values == P_fit.cid.map(parent_of).values).astype(int)
    P_apps = [add_node_feats(build_pairs(a, idf, idf_def), a, ns) for a, ns in zip(apply_list, ns_apps)]
    return P_fit, P_apps


# ----------------------------------------------------------------------------------------------
# GBM
# ----------------------------------------------------------------------------------------------
def feature_cols(P):
    return [c for c in P.columns if c not in META_COLS]


def to_matrix(P, feats, cats=None):
    X = P[feats].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=None if cats is None else cats[c])
    return X


def train_gbm(P_fit, cfg, rounds):
    import lightgbm as lgb
    feats = feature_cols(P_fit)
    X = to_matrix(P_fit, feats)
    cats = {c: X[c].cat.categories for c in CAT_COLS}
    ds = lgb.Dataset(X, P_fit.y.values, categorical_feature=CAT_COLS, free_raw_data=False)
    bsts = []
    for k in range(N_SEEDS):
        params = dict(GBM_BASE); params.update(cfg); params["seed"] = SEED + k
        bsts.append(lgb.train(params, ds, num_boost_round=rounds))
    return bsts, feats, cats


def gbm_raw(model, P, rounds):
    bsts, feats, cats = model
    X = to_matrix(P, feats, cats)
    return np.mean([b.predict(X, num_iteration=rounds, raw_score=True) for b in bsts], axis=0)


# ----------------------------------------------------------------------------------------------
# tree-marginal decoding
# ----------------------------------------------------------------------------------------------
def tree_marginals(S, r):
    """S[i, j]: log-potential of edge parent j -> child i (-inf where absent). Returns edge marginals of
    the distribution over spanning arborescences rooted at r (Matrix-Tree theorem)."""
    n = S.shape[0]
    S = S.copy()
    S[r] = 0.0
    S = S - S.max(1, keepdims=True)  # per-child rescaling leaves the tree distribution unchanged
    W = np.exp(S)
    W[r] = 0.0
    np.fill_diagonal(W, 0.0)
    nr = np.array([k for k in range(n) if k != r])
    Wn = W[np.ix_(nr, nr)]
    L = -Wn.T.copy()
    L[np.arange(len(nr)), np.arange(len(nr))] = W[nr].sum(1)
    try:
        Li = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        Li = np.linalg.pinv(L)
    d = np.diag(Li)
    M = np.zeros((n, n))
    M[nr, r] = W[nr, r] * d
    Mn = Wn * (d[:, None] - Li)
    np.fill_diagonal(Mn, 0.0)
    M[np.ix_(nr, nr)] = Mn
    return np.clip(M, 0.0, 1.0)


class FamilyIndex:
    def __init__(self, P):
        self.fams = []
        for fam, idx in P.groupby("family_id", sort=True).indices.items():
            cid = P.cid.values[idx]; pid = P.pid.values[idx]
            ids = pd.unique(np.concatenate([cid, pid]))
            ix = {e: k for k, e in enumerate(ids)}
            ci = np.array([ix[e] for e in cid]); pj = np.array([ix[e] for e in pid])
            root = pid[P.j_is_root.values[idx] == 1][0]
            self.fams.append((fam, idx, ci, pj, ix[root], ids))


def decode_candidates(P, raw, alpha, fidx):
    """For every child: (child id, best non-root candidate id, its marginal, family id)."""
    cids, bests, margs, fams = [], [], [], []
    for fam, idx, ci, pj, r, ids in fidx.fams:
        n = len(ids)
        nr = np.array([k for k in range(n) if k != r])
        try:
            S = np.full((n, n), -np.inf)
            S[ci, pj] = alpha * raw[idx]
            M = tree_marginals(S, r)
            M[:, r] = -1.0
            np.fill_diagonal(M, -1.0)
            j = M[nr].argmax(1)
            mv = M[nr, j]
            best = np.where(mv >= 0, ids[j], ids[r])
            mv = np.maximum(mv, 0.0)
        except Exception as e:  # never let one family break the run
            log(f"warning: decode failed for family {fam}: {e!r}; placing its members under the root")
            best = np.full(len(nr), ids[r], dtype=object); mv = np.zeros(len(nr))
        cids.append(ids[nr]); bests.append(best); margs.append(mv); fams.append(np.full(len(nr), fam, dtype=object))
    return pd.DataFrame({"cid": np.concatenate(cids), "best": np.concatenate(bests),
                         "marg": np.concatenate(margs), "family_id": np.concatenate(fams)})


def f1_curve(cand, parent_of, root_of, ts):
    """Pooled intermediate-link F1 of 'emit best candidate if marginal > t' for each t."""
    truth = cand.cid.map(parent_of).values
    roots = cand.cid.map(root_of).values
    true_inter = truth != roots
    correct = (cand.best.values == truth)
    n_true = int(true_inter.sum())
    m = cand.marg.values
    out = []
    for t in ts:
        sel = m > t
        tp = int((sel & correct).sum()); fp = int(sel.sum()) - tp; fn = n_true - tp
        out.append(2 * tp / max(1, 2 * tp + fp + fn))
    return np.array(out)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    public_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/public")
    sub_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("working/submission.csv")
    sub_out.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    rd = lambda f: pd.read_csv(public_dir / f, dtype=str, keep_default_na=False)
    test_raw = rd("test_entities.csv")
    # placeholder: every test entity directly under its ultimate parent (schema-valid, scores 0)
    roots_te = test_raw[test_raw.is_ultimate_parent.str.strip() == "1"].set_index("family_id").entity_id
    te_nr = test_raw[test_raw.is_ultimate_parent.str.strip() != "1"]
    placeholder = pd.DataFrame({"entity_id": te_nr.entity_id.values,
                                "parent_entity_id": te_nr.family_id.map(roots_te).fillna("").values})
    placeholder.to_csv(sub_out, index=False)
    log(f"placeholder written: {len(placeholder)} rows")

    train_raw = rd("train_entities.csv")
    parents = rd("train_parents.csv")
    parent_of = dict(zip(parents.entity_id, parents.parent_entity_id))
    tr = prep_entities(train_raw)
    te = prep_entities(test_raw)
    root_of_tr = {}
    for fam, g in tr.groupby("family_id"):
        r = g.entity_id[g.is_root == 1]
        if len(r):
            for e in g.entity_id:
                root_of_tr[e] = r.iloc[0]
    log(f"train {tr.family_id.nunique()} families / {len(tr)} entities; "
        f"test {te.family_id.nunique()} families / {len(te)} entities")
    log(f"plan: {N_FOLDS}-fold family CV x {len(GBM_GRID)} GBM config x {N_SEEDS} seeds x {MAX_ROUNDS} rounds "
        f"(checkpoints {ROUND_CHECKPOINTS}), decode grid {len(ALPHA_GRID)} alphas x {len(T_GRID)} thresholds, "
        f"test scored by the average of the selected configuration's {N_FOLDS} fold models")

    # ---- family-grouped CV of the whole pipeline -> out-of-fold raw scores per (config, rounds)
    fm = family_folds(tr.family_id.unique(), N_FOLDS, SEED)
    fold = tr.family_id.map(fm).values
    oof_P, oof_raw, te_raw, P_te = [], {}, {}, None
    for f in range(N_FOLDS):
        tf = time.time()
        fit_e, val_e = tr[fold != f], tr[fold == f]
        # test pairs are transformed with this fold's fitted transforms (train-only fit, test inference only)
        P_fit, (P_val, P_te_f) = make_features(fit_e, parent_of, [val_e, te])
        if P_te is None:
            P_te = P_te_f[["family_id", "cid", "pid", "j_is_root"]].reset_index(drop=True)
        key_te = pd.MultiIndex.from_arrays([P_te.cid.values, P_te.pid.values])
        oof_P.append(P_val[["family_id", "cid", "pid", "j_is_root"]])
        for ci, cfg in enumerate(GBM_GRID):
            tc = time.time()
            model = train_gbm(P_fit, cfg, MAX_ROUNDS)
            for rounds in ROUND_CHECKPOINTS:
                oof_raw.setdefault((ci, rounds), []).append(gbm_raw(model, P_val, rounds))
                r_te = pd.Series(gbm_raw(model, P_te_f, rounds),
                                 index=pd.MultiIndex.from_arrays([P_te_f.cid.values, P_te_f.pid.values]))
                te_raw.setdefault((ci, rounds), []).append(r_te.reindex(key_te).fillna(0.0).values)
            log(f"fold {f} config {ci}: {len(P_fit)} train pairs, gbm {time.time() - tc:.1f}s")
        log(f"fold {f} done in {time.time() - tf:.1f}s")
    OP = pd.concat(oof_P, ignore_index=True)
    fidx = FamilyIndex(OP)

    # ---- in-script search of GBM config / rounds / decode temperature / threshold on OOF F1
    best = (-1.0, None)
    for key, parts in oof_raw.items():
        raw = np.concatenate(parts)
        for alpha in ALPHA_GRID:
            cand = decode_candidates(OP, raw, alpha, fidx)
            curve = f1_curve(cand, parent_of, root_of_tr, T_GRID)
            k = int(np.argmax(curve))
            if curve[k] > best[0]:
                best = (float(curve[k]), dict(config=key[0], rounds=key[1], alpha=alpha, t=float(T_GRID[k])))
        log(f"config {key[0]} rounds {key[1]}: best so far {best[0]:.4f} {best[1]}")
    sel = best[1]
    log(f"selected {sel} with out-of-fold intermediate-link F1 {best[0]:.4f}")

    # ---- test: average the raw scores of the selected configuration's fold models (model ensembling),
    # then decode every test family on its own
    raw_te = np.mean(te_raw[(sel["config"], sel["rounds"])], axis=0)
    cand = decode_candidates(P_te, raw_te, sel["alpha"], FamilyIndex(P_te))
    chosen = dict(zip(cand.cid[cand.marg > sel["t"]], cand.best[cand.marg > sel["t"]]))
    log(f"test: {len(chosen)} intermediate links emitted for {len(cand)} children")

    sub = placeholder.copy()
    sub["parent_entity_id"] = [chosen.get(e, p) for e, p in zip(sub.entity_id, sub.parent_entity_id)]
    # sanity checks (warn only)
    if sub.entity_id.duplicated().any():
        log("warning: duplicate entity ids in submission")
    fam_of = dict(zip(te.entity_id, te.family_id))
    bad = [e for e, p in zip(sub.entity_id, sub.parent_entity_id) if p and fam_of.get(p) != fam_of.get(e)]
    if bad:
        log(f"warning: {len(bad)} parents outside family; resetting them to the root")
        badset = set(bad)
        sub.loc[sub.entity_id.isin(badset), "parent_entity_id"] = placeholder.loc[sub.entity_id.isin(badset), "parent_entity_id"]
    sub[["entity_id", "parent_entity_id"]].to_csv(sub_out, index=False)
    log(f"submission written to {sub_out}: {len(sub)} rows")


if __name__ == "__main__":
    main()
