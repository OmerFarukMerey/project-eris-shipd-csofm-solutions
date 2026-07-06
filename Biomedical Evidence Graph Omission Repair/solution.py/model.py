"""Fit per-family relation models (LR + ComplementNB ensemble) and decode."""
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from core import PREDS_BY_FAM, FAMILY_OF, RELATION_LABELS, scoring_family

FAMILIES = ["chem_disease", "chem_gene", "gene_disease"]

def _vec(min_df=2, ngram=(1, 2), max_features=40000):
    return TfidfVectorizer(min_df=min_df, ngram_range=ngram, max_features=max_features,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")

def build_matrix(rec, vecs=None, scaler=None, fit=False, use_char=True):
    """Return (X_full, X_text). X_text is non-negative (for NB); X_full adds scaled numeric."""
    if fit:
        tf_both = _vec(); tf_union = _vec(); tf_btw = _vec(min_df=2, ngram=(1, 3))
        Xb = tf_both.fit_transform(rec["both"]); Xu = tf_union.fit_transform(rec["union"])
        Xw = tf_btw.fit_transform(rec["between"])
        parts = [Xb, Xu, Xw]
        vecs = {"tf_both": tf_both, "tf_union": tf_union, "tf_btw": tf_btw}
        if use_char:
            tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                      max_features=16000, sublinear_tf=True)
            parts.append(tf_char.fit_transform(rec["both"])); vecs["tf_char"] = tf_char
        Xtext = sparse.hstack(parts).tocsr()
        scaler = StandardScaler().fit(rec["num"])
        Xfull = sparse.hstack(parts + [sparse.csr_matrix(scaler.transform(rec["num"]))]).tocsr()
        return Xfull, Xtext, vecs, scaler
    parts = [vecs["tf_both"].transform(rec["both"]), vecs["tf_union"].transform(rec["union"]),
             vecs["tf_btw"].transform(rec["between"])]
    if "tf_char" in vecs:
        parts.append(vecs["tf_char"].transform(rec["both"]))
    Xtext = sparse.hstack(parts).tocsr()
    Xfull = sparse.hstack(parts + [sparse.csr_matrix(scaler.transform(rec["num"]))]).tocsr()
    return Xfull, Xtext

def _fit_lr(X, y, C):
    return LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear").fit(X, y)

def _fit_nb(X, y):
    return ComplementNB().fit(X, y)

def fit_models(rec, C_text=4.0, use_char=True, w_nb=0.35, n_jobs=8, verbose=False):
    fam_arr = rec["family"]
    Xfull, Xtext, vecs, scaler = build_matrix(rec, fit=True, use_char=use_char)
    idx_by_fam = {fam: np.where(fam_arr == fam)[0] for fam in FAMILIES}
    lr_tasks = []; lr_meta = []
    nb_tasks = []; nb_meta = []
    const_preds = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        idx = idx_by_fam[fam]
        gold = [rec["gold_preds"][i] for i in idx]
        y_any = np.array([1 if len(g) > 0 else 0 for g in gold])
        if 0 < y_any.sum() < len(y_any):
            lr_tasks.append((Xfull[idx], y_any, C_text)); lr_meta.append((fam, "__gate__"))
        for p in PREDS_BY_FAM[fam]:
            y = np.array([1 if p in g else 0 for g in gold])
            if y.sum() < 3:
                const_preds[fam][p] = float(y.mean()); continue
            lr_tasks.append((Xfull[idx], y, C_text)); lr_meta.append((fam, p))
            nb_tasks.append((Xtext[idx], y)); nb_meta.append((fam, p))
    lr_fitted = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_fit_lr)(*t) for t in lr_tasks)
    nb_fitted = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_fit_nb)(*t) for t in nb_tasks)
    models = {fam: {"gate": None, "lr": {}, "nb": {}, "const": const_preds[fam]} for fam in FAMILIES}
    for (fam, key), m in zip(lr_meta, lr_fitted):
        if key == "__gate__": models[fam]["gate"] = m
        else: models[fam]["lr"][key] = m
    for (fam, key), m in zip(nb_meta, nb_fitted):
        models[fam]["nb"][key] = m
    return {"vecs": vecs, "scaler": scaler, "models": models, "w_nb": w_nb, "use_char": use_char}

def predict_probs(M, rec):
    Xfull, Xtext = build_matrix(rec, vecs=M["vecs"], scaler=M["scaler"], fit=False,
                                use_char=M.get("use_char", True))
    fam_arr = rec["family"]; n = len(rec["id"]); w = M["w_nb"]
    pred_prob = [dict() for _ in range(n)]
    gate_prob = np.zeros(n)
    for fam in FAMILIES:
        idx = np.where(fam_arr == fam)[0]
        if len(idx) == 0: continue
        Xf = Xfull[idx]; Xt = Xtext[idx]
        fm = M["models"][fam]
        gp = fm["gate"].predict_proba(Xf)[:, 1] if fm["gate"] is not None else np.zeros(len(idx))
        for j, ridx in enumerate(idx): gate_prob[ridx] = gp[j]
        for p in PREDS_BY_FAM[fam]:
            if p in fm["const"]:
                pv = np.full(len(idx), fm["const"][p])
            else:
                lr = fm["lr"][p].predict_proba(Xf)[:, 1]
                nb = fm["nb"][p].predict_proba(Xt)[:, 1]
                pv = (1 - w) * lr + w * nb
            for j, ridx in enumerate(idx): pred_prob[ridx][p] = pv[j]
    return pred_prob, gate_prob


def decode(rec, pred_prob, gate_prob, seed_map, thresholds, gate_alpha=0.0,
           gate_min=0.0, emit_argmax=False, argmax_gate=1.1, ids=None):
    if ids is None:
        ids = sorted(set(rec["id"]))
    out = {i: set() for i in ids}
    n = len(rec["id"])
    for r in range(n):
        did = rec["id"][r]
        if did not in out: continue
        s = rec["s"][r]; o = rec["o"][r]; gp = gate_prob[r]
        if gp < gate_min: continue
        seeds = seed_map.get(did, set())
        best = None; emitted = 0
        for p, prob in pred_prob[r].items():
            sc = prob * (gp ** gate_alpha) if gate_alpha > 0 else prob
            if best is None or sc > best[0]: best = (sc, p)
            if sc >= thresholds.get(scoring_family(p), 0.5):
                t = (s, p, o)
                if t not in seeds: out[did].add(t); emitted += 1
        if emit_argmax and emitted == 0 and gp >= argmax_gate and best is not None:
            t = (s, best[1], o)
            if t not in seeds: out[did].add(t)
    return out
