"""Grouped CV + threshold tuning against the exact competition metric."""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from core import compute_score, SCORING_FAMILIES, scoring_family
from features import build_frame
from model import fit_models, predict_probs, decode

def subset_rec(rec, mask):
    out = {}
    for k in ["id", "family", "s", "o", "both", "union", "between"]:
        out[k] = rec[k][mask]
    out["num"] = rec["num"][mask]
    out["gold_preds"] = [rec["gold_preds"][i] for i in np.where(mask)[0]]
    out["seed_preds"] = [rec["seed_preds"][i] for i in np.where(mask)[0]]
    return out

def tune_thresholds(rec, pred_prob, gate_prob, seed_map, gold_patch, ids,
                    gate_alpha=0.0, gate_min=0.0, init=0.5, verbose=False):
    grid = np.round(np.arange(0.15, 0.86, 0.02), 3)
    th = {f: init for f in SCORING_FAMILIES}
    def score_with(th):
        pred = decode(rec, pred_prob, gate_prob, seed_map, th, gate_alpha, gate_min, ids)
        return compute_score(pred, {i: gold_patch[i] for i in ids})["score"]
    best = score_with(th)
    for it in range(3):
        improved = False
        for f in SCORING_FAMILIES:
            base = th[f]; bestf = base; bests = best
            for v in grid:
                th[f] = v
                s = score_with(th)
                if s > bests:
                    bests = s; bestf = v
            th[f] = bestf
            if bests > best + 1e-9:
                best = bests; improved = True
        if verbose:
            print("   iter %d score %.4f th=%s" % (it, best, {k: round(v,2) for k,v in th.items()}))
        if not improved:
            break
    return th, best

def run_cv(train_csv, n_folds=5, gate_alpha=0.0, gate_min=0.0, seed=0):
    df = pd.read_csv(train_csv).fillna("")
    t0 = time.time()
    print("Building full frame...")
    rec, gold_patch, gold_full, seed_map = build_frame(df)
    print("  candidates:", len(rec["id"]), "time %.1fs" % (time.time()-t0))
    ids_all = df["id"].values
    gkf = GroupKFold(n_splits=n_folds)
    # out-of-fold containers
    n = len(rec["id"])
    oof_pred = [None]*n
    oof_gate = np.zeros(n)
    doc_to_rows = {}
    for i, did in enumerate(rec["id"]):
        doc_to_rows.setdefault(did, []).append(i)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=ids_all)):
        tr_ids = set(df["id"].values[tr_idx]); va_ids = set(df["id"].values[va_idx])
        tr_mask = np.array([i in tr_ids for i in rec["id"]])
        va_mask = ~tr_mask
        tr_rec = subset_rec(rec, tr_mask)
        va_rec = subset_rec(rec, va_mask)
        M = fit_models(tr_rec)
        pp, gp = predict_probs(M, va_rec)
        va_rows = np.where(va_mask)[0]
        for j, ridx in enumerate(va_rows):
            oof_pred[ridx] = pp[j]; oof_gate[ridx] = gp[j]
        print("  fold %d done  (%.1fs)" % (fold, time.time()-t0))
    ids = sorted(gold_patch.keys())
    th, best = tune_thresholds(rec, oof_pred, oof_gate, seed_map, gold_patch, ids,
                               gate_alpha=gate_alpha, gate_min=gate_min, verbose=True)
    # sweep emit_argmax
    GOLD = {i: gold_patch[i] for i in ids}
    best_arg = None; best_s = best
    for ag in [None, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        if ag is None:
            pred = decode(rec, oof_pred, oof_gate, seed_map, th, gate_alpha, gate_min, False, 1.1, ids)
        else:
            pred = decode(rec, oof_pred, oof_gate, seed_map, th, gate_alpha, gate_min, True, ag, ids)
        s = compute_score(pred, GOLD)["score"]
        print("   emit_argmax %-5s -> %.4f" % (str(ag), s))
        if s > best_s: best_s = s; best_arg = ag
    emit_argmax = best_arg is not None
    argmax_gate = best_arg if best_arg is not None else 1.1
    pred = decode(rec, oof_pred, oof_gate, seed_map, th, gate_alpha, gate_min, emit_argmax, argmax_gate, ids)
    print("=== OOF FINAL (gate_alpha=%.1f gate_min=%.2f emit_argmax=%s@%s) ===" % (
        gate_alpha, gate_min, emit_argmax, argmax_gate))
    comp = compute_score(pred, {i: gold_patch[i] for i in ids}, verbose=True)
    print("  fam_scores:", {k: round(v,3) for k,v in comp["fam_scores"].items()})
    print("  bucket_means:", {k: round(v,3) for k,v in comp["bucket_means"].items()})
    print("  thresholds:", {k: round(v,3) for k,v in th.items()})
    import json, pickle
    json.dump({"thresholds": th, "_cfg": {"gate_alpha": gate_alpha, "gate_min": gate_min,
               "emit_argmax": emit_argmax, "argmax_gate": argmax_gate},
               "_oof_score": comp["score"]}, open("thresholds.json", "w"), indent=2)
    pickle.dump({"rec_light": {k: rec[k] for k in ["id","family","s","o"]},
                 "oof_pred": oof_pred, "oof_gate": oof_gate,
                 "seed_map": seed_map, "gold_patch": gold_patch},
                open("oof.pkl", "wb"))
    return comp, th, (rec, oof_pred, oof_gate, seed_map, gold_patch)

if __name__ == "__main__":
    ga = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    gm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    run_cv("../dataset/public/train.csv", gate_alpha=ga, gate_min=gm)
