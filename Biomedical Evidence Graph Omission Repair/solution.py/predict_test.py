"""Fit on full train, predict test, write submission.csv."""
import sys, json, time
import numpy as np
import pandas as pd
from core import serialize_triples, RELATION_LABELS, FAM_ENDPOINTS, FAMILY_OF, etype
from features import build_frame
from model import fit_models, predict_probs, decode

def valid_triple(s, p, o, inv):
    if s not in inv or o not in inv:
        return False
    fam = FAMILY_OF.get(p)
    if fam is None:
        return False
    st, ot = FAM_ENDPOINTS[fam]
    return etype(s) == st and etype(o) == ot

def main(thresholds, gate_alpha=0.0, gate_min=0.0, emit_argmax=False, argmax_gate=1.1,
         out_path="../working/submission.csv", use_char=True, seeds=(0,)):
    t0 = time.time()
    train = pd.read_csv("../dataset/public/train.csv").fillna("")
    test = pd.read_csv("../dataset/public/test.csv").fillna("")
    print("building train frame..."); tr_rec, _, _, _ = build_frame(train)
    print("building test frame...");  te_rec, _, _, te_seed = build_frame(test)
    print("  frames built %.1fs" % (time.time()-t0))
    # ensemble over seeds (bagging of the same deterministic fit is a no-op; kept for extensibility)
    n = len(te_rec["id"])
    agg = [dict() for _ in range(n)]
    agg_gate = np.zeros(n)
    for si in seeds:
        M = fit_models(tr_rec)
        pp, gp = predict_probs(M, te_rec)
        for r in range(n):
            for k, v in pp[r].items():
                agg[r][k] = agg[r].get(k, 0.0) + v / len(seeds)
        agg_gate += gp / len(seeds)
        print("  fit+predict seed %d done %.1fs" % (si, time.time()-t0))
    ids = list(test["id"].values)
    pred = decode(te_rec, agg, agg_gate, te_seed, thresholds, gate_alpha, gate_min,
                  emit_argmax, argmax_gate, ids)
    # validate & serialize
    inv_map = {r["id"]: set(r["entity_inventory"].split()) for _, r in test.iterrows()}
    rows = []
    n_edges = 0
    for did in ids:
        triples = []
        for (s, p, o) in sorted(pred.get(did, set())):
            if valid_triple(s, p, o, inv_map[did]):
                triples.append((s, p, o))
        n_edges += len(triples)
        rows.append({"id": did, "relation_patch": serialize_triples(triples)})
    sub = pd.DataFrame(rows, columns=["id", "relation_patch"])
    assert len(sub) == len(test) == 160
    assert sub["id"].tolist() == ids
    sub.to_csv(out_path, index=False)
    print("wrote %s  rows=%d edges=%d nonempty=%d  (%.1fs)" % (
        out_path, len(sub), n_edges, (sub["relation_patch"] != "").sum(), time.time()-t0))

if __name__ == "__main__":
    th = json.load(open("thresholds.json"))
    cfg = th.get("_cfg", {})
    main(th["thresholds"], gate_alpha=cfg.get("gate_alpha", 0.0),
         gate_min=cfg.get("gate_min", 0.0), emit_argmax=cfg.get("emit_argmax", False),
         argmax_gate=cfg.get("argmax_gate", 1.1))
