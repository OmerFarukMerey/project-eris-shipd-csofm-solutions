"""Offline experiments on saved OOF predictions (oof.pkl)."""
import pickle, numpy as np
from core import compute_score, scoring_family, SCORING_FAMILIES, FAMILY_OF, PREDS_BY_FAM

D = pickle.load(open("oof.pkl", "rb"))
rec = D["rec_light"]; oof_pred = D["oof_pred"]; oof_gate = D["oof_gate"]
seed_map = D["seed_map"]; gold_patch = D["gold_patch"]
ids = sorted(gold_patch.keys())
N = len(rec["id"])
GOLD = {i: gold_patch[i] for i in ids}

# group row indices by doc
rows_by_doc = {}
for r in range(N):
    rows_by_doc.setdefault(rec["id"][r], []).append(r)

def decode(fam_th, gate_min=0.0, gate_alpha=0.0, emit_argmax=False, argmax_gate=1.1,
           per_pred_th=None):
    out = {i: set() for i in ids}
    for r in range(N):
        did = rec["id"][r]
        s = rec["s"][r]; o = rec["o"][r]; gp = oof_gate[r]
        if gp < gate_min:
            continue
        seeds = seed_map.get(did, set())
        best = None
        emitted = 0
        for p, prob in oof_pred[r].items():
            sc = prob * (gp ** gate_alpha) if gate_alpha > 0 else prob
            th = per_pred_th[p] if per_pred_th else fam_th[scoring_family(p)]
            if best is None or sc > best[0]:
                best = (sc, p, prob)
            if sc >= th:
                t = (s, p, o)
                if t not in seeds:
                    out[did].add(t); emitted += 1
        if emit_argmax and emitted == 0 and gp >= argmax_gate and best is not None:
            t = (s, best[1], o)
            if t not in seeds:
                out[did].add(t)
    return out

def score(pred):
    return compute_score(pred, GOLD)

def diag(fam_th, **kw):
    c = score(decode(fam_th, **kw))
    print("score %.4f | exact %.3f docmac %.3f pair %.3f fammac %.3f worst %.3f" % (
        c["score"], c["exact_edge_f1"], c["document_macro_f1"], c["endpoint_pair_f1"],
        c["relation_family_macro_f1"], c["worst_patch_density_f1"]))
    print("   fam:", {k: round(v,3) for k,v in c["fam_scores"].items()})
    print("   buckets:", {k: round(v,3) for k,v in c["bucket_means"].items()})
    return c

BASE = {'chem_disease':0.31,'gene_disease':0.43,'cg_activity':0.29,'cg_expression':0.31,'cg_other':0.85}

def coord_ascent(init, grid=None, keys=None, **decode_kw):
    if grid is None: grid = np.round(np.arange(0.12,0.90,0.01),3)
    if keys is None: keys = SCORING_FAMILIES
    th = dict(init)
    best = score(decode(th, **decode_kw))["score"]
    for it in range(4):
        improved=False
        for f in keys:
            b0=th[f]; bestf=b0; bests=best
            for v in grid:
                th[f]=v; s=score(decode(th, **decode_kw))["score"]
                if s>bests: bests=s; bestf=v
            th[f]=bestf
            if bests>best+1e-9: best=bests; improved=True
        if not improved: break
    return th, best

if __name__ == "__main__":
    import sys
    print("=== BASE thresholds ==="); diag(BASE)
    print("\n=== retune family th (fine grid) ===")
    th,b = coord_ascent(BASE); print("best %.4f"%b, {k:round(v,2) for k,v in th.items()}); diag(th)
    print("\n=== + emit_argmax for confident-gate pairs (argmax_gate sweep) ===")
    for ag in [0.5,0.6,0.7,0.8,0.9]:
        c=score(decode(th, emit_argmax=True, argmax_gate=ag))
        print("  argmax_gate %.2f -> %.4f (pair %.3f exact %.3f)"%(ag,c["score"],c["endpoint_pair_f1"],c["exact_edge_f1"]))
