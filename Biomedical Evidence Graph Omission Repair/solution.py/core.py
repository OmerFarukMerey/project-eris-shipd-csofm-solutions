"""Core utilities: parsing, schema, candidate generation, context, scorer."""
import re
import numpy as np
from collections import Counter, defaultdict

# ---------------- Schema ----------------
RELATION_LABELS = [
    "chem_disease:marker/mechanism", "chem_disease:therapeutic",
    "chem_gene:affects^activity", "chem_gene:decreases^activity", "chem_gene:increases^activity",
    "chem_gene:affects^binding",
    "chem_gene:affects^expression", "chem_gene:decreases^expression", "chem_gene:increases^expression",
    "chem_gene:affects^localization",
    "chem_gene:affects^metabolic_processing", "chem_gene:decreases^metabolic_processing", "chem_gene:increases^metabolic_processing",
    "chem_gene:affects^transport", "chem_gene:decreases^transport", "chem_gene:increases^transport",
    "gene_disease:marker/mechanism", "gene_disease:therapeutic",
]
FAMILY_OF = {r: r.split(":")[0] for r in RELATION_LABELS}
FAM_ENDPOINTS = {"chem_disease": ("CHEM", "DISEASE"),
                 "chem_gene": ("CHEM", "GENE"),
                 "gene_disease": ("GENE", "DISEASE")}
PREDS_BY_FAM = defaultdict(list)
for r in RELATION_LABELS:
    PREDS_BY_FAM[FAMILY_OF[r]].append(r)

# 5 scoring families
def scoring_family(pred):
    if pred.startswith("chem_disease"): return "chem_disease"
    if pred.startswith("gene_disease"): return "gene_disease"
    # chem_gene subtypes
    sub = pred.split("^")[1] if "^" in pred else pred
    if sub == "activity": return "cg_activity"
    if sub == "expression": return "cg_expression"
    return "cg_other"

SCORING_FAMILIES = ["chem_disease", "gene_disease", "cg_activity", "cg_expression", "cg_other"]

def etype(e):
    return e.split("_")[0]

# ---------------- Parsing ----------------
def parse_triples(cell):
    if not isinstance(cell, str):
        return []
    cell = cell.strip()
    if not cell:
        return []
    out = []
    for part in cell.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split("|")
        if len(bits) != 3:
            continue
        s, p, o = bits[0].strip(), bits[1].strip(), bits[2].strip()
        out.append((s, p, o))
    return out

def serialize_triples(triples):
    # unique, order-stable
    seen = set(); out = []
    for t in triples:
        if t in seen: continue
        seen.add(t); out.append("%s|%s|%s" % t)
    return " ; ".join(out)

MENTION_RE = re.compile(r"\[([A-Z]+_\d+)\]")

def split_sentences(text):
    # crude sentence splitter on . ! ? followed by space
    return re.split(r"(?<=[.!?])\s+", text)

def doc_structures(text, inventory):
    """Precompute per-doc: sentences, per-entity sentence indices, mention counts, title tokens."""
    sents = split_sentences(text)
    ent_sents = defaultdict(set)   # entity -> set of sentence idx
    ent_count = Counter()
    sent_ents = []                 # per sentence: set of entities present
    for i, s in enumerate(sents):
        ms = MENTION_RE.findall(s)
        sent_ents.append(set(ms))
        for m in ms:
            ent_sents[m].add(i)
            ent_count[m] += 1
    return {"sents": sents, "ent_sents": ent_sents, "ent_count": ent_count,
            "sent_ents": sent_ents, "n_sents": len(sents)}

def _mask_sentence(sent, s, o):
    """Replace entity mentions with role markers for a pair (s,o)."""
    def repl(m):
        e = m.group(1)
        if e == s: return " esubj "
        if e == o: return " eobj "
        t = etype(e)
        if t == "CHEM": return " echem "
        if t == "GENE": return " egene "
        if t == "DISEASE": return " edis "
        return " eent "
    x = MENTION_RE.sub(repl, sent)
    x = x.replace("[NUM]", " enum ")
    return x.lower()

_TOK_RE = re.compile(r"\[[A-Z]+_\d+\]|\[NUM\]|\w+|[^\s\w]")

def _between_context(sent, s, o, win=6):
    """Tokens strictly between the closest s/o mention pair, plus a small window."""
    toks = _TOK_RE.findall(sent)
    spos = [i for i, t in enumerate(toks) if t == "[" + s + "]"]
    opos = [i for i, t in enumerate(toks) if t == "[" + o + "]"]
    if not spos or not opos:
        return ""
    best = None
    for a in spos:
        for b in opos:
            d = abs(a - b)
            if best is None or d < best[0]:
                best = (d, a, b)
    _, a, b = best
    lo, hi = min(a, b), max(a, b)
    seg = toks[max(0, lo - win): hi + win + 1]
    joined = " ".join(seg)
    return _mask_sentence(joined, s, o)

def pair_contexts(ds, s, o):
    """Return (both_ctx, union_ctx, between_ctx, both_idx, union_idx) for a pair."""
    ss = ds["ent_sents"].get(s, set())
    oo = ds["ent_sents"].get(o, set())
    both_idx = sorted(ss & oo)
    union_idx = sorted(ss | oo)
    sents = ds["sents"]
    both = " ".join(_mask_sentence(sents[i], s, o) for i in both_idx)
    union = " ".join(_mask_sentence(sents[i], s, o) for i in union_idx)
    between = " ".join(x for x in (_between_context(sents[i], s, o) for i in both_idx) if x)
    return both, union, between, both_idx, union_idx

# ---------------- Candidate generation ----------------
def candidate_pairs(inventory):
    ents = inventory.split()
    chems = [e for e in ents if e.startswith("CHEM")]
    genes = [e for e in ents if e.startswith("GENE")]
    diseases = [e for e in ents if e.startswith("DISEASE")]
    pairs = []  # (family, subj, obj)
    for c in chems:
        for d in diseases:
            pairs.append(("chem_disease", c, d))
    for c in chems:
        for g in genes:
            pairs.append(("chem_gene", c, g))
    for g in genes:
        for d in diseases:
            pairs.append(("gene_disease", g, d))
    return pairs

# ---------------- Scorer ----------------
def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return 2.0 * tp / denom

def _set_f1(P, G):
    tp = len(P & G); fp = len(P - G); fn = len(G - P)
    return _f1(tp, fp, fn)

def compute_score(pred_patches, gold_patches, verbose=False):
    """pred_patches, gold_patches: dict id -> set of (s,p,o) triples (the PATCH)."""
    ids = list(gold_patches.keys())
    # exact_edge_f1 (micro over doc-edges)
    TP = FP = FN = 0
    for i in ids:
        P = pred_patches.get(i, set()); G = gold_patches[i]
        TP += len(P & G); FP += len(P - G); FN += len(G - P)
    exact_edge_f1 = _f1(TP, FP, FN)
    # document_macro_f1
    doc_f1 = {i: _set_f1(pred_patches.get(i, set()), gold_patches[i]) for i in ids}
    document_macro_f1 = float(np.mean([doc_f1[i] for i in ids]))
    # endpoint_pair_f1 (micro, ignore predicate)
    TP = FP = FN = 0
    for i in ids:
        P = set((s, o) for s, p, o in pred_patches.get(i, set()))
        G = set((s, o) for s, p, o in gold_patches[i])
        TP += len(P & G); FP += len(P - G); FN += len(G - P)
    endpoint_pair_f1 = _f1(TP, FP, FN)
    # relation_family_macro_f1 (per family micro F1, averaged)
    fam_scores = []
    for fam in SCORING_FAMILIES:
        TP = FP = FN = 0
        for i in ids:
            P = set(t for t in pred_patches.get(i, set()) if scoring_family(t[1]) == fam)
            G = set(t for t in gold_patches[i] if scoring_family(t[1]) == fam)
            TP += len(P & G); FP += len(P - G); FN += len(G - P)
        fam_scores.append(_f1(TP, FP, FN))
    relation_family_macro_f1 = float(np.mean(fam_scores))
    # worst_patch_density_f1
    buckets = {"none": [], "small": [], "medium": [], "dense": []}
    for i in ids:
        n = len(gold_patches[i])
        b = "none" if n == 0 else "small" if n <= 2 else "medium" if n <= 5 else "dense"
        buckets[b].append(doc_f1[i])
    bucket_means = {b: (float(np.mean(v)) if v else 1.0) for b, v in buckets.items()}
    worst_patch_density_f1 = min(bucket_means.values())
    score = (0.40 * exact_edge_f1 + 0.20 * document_macro_f1 + 0.15 * endpoint_pair_f1
             + 0.15 * relation_family_macro_f1 + 0.10 * worst_patch_density_f1)
    comp = dict(exact_edge_f1=exact_edge_f1, document_macro_f1=document_macro_f1,
                endpoint_pair_f1=endpoint_pair_f1, relation_family_macro_f1=relation_family_macro_f1,
                worst_patch_density_f1=worst_patch_density_f1, score=score,
                fam_scores=dict(zip(SCORING_FAMILIES, fam_scores)),
                bucket_means=bucket_means)
    if verbose:
        for k in ["exact_edge_f1","document_macro_f1","endpoint_pair_f1","relation_family_macro_f1","worst_patch_density_f1","score"]:
            print("  %-26s %.4f" % (k, comp[k]))
    return comp
