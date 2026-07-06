"""Build per-candidate-pair feature frames."""
import numpy as np
from collections import Counter
from core import (parse_triples, doc_structures, pair_contexts, candidate_pairs,
                  PREDS_BY_FAM, FAMILY_OF, RELATION_LABELS, etype)

REL_INDEX = {r: i for i, r in enumerate(RELATION_LABELS)}

# Cue lexicons (substring match on lowercased masked context) -> directional/axis signals
CUE_GROUPS = {
    "inc": ["increas", "induc", "upregulat", "up-regulat", "up regulat", "enhanc", "elevat",
            "stimulat", "activat", "promot", "augment", "overexpress", "over-express", "higher",
            "raise", "raised", "greater", "gain"],
    "dec": ["decreas", "reduc", "downregulat", "down-regulat", "down regulat", "inhibit", "suppress",
            "block", "attenuat", "lower", "deplet", "diminish", "impair", "abrogat", "abolish",
            "antagoni", "loss", "deficien", "knockdown", "silenc", "repress", "prevent"],
    "expr": ["express", "mrna", "transcription", "transcript", "translation", "protein level",
             "gene expression", "immunoblot", "western", "qpcr", "levels of"],
    "activ": ["activ", "function", "catalyt", "enzymatic", "kinase activity", "phosphoryl"],
    "bind": ["bind", "affinit", "interact", "ligand", "agonist", "antagonist", "receptor", "docking", "complex"],
    "metab": ["metaboli", "degrad", "catabol", "hydroxylat", "oxidat", "cleav", "conjugat", "glucuronid", "biotransform"],
    "transp": ["transport", "uptake", "efflux", "secret", "import", "export", "influx", "absorption"],
    "local": ["localiz", "translocat", "accumulat", "nuclear", "cytoplasm", "membrane", "redistribut"],
    "therap": ["treat", "therap", "therapy", "efficac", "ameliorat", "improv", "alleviat", "beneficial",
               "protect", "recover", "remission", "clinical trial", "administ", "dose", "efficacy"],
    "marker": ["induc", "caus", "associat", "risk", "develop", "toxic", "adverse", "side effect",
               "carcinogen", "damage", "injury", "pathogenesis", "contribut", "lead to", "result in"],
}
CUE_KEYS = list(CUE_GROUPS.keys())

def cue_counts(text):
    v = np.zeros(len(CUE_KEYS), dtype=np.float32)
    if not text:
        return v
    for i, k in enumerate(CUE_KEYS):
        c = 0
        for w in CUE_GROUPS[k]:
            c += text.count(w)
        v[i] = np.log1p(c)
    return v

def build_frame(df):
    """For every candidate pair in every doc, produce a record.
    Returns dict of parallel lists keyed by fields, plus gold graph per doc.
    """
    rec = {k: [] for k in ["id", "family", "s", "o", "both", "union", "between",
                           "num", "gold_preds", "seed_preds"]}
    gold_patch = {}   # id -> set of patch triples
    gold_full = {}    # id -> set of seed+patch triples
    seed_map = {}     # id -> set of seed triples
    for _, r in df.iterrows():
        did = r["id"]
        inv = r["entity_inventory"]
        ds = doc_structures(r["masked_evidence"], inv)
        seeds = parse_triples(r.get("seed_relations", ""))
        patches = parse_triples(r.get("relation_patch", "")) if "relation_patch" in df.columns else []
        full = set(seeds) | set(patches)
        gold_patch[did] = set(patches)
        gold_full[did] = full
        seed_map[did] = set(seeds)
        # per-pair seed predicate map + degrees
        seed_pair_preds = {}
        s_subj_deg = Counter(); o_obj_deg = Counter()
        any_seed_ent = set()
        fam_seed_count = Counter()
        for (ss, pp, oo) in seeds:
            seed_pair_preds.setdefault((ss, oo), set()).add(pp)
            s_subj_deg[ss] += 1; o_obj_deg[oo] += 1
            any_seed_ent.add(ss); any_seed_ent.add(oo)
            fam_seed_count[FAMILY_OF.get(pp, "?")] += 1
        full_pair_preds = {}
        for (ss, pp, oo) in full:
            full_pair_preds.setdefault((ss, oo), set()).add(pp)
        ents = inv.split()
        n_chem = sum(e.startswith("CHEM") for e in ents)
        n_gene = sum(e.startswith("GENE") for e in ents)
        n_dis = sum(e.startswith("DISEASE") for e in ents)
        n_ent = len(ents)
        n_seed = len(seeds)
        n_sents = max(ds["n_sents"], 1)
        for (fam, s, o) in candidate_pairs(inv):
            both, union, between, both_idx, union_idx = pair_contexts(ds, s, o)
            s_sents = ds["ent_sents"].get(s, set())
            o_sents = ds["ent_sents"].get(o, set())
            # min sentence distance
            if s_sents and o_sents:
                mind = min(abs(a - b) for a in s_sents for b in o_sents)
            else:
                mind = 99
            s_first = min(s_sents) if s_sents else 99
            o_first = min(o_sents) if o_sents else 99
            sp = seed_pair_preds.get((s, o), set())
            seed_onehot = np.zeros(len(RELATION_LABELS), dtype=np.float32)
            for p in sp:
                if p in REL_INDEX: seed_onehot[REL_INDEX[p]] = 1.0
            fam_oh = [float(fam == f) for f in ("chem_disease", "chem_gene", "gene_disease")]
            num = np.array([
                np.log1p(ds["ent_count"].get(s, 0)),
                np.log1p(ds["ent_count"].get(o, 0)),
                np.log1p(len(both_idx)),
                np.log1p(len(union_idx)),
                float(len(both_idx) > 0),
                float(mind), float(min(mind, 5)),
                float(0 in both_idx),        # co-occur in title
                float(0 in s_sents), float(0 in o_sents),
                s_first / n_sents, o_first / n_sents,
                np.log1p(n_chem), np.log1p(n_gene), np.log1p(n_dis), np.log1p(n_ent),
                np.log1p(n_seed),
                float(len(sp) > 0),                       # pair already has a seed edge
                float(s in any_seed_ent), float(o in any_seed_ent),
                np.log1p(s_subj_deg.get(s, 0)), np.log1p(o_obj_deg.get(o, 0)),
                np.log1p(fam_seed_count.get(fam, 0)),
                *fam_oh,
            ], dtype=np.float32)
            cue_src = both if both else union
            num = np.concatenate([num, seed_onehot, cue_counts(cue_src)])
            gp = full_pair_preds.get((s, o), set())  # target uses FULL gold (seed+patch)
            rec["id"].append(did); rec["family"].append(fam)
            rec["s"].append(s); rec["o"].append(o)
            rec["both"].append(both); rec["union"].append(union); rec["between"].append(between)
            rec["num"].append(num)
            rec["gold_preds"].append(gp)
            rec["seed_preds"].append(sp)
    rec["num"] = np.vstack(rec["num"]) if rec["num"] else np.zeros((0, 1))
    for k in ["id", "family", "s", "o", "both", "union", "between"]:
        rec[k] = np.array(rec[k], dtype=object)
    return rec, gold_patch, gold_full, seed_map

NUM_FEATURE_NAMES = [
    "log_s_cnt","log_o_cnt","log_both","log_union","has_both","mind","mind5",
    "both_title","s_title","o_title","s_first","o_first",
    "log_nchem","log_ngene","log_ndis","log_nent","log_nseed",
    "pair_has_seed","s_in_seed","o_in_seed","log_s_subjdeg","log_o_objdeg","log_fam_seed",
    "fam_cd","fam_cg","fam_gd",
] + ["seed_" + r for r in RELATION_LABELS] + ["cue_" + k for k in CUE_KEYS]
