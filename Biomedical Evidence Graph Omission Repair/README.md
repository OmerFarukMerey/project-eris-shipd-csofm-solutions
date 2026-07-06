# Biomedical Evidence Graph Omission Repair

## The problem

Each document is a biomedical abstract with entities masked as `[CHEM_XX]`, `[GENE_XX]`, `[DISEASE_XX]`
tokens, a partial "seed" relation graph already extracted (`seed_relations`, triples of
`SUBJECT|RELATION_LABEL|OBJECT`), and an `entity_inventory` of every entity mentioned. The seed graph is
incomplete on purpose: some true relations were withheld. The task is to recover the missing edges — the
**`relation_patch`** — for 160 held-out test documents, given 363 labeled training documents.

Relations only exist between three endpoint-type pairs, each with its own label set:

| Family | Endpoint types | Example labels |
|---|---|---|
| `chem_disease` | CHEM → DISEASE | `marker/mechanism`, `therapeutic` |
| `chem_gene` | CHEM → GENE | `increases/decreases/affects` × `activity/expression/binding/localization/metabolic_processing/transport` |
| `gene_disease` | GENE → DISEASE | `marker/mechanism`, `therapeutic` |

18 relation labels in total. Test documents are bucketed by how many patch edges they're missing (`none`,
`small`, `medium`, `dense` — 14/41/54/51 documents), which matters because the grader explicitly scores the
*worst* bucket.

### How it's scored

The grader compares predicted vs. gold patch triples per document:

| Weight | Component | What it rewards |
|---|---|---|
| 0.40 | `exact_edge_f1` | micro F1 over exact `(subject, relation, object)` triples |
| 0.20 | `document_macro_f1` | per-document triple-set F1, averaged across documents |
| 0.15 | `endpoint_pair_f1` | micro F1 over `(subject, object)` pairs, ignoring the relation label |
| 0.15 | `relation_family_macro_f1` | macro F1 over 5 scoring families: `chem_disease`, `gene_disease`, and chem_gene split into `cg_activity`, `cg_expression`, `cg_other` |
| 0.10 | `worst_patch_density_f1` | the *minimum* of the four density-bucket mean F1s — a hard floor on the worst bucket |

This exact formula (reimplemented in `core.py::compute_score`) is what every modeling decision below is
tuned against, not a generic proxy.

## The solution

A **candidate-and-classify pipeline**: enumerate every plausible edge, build features for each candidate
from the masked text and the seed graph, score each relation label independently, then threshold and decode.

1. **Candidate generation.** For every document, form all `CHEM × DISEASE`, `CHEM × GENE`, and
   `GENE × DISEASE` pairs from its entity inventory — the full Cartesian product per family, since the
   schema forbids relations outside these three endpoint-type combinations anyway.

2. **Context extraction per pair.** Sentences are split, and for each candidate pair three text views are
   built: `both` (sentences mentioning both entities), `union` (sentences mentioning either), and `between`
   (the token window strictly between the closest mention pair). Entity mentions are masked to role/type
   markers (`esubj`, `eobj`, `echem`, `egene`, `edis`, `enum`) instead of raw IDs, so the model learns
   entity-agnostic patterns that transfer to unseen entities.

3. **Feature engineering** (`features.py`), per candidate pair:
   - ~26 numeric features: mention counts, sentence co-occurrence/distance, title co-occurrence, and
     **seed-graph structural features** — does this exact pair already have a seed edge, is either entity
     already seed-connected, the entity's seed degree, the family's seed-edge count in the document.
   - An 18-dim seed-onehot of which relation labels (if any) the seed graph already assigns to this pair.
   - 10 hand-built **cue-lexicon** features: log-count of direction/mechanism keyword hits (increase,
     decrease, expression, activity, binding, metabolic, transport, localization, therapeutic, marker) in
     the pair's context text.
   - Text: 3 TF-IDF vectorizers (word 1-2gram on `both`/`union`, word 1-3gram on `between`) plus a
     char 3-5gram TF-IDF on `both`, all concatenated with the scaled numeric features.

4. **Per-family models** (`model.py`). For each of the 3 endpoint families: a logistic-regression "gate"
   (does this pair have *any* relation at all), and per relation label an ensemble of a class-weighted
   `LogisticRegression` (full text+numeric matrix) and a `ComplementNB` (text-only, non-negative), blended
   65/35. Labels with fewer than 3 positive training examples in a fold fall back to a constant equal to
   their empirical rate rather than fitting an unstable classifier.

5. **Decoding.** Each family has its own tuned probability threshold (5 thresholds: `chem_disease`,
   `gene_disease`, `cg_activity`, `cg_expression`, `cg_other`) — emit an edge if its label probability
   clears its family's threshold and the pair isn't already a seed edge. An `emit_argmax` fallback emits the
   single best-scoring label for a pair when nothing clears its threshold but the pair's gate probability is
   high (≥0.55), trading a bit of precision for recall on plausible-but-under-threshold pairs.

6. **Validation** (`cv.py`). `GroupKFold` (5-fold, grouped by document, so a document's candidate pairs
   never split across train/val). Thresholds and the `emit_argmax` gate are tuned directly against the exact
   competition scorer via coordinate ascent over out-of-fold predictions.

7. **Iteration tooling.** `cv.py` caches out-of-fold predictions to `oof.pkl`; `analyze.py` reloads that
   cache to re-sweep thresholds/gates without re-training anything, which is how the current thresholds were
   found.

## Results (5-fold grouped OOF, from `run.log` / `thresholds.json`)

| Metric | Score |
|---|---|
| **Composite score** | **0.3701** |
| exact_edge_f1 | 0.3499 |
| document_macro_f1 | 0.3461 |
| endpoint_pair_f1 | 0.5522 |
| relation_family_macro_f1 | 0.3228 |
| worst_patch_density_f1 | 0.2963 |

Per-family F1: `chem_disease` 0.462, `gene_disease` 0.465, `cg_activity` 0.249, `cg_expression` 0.312,
`cg_other` 0.126 — the `chem_gene` sub-labels (especially binding/localization/metabolic/transport, lumped
into `cg_other`) are the hardest, both rarer and less lexically distinctive than the disease-relation
families. Per-density-bucket F1: `none` 0.296, `small` 0.322, `medium` 0.363, `dense` 0.357 — the `none`
bucket (documents with no missing edges at all) is the worst, i.e. correctly predicting an *empty* patch is
the main thing dragging down `worst_patch_density_f1`.

Tuned thresholds: `chem_disease` 0.21, `gene_disease` 0.31, `cg_activity` 0.19, `cg_expression` 0.21,
`cg_other` 0.59, with `emit_argmax` enabled at gate ≥ 0.55.

## Files

`solution.py/` is a small package (not a single script), split by concern:

| File | Purpose |
|---|---|
| `core.py` | Relation schema, triple parsing/serialization, candidate-pair generation, context extraction, and the exact scorer (`compute_score`). |
| `features.py` | Builds the per-candidate-pair feature frame (numeric + seed-onehot + cue-lexicon + text context fields). |
| `model.py` | Fits the per-family LR+NB ensembles, predicts probabilities, and decodes probabilities into final triples. |
| `cv.py` | Grouped 5-fold CV runner; tunes per-family thresholds and the `emit_argmax` gate against the exact scorer; writes `thresholds.json` and `oof.pkl`. |
| `analyze.py` | Offline scratchpad that reloads `oof.pkl` to re-tune thresholds/gates without retraining. |
| `predict_test.py` | Fits on the full 363-document train set, validates every emitted triple against the document's entity inventory and endpoint-type schema, and writes `submission.csv`. |
| `thresholds.json` | Tuned per-family thresholds + decode config + OOF score, produced by `cv.py`, consumed by `predict_test.py`. |
| `oof.pkl` | Cached out-of-fold predictions/gold labels, produced by `cv.py`, consumed by `analyze.py`. |
| `run.log` | Captured stdout from the last `cv.py` run. |

## How to run

Imports and dataset paths are relative, so run from inside `solution.py/`:

```bash
cd "Biomedical Evidence Graph Omission Repair/solution.py"
python cv.py            # 5-fold grouped CV; tunes thresholds -> thresholds.json, oof.pkl
python predict_test.py  # fits on full train, writes ../working/submission.csv
```

Requires `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`. Output columns: `id, relation_patch`, with
triples serialized as `SUBJECT|RELATION_LABEL|OBJECT` joined by `" ; "`.

Only the provided public fields are used — no entity/relation lookups outside `train.csv`, and every
emitted triple is validated against the document's own entity inventory and the family's endpoint-type
schema before being written.
