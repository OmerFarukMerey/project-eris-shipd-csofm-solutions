# Project Eris — My Solutions

My solutions to **Project Eris** quests on [shipd.ai](https://shipd.ai). Every folder is a
self-contained challenge: `solution.py` (the graded deliverable, run as
`python3 solution.py <public_dir> <submission_out>`), `readme.txt` with the full write-up,
`PROBLEM.md` where the platform supplied one, and a `working/` directory holding the produced
submission. Challenge datasets are not redistributed (see `.gitignore`).

Each solution trains a real model from the supplied training data on every invocation — no
pretrained lookup tables, no hardcoded answer rules, no external corpora.

## Results

11 challenges placed, all on the podium.

| # | Challenge | Rank | Local score |
|---|---|---|---|
| 1 | [Docstring Gap Restoration](Docstring%20Gap%20Restoration/readme.txt) | 🥇 1st | 0.534 char n-gram F |
| 2 | [Polyphonic Vocal Passage Event Recovery](Polyphonic%20Vocal%20Passage%20Event%20Recovery/readme.txt) | 🥇 1st | 0.200 holdout |
| 3 | [Adverse Event Reaction Code Recommendation](Adverse%20Event%20Reaction%20Code%20Recommendation/readme.txt) | 🥇 1st | 0.307 balanced MAP@5 |
| 4 | [Coastal Sensor Signature Recommendation](Coastal%20Sensor%20Signature%20Recommendation/readme.txt) | 🥇 1st | 0.496 CV / 0.520 LB |
| 5 | [Biocatalytic Product Recommendation](Biocatalytic%20Product%20Recommendation%3A%20Ranking%20Candidates%20by%20Enzyme%20Relevance/readme.txt) | 🥈 2nd | 0.258 final (0.836 quality) |
| 6 | [Ornament Sequence Recovery from Lossy Performance Views](Ornament%20Sequence%20Recovery%20from%20Lossy%20Performance%20Views/readme.txt) | 🥈 2nd | 72.6 holdout |
| 7 | [Anonymized Vocal Fragment Routing](Anonymized%20Vocal%20Fragment%20Routing/readme.txt) | 🥈 2nd | 0.443 reranked |
| 8 | [Lean Proof Patch Recovery](Lean%20Proof%20Patch%20Recovery/readme.txt) | 🥈 2nd | 0.424 3-fold mean |
| 9 | [Catalan Administrative Discourse Operator Reconstruction](Catalan%20Administrative%20Discourse%20Operator%20Reconstruction/readme.txt) | 🥉 3rd | 0.558 OOF |
| 10 | [Cross-Lead ECG Wave Landmark Recovery](Cross-Lead%20ECG%20Wave%20Landmark%20Recovery/readme.txt) | 🥉 3rd | 0.710 grouped 5-fold |
| 11 | [Biomedical Concept Evidence Ranking](Biomedical%20Concept%20Evidence%20Ranking/readme.txt) | 🥉 3rd | 0.645 OOF composite |

Scores are on my own held-out validation unless noted as a leaderboard (LB) figure; each metric is
challenge-specific, so numbers are not comparable across rows.

---

## The challenges

### 🥇 Docstring Gap Restoration

**Problem.** Generate the literal text that replaces a `[GAP]` marker inside a Python docstring.
Scoring is character n-gram F-score over orders 1–6, so partial lexical overlap earns credit even
without an exact match.

**Solution.** Fine-tune only the last four decoder blocks of CodeT5-base in its native sentinel
span-denoising format — matching the pretraining objective rather than fighting it, which is also
the right compute/variance tradeoff on CPU. Masked documentation is placed before the code so it
survives truncation. Learning rate is searched on one train-only partition and decoding on another,
with complete-document groups so duplicated sentences cannot straddle the split. Deeper decoder
adaptation, retrieval hints, a frequent-span classifier, and beam routing all failed to generalize.

### 🥇 Polyphonic Vocal Passage Event Recovery

**Problem.** Reconstruct a fixed number of missing vocal events — onsets, durations, pitch/rest, and
ties — from the surrounding melody and chord context. Scored on event matching, exact sequence, and
edit similarity.

**Solution.** Factorize into rhythm and pitch. A train-fitted duration model generates k-best rhythm
paths by dynamic programming; a CatBoost QuerySoftMax model reranks them; independent and
conditional pitch rankers score pitch/rest states; Viterbi combines emissions and transitions.
Visible boundary ties are enforced as hard constraints. Generating with a cheap structured prior and
then *learning to rank* the hard alternatives beat direct independent duration prediction, which lost
the global tiling structure (0.152 → 0.200).

### 🥇 Adverse Event Reaction Code Recommendation

**Problem.** Recommend up to five reaction codes per adverse-event report, where test reports are
temporally later than training. Frequency-balanced MAP@5 raises the value of rare codes.

**Solution.** Expand each report against all 90 codes, then feed cross-fitted balanced/unbalanced
logistic and ComplementNB predictions, token lifts, report features, and code statistics into a
LightGBM LambdaRank model. Folds are rolling-origin so validation models future prediction. The real
lesson here was diagnostic: random inner OOF manufactured *interpolation*-quality meta-features for
an *extrapolation* task, and adjacent-quarter validation failed to represent longer deployment gaps —
three locally-approved changes regressed on the real grader. Final deliverable is a diversity
rank-average of two differently-biased configurations.

### 🥇 Coastal Sensor Signature Recommendation

**Problem.** Rank five candidate summaries for a hidden six-hour coastal sensor segment using only
the before/after context. Frequency-balanced MAP@5 over candidate patterns.

**Solution.** Predict the hidden segment's 11 summary statistics from context with cross-fitted
LightGBM and Ridge regressors, then difference those predictions against every candidate. Add
physics-inspired estimates and k-NN label votes, and let a LambdaRank model combine everything; a
small frequency penalty counters the metric/popularity mismatch. Biggest single gain came from
handing the regressors regime fields already present in the query. Passing a second, differently
biased Ridge model in *as a feature* helped; fixed score-level blending did not — feature-level
fusion lets the final model learn conditional trust.

### 🥈 Biocatalytic Product Recommendation: Ranking Candidates by Enzyme Relevance

**Problem.** Score how strongly an enzyme-reaction context (substrate SMILES + EC number) supports a
candidate product molecule. Negatives are deliberately *hard* — a real product paired with the wrong
enzyme class from the same top-level family — so surface similarity between candidate and substrates
is uninformative. The metric mixes ranking AUC, calibration, and hard-negative AUC behind a steep
gate.

**Solution.** Represent each reaction in complementary views: character and SMILES-lexical TF-IDF,
reaction-*difference* text (candidate aligned to its closest substrate component, gained/lost
n-grams tokenized), Weisfeiler-Lehman molecular graph fingerprints, and numeric reaction features.
Positive-EC prototypes and a direct graph×EC logistic interaction model enzyme-conditioned activity;
CatBoost combines the heterogeneous margins. Validation groups by candidate SMILES so product
memorization can't masquerade as generalization. A fine-tuned chemical text encoder and a pairwise
neural hard-negative objective were both weaker on the untouched fold and cut entirely.

### 🥈 Ornament Sequence Recovery from Lossy Performance Views

**Problem.** Recover hidden pitch, gap, duration, and multiplicity values from complementary
symbolic views plus acoustic features, while preserving every visible component. Metric blends token
edit similarity, component similarity, and bigram overlap.

**Solution.** An ensemble of CatBoost, ExtraTrees, withheld-only bidirectional GRUs, all-label
leave-one-out GRU/LSTM models, and target-excluded all-label trees. The key idea is **architectural
target exclusion**: the head may use acoustic context and the opposite symbolic view *at* the current
event, but recurrent states from its own view stop before and resume after that event. That uses
every natural label without leaking the target into its own features, and avoids artificial masking.
Holdout climbed from 47.4 (simple view fusion) to 72.6.

### 🥈 Anonymized Vocal Fragment Routing

**Problem.** Choose and order anonymous musical fragments so the total duration exactly fills a gap
and the route matches hidden musical continuity.

**Solution.** Treat candidate *recall* and candidate *selection* as separate bottlenecks. A LightGBM
next-fragment classifier and a transformer pointer network generate complementary routes; a
route-length classifier predicts plausible cardinality; subset-sum pruning guarantees duration
feasibility; beams deduplicate by event sequence. Four route-level rankers then score the union
against exact local metric targets. Individually the generators score ~0.386/0.354, but their union
oracle reaches ~0.675 — neither is strong alone, together they make a strong candidate pool.
Event-distinct beams fixed severe crowding by aliases carrying identical events (oracle
0.588 → 0.648).

### 🥈 Lean Proof Patch Recovery

**Problem.** Given a broken Lean proof, compiler feedback, and proof state, predict the replacement
start line, deletion count, and inserted line. Text similarity dominates the metric, with location,
line-LCS, and exact-patch bonuses.

**Solution.** A line classifier proposes replacement locations. Candidate insertions are pooled from
copied lines, generic tactics, train-only nearest neighbors, alias remapping, and slotted templates.
A tactic-family classifier plus CatBoost regression rank `(start, insertion)` pairs jointly against a
metric-shaped target. Candidate diversity and joint ranking carried the gain.

### 🥉 Catalan Administrative Discourse Operator Reconstruction

**Problem.** Six discourse connectives have been removed from an administrative document and all
other words replaced with stable, collision-prone codes, so no external lookup can help. Predict the
six opaque operator tokens (`O00`–`O25`) in order. Each row also supplies a finite-field constraint:
the six operators generate 2×2 matrices mod 17 whose ordered product must reproduce a given
`boundary_matrix`.

**Solution.** Three independently trained families vote per position — a focused local-phrase TF-IDF
+ multinomial logistic classifier, a boundary-aware classifier with side/distance/position features,
and a compact from-scratch Transformer over learned bucket embeddings. Decoding uses posterior
*marginals* under the finite-field constraint rather than max-score, which preserved the entire 15%
matrix term while improving accuracy. Five-fold OOF is grouped by connected components of exact
shared gap sections. Notably, context-free matrix search is not a prediction method — it can earn the
matrix term while failing to identify the actual discourse path.

### 🥉 Cross-Lead ECG Wave Landmark Recovery

**Problem.** Given one transformed context ECG lead, predict P/QRS/T onset, peak, and offset
landmarks for a *different* lead. Scoring is 60% event F1 at ±8 samples, 30% at ±20, and 10%
sequence-order LCS, with patient-held-out test records.

**Solution.** Classical signal processing proposes beats and context anchors but never emits a final
answer — it serves as a coordinate system and candidate generator. ExtraTrees and CatBoost
classifiers decide target-lead landmark *presence*; regressors predict target-lead timing *offsets*.
Compact structural features and dense waveform neighborhoods turn out to be complementary.
Patient-grouped OOF predictions drive per-landmark blend and threshold search. CatBoost specialists
lifted grouped OOF from 0.696 to 0.710, mostly on faint landmarks and timing.

### 🥉 Biomedical Concept Evidence Ranking

**Problem.** Rank candidate biomedical documents for a query. The composite is deliberately
top-heavy: NDCG@3, fully-relevant hit at rank one, rare concepts, lexical-overlap distractors, long
queries, and worst-track performance.

**Solution.** Fit every text transform only on train-referenced documents. Build lexical TF-IDF/BM25,
LSA, PPMI token embeddings, ColBERT-style soft MaxSim, NMF topics, and within-slate
semantic-versus-lexical contrast features, then train a single CatBoost YetiRank model over five
query-grouped folds. Against adversarial lexical distractors the winning move was explicitly
modelling the residual "semantic similarity beyond overlap" instead of adding more overlap variants.
A single sharp ranker beat model blending — averaging diluted confident rank-one choices.

---

## Repository layout

```
<Challenge Name>/
├── solution.py      the graded deliverable (trains from scratch each run)
├── readme.txt       full write-up: contract, approach, validation, leakage audit
├── PROBLEM.md       platform-supplied problem statement (where provided)
├── dataset/public/  challenge data (gitignored, not redistributed)
└── working/         produced submission.csv
```

- [`problems_and_approaches.md`](problems_and_approaches.md) — the cross-problem method write-up:
  contract-first, evidence-guided structured learning, and the lessons that generalized across every
  challenge.
- [`PROMPT.md`](PROMPT.md) — the working contract used while solving.
- `template/` — scaffolding for starting a new challenge.

## License

[MIT](LICENSE). Challenge datasets remain Shipd challenge content under
[shipd.ai/terms](https://shipd.ai) and are not included here.
