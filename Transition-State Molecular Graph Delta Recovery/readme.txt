Transition-State Molecular Graph Delta Recovery — solution notes
===============================================================

RUNTIME CONTRACT
----------------
Platform invocation:  python3 solution.py <public_dir> <submission_out>
  public_dir     = sys.argv[1]  (contains train.csv, test.csv)
  submission_out = sys.argv[2]  (path to write the submission CSV)
Robust to invocation with NO arguments (some platform checks run the script bare):
find_public_dir() uses argv[1] when valid, else probes ./dataset/public, ./public,
., ./dataset, ./data, then a shallow walk, for a dir with train.csv + test.csv. The
CSV is written to argv[2] when given AND always to ./working/submission.csv (the
path PROBLEM.md names); output parent dirs are created first. All preprocessing,
model TRAINING, and inference happen inside this one script, from the raw CSVs, on
every run. CPU-only; no GPU/accelerator, no network, no external data, no runtime
installs, standard preinstalled libraries only (numpy, pandas, torch). Deterministic
(fixed seeds). Runtime ~6 s.


PROBLEM
-------
Each row packages four INDEPENDENT atom-pair evidence records ("probes"), each from
a different reaction. For every probe we emit its molecular-graph edit — whether the
reactant bond is broken and/or a product bond is formed, with exact old/new orders —
as a JSON program with broken_bonds and formed_bonds lists. Released per-probe
evidence is a handful of discretized fields (elements, formal charges, aromatic
count, reactant degree bands, reactant bond order, reactant distance band, transition
motion band). Whole molecules, atom maps, SMILES, and 3-D coordinates are withheld.

Grading maps every probe handle to its public symmetry-group index and builds two
multisets of (group_index, order): one from broken_bonds, one from formed_bonds. Row
fidelity is 1 only if BOTH multisets exactly equal the hidden ones, else 0. Final
score = mean row fidelity over the 300 test rows.


KEY STRUCTURAL FACTS (established from train.csv)
-------------------------------------------------
1. A broken bond's `order` is ALWAYS the probe's reactant_bond_order (verified: 0
   mismatches over 3120 probes). No probe ever appears twice in either list.
2. So the entire target for a probe collapses to ONE quantity: the product bond
   order p (0.0 == bond absent). With the known reactant order r:
       p == r         -> unchanged;   r>0 and p==0   -> pure cleavage;
       r==0 and p>0   -> pure formation;   r>0,p>0,p!=r -> order replacement.
   The learning problem is therefore: predict p per probe, then serialize.
3. A bond order only ever steps one rung on the ladder 0-1-2-3 (r=0->{0,1},
   r=1->{0,1,2}, r=2->{1,2,3}, r=3->{2,3}); 1.5 (aromatic) never changes.
4. The evidence is deliberately coarse. Within r=0, formation is ~50/50 and is
   balanced across element/degree/charge, so those fields are nearly uninformative
   for r=0; the usable signal is the (distance, motion) geometry. For r>0 the motion
   band (extension => cleavage) plus element/degree (whether an increase to a double
   bond is chemically possible; H-X bonds never strengthen) carry the signal. r=2 is
   an essentially even weaken-vs-stay split. Changes are independent across probes
   (~Binomial(4, 0.5)), so there is no row-level count constraint.

These facts are used to define the prediction TARGET (p) and the output
serialization — they are not hardcoded answers; the mapping from evidence to p is
learned by the model below.


APPROACH / MODEL ARCHITECTURE
-----------------------------
A SEPARATE feed-forward neural network is trained per reactant order r, from
scratch, inside the submission script, on the provided train split. Specializing by
r beats one joint network in cross-validation (+0.26% row fidelity, paired, lower
variance) because the informative evidence differs sharply by r and several released
fields are deliberately balanced (uninformative) for a given r.

* Per-r feature selection (validated by CV ablation): the r=0 formation net is fed
  only (distance, motion, formal_charge) — element and degree are ~50/50 balanced
  for r=0, so excluding them stops the net fitting that noise; the r=1 net is fed
  (motion, element, degree, distance, charge); the r=2 net is fed (element, degree).
* Feature encoding: the chosen fields are one-hot encoded. Every vocabulary is built
  on TRAIN ONLY; a value seen only at inference (e.g. the N-N element pair, absent
  from train) maps to a per-field "unknown" slot rather than causing any test fit.
* Network: per r, a 2-hidden-layer net (Linear 64 -> ReLU -> Dropout 0.3 -> Linear
  64 -> ReLU -> Dropout 0.3 -> Linear |classes|), softmax over exactly the product
  orders OBSERVED for that r in train (learned, not hardcoded) so impossible
  one-rung-plus transitions get zero mass automatically. PyTorch, full-batch, Adam
  (lr 5e-3, wd 1e-4), cross-entropy, 250 epochs, fixed seeds.
* Ensemble: 5 nets per r (different seeds), class probabilities averaged (training /
  ensembling multiple models in one script is allowed; it stabilizes the decode).
* Fallback: an r with fewer than 20 train probes (e.g. r=3, which never occurs in
  test) uses that r's empirical prior instead of a net.
* A training-time wall-clock safeguard stops early and proceeds to inference if a
  large budget is ever exceeded (never triggered here; training takes ~6 s).

One DECODE step sits on top of the learned probabilities and only SUPPORTS the model:
* Symmetry-group multiset decoding: members of a group share identical evidence, so
  the model produces one probability vector for the group. Because grading compares
  group-index multisets, we emit the maximum-likelihood MULTISET of product orders
  of the group's size under i.i.d. draws from the model (enumerate multisets,
  weight by the multinomial coefficient). For a size-2 group at ~50/50 this yields
  {form, noform} instead of {form, form} — the split the multiset grader rewards.
  (The old separate "reachable-order mask" is now intrinsic: each per-r net can only
  output orders observed for that r.)


VALIDATION STRATEGY
-------------------
Row-grouped 5-fold cross-validation (all four probes of a row kept in one fold),
reimplementing the exact group-index multiset metric the grader uses, averaged over
multiple shuffle seeds. Everything (vocabularies, networks, per-r classes/priors) is
fit on the training folds only; validation rows are used for prediction only.

    row fidelity (grouped 5-fold CV)   ~13.0-13.2%   (per-r nets, +/- 0.4)
    per-r nets vs one joint net        +0.26% paired (wins 4/6 seeds, lower variance)

Ceiling check: decoding each row with the in-sample empirical conditional as a
stand-in for the true distribution gives an optimistic Bayes ceiling of ~15.3%. The
CV result sits just under it, so the remaining gap is dominated by irreducible label
noise from the intentionally lossy discretization, not model capacity. The symmetry-
group multiset decode is the lever that lifts the graded metric: it lowers per-probe
accuracy (~57% vs ~59%) but raises row fidelity (from ~11.9% with argmax-copy to
~13%), because ~29% of rows contain a non-singleton symmetry group. (A first
compliant submission using a single joint net scored 0.1923 on the public test,
which sits above the CV rate because the test draw is r=0-heavy; the per-r net is
CV-better in expectation and is the shipped model.)


WHAT WORKED
-----------
* Collapsing the whole target to a single per-probe product-order classification
  once broken-order == reactant-order was verified — turns a variable-length JSON
  generation task into a clean, learnable 5-class problem.
* Trained feed-forward nets over one-hot evidence, SPECIALIZED PER REACTANT ORDER:
  giving each r only its informative fields (so r=0 ignores the balanced-noise
  element/degree) beat a single joint net by +0.26% paired CV and reduced variance.
* Ensembling 5 seeds per r for calibrated probabilities feeding the multiset decode.
* Max-likelihood multiset decoding for symmetry groups (+~1% row fidelity); the
  per-r output classes intrinsically enforce the one-rung reachable-order constraint.


WHAT DID NOT WORK
-----------------
* Pushing per-probe accuracy past ~59%: the r=0 "mid"-distance formation decision and
  the r=2 weaken-vs-stay decision are ~50/50 in the released evidence and cannot be
  beaten above chance — a property of the deliberately coarse bands, confirmed by the
  ~15% Bayes ceiling.
* Adding element/degree/charge as strong drivers of the r=0 decision: they are
  balanced ~50/50 there and add noise rather than signal.
* Looking for a row-level count constraint: none exists (changes are independent,
  Binomial(4, 0.5)).
* Pushing past ~13% CV / ~59.7% per-probe: an exhaustive, cross-validated sweep found
  that NONE of the following moved the metric beyond noise (all ~13.0-13.3% CV):
  explicit interaction/cross features; LightGBM instead of the NN; an NN+LightGBM
  probability blend; larger NN ensembles (7, 12 nets were flat-to-worse vs 5); an
  empirical (non-i.i.d.) group model — symmetry-group members are in fact i.i.d.
  (r=0 size-2 pairs observed 20/31/17 vs i.i.d.-expected 18.4/34/15.6), so the
  multiset decoder is already optimal. The ONE change that did help (and is shipped)
  was per-reactant-order specialization (+0.26% paired). The evidence is otherwise
  saturated: e.g. probes with byte-identical released features "C-C, 3+3+, stable"
  split 303 stay / 208 strengthen, an irreducible ambiguity. A single joint net
  scored 0.1923 on the public test (above the ~13% CV rate because the test draw is
  r=0-heavy: 60% vs 53% train, with 2x as many easy all-r=0 rows); the per-r net is
  CV-better in expectation and is the shipped model.


LEAKAGE STATEMENT
-----------------
Every transform and statistic is fit on TRAIN ONLY; test is used for inference only.
Specifically:
* the per-r one-hot vocabularies, the network weights, and the per-r class sets /
  priors are all computed from train_rows before test.csv is read;
* read_targets() (which parses target_patch_json) is called ONLY on the train
  DataFrame; the test path (load_rows) reads only bond_probe_panel_json and
  answer_constraints_json and never touches any answer/label column;
* no train+test concatenation, no popularity/frequency statistic over test, no
  pseudo-labelling, no test-time adaptation, no threshold/hyperparameter selection on
  test; unknown test categories fall back to an explicit unknown slot;
* random seeds are fixed (numpy + torch); the script reads only files under
  public_dir and writes only the submission path(s).

Every test-data use in solution.py:
  1. Read test.csv AFTER the models are fully trained.
  2. Per row, featurize each probe with its reactant order's train-built vocabulary.
  3. Call the frozen per-r ensemble to get product-order probabilities (predict).
  4. Multiset-decode per symmetry group, serialize, and write the CSV. No state is
     computed across test rows.


PRE-SUBMIT AUDIT / OUTPUT CHECK
-------------------------------
Run from the challenge directory:

    python3 solution.py ./dataset/public ./working/submission.csv

The generated submission was verified to have:
* exactly 300 rows and exactly the columns id,target_patch_json;
* unique ids exactly matching test.csv;
* valid JSON; each entry has keys {probe_id, order} with a nonzero order in
  {1.0,1.5,2.0,3.0}; every broken order equals that probe's reactant order;
* deterministic output across repeated runs (fixed seeds).
* solution.py is the ONLY .py file in the challenge directory and imports no local
  modules.

Compliance note (Project Eris guidebook): the solution's core is a set of genuinely
trained neural networks (one per reactant order) — removing them leaves no working
solution (guidebook 4.3 test: a no-model per-r prior baseline scores ~5% row fidelity
vs ~13% with the nets). The per-r class restriction and the symmetry-group multiset
decode are output-formatting steps that consume the models' learned probabilities;
they support the models and do not replace them.
