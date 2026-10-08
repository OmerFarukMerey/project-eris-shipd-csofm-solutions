Henkin Dependency Recovery - solution notes
============================================

Run:  python3 solution.py <public_dir> <submission_out>
Only inputs read: <public_dir>/formulas.csv, test.csv, test_hints.csv, train_labels.csv.gz, formulas/*.dqx.gz.
Only output written: <submission_out>. No downloads, no pretrained weights, no external data.


1. Task facts taken from PROBLEM.md
-----------------------------------
Submission schema: CSV with exactly two columns, in this order: id,deps. One row per test.csv id (8,791 rows).
  deps = space-separated universal variable ids of the query's own formula, or the literal {} for the empty set.
  The script writes ids only from that formula's "u" line, and writes {} for an empty prediction.
Metric: for each query s(e) = sqrt( J(P,D) * J(U-P, U-D) ), with J(empty, empty) = 1.
  The final score is a weighted mean: the 8 scored families have equal weight, formulas have equal weight
  within a family, and queries have equal weight within a formula. cnf_lifted is train-only and unscored.
  The in-script validation implements exactly this weighting.
Domain: guidebook 5.6 (formal logic / graph structure, no dedicated section). PROBLEM.md forbids pretrained
  models and downloads, so I treat it as from-scratch (5.5): every model is trained from scratch inside the script
  on the provided data. It is not a fine-tuning challenge.
Challenge rules followed: no outside copies of the benchmarks, no undoing of the renumbering, no hand-labelling.
  The test hint lines (test_hints.csv) are a provided input. The task statement explicitly frames them as partial
  supervision ("Survivors are given to you as hints. Your job is to reconstruct the lost lines").
Runtime/hardware: PROBLEM.md does not state them, so I use the guidebook ceiling of 1.5 h (5,400 s).


2. Approach
-----------
The task is framed as link prediction (existential -> universal) on the clause graph, with partial supervision
from the surviving hint lines of the same formula.

Training episodes: train formulas carry full labels. For each train formula the script simulates the documented
damage model 4 times (fixed seeds): each line survives with probability 8% and becomes a hint, and the queries
are up to 400 lost strict lines plus about half as many (at least 3) lost full lines. The 8%, 400 and "half / at
least 3" values come from the PROBLEM.md description of how test was built. They define the training
distribution only and do not fix any output. The result is 916 train episodes and 49,959 training queries.

Per-formula structure (deterministic transforms of each formula's own CNF, with nothing fitted):
  - per-variable counts: degree, polarity counts, clause-length counts, number of universal neighbours
  - Tseitin gate detection: how often a variable is the output or an input of an AND/OR definition; equivalences
  - BFS distances from every universal to every variable, both over all paths and over paths that avoid other
    universals; universal co-occurrence counts
  - polarity-aware Weisfeiler-Lehman colours (4 rounds, vectorised multiset hashing)
  - one-hop and signed two-hop clause-shape signatures
  - random-walk profiles from every universal (1-4 steps)

Hint-dependent features, built only from the query's own formula and that formula's hint lines:
  - kNN votes from hinted existentials (distance-profile and feature similarity)
  - same-WL-colour hint votes
  - random-walk propagation of hint memberships over the existential clause graph (1-3 hops)
  - signed (same-sign and opposite-sign) co-occurrence votes

Models (all LightGBM, trained in-script):
  Stage S - "same dependency set" model on (query, hinted existential) pairs, about 45 pairwise features (WL
    colour equality, signature / profile / walk distances, shared clauses, two-hop overlap, ...). Its outputs are
    turned into learned votes: a similarity-weighted membership vote per universal, the nearest-hint set,
    noisy-or, and the fraction of full hints. On train these are out-of-fold (5 grouped folds); for test they
    come from the model refit on all train episodes.
  Gate  - P(query depends on all universals), per query, using the query-level features plus the learned votes.
  Pair  - P(u in D | query is strict), per (query, universal), using the pair features plus the learned votes.
          Trained on strict queries only.
  The gate and the pair model are each an average of 3 LightGBM models that differ only in their random seeds.

Decoding (per query, using only that query's model outputs):
  1. Candidates are the pair model's top-k sets, k = 0..|U|-1.
  2. The expected metric of each candidate is estimated by Monte Carlo: 256 samples from the pair model's
     Bernoulli posterior, with a fixed seed per query (crc32 of formula id and variable).
  3. Predict "all universals" if p_full + b > (1 - p_full) * E[best strict set]; otherwise output the best
     strict set.
  The single decode knob b is searched in-script on the out-of-fold train predictions against the exact metric
  (grid -0.30..0.30). In the final run it came out at b = -0.3 (+0.002 over b = 0). On the nested
  design-held-out evaluation the curve is flat within +-0.001, so this knob barely matters.

Static plan, every run: 4 draws x 229 train formulas; 5 grouped folds for stage S and for gate/pair OOF;
3 seeds per gate/pair model; final refit of all models; LightGBM rounds 300 (sim), 300 (gate), 400 (pair);
256 MC samples per query.

Deterministic execution:
  - The script never reads the clock (no time module), and it has no try/except fallback paths, worker pools
    or GPU code.
  - Nothing branches on the environment (hardware, thread count, elapsed time or download success).
  - Thread counts are pinned: OMP/OPENBLAS/MKL/NUMEXPR are set to 8 before numpy/scipy are imported, and
    LightGBM uses num_threads=8.
  - LightGBM runs with deterministic=True and force_row_wise=True, with seed, bagging_seed,
    feature_fraction_seed and data_random_seed all fixed (0, 1, 2 for the three ensemble members).
  - Every random draw uses an explicitly seeded numpy Generator: crc32-derived seeds for the training episodes
    and for each query's Monte-Carlo decode.
  - The schema-valid "all universals" placeholder is written unconditionally right after reading test.csv, then
    overwritten unconditionally with the model predictions.
  - Check: two full runs with different PYTHONHASHSEED values produced byte-identical submission.csv files
    (sha256 0328f473...6c7c).

Measured runtime on a 16-core laptop: 333 s wall-clock for the whole script, measured from outside (the script
itself contains no timers). Against the 5,400 s ceiling this leaves about 94% margin.


3. Validation
-------------
In-script validation: folds are grouped by family + number of universals, and each random_dqbf formula is its
own group. The score is out-of-fold over all 916 train episodes, using the exact PROBLEM.md metric and weighting
(cnf_lifted excluded):

  OOF score 0.7941 (0.7922 with b = 0)
    bloem_synthesis      0.989
    bounded_synthesis    0.896
    partial_equivalence  0.836
    ramsey               0.821
    random_dqbf          0.477
    scholl_henkin        0.711
    succinct_graph       0.693
    tentrup_synthesis    0.930

This in-script CV is optimistic: test formulas come from designs that never appear in train. The previous
version (2 draws, single model) reported 0.797 here but scored 0.7103 on the leaderboard.

Offline yardstick for design shift (development only, not part of the script):
  - Within each family, train formulas are clustered into "designs" on scale-free CNF descriptors (fraction of
    2/3/4+-literal clauses, log clause/variable ratio, log universal/variable ratio, gate-output fraction,
    degree-0 fraction, mean clause length).
  - Whole designs are held out (outer 5 folds). The full pipeline runs inside each training part, including the
    stage-S stacking on #universal folds exactly as in the script.
  - bloem_synthesis is a single design, so it uses #universal groups; random_dqbf is held out per formula.
  - The previous version scored 0.7213 on this evaluation (leaderboard 0.7103).
  - Every change was scored on the same held-out draws, under both this design hold-out and the #universal
    grouping. It was kept only if it improved both.

                                      design held out   #universal groups
  previous version (2 draws, 1 seed)       0.7213            0.7936
  this version (4 draws, 3 seeds)          0.7302            0.7990

References: predicting all universals for every query scores about 0.33. Run-to-run model noise is about +-0.005
on these evaluations, and per-family scores swing much more (few designs per family).


4. Leakage statement
--------------------
- Every model (stage S, gate, pair) and the decode knob are fit on train episodes only. Test rows only go
  through transform -> predict.
- Train and test are never concatenated. No scaler, encoder, vectorizer, PCA or clustering is fitted anywhere.
  The per-formula features are deterministic functions of one formula's own CNF (BFS, WL hashing, gate
  detection, walks), with no state carried across formulas.
- Each test query's features use only:
  (a) its own formula's CNF
  (b) its own formula's provided hint lines (an input file)
  (c) the query variable itself
  A query's features and prediction do not depend on which other queries exist: the hint subsampling is seeded
  from the hints only, and the Monte-Carlo seed is per query.
- No statistic is computed over test predictions. There is no calibration to a test distribution, no
  pseudo-labelling, and no use of the query count per formula.
- Risk note: the test hints are used as inputs. The task explicitly provides them as partial supervision for
  reconstructing the lost lines, and they are used per formula exactly as simulated hints are used in training.


5. Hardcoding statement
-----------------------
- No discovered generation rule is encoded as an output. Examples of rules that are NOT asserted:
  "Tseitin outputs depend on everything", "the s- and s'-copies split the universals", "random-DQBF co-occurring
  universals are dependencies". Each of these is only a feature, and the trained gate and pair models decide.
- Constants:
  * damage-model values (8%, 400, half / at least 3): from the PROBLEM.md specification, used only to simulate
    training episodes
  * feature-engineering caps (distance clip 30, hint subsample sizes)
  * standard LightGBM settings (learning rate 0.05, leaves 31/63, not tuned offline)
  * the decode bias b, searched in-script
  No tuned threshold or blend weight is pasted in.
- Strip-the-ML test: with all trained models removed, the pipeline produces no prediction at all. The decoder
  needs the gate probability and the pair-model probabilities; the only remaining output path is the
  placeholder "all universals" file. Hint votes and WL matches are features into the models, never answers.
- Lookups present: family -> integer code (a categorical model input) and universal id -> column index
  (bookkeeping). Neither decides an output.


6. What worked / what did not
-----------------------------
Early development (in-script #universal CV):
  - LightGBM gate + pair models with distance/gate/co-occurrence features: 0.737
  - polarity-aware WL colours and same-colour hint votes: 0.755 (bounded 0.77->0.83, tentrup 0.75->0.84)
  - random-walk hint propagation and walk proximity: ramsey 0.72->0.80
  - signed co-occurrence votes: small, kept (0.780->0.781)
  - learned query-hint "same set" stage: 0.781->0.796 (bounded 0.85->0.90, tentrup 0.89->0.94)
  - expected-metric decoding (instead of a 0.5 threshold): large gain on random_dqbf / partial_equivalence
  - dropped: a set-level reranker over {top-k sets} plus {hinted sets} (+0.001 to +0.004, noise level)
Second round (nested design-held-out / #universal evaluation, same held-out draws; baseline 0.7213 / 0.7936):
  - kept: 4 simulated draws instead of 2 -> 0.7284 / 0.7969
  - kept: 4 draws + 3-seed gate/pair ensemble -> 0.7302 / 0.7990 (shipped)
  - not shipped:
    * 6 draws + 3 seeds -> 0.7262 / 0.8063 (worse on the design hold-out, so 4 draws was kept)
    * XOR-triple features for the gate -> 0.7216 / 0.7912 (gate log-loss improved, final score did not)
    * family-balanced training weights -> 0.7164 / 0.7893
    * dropping the unscored cnf_lifted family from training -> 0.7243 / 0.7811
    * averaging pair probabilities within hint-revealed universal blocks -> 0.7215 / 0.7919
    * stacking stage S on design-grouped folds (an earlier variant) -> scored below 0.7103 on the leaderboard;
      it makes the similarity features look noisier in training than they are at test time
  - diagnosis: on unseen designs the full-vs-strict gate is the main loss (a perfect gate adds about +0.04,
    mostly succinct_graph, ramsey and partial_equivalence). The second Ramsey design in train (one unique set per
    strict variable) has no counterpart when the first is held out, so Ramsey transfer is weak.
Known limits:
  - random_dqbf: dependencies beyond direct co-occurrence look random. On test the model mostly predicts {}
    for these, which matches the high empty-set rate in that family's test hints.
  - succinct_graph and parts of bounded_synthesis contain structurally symmetric copies (e.g. state vs
    next-state annotations) that no local structure can tell apart without a nearby hint.
