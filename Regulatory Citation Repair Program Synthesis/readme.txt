Regulatory Citation Repair Program Synthesis — solution notes
==============================================================

APPROACH
--------
The script learns a one-operation JSON patch program from the 5,000 training
rows. It first predicts the repair operation family from the amendment note,
then ranks valid row-local node or range candidates for that family. Every
prediction edits only the citation alias named in the amendment note.

Training-data analysis established the output constraints:

1. Every canonical program has one operation.
2. The amendment note identifies the scored citation and anchor using stable
   syntax, but its repair-pressure wording must be learned.
3. SET_SCOPE always points to the row's unique scope root.
4. Other families require a probabilistic candidate model: target choice
   depends on heading tags, size, hierarchy order, anchor position, current
   citation state, and the other local citation cards.

The submission therefore always emits exactly one operation. This is both
valid and efficient: when the repaired state is exact, ExactOps and Efficiency
also receive full credit.

MODEL / ALGORITHM
-----------------
Operation-family model:

- CountVectorizer word unigrams/bigrams followed by multinomial logistic
  regression.
- Vocabulary and coefficients are fit inside solution.py on train.csv only.
- Citation and anchor aliases are extracted with regex because their syntax is
  deterministic; regex does not choose the repair family or target.
- Five-fold operation-family accuracy is 1.0000.

Node-choice model (SET_UNIT, ADD_TARGET, DROP_TARGET):

- A shared conditional-logit/grouped-softmax ranker trained on 3,000 rows.
- Candidate sets are all section nodes for SET_UNIT and ADD_TARGET, and the
  current ordered target list for DROP_TARGET.
- Shared node features encode learned train-only heading-tag vocabulary,
  length tags, size bucket, multi-tag counts, tag-by-size interactions, and
  relation to the containing scope.
- Family-specific features encode signed anchor distance, hierarchy position,
  source distance, current-target distance/membership/span, and use by other
  citation cards. DROP_TARGET also gets ordered-list, size-extreme,
  distance-extreme, and ordinal-extreme features.
- Grouped likelihood is optimized deterministically with bounded L-BFGS and
  L2=0.0003.

Range model (SET_RANGE):

- Conditional logit over every ordered section pair with start ordinal <= end
  ordinal.
- Because a SET_RANGE repair replaces the entire target list, the two
  endpoints' salience blocks (train-fitted tag/size plus hierarchy position and
  anchor/source relation) are POOLED (summed), and the endpoints' current-target
  relation features are dropped as noise. Endpoint direction is instead carried
  by pair features: width, whether the span contains the anchor/source, and
  signed anchor offsets for both the start and the end ordinal.
- Pooling the endpoints and pruning the stale current-target and
  cross-citation-overlap features lowered estimation variance on the ~1,000
  range rows. A leaderboard-aware decoder then lifted the range family from
  0.3735 to 0.3752 in five-fold CV.
- L2=0.01. Out-of-fold calibration selected temperature 1.3.
- Decoding uses:
    0.60 * P(pair)
    + (0.40 / 3) * (P(start endpoint) + P(end endpoint)).
  Pure expected-row-score decoding uses pair weight 0.75. The lower 0.60 weight
  was selected against the complete metric (including its lowest-quartile
  term), and reduces double-endpoint misses on uncertain range rows.

SET_SCOPE uses the unique scope node. This deterministic structural step
supports the trained models; learned models choose the operation family and
all non-trivial node/range targets.

VALIDATION
----------
Five-fold KFold validation on train only, shuffled with seed 42. Each fold
refits the text vocabulary/classifier, feature schema, node ranker, and range
ranker without using its validation rows. Scoring simulates operation
application, SemanticF1, ExactFinal, ExactOps, Efficiency, worst-family mean,
and the lowest-quartile tail mean.

Per family (mean row score / ExactFinal rate):

  SET_SCOPE    1.0000 / 1.0000
  DROP_TARGET  0.7213 / 0.6790
  SET_UNIT     0.5494 / 0.4850
  ADD_TARGET   0.5535 / 0.4540
  SET_RANGE    0.3752 / 0.2740

  mean row score          0.6399
  worst-family mean       0.3752
  lowest-25% tail mean    0.1247
  estimated final score   0.5354

Score history (train-only five-fold estimate): linear baseline 0.5272; adding
tag-by-size node interactions, stronger node regularization, and calibrated
range decoding reached 0.5305; pooling range endpoints and pruning stale range
features reached 0.5348; leaderboard-aware range decoding reached 0.5354. The
previous 0.5348 version scored 0.5445 on the challenge.

LEAKAGE STATEMENT
-----------------
- solution.py reads and completely fits on train.csv before it reads test.csv.
- The operation vectorizer/classifier, heading-tag and length-tag
  vocabularies, both rankers, regularization choices, and calibration
  temperature are all fit or selected using train only.
- test.csv is used only by op_model.predict (its pipeline transform plus
  predict), row-wise JSON parsing/feature transformation, and ranker matrix
  prediction.
- Every structured feature for a test row is computed from that one row only.
  No aggregate over multiple test rows is computed.
- There is no train+test concatenation, merge, append, fitting, calibration,
  pseudo-labeling, distribution matching, or test-derived vocabulary.
- Unknown note words or node tags are ignored by train-fitted vocabularies.

WHAT WORKED / WHAT DID NOT
--------------------------
Worked:

- Learning the amendment-pressure vocabulary instead of hardcoding its
  phrase-to-operation mapping.
- Grouped softmax, which trains on the actual within-row candidate decision.
- Sharing tag/size salience across the three node-choice families.
- Signed anchor-distance features, especially ADD_TARGET's directional bias.
- Tag-by-size interactions and a small L2 penalty for node choice.
- Leaderboard-aware range decoding with train-only temperature and weight
  calibration.

Did not improve five-fold validation:

- LightGBM/CatBoost-style boosted rankers. A correctly trained nonlinear
  residual ranker also hurt UNIT/ADD; a DROP-only residual gained one fold and
  lost four.
- Keeping range endpoints separate, composite endpoint likelihood, expanded
  role/boundary pair features, and repeated-fold bagging. Despite a real tag
  asymmetry (starts favor semantic headings, ends favor plain length-tagged
  sections), each raised variance or diluted the full-data fit.
- Exact-hierarchy empirical priors: only about 15% of validation rows had a
  same-family hierarchy repeat, and repeated labels were stochastic.
- Adjacent-L2 range ensembles: effectively neutral relative to one L2=0.01 fit.
- Family-weighted node likelihood and partially family-specific tag salience.
- Signed first/last stale-target features for UNIT/ADD improved seed-42 CV but
  regressed on a second shuffled split, so they were not shipped.
- Deterministic target rules; the learned probabilistic rankers were stronger.

PRE-SUBMIT AUDIT
----------------
- Test dataframe use sites: read after all fitting; op_model.predict;
  parse_rows row-wise transform; node_family_feats/range_feats row-wise
  transforms; matrix-score prediction. No other test use exists.
- No statistic from test rows flows into a vocabulary, feature schema,
  estimator, regularization value, temperature, or threshold.
- No train+test concatenation exists.
- Every emitted program has exactly the top-level key "ops", contains one
  allowed operation, and uses aliases present in that row.
- Only the citation named in the note is edited, preventing collateral damage.
- Seed is fixed at 42. Optimization is deterministic.
- The script reads only from public_dir and writes only submission_out,
  creating the output parent directory first.
- solution.py is self-contained and imports no local files.
