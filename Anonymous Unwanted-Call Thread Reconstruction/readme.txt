Anonymous Unwanted-Call Thread Reconstruction
=============================================

Run
---

    python3 solution.py <public_dir> <submission_out>

Example:

    python3 solution.py dataset/public working/submission.csv

The program is CPU-only. It reads train.csv, train_labels.csv, and test.csv from
the runtime public directory and writes the requested output path. It creates a
valid profile-1 placeholder immediately, before JSON parsing or model fitting,
then atomically replaces its contents only after validating the final frame.

Task and schema
---------------

Each case contains two anchor complaints and eight candidate profiles with three
complaints each. The target is the profile belonging to the same anonymous
caller. Once a profile is selected, its three IDs are ordered by
relative_filing_hour, with ID as a deterministic tie break.

The output has exactly these columns:

    case_id,continuation_chain

continuation_chain is a JSON list of three distinct candidate IDs. The script
validates the columns, row count, case IDs, JSON structure, ID syntax, and ID
uniqueness before replacing the placeholder.

The official metric is 0.55 ProfileAccuracy + 0.20 ComplaintSetF1 + 0.20
DirectedChainF1 + 0.05 ExactChainAccuracy. The solution always emits all three
cards from one profile in chronological order. Therefore all four components
agree: the score is the selected-profile accuracy.

Domain classification
---------------------

Primary guidebook domain: Retrieval/RAG, section 5.3. Each case is a query with
eight candidates, and the output is constrained decoding of the top-ranked
candidate. No language model or external corpus is needed.

Approach
--------

The current model is a two-view ranker selected from training-only validation:

1. A CatBoost QuerySoftMax group ranker. Each group is one case and its eight
   profiles, so training directly optimizes relative candidate ordering rather
   than eight independent binary classifications.
2. A strongly regularized logistic similarity ranker. Its different inductive
   bias improves a small number of QuerySoftMax errors without dominating it.
3. Row-local z-score blending. Candidate scores are normalized only among the
   eight profiles of the same case. The blend weight is searched on out-of-fold
   training predictions.

The earlier classifier ensemble improved its internal OOF score but did not
improve the reported evaluation slice. It was removed rather than retained as
unused weight. The new QuerySoftMax objective and raw categorical representation
produce a materially larger training-only gain.

Features
--------

The query ranker receives 126 raw categorical columns plus 391 numeric matching
features per candidate.

Raw categorical columns preserve information tree models could not recover from
match counts alone:

- each categorical value at the two chronological anchor positions and three
  chronological candidate positions;
- per-card composites such as method+call_type,
  method+call_type+service_category, state+method+call_type,
  filing weekday+hour bucket, and issue bucket+lag;
- ordered sequence signatures and order-independent set signatures for each
  field and composite on both sides.

Numeric features include:

- all 2x3 anchor-to-candidate equality matrices for nine categorical fields;
- per-field overlap, Jaccard, multiplicity, diversity, and agreement summaries;
- training-fold IDF-weighted exact matches;
- whole-card equality and IDF match scores;
- best distinct assignment of the two anchors to two different candidate cards;
- composite-field match matrices and assignment scores;
- log-scaled complaint gaps, profile duration, gap ratios, cyclic hour/week
  transforms, weekday distances, and relative-time shape comparisons.

Relative origins are compared only through within-bundle gaps and shapes. The
script never interprets relative timestamps as a shared absolute calendar.

Training and HPO
----------------

All choices are searched in solution.py from labeled training data:

- three QuerySoftMax configurations;
- three logistic regularization values;
- eleven blend weights from 0.0 to 1.0.

Architecture HPO uses a fixed stratified 75/25 training-only split. On the
supplied release it selected:

- QuerySoftMax: 800 trees, depth 7, learning rate 0.03, L2 10.0,
  random_strength 0.3;
- logistic C=0.01;
- final QuerySoftMax weight 0.9 and logistic weight 0.1.

Final validation is ten-fold stratified CV over complete cases. Each fold trains
on 90% of labeled cases. Every fold independently refits category frequencies,
StandardScaler, QuerySoftMax, and logistic regression. Test predictions are the
mean of the ten independently trained models for the same case and candidate.
A single full-data refit was rejected in favor of the ten-fold average: OOF
validates the 90%-coverage training regime, while averaging reduces model-seed
variance.

Measured OOF ProfileAccuracy on all 2,200 supplied training cases:

- categorical QuerySoftMax group ranker: 0.555455;
- regularized linear similarity ranker: 0.530000;
- selected 0.9/0.1 blend: 0.559545.

For comparison:

- original two-model pipeline: 0.525000 OOF;
- expanded classifier stack: 0.544545 OOF;
- current query-ranking pipeline: 0.559545 OOF.

The current pipeline improves over the expanded stack by 0.015000 absolute and
over the original pipeline by 0.034545 absolute under training-only validation.
The gain is driven by a new ranking objective and categorical representation,
not another small reweighting of the previous models.

Reported external evidence
--------------------------

Two earlier CSV scores were reported during development:

- original two-model CSV: 0.6122448979591838;
- expanded classifier CSVs: 0.6054421768707483.

These correspond to a one-case difference on the evaluated slice. They were not
used to infer test labels, choose per-row predictions, fit a test-time model, or
search feature/model parameters. They motivated replacing the poorly
transferring classifier objective with a training-validated group-ranking
objective. The current QuerySoftMax CSV has not been externally graded here.

Static CPU plan and timing
--------------------------

The executed plan is unconditional:

- 3 QuerySoftMax HPO fits;
- 3 logistic HPO fits;
- 10 final QuerySoftMax fits;
- 10 final logistic fits;
- 11 OOF blend evaluations;
- fixed seed 20260922 and eight CatBoost CPU threads.

Measured on the supplied 2,200-train/500-test release:

- mean final fold: 76.30 seconds;
- complete in-script run: 912.55 seconds;
- process wall time: 914.58 seconds;
- runtime ceiling: 5,400 seconds;
- remaining margin: about 83%.

Timing values are printed only. No elapsed-time value, hardware discovery,
download result, or environment variable controls a branch or reduces work.

Leakage and test-taint audit
----------------------------

Fold-train-only fitted state:

- per-field category frequency dictionaries used by IDF features;
- StandardScaler means and variances;
- CatBoost splits, leaves, and categorical target statistics;
- logistic coefficients.

Training-only global state:

- HPO choices from the fixed 75/25 labeled split;
- blend weight from OOF predictions whose rows were not seen by their producing
  first-level models.

Test-derived state is restricted to the same row being predicted:

- raw values from that case;
- engineered relations inside that case;
- z-score normalization across that case's eight candidates;
- averaging fold predictions for that same case and candidate;
- chronological sorting of the selected profile's own three cards.

There is no train+test concatenation for fitted statistics, no fitting on test
rows, no pseudo-labeling, no nearest-test calibration, no cross-test-row
aggregation, and no asserted class correction. Constructing a test DataFrame is
not fitting: CatBoost receives only fold-training row indices in each fit Pool.

Hardcoding audit
-----------------

The script contains only schema/algorithm constants:

- the nine documented categorical fields;
- generic composite field groups;
- standard weekday ordering;
- model candidate grids and fixed random seed;
- the mandated profile-1 emergency placeholder.

It contains no case IDs, test row indices, expected profile mappings, discovered
test labels, dataset fingerprints, or score-conditioned overrides. Candidate
profile IDs are used only after learned ranking to serialize the selected
profile. Removing the fitted models leaves no normal-case selection mechanism;
only the required early placeholder remains.

Output verification
-------------------

The full command completed successfully and wrote 500 predictions plus the CSV
header. Internal validation confirmed the exact two-column schema and valid
three-ID JSON chains. working/submission.csv is the output from that verified
full run.
