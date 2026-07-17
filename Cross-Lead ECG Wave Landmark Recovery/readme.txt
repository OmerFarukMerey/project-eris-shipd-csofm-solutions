Cross-Lead ECG Wave Landmark Recovery — solution notes
======================================================

PROBLEM
-------
Each row contains a transformed 2-second, 500 Hz context-lead ECG signal and the
identity of a different withheld target lead. The output is the variable-length,
sample-sorted sequence of P/QRS/T onset, peak, and offset landmarks for that
target lead. The official row metric is:

    0.60 * F1 at +/-8 samples
  + 0.30 * F1 at +/-20 samples
  + 0.10 * sequence-order LCS similarity

The train/test split is by patient record, so validation must also keep every
window and target row from one record in one fold.


APPROACH
--------
The solution is a supervised beat-level landmark model trained from scratch on
the supplied train split. A generic ECG transform proposes candidate heartbeats
and context-lead landmark anchors. Those anchors are not emitted directly. Two
learned Extra-Trees model families decide which target-lead landmarks exist and
regress their target-lead sample positions.

Pipeline:

1. Per signal, apply zero-phase high-pass, low-pass, 7-28 Hz band-pass, 8 Hz
   low-pass, derivative, and derivative-energy transforms. Every normalization is
   based only on that one observed signal.
2. Find generic QRS candidates from a derivative-energy envelope. Around each
   candidate, create generic context P/QRS/T anchors with no challenge-specific
   first/last-record suppression rule.
3. Associate train target events with candidate beats. This produces one
   supervised beat example with nine presence labels and up to nine target sample
   offsets.
4. Fit two complementary Extra-Trees ensembles:
   * compact model: beat geometry, train-vocabulary context/target lead one-hots,
     window/beat position, and generic landmark anchors;
   * full model: all compact inputs plus 370 multi-scale waveform samples.
5. Fit CatBoost specialist models: nine balanced presence classifiers use native
   train-derived lead/window/beat categories, while nine MAE timing regressors
   refine individual landmarks from the compact feature view. CatBoost is forced
   to CPU mode and writes no files.
6. Generate patient-grouped out-of-fold predictions for every model. Train-only
   coordinate search selects the compact/full Extra-Trees blend, per-landmark
   CatBoost blend weights, and event thresholds against the official metric.
7. Refit every selected model family on all train beats, then transform, predict,
   decode, and discard one test row at a time.

The trained models are essential: the signal code only proposes beat/landmark
features. Event presence and final target-lead locations come from fitted
classifiers and regressors; removing the learned models leaves no submission
prediction pipeline.


MODEL AND FEATURE DETAILS
-------------------------
Extra-Trees base models:
* one nine-output ExtraTreesClassifier with class_weight="balanced"
* compact/full classifier min_samples_leaf=2/8, max_features=0.65
* three three-output ExtraTreesRegressor timing models per view
* compact/full regressor min_samples_leaf=1/5, max_features=0.75
* fold-only median imputation for a missing edge onset/offset
* 320 trees per estimator

CatBoost specialist models:
* nine CatBoostClassifier models with balanced train-fold class weights
* native categorical context lead, target lead, window, beat ordinal, and count
* nine CatBoostRegressor models with MAE loss, trained only where the landmark is
  present
* depth 6, learning rate 0.04, 450 iterations, CPU task type
* every CatBoost file writer, GPU path, and external-data path disabled

Waveform features:
* robust per-signal 95th-percentile amplitude normalization
* high-passed morphology, 35 Hz low-pass morphology, 7-28 Hz QRS band, 8 Hz
  P/T morphology, low-pass gradient, and smoothed QRS energy
* fixed sample neighborhoods from 240 samples before through 320 samples after a
  candidate, with denser sampling around QRS

Structural features:
* candidate location and distance to the signal edge
* previous/next RR interval, beat count, beat ordinal, and remaining-beat count
* context-lead and target-lead one-hot vectors whose vocabulary is built from
  train only; an unknown inference token maps to all zeros
* window index as a model input, not a hardcoded label-generation rule
* generic context onset/peak/offset anchors and availability indicators

All final samples are rounded to integers, clipped to [0,999], deduplicated by
(sample,wave,landmark), and sorted by sample. The problem guarantees at least six
true events per retained row, so a conservative row is completed to six events
from its highest-confidence row-local model wave groups.


VALIDATION
----------
The script reimplements maximum-cardinality matching per event class, F1 at both
official tolerances, and LCS order similarity. It runs deterministic 5-fold
GroupKFold validation with the patient record parsed from signal_file as the
group. Thus no patient, signal, window, or duplicated context waveform crosses a
fold boundary.

Observed during the final end-to-end run:

    grouped_5fold_cv score=0.710325
    F1_8                  =0.619148
    F1_20                 =0.836461
    order_similarity      =0.878980
    compact Extra presence=0.50
    compact Extra timing  =0.40
    mean Cat presence     =0.56
    mean Cat timing       =0.60

These are out-of-fold model predictions. Extra-Trees weights, per-landmark
CatBoost weights, and event thresholds are selected on complete train-only OOF
predictions, then every final model is refit on all train beats. No test row
participates in model selection or calibration.


WHAT WORKED
-----------
* Beat-level supervised learning: candidate anchors give strong translation
  invariance while the model learns target-lead presence and timing transfer.
* Combining a compact structural model with a regularized waveform model. In a
  representative held-out fold during development, either view alone scored
  about 0.638, while their averaged predictions scored about 0.652.
* Removing hand-authored record-edge label rules. Letting the classifier learn
  boundary behavior from grouped train data improved the compliant model.
* Patient-grouped validation. Row-random validation would leak repeated patient
  windows and substantially overstate generalization.
* Out-of-fold threshold calibration against the actual evaluation metric.
* CatBoost specialists raised grouped OOF score from 0.6963 to 0.7103, mainly by
  improving exact timing and faint-wave presence decisions.


WHAT DID NOT WORK
-----------------
* A deterministic delineator plus lead-offset table was weaker and, as a primary
  solver, would not meet the guidebook's real-training requirement. It was
  replaced by supervised presence and timing models; only generic signal anchors
  remain as input features.
* A dense 1D neural heatmap prototype was prohibitively slow on the local CPU and
  was not used in the submitted solution.
* A high-dimensional waveform forest alone overfit patient morphology on the
  hardest fold. The compact/full ensemble was more accurate and stable.
* Hardcoded first/last-record beat suppression was intentionally removed. The
  final code has no mapping from row id, record id, asset name, or exact evidence
  row to an answer.


LEAKAGE AND COMPLIANCE AUDIT
----------------------------
All fitting and calibration occur before test.csv is opened:

* train NPZ files supply signal transforms, the lead-token/category vocabulary,
  supervised beat examples, class balancing, fold-only missing-target medians,
  every Extra-Trees and CatBoost model, all blend weights, and OOF thresholds;
* GroupKFold uses train patient records only;
* there is no train+test concatenation, merge, append, pseudo-labeling,
  self-training, test-time adaptation, or test-distribution calibration;
* no scaler, encoder, vocabulary, threshold, feature selector, or statistic is
  fit on test or train+test;
* all random_state values and the NumPy seed are fixed;
* all processing is CPU-only; no GPU, MPS, CUDA, remote API, network, package
  installation, external data, pretrained challenge weights, or local helper
  module is used.

Every test-data use in solution.py:

1. Read test.csv after OOF calibration and final model fitting have completed.
2. Iterate one row at a time and load only that row's NPZ context signal.
3. Apply deterministic filters and row-local median/percentile normalization to
   that row's own signal. These per-sample transforms create no cross-row state.
4. Apply the train-built token encoder with an explicit all-zero unknown fallback.
5. Build one small feature matrix containing only that row's candidate beats and
   call predict_proba/predict on the frozen train-fit models.
6. Decode and discard that row before touching the next test row; the safety path
   compares only candidate beats belonging to the current row.
7. Copy the current test id to its output row and write the requested CSV after
   every row has completed.

No count, vocabulary, distribution, or fitted state is computed across test rows.
The script reads only files under public_dir and writes only submission_out.


REPRODUCTION AND OUTPUT AUDIT
-----------------------------
Run from the challenge directory:

    python3 solution.py ./dataset/public ./working/submission.csv

Local environment: Python, NumPy, pandas, SciPy, scikit-learn, and CatBoost; CPU
only. Observed local runtime was 18 minutes 1 second on an Apple M4 Pro with
strict row-at-a-time test inference.

The generated submission was checked to have:
* exactly 703 rows and exactly the columns id,answer_json;
* unique ids exactly matching test.csv;
* valid JSON with allowed waves/landmarks and integer samples in [0,999];
* unique, sample-sorted events (6 to 25 events per row, mean 13.72).

Current submission SHA-1:

    03cbef182fc140120612d950709eab8709b06a51

solution.py is the only .py file in the challenge directory and imports no local
files.
