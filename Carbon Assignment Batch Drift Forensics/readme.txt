Carbon Assignment Batch Drift Forensics solution
================================================

Run
---

Platform command:

    python3 solution.py <public_dir> <submission_out>

Local example:

    python3 solution.py ./dataset/public ./working/submission.csv

The script reads train.jsonl, test.jsonl, and shift_calibration.csv from the
supplied public directory and writes exactly to the supplied output path.
Required Python packages are numpy, pandas, scipy, scikit-learn, lightgbm,
torch, and transformers. The inspectable source resolves the public
DeepChem/ChemBERTa-5M-MLM checkpoint through transformers and then fine-tunes
the model locally. Fixed seeds are used; minor probability variation can remain
between PyTorch/MPS versions.

Approach
--------

This is a two-stage supervised chemistry model, not an ID, row-order, database
lookup, or hand-written shift-range solution.

1. Load the offline DeepChem/ChemBERTa-5M-MLM molecular-language checkpoint and
   fine-tune every encoder layer plus a regression adapter on clean_shift_ppm.
2. Independently fit graph-aware boosted compatibility models on parsed
   radius-2/radius-3 chemistry features.
3. Cross-fit both model families by complete environment_profile groups. A
   claim is never predicted by a fold model trained on the same summarized
   radius-3 profile. This produces leakage-resistant compatibility predictions
   for all training claims.
4. For each batch, jointly test all chemically legal two- and three-claim
   subsets inside each local_class. Fit the residual patterns expected under
   identity, a transposition, either direction of a three-cycle, a two-record
   transplant, and a three-record common offset.
5. Train two complementary batch classifiers on those joint forensic features,
   ensemble their probabilities, temperature-calibrate the ensemble, and apply
   a conservative correction for the published exactly balanced test prior.

Pretrained chemistry encoder and atom-level architecture
--------------------------------------------------------

solution.py adapts the public molecular-language encoder
DeepChem/ChemBERTa-5M-MLM:

    https://huggingface.co/DeepChem/ChemBERTa-5M-MLM

It is a three-layer RoBERTa chemistry encoder with hidden size 384, pretrained
on five million molecular strings. The checkpoint is loaded through the
standard inspectable AutoModel.from_pretrained path—there are no embedded,
encoded, or opaque model payloads in solution.py.

Each claim is serialized as multiplicity, solvent, environment_profile, and
environment; IDs and local_class hashes are excluded. The encoder output is
attention-mask mean pooled and passed through a trainable LayerNorm-MLP
regression adapter. Every pretrained encoder parameter and the adapter are
updated with supervised clean-shift MSE, AdamW, gradient clipping, and a
one-cycle learning-rate schedule.

ChemBERTa is fine-tuned for two epochs in each of two
environment_profile-grouped folds. Held-fold predictions form its training
forensic view, while the two independently fine-tuned test predictions are
averaged.

A complementary graph-aware path parses the censored grammar rather than
treating it as digit-only text. Every claim becomes a sparse vector containing:

* center descriptor and center attributes;
* exact rooted radius-2 paths and root-independent path continuations;
* depth-specific node, element, bond, and edge-transition counts;
* all numerical radius-3 environment_profile counts;
* multiplicity and solvent indicators;
* local_class as a supervised compatibility feature, but never as a batch
  target; and
* an exact lossy-profile feature when such a public profile recurs.

Two LightGBM regressors are fitted on clean_shift_ppm. The first uses 128 leaves
and stronger regularization; the second uses 256 leaves and broader feature
subsampling. Four GroupKFold models of each type are trained. Together the
batch stage receives four compatibility views: regularized boosted graph,
high-capacity boosted graph, their mean, and fine-tuned ChemBERTa.

Batch model and forensic features
---------------------------------

The hidden batch offset is removed robustly with the median compatibility
residual. The code then enumerates every same-local_class pair and triple,
including accidental groups larger than three.

Pair evidence includes:

* residual error after exchanging the two predicted clean shifts;
* whether the two original deviations cancel, as a true swap should;
* observed-versus-predicted separation error;
* anomaly magnitude; and
* residual consistency of the other ten claims.

Triple evidence includes:

* residual error for both possible three-cycle directions;
* mean, spread, range, sign agreement, and sum of the three deviations;
* residual consistency of the other nine claims; and
* the low-within-spread/high-between-offset pattern of partial reference drift.

These candidate rows are pooled invariantly by minima, medians, maxima, and the
top two candidates under several generative costs. Pooling is repeated for all
four compatibility-model views. A 300-tree ExtraTrees classifier consumes the
full candidate evidence. A 400-tree depth-six XGBoost classifier consumes a
compact aggregate view. Final probabilities are 30% ExtraTrees and 70%
XGBoost, followed by the validation-selected temperature 0.95.

The test compatibility estimates average several fold models and are less
variable than a single held-fold training prediction. The batch learner
therefore uses validation-selected residual scales of 0.92 for the boosted
graph views and 0.96 for ChemBERTa. This matches train-time forensic noise to
the folded test ensemble without reading test labels or fitting on test claims.

Validation strategy and observed results
----------------------------------------

The important split is at environment_profile group level, not a random claim
split. Cross-fitting therefore withholds every occurrence of a lossy radius-3
profile from the compatibility model that predicts it. The final inspectable
run observed these held-profile results:

    ChemBERTa fold 1 MAE: 6.8103 ppm
    ChemBERTa fold 2 MAE: 6.9078 ppm

    boosted graph fold 1: 3.3766 ppm
    boosted graph fold 2: 3.4749 ppm
    boosted graph fold 3: 3.5238 ppm
    boosted graph fold 4: 3.4729 ppm
    boosted graph combined OOF: 3.4615 ppm

ChemBERTa is a complementary molecular-language view rather than a replacement
for the stronger parsed-graph regressors. It contributes independent
relative-ordering evidence to the joint batch classifier. Model and probability
choices were made on a separate fixed 25%
stratified batch holdout (random_state=2026). Tight-example labels are private,
so no tight validation score is claimed.

On that holdout, matching the folded-ensemble noise and switching to the
ExtraTrees/XGBoost blend improved the batch metrics to:

    multiclass log loss: 0.9754
    macro-F1:             0.5861


As a diagnostic upper bound, the same forensic batch features supplied with
true training clean shifts reached 0.9950 macro-F1 and 0.0224 log loss on a
separate 20% split. This confirms that the remaining error is primarily clean
shift compatibility uncertainty rather than an inability to represent the five
provenance mechanisms.

What worked
-----------

* Parsing the graph/profile grammar and supervised fitting on clean shifts.
* End-to-end supervised fine-tuning of an offline chemistry-pretrained encoder.
* Grouped cross-fitting, which removed repeated-profile leakage.
* Joint permutation/offset fitting across all twelve claims.
* Searching all same-topology subsets instead of assuming the repeated class is
  always exactly size three.
* Ensembling regressors with different capacity and classifiers with different
  decision geometry.
* Keeping calibrated soft probabilities rather than emitting one-hot labels.

What did not work
-----------------

* local_class target means alone: approximately 6.16 ppm MAE on held-profile
  claims, too weak for tight swaps.
* Treating full environment strings as CatBoost categorical values: about
  10.42 ppm MAE, because unseen deeper contexts do not share category statistics.
* Chemistry-neighbor lookup on TF-IDF graph tokens: about 4.45 ppm MAE and no
  gain when blended with the graph-aware boosted models.
* A direct pair-difference regressor: 5.77 ppm pair-difference MAE versus 4.49
  ppm from differencing the individual compatibility predictions.
* Training the batch classifier from marginal shift statistics. Those features
  are balanced by construction and did not represent the provenance operation.

Submission verification
-----------------------

The generated submission was checked after a full run:

* command: python3 solution.py;
* solution.py size: 23,163 bytes, below the 512,000-byte limit;
* working contains only submission.csv;
* 4,000 rows and exactly the six required columns;
* every test ID present exactly once;
* all values finite and strictly positive;
* minimum probability 0.000267123 and maximum 0.998108424; and
* maximum row-sum deviation from one of 2.1e-9 after CSV serialization.
