INTRAOPERATIVE ECG BEAT-AND-RHYTHM TRANSCRIPTION
================================================

Problem, domain, and submission contract
----------------------------------------
This is a biomedical time-series challenge and falls under Solver Guidebook section 5.6
(Biology, Chemistry, and Other Domains Without Their Own Section). PROBLEM.md does not
categorize it as Fine-tuning and imposes no model-family restriction. The solution trains
its ECG network from scratch using only the supplied public training waveforms.

The output is a CSV with exactly these columns, in this order:

    id,rhythm_family,beats

* id is the test.csv id string and every test id appears exactly once.
* rhythm_family is exactly one of the nine manifest rhythm strings.
* beats is a JSON string containing an ordered list of at most 64 events. Each event is
  exactly [sample_index,"beat_type"], where sample_index is a JSON integer in [0,750)
  and beat_type is exactly N, S, V, or U. The script uses compact JSON, for example
  [[71,"N"],[162,"V"],[278,"N"]]. It emits [] only when no event is predicted (or as
  the required emergency placeholder before training completes).

pandas performs the required CSV quoting around JSON strings. The final audit verifies
column order, row count, exact id order, id uniqueness, rhythm labels, JSON syntax, event
count, integer/range constraints, type labels, and strict sample ordering before replacing
the early placeholder.

Exact metric
------------
Higher is better, with

    score = 0.20 * event_micro_F1
          + 0.30 * beat_type_macro_F1
          + 0.30 * rhythm_macro_F1
          + 0.10 * rare_beat_macro_F1
          + 0.10 * rare_rhythm_macro_F1.

A beat matches only a gold beat of the same type within 10 samples. Candidate pairs are
processed from smallest timing error upward and matched one-to-one. event_micro_F1 pools
all beat types. beat_type_macro_F1 averages the event F1 for N, S, V, and U when present.
rare_beat_macro_F1 averages V and U. rhythm_macro_F1 is ordinary class-F1 macro averaging
across the nine present rhythm families. rare_rhythm_macro_F1 averages
atrioventricular_block, supraventricular_tachyarrhythmia,
ventricular_tachyarrhythmia, and wandering_multifocal_atrial_rhythm. The implementation
uses this exact metric for every output-decoder and ensemble search.

Approach
--------
1. Each waveform is transformed independently. It is median-centered and divided by its
   own RMS, then represented by normalized amplitude, first derivative, and absolute first
   derivative channels. No population-level scaler is needed.
2. A 96-channel gated dilated residual TCN processes all 750 samples without temporal
   downsampling. Its dilation schedule gives nearly whole-window context while retaining
   sample-level output resolution.
3. A detector head learns a one-dimensional R-peak heatmap. Training targets are Gaussian
   heatmaps whose radius comes from the metric tolerance; a CenterNet-style focal loss
   penalizes missed centers and false peaks.
4. A beat head predicts N/S/V/U logits at every sample. Cross-entropy is evaluated at gold
   event positions. A rhythm head combines attention, mean, and maximum pooling from the
   TCN with a learned convolutional encoding of the detector/type probability sequence.
   Thus beat order and learned morphology evidence inform the window rhythm directly.
5. All three variants use square-root class balancing. Two are independent ordinary seeds;
   the third is trained with mild crop translation, polarity, smooth gain/filter, and noise
   augmentation. Translation moves detector and beat-type targets with the waveform. These
   nuisance transforms match the transform family published in PROBLEM.md; they never use
   a label-dependent rule. Three patient-proxy folds produce nine neural models and OOF
   predictions. Early stopping uses validation multitask loss.
6. Original-polarity and inverted-polarity predictions are equally averaged per window.
   This is ordinary per-sample TTA and does not inspect other test rows.
7. On train-only OOF predictions, the script searches neural blend weights, detector
   thresholds, NMS distance, per-beat thresholds, beat-logit biases, and rhythm-logit
   biases against the exact challenge metric. A five-unit simplex gives 0.20 blend
   resolution. Nothing selected by a prior run is pasted into solution.py.
8. A complementary CatBoost rhythm model consumes per-window waveform summaries, RR/type
   sequence statistics, and morphology statistics around transcribed beats. Within each
   OOF fold it trains on that fold's gold training sequences and predicts the validation
   rows from cross-fitted neural transcriptions. A final model trains on all labeled gold
   sequences and predicts test rows from neural transcriptions. This train/predict contract
   is therefore tested OOF rather than assumed.
9. The rhythm search jointly considers all neural rhythm probabilities and the CatBoost
   probabilities. The final test prediction retains the calibrated fold ensembles; there
   is no differently calibrated neural full-data refit.
10. Learned detector local maxima are greedily suppressed by searched confidence and
   distance settings, typed by the learned beat head, and sorted. Signal-processing code
   never emits a beat or rhythm label.

Feature engineering
-------------------
The TCN input is limited to the three per-window channels above. The CatBoost rhythm model
also receives deterministic features computed independently within a row: FFT magnitudes,
amplitude/difference quantiles, pooled energy, beat counts/transitions, RR variability, and
aggregate beat morphology. Gold beat events are used only for CatBoost training rows;
cross-fitted neural events are used for OOF validation, and neural events are used for test.
Separate median-QRS descriptors are computed from training rows only to construct validation
groups. No filename, id pattern, external patient identifier, external recording, pretrained
weight, retrieval result, or synthetic training data is used.

Validation strategy and observed score
--------------------------------------
No patient ids are supplied, and an earlier rhythm-stratified morphology grouping proved
optimistic (OOF 0.764 versus a 0.678 leaderboard). A later neural full-data refit scored
0.6562 because OOF decoder parameters calibrated on fold ensembles were applied to a
differently calibrated two-model refit; that path was deleted. The current script builds a
per-window median-QRS template plus generic spectral descriptors from training signals only,
then binds each window with its five nearest global morphology neighbours without consulting
rhythm labels. A deterministic class/size-balanced assignment creates folds of 1,074, 1,074,
and 1,073 rows.

Every neural training row is predicted out of fold. For the second-stage rhythm learner,
each outer fold chooses its beat blend and decoder from the other folds, then generates the
held-out fold's sequence features. CatBoost trains on gold features from the outer training
rows and predicts those transcribed held-out features. An additional nested prototype check,
where augmentation and CatBoost blend/decoder choices were selected on two folds and scored
on the third, improved the weighted score by 0.0211 over the same base predictions.

The final local end-to-end run obtained:

    final weighted OOF score       0.793178
    event_micro_F1                 0.934355
    beat_type_macro_F1             0.759186
    rhythm_macro_F1                0.812039
    rare_beat_macro_F1             0.577482
    rare_rhythm_macro_F1           0.771916

Selected beat F1 values were N 0.961365, S 0.920414, V 0.858827, and U 0.296137.
Selected rhythm F1 values were sinus_rhythm 0.838649,
atrial_fibrillation_flutter 0.954476, patterned_atrial_ectopy 0.848558,
patterned_ventricular_ectopy 0.918567, sinus_node_dysfunction 0.660436,
atrioventricular_block 0.792208, supraventricular_tachyarrhythmia 0.751678,
ventricular_tachyarrhythmia 0.930233, and wandering_multifocal_atrial_rhythm 0.613546.
The run trained nine neural fold models, three CatBoost OOF models, and one final CatBoost
model; it wrote a schema-valid 1,107-row submission in 1,985.18 seconds on Apple MPS. A
3,000-second wall-clock guard stops launching models and preserves the best completed neural
ensemble if an optional stage cannot finish.

What worked and what did not
----------------------------
The prior neural full-data refit reduced the observed leaderboard from 0.678 to 0.6562; more
rows did not compensate for losing six-model ensemble diversity and applying OOF calibration
to a new probability distribution. It is removed. The final OOF beat search assigned 80% to
the augmented seed and 20% to one plain seed. The rhythm search assigned 60% to the augmented
seed, 20% to a plain seed, and 20% to the event-feature CatBoost model. Thus both additions
survived the exact metric search rather than being forced into the output.

Inverse-frequency balancing, a neural candidate-patch refiner, and a CatBoost candidate
refiner were prototyped. Inverse balancing and the patch CNN received zero blend weight. The
candidate CatBoost gain shrank to 0.0016 under nested calibration and reduced U F1, so it was
not shipped. U has only 158 train events and remains the hardest beat type. Global
morphology-neighbour grouping is more conservative than random windows, but it cannot
guarantee true patient grouping because private patient ids are unavailable.

Leakage audit: complete test-taint trace
---------------------------------------
The final cold review traced every test-derived value:

* test_frame supplies a row's id and signal path. It is never joined or concatenated with
  train_frame. The only cross-row operations on ids are final schema completeness checks;
  they do not alter predictions.
* test_signals is created by iterating rows. Median, RMS, gradient, finite-value repair, and
  optional length interpolation use only that row. Stacking rows is batching, not fitting.
* A trained ECGTranscriber transforms each test row and predicts detector, type, and rhythm
  probabilities. GroupNorm has no cross-batch fitted statistics. Polarity TTA operates on
  that same row.
* test_probabilities and the *_sum arrays are added only across trained models for the same
  row. Division is by the number of models. variant_results and searched blend operations
  likewise combine models elementwise for the same row; no test-row mean, frequency,
  histogram, quantile, sort, class count, or target-distribution correction is computed.
* test_detector/test_type become local maxima and NMS events one row at a time.
  test_rhythm becomes an argmax one row at a time. build_submission serializes one row at
  a time. The final audit reads rows only to verify schema and never feeds information back
  into a prediction.
* CatBoost OOF models train only on each fold's labeled training rows, while their validation
  features come from cross-fitted neural beats. The final CatBoost model trains on all
  labeled rows and applies the already-fitted model to each test row's neural transcription.
  No test row participates in training, feature fitting, or calibration.

All heatmap targets, label encoders, class counts/weights, morphology descriptors,
standardization statistics, validation groups, early stopping, blend weights, thresholds,
distances, and logit biases are fit or searched using training data only. Test is used only
for per-row transformation and model inference. There is no train+test concatenation,
pseudo-labeling, test-time fitting, test-distribution calibration, or cross-test-row
feature. No vectorizer, tokenizer, PCA, clustering model, scaler, or encoder is fit on test.

Hardcoding and real-ML audit
----------------------------
No discovered data-generation pattern is hardcoded. Every data-dependent statistic and
every output-calibration/decode constant is learned from fold-training data or searched
in-script on train-only OOF predictions. Fixed asserted values are limited to the published
schema/transform family, the mandated runtime guard/seed, and ordinary neural/classical-ML
architecture, augmentation, and optimization choices; none maps an input pattern to an
answer.

The cold review found these dictionaries/lookups and control paths:

* manifest label lists, rhythm_to_index, and beat_to_index are schema/target encodings. They
  do not map waveform content to a prediction; trained neural and CatBoost models learn the
  mappings.
* model-result dictionaries store probability tensors from trained models.
* threshold, distance, class-bias, and simplex blend grids are search spaces. Final values
  are selected inside each run by the exact train OOF metric.
* if-chains handle devices, malformed/missing files, early stopping, the wall clock, schema
  validation, and per-row failure containment. The required early placeholder and emergency
  exception fallback use a rhythm frequency learned from train and an empty event list;
  they are never the successful model path and are not a usable transcription system.
* There is no phrase/id/filename-to-label table, regex mapping, template, retrieval system,
  n-gram/frequency answer generator, hand peak detector that emits events, or rule that
  emits a rhythm or beat type.

Strip-the-ML result: with every trained ECGTranscriber and CatBoost model removed, there are
no detector/type/rhythm probabilities, candidate events, or successful-path predictions.
Only the mandated schema-valid emergency placeholder remains; it has empty beats and is not
a usable answer. Trained models produce every submitted beat location, beat type, and rhythm;
NMS and JSON serialization only convert learned probabilities into the required format.

Housekeeping and reproducibility
--------------------------------
Random, NumPy, PyTorch, DataLoader, fold, and model seeds are fixed. solution.py imports no
local code, loads no prior fitted artifact, and is the only Python source file in the
challenge directory. It reads only public_dir, writes only submission_out, creates the
output parent, writes the early placeholder before heavy work, catches fold and per-row
prediction failures, and verifies the complete final schema before overwrite.
