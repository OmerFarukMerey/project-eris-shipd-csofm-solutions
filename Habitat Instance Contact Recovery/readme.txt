Habitat Instance Contact Recovery — solution notes

1. Challenge contract

Domain: computer vision / object detection under Solver Guidebook section 5.2. The required system is a trained image instance detector plus a learned within-image contact predictor. This is not a Fine-tuning-category or From-Scratch-category challenge; the solution nevertheless genuinely fine-tunes a general-purpose COCO backbone in-script.

Submission schema, exactly:

  Column 1: id
  Column 2: instances

There are no other columns. Every test id appears exactly once. instances is a JSON list serialized into one CSV field. The empty value is exactly []. A nonempty value has this shape:

  [{"instance_id":"p1","bbox":[x_min,y_min,x_max,y_max],"contacts":["p2"]}]

instance_id is a row-local unique string. bbox contains four finite normalized numbers in [0,1], ordered [x_min,y_min,x_max,y_max], with positive width and height. contacts is a JSON list of row-local instance_id strings; it has no self-reference. The implementation writes undirected edges from both endpoints. The CSV header order is exactly id,instances.

Evaluation: predicted boxes are matched one-to-one by Hungarian assignment at IoU >= 0.50. Per-image detection F1 is averaged within the empty, one-instance, and two-or-more-instance hidden strata, then the three stratum means are averaged:

  detection_score = mean(empty_f1, single_f1, multi_f1)

For multi-instance images, mapped predicted contact edges are evaluated together with matched-node coverage. Image topology quality is averaged separately over no-contact and contact images, then:

  topology_score = mean(no_contact_topology, contact_topology)
  final_score = sqrt(detection_score * topology_score)

Either zero component makes the final score zero. Higher is better. The local metric implementation uses the arithmetic mean of edge F1 and matched-node coverage for the problem statement's “combines” operation; the problem does not provide a more specific within-image combination formula.

Challenge-specific rules followed: only supplied public data are used; the source corpus, source filenames, and hidden annotations are not recovered or queried externally. Hidden images are session-held-out and include empty scenes and look-alikes, so the model is trained to localize image content rather than infer labels from identifiers.

2. Approach

Image model

The script constructs torchvision Faster R-CNN with a ResNet-50 FPN v2 backbone initialized from allowed general-purpose COCO weights. The full backbone, proposal network, ROI feature extractor, classifier, and box regressor are fine-tuned in-script. The classifier is replaced by a two-class background/target head. The proposal head is replaced and trained with five FPN scales (8, 16, 32, 64, and 128 pixels) and five aspect ratios (1/3, 1/2, 1, 2, and 3). This supports the observed scale and elongation range without using a label lookup or image-id feature.

Images remain at 512 by 512 through the detector transform. Train-only random horizontal flip, vertical flip, and 90-degree rotation augmentations transform each supplied image and its boxes together. No generated or composited records and no external images are used. AdamW, cosine learning-rate decay, gradient clipping, and CUDA mixed precision train the model. The default full-data run uses a 12-epoch fit on the training fold followed, when the wall-clock guard permits, by a low-rate two-epoch fit on all labeled rows. The schedule scales down automatically for tiny smoke datasets.

Contact model

Every labeled pair of boxes in the detector-training fold becomes one contact example. Seventeen symmetric geometry features describe gaps, center displacement, overlap, relative scale, enclosing extent, and box area. A train-fitted QuantileTransformer maps each feature to a bounded normal-score representation before a class-balanced logistic classifier. This keeps detector-box outliers numerically finite. Logistic regularization is selected in-script by cross-validation over seven logarithmically spaced candidates. The learned probability, not a geometric if-rule, produces each contact decision.

The detector confidence threshold and contact probability threshold are jointly searched in-script on the train-only validation fold. Candidate values are quantiles of validation model scores, including the observed endpoints. Each candidate pair is evaluated against the stated stratified geometric-mean objective. No threshold was tuned offline or copied from a leaderboard result.

3. Validation

The default split is deterministic and train-only: 16% is held out, stratified into empty, single, multi-instance without contacts, and multi-instance with contacts. No capture-session identifier is supplied, so a true session-group split cannot be formed; stratifying all metric-critical groups is the strongest non-leaking split available from the columns provided. Hungarian matching uses IoU >= 0.50. Detection is scored equally over empty/single/multi strata. Topology is scored equally over multi-instance no-contact/contact strata using edge F1 and matched-node coverage. The two components are combined geometrically. The script prints the full component and stratum scores before any all-data refit.

Observed local development check: 0.297610 final, 0.342857 detection, and 0.258333 topology on a 20-image stratified holdout from a balanced 128-image subset. This check intentionally used one detector epoch to prove the complete CPU execution path; its detection stratum values were [1.0, 0.0, 0.0285714] and topology stratum values were [0.5, 0.0166667]. It is not presented as the expected score of the default 12-epoch, 1,919-image A10G run. The default run computes and logs its own correctly scoped holdout score rather than relying on this pasted result.

The full local command was also exercised end-to-end on an eight-training-image/two-test-image smoke dataset. It wrote exactly two rows with header id,instances, valid JSON lists, and no malformed values. A larger 128-training-image/four-test-image execution completed in 635.7 script-reported seconds and wrote four valid rows.

4. Leakage audit

All fitted state comes from labeled train rows only:

- Detector weights: fitted on the training fold; the optional final refit uses all labeled train rows.
- Image augmentations: sampled independently per supplied training image and never fitted on test.
- Contact feature scaler and classifier: fitted only on labeled training pairs, first excluding the validation fold and optionally refitted on all labeled train pairs.
- Logistic regularization: cross-validated only on labeled training pairs.
- Detection/contact output thresholds: searched only on the held-out labeled train fold.
- No vectorizer, PCA, clustering model, encoder, imputer, or test-derived statistic exists.

Test-taint trace:

- Raw test.csv produces test_rows (id and image_path).
- Each row's image_path produces only that row's decoded RGB tensor and dimensions.
- detector(tensor) produces only that row's boxes and scores. Batching is ordinary independent model inference.
- That row's dimensions normalize only its own boxes.
- That row's boxes produce its own pair features and learned contact probabilities.
- That row's detector/contact thresholds produce its own JSON list.
- The per-row lists are written beside the original ids.

The script never computes a test prediction count, mean, quantile, class balance, calibration, vocabulary, or any other cross-row test statistic. The only whole-file operations are preserving input order, verifying row count/ID uniqueness for schema completeness, and writing the CSV. There is no train+test concatenation, pseudo-labeling, self-training, test-time fitting, or test-distribution adjustment. Test data are used only by transform(test row) and predict(test row).

5. Hardcoding / real-ML audit

No discovered generation pattern, source identity, phrase/token mapping, output-family lookup, template answer, fixed contact-distance decision, or test-distribution correction is encoded. No frequency table or retrieval result emits an answer.

Output-affecting structures were inspected:

- Annotation dictionaries and instance-id dictionaries only parse labeled graph structure and serialize the required schema; they do not infer a test answer.
- The empty/single/multi if-chain only builds the train-only split and validation metric strata.
- Device and wall-clock if-chains only control compute scheduling and safety.
- Box validity checks enforce the published submission schema; they do not locate an organism.
- Sequential p1, p2, ... values are arbitrary row-local identifiers required to reference model-predicted nodes. They carry no semantic prediction.
- Pair loops enumerate model-predicted nodes. The fitted contact classifier supplies edge probabilities; there is no asserted gap threshold.
- Exception fallbacks emit [] only to preserve a schema-valid row after a genuine row failure. They do not provide useful recovered instances.

No offline-tuned prediction constant is pasted into the program. The values that turn learned scores into boxes/edges—the detector and contact thresholds—are searched in-script against the challenge metric. Contact regularization is also searched in-script. Remaining numeric settings are published schema/evaluation constants (normalization range and IoU 0.50), standard model/optimizer architecture settings, deterministic seeds, or runtime controls; none asserts an answer pattern.

Strip-the-ML test: with all trained models removed, the pipeline produces only the initial all-empty safety placeholder. It cannot locate a target, recover a box, or produce a contact edge. With the detector removed, the learned contact head has no nodes to connect. With the contact model removed, no edge probabilities are available. Therefore the trained models produce the answer; rules, retrieval, regex, and templates do not.

Cold-reviewer pass: a fresh line-by-line read found no encoded generation pattern. The detector produces node existence and coordinates; LogisticRegressionCV produces contact probabilities; train-holdout search produces both decode thresholds. No rule or lookup emits a node or edge. The only test-derived values read across the submission are row count and IDs for required completeness checks; model scores, boxes, pair features, and contact probabilities remain confined to their originating row. No train+test concatenation exists.

6. Robustness and reproducibility

Random seeds are fixed for Python, NumPy, PyTorch, CUDA, data-loader shuffling, and contact cross-validation. The script is self-contained and has no local imports. Its explicit data reads stay under public_dir and its explicit output write is submission_out (the allowed torchvision weight loader may use its standard framework cache). A schema-valid all-empty submission is written immediately after test.csv is read, before ML imports or training. A 3,000-second wall-clock guard stops launching training epochs, and a 2,600-second guard skips optional all-data refitting to reserve inference time. Bad training batches are skipped; unreadable or failed test rows receive [] without aborting other rows. Dataset row count differences are warnings, not assertions.

Before replacing the placeholder, the script verifies prediction count, test-ID uniqueness, per-row list type, unique instance IDs, finite positive boxes, valid row-local contacts, and no self-links. It writes only after all checks pass.

7. What worked and what did not

Worked:

- A real fine-tuned FPN detector directly learns the telemetry appearance and supports empty images.
- Small custom anchors cover the tiny annotated boxes while FPN levels retain large-object coverage.
- Metric-stratified validation prevents the abundant easy count groups from alone selecting thresholds.
- Learned pair geometry gives a compact, trainable contact model and avoids encoding the published distance condition as the answer.
- Early placeholder, per-row fallbacks, and final schema checks make failures scoreable rather than malformed.

Did not work / limitations observed:

- A one-epoch 128-image CPU check localized crowded/single objects poorly; this is why the production path trains twelve epochs on all 1,919 supplied examples rather than using the smoke configuration as a model choice.
- Capture-session group validation is impossible because no session/group column is supplied. The stratified split may therefore be optimistic relative to hidden held-out sessions.
- The problem statement does not specify the exact arithmetic used to combine edge F1 and node coverage within one image. The validation implementation uses their arithmetic mean and keeps the two published topology groups separate.
- The first LBFGS contact-classifier experiment produced numerical warnings on the reduced subset. The delivered estimator uses train-fitted quantile normalization and liblinear LogisticRegressionCV; a full 5,294-pair fit was checked to produce finite probabilities, including high probabilities on labeled positive contacts.
