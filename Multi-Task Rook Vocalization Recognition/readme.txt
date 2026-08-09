Multi-Task Rook Vocalization Recognition
=======================================

Problem, domain, and output contract
------------------------------------
This is an animal-bioacoustics/audio challenge, so it falls under the guidebook's Bio/Chem/Other domain (section 5.6). The submitted model is nevertheless trained entirely from scratch on the supplied audio, following the more conservative section 5.5 constraints: it loads no pretrained weights, external data, external labels, or external services.

The CSV schema is exactly 25 columns in this order:

id, det_score, ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, ind_11, ind_12, ind_13, ind_14, ct_0, ct_1, ct_2, ct_3, ct_4, ct_5, ct_6, ct_7

For test row i, id is the exact string "t" followed by i as five zero-padded decimal digits (for example, t00000). Every predicted field is written as a finite decimal floating-point score. det_score is a per-row sigmoid probability. The 15 ind_* values and eight ct_* values are per-row softmax probabilities in numeric class-id order. Every test id is emitted exactly once, including for background windows.

Metric
------
RookScore is the arithmetic mean of three terms:

1. Detection efficiency: for each false-alarm rate a in {0.05, 0.10}, choose the det_score threshold above which fraction a of background windows fall, measure the fraction of weak/faint call windows above that threshold, then average the two efficiencies.
2. Individual agreement: mean reciprocal rank of the true individual among ind_0 through ind_14 on call windows. Exact ties use their average rank.
3. Call-type agreement: the same mean reciprocal rank among ct_0 through ct_7, restricted to call windows with a scored call type.

All three terms and their mean are in [0, 1], and higher is better. Only within-head ordering matters.

Approach
--------
solution.py memory-maps each 16 kHz waveform. A fixed, differentiable-but-detached signal frontend computes an 80-bin log-mel power spectrogram with a 512-sample FFT, 400-sample Hann window, and 160-sample hop. The frontend supplies two channels: absolute log-mel energy and a per-window standardized copy. Absolute energy supports faint-call detection; row-local standardization supplies session-robust spectral shape. FFT and power calculations stay in float32 even during CUDA mixed-precision training.

A from-scratch 2-D residual CNN shares five residual blocks, then gives detection, individual recognition, and call-type recognition separate two-block residual branches. The classification branches train only on rows carrying their corresponding labels, so background and unscored calls do not update task-specific BatchNorm statistics. Each branch uses learned attentive statistics pooling over a weighted mean and standard deviation. The individual head produces a normalized 192-dimensional embedding and learned-scale cosine logits; cross-entropy plus within-batch supervised contrastive loss pulls calls from the same rook together across recording sessions. A matched experiment showed that angular metric learning helped individual recognition but hurt call-type recognition, so the call-type head deliberately retains an unconstrained learned linear classifier and cross-entropy. Label-preserving augmentation applies per-row random gain and polarity, temporal roll, and train-only time/frequency masking. AdamW, cosine learning-rate decay, gradient clipping, early stopping, and fixed seeds are used.

The script first trains one recording-session holdout model and selects its epoch count against train-only RookScore. If measured runtime permits, it then retrains up to two independently seeded models from scratch on every supplied training session for exactly that selected epoch count; these full-data models form the inference ensemble. If the 3000-second guard prevents a complete full-data model, the validated holdout model is retained instead. Model averaging is only across models for the same test row and never aggregates across test rows.

Validation
----------
Validation uses training data only. A deterministic in-script search constructs an approximately 18% holdout of entire recording-session groups. Its objective preserves all feasible label coverage in the training fold, then validation coverage and train-distribution balance. No recording session appears in both folds. Early stopping selects the checkpoint against the exact individual and call-type MRR terms and the two specified false-alarm operating points. The public training data has no flag identifying the metric's hidden weak-call subset, so the validation detection term uses all held-out call rows as the explicitly documented proxy; it does not invent a weak-call rule.

Reference train-only experiments on the same 3,954-row, six-session holdout:

Attentive linear-head model:
- selected epoch: 6
- detection proxy: 0.7352
- individual MRR: 0.5508
- call-type MRR: 0.5092
- validation RookScore proxy: 0.5984

Angular metric learning on both recognition heads:
- selected epoch: 7
- detection proxy: 0.7382
- individual MRR: 0.5892
- call-type MRR: 0.4695
- validation RookScore proxy: 0.5990
- non-finite/skipped batches: 0

The final hybrid keeps the experimentally stronger angular individual head and linear call-type head. The full-data retraining stage uses the epoch count selected in-script; it never selects against test predictions. On CPU the wall-clock guard correctly retains the validated model. The A10G path uses CUDA mixed precision and launches a full-data member only when measured runtime leaves the required inference buffer.

Leakage audit
-------------
Every fitted object and statistic uses training rows only. This includes the group split, class/label coverage calculations, early stopping, convolution and head weights, optimizer state, and BatchNorm running statistics. The mel filter is an analytic signal transform, not fitted data. The per-window log-mel mean and variance use samples from that same waveform only.

Test-taint trace:
- test_waveforms is read from test_X.npy only after all required paths are checked; only its dimensionality and row count are inspected before the required placeholder is written.
- A contiguous test slice becomes batch_array, then batch/wave.
- torch.nan_to_num is elementwise. STFT, mel projection, log, and standardization operate independently inside each waveform. The CNN is in eval mode, so BatchNorm uses training running statistics rather than test-batch statistics.
- sigmoid is applied independently to each row; softmax is only across the 15 individual classes or eight call-type classes within that row.
- If multiple trained models fit the time budget, their probabilities are averaged only for the same row.
- det_out, ind_out, and ct_out merely store those row predictions. CSV formatting and finite-value checking are performed one row at a time.

No test-derived variable is used to fit, tune, select, calibrate, normalize across rows, sort, count predictions, choose a threshold, or alter another row. There is no train+test concatenation. There is no pseudo-labeling or test-time adaptation. Test data is used only for fixed transform and model prediction.

Hardcoding and real-ML audit
----------------------------
There is no phrase/token mapping, answer dictionary, per-id lookup, regex mapping, template, retrieved answer, generation-pattern rule, or class-balancing correction. No discovered generation pattern is hardcoded. The class counts and column order are the declared output schema, not learned answer logic. Session ids are never asserted; validation sessions are selected by an in-script train-only search.

There are no pasted-in output thresholds, temperatures, blend weights, class priors, or decode knobs. Scores are produced by trained neural heads. Model-selection values are searched in-script against the train-only group holdout. Fixed architecture dimensions, optimizer settings, signal-processing parameters, and augmentation ranges are a priori training configuration rather than asserted answers; they do not directly decide a class or encode a discovered dataset regularity. Every constant that calibrates, decodes, or otherwise decides a real prediction is learned or searched in-script.

The only asserted scores are equal, neutral probabilities in the mandatory early placeholder and exceptional per-row safety fallback: 0.5 for binary detection, 1/15 for each individual, and 1/8 for each call type. They contain no answer information and are overwritten by model scores on every successful row. They exist solely to satisfy the required robustness contract.

Strip-the-ML result: with all trained models removed, the pipeline produces only the schema-valid, uniform placeholder/failure fallback. It produces no discriminative or still-usable answers. The trained multi-task network, not a rule, lookup, retrieval system, or template, produces every substantive score.

What worked and what did not
----------------------------
The first baseline's 0.5746 score showed detection was substantially stronger than the weakly related recognition tasks. Task-specific residual branches and attentive statistics pooling improved the clean score to 0.5984. The controlled angular-head experiment then improved individual MRR from 0.5508 to 0.5892 but reduced call-type MRR from 0.5092 to 0.4695. The final model therefore uses angular supervised metric learning only where validation supports it: rook identity. Call type retains the stronger linear classifier. The A10G path additionally recovers the 18% holdout through train-only-selected full-data retraining rather than leaving every inference model without those sessions.

Local PyTorch MPS training showed backend-specific non-finite gradients and was not used for any reported result. The final script uses CUDA on the grading A10G and stable CPU as its fallback; fixed frontend computation is detached and forced to float32. Clean CPU runs had zero skipped batches. No test-derived calibration or distribution adjustment was introduced.

Operational audit
-----------------
Random seeds are fixed for Python, NumPy, PyTorch, CUDA, split selection, and DataLoader order. The script reads only files beneath public_dir and writes only submission_out. It writes a complete placeholder before heavy training, catches post-placeholder and per-row failures, uses a 3000-second wall-clock training guard, and verifies row prediction shapes and every output value. solution.py is self-contained, imports no local module, and is the only Python file in the challenge directory.
