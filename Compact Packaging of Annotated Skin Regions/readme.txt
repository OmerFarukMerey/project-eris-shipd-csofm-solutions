Compact Packaging of Annotated Skin Regions - solution notes
=============================================================

Run:  python3 solution.py <public_dir> <submission_out>
solution.py is the only .py file in this directory, is self-contained and reads both paths from sys.argv.
Shipped version (v2): solution.py sha256 e57719def21f7277d3d57bdc3ab15e8e5621a560041424562a88f8488c0e51b3
working/submission.csv was produced by exactly this file (see "Execution").


1. Task facts taken from PROBLEM.md
-----------------------------------
Submission schema: header exactly  case_id,regions,crop_plan  (in that order), one row per test case_id
(546 rows). regions and crop_plan are JSON strings of [left, top, right, bottom] JSON integers in 0..1024,
positive width/height, no duplicate boxes, <= 64 region boxes, <= 3 crop boxes, <= 8192 characters.
Any invalid box field zeroes the whole case, so every box is integer-rounded, clipped, de-duplicated and
capped in code, and the written file is re-read and checked at the end.

Metric (per case, averaged over cases):
  0.35 * RegionF1          (mean of the one-to-one-matching F1 at IoU 0.50 and 0.75)
+ 0.20 * CompleteCoverage  (fraction of reference regions fully inside one crop)
+ 0.45 * CompactUtility    (coverage * min(1, D_ref / D); D = summed crop area, D_ref = summed area of the
                            documented greedy reference plan of the true regions)

Domain: computer vision / object detection (guidebook 5.2). PROBLEM.md allows general-purpose pretrained
vision models and crop optimisation using the training labels. Not used: anything from the withheld
photographs, IDs / file order / metadata, participant identity, test-time fitting of any kind.
Hardware/runtime: single A10G, 1.5 h end to end.


2. Approach
-----------
(a) Detector, trained in-script on train.csv only
    - Backbone: timm convnext_tiny.fb_in22k (general ImageNet-22k weights from the Hugging Face hub;
      the only internet use), features_only, FPN top-down neck to stride 4 (128 channels).
    - Heads at stride 4: CenterNet heatmap (penalty-reduced focal loss, elliptic Gaussian targets),
      FCOS-style l/t/r/b distances (GIoU loss on cells inside a box, Gaussian-weighted), and a per-edge
      Laplace log-scale head (NLL on the detached edge errors) that predicts each box edge's uncertainty.
    - Input 768x640 RGB (native size). Augmentation on the GPU: horizontal flip, isotropic scale
      0.75-1.33 with translation, brightness/contrast/colour jitter. No vertical flip (see section 5).
    - AdamW lr 3e-4, wd 0.05, batch 8, 3% warm-up + cosine, fp16 autocast, EMA of weights (0.999).
    - Inference: peak = 3x3 local max of the heatmap; box = l/t/r/b at the peak plus a heat-weighted
      3x3 "voted" box; flip TTA (none, horizontal) averaged per image.
(b) 3-fold training. Each fold model predicts its held-out third (out-of-fold, OOF) and the test
    images. Test maps = average of the 3 fold models (per image).
(c) Calibration models fitted on the OOF detections (train labels only), all logistic regressions
    (standardised inputs):
      P(detection is a distinct annotated region at IoU >= 0.5)  - used by the crop decoder,
      P(distinct match at IoU >= 0.5) and P(... IoU >= 0.75)       - used by the region decoder.
    Features per detection, computed from that image's own detections only: score logit, predicted edge
    uncertainty, box size, relative uncertainty, rank in the image, number of confident detections in
    the image, top score of the image, distance to the nearest other detection.
(d) Region list = expected-F1 decoding per image: the top-j detections, with j maximising
    (sum q50 + w75 * sum q75) / (j + sum q50 + mhat).
(e) Crop plan = Monte-Carlo expected-metric decoding per image:
    - 128 fixed random "worlds" (the same draws for every image): in each world a detection is real with
      its calibrated probability and its true edges are the predicted edges plus kappa x predicted Laplace
      scale x noise; each world's D_ref is computed with the documented reference procedure.
    - candidate plans: for every margin level a and every j <= 16, the exact minimum-total-area partition
      of the top-j expanded detections (margin m0 + a x edge scale) into <= 3 rectangles (subset DP;
      groups are greedily pre-merged above 8 boxes), then 2 rounds of add/drop-one-detection local search
      around the best plan.
    - the plan with the highest expected 0.2*coverage + 0.45*coverage*min(1, D_ref/D) is delivered.
(f) Knob search, in-script, on the OOF predictions against the PROBLEM.md metric:
    region: candidate source (peak/voted box), NMS (off/0.6), mhat, w75 (grid search);
    crop: kappa, m0, mhat, prob scale, margin-level set, NMS (coordinate ascent).

Feature engineering: none on raw pixels outside the CNN. The calibration features are the detector's own
outputs (scores, predicted uncertainty, predicted boxes) for the same image.


3. Validation (train only)
--------------------------
3-fold random split of train.csv rows (seeded). OOF detections for all 2184 training images.
Honest decode estimate printed by the script: fit the calibrators and search the knobs on one random half
of the OOF rows, score the other half, and swap.

Final run of the shipped solution (local RTX 5070 Laptop GPU, see "Execution"):
  half 0: searched-on 0.5602 -> held-out 0.5615 (RegionF1 0.5504, crop part 0.3688)
  half 1: searched-on 0.5627 -> held-out 0.5596 (RegionF1 0.5461, crop part 0.3685)
  OOF held-out estimate (mean)              : 0.5605   (v1 shipped run: 0.5454)
  OOF in-sample with the final knobs        : 0.5612 (RegionF1 0.5495, crop part 0.3688)
  final knobs: region {src vbox, nms off, mhat 0.0, w75 1.5};
               crop {nms off, kappa 1.5, m0 3.0, mhat 1.0, prob scale 1.0, levels 0,1,2,3}
("crop part" = 0.2*coverage + 0.45*utility, max 0.65.)
Caveat: there are no participant IDs and PROBLEM.md forbids recovering them, so the folds are random over
images; tiles of the same person can sit in different folds, which may make the OOF estimate optimistic
relative to the participant-separated test set.

Development measurements (held-out halves of OOF predictions):
  fold 0 of a 4-fold split (546 images):
    threshold decoders (global score threshold + fixed margins), resnet34d        0.51
    + Bayes crop decoder (crop part 0.330 -> 0.353)                               +0.02
    + expected-F1 region decoder (RegionF1 0.510 -> 0.535)                        +0.009
    backbones with the full decoder: resnet50d 0.531, resnet34d 0.540, convnext_tiny 0.557
  separate 3-fold convnext_tiny OOF dump (2184 images, 4 held-out halves), decoder changes v1 -> v2:
    logistic calibration with uncertainty/size features (crop part)                +0.004
    add/drop local search over plans (crop part)                                  +0.003
    128 instead of 64 Monte-Carlo worlds (crop part)                              +0.002
    per-image context features (total)                                            +0.002
    greedy pre-merge above 8 instead of 12 boxes                                  same score, ~10x faster
  v2 decoder on that dump: 0.5534 held-out mean; the v1 shipped run (old decoder, 16 epochs) printed
  0.5454 on its own OOF. The v1 submission scored 0.6204 in the platform's CSV check; that number was not
  used for any choice in the script.


4. Plan, determinism, timing
----------------------------
Static plan, identical on every run: convnext_tiny, 3 folds x 1 seed x 14 epochs, batch 8, 768x640,
TTA none + horizontal flip, fixed search grids. No branch depends on time, hardware or CPU count.
torch.cuda.is_available() only chooses the device; time.time() is only printed.

Deterministic execution: seeds fixed for random / numpy / torch; cuDNN autotuning off
(cudnn.benchmark = False), cudnn.deterministic = True, torch.use_deterministic_algorithms(True) and
CUBLAS_WORKSPACE_CONFIG=:4096:8. A strict-mode run (no warn_only) raised no error on any training or
inference op, and two complete runs of the pipeline on the same train-only subset produced byte-identical
submissions (same sha256), with no non-deterministic-op warnings. Inference uses fixed-shape zero-padded
batches of 8 and all decoding is per image; the Monte-Carlo draws are fixed arrays from the seed, and the
numba-parallel evaluator processes rows independently, so thread count does not change any result.

Timings (observational only):
Final run, RTX 5070 Laptop GPU (8 GB) + 32 GB RAM, deterministic mode:
  image loading                       7 s
  training per fold                   1419-1544 s (13.2-14.4 img/s); 3 folds 4416 s
  OOF + test inference per fold       33-40 s
  calibration + decode search + test decode   65 s (numba, all CPU cores)
  total                               4596 s (76.6 min)
Earlier measurement on a Kaggle Tesla T4 (v1, same model, 16 epochs, autotuned cuDNN): 15.2 img/s.
Deterministic kernels measured ~5% slower than autotuned ones on the laptop (14.3 vs 15.0 img/s).
A10G projection (estimate, not measured): the GPU part should run ~2x faster than the T4, i.e. about
33-36 min for 3 x 14 epochs including the deterministic slowdown, plus ~1 min inference. The decode
stage is CPU-bound: 65 s here; on a 4-vCPU machine it should take ~4-5 min (v1's search ran ~20x slower
on Kaggle's 4 vCPUs than locally, and v2's search does ~1/5 of v1's work per evaluation and fewer
evaluations). Projected total ~40-45 min against the 90-minute ceiling (54-minute safety target).


5. What worked / what did not
-----------------------------
Worked:
- Expected-metric decoding instead of global thresholds: images with only 4-5 regions (46% of train)
  were losing most of their utility to false positives forcing merged crops (delivered area ~3.5x the
  reference). Choosing the plan by Monte-Carlo expected score fixed most of that.
- Edge-uncertainty head: its predicted Laplace scale tracks the real edge error (corr 0.47, versus 0.32
  for box size) and drives per-image crop margins and the IoU-0.75 probability.
- Removing vertical-flip augmentation: the annotations sit ~1.3 units (0.8 px) above the image content.
  Vertical flips made the model average the offset away (top/bottom edge bias -1.47/+1.18), and removing
  them removed the bias (-0.33/+0.05). Matches with IoU >= 0.75 rose from 56% to 60%.
- Exact <= 3-rectangle min-area partition beats the greedy reference plan on 16% of train images (1.7%
  less area on average), leaving room for safety margins at no utility cost.
- Decode speed: v1's crop search took ~34 min on a 4-vCPU machine. v2 caps the exact DP at 8 boxes (same
  score) and uses one search sweep for the held-out halves, which brings the search to a few minutes.
Did not help: resnet50d (worse than resnet34d here), 256 instead of 128 Monte-Carlo worlds (no gain),
NMS on the crop candidates (rarely selected).


6. Leakage statement
--------------------
Every fitted object is fitted on training data only:
- detector weights: training images and their region labels (fold-wise);
- standardisers, logistic calibrators and every decode knob: out-of-fold predictions on training images
  and their labels;
- no scaler, vocabulary, statistic or threshold is computed from test images.
Test images only go through transform (fixed resize-if-needed) and predict. The forward pass always uses
fixed-shape zero-padded batches of 8. All post-processing (fold averaging, TTA averaging, peak extraction,
NMS, per-image calibration features, region count, crop plan) uses only the image's own predictions.
There is no train+test concatenation, no pseudo-labelling and no calibration against the test
distribution.
Batch-independence check: running test inference and decoding in original vs reversed order (different
batch companions, uneven last batch) gave bit-identical peak scores/boxes/scales (max |diff| = 0) and
21/21 identical crop plans.


7. Hardcoding statement and strip-the-ML test
---------------------------------------------
No discovered pattern of the data is hardcoded. The region boxes come from the trained detector, the
probabilities from trained calibrators, and every decode knob is searched in-script on OOF data.
- The documented reference-plan procedure is used only to compute D_ref, the metric's own denominator.
  It is used when scoring OOF predictions and inside each Monte-Carlo world of the expected-score decoder.
  The delivered crops are never the reference plan; they are minimum-area partitions of model-selected
  detections chosen by expected score.
- Fixed constants are model and algorithm settings (architecture, loss weights, augmentation ranges,
  epochs, Monte-Carlo world count, candidate limits 0.05 score / 20 crop candidates / 16 inclusion levels,
  local-search rounds). None is a decode threshold copied from an offline result.
Strip-the-ML test: with every trained model removed there are no detections, so every region list is
empty and every crop plan is the full-image fallback [0,0,1024,1024]. The pipeline produces no usable
answers without the trained models.


Execution
---------
Final run of the shipped v2: local machine (NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB), run at the
user's request because the Kaggle account's weekly GPU quota (45 h) was used up. Command:
  python -u solution.py "<challenge>/dataset/public" ".kaggle_runs/<slug>/local_run/submission.csv"
Exit code 0; solution.py sha256 e57719de... was identical before and after the run. The output passed
the local schema checks (546 rows, exact header, unique IDs equal to test.csv, integer in-range boxes,
no duplicates, <= 64 regions, <= 3 crops; mean 6.54 regions and 3.00 crops per image) and was copied
byte-identical to working/submission.csv.

Earlier remote execution (v1, superseded):
Platform: Kaggle private script kernel
Accelerator: NvidiaTeslaT4
Internet: enabled (only for downloading the timm/Hugging Face convnext_tiny.fb_in22k backbone weights)
Dataset handle: omerfarukmerey/eris-compact-packaging-of-annotated-skin-re-data (private; the slug is
  truncated to fit Kaggle's 50-character limit)
Kernel handle: omerfarukmerey/eris-compact-packaging-of-annotated-skin-re-solver (private), version 1
Fixed plan (v1): 3 folds x 1 seed x 16 epochs, batch 8, convnext_tiny.fb_in22k, 768x640, TTA none+hflip
Status COMPLETE in 6871 s on the T4 (decode search ~34 min of that); OOF held-out 0.5454; that
submission scored 0.6204 in the platform's CSV check. The kernel only carries solution.py (base64,
sha256-checked) and runs python3 solution.py <public_dir> /kaggle/working/submission.csv.


Known risks
-----------
- Calibration transfer: the calibrators and knobs are fitted on single-fold-model OOF scores but applied
  to the 3-model average on test. Averaging mostly lowers the scores of inconsistent (likely false)
  peaks, so the effect should be conservative, but it is not measured.
- Random image-level folds can overstate the held-out score (see section 3).
- v2 has not been executed on Kaggle (quota exhausted). Its training code is identical to v1 apart from
  14 instead of 16 epochs and the deterministic-kernel flags; v1 ran cleanly on the Kaggle image
  (torch 2.10). The decode changes are numpy/numba/scikit-learn only.
- The A10G runtime is a projection from T4 and laptop measurements, not a measurement.
