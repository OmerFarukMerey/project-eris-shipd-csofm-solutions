Anonymous Visual Operator Relay — solution notes
=================================================

PROBLEM (from PROBLEM.md)
-------------------------
Each case is one RGB contact-sheet PNG (456x432) holding 10 grayscale 96x96 panels:
four support (source,result) demonstrations, one query source, and a low-detail
context strip. Exactly three supports share one hidden spatial operator; one is a
plausible decoy from a different operator. The transformed query is never shown.
For every case we predict:
  - p_support_0..3  : probability each support follows the shared operator (decoy = 0)
  - increase_rle    : query pixels that meaningfully increase under the operator
  - decrease_rle    : query pixels that meaningfully decrease (disjoint from increase)
  - p_relay         : probability the whole decoded response is exactly correct

Submission columns (exact order, sample_submission.csv is authoritative):
  sample_id,p_support_0,p_support_1,p_support_2,p_support_3,increase_rle,decrease_rle,p_relay
RLE is 1-indexed, row-major, "start length ...", empty field = empty mask.

METRIC (from PROBLEM.md, weights sum to 1.0; masks dominate = 0.70):
  final = 100 * ( 0.20*support_utility + 0.22*increase_utility + 0.22*decrease_utility
                + 0.16*change_utility + 0.10*direction_utility + 0.10*relay_utility )
  support_utility = 0.60*mean_soft_support_iou + 0.40*exact_support_accuracy
  increase/decrease_utility = mean F1 (both-empty->1, one-empty->0)
  change_utility = mean IoU of the (inc OR dec) union
  direction_utility = mean 3-class pixel accuracy over the evaluated union
  relay_utility = 0.50*exp(-relay_logloss) + 0.50*exact_relay_accuracy
The exact metric is re-implemented in solution.py (final_score) and used for in-script
threshold search and holdout reporting.

Guidebook domain: CV / few-shot from-scratch (guidebook 5.2 / 5.5). A genuinely trained
model must PRODUCE the answers.

KEY DATA FINDINGS (measured in-script; drove the design)
--------------------------------------------------------
1. Panel geometry (verified by averaging many sheets): columns x0=[12,116,220,324],
   rows y0=[30,152,274], each 96x96; query at (row y0=274, x0=12). Intensity = mean(RGB)
   (all content panels are grayscale, R=G=B). Context strip is ignored (low value).
2. ~48% of increase masks and ~44% of decrease masks are EMPTY. Because F1=0 on any
   empty/non-empty mismatch, predicting emptiness correctly is the single largest lever
   (worth ~+20 of the 100 points).
3. The support source->result diffs almost perfectly transfer to the query's DIRECTION
   behaviour: whether the query increases/decreases is decided by whether the (three
   consistent) supports increase/decrease. A trained LogisticRegression on order-invariant
   support-diff summaries reaches ~1.00 held-out emptiness accuracy for both directions.
4. Decoy detection is NOT solvable from marginal statistics (the decoy is designed to match
   changed-pixel count/area/direction balance). The signal is CROSS-SUPPORT consistency:
   each support's source-context-conditioned change signature relative to the consensus of
   the other three (key feature: own-distance-to-consensus minus consensus-spread). A
   LightGBM on these ~51 odd-one-out features reaches AUC ~0.93 / top-1 ~0.79.
5. Spatial localization (WHERE the change lands) is the genuinely hard part. Even with the
   true supports and rich features, per-pixel AUC caps ~0.86 and non-empty F1 ~0.45-0.60 —
   some operators are under-determined from three demos. This is an information limit, not
   a model limit. A query-conditioned CNN with a large receptive field does best.

APPROACH — three trained models produce the answers
----------------------------------------------------
1) EMPTINESS GATE (sklearn LogisticRegression x2, one per direction).
   Features: order-invariant summaries of the four support diffs at a change threshold tau
   (sorted per-support increase/decrease fractions, #present self/other, the four
   (present-self, present-other) signature counts, mean odd-direction fraction over
   self-absent supports). tau is SEARCHED on the train fold. Decides, per direction and
   per case, whether to emit an empty mask.

2) SPATIAL CNN (from-scratch U-Net few-shot segmentation; PyTorch, ~1.3M params).
   - A shared encoder maps each panel (intensity + 6 cheap morphological basis channels:
     dilation, erosion, boundary, top-hat, local-std) to a 96x96 feature map + a global
     vector. The morphological channels give the decoder a geometric basis so it can SELECT
     dilate/erode/remove-small behaviour rather than synthesize it.
   - Per support, a soft change map from sign(result-source) at threshold tau pools source
     features into per-class (increase/decrease/none) prototypes and a 128-d operator
     descriptor.
   - Leave-one-out cosine consistency of the four descriptors yields decoy-resistant soft
     support weights; the operator prototypes and a global operator embedding are aggregated
     with those weights (so the operator stays clean even when the decoy is not identified).
   - A query decoder takes the query features, cosine-similarity-to-prototype maps, and a
     FiLM conditioning from the operator embedding, and emits a per-pixel 3-class softmax
     {none, increase, decrease}. The 3-class softmax makes increase/decrease STRUCTURALLY
     disjoint. Loss = focal 3-class + masked soft-dice (empty targets skipped) + presence
     BCE + a leave-one-out consistency BCE against the support labels.
   - D4 (flip + rot90) augmentation is applied jointly to all panels, the query and the
     masks (the operators are reflection/rotation compatible; no intensity jitter, which
     would flip the sign of the change).
   Decode: for each direction the gate first decides empty vs non-empty; if non-empty, the
   CNN softmax is turned into a mask. The decode config (presence cut, structure-percentile
   trim, and either a global probability threshold OR a per-case top-K count derived from the
   support change fill) is SEARCHED on the train holdout against the real increase+decrease F1.

3) DECOY / SUPPORT (LightGBM on cross-support odd-one-out consistency features).
   Per support: a 16-d source-context-conditioned change signature (increase/decrease
   fractions at interior/boundary/exterior structure pixels, in 4 brightness bins, and at
   high-edge pixels) plus 11 marginal stats, then odd-one-out features relative to the other
   three supports (distance/median-distance/cosine to consensus, consensus spread, the key
   distance-minus-spread term, per-dim deviation, odd-one-out rank). p_support_i = 1 - decoy
   probability. Falls back to a class-balanced LogisticRegression if LightGBM is unavailable.

RELAY CONFIDENCE
   A LogisticRegression calibrator is fit on the held-out fold's ACTUALLY-achieved relay
   outcome (recomputed with the exact metric on our own holdout predictions) using confidence
   features (support-probability margin, gate probabilities, CNN presence probabilities,
   predicted mask areas, both-empty flag). If the holdout has too few positives it falls back
   to a calibrated constant equal to the holdout relay base rate.

VALIDATION
----------
Random 85/15 split of TRAIN (case-level; each contact sheet is one case, so there is no
group leakage). All thresholds and calibrators are fit on the 85% and evaluated on the 15%
holdout using the exact PROBLEM.md metric (final_score in solution.py). The script prints the
full holdout breakdown each run. tau, the decode config, and the relay calibrator are all
searched/fit in-script on that holdout — no constant is pasted in from offline.
(Local MPS/CPU runs use a reduced-epoch smoke config; the platform GPU trains the full model.)

LEAKAGE STATEMENT
-----------------
Every model (both LogisticRegression gates, the LightGBM decoy model, the CNN, the relay
calibrator, all StandardScalers) is fit on TRAIN rows only. Test is used strictly for
per-row transform/predict: each test case's own four support panels and query panel produce
its prediction. No statistic, threshold, vocabulary, scaler, or calibration is computed from
test rows; there is no train+test concatenation anywhere; there is no cross-test-row
aggregation. Support-fill fractions used by the decoder are computed per case from that
case's own panels.

HARDCODING STATEMENT
--------------------
No discovered generation pattern is hardcoded. The emptiness decision is a trained
LogisticRegression (not an asserted rule), the decoy decision is a trained LightGBM, and the
masks are produced by the trained CNN. Every constant that affects an output — the change
threshold tau, the decode presence cut / structure-percentile / probability-threshold /
top-K ratio, and the relay calibration — is SEARCHED or FIT in-script on the train holdout
against the real metric, never pasted in. Panel crop coordinates and the RLE format are
fixed dataset facts (image geometry), not learned regularities.

STRIP-THE-ML TEST
-----------------
Remove every trained model and the pipeline produces nothing usable: with no LogisticRegression
gate there is no empty/non-empty decision; with no CNN there are no per-pixel masks; with no
LightGBM/LogReg there are no support probabilities. The trained models PRODUCE the answers; the
crafted features and morphological channels only SUPPORT them as inputs.

ROBUSTNESS
----------
A schema-valid placeholder submission (empty masks, p_support=0.75, p_relay=0.10) is written to
submission_out immediately after reading test, then overwritten with real predictions. Training
has a wall-clock guard (stops launching epochs past RELAY_TRAIN_DEADLINE); inference has a hard
deadline that fills any remaining rows with the fallback prediction (and skips straight to the on-disk
placeholder if the deadline is exceeded before test inference begins). Per-row prediction failures
fall back to a valid row rather than aborting the run. Random seeds are fixed (python, numpy, torch).
The script reads only public_dir and writes only submission_out.

The env vars RELAY_SMOKE / RELAY_DEVICE / RELAY_EPOCHS / RELAY_TRAIN_DEADLINE / RELAY_HARD_DEADLINE are
LOCAL testing knobs only and are unset on the platform. In SMOKE mode the run intentionally truncates
train/test, so a local smoke submission has fewer than 1500 rows — this never triggers on the platform,
which always produces the full-length submission.

WHAT WORKED / WHAT DID NOT
--------------------------
- Worked: the emptiness gate (~+20 pts, near-perfect and cheap); the odd-one-out LightGBM decoy
  (top-1 ~0.79 vs 0.25 chance); the D4 augmentation; decoy-resistant leave-one-out operator
  aggregation (decouples mask quality from decoy-ID accuracy); the 3-class softmax for structural
  disjointness.
- Modest: the spatial CNN lifts increase/decrease F1 on non-empty cases above the intensity-only
  baseline, but the operator is genuinely under-determined from three demonstrations, so non-empty
  F1 tops out ~0.5-0.6. Prototype-only matching underperforms a discriminative (FiLM + nonlinear
  decoder) head, which is why both are combined.
- Did not help / avoided: raising the change threshold C (misses faint real changes); aggressive
  gradient/detail structure gates (recall collapses); intensity/contrast augmentation (flips the
  sign of the change); more brightness/edge bins in the decoy signature (per-bin noise). Relay
  targets are rare (they require exact supports AND inc/dec F1>=0.90 AND change_iou>=0.90), so
  p_relay is calibrated low and relay utility comes mostly from calibration, not from hits.
