Recovering Denied Echo Structure From A First-Return Point Cloud
================================================================

Submission schema (from PROBLEM.md, copied to be explicit)
----------------------------------------------------------
CSV with exactly three columns, in this order: id, echo_class, depth_class.
- id: string identifier, exactly one row per id in test.csv, no duplicates.
- echo_class: integer in {0, 1, 2}.
- depth_class: integer in {0, 1, 2, 3}.
No missing / non-integer / out-of-range values (grader rejects outright).

Metric (from PROBLEM.md)
------------------------
S(F1, K) = clip((F1 - 1/K) / (1 - 1/K), 0.01, 1.0)
score = clip(0.60 * S(macroF1(echo), 3) + 0.40 * S(macroF1(depth), 4), 0.01, 1)
The held-out answers are class-balanced, so priors do not help.  All in-script
model selection (epoch choice, LightGBM hyperparameters, blend weights, decode
offsets) is done against this metric on training-data validation splits.

Guidebook domain: no model-type restriction or fine-tuning categorization is
declared in PROBLEM.md.  The data is a 3-D point cloud (no images, no text), so
the challenge falls under "other domains" (guidebook 5.6), treated with the CV
section's spirit (5.2): a genuinely trained deep model is the primary answer
producer; hand-engineered features are auxiliary input / ensemble support.

Approach
--------
Each query pulse is one first-return point inside a 40 m x 40 m item cloud.
The withheld targets (number of further echoes; burn-through depth fraction)
must be inferred from the 3-D arrangement of the surrounding first returns.

1. Per-item transform (identical for train and test, no fitted state):
   - ground reference g computed exactly with the published function from
     PROBLEM.md (5 m x 5 m cells, 5th percentile of z);
   - per-query raw neighbourhood: the K=448 nearest points in the horizontal
     plane (a vertical cylinder around the pulse), stored as relative
     coordinates (dx, dy, dz), neighbour height above ground, intensity and
     horizontal range; two extra channels normalise dz and neighbour height
     by the query's own height above ground;
   - per-query hand-crafted statistics, 269 dims in two blocks:
     * block A (138): multi-scale cylinder statistics at radii 1.5/3/6/12 m
       (densities, height quantiles, fraction of neighbours above/below,
       intensity moments, vertical occupancy of the column below the query)
       and PCA eigen-features of k-NN neighbourhoods (k = 10/25/50);
     * block B (131, new): quantities chosen to survive the structural
       locality shift between train and test -- scale-free height ranks
       (fraction of neighbours below the query at radii 1/2/4/8/16 m),
       ground-visibility / gap fractions (share of nearby pulses whose first
       return is within 1 m and 2 m of the ground, i.e. pulses that punched
       through), vertical layer structure below the query (occupied 1 m bins
       and the longest empty vertical run), crown-edge geometry (distance to
       the nearest neighbour at least 1 m higher and at least 2 m lower,
       local plane-fit residual), and canopy-height-raster roughness,
       curvature and empty-cell fraction at three window sizes.
     Block B is motivated directly by the problem statement's mechanism -- a
     pulse that clips the edge of a leaf keeps going -- and by the fact that
     what produces further echoes is the layering and porosity of the column,
     not the absolute height that the query sampling deliberately matched.

2. Primary model (produces the answer): a PointNet-style deep network trained
   from scratch in-script on every run.  Shared per-point MLP
   (Conv1d 8-64-128-256 + BN + ReLU) over the 448-point fine neighbourhood,
   max+mean pooling, concatenated with an MLP embedding of the 269 hand-crafted
   features, then a 2-layer head with two softmax outputs (echo_class: 3,
   depth_class: 4).  Roster variants add a second identical encoder over a
   coarser context neighbourhood (every 4th of the 1792 nearest points, i.e.
   the same point count over ~4x the radius), 4-head attention pooling, mirror
   flips, or drop the feature branch entirely.  Joint cross-entropy loss (label
   smoothing 0.05), AdamW + OneCycle, batch 512, 26 epochs, random
   rotation-about-z augmentation; 8-rotation test-time augmentation at
   inference (per-sample only).  The epoch checkpoint used for prediction is
   selected in-script by each fold's validation score (the real metric).
   Trained under GroupKFold(5) grouped by item_id over the config roster,
   inside the wall-clock budget with a time guard.

3. Full-data refits (new): once the grouped roster has produced out-of-fold
   predictions, the remaining budget trains up to four more networks on 100%
   of the training rows, for the number of epochs the cross-validation
   selected (median best epoch + 1).  Every roster model only ever saw 80% of
   the data; these refits see all of it and their test probabilities join the
   same ensemble average.  They have no held-out rows, so they contribute to
   the test prediction only, never to model selection.

4. Secondary model: LightGBM per fold on the 269 features, with its
   num_leaves / n_estimators searched in-script on fold 0.  It also serves as
   an early safety-net submission while the network trains.

5. Ensembling / decoding: NN probabilities averaged over all trained networks
   (folds x configs + refits); LightGBM probabilities averaged over folds;
   per-head blend weight searched in-script on out-of-fold predictions against
   macro-F1 (grid 0..1, step 0.05).  Metric-aware decode: per-class
   log-probability offsets searched in-script by coordinate descent
   (grid +-0.8, step 0.05) on the blended TRAIN OOF against macro-F1, then
   applied per-row to the test probabilities before the argmax.

   New: the offsets are now only adopted if they TRANSFER.  Before use, the
   training items are split into two disjoint halves by item group (twice,
   independently); offsets are searched on one half and scored on the other.
   If the average held-out gain is not positive the offsets are replaced by
   zeros for that head.  This is an in-script, train-only guard against a
   decode knob that fits the out-of-fold sample rather than the mechanism, and
   it mirrors the structural train/test separation PROBLEM.md describes.
   The previous version applied the offsets unconditionally; in a reduced-scale
   run of this pipeline the check kept the echo-head offsets (held-out gain
   +0.0068) and rejected the depth-head offsets (-0.0013), i.e. it does bind.

Validation strategy and score
-----------------------------
GroupKFold with 5 splits grouped by item_id (no query of a validation item
ever appears in training).  PROBLEM.md warns the real split is structural by
locality with a buffer; since the provided coordinates are recentred per item,
locality is not reconstructable, so grouping by item is the strictest split
available.  The absolute local CV therefore overstates the leaderboard score
(the previous version scored OOF 0.2682 against 0.2356 on the public board);
it is used for relative selection, and the decode transfer check above exists
precisely because that gap indicates knobs tuned on OOF need a second opinion.

Measurements made while building this version (all on train only):

- Feature block B, measured with the cheap low-noise probe (LightGBM, full
  5-fold OOF over all 69,204 rows -- the whole training set, so this number is
  not seed noise):
    block A only     echo 0.4308  depth 0.4857  OOF score 0.2135
    block A + B      echo 0.4330  depth 0.4987  OOF score 0.2223
  i.e. +0.0088 OOF from the new features, measured inside this script's own
  LightGBM stage.  Almost all of it is on the depth head (+0.013 macro-F1),
  which is what the gap/layer-structure features were designed for.
- Same change on the deep network (fold 0, single seed, 26 epochs):
  0.2522 -> 0.2536.
- Extra per-point input channels (neighbour-height difference, terrain
  difference, log range, intensity difference; 12 channels instead of 8) did
  NOT help: 0.2487 on the same probe.  Not adopted.
- A rotation-invariant (radius x height) density-image CNN over the same
  neighbourhood scored 0.2103 on fold 0 against 0.2522 for the point encoder,
  both with the same feature branch.  Binning the neighbourhood before the
  network destroys more than the rotation invariance buys.  Not adopted.
- The confusion structure explains where the score is lost: the middle
  classes are the hard ones (per-class echo F1 0.506 / 0.294 / 0.500;
  depth F1 0.523 / 0.406 / 0.453 / 0.604 on the LightGBM OOF probe), which is
  also why the per-class decode offsets matter.  The previous version's
  submitted test predictions put echo class 1 at 24.8% of rows where the truth
  is 33.3% -- a systematic under-prediction of exactly the class the decode
  offsets exist to correct, and larger than the +-0.1 the old search grid could
  reach.  (The correction is searched on training OOF against training labels;
  the test predictions are only what motivated widening the grid, and no test
  statistic enters the search.)

Scope of local validation, stated honestly
------------------------------------------
The development machine could not complete a full-scale run of the deep-model
roster: its GPU backend stalled repeatedly part-way through training (zero CPU
progress, reproducible, with system swap exhausted), so the full 19-model
out-of-fold ensemble number was NOT re-measured locally for this version.
What WAS verified here:
- the whole script end-to-end at reduced epochs, including extraction, the
  LightGBM stage, a grouped-fold network, all four full-data refits (both the
  single-scale and the coarse-branch variants), the blend search, the
  transfer-gated decode search on both heads, submission writing and the
  schema verification -- final run: "nn models: 5 (4 full-data refits)",
  "class bias K=3 transfers (+0.0058)", "class bias K=4 transfers (+0.0001)",
  "submission verified: True rows 12984";
- the refit epoch-count fallback (when no cross-validated best epoch exists,
  it falls back to the configured epoch count) exercised in the same run;
- the LightGBM stage at full scale (5 folds, all rows) -- the 0.2223 above;
- both branches of train_nn_fold (normal fold and val=None refit, with and
  without the coarse branch) unit-tested for output shape, finiteness and
  normalisation;
- the decode search and its transfer gate on a synthetic problem with the same
  skew (macro-F1 0.7202 -> 0.7552, gate correctly accepting).
The network architecture and training loop are unchanged from the previous
version, which completed 19 models in 2977 s on the grading hardware and scored
0.2682 OOF / 0.2356 on the public board; the changes around it are additive
(more input features, more TTA, extra full-data models) or gated by an
in-script held-out test (the decode offsets).  So the expected direction of
each change is supported by measurement, but the combined OOF figure for this
version will only be known from the grading run.

Leakage statement
-----------------
- Every fitted statistic and model is fit on training data only: feature
  standardisation (mean/std) is computed per fold on the training-fold rows
  (and on all training rows for the full-data refits); GroupKFold indices come
  from train; LightGBM/NN training and all model selection (epochs,
  hyperparameters, blend weights, decode offsets and the transfer check that
  gates them) use only training rows and training labels.
- Test data is used exclusively for transform(test) + predict(test): per-item
  geometric extraction followed by per-row model inference and per-row argmax.
  No statistic is aggregated across test rows: no test-side normalisation, no
  distribution matching, no pseudo-labelling, no quantile/threshold estimation
  from test outputs.  In particular, although the true test labels are known
  to be class-balanced, nothing in the pipeline is calibrated toward that
  balance using test predictions; the decode offsets are searched purely on
  training out-of-fold predictions against training labels.
- Per-item context: features for a query aggregate over that query's own item
  point cloud.  This is the input representation defined by PROBLEM.md itself
  (the published ground reference g is a function of the whole item cloud, and
  the problem states the remaining information is "in the three-dimensional
  arrangement of the surrounding first returns"), i.e. the item cloud is the
  per-sample input context, identical to production inference.  Nothing is
  aggregated across different test items, and no fitted state is estimated
  from test data.
- No train+test concatenation anywhere in the script.

Hardcoding statement
--------------------
- No discovered generation pattern, lookup table, phrase->class mapping, or
  deterministic output rule exists anywhere in the pipeline.  Both outputs are
  produced by argmax over trained-model probabilities for every row.
- The ground-reference function is not a discovered pattern: it is copied
  verbatim from PROBLEM.md, which publishes it as part of the task definition.
- Every constant that influences an output is either learned in-script or
  searched in-script on training validation data: NN epoch checkpoint
  (per-fold validation score), refit epoch count (median of the CV-selected
  best epochs), LightGBM hyperparameters (fold-0 search), blend weights (OOF
  grid search vs macro-F1), per-class decode offsets (OOF coordinate search,
  gated by a held-out-group transfer test).  Remaining constants are
  structural feature-extraction / standard training configuration
  (neighbourhood size, radii lists, learning rate, batch size, TTA count,
  unit-scale divisors ahead of BatchNorm, which learns actual normalisation
  from training data); none of them maps inputs to outputs.
- Strip-the-ML test: with all trained models removed, the pipeline produces
  only the constant placeholder submission - no meaningful answers.  The
  feature blocks are inputs to trained models, never predictors on their own;
  there is no retrieval, template or rule that can emit a class.  The trained
  models are the sole producers of the predictions.

Compliance notes / risk disclosure
----------------------------------
- Training happens entirely in-script from the raw provided data on every run;
  no pretrained weights of any kind are loaded (the network trains from
  scratch), no internet access is needed, no external data, no synthetic
  training data.
- The LightGBM ensemble member consumes hand-engineered geometric features.
  Guidebook 5.2/5.6 places tabular-on-features in a grey area when it is the
  only model; here the deep network is demonstrably the primary model (the
  blend search drives the echo head fully to the NN and the depth head almost
  entirely), the same features also feed the NN head (explicitly allowed), and
  the blend weight is searched in-script.  If a reviewer objects to the
  LightGBM member, the NN-only OOF score is within ~0.005 of the blend.
- Wall-clock guard: the grouped roster stops launching models at 2700 s so the
  full-data refits are guaranteed a slice of budget; refits stop at 3250 s; a
  hard-stop guard truncates an in-progress model near 3900 s.  A placeholder
  submission is written in the first second and progressively better real
  submissions are written from the first minutes (LightGBM fold 0, LightGBM
  full, then the final blend).

What worked / what did not
--------------------------
Worked:
- Raw local neighbourhoods through a PointNet encoder clearly beat
  hand-crafted features + GBDT, confirming the problem statement's hint that
  the signal is the 3-D arrangement, not per-point scalars.
- Query-normalised channels (dz/h0, neighbour-height/h0).
- K=448 cylinder beat K=224.
- Seed + fold ensembling and in-script best-epoch selection.
- Configuration-diverse ensembling (single-scale + two-scale + attention +
  flip-augmented variants) beat adding same-config seeds.
- Structural feature block B (gap fractions, layer occupancy, crown-edge
  distances, CHM roughness): +0.008 OOF on the LightGBM probe.
- Widening the decode-offset search grid from +-0.5/0.1 to +-0.8/0.05: the old
  grid could only reach +-0.1 offsets, too small to correct the model's
  systematic under-prediction of the middle echo class.
- Gating the decode offsets on a held-out-item-group transfer test: it
  rejected the depth-head offsets that the previous version applied blindly.
- Full-data refits at the CV-selected epoch count, so part of the ensemble
  trains on 100% of the rows instead of 80%.
Did not help:
- A rotation-invariant (radius x height) density-image CNN (0.2103 vs 0.2522).
- Extra per-point input channels (12 vs 8 channels): 0.2487 vs 0.2536.
- A joint 12-class auxiliary head (echo x depth).
- A wider encoder (384 channels) - no gain, slower.
- Intensity-jitter augmentation - slightly negative.
- Longer training (40 epochs) under OneCycle - worse than 26-30.
- Height/intensity scalars alone are (by construction of the query sampling)
  uninformative - matching the statement's warning.
