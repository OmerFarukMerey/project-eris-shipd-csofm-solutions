KineScope: Multimodal RAG Evidence Calibration -- solution notes (version 2)
=============================================================================

PROBLEM (as read from PROBLEM.md)
---------------------------------
Each row pairs a query event representation (left_0..left_63) with a retrieved
evidence representation (right_0..right_63); both blocks share one anonymous
coordinate system and every occurrence carries independent nuisance noise.
The target is a continuous support score in [0, 1] for the claim "the query
event is more persistent than the retrieved event": ~0 = evidence contradicts,
0.5 = neutral (similar persistence), ~1 = strongly supports.  Swapping the two
blocks reverses the relation.  Evaluation sessions are disjoint from training
sessions, so the evaluator has to learn a transferable query-evidence relation.

Submission schema (exact): CSV with header  id,prediction  in that order, one
row per test id (800 rows, each id exactly once, no unknown ids), prediction a
finite float in [0, 1].  Row order is irrelevant (grader aligns by id).

Metric: RMSE skill against the neutral prediction,
    score = max(0, 1 - RMSE / null_RMSE),  null_RMSE = RMSE of predicting 0.5.
Higher is better; predicting 0.5 everywhere scores exactly 0.  The same
formula is applied independently to the public (255) and private (545) rows.

Guidebook domain: RAG / Retrieval (section 5.3) -- the submission is a learned
re-ranker / evidence evaluator, so the model that produces the score must be a
genuinely trained ranking model.  PROBLEM.md restrictions honoured: only the
released files are used; no external media, labels, datasets, APIs, pretrained
event models, embeddings or checkpoints; the script never tries to reconnect
occurrences, events or sessions; ids, row order, hashes, byte layout and numeric
formatting are never used; no leaderboard probing; the submission is generated
automatically from train.csv and the test features.

Runtime: PROBLEM.md states no ceiling or hardware, so the conservative guidebook
values are used (1 hour) and the plan is sized to fit on CPU as well as GPU.


WHAT WENT WRONG IN VERSION 1 AND WHAT THE DATA ACTUALLY LOOKS LIKE
------------------------------------------------------------------
Version 1 (equal blend of a Siamese MLP, an RBF network and a Siamese kernel
ridge, probit link) had a random-row out-of-fold skill of 0.68 but scored 0.362
on the public leaderboard.  Offline diagnostics on the TRAINING set explained
the gap (these diagnostics are analysis only; nothing of them is in solution.py):

* Every training event vector has a near neighbour at distance ~1.5 while
  unrelated events are ~10 apart: the 7,260 occurrences come from a few hundred
  source events (~20 noisy occurrences each), and pairs linked through shared
  events fall into 11 groups of 330 rows = the 11 training sessions.
* Random-row cross-validation therefore rewards memorising source events.  The
  honest protocol is leave-one-session-out (LOSO, 11 folds): version 1's kernel
  member scored 0.31 there, a linear difference model 0.35, a wide MLP 0.37.
  Holding out new events inside known sessions gives ~0.43, so most of the loss
  is generalisation to unseen events, the remainder is session shift.
* Per-session LOSO skill varies from ~0.3 to ~0.6 for every model, so public and
  private scores depend strongly on which sessions they contain.


APPROACH (version 2)
--------------------
One model family, chosen for transfer to unseen events and sessions:

    g(query, evidence) = s(query) - s(evidence),      prediction = Phi(g)

s(.) is a trained per-event scorer shared by both blocks (RankNet / Thurstone
form) and Phi the standard normal CDF, so prediction(L, R) + prediction(R, L) = 1
exactly by construction (verified in the log to 3e-8).  The scorer is

    s(x) = MLP( gate * standardise(x) )

* gate: a learned per-dimension input scale with an L1 penalty (soft feature
  relevance; the persistence signal lives in mid-variance directions while the
  top-variance directions carry acquisition-condition variation);
* MLP: 3-4 hidden layers of 512 SiLU units, dropout 0.5, AdamW with weight decay
  0.1, one-cycle schedule, fixed 100-150 epochs, batch 128, learning rate 1e-3;
* loss: squared error directly in probability space (the metric's space) plus
  the gate penalty;
* standardiser fitted on the training(-fold) events of both blocks.

All training happens inside solution.py, on every run, from train.csv only.

In-script model selection with a shift-aware validation (train only):
* pairs are grouped by k-means (k=8) clusters of their midpoints (pseudo
  acquisition conditions) and held out with 4-fold GroupKFold;
* for every validation pair the distance of each block to the nearest
  training-fold event is computed; the (bimodal) distance distribution is split
  in-script with 1-D 2-means and only pairs beyond the midpoint ("far rows",
  i.e. events unseen in the training fold) are scored.  This mimics disjoint
  evaluation sessions without reconnecting occurrences or sessions;
* grid (hidden, depth, dropout, weight decay, gate L1, epochs):
      (512, 4, 0.5, 0.1, 0.05, 150), (512, 3, 0.5, 0.1, 0.05, 150),
      (512, 4, 0.5, 0.1, 0.05, 100)
  scored by far-row skill; the best configuration is refit on all 3,630 rows
  with seeds 0, 1, 2 and the three predictions are averaged.
* A schema-valid neutral (0.5) placeholder is written immediately after reading
  test.csv and overwritten at the end; non-finite predictions fall back to 0.5
  per row; a pipeline exception leaves the placeholder standing.

Feature engineering: none beyond the train-fitted standardiser and the learned
gate; the network consumes the raw 64-d vectors of each block.


VALIDATION STRATEGY AND SCORES (train only)
-------------------------------------------
Offline development used leave-one-session-out (11 folds) on train; the script
itself uses the cluster-grouped far-row CV described above, which ranks the
candidate configurations in the same order as LOSO (checked offline: depth 2
0.400 < depth 3 0.420 < depth 4 0.440 < depth 4 with 150 epochs 0.450 on the
far-row proxy, versus 0.436 < 0.467 < 0.479 < 0.482 under LOSO).

LOSO skill (metric of PROBLEM.md) of the main candidates:

    gated Siamese MLP 512x4, wd 0.1, dropout 0.5, 150 epochs   0.482  (selected)
    same, 3 seeds averaged                                     0.481
    same with 200 epochs / width 768 / gate 0.03 / depth 6     0.476 / 0.477 / 0.475 / 0.463
    gated Siamese MLP 512x3, 100 epochs                        0.467
    gated Siamese MLP 512x2, 100 epochs                        0.436
    small ungated MLP 128x2, wd 0.1                            0.414
    wide MLP 1024x2, wd 1e-2, 200 epochs (version-1 style)     0.368
    Siamese kernel ridge (best smooth setting)                 0.364
    version-1 kernel ridge setting                             0.313
    linear model on left-right (probit)                        0.348

Per-session LOSO skill of the selected model: 0.53 0.56 0.36 0.43 0.55 0.35 0.59
0.52 0.51 0.48 0.47 -- the spread is a property of the sessions, not of seeds.

Things that did NOT transfer (all tested under session-level CV): difference-
only models (cancelling a shared offset), removal of between-cluster
directions, whitening or PCA-reduced inputs, session-conditioned linear kernels,
bilinear/quadratic scorers, ratio and log-linear scorers, Gaussian-process ARD,
group-DRO reweighting, leave-cluster-out bagging, input noise, blends with the
linear or kernel models (weights go to zero once the gated MLP is deep).

After the public score of this version (0.436) a further 15-recipe LOSO batch
was run on a Kaggle T4 (research kernel, not the submission): sharpness-aware
minimisation (0.480), 1024 wide with dropout 0.6 (0.480), batch 64 (0.475),
GELU (0.475), heavier decay (0.475), slower schedule (0.475), DRO (0.468), EMA
weights (0.462), group-lasso input relevance without the gate (0.438-0.448),
residual scorers of depth 6-8 (0.431-0.434); the shipped recipe re-measured at
0.478 on the GPU and the best pairwise average reached 0.481.  The family is
saturated at ~0.48 LOSO; the remaining differences are inside the noise of an
11-session estimate, so the shipped model was kept.

In-script far-row skills from the final Kaggle run (pseudo-groups of sizes
633/334/660/331/491/330/418/433, far threshold 5.33 found in-script, 2,690 far
rows of 3,630):
    (512, 4, 0.5, 0.1, 0.05, 150)   far-row 0.4532   all rows 0.4838   <- selected
    (512, 3, 0.5, 0.1, 0.05, 150)   far-row 0.4453   all rows 0.4778
    (512, 4, 0.5, 0.1, 0.05, 100)   far-row 0.4350   all rows 0.4686
Far-row OOF RMSE per target band of the selected model: 0.148 / 0.176 / 0.153 /
0.170 / 0.149 (mean prediction 0.19 / 0.38 / 0.51 / 0.62 / 0.82 against mean
target 0.10 / 0.30 / 0.50 / 0.70 / 0.91): calibrated, mildly shrunk towards 0.5.
Learned gate after the full refit: mean |gate| 0.09, about 40 of 64 dimensions
below 0.1 and none above 0.5 (the gate acts as a strong relevance re-weighting
of the input, compensated by the first layer).


STATIC PLAN AND MEASURED TIMING (observational only; nothing gates on time)
-----------------------------------------------------------------------------
Plan: k-means pseudo-groups (8) + nearest-neighbour far-row mask; 3 grid
configurations x 4 folds; final refit 3 seeds; 100-150 epochs, batch 128.
Nothing in the plan depends on hardware, elapsed time, quota or download
success.  torch.set_num_threads() and device placement (GPU if present,
otherwise CPU) are the only environment-dependent knobs and affect speed only.
Measured on the Kaggle T4 kernel (final run, wall clock 187 s in total):
    grid config 1 (512x4, 150 ep)   4 folds   53.5 s  (13.4 s per fold)
    grid config 2 (512x3, 150 ep)   4 folds   41.3 s  (10.3 s per fold)
    grid config 3 (512x4, 100 ep)   4 folds   31.7 s  ( 7.9 s per fold)
    final refits                    3 seeds   16 s each on 3,630 rows
    k-means groups + far mask                  0.3 s
On a 4-vCPU CPU (no GPU) the earlier version of this script ran 4.3x slower
than a laptop and about 5-6x slower than the T4 for comparable networks, so the
CPU projection for this plan is roughly 20 minutes, inside the 1-hour ceiling
with more than 60% margin.  These numbers are observational; the plan is fixed.


LEAKAGE STATEMENT
-----------------
* Test rows are used for inference only.  Taint list: the raw test feature
  blocks, the three per-seed predictions and their average.  Every operation is
  per-row: a train-fitted standardiser, a forward pass of a trained Siamese
  network, a fixed 1/3 average across seeds, clipping, writing the row.  No
  statistic, scaler, cluster, threshold or calibration is fitted on, counted
  over, or adjusted to test rows; train and test are never concatenated.  The
  only cross-row reads of test predictions are the mean/std/min/max in one log
  line, which influence nothing.
* The validation machinery (k-means groups, nearest-neighbour distances, the
  far threshold) uses training rows only; inside each fold the standardiser is
  fitted on the fold's training partition and the far mask is computed against
  the fold's training events.
* The script reads only <public_dir>/train.csv and test.csv and writes only
  <submission_out>.


HARDCODING STATEMENT
--------------------
* No discovered generation pattern is encoded as a rule: no lookup table,
  if-chain, template or regex maps inputs to outputs, and the script never
  reconnects occurrences, events or sessions.
* Everything that shapes a prediction is learned (gate, network weights,
  standardiser statistics) or searched in-script on train-only validation (the
  configuration among the grid).  Remaining constants are the static plan
  (grid, folds, seeds, epochs, batch size, learning rate, k=8 pseudo-groups),
  the neutral value 0.5 defined by the problem, the probit link (a modelling
  choice) and the equal 1/3 seed averaging.  The far-row threshold is computed
  in-script from the training distances, not asserted.
* Swap antisymmetry is enforced by the architecture, not by asserting labels or
  generating augmented rows.  No synthetic data, no input-noise augmentation.
* Strip-the-ML test: with the trained networks removed the pipeline produces
  only the neutral placeholder 0.5 for every row, i.e. a score of exactly 0.


COLD-REVIEWER PASS
------------------
Searched for (1) a hardcoded generation rule: none; the grid is searched and the
1/3 seed weights are a fixed default.  (2) What a non-model component produces:
only the 0.5 placeholder.  (3) Test-derived variables read across rows: the
summary log line only.  Grep of time./cpu_count/cuda: every hit is a log line,
the thread-count knob or device placement; no if-statement gates training or
prediction on any of them; no monotonic/perf_counter/deadline/budget usage.


REMOTE EXECUTION
----------------
Platform: Kaggle private script kernel (transport wrapper lives outside the
challenge directory in .kaggle_runs/kinescope-multimodal-rag-evidence-calibration/;
it locates the private dataset, materialises the exact solution.py bytes and
runs  python3 solution.py <public_dir> <submission_out>).
Accelerator: NvidiaTeslaT4 (neural training); the same plan runs on CPU.
Internet: disabled.
Dataset handle: omerfarukmerey/eris-kinescope-multimodal-rag-evidence-calib-data (private)
Kernel handle:  omerfarukmerey/eris-kinescope-multimodal-rag-evidence-solver (private)
Fixed plan: 8 pseudo-groups, 4-fold group CV on far rows, 3 grid configs,
3 final seeds, 100-150 epochs, batch 128.
Kaggle run: kernel version 4, status COMPLETE, device cuda, solution.py wall
time 187 s (observational).  The remote solution.py is byte-identical to the
solution.py in this directory.  The downloaded submission.csv passed the schema
validation (header id,prediction; 800 rows; every test id exactly once; all
finite in [0.016, 0.977]; mean 0.492, std 0.238) and was copied unchanged to
working/submission.csv.  Version-1 kernel runs 1-2 failed inside the wrapper
(dataset path discovery); run 3 was the version-1 model (public LB 0.362).


WHAT WORKED / WHAT DID NOT
--------------------------
Worked: building the swap antisymmetry into the model; the Siamese per-event
scorer; the probability-space loss; strong weight decay + dropout + an L1 input
gate + a short fixed schedule, which stop the network from memorising the
training events; depth 3-4 (each extra layer up to four added ~0.01-0.03 LOSO);
a session-level validation protocol, without which none of this was visible.
Did not help: seed averaging (neutral), kernel and RBF models (they revert to
0.5 off-distribution and memorise on-distribution), every feature-space
"session normalisation" attempt, gradient-boosted trees (0.45 even on random CV).
Risk/limitation: LOSO skill 0.48 is an average over sessions ranging 0.35-0.60;
the public split (255 rows, roughly one session) can land anywhere in that range.
