Vector Fragment Route Program Repair — solution notes
======================================================

APPROACH
--------
The task is to reconstruct, for each drawing, the hidden route program
(start fragment, successor links, per-fragment orientation) over shuffled
vector fragments. I treat it as structured sequence decoding driven by
machine-learned scoring models: all route decisions (which fragment starts,
which fragment follows, which direction each fragment is traversed) are made
by LightGBM models trained inside the submission script on train.csv only,
combined by a beam search that enforces the only hard constraint of the task
(the links must form one simple path visiting every fragment exactly once).

Data insights that shaped the features (all measured on train only):
- Each row's fragments group into `component_count` spatial components; the
  canonical route interleaves components (braids) rather than finishing one
  component first, so the models get state features describing per-component
  visit history (remaining counts, steps-since-visited, etc.) that let them
  learn the braiding convention from data.
- Fragments of a component converge on a shared "hub" region (most points of
  each fragment bunch near a common location). An oriented fragment runs from
  a distal head, over one long jump, into a squiggle near the hub. Hub-relative
  distances/angles are therefore the core geometric features.
- The layout_hint families put components in characteristic places (left/right,
  top/bottom, outer/inner by bounding-box margin, center/sides, diagonal), so
  cluster-rank-along-axis features (rank by centroid x, y, x+y, x-y, distance
  from canvas center, margin, reach) let the models learn which component
  starts and how the cycle proceeds.

MODEL ARCHITECTURE / ALGORITHM
------------------------------
Five learned stages, trained end-to-end inside solution.py (LightGBM, with
LightGBM+XGBoost ensembles for the two heaviest models):

0. outer_inner split model — for outer_inner rows the two components are
   concentric, so KMeans is unreliable. A classifier scores every margin-sorted
   candidate partition (features: margin gap at the boundary, per-side margin /
   reach / size stats, hub separation). Supervised labels are derived from the
   train routes (which partition is consistent with the observed route); at
   test time the model alone picks the partition (99.2% top-1 in grouped CV).
   Out-of-fold on train rows, full-train model for test rows.

1. Orientation model — P(shown point order == route direction) per fragment,
   from direction-sensitive geometry (endpoint-to-hub/anchor distances, where
   the longest segment sits in the sequence, prefix/suffix path length, local
   point density near each end, signed turning). Trained first; out-of-fold
   (4-fold) predictions on train rows provide "predicted-orientation" features
   (e.g. distal-endpoint identity) to the downstream models without leakage;
   a full-train model predicts test fragments. OOF fragment-level accuracy ~99.9%.

2. Pairwise comparator — P(fragment a precedes fragment b) for same-component
   pairs, trained on both pair directions (symmetrized) and predicted as the
   average of P(a before b) and 1 - P(b before a); ensembled over two LightGBM
   seeds and one XGBoost model. Its out-of-fold predictions define a
   per-component max-likelihood total order (exact Held-Karp DP over the pair
   log-probabilities); the rank of each fragment in that order becomes a
   feature for the start/policy models, and the pair log-probabilities are also
   added to the beam score (weight alpha=0.6, tuned on the holdout).

3. Start model — P(fragment is the route start), from fragment geometry,
   hub-relative features, within-component ranks, cluster-rank features and
   the DP-order rank.

4. Successor policy — P(candidate is next | current fragment, decode state),
   trained by imitation of the true routes (teacher forcing over every step,
   true successor = positive, all other remaining fragments = negatives);
   ensembled over two LightGBM seeds and one XGBoost model.
   State features include per-component remaining counts, which component was
   least-recently visited, relations to the last and second-to-last fragment
   taken from the candidate's component, and remaining-peer rank features.

Decoding: beam search (width 32). Beam score = log P(start) + sum of
per-step normalized log P(successor) + alpha * pairwise-order log-probability
terms. outer_inner rows use the partition chosen by the stage-0 split model
(measured: the true outer/inner partition is margin-monotone in ~99.5% of
train rows). Orientations are predicted independently per fragment (argmax).

FEATURE ENGINEERING
-------------------
Per fragment: bbox geometry (center, w/h, diagonal, margin to canvas edge,
reach from canvas center), path length, chord, point count, signed/absolute
turning, endpoint coordinates and deltas, longest-segment length and relative
position, own-anchor (densest own point) distances, endpoint-local densities.
Per component (per-row unsupervised KMeans clustering, k = component_count
given in the data; feature space chosen by layout family — bbox-margin for
outer_inner, x-weighted centroids for center_sides, centroids otherwise):
hub = densest pooled point (refined by predicted-orientation squiggle tails),
component size, centroid, cluster ranks along the layout axes.
Hub-relative per fragment: first/last/distal/proximal endpoint distances,
angles, predicted-orientation variants of those, within-component ranks by
several keys, pairwise min point-set distances and anchor distances.
Multi-metric ordering features: the predicted distal head's distance from the
hub under several metrics — isotropic, fixed anisotropic weights (0.15/0.3/2 on
dx^2), L1, component-extent-normalized ((dx/width)^2 + (dy/height)^2) and
point-cloud-covariance-normalized — plus within-component ranks under these.
Diagnostic motivation: no single metric sorts every component, but for ~73% of
hard components some anisotropic weight makes the true order fully ascending,
so the models are given the family and learn which metric applies per layout.

VALIDATION STRATEGY
-------------------
Random 1400-row holdout from the 7000 train rows (fixed seed, ~20%), scored
with a local re-implementation of the exact row metric and final aggregation
(0.08 PairwisePrecedence + 0.42 AdjacentLinkF1 + 0.08 LCS + 0.07
OrientationAccuracy + 0.35 ExactProgram; final = 0.70 mean + 0.20 tail20% +
0.10 long-rows). Models for the holdout evaluation were trained only on the
other 5600 rows (orientation/pair OOF layers nested inside that split).

Holdout results (v2 pipeline as submitted, trained on 5600 rows):
  final = 0.648   (mean 0.750, tail20 0.268, long 0.696)
  exact sequence = 61.7% of rows, all-orientations-correct = 99.1% of rows
The submitted script trains on all 7000 rows, which should score slightly
above the holdout estimate (the previous submission scored 0.6002 public LB
from a 0.5905 holdout). Progression: GBM policy baseline 0.432 -> + state/angle
features 0.468 -> + predicted-orientation features, pairwise beam term 0.580 ->
+ DP-order features 0.588 -> + capacity 0.590 [previous submission, LB 0.6002]
-> + multi-metric ordering features 0.600 -> + learned outer_inner split model
0.617 -> + model ensembles, beam 32 0.622 -> + symmetrized pair comparator
0.648.

LEAKAGE STATEMENT
-----------------
Every model, statistic, scaler and vocabulary is fit on train.csv only.
Test rows are used exclusively for per-row transform (geometry parsing,
per-row KMeans clustering of that single row's fragments using the row's own
given component_count, feature computation) and predict (model inference,
beam decoding). No statistic is computed across test rows; there is no
train+test concatenation anywhere; no pseudo-labeling, no test-time
calibration, no feature/hyperparameter selection on test. The per-row
clustering of a test row uses only that row's own fragment geometry, which is
single-sample inference. Random seeds are fixed (LightGBM random_state=0,
deterministic KMeans seeding); the script reads only from the provided
public_dir and writes only submission_out.

WHAT WORKED / WHAT DID NOT
--------------------------
Worked:
- Letting the policy model learn the component-interleaving (braid) convention
  from visit-history state features instead of hardcoding any rule.
- Feeding predicted-orientation ("which end is the distal head") into the
  ordering models — biggest single jump (+0.11 final): it upgrades the key
  ordering feature (distal distance from hub) from ~88% correct to ~99.9%.
- Same-component pairwise comparator as an extra beam-score term, and its
  Held-Karp max-likelihood order ranks as features.
- Learned outer_inner partition model (99.2% top-1) replacing likelihood-based
  multi-partition decode - fixed nearly the whole outer_inner failure bucket.
- Symmetrized pair training + bidirectional prediction averaging: pair accuracy
  92.5% -> 93.3%, holdout final +0.026 - the single biggest late-stage win.
- Multi-metric (anisotropic / extent-normalized) distance features: pair
  accuracy 88% -> 92%.
- Independent per-fragment orientation classifier (99.9% fragment-level).

Did not work / dead ends:
- Any single geometric sort key for within-component order (distance from hub,
  angle, spoke length, path length, all-points max distance, L1/Chebyshev
  variants, hub-position grid search): the underlying order is only ~56-76%
  concordant with the best key; this remains the main residual error source.
- Endpoint-continuity chaining (consecutive fragments do not connect spatially).
- Cross-component stroke correspondence (components are independent).
- Clustering on per-fragment anchor points and EM hub-reassignment refinement
  (both worse than layout-aware KMeans).
- Refined squiggle-tail hubs helped feature quality marginally but did not
  move the holdout score.
