Vector Stroke Gap Reconstruction
================================

Overview
--------

The solution reconstructs each missing stroke as a length-constrained sequence of
32x32 grid cells. It uses only the released train.csv labels and the public test
fields. It does not use IDs as model features, external drawings, hidden metadata,
a network connection, or a GPU.

Run from this directory:

    python solution.py

The command reads dataset/public/train.csv and dataset/public/test.csv and writes:

    working/submission.csv

Paths are resolved relative to solution.py, so this also works from the repository
root:

    python "Vector Stroke Gap Reconstruction/solution.py"

For local evaluation on all labeled rows:

    python solution.py --validate

Algorithm / model architecture
------------------------------

This is a geometric template model with CPU LightGBM ranking and classification.
Its stages are template construction, symmetry-normalized retrieval, candidate
path ranking, candidate-cell classification, and connected sequence decoding.

1. Complete the released training strokes

For each train row, the gapped stroke is restored with:

    prefix + answer_json["hidden_cells"] + suffix

All other `points` strokes are already complete. Every resulting stroke is legal
training data and becomes a source of local path templates.

2. Extract candidate gap windows

For each query length n (8 through 20 in the released data), every complete train
stroke is scanned for windows with:

* four visible cells ending at the left gap endpoint,
* n candidate hidden cells,
* the right gap endpoint, and
* four additional visible suffix cells (five suffix cells including the endpoint).

A candidate therefore has the same observable 4+5 context layout as a query. Its
feature vector contains 18 signed integers: the last four prefix coordinates
relative to the left endpoint and the first five suffix coordinates relative to
the right endpoint. Its target is the ordered n-cell path relative to the left
endpoint.

3. Canonicalize rotations and reflections

The eight dihedral transforms of the square (four rotations and their reflections)
are considered for the vector between the visible endpoints. The first transform
that makes the transformed displacement lexicographically maximal is selected.
The displacement, context, and candidate path are all mapped through that same
integer transform.

The lookup key is:

    (missing_count, canonical_delta_row, canonical_delta_column)

This gives exact endpoint and length compatibility while sharing examples across
rotated and mirrored strokes. Only canonical keys needed by the active query set
are materialized.

4. Retrieve compatible paths

Within the compatible key, squared Euclidean distance is computed over the 18
context features. The 60 nearest candidates are selected deterministically.
Candidate paths are transformed back to the query orientation, translated to the
query endpoint, and clipped to the 32x32 grid.

5. Rank paths, classify cells, and decode a connected sequence

The train set is reconstructed in leave-one-row-out mode: when generating examples
for row i, every template sourced from row i is excluded. Each of the 199,588
retrieved paths receives its actual published row score, quantized to 0..100
relevance, as a LightGBM LambdaRank target.

The ranker uses 34 public geometric features: context distance and rank, neighbor
consensus, distance from weighted-mean and Hermite paths, turns, duplicates,
invalid raster steps, endpoint-direction continuity, gap length, displacement,
and local prefix/suffix velocity. Five deterministic row folds produce honest
out-of-fold ranks for training the next stage; the five models are averaged for
test inference.

At each missing time step, every distinct cell proposed by the 60 candidates
becomes a classification option. Two CPU LightGBM classifiers learn from 368,689
options using the same 46 public geometric features: five context-distance vote
masses, path-ranker vote mass, candidate count/rank/distance statistics, absolute
and endpoint-relative coordinates, Hermite/mean-path residuals, gap position,
stroke metadata, and local geometry. IDs and prompt text are never features.

The first classifier targets the exact cell at the current sequence position. The
second targets whether a proposed cell occurs anywhere in the true hidden set,
which supplies a metric-aware signal for both set F1 and small alignment shifts.
Exact context matches (nearest squared context distance zero) use the positional
classifier alone. For inexact matches, the decoder uses a geometric probability
blend with 0.3 positional and 0.7 set-membership weight. This distance gate and
blend weight were selected with five-fold row-level cross-validation.

The resulting log-probabilities are decoded jointly with dynamic programming.
Transitions are penalized unless consecutive cells are 8-neighbors; the same
constraint is applied from the visible prefix endpoint and into the visible suffix
endpoint. This prevents independent high-probability cells from creating jumps,
while retaining a finite fallback when no fully connected option path exists.

6. Fallback

Every released test key had training candidates. For robustness, a query with no
eligible template uses cubic Hermite interpolation. Endpoint derivatives are chord
estimates from the nearest four visible cells on each side. The curve is evaluated
at exactly n positions, rounded, and clipped. The fallback therefore always obeys
the required length and cell range.

Feature engineering
-------------------

The final model uses:

* exact missing length;
* canonical endpoint displacement;
* four prefix cells relative to the left endpoint;
* five suffix cells relative to the right endpoint;
* all eight rotations/reflections through canonicalization;
* context distance, neighbor rank, and per-step consensus support;
* candidate deviation from weighted-mean and Hermite paths;
* raster continuity, turn, duplicate, and endpoint-direction features;
* cell-level distance/rank vote masses and coordinate residuals;
* public stroke count, point count, stroke index, and gap position;
* ordered candidate coordinates at each missing time step;
* 8-neighbor transition compatibility with both visible endpoints; and
* fixed-grid boundary clipping after transferring a candidate.

Absolute row IDs and prompt text are excluded. Absolute query location is used only
for geometric features and to translate relative paths back onto the grid. Other
complete strokes contribute legal training paths but are not semantic labels.

Implementation and resource use
-------------------------------

The template index is built in two passes. The first pass counts candidates for the
required keys. The second allocates compact NumPy arrays and fills them directly:

* int8 context features;
* int8 relative paths; and
* int32 source-row indices.

This avoids retaining millions of Python objects. Training uses 199,588
candidate-path examples and 368,689 candidate-cell options. The released test
index contains 1,553,407 candidates and occupies 72.4 MB in packed arrays.
Observed deterministic end-to-end runtime for the final dual-classifier version
was about 121 seconds on the supplied CPU: 81 seconds for path-ranker and cell
classifier fitting, 14 seconds for the test index, and the remainder for candidate
generation, option scoring, and connected decoding. No GPU code is present.

Validation strategy and result
------------------------------

`python solution.py --validate` first performs leave-one-row-out candidate
construction for all 3,328 train rows. When scoring row i, every path extracted
from any stroke in row i is excluded. Rows are assigned to five deterministic
folds. The path-ranking score and both cell probabilities used for a validation
row come from models trained only on the other four folds.

The scoring implementation follows the published metric exactly: set F1, ordered
LCS, exponential length score, and first/last Manhattan endpoint score.

Observed five-fold distance-gated metric-aware decoding result:

    rows          3328
    final score   0.668632
    set F1        0.580502
    ordered LCS   0.576231
    length score  1.000000
    endpoint      0.907520

The previous positional connected decoder scored 0.666960 on the same rows.
Adding the set-membership objective and public context-distance gate contributes
0.001672 locally, with gains in both set overlap and ordered LCS. The earlier
path-quality regressor scored 0.660819.

What worked
-----------

* Straight endpoint interpolation was a useful format baseline but scored only
  about 0.383 because it misses curvature and corners.
* Cubic Hermite interpolation with short local tangents improved to about 0.569.
* Symmetry-normalized nearest-template coordinate averaging reached about 0.643.
* Unlearned weighted cell consensus reached 0.652650 leave-one-row-out.
* Whole-path quality regression raised validation to 0.660819.
* LambdaRank plus positional cell classification and connected decoding reached
  0.666960.
* Metric-aware set-membership classification with distance-gated blending reached
  0.668632, improving set overlap and sequence ordering without changing length.
* Four prefix and five suffix cells gave enough local direction information while
  preserving a large candidate pool.
* Rotation/reflection canonicalization improved reuse without floating-point
  alignment or learned augmentation.

What did not improve the final model
------------------------------------

* Longer visible contexts made nearest-neighbor matching sparse and reduced the
  validation score.
* Stroke length, gap position, stroke index, and whole-sketch point-count metadata
  gave only tiny, unstable changes and were omitted.
* First- and second-order direction n-gram costs made path search slower without a
  consistent gain.
* Gradient-boosted direct coordinate residual regression underperformed the
  template consensus; tree models worked only when ranking real paths and
  classifying cells proposed by those paths.
* A small CPU U-Net could predict a plausible hidden-cell heatmap, but reranking or
  blending it with templates produced no reliable score gain and added training
  time and a heavy dependency.
* Whole-sketch occupancy and distance-transform features, off-template neighboring
  cell expansion, CatBoost ensembling, and a learned transition classifier were
  also evaluated. Their honest out-of-fold scores did not exceed the compact
  distance-gated dual-LightGBM decoder.
* Extra uncertainty cells were not emitted. Training confirmed that every length
  hint is exact, and changing output length sacrifices the guaranteed length term.

Output integrity
----------------

Before writing, the program verifies every prediction length and every cell's
syntax and grid range. It writes a temporary CSV, flushes and fsyncs it, then
atomically replaces working/submission.csv. The verified metric-aware connected
decoder submission contained 1,792 unique IDs in test order and had SHA-256:

    c754e2977108969075db25970b54c7d794aa7c1fb923aad605f778816529d6ed
