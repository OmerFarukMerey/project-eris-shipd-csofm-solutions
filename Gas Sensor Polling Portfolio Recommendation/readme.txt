Gas Sensor Polling Portfolio Recommendation
===========================================
Entry point:  python3 solution.py <public_dir> <submission_csv>
Both paths come from sys.argv; nothing is hardcoded.  A schema-valid placeholder is
written to <submission_csv> immediately after test.csv is read and overwritten with the
real predictions at the end.

COMPUTE: CPU ONLY.  PROBLEM.md states "Local CPU signal processing and locally executed
learned models are allowed", so the script neither requests nor uses any accelerator -
there is no torch.cuda / torch.backends.mps reference and no .to(device) call anywhere in
the file.  Model sizes and the seed count were chosen so the whole run fits the wall-clock
budget on CPU alone (~1850 s on 8 threads), and an adaptive guard measures the slowest
network so far and refuses to start a run that could not finish in time.


1. SUBMISSION SCHEMA (from PROBLEM.md, verified programmatically before finishing)
----------------------------------------------------------------------------------
Exactly four columns, in this order:

    case_id, polling_mask, recovery_collision_set, clearance_order

  * case_id                 every test id exactly once, no whitespace padding
  * polling_mask            12 characters from {K, .}, exactly six K; position 1 = S01
  * recovery_collision_set  exactly four unique edges joined by '|', each "Sxx~Syy" with
                            the lower alias first, tokens in ascending lexical order,
                            <= 160 characters
  * clearance_order         '>'-joined permutation of S01..S12, earliest clearance first,
                            <= 100 characters

The script re-reads the file it wrote and asserts row count, column names/order, unique
ids, mask length + K count, four unique sorted canonical edges, and permutation validity.
Every per-row decode is wrapped in try/except and falls back to a schema-valid value, so
one bad row can never invalidate the file.


2. METRIC
---------
Score = 0.55 * PortfolioUtilityScore + 0.25 * CollisionGraphScore + 0.20 * ClearanceOrderScore

  PortfolioUtilityScore  mean over cases of normalised_utility^3, where the utility
                         0.55*base + 0.30*coverage + 0.15*(1-redundancy) is computed from
                         HIDDEN per-sensor risk r_i, a hidden 4-column descriptor d_i and a
                         hidden pairwise similarity s_ij, normalised against the min and max
                         over all 924 six-sensor portfolios of that case.
  CollisionGraphScore    0.85 * edge-F1 + 0.15 * exact set match
  ClearanceOrderScore    0.90 * pairwise agreement over the 66 alias pairs
                         + 0.10 * exact order match

Domain: this is guidebook 5.6 (Bio/Chem/Other) - a signal/time-series structured-prediction
problem.  No model-type restriction is declared in PROBLEM.md, so locally executed models
trained in-script on CPU are used throughout.  No pretrained weights of any kind are
loaded and nothing is downloaded; the only inputs are train.csv, test.csv and the packets.

Challenge-specific rules, taken one line at a time from the "What Not To Use" section:

  * "case IDs, packet filenames, byte sizes, hashes, row order, source ordering" - none of
    these reach a model.  The only input to every feature is the numeric packet array;
    case_id appears solely as an output column and sensor_packet_path solely as the path
    passed to np.load.  The sensor's own index inside the packet (S01..S12) is also never
    used as a feature, since PROBLEM.md says the aliases are independently assigned.
  * "external copies of the original recordings / source features" - nothing external is
    read, fetched or matched against.
  * "exact or approximate lookup tables from packet fingerprints to targets" - there is no
    lookup, nearest-neighbour retrieval or memorisation step of any kind.
  * "hidden answer structure, grader behavior, malformed CSV handling, duplicate rows,
    submission ordering" - none of these are touched.  The blend-weight search uses an
    ordinary supervised validation objective computed from held-out TRAINING labels
    (section 5); it assumes nothing about how the grader is implemented, and the
    submission is written in the test.csv row order with one row per id.
  * "Do not call hosted or closed-model APIs during inference.  Local CPU signal
    processing and locally executed learned models are allowed." - no network call is made
    at any point, and the run is CPU-only as described at the top of this file.


3. WHAT THE DATA LOOKS LIKE, AND THE MODELLING IDEA
---------------------------------------------------
Each packet is 12 x 3 x 64: baseline (t=0..23) then exposure (t=24..63) of a
baseline-normalised log-resistance, its first difference, and a phase marker.  Recovery is
not shown.  A random per-sensor gain is applied after normalisation, so amplitude is a
weak clue while the *shape* of the approach to steady state is gain-free.

Three statistics of the TRAINING LABELS motivated the architecture.  They are exploratory
findings that told me what to model; none of them appears anywhere in the code, and the
models are free to contradict all three (the selection head is trained on the portfolio
label independently of the clearance head, and the pair head on the edge labels):

  * the labelled portfolio is almost exactly "the six latest-clearing sensors" - 72.6% of
    training rows match the top-six of the labelled clearance order exactly, and the
    disagreements are overwhelmingly a single swap across the 6/7 boundary;
  * the four collision edges are almost always adjacent in the clearance order (79% at
    rank distance 1, 18% at distance 2), i.e. the hidden s_ij behaves like a decreasing
    function of the gap between two sensors' recovery times;
  * 63% of training edge sets are exactly the four smallest gaps of a single 1-D latent.

So all three outputs are driven by one latent per-sensor recovery variable plus a latent
pairwise similarity.  The pipeline learns both and decodes them.


4. FEATURES  (117 per sensor -> 585 after within-case context)
--------------------------------------------------------------
All features are computed from ONE case's own packet.  Nothing is pooled across rows, so
the identical code path runs on train and on test.

Per sensor:
  * baseline block: mean, log std, slope, quadratic fit, log per-sample noise, range,
    head/tail means, drift (tail minus head), drift normalised by noise and by the
    exposure amplitude.  The baseline window is the tail of the *previous* recovery, so
    its decay is a direct observation of the quantity being predicted: an AR(1) fit and a
    kinetic regression on the mean-removed baseline are included.
  * amplitude block: signed steady state, log amplitude (floored at max(3*baseline std,
    2e-3) so dead channels do not blow up), sign, dead flag, log SNR, peak ratio.
  * gain-free kinetics of the normalised response ne(t) = exposure / steady state:
    17 sampled values, six threshold-crossing times, three integral time constants,
    a kinetic regression (dne ~ k*(1-ne), k = 1-exp(-1/tau)), late/mid slopes, roughness,
    settle residual, cumulative-change fractions.
  * AR(1) and AR(2) linear-prediction fits, on both the residual-to-asymptote and the
    derivative channel: coefficients, both pole magnitudes, both implied time constants,
    an oscillatory flag and the fit residual.  This is a robust two-exponential estimator.
  * derivative moments: centroid, spread, max and argmax.
  * time-warp features: the case's own median normalised response is used as a template,
    resampled on a 29-point log-alpha grid, and each sensor's best log time-scale is found
    by parabolic refinement of the argmin - with and without a free amplitude, on both the
    response and the derivative.  This measures "how much slower than its neighbours is
    this sensor" without any amplitude dependence, and was one of the two largest feature
    wins.

Within-case context (this is what makes the 12 sensors comparable, since the analyte and
the recording differ per case): each feature is also emitted as a within-case z-score, a
within-case rank, and the case's own mean and standard deviation of that feature.

Pair features (66 per case): absolute differences / min / max / difference-rank of the
z-scored sensor features, the four per-sensor model scores turned into |dz|, |drank|, mean
z, mean rank, min rank, cosine similarity and log L2 distance of the normalised response,
derivative and log-residual traces, the residual of a free-scale projection of one trace
onto the other, and the within-case ranks of all of those scalars.

  * all-pairs time warp.  Every sensor's trace is resampled on a 25-point log-alpha grid
    and matched against every OTHER sensor's trace (both directions, with and without a
    free amplitude), giving the relative time-scale and residual for each ordered pair.
    Asking "do these two have the same kinetics" directly is sharper than comparing both
    to a shared template, and it is exactly what the hidden pairwise recovery similarity
    should track.  Worth +0.004 edge F1 on its own.  Computed as a chunked einsum over
    a precomputed warp bank, ~2 s for all 3500 cases.


5. MODELS
---------
Stage 1a - LightGBM on the per-sensor context features
    * regressor on the clearance rank        -> per-sensor clearance score
    * classifier on the polling label        -> per-sensor selection score
    5-fold; test predictions are the average of the five fold models.

Stage 1b - a permutation-equivariant set transformer, trained from scratch in-script on CPU
    Tokens are the twelve aliases, with NO positional encoding, which matches the fact
    that alias assignment is arbitrary within a case.  Two-layer pre-norm encoder,
    d=160, 4 heads.  Three heads:
      t  per-sensor clearance logit, trained with a pairwise logistic loss over all 66
         alias pairs - this optimises the ClearanceOrderScore pairwise-agreement term
         directly rather than through a rank regression proxy;
      m  per-sensor selection logit (BCE against the labelled portfolio);
      p  a pair head over [h_i+h_j, |h_i-h_j|, |t_i-t_j|, t_i+t_j] trained with a
         four-hot softmax cross-entropy over the 66 pairs.
    Four seeds x 5 folds = 20 networks, averaged.  Seed averaging is worth about +0.010
    pairwise agreement over a single seed, so it is the cheapest real gain in the whole
    pipeline.  A second configuration with a dilated 1-D CNN stem over the raw normalised
    traces was built and benchmarked; it is NOT in the final script because on CPU it
    costs 443 s per fold-seed (6640 s for the full sweep) against 44 s for the tabular
    network, for +0.005 on the portfolio validation objective.  Dropping it and spending a
    quarter of that time on a fourth seed instead came out level (section 6).

Stage 2 - two boosted pair models on the pair features above (which include the stage-1
    per-sensor scores), 5-fold, predicting whether a pair is one of the four collision
    edges:
      * a binary classifier, and
      * a lambdarank model over groups of 66, because exactly four of the 66 pairs are
        edges in every case, so the decision is competitive within the case rather than
        66 independent yes/no calls.
    The two are blended with the transformer's pair head at searched weights.  Alone the
    listwise model is the weaker of the two (row 0.174 vs 0.181), but it disagrees in
    useful places and the blend beats either.

Decoding
    clearance_order  aliases sorted by ascending blended clearance score
    polling_mask     the six highest blended selection scores
    collision set    the four highest blended pair scores
    Blends are z-scored per row and combined with weights found by an in-script search.

Blend-weight search (in-script, on out-of-fold TRAIN predictions, against the real metric)
    * clearance weight            maximises 0.90*pairwise agreement + 0.10*exact match
    * two collision weights       maximise 0.85*edge F1 + 0.15*exact set match
    * two portfolio weights       maximise a label-calibrated surrogate for
                                  PortfolioUtilityScore (see below)
    An 11-point grid per weight; nothing is pasted in from an offline run.

The portfolio validation objective.  PortfolioUtilityScore depends on r_i, d_i and s_ij,
which are not published, so it cannot be computed locally at all - and a plain exact-match
rate is too coarse to pick blend weights with, because it treats "swapped two nearly-tied
sensors" and "dropped the single most important sensor" as equally wrong.  The search
therefore uses an ordinary supervised validation measure built from the held-out fold's
own TRAINING labels: two per-sensor value vectors, each normalised over all 924 portfolios
and cubed, each giving the labelled portfolio a value of exactly 1.0:
    (a) clearance rank + 12*label, so a swap at the selection boundary is cheap;
    (b) the label indicator alone, so every non-labelled sensor is equally bad.
(a) is the optimistic end, (b) the pessimistic end; the search maximises their mean.  It
is a validation instrument only - it is never a model input, never sees test data, and
assumes nothing about the grader's implementation.  It is reported below as "portfolio
surrogate" and is deliberately conservative: the actual leaderboard score came in well
above what it predicted.


6. VALIDATION
-------------
5-fold KFold over training cases (fixed seed).  Out-of-fold numbers from the final script:

    ClearanceOrderScore   0.6817      (pairwise agreement 0.757, exact match ~0)
    CollisionGraphScore   0.1877      (edge F1 0.220, exact set match 0.002)
    Portfolio surrogate   0.4798      (exact mask match 0.0853, mean overlap 4.475 / 6)
    -------------------------------------------------------------------------------
    Weighted total        0.4472      with the surrogate standing in for the 0.55 term

These figures reproduce exactly run to run (CPU only, fixed seeds).  The predecessor of
this build - same pipeline without the all-pairs warp features and the listwise pair model
- scored 0.6819 / 0.1805 / 0.4809 -> 0.4460, and it was submitted and scored 0.5558 on the
real metric.  So the improvement here is confined to the collision term, +0.007 row score,
worth about +0.002 on the weighted total.

Backing the real components out of that 0.5558, using the CV clearance and collision
values, puts the true PortfolioUtilityScore near 0.68 - far above the 0.446 the surrogate
predicted.  The whole gap is in the 0.55-weighted portfolio term, i.e. near-miss
portfolios lose much less hidden utility than even the optimistic half of the bracket
assumed.  The surrogate is therefore useful for RANKING blend weights but should not be
read as a score estimate.

Why this build stops here.  The three outputs all bottom out on the same per-sensor
latent, and it is resolution-limited rather than model-limited.  Bucketing the 66 pairs of
every case by the predicted score gap gives a clean monotone calibration curve - 0.53
accuracy in the closest decile rising to 0.97 in the widest, and 0.98 on the most
confident 2% - and bucketing by TRUE rank distance gives 0.59 for adjacent sensors rising
to 0.97 for the two extremes.  There is no plateau in the confident buckets, which is what
label noise or a wrong model class would look like; it is the signature of a continuous
latent estimated with noise comparable to the spacing between adjacent ranks.  Closing the
remaining gap needs a sharper latent, and six independent feature families and three model
reformulations (section 9) all failed to provide one.

Caveat on the split, stated plainly: PROBLEM.md says all cases from one acquisition day
stay in one split and each source recording contributes five cases.  The recording id is
not published.  I tested whether the groups are recoverable (nearest-neighbour structure
of the case-level mean normalised response): only 15% of cases have a mutually consistent
top-4 neighbourhood, so no reliable grouping exists to build a GroupKFold from.  Plain
KFold is therefore used and the reported CV may be mildly optimistic relative to a
day-disjoint test split.  Since every model choice was made by comparing options under the
same split, the ranking of options should be unaffected.

Reference points measured along the way: a random valid submission scores about 0.06 on
the collision component and 0.50 on clearance; picking the four collision edges uniformly
from the adjacent pairs of the TRUE clearance order gives edge F1 0.283, which is roughly
the ceiling for any method that only knows the order and not the gaps.


7. LEAKAGE STATEMENT
--------------------
Every transform and every statistic is fit on train only; test is used for transform and
predict exclusively.

  * There is no pd.concat / merge / append of train and test anywhere.  build_features()
    is called separately on the train array and on the test array.
  * All feature engineering is per-row: every statistic (z-score, within-case rank, case
    mean/std, warp template) is computed across the twelve sensors OF A SINGLE CASE, which
    is that row's own input.  No feature reads another row.
  * The neural network's input standardisation (mu, sd) is computed from the TRAIN feature
    tensor only and applied unchanged to test.
  * LightGBM and the networks are fit on training folds only; test predictions are
    averages of those fitted models.
  * zrow() standardises within one row across its 12 sensors (or 66 pairs) - again per
    row, never across the test set.
  * All blend weights are searched on out-of-fold TRAIN predictions scored against TRAIN
    labels.  No test prediction is counted, thresholded, calibrated or normalised against
    any test-set distribution.
  * No pseudo-labelling, no self-training, no test-derived vocabulary or frequency table.

Taint trace: Xte -> FA_te/Xc_te -> Zte -> {R_te, Sel_te, T_te, M_te, P_te} -> PF_te ->
Ed_te -> clear_te / mask_te / edge_te -> per-row argsort -> strings.  Every operation in
that chain is per-row; the only cross-row objects touching test are the fitted models
themselves, which were fit on train.


8. HARDCODING STATEMENT
-----------------------
No discovered generation pattern is hardcoded.

  * The structural facts in section 3 (portfolio == six latest clearers; edges == adjacent
    in the order) are NOT written into the code as rules.  They are stated here as the
    motivation for learning a shared latent, and the models are free to disagree with them
    - the selection head is trained on the portfolio label independently of the clearance
    head, and the pair head is trained on the edge labels, so nothing forces the
    "adjacent" or "top six" structure.
  * There is no dict / lookup table / if-chain mapping any input to any output.  The only
    constant tables in the file are PAIRS, SUBSETS, PIDX, SUB_IDX, SUB_ONEHOT and ALIASES,
    which are enumerations of the output grammar produced by itertools.combinations - pure
    combinatorics, independent of the data.
  * Every constant that influences an output is either learned (model parameters) or
    searched in-script on a train holdout (the four blend weights).  Nothing was tuned
    offline and pasted in.  Feature-engineering constants (clip bounds, window lengths,
    the log-alpha warp grid, the amplitude floor) are signal-processing choices applied
    identically to train and test; none of them decides an output.
  * Strip-the-ML test: with all trained models removed, clear_te / mask_te / edge_te do
    not exist and the pipeline emits only the constant placeholder row (KKKKKK......,
    S01~S02|S03~S04|S05~S06|S07~S08, S01>...>S12) for every case - i.e. it produces
    nothing usable.  The trained models produce the answers; the decode is a plain argsort
    of their scores.
  * A subset-level reranker and a redundancy-penalised set decode were both built and both
    REJECTED on out-of-fold evidence (section 9), so no hand-built candidate-selection
    layer sits on top of the models.


9. WHAT WORKED / WHAT DID NOT
-----------------------------
Worked
  * Framing all three targets as one latent recovery variable + one latent similarity.
  * Gain-free shape features.  The single strongest raw feature is the exposure time
    constant from the log-residual fit (per-case Spearman 0.37 with clearance rank); the
    integral / kinetic / AR(2)-pole estimators of the same quantity add on top.
  * Baseline drift.  The pre-exposure window is the tail of the previous recovery and its
    slope is among the highest-importance features.
  * Time-warp against the case's own median response: +0.003 pairwise agreement.
  * Within-case context (z-score, rank) and case-level mean/std: +0.004 agreement.
  * The pairwise-logistic ranking loss and the permutation-equivariant transformer.  Alone
    the network is slightly weaker than LightGBM (0.750 vs 0.749 agreement) but the blend
    reaches 0.757, and seed averaging inside the network is worth +0.010 on its own.
  * Trace cosine similarity is the top pair feature, ahead of the predicted score gap.

Did not work (all measured out-of-fold, all removed)
  * A structured utility head that predicted r_i, d_i and s_ij and took a softmax over all
    924 portfolios: exact mask match 0.05-0.067 versus 0.087 for plain top-six of a
    per-sensor score.  The 924-way cross-entropy is a much harder optimisation than the
    per-sensor marginals and lost more than the interaction terms gained.
  * Penalising within-portfolio redundancy at decode time using the pair model: monotonic
    damage at every strength tried (exact 0.077 -> 0.043 as the penalty grew).  The pair
    model outputs "is this a top-4 pair", which is a very peaked transform of s_ij and a
    poor stand-in for it.
  * A subset reranker over the top-24 candidate portfolios with set-level features
    (redundancy, coverage proxies from feature PCA, swap patterns).  The true portfolio is
    in the top-24 for 56% of cases, but the reranker never beat simply taking the top
    candidate (surrogate 0.4819 -> 0.4812 at best).
  * A second stacking pass for the pair model with graph-structure features (endpoint
    degree, triangle closure): edge F1 0.213 -> 0.208.
  * PCA of the raw traces and feeding raw trace samples to LightGBM: both slightly
    negative.
  * LightGBM lambdarank instead of rank regression: 0.744 vs 0.746 agreement.
  * Averaging three differently-parameterised LightGBM configurations: +0.002 agreement on
    its own, but exactly zero after the network blend, so it was dropped to save runtime.
  * A Bradley-Terry pairwise ranker (predict P(rank_i > rank_j) from signed feature
    differences and the signed relative warp, then Borda-aggregate to a per-sensor score):
    0.733 agreement versus 0.757 for the pointwise ensemble, and it degraded the blend
    monotonically at every weight.  The pointwise models already see within-case z-scores
    and ranks, so the explicit difference representation adds nothing.
  * Ordinal decomposition, P(rank >= k) for k in {3,6,9} blended with the regressor:
    +0.002 agreement for the boosted stage alone (0.7475 -> 0.7497) but +0.0003 after the
    network blend, for three extra model fits.  Not worth the runtime.
  * Sensor-level aggregates of the all-pairs warp matrix (row/column mean, median, IQR,
    min, max of the relative time-scales): 0.7475 -> 0.7444.  120 highly-correlated extra
    columns dilute the feature subsampling; the same information helps only at pair level.
  * "Extremeness" features aimed at the utility's coverage term (count of feature
    dimensions where a sensor is the case max/min, max |z|, distance to the case centroid
    and to the nearest other sensor): 0.7475 -> 0.7463, surrogate 0.4807 -> 0.4771.
  * A fifth network seed: the network itself improved (clearance 0.6758 -> 0.6764) but the
    end-to-end numbers moved by more than that in both directions, so it was reverted to
    the four-seed configuration that the reported figures were measured on.
  * A second network configuration with a dilated 1-D CNN stem over the raw traces: worth
    +0.005 on the portfolio objective, but 443 s per fold-seed on CPU versus 44 s for the
    tabular network.  Not affordable inside a CPU-only budget, and a fourth seed of the
    cheap network recovered the difference.

Runtime: about 1850 s locally, CPU only, 8 torch threads / 14 cores.  Breakdown: features
3 s, LightGBM clearance 162 s, LightGBM selection 192 s, 20 networks 1103 s, two pair
models 381 s, blend search 11 s.  The wall-clock guard is adaptive - it tracks the slowest
trained so far and refuses to start another one if it could not finish before 3200 s,
while always training at least one network per fold, so the run cannot overrun the
platform ceiling and cannot end up shipping only the placeholder.
