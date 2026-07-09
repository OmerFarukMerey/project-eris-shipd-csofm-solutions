Coastal Sensor Signature Recommendation - readme
==================================================

Run: python solution.py  (from this directory)
Output: ./working/submission.csv (630 rows, columns query_id,rec_1..rec_5)
Runtime: roughly 9.25 minutes (5-fold CV, each fold nesting its own LightGBM
+ Ridge regressor out-of-fold steps, plus a 3-seed final ensemble fit).


APPROACH
--------
Each query hides a real 6-hour sensor segment between 12 hours of "before"
context (tm12..tm01) and 12 hours of "after" context (tp06..tp17). The
candidate catalog (650 rows) gives, for each candidate hidden segment, only
aggregate summary statistics (mean/delta/range of wave height, mean period,
mean/delta wind speed, mean/delta pressure, mean wave direction as sin/cos)
plus coarse profile bands.

The solution went through six rounds, each validated against the real
leaderboard, not just local CV (round 6 not yet leaderboard-scored as of
writing):
  round 1 (aggregate-stat matching + LightGBM ranker):           CV 0.319, leaderboard 0.320.
  round 2 (+ k-NN neighbor-vote features, tuned LightGBM):        CV 0.373, leaderboard 0.384.
  round 3 (+ supervised pseudo-hidden-summary regressor, +
           frequency penalty):                                   CV 0.453, leaderboard 0.4605.
  round 4 (re-tuned frequency penalty; ~10 other ideas tried
           and rejected with evidence -- see below):              CV 0.453, leaderboard 0.4606.
  round 5 (regressor sees the query's own bands/season, and
           more OOF inner folds):                                CV 0.488, leaderboard 0.520.
  round 6 (+ a second, differently-biased Ridge regressor,
           fed as separate features alongside the LightGBM one):  CV 0.496.

Round 4 was a deliberately thorough attempt to push past 0.46: nearly every
standard lever for this kind of pairwise ranking problem was tried (balanced/
frequency-aware k-NN features three different ways, inverse-frequency
training weights, rank-normalized blends with 4 different alternative
scorers, two alternative ranking objectives, a stricter lambdarank
truncation setting). All but one were tested and rejected with CV evidence,
and the round-4 leaderboard score (0.4606) confirmed the CV prediction that
none of it would move the needle much.

Round 5 found the actual next lever by going back to the regressor rather
than trying to recombine what already existed: it was only ever given the
216 raw context columns, never the query's own season/hour/profile bands,
even though those same bands are literally present in the candidate catalog
it's trying to match against. Adding them was worth +0.033 CV on its own,
by far the largest single change since the regressor itself was introduced
in round 3, and the round-5 leaderboard score (0.520) beat CV's own
prediction (0.488) for the first time -- a sign the model was still
comfortably below any real ceiling.

Round 6 started from a recall@K diagnostic (recall@50 = 0.981, see below)
showing the model was overwhelmingly losing points to ORDERING errors, not
retrieval failures -- the true relevant patterns are almost always already
near the top of the ranking, just not quite in the top 5. That reframing
pointed at levers that sharpen ordering specifically. Most of them (a
candidate-candidate co-occurrence reranker, chained/stacked regression
targets, a third k-NN block built on the regressor's own outputs) were
tested and rejected with CV evidence despite being independently
recommended as top picks by outside review. The one that worked was adding
a second regressor of a genuinely different kind (Ridge, not another
LightGBM) as a separate feature block: CV 0.488 -> 0.496. See "What worked
and what did not (round 6)" for the full account.

Core idea, updated after round 2's plateau:
  1. Estimate what the query's hidden-segment summary statistics probably
     looked like, two ways:
     (a) hand-crafted heuristics from the boundary context (naive average,
         one-step linear extrapolation) -- physics-motivated but uncalibrated.
     (b) a supervised regressor trained to predict those same statistics
         directly from the label information available in the training set
         (round 3's big lever -- see below).
     Both are kept as separate features; the regressor dominates.
  2. Add a k-NN "neighbor vote" signal: 636 of the 650 candidates are reused
     as an answer ~14 times on average across the 1750 training queries, so
     "which candidates were correct for similar-looking training queries"
     carries information beyond aggregate-stat matching alone.
  3. Turn all of this into a (query, candidate) pairwise ranking problem,
     learned with a LightGBM LambdaRank model, plus a light post-hoc
     frequency penalty aligned with the frequency-balanced eval metric.

A pure hand-weighted nearest-neighbor baseline was built first per the
"strong baseline first" instruction (CV MAP@5 ~0.054 throughout), and beaten
decisively by the learned ranker in every round.


MODEL ARCHITECTURE / ALGORITHM
-------------------------------
Three model families, feeding one final ranker:

1. Supervised pseudo-hidden-summary regressors (round 3's new piece;
   inputs extended in round 5).
   For each of the 11 candidate summary fields (cand_wvht_mean, _delta,
   _range, cand_dpd_mean, cand_apd_mean, cand_wspd_mean/_delta,
   cand_pres_mean/_delta, cand_mwd_sin/cos_mean), a separate small
   LGBMRegressor (n_estimators=200, num_leaves=15) is trained to predict a
   "pseudo target": the average of that field across a training query's 5
   TRUE relevant candidates. This directly learns "given this context, what
   did the hidden segment's aggregate stats probably look like" from real
   label information, instead of relying on a hand-built physics formula
   that has never seen a label. A held-out feasibility check confirmed the
   payoff before committing to it: correlation with the true pseudo-target
   was 0.95 (regressor) vs 0.75 (naive/v2 heuristic) for wave-height mean,
   and 0.92 vs 0.84 for wind-speed mean, on a random 80/20 holdout.
   Input features: the full 216-column raw before/after context, PLUS (as
   of round 5) the query's own season_sin/cos, hour_band, swell_band,
   wind_band, and water_temp_band. The regressor originally only saw the
   raw context; adding these 6 extra columns was round 5's single biggest
   lever (CV 0.452 -> 0.485) since they tell the regressor which physical
   regime the context sits in, not just its shape -- exactly the same bands
   present in the candidate catalog it needs to match against. Adding the
   hand-crafted summary/slope features as further regressor inputs on top
   was also tried and found NOT to help (see round 5 notes).
   Leakage control: every regressor prediction used as a feature for a
   TRAINING row is an out-of-fold prediction (6-way inner KFold within the
   ranker's own training fold; tuned up from 4 in round 5, a small further
   gain) -- no row's estimate is ever produced by a model that saw that
   row's own label. For validation/test rows, the regressor is refit on
   100% of the corresponding training pool with no such restriction needed,
   since validation/test rows never overlap it.

2. A second regressor of a genuinely different kind (round 6): the same 11
   pseudo-targets, the same REG_INPUT_COLS input space (216 raw context +
   6 bands, standardized), but fit with Ridge regression (alpha=3.0,
   tuned over {1, 3, 10}) instead of LightGBM. The rationale: LGBM trees
   split on axis-aligned thresholds and can't represent a smooth trend
   extrapolated across the hidden gap, while a linear model does that
   natively -- the two are biased in different, complementary ways. Its
   predictions are diffed against the candidate fields into their OWN
   separate pair-feature block (ridge_diff_*/ridge_absdiff_*), fed to the
   ranker ALONGSIDE the LightGBM regressor's block, not blended with it.
   This is the key structural difference from every rejected blending
   attempt in rounds 3-4: there, two already-final ranking SCORES were
   combined with a fixed external weight (which always lost information).
   Here, two regressors' raw predictions are both handed to the SAME
   ranker as input features, and gradient boosting itself learns the
   (nonlinear, conditional) combination rule -- which is far more
   expressive than any fixed blend weight, and is why this succeeded
   where score-level blending never once did. Same leakage-safe nested-OOF
   procedure as the LightGBM regressor.

3. LightGBM LGBMRanker, objective="lambdarank", metric="ndcg".
   n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=30,
   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0. Trained on an
   explicit (query x candidate) pair table: every training query crossed
   with all 650 candidates (1,750 x 650 = 1,137,500 rows), labeled 1 if that
   candidate is among the query's 5 relevant_pattern_ids, else 0, grouped
   into per-query blocks of 650 so LightGBM optimizes NDCG within each block
   -- the "rank the catalog, take the top 5" structure the task requires.
   The final submission averages predictions from 3 fixed seeds (42/43/44)
   for a small, cheap variance reduction; CV itself uses a single seed for
   speed.

A light post-hoc frequency penalty (score -= 0.07 * log1p(candidate's
training frequency)) is applied to the ranker's output before ranking, to
counter the model's tendency to favor popular candidates in a metric that
weights every distinct target pattern equally regardless of frequency (see
Validation). It is NOT added as a training feature -- that was tried and
made things worse (see "did not work"). The 0.07 constant was re-tuned in
round 4 (down from round 3's 0.2) specifically in the 0.03-0.10 range
flagged as worth rechecking once the regressor features were added, since
they already remove most of the popularity bias the penalty exists to
correct -- see round 4 notes below.


FEATURE ENGINEERING
--------------------
Roughly 120 features per (query, candidate) pair, in four blocks:

1. Hand-crafted hidden-segment estimates (naive boundary average + one-step
   linear extrapolation, wave-height range/volatility proxies), diffed
   against the matching candidate field -- unchanged from round 2. Circular
   season/hour similarity (season_band is a clean 8-bin, 45-degree encoding
   of atan2(season_sin, season_cos), but the catalog has zero patterns in
   bands 6-7, so candidates' season_band is converted back to a bin-center
   angle and compared continuously rather than via exact-band equality).
   Profile bands (swell/wind/water_temp): equality + absolute difference.

2. Regressor-based estimates (see Model Architecture): diff + absolute diff
   against the matching candidate field, for all 11 summary fields, from
   BOTH the LightGBM regressor and (as of round 6) the Ridge regressor as
   two separate blocks. These features dominate the model's top-15
   importances: reg_absdiff_cand_pres_mean, reg_absdiff_cand_wspd_mean,
   and reg_absdiff_cand_wvht_mean (LightGBM regressor) are the top 3, with
   ridge_absdiff_cand_wspd_mean and ridge_absdiff_cand_wvht_mean (Ridge
   regressor) also in the top 10 -- confirming the two regressors are
   contributing distinct, non-redundant signal rather than one making the
   other superfluous.

3. k-NN neighbor-vote features (unchanged from round 2): for every query,
   find its K nearest OTHER training queries (leave-one-out for training
   rows) at 4 scales (K = 10, 30, 60, 120), in two similarity spaces (full
   216-column raw context, recency-weighted; and the compact engineered
   summary-estimate space), and compute what fraction of those neighbors
   had each candidate in their relevant set.

4. Auxiliary query-only context with no direct candidate counterpart
   (air/water temperature boundary estimates, raw trend slopes).

All pairwise features are built with vectorized numpy broadcasting (query
matrix x candidate matrix), not per-row Python loops; only the regressor's
own fitting loop (11 targets x a handful of folds) is a Python loop, and it's
the majority of the pipeline's ~7-minute runtime.


VALIDATION STRATEGY
---------------------
The official metric (frequency-balanced MAP@5) was reimplemented exactly:
  AP@5(row) = sum(precision_at_rank for hits in ranks 1..5) / 5
  score = mean over distinct target patterns of
          ( mean AP@5 over rows where that pattern is relevant )
This balancing means a row contributes to the running average of all 5 of
its relevant patterns, so rare patterns are not drowned out by common ones.
This local harness has been checked against the real leaderboard five times
now: round 1 scored 0.319 locally / 0.320 on the leaderboard, round 2 scored
0.373 locally / 0.384 on the leaderboard, round 3 scored 0.453 locally /
0.4605 on the leaderboard, round 4 scored 0.453 locally / 0.4606 on the
leaderboard (correctly predicting round 4's changes wouldn't move the
needle), and round 5 scored 0.488 locally / 0.520 on the leaderboard -- CV
has run slightly conservative in every round, by as much as 0.03 in round 5,
so round 6's local 0.496 should be read as a floor, not a ceiling, on the
next leaderboard score.

5-fold CV over the 1750 training queries (KFold, shuffled, fixed seed).
For each fold: the regressor's nested out-of-fold step runs entirely inside
that fold's training portion (no access to the held-out validation queries
at any point), the k-NN votes and ranker train only on that portion, and the
held-out fold's queries are scored against all 650 candidates.

Round-by-round CV (mean frequency-balanced MAP@5 across 5 folds):
  baseline (weighted nearest-neighbor):                      0.054
  round 1 (aggregate-stat matching + LGBM):                  0.319
  round 2 (+ k-NN votes, tuned LGBM, 3-seed ensemble):        0.373
  round 3 (+ regressor features, + frequency penalty=0.2):    0.453
  round 4 (frequency penalty re-tuned to 0.07):               0.453
  round 5 (regressor sees bands/season, n_inner 4->6):        0.488
  round 6 (+ Ridge regressor as a second feature block):      0.496  <- final

Per-fold scores were consistent across rounds 3-6 (roughly 0.44-0.52), no
sign of a lucky split or overfit.

A round 6 recall@K diagnostic (see below) also confirms the round 4 stress
test's read on fold variance from a different angle: recall@50 is 0.981,
meaning the underlying retrieval signal is stable and near-saturated:
per-fold MAP@5 swings mostly come from how the last mile of ordering shakes
out on a given fold's mix of rare/common target patterns, not from the
model finding fundamentally different candidates fold to fold.

A round 4 stress test looked at whether this per-fold variance (0.44-0.47)
reflects real model instability or just metric noise: each validation fold
covers 518-553 of the catalog's 636 distinct training patterns, but 100-136
of those (roughly a fifth to a quarter) appear in only a single validation
query for that fold, meaning that pattern's contribution to the fold's
balanced average is one noisy hit-or-miss observation rather than an
averaged estimate. This is expected sampling behavior given the metric's
per-pattern averaging on a modestly sized dataset, not evidence the model
itself is unstable.


WHAT WORKED AND WHAT DID NOT (round 3)
-----------------------------------------
Round 2 topped out at 0.373 CV / 0.384 leaderboard, short of the 0.5 target.
The following were tried, in the order tested, to close that gap further.

Worked:
- Supervised pseudo-hidden-summary regressors: by far the biggest lever
  found in this project (CV 0.373 -> 0.452, +21% relative). The insight was
  that every earlier hidden-segment estimate (naive average, linear
  extrapolation, cubic spline -- all from round 2) was a hand-built physics
  heuristic that never once looked at a training label, even though the
  labels directly reveal what a query's true hidden segment probably looked
  like (via its 5 relevant candidates' own summary stats). Training a
  regressor on that pseudo-target closed most of the estimation-error gap
  that round 2's readme had diagnosed as a likely hard ceiling -- it wasn't
  a ceiling, just a heuristic-vs-learned gap.
- A light post-hoc frequency penalty (score -= lambda * log1p(candidate
  training frequency), lambda tuned via CV grid search to ~0.2-0.25) gave a
  small, real gain BEFORE the regressor features were added (+0.004 CV,
  tested across a lambda grid from 0 to 1.2 with a clear interior optimum,
  not a monotonic edge effect). Kept in the final model as a low-risk safety
  net, though its effect is now marginal (~+0.0005 CV) since the regressor
  features already fix most of the popularity-bias problem it was
  correcting for.

Did not work / was not worth it:
- "Balanced" k-NN votes (log-lift over each candidate's base rate, meant to
  align the vote feature with the frequency-balanced metric's equal
  weighting of rare and common patterns) made things clearly worse (CV
  ~0.32-0.37 vs 0.373 for plain vote fractions), both as a full replacement
  and as an additional feature alongside the raw vote. The transform
  amplifies noise for rare candidates specifically -- exactly the candidates
  whose neighbor-vote estimate is least reliable to begin with (fewest
  neighbor observations), so "correcting" for their rarity mostly just
  amplifies estimation noise rather than signal. A gentler additive
  (vote - prior) version was also tried and still underperformed the raw
  vote.
- Adding candidate training frequency as a direct ranker TRAINING feature
  (as opposed to a post-hoc penalty) made things worse (CV 0.368 vs 0.373
  without it) -- the model learns to exploit popularity as a predictive
  signal during training (popular candidates genuinely are positive labels
  more often), which is exactly the bias the frequency-balanced metric
  penalizes. The post-hoc penalty works precisely because it's applied
  after training, where the model can't learn to route around it.
- A per-query rank-normalized blend of the LightGBM ranker with the
  nearest-neighbor baseline was tried at a range of blend weights (0.5 to
  0.98 LightGBM / rest baseline). Every weight tested underperformed pure
  LightGBM, monotonically worse as more baseline weight was added -- the
  baseline is too weak a signal at any blend ratio, confirming the same
  finding from round 2's z-score blend with a different blending method.
- Larger regressor capacity (num_leaves 15->20, n_estimators 200->300) and
  more inner OOF folds (4->5) together cost roughly 2x the runtime for a
  slightly WORSE CV score (0.451 vs 0.452) -- likely mild overfitting from
  the extra capacity on a modest-sized per-fold training set. The original,
  smaller/faster regressor config was kept.

Why round 2's diagnosed "ceiling" was not actually a ceiling:
Round 2's readme concluded, based on real diagnostic evidence (rare-pattern
recall, per-pattern-frequency AP buckets, band-match rates vs. chance), that
the residual gap to 0.5 was "bounded by information loss in the anonymized
features themselves." That evidence was real, but the conclusion was too
strong: it correctly showed the hand-crafted estimators had hit their
ceiling, not that the *information* had. The training labels themselves
encode a much more accurate implicit mapping from context to hidden-segment
stats than any hand-built formula does, and a regressor can extract it. The
lesson generalizes: when an estimation feature is built from a heuristic
and labels are available, try learning the estimator from the labels before
concluding the underlying information is exhausted.


WHAT WORKED AND WHAT DID NOT (round 4)
-----------------------------------------
Round 3 scored 0.4605 on the leaderboard. Round 4 was a deliberate, thorough
push to close the remaining gap to 0.5, testing essentially every standard
lever for pairwise learning-to-rank problems. All comparisons below reuse a
single expensive shared feature cache (pair matrix + regressor OOF features,
built once) so that different knn/model/post-processing variants could be
compared on a controlled, apples-to-apples basis rather than across
separately retrained end-to-end runs (which have their own ~0.002-0.003
run-to-run noise from LightGBM's multithreaded histogram building, even
with a fixed random_state -- see the note on the frequency-penalty result
below).

Worked (small, kept):
- Re-tuning the frequency-penalty lambda specifically in the 0.03-0.10 range
  (as opposed to round 3's 0.2, tuned before the regressor features
  existed): 0.07 is the new optimum (controlled-comparison CV 0.4563 ->
  0.4568), confirming the intuition that once the regressor features
  already remove most of the popularity bias, a lighter touch is better.
  This is a real but very small effect, likely near or below the noise
  floor of a full retrain (the two full end-to-end runs of solution.py, one
  at lambda=0.2 and one at 0.07, produced identical 4-decimal CV numbers --
  LightGBM's own run-to-run training noise from multithreading is
  comparable in size to this effect). Kept anyway since the controlled
  comparison showed a real, reproducible direction, and it cannot hurt.

Did not work (all tested with CV evidence, all rejected):
- lambdarank_truncation_level=5 (focuses the LambdaRank gradient on top-5
  pairs specifically, matching the eval depth): made things clearly WORSE
  (CV 0.441 vs 0.456) -- the model apparently benefits from learning to
  rank the full candidate pool, not just the top 5, probably because the
  full-list signal helps it place the truly-relevant-but-not-yet-top-ranked
  candidates correctly during training.
- Inverse-frequency sample weights on positive (query, candidate) training
  pairs (weight rare-candidate positives more heavily in the LambdaRank
  loss, as a training-time analog of the post-hoc penalty): both a 1/sqrt
  (freq) and a 1/freq weighting hurt CV (0.454 and 0.442 vs 0.457
  unweighted) -- rare candidates already have very few positive examples,
  and up-weighting them seems to encourage overfitting to that small, noisy
  sample rather than improving generalization. This is a different failure
  mode from the post-hoc penalty (which works specifically because it's
  applied AFTER training, not during it).
- A "regressor-distance" scorer -- a simple non-learned nearest-neighbor
  score built directly from the regressor's predicted estimates (bypassing
  the LGBM ranker entirely) -- was tested standalone (CV 0.220, far below
  the full ranker's 0.456) and blended with the ranker via per-query rank
  normalization at many weights. Every blend weight tested, even a 5%
  contribution, made things worse than the pure ranker. The same
  monotonic-decline pattern held for a 3-way blend adding the nearest-
  neighbor baseline as well.
- Two alternative ranking objectives on the identical feature set: LightGBM
  rank_xendcg (standalone CV 0.390) and XGBoost rank:map (standalone CV
  0.423) were both clearly weaker than the main LambdaRank model (0.456),
  and blending either (or both) in via rank normalization, at every weight
  tried, underperformed the pure LambdaRank model.
- "Balanced" k-NN votes were retried a third way (after two rejected
  variants in round 3): a Bayesian-shrinkage-smoothed lift (smoothing the
  raw neighbor vote toward each candidate's prior before computing the
  lift, meant to avoid the noise blowup that sank the earlier log-lift and
  additive variants) still hurt clearly (CV 0.399 vs 0.457 with plain vote
  fractions), even added alongside the raw votes rather than replacing
  them. Across three independent implementations now, frequency-adjusting
  the k-NN vote feature has never once helped in this problem -- this
  avenue is considered closed.
- CatBoost (YetiRank/QuerySoftMax) was not run. By the time it came up in
  the priority list, three different alternative model architectures/
  objectives (XGBoost rank:ndcg in round 3, LightGBM rank_xendcg, XGBoost
  rank:map) and 7+ blend configurations had all shown the same pattern:
  every alternative or blended model underperforms the single well-tuned
  LightGBM LambdaRank ranker on this feature set. Given that consistency,
  spending the runtime on a fourth alternative architecture was judged
  unlikely to break the pattern; flagged here rather than silently skipped.
- A neural top-50 reranker was not attempted, per the user's own
  deprioritization of it as a later-stage idea, and because nothing in this
  round's results suggested the bottleneck is model architecture rather
  than the ranking objective already converging on the same information the
  regressor and k-NN features provide.

Overall read on round 4: the model was already close to a local optimum for
this feature set going into the round. Every lever that tried to make the
model attend MORE to rare/unpopular candidates (balanced votes three ways,
inverse-frequency training weights, frequency as a training feature in
round 3) made things worse, while the ONE thing that reliably helps with
rare candidates is a penalty applied strictly after training. Every attempt
to combine the strong LightGBM ranker with a second, weaker signal (any of
5 different scorers/objectives tried) also made things worse -- this
ranker, on this feature set, does not appear to have complementary blind
spots that a second model can fill in via simple blending. Further gains
most likely require new information (a genuinely different feature source)
rather than new ways of combining or reweighting the information already
present.


WHAT WORKED AND WHAT DID NOT (round 5)
-----------------------------------------
Round 4 closed with the reflection that further gains "most likely require
new information... rather than new ways of combining... the information
already present." Round 5 tested that reflection directly: instead of more
blending/reweighting tricks, go back and check whether every existing
information source was actually being fully used. It wasn't -- the
regressor, the single biggest lever in the whole project, had been built
using only the 216 raw context columns, never the query's own bands.

Worked (the big one):
- Giving the regressor the query's own season_sin/cos, hour_band,
  swell_band, wind_band, and water_temp_band as additional inputs (on top
  of the 216 raw context columns it already had) was worth +0.033 CV
  (0.452 -> 0.485) on its own, consistent across all 5 folds -- the
  largest single change since the regressor itself was introduced. This
  wasn't a new idea so much as an oversight being fixed: the regressor
  predicts candidate summary fields, and the candidate catalog itself
  carries these exact same bands, so telling the regressor which regime
  (e.g. which swell_band) the query sits in directly sharpens what
  aggregate stats to expect, on top of the raw shape of the surrounding
  context it already saw.
- Increasing the regressor's inner OOF fold count from 4 to 6 (holding
  regressor capacity fixed, to isolate this from round 3's rejected
  "more folds + more capacity together" experiment) gave a further real
  gain (0.485 -> 0.488) -- more training data per inner fold produces
  better out-of-fold estimates. n_inner=8 was also tried and was WORSE
  (0.482) than n_inner=6, so this is a genuine sweet spot, not a
  monotonic "more is better" effect.

Did not work / was not worth it (tested on top of the bands/season win):
- Also feeding the regressor the hand-crafted summary/slope features
  (naive and v2 means/deltas/ranges) alongside the raw context and bands
  made things slightly WORSE (CV 0.484 vs 0.485) -- the regressor, given
  the raw context, can already reconstruct anything those hand-crafted
  features compute from it; adding them back in as separate inputs mostly
  just adds redundant, higher-variance columns for a model already fit on
  a modest amount of per-fold data.
- Regressor seed-ensembling (averaging 3 seeds per regressor fit, the same
  cheap trick that helps the final ranker) gave essentially no change
  (0.484 vs 0.485) for roughly double the runtime -- unlike the ranker,
  the regressor's own randomness doesn't appear to be a meaningful source
  of variance here. Not included in the final model.
- Using the median instead of the mean of the 5 relevant candidates' fields
  as the pseudo-target was clearly worse (0.480 vs 0.488) -- the mean
  matches how the candidates' own summary fields were most likely computed
  in the first place (a straightforward arithmetic average over each
  candidate's real 6-hour segment), so it's the more consistent target to
  fit against.

Where this leaves things: round 5's lesson mirrors round 3's -- the
highest-leverage move was making sure a component that already existed and
already worked (the regressor) had access to information that was sitting
right there in the same dataframe the whole time, rather than inventing a
new signal or a cleverer way to combine existing ones. If further gains are
wanted, the same question is worth re-asking for every other feature block
in the pipeline: does it see every legitimately available input, or only
some of them?


WHAT WORKED AND WHAT DID NOT (round 6)
-----------------------------------------
Round 5 scored 0.520 on the leaderboard, beating its own CV estimate (0.488)
for the first time in the project -- a sign there was still real room left.
Round 6 opened with a diagnostic rather than another feature idea.

Recall@K diagnostic (run first, per outside review's own suggestion that it
should be):
  recall@5:   0.565
  recall@10:  0.780
  recall@20:  0.916
  recall@50:  0.981
  recall@100: 0.994
By the time the shortlist reaches 50 candidates, 98% of the true relevant
patterns are already present. The model is very rarely failing to find the
right candidates at all -- it is failing to rank them into the top 5 when
they're already nearby. This reframed the rest of the round: levers that
specifically sharpen ORDERING among near-miss candidates were prioritized
over further retrieval-side feature work.

A related, quick diagnostic worth restating clearly since it was asked
about again this round: the band-hard-filter hypothesis (do relevant
candidates always share the query's swell/wind/water_temp bands, such that
the pool could be filtered before ranking?) was already tested in round 2
with a direct, decisive answer -- no. Only ~30% of a query's 5 relevant
candidates match all three profile bands simultaneously, 516 of 1750
training queries have ZERO of their 5 matching on all three bands at once,
and hour_band's match rate (27%) is statistically indistinguishable from
chance. The regressor's use of these bands (round 5) is real evidence they
carry SOFT signal, not evidence of a hard filter -- those are different
claims, and only the soft-signal one is supported by the data. No new work
was needed here; restated for the record since it came up again.

Worked:
- A second, differently-biased regressor (Ridge, alpha=3.0) on the same
  inputs and pseudo-targets as the LightGBM regressor, fed to the ranker as
  its own separate pair-feature block rather than blended with the LightGBM
  regressor's score: CV 0.488 -> 0.496 (alpha grid: 10.0 -> 0.492, 3.0 ->
  0.496, 1.0 -> 0.495; 3.0 is the sweet spot). This is the one place in the
  whole project where "add a second model" has helped -- see Model
  Architecture for why this differs structurally from every rejected
  blending attempt in rounds 3-4 (feature-level combination learned by the
  ranker itself, not a fixed external blend weight on final scores).

Did not work / was not worth it (all recommended by outside review as
likely wins, all tested with CV evidence, all rejected):
- Candidate-candidate co-occurrence reranking: build C = R^T R from
  training labels (how often each pair of candidates co-occurs in a
  relevant set of 5), row-normalize, and boost each candidate's score by a
  weighted average of its "clique partners'" scores for that query
  (s_final = s + alpha * s @ C^T). This was independently the #1
  recommendation from two different outside reviews of the project. It
  hurt CV MONOTONICALLY as alpha increased from 0 -- even alpha=0.1 already
  cost -0.001, and by alpha=3.0 CV had collapsed to 0.339. Most likely
  explanation: candidates that co-occur across MANY training queries tend
  to just be generically common/compatible patterns, so propagating score
  through this graph mostly re-injects a popularity bias back into a model
  whose whole recent history (rounds 3-4) has been about carefully removing
  exactly that bias. A graph built from per-QUERY local structure (e.g.
  only counting co-occurrences among a query's k-NN neighbors) might behave
  differently, but the simple global version tested here is a clear no.
- Target-stacking / 2-round chained regression: use round-1 OOF predictions
  of all 11 regressor targets as extra input features for a second-round
  refit of each target (meant to let the model exploit that wave height,
  wind, and pressure are physically coupled). Implemented with a fully
  leakage-safe two-stage nested-OOF scheme. Hurt CV (0.478 vs 0.488) --
  the raw 216-column context already implicitly encodes whatever
  cross-target correlation exists, so the extra stacking layer mostly adds
  the first round's own OOF estimation noise as new input variance rather
  than new signal.
- A third k-NN neighbor-vote block built on the regressor's own predicted
  11-dimensional target space (in addition to the existing raw-context and
  hand-crafted-summary similarity spaces): hurt CV (0.484 vs 0.488) --
  this space is highly correlated with the regressor's own direct
  pair-diff features already in the model, so it mostly added redundant,
  noisier information rather than a genuinely new similarity signal.
- Re-tuning the frequency-penalty lambda again on the round-5 model: the
  optimum barely moved (0.07 -> 0.10, CV 0.4877 -> 0.4881, a ~0.0004
  difference within noise). This lever has now been re-checked twice and
  is essentially exhausted -- each successive round of real feature
  improvements shrinks its effect further, as expected since it exists to
  correct a bias the other features increasingly remove on their own.
- Not retried this round, but re-confirmed as still correctly rejected from
  round 4 with matching outside-review suggestions: a direct
  regressor-distance score blended with the ranker (rejected in round 4:
  standalone CV 0.220, every blend weight worse than pure ranker), and
  lambdarank_truncation_level=5 / alternative objectives as ensemble
  members (rejected in round 4: truncation hurt CV 0.441 vs 0.456;
  rank_xendcg and XGBoost rank:map both weaker standalone and every blend
  worse). Both were suggested again this round; the round 4 evidence
  against them stands and was not worth re-running.

Overall read on round 6: the diagnostic-first approach paid off by ruling
out an entire category of post-hoc reranking ideas quickly (co-occurrence,
stacking, extra kNN) rather than partially validating each one against
intuition alone -- all three are individually plausible and were
independently recommended, and all three hurt. The one idea that worked
(Ridge) succeeded for a specific, identifiable structural reason (feature-
level combination via the ranker's own boosting, not a fixed score blend)
that is worth using as a template: if a "second model" idea is under
consideration, feed its predictions in as ranker input features rather
than blending its final scores, since the latter has now failed at least
8 separate times across three rounds and the former has succeeded once.
