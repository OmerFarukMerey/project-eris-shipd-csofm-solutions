Adverse Event Reaction Code Recommendation - readme
====================================================

Run: python solution.py  (from this directory)
Output: ./working/submission.csv -- THE deliverable (1640 rows, columns
        report_id,recommended_reactions, up to 5 pipe-delimited reaction_code values per
        row, ranked most-to-least likely). As of round 10 this is the GRADER-PROVEN
        DIVERSITY ENSEMBLE (round 9's rank-average of the round-5 anchor and the
        round-8 config scored 0.3072 on the real grader, the project's new best) plus
        EXACTLY ONE change: a third member added to the rank-average (round-5 anchor +
        GBM family, whose gate on the anchor base passed on both seeds). Candidates are
        built as one attributable change from the latest grader-proven anchor; no local
        hyperparameter search touches the deliverable (rounds 3, 6 and 8 all regressed
        on the real grader despite winning local CV).
Also written, diagnostic only (not the deliverable): ./working/variants/*.csv -- 4 files:
        submission_variant_ensemble_r5_r8.csv = byte-exact reproduction of the graded
        0.3072 ensemble (md5-verified) -- the fallback / current anchor;
        submission_variant_ensemble4.csv = all four member configs rank-averaged (adds
        the round-7 config as a fourth diversity axis);
        submission_variant_ensemble_w21.csv = the graded pair reweighted 2:1 toward its
        stronger member (r5 anchor, 0.3049 solo, vs r8, 0.2977 solo);
        submission_variant_round5_exact.csv = byte-exact 0.3049 single-model fallback.
Runtime: ~3-4 minutes (four pinned-config member fits -- r5 anchor, r8 config, r5+GBM,
         r7 config -- each a full OOF feature pass + 5-seed ranker fit, then cheap
         rank-average combinations; no hyperparameter search).

Score history (real grader, not local CV):
  round 1 (LR + code-static ranker, 2 CV folds):                                    0.2951
  round 2 (+ richer features, graded labels, PMI rerank, 3 CV folds):                0.3036
  round 3 (+ recency weighting, stricter fold-acceptance rule):                     ~0.29  (REGRESSION)
  round 4 (diagnosed + fixed the regression; offered 2 new levers as variants,
           not yet folded into default):                                          not resubmitted
  round 5 (per-code OOF bias calibration, gamma=0.5, REAL-GRADER CONFIRMED
           and promoted to the shipped default):                                  0.3049
  round 6 (gap-aware bias calibration -- REGRESSION, a second CV-approved
           correction-layer lever that did not transfer to the real grader):        0.302
  round 7 (fixed temporal leakage in the inner OOF splits -- a signal-layer
           fix, not another correction-layer lever; pipeline simplified):         not submitted (superseded by round 8)
  round 8 (+ per-code GBM base-learner family, the sole survivor of a 5-lever
           experiment sweep + combination verification; dual-path search;
           blocked OOF; binary labels; lambda=0.22 -- a THIRD regression):          0.2977
  round 9 (re-anchored to the round-5 config; local search retired from the
           deliverable path; diversity-ensemble variant = rank-average of the
           round-5 anchor and the round-8 config -- NEW BEST, the first
           genuine improvement over the anchor in six graded attempts):           0.3072
  round 10 (this round: the grader-proven ensemble mechanism widened -- all
           four member configs built, graded 0.3072 pair reproduced byte-
           exactly as fallback, primary = that pair + the r5+GBM member,
           plus 4-member and reweighted variants):                                not yet submitted
Rounds 3 and 6 were both real, submitted regressions, not just disappointing local-CV reads
-- see "ROUND 4" and "ROUND 7" below for the full account of each, since together they are
the most important methodological lesson in this project: a correction/calibration-layer
lever that only wins on CV (even carefully, multi-seed, multi-fold validated CV) has now
failed to transfer to the real grader TWICE. Local CV vs. real-grader history: round 1
(local 0.2388 vs real 0.2951, gap +0.056), round 2 (local 0.2657 vs real 0.3036, gap +0.038),
round 3 (local 0.2663 vs real ~0.29 -- local CV UP, real score DOWN), round 5 (bias
calibration: local CV -- short-gap AND long-gap -- and the real grader all agreed, the
first full three-way agreement in this project), round 6 (gap-aware calibration: local CV
agreed across 3 ranker seeds, real grader disagreed -- the SAME failure mode as round 3, on
a mechanism that was validated more carefully, not less). Every architecture decision is
still made by LOCAL CV DELTA on matched folds, but round 4's long-gap check fold -- while a
genuine improvement over having no gap-length check at all -- turned out to be reusable
enough across many different lever tests (recency decay, rank blend, gap-aware gamma, and
round 7's label-SVD re-test) that round 7 stopped trusting a "long-gap check says yes" verdict
in isolation: a SINGLE reused fold approved by many different hypotheses is itself a subtle
multiple-comparisons risk, not independent confirmation each time. Round 7's structural fix
was instead validated primarily against the more heavily-averaged 3-fold short-gap harness.


ROUND 4: DIAGNOSING A REAL REGRESSION
---------------------------------------
Round 3 shipped with local CV improving (0.2657 -> 0.2663) and scored ~0.29 on the real
grader -- WORSE than round 2's 0.3036. This is a genuine methodology failure, not just an
underwhelming result, and it was root-caused before touching anything else.

Diagnosis: every CV fold used through round 3 (A, B, C) tests only a 1-quarter ADJACENT gap
(train through quarter N, validate quarter N+1). But the real final model trains on all of
2023 (through Q4) and predicts across all of 2024 (Q1-Q4) -- a 1-to-4-quarter gap. Round 3's
one adopted lever, recency weighting (a per-report sample_weight decaying with quarters-back
from the most recent training quarter), was selected purely on adjacent-quarter folds where
it looked good. A new "skip-gap" check fold was added to test this directly: train on
Q1+Q2 (which has real quarter-to-quarter age variation, unlike training on Q1 alone), validate
on Q4, deliberately SKIPPING Q3 to create a genuine 2-quarter gap. Result: decay=0.55 (the
value round 3 shipped) scored 0.2389 on this skip-gap fold vs. 0.2412 for no reweighting at
all -- WORSE, despite winning cleanly on every adjacent-quarter fold. Recency weighting
tuned on a short gap makes the model lean hardest on precisely the training slice (the most
recent quarter) that is least informative about a target 2-4 quarters further out -- helpful
for a 1-quarter extrapolation, actively harmful for a longer one. That mismatch, not some
other change, is the most likely cause of the regression.

Fix: LONGGAP_FOLD_SPEC (train Q1+Q2, validate Q4) is now a REQUIRED additional gate in
run_search()'s recency-decay sweep -- a candidate decay must pass the normal short-gap
folds_improve rule AND not regress on this skip-gap fold relative to no reweighting. Rerun
under this gate, decay=0.55 is correctly rejected and the search falls back to decay=1.0
(no reweighting), which exactly reproduces round 2's config end to end (verified: the
regenerated submission.csv is byte-identical to the saved round-2 output). This is now
./working/submission.csv again -- the primary deliverable is back to the last real,
confirmed-good configuration.

Two further levers, suggested for this round, were then designed and tested with the new
discipline in mind: evaluated on BOTH the short-gap 3-fold harness AND the skip-gap check
fold from the start, not short-gap alone.
  - LR/ranker rank blend (blend_rank_scores, RANK_BLEND_W=0.15): an OUTPUT-level ensemble --
    blend the final ranker score's in-report RANK with the standalone balanced-LR's own rank
    (distinct from feeding LR in as an input feature, which the ranker already does). Smooth,
    non-jagged response across w (unlike recency decay's jagged one): small, consistent
    short-gap cost (mean 0.2627 -> ~0.2623-0.2625 across w=0.1-0.25) traded for a real
    skip-gap gain (0.2412 -> 0.2474-0.2486 across the same range). This is exactly the
    "hedge toward the real generalization distance" effect recency weighting was supposed to
    provide but didn't -- a simpler, more robust standalone signal blended in at the output
    level costs a little on the gap CV over-samples (adjacent-quarter) and helps on the gap
    that actually matters (longer-range).
  - Per-code OOF bias calibration (compute_oof_bias / apply_bias_calibration, BIAS_CAL_GAMMA
    =0.5): class_weight="balanced" on the per-code LR systematically inflates predicted
    probabilities relative to true empirical rates (confirmed: the true-rate-minus-OOF-rate
    bias is <= 0 for every code, in every fold checked). Correcting the final ranker score by
    this per-code bias (distinct from the frequency penalty's single smooth function of
    log-count -- this is empirical, per-code, from real OOF evidence) improved BOTH regimes
    simultaneously at gamma=0.5: short-gap mean 0.2627 -> 0.2636, skip-gap 0.2412 -> 0.2433.
    No tradeoff at this setting, unlike the rank blend.

Given local CV has now been shown untrustworthy for extrapolation ONCE already (round 3),
NEITHER new lever was folded into the default pipeline in round 4 despite validating on two
fold flavors instead of one. Both were offered only as isolated diagnostic variants so a
real-grader score on each would cleanly attribute to just that one lever -- see "ROUND 5"
below for what happened when the user actually tested them.


ROUND 5: BIAS CALIBRATION CONFIRMED REAL, PROMOTED TO DEFAULT; FURTHER PUSH INCONCLUSIVE
--------------------------------------------------------------------------------------------
The user submitted submission_variant_biascal.csv (gamma=0.5, bias calibration alone, no rank
blend) and it scored 0.3049 -- a genuine, real-grader-confirmed improvement over round 2's
0.3036. This is the first lever in this whole project to have full three-way agreement:
short-gap CV, the round-4 long-gap check fold, AND the real grader all say it helps. Given
that, gamma=0.5 bias calibration is now ON BY DEFAULT (BIAS_CAL_GAMMA=0.5 in fit_final's own
signature) -- it is no longer a diagnostic-only variant, it is part of ./working/submission.csv.

Asked to push further, two follow-ups were explored, both BEFORE touching the default again
(given round 3's lesson about shipping on CV alone):

1. A finer gamma sweep (0.5 to 6.0), validated on the short-gap 3-fold harness AND the
   round-4 long-gap (2-quarter, train Q1+Q2 -> val Q4) check. Short-gap performance peaks at
   gamma=0.5 and gently declines afterward; the 2-quarter long-gap check keeps improving
   through gamma=5.0 (0.2412 -> 0.2507) before dropping at 6.0. The two checks disagree on
   the "right" gamma.

2. A THIRD check fold was added to break the tie: train Q1 only, validate Q4 (a genuine
   3-quarter gap -- this one doesn't need age variation in training the way recency
   weighting did, so training on a single quarter is fine here, unlike that earlier flawed
   diagnostic). This fold peaks at a more moderate gamma=1.0 (0.2085 -> 0.2108) and is roughly
   flat-to-declining beyond that -- a DIFFERENT optimum than the 2-quarter check's gamma=5.
   Reality check on why this matters: the real final model trains through Q4 2023 and
   predicts across all of 2024 -- Q1 2024 is a 1-quarter gap (~20% of test rows), but Q2/Q3/Q4
   2024 are 2/3/4-quarter gaps (~80% combined). The short-gap-only CV harness this whole
   project has leaned on most heavily actually UNDER-represents the dominant real regime.

3. Stacking bias calibration with the rank blend was also tried (various gamma/w
   combinations). None cleanly dominated gamma=0.5 alone across all three fold flavors --
   the 2-quarter check improved further with rank blend added, but the 3-quarter check got
   WORSE than even the no-calibration baseline at several rank-blend weights. This also
   exposed a problem with the rank blend's earlier round-4 story: it was framed as a general
   "hedge toward longer gaps," but it only helps the 2-quarter check -- it actively hurts the
   3-quarter one. With just one fold per gap length (not averaged like the short-gap harness),
   each long-gap number is itself a noisy point estimate, and "helps gap X, hurts gap Y" is
   not the same as "generally helps longer gaps."

Given (a) no configuration beyond gamma=0.5 alone improves ALL three fold flavors at once --
every option tested trades one regime for another -- and (b) gamma=0.5 is the ONLY option
with real-grader ground truth, gamma=0.5 remains the shipped default. Pushing further on CV
evidence alone here would repeat exactly the round-3 mistake (trusting a change that wins on
some validation slice while it's genuinely unclear whether it wins on the one that matters).
Two new variants are offered instead, each a single clean next test:
  - submission_variant_biascal_gamma1.csv (gamma=1.0 alone): the most different profile from
    gamma=0.5 -- better on the 3-quarter check, roughly flat on the 2-quarter check, slightly
    worse on short-gap. A genuinely uncertain bet; real-grader feedback on it would directly
    resolve whether pushing gamma higher continues to help or was already past the peak.
  - submission_variant_rankblend.csv (bias-cal gamma=0.5 + rank blend w=0.15, stacked): tests
    whether rank blend adds anything on top of the now-improved baseline, given its mixed
    long-gap evidence.
  - submission_variant_no_biascal.csv: the pre-round-5 baseline (gamma=0.0), kept for a clean
    before/after comparison if useful.


ROUND 6: GAP-AWARE BIAS CALIBRATION (exploiting a signal that was sitting unused: test.csv's
own report_period)
--------------------------------------------------------------------------------------------
Asked to push further after round 5's confirmed +0.0013, a genuinely new angle rather than
more gamma-tuning: test.csv carries report_period, so for every test row the exact
quarter-gap from the training cutoff (y2023_q4) is KNOWN -- y2024_q1 is a 1-quarter gap,
q2 a 2-quarter gap, q3 a 3-quarter gap, q4 a 4-quarter gap. Nothing in the pipeline had used
that per-row information before; every post-hoc correction so far applied one number to
every row uniformly. Round 5's own gamma sweep hinted the "right" correction strength isn't
uniform across gap lengths (short-gap CV prefers gamma~0.5, the round-4 long-gap check kept
improving through gamma~5) -- which is exactly the kind of thing a per-row, gap-conditioned
correction can exploit directly instead of averaging away.

Validated via a combined-gap simulation that mimics the real mixed-gap test set at smaller
scale: train on Q1+Q2 (standing in for "all available history"), then evaluate JOINTLY on
Q3+Q4 combined (Q3 rows = gap 1, Q4 rows = gap 2 from the Q2 training cutoff) -- the closest
thing constructible from only 4 quarters of train.csv to "one model, predictions spanning
multiple gap lengths at once," which is what the real final submission actually does across
2024's 4 quarters. Applying gap-matched gammas (0.5 for the gap-1 rows, higher for the gap-2
rows) beat every UNIFORM gamma tried, including the best uniform one:
  uniform gamma=0.0 (no cal)     : combined score 0.2326
  uniform gamma=0.5 (shipped)    : combined score 0.2336
  uniform gamma=1.0/2.0          : combined score 0.2335 (both, i.e. no better than 0.5)
  gap-aware (gap1=0.5, gap2=3.0) : combined score 0.2346
  gap-aware (gap1=0.5, gap2=5.0) : combined score 0.2373  <- best found
Checked for robustness across 3 ranker seeds (42/43/44) before trusting it, given round 3's
lesson about single noisy long-gap evidence -- gap-aware beat uniform gamma=0.5 in all three
(+0.0037, +0.0020, +0.0040 respectively). This is a real, repeatable effect, not a
seed-specific artifact.

Implementation (gap_aware_gamma / apply_bias_calibration in solution.py): a per-test-row
gamma vector from each row's own report_period, via GAP_GAMMA_BY_QUARTER = {1: 0.5, 2: 3.0,
3: 1.5, 4: 2.0}. Only gaps 1 and 2 have solid same-training-size evidence (the simulation
above); gap 3's only direct check (train Q1 alone -> val Q4) confounds gap length with a
much smaller training set (823 vs. 1626 rows), so its apparent "wants less gamma than gap 2"
finding is treated with real skepticism, not taken at face value. Gap 4 has NO direct
evidence at all (train.csv only spans 4 quarters, so a 4-quarter gap can't be constructed
locally). The gamma=1.5/2.0 values for gaps 3/4 are therefore a conservative interpolation
between the well-evidenced gap-1 and gap-2 endpoints, not a confident extrapolation of the
gap-2 peak (gamma=5-7) -- deliberately not chasing the single most extreme point found,
consistent with every other lesson this project has learned about single-fold long-gap
evidence being noisier than it looks.

Given this is a more elaborate mechanism than the already-confirmed uniform case (more
free parameters, weaker evidence for 2 of its 4 pieces), it was offered as
submission_variant_biascal_gapaware.csv rather than promoted straight to the default,
exactly as uniform gamma=0.5 was handled in round 5 before ITS promotion. UPDATE (round 7):
this variant scored 0.302 on the real grader -- WORSE than the plain gamma=0.5 default
(0.3049), despite the 3-seed local validation. It is withdrawn; see ROUND 7 below for the
diagnosis and what changed as a result.


ROUND 7: FIXING TEMPORAL LEAKAGE IN THE BASE-LEARNER OOF SPLITS (a signal-layer fix, not
another correction-layer lever)
--------------------------------------------------------------------------------------------
Two rounds in a row (3 and 6) shipped a lever that won cleanly on local CV -- including,
for round 6, three independent ranker seeds -- and then lost on the real grader. That is a
pattern, not bad luck twice, and two independent outside reviews converged on the same
root-cause diagnosis before any further correction-layer tuning was attempted.

THE BUG: every per-code base learner (balanced LR, unbalanced LR, ComplementNB) and the
per-token lift features were built OOF using `KFold(shuffle=True)` on each fold's train
split. Shuffling ignores report_period entirely -- for fold C (train = Q1+Q2+Q3), a Q1
row's OOF meta-feature could easily be produced by a model trained on a MIX that includes
Q2/Q3 rows. That is an INTERPOLATION task (filling a gap within a window the model has
already seen contemporaneous data from). But the ranker's real job -- and every val/test
prediction in this whole pipeline -- is EXTRAPOLATION: predict a genuinely later, unseen
period from only earlier data. Random-shuffled OOF therefore made the ranker's OWN TRAINING
META-FEATURES systematically easier/cleaner than what the same mechanism produces at real
prediction time -- a train/serve quality mismatch invisible to any metric computed only on
correctly-temporal validation SPLITS, because the leak lives inside how each split's own
training-row features are constructed, not in which rows land in which split.

THE FIX (period_block_splits / fit_percode_oof / fit_token_lift_oof in solution.py):
leave-one-quarter-out splits instead of random KFold, for every base-learner OOF and the
token-lift OOF. Falls back to random KFold only when a fold's train split is itself a
single quarter (e.g. fold A), where quarter-blocking isn't definable. Verified this is not
a no-op: re-running the full search afterward, the ranker's own hyperparameter search
landed somewhere genuinely different (use_graded flipped from True to False, min_child_samples
15->10) -- the underlying signal actually changed, it isn't the same numbers recomputed.
Local CV (short-gap 3-fold, pre-bias-cal) dropped slightly, 0.2657 -> ~0.2612-2633 depending
on exact run -- consistent with the leakage theory's own prediction (a previously inflated,
interpolation-assisted number coming down toward something more honest) rather than a red
flag on its own.

DIAGNOSIS BEFORE MORE MODELING (per outside review advice, done as pure analysis, no
training): decomposed fold-C validation performance two ways.
  - Weighted recall@5 by relevant-code train_frequency_bucket: count_250_plus=0.729,
    count_100_249=0.475, count_40_99=0.269, count_15_39=0.177. A clean, roughly 4x gap
    between the most and least common codes -- the loss is concentrated in the RAREST
    codes, exactly the "codes that never crack the top 5" pattern the reviewer predicted.
  - Test-vocabulary OOV coverage against train, per token column: 0.6-2.2% of test token
    OCCURRENCES (not unique tokens -- occurrence-weighted is what actually matters) are
    unseen in train, across suspect/concomitant/indication profiles. This is NOT a
    meaningful source of loss; vocabulary drift is a non-issue here, so fitting vectorizers
    on train+test features (legitimate, no labels involved) was not pursued further.

Given the rare-code concentration, two structural additions were re-examined with the fix
in place (both had been rejected or not yet tried under the old, leaky OOF):
  - Label-matrix SVD (round 4 rejected this on the leaky OOF; re-tested here with the
    ridge-regression report-factor OOF ALSO quarter-blocked). Result: the SAME suspicious
    pattern as before -- helps the long-gap check fold (+0.004 to +0.005 at rank 8-15) but
    only 1 of 3 short-gap folds (fold C), the other two flat-to-worse. Given round 6 just
    demonstrated that exact pattern ("long-gap check says yes") failing on the real grader,
    this was NOT adopted or even offered as a variant -- the bar for trusting the long-gap
    check alone is now much higher than it was two rounds ago.
  - Isotonic-calibration x frequency-weight ranking (replacing freq-penalty/bias-cal/
    gap-aware with the theoretically-motivated Bayes-ish ranking `p_hat(code|report) x
    weight(code)`, per outside review): implemented with a quarter-blocked OOF ranker score
    pass (~4 extra ranker fits per fold) feeding per-code isotonic regression, tested on the
    more heavily-averaged 3-fold short-gap harness rather than the single long-gap fold.
    Result: DECISIVELY worse than the plain raw ranker score in every combination tried
    (raw score 0.2612 vs. isotonic-alone 0.2501 vs. isotonic x freq-weight 0.2435 -- the
    theoretically cleanest option was the worst empirically). Not adopted. A theoretically
    elegant replacement is not automatically an empirical improvement; this was checked
    rather than assumed, same as everything else in this project.

SIMPLIFICATION (per both outside reviews' explicit request): RECENCY_DECAY_GRID emptied
(the sweep consistently landed on "no reweighting" anyway once gated by the long-gap check,
and its non-gated ancestor caused round 3's regression -- pure overhead now, no observed
benefit ever since the gate was added). The B+C-only search and its foldBC variant were
dropped (never adopted as anything but a diagnostic). The variant list was cut from 8 files
to 2 (current pipeline, and bias-cal forced off) -- per the explicit warning that many
variant files per run risks spending scarce real-grader submissions on questions CV has
already shown it cannot reliably answer (gamma 0.5 vs. 1.0, rank blend, gap-awareness).
Future submissions should go to STRUCTURAL candidates one at a time, biggest expected gain
first, not more correction-layer micro-variants.

NOT YET DONE as of round 7 (all four were then tested properly in round 8, see below):
per-column separate base learners, hashed cross-profile interaction features, NB-SVM-style
log-count-ratio features, and per-code LightGBM classifiers as a fourth OOF base learner.


ROUND 8: THE SIGNAL-LAYER EXPERIMENT SWEEP -- ONE SURVIVOR OF FIVE (per-code GBM base
learners), PLUS A GREEDY-SEARCH PATH-DEPENDENCE FIX IT EXPOSED
--------------------------------------------------------------------------------------------
With the OOF leakage fixed, all four outstanding signal-layer candidates from the outside
reviews (plus a fifth, configuration-rank ensembling) were run as FIVE PARALLEL, ISOLATED
EXPERIMENTS, each A/B-testing exactly one lever against the identical fixed baseline
(round-7 config: binary labels, nl=7, lr=0.03, mcs=10, trunc=30) on the same 3 temporal
folds, all inside a single python process per experiment (LightGBM numbers are only
comparable within one process on this machine), all with quarter-blocked OOF and the same
anti-leakage rules, all under the same acceptance bar (mean up AND >=2/3 folds up). The
winners then went to a separate combination-verifier run: each winner solo, all winners
combined, leave-one-out, on TWO ranker seeds (42/43).

Results, honestly reported (baseline mean 0.261844; folds [0.29848, 0.22393, 0.26312]):
  - Per-column LRs (suspect-only / indication-only / both): FAILED. Every variant below
    baseline on mean (best -0.0014); each improved only 1/3 folds. The column-pure signals
    are redundant with the existing all-column LR/NB families.
  - NB-SVM (log-count-ratio weighted LR, r recomputed inside every OOF split since it is
    label-derived): FAILED. Both variants below baseline (best -0.0014), 1/3 folds each.
    Single-quarter fold A degrades most -- rarest-code r estimates are noisiest there.
  - Hashed suspect x indication interactions (FeatureHasher 2^16 -> per-code balanced LR):
    MARGINAL PASS solo (mean +0.0005, 2/3 folds; pairs-only variant; the full singles+
    crosses variant failed), and the experiment agent itself flagged it as fragile
    (consistently hurts the smallest-train fold). The combination verifier then showed it
    FAILS the two-seed bar solo (seed 43: 1/3 folds) and makes the combined set WORSE than
    pcgbm alone. Rejected -- correctly caught by the multi-seed verification layer that
    rounds 3/6 taught this project to require.
  - Configuration-rank ensembling (4 diverse ranker configs, rank-averaged): FAILED.
  - PER-CODE LIGHTGBM CLASSIFIERS (pcgbm): PASSED CLEANLY -- the only one. Small
    LGBMClassifiers (80 trees, lr 0.1, 7 leaves, deterministic, quarter-blocked OOF), one
    per code, on the existing full report-feature matrix; feature blocks "gbm" (prob) +
    "gbm_rank" (in-report rank). Improved ALL 3 folds on seed 42 (mean +0.003491) AND all
    3 on seed 43 (+0.003192) in the combination verifier, which also reproduced the
    experiment's numbers exactly to full precision. This is the nonlinear base learner the
    reviews predicted would capture token x token / token x demographic interactions the
    three linear families cannot; the gain grows with training-set size across folds
    (+0.0002 / +0.0021 / +0.0082), which is the right shape for the real final model
    (trains on all 4 quarters, more than any CV fold).
Integrated into compute_model_features exactly as validated: prob + rank blocks only (no
logit block -- the evidence covers precisely the tested configuration), pinned n_jobs, same
quarter-blocked OOF as the linear families.

THE PATH-DEPENDENCE BUG THE NEW FEATURE EXPOSED: the first integrated run landed at mean
0.2611 -- WORSE than the experiment's fixed-config 0.2653 -- because the old search decided
binary-vs-graded labels ONCE, up front, at untuned base params (nl=15, lr=0.05, trunc=10),
and the GBM feature happened to flip that particular comparison to graded; the single
greedy path then locked in a graded config that never reached the binary optimum. Fix: the
coordinate search now runs as a FULL SEPARATE PATH per label scheme (binary and graded each
get their own truncation + coordinate sweeps), plus the experiment-validated anchor config
is evaluated explicitly as a third candidate, and the best ENDPOINT wins. Result: binary
path endpoint 0.2650 (vs graded 0.2611, vs anchor 0.2642 -- the binary path found a
slightly better mcs=20 than the anchor's mcs=10), and with the post-hoc freq penalty
(lambda=0.22) + PMI rerank (alpha=0.08) the final local CV is 0.2696, folds
[0.30357, 0.22872, 0.27655] -- the best 3-fold number this project has recorded, with the
biggest gains exactly where the round-7 diagnostic located the loss (the largest fold).

Process note: this round was run as an orchestrated parallel workflow (5 experiment agents
+ 1 combination-verifier agent), each agent reporting exact fold numbers from a
single-process A/B -- the discipline that four of five plausible, outside-recommended
levers were REJECTED on evidence (one of them after passing its own solo bar) is the point;
the surviving lever carries the strongest pre-submission evidence of anything shipped since
round 5.

UPDATE: round 8 scored 0.2977 on the real grader -- the THIRD straight regression from a
locally-validated change (after rounds 3 and 6). See ROUND 9.


ROUND 9: RE-ANCHORING TO THE GRADER -- LOCAL SEARCH RETIRED FROM THE DELIVERABLE PATH
--------------------------------------------------------------------------------------------
Round 8's real score (0.2977) despite the best local CV ever recorded (0.2696) settles a
question this project has been circling since round 3: the 3-fold 2023 harness is NOT a
reliable arbiter for architecture-level decisions about the 2024 test set. Every real-grader
submission that deviated from the round-5 configuration has scored WORSE than it -- recency
weighting (~0.29), gap-aware calibration (0.302), and now the round-7/8 stack (0.2977), each
of which won local CV, some across multiple seeds and fold flavors. Three independent
failures with three different mechanisms is a property of the VALIDATION SETUP, not bad luck:
the local->real gap is large (+0.038 to +0.056 across rounds) and evidently config-dependent,
so optimizing local CV re-ranks configs in ways the grader does not honor.

Round 8 also confounded attribution by bundling three changes (blocked OOF + GBM family +
search-selected binary/lambda=0.22) into one submission -- so it cannot even be said WHICH
of them hurt. Round 9 fixes the process:

  1. THE ANCHOR IS REPRODUCIBLE, BYTE-FOR-BYTE. The round-5 configuration (graded labels,
     shuffled inner OOF, lambda=0.02, alpha=0.08, bias-cal gamma=0.5) is now pinned in code
     as R5_ANCHOR_CONFIG, with module flags (INNER_OOF_BLOCKED, INCLUDE_GBM_FAMILY) that
     make the historical behavior selectable. The reproduction was verified byte-identical
     (md5 aa79249c62951af941055986d8420081) to the stored copy of the output that scored
     0.3049 -- this is variants/submission_variant_round5_exact.csv, the permanent fallback.
     (Note on the OOF flag: "shuffled inner OOF" uses only TRAIN labels -- neither setting
     touches test information; round 7's "leakage" framing was about intra-train temporal
     structure, a modeling choice, not a competition-rules issue. The blocked variant has
     never been graded in isolation and both graded submissions containing it lost to the
     shuffled-OOF anchor, so the anchor keeps its original scheme.)
  2. THE PRIMARY IS ONE ATTRIBUTABLE CHANGE FROM THE ANCHOR: + the per-code GBM family,
     nothing else. Not the round-8 stack -- the anchor's own label scheme (graded), its own
     inner-OOF scheme (shuffled), its own small lambda (0.02). The GBM family was
     re-gate-tested on THIS exact base before shipping: seed 42 mean +0.0019 (2/3 folds),
     seed 43 mean +0.0031 (3/3 folds). If this submission beats 0.3049, the GBM signal is
     real on the grader too; if it loses, the fallback is one file away and the GBM family
     is retired with clean attribution either way.
  3. THE DIVERSITY ENSEMBLE, the one standard ranking-competition lever never yet tried on
     this grader: variants/submission_variant_ensemble_r5_r8.csv rank-averages the round-5
     anchor scores with the round-8 config scores -- two models both known ~0.30 on the
     real grader that differ in inner-OOF scheme, label scheme, feature set, and penalty
     strength. Ensembles of diverse same-strength rankers routinely beat both members;
     unlike every failed lever so far, this does not ask local CV to arbitrate anything.
  4. run_search() is RETIRED from the deliverable path (kept in the file as a diagnostic
     tool only). main() now builds the three outputs above from pinned configs, no search.

RESEARCH NOTES (user-requested web survey; context, not shipped claims): the metric here is
essentially a propensity-weighted MAP@5, and the extreme-multi-label literature (PfastreXML,
propensity-scored probabilistic label trees) addresses exactly this shape of problem --
optimizing propensity-scored nDCG/precision directly at TRAIN time so tail labels get
priority, with inverse-propensity weights of the form p_l = 1 + C(n_l + B)^-A rather than
the plain 1/sqrt(n) this metric uses. The project's graded-label_gain mechanism is a crude
version of exactly this idea (and is part of the anchor). The literature's cleaner
formulation -- propensity-weighted TRAINING of the ranker itself (weights inside the
LambdaRank gain, tuned to the metric's own 1/sqrt form) -- is the most principled untried
lever if the grader confirms the GBM primary; but per this project's history, it would be
built as one attributable change from whatever anchor the grader has most recently blessed,
never shipped on local CV evidence alone.

OUTCOME: the round-9 ensemble variant scored 0.3072 on the real grader -- NEW BEST (+0.0023
over the 0.3049 anchor), the first genuine improvement in six graded perturbation attempts.
The r5+GBM primary was not graded solo; it is folded into round 10's ensembles as a member.


ROUND 10: WIDENING THE GRADER-PROVEN ENSEMBLE
--------------------------------------------------------------------------------------------
Round 9's key result: rank-averaging two ~0.30 models with maximum configuration diversity
(round-5 anchor 0.3049 solo + round-8 config 0.2977 solo -> 0.3072 combined) is the first
mechanism the grader itself has rewarded since round 5. Note what made it work where six
single-model perturbations failed: it never asked local CV to arbitrate anything -- both
members' strengths were known from the grader, and the combination exploits their error
DIVERSITY (different inner-OOF scheme, label scheme, feature set, penalty strength) rather
than betting on any one config being "better."

Round 10 extends the same mechanism by one attributable step at a time. All four member
models are now built by main() from pinned configs:
  A. r5 anchor        (real 0.3049 solo; shuffled OOF, graded, no GBM, lambda=0.02)
  B. r8 config        (real 0.2977 solo; blocked OOF, binary, +GBM, lambda=0.22)
  C. r5+GBM           (ungraded solo; the anchor plus the GBM family, whose gate on this
                       exact base passed on both seeds: +0.0019/+0.0031 local mean)
  D. r7 config        (ungraded solo; blocked OOF, binary, no GBM, lambda=0.08)
Deliverables:
  - PRIMARY submission.csv = rank-average(A, B, C): the graded 0.3072 pair plus one new,
    differently-biased member -- the same move the grader just rewarded, taken one step
    further.
  - variant ensemble_r5_r8 = rank-average(A, B): byte-exact reproduction of the graded
    0.3072 file (md5 dc766cdbac38919909ded3b46dd41c71) -- the fallback / current anchor.
  - variant ensemble4 = rank-average(A, B, C, D): the widest available diversity.
  - variant ensemble_w21 = (2A + B)/3: the graded pair reweighted toward its stronger
    member -- tests whether equal weighting under-uses the 0.3049 member.
Suggested grading order: PRIMARY first (3-member), then ensemble4, then ensemble_w21 --
each answers one question about how far the ensemble mechanism stretches.


PROBLEM
-------
Each report is an anonymized adverse-event signature (salted lossy bucket profiles for
drug/indication/route/action tokens, plus coarse categoricals for country, reporter
qualification, sex, age, seriousness flags, and drug count). Given a test report, recommend
up to 5 of a fixed universe of 90 allowed reaction codes, ranked by likelihood. Scored by
frequency-balanced MAP@5: each hidden relevant code gets weight 1/sqrt(hidden_count(code)),
so correctly recommending a rare code is worth substantially more than a common one.

EDA (verified directly against the CSVs, not assumed; re-verified again this round before
acting on any outside advice):
  - train.csv = 3269 rows, all report_period in {y2023_q1..q4}; test.csv = 1640 rows, all
    {y2024_q1..q4} -- a genuine earlier/later temporal split.
  - Every one of the 90 allowed codes appears in train.csv's reaction_targets (count range
    26-300, matching allowed_reactions.csv's own train_frequency_bucket cutoffs); no label
    noise to filter.
  - route_profile and action_profile are 100% empty in both train and test, zero vocabulary
    either side -- re-checked yet again this round (independently, not just recalled from
    earlier rounds). Still empty, so still excluded.
  - report_period is a trap: train is entirely y2023_qX, test is entirely y2024_qX, so the
    two never share a category -- every test row would map to a model's "other" bucket for
    this raw feature. Fixed in round 2 by deriving a year-agnostic "quarter" (q1..q4) column.


APPROACH / MODEL ARCHITECTURE
------------------------------
One long (report, code) pair table -- every report considered against all 90 fixed candidate
codes (~294K train pair-rows on the full train set) -- with every signal fed in as an INPUT
FEATURE to one LightGBM LambdaRank model, rather than hand-blending separate models' final
scores (a sibling project, "Coastal Sensor Signature Recommendation", documented the latter
failing repeatedly on a similarly frequency-balanced metric).

Per-row features:
  - Report-side: one-hot categoricals (quarter, country buckets, reporter qualification, sex,
    age bucket, drug count bucket) + multi-hot token bags (seriousness, suspect/concomitant
    drug, indication) with missingness/count extras.
  - Three differently-biased per-code classifier families (balanced LR, unbalanced LR,
    ComplementNB), each contributing raw probability + logit + in-report rank (9 columns),
    all leak-free OOF via a 4-inner-fold scheme.
  - Per-token-column naive-Bayes-style max/mean log-lift features (captures "one strongly
    indicative token" that regularized LR averages away).
  - Code-side statics: log1p(train count), raw frequency, one-hot train_frequency_bucket,
    and a 90-dim code-identity one-hot.
Training signal: graded LambdaRank relevance labels (grade 1-4 by train_frequency_bucket,
label_gain = that grade's mean frequency-balanced-metric weight from the fold's own train
counts) instead of binary 0/1, so LambdaRank's own NDCG-style objective pushes rare relevant
codes toward the top during training. A per-report recency-decay sample_weight (new this
round, see below) further reweights training rows by how recent their quarter is.
Post-hoc: a small frequency penalty sweep (usually near-0, see "what did not work") followed
by a PMI (pointwise mutual information) code-code co-occurrence rerank -- boosts each
candidate's score by alpha * mean PMI with the ranker's own current top-2 predicted codes.

Final local validation (3-fold average frequency-balanced MAP@5): 0.2663, up from round 2's
0.2657 and round 1's 0.2388 (2-fold; see "Validation strategy" for the apples-to-apples check
on comparable folds).


WHAT'S NEW IN ROUND 3 (recency weighting was later found to REGRESS the real
score and was reverted in round 4 -- see "ROUND 4" above for the full story.
Kept below as-written for an honest record of what the round-3 evidence
looked like at the time, including why it was initially convincing.)
------------------------------------------------------------------------
The user supplied a detailed, numbered list of further levers to try, several coming from
independent outside review. Every one was checked against local CV evidence -- including
ones that sounded highly plausible mechanistically -- rather than assumed to transfer. Most
did not hold up; this is reported honestly below rather than only reporting the ones that
worked.

A stricter acceptance rule was adopted first, before evaluating anything else: a change is
only adopted if it improves the MEAN **and** strictly improves more than half the folds (2 of
3), not just the mean. On a validation set this small (792-851 rows per fold), a change can
win on mean by helping one noisy fold a lot while quietly hurting the other two -- exactly
the failure mode a much larger frequency-penalty value produced in round 2 (see that round's
notes). This rule is now baked into every search decision in solution.py's run_search(), not
just applied by eyeball on ad hoc checks. It turned out NOT to be sufficient on its own --
see round 4 -- because all three folds it was applied to shared the same blind spot
(adjacent-quarter gaps only).

ADOPTED AT THE TIME, LATER REVERTED (see ROUND 4):
  - Recency weighting: a per-report sample_weight = decay^(quarters back from that fold's
    most recent train quarter), decay=0.55 selected by sweeping {0.7, 0.55, 0.5, 0.3} inside
    the actual search (not an isolated script). Fold A (which trains on a single quarter, Q1
    only) is structurally invariant to this by construction -- no age variation within a
    single quarter to weight by -- so it ties exactly at every decay value, which is itself a
    useful sanity check that the implementation is correct. Folds B and C both improve
    (0.2263->0.2271 to 0.2274 range across reruns, i.e. depending on decay setting;
    0.2636->0.2664ish), lifting the mean from 0.2627 to 0.2639 before PMI rerank, and to
    0.2663 after (PMI's own best alpha also shifted slightly, 0.08->0.1, once recency
    weighting was in place). Checked for robustness across two different ranker seeds before
    trusting it, because the response across decay values is jagged rather than smooth (only
    a few discrete "age" values exist per fold, unlike a continuous dial like PMI's alpha) --
    decay=0.55 was the best point under both seeds tested, which is what made it trustworthy
    at the time DESPITE the jaggedness. In hindsight the jaggedness itself, plus the fact that
    every fold used to validate it shared the same 1-quarter-gap structure, should have been
    treated as a bigger warning sign than it was -- round 4's skip-gap check fold showed this
    exact decay value scoring WORSE than no reweighting at a genuine 2-quarter gap, which is
    much closer to what the real final model (train through Q4 2023, predict all of Q1-Q4
    2024) actually faces. The post-hoc frequency penalty converged to lambda=0.0 once recency
    weighting was in the mix (previously 0.02); this is now understood as recency weighting
    absorbing/masking some of the same popularity-drift correction the penalty provides,
    rather than genuinely superseding it.

REJECTED WITH EVIDENCE (all genuinely tried, not skipped):
  - Native categorical code_id (LightGBM's categorical_feature, replacing or supplementing the
    90-dim one-hot). The mechanistic argument was sound (native categorical splits partition
    all 90 codes far more efficiently than one-hot at shallow tree depth) but the one-hot
    version won on 2 of 3 folds against both a categorical-only and a "both" variant. Possible
    explanation: at num_leaves=7 the trees are shallow enough, and the per-code classifier
    features already carry enough code-specific signal, that the extra split-efficiency
    wasn't the binding constraint -- and LightGBM's categorical split search (partitioning by
    gradient statistics) may add variance of its own with only 823-2477 training reports.
  - Extending NUM_LEAVES_GRID to 63/127 (with min_child_samples up to 80 paired at each, per
    the advice that the old optimum may have moved with a richer feature set): lost
    decisively and monotonically to num_leaves=7 at every setting tried. With only
    823-2477 rows per fold, deeper trees just overfit -- the feature set got richer, but the
    row count didn't, and the row count is what actually bounds safe tree depth here.
  - Low-rank factorization of the label matrix (TruncatedSVD of Y at rank 8/15/25/40 for code
    embeddings, OOF ridge-regressed report factors, dot-product fed as a ranker feature plus
    embedding dims appended to code_static) -- the one lever the reviewer explicitly flagged
    as "the biggest untapped signal." Consistent negative at every rank tested: fold B
    improved but folds A and C both got worse, every time. Likely cause: the rare codes this
    was meant to help (min count 26) don't have enough co-occurrence signal at fold scale
    (Q1 alone is 823 rows) to support a stable embedding, and the ridge-regression step from
    report features to SVD factors isn't predictive enough for held-out rows to add signal
    rather than noise.
  - Extending the per-token log-lift machinery to the scalar categoricals (patient_sex,
    patient_age_bucket, reporter_qualification, country buckets, drug_count_bucket) --
    described as "nearly copy-paste," and it was: the same fit_token_lift/apply_token_lift
    functions work unmodified since these columns are single-token strings. Only 1 of 3 folds
    improved; not adopted.
  - Hyperparameter squeeze (colsample_bytree, min_child_samples, reg_lambda coordinate sweeps).
    Notably, an ISOLATED ad hoc script test first suggested colsample_bytree=1.0 won cleanly
    on all 3 folds -- but this did NOT replicate when swept inside the actual pipeline's own
    consistent run (same process, same contexts, same pair tables): there, colsample=1.0 only
    won 1 of 3 folds and was correctly rejected by the folds-improve rule. The most likely
    explanation is LightGBM's `deterministic=True` guarantee only holds for a FIXED thread
    count -- it does not guard against auto-detected thread count varying between separate
    process invocations under different system load, which can shift floating-point
    summation order and hence which splits get chosen. Lesson: trust hyperparameter
    comparisons made WITHIN one consistent process/run over comparisons made by
    cross-referencing a separate ad hoc script's numbers against the main pipeline's, even
    with fixed seeds. None of colsample_bytree, min_child_samples, or reg_lambda changed
    from their round-2 values once evaluated this way.
  - Finer LambdaRank relevance grades (quantile-binning per-code 1/sqrt(train_count) into 6,
    8, or 12 grades instead of the current 4 train_frequency_bucket-based grades). None beat
    the current scheme on mean; a quantile-based 4-grade variant nominally passed the
    folds-improve check (2 of 3) but still lost on mean (0.2622 vs. 0.2627), so it was
    rejected under the combined bar. Plausible reason: with only 26-300 positives per code
    to begin with, finer grade resolution mostly adds noise to the training objective's gain
    structure rather than genuine extra signal.
  - A shared-representation multilabel MLP (sklearn MLPClassifier fit directly on the
    indicator matrix Y, one hidden layer, several hidden-size/regularization combinations
    tried) as a fourth differently-biased base learner, OOF like the other three. Every
    single configuration tested underperformed the no-MLP baseline -- a clean, decisive
    rejection, not a close call. With only 823-2477 training rows, a neural net (even a small
    one) doesn't have enough data to learn a shared representation that beats the existing
    per-code linear/NB models plus lift features.
  - k-NN neighbor-vote feature as a ranker input: re-checked from scratch under the full
    round-2/3 feature set (not assumed still true from earlier rounds) -- only 1 of 3 folds
    improved, confirming the original rejection still holds even with richer features
    surrounding it.


FEATURE ENGINEERING
--------------------
  - Dropped: route_profile, action_profile (100% empty, re-verified this round).
  - quarter (derived from report_period, generalizes across years) + primary_country_bucket,
    occur_country_bucket, reporter_qualification, patient_sex, patient_age_bucket,
    drug_count_bucket: one-hot, with any category below 10 occurrences in that fold's training
    split (or unseen in training entirely) bucketed into a shared "other" column.
  - Token-bag columns (seriousness_profile, suspect/concomitant-drug, indication): binary
    multi-hot on train-only vocabulary, plus a missingness flag and token-count numeric per
    column, plus the max/mean log-lift features.
  - A separate compact representation (scalar one-hot + ~40-component TruncatedSVD of the
    token-bag matrix, L2-normalized) is used only for the k-NN cosine-similarity ablation --
    the raw multi-hot space is too high-dimensional relative to fold size for raw-space
    cosine distance to be meaningful.
  - Code-side: log1p(train count), raw frequency, one-hot train_frequency_bucket, and a 90-dim
    code-identity one-hot.


VALIDATION STRATEGY
--------------------
NOTE: the "Three-fold results" and "Comparing to round 2" numbers just below are round 3's
AT-THE-TIME results (recency decay=0.55 adopted, final mean 0.2663) and are kept for the
historical record. The actual current pipeline reverted recency weighting after round 4's
diagnosis (see "ROUND 4" above) -- current local CV is back to round 2's 0.2627/0.2657
(no-penalty/with-PMI), with a fourth, skip-gap check fold now also part of the harness
specifically to catch what these three folds alone could not.

Three rolling-origin temporal folds (never random k-fold, matching the real earlier/later
split):
  Fold A: train y2023_q1        (823 rows)  -> validate y2023_q2 (803 rows)
  Fold B: train y2023_q1+q2    (1626 rows)  -> validate y2023_q3 (851 rows)
  Fold C: train y2023_q1+q2+q3 (2477 rows)  -> validate y2023_q4 (792 rows)
Fold A is the noisiest of the three (least training data, sparsest rare-code counts, and
structurally invariant to recency weighting since it's a single quarter) but including it
in the average reduces the chance that a 2-fold search overfits to one fold's idiosyncrasies.
The frequency-balanced MAP@5 metric is reimplemented directly from the challenge spec,
evaluated per fold with weight(code) computed only from that fold's OWN validation-split
label counts (never from train counts).

Three-fold results (evidence base for round 3's choices):
  Popularity-only (same top-5 every row)              : 0.0860  (folds: 0.1209, 0.0967, 0.0405)
  Per-code LR-only (balanced)                         : 0.2284  (folds: 0.2600, 0.1920, 0.2332)
  k-NN neighbor-vote only (k=40)                       : 0.1940
  Ranker, graded labels, tuned (round 2 end state)      : 0.2627  (folds: 0.2982, 0.2263, 0.2636)
  Ranker, + recency decay=0.55                          : 0.2639  (folds: 0.2982, 0.2271, 0.2664)
  Ranker, + freq penalty (lambda=0.0, i.e. no-op now)   : 0.2639
  Ranker, + PMI co-occurrence rerank (alpha=0.1)        : 0.2663  <- final

Comparing to round 2 on the same 3 folds: fold A ties exactly (0.2982, expected -- recency
weighting cannot affect a single-quarter training fold), fold B improves (0.2263 -> 0.2277),
fold C improves (0.2636 -> 0.2695). Two of three folds genuinely better, one structurally
unchanged -- this is what "adopt only if it clears the folds-improve bar" is meant to catch
and did catch correctly here (as opposed to the several ideas above that only won on mean).


WHAT WORKED (round 1 + round 2, still standing)
--------------------------------------------------
  - The ranker architecture itself: every standalone signal is far weaker than the LightGBM
    ranker that combines them (0.2284 for LR alone, 0.1940 for kNN alone, 0.0860 for
    popularity alone, vs. 0.2663 final) -- roughly 3x the naive popularity floor.
  - class_weight="balanced" on the per-code logistic regressions, and a leak-free 4-inner-fold
    OOF pass for every per-code classifier feature (LR, unbalanced LR, NB) so the ranker's
    training rows never see a code-probability derived from their own label.
  - Rare-category bucketing (train count < 10 -> "other") for the high-cardinality country
    columns.
  - Deriving "quarter" instead of using raw report_period (see EDA above).
  - PMI co-occurrence reranking: distinct from a raw co-occurrence-COUNT graph (which a
    sibling project found hurt CV monotonically, since codes co-occurring across many reports
    mostly reflects generic popularity) -- PMI explicitly divides out each code's marginal
    frequency, and gives a smooth, non-degenerate, all-three-folds-improve-together peak.
  - Graded LambdaRank relevance labels over binary + a post-hoc-only frequency penalty: more
    principled (shapes the training objective directly), though the margin over binary was
    small on its own (~+0.0005) until recency weighting and PMI rerank compounded on top.


WHAT DID NOT WORK (round 1 + round 2, for continuity -- full detail in earlier notes)
------------------------------------------------------------------------------------
  - route_profile / action_profile: 100% empty, re-verified three times now across three
    rounds. Still excluded.
  - k-NN neighbor-vote feature and an unbounded post-hoc frequency penalty as ranker inputs:
    both rejected in round 1/2, both re-confirmed rejected in round 3 under the richer
    feature set (see above).
  - An unbounded frequency-penalty lambda search finds an apparent optimum near 1.2 where the
    penalty term dominates the ranker's own score and the two validation folds diverge
    sharply (~0.29 vs. ~0.22) -- a clear overfitting signature, caught and rejected in round 2,
    and the same caution (checking whether ALL folds move together, not just the mean) is
    what round 3's stricter acceptance rule now enforces automatically everywhere.
  - Rare-code-aware sample weights stacked ON TOP of graded relevance labels (round 2): made
    things worse, confirming the two mechanisms double-count the same rarity correction rather
    than adding new information. (Recency-weighting sample_weight, added in round 3, is a
    different axis -- reweighting by TIME, not by code rarity -- which is why it does not
    hit the same double-counting problem.)
  - n_estimators=300 already near-optimal at CV-fold scale in round 2 (500/800 both overfit
    the smaller folds); re-confirmed in round 3's own coordinate search.
