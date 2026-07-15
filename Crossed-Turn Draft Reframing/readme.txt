Crossed-Turn Draft Reframing solution
=====================================

Run
---
From this directory:

    python solution.py

The script reads dataset/public/train.csv and dataset/public/test.csv and writes
working/submission.csv. It uses only supplied rows. It does not recover
plaintext, identify participants, reverse-map IDs, query external corpora, or
use hidden test outcomes.

Overview
--------
The solution is now a two-pass, cross-fitted opaque-token alignment model.
Length geometry is useful, but the central signal is whether protected draft
content appears to survive in the protected final reply. Literal token
intersection is impossible because draft and reply use different keyed code
spaces. The model therefore learns statistical code-to-code associations from
recurring draft/reply co-occurrence without assigning any code a plaintext
meaning.

Pass 1: cross-field alignment
-----------------------------
Separate binary document-term matrices are built for draft and reply unigrams.
For fit rows, the association matrix is:

    A = D.T @ R

Each nonzero A[d,r] is divided by sqrt(df_d[d] * df_r[r]), producing a
cosine-style co-occurrence association rather than a raw-frequency match.

Two mappings are cross-fitted:

1. A label-free map using every eligible fit row.
2. A clean-pair map using only labeled non-reframed fit rows. These rows are
   less likely to mix abandoned draft content into the correspondence map.

For every training dyad, mappings are generated while excluding that complete
dyad. Test mappings use all applicable training rows. No training row creates
its own association feature.

First-pass features include:

- mean, maximum, and quantiles of association to reply codes actually present;
- association relative to each draft code's best learned counterpart;
- top-1 and top-3 mapped-code survival;
- mapped-code coverage and confidence;
- confidence-thresholded missing-code fractions;
- occurrence- and inverse-frequency-weighted survival;
- reverse reply-to-draft correspondence;
- forward/reverse union and intersection;
- consensus and differences between label-free and clean-pair maps.

The same procedure is independently applied to protected bigram codes. Bigram
alignment contributes sequence-persistence evidence. All operations remain on
opaque codes; no plaintext dictionary is constructed.

Pass 2: pseudo-clean test adaptation
------------------------------------
The first-pass model produces grouped out-of-fold risks for train and initial
risks for test. Rows below risk thresholds 0.20 and 0.25 are treated as
pseudo-clean correspondence evidence. This selection is conservative: on the
training OOF predictions, the large low-risk region is overwhelmingly negative.

A second draft/reply association is then induced jointly from recurring
pseudo-clean train and test rows. Crucially, leave-one-row-out subtraction
removes each scored row's own draft/reply cross-product before its mapping is
chosen. A row therefore cannot make arbitrary codes appear corresponding merely
because they coexist in that row.

The adaptive features include:

- top-1 and top-2 mapped-code survival;
- mapping coverage;
- survival conditional on coverage;
- confidence-weighted top-1 and top-2 survival;
- agreement across the 0.20 and 0.25 pseudo-clean thresholds;
- consensus with the original dyad-held-out alignment.

This pass expands correspondence coverage for recurring test vocabulary while
rejecting the noisier strategy of adding every test row to the map.

Final risk model
----------------
Balanced logistic regressions combine alignment with composition geometry and
sparse protected-code indicators. The initial cross-fit uses C=0.07. The final
ensemble averages C values 0.05, 0.15, and 0.30.

Geometry includes:

- total, unigram, bigram, and unique-token counts for every text field;
- unigram/bigram repetition and log word counts;
- reply-minus-draft and absolute word-count differences;
- smoothed reply/draft, draft/reply, context/draft, and context/reply ratios;
- shorter/same-or-shorter indicators;
- extent, revision, age, pause, and snapshot interactions;
- coarse draft-length, reply-length, and length-ratio bins;
- repeated-final-reply group size, draft variation, ranks, and overlap.

The sparse representation has two components:

- field-prefixed binary TF-IDF codes using C_, D_, and R_ prefixes; and
- recurring draft-code/reply-code cross-product indicators.

Ordinary TF-IDF uses min_df=3. Pair indicators use min_df=2. Both blocks are
scaled by 4. Their vocabularies and the unsupervised encoder are fitted on train
plus test; no test target exists or is used.

The final ranking starts with 95% second-pass ensemble rank and 5% adaptive
alignment rank. It then applies two deliberately small ordering refinements.
Rows with strong independent survival evidence (reverse survival above 0.40 or
pseudo-clean confidence-weighted survival of at least 0.65) receive a 0.15
demotion; this is not a hard zero, so a strong model score can still override
the rule. Repeated exact replies receive a 0.005 within-cluster rank adjustment.
The result is percentile-ranked, and empty drafts receive risk 0 exactly as
required by the published target definition.

Validation
----------
Every supervised validation split holds out complete thread_id dyads. Five
repeated 5-fold StratifiedGroupKFold partitions are used.

Current script output:

    Initial grouped CV lift: 0.932829
    Two-pass grouped CV lift: 0.951621

The ungated second pass scored approximately 0.947. Its development repeats
ranged approximately from 0.928 to 0.954, with a leave-one-dyad-out
second-stage score around 0.952. The adaptive association uses grouped OOF
predictions and leave-one-row-out co-occurrence, so neither a row's target nor
its own token pair enters its adaptive mapping.

The final survival-evidence demotion is intentionally soft and separated from
model fitting. Both evidence sources individually had zero positives above the
chosen cut in the 69-positive training set, and their union remained
positive-free. The 0.65 adaptive cut leaves margin above the largest observed
positive value (about 0.598) rather than using that boundary directly.

What improved after the 0.84 submission
---------------------------------------
- Added conservative pseudo-clean adaptation to recurring test vocabulary.
- Excluded every scored row's own co-occurrence from adaptive mapping.
- Used top-2 matching to reduce arbitrary top-1 ties between opaque codes.
- Added confidence-weighted adaptive survival and threshold consensus.
- Added a second-pass model and a small evidence-based adaptive rerank.
- Softly demoted rows with two kinds of strong draft-survival evidence.
- Added a tiny within-repeated-reply ordering adjustment.

The previously submitted model validated near 0.933 under the same standard
protocol. The complete current pipeline raises that result to approximately
0.952.

What did not help
-----------------
- Unfiltered transductive test mapping reduced grouped validation quality.
- Soft weighting of every row was weaker than conservative pseudo-clean cuts.
- Context/draft and context/reply cross-products added noise.
- Latent bigram reconstruction from inferred word constituents was weaker than
  direct association.
- CatBoost, LightGBM, random forests, ExtraTrees, and linear SVMs underperformed
  regularized logistic regression.
- Token target encoding, Naive-Bayes ratios, greedy one-to-one matching, and
  inverse-length association weighting overfit or reduced mapping quality.
