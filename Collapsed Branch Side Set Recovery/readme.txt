Collapsed Branch Side Set Recovery - Solution
=============================================

PROBLEM
-------
Each row is a collapsed local neighbourhood of a phylogenetic tree.  One
internal branch was contracted into the placeholder node X, merging the two
sides of that branch into a single polytomy.  Endpoint taxa are anonymised as
row-local tokens (u00, u01, ...).  Evidence is a SPARSE set of endpoint pairs,
each with an integer hop count (topological distance in the original tree) and
an optional within-row branch-length rank (0..7, or null).  For every row we
must output the EXACT set of endpoint tokens on the anchor's side of the hidden
branch.  Scoring is exact set-match accuracy averaged over rows.

The true side is a valid bipartition part: it contains the anchor, has >= 2
tokens and <= floor(n/2) of the n incident tokens.


APPROACH  (constrained set search + TWO learned models)
-------------------------------------------------------
1. Candidate enumeration.  For each row we enumerate every VALID candidate side
   (contains the anchor, size 2..floor(n/2)).  The true side is always one of
   them, so the task reduces to scoring/ranking candidates.  Enumeration is
   capped at 4000 candidates for the rare very large rows (n up to 16).

2. Per-EDGE model (LightGBM).  A classifier trained on the observed evidence
   pairs predicts whether a pair is same-side or crossing, from its hops/rank
   AND its position RELATIVE to the other pairs in its row (hops minus row-mean,
   the pair's hop-rank within the row, hops/row-mean, is-row-min/max, etc.).
   The hidden branch adds a roughly constant offset to every crossing pair, so
   whether a pair crosses is best judged relative to the row's own scale.  This
   lifts same/cross AUC from 0.955 (a plain frequency table) to 0.983.  The
   per-pair log-odds it outputs are the main evidence signal fed to step 3.

3. Per-CANDIDATE model (LightGBM, binary).  For each candidate we build ~79
   relational, permutation-invariant features:
   - the per-edge log-odds aggregated over the three pair classes the candidate
     cut induces (within-side / within-complement / across), plus their
     ordering consistency and separation margin;
   - a generative log-likelihood-ratio and a size prior P(|side| | n);
   - hop/rank distribution stats for each pair class; within-row hop/rank
     ordering consistency (a correct bipartition ranks every same-side pair
     below every crossing pair); "violation" counts (e.g. a cherry, hops==2,
     split across the cut); anchor-incident-edge consistency.
   The true side is label 1, sampled competitors label 0 (120 random negatives
   per row).  8 seeds are bagged; the argmax candidate is predicted.

   The per-edge log-odds used to featurise TRAINING rows are produced
   out-of-fold (two-way split) so the candidate model never trains on in-sample
   edge scores; test rows use an edge model fit on all of train.

MODEL / ALGORITHM
-----------------
Two LightGBM models.  Edge model: 400 trees, lr 0.03, num_leaves 31.  Candidate
model: 700 trees, lr 0.03, num_leaves 31, min_child 100, L2=5, L1=2,
subsample/colsample 0.8/0.7 - deliberately REGULARISED because the effective
sample size is only ~3,442 independent rows (the ~121 candidates per row are
highly correlated).  8-seed bagging stabilises the near-tied argmax.  Runtime
~5.5 min end-to-end on CPU.


VALIDATION STRATEGY
-------------------
5-fold cross-validation over the 3,442 training rows (fixed shuffle seed).  Per
fold, the generative tables, size prior and LightGBM models are fit on the 4
training folds only and evaluated by exact set-match on the held-out fold.

CV (exact set-match accuracy) ladder:
   - shortest-path oracle-size baseline ......... 0.08  (evidence graph is a
                                                         disconnected matching)
   - generative naive-Bayes partition search .... 0.37
   - LightGBM ranker (base features) ............ 0.44
   - LightGBM ranker (enriched + regularised) ... 0.46
   - + per-edge model (row-relative features) ... 0.465 (final; +0.004 over the
                                                   no-edge model on identical
                                                   folds/seeds, +0.008 in a
                                                   controlled 3-fold A/B)

WHY THE PUBLIC LB (0.4305) IS BELOW CV, AND THE CEILING
-------------------------------------------------------
This was investigated carefully:
 * Adversarial validation (classifier trained to tell train from test rows)
   gives AUC = 0.50 - train and test are the SAME distribution, no shift.  So
   the 5-fold CV (~0.46) should reflect true accuracy on the FULL test set.  The
   public LB is only a subset of test; 0.43 is ~1.4 sigma below 0.46 for a
   subset of that size, i.e. ordinary sampling variance.  (Per the guidebook,
   the public LB is a soft signal, not the final measure.)
 * The task has a genuine ceiling around 0.46-0.52.  Because the evidence is a
   sparse (often disconnected) near-matching, MANY bipartitions induce the same
   same/cross labelling on the observed pairs and therefore the same score - the
   true side is only the UNIQUE evidence-best for 12% of rows, and is merely
   tied-for-best for 63%.  The dominant residual is the "orientation" tie: for a
   crossing pair (a,b) the evidence cannot say which of a,b sits on the anchor's
   side, and we verified there is NO token-level signal to break it
   (anchor-side vs other-side tokens are statistically identical in degree and
   edge-hops).  Even an oracle that is told the correct side-SIZE reaches only
   0.52.

LEAKAGE STATEMENT
-----------------
Every fitted object - the P(hops,rank | same/cross) tables, the size prior, and
the LightGBM models - is fit on TRAINING ROWS ONLY.  Test rows are used only to
(a) enumerate their own candidate sides and (b) receive per-row predictions
(inference).  No train+test concatenation, no statistic computed over test rows,
no test-derived thresholds/hyperparameters, no pseudo-labelling.  Random seeds
fixed.  The script reads only public_dir and writes only the submission CSV.

WHAT WORKED
-----------
- Reframing as constrained candidate-set search (true side always enumerated).
- Joint (hops, length_rank) generative LLR + size prior as the core signal;
  length_rank alone is very strong (rank 0 -> 96% same-side, rank 7 -> 9%).
- Splitting evidence pairs into within-side / within-complement / across-cut and
  adding within-row ordering-consistency features.
- Regularising the GBM and bagging seeds (reduce overfit on ~3.4k effective
  rows; stabilise the near-tied argmax).

WHAT DID NOT HELP (tested, dropped)
-----------------------------------
- Shortest-path anchor closeness (graph is disconnected).
- Using ALL candidates as negatives (easy negatives swamp the boundary).
- Hard-negative mining (one-token-off candidates).
- Explicit S-vs-complement tightness asymmetry features.
- A dedicated row-level size classifier (0.75 vs the model's implicit 0.82).
- Blending the generative score onto the LGB score (monotonically hurt).
- Lexicographic "evidence-first, LGB-for-ties" selection (hurt badly, 0.37):
  the LGB's holistic balancing of evidence and size is already optimal.
- Ranking objective (lambdarank) - slightly worse than binary.
