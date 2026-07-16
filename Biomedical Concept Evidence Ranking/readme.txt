Biomedical Concept Evidence Ranking solution
============================================

Usage
-----

    python3 solution.py <public_dir> <submission_out>

Example:

    python3 solution.py ./dataset/public ./working/submission.csv

The script reads only documents.csv, train.csv, and test.csv from public_dir. It
creates the output directory when needed and writes exactly the two required
submission columns, then validates every slate and candidate set before writing.
No paths, document IDs, slate IDs, labels, or candidate orders are hardcoded, and
no external corpus or recovered biomedical identifier is used.


The scoring metric (what we actually optimize)
----------------------------------------------

Score = 0.50*NDCG@3 + 0.20*TopRelevantHit@1 + 0.10*RareConceptNDCG@3
      + 0.10*OverlapDistractorNDCG@3 + 0.05*LongQueryNDCG@3 + 0.05*WorstTrackScore

Two structural facts drive every design choice:

1. TopRelevantHit@1 (20%) rewards putting a gain==2 ("completely relevant") doc
   at rank 1, and NDCG@3 uses exponential gains (2^g - 1), so a gain-2 doc is
   worth 3x a gain-1 doc and rank 1 dominates. The metric is extremely
   top-heavy: a single confident rank-1 pick matters far more than the tail.

2. OverlapDistractor + WorstTrack (15%) are, by construction, slates where a
   partially-relevant (gain==1) candidate has HIGHER raw token overlap with the
   query than the best gain==2 candidate. On these slates raw lexical overlap
   points at the wrong answer; only semantic signal separates the classes.

The metric is reconstructed exactly on the training labels (composite_metric /
track_flags in solution.py) so model and feature choices are tuned against the
real objective, not a proxy. OverlapDistractor and LongQuery flags are computed
exactly; RareConcept is approximated by the top quartile of queries by rare-token
fraction.


Approach
--------

A single, sharp CatBoost YetiRank ranker over an augmented feature set. Steps:

1. Index all public documents by doc_id.
2. Build 157 pair features: 91 lexical / latent-semantic / length / rarity
   (build_features), 11 "semantic-beyond-lexical" (build_semantic_features),
   9 learned token-embedding cosines (build_embedding_features), 17 NMF
   topic-distribution features (build_nmf_features), 8 slate-relative distractor
   discriminators (build_slate_features), 12 soft token-matching / MaxSim
   features (build_softmatch_features), and 9 document-graph / centrality
   features (build_graph_features).
3. Split training data into five folds by query_doc_id (GroupKFold). Both public
   slates of a training query stay in the same fold, so no query leaks across
   the fold boundary and validation mimics the hidden "unseen query" regime.
4. Train one CatBoostRanker per fold (YetiRank, NDCG@3 early stopping). Produce
   out-of-fold predictions for validation and average the five fold models' raw
   test scores for the submission.
5. Sort each slate by score and validate that its 12 candidates occur once each.

Deliberately NOT done: model blending, seed bagging, or mixing in a lexical
retrieval anchor / gain-2 classifier. See "What worked and what did not".


Model architecture / ranking algorithm
--------------------------------------

CatBoostRanker, loss YetiRank, depth 6, learning rate 0.04, L2 leaf reg 5.0,
random strength 0.5, up to 600 iterations, 70-round early stopping on held-out
NDCG@3, one model per query-grouped fold. Test scores are the mean of the five
fold models' raw outputs (standard cross-validation bagging of a single
configuration, which never reorders candidates within a slate).

Only document-derived signals are used. An out-of-fold candidate-history target
feature is available behind USE_CANDIDATE_HISTORY=1 but is OFF by default: test
queries have 0% overlap with training queries and test candidates only ~5%
overlap, so candidate history has ~30% coverage in cross-validation but ~5% at
test — it inflates CV without transferring.


Feature engineering
-------------------

Base features (build_features, 91):
- TF-IDF cosine over all tokens; separate title-title, title-abstract,
  abstract-title, abstract-abstract field cosines.
- Two title-boosted (x2, x4) robust TF-IDF cosines.
- Unique-token intersection, IDF-weighted intersection, Jaccard, directional
  containment, field overlaps, directional BM25.
- Bigram/trigram cosine and overlap.
- Truncated-SVD (ARPACK, 192-dim) latent cosines at several ranks plus
  field-crossed latent cosines.
- Per-document statistics: original vs actual length, truncation flag, unique
  ratio, IDF mean / RMS / max / p90, title vs abstract IDF, token-number
  summaries, title-abstract self-overlap.

Semantic-beyond-lexical features (build_semantic_features, 11) — added to lift
the OverlapDistractor and long-query tracks that raw overlap cannot:
- max_shared_idf, top3_shared_idf: the IDF (rarity) of the single most / top-3
  most discriminative SHARED tokens. A distractor shares only common tokens;
  true evidence shares a rare concept. Strongest new signals.
- allcos_minus_jacc, sem_minus_jacc, sem_minus_qcontain, tt_sem_minus_jacc:
  semantic similarity (TF-IDF or 160-dim ARPACK LSA cosine) MINUS lexical
  overlap (Jaccard / containment). Positive when two documents are topically
  close despite little literal token sharing — the distractor-beating residual.
- lsa160_cos, lsa_qtitle_cabs, lsa_qabs_ctitle: full-text and cross-field
  (query-title vs candidate-abstract and vice-versa) latent-topic cosines.
- rare_cos_r75, rare_recall_r90: retrieval restricted to rare concepts (top 25%
  / 10% IDF), i.e. matching on the concepts that actually discriminate.

Measured under matched conditions this augmentation lifts the OverlapDistractor
track by about +0.004 and NDCG@3 by about +0.002 on a single ranker.

Learned token embeddings (build_embedding_features, 9) — the largest single win:
- A token co-occurrence matrix (whole-document) is turned into PPMI and factored
  by randomized SVD into 128-dim token embeddings (built from scratch; no gensim).
  Document embeddings are IDF-weighted, L2-normalized means per field.
- Features: query-vs-candidate embedding cosine (full / title / abstract /
  cross-field) and embedding-minus-lexical residuals. Embeddings link related
  biomedical concepts that share no literal token, so they lift every track,
  especially the hard ones (rare +0.019, distractor/worst +0.016, and NDCG@3 and
  TopRelevantHit@1 by +0.011 each on a single ranker).

NMF topic model (build_nmf_features, 17) — a non-negative semantic view
complementary to LSA:
- A 64-topic NMF is fit on the combined-text TF-IDF. Features: topic cosine,
  Jensen-Shannon / Hellinger / Bhattacharyya affinities between the two topic
  distributions, per-document topic entropy, shared-dominant-topic indicators,
  directional cross-topic mass, and topic-minus-lexical residuals.
- Chiefly sharpens rank-1 quality: TopRelevantHit@1 +0.013 and NDCG@3 +0.004.

Slate-relative distractor discriminators (build_slate_features, 8) — the signed
gap between a candidate's WITHIN-SLATE rank on semantic signals (embedding /
topic / TF-IDF / LSA cosine) and its within-slate rank on lexical signals
(Jaccard / BM25 / overlap). A lexical distractor ranks high on token overlap but
low on semantics, so the gap has the opposite sign from genuine evidence. These
are computed from the already-built feature columns plus slate grouping (no
labels); +0.0023 composite on the single ranker in matched conditions.

Soft token matching / MaxSim (build_softmatch_features, 12) — a ColBERT-style
semantic overlap that is the strongest new signal of the latest round:
- Token embeddings (the same PPMI + SVD as the embedding block) are L2-normalized;
  each document is restricted to its top-IDF tokens per field. For each query
  token, take the MAXIMUM cosine to any candidate token, then aggregate as
  IDF-weighted mean, mean, and coverage@0.5 (both directions, per field), plus a
  "soft-only" residual over query tokens that have NO exact candidate match.
- Unlike a pooled document-vector cosine, MaxSim rewards a query concept that has
  a close semantic neighbour among the candidate's concepts even with zero
  literal overlap, so it directly beats lexical distractors. It lifts every track
  (+0.004 composite, and rare / distractor / long / top-1 all positive) with no
  overfit (per-fold gap ~0.0001) and is fully label-free.

Document-graph / centrality (build_graph_features, 9) — a structural axis
distinct from all the pairwise cosines. A k-NN document-similarity graph is built
over an LSA space, and the block emits candidate quality priors (PageRank,
hubness, clustering), query-candidate proximity (mutual-kNN, common neighbours,
Adamic-Adar), and the candidate's GLOBAL similarity-rank to the query among all
28k documents. That global rank is a query-comparable calibration that should
help the unseen test queries specifically. Only the non-similarity-overlap
columns are kept: the raw neighbour-similarity / graph-cosine columns duplicated
the embedding cosines and pulled rank-1 down, whereas the pruned 9-feature set
lifts the OverlapDistractor / WorstTrack by +0.007 and nudges NDCG@3 and
TopRelevantHit@1 up in matched conditions (+0.0023 composite). Label-free.


Validation strategy
-------------------

Five-fold GroupKFold on query_doc_id, so validation queries are never seen in
training — the same shift the hidden test imposes. The exact composite metric
(all six components) is recomputed on the pooled out-of-fold predictions, and
overfitting is checked by comparing the pooled composite against the per-fold
mean (gap ~0.0001-0.0002). Out-of-fold composite of the shipped 148-feature
ranker is ~0.651 (NDCG@3 0.653, TopRelevantHit@1 0.674). Leaderboard progression
of the feature rounds: 102 feats 0.3501, 128 feats (embeddings+NMF) 0.3524, 136
feats (+ slate-relative), 148 feats (+ soft token matching) 0.3556, 157 feats
(+ graph/centrality) shipped here. The
transfer ratio is shrinking (the 128-feature round gained +0.0147 OOF but only
+0.0023 on the hidden test), so recent rounds prioritise GENUINELY NEW,
shift-targeting signal (soft semantic matching, graph structure) over more
variations on existing cosines, which increasingly only overfit the public split.

Cross-validation is treated with suspicion: the hidden test is heavily shifted
(0% query overlap, distractor / rare-concept / long-query tracks), so CV
overstates absolute score and differences below ~0.003 are within run-to-run
CatBoost threading nondeterminism. Choices are therefore made on robustness
principles, not on chasing sub-0.003 CV deltas.


How the final design was chosen (multi-agent search)
----------------------------------------------------

A parallel multi-agent workflow trained seven diverse base learners (CatBoost /
LightGBM / XGBoost rankers and gain==2 / gain>=1 classifiers), ran two explorer
agents on the hard tracks, greedily searched the composite-optimal ensemble, and
adversarially verified the winner. Findings that shaped the shipped model:

- Every diverse base learner scored below the single CatBoost YetiRank ranker on
  composite (LightGBM ranker 0.622, XGBoost 0.620, classifiers 0.60-0.61 vs
  0.631). They add nothing this metric rewards.
- Blending / seed-averaging LOWERS the composite. z-score-averaging dilutes the
  confident rank-1 pick, which the 20% TopRelevantHit@1 and top-heavy NDCG@3
  punish. The ensemble search converged to a single model; adversarial
  verification independently recommended one model.
- The only transferable gain was the semantic-beyond-lexical feature block,
  which improves the distractor / long tracks on a single ranker. That block is
  what was integrated.

A second workflow then expanded features (five parallel families: token
embeddings, slate-relative rank gaps, NMF topics, deep rare-concept, long-query
/ passage). All five helped, but forward-selection + adversarial verification
kept only the two whose gains survived stacking without overfitting: token
embeddings (+0.0122 composite alone, biggest hard-track lift) and NMF topics
(+0.0057, mostly TopRelevantHit@1). Combined they take the OOF composite from
0.6287 to 0.6434 with a per-fold overfit gap of 0.0002. This round was guided by
a measured fact: on the previous submission a feature block that looked like OOF
noise still gained +0.0125 on the hidden leaderboard, i.e. hard-track feature
quality transfers far better than the pooled OOF composite suggests.


What worked and what did not
----------------------------

Worked:
- A single sharp YetiRank ranker over document-content features.
- Query-grouped folds that reproduce the unseen-query shift.
- Learned token embeddings (PPMI + SVD) — the biggest lever, lifting every track.
- NMF topic-distribution affinities, mainly sharpening rank-1 (TopRelevantHit@1).
- Slate-relative semantic-vs-lexical rank gaps (built from embedding/topic vs
  lexical standings within a slate).
- Soft token matching (ColBERT-style MaxSim in embedding space) — the strongest
  new signal of the latest round; lifts every track, fully unsupervised.
- Document-graph centrality / proximity priors (pruned to the structural + global-
  rank features), a new axis that lifts the distractor track by +0.007.
- Semantic-minus-lexical and rare-shared-IDF features for the distractor track.
- Reconstructing the exact composite metric to tune against the real objective.
- ARPACK SVD to avoid the float32 randomized-SVD overflow (which can emit NaNs
  and fail the finite-value guard on some sklearn builds).

Did not work / rejected:
- Blending the ranker with a robust lexical anchor, a gain-2 probability model,
  or other model families: all lowered the composite by diluting rank 1.
- Seed-averaging multiple CatBoost rankers: same dilution, lower tophit.
- A dedicated gain==2 classifier to lift TopRelevantHit@1: could not beat the
  ranker's own top score (0.654); no available signal resolves the last gain-2
  pick better than the ranker itself.
- Out-of-fold candidate-history: strong in CV, ~5% test coverage, does not
  transfer (kept behind an env flag, off by default).
- LightGBM / XGBoost rankers as primary models: individually weaker and, when
  blended, dilutive.

Rejected (helped OOF alone but dropped in forward-selection/ablation, or made
things worse): deep rare-concept ladder and long-query / passage features (did
not survive stacking on embeddings + NMF); higher-dimensional (256) token
embeddings (redundant with the 128-dim block, -0.0024); and every ranker
hyperparameter variation tried (row/feature bagging, MVS, depth 5/7, 900
iterations, YetiRankPairwise) — all lost to the base config, because row/feature
subsampling dilutes the confident rank-1 pick this metric rewards. The base
YetiRank config (depth 6, lr 0.04, l2 5, 600 iters) is a firm local optimum on
the model side; remaining headroom is in features, not the model.

New-signal round (5 fresh families, forward-selected): soft token matching won
(+0.004, all tracks up). Windowed-context embeddings and alternative
factorizations each helped ~+0.002 alone but overlap the existing embedding
cosines and did not survive stacking. A leakage-safe STACKED gain-2/gain-1 meta
block (strict out-of-fold) clearly HURT (-0.013), reconfirming that gain-2
classification signal does not help this ranker even as slate-relative features.

Graph follow-up: the graph/centrality family tied soft matching (+0.004) but the
FULL block traded TopRelevantHit@1 (-0.008) and rare (-0.005) for the distractor
track, because its raw neighbour-similarity / graph-cosine columns duplicate the
embedding cosines and add noise at rank 1. Pruning to only the structural + quality
+ global-rank columns (build_graph_features, 9) removes that regression: it keeps
the +0.007 distractor/worst lift while nudging NDCG@3 and TopRelevantHit@1 up
(+0.0023 composite in matched conditions), so the lean subset is shipped.

Leaderboard progression: lexical-heavy ~0.33 -> pure supervised ranker 0.3376
-> single ranker + semantic-beyond-lexical features 0.3501. The consistent
lesson is that more/better supervised semantic features transfer, so this round
pushed feature quality (embeddings, topics) rather than model complexity.


Reproducibility and resource notes
----------------------------------

Every stochastic component has a fixed seed and folds are deterministic
(GroupKFold, no shuffle). Feature fitting uses only the public document
collection; the only labels used are the training relevance gains inside the
supervised folds. Sparse matrices back the token features; only the compact
102-column pair table and the 160/192-dim document representations are dense.
A full run (feature build + five folds) takes a few minutes on a multi-core CPU.
