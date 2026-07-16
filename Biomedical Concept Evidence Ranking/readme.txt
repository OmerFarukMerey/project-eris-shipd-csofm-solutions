Biomedical Concept Evidence Ranking solution
============================================

Usage
-----

    python3 solution.py <public_dir> <submission_out>

Example:

    python3 solution.py ./dataset/public ./working/submission.csv

The script reads only documents.csv, train.csv, and test.csv from public_dir,
creates the output parent directory, writes exactly the two required submission
columns, and validates every slate before writing. No paths, IDs, labels, or
candidate orders are hardcoded; no external corpus is used.


The scoring metric (what we optimize)
-------------------------------------

Score = 0.50*NDCG@3 + 0.20*TopRelevantHit@1 + 0.10*RareConceptNDCG@3
      + 0.10*OverlapDistractorNDCG@3 + 0.05*LongQueryNDCG@3 + 0.05*WorstTrackScore

Two facts drive the design: (1) the metric is extremely top-heavy — NDCG@3 uses
exponential gains and TopRelevantHit@1 rewards a gain==2 doc at rank 1, so a
single confident top pick dominates; (2) the OverlapDistractor / WorstTrack
slates are built so a partially-relevant candidate has HIGHER raw token overlap
than the best fully-relevant one, so only semantic (non-lexical) signal separates
the classes. The exact composite is reconstructed on the training labels
(composite_metric / track_flags) and used to evaluate choices.


Approach
--------

A single, sharp CatBoost YetiRank ranker over a document-content feature set.

1. Determine the fitting corpus = documents referenced by train.csv (query or
   candidate). 23,305 of the 28,411 documents; the other ~5,100 are referenced
   only by test and are NEVER fit on.
2. build_doc_space: fit every vectorizer / IDF / SVD / PPMI embedding / NMF model
   and every corpus statistic ON THE TRAIN CORPUS, then transform ALL documents
   to per-document representations.
3. pair_features: compute row-wise query/candidate features by indexing the
   train-fitted per-document representations. Run once for train pairs and once
   for test pairs (never concatenated).
4. add_slate_features: within-slate (per-example) distractor discriminators.
5. Five query-grouped CatBoost YetiRank folds; average the fold models' raw test
   scores; sort each slate; validate.

A single model (not a blend) is used deliberately: blending or seed-averaging was
repeatedly measured to LOWER this composite because it dilutes the confident
rank-1 pick that the top-heavy metric rewards.


Model architecture / algorithm
------------------------------

CatBoostRanker, YetiRank loss, depth 6, learning rate 0.04, L2 leaf reg 5.0, up
to 600 iterations, 70-round early stopping on held-out NDCG@3, one model per
query-grouped fold. Test scores are the mean of the five fold models' raw
predictions (cross-validation bagging of one configuration, which never reorders
candidates within a slate). Extensive tuning (row/feature bagging, depth 5/7, 900
iterations, YetiRankPairwise) all lost to this configuration.


Feature engineering (106 features, all train-fit)
-------------------------------------------------

- Lexical: all-token and field (title/abstract) TF-IDF cosines; two title-boosted
  robust TF-IDF cosines; unique-token overlap, IDF-weighted overlap, Jaccard,
  directional containment; field overlaps; directional BM25 (average length from
  train docs); bigram/trigram cosine; rare-token overlap and query rare-fraction;
  query/candidate length features; per-document IDF/length statistics.
- Latent semantic (LSA): TruncatedSVD (ARPACK) cosines at several ranks and
  field-crossed title/abstract latent cosines.
- Semantic-beyond-lexical: rarity of the most-discriminative shared token; LSA/
  TF-IDF cosine minus lexical overlap residuals; rare-concept-restricted cosine
  and recall. These beat the overlap-distractor slates.
- Learned token embeddings: a PPMI token co-occurrence matrix (over TRAIN docs)
  factored by SVD into 128-d token vectors; IDF-weighted document embeddings;
  query/candidate cosines (full/title/abstract/cross) and embedding-minus-lexical
  residuals. Embeddings link related concepts sharing no literal token.
- Soft token matching (ColBERT-style MaxSim): for each query token, the max
  embedding cosine to any candidate token, aggregated (IDF-mean, coverage, and a
  soft-only residual over query tokens with no exact match), both directions and
  per field. The strongest single semantic-overlap signal.
- NMF topic distributions: 64-topic NMF; topic cosine, Jensen-Shannon / Hellinger
  / Bhattacharyya affinities, entropy, dominant-topic and cross-topic mass.
- Slate-relative: within-slate rank/z gaps between semantic and lexical standings
  (a lexical distractor ranks high on overlap but low on semantics).


Validation strategy
-------------------

Five-fold GroupKFold on query_doc_id (built from train.csv only), so validation
queries are never seen in training — mirroring the hidden test's unseen-query
shift. The exact composite metric (all six components) is recomputed on the
pooled out-of-fold predictions; overfitting is checked via the pooled-vs-per-fold
gap. Out-of-fold composite of the shipped model is ~0.645 (NDCG@3 ~0.650,
TopRelevantHit@1 ~0.659). CV is treated with caution: the hidden test is heavily
shifted, so absolute CV overstates the leaderboard and small deltas are within
CatBoost thread nondeterminism.


LEAKAGE STATEMENT
-----------------

Every transform and statistic is fit on train-referenced documents only; test
documents are used for inference (transform / predict) only.

- Fitting corpus = documents referenced by train.csv. Test-only documents are
  excluded from every fit (fit_rows in main / build_doc_space).
- Fit on train docs, transform all docs: TF-IDF vocabulary and IDF, the title and
  abstract field vectorizers, the bigram/trigram vectorizer, both LSA SVDs, the
  NMF model, and the PPMI token embedding (its co-occurrence matrix is
  binary[fit_rows] — train documents only). BM25 average length and all rare-token
  IDF quantiles are computed from train documents only.
- Test documents contribute NO fitted state, vocabulary, count, co-occurrence, or
  corpus statistic. Tokens appearing only in test are out-of-vocabulary and are
  ignored (the vectorizers drop them; the manual shared-token loop skips them).
- No pd.concat / merge / append of train and test anywhere. Train and test pair
  features are computed in two independent passes over the shared train-fitted
  document space.
- Slate-relative features (rank/z) are computed strictly within a single slate
  (one ranking example) — never pooled across different slates — so they are valid
  per-example inference for test slates.
- The model is trained on train rows and predicts test rows. No pseudo-labeling,
  no test-derived feature/threshold/hyperparameter selection. All seeds fixed.


Pre-submit audit (mandatory checklist)
--------------------------------------

- Uses of the test dataframe: (a) map its doc IDs to rows, (b) pair_features =
  transform-only feature computation from the train-fitted document space, (c)
  model.predict, (d) build/validate the submission. All are transform or predict.
- No statistic computed from test rows flows into features, preprocessing, or
  fitting: every fit uses fit_rows (train docs) or train pair rows; BM25 length,
  IDF quantiles, PPMI co-occurrence are train-doc-only.
- No train+test concatenation anywhere (verified by inspection; the only concat
  calls join feature COLUMNS within a single set).
- Random seeds fixed (SEED). The script reads only public_dir and writes only
  submission_out.


What worked and what did not
----------------------------

Worked: a single sharp YetiRank ranker; query-grouped folds; soft token matching
(MaxSim) and learned token embeddings as semantic-overlap signals for the
distractor track; semantic-minus-lexical and rare-shared-IDF features;
reconstructing the exact composite metric to tune against.

Did not work / rejected: blending or seed-averaging (dilutes rank-1); a gain==2
stacked classifier signal (hurts); row/feature bagging and alternate ranker
configs (all below the base). Compared to an earlier version that fit its
vectorizers/embeddings on the full document corpus (train+test documents), fitting
on train documents only costs ~0.005 out-of-fold composite. That cost is accepted:
per the challenge's leakage rules, test data is inference-only, so generalization
and a valid submission beat a marginally higher but non-compliant score.


Reproducibility
---------------

All stochastic components have fixed seeds and GroupKFold is deterministic. Sparse
matrices back the token features; only the compact 106-column pair tables and the
dense document embeddings are materialized. A full run takes a few minutes on a
multi-core CPU.
