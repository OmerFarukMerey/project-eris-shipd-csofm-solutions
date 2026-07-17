Anonymized Vocal Fragment Routing solution

Approach
--------
The solution trains complementary next-fragment models, generates only duration-feasible routes, and learns a route-level reranker against the supplied composite metric. All training occurs inside solution.py on train.csv.

1. Parse notes/rests, durations, trusted prefix/suffix context, and the 16 row-local bank fragments.
2. Fit train-only melodic interval, duration-transition, event-kind, and token-transition statistics.
3. Train a LightGBM next-fragment classifier from teacher-forced route steps.
4. Train a transformer pointer network from scratch. It jointly encodes the 10 prefix events, 10 suffix events, 16 bank fragments, remaining duration, and causal decoder history, then points to an unused bank fragment.
5. Train a multiclass LightGBM route-length prior. Its inputs include span, bank/context duration summaries, rest counts, and row-local dynamic-programming counts of duration-compatible subsets by cardinality.
6. Run subset-sum-pruned beam search with both sequence models. Beams and final pools are deduplicated by event sequence rather than arbitrary alias identity, while retaining unique aliases in every output route. The pool also preserves candidates across route lengths learned from train.
7. Union both candidate pools, cross-score every route with both sequence models, and select the final route with a held-out-tuned ensemble of LightGBM regression and group-ranking objectives trained on the exact local row metric.

Model architecture / algorithm
------------------------------
Pointer network:
- 192-dimensional learned pitch, duration, segment, position, pitch-band, remaining-duration, and decoder-step embeddings.
- Four pre-normalized transformer encoder layers, four attention heads, 384-dimensional feed-forward blocks, and 0.1 dropout.
- Causal masking over decoder positions; context and bank positions remain visible.
- Learned query/key pointer projections over the 16 bank slots.
- Teacher forcing, feasible-candidate masking, pitch-shift augmentation from -3 to +3 semitones, AdamW, gradient clipping, and OneCycle learning-rate scheduling.

Tree models:
- Binary LightGBM next-fragment classifier.
- Multiclass LightGBM route-length classifier.
- Four route rerankers: L2 regression, L1 regression, LambdaRank, and XE-NDCG. Per-row score normalization makes their scales comparable; blend weights are selected only on held-out training candidates.
- Each tree model tunes its boosting length with train-only, overlap-blocked early stopping, then refits all rows available to that model for the selected number of rounds.

Decoder:
- Exact subset-sum feasibility pruning after every candidate step.
- No alias can be reused.
- Completed candidates match span_units exactly.
- Event-distinct beam states prevent anonymous duplicate aliases from crowding out musically different routes.
- A deterministic subset-sum fallback guarantees an exact-duration, unique-alias route whenever the learned beams return no route.

Feature engineering
-------------------
The step model uses candidate pitch/rest type, duration, remaining span, completion and feasibility flags, one- and two-step intervals, duration ratios, local context motifs, context pitch distances, train-only interval/duration/kind/token likelihoods, suffix-boundary compatibility, and within-row duplicate-event counts.

The route-length model uses span, bank/context duration distributions, rest counts, and log counts of row-local bank subsets that reach span_units at each possible cardinality. These subset counts are computed independently for one row during inference; they are not test-set-level statistics.

The reranker uses route length/rhythm, interval and contour summaries, rest behavior, prefix/suffix boundary compatibility, motif support, both models' total/per-step/minimum log probabilities and ranks, cross-model agreement, and the learned route-length probability.

Validation strategy and score
-----------------------------
Validation uses connected components of exact 12-event overlaps after transposition normalization, computed from training rows only. This produced 219 overlap groups, greedily balanced into two disjoint folds of 1,800 rows each.

For each outer fold, all statistics and all three generator models were fit only on the other fold. The held-out fold generated out-of-fold candidate pools. The rerankers were trained on fold-0 candidates; objective/blend selection used only fold-1 candidates.

Observed full-data baseline run:
- GBM first candidate: 0.38591 mean row score.
- Pointer-network first candidate: 0.35398.
- Candidate-pool oracle: 0.67534.
- L2 reranker selection: 0.43293 overlap-blocked mean row score.
- Public CSV score reported for that baseline: 0.44299.
- End-to-end runtime: 1,623.9 seconds on local Apple MPS for 3,600 train and 1,200 test rows.

Observed 600-row end-to-end ranking smoke test for the final ensemble:
- L2: 0.40440; L1: 0.40366; LambdaRank: 0.40950; XE-NDCG: 0.40482.
- Held-out-selected L1 + LambdaRank blend: 0.41176.
- The final full-data run was stopped before completion at the user's request; no unobserved full-data score is claimed.

What worked and what did not
----------------------------
Worked:
- Exact duration feasibility pruning substantially reduces the pointer search space without fitting on test.
- A learned route-length prior outperformed a fixed span/mean-duration estimate in grouped validation (0.468 versus 0.268 length accuracy in the development diagnostic).
- Event-distinct beams fixed severe crowding by aliases carrying identical events. On a fixed 400-row grouped holdout diagnostic, candidate oracle score increased from 0.58834 to 0.64786 and exact-event-route recall increased from 0.0750 to 0.1325.
- Combining neural and tree candidates raised the available candidate oracle well above either generator's first route; group-ranking plus regression blending improved held-out route selection over the regression baseline in the final smoke test.
- Preserving the remaining-span embedding at the pointer start token made span visible from the first decoding decision.

Did not work:
- Duration matching alone is ambiguous and was not competitive.
- A fixed expected duration per route event produced poor route-length predictions and was removed.
- Globally ranked beams without event-level deduplication spent most candidate slots on interchangeable alias variants.
- The pointer model alone underperformed the step GBM on first-route score, so it is retained as a complementary learned candidate generator rather than the sole selector.

Leakage and compliance statement
--------------------------------
Every fitted object and corpus statistic is trained from train rows only: overlap grouping, validation folds, GlobalStats, LightGBM models, transformer weights, early-stopping decisions, objective/blend selection, and reranker labels. There is no train/test concatenation, merge, pseudo-labeling, calibration, distribution estimation, vocabulary fitting, or threshold/hyperparameter selection using test rows.

Test data is used only as follows:
1. test.csv is read and each row is parsed with the fixed parser (transform).
2. For one row at a time, fixed train-fitted models compute route-length probabilities, sequence probabilities, candidate routes, and reranker predictions (transform/predict). Row-local feasibility and subset counts inspect only that row's own 16-fragment bank.
3. Test IDs are copied unchanged into the submission beside their predictions.

No statistic aggregated across test rows flows into preprocessing, features, model fitting, calibration, or route choice. Random seeds are fixed at 42 (with deterministic LightGBM settings). The script reads only train.csv and test.csv beneath public_dir and writes only the platform-supplied submission_out path. It imports no local modules and is the only .py file in the challenge directory.

Reproduction
------------
python3 solution.py <public_dir> <submission_out>

The local contract run `python3 solution.py dataset/public working/submission.csv` produced a 1,200-row CSV with exactly the columns id,predicted_route. A structural audit found zero empty routes, unknown aliases, duplicate aliases, ID/order mismatches, or span-duration mismatches.
