Statutory Outline Reconstruction
================================

Submission contract and domain
------------------------------

This is an NLP / sequence-structure task under Solver Guidebook section 5.1. It is not declared a Fine-tuning or From-Scratch task.

solution.py writes exactly two CSV columns in this order:

1. act_id: the unchanged test act identifier, exactly once per test row.
2. parents_json: a JSON integer array with exactly one entry per provision. Entry i is -1 or an earlier index in [0, i-1]. The script uses compact json.dumps output and lets pandas apply CSV quoting.

The challenge permits either column order. The produced order is always act_id,parents_json.

Metric
------

Each act's raw score is 0.55 parent-link accuracy + 0.25 depth agreement + 0.20 sibling pair-counting F1. Depth agreement uses the reference-tree span max(maximum true depth, 1). The reported act score is max(0, (raw - trivial) / (1 - trivial)), where trivial is the raw score of the all-top-level forest for that act. The final score is the unweighted mean across acts.

Model
-----

The upgraded solution treats the target as a trained sequence-labeling problem rather than making independent parent choices.

1. It loads the general-purpose microsoft/deberta-v3-small backbone only from bundled local Hugging Face files (`local_files_only=True`). It never downloads a weight. The backbone remains frozen and independently transforms each provision.
2. Each provision is represented by masked mean pooling over its contextual DeBERTa token states, plus generic length, character-class, and punctuation-count features.
3. A three-layer bidirectional GRU, trained from scratch on the released acts, contextualizes the complete provision sequence.
4. One learned head predicts each provision's depth. A second learned head predicts the depth transition between adjacent provisions from the two contextual states, their product, and absolute difference.
5. Viterbi decoding finds the highest-scoring valid depth sequence using both learned heads. Validity requires the first depth to be zero and a next depth to increase by at most one. This is the preorder-tree invariant, not a data-mined phrase/template mapping.
6. A depth sequence uniquely determines parents in preorder: a non-root provision attaches to the latest earlier provision one level shallower. All structural decisions therefore come from trained depth and transition logits.
7. The transition-logit weight is searched in-script over 0, 0.25, 0.5, 1, 2, and 4 on the train-only holdout against the exact challenge metric.
8. The three highest-scoring training snapshots are retained. The script searches whether the best one, best two, or all three should be averaged, again using only the train holdout and exact metric. Selected model logits are averaged before Viterbi decoding.

Training uses an act-balanced objective: depth and transition cross-entropies are averaged within each act before averaging acts. This aligns fitting with the final metric's equal act weighting instead of allowing long acts to dominate.

Validation
----------

A fixed-seed 15% split holds out 562 complete acts; 3,182 complete acts are used for fitting. No provision from a held-out act enters model training. Up to 26 epochs are evaluated against the exact normalized challenge metric.

Observed local selection:

- epoch 26 individual snapshot: 0.606350;
- epoch 24 individual snapshot: 0.600172;
- selected two-snapshot logit ensemble: 0.613320;
- selected transition weight: 4.0;
- complete end-to-end runtime on Apple MPS: 1,983.9 seconds.

The previous pointer baseline documented in this directory validated at 0.535276 and scored 0.56 on the leaderboard. The new train-only validation result is 0.078044 higher. A leaderboard score of 0.70 is not claimed without an actual submission result; 0.613320 is the measured evidence.

The generated local submission has 1,588 unique matching test ids, exact array lengths, no empty arrays, and only well-founded forests. Predicted root share is 0.25548 and maximum predicted depth is 4.

What worked and what did not
----------------------------

Independent pointer edges were internally inconsistent: locally they reached 0.535276 and sometimes implied depth 7. Explicitly learning depth transitions and enforcing one coherent preorder sequence produced the largest gain. Frozen DeBERTa features improved drafting-language representation without using external legal data. Equal act weighting improved the challenge metric. Snapshot logit averaging raised validation from 0.606350 to 0.613320 without additional fitted constants.

A compact LightGBM parent ranker reached only 0.401873 on the exact metric. A frozen-DeBERTa pointer reached 0.567858. A depth-only GRU reached 0.528111. A depth-transition GRU with provision-weighted loss reached 0.596691. A joint Transformer pointer/depth/transition model reached 0.579433. These experiments were not retained in the deliverable.

Leakage audit
-------------

Test-taint trace:

1. test_frame is read from the supplied test.csv.
2. Each provisions_json value is parsed independently into one test_texts entry. A same-length all-root placeholder is immediately written.
3. The already-loaded frozen tokenizer/backbone independently performs transform(test provision). Batching is compute batching only: the backbone has no fitted update and no operation reduces across test acts.
4. Each test act receives generic surface features independently. No vocabulary, scaler, cutoff, class count, architecture size, checkpoint, or decoder weight is fitted from test.
5. Selected frozen snapshots independently produce per-act depth and transition logits. Averaging occurs across models for the same act, never across test acts.
6. Viterbi and parent conversion use only one act's logits. No test-wide count, mean, sort, quantile, balance, calibration, adaptation, or threshold exists.
7. Final cross-row operations only validate output row count, id uniqueness/equality, and schema; they never modify predictions.

There is no train+test concatenation. The holdout split, model fitting, epoch selection, snapshot selection, transition weight, depth class count, and maximum-node capacity derive from train only. Test is used only for transform and predict.

Hardcoding / real-ML audit
--------------------------

- No act id, row order, phrase, keyword, statutory designator, file hash, or source lookup maps to an answer.
- TOKEN_PATTERN is only a generic feature tokenizer; it emits no labels.
- Punctuation lists and numeric bounds are generic representation/engineering constants fed into a trained model.
- The Viterbi transition constraint and depth-to-parent conversion are formal preorder-tree invariants. They cannot choose useful depths without learned model logits.
- Epoch, snapshot count, and answer-combination weight are searched in-script against the exact metric on train-only validation.
- The emergency all-root output is the challenge's declared zero-score degenerate forest and is used only to guarantee valid output after a failure.

No discovered generation pattern is hardcoded. Strip-the-ML result: removing the trained GRU and its depth/transition heads leaves only the schema-safe all-root zero-score placeholder. The decoder has no usable depth sequence and cannot reconstruct an outline. Trained model outputs produce every non-fallback parent.

Challenge-specific compliance
-----------------------------

The script does not retrieve Statutes at Large, hidden acts, external corpora, source publications, external APIs, hosted inference, or specialized legal weights. The optional backbone is general-purpose and must already be bundled. It is loaded with network access disabled. If unavailable, the valid placeholder remains rather than downloading a model.

Operational audit
-----------------

- Python, NumPy, Torch, CUDA, split, and loader seeds are fixed.
- A schema-valid placeholder is written before embedding or training.
- The 3,150-second guard stops launching training with inference margin.
- Malformed/noisy test rows and inference failures retain per-act valid fallbacks.
- solution.py reads only train.csv and test.csv under public_dir and writes only submission_out.
- solution.py is self-contained, imports no local module, and is the only Python file in the challenge directory.
