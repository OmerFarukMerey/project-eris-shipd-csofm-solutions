Interleaved Loanword Stream Deconvolution
=========================================

Problem classification
----------------------

This is an NLP / sequence-to-sequence challenge governed by Solver Guidebook §5.1. It is not declared a Fine-tuning challenge. The submitted model is trained from scratch on the supplied train.csv; it loads no pretrained model or external weight.

The task is joint constrained generation, not classification or candidate-ID prediction. For each row, the solver must recover three complete protected-glyph streams whose stable left-to-right interleaving is the supplied mixed stream.

Exact submission contract
-------------------------

The CSV has exactly two columns, in this order:

    id,answer_json

There is exactly one row for every test ID, in test.csv order. IDs are copied without modification. There are no missing, duplicate, extra, unknown, or null IDs.

answer_json is a JSON string with exactly this object shape:

    {"lexeme_streams":[["g01","g02"],["g03"],["g04","g04"]]}

The value of lexeme_streams is exactly three glyph-token arrays in the displayed lexeme_slots order. Each array has the slot's target_length and unique_glyph_count. Across the three arrays, the glyph multiset equals mixed_glyph_stream, and scanning the mixed stream can reproduce the arrays while preserving each array's internal order. solution.py serializes compact JSON; csv.DictWriter performs the required CSV quoting.

Evaluation metric
-----------------

The three predicted streams and three gold streams are flattened with stream-boundary tokens. For one row,

    row_score = 1 - edit_distance(predicted_tokens, gold_tokens)
                    / max(len(predicted_tokens), len(gold_tokens), 1)

The final score is the arithmetic mean of row_score across rows. Scores for structurally valid files range from 0 to 1 and higher is better. There is no tail term, worst-group term, subgroup term, classification term, set-F1 term, LCS term, candidate-selection term, or weighted secondary metric. Structurally invalid output is rejected rather than assigned a normal low score.

Challenge-specific rules followed
---------------------------------

- The script trains a genuine task-specific neural sequence model on train.csv on every run.
- It uses only train.csv and test.csv below the supplied public_dir. There is no external WOLD lookup, dictionary, web lookup, hosted API, remote inference, source reconstruction, external dataset, synthetic training dataset, or runtime download.
- It does not use TF-IDF, BM25, fuzzy matching, fixed n-gram overlap, a Markov table, opaque-ID logic, row-order logic, manual row patches, or a rule-only main solution.
- Test data is inference-only. There is no pseudo-labeling, test-time fitting/adaptation, or test-distribution calibration.
- All task-specific training is CPU-only and occurs inside solution.py.

Approach
--------

1. Parse each labeled train packet into three individual glyph streams and their recipient, semantic field, concept code, target length, and unique-count descriptors.
2. Fit all vocabularies on training data only. Unknown validation/test values map to reserved unknown embeddings.
3. Train a conditional neural language model in both reading directions. The same observed gold sequence supplies a forward likelihood and a reverse likelihood; no new or synthetic target sequence is created.
4. At inference, scan one mixed stream from left to right. A constrained beam appends each observed token to one of three neural stream states. It permits only states that can still satisfy all displayed length and unique-count constraints. Thus every retained complete hypothesis is a valid order-preserving partition of that row's mixed tokens.
5. Generate candidates with the learned forward and reverse scorers, then rescore each complete three-stream hypothesis in both directions. A train-holdout search chooses the directional weight. The trained model's likelihood chooses the final generated streams.
6. Verify length, unique-count, multiset, and three-way interleaving invariants before writing the final file.

Model architecture and features
-------------------------------

The model is a compact conditional GRU language model trained from scratch. At every autoregressive step it receives:

- a global protected-glyph embedding;
- a recipient-specific glyph embedding;
- learned recipient, semantic-field, and concept embeddings;
- learned target-length and unique-glyph-count embeddings;
- a learned forward/reverse direction embedding.

A condition MLP initializes and accompanies the GRU state. The output layer combines a shared neural softmax projection with a learned recipient-specific output adapter and bias. This lets the recurrent dynamics share sequence knowledge while allowing protected glyphs to have recipient-specific behavior. Concept dropout, label smoothing, gradient clipping, weight decay, and early stopping regularize the small training set.

The decoder does not retrieve or instantiate training forms. It maintains one GRU state per slot in every beam hypothesis. A token assignment receives the model's learned next-token log probability, updates that slot's neural state, and is pruned only when a displayed structural constraint would become impossible. Complete candidates receive learned EOS probabilities and exact forward/reverse neural sequence scores.

Validation and in-script search
-------------------------------

The split is built from train.csv only. It is recipient-stratified and holds out about 20% of rows. Rows sharing an exact (recipient_code, concept_code) pair are unioned before splitting, so an exact protected lexeme pair cannot leak between the fitting and validation sides. On the supplied data this produced 217 fitting rows and 54 validation rows; all three streams stay together with their row.

Six model configurations are trained in-script. The grid varies fixed random seed, hidden width, condition width, learning rate, weight decay, concept dropout, and label smoothing. Validation NLL selects the checkpoint epoch within each run. The real challenge metric, computed on complete constrained validation deinterleavings, selects the model configuration. A second in-script coordinate search evaluates beam widths {80, 160, 320}, retained-candidate counts {8, 16, 24}, and forward weights from 0.0 through 1.0 in increments of 0.1. No public-leaderboard or test statistic enters this selection.

Observed local run:

    python3 solution.py ./dataset/public ./working/submission.csv

    selected epoch=12 beam=160 keep=24 forward_weight=0.6
    validation_score=0.522996
    final training epochs=12
    wrote 95 predictions; elapsed=128.9s

After selection, a fresh model with the chosen configuration is trained on all 271 labeled rows for the selected epoch count. The reported validation score is the mean normalized token-edit similarity on the untouched 54-row grouped holdout, before full-data retraining.

What worked and what did not
----------------------------

- The recipient-specific neural input embeddings and output adapters materially improved held-out deinterleaving. Shared output weights alone blurred unrelated protected alphabets.
- Joint forward/reverse training exposed both prefix and suffix regularities. Generating in both directions increased the set of plausible complete partitions; the validation metric selected their final likelihood blend.
- Enforcing the supplied constraints inside beam search was essential. It prevents a high-likelihood partial word from consuming tokens needed to make another slot valid.
- A constraint-only diagnostic produced structurally valid but lexically uninformed partitions and scored about 0.354 on the same style of holdout. It is retained only as the required early placeholder and emergency per-row safety path, not as the successful prediction method.
- Larger beams were not automatically better. The in-script metric search selected width 160 on the observed run instead of asserting a decode size.
- No retrieval, fixed n-gram table, or deterministic concept-to-form rule was retained.

Leakage audit
-------------

Training/validation data flow:

- Inner-train rows -> inner-train glyph/recipient/field/concept/size vocabularies -> HPO models.
- Held-out train rows -> transform with those already-fitted vocabularies -> validation prediction and metric only.
- All train rows -> newly fitted full-train vocabularies -> final model.

Test taint list and operations:

1. raw_test is read from test.csv and parsed into test_rows.
2. For one row, its mixed glyphs and three descriptors are transformed with full-train vocabularies. Unknown values use reserved indices.
3. That same row creates token bit masks, suffix feasibility counts, forward/reverse beam states, model log probabilities, and a candidate pool.
4. That same row's candidate scores are blended with the weight selected on the train holdout; argmax chooses that row's streams.
5. Those streams are structurally verified, serialized to answer_json, and appended for output.

Every predictive operation in steps 2-4 is per-row. No variable derived from one test row is used to predict another test row. There is no test-row mean, count, frequency, vocabulary, normalization, sorting across rows, quantile, class-balance correction, threshold fitting, pseudo-label, or calibration. The only cross-row test operations are non-predictive output bookkeeping required by the contract: preserving file order, checking ID uniqueness/completeness, checking row count, and writing the CSV.

There is no train+test concatenation. Every tokenizer/vocabulary/encoder, embedding, neural weight, regularizer selection, checkpoint epoch, beam setting, and directional weight is fit or selected using train rows only. Test is used only by transform(row), predict(row), structural verification, and serialization.

Hardcoding / real-ML audit
--------------------------

No discovered generation pattern is hardcoded. No phrase/token/concept/recipient-to-answer dictionary, ID map, manual output, row patch, form template, regex mapping, candidate ID, or learned-offline weight is present.

Dictionaries and lookup-like objects that influence execution are limited to:

- train-fitted symbol-to-index vocabularies for glyphs, recipients, fields, and concepts;
- per-row token-to-bit maps and suffix counters used only to enforce the explicitly supplied structural constraints;
- candidate dictionaries used only to deduplicate outputs already generated by the neural model;
- the in-script HPO grid, whose winning model/decode values are selected against the train-only holdout metric.

The structural checks assert only rules stated in PROBLEM.md: three streams, displayed lengths and unique counts, conserved glyph multiset, and order-preserving interleaving. They do not assert a loanword form or construction artifact. Every data-dependent/tuned constant that chooses among deinterleavings--model configuration, checkpoint epoch, beam width, candidate retention, and directional blend--is learned or searched in-script. Remaining literals define the required schema, resource-safe architecture/search ranges, batching, deterministic seeds, and wall-clock safeguards; none maps an input pattern to an answer.

Strip-the-ML result: with all trained models removed, the competitive prediction pipeline produces no learned or model-selected answer. Only the prompt-mandated schema-valid early placeholder/emergency partition remains; it has no learned lexical scorer and is overwritten after successful neural training and inference. The final successful path's complete sequences are generated and selected by trained GRU likelihoods, with rules serving only as hard validity constraints.

Cold-reviewer pass
------------------

Question: Where is the hardcoded generation pattern?

Result: none found. The only deterministic output constructor is structural_partition, used before heavy work for the mandatory valid placeholder and after a caught per-row failure. It reads only that row's explicit length, unique-count, multiset, and interleaving constraints. It contains no learned-offline form, recipient rule, concept rule, test statistic, opaque ID, or row patch. The successful run emitted no per-row fallback warnings; final answers came from the trained decoder.

Question: What produces the answer?

Result: learned recipient-conditioned forward/reverse GRU token likelihoods generate, score, and select complete stream hypotheses. Beam logic only removes structurally impossible hypotheses. There is no retrieval/template candidate source.

Question: Which test-derived variable is read across rows?

Result: none in the predictive path. test_rows and predictions are traversed across rows only to maintain submission order and perform required schema/ID/completeness checks. No cross-row value feeds a model, transform, score, threshold, blend, or row prediction.

Housekeeping and robustness audit
---------------------------------

- Random seeds and split RNG are fixed. CPU training is deterministic for the used operations.
- The platform arguments are honored exactly: public_dir is sys.argv[1], and submission_out is sys.argv[2]. No dataset/public or working/submission.csv path is embedded in solution.py.
- The script reads only train.csv and test.csv below public_dir and writes only submission_out.
- A fully structure-valid placeholder is written before importing PyTorch or starting training.
- Per-test-row model/decoder failures are caught and replaced by a verified structure-valid fallback; one noisy row cannot terminate the inference loop.
- Dataset row counts are never asserted. Malformed training rows are warned about and skipped.
- A 3000-second training guard stops further epochs, and a 2850-second launch cutoff prevents full-data retraining from starting too late. Inference and final CSV writing remain outside the training schedule.
- Before final write, every row is checked for three streams, exact lengths, exact unique counts, conserved multiset, and valid interleaving. After write, IDs, order, columns, and nonempty predictions are checked.
- The independent local audit found 95 submission rows, columns exactly [id, answer_json], 95 unique IDs, and zero structural errors.
- solution.py is the only .py file in the challenge directory. It is self-contained and imports no local module.
