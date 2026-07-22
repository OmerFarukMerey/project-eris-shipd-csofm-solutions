Phase-Conditioned Procedure Span Infilling
==========================================

Problem, domain, schema, and metric
----------------------------------
This is an NLP / sequence-to-sequence challenge under Solver Guidebook section 5.1. PROBLEM.md does not classify it as Fine-tuning or From-Scratch. Its challenge-specific CPU-only rule overrides generic accelerator availability: solution.py explicitly places every model and tensor on CPU, never uses MPS/Metal/GPU execution, runs Hugging Face offline with local-files-only weights, and performs no network access.

The UTF-8 submission has exactly these columns in this order:

    id,answer_json

There is exactly one row per unique test id, in test.csv order. answer_json is compact JSON of the exact form

    {"missing_tokens":["token_one","token_two"]}

The list length equals missing_token_count, order is significant, and every item matches [a-z][a-z0-9_]{0,31}. The specification allows either CSV column order; this solver consistently emits id then answer_json.

For true list T and prediction P, position_accuracy is the exact positional-match fraction; token_F1 is duplicate-aware multiset F1 using Counter intersection multiplicity; token_LCS is LCS(T,P)/|T|; and exact_span is one only for an identical ordered list. The official row score is

    0.55*exact_span + 0.25*position_accuracy + 0.10*token_F1 + 0.10*token_LCS

The final score is the arithmetic mean of all row scores. There is no tail, subgroup, or worst-group term.

Why the first version was replaced
----------------------------------
The first version selected its models and confidence threshold on one 601/165 GroupShuffleSplit. That split scored 0.232900 locally, but the submitted result reported by the user was 0.191. A train-only diagnosis found that the one split was optimistic: 24 of its 165 validation rows had a normalized complete step also represented in the fit partition. Those 24 rows scored about 0.859 while the remaining rows scored about 0.070, so a small change in duplicate allocation materially shifted the mean. The single validation fit also calibrated a span confidence threshold and then applied it to a larger full-data classifier whose confidence scale differed.

The replacement uses complete three-fold out-of-fold predictions for every model-selection decision. Exact pattern_context_tokens identifies a source-procedure group; GroupKFold keeps every row from that observed procedure together. The folds are 510/256, 511/255, and 511/255 train/validation rows. This removes dependence on one lucky holdout and keeps training size and confidence calibration consistent between validation and final fold models.

Approach and architecture
-------------------------
The primary model is bert-base-uncased genuinely fine-tuned in-script as a masked-token span infiller. The target gap becomes exactly missing_token_count [MASK] positions. Input order is target step, target phase and flow position, phase sequence, prerequisites, consequences, and pattern context. A hidden sibling gap has a separate marker and is never revealed.

Normalized training words that would otherwise split into WordPieces become atomic tokenizer additions. Each new embedding starts as the mean of the backbone's original WordPiece embeddings. Every BERT layer and the masked-language-model head is then optimized on the supplied labels. Candidate output scopes are built from each fold's training rows only. Epoch and scope are selected by that fold's held-out official metric. The final test token logits are a per-row probability ensemble of three independently trained fold models; every labelled row is used by two models and excluded from one model's validation predictions.

A supervised full-span SGD logistic classifier supplies complementary exact-span evidence. It directly predicts the complete ordered answer string as its learned class label from structured TF-IDF features. Three such classifiers are trained on the same outer folds. Their predicted spans, class-count-normalized confidence, and agreement are combined only when a train-only nested cross-validation search says to use them; otherwise BERT produces the answer.

Nested span tuning mirrors final inference without label leakage. For each outer validation fold, three inner models are trained only on subsets of the outer training partition and predict the untouched outer validation rows. The in-script search jointly selects SGD alpha, confidence reducer, minimum agreeing-model count, and confidence threshold against the exact official metric. The completed run selected alpha 3e-6, mean confidence, two agreeing models, and 178 span-selected OOF rows. No selected predictive constant is pasted from offline experimentation.

Feature engineering
-------------------
BERT sees explicit field boundaries, phase, flow position, target masks, sibling-gap marker, visible target-step text, phase sequence, prerequisites, consequences, and procedure context.

The span classifier receives trained TF-IDF features for output count, phase, flow position, gap index, output slot, raw and field-prefixed step tokens, the nearest eight tokens on each side tagged by distance, phase sequence, and separately prefixed prerequisite/consequence/context tokens. TF-IDF only supplies features to a trained classifier; it does not retrieve, instantiate, or emit an answer by itself.

Validation results
------------------
The completed revised run reported:

* Fold 1 BERT score: 0.182968 (epoch 8, answer scope).
* Fold 2 BERT score: 0.195820 (epoch 8, train_tokens scope).
* Fold 3 BERT score: 0.194060 (epoch 8, train_tokens scope).
* Complete three-fold neural OOF score: 0.190939.
* Nested-cross-fitted BERT/span score after in-script HPO: 0.227452.

Unlike the superseded 0.232900 number, 0.227452 uses predictions covering all 766 labelled rows from models that did not train on the row being scored. The span-selector gain is positive in every outer fold. Test model averaging itself cannot be scored without test labels; it is ordinary probability ensembling across independently trained models, not test calibration.

Leakage audit
-------------
Every learned state is fit on train.csv only. Outer-fold tokenizers, tokenizer additions, candidate vocabularies, BERT weights, TF-IDF vocabularies/IDF values, SGD classes/weights, epochs, scopes, alphas, confidence statistic, vote count, and threshold are all fit or selected without test data. Inner span models never see their outer validation targets. There is no train+test concatenation.

The test-taint trace is:

    test.csv
      -> test_frame / test_records
      -> per-row BERT serialization and per-row span feature text
      -> fold tokenizer.transform / TF-IDF transform
      -> each trained fold model's per-row logits or span probability
      -> per-row cross-model probability average and span vote
      -> missing-token list
      -> answer_json and submission.csv

All predictive test operations are per-row. Batch padding is ordinary independent batch inference. Models are averaged across models, never across test rows. No test-derived mean, count, quantile, sorting, clustering, vocabulary, threshold, calibration, pseudo-label, class balance, or fitted state exists. The only cross-row operation involving output is the mandated row-count/id/schema integrity audit, which does not change semantic predictions.

Hardcoding and strip-the-ML audit
--------------------------------
There is no id-to-answer, phrase-to-token, keyword-to-label, alias/index arithmetic, evidence lookup, exact-record answer table, regex answer rule, frequency template, or sample-submission use. Dictionaries encode parsed JSON, model metadata, candidate columns, and HPO results. Class-label maps decode a trained classifier's selected class; without classifier probabilities they select nothing. If-chains enforce schema, validation-selected routing, runtime safety, and malformed-row recovery rather than semantic mappings.

Fixed constants are the random seed, allowed general-purpose backbone, architecture/resource caps, fold count, CPU/offline controls, metric coefficients from PROBLEM.md, runtime guard, and search-space boundaries. Predictive alpha, BERT epoch, output scope, confidence reducer, vote count, and threshold are selected in-script from train-only out-of-fold scores.

The early all-"unknown" list is the non-meaningful crash placeholder required by PROMPT.md. Successful inference overwrites it. The independently verified revised output contains zero "unknown" tokens.

Strip-the-ML result: removing the fine-tuned BERT models and trained SGD classifiers leaves field serialization, TF-IDF plumbing, schema checks, and the non-meaningful crash placeholder. Those components cannot produce a usable answer. Every meaningful token or full span is selected by a trained model.

Cold-reviewer pass: after the revised end-to-end run, all 968 source lines were read afresh while assuming a hardcoded generation pattern and test leak existed. Every fit call was traced to outer- or inner-training indices. Searches found no train-test join, no test-derived fit/count/mean/quantile/sort/calibration, no row-id answer key, no sample-submission access, and no phrase/keyword/alias/evidence mapping. Test-derived values flow only through per-row transforms, trained-model predictions, same-row cross-model ensembling, serialization, and the non-predictive integrity audit.

Robustness, runtime, and output verification
-------------------------------------------
solution.py reads only public_dir and writes only submission_out; it imports no local code. It writes the complete placeholder immediately after reading test.csv. Missing required input is the only deliberate hard failure. Model-stage failures preserve a valid output, malformed rows receive valid fallback lists, and no fixed dataset-size assertion can abort a grading run.

NumPy, PyTorch, scikit-learn, fold construction, and data-loader seeds are fixed. The 3,000-second wall-clock guard reserves inference/output time and can stop later epochs while retaining the best train-only validation state.

The completed revised command

    python3 solution.py ./dataset/public ./working/submission.csv

finished in 2,495.3 seconds on CPU. Its in-script audit passed for 413 rows. An independent post-run audit confirmed exact column order, exact row count, unique and ordered ids, correct JSON root, exact required token counts, valid token regexes, and zero retained placeholder tokens.

solution.py is the only Python source file in the challenge directory.
