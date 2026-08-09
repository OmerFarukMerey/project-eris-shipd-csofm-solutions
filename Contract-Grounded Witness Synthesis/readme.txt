Contract-Grounded Witness Synthesis — solution notes

1. Problem classification and exact output contract

This is an NLP/code-understanding challenge under Solver Guidebook section 5.1. It is not declared a Fine-tuning or From-Scratch challenge. The challenge-specific restrictions additionally require CPU-only computation, prohibit retrieval or matching of original MBPP material, and permit episode-local static analysis or sandboxed execution of the released implementations.

The output is CSV with exactly these columns, in this order:

    episode_id,predicted_witness_suite

`episode_id` is copied exactly from test.csv; every test ID occurs exactly once and there are no missing, duplicate, unknown, or extra IDs. `predicted_witness_suite` is a JSON-encoded array containing one, two, or three distinct strings. Each string is copied byte-for-byte from that episode's witness_bank_json. The selected strings are lexicographically sorted.

The private metric is:

    canonical_accuracy = fraction whose suite uniquely certifies the hidden correct candidate
    identity_skill = max(0, (canonical_accuracy - 0.25) / 0.75)
    row_certificate_quality = 0 for a wrong/non-unique certificate, otherwise
                              min(1, hidden_minimum_suite_size / submitted_suite_size)
    exact_suite = 1 only for a correct certificate whose sorted array exactly equals gold
    score = 0.60 * identity_skill
          + 0.25 * mean(row_certificate_quality)
          + 0.15 * mean(exact_suite)

The score is clipped to [0, 1] and maximized. This has no tail, worst-group, or subgroup term.

2. Approach and trained model

The script first executes each training gold witness suite against all four released candidates in isolated, resource-limited Python subprocesses. The sole survivor supplies the training label. All 394 local training episodes yielded one survivor.

Each candidate is represented by model inputs derived from:

- unigram and bigram acceptance-card cue tokens;
- normalized Python token unigrams, bigrams, and trigrams from the implementation;
- candidate-relative mutation spans found by token alignment, including selected token, alternatives, and immediate context; and
- explicit conjunction features between contract cues and code/mutation atoms.

Opaque episode markers are removed. Local alpha-normalized variable names and quoted string literals are normalized. Numeric source tokens remain source-derived feature values. FeatureHasher provides a stateless sparse encoding; it learns no corpus vocabulary or statistics.

A pairwise LinearSVC is genuinely trained in-script. For every labeled episode it learns from the sparse difference between the correct candidate and each alternative, plus the reverse difference. At inference the trained weight vector scores all four candidates. The highest-scored candidate for which an executable certificate exists is selected. Thus the trained model, not a lookup, template, regex, assertion frequency, or fixed position, produces the candidate identity.

For the model-selected candidate, the script executes the released witness bank against all candidates in subprocess isolation. It exhaustively checks legal suites of size one, then two, then three. At the first feasible size it emits the lexicographically smallest tuple that preserves the selected candidate and rejects the other three. This decoder implements the released certificate definition; it does not decide semantic correctness.

3. Validation

The validation split is made from training data only. GroupShuffleSplit holds out complete contract groups, preventing multiple mutation episodes from the same released source/contract from crossing the split. With seed 2026, the local holdout contains 93 episodes from 36 contract groups.

The SVC regularization value is not pasted in. The script searches C over a declared grid on this holdout against the actual challenge formula. Because the executable minimum-suite decoder has quality and exactness 1 whenever identity is correct, this validation objective is 0.60 * identity_skill + 0.40 * canonical_accuracy. The local end-to-end run selected C=0.003 and reported:

    holdout canonical_accuracy: 0.720430
    projected challenge metric: 0.664516

As an additional decoder check during development, exhaustive synthesis recovered the exact released gold suite for all 394 training episodes. The final local smoke run wrote 158/158 test rows in 17.3 seconds. An independent post-run audit found valid bank membership and exactly one executable surviving candidate for every one of the 158 emitted suites; suite sizes were 138 one-assertion and 20 two-assertion certificates.

4. Leakage audit

Test-taint trace:

- Raw test row -> that row's contract, four candidate strings, and witness bank.
- The row's contract/candidates -> four row-local feature dictionaries -> stateless FeatureHasher transform -> four SVC scores.
- The row's candidates/bank -> four independent sandbox behavior vectors.
- The four scores are sorted only within the same episode. The decoder reads only that episode's four behavior vectors and its own bank.
- The selected bank strings -> that row's JSON prediction.

Feature transformation and SVC prediction are batched for efficiency, but FeatureHasher is stateless and every sparse row is independent. There is no mean, count, normalization, sorting, calibration, vocabulary construction, threshold selection, or other aggregation across test episodes. The only cross-row test operation is final schema validation of row count and duplicate IDs; it cannot alter a prediction.

There is no train+test concatenation. The SVC, label recovery, validation split, and C selection use training rows only. No tokenizer vocabulary, TF-IDF model, embedding model, scaler, encoder, PCA/SVD/NMF, clustering model, or statistic is fit on test. Test is used only for row-local transform, predict, sandboxed candidate execution, certificate decoding, and output validation.

5. Hardcoding and real-ML audit

No discovered generation pattern, phrase-to-label map, token-to-candidate map, candidate-position prior, alias arithmetic, output template, witness index, or test-distribution correction is hardcoded.

Audit of constructs that can look suspicious:

- Dictionaries hold JSON payloads, schema columns, or sparse feature values. None maps an input phrase/token to an answer.
- Regex is used only for generic lexical tokenization and identifier/number normalization before the trained model.
- If-chains enforce parsing, sandbox failure handling, suite legality, and output validity. None assigns semantic labels from input content.
- The assertion combinations are generated episode-locally from the released bank after model scoring. They cannot select a semantically correct candidate without the trained ranker.
- The early bank[0] value is a format-only crash placeholder/fallback required by the runtime contract. It is overwritten on a successful run and is not presented as a meaningful prediction.
- Performance-sensitive model regularization is searched in-script on the train holdout. There is no blend weight, probability calibration, class threshold, decode threshold, or test-tuned constant. The fixed seed, hashing capacity, validation fraction, subprocess limits, wall guard, and optimizer iteration ceiling are reproducibility/capacity/safety settings; none encodes an answer or discovered target regularity.

Strip-the-ML result: with the trained LinearSVC removed, the pipeline has no candidate scores or candidate ordering and therefore cannot choose which behavior satisfies the contract. It can only leave the deliberately format-only early placeholder. It does not produce still-usable contract-grounded answers. The trained model produces the identity decision; execution supplies behavior evidence and the exhaustive decoder serializes the smallest legal certificate for that decision.

6. Robustness and housekeeping

The script reads public_dir and submission_out from sys.argv and creates the output parent directory. It writes a schema-valid placeholder immediately after reading and validating test.csv, before sandboxing or training. Candidate execution uses isolated `python -I` subprocesses with CPU, address-space, file-size, descriptor, and parent wall-time limits. A failed candidate/assertion returns a behavior failure instead of terminating the row. Candidate order is model-ranked, so the decoder can continue to the next model-scored certifiable candidate. A final per-row schema/bank-membership audit runs before the real predictions overwrite the placeholder.

The 3000-second wall-clock guard stops launching hyperparameter/final training and moves to inference with the already trained holdout model. The local path contains exactly one Python file, solution.py; it is self-contained and imports nothing local. Random seeds are fixed. Parent-script file access is limited to reading train.csv/test.csv under public_dir and writing submission_out.

What worked: contract/code conjunctions plus candidate-relative mutation context, pairwise ranking, and group-aware regularization selection. Full code context improved validation over mutation spans alone. Exact episode-local execution makes certificate minimization deterministic and gives exact-suite credit whenever identity is correct.

What did not work: assertion-behavior text by itself was too weak to infer the controlled semantic card, and adding it to the identity feature space did not improve grouped validation. It was removed from the ranker. The assertion bank remains only as executable evidence for certificate construction, where it is exact and useful.
