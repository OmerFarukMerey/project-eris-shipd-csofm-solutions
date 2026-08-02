Catalan Administrative Discourse Operator Reconstruction
========================================================

1. Problem contract
-------------------

Guidebook domain: NLP / sequence-to-sequence (§5.1). PROBLEM.md does not classify this as a
Fine-tuning challenge and does not require a pretrained backbone. The solution therefore trains a
compact neural text encoder from scratch on the supplied training set, alongside two trained
linear text classifiers. It does not download weights, access the internet, or use external data.

Exact submission schema, including order:

    sample_id,operator_path

- sample_id is one exact test.csv ID. Every test ID appears exactly once; no ID may be empty or
  duplicated, and the submitted ID set must equal the test ID set.
- operator_path is one plain string containing exactly six tokens separated by single spaces.
  Every token must be one of O00 through O25. Repeats are valid. There are no brackets, commas, or
  JSON syntax in this field.
- The file is UTF-8 CSV. A malformed row scores zero, while malformed columns or IDs reject the
  entire submission.

The exact mean-row evaluation metric is:

    0.45 * A_position
  + 0.20 * F_edge
  + 0.15 * S_edit
  + 0.15 * A_matrix
  + 0.05 * I_exact

A_position is six-position token accuracy. F_edge is multiset F1 over the five directed adjacent
operator pairs; absolute edge position is ignored, direction and repeated-pair multiplicity are
retained. S_edit is 1 - token_Levenshtein_distance/6. A_matrix is accuracy over the four flattened
cells of the predicted ordered matrix product. I_exact is complete six-token-path accuracy. There
is no hidden tail, worst-group, or subgroup term.

Challenge-specific restrictions followed: one NVIDIA GPU, at most 10 GB VRAM, expected runtime at
most 1.5 hours, supplied challenge files only, no external corpora/source copies/reverse lookup,
no use of IDs or row order as features, and no internet/package installation. The public
operator_algebra.json procedure is used exactly: SHA-256-derived invertible matrices modulo 17,
identity start, and ordered right multiplication.

2. Approach
-----------

All learning starts from raw train.csv on every invocation.

Parsing and representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each row is parsed into its six G1..G6 sections. For every gap the model retains up to the supplied
20-token left window and 28-token right window around MASK. Public w[0-9a-f]{4} buckets map to
embedding rows by their specified hexadecimal bucket ID; this is input parsing, not a token-to-
operator mapping. Malformed input sections become empty context rather than terminating an entire
run.

Three independently trained model families provide class logits for all six positions:

1. Focused phrase text classifier. A train-fitted TF-IDF transform over local word unigrams,
   bigrams, and trigrams feeds multinomial logistic regression. Its left/right context window is
   selected in-script on the calibration fold; every transform and classifier is fitted on fold
   training rows only.
2. Boundary-aware text classifier. Deterministic feature extraction marks left/right side,
   boundary-distance bins, gap position, and local one-to-three-token phrases. A separate
   train-fitted TF-IDF transform feeds a separate multinomial logistic regression. These are
   features into a trained classifier; no phrase or n-gram table is instantiated as an answer.
3. Compact neural sequence encoder. A 96-dimensional learned bucket embedding and learned relative
   positions feed a shared two-layer local Transformer. The MASK state from each gap is augmented
   by a learned G1..G6 position embedding, then all six states pass through a two-layer row
   Transformer before a learned 26-class head. The model has about 1.17 million parameters and is
   trained from scratch with cross-entropy, label smoothing, AdamW, gradient clipping, and
   train-fold early stopping. Two fixed, reproducible training seeds are averaged per fold.

Five GroupKFold models are trained. Rows sharing any exact participant-visible gap section are
connected into one group, so such overlaps never cross a training/validation boundary. Test logits
are averaged across fold models at the same row and class; no operation averages or normalizes
across test rows.

Metric-aware model and decoder selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The logistic-regression C value and focused left/right context window are selected in-script on a
train-only calibration fold, scored with the complete challenge metric after constrained decoding.
The search evaluates five regularization values and five context-window configurations, then
re-searches regularization for the selected window. On the observed run it selected C=3.87298 and
12 left plus 20 right tokens. Neither result is pasted into solution.py; both are rediscovered on
every invocation.

Out-of-fold logits determine each model family's scale. The script searches all 0.1-resolution
simplex blends plus independent argmax, exact maximum-score decoding, and posterior-marginal
decoding against the complete stated metric. Marginal-decoder temperature is also searched
in-script. The observed run selected focused TF-IDF 0.30, boundary TF-IDF 0.30, neural Transformer
0.40, posterior-marginal decoding, and temperature 0.303143. These are train-only search results,
not hardcoded output rules.

For constrained decoding, each row's trained-model logits score paths. A meet-in-the-middle
procedure joins 26^3 prefixes and suffixes whose ordered composition equals that row's
boundary_matrix. Maximum-score mode selects the highest-logit legal path. Marginal mode enumerates
the roughly four thousand legal paths, forms a temperature-scaled posterior from the trained
logits, computes per-position operator marginals, and returns the legal path with greatest expected
position agreement. The algebra only restricts the feasible set; trained-model probabilities rank
and marginalize it. There is no context-free path lookup or template emission.

3. Validation
-------------

Strategy: five-fold out-of-fold validation on train.csv only, grouped by connected components of
exact shared gap sections. This produced 2,065 groups and out-of-fold predictions for all 2,108
rows. Neural checkpoint selection uses only the relevant fold's held-out training rows. The hidden
source-family metadata used to create the official train/test split is not participant-visible;
exact-gap grouping enforces the strongest directly observable overlap restriction without
inventing test-derived groups.

Observed full out-of-fold score from the required local end-to-end command:

    0.557557

This is the exact challenge metric, including position, directed-edge multiset F1, token edit,
matrix-cell, and exact-path terms. This improves the previous pipeline's observed 0.549668 OOF
score by 0.007889. The selected marginal decoder maintained full matrix agreement on every scored
out-of-fold row.

The improved full local command completed successfully on Apple MPS in 643.4 seconds, trained all
five folds and ten neural models, and wrote 702 predictions. The submitted A10G is the intended
faster GPU. The model is small relative to the 10 GB limit. A 3,000-second wall-clock guard
prevents new fold/seed training from starting and proceeds to inference with completed models.

4. Leakage audit
----------------

Training data flow
~~~~~~~~~~~~~~~~~~

- Exact-gap groups are constructed only from train context_sequence.
- Every TF-IDF vocabulary, document frequency, IDF, normalization configuration, and feature
  selection is fitted on a fold's training rows. validation/test call transform only.
- Logistic classifiers and neural weights are fitted only from training rows and targets.
- Early stopping, C and context-window search, family logit scales, blend weights, decoder mode,
  and marginal temperature use only held-out or out-of-fold train predictions and train labels.
- No train/test concatenation, append, merge, joint vocabulary, joint count, joint embedding fit,
  clustering, pseudo-labeling, or test-time adaptation exists.

Test-taint trace
~~~~~~~~~~~~~~~~

Raw test.csv produces the following tainted values:

1. test.sample_id -> early placeholder IDs -> final submission IDs. The only cross-row operations
   are required completeness, duplicate, and exact-ID-set integrity checks.
2. test.context_sequence -> per-row parsed sections -> per-row raw and boundary feature strings ->
   train-fitted vectorizer.transform -> classifier decision scores.
3. test.context_sequence -> per-row 6x49 bucket tensor -> trained Transformer predict -> neural
   logits.
4. Each family's test logits -> equal averaging across MODELS at the same row/class -> division by
   a scale learned from train OOF logits -> searched family-weighted logits for that same row.
5. test.constraint_seed plus test.boundary_matrix -> that row's public operator matrices and legal
   path set -> trained-logit maximum-score or posterior-marginal decode for that same row.
6. Per-row decoded token IDs -> per-row operator_path string -> output CSV.

No tainted value is reduced across test rows. There is no test mean/std, bincount, argmax-count,
quantile, sort, class-balance adjustment, calibration, threshold fitting, vocabulary fitting, or
feature selection. Batch transform/predict is only an execution container: TF-IDF uses train IDF
and row-local L2 normalization, and Transformer inference has no batch normalization. Arrays named
*_sum accumulate model predictions pointwise for each row; they sum across models, never rows.

5. Hardcoding / real-ML audit
-----------------------------

No discovered generation pattern is hardcoded. Every answer-bearing parameter is trained or
searched in-script. Remaining literals are public protocol fields, model architecture,
reproducibility, resource limits, or robustness settings; none asserts an operator label from an
input phrase.

Dictionaries, mappings, conditions, templates, and constants influencing the pipeline:

- token_to_id is built from operator_algebra.json's allowed-token list. It is a protocol encoder,
  not a phrase-to-label mapping.
- The finite-field matrix mapping is recomputed from each row's public seed and the published
  SHA-256 procedure. It supplies a legal constraint; trained logits rank the legal paths.
- The exact-section owner dictionary is train-only validation grouping and never predicts a label.
- TF-IDF vocabularies/IDFs, logistic coefficients/intercepts, neural embeddings/attention/heads,
  checkpoint epochs, context window, logit scales, C, family weights, decoder mode, and posterior
  temperature are LEARNED or SEARCHED.
- LEFT/MASK/RIGHT/NEXT_GAP parsing and w-code decoding follow the public input grammar. Side and
  local phrase features enter trained classifiers; no feature maps directly to O00..O25.
- Device, error, wall-clock, fold, and per-row validity if-chains control execution only. They do
  not contain a phrase/token/operator family mapping.
- On a constrained-decoder exception, fallback is that row's trained-model argmax, not a constant
  class.
- The only asserted output token is operator_algebra.json's first allowed token in the mandatory
  early crash placeholder. It is written before heavy work solely to satisfy the robustness
  contract and is overwritten after successful training. It never post-processes a successful
  model prediction.
- Fixed seeds make stochastic training reproducible. Model width, layer count, dropout, optimizer,
  fold count, feature ranges, and search grids are ordinary model/training design, not constants
  discovered from labels or test predictions.

Strip-the-ML result: with all trained classifiers and neural models removed, there are no class
logits and the ensemble/decoder cannot produce a ranked path. The only remaining CSV is the
explicitly mandated crash placeholder, not usable model answers. Matrix enumeration alone is never
called as a context-free answer generator. The trained models produce the answer; phrase features
and public algebra only support them.

6. Robustness and housekeeping
------------------------------

- The script reads public_dir and writes only submission_out, both taken from sys.argv.
- submission_out.parent is created before writing.
- A schema-valid placeholder is written immediately after test.csv is read and before train loading
  or training.
- Missing required input files are the only deliberate pre-placeholder hard failure.
- Fold/model failures retain completed trained models; per-row decode failures use per-row model
  argmax and continue.
- No dataset-size assertion is used. Row count is read from the supplied files.
- Before final write, the script checks exact column names/order, row count, ID completeness and
  uniqueness, ID-set equality, nonempty paths, path length, and allowed tokens.
- Random seeds are fixed. solution.py is self-contained and imports no local module.
- solution.py is the only Python source file in the challenge directory.

7. What worked and what did not
-------------------------------

What worked:

- Searching a tighter phrase window removed distant document noise while preserving the lexical
  evidence nearest MASK; the observed calibration search selected 12 left and 20 right tokens.
- Local phrase n-grams learned by multinomial text classifiers captured stable coded-word
  collocations better than a neural encoder alone on only 2,108 rows.
- The small Transformer supplied complementary ordered six-gap evidence; the real-metric search
  retained it at 0.40 ensemble weight despite weaker standalone position accuracy.
- Posterior-marginal finite-field decoding improved over maximum-score decoding while preserving
  the entire 15% matrix term. The temperature was selected against the real metric in-script.
- Two differently represented trained text models remained complementary at 0.30 weight each.

What did not work in train-only development probes:

- A row-wide TF-IDF representation diluted the target boundary and reached about 0.249 position
  accuracy on the development fold, below the local 1-3-gram model at about 0.308.
- A recurrent encoder and a hashed neural n-gram encoder overfit the small training fold and reached
  only about 0.25 and 0.23 position accuracy respectively.
- A separately learned first-order operator transition matrix reduced the exact-decoder metric, so
  it was deleted rather than shipped.
- Context-free matrix search was not considered a prediction method: it can earn matrix agreement
  but cannot identify the labelled discourse path and would fail the real-ML requirement.

8. Cold-reviewer conclusion
---------------------------

The final fresh source review found no hardcoded generation pattern: no O-token literal occurs in
solution.py, no phrase/token lookup emits a class, and the public matrix procedure is only invoked
with trained ensemble logits. The trained classifiers and Transformer produce every successful-run
answer. The only test-derived values read across rows are required output-integrity checks; all
prediction arithmetic is per row or pointwise averaging across models. No train/test concatenation
or test-fitted state exists.

The observed submission.csv has exactly 702 rows, columns sample_id,operator_path in that order, the
exact test ID set with no duplicates, six allowed tokens in every path, and all 702 predicted matrix
products equal their corresponding supplied boundary_matrix.
