Route and Read: Joint Tool Selection and Readiness
===================================================

Run:  python3 solution.py <public_dir> <submission_out>
Hardware: CPU only (DEVICE = torch.device("cpu"), 8 torch threads). No GPU code path.

1. Problem facts taken from PROBLEM.md
--------------------------------------
Submission schema: exactly two columns, in this order: id,prediction
  - id: every id of test.csv, one row each (300 rows)
  - prediction: an integer in {0,1,2,3,4,5}
      label = k       (k = 0,1,2) -> candidate k is relevant and its arguments are ready
      label = k + 3                -> candidate k is relevant and the call is not ready
                                      (a missing required value or a value outside its constraint)
Metric: plain multiclass accuracy over all test rows. There is no worst-group, subgroup or
  partial-credit term, and every routing or readiness mistake costs the same.
Domain: NLP (guidebook section 5.1). This is a bilingual (en/zh) semantic matching task plus a
  schema check. There is no model-type restriction and no Fine-tuning categorisation, so I used
  pretrained Hugging Face sentence-encoder backbones, fine-tuned in-script.
Challenge rules: use only the public files; no private answers, construction artifacts, hidden
  metadata, source lookup or web services. PROBLEM.md states that raw lengths, row ids,
  language, tool count and schema size are not intended label signals, so none of them is a
  model feature.

2. Approach
-----------
The 6-way label factorises into a routing part (which of the 3 candidates) and a readiness
part (ready or not ready). One network models both jointly:

    log p(label = k + 3j) = log softmax(route_scores)[k] + log softmax(ready_logits)[j]

Routing (the hard half), backbones a and b: a bi-encoder, i.e. a fine-tuned multilingual
  sentence encoder with mean pooling and L2 normalisation.
  score_k = cos(embed(request), embed(candidate_text_k)) / 0.05.
  The training loss is cross-entropy over the row's own three candidates plus the candidates of
  the other 15 rows in the batch (in-batch negatives).
Readiness: a small MLP head (21 -> 16 -> 2) trained jointly with the encoder (loss = routing CE
  + readiness CE). Its input is the schema-validation features described below.

Three backbones, each a full joint model with its own readiness head:
  a) sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Candidate text = root
     description + " | " + the schema's (nested) parameter descriptions, max 128 tokens.
  b) sentence-transformers/paraphrase-multilingual-mpnet-base-v2. Candidate text = root
     description only, max 64 tokens.
  c) BAAI/bge-reranker-base, a pretrained cross-encoder (reranker): request and candidate root
     description are read together and give one relevance logit per candidate; the training
     loss is cross-entropy over the row's three candidates. Max 96 tokens.
Their joint 6-way log-probabilities are blended as w_a*a + w_b*b + w_c*c with the weights on
the simplex, and the prediction is the argmax.

Training: AdamW (encoder lr 2e-5 with weight decay 0.01; head lr 1e-2), batch 16 rows, linear
warm-up over the first 10% of steps then linear decay, gradient clipping at 1.0, 2 epochs, fixed
seed 0.

Early stopping, epoch selection and blend weight, all done in-script:
  - holdout = the first fold of a StratifiedKFold(5, shuffle, seed 0) on target x language
    (300 rows); the models train on the other 1200 rows;
  - after every epoch each backbone predicts the holdout and the test set. The epoch with the
    best holdout joint accuracy is kept (its test predictions are the ones used);
  - the blend weights are grid-searched over the 66-point simplex grid with step 0.1 on the
    holdout against joint accuracy (the PROBLEM.md metric).

3. Feature engineering / preprocessing
---------------------------------------
- The dialogue is split into the free-text request and the supplied-argument JSON (the request
  is everything before the first "{").
- Chinese requests are rendered as two-character fragments joined by the enumeration comma
  "、". The commas are removed so the tokenizer sees contiguous text again. This is input text
  cleaning only; English rows contain no such character, so they are unchanged.
- Candidate text: the root "description", plus, for backbone (a), the descriptions of every
  nested parameter in document order. A parameter description that all three candidates of the
  same row share is dropped, because it cannot tell them apart. This is computed per row, with
  no hard-coded strings.
- Schema-validation features (per candidate, by a recursive JSON-schema walk of the supplied
  arguments): missing required keys (top level and nested), number of required keys, type
  mismatches, enum violations, minimum/maximum violations, and the number of checked values.
  The three candidates' vectors are summarised as mean/max/min and log1p-transformed (21
  numbers). These are inputs to the learned readiness head; they are not used as a rule.
- Parameter names are opaque row-local handles and are never used as text.

4. Validation
-------------
Strategy: stratified (target x language) split. A label-free check showed that a random
stratified split reproduces test's overlap with train. Measured per test row as the number of
candidate descriptions also seen in train:
  test:         0/1/2/3 = 0.20/0.49/0.24/0.06
  random 5-fold: 0.26/0.46/0.23/0.05
Grouping by description gave less overlap than test. A candidate that was the target in the
training fold is the target again only 40% of the time (chance is 33%), so memorising tools
gives little.

In-script holdout result of the final script (300 rows):
  MiniLM (root+params): epoch 1 0.6433, epoch 2 0.6600 (epoch 2 kept); en 0.607, zh 0.713
  mpnet (root):        epoch 1 0.6133, epoch 2 0.6400 (epoch 2 kept); en 0.573, zh 0.707
  bge-reranker (root): epoch 1 0.6000, epoch 2 0.5967 (epoch 1 kept); en 0.527, zh 0.673
  blend search, w = (MiniLM, mpnet, bge), top 5 of 66:
    (0.6, 0.0, 0.4) .680 | (0.3, 0.1, 0.6) .673 | (0.3, 0.3, 0.4) .673 |
    (0.3, 0.4, 0.3) .673 | (0.5, 0.0, 0.5) .673;  equal weights .663
  final blend (w = 0.6/0.0/0.4): JOINT ACCURACY 0.6800 (routing 0.6800, readiness 1.0000;
    en 0.620, zh 0.740)
  w was picked on the same 300 rows, so the blend's 0.680 is slightly optimistic. On the same
  holdout, the two-backbone blend (MiniLM + mpnet, w=0.4) scored 0.667 and the version before
  that (no zh separator removal, root text only) scored 0.620.

Offline experiments (folds 0+1 of a stratified 5-fold, 600 rows; routing accuracy):
  zero-shot (no training), root description: MiniLM 0.503, mpnet 0.514, e5-base 0.481,
    LaBSE 0.447, e5-small 0.446
  fine-tuned MiniLM, root text                        0.622
  + Chinese separator removal                         0.643   (zh 0.65 -> 0.71)
  + parameter descriptions in candidate text          0.657   (en 0.577 -> 0.613)
  fine-tuned mpnet, root text, separator removal      0.635
  blend of the two (w = 0.3..0.6)                     0.662 - 0.672
Readiness: a learned classifier on the validation features is 100% in 5-fold CV and 100% on
the holdout. The joint accuracy therefore equals the routing accuracy.
Holdout numbers are relative indicators, not test-score estimates: fold-to-fold spread is
about +/-0.03 at 300 rows.

5. Runtime plan (static; nothing depends on the clock or the environment)
--------------------------------------------------------------------------
Fixed plan: 3 backbones x 2 epochs on 1200 rows, a holdout and test prediction after every
epoch, then a 66-point blend search.
Measured on an Apple M4 Pro CPU with 8 threads: MiniLM (root+params) about 135 s/epoch, mpnet
(root) about 115 s/epoch.
Measured end to end: 754 s wall clock (12.6 min) with all three backbones; the two-backbone
version took 532 s.
That is far below the 1.5 h guideline ceiling, which leaves room for a several-times-slower
grading CPU. Peak memory is 10.0 GB.
Backbone weights are downloaded from the Hugging Face hub (allowed by guidebook 3.3) and are
then fine-tuned in-script.

6. Leakage statement
--------------------
- Test rows are only parsed row by row (request, candidate texts and the row's own validation
  counts) and passed through predict(). Every test score depends only on that row's own
  content.
- No train+test concatenation anywhere. The train and test parse results are kept in separate
  dicts.
- Nothing is fitted on test: there is no scaler, vectorizer or encoder statistic at all (the
  tokenizers are the pretrained ones). Fine-tuning, epoch selection and the blend weight use
  train rows only (fit split and holdout).
- No pseudo-labelling, no calibration to a test distribution, and no aggregation over test
  predictions (no counts, no class-balance adjustment). The final argmax is per row.

7. Hardcoding statement
-----------------------
- No phrase-to-tool lookup, no keyword rule, no template. Routing is produced entirely by the
  fine-tuned encoders.
- Readiness is produced by a learned head. The validator only counts schema violations as
  input features, and the head learns how those counts map to ready / not ready.
- Every decision knob is searched in-script on the holdout: the epoch kept per backbone and the
  blend weight. The remaining constants are ordinary training hyperparameters (lr, batch,
  temperature, epochs, max lengths); none is a decode threshold or a tuned output bias.
- Strip-the-ML test: with every trained model removed, the pipeline produces no prediction at
  all. It has no routing scores and no readiness decision, only raw validation counts and text
  strings; the placeholder file is all 0. The trained models produce the answer.

8. What worked / what did not
-----------------------------
Worked:
- Fine-tuning the bi-encoder: zero-shot 0.50 -> 0.62.
- Removing the Chinese fragment separators: zh +5 points after fine-tuning.
- Parameter descriptions in the candidate text for the small backbone: helps English.
- In-batch negatives: more stable across epochs.
- Blending different backbones: about +1 to +2 points, then +1.3 more from the reranker.
Did not work:
- A cross-encoder built from MiniLM (request + candidate jointly): 0.46, and 3.5x slower. A
  pretrained reranker (bge-reranker-base) as the cross-encoder works: 0.60 alone, and it is the
  partner that lifts the blend from 0.667 to 0.680.
- e5-base as a backbone: 0.61, below mpnet and MiniLM.
- Lexical overlap between request and candidates is at chance (0.33). The task is built so
  surface terms mislead.
- Training past 1-2 epochs overfits: loss keeps falling while holdout accuracy flattens or
  drops.
Not used, on purpose: the description language (in a few zh rows the target's description is
English, a construction artifact), schema size, and similarity between candidates (at chance
anyway).

9. Pre-submit audit summary
---------------------------
Part A (test taint): te_data -> predict() -> test_lp -> blend -> per-row argmax -> CSV. No
  operation reads across test rows.
Part B (hardcoding): every dict, constant and branch was checked. PYTYPES/NUMERIC map JSON-schema
  type names to Python types for the validator's input features and decide no output.
  BACKBONES/lr/epochs are training hyperparameters. BLEND_STEPS sets the search space of an
  in-script search.
Part C (housekeeping):
  - seeds are fixed;
  - the script reads only public_dir and writes only submission_out;
  - solution.py is the only .py file and has no local imports;
  - the placeholder submission is written right after reading test;
  - an AST audit finds no try/except, raise, assert or while. The only `if` statement is the
    __main__ guard;
  - no time/clock reads, no environment reads (environment variables are only assigned), and
    no cuda/cpu_count checks;
  - torch.use_deterministic_algorithms(True) is set, with fixed thread counts.
  - submission SHA-256 of the local run:
    0c7d9b07721e988885648010407a4a6503a23965f116f65330aa3934bec1e9f4
