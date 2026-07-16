Historical Newspaper OCR Forensics solution
============================================

Run
---

    python3 solution.py <public_dir> <submission_out>

The platform supplies both paths. The script reads train.csv and test.csv from
public_dir, trains only on released data, and writes exactly to submission_out.
It uses fixed random seeds and CPU implementations throughout.

Approach
--------

This is candidate-anchored verification and repair, not unrestricted OCR. The
system never decodes a line image into free text. It starts with candidate_text,
enumerates small candidate-to-truth edits observed in the released training
set, and asks both a language model and a visual matching model which supplied
hypothesis best matches the line.

1. Learn the corruption channel from candidate_text/corrected_text pairs using
   exact SequenceMatcher edit spans.
2. Learn an interpolated character 2--5 gram model from trusted training
   corrected_text.
3. For each candidate, enumerate only localized hypotheses permitted by the
   documented categories:
   - observed character substitutions, including historical confusions;
   - insertion or removal of one ordinary character;
   - insertion/removal/replacement of spaces and punctuation;
   - observed diacritic changes;
   - pairs of non-overlapping edits from different categories for mixed.
4. Keep the eight best character- and word-ranked hypotheses per error type.
5. Score those hypotheses against the image with the candidate-conditioned
   visual matcher.
6. Feed language, edit, text-shape, and visual score features to a CatBoost
   multiclass classifier. The predicted type selects its corresponding repair;
   the language and visual scores rerank the eight localized alternatives.
7. Preserve candidate_text byte-for-byte whenever the prediction is none.

Model architecture / algorithm
------------------------------

Character language model
~~~~~~~~~~~~~~~~~~~~~~~~

The language score interpolates smoothed 2-, 3-, 4-, and 5-character
probabilities. Boundary markers preserve line-initial and line-final evidence.
A one-edit hypothesis is scored by recomputing only the affected n-grams; this
makes exhaustive localized insertion/deletion search practical without changing
the score. The language model is used as evidence, not as a generator, so
unusual historical spellings remain available unchanged.

Word language evidence
~~~~~~~~~~~~~~~~~~~~~~

A released-corpus word unigram/bigram model complements character n-grams.
Type-specific weights were selected on held-out data: stronger word evidence
helps missing-character and substitution repairs, while spacing keeps the
character-only score to avoid over-normalizing historical compounds.

Candidate-conditioned visual matcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Images are converted to grayscale darkness, percentile-normalized, and resized
to height 32 while retaining aspect ratio. No preprocessing step recognizes or
outputs text.

A small CPU convolutional encoder has channels 8 -> 16 -> 32 and produces a
horizontal feature sequence. For each character of a supplied text hypothesis,
heuristic proportional glyph widths determine a monotonic expected x-position.
Seven neighboring visual features are gathered around that position. They are
combined with embeddings of the supplied previous/current/next characters.
Token compatibility scores are summarized using mean, minimum, bottom-three,
maximum, and variance statistics, then mapped to one line/hypothesis score.

The visual model has no CTC loss, decoder, vocabulary-output head, beam search,
or route for generating a complete transcript. It can only compare a supplied
candidate-derived hypothesis with an image. Training uses a pairwise ranking
loss between each trusted corrected text and its actual corrupted candidate.
Correct training rows receive one synthetic, localized candidate-like negative.
Localized token supervision marks only the candidate positions implicated by
the trusted edit. This trains the visual encoder to distinguish a single wrong
glyph instead of relying only on a line-level ranking loss.


Decision model
~~~~~~~~~~~~~~

A one-vs-rest ensemble of eight depth-6 CatBoost models predicts the canonical
error classes. Independent detectors improved the rare punctuation and mixed
classes over the multiclass baseline. Training uses a disjoint calibration
split plus a low-weight, class-balanced supplemental sample. Final assignment
enforces the released generation proportions.

Feature engineering
-------------------

Per-row features include:

- candidate length plus character- and word-LM scores per character;
- counts of spaces, punctuation, non-ASCII characters, capitals, and digits;
- for every error type, best score improvement over the unchanged candidate;
- best-versus-second hypothesis margin and hypothesis availability;
- edit position, boundary indicators, old/new span lengths, edit operation,
  and whether the edit changes space or diacritic content;
- visual score, visual improvement over unchanged candidate, visual margin,
  and the visual score of the language-first repair.

The released generation proportions are used for a globally calibrated final
assignment. This materially improved punctuation, deletion, and mixed recall
and raised the held-out weakest-component score.

Validation strategy
-------------------

A reproducible stratified 60/20/20 split is used by the validation function:

- 60% builds the language model, corruption channel, and visual matcher;
- 20% trains CatBoost, augmented by low-weight balanced training examples;
- 20% is held out for all reported decisions.

The local report implements the challenge's class-balanced character repair,
exact-text, exact-row, correctness macro-F1, error-type macro-F1, weakest
component, and final weighted score.

Best measured held-out configuration (proportion-calibrated classes,
visual/language reranking weight 2.0):

- challenge score: 0.6316
- correctness macro-F1: 0.7750
- error-type macro-F1: 0.6248
- balanced strict-character repair: 0.6800
- balanced exact-text accuracy: 0.5935
- balanced exact-row accuracy: 0.5935

What worked
-----------

- Treating candidate_text as the anchor reduced repair to a small, auditable
  hypothesis set.
- The learned confusion channel gave nearly complete one-edit coverage.
- Character n-grams were especially effective for substitution, insertion, and
  spacing repair while retaining historical spelling.
- Incremental edit scoring reduced feature-generation cost substantially.
- Visual compatibility features improved the balanced validation score over the
  language-only baseline, particularly for minority error decisions.
- Training the final classifier on held-out calibration rows avoided direct
  target leakage from the trusted corrections used by the language model.

What did not work
-----------------

- Pure image/text ranking was weaker than language-first ranking for selecting
  the exact character; visual scores are therefore classifier evidence and a
  secondary reranker rather than the sole decision.
- A direct character-TF-IDF error classifier was substantially weaker than the
  hypothesis-delta features, so it was not included.
- Mixed errors remain the hardest class: composing two localized changes expands
  ambiguity, and aggressive mixed predictions damaged the none class.
- A language-only baseline reached reasonable raw accuracy but underperformed on
  the class-balanced/weakest-component objective.

Submission validation
---------------------

The generated file was checked after the final run: 5,000 rows; exact required
column order; sample-submission ID order; unique IDs; canonical error labels;
0/1 is_correct values; non-empty text; maximum text length 104; logical
none/is_correct consistency; and exact candidate preservation on every row
predicted correct.
