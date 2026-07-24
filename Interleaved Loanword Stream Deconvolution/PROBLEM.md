Interleaved Loanword Stream Deconvolution
Domain: Seq2Seq

Overview
Each example contains three protected loanword forms from one recipient-language vocabulary. Their glyph streams have been interleaved into one mixed stream while preserving the internal left-to-right order of every hidden form.

The input supplies:

one mixed protected-glyph sequence;

three ordered lexeme slots;

the target length and number of distinct glyphs for each slot;

protected concept information and semantic fields;

an opaque recipient-language code.

The output is the three original variable-length glyph sequences in lexeme-slot order. This is a constrained sequence deconvolution problem: the output glyphs must collectively use every input glyph occurrence exactly once, and the three outputs must be capable of reproducing the displayed mixed stream through order-preserving interleaving.

The target is not a row class, candidate ID, donor label, or per-token category. Solvers must generate three complete sequences.

Number of Rows
The prepared dataset contains exactly 366 unique sequence examples:

| Split | File | Rows | Hidden lexeme streams |

|---|---|---:|---:|

| Training | train.csv | 271 | 813 |

| Evaluation | test.csv | 95 | 285 |

| Total | | 366 | 1,098 |

Every row contains three output streams. Individual stream lengths range from 2 to 14 protected glyphs.

sample_submission.csv and the hidden answers mirror the same 95 evaluation IDs and are not additional examples.

Files
train.csv — mixed streams and labeled deinterleavings.

test.csv — mixed streams without target sequences.

sample_submission.csv — structurally valid example output.

Input Columns
id: opaque row identifier.

prompt: transduction instruction.

deinterleaving_packet_json: mixed stream, recipient code, and three lexeme-slot descriptors.

mixed_length: total number of glyph tokens in the mixed stream.

slot_count: always 3.

answer_json: training-only output.

Input JSON Schema

{

  "recipient_code": "recipient_xxx",

  "mixed_glyph_stream": ["g02", "g07", "g05", "g01", "g03", "g05", "g04"],

  "lexeme_slots": [

    {

      "slot_index": 0,

      "concept_code": "concept_a12",

      "semantic_field": "Food and drink",

      "target_length": 2,

      "unique_glyph_count": 2

    },

    {

      "slot_index": 1,

      "concept_code": "concept_b34",

      "semantic_field": "Animals",

      "target_length": 3,

      "unique_glyph_count": 3

    },

    {

      "slot_index": 2,

      "concept_code": "concept_c56",

      "semantic_field": "The body",

      "target_length": 2,

      "unique_glyph_count": 1

    }

  ]

}

Original spellings, language names, source identifiers, and output forms are not exposed directly.

Output Schema
answer_json must contain exactly:


{

  "lexeme_streams": [

    ["g02", "g01"],

    ["g07", "g03", "g04"],

    ["g05", "g05"]

  ]

}

The three lists must follow the displayed lexeme_slots order.

A valid prediction must:

contain exactly three glyph-token lists;

match every displayed target_length;

use the same glyph multiset as mixed_glyph_stream;

form a valid order-preserving deinterleaving of that mixed stream.

Submit exactly two CSV columns:


id,answer_json

ild_example,"{""lexeme_streams"":[[""g01""],[""g02""],[""g03""]]}"

Missing, duplicate, unknown, extra, or null row IDs are rejected. Malformed JSON and structurally impossible deinterleavings are rejected.

Evaluation
The three streams are flattened with boundary tokens and evaluated using one normalized token-level edit similarity:


row_score = 1 - edit_distance(predicted_tokens, gold_tokens)

                  / max(len(predicted_tokens), len(gold_tokens), 1)

final_score = mean(row_score)

For structurally valid submissions, scores range from 0 to 1, and higher is better. Exact deinterleaving receives 1. Invalid or malformed submissions are rejected by the strict grader rather than assigned a normal metric value. The platform grading configuration must therefore use negative infinity as the minimum bound and 1 as the maximum bound. There is no classification accuracy term, set-F1 term, LCS term, candidate-selection term, or weighted metric composition.

Expected Approaches
A compliant solution should train a sequence model on train.csv and jointly decode each mixed stream under the three displayed length constraints. Suitable CPU approaches include compact recurrent language models, character-level transformers trained from scratch, neural sequence scorers with beam search, or other learned constrained decoders.

Prohibited Approaches
external WOLD lookup, dictionaries, web search, or source reconstruction;

external datasets, hosted APIs, remote inference, or runtime downloads;

hardcoded evaluation outputs, manual answer injection, ID maps, or row patches;

exploiting row order, opaque IDs, or deterministic construction artifacts;

fitting on test rows, pseudo-labeling, test-time adaptation, or test-distribution calibration;

TF-IDF at any stage;

BM25, fixed n-gram overlap, fuzzy matching, or a rule-only main solution;

a submission script with no genuine task-specific model training on train.csv.

All task-specific fitting must happen inside the submitted CPU solution using only the public challenge files.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.