Overview
Build a system that expands an editor-style historical-Polish search query into the spellings that may appear in a source document.

For each test row, you receive a short sequence of query tokens written in editorial spelling. Your submission must return an aligned variant lattice: for every query-token position, give one to three possible source spellings and a probability for each one. A variant lattice is simply a token-by-token list of ranked spelling candidates. It is not a translation, summary or unaligned generated sentence.

In plain terms:


input:  editorial query tokens + linguistic codes + three examples from the same source profile

output: source-spelling candidates with probabilities at each token position

The prepared rows come from human-annotated 17th- and 18th-century Polish source documents in a public digital-humanities corpus. In that corpus, each accepted token is aligned with two spellings:

an editor-facing transcription, used here as query_tokens;

a source-preserving spelling, used here as observed_tokens in training and as the hidden target in testing.

Rows are created by extracting aligned sentence-level token sequences from those annotations. For each source document, the preparation script selects three separate aligned sentences as public calibration examples, then uses other aligned sentences from the same document as prediction rows. Broken XML files, rejected segmentation alternatives, malformed tokens, and invalid alignments are skipped during preparation.

A profile_id represents one anonymized source document or work. Rows with the same profile come from the same source and therefore tend to share spelling habits, printer conventions, abbreviation behavior, and editorial/transcription patterns. Training and test profiles are disjoint, but every test profile includes three public calibration examples from that profile. The split therefore tests few-shot transfer to a new source profile rather than memorization of a source seen in training.

A small row-level expansion_budget limits how many extra candidates you may add beyond the mandatory one candidate per token. The model must decide where uncertainty is useful. Spending alternatives on every token is invalid, while predicting only one spelling everywhere may miss important historical variants.

This models a practical archive-search workflow. Researchers often search with modernized or editorial spelling, while source records may preserve older spelling conventions. A useful expansion system improves recall without flooding the search engine with too many noisy alternatives. Ordinary historical normalization is only a partial building block: this task runs in the reverse direction, adapts to a source profile from examples, and evaluates calibrated aligned candidate sets rather than one deterministic normalized string.

Your complete solution must run on one NVIDIA A10G GPU with 24 GB of VRAM and finish within 1 hour, including training, validation, inference, and writing the submission.

Dataset

dataset/public/

├── train.csv

├── test.csv

└── sample_submission.csv

JSON-valued columns use compact JSON. All token positions are aligned. The validated release contains 3,500 training rows and 900 test rows; the test rows span 200 anonymous profiles.

train.csv

case_id,profile_id,query_tokens,analysis_codes,expansion_budget,calibration_examples,observed_tokens

Columns:

case_id

Data type: string

Opaque row identifier, unique within train.csv.

profile_id

Data type: string

An anonymized source-profile identifier. One profile corresponds to one source document or work, not to a person. Rows with the same value share source-specific spelling behavior and the same three calibration examples.

query_tokens

Data type: JSON array of Unicode strings

Editorially spelled input sequence. These are the spellings a researcher might naturally type into an archive search system.

analysis_codes

Data type: JSON array of arrays of strings

Linguistic feature codes aligned to query_tokens. Each inner array contains one or more categorical codes for the corresponding token.

The codes are derived from the token's morphosyntactic analysis in the source annotations. They represent broad word-class information and, when available, inflectional features such as case, number, gender, person, tense, mood, degree, or related grammatical categories.

The exact tag labels are anonymized into stable g_... codes. Identical codes have identical meaning everywhere in the public files, but the code strings have no numeric order or direct human-readable label. Treat them as categorical features.

expansion_budget

Data type: integer

Maximum number of extra candidates allowed across the row, beyond one mandatory candidate at every token position. Values are between 2 and 8 and depend only on public sequence length.

calibration_examples

Data type: JSON array of three objects

Each object has an aligned query_tokens array and observed_tokens array from the same profile_id. Calibration sequences are distinct from the row's requested sequence. They show how this profile tends to spell some editorial tokens in source form.

observed_tokens

Data type: JSON array of Unicode strings

The aligned source-profile spelling sequence to learn for this row.

For every training row:


len(query_tokens) == len(analysis_codes) == len(observed_tokens)

Within every calibration object:


len(query_tokens) == len(observed_tokens)

test.csv

case_id,profile_id,query_tokens,analysis_codes,expansion_budget,calibration_examples

The columns have the same meanings as in train.csv, but the requested observed sequence is not included. The public expansion_budget is part of the submission contract and does not depend on the hidden spelling sequence. Training and test profile_id sets are disjoint. The calibration examples make each held-out profile locally observable, so the split tests few-shot transfer rather than unsupported convention guessing.

Every test case_id must appear exactly once in the submission.

sample_submission.csv

case_id,variant_lattice

The sample contains a valid one-candidate lattice filled with a deliberately incorrect placeholder. It demonstrates serialization and scores exactly 0.0.

Shortened data examples
The examples below are shortened for readability. Actual public rows are CSV rows, and the JSON fields use the same shapes. Code values shown here are illustrative examples of the public g_... format.

A training row fragment:


case_id: avl_1f4d8c2a9306b7e4d512

profile_id: p_9c083fd44b2a6f071a

query_tokens: ["który", "jest", "panem", "?"]

analysis_codes: [

  ["g_8f2b4c91aa", "g_3a16db9240"],

  ["g_17e0d6a531"],

  ["g_4ad19b0c77", "g_a24861fb02"],

  ["g_2ab08ef591"]

]

expansion_budget: 2

calibration_examples: [

  {"query_tokens": ["który", "był"], "observed_tokens": ["ktory", "był"]},

  {"query_tokens": ["jest", "to"], "observed_tokens": ["iest", "to"]},

  {"query_tokens": ["pan", "mój"], "observed_tokens": ["pan", "moy"]}

]

observed_tokens: ["ktory", "iest", "panem", "?"]

A matching test row has the same public fields but omits observed_tokens:


case_id: avl_45bc2e8a0f19d320ab6d

profile_id: p_3e64a8d0b296c107

query_tokens: ["który", "jest", "dobry", "."]

analysis_codes: [

  ["g_8f2b4c91aa", "g_3a16db9240"],

  ["g_17e0d6a531"],

  ["g_91dcef3740"],

  ["g_7f26311db5"]

]

expansion_budget: 2

calibration_examples: [

  {"query_tokens": ["który", "człowiek"], "observed_tokens": ["ktory", "człowiek"]},

  {"query_tokens": ["jest", "taki"], "observed_tokens": ["iest", "taki"]},

  {"query_tokens": ["mój", "dom"], "observed_tokens": ["moy", "dom"]}

]

For this test row, a valid lattice would contain four position lists, one for each token in query_tokens.

Submission format
Write predictions to:


working/submission.csv

The CSV must contain exactly these columns in this order:


case_id,variant_lattice

variant_lattice is a JSON array with one entry per query_tokens position. Each position is a list of one to three candidate objects:


[

  [

    {"text": "ktory", "prob": 0.65},

    {"text": "który", "prob": 0.35}

  ],

  [

    {"text": "iest", "prob": 0.70},

    {"text": "jest", "prob": 0.30}

  ],

  [

    {"text": "dobry", "prob": 1.0}

  ],

  [

    {"text": ".", "prob": 1.0}

  ]

]

For every token position:

the candidate list must contain between 1 and 3 objects;

every object must contain exactly text and prob;

text must be a unique, non-empty, trimmed Unicode string of at most 160 characters and must not contain control or surrogate characters;

prob must be a finite number in [0.01, 1.0];

probabilities must be in non-increasing order and sum to 1.0 within 1e-6;

when probabilities tie, list order determines the top-1 candidate.

Across the row, define:


extra_candidates = Σ_positions (number_of_candidates_at_position - 1)

extra_candidates must not exceed that row's public expansion_budget. A one-candidate prediction at every position always uses budget 0. The budget is global: using a second candidate at one position consumes one unit, and using three candidates consumes two units.

The outer list length must equal the corresponding query_tokens length. Rows may appear in any order. Extra columns, missing rows, duplicate identifiers, malformed JSON, invalid probabilities, or incorrect sequence lengths invalidate the submission.

CSV example:


case_id,variant_lattice

avl_45bc2e8a0f19d320ab6d,"[[{""text"":""ktory"",""prob"":0.65},{""text"":""który"",""prob"":0.35}],[{""text"":""iest"",""prob"":0.70},{""text"":""jest"",""prob"":0.30}],[{""text"":""dobry"",""prob"":1.0}],[{""text"":""."",""prob"":1.0}]]"  
avl_45bc2e8a0f19d320ab12,"[[{""text"":""ktory"",""prob"":0.55},{""text"":""który"",""prob"":0.45}],[{""text"":""iest"",""prob"":0.70},{""text"":""jest"",""prob"":0.30}],[{""text"":""dobry"",""prob"":1.0}],[{""text"":""."",""prob"":1.0}]]"

Evaluation
The score rewards accurate spellings, calibrated uncertainty, transfer to unseen query-to-source spelling pairs, complete phrase coverage, and preservation of tokens that should not change. The expansion budget is a hard validity constraint rather than a separately weighted score component, so a solver cannot gain coverage by flooding every position with alternatives.

Per-position credit
For one position, let the submitted candidates be (v_j, p_j) and let t be the true observed token. Define:


p_true = sum of p_j for candidates where v_j == t

probability_credit = max(0, 2 × p_true - Σ_j p_j²)

probability_credit is 1 for a probability-1 correct candidate and 0 when the true token is absent. Splitting probability across unnecessary candidates lowers the credit.

Character similarity is:


char_similarity(v, t) = max(0, 1 - levenshtein(v, t) / max(len(v), len(t), 1))

The probability-weighted character score for one position is:


expected_char = Σ_j p_j × char_similarity(v_j, t)

Position groups
For a row:

changed positions are positions where the public query token differs from the true observed token;

unchanged positions are positions where they are equal;

novel-variant positions are changed positions whose exact query_token → observed_token pair is not shown in any public training target or public calibration example.

Novel-variant positions affect scoring only through the novel_probability and novel_top1 components, which together make up 35% of the row score. They are included to measure whether a model can compose known spelling behavior rather than merely copy a seen token-pair table.

To keep this learnable, a novel-variant position is eligible only when its character-level edit pattern has support in the public training data. This pattern is called an edit signature. It is computed by case-folding the query token and observed token, aligning their characters, and recording only the non-matching edit operations, such as substitutions, insertions, deletions, or case-only differences. For example, accent loss, ji style substitutions, or a recurring inserted character can form reusable edit patterns. The exact signature is not something you submit; it is only a construction check ensuring that novel positions require transfer from recurring public transformations, not guessing a completely unseen spelling rule.

The prepared test set contains at least two novel-variant positions per row. These positions use the same character inventory and recurring transformation structure as the public training data.

Row score
Define:


changed_probability = mean probability_credit on changed positions

changed_top1 = exact top-1 accuracy on changed positions

changed_char = mean expected_char on changed positions

novel_probability = mean probability_credit on novel-variant positions

novel_top1 = exact top-1 accuracy on novel-variant positions

full_changed_coverage = 1 if every changed target occurs anywhere in its candidate list, else 0

unchanged_probability = mean probability_credit on unchanged positions

The row score is:


0.20 × changed_probability

+ 0.10 × changed_top1

+ 0.10 × changed_char

+ 0.25 × novel_probability

+ 0.10 × novel_top1

+ 0.15 × full_changed_coverage

+ 0.10 × unchanged_probability

The component weights are:

| Component | Weight | What it measures |

|---|---:|---|

| changed_probability | 0.20 | Probability assigned to correct changed-token spellings |

| changed_top1 | 0.10 | Whether the best candidate is exactly correct on changed tokens |

| changed_char | 0.10 | Character-level closeness on changed tokens |

| novel_probability | 0.25 | Probability quality on held-out query-to-observed pairs |

| novel_top1 | 0.10 | Top-1 exact accuracy on held-out query-to-observed pairs |

| full_changed_coverage | 0.15 | Whether every changed token is covered somewhere in the lattice |

| unchanged_probability | 0.10 | Preservation of tokens that should stay unchanged |

Row scores are averaged within each profile_id. The final score is the unweighted mean of those profile scores multiplied by 100. Profile-macro averaging prevents large profiles from dominating evaluation.

Not allowed
External APIs.

External historical spelling pairs, historical corpora, specialist dictionaries, or manually created test transcriptions.

Searching for or re-identifying public source passages to recover test spellings.

Hard-coding predictions by case_id, profile_id, row order, or filename.

Exploiting serialization details, preparation behavior, grader behavior, or identifiers instead of modeling the public text and calibration evidence.

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.