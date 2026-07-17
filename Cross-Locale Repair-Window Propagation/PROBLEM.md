Problem Description
Cross-Locale Repair-Window Propagation
Overview
A localized product release sends the same source segment to several locale workflows. When a linguist repairs one locale and records the error family, a release manager needs to identify which regions of the remaining locale drafts deserve review. Sending every token back to a linguist wastes time; missing a corresponding region ships an avoidable defect.

Predict the target locale's repair windows from an approved anchor-locale edit. Each text is represented by a fixed-width privacy sketch: 16 normalized position windows, four collision-prone keyed token buckets per window. The sketches preserve learnable local patterns without exposing original sentences, exact token counts, or human corrections. Every example comes from real expert post-editing; no error text or labels are synthesized.

The private set holds out complete source groups and seven directed locale pairs. Each anchor and target correction shares a human-recorded MQM error family and is verified to touch related, but non-identical, normalized regions. A separate boundary-tolerant preparation screen excludes both unrelated pairs and near-copies. This curation isolates cases where propagation is warranted without turning the answer into a copied anchor mask. A system must transfer repair topology across languages rather than retrieve original sentences or learn one locale route.

Dataset
File descriptions
train.csv -- 17,444 labeled multilingual repair-window observations.

test.csv -- 2,563 unlabeled observations with unseen source groups and held-out directed locale pairs.

sample_submission.csv -- Submission template with random 16-bit predictions.

Column descriptions
id (string) -- Unique 12-character hexadecimal observation identifier.

anchor_locale (string) -- Locale in which a human-approved repair is available.

target_locale (string) -- Different locale whose repair windows must be predicted.

anchor_error_family (string) -- MQM family shared by the verified anchor and target corrections.

source_sketch (string) -- Fixed-width 64-code sketch of the common source segment.

anchor_draft_sketch (string) -- Fixed-width sketch of the anchor translation before review.

anchor_repaired_sketch (string) -- Fixed-width sketch of the approved anchor translation.

anchor_repair_windows (string) -- Sixteen binary digits; 1 marks a normalized anchor window touched by the human edit.

target_draft_sketch (string) -- Fixed-width sketch of the unreviewed target translation.

target_repair_windows (string) -- Sixteen binary digits marking target windows touched by the real human post-edit; present only in train.csv and submissions.

Window 0 is the beginning of a segment and window 15 is the end. All sketches contain exactly four codes per window. Bucket collisions and deterministic decoy codes are intentional privacy constraints.

Latin/Cyrillic text is tokenized into Unicode words and punctuation. Japanese hiragana, katakana, and kanji are tokenized character by character before normalized windows are formed, preventing an unspaced sentence from collapsing into one review unit.

Evaluation
Submissions are scored by locale-balanced Review-Window Utility (RWU). For each row, predicted and true repair windows receive a maximum-weight one-to-one matching. An exact-window match earns 1.0 credit, an immediately adjacent match earns 0.35, and a match farther away earns zero. The adjacent credit reflects limited imprecision at normalized window boundaries without treating shifted localization as fully correct.

Weighted repair-window F1: 80%.

Review-budget fidelity, min(predicted_count, true_count) / max(predicted_count, true_count): 20%.

Rows are averaged within each target locale, then the seven locale scores are averaged equally. Predicting no windows scores zero because every retained target contains a real human edit. Higher is better; the range is 0 to 1.


matched_credit = maximum_weight_one_to_one_matches(

    predicted_windows,

    true_windows,

    exact_credit=1.0,

    adjacent_credit=0.35,

)

precision = matched_credit / max(len(predicted_windows), 1)

recall = matched_credit / max(len(true_windows), 1)

weighted_f1 = harmonic_mean(precision, recall)

budget = min(len(predicted_windows), len(true_windows)) / max(len(predicted_windows), len(true_windows), 1)

row_score = 0.80  *weighted_f1 + 0.20*  budget

score = mean(mean(row_score for rows in locale) for locale in seven_locales)

Submission
Submit one 16-bit repair-window mask for every test row.

id (string) -- Exact identifier from test.csv.

target_repair_windows (string) -- Exactly 16 characters, each 0 or 1.

Example:


id,target_repair_windows

0021586c38db,0000000001100010

00363157b323,0000000000000000

Requirements
The file must contain exactly one row for every test id.

IDs must be unique and match the test IDs exactly.

Columns must be id,target_repair_windows in that order.

Every prediction should match [01]{16}. If a CSV reader strips leading zeros, the grader left-pads an otherwise valid binary value back to 16 positions.

File format: .csv only.

What Not To Use
Do not recover labels from an external copy of LangMark or other answer-bearing translation memory.

Do not hardcode predictions by test ID or row order.

Do not claim an anchor-conditioned system while discarding every anchor field; compare against a target-only ablation.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.