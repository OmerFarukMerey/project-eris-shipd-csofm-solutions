Lean Proof Patch Recovery
Overview
Repair a broken Lean proof card by submitting the line patch that should replace the faulty span. Lean is an interactive theorem prover: a proof script is checked line by line, and a small wrong tactic or missing argument can cause the verifier to stop with an error. Each task row contains one anonymized broken proof snippet, the checker feedback associated with the failure, an approximate failing line, and a short local proof-state hint.

The proof cards preserve line order, indentation, tactic structure, local dependencies, and repeated-symbol relationships, but source filenames, import paths, theorem names, and most identifiers have been replaced with row-local aliases. The snippets are not intended to be compiled against an external Lean library. Your goal is to recover a compact edit patch in the same anonymized vocabulary: the 1-based line where replacement begins, how many existing lines to delete, and the ordered Lean source lines to insert.

Dataset
train.csv: 3238 completed rows with inputs and ground-truth patches.
test.csv: 1743 held-out rows with the same inputs but without answer_json.
sample_submission.csv: Valid baseline CSV with the required columns.
Columns:

id (string): Unique row id.
prompt (string): Natural-language instruction for the row.
broken_proof (string): Anonymized Lean-like source text containing the faulty proof span. Newlines are preserved inside the CSV field.
compiler_feedback (string): Anonymized error or diagnostic text produced near the broken span.
line_hint (integer): Approximate 1-based line number associated with the failure report.
state_hint (string): Anonymized local proof-state text, such as goals, hypotheses, or tactic context available around the failure.
answer_format_json (JSON object): The required output schema. It names the fields replace_start_line, delete_line_count, and insert_lines, and indicates that the first two are integers while insert_lines is a list of source-code strings.
answer_json (JSON object, train only): Ground-truth patch object with replace_start_line, delete_line_count, and insert_lines.
Submission Format
id,answer_json  
leanpatch_example_1,"{""replace_start_line"":8,""delete_line_count"":1,""insert_lines"":[""  simp""]}"  
leanpatch_example_2,"{""replace_start_line"":12,""delete_line_count"":2,""insert_lines"":[""  exact h""]}"  

Submit a UTF-8 CSV with exactly two columns: id and answer_json. The answer_json value must be a valid JSON object serialized as a CSV string.

replace_start_line (integer): 1-based line number in broken_proof where the replacement begins.
delete_line_count (integer): Number of existing lines to remove starting at replace_start_line; must be non-negative.
insert_lines (JSON list of strings): Ordered anonymized proof-source lines to insert at that location. Preserve indentation inside each string and use the row-local aliases shown in the task.
Evaluation
For each row, the submitted patch is compared with the true patch.

location_score averages exponential penalties for start-line and delete-count errors:

location_score = 0.5*exp(-abs(pred_start - true_start)/2) + 0.5*exp(-abs(pred_delete_count - true_delete_count)/2).

For the text component, the inserted lines are joined with newline characters. Let pred_text and true_text be those joined strings. text_score = max(0, 1 - levenshtein_distance(pred_text, true_text) / max(len(pred_text), len(true_text), 1)).

line_lcs compares the inserted-line lists directly: line_lcs = LCS(pred_insert_lines, true_insert_lines) / max(1, len(true_insert_lines)). In this dataset the true patch inserts at least one line; the denominator rule is stated for completeness.

exact_patch is 1 only when replace_start_line, delete_line_count, and the full ordered insert_lines list all match exactly. Otherwise it is 0.

row_score = 0.18*location_score + 0.58*text_score + 0.14*line_lcs + 0.10*exact_patch. Final score is mean row score.

What Not To Use
Hardcoded id-to-patch maps
Manual lookup of held-out proof records
Hidden answer metadata outside the public files
GPU usage for training, inference, feature extraction, or search