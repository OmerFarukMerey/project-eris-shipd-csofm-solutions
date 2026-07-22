Problem Description
Overview
Generate the exact ordered normalized-word span omitted from one phase of a curated multi-step procedure. Each row supplies readable procedure context, prerequisite and consequence terms, the phase sequence, and masked_segments for the target step. A visible segment contains normalized words; a gap contains only its index and length. target_gap_index identifies which hidden span to generate. When one source step supplies two rows, both gaps remain hidden in both rows, preventing sibling-row answer leakage.

Common function words are absent and inflected forms are conservatively normalized, so the output is a compact content-word sequence rather than verbatim prose. Related procedures and exact duplicate source steps remain together across allocation.

Dataset
train.csv: 766 labeled rows.
test.csv: 413 unlabeled rows.
sample_submission.csv: 413 deterministic schema-valid random predictions illustrating the required CSV serialization.
All catalog records connected by any parent-child relation, plus records sharing an exact procedure-step text, are kept on the same side of the allocation boundary.

Column Definitions
id (string; opaque row id)
prompt (string; task instruction)
flow_context_json (JSON object; pattern_context_tokens, prerequisite_tokens, consequence_tokens, phase_sequence, and target_step with flow_position, phase, masked_segments, target_gap_index, and missing_token_count)
missing_token_count (integer; exact number of normalized words to generate)
answer_format_json (JSON object; root name, token pattern, and required count)
answer_json (JSON object in train.csv only; ordered missing normalized words)
Prediction Object
answer_json is {"missing_tokens":[...]}. The list length must equal missing_token_count, order matters, and every item must match [a-z][a-z0-9_]{0,31}.

Submission Format
Submit a UTF-8 CSV with exactly id and answer_json, in either column order, and exactly one row for every test id. IDs must be unique and match the test ids exactly. The examples below demonstrate serialization; their identifiers and values are illustrative.

id,answer_json  
flow_example_01,"{""missing_tokens"":[""credential"",""access""]}"  
flow_example_02,"{""missing_tokens"":[""send"",""request"",""server""]}"  

Malformed JSON or a row that violates its schema receives zero for that row. Grading continues for the remaining rows. Missing or extra columns, a wrong row count, duplicate ids, or a mismatched id set reject the complete submission.

Evaluation
position_accuracy is the fraction of list positions containing the exact true token. token_F1 is duplicate-aware multiset F1: common token multiplicity is summed from the intersection of token counters, precision is common multiplicity divided by submitted length, and recall is common multiplicity divided by true length. token_LCS is longest-common-subsequence length divided by the true length. exact_span is 1 only when the entire ordered list matches.

row_score = 0.55*exact_span + 0.25*position_accuracy + 0.10*token_F1 + 0.10*token_LCS

The final score is the arithmetic mean over all evaluation rows.

Scores range from 0 to 1, and higher is better. A completely correct submission scores 1.

Expected Methods
CPU sparse retrieval, sequence alignment, phase-aware reranking, compact token models, and schema-constrained decoding.

What Not To Use
GPU, TPU, Metal, CUDA, ROCm, or any other accelerator for training, inference, feature extraction, or search
Hosted APIs, remote inference services, or network access during solution execution
Runtime package installation, downloaded code, vendored external code, or remote-code loaders
External datasets, external answer tables, or challenge-specific pretrained checkpoints
Hardcoded mappings from row ids, asset names, aliases, or exact evidence records to answers
Manual labeling of evaluation rows
Solutions must operate on the supplied files with CPU resources only. General-purpose libraries and public general-purpose pretrained weights already present in the execution environment are allowed when they run entirely on CPU and were not trained specifically for this dataset.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.