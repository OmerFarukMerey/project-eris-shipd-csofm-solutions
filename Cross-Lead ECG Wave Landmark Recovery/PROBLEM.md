Overview
Generate the ordered P-wave, QRS-complex, and T-wave onset, peak, and offset landmarks for one missing target lead over a two-second window. The .npz file contains one transformed context lead; target_lead_token identifies the permanently withheld lead for that row. A patient contributes up to eleven target rows distributed over five non-overlapping context windows, and every target lead occurs in only one window. Rows sharing a context window ask about different withheld leads and never expose a target waveform. Only rows containing at least six official target landmarks are retained.

Every retained row from one patient is kept wholly on one side of the allocation boundary. The task requires lead-specific timing transfer and morphology priors from a single observed channel, emitted as a variable-length ordered landmark sequence.

Dataset
train.csv: 1,298 labeled rows.
test.csv: 703 unlabeled rows.
sample_submission.csv: 703 deterministic schema-valid random predictions illustrating the required CSV serialization.
signals/: NPZ signal files referenced by the CSV rows.
All retained withheld-target rows from one patient record are kept on the same side of the allocation boundary.

Column Definitions
id (string; opaque patient-target-window row id)
signal_file (string; relative path to an NPZ file containing an int16 signal array of shape [1000,1] and a 1-item lead_tokens array)
target_lead_token (string; anonymized identity of the withheld target lead)
sample_rate_hz (integer; 500 for every row)
sample_count (integer; 1000 for every row)
prompt (string; task instruction)
answer_format_json (JSON object; event fields and allowed ranges)
answer_json (JSON object in train.csv only; ordered landmark events)
Prediction Object
answer_json is {"events":[...]}. Each event contains integer sample, wave in P|QRS|T, and landmark in onset|peak|offset. Events must be unique and sorted by sample.

Submission Format
Submit a UTF-8 CSV with exactly id and answer_json, in either column order, and exactly one row for every test id. IDs must be unique and match the test ids exactly. The examples below demonstrate serialization; their aliases are illustrative.

id,answer_json  
ecg_example_01,"{""events"":[{""sample"":641,""wave"":""QRS"",""landmark"":""onset""},{""sample"":664,""wave"":""QRS"",""landmark"":""peak""},{""sample"":690,""wave"":""QRS"",""landmark"":""offset""}]}"  
ecg_example_02,"{""events"":[{""sample"":750,""wave"":""P"",""landmark"":""onset""}]}"  

Malformed JSON or a row that violates its schema receives zero for that row. Grading continues for the remaining rows. Missing or extra columns, a wrong row count, duplicate ids, or a mismatched id set reject the complete submission.

Evaluation
For a tolerance d, events are matched by maximum-cardinality one-to-one bipartite matching only when wave and landmark are exact and sample distance is at most d. Precision is matched events divided by submitted events, recall is matched events divided by true events, and F1_d = 2PR/(P+R). order_similarity is the LCS length of the submitted and true (wave, landmark) sequences divided by the longer sequence length.

row_score = 0.60 * F1_8 + 0.30 * F1_20 + 0.10 * order_similarity

Eight and twenty samples correspond to 16 ms and 40 ms at 500 Hz. The final score is the arithmetic mean over all evaluation rows.

Scores range from 0 to 1, and higher is better. A completely correct submission scores 1.

Expected Methods
CPU wavelets, filters, derivative-energy features, lead-conditioned timing transfer, dynamic programming, template models, and compact 1D sequence models.

What Not To Use
GPU, TPU, Metal, CUDA, ROCm, or any other accelerator for training, inference, feature extraction, or search
Hosted APIs, remote inference services, or network access during solution execution
Runtime package installation, downloaded code, vendored external code, or remote-code loaders
External datasets, external answer tables, source-record lookup, or challenge-specific pretrained checkpoints
Hardcoded mappings from row ids, asset names, aliases, or exact evidence records to answers
Manual labeling of evaluation rows
Solutions must operate on the supplied files with CPU resources only. General-purpose libraries and public general-purpose pretrained weights already present in the execution environment are allowed when they run entirely on CPU and were not trained specifically for this dataset.