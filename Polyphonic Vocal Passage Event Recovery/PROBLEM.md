Overview
Generate the exact ordered vocal events missing from a short symbolic score passage. Each row shows six vocal events before the gap, six after it, and accompaniment chords aligned around the same time range. The missing passage contains four to seven source-annotated notes or rests. A prediction must recover onset, duration, pitch or rest, tie state, and the target voice token for every event.

The target is a complete conditional event sequence. Its difficulty comes from phrase continuation, rhythmic placement, harmony, nonlocal repetition, and held-out composers.

Dataset
train.csv: 2,985 labeled rows.
test.csv: 1,605 unlabeled rows.
sample_submission.csv: 1,605 deterministic schema-valid random predictions illustrating the required CSV serialization.
All passages from the same composer are kept on the same side of the allocation boundary.

Column Definitions
id (string; opaque row identifier)
prompt (string; task instruction)
score_context_json (JSON object; target_before, target_after, accompaniment_chords, missing_event_count, and target_voice_token)
missing_event_count (integer; exact number of events that must be generated)
answer_format_json (JSON object; required root, event fields, and rest encoding)
answer_json (JSON object in train.csv only; ordered events list)
Prediction Object
answer_json is {"events":[...]}. Every event has integer onset_tick, positive integer duration_tick, integer pitch (-1 means rest), tie in none|start|stop|continue, and the row's voice_token. Events are ordered by onset.

Submission Format
Submit a UTF-8 CSV with exactly id and answer_json, in either column order, and exactly one row for every test id. IDs must be unique and match the test ids exactly. The examples below demonstrate serialization; their aliases are illustrative.

id,answer_json  
music_example_01,"{""events"":[{""onset_tick"":0,""duration_tick"":2,""pitch"":67,""tie"":""none"",""voice_token"":""voice_a1b2c3""}]}"  
music_example_02,"{""events"":[{""onset_tick"":0,""duration_tick"":4,""pitch"":-1,""tie"":""none"",""voice_token"":""voice_d4e5f6""},{""onset_tick"":4,""duration_tick"":2,""pitch"":72,""tie"":""start"",""voice_token"":""voice_d4e5f6""}]}"  

Malformed JSON or a row that violates its schema receives zero for that row. Grading continues for the remaining rows. Missing or extra columns, a wrong row count, duplicate ids, or a mismatched id set reject the complete submission.

Evaluation
Let an event match when pitch, tie, and voice are exact, onset differs by at most 2 ticks, and duration differs by at most 1 tick. event_F1 uses maximum-cardinality one-to-one bipartite matching between submitted and true events. Precision is matched events divided by submitted events, recall is matched events divided by true events, and F1 is 2PR/(P+R). edit_similarity = 1 - (len(P)+len(T)-2*LCS(P,T))/(len(P)+len(T)), where LCS compares complete event records. exact_sequence is 1 only when every event and its order match exactly.

row_score = 0.45 * exact_sequence + 0.35 * event_F1 + 0.20 * edit_similarity

The final score is the arithmetic mean over all evaluation rows.

Scores range from 0 to 1, and higher is better. A completely correct submission scores 1.

Expected Methods
CPU symbolic n-grams, variable-order Markov models, dynamic programming, harmonic features, compact sequence models, and constrained decoders.

What Not To Use
GPU, TPU, Metal, CUDA, ROCm, or any other accelerator for training, inference, feature extraction, or search
Hosted APIs, remote inference services, or network access during solution execution
Runtime package installation, downloaded code, vendored external code, or remote-code loaders
External datasets, external answer tables, source-record lookup, or challenge-specific pretrained checkpoints
Hardcoded mappings from row ids, asset names, aliases, or exact evidence records to answers
Manual labeling of evaluation rows
Solutions must operate on the supplied files with CPU resources only. General-purpose libraries and public general-purpose pretrained weights already present in the execution environment are allowed when they run entirely on CPU and were not trained specifically for this dataset.