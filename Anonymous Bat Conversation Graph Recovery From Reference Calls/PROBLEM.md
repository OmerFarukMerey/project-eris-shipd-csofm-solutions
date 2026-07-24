Overview
Plain language objective: match anonymous bat callers from reference calls, recover source-documented addressees and interaction contexts, and submit the directed social graph implied by each episode.

You are given short real Egyptian fruit bat vocalization clips arranged into anonymous social episodes. Each episode contains two to four anonymous bats such as Bat_A, Bat_B, and Bat_C. For every anonymous bat, the episode provides reference WAV clips where that bat is known to be the caller. The gallery then contains additional shuffled WAV clips from the same anonymous episode.

Your task is to recover the social call ledger: who called, who the call was directed to when the source supports it, what native interaction context was documented, and what directed caller-to-addressee graph is implied by the gallery. This is not bat-language translation and not ordinary global speaker classification. The anonymous node mapping resets in every episode, so the reference clips are part of the input evidence.

CPU only: solutions must finish within 1.5 hours on 10 CPU cores and 62 GB RAM. Suitable approaches include high-sample-rate log-spectral features, compact Siamese or metric-learning models, 1D CNN/TCN encoders trained from scratch on CPU, nearest-reference scoring, calibrated unknown handling, and a lightweight graph head. Do not require GPUs, hosted APIs, external datasets, runtime downloads, or internet access.

For every test episode, submit:

call_predictions_json: one prediction object per gallery call.
graph_json: directed edge counts and context summaries implied by the episode.
confidence: row-level confidence in [0,1].
What Not To Use / What Not To Do:

Do not use source filenames, source file IDs, original bat IDs, treatment IDs, recording dates, recording channels, archive member order, file sizes, exact timestamps, or source-row lookup.
Do not download, index, or search the upstream bat corpus or annotations to identify hidden test rows.
Do not reduce the task to fixed global bat identity classification, context-only classification, metadata-only classification, or a decorative graph copied from per-call predictions without checking consistency.
Do not invent hunting, translation, semantic-intent, or conversation labels outside the supplied native context taxonomy.
Do not use hosted models, remote audio APIs, proprietary systems, GPU-only dependencies, external labels, runtime downloads, private files, grader internals, hardcoded IDs, or malformed-submission exploits.
Enforcement on invalid approaches: submissions may be rejected before payout if they rely on source lookup, metadata reconstruction, private files, fixed templates, or methods that ignore the episode references and social graph-recovery task.

Task
For each test episode, group the provided gallery calls by episode_id, read the episode's reference_json, and predict a complete row-local social record. The caller field should be one of the anonymous episode nodes, inferred from the reference calls rather than from any global bat name. The addressee field should be one of the same nodes when source evidence supports a directed addressee, or UNKNOWN when the documented source target is unknown or not resolvable. The context field must use the supplied native context vocabulary.

The graph is operational, not decorative: graph_json should summarize the directed caller-to-addressee counts and context counts implied by your per-call predictions. A strong answer keeps the per-call ledger and the graph internally consistent.

Intended Approach
A practical CPU solution can extract high-frequency log-mel or constant-Q style acoustic features from each WAV, learn a compact caller embedding from training episodes, and compare gallery calls to the row-local reference clips with a metric-learning or Siamese-style scorer. Addressee and context heads can use the same acoustic representation plus episode-level priors learned only from the training labels. The graph head can then be constructed from calibrated per-call predictions and checked for consistency.

Reasonable CPU methods include nearest-reference matching with spectral features, gradient-boosted models over bat-call descriptors, compact 1D CNN/TCN encoders trained from scratch on CPU, and lightweight postprocessing for confidence calibration and graph consistency. Use train-only validation folds by episode and real audio/context families to tune thresholds for UNKNOWN, confidence, and graph counts. Do not validate on hidden test answers or external copies of the source corpus.

Dataset
Prepared files:

Item	Description
train.csv	Labeled gallery calls
test.csv	Test gallery calls
train_graphs.csv	Train graph labels
taxonomy.json	Public vocabulary
train/audio/	Train WAV clips
test/audio/	Test WAV clips
sample_submission.csv	Weak valid template
The prepared split contains 36 training episodes and 18 test episodes. The test set is arranged to test generalization across bats, recording conditions, and interaction contexts, while each test episode supplies its own reference calls. Public IDs, audio paths, and row order are opaque. Original filenames, source bat IDs, treatment IDs, recording dates, recording channels, and source sample ranges are not public.

All public audio is mono 16-bit WAV at 100,000 Hz. Clips are source-redacted, label-preserving excerpts derived from 250,000 Hz official WAVs and lightly processed to preserve bat-call structure while reducing direct source fingerprinting.

train.csv columns:

Column	Type	Description
episode_id	string	Opaque episode ID
call_id	string	Opaque gallery call ID
audio_path	path	Gallery WAV path
gallery_position	int	Shuffled position
node_set_json	JSON	Episode node labels
reference_json	JSON	Reference clips
clip_duration_sec	float	Public clip duration
sample_rate_hz	int	Always 100000
caller	string	Train caller label
addressee	string	Train addressee label
context	string	Train native context
test.csv columns:

Column	Type	Description
episode_id	string	Opaque episode ID
call_id	string	Opaque gallery call ID
audio_path	path	Gallery WAV path
gallery_position	int	Shuffled position
node_set_json	JSON	Episode node labels
reference_json	JSON	Reference clips
clip_duration_sec	float	Public clip duration
sample_rate_hz	int	Always 100000
reference_json is a JSON list. Each item has reference_id, bat, audio_path, and duration_sec. The bat field is an anonymous node label valid only inside that episode.

Allowed context labels are listed in taxonomy.json: UNKNOWN_CONTEXT, SEPARATION, BITING, FEEDING, FIGHTING, GROOMING, ISOLATION, KISSING, LANDING, MATING_PROTEST, THREAT_LIKE, GENERAL, and SLEEPING.

Submission
Write ./working/submission.csv with exactly these columns in this order:

The submission CSV must contain exactly one row per test episode (18 rows total).

Column	Type	Constraint
episode_id	string	Exact test episode
call_predictions_json	JSON	One object per call
graph_json	JSON	Directed graph object
confidence	float	In [0,1]
call_predictions_json must be a JSON list with exactly one object for every gallery call_id in that episode. Each object must have exactly:

call_id: a test gallery call ID.
caller: one of the episode nodes.
addressee: one of the episode nodes, or UNKNOWN.
context: one allowed context label.
confidence: a number in [0,1].
graph_json must be a JSON object with exactly edges. Each edge has source, target, count, and contexts. source and target are distinct episode nodes. count is the number of predicted directed calls. contexts maps context labels to counts for that directed edge.

Example:

episode_id,call_predictions_json,graph_json,confidence  
bat_ep_example,"[{""call_id"":""call_a"",""caller"":""Bat_A"",""addressee"":""Bat_B"",""context"":""SLEEPING"",""confidence"":0.62}]","{""edges"":[{""source"":""Bat_A"",""target"":""Bat_B"",""count"":1,""contexts"":{""SLEEPING"":1}}]}",0.62  

Every test episode must appear exactly once. Extra, missing, reordered, or duplicate columns; duplicate, missing, or unknown episode IDs; NaN or infinite row confidence; out-of-range row confidence; unreadable CSVs; or inconsistent row counts raise an invalid-submission error. Row-local malformed JSON, oversized JSON cells, invalid call labels, duplicate call IDs, duplicate graph edges, self-edges, invalid graph counts, or graph/call schema mistakes score zero for the affected episode row rather than crashing the whole submission.

Evaluation
Higher is better. Theoretical minimum: 0.0. Theoretical maximum: 1.0. A perfect private submission with all confidences equal to 1.0 scores exactly 1.0.

For each episode, the grader computes:

caller_accuracy   = mean exact caller match  
addressee_accuracy = mean exact addressee match, including UNKNOWN  
context_accuracy  = mean exact native-context match  
call_core         = 0.45*caller_accuracy + 0.25*addressee_accuracy + 0.30*context_accuracy  

For the submitted graph_json, directed edge counts and edge-context counts are compared with the hidden graph using count F1:

count_f1 = 2 * overlap / (predicted_total + true_total)  
overlap  = sum over keys min(predicted_count, true_count)  
graph_score = 0.65*directed_edge_count_f1 + 0.35*edge_context_count_f1  

The submitted graph must also agree with the submitted per-call predictions:

consistency = 0.65*edge_count_f1(graph, calls) + 0.35*edge_context_f1(graph, calls)  
core = 0.80*call_core + 0.15*graph_score + 0.05*consistency  

Confidence only calibrates earned credit:

call_calibration = mean max(0, 1 - abs(call_confidence - per_call_correctness))  
row_calibration  = max(0, 1 - abs(row_confidence - core))  
episode_score    = core * (0.94 + 0.04*row_calibration + 0.02*call_calibration)  

The final score blends average performance with hidden worst-group robustness over real episode families:

final = 0.78 * mean(episode_score) + 0.22 * mean(worst_group_mean per hidden axis)  

Hidden axes cover episode node-count family, native unknown-addressee presence, and dominant source context family. These groups are not public.

Structural file failures raise an invalid-submission error. Row-local malformed JSON, invalid labels, bad call lists, or invalid graph records score 0.0 for the affected episode row. The grader uses generic messages and does not reveal labels, hidden groups, source IDs, split logic, metric internals, or traceback details.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.