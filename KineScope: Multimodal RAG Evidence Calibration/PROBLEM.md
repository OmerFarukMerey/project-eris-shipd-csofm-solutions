Retrieval-augmented systems that answer questions about long field recordings often retrieve compact event representations rather than raw frames. The difficult part is not merely finding something visually similar: the system must decide whether the retrieved evidence supports the temporal relation implied by the query.

KineScope is an LLM Evaluation / multimodal RAG evidence-grounding benchmark. Each row contains two anonymized event representations: a query event in the left_* block and a retrieved evidence event in the right_* block. Predict a continuous support score for the query's comparative claim that its event is more persistent than the retrieved event.

This is not ordinary object recognition and not a generic independent-row tabular task. Evaluation acquisition sessions are disjoint from training sessions, so a useful evaluator must learn a transferable query-evidence relation rather than memorize a recording, event, or retrieval position.

Objective
For every row in test.csv, predict one floating-point prediction in [0, 1]:

values near 0: the retrieved evidence contradicts the claim because the query event is much less persistent;

values near 0.5: the evidence is neutral because the events have similar persistence;

values near 1: the retrieved evidence strongly supports the claim because the query event is much more persistent.

The score is continuous: both ordering and support magnitude matter. Calibration uses a training-only reference distribution; hidden targets never define the scale.

Why this is a RAG evaluation task
The left block is the query representation and the right block is the retrieved candidate representation. The submitted value evaluates the candidate's evidential support for a fixed comparative answer, making this a learned multimodal RAG re-ranker/evaluator. No text generation is required, and the output is not an event class or source identity.

Each released occurrence is a private projection of temporal visual statistics. The query and evidence blocks use the same anonymous coordinate system, but every occurrence has independent nuisance variation. Exact-vector matching cannot reconnect occurrences or identify their parent observations.

Grounding targets combine temporal extent with motion morphology, are normalized within protected acquisition sessions, and are converted to a continuous training-referenced percentile scale. Session averages, camera signatures, a global prior, and duration-only rules are insufficient. Pair order is meaningful: swapping query and evidence reverses the relation. CSV order, identifier spelling, numeric formatting, and vector norms carry no target information.

Dataset Split
Protected acquisition sessions are assigned before any query-evidence pair or occurrence is generated. No source event or protected session crosses train/evaluation or public/private boundaries.

The prepared release contains:

3,630 labelled training query-evidence pairs;

800 evaluation pairs;

255 public and 545 private evaluation pairs;

equal sampling across five broad support bands in training, public, and private partitions.

The evaluation-to-training row ratio is approximately 22%. Every evaluation pair belongs to exactly one visibility partition.

Files
train.csv: id, 64 left_* query features, 64 right_* retrieved-evidence features, and continuous target in [0,1].

test.csv: the same identifiers and features without target.

sample_submission.csv: the exact required id,prediction schema with a varied label-free example.

data_manifest.json: dimensions, target range, neutral baseline, and release row counts.

Feature columns are ordered left_0 through left_63, followed by right_0 through right_63. All feature values are finite floating-point numbers. id is an opaque record key, not a feature.

Modelling Guidance
A strong baseline compares query and evidence with antisymmetric and symmetric features such as left - right, abs(left - right), and coordinate interactions, then learns support calibration from train.csv. Nonlinear pair encoders, Siamese networks, and ensembles can capture additional cross-block interactions.

Validation should hold out complete groups of related training rows. Random row validation can be optimistic because related rows may share acquisition conditions.

Evaluation
The leaderboard uses normalized RMSE skill relative to the neutral evidence prediction 0.5:


RMSE      = sqrt(mean((prediction - target)^2))

null_RMSE = sqrt(mean((0.5 - target)^2))

score     = max(0, 1 - RMSE / null_RMSE)

Scores lie in [0, 1], higher is better, and exact answers score 1.0. Predicting 0.5 everywhere scores exactly 0.0 on full, public, and private answers. Valid but worse predictions are clipped to 0.0.

The grader aligns rows by id, so submission row order has no effect. The same formula is applied independently to public and private answer subsets.

Submission Format
Submit a CSV with exactly these columns in this order:


id,prediction

0123456789abcdefabcd,0.73

abcdef0123456789abcd,0.18

Every test ID must appear exactly once. Predictions must be finite numeric values in [0,1]. Wrong or reordered columns, missing rows, duplicate IDs, unknown IDs, non-numeric values, non-finite values, and out-of-range values raise a structural validation error.

Restrictions
Use only the released challenge files.

Do not use external media, labels, datasets, APIs, pretrained event models, hosted embeddings, or challenge-specific checkpoints.

Do not attempt to identify parent recordings, source events, acquisition sessions, or the upstream dataset.

Do not exploit IDs, row order, hashes, CSV byte layout, numeric formatting, repeated-submission probing, or leaderboard feedback.

Do not adapt a model using hidden labels or manually label evaluation rows.

Generate the final submission automatically from supplied training pairs and evaluation features.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.