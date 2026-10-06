Route and Read: Joint Tool Selection and Readiness
Overview
Each row contains an English or Chinese request and three candidate tool specifications. The candidates describe closely related operations. Predict both which candidate is relevant and whether its arguments are ready to execute.

Objective
For every test row, return the one integer label that jointly identifies the relevant candidate position and its ready/not-ready state.

Labels
Candidate positions are zero-based in the displayed JSON list.

Label	Relevant candidate	Decision
0	0	ready to execute
1	1	ready to execute
2	2	ready to execute
3	0	needs user input or argument repair
4	1	needs user input or argument repair
5	2	needs user input or argument repair
For labels 3--5, the relevant schema has either a missing required value or a supplied value outside its allowed constraint. The prediction is the same in both cases: the call is not ready.

What makes the task difficult
All three tools in a row have the same readiness state. Structural inspection therefore determines only the ready/not-ready half of the label; it cannot reveal which candidate matches the request. The candidates form a deliberately hard same-language semantic neighbourhood, and their names are opaque row-local handles. Request wording and candidate wording are intentionally lexically different, while nearby decoys can share tempting surface terms. Solvers must compare the request meaning with the candidate descriptions and complete schema, then validate the relevant schema.

The test set is a controlled hard-neighbour holdout with class and language coverage checked by the organizers. Hidden test quotas are not disclosed. Complete parent units are indivisible, and exact rendered dialogues, tool sets, feature pairs, and public tool names do not cross the boundary.

Dataset
train.csv: 1,500 labeled rows.
test.csv: 300 unlabeled rows.
sample_submission.csv: required submission shape.
Columns
Columns:

id: opaque row identifier;
language: en or zh;
dialogue: request and supplied arguments;
tool_specifications: JSON list of exactly three candidate schemas;
target: training-only label in {0,1,2,3,4,5}.
Modelling guidance
Parse tool_specifications as JSON and preserve list order. A strong approach first compares the action expressed in dialogue with each candidate's root and nested descriptions, then checks required fields, types, and enum/range constraints for the selected candidate. Raw lengths, row IDs, language, tool count, and schema size are not intended label signals.

Evaluation
The metric is multiclass accuracy. Construction controls prevent any one label from dominating the held-out evaluation, every row has one mutually exclusive decision, and all routing/readiness mistakes have equal operational weight. Accuracy therefore directly measures the fraction of complete joint decisions that are correct. Partial credit would incorrectly reward an unsafe half-decision such as choosing the right tool but missing that its arguments are invalid; class-weighted metrics are unnecessary for this controlled evaluation. Hidden test label quotas are not participant inputs. Scores range from 0 to 1; higher is better. Uniform random guessing has expected accuracy 1/6, or about 0.1667.

The grader aligns by id, accepts integer predictions only in {0,1,2,3,4,5}, and requires the complete test ID set (or the exact hidden answer shard used by the platform).

Submission
Submit exactly two columns in this order:

id,prediction
tut_0123456789abcdef01,4

Include one row for every ID in test.csv.

Intended use and limitations
This is a controlled bilingual benchmark for joint semantic routing and pre-execution readiness. It does not evaluate generated calls, clarification wording, side-effect safety, production tool coverage, or natural class prevalence. Use only the provided public files; private answers, hidden platform metadata, source lookup, and external web services are out of scope.

Restrictions
Do not use private answers, construction artifacts, hidden platform metadata, source lookup, or external web services. Predictions must be based only on the solver-facing public files.

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.