Complaint operations often need to reconcile repeated unwanted-contact reports after direct caller identifiers have been removed. Individual complaints are weak evidence; useful matching depends on agreement across contact method, call type, geography, service category, reporting cadence, and filing behavior.

Each case contains a two-complaint anchor bundle and eight anonymous candidate profiles. Every profile contains three complaints. Exactly one profile shares the anchor's hidden reported caller identifier. Return that profile's three complaint-card IDs in chronological order.

The target is an observed reported-caller thread. It does not assert that a phone number belongs to one verified organization or that every report describes one real-world campaign.

Dataset
The prepared release contains 2,200 training cases and 500 test cases. Each case is based on a distinct normalized reported caller with five source complaints. Caller groups are split before candidate construction; all candidate profiles come from the same partition. Direct caller and advertiser numbers, ticket IDs, cities, ZIP codes, absolute dates, and source row order are not released. Each profile has an independent relative-time origin, preventing calendar proximity from identifying the answer.

train.csv
Column	Type	Description
case_id	string	Unique anonymized training-case identifier.
anchor_bundle	JSON string	Two cards a01 and a02 containing anonymous complaint attributes.
profile_bank	JSON string	Eight profiles p01 through p08, each containing three local complaint cards.
test.csv
test.csv has the same three columns and data types as train.csv.

train_labels.csv
Column	Type	Description
case_id	string	Joins to train.csv.
continuation_chain	JSON string	Three card IDs from the correct profile in chronological order.
sample_submission.csv
A schema-correct example containing every test case_id and a placeholder continuation chain.

Complaint cards contain relative filing day and hour, filing weekday and hour bucket, issue-time bucket, filing-lag bucket, contact method, call type, service category, state, and whether an advertiser number was reported.

The records are public consumer allegations and may be incomplete, duplicated, mistaken, or based on spoofed caller information. The benchmark supports anonymous complaint reconciliation research, not enforcement decisions, attribution of wrongdoing, reverse identification, or consumer risk scoring.

Evaluation
ProfileAccuracy is one only when all submitted cards come from the correct profile. ComplaintSetF1 scores membership. DirectedChainF1 compares chronological edges after adding ANCHOR and END nodes.

Score = 0.55 * ProfileAccuracy + 0.20 * ComplaintSetF1 + 0.20 * DirectedChainF1 + 0.05 * ExactChainAccuracy
All components are macro-averaged over cases. Malformed rows receive zero. Scores are in [0,1]; higher is better.

Submission
Submit a CSV with exactly these columns:

Column	Type	Description
case_id	string	Test case identifier.
continuation_chain	JSON string	Three distinct profile-card IDs in chronological order.
case_id,continuation_chain UC_TE_0123456789abcdef,"[""p04_c02"",""p04_c01"",""p04_c03""]"
Valid card IDs range from p01_c01 through p08_c03. Missing or duplicate case IDs invalidate a submission. Extra rows are ignored for an evaluation slice. Invalid JSON, unknown IDs, repeated IDs, or a list with the wrong length makes that row malformed.

Allowed Methods
Recommendation, metric-learning, set encoders, pair ranking, and constrained decoding.
Models trained only from released challenge files.
Deterministic categorical and relative-time feature engineering.
Prohibited Methods
Reverse-number lookup or searching any external complaint, caller, business, or identity database.
Reconstructing hidden caller numbers, advertiser numbers, ticket IDs, cities, ZIP codes, or absolute dates.
Manual test labeling or sharing hidden test predictions.