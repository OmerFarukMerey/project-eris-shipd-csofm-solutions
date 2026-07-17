Anonymized Vocal Fragment Routing
Overview
Each row is a pointer-style symbolic music reconstruction problem. You are given a verified vocal context before an omitted passage, a verified vocal context after it, and a shuffled bank of anonymous event fragments. Your task is to output the ordered route through the fragment bank that reconstructs the omitted passage.

In plain terms: choose which fragments belong in the missing vocal passage and put them in the correct musical order.

This is not free melody continuation. The answer is a sequence of fragment aliases such as F07 F02 F14, not a generated melody string. The fragment bank contains all true events plus plausible distractors drawn from the same source-score distribution. Train and test examples are split by source score, so hidden test rows come from scores unseen during training.

Task
Submit predicted_route, a space-separated list of fragment aliases. Each alias must appear in that row's fragment_bank. A fragment may be used at most once.

The selected fragments, in your submitted order, define the reconstructed omitted passage.

Input representation
prefix_events and suffix_events are trusted context sequences. Each event token is either a row-local note or a rest:

| token form | meaning |
|---|---|
| `N+03:4` | Note 3 semitones above the row-local reference pitch, lasting 4 sixteenth-note units. |
| `N-07:2` | Note 7 semitones below the row-local reference pitch, lasting 2 sixteenth-note units. |
| `R:4` | Rest lasting 4 sixteenth-note units. |

fragment_bank is a JSON list. Each item has:

| field | type | description |
|---|---|---|
| `fragment` | string | Row-local alias such as `F03`. |
| `event` | string | The event token carried by this fragment. |
| `kind` | string | `N` for note or `R` for rest. |
| `duration` | integer | Duration in sixteenth-note units. |
| `pitch_band` | string | Coarse row-local pitch band: `low`, `mid`, `high`, or `rest`. |

span_units gives the total duration of the omitted passage. It is a constraint, not a complete solution, because many fragment subsets can match the same duration.

Files
train.csv

| column | type | description |
|---|---|---|
| `id` | string | Unique training row ID. |
| `prefix_events` | string | Ten trusted event tokens before the omitted passage. |
| `suffix_events` | string | Ten trusted event tokens after the omitted passage. |
| `fragment_bank` | JSON list string | Shuffled candidate fragments for this row. |
| `span_units` | integer | Total hidden passage duration in sixteenth-note units. |
| `fragment_count` | integer | Number of fragments in the bank. |
| `target_route` | string | Training-only ordered route through true fragments. |

test.csv has the same columns except target_route.

sample_submission.csv

undefined

column	type	description
id	string	Test row ID.
predicted_route	string	Space-separated fragment aliases.

## Example

Input fields:

prefix_events = N+02:4 N+00:2 N-02:2 R:2 N+03:4 N+05:2 N+03:2 N+00:4 N-02:4 R:4 suffix_events = N-03:4 N-02:2 N+00:2 N+02:4 N+00:4 R:4 N-02:2 N-03:2 N-05:4 N-03:4 span_units = 14 fragment_bank = [ {"fragment":"F01","event":"N-05:4","kind":"N","duration":4,"pitch_band":"mid"}, {"fragment":"F02","event":"N+05:2","kind":"N","duration":2,"pitch_band":"mid"}, {"fragment":"F03","event":"R:8","kind":"R","duration":8,"pitch_band":"rest"}, {"fragment":"F04","event":"N+00:4","kind":"N","duration":4,"pitch_band":"mid"}, {"fragment":"F05","event":"N+03:2","kind":"N","duration":2,"pitch_band":"mid"}, {"fragment":"F06","event":"N-03:6","kind":"N","duration":6,"pitch_band":"mid"} ]


A valid route might be:

F04 F02 F05 F06


This route reconstructs the event sequence `N+00:4 N+05:2 N+03:2 N-03:6`, whose total duration is 14.

## Evaluation

Invalid row predictions receive 0 for that row. Structural submission errors are rejected.

For each row, the grader maps your aliases through that row's hidden/private copy of `fragment_bank`, obtains the predicted event sequence, and compares it with the hidden canonical route.

Components:

component	description
AliasSetF1	F1 over selected fragment aliases, ignoring order.
PositionAlias	Fraction of route positions with the exact correct alias.
OnsetTypeF1	F1 over (onset_time, event_type) pairs.
ExactEventF1	F1 over (onset_time, event_type, pitch_offset, duration) tuples.
FramePitchScore	Sixteenth-frame pitch/rest agreement; exact note pitch gets 1.0, pitch within two semitones gets 0.4.
ContourAgreement	F1 over up/same/down contour signs between consecutive selected note events.
DurationTotal	max(0, 1 - abs(predicted_total_duration - true_total_duration) / true_total_duration).

The row score is:

row_score = 0.18 * AliasSetF1

0.17 * PositionAlias
0.15 * OnsetTypeF1
0.22 * ExactEventF1
0.15 * FramePitchScore
0.08 * ContourAgreement
0.05 * DurationTotal

The final score is:

final_score = mean(row_score over all test rows)


The score is bounded in `[0, 1]`. The sample submission scores 0. A perfect oracle scores 1.

## Submission format

The submission must have exactly two columns in this order:

column	type	description
id	string	Test row ID from test.csv.
predicted_route	string	Space-separated fragment aliases, for example F04 F02 F05 F06.

CSV example:

id,predicted_route 0a12bc34de56f789,F04 F02 F05 F06


## What not to use

Do not use source score IDs, composer names, repository paths, row order, or external source lookup. These are absent from solver-facing rows.

Do not treat the fragment aliases as meaningful across rows. `F04` in one row has no relationship to `F04` in another row.

Do not solve by duration alone. Many distractor subsets are duration-compatible.

Do not submit event tokens, JSON programs, natural language, MIDI, MusicXML, or rendered images. Submit only the route aliases.

## Benchmark boundary and originality

This benchmark is not ordinary melody continuation, generic infilling, masked-token prediction, audio transcription, optical music recognition, or classification.

The distinctive structure is anonymous fragment routing: the model must select and order a subset of row-local event fragments under melodic-context and duration constraints. The output space is a variable-length pointer sequence over aliases, and scoring jointly rewards alias selection, exact ordering, rhythmic onset reconstruction, event identity, frame-level pitch/rest behavior, contour, and duration.

A solver that only copies bank order, only matches total duration, or only chooses common note durations should score poorly. Strong solutions need learned sequence modeling over real vocal-line contexts plus combinatorial search over each row's candidate bank.