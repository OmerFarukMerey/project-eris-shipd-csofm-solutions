Vector Fragment Route Program Repair
Overview
Each row contains a small 128 by 128 vector drawing that has been broken into anonymous fragments. The fragment geometry is still visible, but the route metadata that says how the fragments should be connected has been lost.

Your job is to reconstruct that missing route metadata. For every drawing, submit a JSON program that says:

which fragment starts the route;
which fragment follows each previous fragment;
whether each visible fragment should be used as shown or reversed.
This is a structured sequence-to-program challenge over vector geometry. It is not image classification, OCR, handwriting recognition, or a source lookup task. The public rows are generated mosaics with row-local aliases, so there is no character label, filename, or source identifier to recover.

Task
For each test row, submit one JSON object in the predicted_program column.

A valid program has exactly three keys:

{
  "start": "F4",
  "links": [["F4", "F2"], ["F2", "F9"], ["F9", "F1"]],
  "orientations": {"F4": "+", "F2": "-", "F9": "+", "F1": "-"}
}

Meaning:

start is the first row-local fragment alias.
links is a list of directed successor edges. ["F4","F2"] means F2 follows F4.
orientations maps every fragment alias to + or -.
+ means traverse the fragment points in the order shown in fragment_cards.
- means traverse the fragment points in reverse order.
The links must form one simple path that visits every fragment exactly once.

The hidden route is not simply left-to-right, top-to-bottom, or public-card order. Mosaics are generated from multiple spatial components, and the canonical route braids across components. A solution that only sorts fragments by position receives limited credit.

Files and columns
train.csv
&nbsp;

column	type	description
id	string	Unique training row ID.
canvas_size	integer	Coordinate canvas size. Always 128.
fragment_count	integer	Number of fragment cards in the row.
component_count	integer	Number of source components used to form the mosaic.
layout_hint	string	Coarse mosaic layout family: left_right, top_bottom, outer_inner, center_sides, or diagonal.
fragment_cards	JSON list string	Shuffled vector fragment cards. Each card has fragment, points, bbox, and shape_tag.
target_program	JSON object string	Training-only canonical route program with start, links, and orientations.

### `test.csv`


 

| column | type | description |
|---|---|---|
| `id` | string | Unique test row ID. |
| `canvas_size` | integer | Coordinate canvas size. Always `128`. |
| `fragment_count` | integer | Number of fragment cards in the row. |
| `component_count` | integer | Number of source components used to form the mosaic. |
| `layout_hint` | string | Coarse mosaic layout family. |
| `fragment_cards` | JSON list string | Shuffled vector fragment cards. The route program is hidden. |

sample_submission.csv
&nbsp;

column	type	description
id	string	Test row ID.
predicted_program	stringified JSON object	Submitted route program.

## Fragment card schema

Each item inside `fragment_cards` is a JSON object:

&nbsp;

field	type	description
fragment	string	Row-local alias such as F7. Alias numbers are randomized and do not encode route position.
points	list[list[integer]]	Simplified vector control points on the 128 by 128 canvas. The listed order may be forward or reversed relative to the hidden route.
bbox	list[integer]	[x_min, y_min, x_max, y_max] bounding box for the visible points.
shape_tag	string	Coarse geometry tag: horizontal, vertical, sweep, fall, dot, or bent.

Example fragment card:

&nbsp;

{"fragment":"F4","points":[[17,57],[17,56],[18,54],[50,83]],"bbox":[17,53,50,83],"shape_tag":"bent"}


## Target and submission program schema

`target_program` in training data and `predicted_program` in submissions use the same schema:

&nbsp;

key	type	requirement
start	string	Must be exactly one fragment alias from the row.
links	list[list[string,string]]	Must contain exactly fragment_count - 1 directed edges.
orientations	object	Must map every row alias exactly once to + or -.

Additional JSON keys are invalid. Missing aliases, duplicate aliases, duplicate successors, duplicate predecessors, cycles, disconnected paths, unknown aliases, malformed JSON, and empty programs score 0 for that row.

## Evaluation

Invalid row programs receive 0 for that row. Structural submission errors are rejected.

For a valid row, the grader converts the route program into an ordered alias list by starting at `start` and following `links`.

Define:

- `T = [t1, t2, ..., tn]` as the true fragment route.
- `P = [p1, p2, ..., pn]` as the predicted fragment route.
- `true_sign(a)` as the true `+` or `-` orientation for alias `a`.
- `pred_sign(a)` as the submitted `+` or `-` orientation for alias `a`.
- `pos_P(a)` as the 1-based position of alias `a` in `P`.

`PairwisePrecedence` measures how many true before/after relationships are preserved:

PairwisePrecedence = number of true alias pairs (ti, tj) with i < j and pos_P(ti) < pos_P(tj) / (n * (n - 1) / 2)


`AdjacentLinkF1` measures exact local successor recovery:

TrueLinks = {(t1,t2), (t2,t3), ..., (t(n-1),tn)} PredLinks = {(p1,p2), (p2,p3), ..., (p(n-1),pn)}

precision = |TrueLinks intersect PredLinks| / |PredLinks| recall = |TrueLinks intersect PredLinks| / |TrueLinks|

AdjacentLinkF1 = 0 if precision + recall = 0 otherwise 2 * precision * recall / (precision + recall)


`LCS` gives partial credit for preserving a long ordered subsequence:

LCS = length(longest common subsequence of P and T) / n


`OrientationAccuracy` scores the forward/reverse decisions:

OrientationAccuracy = number of aliases a where pred_sign(a) = true_sign(a) / n


`ExactProgram` rewards complete route recovery:

ExactProgram = 1 if P equals T and every orientation sign is correct, else 0


The row score is:

row_score = 0.08 * PairwisePrecedence

0.42 * AdjacentLinkF1
0.08 * LCS
0.07 * OrientationAccuracy
0.35 * ExactProgram

The final score is:

mean_score = mean(row_score over all test rows) tail_score = mean(row_score over the lowest-scoring 20% of test rows) long_score = mean(row_score over rows with at least 10 fragments)

final_score = 0.70 * mean_score

0.20 * tail_score
0.10 * long_score

The score is bounded in `[0, 1]`. The sample submission scores 0. A perfect oracle scores 1.

## Submission format

&nbsp;

The submission must have exactly two columns in this order:

column	type	description
id	string	Test row ID from test.csv.
predicted_program	stringified JSON object	Route program with start, links, and orientations.

CSV example:

&nbsp;

id,predicted_program 0a12bc34de56f789,"{""start"":""F4"",""links"":[[""F4"",""F2""],[""F2"",""F9""],[""F9"",""F1""]],""orientations"":{""F4"":""+"",""F2"":""-"",""F9"":""+"",""F1"":""-""}}"


## What not to use

Do not assume aliases encode route position. `F1` is not necessarily first.

Do not use row order or ID hashes. IDs are arbitrary and do not encode targets.

Do not submit a plain sequence, comma-separated list, natural language answer, SVG, image, or JSON with extra keys.

Do not treat a row as exact source lookup. Public rows are anonymized mosaics and do not contain source filenames, source identifiers, character labels, or repository paths.

Do not ignore orientation signs. Orientation is a scored part of the route program.

Do not assume the route is a plain spatial sort. The canonical path usually alternates between components, so a position-only ordering is intentionally incomplete.

## Benchmark boundary and originality

This challenge is a highly novel and original route-program repair benchmark for shuffled vector fragments. It is deliberately different from common visual or drawing tasks.

A strong solution should learn component-level layout conventions, local geometric continuity, pairwise successor prediction, orientation cues, and constrained graph decoding over variable-size fragment sets.