This is a GPU-oriented computer-vision challenge about recognizing overlapping pictograms and reasoning about which ones are in front. Each row contains one 256 by 256 image made by stacking six anonymous icon instances. The icons partially cover one another, may be rotated or recolored, and are mixed with visual noise.

Your task is to output the six candidate aliases that actually appear in the image, ordered from back layer to front layer.

In plain terms: look at a messy pile of icons, decide which six icons from the candidate list are present, and recover the hidden stacking order. This is not image classification, scalar regression, or ordinary object detection. A valid answer is a short ordered sequence of row-local aliases.

The source artwork comes from OpenMoji. Training and test scenes use disjoint source icons, and public rows do not include source hexcodes, filenames, or stable icon IDs.

Dataset files
train.csv contains 2,160 rows. Each row has:

id: string. Unique training scene ID.
image_path: string. Relative path to the 256 by 256 PNG scene.
scene_card: JSON object. Canvas size, candidate count, present count, and target-order convention.
candidate_cards: JSON list. Twenty-four row-local candidate icon cards.
target_stack: string. Training-only answer: six aliases ordered back to front.
test.csv contains 900 rows. It has the same public columns as train.csv but omits target_stack.

sample_submission.csv contains every test ID with an empty dummy prediction. It is structurally valid and scores 0.

Generated image files are stored under:

images/train/: training PNG scenes.
images/test/: test PNG scenes.
Field schemas
scene_card is a JSON object with:

canvas_size: list of two integers. Always [256, 256].
candidate_count: integer. Always 24.
present_count: integer. Always 6.
target_order: string. Always back_to_front.
output_separator: string. States that aliases should be separated by single spaces.
Each candidate_cards item is a JSON object with:

alias: string. Row-local candidate alias such as I03.
group_hint: string. Broad visual group hint, such as animals-nature, objects, or symbols.
subgroup_hint: string. More specific source subgroup hint.
label_hint: string. Short source-derived natural-language label.
tag_hint: string. A short comma-separated tag hint.
Aliases are row-local. I03 in one row has no relationship to I03 in another row.

target_stack, present only in train.csv, is a space-separated sequence of exactly six aliases. The first alias is the back-most rendered icon. The last alias is the front-most rendered icon.

Example target:


I14 I03 I21 I08 I17 I02


## **Task**

For each test image, submit `predicted_stack`: six candidate aliases ordered from back layer to front layer.

Valid prediction example:

I14 I03 I21 I08 I17 I02


The grader also accepts a JSON list of aliases such as:

["I14","I03","I21","I08","I17","I02"]


Predictions with duplicate aliases, unknown aliases, invented aliases, more than 12 aliases, empty strings, or malformed JSON score 0 for that row.

## **Evaluation**

Structurally invalid submission files are rejected. Examples are missing columns, extra columns, duplicate IDs, unknown IDs in full-answer grading, missing IDs, or wrong column order. The grader aligns rows by ID, not row order.

For each valid row:

- `PresentF1` is F1 between the predicted alias set and the six true present aliases.
- `OrderedPairF1` is F1 over all ordered alias pairs. For a sequence `A B C`, the ordered pairs are `(A,B)`, `(A,C)`, and `(B,C)`.
- `AdjacentLinkF1` is F1 over adjacent ordered pairs. For `A B C`, the adjacent pairs are `(A,B)` and `(B,C)`.
- `ExtremeAccuracy` gives half credit for identifying the two back-most aliases and half credit for identifying the two front-most aliases.
- `ExactStack` is 1 if the full sequence exactly matches the target stack, else 0.

The row score is:

row_score = 0.34 * PresentF1

0.26 * OrderedPairF1
0.20 * AdjacentLinkF1
0.10 * ExtremeAccuracy
0.10 * ExactStack

The 900 hidden rows are balanced across six private rendering families:

- `clear_offset`
- `radial_overlap`
- `tight_occlusion`
- `low_contrast`
- `small_dense`
- `edge_clipped`

Each family has 150 hidden rows. Family labels are private and are used only for subgroup-robust scoring.

The final score is:

final_score = 0.72 * mean(row_score over all rows)

0.18 * worst_family_mean
0.10 * bottom_20_percent_mean

Scores are finite and bounded in `[0, 1]`. The sample submission scores 0. A perfect oracle scores 1.

Participants can compute training metrics locally, but cannot exactly compute hidden `worst_family_mean` or bottom-tail behavior without the private answers.

## **Submission format**

Submit a CSV with exactly two columns in this order:

1. `id`: string. Test row ID.
2. `predicted_stack`: string. Space-separated aliases, back to front.

CSV example:

id,predicted_stack 0a12bc34de56f789,I14 I03 I21 I08 I17 I02


## **Recommended solution approach**

A simple approach can train a CNN or vision transformer to score image/candidate pairs, then use a second model or heuristic to infer layer order from occlusion patterns. Stronger approaches should combine visual-text matching, candidate-set calibration, and pairwise front/back reasoning. The intended hardware is one NVIDIA A10G-class GPU within a 90-minute solution budget.

## **What not to use**

- Do not use row IDs, row order, alias numbers, or candidate-card order as semantic signals.
- Do not assume all candidate aliases are present. Exactly 6 of 24 candidates are present.
- Do not treat this as single-label classification; every row requires a six-item ordered sequence.
- Do not search for source hexcodes, filenames, or exact source IDs. These are intentionally absent from solver-facing files.
- Do not submit bounding boxes, probabilities, natural language, image files, or extra columns.
- Do not optimize only easy separated scenes. Worst-family and bottom-tail scoring penalize failure on tight, low-contrast, small, and clipped scenes.

## **Benchmark boundary**

This differs from ordinary emoji classification and standard object detection. The solver is not asked to name one icon or draw boxes for a fixed class vocabulary. This is a completely original and highly novel task. The row contains a new candidate set, source-disjoint icons, heavy overlap, and a hidden z-order. The benchmark measures candidate grounding plus occlusion-aware sequence recovery.

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.