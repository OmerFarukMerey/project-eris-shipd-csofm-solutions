Evidence-Guided Flood Mask Repair
1. Challenge Overview
Evidence-Guided Flood Mask Repair is a structured prediction challenge about auditing and correcting flood extent maps from multimodal satellite and environmental data.

Flood mapping is used in disaster response, infrastructure planning, emergency coordination, agricultural assessment, and post-event damage analysis. Automated flood masks can be useful at large scale, but they often contain localized errors caused by cloud cover, radar noise, terrain effects, missing observations, confusing land surfaces, or imperfect model boundaries.

In this challenge, each case begins with a draft binary flood mask that is already mostly correct.

Participants do not generate an entirely new flood map from scratch.

Instead, they inspect the supplied evidence and submit a sparse repair plan describing:

Pixels that should be added to the flood class.
Pixels that should be removed from the flood class.
Each case provides:

An eight-channel spatial raster.
An eight-channel validity mask.
A draft binary flood mask.
An audit mask identifying the only region where edits are permitted.
A unique opaque case identifier.
The objective is to determine:

Whether the draft flood mask contains an error.
Where the error is located.
Whether flood pixels should be added or removed.
How far the correction should extend.
Which nearby flood structures should remain unchanged.
Whether the correct action is to make no edit.
Some audit regions contain genuine flood-mask defects.

Other audit regions are already correct.

A successful system must therefore learn both intervention and restraint.

2. Real-World Prediction Scenario
The positive class represents flooded land or water-covered terrain associated with a flood event.

The negative class represents areas that should not be labeled as flooded.

The input raster combines information from radar, optical imagery, terrain, and recent precipitation.

These sources provide complementary evidence.

Radar measurements can detect surface-water and moisture-related changes even when optical observations are degraded by clouds.

Optical channels help distinguish water, vegetation, soil, and built surfaces when observations are valid.

Elevation provides terrain context because flood extent is constrained by local topography.

Recent precipitation provides event context because heavy rainfall can increase the likelihood of flooding.

The draft mask represents an existing flood prediction produced before the repair stage.

The participant’s task is to identify localized inconsistencies between that draft prediction and the available evidence.

3. Core Prediction Problem
This benchmark does not ask only which pixels appear flooded.

It asks which parts of an existing flood prediction are inconsistent with the available evidence and what the smallest justified correction is.

The draft mask acts as a strong prior.

Most draft pixels are already correct and should remain unchanged.

The model must compare:

The current draft flood state.
Radar evidence.
Optical evidence.
Observation validity and missingness.
Elevation and terrain structure.
Recent precipitation context.
Local flood boundaries.
Connected flood structures.
The cost of making an unnecessary edit.
This makes the challenge a constrained flood-mask repair task rather than ordinary semantic segmentation.

4. Distinctive Task Structure
Existing Prediction as an Input
The draft flood mask is the initial state of the final output.

The model must determine which portions of that draft should be trusted.

Hard Intervention Boundary
The audit mask defines where edits are permitted.

Pixels outside the audit mask are immutable.

Sparse Directional Output
Participants submit separate addition and removal masks rather than a complete replacement flood mask.

This preserves the direction of every correction.

Explicit No-Change Cases
Some cases require no additions and no removals.

The model must recognize when the draft is already supported by the available evidence.

Preservation Requirement
Correct flood and non-flood pixels inside the audit region must be protected from unnecessary edits.

Structural Evaluation
The metric evaluates boundaries and connected flood components in addition to pixel overlap.

Robustness Aggregation
The final score rewards consistent performance across all cases, including sparse, mixed, structurally difficult, and no-change cases.

5. Inputs
Every case contains aligned raster, draft-mask, and audit-mask inputs.

Multimodal Raster
The raster has shape 8 × H × W.

The channels are:

Channel 0: Sentinel-1 VV radar backscatter.
Channel 1: Sentinel-1 VH radar backscatter.
Channel 2: Sentinel-2 green reflectance.
Channel 3: Sentinel-2 red reflectance.
Channel 4: Sentinel-2 near-infrared reflectance.
Channel 5: Sentinel-2 shortwave-infrared reflectance.
Channel 6: elevation.
Channel 7: seven-day cumulative precipitation.
All channels are normalized to the interval from 0 to 1.

The values are normalized features rather than original physical units.

Channel 0 and Channel 1 describe radar return strength in VV and VH polarization.

Radar backscatter may help separate open water, rough land surfaces, vegetation, and moisture-related changes.

Channel 2 and Channel 3 describe visible green and red reflectance.

These channels provide information about surface color, vegetation, sediment, soil, and water appearance.

Channel 4 describes near-infrared reflectance.

Water commonly has low near-infrared reflectance, while healthy vegetation often has stronger near-infrared response.

Channel 5 describes shortwave-infrared reflectance.

This channel can help distinguish water and wet surfaces from dry soil, vegetation, and built areas.

Channel 6 describes relative terrain elevation after normalization.

Flooding is influenced by terrain shape, low-lying regions, drainage paths, and local topographic barriers.

Channel 7 describes normalized seven-day cumulative precipitation.

Higher recent rainfall may support the presence of flooding, although precipitation alone does not determine the flood boundary.

Validity Mask
Each raster archive also contains a Boolean validity mask with shape 8 × H × W.

A value of True means that the corresponding channel observation is valid.

A value of False means that the observation is missing or invalid.

Invalid raster positions are filled with zero after normalization.

Models should use the validity mask to distinguish genuine normalized zero values from missing observations.

Draft Flood Mask
The draft mask is a binary image with shape H × W.

Pixel meaning:

0 means non-flood.
Any nonzero value means flood.
The draft mask is already mostly correct.

Audit Mask
The audit mask is a binary image with shape H × W.

Pixel meaning:

0 means editing is forbidden.
Any nonzero value means editing is permitted.
The audit mask identifies where the draft should be reviewed.

It does not reveal which pixels are wrong.

6. Repair Operation
Let draft be the supplied draft flood mask.

Let audit be the supplied audit mask.

Let add be the predicted flood-addition mask.

Let remove be the predicted flood-removal mask.

The repaired flood mask is:

repaired = (draft OR add) AND NOT remove

Pixels in add change from non-flood to flood.

Pixels in remove change from flood to non-flood.

All other pixels preserve their draft state.

The addition and removal masks must be disjoint.

7. Validity Constraints
Every predicted addition must satisfy:

audit = 1
draft = 0
Every predicted removal must satisfy:

audit = 1
draft = 1
For every pixel where audit = 0:

add must equal 0.
remove must equal 0.
repaired must equal draft.
Participants should constrain predictions before encoding them.

Conceptually:

Additions are restricted to editable pixels currently labeled non-flood.
Removals are restricted to editable pixels currently labeled flood.
Any overlap between additions and removals must be removed.
8. Possible Draft Defects
A draft may contain one or more localized flood-mapping defects.

Missing Flood Region
A genuinely flooded region is absent from the draft.

False Flood Region
A non-flood region is incorrectly labeled as flooded.

Boundary Expansion
The flood mask extends beyond the boundary supported by the evidence.

Boundary Contraction
A supported portion of the flood extent is missing.

Local Omission
A flooded branch, corridor, shoreline section, low-lying region, or compact component is absent.

False Island
An unsupported isolated flood component is present.

Broken Connectivity
A valid connection between flooded regions is missing.

False Bridge
Two separate flood components are incorrectly connected.

Incorrect Hole Fill
A valid non-flood hole inside or near the flood region is incorrectly filled.

Incorrect Hole Cut
A false non-flood hole is introduced into a valid flood region.

Terrain-Inconsistent Spill
The draft extends uphill or across terrain in a way that is not supported by the available evidence.

Sensor-Confusion Error
The draft follows a pattern caused by radar noise, optical ambiguity, cloud-related missingness, shadows, wet soil, vegetation, or another surface that resembles flood evidence in only some channels.

Mixed Defect
Multiple compatible error types occur in one case.

No-Change Case
The draft flood mask is already correct inside the audit region.

The correct output contains an empty addition mask and an empty removal mask.

9. Why the Task Is Difficult
Most Pixels Are Already Correct
The imbalance is not only flood versus non-flood.

It is also change versus preservation.

True Edits Are Sparse
A large audit region may contain only a small correction.

Addition and Removal Are Asymmetric
Evidence supporting a missing flood region may differ from evidence supporting removal of a false flood region.

Sensor Modalities May Disagree
Radar, optical imagery, terrain, and rainfall may support different interpretations.

Missingness Is Structured
Clouds, invalid observations, or sensor limitations may be concentrated around difficult areas.

Terrain Matters
A visually plausible flood prediction may be inconsistent with elevation and local drainage structure.

Geometry Matters
A repair may have reasonable pixel overlap while producing an incorrect flood boundary.

Connectivity Matters
A repair may create a false bridge, remove a valid connection, split a flood component, or introduce an isolated region.

Correct Regions May Appear Suspicious
Some difficult-looking audit regions should remain unchanged.

10. Public Directory
The released public directory contains:

audit_masks/
draft_masks/
rasters/
sample_submission.csv
test.csv
train.csv
All paths stored in the CSV files are relative to the public directory.

11. Raster Files
The rasters/ directory contains compressed NumPy archives.

Each .npz file contains:

x
valid
x
Type: float32

Shape: 8 × H × W

Values are normalized to the interval from 0 to 1.

Channel order:

Channel 0: Sentinel-1 VV radar backscatter.
Channel 1: Sentinel-1 VH radar backscatter.
Channel 2: Sentinel-2 green reflectance.
Channel 3: Sentinel-2 red reflectance.
Channel 4: Sentinel-2 near-infrared reflectance.
Channel 5: Sentinel-2 shortwave-infrared reflectance.
Channel 6: normalized elevation.
Channel 7: normalized seven-day cumulative precipitation.
valid
Type: Boolean

Shape: 8 × H × W

A value of True means that the corresponding observation is valid.

A value of False means that the observation is unavailable or invalid.

Invalid positions in x are filled with zero.

Models should use valid to distinguish real zero values from missing observations.

12. Draft Masks
The draft_masks/ directory contains binary PNG files.

Each mask has the same height and width as its corresponding raster.

Pixel meaning:

0 means non-flood.
Any nonzero value means flood.
The draft should be interpreted as image_array greater than zero.

13. Audit Masks
The audit_masks/ directory contains binary PNG files.

Pixel meaning:

0 means editing is forbidden.
Any nonzero value means editing is permitted.
The audit mask should be interpreted as image_array greater than zero.

An audit region may contain:

One genuine flood-mask defect.
Multiple defects.
Several plausible-looking repair locations.
Correct flood structure surrounding a defect.
Disconnected candidate regions.
Missing observations.
Contradictory evidence.
No defect.
14. Training Data
train.csv contains:

case_id
raster_path
draft_mask
audit_mask
height
width
add_rle
remove_rle
case_id
A unique opaque identifier for the repair case.

raster_path
The relative path to the compressed raster archive.

Multiple cases may reference the same parent raster.

draft_mask
The relative path to the draft flood-mask PNG.

audit_mask
The relative path to the audit-mask PNG.

height
The image height in pixels.

width
The image width in pixels.

add_rle
Run-length encoding of pixels that must change from non-flood in the draft to flood in the corrected mask.

An empty string means that no flood pixels need to be added.

remove_rle
Run-length encoding of pixels that must change from flood in the draft to non-flood in the corrected mask.

An empty string means that no flood pixels need to be removed.

When both fields are empty, the case is a valid no-change example.

Empty edit fields may appear as blank CSV cells or may be parsed as missing values by some CSV libraries.

Both are interpreted as empty masks.

Participants are encouraged to load CSV files with automatic missing-value conversion disabled so blank RLE values remain empty strings.

15. Test Data
test.csv contains:

case_id
raster_path
draft_mask
audit_mask
height
width
Participants must predict add_rle and remove_rle for every test case.

The hidden corrections are not included.

16. Sample Submission
sample_submission.csv contains:

case_id
add_rle
remove_rle
The supplied sample submission leaves both edit columns empty.

Submitting it unchanged produces the valid copy-draft baseline.

A correctly formatted submission file may look like this:

case_id,add_rle,remove_rle

case_0a4f93,"112 8 368 4",""

case_193d7c,"","501 6"

case_282aac,"",""

The first example predicts two addition runs and no removals.

The second example predicts one removal run and no additions.

The third example predicts no change.

The case identifiers shown above are illustrative.

A real submission must use the exact case_id values provided in test.csv.

17. Run-Length Encoding
Edit masks use one-indexed row-major run-length encoding.

The binary mask is flattened one row at a time from top to bottom and left to right.

Each positive run is represented by:

start length

Starts are one-indexed.

Multiple runs are separated by spaces.

For example:

4 3 12 2

This represents:

Three positive pixels beginning at flattened position 4.
Two positive pixels beginning at flattened position 12.
An empty mask is represented by an empty string.

A correct encoder should:

Flatten the mask in row-major order.
Detect every consecutive run of positive pixels.
Record the one-indexed start of each run.
Record the number of pixels in each run.
Return an empty string when the mask contains no positive pixels.
A correct decoder should:

Treat blank values and missing values as empty masks.
Require an even number of integer values.
Interpret alternating values as start and length.
Convert starts from one-indexed to zero-indexed positions.
Require positive run lengths.
Reject runs that exceed the declared mask dimensions.
Reject overlapping or unsorted runs.
18. Submission Format
The submission must contain exactly:

case_id
add_rle
remove_rle
Submission requirements:

Include every evaluation case_id.
Include each identifier exactly once.
Do not include unknown identifiers.
Use valid one-indexed row-major RLE.
Keep every run inside the declared dimensions.
Keep all edits inside the audit mask.
Add only where the draft is non-flood.
Remove only where the draft is flood.
Keep addition and removal masks disjoint.
A malformed edit invalidates only the affected case.

Submission-wide structural failures invalidate the full submission.

Examples include:

Incorrect submission columns.
Missing evaluation identifiers.
Duplicate identifiers.
Unknown identifiers.
An empty submission table.
19. Evaluation
Every case is scored independently.

The evaluation measures:

Addition accuracy.
Removal accuracy.
Final repaired-mask agreement.
Boundary agreement.
Preservation of correct draft pixels.
Connected-component agreement.
Robustness across cases.
Addition F1
addition_f1 is the binary F1 score between the hidden addition mask and the predicted addition mask.

If both masks are empty, addition_f1 equals 1.

If exactly one mask is empty, addition_f1 equals 0.

Removal F1
removal_f1 is the binary F1 score between the hidden removal mask and the predicted removal mask.

The same empty-mask rules apply.

Edit Score
edit_score is the average of addition_f1 and removal_f1.

Audit-Region IoU
audit_iou is the intersection-over-union between the repaired prediction and corrected flood target inside the audit region.

Only pixels inside the audit mask are included.

If both evaluated masks contain no flood pixels, audit_iou equals 1.

Boundary Agreement
Flood-boundary pixels are compared inside the audit region with a tolerance of two pixels.

The boundary score is a boundary F1 score.

If both boundary sets are empty, boundary_score equals 1.

If only one boundary set is empty, boundary_score equals 0.

Preservation Score
The preservation score measures the fraction of already-correct draft pixels inside the audit region that remain unchanged.

If there are no already-correct audit pixels, preservation_score equals 1.

Component Agreement
Eligible connected flood components intersecting the audit region are matched one-to-one when their IoU is at least 0.25.

Components smaller than eight pixels are ignored.

The component score is a component-level F1 score.

If neither result contains an eligible component, component_score equals 1.

If only one result contains eligible components, component_score equals 0.

Per-Case Score
The case score is calculated as:

40 percent edit score.
25 percent audit-region IoU.
15 percent boundary agreement.
10 percent preservation score.
10 percent component agreement.
Equivalently:

case_score = 0.40 × edit_score + 0.25 × audit_iou + 0.15 × boundary_score + 0.10 × preservation_score + 0.10 × component_score

Final Score
Let mean_case_score be the mean of all case scores.

Let lower_quartile_score be the 25th percentile of all case scores.

The final score is:

final_score = 100 × (0.85 × mean_case_score + 0.15 × lower_quartile_score)

The result is clipped to the interval from 0 to 100.

20. High-Level Grader Logic
The grader first validates the submission as a whole.

It requires exactly three columns:

case_id
add_rle
remove_rle
It requires every evaluation identifier to appear exactly once.

It rejects duplicate identifiers, unknown identifiers, missing identifiers, and empty submission tables.

If a submission-wide structural requirement fails, the final score is zero.

If submission-wide validation succeeds, the grader processes each evaluation case independently.

For each case, the grader:

Loads the draft flood mask.
Loads the audit mask.
Decodes the hidden addition mask.
Decodes the hidden removal mask.
Decodes the participant addition mask.
Decodes the participant removal mask.
Confirms that additions and removals do not overlap.
Confirms that all edits lie inside the audit mask.
Confirms that additions occur only where the draft is non-flood.
Confirms that removals occur only where the draft is flood.
Reconstructs the hidden corrected flood mask.
Reconstructs the participant’s repaired flood mask.
Computes the pixel, boundary, preservation, and component metrics.
Combines the metrics into a weighted case score.
If a participant prediction is malformed or violates a case-level constraint, that case receives a score of zero while other cases remain eligible for scoring.

After all cases are scored, the grader calculates:

The mean case score.
The 25th percentile case score.
The final score combines both values so that consistent performance is rewarded.

21. What the Metric Rewards
A strong system should:

Detect genuine flood-mask defects.
Predict the correct edit direction.
Recover accurate flood boundaries.
Preserve valid flood and non-flood pixels.
Restore meaningful connected flood structure.
Avoid false islands and false bridges.
Leave correct audit regions unchanged.
Perform consistently across different case types.
The metric does not reward aggressive editing by default.

Unnecessary edits may reduce edit accuracy, preservation, IoU, boundary agreement, and component agreement simultaneously.

22. Modeling Guidance
A practical model may combine:

The eight raster channels.
The eight-channel validity mask.
The draft flood mask.
The audit mask.
Distance to the current flood boundary.
Local elevation variation.
Local rainfall context.
Neighborhood statistics.
Connected-component features.
Missing-observation features.
A natural output formulation is a three-state pixel classifier:

Keep.
Add flood.
Remove flood.
Predictions may then be filtered using:

Audit membership.
Draft-state validity.
Component size.
Boundary consistency.
Morphological consistency.
Terrain consistency.
Topological checks.
Confidence thresholds.
A participant may also model addition and removal using two binary output heads.

23. Validation Guidance
Multiple training cases may reference the same parent raster.

A random row-level train and validation split may place cases derived from the same raster in both partitions.

This can produce overly optimistic validation results.

Participants are strongly encouraged to group validation by raster_path.

A useful development process is:

Establish the copy-draft baseline.
Create a validation split grouped by raster_path.
Train and evaluate a simple baseline.
Confirm that the learned system improves over copy-draft.
Tune addition and removal thresholds independently.
Validate the final encoded submission.
24. Numerical Stability and NaN Prevention
This task contains sparse targets and many valid no-change regions.

Some training crops may contain:

No addition pixels.
No removal pixels.
No changed pixels.
Very few audit pixels.
Only one active class.
Loss implementations must handle these cases safely.

Never Reduce Over an Empty Selection
A masked loss may become NaN when no pixels are selected.

Before averaging losses inside the audit region, verify that at least one pixel is selected.

If no pixel is selected, skip that batch or sample.

Avoid Division by Zero in Class Weights
Per-batch class weights are dangerous when one class is absent.

A calculation that divides the negative count by the positive count becomes infinite when the positive count is zero.

Participants should:

Clamp denominators to at least one.
Limit class weights to a reasonable maximum.
Consider fixed moderate class weights instead of per-batch weights.
Fail Immediately on Non-Finite Values
Before each optimizer update, verify that:

Model inputs are finite.
Model outputs are finite.
The loss is finite.
Gradients are finite.
Model parameters remain finite.
Training should stop immediately if any of these checks fail.

Participants should not continue training after a NaN or infinite gradient.

Participants should not use a checkpoint saved after numerical failure.

Training should restart from clean model weights after the underlying issue is corrected.

Gradient Clipping
Gradient clipping may help prevent unstable updates.

However, clipping does not repair NaN gradients.

A non-finite gradient must be treated as a failure rather than merely clipped.

Safe Loss Behavior
For a three-class model with keep, add, and remove outputs:

Compute per-pixel loss without reduction.
Select only pixels allowed by the chosen training mask.
Confirm that the selection is nonempty.
Average the selected loss.
Confirm that the result is finite before backpropagation.
25. Prediction Safety
Before encoding a prediction, participants should enforce all hard constraints.

The final addition mask should contain only pixels that:

Were predicted as flood additions.
Lie inside the audit mask.
Are non-flood in the draft.
The final removal mask should contain only pixels that:

Were predicted as flood removals.
Lie inside the audit mask.
Are flood in the draft.
Any overlap between additions and removals must be removed.

This filtering should occur even if the model was trained to obey the constraints.

26. Submission Validation Guidance
Before saving the final submission:

Constrain additions and removals using the audit and draft masks.
Remove any overlap.
Encode both masks using one-indexed row-major RLE.
Preserve empty masks as empty strings.
Confirm that the CSV columns are exact.
Confirm that every evaluation identifier appears exactly once.
Decode every RLE and verify its constraints.
A recommended validator should confirm:

The submission has exactly the required columns.
No case_id is missing.
No case_id is duplicated.
No unknown case_id is present.
All required evaluation identifiers are present.
Every RLE contains start-length pairs.
Every start is one-indexed and positive.
Every run length is positive.
Every run remains inside the declared dimensions.
Runs are sorted and do not overlap.
Addition and removal masks are disjoint.
All edits are inside the audit mask.
Additions occur only where the draft is non-flood.
Removals occur only where the draft is flood.
27. Minimal Valid Submission Behavior
A valid copy-draft submission contains every test case_id and leaves both edit fields empty.

This produces no additions and no removals.

Participants should confirm that their complete pipeline can always fall back to this valid output if model training or inference fails.

A failed model should not be allowed to produce malformed RLE, missing rows, invalid edits, or corrupted outputs.

28. Pre-Submission Checklist
Before finalizing a submission:

Confirm that the unchanged sample submission produces a valid CSV.
Evaluate the model on a split grouped by raster_path.
Stop training immediately if inputs, outputs, loss, gradients, or parameters become non-finite.
Reload only a checkpoint saved before any numerical failure.
Apply audit and draft-state constraints before RLE encoding.
Confirm that addition and removal masks are disjoint.
Decode and validate every final RLE.
Confirm that all test identifiers appear exactly once.
Confirm that the CSV columns are exactly case_id, add_rle, and remove_rle.
Compare the final model against the copy-draft baseline.
29. Research Positioning
This benchmark studies flood-map repair rather than independent flood segmentation.

Its defining structure is the combination of:

A mostly correct draft flood prediction.
A broad review region rather than direct error prompts.
Immutable pixels outside that region.
Separate sparse addition and removal outputs.
Explicit no-change examples.
Preservation-aware evaluation.
Boundary-aware evaluation.
Component-aware evaluation.
Robustness-aware aggregation.
The participant is learning a flood-map intervention policy rather than only predicting a final flood class independently at every pixel.

30. Summary
Evidence-Guided Flood Mask Repair asks a model to inspect an existing flood prediction and determine:

Which flood and non-flood regions should be trusted.
Which pixels should be changed.
Whether a change requires adding or removing flood.
How far the correction should extend.
Which valid structures must be preserved.
Whether the correct action is to make no change.
The output is a constrained, sparse, directional flood-mask repair plan.

Submissions are evaluated through:

Edit accuracy.
Final flood-mask agreement.
Boundary fidelity.
Preservation.
Connected-component consistency.
Robustness across cases.
A successful solution must combine multimodal evidence interpretation, flood-boundary reasoning, terrain awareness, conservative intervention, numerical stability, and strict submission validation.