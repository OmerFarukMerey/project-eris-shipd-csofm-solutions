Overview
Locate the annotated skin regions in each photograph and enclose them in at most three rectangular crops. Predict the individual region boxes and the crop boxes. Good crops retain complete regions while including little additional image area.

The photographs are skin-surface tiles captured by a multi-camera total-body photography system. Each tile can contain several small annotated lesions. When assembling an image packet for visual review, sending the entire tile can include much more image area than the regions being discussed; sending a separate tight crop for every region can instead create an unwieldy packet. This task measures the geometric trade-off between those choices. It does not predict malignancy, diagnose a patient or establish that unannotated skin is healthy.

Annotations are existing expert-reviewed boxes. The task keeps images with four through sixteen distinct regions, so a three-crop limit normally requires combining some regions. Photographs are resized to 768 by 640 RGB pixels and may be reflected horizontally together with their annotations. The publisher's participant-separated training and test assignments are retained; no test-source image is moved into training.

The competition environment provides access to a single NVIDIA A10G GPU. The entire pipeline must finish within 1.5 hours, including data loading, training or adaptation, inference, decoding, validation, and submission generation.

Dataset
There are 2,184 training examples and 546 test examples, with 2,730 referenced images. Each row has one photograph. Two exact duplicate-image records were excluded before selecting cases. The test examples are a deterministic selection from the publisher's held-out partition, not augmented copies of training examples.

| File | Content |
|---|---|
| train.csv | case_id, image_path, regions, crop_plan for 2,184 labelled images. |
| test.csv | case_id and image_path for 546 images. |
| sample_submission.csv | All test IDs with empty box lists in both target columns. |
| images/ | 2,730 RGB JPEGs, each 768 pixels wide by 640 pixels high. |

An image_path such as images/rp_0123456789abcdef012345.jpg is relative to the supplied data directory. The crop coordinate system is normalized and does not use these physical image dimensions directly.

| Column | Type | Meaning |
|---|---|---|
| case_id | string | Opaque row identifier; copy without modification. |
| image_path | string | Relative photograph path, present in train.csv and test.csv. |
| regions | JSON-encoded string | List of individual region boxes. Up to 64 predicted boxes are allowed. |
| crop_plan | JSON-encoded string | List of zero to three proposed crop boxes. |

Box Coordinates
Every box is [left, top, right, bottom], using integer coordinates from 0 through 1024 on both axes. The top-left corner is (0,0); the bottom-right image boundary is (1024,1024). Boxes have positive width and height. Their area is (right-left) times (bottom-top). Scale x coordinates by image width/1024 and y coordinates by image height/1024 to display them on the photograph.

For example, [256,256,512,512] covers a quarter of the width and a quarter of the height, starting one quarter of the way from the top-left corner. It does not cover one quarter of the total area.

The supplied region coordinates round outward from the original annotations. The training crop_plan is constructed by repeatedly merging the two box groups with the smallest increase in their combined bounding-rectangle area until at most three groups remain. Initially, each box is its own group and boxes are sorted lexicographically. For group indices i less than j, merge cost is the combined bounding-box area minus the two current bounding-box areas. Ties use the smallest (i,j) pair. The merged group keeps position i and group j is removed. This is a reproducible reference plan, not a claim of global optimality. Better alternative plans can receive full credit.

Submission Format
Write ./working/submission.csv with exactly case_id,regions,crop_plan in that order. Include each test ID once, with no missing or extra IDs or columns. Submission rows can appear in any order.

case_id,regions,crop_plan
rp_0123456789abcdef012345,"[[100,100,120,120],[150,100,170,120],[400,400,420,420],[700,700,720,720]]","[[100,100,170,120],[400,400,420,420],[700,700,720,720]]"

This illustrative row combines two nearby regions into one crop and uses two other crops for distant regions. It is a format example, not a supplied test label.

Both targets are strings containing JSON arrays. Each string is limited to 8,192 characters and nesting depth two. A region list has at most 64 boxes; a crop list has at most three. Coordinates must be JSON integers, not booleans, quoted numbers or floating-point values. Duplicate boxes, out-of-range coordinates, zero-area boxes and reversed corners are invalid. Box order does not matter and overlapping crops are allowed. Each crop is delivered independently, so overlapping pixels consume capacity in every crop containing them.

Wrong column order, duplicate column names, missing/extra rows and invalid or duplicate IDs reject the submission. An invalid box field makes its entire case score zero, including its other target. Empty lists are valid and score zero when both targets are empty.

Evaluation
The Compact Region Packaging Score is the mean case score. Minimum score: 0.0. Maximum score: 1.0. Higher is better. Per case, the score is 0.35 times RegionF1 plus 0.20 times CompleteCoverage plus 0.45 times CompactUtility.

case_score = 0.35 * RegionF1
           + 0.20 * CompleteCoverage
           + 0.45 * CompactUtility
final_score = sum(case_score) / number_of_test_images

The 35% localization term measures whether individual regions were actually found. The 20% coverage term gives explicit credit for not cutting a region. The largest weight, 45%, measures the central packaging objective: retaining regions at a low delivered-pixel cost. A full-image crop gives full coverage but its packaging credit is reduced by the reference plan's area relative to the full image. Wasted pixels are charged against the useful reference packet, not the much larger source photograph.

RegionF1
For two boxes, intersection over union (IoU) is their intersection area divided by their union area. For each threshold in {0.50, 0.75}, form possible predicted/reference matches whose IoU meets that threshold. Find the maximum-cardinality one-to-one matching. If its size is M, with P predicted boxes and T reference boxes, the threshold score is 2M/(P+T). RegionF1 is the average of the two threshold scores. Missing detections and extra detections both lower it. Every retained case has at least four reference regions, so the denominator is nonzero.

CompleteCoverage
A reference region is covered only when all four of its boundaries lie inside one submitted crop, including equality. Covering different portions with separate crops does not count as complete containment. CompleteCoverage is the number of covered reference regions divided by T. All regions are weighted equally regardless of area.

CompactUtility
Let D be the sum of the areas of submitted crop rectangles and D_ref the sum for the reference crop plan. These are normalized delivered-pixel costs at a common spatial sampling rate. Overlapping pixels are charged repeatedly because the crops are separate delivered images. CompactUtility equals CompleteCoverage times min(1, D_ref/D) when D is positive, and zero when no crops are submitted.

D = sum((right-left)*(bottom-top) for each submitted crop)
D_ref = sum((right-left)*(bottom-top) for each reference crop)
CompactUtility = CompleteCoverage * min(1, D_ref / D), if D > 0
CompactUtility = 0, if D = 0

Reference plans cover all reference regions and have positive cost. Any alternative plan that covers every reference region using no more delivered pixels receives CompactUtility one. Delivering exactly twice the reference plan's pixel cost while maintaining complete coverage gives CompactUtility 0.5. A tiny crop covering only a quarter of the regions cannot exceed 0.25, even if it costs less than the reference. The predicted region list does not redefine coverage or reference cost.

Normalization is against the documented feasible reference, not an undisclosed exact optimizer. For example, a reference cost of 32,768 and a submitted full-image crop of cost 1,048,576 give CompactUtility 0.03125; with no region detections, the case score is 0.2140625. Exact valid answers achieve 1.0. Malformed cases receive zero before component aggregation; all remaining cases are averaged equally. There is no frequency weighting or diagnosis-dependent penalty.

Allowed Methods
Train an object detector on the supplied region annotations, then optimize its crop geometry, or learn the outputs jointly. General-purpose pretrained vision models are allowed. Models trained on the withheld photographs or their annotations are not. Crop optimization using the supplied training labels is allowed and need not be differentiable.

What Not To Use
Do not look up original source annotations or identify a held-out photograph through an external image index. Do not exploit IDs, filename order, metadata or file sizes as answer keys. Do not use test labels, test-time adaptation, transductive fitting, test-derived normalization, pseudo-label training on test images or test-based threshold selection. Fit all parameters and select methods using training-only evidence. Do not infer or attempt to recover participant identity. Predictions are for benchmark annotation geometry, not clinical diagnosis or patient decisions.