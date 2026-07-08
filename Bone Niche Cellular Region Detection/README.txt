BONE NICHE CELLULAR REGION DETECTION - APPROACH NOTES
======================================================

WHAT THIS IS
------------
solution.py fine-tunes a torchvision Faster R-CNN object detector on the
prepared bone microscopy patches and writes predicted bounding boxes for
test.csv images to ./working/submission.csv. Only files under
dataset/public/ are used; the pretrained backbone is a generic ImageNet/
COCO torchvision checkpoint (not challenge-specific).

Run with:
    python solution.py
(tested with the anaconda3 Python 3.13 environment: torch 2.8, torchvision
0.23, pandas, numpy, PIL, scikit-learn)


MODEL ARCHITECTURE
-------------------
- Faster R-CNN with a MobileNetV3-Large + FPN backbone
  (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn),
  initialized from COCO-pretrained weights (FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).
- The box predictor head is replaced with a fresh FastRCNNPredictor for
  3 classes (background + osteoblast-associated + osteoclast-associated);
  everything else (backbone, FPN, RPN) is fine-tuned from the COCO checkpoint.
- The model's internal resize transform is pinned to min_size=max_size=512,
  i.e. it runs at the images' native resolution instead of the COCO defaults
  (which would upscale to ~800px, wasting compute with no extra real detail,
  since our source patches are 512x512 to begin with).
- Grayscale patches are replicated across 3 channels to match the
  ImageNet-pretrained backbone's expected input and normalization stats.
- box_detections_per_img is left at 100, which directly satisfies the
  competition's "at most 100 positive-confidence predictions per image" rule
  (torchvision caps total detections per image across both classes there).


WHY FASTER R-CNN / MOBILENET-FPN
---------------------------------
- Two-stage detectors with an FPN handle the wide object-size range in this
  dataset well: class 0 (osteoblast) boxes are ~57x60px median, class 1
  (osteoclast) boxes are ~26x26px median, with a long tail down to ~4px.
  Multi-level FPN anchors (32/64/128/256/512 at each of 3 pyramid levels)
  give reasonable coverage of that range without any custom anchor tuning.
- A COCO-pretrained backbone gives useful low/mid-level texture and edge
  features out of the box, which matters a lot with only ~1140 annotated
  training images.
- MobileNetV3 was chosen over ResNet50 purely for training-time budget on
  the available hardware (see COMPUTE NOTES).


TRAINING SETUP
--------------
- Train/val split: 90/10 random split of train.csv, stratified by a coarse
  bucket of per-image box count (0 / 1-2 / 3-5 / 6-10 / 11+) so the
  validation set isn't accidentally skewed toward empty or very dense images.
  (The prepared CSVs don't expose a source-image grouping column, so a
  plain stratified random split is used instead of a group split.)
- Optimizer: SGD, momentum 0.9, weight_decay 5e-4, base lr 0.005 for
  batch size 4, with linear warmup (~1 epoch) then a step decay
  (x0.1 at epoch 8 and again at epoch 11 of 12).
- Model selection: after every epoch, mAP@0.50 is computed on the held-out
  validation split using the exact scoring rule described in the problem
  (per class, sort by confidence, greedy match to unmatched GT at IoU>=0.5,
  all-point interpolated AP, averaged over class 0 and class 1). The
  checkpoint with the best validation mAP is the one used for test
  inference, not simply the last epoch.


AUGMENTATION (KEY DESIGN CHOICE)
---------------------------------
With only ~1140 annotated training images, augmentation was the main lever
for robustness given the problem's own framing ("stay robust to image-to-
image variation in contrast, background, tissue texture, and object
density"). All augmentation is implemented directly with numpy/PIL
(no albumentations/cv2 dependency) and boxes are transformed in lockstep
with the image:
- Random horizontal flip, vertical flip, and random 90-degree rotation
  (0/90/180/270). Fluorescence microscopy patches have no canonical
  orientation, so all 8 dihedral symmetries are valid label-preserving
  transforms - cheap, safe augmentation that directly multiplies the
  effective dataset size.
- Random crop-and-resize ("zoom" augmentation, p=0.7): crop a random square
  region (70-100% of the image side) and resize back to 512x512. Boxes are
  clipped to the crop and dropped if less than 30% of their original area
  survives or they shrink below 2px. This simulates the object-density /
  scale variation called out in the problem statement and is a much
  higher-value augmentation for small-object detection than color jitter
  alone.
- Brightness/contrast/gamma jitter plus light Gaussian noise, applied to
  the float image, to cover the "image-to-image variation in contrast [and]
  background" the problem explicitly warns about.
No augmentation is applied to the validation or test images.


EVALUATION METRIC IMPLEMENTATION
----------------------------------
Since the task's grading rule is a specific greedy-matching mAP@0.50 (not
COCO's multi-IoU-averaged mAP), a small self-contained implementation
(box_iou_matrix / compute_ap_for_class / evaluate_map in solution.py)
reproduces it directly instead of depending on pycocotools: for each class,
predictions are sorted by confidence, each is matched to the
highest-IoU unmatched ground-truth box of the same class/image if
IoU >= 0.5, duplicate matches count as false positives, and AP is the
all-point-interpolated area under the resulting precision/recall curve.
The final score reported during training is the mean of the two per-class
APs, mirroring the competition metric exactly.


COMPUTE NOTES / WHAT DIDN'T WORK
-----------------------------------
- This machine has no CUDA GPU but does expose Apple's MPS backend.
  MPS was tried first and rejected: Faster R-CNN's RPN/RoIAlign stages
  produce a different number of proposals/boxes on almost every batch,
  and PyTorch's MPS backend recompiles Metal shaders whenever a new tensor
  shape is seen. In a benchmark, the very first real training step after
  warmup took 56 seconds on MPS (vs ~1.2s on CPU for the same batch) - i.e.
  MPS is actively counterproductive for this specific model shape profile,
  even though it works fine for fixed-shape CNNs. Training therefore runs
  on CPU (with CUDA still auto-selected if this script is ever run on a
  machine that has it), which gave a stable, predictable ~1.2s/batch
  (batch size 4) on a 14-core Apple M4 Pro.
- fasterrcnn_mobilenet_v3_large_320_fpn (the torchvision variant tuned to
  downscale to 320px for speed) was benchmarked too: it was only marginally
  faster than running the plain mobilenet_v3_large_fpn backbone at native
  512px (0.27s vs 0.31s per image), and shrinking already-small osteoclast
  boxes (~26px) further down to ~16px seemed like a bad trade for that small
  a speed gain, so native 512px resolution was kept.
- ResNet50-FPN (v2) was considered for accuracy but not used: it's a
  noticeably heavier backbone than MobileNetV3, and with this dataset's
  size and this machine's CPU-only budget, the extra epochs affordable
  with MobileNetV3 in the same wall-clock time were judged more valuable
  than a bigger backbone trained for fewer epochs. This is a
  compute-budget-driven choice, not an accuracy ceiling - a GPU box would
  likely favor ResNet50-FPN.
- Batch size beyond 4 (tested up to 8) didn't improve per-image throughput
  on CPU meaningfully, since Faster R-CNN's per-image RPN/RoI post-processing
  is a Python-level loop that doesn't batch well; batch size 4 was kept for
  simplicity.


INFERENCE / SUBMISSION
------------------------
- The best (by validation mAP@0.5) checkpoint is loaded and run once over
  every image in test.csv.
- Predicted boxes are clipped to the 512x512 image bounds, degenerate boxes
  (x_max<=x_min or y_max<=y_min) are dropped, and predictions are capped at
  100 per image by confidence (matching the submission requirement).
- Confidences are left as raw model scores (not thresholded) because the
  mAP@0.5 metric integrates over the full precision/recall curve, so
  including lower-confidence tail predictions doesn't hurt the score and
  can only help recall.
- Output columns are exactly id,class_id,confidence,x_min,y_min,x_max,y_max,
  written to ./working/submission.csv.
