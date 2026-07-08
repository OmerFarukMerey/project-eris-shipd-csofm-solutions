"""
Bone Niche Cellular Region Detection - solution.

Fine-tunes a torchvision Faster R-CNN (MobileNetV3-Large-FPN, COCO-pretrained
backbone) on the prepared bone microscopy patches to localize osteoblast-
associated (class 0) and osteoclast-associated (class 1) regions, then runs
inference on the test set and writes ./working/submission.csv.

Run with:  python solution.py
(uses only the prepared public dataset files under dataset/public/)
"""

import os
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --------------------------------------------------------------------------
# Paths & config
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "public")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
WORKING_DIR = os.path.join(BASE_DIR, "working")

SEED = 42
IMG_SIZE = 512
NUM_CLASSES = 3  # background + class0 + class1 (torchvision label space)

# NOTE on device choice: MPS was tried on the Apple Silicon dev machine, but
# empirically the MPS backend recompiles Metal shaders whenever Faster R-CNN's
# dynamic-shape ops (RPN proposal count, RoIAlign box count) change between
# iterations, which stalled a single training step for 50+ seconds. CPU
# inference/training with multi-threaded BLAS was far more predictable there.
# On a CUDA GPU (e.g. the RTX 5070 this is meant to be transferred to) none of
# that applies, so CUDA is used whenever available, with batch size/workers/lr
# scaled up accordingly.
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = int(os.environ.get("BONE_BATCH_SIZE", 16 if USE_CUDA else 4))
NUM_WORKERS = int(os.environ.get("BONE_NUM_WORKERS", 8 if USE_CUDA else 0))
# data prep (~ms) is negligible next to ~1.2s/batch CPU model compute, so
# CPU runs keep everything in the main process; on GPU, workers overlap image
# decode/augmentation with GPU compute instead of leaving the GPU data-starved.
NUM_EPOCHS = int(os.environ.get("BONE_NUM_EPOCHS", 12))
BASE_LR = float(os.environ.get("BONE_BASE_LR", 0.00125 * BATCH_SIZE))  # linear LR scaling rule, tuned at bs=4 -> lr=0.005
WEIGHT_DECAY = 5e-4
WARMUP_ITERS = 300
VAL_FRACTION = 0.1
EVAL_EVERY = 1

CHECKPOINT_PATH = os.path.join(WORKING_DIR, "model_best.pt")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

if USE_CUDA:
    torch.backends.cudnn.benchmark = True  # fixed 512x512 input size -> let cudnn autotune conv algos


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Augmentation (numpy/PIL based, boxes as float arrays [x_min,y_min,x_max,y_max])
# --------------------------------------------------------------------------

def hflip(img, boxes):
    w = img.shape[1]
    img = np.ascontiguousarray(img[:, ::-1])
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img, boxes


def vflip(img, boxes):
    h = img.shape[0]
    img = np.ascontiguousarray(img[::-1, :])
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    return img, boxes


def rot90(img, boxes, k):
    """Rotate image by 90*k degrees CCW (numpy convention) and boxes to match."""
    h, w = img.shape[:2]
    img = np.ascontiguousarray(np.rot90(img, k))
    if len(boxes) == 0:
        return img, boxes
    boxes = boxes.copy()
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if k == 1:
        nx_min, nx_max = y_min, y_max
        ny_min, ny_max = w - x_max, w - x_min
    elif k == 2:
        nx_min, nx_max = w - x_max, w - x_min
        ny_min, ny_max = h - y_max, h - y_min
    elif k == 3:
        nx_min, nx_max = h - y_max, h - y_min
        ny_min, ny_max = x_min, x_max
    else:
        return img, boxes
    boxes = np.stack([nx_min, ny_min, nx_max, ny_max], axis=1)
    return img, boxes


def random_crop_resize(img, boxes, labels, out_size, scale_range=(0.7, 1.0), min_keep_area_frac=0.3):
    """Zoom-in augmentation: crop a random square sub-region then resize back to out_size."""
    h, w = img.shape[:2]
    side = int(round(random.uniform(*scale_range) * min(h, w)))
    side = max(32, min(side, min(h, w)))
    x0 = random.randint(0, w - side)
    y0 = random.randint(0, h - side)
    crop = img[y0:y0 + side, x0:x0 + side]

    if len(boxes):
        orig_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
        shifted = boxes.copy()
        shifted[:, [0, 2]] -= x0
        shifted[:, [1, 3]] -= y0
        clipped = shifted.copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, side)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, side)
        new_w = clipped[:, 2] - clipped[:, 0]
        new_h = clipped[:, 3] - clipped[:, 1]
        new_area = np.maximum(new_w, 0) * np.maximum(new_h, 0)
        keep = (new_w >= 2) & (new_h >= 2) & (new_area >= min_keep_area_frac * np.maximum(orig_area, 1e-6))
        boxes = clipped[keep]
        labels = labels[keep]

    pil_crop = Image.fromarray(crop)
    pil_crop = pil_crop.resize((out_size, out_size), Image.BILINEAR)
    out_img = np.array(pil_crop)

    if len(boxes):
        scale = out_size / side
        boxes = boxes * scale

    return out_img, boxes, labels


def color_jitter(img_float):
    """img_float: HxW float32 in [0,1]. Simulates contrast/brightness/gamma variation."""
    contrast = random.uniform(0.8, 1.25)
    brightness = random.uniform(-0.12, 0.12)
    gamma = random.uniform(0.85, 1.2)
    img = (img_float - 0.5) * contrast + 0.5 + brightness
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, gamma)
    if random.random() < 0.3:
        noise_sigma = random.uniform(0.0, 0.02)
        img = img + np.random.normal(0, noise_sigma, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def augment(img, boxes, labels):
    """img: HxW uint8 numpy array. boxes: (N,4) float array. labels: (N,) int array."""
    if random.random() < 0.5:
        img, boxes = hflip(img, boxes)
    if random.random() < 0.5:
        img, boxes = vflip(img, boxes)
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        img, boxes = rot90(img, boxes, k)
    if random.random() < 0.7:
        img, boxes, labels = random_crop_resize(img, boxes, labels, out_size=IMG_SIZE)

    img_float = img.astype(np.float32) / 255.0
    img_float = color_jitter(img_float)
    return img_float, boxes, labels


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class BoneDataset(Dataset):
    def __init__(self, ids, ann_by_id, img_dir, train):
        self.ids = list(ids)
        self.ann_by_id = ann_by_id
        self.img_dir = img_dir
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, f"{img_id}.png")).convert("L")
        img = np.array(img)

        boxes, labels = self.ann_by_id.get(img_id, (np.zeros((0, 4), dtype=np.float32),
                                                      np.zeros((0,), dtype=np.int64)))
        boxes = boxes.astype(np.float32).copy()
        labels = labels.astype(np.int64).copy()

        if self.train:
            img_float, boxes, labels = augment(img, boxes, labels)
        else:
            img_float = img.astype(np.float32) / 255.0

        img3 = np.stack([img_float, img_float, img_float], axis=0)
        img_t = torch.from_numpy(img3.astype(np.float32))

        if len(boxes):
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": img_id}
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_ann_by_id(ann_df):
    ann_by_id = {}
    for img_id, group in ann_df.groupby("id"):
        boxes = group[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float32)
        labels = (group["class_id"].to_numpy(dtype=np.int64) + 1)  # shift: 0/1 -> 1/2 (torchvision bg=0)
        ann_by_id[img_id] = (boxes, labels)
    return ann_by_id


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(num_classes=NUM_CLASSES):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        min_size=IMG_SIZE,
        max_size=IMG_SIZE,
        box_score_thresh=0.001,
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --------------------------------------------------------------------------
# mAP@0.5 evaluation (matches the challenge's grading rule)
# --------------------------------------------------------------------------

def box_iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_ap_for_class(preds, gts, iou_thresh=0.5):
    """preds: list of (image_id, score, box). gts: dict image_id -> list of boxes."""
    if sum(len(v) for v in gts.values()) == 0:
        return None  # no positives for this class in this split
    preds = sorted(preds, key=lambda x: -x[1])
    matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gts.items()}
    n_gt = sum(len(v) for v in gts.values())

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img_id, score, box) in enumerate(preds):
        gt_boxes = gts.get(img_id, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou_matrix(np.array([box]), gt_boxes)[0]
        best_j = np.argmax(ious)
        best_iou = ious[best_j]
        if best_iou >= iou_thresh and not matched[img_id][best_j]:
            tp[i] = 1
            matched[img_id][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # all-point interpolation (COCO/VOC-2010+ style)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


@torch.no_grad()
def evaluate_map(model, loader, device):
    model.eval()
    preds_by_class = {0: [], 1: []}
    gts_by_class = {0: {}, 1: {}}

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            img_id = target["image_id"]
            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()
            for c in (0, 1):
                cls_boxes = gt_boxes[gt_labels == (c + 1)]
                gts_by_class[c][img_id] = cls_boxes

            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                c = int(label) - 1
                if c in (0, 1):
                    preds_by_class[c].append((img_id, float(score), box))

    aps = {}
    for c in (0, 1):
        ap = compute_ap_for_class(preds_by_class[c], gts_by_class[c], iou_thresh=0.5)
        aps[c] = ap if ap is not None else 0.0
    mean_ap = (aps[0] + aps[1]) / 2.0
    return {"map": mean_ap, "ap0": aps[0], "ap1": aps[1]}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, lr_scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def make_warmup_then_multistep(optimizer, warmup_iters, milestones_epochs, iters_per_epoch, gamma=0.1):
    milestones_iters = [m * iters_per_epoch for m in milestones_epochs]

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        factor = 1.0
        for m in milestones_iters:
            if it >= m:
                factor *= gamma
        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Inference / submission
# --------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, test_df, img_dir, device, max_per_image=100):
    model.eval()
    rows = []
    for _, row in test_df.iterrows():
        img_id = row["id"]
        img = Image.open(os.path.join(img_dir, f"{img_id}.png")).convert("L")
        img = np.array(img).astype(np.float32) / 255.0
        img3 = np.stack([img, img, img], axis=0)
        img_t = torch.from_numpy(img3.astype(np.float32)).to(device)

        output = model([img_t])[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        order = np.argsort(-scores)[:max_per_image]
        for j in order:
            x_min, y_min, x_max, y_max = boxes[j]
            x_min = float(np.clip(x_min, 0, IMG_SIZE))
            y_min = float(np.clip(y_min, 0, IMG_SIZE))
            x_max = float(np.clip(x_max, 0, IMG_SIZE))
            y_max = float(np.clip(y_max, 0, IMG_SIZE))
            if x_max <= x_min or y_max <= y_min:
                continue
            class_id = int(labels[j]) - 1
            if class_id not in (0, 1):
                continue
            rows.append({
                "id": img_id,
                "class_id": class_id,
                "confidence": round(float(scores[j]), 5),
                "x_min": round(x_min, 2),
                "y_min": round(y_min, 2),
                "x_max": round(x_max, 2),
                "y_max": round(y_max, 2),
            })
    return pd.DataFrame(rows, columns=["id", "class_id", "confidence", "x_min", "y_min", "x_max", "y_max"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    set_seed()
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    ann_df = pd.read_csv(os.path.join(DATA_DIR, "train_annotations.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    ann_by_id = build_ann_by_id(ann_df)

    box_counts = train_df["id"].map(lambda i: len(ann_by_id.get(i, (np.zeros((0, 4)),))[0]))
    bucket = pd.cut(box_counts, bins=[-1, 0, 2, 5, 10, 1000], labels=[0, 1, 2, 3, 4])

    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(
        train_df["id"].tolist(), test_size=VAL_FRACTION, random_state=SEED, stratify=bucket
    )

    print(f"train images: {len(train_ids)}, val images: {len(val_ids)}", flush=True)

    train_ds = BoneDataset(train_ids, ann_by_id, IMAGES_DIR, train=True)
    val_ds = BoneDataset(val_ids, ann_by_id, IMAGES_DIR, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=collate_fn, drop_last=True, pin_memory=USE_CUDA,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=USE_CUDA,
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"device: {DEVICE}  batch_size: {BATCH_SIZE}  num_workers: {NUM_WORKERS}  base_lr: {BASE_LR}", flush=True)
    model = build_model().to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    iters_per_epoch = len(train_loader)
    lr_scheduler = make_warmup_then_multistep(
        optimizer, warmup_iters=min(WARMUP_ITERS, iters_per_epoch), milestones_epochs=[8, 11],
        iters_per_epoch=iters_per_epoch,
    )

    best_map = -1.0
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, lr_scheduler)
        msg = f"epoch {epoch+1}/{NUM_EPOCHS} train_loss {train_loss:.4f} time {time.time()-t0:.1f}s"

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            metrics = evaluate_map(model, val_loader, DEVICE)
            msg += f" | val_mAP@0.5 {metrics['map']:.4f} (cls0 {metrics['ap0']:.4f} cls1 {metrics['ap1']:.4f})"
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                msg += " [saved best]"
        print(msg, flush=True)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"loaded best checkpoint (val_mAP@0.5={best_map:.4f}) for inference", flush=True)
    else:
        print("no checkpoint saved (unexpected); using final epoch weights", flush=True)

    submission = run_inference(model, test_df, IMAGES_DIR, DEVICE)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {len(submission)} rows to {SUBMISSION_PATH}", flush=True)


if __name__ == "__main__":
    main()
