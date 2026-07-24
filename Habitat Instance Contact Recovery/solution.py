#!/usr/bin/env python3
"""Train an instance detector and learned contact head, then create submission.csv."""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SEED = 1729
TRAINING_DEADLINE_SECONDS = 3000.0
REFIT_DEADLINE_SECONDS = 2600.0


def log(message):
    print(message, flush=True)


def read_test_rows(public_dir):
    path = public_dir / "test.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "image_path"]:
            log(f"warning: unexpected test columns {reader.fieldnames}")
        rows = list(reader)
    return rows


def write_submission(path, test_rows, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "instances"])
        writer.writeheader()
        for row, prediction in zip(test_rows, predictions):
            try:
                encoded = json.dumps(prediction, separators=(",", ":"), allow_nan=False)
            except Exception as exc:
                log(f"warning: invalid prediction for {row.get('id')}: {exc}")
                encoded = "[]"
            writer.writerow({"id": row.get("id", ""), "instances": encoded})


def seed_everything(seed, torch):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def load_training_rows(public_dir, pd):
    path = public_dir / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    frame = pd.read_csv(path, dtype={"id": str, "image_path": str})
    required = ["id", "image_path", "instances"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"required training columns are missing: {missing}")
    parsed = []
    for row in frame.itertuples(index=False):
        try:
            objects = json.loads(row.instances)
            if not isinstance(objects, list):
                objects = []
        except Exception as exc:
            log(f"warning: invalid training annotation for {row.id}: {exc}")
            objects = []
        parsed.append({"id": str(row.id), "image_path": str(row.image_path), "instances": objects})
    return parsed


def annotation_stratum(objects):
    count = len(objects)
    if count == 0:
        return "empty"
    if count == 1:
        return "single"
    has_contact = any(bool(obj.get("contacts", [])) for obj in objects)
    return "multi_contact" if has_contact else "multi_no_contact"


def make_split(rows, np):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(rows))
    strata = np.asarray([annotation_stratum(row["instances"]) for row in rows])
    validation_size = max(1, int(round(0.16 * len(rows))))
    try:
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=validation_size,
            random_state=SEED,
            shuffle=True,
            stratify=strata,
        )
    except ValueError:
        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(indices)
        valid_idx = shuffled[:validation_size]
        train_idx = shuffled[validation_size:]
    return train_idx.tolist(), valid_idx.tolist()


def clean_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(value == value and abs(value) != float("inf") for value in values):
        return None
    x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in values]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class HabitatDataset:
    def __init__(self, rows, indices, public_dir, augment=False):
        self.rows = rows
        self.indices = list(indices)
        self.public_dir = public_dir
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        import torch

        row_index = self.indices[position]
        row = self.rows[row_index]
        image_path = self.public_dir / row["image_path"]
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            log(f"warning: failed to read training image {image_path}: {exc}")
            image = Image.new("RGB", (512, 512))

        boxes = []
        for obj in row["instances"]:
            box = clean_box(obj.get("bbox"))
            if box is not None:
                boxes.append(box)

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes = [[1.0 - x2, y1, 1.0 - x1, y2] for x1, y1, x2, y2 in boxes]
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                boxes = [[x1, 1.0 - y2, x2, 1.0 - y1] for x1, y1, x2, y2 in boxes]
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.ROTATE_90)
                boxes = [[y1, 1.0 - x2, y2, 1.0 - x1] for x1, y1, x2, y2 in boxes]

        width, height = image.size
        absolute = [[x1 * width, y1 * height, x2 * width, y2 * height] for x1, y1, x2, y2 in boxes]
        box_tensor = torch.as_tensor(absolute, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(absolute),), dtype=torch.int64)
        area = (
            (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
            if len(absolute)
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": box_tensor,
            "labels": labels,
            "image_id": torch.tensor([row_index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(absolute),), dtype=torch.int64),
        }
        tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        return tensor, target


def collate_detection(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id * 104729) % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except Exception:
        pass


def build_detector(torch):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import RPNHead

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    try:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            min_size=512,
            max_size=512,
            box_detections_per_img=40,
        )
        log("loaded COCO-pretrained ResNet-50-FPN-v2 backbone")
    except Exception as exc:
        log(f"warning: pretrained weights unavailable, training from random initialization: {exc}")
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=512,
            max_size=512,
            box_detections_per_img=40,
        )

    anchors = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((1.0 / 3.0, 0.5, 1.0, 2.0, 3.0),) * 5,
    )
    model.rpn.anchor_generator = anchors
    model.rpn.head = RPNHead(256, anchors.num_anchors_per_location()[0], conv_depth=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.detections_per_img = 40
    return model


def make_loader(dataset, torch, shuffle, batch_size):
    workers = min(4, max(0, (os.cpu_count() or 2) // 2))
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_detection,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def train_detector(model, loader, device, torch, started_at, epochs, learning_rate):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    completed = 0
    model.train()
    for epoch in range(epochs):
        if time.monotonic() - started_at >= TRAINING_DEADLINE_SECONDS:
            log("wall-clock guard: stopping detector training")
            break
        total_loss = 0.0
        accepted_batches = 0
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    log("warning: skipped non-finite training loss")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().cpu())
                accepted_batches += 1
            except Exception as exc:
                if device.type == "cuda" and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                log(f"warning: skipped failed training batch: {exc}")
        scheduler.step()
        completed += 1
        mean_loss = total_loss / max(1, accepted_batches)
        log(f"epoch {epoch + 1}/{epochs}: loss={mean_loss:.5f}")
    return completed


def image_to_tensor(image_path, torch):
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
    return tensor, width, height


def prediction_to_normalized(output, width, height, np):
    boxes = output["boxes"].detach().cpu().numpy().astype(np.float64, copy=False)
    scores = output["scores"].detach().cpu().numpy().astype(np.float64, copy=False)
    if boxes.size:
        boxes[:, [0, 2]] /= float(width)
        boxes[:, [1, 3]] /= float(height)
        boxes = np.clip(boxes, 0.0, 1.0)
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    return {"boxes": boxes[valid], "scores": scores[valid]}


def predict_rows(model, rows, indices, public_dir, device, torch, np, batch_size):
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            tensors = []
            dimensions = []
            valid_positions = []
            batch_results = [{"boxes": np.empty((0, 4)), "scores": np.empty((0,))} for _ in batch_indices]
            for local_position, row_index in enumerate(batch_indices):
                path = public_dir / rows[row_index]["image_path"]
                try:
                    tensor, width, height = image_to_tensor(path, torch)
                    tensors.append(tensor.to(device, non_blocking=True))
                    dimensions.append((width, height))
                    valid_positions.append(local_position)
                except Exception as exc:
                    log(f"warning: failed to read inference image {path}: {exc}")
            if tensors:
                try:
                    outputs = model(tensors)
                    for output, (width, height), local_position in zip(outputs, dimensions, valid_positions):
                        batch_results[local_position] = prediction_to_normalized(output, width, height, np)
                except Exception as batch_exc:
                    log(f"warning: batch inference failed, retrying rows: {batch_exc}")
                    for tensor, (width, height), local_position in zip(tensors, dimensions, valid_positions):
                        try:
                            output = model([tensor])[0]
                            batch_results[local_position] = prediction_to_normalized(output, width, height, np)
                        except Exception as row_exc:
                            row_index = batch_indices[local_position]
                            log(f"warning: inference failed for {rows[row_index]['id']}: {row_exc}")
            results.extend(batch_results)
    return results


def pair_features(box_a, box_b, np):
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    aw, ah = max(ax2 - ax1, 1.0e-7), max(ay2 - ay1, 1.0e-7)
    bw, bh = max(bx2 - bx1, 1.0e-7), max(by2 - by1, 1.0e-7)
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    gap_x = max(ax1 - bx2, bx1 - ax2, 0.0)
    gap_y = max(ay1 - by2, by1 - ay2, 0.0)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    enclosing_w = max(ax2, bx2) - min(ax1, bx1)
    enclosing_h = max(ay2, by2) - min(ay1, by1)
    values = [
        gap_x,
        gap_y,
        (gap_x * gap_x + gap_y * gap_y) ** 0.5,
        abs(acx - bcx),
        abs(acy - bcy),
        abs(acx - bcx) / max(aw, bw),
        abs(acy - bcy) / max(ah, bh),
        inter_w / min(aw, bw),
        inter_h / min(ah, bh),
        intersection / max(union, 1.0e-12),
        min(aw, bw) / max(aw, bw),
        min(ah, bh) / max(ah, bh),
        min(aw * ah, bw * bh) / max(aw * ah, bw * bh),
        enclosing_w,
        enclosing_h,
        np.log(max(aw * ah, 1.0e-12)),
        np.log(max(bw * bh, 1.0e-12)),
    ]
    return values


def training_pairs(rows, indices, np):
    features = []
    labels = []
    for row_index in indices:
        objects = row_boxes_and_edges(rows[row_index])
        boxes, edges = objects
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                features.append(pair_features(boxes[first], boxes[second], np))
                labels.append(int((first, second) in edges))
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def fit_contact_model(rows, indices, np):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    features, labels = training_pairs(rows, indices, np)
    if len(features) == 0 or len(np.unique(labels)) < 2:
        log("warning: insufficient pair labels for contact model")
        return None
    class_counts = np.bincount(labels)
    folds = min(3, int(class_counts.min()))
    if folds < 2:
        log("warning: insufficient pair labels for contact cross-validation")
        return None
    model = make_pipeline(
        QuantileTransformer(
            n_quantiles=min(256, len(features)),
            output_distribution="normal",
            random_state=SEED,
        ),
        LogisticRegressionCV(
            Cs=np.logspace(-2.0, 2.0, 7),
            cv=folds,
            scoring="f1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        ),
    )
    model.fit(features, labels)
    log(f"trained contact model on {len(labels)} labeled pairs")
    return model


def pair_probabilities(boxes, contact_model, np):
    count = len(boxes)
    probabilities = np.zeros((count, count), dtype=np.float64)
    if contact_model is None or count < 2:
        return probabilities
    pairs = []
    locations = []
    for first in range(count):
        for second in range(first + 1, count):
            pairs.append(pair_features(boxes[first], boxes[second], np))
            locations.append((first, second))
    try:
        from scipy.special import expit

        features = contact_model.steps[0][1].transform(np.asarray(pairs, dtype=np.float64))
        classifier = contact_model.steps[-1][1]
        logits = np.einsum("ij,j->i", features, classifier.coef_[0]) + classifier.intercept_[0]
        predicted = expit(logits)
        if not np.isfinite(predicted).all():
            raise ValueError("non-finite learned contact probability")
    except Exception as exc:
        log(f"warning: contact inference failed: {exc}")
        return probabilities
    for probability, (first, second) in zip(predicted, locations):
        probabilities[first, second] = probability
        probabilities[second, first] = probability
    return probabilities


def row_boxes_and_edges(row):
    boxes = []
    ids = []
    for obj in row["instances"]:
        box = clean_box(obj.get("bbox"))
        if box is not None:
            boxes.append(box)
            ids.append(str(obj.get("instance_id", len(ids))))
    id_to_index = {instance_id: index for index, instance_id in enumerate(ids)}
    edges = set()
    for first, obj in enumerate(row["instances"]):
        if first >= len(boxes):
            break
        for contact in obj.get("contacts", []):
            second = id_to_index.get(str(contact))
            if second is not None and second != first:
                edges.add(tuple(sorted((first, second))))
    return boxes, edges


def iou_matrix(predicted, truth, np):
    if len(predicted) == 0 or len(truth) == 0:
        return np.zeros((len(predicted), len(truth)), dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    g = np.asarray(truth, dtype=np.float64)
    left = np.maximum(p[:, None, :2], g[None, :, :2])
    right = np.minimum(p[:, None, 2:], g[None, :, 2:])
    sizes = np.maximum(0.0, right - left)
    intersection = sizes[:, :, 0] * sizes[:, :, 1]
    p_area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    g_area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = p_area[:, None] + g_area[None, :] - intersection
    return intersection / np.maximum(union, 1.0e-12)


def match_boxes(predicted, truth, np):
    from scipy.optimize import linear_sum_assignment

    matrix = iou_matrix(predicted, truth, np)
    if matrix.size == 0:
        return {}
    pred_indices, truth_indices = linear_sum_assignment(-matrix)
    return {
        int(pred): int(target)
        for pred, target in zip(pred_indices, truth_indices)
        if matrix[pred, target] >= 0.5
    }


def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else (2.0 * tp) / denominator


def evaluate_predictions(rows, indices, raw_predictions, pair_matrices, detection_threshold, contact_threshold, np):
    detection_values = {"empty": [], "single": [], "multi": []}
    topology_values = {"no_contact": [], "contact": []}
    for local, row_index in enumerate(indices):
        truth_boxes, truth_edges = row_boxes_and_edges(rows[row_index])
        raw = raw_predictions[local]
        keep = raw["scores"] >= detection_threshold
        predicted_boxes = raw["boxes"][keep]
        kept_indices = np.flatnonzero(keep)
        matches = match_boxes(predicted_boxes, truth_boxes, np)
        tp = len(matches)
        fp = len(predicted_boxes) - tp
        fn = len(truth_boxes) - tp
        stratum = "empty" if len(truth_boxes) == 0 else ("single" if len(truth_boxes) == 1 else "multi")
        detection_values[stratum].append(f1_from_counts(tp, fp, fn))

        if len(truth_boxes) >= 2:
            predicted_edges = set()
            matrix = pair_matrices[local]
            for first in range(len(predicted_boxes)):
                for second in range(first + 1, len(predicted_boxes)):
                    if matrix[kept_indices[first], kept_indices[second]] >= contact_threshold:
                        predicted_edges.add((first, second))
            edge_tp = 0
            for first, second in predicted_edges:
                if first in matches and second in matches:
                    mapped = tuple(sorted((matches[first], matches[second])))
                    if mapped in truth_edges:
                        edge_tp += 1
            edge_fp = len(predicted_edges) - edge_tp
            edge_fn = len(truth_edges) - edge_tp
            edge_f1 = f1_from_counts(edge_tp, edge_fp, edge_fn)
            node_coverage = len(matches) / len(truth_boxes)
            topology = 0.5 * (edge_f1 + node_coverage)
            group = "contact" if truth_edges else "no_contact"
            topology_values[group].append(topology)

    detection_parts = [
        float(np.mean(detection_values[group])) if detection_values[group] else 0.0
        for group in ("empty", "single", "multi")
    ]
    topology_parts = [
        float(np.mean(topology_values[group])) if topology_values[group] else 0.0
        for group in ("no_contact", "contact")
    ]
    detection_score = float(np.mean(detection_parts))
    topology_score = float(np.mean(topology_parts))
    final_score = float((max(0.0, detection_score) * max(0.0, topology_score)) ** 0.5)
    return final_score, detection_score, topology_score, detection_parts, topology_parts


def searched_candidates(values, np, count=11):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.asarray([0.5], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, min(count, len(finite)))
    candidates = np.quantile(finite, quantiles)
    candidates = np.concatenate(
        [
            [np.nextafter(float(np.min(finite)), -np.inf)],
            candidates,
            [np.nextafter(float(np.max(finite)), np.inf)],
        ]
    )
    return np.unique(candidates)


def tune_thresholds(rows, valid_indices, predictions, contact_model, np):
    pair_matrices = [pair_probabilities(prediction["boxes"], contact_model, np) for prediction in predictions]
    detection_values = [value for prediction in predictions for value in prediction["scores"]]
    pair_values = []
    for matrix in pair_matrices:
        if len(matrix) >= 2:
            pair_values.extend(matrix[np.triu_indices(len(matrix), 1)].tolist())
    detection_candidates = searched_candidates(detection_values, np)
    contact_candidates = searched_candidates(pair_values, np)
    best = None
    for detection_threshold in detection_candidates:
        for contact_threshold in contact_candidates:
            metrics = evaluate_predictions(
                rows,
                valid_indices,
                predictions,
                pair_matrices,
                float(detection_threshold),
                float(contact_threshold),
                np,
            )
            key = (metrics[0], metrics[1], metrics[2], float(detection_threshold), float(contact_threshold))
            if best is None or key > best[0]:
                best = (key, metrics)
    detection_threshold = best[0][3]
    contact_threshold = best[0][4]
    metrics = best[1]
    log(
        "validation: "
        f"score={metrics[0]:.6f} detection={metrics[1]:.6f} topology={metrics[2]:.6f} "
        f"det_threshold={detection_threshold:.6f} contact_threshold={contact_threshold:.6f}"
    )
    log(f"validation detection strata={metrics[3]} topology strata={metrics[4]}")
    return detection_threshold, contact_threshold, metrics


def format_row_prediction(raw, contact_model, detection_threshold, contact_threshold, np):
    keep = raw["scores"] >= detection_threshold
    boxes = raw["boxes"][keep]
    probabilities = pair_probabilities(boxes, contact_model, np)
    contacts = [[] for _ in range(len(boxes))]
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if probabilities[first, second] >= contact_threshold:
                contacts[first].append(f"p{second + 1}")
                contacts[second].append(f"p{first + 1}")
    instances = []
    for index, box in enumerate(boxes):
        cleaned = clean_box(box.tolist())
        if cleaned is None:
            continue
        instances.append(
            {
                "instance_id": f"p{index + 1}",
                "bbox": [round(float(value), 6) for value in cleaned],
                "contacts": contacts[index],
            }
        )
    valid_ids = {instance["instance_id"] for instance in instances}
    for instance in instances:
        instance["contacts"] = [contact for contact in instance["contacts"] if contact in valid_ids]
    return instances


def validate_submission(test_rows, predictions):
    if len(predictions) != len(test_rows):
        return False, f"row mismatch: {len(predictions)} predictions for {len(test_rows)} test rows"
    ids = [row.get("id") for row in test_rows]
    if len(ids) != len(set(ids)) or any(not identifier for identifier in ids):
        return False, "test IDs are missing or duplicated"
    for row, instances in zip(test_rows, predictions):
        if not isinstance(instances, list):
            return False, f"instances is not a list for {row.get('id')}"
        instance_ids = [instance.get("instance_id") for instance in instances]
        if len(instance_ids) != len(set(instance_ids)):
            return False, f"duplicate instance ID for {row.get('id')}"
        id_set = set(instance_ids)
        for instance in instances:
            box = clean_box(instance.get("bbox"))
            if box is None:
                return False, f"invalid box for {row.get('id')}"
            contacts = instance.get("contacts", [])
            if any(contact not in id_set or contact == instance.get("instance_id") for contact in contacts):
                return False, f"invalid contact for {row.get('id')}"
    return True, "ok"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    started_at = time.monotonic()
    public_dir = Path(sys.argv[1]).expanduser().resolve()
    submission_out = Path(sys.argv[2]).expanduser().resolve()
    test_rows = read_test_rows(public_dir)
    write_submission(submission_out, test_rows, [[] for _ in test_rows])
    log(f"wrote schema-valid placeholder with {len(test_rows)} rows")

    import numpy as np
    import pandas as pd
    import torch

    seed_everything(SEED, torch)
    rows = load_training_rows(public_dir, pd)
    if len(rows) != 1919:
        log(f"warning: observed {len(rows)} training rows; continuing without a size assumption")
    train_indices, valid_indices = make_split(rows, np)
    log(f"split: {len(train_indices)} train, {len(valid_indices)} validation")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size = 4
    else:
        device = torch.device("cpu")
        batch_size = 2
    log(f"device={device}, batch_size={batch_size}")

    model = build_detector(torch).to(device)
    training_dataset = HabitatDataset(rows, train_indices, public_dir, augment=True)
    training_loader = make_loader(training_dataset, torch, shuffle=True, batch_size=batch_size)
    completed = train_detector(
        model,
        training_loader,
        device,
        torch,
        started_at,
        epochs=max(1, min(12, len(rows) // 128)),
        learning_rate=2.0e-4,
    )
    if completed == 0:
        log("warning: detector completed no training epoch; retaining placeholder")
        return

    contact_model = fit_contact_model(rows, train_indices, np)
    if contact_model is None:
        log("warning: contact model did not train; retaining placeholder")
        return
    validation_predictions = predict_rows(
        model,
        rows,
        valid_indices,
        public_dir,
        device,
        torch,
        np,
        batch_size,
    )
    detection_threshold, contact_threshold, _ = tune_thresholds(
        rows,
        valid_indices,
        validation_predictions,
        contact_model,
        np,
    )

    if time.monotonic() - started_at < REFIT_DEADLINE_SECONDS:
        all_indices = list(range(len(rows)))
        refit_dataset = HabitatDataset(rows, all_indices, public_dir, augment=True)
        refit_loader = make_loader(refit_dataset, torch, shuffle=True, batch_size=batch_size)
        train_detector(
            model,
            refit_loader,
            device,
            torch,
            started_at,
            epochs=max(1, min(2, len(rows) // 512)),
            learning_rate=5.0e-5,
        )
        refit_contact_model = fit_contact_model(rows, all_indices, np)
        if refit_contact_model is not None:
            contact_model = refit_contact_model
    else:
        log("wall-clock guard: skipped full-data refit")

    inference_rows = [
        {"id": str(row.get("id", "")), "image_path": str(row.get("image_path", "")), "instances": []}
        for row in test_rows
    ]
    test_predictions = predict_rows(
        model,
        inference_rows,
        list(range(len(inference_rows))),
        public_dir,
        device,
        torch,
        np,
        batch_size,
    )
    formatted = []
    for row, raw in zip(test_rows, test_predictions):
        try:
            formatted.append(
                format_row_prediction(raw, contact_model, detection_threshold, contact_threshold, np)
            )
        except Exception as exc:
            log(f"warning: post-processing failed for {row.get('id')}: {exc}")
            formatted.append([])

    valid, message = validate_submission(test_rows, formatted)
    if not valid:
        log(f"warning: final predictions failed validation ({message}); retaining placeholder")
        return
    write_submission(submission_out, test_rows, formatted)
    log(
        f"wrote {len(formatted)} predictions to {submission_out}; "
        f"elapsed={time.monotonic() - started_at:.1f}s"
    )


if __name__ == "__main__":
    main()
