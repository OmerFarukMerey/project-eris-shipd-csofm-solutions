#!/usr/bin/env python3
"""End-to-end neural ECG beat and rhythm transcription."""

import json
import math
import random
import sys
import time
import warnings
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 20260720
SIGNAL_LENGTH = 750
MAX_EVENTS = 64
MAX_EPOCHS = 30
MIN_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 6
TRAINING_CUTOFF_SECONDS = 3000.0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_json_events(value):
    events = json.loads(value) if isinstance(value, str) else value
    output = []
    for event in events:
        if not isinstance(event, (list, tuple)) or len(event) != 2:
            continue
        output.append((int(event[0]), str(event[1])))
    return output


def load_one_signal(path, length):
    values = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if values.size != length:
        if values.size < 2:
            values = np.zeros(length, dtype=np.float32)
        else:
            source_grid = np.linspace(0.0, 1.0, values.size)
            target_grid = np.linspace(0.0, 1.0, length)
            values = np.interp(target_grid, source_grid, values).astype(np.float32)
    finite = np.isfinite(values)
    if not finite.all():
        replacement = float(np.median(values[finite])) if finite.any() else 0.0
        values = values.copy()
        values[~finite] = replacement
    return values


def signal_to_channels(values):
    """Per-window deterministic transform; it never fits cross-row state."""
    values = np.asarray(values, dtype=np.float32)
    center = float(np.median(values))
    centered = values - center
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(scale) or scale < 1e-5:
        scale = 1.0
    normalized = centered / scale
    gradient = np.gradient(normalized).astype(np.float32)
    return np.stack(
        [normalized.astype(np.float32), gradient, np.abs(gradient)], axis=0
    )


def load_signal_frame(frame, public_dir, length, tolerate_errors):
    channels = np.zeros((len(frame), 3, length), dtype=np.float32)
    failures = np.zeros(len(frame), dtype=bool)
    for row_number, relative_path in enumerate(frame["signal"].astype(str)):
        try:
            channels[row_number] = signal_to_channels(
                load_one_signal(public_dir / relative_path, length)
            )
        except Exception as exc:
            if not tolerate_errors:
                raise RuntimeError(
                    f"failed to read required training signal {relative_path}: {exc}"
                ) from exc
            failures[row_number] = True
            log(f"warning: test row {row_number} signal failed; using a zero waveform")
    return channels, failures


def waveform_feature_rows(channels):
    """Per-window waveform descriptors; no statistic is fit across rows."""
    signal = channels[:, 0].astype(np.float64)
    spectrum = np.log1p(np.abs(np.fft.rfft(signal, axis=1))[:, 1:65])
    quantiles = np.quantile(
        signal, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=1
    ).T
    differences = np.diff(signal, axis=1)
    difference_quantiles = np.quantile(
        differences, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=1
    ).T
    usable = signal[:, : (signal.shape[1] // 30) * 30]
    pooled_energy = np.mean(
        np.abs(usable.reshape(len(signal), -1, 30)), axis=2
    )
    return np.concatenate(
        [spectrum, quantiles, difference_quantiles, pooled_energy], axis=1
    ).astype(np.float32)


def morphology_features(channels):
    """Train-only standardized descriptors for morphology-neighbour OOF groups."""
    features = waveform_feature_rows(channels).astype(np.float64)
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(scale, 1e-6)).astype(np.float32)


def patient_proxy_features(channels, parsed_events, radius=40):
    """Train-only median QRS shapes for conservative patient-proxy grouping."""
    signal = channels[:, 0].astype(np.float64)
    templates = np.zeros((len(signal), radius * 2 + 1), dtype=np.float64)
    for row_number, events in enumerate(parsed_events):
        snippets = []
        for position, _ in events:
            if position < radius or position + radius >= signal.shape[1]:
                continue
            snippet = signal[
                row_number, position - radius : position + radius + 1
            ].copy()
            edge = np.concatenate([snippet[:10], snippet[-10:]])
            snippet -= np.median(edge)
            if snippet[radius] < 0:
                snippet *= -1.0
            snippet /= max(float(np.sqrt(np.mean(snippet * snippet))), 1e-6)
            snippets.append(snippet)
        if snippets:
            templates[row_number] = np.median(np.stack(snippets), axis=0)
    template_mean = templates.mean(axis=0, keepdims=True)
    template_scale = templates.std(axis=0, keepdims=True)
    templates = (templates - template_mean) / np.maximum(template_scale, 1e-6)
    generic = morphology_features(channels)
    return np.concatenate([templates, generic[:, :64]], axis=1).astype(np.float32)


def make_morphology_groups(features, group_size=6):
    """Bind nearest QRS morphologies without consulting rhythm labels."""
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(features, metric="euclidean")
    neighbours = np.argsort(distances, axis=1, kind="stable")
    nearest_distance = distances[np.arange(len(features)), neighbours[:, 1]]
    priority = np.argsort(nearest_distance, kind="stable")
    assigned = np.zeros(len(features), dtype=bool)
    groups = np.full(len(features), -1, dtype=np.int32)
    next_group = 0
    for anchor in priority:
        if assigned[anchor]:
            continue
        members = [int(anchor)]
        for neighbour in neighbours[anchor, 1:]:
            if not assigned[neighbour]:
                members.append(int(neighbour))
                if len(members) == group_size:
                    break
        assigned[members] = True
        groups[members] = next_group
        next_group += 1
    return groups


def grouped_stratified_folds(features, labels, requested_folds, seed):
    """Balance global morphology groups while keeping every group in one fold."""
    label_values = sorted(set(labels.tolist()))
    minimum_class_count = min(Counter(labels.tolist()).values())
    fold_count = min(requested_folds, minimum_class_count)
    if fold_count < 2:
        return []
    label_to_index = {label: index for index, label in enumerate(label_values)}
    numeric_labels = np.asarray([label_to_index[label] for label in labels])
    groups = make_morphology_groups(features)
    unique_groups = np.unique(groups)
    group_counts = np.zeros(
        (len(unique_groups), len(label_values)), dtype=np.int32
    )
    group_sizes = np.zeros(len(unique_groups), dtype=np.int32)
    for group_index, group in enumerate(unique_groups):
        members = np.flatnonzero(groups == group)
        group_counts[group_index] = np.bincount(
            numeric_labels[members], minlength=len(label_values)
        )
        group_sizes[group_index] = len(members)
    total_counts = group_counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique_groups))
    rng.shuffle(order)
    rarity = np.max(group_counts / np.maximum(total_counts, 1), axis=1)
    order = order[np.argsort(-rarity[order], kind="stable")]
    fold_counts = np.zeros((fold_count, len(label_values)), dtype=np.int32)
    fold_sizes = np.zeros(fold_count, dtype=np.int32)
    group_fold = np.full(len(unique_groups), -1, dtype=np.int16)
    for group_index in order:
        best_choice = None
        for fold in range(fold_count):
            fold_counts[fold] += group_counts[group_index]
            fold_sizes[fold] += group_sizes[group_index]
            class_imbalance = float(
                np.mean(
                    np.std(
                        fold_counts / np.maximum(total_counts, 1),
                        axis=0,
                    )
                )
            )
            size_imbalance = float(np.std(fold_sizes / max(len(labels), 1)))
            objective = class_imbalance + 0.1 * size_imbalance
            fold_counts[fold] -= group_counts[group_index]
            fold_sizes[fold] -= group_sizes[group_index]
            choice = (objective, int(fold_sizes[fold]), fold)
            if best_choice is None or choice < best_choice:
                best_choice = choice
        destination = best_choice[2]
        group_fold[group_index] = destination
        fold_counts[destination] += group_counts[group_index]
        fold_sizes[destination] += group_sizes[group_index]
    row_fold = np.full(len(labels), -1, dtype=np.int16)
    for group_index, group in enumerate(unique_groups):
        row_fold[groups == group] = group_fold[group_index]
    folds = []
    for fold in range(fold_count):
        valid_indices = np.flatnonzero(row_fold == fold)
        train_indices = np.flatnonzero(row_fold != fold)
        folds.append((train_indices, valid_indices))
    return folds


def build_training_targets(frame, beat_labels, rhythm_to_index, tolerance, length):
    beat_to_index = {label: index for index, label in enumerate(beat_labels)}
    detector = np.zeros((len(frame), length), dtype=np.float32)
    beat_type = np.full((len(frame), length), -100, dtype=np.int64)
    rhythm = np.asarray(
        [rhythm_to_index[value] for value in frame["rhythm_family"]], dtype=np.int64
    )
    parsed_events = []
    sigma = max(float(tolerance) / 3.0, 1.0)
    radius = max(int(tolerance), 1)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2).astype(np.float32)
    for row_number, raw_events in enumerate(frame["beats"]):
        events = safe_json_events(raw_events)
        clean_events = []
        for position, label in events:
            if label not in beat_to_index or not 0 <= position < length:
                continue
            clean_events.append((position, label))
            lo = max(0, position - radius)
            hi = min(length, position + radius + 1)
            kernel_lo = lo - (position - radius)
            kernel_hi = kernel_lo + (hi - lo)
            detector[row_number, lo:hi] = np.maximum(
                detector[row_number, lo:hi], kernel[kernel_lo:kernel_hi]
            )
            beat_type[row_number, position] = beat_to_index[label]
        parsed_events.append(clean_events)
    return detector, beat_type, rhythm, parsed_events


class ECGDataset(Dataset):
    def __init__(self, signals, detector, beat_type, rhythm, indices, augment=False):
        self.signals = signals
        self.detector = detector
        self.beat_type = beat_type
        self.rhythm = rhythm
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        values = torch.from_numpy(self.signals[index])
        detector = torch.from_numpy(self.detector[index])
        beat_type = torch.from_numpy(self.beat_type[index])
        if self.augment:
            raw = values[0].clone()
            detector = detector.clone()
            beat_type = beat_type.clone()
            shift = int(torch.randint(-7, 8, (1,)).item())
            if shift > 0:
                raw = torch.cat([raw[:1].expand(shift), raw[:-shift]])
                detector = torch.cat([detector.new_zeros(shift), detector[:-shift]])
                beat_type = torch.cat(
                    [
                        torch.full((shift,), -100, dtype=beat_type.dtype),
                        beat_type[:-shift],
                    ]
                )
            elif shift < 0:
                width = -shift
                raw = torch.cat([raw[width:], raw[-1:].expand(width)])
                detector = torch.cat([detector[width:], detector.new_zeros(width)])
                beat_type = torch.cat(
                    [
                        beat_type[width:],
                        torch.full((width,), -100, dtype=beat_type.dtype),
                    ]
                )
            if torch.rand(()) < 0.5:
                raw = -raw
            if torch.rand(()) < 0.8:
                coordinate = torch.linspace(-1.0, 1.0, len(raw))
                linear = (torch.rand(()) - 0.5) * 0.30
                quadratic = (torch.rand(()) - 0.5) * 0.18
                raw = raw * (
                    1.0
                    + linear * coordinate
                    + quadratic * (coordinate.square() - 1.0 / 3.0)
                )
            if torch.rand(()) < 0.5:
                smoothed = F.avg_pool1d(
                    F.pad(raw[None, None], (2, 2), mode="reflect"),
                    kernel_size=5,
                    stride=1,
                ).squeeze()
                mixture = torch.rand(()) * 0.35
                raw = (1.0 - mixture) * raw + mixture * smoothed
            raw = raw + (torch.rand(()) * 0.025) * torch.randn_like(raw)
            raw = raw - torch.median(raw)
            raw = raw / torch.sqrt(torch.mean(raw.square()) + 1e-6)
            gradient = torch.gradient(raw)[0]
            values = torch.stack([raw, gradient, torch.abs(gradient)])
        return (
            values,
            detector,
            beat_type,
            torch.tensor(self.rhythm[index], dtype=torch.long),
        )


class DilatedResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(12, width)
        self.dilated = nn.Conv1d(
            width,
            width * 2,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.project = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = self.norm(values)
        residual = F.glu(self.dilated(residual), dim=1)
        residual = self.dropout(residual)
        return values + self.project(residual)


class ECGTranscriber(nn.Module):
    def __init__(self, input_channels, beat_classes, rhythm_classes, detector_prior):
        super().__init__()
        width = 96
        self.stem = nn.Conv1d(input_channels, width, kernel_size=9, padding=4)
        dilations = (1, 2, 4, 8, 16, 32, 64, 1, 4, 16)
        self.blocks = nn.ModuleList(
            [DilatedResidualBlock(width, dilation, 0.08) for dilation in dilations]
        )
        self.final_norm = nn.GroupNorm(12, width)
        self.detector_head = nn.Sequential(
            nn.Conv1d(width, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.type_head = nn.Sequential(
            nn.Conv1d(width, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, beat_classes, kernel_size=1),
        )
        self.event_encoder = nn.Sequential(
            nn.Conv1d(beat_classes + 1, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=9, padding=16, dilation=4),
            nn.GELU(),
        )
        self.rhythm_attention = nn.Conv1d(width, 1, kernel_size=1)
        self.rhythm_head = nn.Sequential(
            nn.Linear(width * 3 + 64, width * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(width * 2, rhythm_classes),
        )
        prior = float(np.clip(detector_prior, 1e-4, 1.0 - 1e-4))
        nn.init.constant_(self.detector_head[-1].bias, math.log(prior / (1.0 - prior)))

    def forward(self, values):
        features = self.stem(values)
        for block in self.blocks:
            features = block(features)
        features = F.gelu(self.final_norm(features))
        detector_logits = self.detector_head(features).squeeze(1)
        type_logits = self.type_head(features)
        detector_probability = torch.sigmoid(detector_logits).unsqueeze(1)
        type_probability = torch.softmax(type_logits, dim=1)
        event_sequence = torch.cat(
            [detector_probability, detector_probability * type_probability], dim=1
        )
        event_features = self.event_encoder(event_sequence)
        event_average = torch.mean(event_features, dim=-1)
        event_maximum = torch.amax(event_features, dim=-1)
        attention = torch.softmax(self.rhythm_attention(features), dim=-1)
        attended = torch.sum(features * attention, dim=-1)
        average = torch.mean(features, dim=-1)
        maximum = torch.amax(features, dim=-1)
        rhythm_logits = self.rhythm_head(
            torch.cat(
                [attended, average, maximum, event_average, event_maximum], dim=1
            )
        )
        return detector_logits, type_logits, rhythm_logits


def class_balanced_weights(counts, device, exponent):
    counts = np.asarray(counts, dtype=np.float64)
    positive = counts > 0
    raw = np.ones_like(counts)
    ratio = counts[positive].sum() / (positive.sum() * counts[positive])
    raw[positive] = np.power(ratio, exponent)
    normalizer = float(np.sum(raw[positive] * counts[positive]) / counts[positive].sum())
    raw /= max(normalizer, 1e-8)
    return torch.tensor(raw, dtype=torch.float32, device=device)


def heatmap_focal_loss(logits, targets):
    probabilities = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    positive = targets.eq(1.0)
    negative = targets.lt(1.0)
    negative_weight = torch.pow(1.0 - targets, 4.0)
    positive_loss = -torch.log(probabilities) * torch.pow(1.0 - probabilities, 2.0)
    negative_loss = (
        -torch.log(1.0 - probabilities)
        * torch.pow(probabilities, 2.0)
        * negative_weight
    )
    positive_loss = positive_loss[positive].sum()
    negative_loss = negative_loss[negative].sum()
    count = positive.sum().clamp(min=1)
    return (positive_loss + negative_loss) / count


def multitask_loss(outputs, detector_target, type_target, rhythm_target, type_weight, rhythm_weight):
    detector_logits, type_logits, rhythm_logits = outputs
    detector_loss = heatmap_focal_loss(detector_logits, detector_target)
    type_loss = F.cross_entropy(
        type_logits, type_target, weight=type_weight, ignore_index=-100
    )
    rhythm_loss = F.cross_entropy(rhythm_logits, rhythm_target, weight=rhythm_weight)
    return detector_loss + type_loss + rhythm_loss, (
        float(detector_loss.detach().cpu()),
        float(type_loss.detach().cpu()),
        float(rhythm_loss.detach().cpu()),
    )


def validation_loss(model, loader, device, type_weight, rhythm_weight, amp_enabled):
    model.eval()
    total = 0.0
    count = 0
    components = np.zeros(3, dtype=np.float64)
    with torch.inference_mode():
        for signals, detector, beat_type, rhythm in loader:
            signals = signals.to(device, non_blocking=True)
            detector = detector.to(device, non_blocking=True)
            beat_type = beat_type.to(device, non_blocking=True)
            rhythm = rhythm.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss, pieces = multitask_loss(
                    model(signals), detector, beat_type, rhythm, type_weight, rhythm_weight
                )
            batch_count = signals.shape[0]
            total += float(loss.detach().cpu()) * batch_count
            components += np.asarray(pieces) * batch_count
            count += batch_count
    return total / max(count, 1), components / max(count, 1)


def fit_fold(
    fold_number,
    train_indices,
    valid_indices,
    signals,
    detector,
    beat_type,
    rhythm,
    device,
    balance_power,
    variant_number,
    augment,
    started_at,
):
    seed_everything(SEED + fold_number + 1000 * variant_number)
    batch_size = 64 if device.type != "cpu" else 32
    train_dataset = ECGDataset(
        signals, detector, beat_type, rhythm, train_indices, augment=augment
    )
    valid_dataset = ECGDataset(
        signals, detector, beat_type, rhythm, valid_indices
    )
    generator = torch.Generator().manual_seed(SEED + fold_number + 1000 * variant_number)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rhythm_counts = np.bincount(rhythm[train_indices], minlength=int(rhythm.max()) + 1)
    type_values = beat_type[train_indices]
    type_counts = np.bincount(
        type_values[type_values >= 0], minlength=4
    )
    rhythm_weight = class_balanced_weights(rhythm_counts, device, balance_power)
    type_weight = class_balanced_weights(type_counts, device, balance_power)
    event_count = int(np.sum(type_values >= 0))
    detector_prior = event_count / max(len(train_indices) * detector.shape[1], 1)
    model = ECGTranscriber(
        signals.shape[1], len(type_counts), len(rhythm_counts), detector_prior
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1.0e-5
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            signal_batch, detector_batch, type_batch, rhythm_batch = batch
            signal_batch = signal_batch.to(device, non_blocking=True)
            detector_batch = detector_batch.to(device, non_blocking=True)
            type_batch = type_batch.to(device, non_blocking=True)
            rhythm_batch = rhythm_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss, _ = multitask_loss(
                    model(signal_batch),
                    detector_batch,
                    type_batch,
                    rhythm_batch,
                    type_weight,
                    rhythm_weight,
                )
            if not torch.isfinite(loss):
                log(f"warning: fold {fold_number + 1} skipped a non-finite batch")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_count = signal_batch.shape[0]
            running += float(loss.detach().cpu()) * batch_count
            seen += batch_count
        scheduler.step()
        current_loss, pieces = validation_loss(
            model, valid_loader, device, type_weight, rhythm_weight, amp_enabled
        )
        improved = current_loss < best_loss - 1e-4
        if improved:
            best_loss = current_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        log(
            f"variant {variant_number + 1} fold {fold_number + 1} "
            f"epoch {epoch + 1:02d}: train={running / max(seen, 1):.4f} "
            f"valid={current_loss:.4f} (det={pieces[0]:.3f}, "
            f"type={pieces[1]:.3f}, rhythm={pieces[2]:.3f})"
        )
        if (
            epoch + 1 >= MIN_EPOCHS
            and stale_epochs >= EARLY_STOPPING_PATIENCE
        ):
            break
        if time.monotonic() - started_at >= TRAINING_CUTOFF_SECONDS:
            log("wall-clock guard reached during training; moving to inference")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model




def predict_probabilities(model, signals, device, batch_size=128, polarity_tta=True):
    model.eval()
    detector_outputs = []
    type_outputs = []
    rhythm_outputs = []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            values = torch.from_numpy(signals[start : start + batch_size]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                detector_logits, type_logits, rhythm_logits = model(values)
                detector_probability = torch.sigmoid(detector_logits)
                type_probability = torch.softmax(type_logits, dim=1)
                rhythm_probability = torch.softmax(rhythm_logits, dim=1)
                if polarity_tta:
                    inverted = values.clone()
                    inverted[:, :2] = -inverted[:, :2]
                    inverted_detector, inverted_type, inverted_rhythm = model(inverted)
                    detector_probability = (
                        detector_probability + torch.sigmoid(inverted_detector)
                    ) * 0.5
                    type_probability = (
                        type_probability + torch.softmax(inverted_type, dim=1)
                    ) * 0.5
                    rhythm_probability = (
                        rhythm_probability + torch.softmax(inverted_rhythm, dim=1)
                    ) * 0.5
            detector_outputs.append(detector_probability.float().cpu().numpy())
            type_outputs.append(type_probability.float().cpu().numpy())
            rhythm_outputs.append(rhythm_probability.float().cpu().numpy())
    return (
        np.concatenate(detector_outputs, axis=0),
        np.concatenate(type_outputs, axis=0),
        np.concatenate(rhythm_outputs, axis=0),
    )


def prepare_peak_candidates(detector_probability, type_probability):
    rows = []
    for detector_row, type_row in zip(detector_probability, type_probability):
        left = np.empty_like(detector_row)
        right = np.empty_like(detector_row)
        left[0] = -np.inf
        left[1:] = detector_row[:-1]
        right[-1] = -np.inf
        right[:-1] = detector_row[1:]
        positions = np.flatnonzero(
            (detector_row >= left) & (detector_row > right)
        ).astype(np.int16)
        scores = detector_row[positions].astype(np.float32)
        log_types = np.log(
            np.maximum(type_row[:, positions].T, 1e-8)
        ).astype(np.float32)
        rows.append((positions, scores, log_types))
    return rows


def decode_candidate_rows(candidates, thresholds, minimum_distance, type_bias):
    predictions = []
    for positions, scores, log_types in candidates:
        if len(positions) == 0:
            predictions.append([])
            continue
        predicted_types = np.argmax(log_types + type_bias[None, :], axis=1)
        eligible = np.flatnonzero(scores >= thresholds[predicted_types])
        if len(eligible) == 0:
            predictions.append([])
            continue
        ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
        accepted = []
        for candidate in ranked:
            position = int(positions[candidate])
            if all(abs(position - other[0]) >= minimum_distance for other in accepted):
                accepted.append((position, int(predicted_types[candidate])))
                if len(accepted) == MAX_EVENTS:
                    break
        accepted.sort(key=lambda event: event[0])
        predictions.append(accepted)
    return predictions


def event_true_positive_count(predicted, truth, beat_type, tolerance):
    predicted_positions = [position for position, label in predicted if label == beat_type]
    true_positions = [position for position, label in truth if label == beat_type]
    pairs = []
    for predicted_index, predicted_position in enumerate(predicted_positions):
        for true_index, true_position in enumerate(true_positions):
            error = abs(predicted_position - true_position)
            if error <= tolerance:
                pairs.append((error, predicted_index, true_index))
    pairs.sort()
    used_predicted = set()
    used_truth = set()
    matches = 0
    for _, predicted_index, true_index in pairs:
        if predicted_index in used_predicted or true_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(true_index)
        matches += 1
    return matches, len(predicted_positions), len(true_positions)


def f1_from_counts(true_positive, predicted_count, truth_count):
    denominator = predicted_count + truth_count
    return 2.0 * true_positive / denominator if denominator else 0.0


def beat_metric(predictions, truths, beat_labels, rare_labels, tolerance):
    totals = np.zeros((len(beat_labels), 3), dtype=np.int64)
    for predicted, truth in zip(predictions, truths):
        predicted_named = [(p, beat_labels[t]) for p, t in predicted]
        for type_index, label in enumerate(beat_labels):
            matched, predicted_count, truth_count = event_true_positive_count(
                predicted_named, truth, label, tolerance
            )
            totals[type_index] += (matched, predicted_count, truth_count)
    per_type = np.asarray(
        [f1_from_counts(*row) for row in totals], dtype=np.float64
    )
    total_matched, total_predicted, total_truth = totals.sum(axis=0)
    micro = f1_from_counts(total_matched, total_predicted, total_truth)
    present = totals[:, 2] > 0
    macro = float(per_type[present].mean()) if present.any() else 0.0
    rare_indices = [beat_labels.index(label) for label in rare_labels if label in beat_labels]
    rare = float(per_type[rare_indices].mean()) if rare_indices else 0.0
    weighted = 0.20 * micro + 0.30 * macro + 0.10 * rare
    return weighted, {
        "event_micro_F1": micro,
        "beat_type_macro_F1": macro,
        "rare_beat_macro_F1": rare,
        "beat_type_F1": dict(zip(beat_labels, per_type.tolist())),
    }


def rhythm_metric(probabilities, truths, rhythm_labels, rare_labels, bias):
    predicted = np.argmax(
        np.log(np.maximum(probabilities, 1e-8)) + bias[None, :], axis=1
    )
    per_class = []
    for class_index in range(len(rhythm_labels)):
        true_positive = int(np.sum((predicted == class_index) & (truths == class_index)))
        predicted_count = int(np.sum(predicted == class_index))
        truth_count = int(np.sum(truths == class_index))
        per_class.append(f1_from_counts(true_positive, predicted_count, truth_count))
    per_class = np.asarray(per_class, dtype=np.float64)
    present = np.asarray(
        [np.any(truths == index) for index in range(len(rhythm_labels))]
    )
    macro = float(per_class[present].mean()) if present.any() else 0.0
    rare_indices = [
        rhythm_labels.index(label) for label in rare_labels if label in rhythm_labels
    ]
    rare = float(per_class[rare_indices].mean()) if rare_indices else 0.0
    weighted = 0.30 * macro + 0.10 * rare
    return weighted, {
        "rhythm_macro_F1": macro,
        "rare_rhythm_macro_F1": rare,
        "rhythm_F1": dict(zip(rhythm_labels, per_class.tolist())),
        "predicted": predicted,
    }


def calibrate_rhythm(probabilities, truths, rhythm_labels, rare_labels):
    bias = np.zeros(len(rhythm_labels), dtype=np.float32)
    best_score, _ = rhythm_metric(
        probabilities, truths, rhythm_labels, rare_labels, bias
    )
    grid = np.linspace(-1.5, 1.5, 13, dtype=np.float32)
    for _ in range(2):
        changed = False
        for class_index in range(1, len(rhythm_labels)):
            local_best = best_score
            local_value = float(bias[class_index])
            for value in grid:
                proposal = bias.copy()
                proposal[class_index] = value
                score, _ = rhythm_metric(
                    probabilities, truths, rhythm_labels, rare_labels, proposal
                )
                if score > local_best + 1e-10:
                    local_best = score
                    local_value = float(value)
            if local_best > best_score + 1e-10:
                bias[class_index] = local_value
                best_score = local_best
                changed = True
        if not changed:
            break
    return bias


def calibrate_beats(candidates, truths, beat_labels, rare_labels, tolerance):
    type_bias = np.zeros(len(beat_labels), dtype=np.float32)
    threshold = np.full(len(beat_labels), 0.30, dtype=np.float32)
    best_distance = max(int(tolerance * 2), 1)
    best_score = -1.0
    threshold_grid = np.linspace(0.05, 0.75, 15, dtype=np.float32)
    distance_grid = sorted(
        set(max(1, int(round(tolerance * factor))) for factor in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0))
    )
    for distance in distance_grid:
        for value in threshold_grid:
            proposal_threshold = np.full(len(beat_labels), value, dtype=np.float32)
            predictions = decode_candidate_rows(
                candidates, proposal_threshold, distance, type_bias
            )
            score, _ = beat_metric(
                predictions, truths, beat_labels, rare_labels, tolerance
            )
            if score > best_score:
                best_score = score
                threshold = proposal_threshold
                best_distance = distance
    bias_grid = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
    for class_index in range(1, len(beat_labels)):
        local_best = best_score
        local_value = float(type_bias[class_index])
        for value in bias_grid:
            proposal_bias = type_bias.copy()
            proposal_bias[class_index] = value
            predictions = decode_candidate_rows(
                candidates, threshold, best_distance, proposal_bias
            )
            score, _ = beat_metric(
                predictions, truths, beat_labels, rare_labels, tolerance
            )
            if score > local_best + 1e-10:
                local_best = score
                local_value = float(value)
        type_bias[class_index] = local_value
        best_score = local_best
    lower = max(0.02, float(threshold[0]) - 0.25)
    upper = min(0.95, float(threshold[0]) + 0.25)
    class_threshold_grid = np.linspace(lower, upper, 15, dtype=np.float32)
    for class_index in range(len(beat_labels)):
        local_best = best_score
        local_value = float(threshold[class_index])
        for value in class_threshold_grid:
            proposal_threshold = threshold.copy()
            proposal_threshold[class_index] = value
            predictions = decode_candidate_rows(
                candidates, proposal_threshold, best_distance, type_bias
            )
            score, _ = beat_metric(
                predictions, truths, beat_labels, rare_labels, tolerance
            )
            if score > local_best + 1e-10:
                local_best = score
                local_value = float(value)
        threshold[class_index] = local_value
        best_score = local_best
    return threshold, best_distance, type_bias


def simplex_blend_grid(component_count, units=5):
    if component_count < 1:
        return []
    return [
        np.asarray(values, dtype=np.float32) / float(units)
        for values in product(range(units + 1), repeat=component_count)
        if sum(values) == units
    ]


def select_beat_decoder(
    variant_results,
    rows,
    truths,
    blend_grid,
    beat_labels,
    rare_labels,
    tolerance,
    log_results=False,
):
    best_score = -1.0
    best = None
    for blend_weights in blend_grid:
        detector_probability = sum(
            result["oof_detector"][rows] * float(weight)
            for result, weight in zip(variant_results, blend_weights)
        )
        type_probability = sum(
            result["oof_type"][rows] * float(weight)
            for result, weight in zip(variant_results, blend_weights)
        )
        candidates = prepare_peak_candidates(detector_probability, type_probability)
        thresholds, minimum_distance, type_bias = calibrate_beats(
            candidates, truths, beat_labels, rare_labels, tolerance
        )
        predictions = decode_candidate_rows(
            candidates, thresholds, minimum_distance, type_bias
        )
        score, parts = beat_metric(
            predictions, truths, beat_labels, rare_labels, tolerance
        )
        if log_results:
            log(
                f"OOF beat blend weights={blend_weights.tolist()}: "
                f"weighted={score:.6f}, per-type={parts['beat_type_F1']}"
            )
        if score > best_score:
            best_score = score
            best = (
                blend_weights.copy(),
                thresholds,
                minimum_distance,
                type_bias,
                parts,
            )
    return best_score, best


def cross_fitted_named_events(
    variant_results,
    folds,
    blend_grid,
    parsed_events,
    beat_labels,
    rare_labels,
    tolerance,
):
    named_events = [None] * len(parsed_events)
    for fold_number, (train_indices, valid_indices) in enumerate(folds):
        train_truth = [parsed_events[index] for index in train_indices]
        _, selected = select_beat_decoder(
            variant_results,
            train_indices,
            train_truth,
            blend_grid,
            beat_labels,
            rare_labels,
            tolerance,
        )
        blend, thresholds, distance, type_bias, _ = selected
        detector_probability = sum(
            result["oof_detector"][valid_indices] * float(weight)
            for result, weight in zip(variant_results, blend)
        )
        type_probability = sum(
            result["oof_type"][valid_indices] * float(weight)
            for result, weight in zip(variant_results, blend)
        )
        predictions = decode_candidate_rows(
            prepare_peak_candidates(detector_probability, type_probability),
            thresholds,
            distance,
            type_bias,
        )
        for row_number, events in zip(valid_indices, predictions):
            named_events[row_number] = [
                (position, beat_labels[type_index])
                for position, type_index in events
            ]
        log(
            f"cross-fitted rhythm features fold {fold_number + 1}: "
            f"beat blend={blend.tolist()}"
        )
    return named_events


def sequence_features(event_rows, beat_labels, length):
    beat_to_index = {label: index for index, label in enumerate(beat_labels)}
    output = []
    for events in event_rows:
        events = sorted(
            (int(position), str(label))
            for position, label in (events or [])
            if str(label) in beat_to_index and 0 <= int(position) < length
        )
        positions = np.asarray([position for position, _ in events], dtype=np.float32)
        types = np.asarray(
            [beat_to_index[label] for _, label in events], dtype=np.int64
        )
        counts = np.bincount(types, minlength=len(beat_labels)).astype(np.float32)
        proportions = counts / max(len(events), 1)
        intervals = np.diff(positions)
        if len(intervals):
            interval_difference = np.diff(intervals)
            interval_features = np.asarray(
                [
                    float(np.mean(intervals)),
                    float(np.std(intervals)),
                    float(np.min(intervals)),
                    float(np.max(intervals)),
                    *np.quantile(intervals, [0.10, 0.25, 0.50, 0.75, 0.90]),
                    float(np.mean(np.abs(interval_difference)))
                    if len(interval_difference)
                    else 0.0,
                    float(np.std(interval_difference))
                    if len(interval_difference)
                    else 0.0,
                ],
                dtype=np.float32,
            )
        else:
            interval_features = np.zeros(11, dtype=np.float32)
        edge_features = np.asarray(
            [
                float(positions[0]) if len(positions) else 0.0,
                float(length - positions[-1]) if len(positions) else 0.0,
                float(len(positions)),
            ],
            dtype=np.float32,
        )
        transitions = np.zeros(
            (len(beat_labels), len(beat_labels)), dtype=np.float32
        )
        for first, second in zip(types[:-1], types[1:]):
            transitions[first, second] += 1.0
        transitions /= max(len(types) - 1, 1)
        output.append(
            np.concatenate(
                [
                    counts,
                    proportions,
                    interval_features,
                    edge_features,
                    transitions.ravel(),
                ]
            )
        )
    return np.stack(output).astype(np.float32)


def rhythm_morphology_features(channels, event_rows, beat_labels, radius=40):
    beat_to_index = {label: index for index, label in enumerate(beat_labels)}
    output = []
    for values, raw_events in zip(channels, event_rows):
        events = sorted(
            (int(position), str(label))
            for position, label in (raw_events or [])
            if str(label) in beat_to_index and 0 <= int(position) < values.shape[1]
        )
        signal = values[0]
        padded = np.pad(signal, (radius, radius), mode="reflect")
        patches = []
        scalars = []
        type_indices = []
        for position, label in events:
            patch = padded[position : position + 2 * radius + 1].astype(
                np.float64, copy=True
            )
            patch -= np.median(np.concatenate([patch[:8], patch[-8:]]))
            raw_amplitude = float(patch[radius])
            center = patch[radius - 4 : radius + 5]
            if center[np.argmax(np.abs(center))] < 0.0:
                patch *= -1.0
            rms = max(float(np.sqrt(np.mean(patch * patch))), 1e-6)
            normalized = patch / rms
            gradient = np.diff(normalized)
            patches.append(normalized[::5])
            type_indices.append(beat_to_index[label])
            scalars.append(
                [
                    raw_amplitude,
                    float(np.ptp(patch)),
                    rms,
                    float(np.max(patch)),
                    float(np.min(patch)),
                    float(np.max(np.abs(gradient))),
                    float(np.mean(np.abs(gradient))),
                    float(np.mean(np.abs(normalized[: radius // 2]))),
                    float(np.mean(np.abs(normalized[radius // 2 : radius]))),
                    float(
                        np.mean(
                            np.abs(
                                normalized[
                                    radius + 1 : radius + 1 + radius // 2
                                ]
                            )
                        )
                    ),
                    float(
                        np.mean(
                            np.abs(normalized[radius + 1 + radius // 2 :])
                        )
                    ),
                    float(np.sum(np.abs(normalized[radius - 8 : radius + 9]))),
                    float(np.argmax(patch) - radius),
                    float(np.argmin(patch) - radius),
                ]
            )
        if patches:
            patch_array = np.asarray(patches)
            scalar_array = np.asarray(scalars)
            type_array = np.asarray(type_indices)
            median_patch = np.median(patch_array, axis=0)
            deviations = np.sqrt(
                np.mean((patch_array - median_patch[None, :]) ** 2, axis=1)
            )
            aggregate = np.concatenate(
                [
                    np.mean(patch_array, axis=0),
                    np.std(patch_array, axis=0),
                    np.quantile(scalar_array, [0.10, 0.50, 0.90], axis=0).ravel(),
                    np.mean(scalar_array, axis=0),
                    np.std(scalar_array, axis=0),
                    np.quantile(deviations, [0.0, 0.25, 0.50, 0.75, 1.0]),
                    [float(np.mean(deviations)), float(np.std(deviations))],
                ]
            )
            typed = []
            for type_index in range(len(beat_labels)):
                selected = type_array == type_index
                if np.any(selected):
                    typed.extend(
                        [
                            float(np.mean(selected)),
                            float(np.mean(deviations[selected])),
                            float(np.std(deviations[selected])),
                            float(np.mean(scalar_array[selected, 1])),
                            float(np.mean(scalar_array[selected, 5])),
                            float(np.mean(scalar_array[selected, 6])),
                        ]
                    )
                else:
                    typed.extend([0.0] * 6)
            if len(patch_array) > 1:
                pair_distance = np.sqrt(
                    np.mean(np.diff(patch_array, axis=0) ** 2, axis=1)
                )
                alternation = [
                    float(np.mean(pair_distance)),
                    float(np.std(pair_distance)),
                    float(np.median(pair_distance)),
                    float(np.mean(pair_distance[::2])),
                    float(np.mean(pair_distance[1::2]))
                    if len(pair_distance) > 1
                    else 0.0,
                ]
            else:
                alternation = [0.0] * 5
        else:
            aggregate = np.zeros(111, dtype=np.float64)
            typed = [0.0] * (6 * len(beat_labels))
            alternation = [0.0] * 5
        positions = np.asarray([position for position, _ in events], dtype=np.float64)
        intervals = np.diff(positions)
        if len(intervals):
            median_interval = max(float(np.median(intervals)), 1.0)
            normalized_intervals = intervals / median_interval
            interval_difference = np.diff(normalized_intervals)
            timing = [
                float(len(positions)),
                float(np.mean(intervals)),
                float(np.std(intervals)),
                float(np.std(intervals) / max(np.mean(intervals), 1.0)),
                float(np.min(intervals)),
                float(np.max(intervals)),
                *np.quantile(intervals, [0.10, 0.25, 0.50, 0.75, 0.90]),
                float(np.sqrt(np.mean(np.diff(intervals) ** 2)))
                if len(intervals) > 1
                else 0.0,
                float(np.mean(np.abs(interval_difference)))
                if len(interval_difference)
                else 0.0,
                float(np.std(interval_difference))
                if len(interval_difference)
                else 0.0,
                float(np.mean(intervals < 0.8 * median_interval)),
                float(np.mean(intervals > 1.2 * median_interval)),
                float(np.mean(normalized_intervals[::2])),
                float(np.mean(normalized_intervals[1::2]))
                if len(normalized_intervals) > 1
                else 0.0,
            ]
        else:
            timing = [float(len(positions))] + [0.0] * 17
        output.append(
            np.concatenate([aggregate, typed, alternation, timing]).astype(
                np.float32
            )
        )
    return np.nan_to_num(np.stack(output), copy=False)


def rhythm_feature_matrix(channels, event_rows, beat_labels, length):
    return np.concatenate(
        [
            waveform_feature_rows(channels),
            sequence_features(event_rows, beat_labels, length),
            rhythm_morphology_features(channels, event_rows, beat_labels),
        ],
        axis=1,
    ).astype(np.float32)


def fit_rhythm_feature_stack(
    training_signals,
    test_signals,
    gold_events,
    oof_events,
    test_events,
    rhythm_target,
    rhythm_labels,
    beat_labels,
    folds,
    length,
    started_at,
):
    from catboost import CatBoostClassifier

    gold_features = rhythm_feature_matrix(
        training_signals, gold_events, beat_labels, length
    )
    oof_features = rhythm_feature_matrix(
        training_signals, oof_events, beat_labels, length
    )
    test_features = rhythm_feature_matrix(
        test_signals, test_events, beat_labels, length
    )
    oof_probability = np.full(
        (len(training_signals), len(rhythm_labels)), np.nan, dtype=np.float32
    )
    common_parameters = {
        "iterations": 1000,
        "depth": 7,
        "learning_rate": 0.035,
        "loss_function": "MultiClass",
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 7.0,
        "random_strength": 0.7,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    trained_models = 0
    for fold_number, (train_indices, valid_indices) in enumerate(folds):
        if time.monotonic() - started_at >= TRAINING_CUTOFF_SECONDS:
            raise RuntimeError("wall-clock guard reached during rhythm feature stack")
        model = CatBoostClassifier(
            **common_parameters, random_seed=SEED + 40000 + fold_number
        )
        model.fit(gold_features[train_indices], rhythm_target[train_indices])
        probability = model.predict_proba(oof_features[valid_indices])
        classes = np.asarray(model.classes_, dtype=np.int64)
        oof_probability[
            valid_indices[:, None], classes[None, :]
        ] = probability.astype(np.float32)
        trained_models += 1
    if not np.isfinite(oof_probability).all():
        raise RuntimeError("rhythm feature stack produced incomplete OOF predictions")
    if time.monotonic() - started_at >= TRAINING_CUTOFF_SECONDS:
        raise RuntimeError("wall-clock guard reached before final rhythm feature model")
    final_model = CatBoostClassifier(
        **common_parameters, random_seed=SEED + 49000
    )
    final_model.fit(gold_features, rhythm_target)
    final_raw = final_model.predict_proba(test_features)
    final_probability = np.zeros(
        (len(test_signals), len(rhythm_labels)), dtype=np.float32
    )
    final_classes = np.asarray(final_model.classes_, dtype=np.int64)
    final_probability[:, final_classes] = final_raw.astype(np.float32)
    trained_models += 1
    return {
        "oof_rhythm": oof_probability,
        "test_rhythm": final_probability,
        "name": "event-feature CatBoost",
    }, trained_models


def write_placeholder(test_frame, output_path, rhythm_label, columns):
    placeholder = pd.DataFrame(
        {
            columns[0]: test_frame["id"].astype(str),
            columns[1]: rhythm_label,
            columns[2]: "[]",
        }
    )
    placeholder.to_csv(output_path, index=False)


def build_submission(
    test_frame,
    rhythm_indices,
    event_predictions,
    rhythm_labels,
    beat_labels,
    fallback_rhythm,
):
    rhythm_output = []
    beat_output = []
    for row_number in range(len(test_frame)):
        try:
            rhythm_index = int(rhythm_indices[row_number])
            rhythm_label = rhythm_labels[rhythm_index]
            named_events = [
                [int(position), beat_labels[int(type_index)]]
                for position, type_index in event_predictions[row_number]
            ]
            rhythm_output.append(rhythm_label)
            beat_output.append(json.dumps(named_events, separators=(",", ":")))
        except Exception as exc:
            log(f"warning: prediction fallback for test row {row_number}: {exc}")
            rhythm_output.append(fallback_rhythm)
            beat_output.append("[]")
    return pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "rhythm_family": rhythm_output,
            "beats": beat_output,
        }
    )


def audit_submission(submission, test_frame, rhythm_labels, beat_labels, length):
    expected_columns = ["id", "rhythm_family", "beats"]
    problems = []
    if submission.columns.tolist() != expected_columns:
        problems.append("incorrect columns")
    if len(submission) != len(test_frame):
        problems.append("incorrect row count")
    if submission["id"].astype(str).tolist() != test_frame["id"].astype(str).tolist():
        problems.append("ids are not in exact test order")
    if submission["id"].duplicated().any():
        problems.append("duplicate ids")
    for row_number, row in submission.iterrows():
        if row["rhythm_family"] not in rhythm_labels:
            problems.append(f"invalid rhythm at row {row_number}")
            continue
        try:
            events = json.loads(row["beats"])
            positions = []
            if not isinstance(events, list) or len(events) > MAX_EVENTS:
                raise ValueError("invalid event list")
            for event in events:
                if not isinstance(event, list) or len(event) != 2:
                    raise ValueError("invalid event")
                position, label = event
                if isinstance(position, bool) or not isinstance(position, int):
                    raise ValueError("non-integer position")
                if not 0 <= position < length or label not in beat_labels:
                    raise ValueError("event outside schema")
                positions.append(position)
            if positions != sorted(positions) or len(positions) != len(set(positions)):
                raise ValueError("events not strictly ordered")
        except Exception as exc:
            problems.append(f"malformed beats at row {row_number}: {exc}")
        if len(problems) >= 10:
            break
    return problems


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    started_at = time.monotonic()
    seed_everything(SEED)
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    manifest_path = public_dir / "task_manifest.json"
    for required_path in (train_path, test_path, manifest_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"missing required input: {required_path}")
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    rhythm_labels = list(manifest["rhythm_families"])
    beat_labels = list(manifest["beat_types"])
    rare_rhythm_labels = list(manifest["rare_rhythm_families"])
    rare_beat_labels = list(manifest["rare_beat_types"])
    submission_columns = list(manifest["submission_columns"])
    tolerance = int(manifest["event_tolerance_samples"])
    fallback_rhythm = (
        train_frame["rhythm_family"].value_counts().index[0]
        if len(train_frame)
        else rhythm_labels[0]
    )
    write_placeholder(
        test_frame, submission_out, fallback_rhythm, submission_columns
    )
    log(f"wrote early schema-valid placeholder for {len(test_frame)} test rows")
    required_train_columns = {"id", "signal", "rhythm_family", "beats"}
    required_test_columns = {"id", "signal"}
    if not required_train_columns.issubset(train_frame.columns):
        raise ValueError("train.csv is missing required columns")
    if not required_test_columns.issubset(test_frame.columns):
        raise ValueError("test.csv is missing required columns")
    rhythm_to_index = {label: index for index, label in enumerate(rhythm_labels)}
    unknown_rhythm = set(train_frame["rhythm_family"]) - set(rhythm_labels)
    if unknown_rhythm:
        raise ValueError(f"training data contains unknown rhythms: {sorted(unknown_rhythm)}")
    log("loading and transforming training waveforms")
    training_signals, _ = load_signal_frame(
        train_frame, public_dir, SIGNAL_LENGTH, tolerate_errors=False
    )
    log("loading test waveforms for inference-only transforms")
    test_signals, _ = load_signal_frame(
        test_frame, public_dir, SIGNAL_LENGTH, tolerate_errors=True
    )
    detector_target, type_target, rhythm_target, parsed_events = build_training_targets(
        train_frame,
        beat_labels,
        rhythm_to_index,
        tolerance,
        SIGNAL_LENGTH,
    )
    features = patient_proxy_features(training_signals, parsed_events)
    folds = grouped_stratified_folds(
        features,
        train_frame["rhythm_family"].to_numpy(),
        requested_folds=3,
        seed=SEED,
    )
    if not folds:
        log("warning: too few examples for validation folds; placeholder retained")
        return
    fold_sizes = [len(valid) for _, valid in folds]
    log(f"using morphology-neighbour grouped OOF folds: {fold_sizes}")
    device = choose_device()
    if device.type == "cpu":
        warnings.warn("accelerator unavailable; training on CPU")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    log(f"training on {device}")
    row_count = len(train_frame)
    variant_specs = (
        (0.5, False),
        (0.5, False),
        (0.5, True),
    )
    variant_results = []
    total_trained_models = 0
    for variant_number, (balance_power, augment) in enumerate(variant_specs):
        if time.monotonic() - started_at >= TRAINING_CUTOFF_SECONDS:
            log("wall-clock guard reached before next training variant")
            break
        oof_detector = np.full(
            (row_count, SIGNAL_LENGTH), np.nan, dtype=np.float32
        )
        oof_type = np.full(
            (row_count, len(beat_labels), SIGNAL_LENGTH), np.nan, dtype=np.float32
        )
        oof_rhythm = np.full(
            (row_count, len(rhythm_labels)), np.nan, dtype=np.float32
        )
        test_detector_sum = np.zeros(
            (len(test_frame), SIGNAL_LENGTH), dtype=np.float32
        )
        test_type_sum = np.zeros(
            (len(test_frame), len(beat_labels), SIGNAL_LENGTH), dtype=np.float32
        )
        test_rhythm_sum = np.zeros(
            (len(test_frame), len(rhythm_labels)), dtype=np.float32
        )
        trained_models = 0
        for fold_number, (train_indices, valid_indices) in enumerate(folds):
            if time.monotonic() - started_at >= TRAINING_CUTOFF_SECONDS:
                log("wall-clock guard reached before next fold; moving to inference")
                break
            try:
                mode = "augmented" if augment else "plain"
                log(
                    f"starting {mode} variant {variant_number + 1}/"
                    f"{len(variant_specs)} (power={balance_power:.2f}), "
                    f"fold {fold_number + 1}/{len(folds)} "
                    f"({len(train_indices)} train, {len(valid_indices)} validation)"
                )
                model = fit_fold(
                    fold_number,
                    train_indices,
                    valid_indices,
                    training_signals,
                    detector_target,
                    type_target,
                    rhythm_target,
                    device,
                    balance_power,
                    variant_number,
                    augment,
                    started_at,
                )
                valid_probabilities = predict_probabilities(
                    model, training_signals[valid_indices], device
                )
                oof_detector[valid_indices] = valid_probabilities[0]
                oof_type[valid_indices] = valid_probabilities[1]
                oof_rhythm[valid_indices] = valid_probabilities[2]
                test_probabilities = predict_probabilities(model, test_signals, device)
                test_detector_sum += test_probabilities[0]
                test_type_sum += test_probabilities[1]
                test_rhythm_sum += test_probabilities[2]
                trained_models += 1
                total_trained_models += 1
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                log(
                    f"warning: variant {variant_number + 1} fold "
                    f"{fold_number + 1} failed and was skipped: {exc}"
                )
        if trained_models == 0:
            continue
        if variant_number > 0 and trained_models != len(folds):
            log("discarding incomplete optional training variant")
            break
        variant_results.append(
            {
                "name": f"{'augmented' if augment else 'plain'} "
                f"seed {variant_number + 1}",
                "oof_detector": oof_detector,
                "oof_type": oof_type,
                "oof_rhythm": oof_rhythm,
                "test_detector": test_detector_sum / float(trained_models),
                "test_type": test_type_sum / float(trained_models),
                "test_rhythm": test_rhythm_sum / float(trained_models),
            }
        )
    if not variant_results:
        log("no trained model completed; early placeholder retained")
        return
    try:
        valid_mask = np.ones(row_count, dtype=bool)
        for result in variant_results:
            valid_mask &= np.isfinite(result["oof_detector"][:, 0])
        valid_rows = np.flatnonzero(valid_mask)
        if len(valid_rows) == 0:
            log("no OOF rows available for calibration; early placeholder retained")
            return
        log(
            f"searching model blends and decoders on {len(valid_rows)} "
            "train-only OOF predictions"
        )
        neural_blend_grid = simplex_blend_grid(len(variant_results), units=5)
        oof_truth = [parsed_events[index] for index in valid_rows]
        best_beat_score, best_beat = select_beat_decoder(
            variant_results,
            valid_rows,
            oof_truth,
            neural_blend_grid,
            beat_labels,
            rare_beat_labels,
            tolerance,
            log_results=True,
        )
        beat_blend, beat_thresholds, minimum_distance, type_bias, beat_parts = (
            best_beat
        )
        test_detector = sum(
            result["test_detector"] * float(weight)
            for result, weight in zip(variant_results, beat_blend)
        )
        test_type = sum(
            result["test_type"] * float(weight)
            for result, weight in zip(variant_results, beat_blend)
        )
        test_event_predictions = decode_candidate_rows(
            prepare_peak_candidates(test_detector, test_type),
            beat_thresholds,
            minimum_distance,
            type_bias,
        )
        test_named_events = [
            [
                (position, beat_labels[type_index])
                for position, type_index in events
            ]
            for events in test_event_predictions
        ]
        rhythm_components = [
            {
                "name": result["name"],
                "oof_rhythm": result["oof_rhythm"],
                "test_rhythm": result["test_rhythm"],
            }
            for result in variant_results
        ]
        if len(valid_rows) == row_count:
            try:
                log("building cross-fitted beat-sequence rhythm features")
                cross_fitted_events = cross_fitted_named_events(
                    variant_results,
                    folds,
                    neural_blend_grid,
                    parsed_events,
                    beat_labels,
                    rare_beat_labels,
                    tolerance,
                )
                feature_result, feature_model_count = fit_rhythm_feature_stack(
                    training_signals,
                    test_signals,
                    parsed_events,
                    cross_fitted_events,
                    test_named_events,
                    rhythm_target,
                    rhythm_labels,
                    beat_labels,
                    folds,
                    SIGNAL_LENGTH,
                    started_at,
                )
                rhythm_components.append(feature_result)
                total_trained_models += feature_model_count
                log("added cross-fitted event-feature rhythm model")
            except Exception as exc:
                log(f"warning: event-feature rhythm model skipped: {exc}")
        rhythm_blend_grid = simplex_blend_grid(len(rhythm_components), units=5)
        best_rhythm_score = -1.0
        best_rhythm = None
        for blend_weights in rhythm_blend_grid:
            rhythm_probability = sum(
                result["oof_rhythm"][valid_rows] * float(weight)
                for result, weight in zip(rhythm_components, blend_weights)
            )
            rhythm_bias = calibrate_rhythm(
                rhythm_probability,
                rhythm_target[valid_rows],
                rhythm_labels,
                rare_rhythm_labels,
            )
            score, parts = rhythm_metric(
                rhythm_probability,
                rhythm_target[valid_rows],
                rhythm_labels,
                rare_rhythm_labels,
                rhythm_bias,
            )
            log(
                f"OOF rhythm blend weights={blend_weights.tolist()}: "
                f"weighted={score:.6f}, per-class={parts['rhythm_F1']}"
            )
            if score > best_rhythm_score:
                best_rhythm_score = score
                best_rhythm = (blend_weights.copy(), rhythm_bias, parts)
        rhythm_blend, rhythm_bias, rhythm_parts = best_rhythm
        test_rhythm = sum(
            result["test_rhythm"] * float(weight)
            for result, weight in zip(rhythm_components, rhythm_blend)
        )
        log(
            "OOF score="
            f"{best_beat_score + best_rhythm_score:.6f}; "
            f"event_micro={beat_parts['event_micro_F1']:.6f}, "
            f"beat_macro={beat_parts['beat_type_macro_F1']:.6f}, "
            f"rhythm_macro={rhythm_parts['rhythm_macro_F1']:.6f}, "
            f"rare_beat={beat_parts['rare_beat_macro_F1']:.6f}, "
            f"rare_rhythm={rhythm_parts['rare_rhythm_macro_F1']:.6f}"
        )
        log(
            f"searched decoder: beat_blend={beat_blend.tolist()}, "
            f"rhythm_components="
            f"{[component['name'] for component in rhythm_components]}, "
            f"rhythm_blend={rhythm_blend.tolist()}, "
            f"thresholds={beat_thresholds.tolist()}, "
            f"minimum_distance={minimum_distance}, type_bias={type_bias.tolist()}, "
            f"rhythm_bias={rhythm_bias.tolist()}"
        )
        test_rhythm_indices = np.argmax(
            np.log(np.maximum(test_rhythm, 1e-8)) + rhythm_bias[None, :], axis=1
        )
        submission = build_submission(
            test_frame,
            test_rhythm_indices,
            test_event_predictions,
            rhythm_labels,
            beat_labels,
            fallback_rhythm,
        )
        submission = submission[submission_columns]
        problems = audit_submission(
            submission, test_frame, rhythm_labels, beat_labels, SIGNAL_LENGTH
        )
        if problems:
            log(f"warning: final audit found {problems}; retaining valid placeholder")
            return
        submission.to_csv(submission_out, index=False)
        log(
            f"wrote complete submission to {submission_out} "
            f"({len(submission)} rows, {total_trained_models} models)"
        )
    except Exception as exc:
        log(f"warning: post-training inference failed; valid placeholder retained: {exc}")


if __name__ == "__main__":
    main()
