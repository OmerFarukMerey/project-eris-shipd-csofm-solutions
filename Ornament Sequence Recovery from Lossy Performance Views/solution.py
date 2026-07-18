#!/usr/bin/env python3
"""Train a compliant lossy-view sequence ensemble and write the submission."""

import copy
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


SEED = 2026
MAX_LENGTH = 32
# Stop starting new optional training once this many wall-clock seconds have
# elapsed, leaving headroom for final retraining and inference inside the
# platform's 1.5 hour ceiling (Solver Guidebook 3.5). The required two seeds of
# every neural family always run; extra seeds are trained only when time allows.
OPTIONAL_TRAIN_DEADLINE_SECONDS = 3300
REQUIRED_NEURAL_SEEDS = 2
COMPONENT_CLASSES = (
    np.arange(-24, 25, dtype=np.int16),
    np.arange(0, 6, dtype=np.int16),
    np.arange(0, 6, dtype=np.int16),
    np.arange(1, 5, dtype=np.int16),
)
CLASS_OFFSETS = (-24, 0, 0, 1)
# Feature slices of the current event's own view tokens inside the enhanced tree
# feature vector (five-event acoustic window = 95, current symbolic block at
# offset 4 of the nine-event window). Zeroing a component group's own current
# tokens lets an all-label tree learn from every labeled event without ever
# seeing the value it must predict.
CURRENT_PITCH_SLICE = slice(127, 131)
CURRENT_TIMING_SLICE = slice(131, 135)


def parse_views(pitch_text, timing_text):
    pitch = []
    for token in pitch_text.split():
        p = None if token[1] == "?" else int(token[1:4])
        m = None if token[-1] == "?" else int(token[-1])
        pitch.append((p, m))

    timing = []
    for token in timing_text.split():
        g = None if token[1] == "?" else int(token[1])
        d = None if token[-1] == "?" else int(token[-1])
        timing.append((g, d))

    if len(pitch) != len(timing):
        raise ValueError("pitch_view and timing_view lengths differ")
    return pitch, timing


def parse_target(text):
    events = []
    for token in text.split():
        events.append((int(token[1:4]), int(token[6]), int(token[9]), int(token[12])))
    return events


def infer_chroma_shift(chroma, pitch_view):
    """Align one excerpt using only pitches visible in that same excerpt."""
    visible = [(i, p) for i, (p, _) in enumerate(pitch_view) if p is not None]
    scores = np.zeros(12, dtype=np.float32)
    for shift in range(12):
        scores[shift] = sum(chroma[i, (p + shift) % 12] for i, p in visible)
    shift = int(np.argmax(scores))
    second = np.partition(scores, -2)[-2]
    confidence = float((scores[shift] - second) / (abs(scores[shift]) + 1e-6))
    return shift, confidence


def make_row_inputs(acoustic_row, pitch_view, timing_view):
    """Build tree features and compact neural inputs for one independent row."""
    n = len(pitch_view)
    chroma = acoustic_row[:n, :12].astype(np.float32, copy=False)
    shift, shift_confidence = infer_chroma_shift(chroma, pitch_view)
    aligned = chroma[:, (np.arange(12) + shift) % 12]
    acoustic = np.concatenate(
        (aligned, acoustic_row[:n, 12:18].astype(np.float32, copy=False)), axis=1
    )

    component_values = (
        [p for p, _ in pitch_view],
        [g for g, _ in timing_view],
        [d for _, d in timing_view],
        [m for _, m in pitch_view],
    )
    component_scales = (24.0, 5.0, 5.0, 4.0)
    row_mean = acoustic.mean(axis=0)
    row_std = acoustic.std(axis=0)
    zero_acoustic = np.zeros(18, dtype=np.float32)
    feature_rows = []

    for position in range(n):
        features = []

        # Local acoustic evidence and explicit sequence boundaries.
        for offset in range(-2, 3):
            neighbor = position + offset
            if 0 <= neighbor < n:
                features.extend(acoustic[neighbor])
                features.append(1.0)
            else:
                features.extend(zero_acoustic)
                features.append(0.0)

        # Visible symbolic evidence in a local sequence window.
        for offset in range(-4, 5):
            neighbor = position + offset
            if 0 <= neighbor < n:
                p, m = pitch_view[neighbor]
                g, d = timing_view[neighbor]
                features.extend(
                    (
                        0.0 if p is None else p / 24.0,
                        float(p is not None),
                        0.0 if m is None else (m - 1) / 3.0,
                        float(m is not None),
                        0.0 if g is None else g / 5.0,
                        float(g is not None),
                        0.0 if d is None else d / 5.0,
                        float(d is not None),
                    )
                )
            else:
                features.extend((0.0,) * 8)

        features.extend(
            (
                position / 31.0,
                (n - 1 - position) / 31.0,
                n / 32.0,
                shift_confidence,
            )
        )

        # Differences expose onset/duration changes directly to the tree models.
        previous = acoustic[position - 1] if position > 0 else acoustic[position]
        following = acoustic[position + 1] if position + 1 < n else acoustic[position]
        features.extend(acoustic[position] - previous)
        features.extend(following - acoustic[position])
        features.extend(row_mean)
        features.extend(row_std)

        # Two visible anchors on each side provide full-excerpt context without
        # filling or estimating any withheld value.
        for values, scale in zip(component_values, component_scales):
            left = [k for k in range(position - 1, -1, -1) if values[k] is not None][:2]
            right = [k for k in range(position + 1, n) if values[k] is not None][:2]
            for side in (left, right):
                for rank in range(2):
                    if rank < len(side):
                        neighbor = side[rank]
                        features.extend(
                            (values[neighbor] / scale, abs(neighbor - position) / 31.0, 1.0)
                        )
                    else:
                        features.extend((0.0, 1.0, 0.0))
            if left and right:
                left_position, right_position = left[0], right[0]
                interpolated = (
                    values[left_position] * (right_position - position)
                    + values[right_position] * (position - left_position)
                ) / (right_position - left_position)
                features.extend((interpolated / scale, 1.0))
            else:
                features.extend((0.0, 0.0))

        feature_rows.append(features)

    categories = np.asarray(
        [
            (
                49 if p is None else p + 24,
                6 if g is None else g,
                6 if d is None else d,
                4 if m is None else m - 1,
            )
            for (p, m), (g, d) in zip(pitch_view, timing_view)
        ],
        dtype=np.int64,
    )
    return np.asarray(feature_rows, dtype=np.float32), acoustic, categories


class SequenceDataset(Dataset):
    def __init__(self, acoustic, categories, missing, targets, lengths, row_ids, mean, std):
        self.acoustic = acoustic
        self.categories = categories
        self.missing = missing
        self.targets = targets
        self.lengths = lengths
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, index):
        row = int(self.row_ids[index])
        normalized = (self.acoustic[row] - self.mean) / self.std
        return (
            torch.from_numpy(normalized.astype(np.float32, copy=False)),
            torch.from_numpy(self.categories[row]),
            torch.from_numpy(self.missing[row]),
            torch.from_numpy(self.targets[row]),
            torch.tensor(self.lengths[row], dtype=torch.int64),
        )


class BidirectionalSequenceModel(nn.Module):
    """A from-scratch BiGRU that learns dependencies across the full excerpt."""

    def __init__(self, hidden_size=160, dropout=0.18, rnn_type="gru"):
        super().__init__()
        rnn = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.acoustic_projection = nn.Sequential(
            nn.Linear(18, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.component_embeddings = nn.ModuleList(
            (nn.Embedding(50, 20), nn.Embedding(7, 8), nn.Embedding(7, 8), nn.Embedding(5, 6))
        )
        self.position_embedding = nn.Embedding(MAX_LENGTH, 16)
        self.input_projection = nn.Sequential(
            nn.Linear(122, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.encoder = rnn(
            hidden_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(2 * hidden_size),
        )
        self.heads = nn.ModuleList(
            nn.Linear(2 * hidden_size, size) for size in (49, 6, 6, 4)
        )

    def forward(self, acoustic, categories, lengths):
        batch_size = acoustic.shape[0]
        positions = torch.arange(MAX_LENGTH, device=acoustic.device)[None].expand(
            batch_size, -1
        )
        parts = [self.acoustic_projection(acoustic)]
        parts.extend(
            embedding(categories[:, :, component])
            for component, embedding in enumerate(self.component_embeddings)
        )
        parts.append(self.position_embedding(positions))
        encoded = self.input_projection(torch.cat(parts, dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=MAX_LENGTH
        )
        encoded = self.output_projection(encoded)
        return [head(encoded) for head in self.heads]


class LeaveOneOutSequenceModel(nn.Module):
    """Learn every label without exposing its current same-view token."""

    def __init__(self, dropout=0.16, rnn_type="gru"):
        super().__init__()
        rnn = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.acoustic_projection = nn.Sequential(
            nn.Linear(18, 80), nn.GELU(), nn.LayerNorm(80)
        )
        self.acoustic_encoder = rnn(
            80,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.component_embeddings = nn.ModuleList(
            (nn.Embedding(50, 20), nn.Embedding(7, 8), nn.Embedding(7, 8), nn.Embedding(5, 6))
        )
        self.position_embedding = nn.Embedding(MAX_LENGTH, 16)
        self.pitch_forward = rnn(26, 64, batch_first=True)
        self.pitch_backward = rnn(26, 64, batch_first=True)
        self.timing_forward = rnn(16, 48, batch_first=True)
        self.timing_backward = rnn(16, 48, batch_first=True)
        self.pitch_projection = nn.Sequential(
            nn.LayerNorm(416),
            nn.Linear(416, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
        )
        self.timing_projection = nn.Sequential(
            nn.LayerNorm(394),
            nn.Linear(394, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
        )
        self.heads = nn.ModuleList(
            (nn.Linear(256, 49), nn.Linear(256, 6), nn.Linear(256, 6), nn.Linear(256, 4))
        )

    @staticmethod
    def packed_encode(encoder, values, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=MAX_LENGTH
        )
        return encoded

    @staticmethod
    def reverse_valid(values, lengths):
        reversed_values = torch.zeros_like(values)
        for row, length in enumerate(lengths.cpu().tolist()):
            reversed_values[row, :length] = values[row, :length].flip(0)
        return reversed_values

    def leave_one_out_context(self, values, lengths, forward_encoder, backward_encoder):
        forward = self.packed_encode(forward_encoder, values, lengths)
        reversed_values = self.reverse_valid(values, lengths)
        backward = self.packed_encode(backward_encoder, reversed_values, lengths)
        backward = self.reverse_valid(backward, lengths)

        left_context = torch.zeros_like(forward)
        left_context[:, 1:] = forward[:, :-1]
        right_context = torch.zeros_like(backward)
        right_context[:, :-1] = backward[:, 1:]
        return torch.cat((left_context, right_context), dim=-1)

    def forward(self, acoustic, categories, lengths):
        batch_size = acoustic.shape[0]
        acoustic_context = self.packed_encode(
            self.acoustic_encoder, self.acoustic_projection(acoustic), lengths
        )
        embedded = [
            embedding(categories[:, :, component])
            for component, embedding in enumerate(self.component_embeddings)
        ]
        positions = self.position_embedding(
            torch.arange(MAX_LENGTH, device=acoustic.device)[None].expand(batch_size, -1)
        )

        pitch_context = self.leave_one_out_context(
            torch.cat((embedded[0], embedded[3]), dim=-1),
            lengths,
            self.pitch_forward,
            self.pitch_backward,
        )
        timing_context = self.leave_one_out_context(
            torch.cat((embedded[1], embedded[2]), dim=-1),
            lengths,
            self.timing_forward,
            self.timing_backward,
        )

        # Pitch/multiplicity heads see neighboring pitch-view tokens and the
        # current timing view, never the current pitch-view label. Timing heads
        # use the exact converse. Thus all labeled positions are safe to train.
        pitch_hidden = self.pitch_projection(
            torch.cat(
                (
                    acoustic_context,
                    pitch_context,
                    embedded[1],
                    embedded[2],
                    positions,
                ),
                dim=-1,
            )
        )
        timing_hidden = self.timing_projection(
            torch.cat(
                (
                    acoustic_context,
                    timing_context,
                    embedded[0],
                    embedded[3],
                    positions,
                ),
                dim=-1,
            )
        )
        return [
            self.heads[0](pitch_hidden),
            self.heads[1](timing_hidden),
            self.heads[2](timing_hidden),
            self.heads[3](pitch_hidden),
        ]


def set_torch_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def acoustic_statistics(acoustic, lengths, row_ids):
    valid = np.concatenate([acoustic[i, : lengths[i]] for i in row_ids], axis=0)
    return valid.mean(axis=0), valid.std(axis=0) + 1e-4


def sequence_loss(logits, targets, missing):
    weights = (1.35, 1.0, 1.0, 0.8)
    losses = []
    for component, component_logits in enumerate(logits):
        selected = missing[:, :, component]
        losses.append(
            F.cross_entropy(
                component_logits[selected], targets[:, :, component][selected]
            )
        )
    return sum(weight * loss for weight, loss in zip(weights, losses))


def train_sequence_with_validation(
    acoustic,
    categories,
    missing,
    targets,
    lengths,
    train_rows,
    validation_rows,
    mean,
    std,
    seed,
    rnn_type="gru",
):
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BidirectionalSequenceModel(rnn_type=rnn_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        SequenceDataset(
            acoustic, categories, missing, targets, lengths, train_rows, mean, std
        ),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        SequenceDataset(
            acoustic, categories, missing, targets, lengths, validation_rows, mean, std
        ),
        batch_size=128,
        shuffle=False,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(40):
        model.train()
        for batch in train_loader:
            batch_acoustic, batch_categories, batch_missing, batch_targets, batch_lengths = (
                value.to(device) for value in batch
            )
            loss = sequence_loss(
                model(batch_acoustic, batch_categories, batch_lengths),
                batch_targets,
                batch_missing,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch_acoustic, batch_categories, batch_missing, batch_targets, batch_lengths = (
                    value.to(device) for value in batch
                )
                validation_loss += sequence_loss(
                    model(batch_acoustic, batch_categories, batch_lengths),
                    batch_targets,
                    batch_missing,
                ).item()
                batches += 1
        validation_loss /= batches
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 9:
            break

    model.load_state_dict(best_state)
    return model.cpu(), best_epoch + 1


def train_sequence_full(
    acoustic, categories, missing, targets, lengths, mean, std, seed, epochs, rnn_type="gru"
):
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BidirectionalSequenceModel(rnn_type=rnn_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        SequenceDataset(
            acoustic,
            categories,
            missing,
            targets,
            lengths,
            np.arange(len(lengths)),
            mean,
            std,
        ),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch_acoustic, batch_categories, batch_missing, batch_targets, batch_lengths = (
                value.to(device) for value in batch
            )
            loss = sequence_loss(
                model(batch_acoustic, batch_categories, batch_lengths),
                batch_targets,
                batch_missing,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.cpu()


def all_label_loss(logits, targets, lengths):
    valid = torch.arange(MAX_LENGTH, device=lengths.device)[None] < lengths[:, None]
    weights = (1.35, 1.0, 1.0, 0.8)
    losses = [
        F.cross_entropy(component_logits[valid], targets[:, :, component][valid])
        for component, component_logits in enumerate(logits)
    ]
    return sum(weight * loss for weight, loss in zip(weights, losses))


def train_leave_one_out_with_validation(
    acoustic,
    categories,
    missing,
    targets,
    lengths,
    train_rows,
    validation_rows,
    mean,
    std,
    seed,
    rnn_type="gru",
):
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeaveOneOutSequenceModel(rnn_type=rnn_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    train_loader = DataLoader(
        SequenceDataset(
            acoustic, categories, missing, targets, lengths, train_rows, mean, std
        ),
        batch_size=48,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        SequenceDataset(
            acoustic, categories, missing, targets, lengths, validation_rows, mean, std
        ),
        batch_size=96,
        shuffle=False,
    )

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(40):
        model.train()
        for batch in train_loader:
            batch_acoustic, batch_categories, _, batch_targets, batch_lengths = (
                value.to(device) for value in batch
            )
            loss = all_label_loss(
                model(batch_acoustic, batch_categories, batch_lengths),
                batch_targets,
                batch_lengths,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch_acoustic, batch_categories, _, batch_targets, batch_lengths = (
                    value.to(device) for value in batch
                )
                validation_loss += all_label_loss(
                    model(batch_acoustic, batch_categories, batch_lengths),
                    batch_targets,
                    batch_lengths,
                ).item()
                batches += 1
        validation_loss /= batches
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 8:
            break

    model.load_state_dict(best_state)
    return model.cpu(), best_epoch + 1


def train_leave_one_out_full(
    acoustic, categories, missing, targets, lengths, mean, std, seed, epochs, rnn_type="gru"
):
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeaveOneOutSequenceModel(rnn_type=rnn_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    loader = DataLoader(
        SequenceDataset(
            acoustic,
            categories,
            missing,
            targets,
            lengths,
            np.arange(len(lengths)),
            mean,
            std,
        ),
        batch_size=48,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch_acoustic, batch_categories, _, batch_targets, batch_lengths = (
                value.to(device) for value in batch
            )
            loss = all_label_loss(
                model(batch_acoustic, batch_categories, batch_lengths),
                batch_targets,
                batch_lengths,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.cpu()


def decide_classes(probabilities, component, similarity_weight):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite model probability")
    values = COMPONENT_CLASSES[component].astype(np.float64)
    if component == 0:
        kernel = np.exp(-np.abs(values[:, None] - values[None, :]) / 2.0)
    elif component in (1, 2):
        kernel = np.exp(-np.abs(values[:, None] - values[None, :]) / 1.25)
    else:
        kernel = np.eye(len(values), dtype=np.float64)
    # Avoid platform BLAS issues observed for mixed-layout probability arrays.
    expected_similarity = np.einsum(
        "ni,ij->nj", probabilities, kernel, optimize=False
    )
    utility = (
        (1.0 - similarity_weight) * probabilities
        + similarity_weight * expected_similarity
    )
    return COMPONENT_CLASSES[component][np.argmax(utility, axis=1)]


def event_similarity(first, second):
    return (
        0.50 * np.exp(-abs(first[0] - second[0]) / 2.0)
        + 0.20 * np.exp(-abs(first[1] - second[1]) / 1.25)
        + 0.20 * np.exp(-abs(first[2] - second[2]) / 1.25)
        + 0.10 * (first[3] == second[3])
    )


def sequence_score(predicted, target):
    n, m = len(predicted), len(target)
    previous = np.arange(m + 1)
    for i, predicted_event in enumerate(predicted, 1):
        current = np.empty(m + 1, dtype=np.int16)
        current[0] = i
        for j, target_event in enumerate(target, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (predicted_event != target_event),
            )
        previous = current
    exact = max(0.0, 1.0 - previous[m] / max(n, m, 1))

    previous_similarity = np.zeros(m + 1, dtype=np.float32)
    for predicted_event in predicted:
        current_similarity = np.zeros(m + 1, dtype=np.float32)
        for j, target_event in enumerate(target, 1):
            current_similarity[j] = max(
                previous_similarity[j],
                current_similarity[j - 1],
                previous_similarity[j - 1]
                + event_similarity(predicted_event, target_event),
            )
        previous_similarity = current_similarity
    aligned = 2.0 * previous_similarity[m] / (n + m) if n and m else 0.0

    predicted_bigrams = Counter(zip(predicted, predicted[1:]))
    target_bigrams = Counter(zip(target, target[1:]))
    overlap = sum((predicted_bigrams & target_bigrams).values())
    precision = overlap / max(n - 1, 1)
    recall = overlap / max(m - 1, 1)
    bigram = (
        2.0 * precision * recall / (precision + recall)
        if precision and recall
        else 0.0
    )
    return 100.0 * (0.40 * exact + 0.40 * aligned + 0.20 * bigram)


def validation_metric(probabilities, utility_weights, targets, missing, lengths):
    predictions = np.column_stack(
        [
            decide_classes(probabilities[component], component, utility_weights[component])
            for component in range(4)
        ]
    )
    predictions[~missing] = targets[~missing]
    scores = []
    start = 0
    for length in lengths:
        end = start + length
        scores.append(
            sequence_score(
                [tuple(event) for event in predictions[start:end]],
                [tuple(event) for event in targets[start:end]],
            )
        )
        start = end
    return float(np.mean(scores))


def predict_sequence_probabilities(model, acoustic, categories, lengths, row_ids, mean, std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    probabilities = [[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0, len(row_ids), 128):
            selected = np.asarray(row_ids[start : start + 128], dtype=np.int64)
            batch_acoustic = torch.from_numpy(
                ((acoustic[selected] - mean) / std).astype(np.float32)
            ).to(device)
            batch_categories = torch.from_numpy(categories[selected]).to(device)
            batch_lengths = torch.from_numpy(lengths[selected]).to(device)
            logits = model(batch_acoustic, batch_categories, batch_lengths)
            for row_in_batch, length in enumerate(batch_lengths.cpu().numpy()):
                for component in range(4):
                    probabilities[component].append(
                        F.softmax(logits[component][row_in_batch, :length], dim=-1)
                        .cpu()
                        .numpy()
                    )
    return [np.concatenate(component_rows, axis=0) for component_rows in probabilities]


def full_probabilities(model, features, component):
    source = model.predict_proba(features)
    full = np.zeros((len(features), len(COMPONENT_CLASSES[component])), dtype=np.float32)
    class_to_column = {int(value): j for j, value in enumerate(COMPONENT_CLASSES[component])}
    for source_column, value in enumerate(model.classes_):
        full[:, class_to_column[int(value)]] = source[:, source_column]
    return full


def fit_tree_pair(features, targets, missing, event_rows, allowed_rows, component, final):
    selected = missing[:, component] & np.isin(event_rows, allowed_rows)
    extra_model = ExtraTreesClassifier(
        n_estimators=450 if final else 350,
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED + component,
    )
    extra_model.fit(features[selected], targets[selected, component])
    boosted_model = CatBoostClassifier(
        iterations=500 if final else 400,
        depth=8,
        learning_rate=0.07 if final else 0.08,
        loss_function="MultiClass",
        l2_leaf_reg=6.0,
        random_seed=SEED + component,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )
    boosted_model.fit(features[selected], targets[selected, component])
    return extra_model, boosted_model


def target_excluded_features(features, component):
    """Zero the current event's own view tokens for a component group."""
    excluded = features.copy()
    if component in (0, 3):
        excluded[:, CURRENT_PITCH_SLICE] = 0.0
    else:
        excluded[:, CURRENT_TIMING_SLICE] = 0.0
    return excluded


def fit_all_label_tree_pair(
    features, targets, event_rows, allowed_rows, component, final
):
    """Train on every labeled event, with the current view token excluded."""
    selected = np.isin(event_rows, allowed_rows)
    excluded = target_excluded_features(features[selected], component)
    extra_model = ExtraTreesClassifier(
        n_estimators=500 if final else 400,
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED + 40 + component,
    )
    extra_model.fit(excluded, targets[selected, component])
    boosted_model = CatBoostClassifier(
        iterations=500 if final else 400,
        depth=8,
        learning_rate=0.08,
        loss_function="MultiClass",
        l2_leaf_reg=6.0,
        random_seed=SEED + 40 + component,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )
    boosted_model.fit(excluded, targets[selected, component])
    return extra_model, boosted_model


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    start_time = time.time()

    def optional_training_allowed(trained_count):
        # The first REQUIRED_NEURAL_SEEDS seeds always run; further seeds start
        # only while wall-clock headroom remains for retraining and inference.
        if trained_count < REQUIRED_NEURAL_SEEDS:
            return True
        return time.time() - start_time < OPTIONAL_TRAIN_DEADLINE_SECONDS

    train = pd.read_csv(public_dir / "train.csv")
    train_acoustic_file = np.load(public_dir / "train_features.npz")["acoustic"]
    if len(train) != len(train_acoustic_file):
        raise ValueError("training CSV and acoustic rows are not aligned")

    tree_rows = []
    targets_by_row = []
    missing_by_row = []
    acoustic_padded = np.zeros((len(train), MAX_LENGTH, 18), dtype=np.float32)
    categories_padded = np.zeros((len(train), MAX_LENGTH, 4), dtype=np.int64)
    missing_padded = np.zeros((len(train), MAX_LENGTH, 4), dtype=bool)
    targets_padded = np.zeros((len(train), MAX_LENGTH, 4), dtype=np.int64)
    lengths = np.zeros(len(train), dtype=np.int64)

    for row_number, row in train.iterrows():
        pitch_view, timing_view = parse_views(row["pitch_view"], row["timing_view"])
        target_events = parse_target(row["target_sequence"])
        if len(target_events) != len(pitch_view):
            raise ValueError("training view and target lengths differ")
        tree_features, neural_acoustic, categories = make_row_inputs(
            train_acoustic_file[row_number], pitch_view, timing_view
        )
        n = len(target_events)
        target_array = np.asarray(target_events, dtype=np.int16)
        target_indices = target_array.copy()
        target_indices[:, 0] += 24
        target_indices[:, 3] -= 1
        row_missing = np.asarray(
            [
                (p is None, g is None, d is None, m is None)
                for (p, m), (g, d) in zip(pitch_view, timing_view)
            ],
            dtype=bool,
        )
        tree_rows.append(tree_features)
        targets_by_row.append(target_array)
        missing_by_row.append(row_missing)
        acoustic_padded[row_number, :n] = neural_acoustic
        categories_padded[row_number, :n] = categories
        missing_padded[row_number, :n] = row_missing
        targets_padded[row_number, :n] = target_indices
        lengths[row_number] = n

    tree_features = np.concatenate(tree_rows, axis=0)
    targets = np.concatenate(targets_by_row, axis=0)
    missing = np.concatenate(missing_by_row, axis=0)
    event_rows = np.repeat(np.arange(len(train)), lengths)

    # All architecture/blend selection is performed on labeled training data only.
    development_rows, validation_rows = train_test_split(
        np.arange(len(train)),
        test_size=0.20,
        random_state=SEED,
        stratify=train["capture_profile"],
    )
    development_mean, development_std = acoustic_statistics(
        acoustic_padded, lengths, development_rows
    )

    validation_tree_features = np.concatenate(
        [tree_rows[row] for row in validation_rows], axis=0
    )
    validation_targets = np.concatenate(
        [targets_by_row[row] for row in validation_rows], axis=0
    )
    validation_missing = np.concatenate(
        [missing_by_row[row] for row in validation_rows], axis=0
    )

    validation_extra_probabilities = []
    validation_boosted_probabilities = []
    for component in range(4):
        extra_model, boosted_model = fit_tree_pair(
            tree_features,
            targets,
            missing,
            event_rows,
            development_rows,
            component,
            final=False,
        )
        validation_extra_probabilities.append(
            full_probabilities(extra_model, validation_tree_features, component)
        )
        validation_boosted_probabilities.append(
            full_probabilities(boosted_model, validation_tree_features, component)
        )

    neural_specs = (
        (SEED, "gru"),
        (SEED + 5, "gru"),
        (SEED + 10, "gru"),
    )
    trained_neural_specs = []
    validation_neural_models = []
    selected_epochs = []
    for neural_seed, rnn_type in neural_specs:
        if not optional_training_allowed(len(validation_neural_models)):
            break
        model, epochs = train_sequence_with_validation(
            acoustic_padded,
            categories_padded,
            missing_padded,
            targets_padded,
            lengths,
            development_rows,
            validation_rows,
            development_mean,
            development_std,
            neural_seed,
            rnn_type,
        )
        trained_neural_specs.append((neural_seed, rnn_type))
        validation_neural_models.append(model)
        selected_epochs.append(epochs)
    validation_neural_runs = [
        predict_sequence_probabilities(
            model,
            acoustic_padded,
            categories_padded,
            lengths,
            validation_rows,
            development_mean,
            development_std,
        )
        for model in validation_neural_models
    ]
    validation_neural_probabilities = [
        np.mean([run[component] for run in validation_neural_runs], axis=0)
        for component in range(4)
    ]

    # Three GRU seeds form the proven core; one LSTM seed adds architectural
    # diversity so the averaged family decorrelates errors better than more
    # same-architecture seeds (which plateaued in validation).
    leave_one_out_specs = (
        (SEED + 15, "gru"),
        (SEED + 27, "gru"),
        (SEED + 33, "gru"),
        (SEED + 41, "lstm"),
    )
    trained_leave_one_out_specs = []
    validation_leave_one_out_models = []
    selected_leave_one_out_epochs = []
    for neural_seed, rnn_type in leave_one_out_specs:
        if not optional_training_allowed(len(validation_leave_one_out_models)):
            break
        model, epochs = train_leave_one_out_with_validation(
            acoustic_padded,
            categories_padded,
            missing_padded,
            targets_padded,
            lengths,
            development_rows,
            validation_rows,
            development_mean,
            development_std,
            neural_seed,
            rnn_type,
        )
        trained_leave_one_out_specs.append((neural_seed, rnn_type))
        validation_leave_one_out_models.append(model)
        selected_leave_one_out_epochs.append(epochs)
    validation_leave_one_out_runs = [
        predict_sequence_probabilities(
            model,
            acoustic_padded,
            categories_padded,
            lengths,
            validation_rows,
            development_mean,
            development_std,
        )
        for model in validation_leave_one_out_models
    ]
    validation_leave_one_out_probabilities = [
        np.mean(
            [run[component] for run in validation_leave_one_out_runs], axis=0
        )
        for component in range(4)
    ]

    validation_all_label_extra_probabilities = []
    validation_all_label_boosted_probabilities = []
    for component in range(4):
        extra_model, boosted_model = fit_all_label_tree_pair(
            tree_features,
            targets,
            event_rows,
            development_rows,
            component,
            final=False,
        )
        component_validation = target_excluded_features(
            validation_tree_features, component
        )
        validation_all_label_extra_probabilities.append(
            full_probabilities(extra_model, component_validation, component)
        )
        validation_all_label_boosted_probabilities.append(
            full_probabilities(boosted_model, component_validation, component)
        )

    # HPO occurs inside the submitted script and uses labeled validation rows
    # only. First select the tree/withheld-loss ensemble.
    selected_weights = []
    validation_base_probabilities = []
    for component in range(4):
        component_missing = validation_missing[:, component]
        best_accuracy = -1.0
        best_weights = (0.5, 0.0)
        best_probability = None
        for boosted_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            tree_probability = (
                boosted_weight * validation_boosted_probabilities[component]
                + (1.0 - boosted_weight) * validation_extra_probabilities[component]
            )
            for neural_weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                probability = (
                    (1.0 - neural_weight) * tree_probability
                    + neural_weight * validation_neural_probabilities[component]
                )
                prediction = (
                    np.argmax(probability, axis=1) + CLASS_OFFSETS[component]
                )
                accuracy = np.mean(
                    prediction[component_missing]
                    == validation_targets[component_missing, component]
                )
                if accuracy > best_accuracy:
                    best_accuracy = float(accuracy)
                    best_weights = (boosted_weight, neural_weight)
                    best_probability = probability
        selected_weights.append(best_weights)
        validation_base_probabilities.append(best_probability)

    # Then select the complementary all-label leave-one-out learner. Its
    # architecture, rather than synthetic masking, prevents target exposure.
    selected_leave_one_out_weights = []
    validation_final_probabilities = []
    for component in range(4):
        component_missing = validation_missing[:, component]
        best_accuracy = -1.0
        best_weight = 0.0
        best_probability = None
        for leave_one_out_weight in (
            0.0,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
        ):
            probability = (
                (1.0 - leave_one_out_weight)
                * validation_base_probabilities[component]
                + leave_one_out_weight
                * validation_leave_one_out_probabilities[component]
            )
            prediction = np.argmax(probability, axis=1) + CLASS_OFFSETS[component]
            accuracy = np.mean(
                prediction[component_missing]
                == validation_targets[component_missing, component]
            )
            if accuracy > best_accuracy:
                best_accuracy = float(accuracy)
                best_weight = leave_one_out_weight
                best_probability = probability
        selected_leave_one_out_weights.append(best_weight)
        validation_final_probabilities.append(best_probability)
        print(
            f"component {component}: validation_accuracy={best_accuracy:.6f}, "
            f"catboost_weight={selected_weights[component][0]:.2f}, "
            f"withheld_neural_weight={selected_weights[component][1]:.2f}, "
            f"all_label_weight={best_weight:.2f}"
        )

    # Finally fold in the all-label target-excluded tree learner. It trains on
    # every labeled event, so it complements the withheld-only models.
    selected_all_label_tree_weights = []
    for component in range(4):
        component_missing = validation_missing[:, component]
        best_accuracy = -1.0
        best_weights = (0.5, 0.0)
        best_probability = validation_final_probabilities[component]
        for boosted_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            tree_probability = (
                boosted_weight
                * validation_all_label_boosted_probabilities[component]
                + (1.0 - boosted_weight)
                * validation_all_label_extra_probabilities[component]
            )
            for tree_weight in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                probability = (
                    (1.0 - tree_weight) * validation_final_probabilities[component]
                    + tree_weight * tree_probability
                )
                prediction = np.argmax(probability, axis=1) + CLASS_OFFSETS[component]
                accuracy = np.mean(
                    prediction[component_missing]
                    == validation_targets[component_missing, component]
                )
                if accuracy > best_accuracy:
                    best_accuracy = float(accuracy)
                    best_weights = (boosted_weight, tree_weight)
                    best_probability = probability
        selected_all_label_tree_weights.append(best_weights)
        validation_final_probabilities[component] = best_probability
        print(
            f"component {component}: all_label_tree_catboost_weight="
            f"{best_weights[0]:.2f}, all_label_tree_weight={best_weights[1]:.2f}, "
            f"validation_accuracy={best_accuracy:.6f}"
        )

    # The published metric gives graded credit for nearby component values.
    # Select its probability/similarity tradeoff directly on train-only data.
    selected_utility_weights = [0.0, 0.0, 0.0, 0.0]
    for component in range(4):
        best_score = -1.0
        best_weight = 0.0
        for utility_weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            candidate = selected_utility_weights.copy()
            candidate[component] = utility_weight
            score = validation_metric(
                validation_final_probabilities,
                candidate,
                validation_targets,
                validation_missing,
                lengths[validation_rows],
            )
            if score > best_score:
                best_score = score
                best_weight = utility_weight
        selected_utility_weights[component] = best_weight
    print(
        "validation_metric="
        f"{validation_metric(validation_final_probabilities, selected_utility_weights, validation_targets, validation_missing, lengths[validation_rows]):.6f}, "
        f"utility_weights={selected_utility_weights}"
    )

    # Retrain every selected model family on all labeled rows.
    all_rows = np.arange(len(train))
    final_extra_models = []
    final_boosted_models = []
    for component in range(4):
        extra_model, boosted_model = fit_tree_pair(
            tree_features,
            targets,
            missing,
            event_rows,
            all_rows,
            component,
            final=True,
        )
        final_extra_models.append(extra_model)
        final_boosted_models.append(boosted_model)

    full_mean, full_std = acoustic_statistics(acoustic_padded, lengths, all_rows)
    final_neural_models = []
    for (neural_seed, rnn_type), epochs in zip(trained_neural_specs, selected_epochs):
        if not optional_training_allowed(len(final_neural_models)):
            break
        final_neural_models.append(
            train_sequence_full(
                acoustic_padded,
                categories_padded,
                missing_padded,
                targets_padded,
                lengths,
                full_mean,
                full_std,
                neural_seed,
                epochs,
                rnn_type,
            )
        )

    final_leave_one_out_models = []
    for (neural_seed, rnn_type), epochs in zip(
        trained_leave_one_out_specs, selected_leave_one_out_epochs
    ):
        if not optional_training_allowed(len(final_leave_one_out_models)):
            break
        final_leave_one_out_models.append(
            train_leave_one_out_full(
                acoustic_padded,
                categories_padded,
                missing_padded,
                targets_padded,
                lengths,
                full_mean,
                full_std,
                neural_seed,
                epochs,
                rnn_type,
            )
        )

    final_all_label_extra_models = []
    final_all_label_boosted_models = []
    for component in range(4):
        extra_model, boosted_model = fit_all_label_tree_pair(
            tree_features,
            targets,
            event_rows,
            all_rows,
            component,
            final=True,
        )
        final_all_label_extra_models.append(extra_model)
        final_all_label_boosted_models.append(boosted_model)

    # Test is loaded only after every fit, selection, and training statistic is final.
    test = pd.read_csv(public_dir / "test.csv")
    test_acoustic_file = np.load(public_dir / "test_features.npz")["acoustic"]
    if len(test) != len(test_acoustic_file):
        raise ValueError("test CSV and acoustic rows are not aligned")

    test_tree_rows = []
    test_views = []
    test_lengths = np.zeros(len(test), dtype=np.int64)
    test_acoustic_padded = np.zeros((len(test), MAX_LENGTH, 18), dtype=np.float32)
    test_categories_padded = np.zeros((len(test), MAX_LENGTH, 4), dtype=np.int64)
    for row_number, row in test.iterrows():
        pitch_view, timing_view = parse_views(row["pitch_view"], row["timing_view"])
        tree_row, neural_acoustic, categories = make_row_inputs(
            test_acoustic_file[row_number], pitch_view, timing_view
        )
        n = len(pitch_view)
        test_tree_rows.append(tree_row)
        test_views.append((pitch_view, timing_view))
        test_lengths[row_number] = n
        test_acoustic_padded[row_number, :n] = neural_acoustic
        test_categories_padded[row_number, :n] = categories
    test_tree_features = np.concatenate(test_tree_rows, axis=0)

    neural_test_runs = [
        predict_sequence_probabilities(
            model,
            test_acoustic_padded,
            test_categories_padded,
            test_lengths,
            np.arange(len(test)),
            full_mean,
            full_std,
        )
        for model in final_neural_models
    ]
    neural_test_probabilities = [
        np.mean([run[component] for run in neural_test_runs], axis=0)
        for component in range(4)
    ]
    leave_one_out_test_runs = [
        predict_sequence_probabilities(
            model,
            test_acoustic_padded,
            test_categories_padded,
            test_lengths,
            np.arange(len(test)),
            full_mean,
            full_std,
        )
        for model in final_leave_one_out_models
    ]
    leave_one_out_test_probabilities = [
        np.mean(
            [run[component] for run in leave_one_out_test_runs], axis=0
        )
        for component in range(4)
    ]

    predicted_components = []
    for component in range(4):
        boosted_weight, neural_weight = selected_weights[component]
        tree_probability = (
            boosted_weight
            * full_probabilities(
                final_boosted_models[component], test_tree_features, component
            )
            + (1.0 - boosted_weight)
            * full_probabilities(final_extra_models[component], test_tree_features, component)
        )
        base_probability = (
            (1.0 - neural_weight) * tree_probability
            + neural_weight * neural_test_probabilities[component]
        )
        leave_one_out_weight = selected_leave_one_out_weights[component]
        probability = (
            (1.0 - leave_one_out_weight) * base_probability
            + leave_one_out_weight * leave_one_out_test_probabilities[component]
        )

        all_label_boosted_weight, all_label_tree_weight = (
            selected_all_label_tree_weights[component]
        )
        component_test_features = target_excluded_features(
            test_tree_features, component
        )
        all_label_tree_probability = (
            all_label_boosted_weight
            * full_probabilities(
                final_all_label_boosted_models[component],
                component_test_features,
                component,
            )
            + (1.0 - all_label_boosted_weight)
            * full_probabilities(
                final_all_label_extra_models[component],
                component_test_features,
                component,
            )
        )
        probability = (
            (1.0 - all_label_tree_weight) * probability
            + all_label_tree_weight * all_label_tree_probability
        )
        predicted_components.append(
            decide_classes(
                probability, component, selected_utility_weights[component]
            )
        )
    predicted_components = np.column_stack(predicted_components)

    sequences = []
    start = 0
    for length, (pitch_view, timing_view) in zip(test_lengths, test_views):
        row_prediction = predicted_components[start : start + length]
        tokens = []
        for position, ((p, m), (g, d)) in enumerate(zip(pitch_view, timing_view)):
            learned = row_prediction[position]
            pitch = int(learned[0]) if p is None else p
            gap = int(learned[1]) if g is None else g
            duration = int(learned[2]) if d is None else d
            multiplicity = int(learned[3]) if m is None else m
            tokens.append(f"P{pitch:+03d}_G{gap}_D{duration}_M{multiplicity}")
        sequences.append(" ".join(tokens))
        start += length

    submission = pd.DataFrame(
        {"sample_id": test["sample_id"].astype(np.int64), "target_sequence": sequences}
    )
    if (
        start != len(predicted_components)
        or len(submission) != len(test)
        or submission["sample_id"].duplicated().any()
        or submission["target_sequence"].eq("").any()
    ):
        raise ValueError("invalid submission")

    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)


if __name__ == "__main__":
    main()
