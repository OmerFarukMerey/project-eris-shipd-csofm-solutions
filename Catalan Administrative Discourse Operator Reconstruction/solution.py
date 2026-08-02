#!/usr/bin/env python3
"""Train-only discourse-operator reconstruction with algebra-constrained decoding."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset


BASE_SEED = 20260801
TRAIN_GUARD_SECONDS = 3000.0
N_FOLDS = 5
NEURAL_SEEDS_PER_FOLD = 2
MAX_NEURAL_EPOCHS = 24
NEURAL_PATIENCE = 5
LINEAR_MAX_FEATURES = 180_000
WORD_BUCKETS = 8192
PAD_ID = 0
UNKNOWN_ID = 1
MASK_ID = 2


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_context(context: object, path_length: int) -> tuple[list[str], list[tuple[list[str], list[str]]]]:
    """Parse one row without allowing malformed sections to abort the run."""
    text = "" if pd.isna(context) else str(context)
    source_sections = text.split(" NEXT_GAP ") if text else []
    raw_sections: list[str] = []
    parts: list[tuple[list[str], list[str]]] = []
    for gap_index in range(path_length):
        section = source_sections[gap_index] if gap_index < len(source_sections) else ""
        tokens = section.split()
        left: list[str] = []
        right: list[str] = []
        try:
            left_marker = tokens.index("LEFT")
            mask_marker = tokens.index("MASK", left_marker + 1)
            right_marker = tokens.index("RIGHT", mask_marker + 1)
            left = tokens[left_marker + 1 : mask_marker]
            right = tokens[right_marker + 1 :]
        except (ValueError, IndexError):
            pass
        if not section:
            section = f"G{gap_index + 1} LEFT MASK RIGHT"
        raw_sections.append(f"POS{gap_index + 1} {section}")
        parts.append((left[-20:], right[:28]))
    return raw_sections, parts


def word_bucket_id(token: str) -> int:
    """Map the public w[0-9a-f]{4} grammar to a stable embedding row."""
    if len(token) != 5 or token[0] != "w":
        return UNKNOWN_ID
    try:
        value = int(token[1:], 16)
    except ValueError:
        return UNKNOWN_ID
    return value + 3 if 0 <= value < WORD_BUCKETS else UNKNOWN_ID


def encode_neural_inputs(parsed_rows: Sequence[Sequence[tuple[list[str], list[str]]]], path_length: int) -> torch.Tensor:
    encoded = np.zeros((len(parsed_rows), path_length, 49), dtype=np.int64)
    encoded[:, :, 20] = MASK_ID
    for row_index, row in enumerate(parsed_rows):
        for gap_index, (left, right) in enumerate(row[:path_length]):
            left_ids = [word_bucket_id(token) for token in left[-20:]]
            right_ids = [word_bucket_id(token) for token in right[:28]]
            encoded[row_index, gap_index, 20 - len(left_ids) : 20] = left_ids
            encoded[row_index, gap_index, 21 : 21 + len(right_ids)] = right_ids
    return torch.from_numpy(encoded)


def engineered_gap_text(parts: tuple[list[str], list[str]], gap_index: int) -> str:
    """Create local phrase, side, and boundary-distance features for a trained classifier."""
    left, right = parts
    left = left[-20:]
    right = right[:28]
    sequence = left + ["MASK"] + right
    features = [f"U={token}" for token in sequence]
    features.extend(f"B={sequence[i]}~{sequence[i + 1]}" for i in range(len(sequence) - 1))
    features.extend(
        f"T={sequence[i]}~{sequence[i + 1]}~{sequence[i + 2]}" for i in range(len(sequence) - 2)
    )
    features.extend(f"L={token}" for token in left)
    features.extend(f"R={token}" for token in right)
    features.extend(f"LD{min(distance, 6)}={token}" for distance, token in enumerate(reversed(left), 1))
    features.extend(f"RD{min(distance, 6)}={token}" for distance, token in enumerate(right, 1))
    features.append(f"POS={gap_index + 1}")
    return " ".join(features)

def build_focused_rows(
    parsed_rows: Sequence[Sequence[tuple[list[str], list[str]]]],
    left_window: int,
    right_window: int,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in parsed_rows:
        focused: list[str] = []
        for gap_index, (left, right) in enumerate(row):
            left_text = " ".join(left[-left_window:])
            right_text = " ".join(right[:right_window])
            focused.append(f"POS{gap_index + 1} LEFT {left_text} MASK RIGHT {right_text}")
        rows.append(focused)
    return rows


def flatten_rows(rows: Sequence[Sequence[str]], indices: Iterable[int]) -> list[str]:
    return [rows[int(row_index)][gap_index] for row_index in indices for gap_index in range(len(rows[int(row_index)]))]


def build_exact_gap_groups(contexts: Sequence[object]) -> np.ndarray:
    """Keep rows sharing a participant-visible gap in the same validation fold."""
    count = len(contexts)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    owner: dict[str, int] = {}
    for row_index, context in enumerate(contexts):
        text = "" if pd.isna(context) else str(context)
        for section in text.split(" NEXT_GAP "):
            if section in owner:
                union(row_index, owner[section])
            else:
                owner[section] = row_index
    return np.asarray([find(index) for index in range(count)], dtype=np.int64)


class AlgebraDecoder:
    """Public finite-field construction and exact model-scored meet-in-the-middle decoder."""

    def __init__(self, specification: dict[str, object]):
        self.modulus = int(specification["modulus"])
        self.tokens = [str(token) for token in specification["operator_tokens"]]
        self.path_length = int(specification["path_length"])
        self.identity = np.asarray(specification["identity_matrix"], dtype=np.int16)
        operator_count = len(self.tokens)
        if self.path_length != 6:
            raise ValueError("This challenge requires a six-token algebra path")
        self.triples = np.stack(
            np.unravel_index(np.arange(operator_count**3), (operator_count, operator_count, operator_count)),
            axis=1,
        ).astype(np.int16)
        self.key_space = self.modulus**4
        self.inverse_scalars = np.zeros(self.modulus, dtype=np.int16)
        for value in range(1, self.modulus):
            self.inverse_scalars[value] = pow(value, -1, self.modulus)

    def operator_matrices(self, seed: str) -> np.ndarray:
        matrices: list[list[int]] = []
        for token in self.tokens:
            counter = 0
            while True:
                digest = hashlib.sha256(f"FFDOR1|{seed}|{token}|{counter}".encode("ascii")).digest()
                a, b, c, d = (byte % self.modulus for byte in digest[:4])
                determinant = (a * d - b * c) % self.modulus
                if determinant and [a, b, c, d] != self.identity.tolist():
                    matrices.append([a, b, c, d])
                    break
                counter += 1
        return np.asarray(matrices, dtype=np.int16)

    def multiply(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        modulus = self.modulus
        return np.stack(
            (
                (first[..., 0] * second[..., 0] + first[..., 1] * second[..., 2]) % modulus,
                (first[..., 0] * second[..., 1] + first[..., 1] * second[..., 3]) % modulus,
                (first[..., 2] * second[..., 0] + first[..., 3] * second[..., 2]) % modulus,
                (first[..., 2] * second[..., 1] + first[..., 3] * second[..., 3]) % modulus,
            ),
            axis=-1,
        ).astype(np.int16)

    def matrix_key(self, matrix: np.ndarray) -> np.ndarray:
        modulus = self.modulus
        values = matrix.astype(np.int32)
        return (((values[..., 0] * modulus + values[..., 1]) * modulus + values[..., 2]) * modulus + values[..., 3]).astype(
            np.int32
        )

    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        determinant = (matrix[..., 0] * matrix[..., 3] - matrix[..., 1] * matrix[..., 2]) % self.modulus
        multiplier = self.inverse_scalars[determinant]
        return (
            np.stack(
                (
                    matrix[..., 3] * multiplier,
                    -matrix[..., 1] * multiplier,
                    -matrix[..., 2] * multiplier,
                    matrix[..., 0] * multiplier,
                ),
                axis=-1,
            ).astype(np.int16)
            % self.modulus
        )

    def prepare(self, seed: str, target: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrices = self.operator_matrices(seed)
        triples = self.triples
        products = self.multiply(
            self.multiply(matrices[triples[:, 0]], matrices[triples[:, 1]]), matrices[triples[:, 2]]
        )
        keys = self.matrix_key(products)
        target_matrix = np.broadcast_to(np.asarray(target, dtype=np.int16), products.shape)
        required_suffix = self.multiply(self.inverse(products), target_matrix)
        return keys, self.matrix_key(required_suffix), matrices

    def decode_prepared(self, logits: np.ndarray, prepared: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        keys, required_keys, _ = prepared
        triples = self.triples
        prefix_scores = (
            logits[0, triples[:, 0]] + logits[1, triples[:, 1]] + logits[2, triples[:, 2]]
        ).astype(np.float64)
        suffix_scores = (
            logits[3, triples[:, 0]] + logits[4, triples[:, 1]] + logits[5, triples[:, 2]]
        ).astype(np.float64)
        best_suffix_score = np.full(self.key_space, -np.inf, dtype=np.float64)
        np.maximum.at(best_suffix_score, keys, suffix_scores)
        best_suffix_index = np.full(self.key_space, -1, dtype=np.int32)
        winners = np.flatnonzero(suffix_scores == best_suffix_score[keys])
        best_suffix_index[keys[winners]] = winners
        totals = prefix_scores + best_suffix_score[required_keys]
        prefix_index = int(np.argmax(totals))
        suffix_index = int(best_suffix_index[required_keys[prefix_index]])
        if suffix_index < 0:
            raise RuntimeError("No algebra-compatible suffix found")
        return np.concatenate((triples[prefix_index], triples[suffix_index])).astype(np.int64)

    def decode(self, logits: np.ndarray, seed: str, target: Sequence[int]) -> np.ndarray:
        return self.decode_prepared(logits, self.prepare(seed, target))

    def enumerate_prepared(self, prepared: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        keys, required_keys, _ = prepared
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        left = np.searchsorted(sorted_keys, required_keys, side="left")
        right = np.searchsorted(sorted_keys, required_keys, side="right")
        counts = right - left
        total = int(counts.sum())
        if total == 0:
            raise RuntimeError("No algebra-compatible paths found")
        prefix_indices = np.repeat(np.arange(len(self.triples), dtype=np.int32), counts)
        starts = np.repeat(left, counts)
        group_bases = np.repeat(np.cumsum(counts) - counts, counts)
        sorted_positions = starts + np.arange(total, dtype=np.int64) - group_bases
        suffix_indices = order[sorted_positions]
        return np.concatenate(
            (self.triples[prefix_indices], self.triples[suffix_indices]), axis=1
        ).astype(np.int16)

    def decode_marginal_prepared(
        self,
        logits: np.ndarray,
        prepared: tuple[np.ndarray, np.ndarray, np.ndarray],
        temperature: float,
    ) -> np.ndarray:
        paths = self.enumerate_prepared(prepared)
        positions = np.arange(self.path_length)
        scores = logits[positions[None, :], paths].sum(axis=1).astype(np.float64)
        posterior = np.exp((scores - scores.max()) / max(float(temperature), np.finfo(float).eps))
        posterior /= posterior.sum()
        marginals = np.zeros((self.path_length, len(self.tokens)), dtype=np.float64)
        for position in range(self.path_length):
            np.add.at(marginals[position], paths[:, position], posterior)
        utilities = marginals[positions[None, :], paths].sum(axis=1)
        return paths[int(np.argmax(utilities))].astype(np.int64)

    def decode_marginal(
        self,
        logits: np.ndarray,
        seed: str,
        target: Sequence[int],
        temperature: float,
    ) -> np.ndarray:
        return self.decode_marginal_prepared(logits, self.prepare(seed, target), temperature)

    def product_from_matrices(self, matrices: np.ndarray, path: Sequence[int]) -> np.ndarray:
        product = self.identity.copy()
        for token_id in path:
            product = self.multiply(product, matrices[int(token_id)])
        return product


class GapTransformer(nn.Module):
    """Shared local encoder followed by a six-gap sequence encoder."""

    def __init__(self, label_count: int, path_length: int, model_width: int = 96, dropout: float = 0.15):
        super().__init__()
        self.word_embedding = nn.Embedding(WORD_BUCKETS + 3, model_width, padding_idx=PAD_ID)
        self.relative_embedding = nn.Embedding(49, model_width)
        self.gap_embedding = nn.Embedding(path_length, model_width)
        self.input_norm = nn.LayerNorm(model_width)
        local_layer = nn.TransformerEncoderLayer(
            model_width,
            nhead=4,
            dim_feedforward=model_width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=2, enable_nested_tensor=False)
        row_layer = nn.TransformerEncoderLayer(
            model_width,
            nhead=4,
            dim_feedforward=model_width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=2, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(model_width)
        self.classifier = nn.Linear(model_width, label_count)
        self.register_buffer("relative_ids", torch.arange(49), persistent=False)
        self.register_buffer("gap_ids", torch.arange(path_length), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, gap_count, sequence_length = inputs.shape
        flat_inputs = inputs.reshape(batch_size * gap_count, sequence_length)
        states = self.word_embedding(flat_inputs) + self.relative_embedding(self.relative_ids)[None, :, :]
        states = self.input_norm(states)
        states = self.local_encoder(states, src_key_padding_mask=flat_inputs.eq(PAD_ID))
        gap_states = states[:, 20, :].reshape(batch_size, gap_count, -1)
        gap_states = gap_states + self.gap_embedding(self.gap_ids)[None, :, :]
        gap_states = self.row_encoder(gap_states)
        return self.classifier(self.output_norm(gap_states))


def predict_neural(model: nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs.append(model(batch).float().cpu())
    return torch.cat(outputs).numpy().astype(np.float32)


def train_neural_fold(
    all_inputs: torch.Tensor,
    targets: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_inputs: torch.Tensor,
    label_count: int,
    path_length: int,
    seed: int,
    device: torch.device,
    run_started: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    seed_everything(seed)
    model = GapTransformer(label_count, path_length).to(device)
    train_batch_size = 64 if device.type == "cuda" else 32
    inference_batch_size = 128 if device.type == "cuda" else 64
    train_loader = DataLoader(
        TensorDataset(all_inputs[train_indices], targets[train_indices]),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(all_inputs[validation_indices], targets[validation_indices]),
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_NEURAL_EPOCHS, eta_min=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, MAX_NEURAL_EPOCHS + 1):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=device.type == "cuda")
            batch_targets = batch_targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(batch_inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, label_count), batch_targets.reshape(-1), label_smoothing=0.04
                )
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        validation_outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_inputs, _ in validation_loader:
                batch_inputs = batch_inputs.to(device, non_blocking=device.type == "cuda")
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    validation_outputs.append(model(batch_inputs).float().cpu())
        validation_logits = torch.cat(validation_outputs)
        accuracy = float((validation_logits.argmax(-1) == targets[validation_indices]).float().mean())
        if accuracy > best_accuracy + 1e-6:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        elif epoch - best_epoch >= NEURAL_PATIENCE:
            break
        if time.monotonic() - run_started >= TRAIN_GUARD_SECONDS and best_state is not None:
            break

    if best_state is None:
        raise RuntimeError("Neural training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_logits = predict_neural(model, all_inputs[validation_indices], device, inference_batch_size)
    test_logits = predict_neural(model, test_inputs, device, inference_batch_size)
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return validation_logits, test_logits, best_epoch, best_accuracy


def make_raw_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        token_pattern=r"\S+",
        ngram_range=(1, 3),
        min_df=2,
        max_features=LINEAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )


def make_engineered_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        token_pattern=r"\S+",
        ngram_range=(1, 1),
        min_df=2,
        max_features=LINEAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )


def make_linear_model(regularization: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(regularization),
        solver="lbfgs",
        max_iter=220,
        random_state=BASE_SEED,
        n_jobs=-1,
    )


def align_decision_scores(
    model: LogisticRegression,
    features: object,
    row_count: int,
    path_length: int,
    token_to_id: dict[str, int],
) -> np.ndarray:
    raw = np.asarray(model.decision_function(features), dtype=np.float32).reshape(row_count, path_length, -1)
    aligned = np.full((row_count, path_length, len(token_to_id)), -1e4, dtype=np.float32)
    for source_index, token in enumerate(model.classes_):
        if str(token) in token_to_id:
            aligned[:, :, token_to_id[str(token)]] = raw[:, :, source_index]
    return aligned


def token_levenshtein(first: Sequence[int], second: Sequence[int]) -> int:
    previous = list(range(len(second) + 1))
    for row_index, first_token in enumerate(first, 1):
        current = [row_index]
        for column_index, second_token in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + int(first_token != second_token),
                )
            )
        previous = current
    return previous[-1]


def mean_challenge_metric(
    predictions: np.ndarray,
    targets: np.ndarray,
    boundaries: np.ndarray,
    decoder: AlgebraDecoder,
    prepared: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    exact_matrix: bool,
) -> float:
    row_scores: list[float] = []
    for row_index, (prediction, target) in enumerate(zip(predictions, targets)):
        position_accuracy = float(np.mean(prediction == target))
        predicted_edges = Counter(zip(prediction[:-1], prediction[1:]))
        target_edges = Counter(zip(target[:-1], target[1:]))
        edge_f1 = sum((predicted_edges & target_edges).values()) / 5.0
        edit_similarity = 1.0 - token_levenshtein(prediction.tolist(), target.tolist()) / 6.0
        if exact_matrix:
            matrix_accuracy = 1.0
        else:
            product = decoder.product_from_matrices(prepared[row_index][2], prediction)
            matrix_accuracy = float(np.mean(product == boundaries[row_index]))
        exact_path = float(np.array_equal(prediction, target))
        row_scores.append(
            0.45 * position_accuracy
            + 0.20 * edge_f1
            + 0.15 * edit_similarity
            + 0.15 * matrix_accuracy
            + 0.05 * exact_path
        )
    return float(np.mean(row_scores))


def tune_linear_regularization(
    raw_rows: Sequence[Sequence[str]],
    target_tokens: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seeds: Sequence[str],
    boundaries: np.ndarray,
    decoder: AlgebraDecoder,
    token_to_id: dict[str, int],
) -> float:
    vectorizer = make_raw_vectorizer()
    train_text = flatten_rows(raw_rows, train_indices)
    validation_text = flatten_rows(raw_rows, validation_indices)
    train_features = vectorizer.fit_transform(train_text)
    validation_features = vectorizer.transform(validation_text)
    train_labels = target_tokens[train_indices].reshape(-1)
    validation_targets = np.asarray([[token_to_id[token] for token in row] for row in target_tokens[validation_indices]])
    validation_boundaries = boundaries[validation_indices]
    prepared = [decoder.prepare(seeds[int(index)], boundaries[int(index)]) for index in validation_indices]
    candidates = np.geomspace(0.75, 20.0, num=5)
    best_score = -math.inf
    best_value = float(candidates[0])
    for candidate in candidates:
        model = make_linear_model(float(candidate))
        model.fit(train_features, train_labels)
        logits = align_decision_scores(
            model, validation_features, len(validation_indices), decoder.path_length, token_to_id
        )
        predictions = np.asarray(
            [decoder.decode_prepared(logits[row_index], prepared[row_index]) for row_index in range(len(prepared))]
        )
        score = mean_challenge_metric(
            predictions, validation_targets, validation_boundaries, decoder, prepared, exact_matrix=True
        )
        log(f"regularization candidate C={candidate:.4g}: constrained metric={score:.6f}")
        if score > best_score:
            best_score = score
            best_value = float(candidate)
    log(f"selected C={best_value:.6g} on the train-only calibration fold")
    return best_value

def tune_primary_configuration(
    raw_rows: Sequence[Sequence[str]],
    parsed_rows: Sequence[Sequence[tuple[list[str], list[str]]]],
    target_tokens: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seeds: Sequence[str],
    boundaries: np.ndarray,
    decoder: AlgebraDecoder,
    token_to_id: dict[str, int],
) -> tuple[list[list[str]], tuple[int, int], float]:
    baseline_regularization = tune_linear_regularization(
        raw_rows,
        target_tokens,
        train_indices,
        validation_indices,
        seeds,
        boundaries,
        decoder,
        token_to_id,
    )
    window_candidates = ((4, 8), (6, 12), (8, 16), (12, 20), (20, 28))
    validation_targets = np.asarray(
        [[token_to_id[token] for token in row] for row in target_tokens[validation_indices]]
    )
    validation_boundaries = boundaries[validation_indices]
    prepared = [decoder.prepare(seeds[int(index)], boundaries[int(index)]) for index in validation_indices]
    best_score = -math.inf
    best_window = window_candidates[0]
    best_rows = build_focused_rows(parsed_rows, *best_window)
    for window in window_candidates:
        candidate_rows = build_focused_rows(parsed_rows, *window)
        vectorizer = make_raw_vectorizer()
        train_features = vectorizer.fit_transform(flatten_rows(candidate_rows, train_indices))
        validation_features = vectorizer.transform(flatten_rows(candidate_rows, validation_indices))
        model = make_linear_model(baseline_regularization)
        model.fit(train_features, target_tokens[train_indices].reshape(-1))
        logits = align_decision_scores(
            model, validation_features, len(validation_indices), decoder.path_length, token_to_id
        )
        predictions = np.asarray(
            [decoder.decode_prepared(logits[row_index], prepared[row_index]) for row_index in range(len(prepared))]
        )
        score = mean_challenge_metric(
            predictions, validation_targets, validation_boundaries, decoder, prepared, exact_matrix=True
        )
        log(f"context window left={window[0]}, right={window[1]}: constrained metric={score:.6f}")
        if score > best_score:
            best_score = score
            best_window = window
            best_rows = candidate_rows
    log(f"selected context window left={best_window[0]}, right={best_window[1]}")
    regularization = tune_linear_regularization(
        best_rows,
        target_tokens,
        train_indices,
        validation_indices,
        seeds,
        boundaries,
        decoder,
        token_to_id,
    )
    return best_rows, best_window, regularization


def fit_linear_family(
    vectorizer: TfidfVectorizer,
    train_rows: Sequence[Sequence[str]],
    test_rows: Sequence[Sequence[str]],
    target_tokens: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    regularization: float,
    token_to_id: dict[str, int],
    path_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_text = flatten_rows(train_rows, train_indices)
    validation_text = flatten_rows(train_rows, validation_indices)
    test_text = flatten_rows(test_rows, range(len(test_rows)))
    train_features = vectorizer.fit_transform(train_text)
    validation_features = vectorizer.transform(validation_text)
    test_features = vectorizer.transform(test_text)
    model = make_linear_model(regularization)
    model.fit(train_features, target_tokens[train_indices].reshape(-1))
    validation_logits = align_decision_scores(
        model, validation_features, len(validation_indices), path_length, token_to_id
    )
    test_logits = align_decision_scores(model, test_features, len(test_rows), path_length, token_to_id)
    return validation_logits, test_logits


def integer_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in integer_compositions(total - first, parts - 1):
            yield (first,) + rest


def tune_ensemble_and_decoder(
    family_names: Sequence[str],
    oof_families: Sequence[np.ndarray],
    targets: np.ndarray,
    seeds: Sequence[str],
    boundaries: np.ndarray,
    decoder: AlgebraDecoder,
) -> tuple[np.ndarray, str, float, np.ndarray, list[float], float]:
    common = np.ones(len(targets), dtype=bool)
    for family in oof_families:
        common &= np.isfinite(family).all(axis=(1, 2))
    validation_indices = np.flatnonzero(common)
    if len(validation_indices) == 0:
        raise RuntimeError("No out-of-fold rows are available for decoder tuning")
    validation_targets = targets[validation_indices]
    validation_boundaries = boundaries[validation_indices]
    prepared = [decoder.prepare(seeds[int(index)], boundaries[int(index)]) for index in validation_indices]

    scales: list[float] = []
    normalized: list[np.ndarray] = []
    for family in oof_families:
        scale = float(np.std(family[validation_indices]))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        scales.append(scale)
        normalized.append(family[validation_indices] / scale)

    best_score = -math.inf
    best_weights = np.full(len(normalized), 1.0 / len(normalized), dtype=np.float64)
    best_mode = "independent"
    best_temperature = 1.0
    grid_denominator = 10
    for composition in integer_compositions(grid_denominator, len(normalized)):
        weights = np.asarray(composition, dtype=np.float64) / grid_denominator
        blended = np.zeros_like(normalized[0], dtype=np.float32)
        for weight, family in zip(weights, normalized):
            blended += np.float32(weight) * family

        independent_predictions = blended.argmax(axis=-1)
        independent_score = mean_challenge_metric(
            independent_predictions,
            validation_targets,
            validation_boundaries,
            decoder,
            prepared,
            exact_matrix=False,
        )
        if independent_score > best_score:
            best_score = independent_score
            best_weights = weights.copy()
            best_mode = "independent"

        constrained_predictions = np.asarray(
            [decoder.decode_prepared(blended[row_index], prepared[row_index]) for row_index in range(len(prepared))]
        )
        constrained_score = mean_challenge_metric(
            constrained_predictions,
            validation_targets,
            validation_boundaries,
            decoder,
            prepared,
            exact_matrix=True,
        )
        if constrained_score > best_score:
            best_score = constrained_score
            best_weights = weights.copy()
            best_mode = "exact"

    best_blended = np.zeros_like(normalized[0], dtype=np.float32)
    for weight, family in zip(best_weights, normalized):
        best_blended += np.float32(weight) * family
    for temperature in np.geomspace(0.1, 1.6, num=6):
        marginal_predictions = np.asarray(
            [
                decoder.decode_marginal_prepared(best_blended[row_index], prepared[row_index], float(temperature))
                for row_index in range(len(prepared))
            ]
        )
        marginal_score = mean_challenge_metric(
            marginal_predictions,
            validation_targets,
            validation_boundaries,
            decoder,
            prepared,
            exact_matrix=True,
        )
        log(f"marginal decoder temperature={temperature:.4g}: metric={marginal_score:.6f}")
        if marginal_score > best_score:
            best_score = marginal_score
            best_mode = "marginal"
            best_temperature = float(temperature)

    weight_text = ", ".join(f"{name}={weight:.2f}" for name, weight in zip(family_names, best_weights))
    log(
        f"OOF validation rows={len(validation_indices)}, metric={best_score:.6f}, "
        f"decoder={best_mode}, temperature={best_temperature:.6g}, weights: {weight_text}"
    )
    return best_weights, best_mode, best_score, validation_indices, scales, best_temperature


def write_placeholder(test_ids: Sequence[object], output_path: Path, token: str, path_length: int) -> None:
    placeholder = " ".join([token] * path_length)
    pd.DataFrame({"sample_id": list(test_ids), "operator_path": [placeholder] * len(test_ids)}).to_csv(
        output_path, index=False
    )


def solve(public_dir: Path, output_path: Path) -> None:
    run_started = time.monotonic()
    required = [public_dir / "train.csv", public_dir / "test.csv", public_dir / "operator_algebra.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")

    specification = json.loads((public_dir / "operator_algebra.json").read_text(encoding="utf-8"))
    decoder = AlgebraDecoder(specification)
    tokens = decoder.tokens
    token_to_id = {token: index for index, token in enumerate(tokens)}
    test = pd.read_csv(public_dir / "test.csv")
    if "sample_id" not in test.columns:
        raise ValueError("test.csv is missing required sample_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_placeholder(test["sample_id"].tolist(), output_path, tokens[0], decoder.path_length)
    log(f"wrote early schema-valid placeholder with {len(test)} rows")

    try:
        train = pd.read_csv(public_dir / "train.csv")
        required_train_columns = {"context_sequence", "constraint_seed", "boundary_matrix", "operator_path"}
        required_test_columns = {"context_sequence", "constraint_seed", "boundary_matrix"}
        if not required_train_columns.issubset(train.columns):
            raise ValueError(f"train.csv lacks columns: {sorted(required_train_columns - set(train.columns))}")
        if not required_test_columns.issubset(test.columns):
            raise ValueError(f"test.csv lacks columns: {sorted(required_test_columns - set(test.columns))}")

        target_tokens = np.asarray([str(path).split() for path in train["operator_path"]], dtype=object)
        if target_tokens.ndim != 2 or target_tokens.shape[1] != decoder.path_length:
            raise ValueError("Training targets do not have the algebra path length")
        target_ids = np.asarray([[token_to_id[token] for token in row] for row in target_tokens], dtype=np.int64)
        target_tensor = torch.from_numpy(target_ids)
        train_boundaries = np.asarray([json.loads(value) for value in train["boundary_matrix"]], dtype=np.int16)
        test_boundaries = [json.loads(value) for value in test["boundary_matrix"]]
        train_seeds = train["constraint_seed"].astype(str).tolist()
        test_seeds = test["constraint_seed"].astype(str).tolist()

        train_parsed_records = [parse_context(value, decoder.path_length) for value in train["context_sequence"]]
        test_parsed_records = [parse_context(value, decoder.path_length) for value in test["context_sequence"]]
        train_raw_rows = [record[0] for record in train_parsed_records]
        train_parts = [record[1] for record in train_parsed_records]
        test_parts = [record[1] for record in test_parsed_records]
        train_engineered_rows = [
            [engineered_gap_text(row[gap], gap) for gap in range(decoder.path_length)] for row in train_parts
        ]
        test_engineered_rows = [
            [engineered_gap_text(row[gap], gap) for gap in range(decoder.path_length)] for row in test_parts
        ]
        train_neural_inputs = encode_neural_inputs(train_parts, decoder.path_length)
        test_neural_inputs = encode_neural_inputs(test_parts, decoder.path_length)

        groups = build_exact_gap_groups(train["context_sequence"].tolist())
        group_count = len(np.unique(groups))
        fold_count = min(N_FOLDS, group_count)
        if fold_count < 2:
            raise ValueError("Training data has fewer than two validation groups")
        folds = list(GroupKFold(n_splits=fold_count).split(np.arange(len(train)), groups=groups))
        log(f"using {fold_count}-fold exact-gap-group validation over {group_count} groups")

        calibration_train, calibration_validation = folds[0]
        train_primary_rows, primary_window, regularization = tune_primary_configuration(
            train_raw_rows,
            train_parts,
            target_tokens,
            calibration_train,
            calibration_validation,
            train_seeds,
            train_boundaries,
            decoder,
            token_to_id,
        )
        test_primary_rows = build_focused_rows(test_parts, *primary_window)

        shape = (len(train), decoder.path_length, len(tokens))
        test_shape = (len(test), decoder.path_length, len(tokens))
        oof_raw = np.full(shape, np.nan, dtype=np.float32)
        oof_engineered = np.full(shape, np.nan, dtype=np.float32)
        oof_neural = np.full(shape, np.nan, dtype=np.float32)
        test_raw_sum = np.zeros(test_shape, dtype=np.float32)
        test_engineered_sum = np.zeros(test_shape, dtype=np.float32)
        test_neural_sum = np.zeros(test_shape, dtype=np.float32)
        raw_count = 0
        engineered_count = 0
        neural_count = 0

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        log(f"neural device: {device}")

        for fold_index, (fold_train, fold_validation) in enumerate(folds):
            if fold_index > 0 and time.monotonic() - run_started >= TRAIN_GUARD_SECONDS:
                log("wall-clock guard reached before another fold; moving to inference")
                break
            log(f"training fold {fold_index + 1}/{fold_count}")
            try:
                validation_logits, test_logits = fit_linear_family(
                    make_raw_vectorizer(),
                    train_primary_rows,
                    test_primary_rows,
                    target_tokens,
                    fold_train,
                    fold_validation,
                    regularization,
                    token_to_id,
                    decoder.path_length,
                )
                oof_raw[fold_validation] = validation_logits
                test_raw_sum += test_logits
                raw_count += 1
            except Exception as error:
                log(f"focused linear fold failed and was skipped: {error}")

            try:
                validation_logits, test_logits = fit_linear_family(
                    make_engineered_vectorizer(),
                    train_engineered_rows,
                    test_engineered_rows,
                    target_tokens,
                    fold_train,
                    fold_validation,
                    regularization,
                    token_to_id,
                    decoder.path_length,
                )
                oof_engineered[fold_validation] = validation_logits
                test_engineered_sum += test_logits
                engineered_count += 1
            except Exception as error:
                log(f"engineered linear fold failed and was skipped: {error}")

            fold_neural_validation: list[np.ndarray] = []
            for seed_index in range(NEURAL_SEEDS_PER_FOLD):
                if neural_count > 0 and time.monotonic() - run_started >= TRAIN_GUARD_SECONDS:
                    log("wall-clock guard reached before another neural seed")
                    break
                model_seed = BASE_SEED + fold_index * 101 + seed_index * 1009
                try:
                    validation_logits, test_logits, best_epoch, best_accuracy = train_neural_fold(
                        train_neural_inputs,
                        target_tensor,
                        fold_train,
                        fold_validation,
                        test_neural_inputs,
                        len(tokens),
                        decoder.path_length,
                        model_seed,
                        device,
                        run_started,
                    )
                    fold_neural_validation.append(validation_logits)
                    test_neural_sum += test_logits
                    neural_count += 1
                    log(
                        f"fold {fold_index + 1} neural seed {seed_index + 1}: "
                        f"epoch={best_epoch}, position_accuracy={best_accuracy:.6f}"
                    )
                except Exception as error:
                    log(f"neural fold seed failed and was skipped: {error}")
            if fold_neural_validation:
                oof_neural[fold_validation] = np.mean(fold_neural_validation, axis=0, dtype=np.float32)

        family_names: list[str] = []
        oof_families: list[np.ndarray] = []
        test_families: list[np.ndarray] = []
        if raw_count:
            family_names.append("focused_tfidf")
            oof_families.append(oof_raw)
            test_families.append(test_raw_sum / raw_count)
        if engineered_count:
            family_names.append("boundary_tfidf")
            oof_families.append(oof_engineered)
            test_families.append(test_engineered_sum / engineered_count)
        if neural_count:
            family_names.append("neural_transformer")
            oof_families.append(oof_neural)
            test_families.append(test_neural_sum / neural_count)
        if not family_names:
            log("all trained models failed; retaining the early placeholder")
            return

        weights, decode_mode, validation_score, validation_indices, scales, decode_temperature = (
            tune_ensemble_and_decoder(
                family_names,
                oof_families,
                target_ids,
                train_seeds,
                train_boundaries,
                decoder,
            )
        )
        del validation_score, validation_indices
        test_blend = np.zeros(test_shape, dtype=np.float32)
        for weight, scale, family in zip(weights, scales, test_families):
            test_blend += np.float32(weight / scale) * family

        fallback_ids = test_blend.argmax(axis=-1)
        predictions = fallback_ids.copy()
        if decode_mode in {"exact", "marginal"}:
            for row_index in range(len(test)):
                try:
                    if decode_mode == "marginal":
                        predictions[row_index] = decoder.decode_marginal(
                            test_blend[row_index],
                            test_seeds[row_index],
                            test_boundaries[row_index],
                            decode_temperature,
                        )
                    else:
                        predictions[row_index] = decoder.decode(
                            test_blend[row_index], test_seeds[row_index], test_boundaries[row_index]
                        )
                except Exception as error:
                    log(f"row {row_index} constrained decode failed; using model argmax: {error}")

        paths = [" ".join(tokens[int(token_id)] for token_id in row) for row in predictions]
        valid_token_set = set(tokens)
        for row_index, path in enumerate(paths):
            pieces = path.split(" ")
            if len(pieces) != decoder.path_length or any(piece not in valid_token_set for piece in pieces):
                paths[row_index] = " ".join(tokens[int(token_id)] for token_id in fallback_ids[row_index])

        submission = pd.DataFrame({"sample_id": test["sample_id"].tolist(), "operator_path": paths})
        if submission.columns.tolist() != ["sample_id", "operator_path"]:
            log("submission column check failed; retaining placeholder")
            return
        if len(submission) != len(test) or submission["sample_id"].isna().any():
            log("submission completeness check failed; retaining placeholder")
            return
        if submission["sample_id"].duplicated().any() or set(submission["sample_id"]) != set(test["sample_id"]):
            log("submission ID-set check failed; retaining placeholder")
            return
        if submission["operator_path"].isna().any() or (submission["operator_path"].str.len() == 0).any():
            log("submission prediction check failed; retaining placeholder")
            return
        submission.to_csv(output_path, index=False, encoding="utf-8")
        log(
            f"wrote {len(submission)} model predictions to {output_path}; "
            f"trained families={family_names}, elapsed={time.monotonic() - run_started:.1f}s"
        )
    except Exception as error:
        log(f"heavy pipeline failed; retaining the early placeholder: {type(error).__name__}: {error}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    solve(public_dir, output_path)


if __name__ == "__main__":
    main()
