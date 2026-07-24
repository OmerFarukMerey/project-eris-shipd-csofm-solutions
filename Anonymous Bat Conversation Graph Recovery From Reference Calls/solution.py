#!/usr/bin/env python3
"""Train-only, reference-conditioned bat social graph recovery.

The script deliberately keeps test inference row-local.  Every predictive transform is
stateless or fitted on training rows; the only cross-call test operation is the required
serialization of model-produced calls into an episode graph.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.io import wavfile
import torch
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from torch import nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

SEED = 1729
N_JOBS = max(1, min(10, os.cpu_count() or 1))
N_TREES = 320
SEARCH_STOP_SECONDS = 2400.0
TRAIN_STOP_SECONDS = 3000.0
EPS = 1.0e-8
F1_DIM = 480
F2_DIM = 1374


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def parse_json_list(value: Any) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON list")
    return parsed


def nodes_for_row(row: Any) -> list[str]:
    return [str(x) for x in parse_json_list(row.node_set_json)]


def refs_for_row(row: Any) -> list[dict[str, Any]]:
    return [dict(x) for x in parse_json_list(row.reference_json)]


def safe_nodes(row: Any) -> list[str]:
    try:
        nodes = nodes_for_row(row)
        if nodes:
            return nodes
    except Exception:
        pass
    return ["Bat_A"]


def make_graph(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Losslessly serialize directed model predictions; do not change predictions."""
    edge_counts: Counter[tuple[str, str]] = Counter()
    context_counts: Counter[tuple[str, str, str]] = Counter()
    for call in calls:
        source = str(call["caller"])
        target = str(call["addressee"])
        if target == "UNKNOWN" or source == target:
            continue
        context = str(call["context"])
        edge_counts[(source, target)] += 1
        context_counts[(source, target, context)] += 1
    edges: list[dict[str, Any]] = []
    for source, target in sorted(edge_counts):
        contexts = {
            context: int(context_counts[(source, target, context)])
            for context in sorted(
                c for s, t, c in context_counts if s == source and t == target
            )
        }
        edges.append(
            {
                "source": source,
                "target": target,
                "count": int(edge_counts[(source, target)]),
                "contexts": contexts,
            }
        )
    return {"edges": edges}


def placeholder_frame(test: pd.DataFrame, context_labels: list[str]) -> pd.DataFrame:
    """Mandatory early valid artifact; always overwritten after successful inference."""
    fallback_context = (
        "UNKNOWN_CONTEXT" if "UNKNOWN_CONTEXT" in context_labels else context_labels[0]
    )
    rows: list[dict[str, Any]] = []
    for episode_id in test["episode_id"].drop_duplicates().tolist():
        episode = test.loc[test["episode_id"] == episode_id]
        calls: list[dict[str, Any]] = []
        for row in episode.itertuples(index=False):
            caller = safe_nodes(row)[0]
            calls.append(
                {
                    "call_id": str(row.call_id),
                    "caller": caller,
                    "addressee": "UNKNOWN",
                    "context": fallback_context,
                    "confidence": 0.0,
                }
            )
        rows.append(
            {
                "episode_id": str(episode_id),
                "call_predictions_json": compact_json(calls),
                "graph_json": compact_json({"edges": []}),
                "confidence": 0.0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["episode_id", "call_predictions_json", "graph_json", "confidence"],
    )


def _frames(x: np.ndarray, frame: int = 2048, hop: int = 256) -> np.ndarray:
    if len(x) < frame:
        return np.pad(x, (0, frame - len(x)))[None, :]
    starts = np.arange(0, len(x) - frame + 1, hop)
    return np.stack([x[start : start + frame] for start in starts])


def _quantile_summary(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.mean(values),
            np.std(values),
            *np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
        ],
        dtype=np.float32,
    )


def extract_audio_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Two complementary fixed log-spectral descriptors from one WAV read."""
    sample_rate, raw = wavfile.read(path)
    x = raw.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        raise ValueError(f"empty WAV: {path}")
    x -= np.mean(x)
    frame = 2048
    frames = _frames(x, frame=frame, hop=256)
    window = np.hanning(frame).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32) ** 2

    # Descriptor 1: broad full-Nyquist distribution and temporal modulation.
    spectrum_1 = spectrum[::2]
    edges_1 = np.linspace(1, spectrum_1.shape[1], 65, dtype=int)
    bands_1 = np.stack(
        [
            spectrum_1[:, edges_1[i] : max(edges_1[i] + 1, edges_1[i + 1])].mean(
                axis=1
            )
            for i in range(64)
        ],
        axis=1,
    )
    total_1 = bands_1.sum(axis=1) + 1.0e-12
    norm_1 = bands_1 / total_1[:, None]
    lognorm_1 = np.log(norm_1 + 1.0e-8)
    logenergy_1 = np.log(total_1 + 1.0e-12)
    band_stats_1 = np.concatenate(
        [
            lognorm_1.mean(axis=0),
            lognorm_1.std(axis=0),
            np.quantile(lognorm_1, 0.1, axis=0),
            np.quantile(lognorm_1, 0.5, axis=0),
            np.quantile(lognorm_1, 0.9, axis=0),
        ]
    )
    cep_1 = dct(lognorm_1, type=2, axis=1, norm="ortho")[:, 1:25]
    cep_stats_1 = np.concatenate(
        [
            cep_1.mean(axis=0),
            cep_1.std(axis=0),
            np.quantile(cep_1, 0.1, axis=0),
            np.quantile(cep_1, 0.9, axis=0),
        ]
    )
    freq_1 = (np.arange(64) + 0.5) * (sample_rate / 2.0 / 64.0)
    centroid_1 = (norm_1 * freq_1).sum(axis=1)
    bandwidth_1 = np.sqrt(
        (norm_1 * (freq_1[None, :] - centroid_1[:, None]) ** 2).sum(axis=1)
    )
    flatness_1 = np.exp(np.mean(np.log(bands_1 + 1.0e-12), axis=1)) / (
        np.mean(bands_1, axis=1) + 1.0e-12
    )
    temporal_1 = np.concatenate(
        [
            _quantile_summary(logenergy_1),
            _quantile_summary(centroid_1),
            _quantile_summary(bandwidth_1),
            _quantile_summary(flatness_1),
        ]
    )
    absolute = np.abs(x)
    wave = np.asarray(
        [
            len(x) / float(sample_rate),
            np.sqrt(np.mean(x * x) + 1.0e-12),
            np.max(absolute),
            np.mean(absolute),
            np.std(x),
            np.mean(np.diff(np.signbit(x)) != 0),
            *np.quantile(absolute, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]),
        ],
        dtype=np.float32,
    )
    centered_energy_1 = logenergy_1 - logenergy_1.mean()
    modulation_1 = np.abs(
        np.fft.rfft(centered_energy_1 * np.hanning(len(centered_energy_1)))
    ) ** 2
    modulation_1 /= modulation_1.sum() + 1.0e-12
    mod_edges_1 = np.linspace(0, len(modulation_1), 17, dtype=int)
    mod_bands_1 = np.asarray(
        [
            modulation_1[
                mod_edges_1[i] : max(mod_edges_1[i] + 1, mod_edges_1[i + 1])
            ].sum()
            for i in range(16)
        ],
        dtype=np.float32,
    )
    feature_1 = np.concatenate(
        [wave, band_stats_1, cep_stats_1, temporal_1, mod_bands_1]
    ).astype(np.float32)

    # Descriptor 2: denser low-frequency allocation plus vocal-active frame summaries.
    edge_hz = np.concatenate(
        [np.linspace(0.0, 25000.0, 65), np.linspace(25000.0, 50000.0, 33)[1:]]
    )
    bin_edges = np.clip(
        np.round(edge_hz / (sample_rate / frame)).astype(int), 0, spectrum.shape[1] - 1
    )
    bands_2 = np.stack(
        [
            spectrum[:, bin_edges[i] : max(bin_edges[i] + 1, bin_edges[i + 1])].mean(
                axis=1
            )
            for i in range(len(bin_edges) - 1)
        ],
        axis=1,
    )
    total_2 = bands_2.sum(axis=1) + 1.0e-12
    norm_2 = bands_2 / total_2[:, None]
    lognorm_2 = np.log(norm_2 + 1.0e-9)
    logenergy_2 = np.log(total_2 + 1.0e-12)
    summaries_2: list[np.ndarray] = [
        lognorm_2.mean(axis=0),
        lognorm_2.std(axis=0),
        np.quantile(lognorm_2, 0.1, axis=0),
        np.quantile(lognorm_2, 0.5, axis=0),
        np.quantile(lognorm_2, 0.9, axis=0),
    ]
    order = np.argsort(logenergy_2)
    for fraction in (0.1, 0.25, 0.5):
        count = max(1, int(math.ceil(len(order) * fraction)))
        active = order[-count:]
        summaries_2.extend(
            [lognorm_2[active].mean(axis=0), lognorm_2[active].std(axis=0)]
        )
    voice_2 = np.concatenate(summaries_2)
    cep_2 = dct(lognorm_2, type=2, axis=1, norm="ortho")[:, 1:33]
    top_quarter = order[-max(1, int(math.ceil(len(order) * 0.25))) :]
    cep_stats_2 = np.concatenate(
        [
            cep_2.mean(axis=0),
            cep_2.std(axis=0),
            np.quantile(cep_2, 0.1, axis=0),
            np.quantile(cep_2, 0.9, axis=0),
            cep_2[top_quarter].mean(axis=0),
            cep_2[top_quarter].std(axis=0),
        ]
    )
    freq_2 = (edge_hz[:-1] + edge_hz[1:]) / 2.0
    centroid_2 = (norm_2 * freq_2).sum(axis=1)
    bandwidth_2 = np.sqrt(
        (norm_2 * (freq_2[None, :] - centroid_2[:, None]) ** 2).sum(axis=1)
    )
    flatness_2 = np.exp(np.mean(np.log(bands_2 + 1.0e-12), axis=1)) / (
        np.mean(bands_2, axis=1) + 1.0e-12
    )
    peak_2 = freq_2[np.argmax(norm_2, axis=1)]
    cumulative_2 = np.cumsum(norm_2, axis=1)
    roll50_2 = freq_2[np.argmax(cumulative_2 >= 0.5, axis=1)]
    roll90_2 = freq_2[np.argmax(cumulative_2 >= 0.9, axis=1)]
    temporal_2 = np.concatenate(
        [
            _quantile_summary(logenergy_2),
            _quantile_summary(centroid_2),
            _quantile_summary(bandwidth_2),
            _quantile_summary(flatness_2),
            _quantile_summary(peak_2),
            _quantile_summary(roll50_2),
            _quantile_summary(roll90_2),
            _quantile_summary(centroid_2[top_quarter]),
            _quantile_summary(peak_2[top_quarter]),
            _quantile_summary(roll90_2[top_quarter]),
        ]
    )
    centered_energy_2 = logenergy_2 - logenergy_2.mean()
    modulation_2 = np.abs(
        np.fft.rfft(centered_energy_2 * np.hanning(len(centered_energy_2)))
    ) ** 2
    modulation_2 /= modulation_2.sum() + 1.0e-12
    mod_edges_2 = np.linspace(0, len(modulation_2), 25, dtype=int)
    mod_bands_2 = np.asarray(
        [
            modulation_2[
                mod_edges_2[i] : max(mod_edges_2[i] + 1, mod_edges_2[i + 1])
            ].sum()
            for i in range(24)
        ],
        dtype=np.float32,
    )
    feature_2 = np.concatenate(
        [wave, voice_2, cep_stats_2, temporal_2, mod_bands_2]
    ).astype(np.float32)

    feature_1 = np.nan_to_num(feature_1, nan=0.0, posinf=0.0, neginf=0.0)
    feature_2 = np.nan_to_num(feature_2, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_1.shape != (F1_DIM,) or feature_2.shape != (F2_DIM,):
        raise RuntimeError(
            f"unexpected feature dimensions {feature_1.shape}, {feature_2.shape}"
        )
    return feature_1, feature_2


def collect_audio_paths(frame: pd.DataFrame) -> list[str]:
    paths: dict[str, None] = {}
    for row in frame.itertuples(index=False):
        paths[str(row.audio_path)] = None
        try:
            for ref in refs_for_row(row):
                paths[str(ref["audio_path"])] = None
        except Exception:
            continue
    return list(paths)


def extract_cache(
    public_dir: Path, frame: pd.DataFrame, split_name: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature_1: dict[str, np.ndarray] = {}
    feature_2: dict[str, np.ndarray] = {}
    failures = 0
    for relative in collect_audio_paths(frame):
        try:
            first, second = extract_audio_features(public_dir / relative)
        except Exception as exc:
            failures += 1
            warnings.warn(f"{split_name} audio fallback for {relative}: {exc}")
            first = np.zeros(F1_DIM, dtype=np.float32)
            second = np.zeros(F2_DIM, dtype=np.float32)
        feature_1[relative] = first
        feature_2[relative] = second
    if failures:
        print(f"warning: {failures} {split_name} audio files used zero-vector fallbacks")
    return feature_1, feature_2


def reference_features(row: Any, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    by_node: dict[str, list[np.ndarray]] = defaultdict(list)
    try:
        references = refs_for_row(row)
    except Exception:
        references = []
    for ref in references:
        try:
            path = str(ref["audio_path"])
            by_node[str(ref["bat"])].append(
                cache.get(path, np.zeros(F1_DIM, np.float32))
            )
        except Exception:
            continue
    result: dict[str, np.ndarray] = {}
    for node in safe_nodes(row):
        values = by_node.get(node)
        if values:
            result[node] = np.stack(values)
        else:
            result[node] = np.zeros((1, F1_DIM), dtype=np.float32)
    return result


def build_pair_matrix(
    frame: pd.DataFrame, feature_1: dict[str, np.ndarray], labeled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[bool] = []
    row_ids: list[int] = []
    node_ids: list[str] = []
    groups: list[str] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        gallery = feature_1.get(str(row.audio_path), np.zeros(F1_DIM, np.float32))
        refs = reference_features(row, feature_1)
        for node in safe_nodes(row):
            ref_values = refs[node]
            distances = np.abs(ref_values - gallery)
            ref_mean = ref_values.mean(axis=0)
            values.append(
                np.concatenate(
                    [
                        np.abs(gallery - ref_mean),
                        distances.mean(axis=0),
                        distances.min(axis=0),
                        distances.max(axis=0),
                        [float(len(refs))],
                    ]
                ).astype(np.float32)
            )
            labels.append(bool(labeled and node == str(row.caller)))
            row_ids.append(row_index)
            node_ids.append(node)
            groups.append(str(row.episode_id))
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(row_ids, dtype=np.int32),
        np.asarray(node_ids, dtype=object),
        np.asarray(groups, dtype=object),
    )


def extra_classifier(
    leaf: int, class_weight: str | None = "balanced", seed_offset: int = 0
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=N_TREES,
        min_samples_leaf=int(leaf),
        max_features="sqrt",
        class_weight=class_weight,
        random_state=SEED + seed_offset,
        n_jobs=N_JOBS,
    )


def positive_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(values), dtype=np.float64)
    return probabilities[:, classes.index(1)].astype(np.float64)


def aligned_probability(
    model: ExtraTreesClassifier, values: np.ndarray, class_count: int
) -> np.ndarray:
    raw = model.predict_proba(values)
    result = np.zeros((len(values), class_count), dtype=np.float64)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    return result


def split_count(groups: np.ndarray) -> int:
    return max(2, min(6, len(np.unique(groups))))


def candidate_weights(count: int, steps: int = 4) -> list[np.ndarray]:
    """Generic simplex grid used for in-script ensemble search."""
    result: list[np.ndarray] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            result.append(np.asarray(prefix + [remaining], dtype=float) / steps)
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value, slots - 1)

    visit([], steps, count)
    return result

def normalize_group_scores(
    scores: np.ndarray,
    row_ids: np.ndarray,
    row_count: int,
    temperature: float | None,
) -> np.ndarray:
    """Normalize alternative hypotheses within one gallery call only."""
    result = np.zeros(len(scores), dtype=np.float64)
    for row_index in range(row_count):
        indices = np.flatnonzero(row_ids == row_index)
        values = np.asarray(scores[indices], dtype=np.float64)
        if temperature is None:
            values = np.clip(values, EPS, None)
        else:
            values = values / max(float(temperature), EPS)
            values = np.exp(values - values.max())
        result[indices] = values / values.sum()
    return result


def query_group_sizes(row_ids: np.ndarray) -> np.ndarray:
    """Candidate counts for contiguous query groups used by learning-to-rank."""
    return np.unique(row_ids, return_counts=True)[1].astype(np.int32)


def acoustic_condition_map(
    frame: pd.DataFrame, feature_1: dict[str, np.ndarray]
) -> dict[str, int]:
    """Train-only episode clusters used to expose recording-condition shift in OOF."""
    episode_ids: list[str] = []
    descriptors: list[np.ndarray] = []
    for episode_id, episode in frame.groupby("episode_id", sort=False):
        first_row = next(episode.itertuples(index=False))
        paths = [str(value) for value in episode["audio_path"]]
        paths.extend(str(ref["audio_path"]) for ref in refs_for_row(first_row))
        values = np.stack(
            [
                feature_1.get(path, np.zeros(F1_DIM, dtype=np.float32))
                for path in dict.fromkeys(paths)
            ]
        )
        episode_ids.append(str(episode_id))
        descriptors.append(
            np.concatenate([values.mean(axis=0), values.std(axis=0)])
        )
    matrix = np.stack(descriptors).astype(np.float64)
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1.0e-4)
    component_count = max(2, min(12, len(matrix) - 1))
    reduced = PCA(
        n_components=component_count, svd_solver="full", random_state=SEED
    ).fit_transform(matrix)
    cluster_count = max(2, min(12, len(matrix) // 2))
    labels = KMeans(
        n_clusters=cluster_count,
        n_init=30,
        random_state=SEED,
    ).fit_predict(reduced)
    if len(np.unique(labels)) < 2:
        fallback_count = max(2, min(6, len(episode_ids)))
        labels = np.arange(len(episode_ids), dtype=np.int32) % fallback_count
    return {
        episode_id: int(label)
        for episode_id, label in zip(episode_ids, labels)
    }


def lgb_ranker(depth: int, seed_offset: int) -> LGBMRanker:
    leaves = 7 if depth <= 3 else 15
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=leaves,
        max_depth=depth,
        min_child_samples=15,
        reg_lambda=7.0,
        verbosity=-1,
        n_jobs=N_JOBS,
        random_state=SEED + seed_offset,
    )


def lgb_context_classifier(seed_offset: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=12,
        reg_lambda=5.0,
        class_weight="balanced",
        verbosity=-1,
        n_jobs=N_JOBS,
        random_state=SEED + seed_offset,
    )


class ReferenceMetricNet(nn.Module):
    """Shared train-from-scratch encoder scored against row-local references."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def embed(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.encoder(values), dim=-1)

    def forward(
        self, gallery: torch.Tensor, references: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        gallery_embedding = self.embed(gallery)
        reference_embedding = nn.functional.normalize(
            self.embed(references).mean(dim=2), dim=-1
        )
        scores = torch.einsum(
            "bd,bnd->bn", gallery_embedding, reference_embedding
        ) * 30.0
        return scores.masked_fill(~mask, -1.0e4)


def build_metric_tensors(
    frame: pd.DataFrame,
    feature_1: dict[str, np.ndarray],
    labeled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[str]]]:
    gallery = np.stack(
        [
            feature_1.get(str(row.audio_path), np.zeros(F1_DIM, np.float32))
            for row in frame.itertuples(index=False)
        ]
    ).astype(np.float32)
    references = np.zeros((len(frame), 4, 2, F1_DIM), dtype=np.float32)
    mask = np.zeros((len(frame), 4), dtype=bool)
    targets = np.zeros(len(frame), dtype=np.int64)
    node_orders: list[list[str]] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        nodes = safe_nodes(row)
        node_orders.append(nodes)
        by_node = reference_features(row, feature_1)
        for node_index, node in enumerate(nodes[:4]):
            values = by_node[node]
            if len(values) == 1:
                values = np.repeat(values, 2, axis=0)
            references[row_index, node_index] = values[:2]
            mask[row_index, node_index] = True
            if labeled and node == str(row.caller):
                targets[row_index] = node_index
    return gallery, references, mask, targets, node_orders


def fit_reference_metric(
    gallery: np.ndarray,
    references: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    epochs: int,
    hidden_dim: int,
    embedding_dim: int,
    weight_decay: float,
    seed: int,
) -> tuple[ReferenceMetricNet, np.ndarray, np.ndarray]:
    reference_mask = np.repeat(mask[indices, :, None], 2, axis=2)
    normalization_values = np.concatenate(
        [gallery[indices], references[indices][reference_mask]], axis=0
    )
    mean = normalization_values.mean(axis=0)
    scale = normalization_values.std(axis=0) + 1.0e-3
    gallery_tensor = torch.from_numpy((gallery[indices] - mean) / scale)
    reference_tensor = torch.from_numpy((references[indices] - mean) / scale)
    mask_tensor = torch.from_numpy(mask[indices])
    target_tensor = torch.from_numpy(targets[indices])
    torch.manual_seed(seed)
    model = ReferenceMetricNet(F1_DIM, hidden_dim, embedding_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=weight_decay
    )
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(gallery_tensor, reference_tensor, mask_tensor)
        loss = nn.functional.cross_entropy(logits, target_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model, mean.astype(np.float32), scale.astype(np.float32)


def predict_reference_metric(
    model: ReferenceMetricNet,
    mean: np.ndarray,
    scale: np.ndarray,
    gallery: np.ndarray,
    references: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(
            torch.from_numpy((gallery - mean) / scale),
            torch.from_numpy((references - mean) / scale),
            torch.from_numpy(mask),
        ).numpy()


def metric_oof(
    gallery: np.ndarray,
    references: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    config: tuple[int, int, int, float],
    seed_offset: int,
) -> np.ndarray:
    epochs, hidden_dim, embedding_dim, weight_decay = config
    result = np.full((len(gallery), 4), -1.0e4, dtype=np.float32)
    splitter = GroupKFold(split_count(groups))
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(gallery, targets, groups)
    ):
        model, mean, scale = fit_reference_metric(
            gallery,
            references,
            mask,
            targets,
            train_index,
            epochs,
            hidden_dim,
            embedding_dim,
            weight_decay,
            SEED + seed_offset + fold,
        )
        result[valid_index] = predict_reference_metric(
            model,
            mean,
            scale,
            gallery[valid_index],
            references[valid_index],
            mask[valid_index],
        )
    return result


def metric_to_pair_scores(
    metric_scores: np.ndarray, pair_rows: np.ndarray, row_count: int
) -> np.ndarray:
    result = np.zeros(len(pair_rows), dtype=np.float64)
    for row_index in range(row_count):
        indices = np.flatnonzero(pair_rows == row_index)
        logits = metric_scores[row_index, : len(indices)].astype(np.float64)
        probabilities = np.exp(logits - logits.max())
        result[indices] = probabilities / probabilities.sum()
    return result


def row_candidate_probabilities(
    row_ids: np.ndarray,
    node_ids: np.ndarray,
    scores: np.ndarray,
    row_count: int,
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row_index in range(row_count):
        indices = np.flatnonzero(row_ids == row_index)
        values = np.clip(scores[indices], EPS, None)
        values /= values.sum()
        result.append(
            {str(node_ids[index]): float(value) for index, value in zip(indices, values)}
        )
    return result


def context_matrix(
    frame: pd.DataFrame, feature_2: dict[str, np.ndarray]
) -> np.ndarray:
    return np.stack(
        [
            feature_2.get(str(row.audio_path), np.zeros(F2_DIM, np.float32))
            for row in frame.itertuples(index=False)
        ]
    ).astype(np.float32)


def node_counts(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [len(safe_nodes(row)) for row in frame.itertuples(index=False)],
        dtype=np.float32,
    )


def build_joint_matrix(
    frame: pd.DataFrame,
    feature_1: dict[str, np.ndarray],
    caller_probabilities: list[dict[str, float]],
    context_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    labeled: bool,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    values: list[np.ndarray] = []
    labels: list[bool] = []
    row_ids: list[int] = []
    callers: list[str] = []
    targets: list[str] = []
    groups: list[str] = []
    zero_target = np.zeros(F1_DIM * 4, dtype=np.float32)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        gallery = feature_1.get(str(row.audio_path), np.zeros(F1_DIM, np.float32))
        refs = reference_features(row, feature_1)
        stats = {
            node: (ref_values.mean(axis=0), ref_values.std(axis=0))
            for node, ref_values in refs.items()
        }
        nodes = safe_nodes(row)
        for caller in nodes:
            caller_mean, caller_std = stats[caller]
            caller_base = np.concatenate(
                [gallery, caller_mean, caller_std, np.abs(gallery - caller_mean)]
            )
            possible_targets = ["UNKNOWN"] + [node for node in nodes if node != caller]
            for target in possible_targets:
                if target == "UNKNOWN":
                    target_base = zero_target
                else:
                    target_mean, target_std = stats[target]
                    target_base = np.concatenate(
                        [
                            target_mean,
                            target_std,
                            np.abs(gallery - target_mean),
                            np.abs(caller_mean - target_mean),
                        ]
                    )
                values.append(
                    np.concatenate(
                        [
                            caller_base,
                            target_base,
                            context_probabilities[row_index],
                            [
                                caller_probabilities[row_index].get(caller, EPS),
                                float(unknown_probabilities[row_index]),
                                float(target == "UNKNOWN"),
                                float(len(nodes)),
                            ],
                        ]
                    ).astype(np.float32)
                )
                labels.append(
                    bool(
                        labeled
                        and caller == str(row.caller)
                        and target == str(row.addressee)
                    )
                )
                row_ids.append(row_index)
                callers.append(caller)
                targets.append(target)
                groups.append(str(row.episode_id))
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(row_ids, dtype=np.int32),
        np.asarray(callers, dtype=object),
        np.asarray(targets, dtype=object),
        np.asarray(groups, dtype=object),
    )


def decode_social(
    joint_scores: np.ndarray,
    joint_weight: float,
    row_ids: np.ndarray,
    candidate_callers: np.ndarray,
    candidate_targets: np.ndarray,
    caller_probabilities: list[dict[str, float]],
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    callers: list[str] = []
    targets: list[str] = []
    tops: list[float] = []
    margins: list[float] = []
    chosen_joint: list[float] = []
    caller_weight = 1.0 - joint_weight
    for row_index in range(row_count):
        indices = np.flatnonzero(row_ids == row_index)
        logits = np.asarray(
            [
                joint_weight * math.log(max(float(joint_scores[index]), EPS))
                + caller_weight
                * math.log(
                    max(
                        caller_probabilities[row_index].get(
                            str(candidate_callers[index]), EPS
                        ),
                        EPS,
                    )
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        shifted = logits - logits.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        order = np.argsort(probabilities)
        local = int(order[-1])
        selected = indices[local]
        callers.append(str(candidate_callers[selected]))
        targets.append(str(candidate_targets[selected]))
        tops.append(float(probabilities[local]))
        margins.append(
            float(probabilities[order[-1]] - probabilities[order[-2]])
            if len(order) > 1
            else float(probabilities[order[-1]])
        )
        chosen_joint.append(float(joint_scores[selected]))
    return (
        np.asarray(callers, dtype=object),
        np.asarray(targets, dtype=object),
        np.asarray(tops, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
        np.asarray(chosen_joint, dtype=np.float64),
    )


def count_f1(predicted: Counter[Any], truth: Counter[Any]) -> float:
    predicted_total = sum(predicted.values())
    truth_total = sum(truth.values())
    if predicted_total + truth_total == 0:
        return 1.0
    overlap = sum(
        min(predicted_count, truth.get(key, 0))
        for key, predicted_count in predicted.items()
    )
    return 2.0 * overlap / (predicted_total + truth_total)


def episode_components(
    train: pd.DataFrame,
    predicted_callers: np.ndarray,
    predicted_targets: np.ndarray,
    predicted_contexts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_id, raw_indices in train.groupby("episode_id", sort=False).groups.items():
        indices = np.asarray(list(raw_indices), dtype=int)
        truth = train.iloc[indices]
        caller_accuracy = float(
            np.mean(predicted_callers[indices] == truth["caller"].to_numpy())
        )
        target_accuracy = float(
            np.mean(predicted_targets[indices] == truth["addressee"].to_numpy())
        )
        context_accuracy = float(
            np.mean(predicted_contexts[indices] == truth["context"].to_numpy())
        )
        call_core = (
            0.45 * caller_accuracy
            + 0.25 * target_accuracy
            + 0.30 * context_accuracy
        )
        predicted_edges: Counter[Any] = Counter(
            (predicted_callers[index], predicted_targets[index])
            for index in indices
            if predicted_targets[index] != "UNKNOWN"
        )
        truth_edges: Counter[Any] = Counter(
            (train.iloc[index]["caller"], train.iloc[index]["addressee"])
            for index in indices
            if train.iloc[index]["addressee"] != "UNKNOWN"
        )
        predicted_edge_contexts: Counter[Any] = Counter(
            (
                predicted_callers[index],
                predicted_targets[index],
                predicted_contexts[index],
            )
            for index in indices
            if predicted_targets[index] != "UNKNOWN"
        )
        truth_edge_contexts: Counter[Any] = Counter(
            (
                train.iloc[index]["caller"],
                train.iloc[index]["addressee"],
                train.iloc[index]["context"],
            )
            for index in indices
            if train.iloc[index]["addressee"] != "UNKNOWN"
        )
        graph_score = 0.65 * count_f1(
            predicted_edges, truth_edges
        ) + 0.35 * count_f1(predicted_edge_contexts, truth_edge_contexts)
        core = 0.80 * call_core + 0.15 * graph_score + 0.05
        rows.append(
            {
                "episode_id": str(episode_id),
                "core": float(core),
                "call_core": float(call_core),
                "graph_score": float(graph_score),
                "caller_accuracy": caller_accuracy,
                "addressee_accuracy": target_accuracy,
                "context_accuracy": context_accuracy,
            }
        )
    return pd.DataFrame(rows)


def proxy_final_score(
    components: pd.DataFrame, train: pd.DataFrame, score_column: str = "core"
) -> float:
    episode_meta: list[dict[str, Any]] = []
    for episode_id, episode in train.groupby("episode_id", sort=False):
        contexts = episode["context"].value_counts()
        episode_meta.append(
            {
                "episode_id": str(episode_id),
                "node_count": len(nodes_for_row(next(episode.itertuples(index=False)))),
                "unknown_present": bool((episode["addressee"] == "UNKNOWN").any()),
                "dominant_context": str(contexts.index[0]),
            }
        )
    scored = components.merge(pd.DataFrame(episode_meta), on="episode_id", how="left")
    worst_axis_means: list[float] = []
    for axis in ("node_count", "unknown_present", "dominant_context"):
        worst_axis_means.append(float(scored.groupby(axis)[score_column].mean().min()))
    return float(
        0.78 * scored[score_column].mean() + 0.22 * np.mean(worst_axis_means)
    )



def social_posteriors(
    joint_scores: np.ndarray,
    joint_weight: float,
    row_ids: np.ndarray,
    candidate_callers: np.ndarray,
    candidate_targets: np.ndarray,
    caller_probabilities: list[dict[str, float]],
    row_count: int,
) -> list[dict[tuple[str, str], float]]:
    """Complete row-local social posterior used by the learned graph head."""
    result: list[dict[tuple[str, str], float]] = []
    for row_index in range(row_count):
        indices = np.flatnonzero(row_ids == row_index)
        logits = np.asarray(
            [
                joint_weight * math.log(max(float(joint_scores[index]), EPS))
                + (1.0 - joint_weight)
                * math.log(
                    max(
                        caller_probabilities[row_index].get(
                            str(candidate_callers[index]), EPS
                        ),
                        EPS,
                    )
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        result.append(
            {
                (str(candidate_callers[index]), str(candidate_targets[index])): float(
                    probability
                )
                for index, probability in zip(indices, probabilities)
            }
        )
    return result


def episode_rate_matrix(
    frame: pd.DataFrame,
    posteriors: list[dict[tuple[str, str], float]],
    context_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    labeled: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Permutation-invariant episode features for known-addressee count regression."""
    episode_ids: list[str] = []
    values: list[np.ndarray] = []
    targets: list[float] = []
    for episode_id, raw_indices in frame.groupby(
        "episode_id", sort=False
    ).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int32)
        first_row = next(frame.iloc[indices].itertuples(index=False))
        reference_durations = np.asarray(
            [
                float(ref.get("duration_sec", 0.0))
                for ref in refs_for_row(first_row)
            ],
            dtype=np.float64,
        )
        if len(reference_durations) == 0:
            reference_durations = np.zeros(1, dtype=np.float64)
        gallery_durations = frame.iloc[indices]["clip_duration_sec"].to_numpy(
            dtype=np.float64
        )
        joint_unknown = np.asarray(
            [
                sum(
                    probability
                    for (_, target), probability in posteriors[index].items()
                    if target == "UNKNOWN"
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        unknown = np.asarray(unknown_probabilities[indices], dtype=np.float64)
        values.append(
            np.concatenate(
                [
                    np.asarray(
                        [
                            len(safe_nodes(first_row)),
                            len(indices),
                            gallery_durations.mean(),
                            gallery_durations.std(),
                            reference_durations.mean(),
                            reference_durations.std(),
                            joint_unknown.mean(),
                            joint_unknown.std(),
                            unknown.mean(),
                            unknown.std(),
                        ],
                        dtype=np.float64,
                    ),
                    np.quantile(joint_unknown, [0.1, 0.25, 0.5, 0.75, 0.9]),
                    np.quantile(unknown, [0.1, 0.25, 0.5, 0.75, 0.9]),
                    context_probabilities[indices].mean(axis=0),
                ]
            )
        )
        episode_ids.append(str(episode_id))
        if labeled:
            targets.append(
                float(
                    np.mean(
                        frame.iloc[indices]["addressee"].astype(str).to_numpy()
                        != "UNKNOWN"
                    )
                )
            )
    return (
        episode_ids,
        np.stack(values).astype(np.float64),
        np.asarray(targets, dtype=np.float64),
    )


def fit_episode_rate_head(
    values: np.ndarray, targets: np.ndarray, groups: np.ndarray
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]:
    """Cross-fit and train a low-variance episode known-call regressor."""
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in (1.0, 10.0, 100.0):
        predictions = np.zeros(len(targets), dtype=np.float64)
        splitter = GroupKFold(split_count(groups))
        for train_index, valid_index in splitter.split(values, targets, groups):
            mean = values[train_index].mean(axis=0)
            scale = values[train_index].std(axis=0) + 1.0e-4
            model = Ridge(alpha=alpha)
            model.fit(
                (values[train_index] - mean) / scale, targets[train_index]
            )
            predictions[valid_index] = np.clip(
                model.predict((values[valid_index] - mean) / scale), 0.0, 1.0
            )
        loss = mean_absolute_error(targets, predictions)
        if best is None or loss < best[0]:
            best = (float(loss), alpha, predictions)
    assert best is not None
    mean = values.mean(axis=0)
    scale = values.std(axis=0) + 1.0e-4
    final_model = Ridge(alpha=best[1])
    final_model.fit((values - mean) / scale, targets)
    return final_model, mean, scale, best[2], best[1]


def edge_head_matrix(
    frame: pd.DataFrame,
    feature_1: dict[str, np.ndarray],
    posteriors: list[dict[tuple[str, str], float]],
    caller_probabilities: list[dict[str, float]],
    context_probabilities: np.ndarray,
    episode_ids: list[str],
    rate_hints: np.ndarray,
    labeled: bool,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, str]]]:
    """One equivariant training row per possible directed episode edge."""
    rate_by_episode = {
        episode_id: float(rate)
        for episode_id, rate in zip(episode_ids, rate_hints)
    }
    values: list[np.ndarray] = []
    targets: list[float] = []
    keys: list[tuple[str, str, str]] = []
    for episode_id, raw_indices in frame.groupby(
        "episode_id", sort=False
    ).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int32)
        episode = frame.iloc[indices]
        first_row = next(episode.itertuples(index=False))
        nodes = safe_nodes(first_row)
        references = reference_features(first_row, feature_1)
        reference_means = {
            node: references[node].mean(axis=0) for node in nodes
        }
        context_mean = context_probabilities[indices].mean(axis=0)
        truth: Counter[Any] = Counter()
        if labeled:
            truth = Counter(
                (str(row.caller), str(row.addressee))
                for row in episode.itertuples(index=False)
                if str(row.addressee) != "UNKNOWN"
            )
        for source in nodes:
            for target in nodes:
                if source == target:
                    continue
                edge = (source, target)
                probabilities = np.asarray(
                    [posteriors[index].get(edge, 0.0) for index in indices],
                    dtype=np.float64,
                )
                source_probabilities = np.asarray(
                    [
                        caller_probabilities[index].get(source, 0.0)
                        for index in indices
                    ],
                    dtype=np.float64,
                )
                target_probabilities = np.asarray(
                    [
                        sum(
                            probability
                            for (_, candidate_target), probability in posteriors[
                                index
                            ].items()
                            if candidate_target == target
                        )
                        for index in indices
                    ],
                    dtype=np.float64,
                )
                hard_count = sum(
                    max(posteriors[index], key=posteriors[index].get) == edge
                    for index in indices
                )
                reference_distance = np.abs(
                    reference_means[source] - reference_means[target]
                )
                values.append(
                    np.concatenate(
                        [
                            np.asarray(
                                [
                                    len(nodes),
                                    len(indices),
                                    rate_by_episode[str(episode_id)],
                                    probabilities.sum(),
                                    probabilities.mean(),
                                    probabilities.std(),
                                    probabilities.max(),
                                    np.quantile(probabilities, 0.75),
                                    np.quantile(probabilities, 0.90),
                                    hard_count,
                                    source_probabilities.sum(),
                                    source_probabilities.mean(),
                                    source_probabilities.max(),
                                    target_probabilities.sum(),
                                    target_probabilities.mean(),
                                    target_probabilities.max(),
                                    reference_distance.mean(),
                                    reference_distance.std(),
                                ],
                                dtype=np.float64,
                            ),
                            reference_distance[:12],
                            context_mean,
                        ]
                    )
                )
                keys.append((str(episode_id), source, target))
                if labeled:
                    targets.append(float(truth[edge]))
    return (
        np.stack(values).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        keys,
    )


def make_edge_regressor(kind: str, parameter: int, seed: int) -> Any:
    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=parameter,
            max_features=0.8,
            random_state=SEED + seed,
            n_jobs=N_JOBS,
        )
    return LGBMRegressor(
        objective="poisson",
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=12,
        reg_lambda=5.0,
        verbosity=-1,
        n_jobs=N_JOBS,
        random_state=SEED + seed,
    )


def crossfit_edge_heads(
    values: np.ndarray, targets: np.ndarray, groups: np.ndarray
) -> tuple[list[tuple[str, int]], list[np.ndarray]]:
    specs = [("extra", 3), ("lgb", 3)]
    predictions: list[np.ndarray] = []
    for spec_index, (kind, parameter) in enumerate(specs):
        oof = np.zeros(len(targets), dtype=np.float64)
        splitter = GroupKFold(split_count(groups))
        for train_index, valid_index in splitter.split(values, targets, groups):
            model = make_edge_regressor(kind, parameter, 300 + spec_index)
            model.fit(values[train_index], targets[train_index])
            oof[valid_index] = np.clip(
                model.predict(values[valid_index]), 0.0, None
            )
        predictions.append(oof)
    return specs, predictions


def quota_decode(
    frame: pd.DataFrame,
    posteriors: list[dict[tuple[str, str], float]],
    episode_ids: list[str],
    known_rates: np.ndarray,
    edge_keys: list[tuple[str, str, str]],
    edge_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Globally assign one episode's calls to counts predicted by the graph head."""
    rate_by_episode = {
        episode_id: float(rate)
        for episode_id, rate in zip(episode_ids, known_rates)
    }
    edge_by_key = {
        key: max(float(value), 1.0e-4)
        for key, value in zip(edge_keys, edge_predictions)
    }
    callers = np.empty(len(frame), dtype=object)
    targets = np.empty(len(frame), dtype=object)
    tops = np.zeros(len(frame), dtype=np.float64)
    margins = np.zeros(len(frame), dtype=np.float64)
    chosen = np.zeros(len(frame), dtype=np.float64)
    for episode_id, raw_indices in frame.groupby(
        "episode_id", sort=False
    ).groups.items():
        indices = np.asarray(list(raw_indices), dtype=np.int32)
        first_row = next(frame.iloc[indices].itertuples(index=False))
        nodes = safe_nodes(first_row)
        edges = [
            (source, target)
            for source in nodes
            for target in nodes
            if source != target
        ]
        known_count = int(
            np.clip(
                round(rate_by_episode[str(episode_id)] * len(indices)),
                0,
                len(indices),
            )
        )
        weights = np.asarray(
            [
                edge_by_key[(str(episode_id), source, target)]
                for source, target in edges
            ],
            dtype=np.float64,
        )
        raw_counts = known_count * weights / weights.sum()
        counts = np.floor(raw_counts).astype(np.int32)
        remainder = known_count - int(counts.sum())
        if remainder:
            order = np.argsort(-(raw_counts - counts))
            counts[order[:remainder]] += 1
        slots: list[tuple[str, str]] = []
        for edge, count in zip(edges, counts):
            slots.extend([edge] * int(count))
        slots.extend(
            [("UNKNOWN", str(index)) for index in range(len(indices) - len(slots))]
        )
        costs = np.zeros((len(indices), len(slots)), dtype=np.float64)
        for local_row, row_index in enumerate(indices):
            unknown_score = max(
                probability
                for (_, target), probability in posteriors[row_index].items()
                if target == "UNKNOWN"
            )
            for slot_index, slot in enumerate(slots):
                score = (
                    unknown_score
                    if slot[0] == "UNKNOWN"
                    else posteriors[row_index].get(slot, EPS)
                )
                costs[local_row, slot_index] = -math.log(max(score, EPS))
        assigned_rows, assigned_slots = linear_sum_assignment(costs)
        for local_row, slot_index in zip(assigned_rows, assigned_slots):
            row_index = int(indices[local_row])
            slot = slots[slot_index]
            if slot[0] == "UNKNOWN":
                candidates = [
                    (probability, source)
                    for (source, target), probability in posteriors[
                        row_index
                    ].items()
                    if target == "UNKNOWN"
                ]
                selected_probability, selected_caller = max(candidates)
                selected_target = "UNKNOWN"
            else:
                selected_caller, selected_target = slot
                selected_probability = posteriors[row_index].get(slot, EPS)
            alternatives = sorted(posteriors[row_index].values())
            second = alternatives[-2] if len(alternatives) > 1 else 0.0
            callers[row_index] = selected_caller
            targets[row_index] = selected_target
            tops[row_index] = selected_probability
            margins[row_index] = max(0.0, selected_probability - second)
            chosen[row_index] = selected_probability
    return callers, targets, tops, margins, chosen


def context_decision(
    probabilities: np.ndarray, biases: np.ndarray, context_labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.log(np.clip(probabilities, EPS, None)) + biases[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    order = np.argsort(calibrated, axis=1)
    top_indices = order[:, -1]
    labels = np.asarray([context_labels[index] for index in top_indices], dtype=object)
    tops = calibrated[np.arange(len(calibrated)), top_indices]
    margins = tops - calibrated[np.arange(len(calibrated)), order[:, -2]]
    return labels, tops, margins


def caller_summary(
    probabilities: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    tops: list[float] = []
    margins: list[float] = []
    for row in probabilities:
        values = sorted(row.values())
        tops.append(float(values[-1]))
        margins.append(float(values[-1] - values[-2]) if len(values) > 1 else values[-1])
    return np.asarray(tops), np.asarray(margins)


def call_meta_features(
    social_top: np.ndarray,
    social_margin: np.ndarray,
    caller_probabilities: list[dict[str, float]],
    context_top: np.ndarray,
    context_margin: np.ndarray,
    unknown_probability: np.ndarray,
    counts: np.ndarray,
    predicted_targets: np.ndarray,
    chosen_joint: np.ndarray,
) -> np.ndarray:
    caller_top, caller_margin = caller_summary(caller_probabilities)
    return np.column_stack(
        [
            social_top,
            social_margin,
            caller_top,
            caller_margin,
            context_top,
            context_margin,
            unknown_probability,
            counts,
            predicted_targets == "UNKNOWN",
            chosen_joint,
        ]
    ).astype(np.float64)


def crossfit_call_calibrator(
    values: np.ndarray, targets: np.ndarray, groups: np.ndarray
) -> tuple[Any, np.ndarray, int]:
    leaves = (2, 4, 8, 16)
    best: tuple[float, int, np.ndarray] | None = None
    for leaf in leaves:
        predictions = np.zeros(len(targets), dtype=np.float64)
        splitter = GroupKFold(split_count(groups))
        for train_index, valid_index in splitter.split(values, targets, groups):
            model = ExtraTreesRegressor(
                n_estimators=240,
                min_samples_leaf=leaf,
                max_features=1.0,
                random_state=SEED + 210 + leaf,
                n_jobs=N_JOBS,
            )
            model.fit(values[train_index], targets[train_index])
            predictions[valid_index] = np.clip(
                model.predict(values[valid_index]), 0.0, 1.0
            )
        loss = mean_absolute_error(targets, predictions)
        if best is None or loss < best[0]:
            best = (float(loss), leaf, predictions)
    assert best is not None
    final_model = ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=best[1],
        max_features=1.0,
        random_state=SEED + 230 + best[1],
        n_jobs=N_JOBS,
    )
    final_model.fit(values, targets)
    return final_model, best[2], best[1]


def row_calibration_features(row: Any) -> np.ndarray:
    try:
        refs = refs_for_row(row)
        durations = np.asarray(
            [float(ref.get("duration_sec", 0.0)) for ref in refs], dtype=np.float64
        )
        if len(durations) == 0:
            durations = np.zeros(1, dtype=np.float64)
        return np.asarray(
            [
                float(len(nodes_for_row(row))),
                float(np.mean(durations)),
                float(np.std(durations)),
                float(np.min(durations)),
                float(np.max(durations)),
            ],
            dtype=np.float64,
        )
    except Exception:
        return np.zeros(5, dtype=np.float64)


def fit_row_calibrator(
    train: pd.DataFrame, components: pd.DataFrame
) -> tuple[Any, np.ndarray, str]:
    episode_rows = [
        next(episode.itertuples(index=False))
        for _, episode in train.groupby("episode_id", sort=False)
    ]
    values = np.stack([row_calibration_features(row) for row in episode_rows])
    targets = components["core"].to_numpy(dtype=np.float64)
    splitter = KFold(
        n_splits=max(2, min(6, len(targets))), shuffle=True, random_state=SEED
    )
    candidate_names = ["mean", "extra_2", "extra_4", "extra_8"]
    best: tuple[float, str, np.ndarray] | None = None
    for name in candidate_names:
        predictions = np.zeros(len(targets), dtype=np.float64)
        for train_index, valid_index in splitter.split(values):
            if name == "mean":
                model: Any = DummyRegressor(strategy="mean")
            else:
                leaf = int(name.split("_")[1])
                model = ExtraTreesRegressor(
                    n_estimators=240,
                    min_samples_leaf=leaf,
                    max_features=1.0,
                    random_state=SEED + 250 + leaf,
                    n_jobs=N_JOBS,
                )
            model.fit(values[train_index], targets[train_index])
            predictions[valid_index] = np.clip(
                model.predict(values[valid_index]), 0.0, 1.0
            )
        loss = mean_absolute_error(targets, predictions)
        if best is None or loss < best[0]:
            best = (float(loss), name, predictions)
    assert best is not None
    if best[1] == "mean":
        final_model: Any = DummyRegressor(strategy="mean")
    else:
        leaf = int(best[1].split("_")[1])
        final_model = ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=SEED + 270 + leaf,
            n_jobs=N_JOBS,
        )
    final_model.fit(values, targets)
    return final_model, best[2], best[1]


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    context_labels: set[str],
) -> list[str]:
    errors: list[str] = []
    expected_columns = [
        "episode_id",
        "call_predictions_json",
        "graph_json",
        "confidence",
    ]
    if submission.columns.tolist() != expected_columns:
        errors.append("incorrect submission columns")
        return errors
    expected_episodes = test["episode_id"].drop_duplicates().astype(str).tolist()
    if submission["episode_id"].astype(str).tolist() != expected_episodes:
        errors.append("episode IDs or order differ from test")
    if len(submission) != len(expected_episodes):
        errors.append("incorrect episode row count")
    for output_row in submission.itertuples(index=False):
        episode = test.loc[test["episode_id"].astype(str) == str(output_row.episode_id)]
        if episode.empty:
            errors.append(f"unknown episode {output_row.episode_id}")
            continue
        nodes = set(safe_nodes(next(episode.itertuples(index=False))))
        expected_calls = set(episode["call_id"].astype(str))
        try:
            calls = json.loads(output_row.call_predictions_json)
            graph = json.loads(output_row.graph_json)
        except Exception as exc:
            errors.append(f"unreadable JSON in {output_row.episode_id}: {exc}")
            continue
        if not isinstance(calls, list):
            errors.append(f"call list is not a list in {output_row.episode_id}")
            continue
        if len(calls) != len(episode):
            errors.append(f"call count mismatch in {output_row.episode_id}")
        if {str(call.get("call_id")) for call in calls} != expected_calls:
            errors.append(f"call ID mismatch in {output_row.episode_id}")
        for call in calls:
            if set(call) != {"call_id", "caller", "addressee", "context", "confidence"}:
                errors.append(f"call schema mismatch in {output_row.episode_id}")
                break
            confidence = float(call["confidence"])
            if (
                call["caller"] not in nodes
                or call["addressee"] not in nodes | {"UNKNOWN"}
                or call["addressee"] == call["caller"]
                or call["context"] not in context_labels
                or not np.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                errors.append(f"invalid call value in {output_row.episode_id}")
                break
        if graph != make_graph(calls):
            errors.append(f"graph/call inconsistency in {output_row.episode_id}")
        confidence = float(output_row.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            errors.append(f"invalid row confidence in {output_row.episode_id}")
    return errors


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    started = time.monotonic()
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    required = [public_dir / "test.csv", public_dir / "train.csv", public_dir / "taxonomy.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))

    test = pd.read_csv(public_dir / "test.csv").reset_index(drop=True)
    taxonomy = json.loads((public_dir / "taxonomy.json").read_text(encoding="utf-8"))
    context_labels = [str(label) for label in taxonomy["context_labels"]]
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    early_placeholder = placeholder_frame(test, context_labels)
    early_placeholder.to_csv(submission_out, index=False)
    print(f"wrote early schema-valid placeholder: {submission_out}")

    train = pd.read_csv(public_dir / "train.csv").reset_index(drop=True)
    required_train_columns = {
        "episode_id",
        "call_id",
        "audio_path",
        "node_set_json",
        "reference_json",
        "caller",
        "addressee",
        "context",
    }
    required_test_columns = required_train_columns - {"caller", "addressee", "context"}
    if not required_train_columns.issubset(train.columns):
        raise ValueError("train.csv is missing required columns")
    if not required_test_columns.issubset(test.columns):
        raise ValueError("test.csv is missing required columns")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(N_JOBS)
    print(f"extracting train-only features from {len(train)} labeled calls")
    train_f1, train_f2 = extract_cache(public_dir, train, "train")
    condition_by_episode = acoustic_condition_map(train, train_f1)
    groups = np.asarray(
        [condition_by_episode[str(value)] for value in train["episode_id"]],
        dtype=np.int32,
    )
    folds = split_count(groups)
    print(
        f"validation: {folds}-fold GroupKFold over "
        f"{len(set(condition_by_episode.values()))} train-only acoustic families"
    )

    # Caller candidates combine tree matching, learning-to-rank, and a genuinely trained
    # shared metric encoder. Every model sees row-local references rather than node names.
    pair_x, pair_y, pair_rows, pair_nodes, pair_groups = build_pair_matrix(
        train, train_f1, labeled=True
    )
    pair_groups = np.asarray(
        [condition_by_episode[str(value)] for value in pair_groups],
        dtype=np.int32,
    )
    caller_specs = [(1, "balanced"), (2, "balanced"), (4, "balanced")]
    caller_oof_models: list[np.ndarray] = []
    for spec_index, (leaf, class_weight) in enumerate(caller_specs):
        oof = np.zeros(len(pair_y), dtype=np.float64)
        splitter = GroupKFold(folds)
        for train_index, valid_index in splitter.split(pair_x, pair_y, pair_groups):
            model = extra_classifier(leaf, class_weight, 10 + spec_index)
            model.fit(pair_x[train_index], pair_y[train_index])
            oof[valid_index] = positive_probability(model, pair_x[valid_index])
        caller_oof_models.append(oof)
    best_extra_caller: tuple[float, np.ndarray, np.ndarray] | None = None
    for weights in candidate_weights(len(caller_oof_models), steps=10):
        scores = sum(
            weight * oof for weight, oof in zip(weights, caller_oof_models)
        )
        predictions: list[str] = []
        for row_index in range(len(train)):
            indices = np.flatnonzero(pair_rows == row_index)
            predictions.append(
                str(pair_nodes[indices[np.argmax(scores[indices])]])
            )
        score = accuracy_score(train["caller"], predictions)
        if best_extra_caller is None or score > best_extra_caller[0]:
            best_extra_caller = (float(score), weights, scores)
    assert best_extra_caller is not None
    caller_weights = best_extra_caller[1]

    caller_family_specs: list[tuple[str, Any]] = [("extra", None)]
    caller_family_oof: list[np.ndarray] = [
        normalize_group_scores(
            best_extra_caller[2], pair_rows, len(train), temperature=None
        )
    ]

    caller_rank_oof = np.zeros(len(pair_y), dtype=np.float64)
    splitter = GroupKFold(folds)
    for train_index, valid_index in splitter.split(pair_x, pair_y, pair_groups):
        ranker = lgb_ranker(3, 25)
        ranker.fit(
            pair_x[train_index],
            pair_y[train_index],
            group=query_group_sizes(pair_rows[train_index]),
        )
        caller_rank_oof[valid_index] = ranker.predict(pair_x[valid_index])
    for temperature in (0.25, 0.5, 1.0):
        caller_family_specs.append(("ranker", temperature))
        caller_family_oof.append(
            normalize_group_scores(
                caller_rank_oof, pair_rows, len(train), temperature=temperature
            )
        )

    (
        metric_gallery,
        metric_references,
        metric_mask,
        metric_targets,
        metric_nodes,
    ) = build_metric_tensors(train, train_f1, labeled=True)
    metric_configs = [(120, 128, 48, 0.01)] * 4
    metric_oof_models: list[np.ndarray] = []
    for config_index, config in enumerate(metric_configs):
        logits = metric_oof(
            metric_gallery,
            metric_references,
            metric_mask,
            metric_targets,
            groups,
            config,
            310 + 20 * config_index,
        )
        metric_scores = metric_to_pair_scores(logits, pair_rows, len(train))
        metric_oof_models.append(metric_scores)
        caller_family_specs.append(("metric", config_index))
        caller_family_oof.append(metric_scores)
        if time.monotonic() - started >= SEARCH_STOP_SECONDS:
            break

    best_caller: tuple[float, np.ndarray, np.ndarray] | None = None
    for family_weights in candidate_weights(len(caller_family_oof), steps=4):
        scores = sum(
            weight * oof
            for weight, oof in zip(family_weights, caller_family_oof)
        )
        predictions = []
        for row_index in range(len(train)):
            indices = np.flatnonzero(pair_rows == row_index)
            predictions.append(
                str(pair_nodes[indices[np.argmax(scores[indices])]])
            )
        score = accuracy_score(train["caller"], predictions)
        if best_caller is None or score > best_caller[0]:
            best_caller = (float(score), family_weights, scores)
    assert best_caller is not None
    caller_family_weights = best_caller[1]
    caller_oof = best_caller[2]
    caller_probabilities = row_candidate_probabilities(
        pair_rows, pair_nodes, caller_oof, len(train)
    )

    # Context models use only each gallery WAV.  Classes are aligned when a rare label is
    # absent from one training fold.
    context_to_index = {label: index for index, label in enumerate(context_labels)}
    context_y = np.asarray(
        [context_to_index[str(value)] for value in train["context"]], dtype=np.int32
    )
    context_x = context_matrix(train, train_f2)
    context_specs = [
        ("extra", 1, None),
        ("extra", 2, "balanced"),
        ("lgb", 3, "balanced"),
    ]
    context_oof_models: list[np.ndarray] = []
    for spec_index, (kind, leaf, class_weight) in enumerate(context_specs):
        oof = np.zeros((len(train), len(context_labels)), dtype=np.float64)
        splitter = GroupKFold(folds)
        for train_index, valid_index in splitter.split(context_x, context_y, groups):
            if kind == "extra":
                model: Any = extra_classifier(
                    leaf, class_weight, 30 + spec_index
                )
            else:
                model = lgb_context_classifier(40 + spec_index)
            model.fit(context_x[train_index], context_y[train_index])
            oof[valid_index] = aligned_probability(
                model, context_x[valid_index], len(context_labels)
            )
        context_oof_models.append(oof)
    context_single_scores = [
        accuracy_score(context_y, np.argmax(probabilities, axis=1))
        for probabilities in context_oof_models
    ]
    context_stack_index = int(np.argmax(context_single_scores))
    context_stack_oof = context_oof_models[context_stack_index]

    # A trained unknown-target prior is an input feature to the joint social-call model.
    counts = node_counts(train)
    unknown_y = (train["addressee"].astype(str).to_numpy() == "UNKNOWN").astype(np.int8)
    unknown_x = np.concatenate(
        [context_x, counts[:, None], context_stack_oof], axis=1
    ).astype(np.float32)
    unknown_specs = [2, 8, 12]
    unknown_candidates: list[tuple[float, int, np.ndarray]] = []
    for spec_index, leaf in enumerate(unknown_specs):
        oof = np.zeros(len(train), dtype=np.float64)
        splitter = GroupKFold(folds)
        for train_index, valid_index in splitter.split(unknown_x, unknown_y, groups):
            model = extra_classifier(leaf, "balanced", 50 + spec_index)
            model.fit(unknown_x[train_index], unknown_y[train_index])
            oof[valid_index] = positive_probability(model, unknown_x[valid_index])
        if len(np.unique(unknown_y)) > 1:
            selection_score = roc_auc_score(unknown_y, oof)
        else:
            selection_score = accuracy_score(unknown_y, oof >= 0.5)
        unknown_candidates.append((float(selection_score), leaf, oof))
        if time.monotonic() - started >= SEARCH_STOP_SECONDS:
            break
    unknown_candidates.sort(key=lambda item: item[0], reverse=True)
    _, unknown_leaf, unknown_oof = unknown_candidates[0]

    # Joint model candidates are complete (caller, addressee) hypotheses.  No anonymous
    # label index is a feature: all node information comes from its supplied references.
    joint_x, joint_y, joint_rows, joint_callers, joint_targets, joint_groups = (
        build_joint_matrix(
            train,
            train_f1,
            caller_probabilities,
            context_stack_oof,
            unknown_oof,
            labeled=True,
        )
    )
    joint_groups = np.asarray(
        [condition_by_episode[str(value)] for value in joint_groups],
        dtype=np.int32,
    )
    base_context_labels = np.asarray(
        [context_labels[index] for index in np.argmax(context_stack_oof, axis=1)],
        dtype=object,
    )
    joint_base_specs = [
        ("extra", 2),
        ("extra", 4),
        ("extra", 8),
        ("extra", 12),
        ("ranker", 3),
        ("ranker", 5),
    ]
    joint_specs: list[tuple[str, int, float | None]] = []
    joint_oof_models: list[np.ndarray] = []
    for spec_index, (kind, parameter) in enumerate(joint_base_specs):
        raw_oof = np.zeros(len(joint_y), dtype=np.float64)
        splitter = GroupKFold(folds)
        for train_index, valid_index in splitter.split(
            joint_x, joint_y, joint_groups
        ):
            if kind == "extra":
                model: Any = extra_classifier(
                    parameter, "balanced", 70 + spec_index
                )
                model.fit(joint_x[train_index], joint_y[train_index])
                raw_oof[valid_index] = positive_probability(
                    model, joint_x[valid_index]
                )
            else:
                model = lgb_ranker(parameter, 80 + spec_index)
                model.fit(
                    joint_x[train_index],
                    joint_y[train_index],
                    group=query_group_sizes(joint_rows[train_index]),
                )
                raw_oof[valid_index] = model.predict(joint_x[valid_index])
        if kind == "extra":
            temperature: float | None = None
            normalized = normalize_group_scores(
                raw_oof, joint_rows, len(train), temperature=None
            )
        else:
            # Ranking temperature is searched on grouped OOF social predictions.
            best_temperature: tuple[float, float, np.ndarray] | None = None
            for candidate_temperature in (0.25, 0.5, 1.0):
                candidate = normalize_group_scores(
                    raw_oof,
                    joint_rows,
                    len(train),
                    temperature=candidate_temperature,
                )
                candidate_score = -np.inf
                for blend in (0.25, 0.5, 0.75, 1.0):
                    decoded = decode_social(
                        candidate,
                        blend,
                        joint_rows,
                        joint_callers,
                        joint_targets,
                        caller_probabilities,
                        len(train),
                    )
                    components = episode_components(
                        train, decoded[0], decoded[1], base_context_labels
                    )
                    candidate_score = max(
                        candidate_score, proxy_final_score(components, train)
                    )
                if (
                    best_temperature is None
                    or candidate_score > best_temperature[0]
                ):
                    best_temperature = (
                        float(candidate_score),
                        candidate_temperature,
                        candidate,
                    )
            assert best_temperature is not None
            temperature = best_temperature[1]
            normalized = best_temperature[2]
        joint_specs.append((kind, parameter, temperature))
        joint_oof_models.append(normalized)
        if time.monotonic() - started >= SEARCH_STOP_SECONDS:
            break

    social_candidates: list[tuple[Any, ...]] = []
    for model_weights in candidate_weights(len(joint_oof_models), steps=4):
        blended_joint_oof = sum(
            weight * probabilities
            for weight, probabilities in zip(model_weights, joint_oof_models)
        )
        # A nonzero joint weight is mandatory: trained joint models must produce the
        # addressee rather than candidate ordering or a fixed rule.
        for joint_weight_candidate in (0.25, 0.5, 0.75, 1.0):
            decoded = decode_social(
                blended_joint_oof,
                joint_weight_candidate,
                joint_rows,
                joint_callers,
                joint_targets,
                caller_probabilities,
                len(train),
            )
            predicted_callers, predicted_targets = decoded[:2]
            components = episode_components(
                train, predicted_callers, predicted_targets, base_context_labels
            )
            score = proxy_final_score(components, train)
            social_candidates.append(
                (
                    score,
                    model_weights,
                    joint_weight_candidate,
                    blended_joint_oof,
                    predicted_callers,
                    predicted_targets,
                    decoded[2],
                    decoded[3],
                    decoded[4],
                )
            )
    social_candidates.sort(key=lambda item: item[0], reverse=True)
    (
        _,
        joint_weights,
        joint_weight,
        selected_joint_oof,
        oof_callers,
        oof_targets,
        oof_social_top,
        oof_social_margin,
        oof_chosen_joint,
    ) = social_candidates[0]

    # Search trained context-model blends against the actual train-only episode metric.
    context_blend_best: tuple[
        float, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame
    ] | None = None
    for weights in candidate_weights(len(context_oof_models), steps=4):
        probabilities = sum(
            weight * model_probability
            for weight, model_probability in zip(weights, context_oof_models)
        )
        labels = np.asarray(
            [context_labels[index] for index in np.argmax(probabilities, axis=1)],
            dtype=object,
        )
        components = episode_components(train, oof_callers, oof_targets, labels)
        score = proxy_final_score(components, train)
        if context_blend_best is None or score > context_blend_best[0]:
            context_blend_best = (score, weights, probabilities, labels, components)
    assert context_blend_best is not None
    context_weights = context_blend_best[1]
    context_blend_oof = context_blend_best[2]

    # Generic coordinate search; every class receives the same candidate grid.
    context_biases = np.zeros(len(context_labels), dtype=np.float64)
    bias_score = context_blend_best[0]
    for _ in range(2):
        for class_index in range(len(context_labels)):
            selected_value = context_biases[class_index]
            selected_score = bias_score
            for candidate_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
                trial = context_biases.copy()
                trial[class_index] = candidate_value
                trial_labels = context_decision(
                    context_blend_oof, trial, context_labels
                )[0]
                trial_components = episode_components(
                    train, oof_callers, oof_targets, trial_labels
                )
                trial_score = proxy_final_score(trial_components, train)
                if trial_score > selected_score + 1.0e-12:
                    selected_score = trial_score
                    selected_value = candidate_value
            context_biases[class_index] = selected_value
            bias_score = selected_score

    oof_contexts, oof_context_top, oof_context_margin = context_decision(
        context_blend_oof, context_biases, context_labels
    )

    # A permutation-equivariant graph head predicts episode edge-count quotas from
    # cross-fitted social probabilities. Hungarian assignment then keeps the call ledger
    # and graph exactly consistent while using all calls in the row-local episode.
    train_social_posteriors = social_posteriors(
        selected_joint_oof,
        joint_weight,
        joint_rows,
        joint_callers,
        joint_targets,
        caller_probabilities,
        len(train),
    )
    rate_episode_ids, rate_x, rate_y = episode_rate_matrix(
        train,
        train_social_posteriors,
        context_blend_oof,
        unknown_oof,
        labeled=True,
    )
    rate_groups = np.asarray(
        [condition_by_episode[episode_id] for episode_id in rate_episode_ids],
        dtype=np.int32,
    )
    (
        rate_model,
        rate_mean,
        rate_scale,
        rate_oof,
        rate_alpha,
    ) = fit_episode_rate_head(rate_x, rate_y, rate_groups)
    edge_x, edge_y, edge_keys = edge_head_matrix(
        train,
        train_f1,
        train_social_posteriors,
        caller_probabilities,
        context_blend_oof,
        rate_episode_ids,
        rate_oof,
        labeled=True,
    )
    edge_groups = np.asarray(
        [condition_by_episode[episode_id] for episode_id, _, _ in edge_keys],
        dtype=np.int32,
    )
    edge_specs, edge_oof_models = crossfit_edge_heads(
        edge_x, edge_y, edge_groups
    )
    base_graph_components = episode_components(
        train, oof_callers, oof_targets, oof_contexts
    )
    graph_head_best: tuple[
        float, tuple[str, int] | None, np.ndarray, tuple[np.ndarray, ...]
    ] = (
        proxy_final_score(base_graph_components, train),
        None,
        np.zeros(len(edge_y), dtype=np.float64),
        (
            oof_callers,
            oof_targets,
            oof_social_top,
            oof_social_margin,
            oof_chosen_joint,
        ),
    )
    for edge_spec, edge_oof in zip(edge_specs, edge_oof_models):
        decoded = quota_decode(
            train,
            train_social_posteriors,
            rate_episode_ids,
            rate_oof,
            edge_keys,
            edge_oof,
        )
        components = episode_components(
            train, decoded[0], decoded[1], oof_contexts
        )
        score = proxy_final_score(components, train)
        if score > graph_head_best[0] + 1.0e-12:
            graph_head_best = (
                float(score),
                edge_spec,
                edge_oof,
                decoded,
            )
    selected_edge_spec = graph_head_best[1]
    edge_model: Any = None
    if selected_edge_spec is not None:
        edge_model = make_edge_regressor(
            selected_edge_spec[0], selected_edge_spec[1], 390
        )
        edge_model.fit(edge_x, edge_y)
        (
            oof_callers,
            oof_targets,
            oof_social_top,
            oof_social_margin,
            oof_chosen_joint,
        ) = graph_head_best[3]
    validation_components = episode_components(
        train, oof_callers, oof_targets, oof_contexts
    )

    # Confidence is itself learned from grouped OOF behavior.
    call_correctness = (
        0.45 * (oof_callers == train["caller"].astype(str).to_numpy())
        + 0.25 * (oof_targets == train["addressee"].astype(str).to_numpy())
        + 0.30 * (oof_contexts == train["context"].astype(str).to_numpy())
    ).astype(np.float64)
    meta_x = call_meta_features(
        oof_social_top,
        oof_social_margin,
        caller_probabilities,
        oof_context_top,
        oof_context_margin,
        unknown_oof,
        counts,
        oof_targets,
        oof_chosen_joint,
    )
    call_calibrator, call_confidence_oof, call_leaf = crossfit_call_calibrator(
        meta_x, call_correctness, groups
    )
    row_calibrator, row_confidence_oof, row_calibrator_name = fit_row_calibrator(
        train, validation_components
    )

    episode_scores: list[float] = []
    for episode_index, component in validation_components.iterrows():
        episode_id = str(component["episode_id"])
        indices = np.flatnonzero(train["episode_id"].astype(str).to_numpy() == episode_id)
        call_calibration = float(
            np.mean(
                np.maximum(
                    0.0,
                    1.0 - np.abs(call_confidence_oof[indices] - call_correctness[indices]),
                )
            )
        )
        row_calibration = max(
            0.0, 1.0 - abs(float(row_confidence_oof[episode_index]) - component["core"])
        )
        episode_scores.append(
            float(component["core"])
            * (0.94 + 0.04 * row_calibration + 0.02 * call_calibration)
        )
    validation_components["episode_score"] = episode_scores
    validation_score = proxy_final_score(
        validation_components, train, score_column="episode_score"
    )
    print(
        "OOF validation "
        f"final_proxy={validation_score:.6f} "
        f"core={validation_components['core'].mean():.6f} "
        f"caller={validation_components['caller_accuracy'].mean():.6f} "
        f"addressee={validation_components['addressee_accuracy'].mean():.6f} "
        f"context={validation_components['context_accuracy'].mean():.6f} "
        f"graph={validation_components['graph_score'].mean():.6f}"
    )
    print(
        "selected "
        f"caller_tree_weights={caller_weights.tolist()} "
        f"caller_family_weights={caller_family_weights.tolist()} "
        f"unknown_leaf={unknown_leaf} joint_weights={joint_weights.tolist()} "
        f"joint_weight={joint_weight:.2f} "
        f"context_weights={context_weights.tolist()} "
        f"rate_alpha={rate_alpha:g} edge_head={selected_edge_spec} "
        f"call_calibrator_leaf={call_leaf} row_calibrator={row_calibrator_name}"
    )

    # Final fit starts well before the hard guard.  Search stops at 2400 seconds, leaving
    # at least 600 seconds for these compact models and inference.
    if time.monotonic() - started >= TRAIN_STOP_SECONDS:
        print("warning: training guard reached; preserving early valid placeholder")
        return

    extra_caller_needed = any(
        weight > 0.0 and spec[0] == "extra"
        for spec, weight in zip(caller_family_specs, caller_family_weights)
    )
    caller_models: list[Any] = []
    for spec_index, ((leaf, class_weight), weight) in enumerate(
        zip(caller_specs[: len(caller_weights)], caller_weights)
    ):
        if weight <= 0.0 or not extra_caller_needed:
            caller_models.append(None)
            continue
        model = extra_classifier(leaf, class_weight, 110 + spec_index)
        model.fit(pair_x, pair_y)
        caller_models.append(model)

    caller_rank_needed = any(
        weight > 0.0 and spec[0] == "ranker"
        for spec, weight in zip(caller_family_specs, caller_family_weights)
    )
    caller_rank_model: Any = None
    if caller_rank_needed:
        caller_rank_model = lgb_ranker(3, 125)
        caller_rank_model.fit(
            pair_x, pair_y, group=query_group_sizes(pair_rows)
        )

    metric_full_models: dict[
        int, tuple[ReferenceMetricNet, np.ndarray, np.ndarray]
    ] = {}
    all_train_indices = np.arange(len(train), dtype=np.int32)
    for family_spec, weight in zip(caller_family_specs, caller_family_weights):
        if family_spec[0] != "metric" or weight <= 0.0:
            continue
        config_index = int(family_spec[1])
        if config_index in metric_full_models:
            continue
        epochs, hidden_dim, embedding_dim, weight_decay = metric_configs[
            config_index
        ]
        metric_full_models[config_index] = fit_reference_metric(
            metric_gallery,
            metric_references,
            metric_mask,
            metric_targets,
            all_train_indices,
            epochs,
            hidden_dim,
            embedding_dim,
            weight_decay,
            SEED + 310 + 20 * config_index,
        )

    context_models: list[Any] = []
    for spec_index, ((kind, leaf, class_weight), weight) in enumerate(
        zip(context_specs[: len(context_weights)], context_weights)
    ):
        if weight <= 0.0 and spec_index != context_stack_index:
            context_models.append(None)
            continue
        if kind == "extra":
            model = extra_classifier(leaf, class_weight, 130 + spec_index)
        else:
            model = lgb_context_classifier(140 + spec_index)
        model.fit(context_x, context_y)
        context_models.append(model)

    unknown_model = extra_classifier(unknown_leaf, "balanced", 150)
    unknown_model.fit(unknown_x, unknown_y)
    joint_models: list[Any] = []
    # Cross-fitted stack features prevent the joint models learning in-sample confidence.
    for spec_index, ((kind, parameter, _), weight) in enumerate(
        zip(joint_specs[: len(joint_weights)], joint_weights)
    ):
        if weight <= 0.0:
            joint_models.append(None)
            continue
        if kind == "extra":
            model = extra_classifier(parameter, "balanced", 170 + spec_index)
            model.fit(joint_x, joint_y)
        else:
            model = lgb_ranker(parameter, 180 + spec_index)
            model.fit(
                joint_x, joint_y, group=query_group_sizes(joint_rows)
            )
        joint_models.append(model)

    print("extracting inference-only test features")
    test_f1, test_f2 = extract_cache(public_dir, test, "test")
    test_pair_x, _, test_pair_rows, test_pair_nodes, _ = build_pair_matrix(
        test, test_f1, labeled=False
    )
    test_metric_gallery, test_metric_references, test_metric_mask, _, _ = (
        build_metric_tensors(test, test_f1, labeled=False)
    )

    test_extra_score = np.zeros(len(test_pair_x), dtype=np.float64)
    for weight, model in zip(caller_weights, caller_models):
        if weight > 0.0 and model is not None:
            test_extra_score += weight * positive_probability(model, test_pair_x)
    test_rank_raw: np.ndarray | None = None
    if caller_rank_model is not None:
        test_rank_raw = caller_rank_model.predict(test_pair_x)
    test_metric_scores: dict[int, np.ndarray] = {}
    test_caller_families: list[np.ndarray] = []
    for family_spec, family_weight in zip(
        caller_family_specs, caller_family_weights
    ):
        kind, parameter = family_spec
        if family_weight <= 0.0:
            test_caller_families.append(
                np.zeros(len(test_pair_x), dtype=np.float64)
            )
            continue
        if kind == "extra":
            family_score = normalize_group_scores(
                test_extra_score, test_pair_rows, len(test), temperature=None
            )
        elif kind == "ranker":
            assert test_rank_raw is not None
            family_score = normalize_group_scores(
                test_rank_raw,
                test_pair_rows,
                len(test),
                temperature=float(parameter),
            )
        else:
            config_index = int(parameter)
            if config_index not in test_metric_scores:
                model, mean, scale = metric_full_models[config_index]
                logits = predict_reference_metric(
                    model,
                    mean,
                    scale,
                    test_metric_gallery,
                    test_metric_references,
                    test_metric_mask,
                )
                test_metric_scores[config_index] = metric_to_pair_scores(
                    logits, test_pair_rows, len(test)
                )
            family_score = test_metric_scores[config_index]
        test_caller_families.append(family_score)
    test_caller_score = sum(
        weight * score
        for weight, score in zip(caller_family_weights, test_caller_families)
    )
    test_caller_probabilities = row_candidate_probabilities(
        test_pair_rows, test_pair_nodes, test_caller_score, len(test)
    )

    test_context_x = context_matrix(test, test_f2)
    test_context_model_probabilities: list[np.ndarray] = []
    for spec_index, model in enumerate(context_models):
        if model is None:
            test_context_model_probabilities.append(
                np.zeros((len(test), len(context_labels)), dtype=np.float64)
            )
        else:
            test_context_model_probabilities.append(
                aligned_probability(model, test_context_x, len(context_labels))
            )
    test_context_stack = test_context_model_probabilities[context_stack_index]
    test_counts = node_counts(test)
    test_unknown_x = np.concatenate(
        [test_context_x, test_counts[:, None], test_context_stack], axis=1
    ).astype(np.float32)
    test_unknown_probability = positive_probability(unknown_model, test_unknown_x)

    (
        test_joint_x,
        _,
        test_joint_rows,
        test_joint_callers,
        test_joint_targets,
        _,
    ) = build_joint_matrix(
        test,
        test_f1,
        test_caller_probabilities,
        test_context_stack,
        test_unknown_probability,
        labeled=False,
    )
    test_joint_score = np.zeros(len(test_joint_x), dtype=np.float64)
    for (kind, _, temperature), weight, model in zip(
        joint_specs, joint_weights, joint_models
    ):
        if weight <= 0.0 or model is None:
            continue
        if kind == "extra":
            raw_score = positive_probability(model, test_joint_x)
        else:
            raw_score = model.predict(test_joint_x)
        normalized_score = normalize_group_scores(
            raw_score,
            test_joint_rows,
            len(test),
            temperature=temperature,
        )
        test_joint_score += weight * normalized_score
    test_social = decode_social(
        test_joint_score,
        joint_weight,
        test_joint_rows,
        test_joint_callers,
        test_joint_targets,
        test_caller_probabilities,
        len(test),
    )
    test_callers, test_targets = test_social[:2]

    test_context_blend = sum(
        weight * probabilities
        for weight, probabilities in zip(
            context_weights, test_context_model_probabilities
        )
    )
    test_contexts, test_context_top, test_context_margin = context_decision(
        test_context_blend, context_biases, context_labels
    )
    if selected_edge_spec is not None:
        assert edge_model is not None
        test_social_posteriors = social_posteriors(
            test_joint_score,
            joint_weight,
            test_joint_rows,
            test_joint_callers,
            test_joint_targets,
            test_caller_probabilities,
            len(test),
        )
        test_rate_episode_ids, test_rate_x, _ = episode_rate_matrix(
            test,
            test_social_posteriors,
            test_context_blend,
            test_unknown_probability,
            labeled=False,
        )
        test_known_rates = np.clip(
            rate_model.predict((test_rate_x - rate_mean) / rate_scale),
            0.0,
            1.0,
        )
        test_edge_x, _, test_edge_keys = edge_head_matrix(
            test,
            test_f1,
            test_social_posteriors,
            test_caller_probabilities,
            test_context_blend,
            test_rate_episode_ids,
            test_known_rates,
            labeled=False,
        )
        test_edge_predictions = np.clip(
            edge_model.predict(test_edge_x), 0.0, None
        )
        test_social = quota_decode(
            test,
            test_social_posteriors,
            test_rate_episode_ids,
            test_known_rates,
            test_edge_keys,
            test_edge_predictions,
        )
        test_callers, test_targets = test_social[:2]
    test_meta_x = call_meta_features(
        test_social[2],
        test_social[3],
        test_caller_probabilities,
        test_context_top,
        test_context_margin,
        test_unknown_probability,
        test_counts,
        test_targets,
        test_social[4],
    )
    test_call_confidence = np.clip(call_calibrator.predict(test_meta_x), 0.0, 1.0)

    output_rows: list[dict[str, Any]] = []
    episode_order = test["episode_id"].drop_duplicates().astype(str).tolist()
    for episode_id in episode_order:
        indices = np.flatnonzero(test["episode_id"].astype(str).to_numpy() == episode_id)
        calls: list[dict[str, Any]] = []
        for index in indices:
            row = test.iloc[index]
            nodes = safe_nodes(row)
            caller = str(test_callers[index])
            target = str(test_targets[index])
            context = str(test_contexts[index])
            # Model-based values are retained whenever valid.  These branches only make a
            # corrupt/noisy row schema-safe; they do not alter valid predictions.
            if caller not in nodes:
                caller = max(
                    test_caller_probabilities[index],
                    key=test_caller_probabilities[index].get,
                )
            if target not in nodes + ["UNKNOWN"] or target == caller:
                target = "UNKNOWN"
            if context not in context_labels:
                context = context_labels[int(np.argmax(test_context_blend[index]))]
            confidence = float(test_call_confidence[index])
            if not np.isfinite(confidence):
                confidence = 0.0
            calls.append(
                {
                    "call_id": str(row["call_id"]),
                    "caller": caller,
                    "addressee": target,
                    "context": context,
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                }
            )
        first_row = test.iloc[int(indices[0])]
        row_feature = row_calibration_features(first_row)[None, :]
        row_confidence = float(np.clip(row_calibrator.predict(row_feature)[0], 0.0, 1.0))
        if not np.isfinite(row_confidence):
            row_confidence = 0.0
        output_rows.append(
            {
                "episode_id": episode_id,
                "call_predictions_json": compact_json(calls),
                "graph_json": compact_json(make_graph(calls)),
                "confidence": row_confidence,
            }
        )

    submission = pd.DataFrame(
        output_rows,
        columns=["episode_id", "call_predictions_json", "graph_json", "confidence"],
    )
    errors = validate_submission(submission, test, set(context_labels))
    if errors:
        print("warning: final submission validation failed; early placeholder retained")
        for error in errors[:20]:
            print(" -", error)
        return
    submission.to_csv(submission_out, index=False)
    print(
        f"wrote {len(submission)} validated episode rows to {submission_out} "
        f"in {time.monotonic() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
