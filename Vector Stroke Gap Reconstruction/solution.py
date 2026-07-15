#!/usr/bin/env python3
"""CPU-only solver for Vector Stroke Gap Reconstruction."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

try:
    import numpy as np
    from lightgbm import LGBMClassifier, LGBMRanker
except ImportError as exc:  # fail explicitly rather than changing the tuned algorithm
    raise SystemExit("Vector Stroke Gap Reconstruction requires NumPy and LightGBM") from exc

GRID_SIZE = 32
PREFIX_CONTEXT = 4
SUFFIX_CONTEXT = 5
NEIGHBOR_COUNT = 60
CONTEXT_TEMPERATURE = 4.0
QUALITY_SCALE = 3.0
RERANK_DISTANCE_TEMPERATURE = 64.0
CONNECTIVITY_PENALTY = 20.0
BLEND_EXACT_WEIGHT = 0.3
BLEND_MIN_CONTEXT_DISTANCE = 1.0
CELL_RE = re.compile(r"c(\d{2})_(\d{2})\Z")

# Each tuple is (a, b, c, d) for [[a, b], [c, d]]. The order is part
# of canonical tie-breaking and must remain stable.
TRANSFORMS: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),
    (1, 0, 0, -1),
    (0, -1, 1, 0),
    (0, 1, 1, 0),
    (-1, 0, 0, -1),
    (-1, 0, 0, 1),
    (0, 1, -1, 0),
    (0, -1, -1, 0),
)


@dataclass(slots=True)
class Query:
    row_id: str
    gap: dict
    missing_count: int
    point_count: int
    stroke_count: int
    source_index: int | None
    truth: list[str] | None


@dataclass(slots=True)
class TemplateGroup:
    features: np.ndarray  # (candidate, 18), int8
    paths: np.ndarray  # (candidate, missing_count, 2), int8, canonical relative coords
    sources: np.ndarray  # (candidate,), int32


@dataclass(slots=True)
class NeighborSet:
    distances: np.ndarray  # (candidate,), float64
    paths: np.ndarray  # (candidate, missing_count, 2), int16 world coordinates


@dataclass(slots=True)
class CellOptions:
    features: np.ndarray  # (option, 46), float32
    cells: np.ndarray  # (option, 2), int16
    steps: np.ndarray  # (option,), int8
    slices: list[slice]  # one contiguous option slice per hidden time step


def parse_cell(cell: str) -> tuple[int, int]:
    match = CELL_RE.fullmatch(cell)
    if match is None:
        raise ValueError(f"invalid cell name: {cell!r}")
    row, column = int(match.group(1)), int(match.group(2))
    if not (0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE):
        raise ValueError(f"cell outside {GRID_SIZE}x{GRID_SIZE} grid: {cell!r}")
    return row, column


def format_cell(row: int, column: int) -> str:
    return f"c{row:02d}_{column:02d}"


def cells_to_array(cells: Sequence[str]) -> np.ndarray:
    return np.asarray([parse_cell(cell) for cell in cells], dtype=np.int16)


def canonical_transform(delta_row: int, delta_column: int) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    best_transform = TRANSFORMS[0]
    a, b, c, d = best_transform
    best_delta = (a * delta_row + b * delta_column, c * delta_row + d * delta_column)
    for transform in TRANSFORMS[1:]:
        a, b, c, d = transform
        transformed = (a * delta_row + b * delta_column, c * delta_row + d * delta_column)
        if transformed > best_delta:
            best_transform = transform
            best_delta = transformed
    return best_transform, best_delta


def apply_transform(points: np.ndarray, transform: tuple[int, int, int, int], out: np.ndarray | None = None) -> np.ndarray:
    """Map row-vector points from world coordinates to canonical coordinates."""
    if out is None:
        out = np.empty_like(points, dtype=np.int16)
    a, b, c, d = transform
    out[:, 0] = a * points[:, 0] + b * points[:, 1]
    out[:, 1] = c * points[:, 0] + d * points[:, 1]
    return out


def inverse_transform(points: np.ndarray, transform: tuple[int, int, int, int]) -> np.ndarray:
    """Map row-vector points from canonical coordinates back to world coordinates."""
    a, b, c, d = transform
    result = np.empty_like(points, dtype=np.int16)
    result[..., 0] = a * points[..., 0] + c * points[..., 1]
    result[..., 1] = b * points[..., 0] + d * points[..., 1]
    return result


def read_dataset(path: Path, require_answers: bool) -> tuple[list[dict], list[Query]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"dataset is empty: {path}")

    ids: set[str] = set()
    queries: list[Query] = []
    for source_index, row in enumerate(rows):
        row_id = row.get("id", "")
        if not row_id or row_id in ids:
            raise ValueError(f"missing or duplicate id {row_id!r} in {path}")
        ids.add(row_id)

        sketch = json.loads(row["sketch_json"])
        if sketch.get("grid_size") != GRID_SIZE:
            raise ValueError(f"{row_id}: expected grid_size={GRID_SIZE}")
        gaps = [stroke for stroke in sketch["visible_strokes"] if "prefix" in stroke or "suffix" in stroke]
        if len(gaps) != 1 or "prefix" not in gaps[0] or "suffix" not in gaps[0]:
            raise ValueError(f"{row_id}: expected exactly one prefix/suffix gap stroke")
        gap = gaps[0]
        missing_count = int(row["missing_count_hint"])
        if not (8 <= missing_count <= 20):
            raise ValueError(f"{row_id}: missing_count_hint outside released 8..20 range")
        if len(gap["prefix"]) < PREFIX_CONTEXT or len(gap["suffix"]) < SUFFIX_CONTEXT:
            raise ValueError(
                f"{row_id}: requires at least {PREFIX_CONTEXT} prefix and "
                f"{SUFFIX_CONTEXT} suffix cells"
            )
        if int(gap["hidden_length_hint"]) != missing_count:
            raise ValueError(f"{row_id}: inconsistent hidden length hints")
        for stroke in sketch["visible_strokes"]:
            visible = stroke.get("points")
            if visible is None:
                visible = stroke["prefix"] + stroke["suffix"]
            for cell in visible:
                parse_cell(cell)

        truth: list[str] | None = None
        if require_answers:
            if "answer_json" not in row or not row["answer_json"]:
                raise ValueError(f"{row_id}: missing answer_json")
            answer = json.loads(row["answer_json"])
            if set(answer) != {"hidden_cells"}:
                raise ValueError(f"{row_id}: answer_json must contain only hidden_cells")
            truth = list(answer["hidden_cells"])
            if len(truth) != missing_count:
                raise ValueError(f"{row_id}: answer length differs from hint")
            for cell in truth:
                parse_cell(cell)

        queries.append(
            Query(
                row_id=row_id,
                gap=gap,
                missing_count=missing_count,
                point_count=int(row["point_count_hint"]),
                stroke_count=len(sketch["visible_strokes"]),
                source_index=source_index if require_answers else None,
                truth=truth,
            )
        )
    return rows, queries


def reconstruct_training_strokes(rows: Sequence[dict], queries: Sequence[Query]) -> list[tuple[int, np.ndarray]]:
    strokes: list[tuple[int, np.ndarray]] = []
    for source_index, (row, query) in enumerate(zip(rows, queries, strict=True)):
        sketch = json.loads(row["sketch_json"])
        assert query.truth is not None
        for stroke in sketch["visible_strokes"]:
            if "points" in stroke:
                cells = stroke["points"]
            else:
                cells = stroke["prefix"] + query.truth + stroke["suffix"]
            strokes.append((source_index, cells_to_array(cells)))
    return strokes


def query_key(query: Query) -> tuple[int, int, int]:
    prefix_end = parse_cell(query.gap["prefix"][-1])
    suffix_start = parse_cell(query.gap["suffix"][0])
    _, canonical_delta = canonical_transform(
        suffix_start[0] - prefix_end[0], suffix_start[1] - prefix_end[1]
    )
    return query.missing_count, canonical_delta[0], canonical_delta[1]


def candidate_windows(
    strokes: Sequence[tuple[int, np.ndarray]],
    lengths: Sequence[int],
    required_keys: set[tuple[int, int, int]],
) -> Iterator[tuple[int, np.ndarray, int, int, tuple[int, int, int, int], tuple[int, int, int]]]:
    for source_index, points in strokes:
        point_count = len(points)
        for missing_count in lengths:
            # p is the visible cell immediately before the gap. The largest
            # p leaves the endpoint plus four further suffix context cells.
            for p in range(PREFIX_CONTEXT - 1, point_count - missing_count - SUFFIX_CONTEXT):
                right = p + missing_count + 1
                delta = points[right] - points[p]
                transform, canonical_delta = canonical_transform(int(delta[0]), int(delta[1]))
                key = (missing_count, canonical_delta[0], canonical_delta[1])
                if key in required_keys:
                    yield source_index, points, missing_count, p, transform, key


def build_template_library(
    strokes: Sequence[tuple[int, np.ndarray]], queries: Sequence[Query]
) -> tuple[dict[tuple[int, int, int], TemplateGroup], int]:
    lengths = sorted({query.missing_count for query in queries})
    required_keys = {query_key(query) for query in queries}
    counts = {key: 0 for key in required_keys}

    for _, _, _, _, _, key in candidate_windows(strokes, lengths, required_keys):
        counts[key] += 1

    groups: dict[tuple[int, int, int], TemplateGroup] = {}
    for key, count in counts.items():
        missing_count = key[0]
        groups[key] = TemplateGroup(
            features=np.empty((count, 2 * (PREFIX_CONTEXT + SUFFIX_CONTEXT)), dtype=np.int8),
            paths=np.empty((count, missing_count, 2), dtype=np.int8),
            sources=np.empty(count, dtype=np.int32),
        )

    cursors = {key: 0 for key in required_keys}
    relative_buffers = {
        missing_count: np.empty((PREFIX_CONTEXT + SUFFIX_CONTEXT + missing_count, 2), dtype=np.int16)
        for missing_count in lengths
    }
    transformed_buffers = {missing_count: np.empty_like(buffer) for missing_count, buffer in relative_buffers.items()}

    for source_index, points, missing_count, p, transform, key in candidate_windows(
        strokes, lengths, required_keys
    ):
        right = p + missing_count + 1
        relative = relative_buffers[missing_count]
        relative[:PREFIX_CONTEXT] = points[p - PREFIX_CONTEXT + 1 : p + 1] - points[p]
        relative[PREFIX_CONTEXT : PREFIX_CONTEXT + SUFFIX_CONTEXT] = (
            points[right : right + SUFFIX_CONTEXT] - points[right]
        )
        relative[PREFIX_CONTEXT + SUFFIX_CONTEXT :] = points[p + 1 : right] - points[p]
        transformed = apply_transform(relative, transform, transformed_buffers[missing_count])

        cursor = cursors[key]
        group = groups[key]
        group.features[cursor] = transformed[: PREFIX_CONTEXT + SUFFIX_CONTEXT].reshape(-1)
        group.paths[cursor] = transformed[PREFIX_CONTEXT + SUFFIX_CONTEXT :]
        group.sources[cursor] = source_index
        cursors[key] = cursor + 1

    if any(cursors[key] != counts[key] for key in required_keys):
        raise RuntimeError("template library fill count mismatch")
    return groups, sum(counts.values())


def make_query_feature(query: Query) -> tuple[tuple[int, int, int], tuple[int, int, int, int], np.ndarray, np.ndarray]:
    prefix = cells_to_array(query.gap["prefix"][-PREFIX_CONTEXT:])
    suffix = cells_to_array(query.gap["suffix"][:SUFFIX_CONTEXT])
    start = prefix[-1]
    end = suffix[0]
    transform, canonical_delta = canonical_transform(int(end[0] - start[0]), int(end[1] - start[1]))

    relative = np.empty((PREFIX_CONTEXT + SUFFIX_CONTEXT, 2), dtype=np.int16)
    relative[:PREFIX_CONTEXT] = prefix - start
    relative[PREFIX_CONTEXT:] = suffix - end
    transformed = apply_transform(relative, transform)
    key = (query.missing_count, canonical_delta[0], canonical_delta[1])
    return key, transform, transformed.reshape(-1), start


def select_neighbors(distances: np.ndarray, eligible: np.ndarray, count: int) -> np.ndarray:
    """Select by (distance, candidate index), including deterministic cutoff ties."""
    if len(eligible) <= count:
        selected = eligible.copy()
    else:
        eligible_distances = distances[eligible]
        cutoff = int(np.partition(eligible_distances, count - 1)[count - 1])
        below = eligible[eligible_distances < cutoff]
        tied = eligible[eligible_distances == cutoff]
        selected = np.concatenate((below, tied[: count - len(below)]))
    order = np.lexsort((selected, distances[selected]))
    return selected[order]


def hermite_fallback(query: Query) -> list[str]:
    prefix = cells_to_array(query.gap["prefix"][-PREFIX_CONTEXT:]).astype(np.float64)
    suffix = cells_to_array(query.gap["suffix"][:PREFIX_CONTEXT]).astype(np.float64)
    start, end = prefix[-1], suffix[0]
    start_tangent = (prefix[-1] - prefix[0]) / (len(prefix) - 1)
    end_tangent = (suffix[-1] - suffix[0]) / (len(suffix) - 1)
    duration = query.missing_count + 1
    output: list[str] = []
    for step in range(1, query.missing_count + 1):
        t = step / duration
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        point = h00 * start + h10 * duration * start_tangent + h01 * end + h11 * duration * end_tangent
        point = np.clip(np.rint(point), 0, GRID_SIZE - 1).astype(np.int16)
        output.append(format_cell(int(point[0]), int(point[1])))
    return output


def get_neighbors(
    query: Query,
    groups: dict[tuple[int, int, int], TemplateGroup],
    exclude_source: int | None = None,
) -> NeighborSet | None:
    key, transform, feature, start = make_query_feature(query)
    group = groups.get(key)
    if group is None or len(group.features) == 0:
        return None

    differences = group.features.astype(np.int16) - feature.astype(np.int16)
    differences = differences.astype(np.int32)
    distances = np.einsum("ij,ij->i", differences, differences, dtype=np.int64)
    if exclude_source is None:
        eligible = np.arange(len(group.features), dtype=np.int64)
    else:
        eligible = np.flatnonzero(group.sources != exclude_source)
    if len(eligible) == 0:
        return None

    selected = select_neighbors(distances, eligible, NEIGHBOR_COUNT)
    canonical_paths = group.paths[selected].astype(np.int16)
    world_paths = inverse_transform(canonical_paths, transform)
    world_paths[..., 0] += int(start[0])
    world_paths[..., 1] += int(start[1])
    np.clip(world_paths, 0, GRID_SIZE - 1, out=world_paths)
    return NeighborSet(distances=distances[selected].astype(np.float64), paths=world_paths)


def candidate_quality_features(query: Query, neighbors: NeighborSet) -> np.ndarray:
    """Describe each candidate using only public geometry and neighbor consensus."""
    paths = neighbors.paths.astype(np.float64)
    distances = neighbors.distances
    candidate_count = len(paths)
    start = np.asarray(parse_cell(query.gap["prefix"][-1]), dtype=np.float64)
    end = np.asarray(parse_cell(query.gap["suffix"][0]), dtype=np.float64)
    displacement = end - start
    prefix = cells_to_array(query.gap["prefix"][-PREFIX_CONTEXT:]).astype(np.float64)
    suffix = cells_to_array(query.gap["suffix"][:SUFFIX_CONTEXT]).astype(np.float64)
    prefix_velocity = (prefix[-1] - prefix[0]) / (len(prefix) - 1)
    suffix_velocity = (suffix[-1] - suffix[0]) / (len(suffix) - 1)
    query_features = np.asarray(
        [
            query.missing_count,
            displacement[0],
            displacement[1],
            np.max(np.abs(displacement)),
            np.sum(np.abs(displacement)),
            query.missing_count + 1 - np.max(np.abs(displacement)),
            start[0],
            start[1],
            end[0],
            end[1],
            prefix_velocity[0],
            prefix_velocity[1],
            suffix_velocity[0],
            suffix_velocity[1],
        ],
        dtype=np.float64,
    )

    context_weights = np.exp(
        -(distances - distances[0]) / CONTEXT_TEMPERATURE
    )
    support = np.empty((candidate_count, query.missing_count), dtype=np.float64)
    for step in range(query.missing_count):
        cell_weights: dict[tuple[int, int], float] = {}
        for rank, weight in enumerate(context_weights):
            cell = (int(paths[rank, step, 0]), int(paths[rank, step, 1]))
            cell_weights[cell] = cell_weights.get(cell, 0.0) + float(weight)
        total_weight = float(np.sum(context_weights))
        for rank in range(candidate_count):
            cell = (int(paths[rank, step, 0]), int(paths[rank, step, 1]))
            support[rank, step] = cell_weights[cell] / total_weight

    mean_path = np.average(paths, axis=0, weights=context_weights)
    hermite_path = cells_to_array(hermite_fallback(query)).astype(np.float64)
    previous_direction = prefix[-1] - prefix[-2]
    next_direction = suffix[1] - suffix[0]
    features = np.empty((candidate_count, 34), dtype=np.float32)

    for rank, path in enumerate(paths):
        full_path = np.vstack((start, path, end))
        steps = np.diff(full_path, axis=0)
        turns = np.diff(steps, axis=0)
        path_cells = {tuple(cell) for cell in path.astype(np.int16)}
        features[rank] = np.concatenate(
            (
                query_features,
                np.asarray(
                    [
                        distances[rank],
                        distances[rank] - distances[0],
                        rank,
                        distances[0],
                        np.mean(support[rank]),
                        np.min(support[rank]),
                        np.max(support[rank]),
                        np.median(support[rank]),
                        np.mean(np.abs(path - mean_path)),
                        np.mean((path - mean_path) ** 2),
                        np.max(np.abs(path - mean_path)),
                        np.mean(np.abs(path - hermite_path)),
                        np.mean((path - hermite_path) ** 2),
                        np.max(np.abs(path - hermite_path)),
                        np.mean(np.abs(turns)),
                        np.sum(np.any(turns != 0, axis=1)),
                        np.sum(np.max(np.abs(steps), axis=1) != 1),
                        len(path) - len(path_cells),
                        np.sum(np.abs(steps[0] - previous_direction)),
                        np.sum(np.abs(steps[-1] - next_direction)),
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return features


def vote_neighbors(
    query: Query, neighbors: NeighborSet, predicted_quality: np.ndarray | None = None
) -> list[str]:
    if predicted_quality is None:
        weights = np.exp(
            -(neighbors.distances - neighbors.distances[0]) / CONTEXT_TEMPERATURE
        )
    else:
        logits = QUALITY_SCALE * (predicted_quality - np.max(predicted_quality))
        logits -= (
            neighbors.distances - neighbors.distances[0]
        ) / RERANK_DISTANCE_TEMPERATURE
        weights = np.exp(np.clip(logits, -50.0, 0.0))

    prediction: list[str] = []
    for step in range(query.missing_count):
        votes: dict[tuple[int, int], float] = {}
        for rank, weight in enumerate(weights):
            cell = (
                int(neighbors.paths[rank, step, 0]),
                int(neighbors.paths[rank, step, 1]),
            )
            votes[cell] = votes.get(cell, 0.0) + float(weight)
        best = max(votes, key=votes.get)
        prediction.append(format_cell(*best))
    return prediction




def metric_components(predicted: Sequence[str], truth: Sequence[str]) -> tuple[float, float, float, float, float]:
    predicted_set, truth_set = set(predicted), set(truth)
    if not predicted_set and not truth_set:
        set_f1 = 1.0
    else:
        set_f1 = 2.0 * len(predicted_set & truth_set) / (len(predicted_set) + len(truth_set))

    lcs = [0] * (len(truth) + 1)
    for predicted_cell in predicted:
        previous_diagonal = 0
        for index, truth_cell in enumerate(truth, start=1):
            old = lcs[index]
            if predicted_cell == truth_cell:
                lcs[index] = previous_diagonal + 1
            elif lcs[index - 1] > lcs[index]:
                lcs[index] = lcs[index - 1]
            previous_diagonal = old
    ordered_lcs = lcs[-1] / max(1, len(truth))
    length_score = math.exp(-abs(len(predicted) - len(truth)) / max(2.0, len(truth) / 2.0))

    if predicted and truth:
        predicted_first, predicted_last = parse_cell(predicted[0]), parse_cell(predicted[-1])
        truth_first, truth_last = parse_cell(truth[0]), parse_cell(truth[-1])
        endpoint_score = 0.5 * (
            math.exp(-sum(abs(a - b) for a, b in zip(predicted_first, truth_first)) / 3.0)
            + math.exp(-sum(abs(a - b) for a, b in zip(predicted_last, truth_last)) / 3.0)
        )
    else:
        endpoint_score = 1.0 if not predicted and not truth else 0.0

    row_score = 0.42 * set_f1 + 0.34 * ordered_lcs + 0.12 * length_score + 0.12 * endpoint_score
    return row_score, set_f1, ordered_lcs, length_score, endpoint_score


def row_score(predicted: Sequence[str], truth: Sequence[str]) -> float:
    return metric_components(predicted, truth)[0]


def candidate_path_cells(path: np.ndarray) -> list[str]:
    return [format_cell(int(row), int(column)) for row, column in path]


def make_reranker_dataset(
    queries: Sequence[Query],
    groups: dict[tuple[int, int, int], TemplateGroup],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[slice], list[NeighborSet | None]]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    slices: list[slice] = []
    neighbor_sets: list[NeighborSet | None] = []
    cursor = 0

    for row_index, query in enumerate(queries):
        assert query.truth is not None and query.source_index is not None
        neighbors = get_neighbors(query, groups, exclude_source=query.source_index)
        neighbor_sets.append(neighbors)
        if neighbors is None:
            slices.append(slice(cursor, cursor))
            continue
        features = candidate_quality_features(query, neighbors)
        targets = np.asarray(
            [
                row_score(candidate_path_cells(path), query.truth)
                for path in neighbors.paths
            ],
            dtype=np.float32,
        )
        feature_blocks.append(features)
        target_blocks.append(targets)
        row_blocks.append(np.full(len(features), row_index, dtype=np.int32))
        slices.append(slice(cursor, cursor + len(features)))
        cursor += len(features)

    if not feature_blocks:
        raise RuntimeError("no candidate paths available to train reranker")
    return (
        np.vstack(feature_blocks),
        np.concatenate(target_blocks),
        np.concatenate(row_blocks),
        slices,
        neighbor_sets,
    )


def make_ranker(random_state: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=400,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=2.0,
        label_gain=list(range(101)),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=min(8, os.cpu_count() or 1),
        random_state=random_state,
    )


def make_cell_classifier(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=3.0,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=min(8, os.cpu_count() or 1),
        random_state=random_state,
    )


def make_row_folds(row_count: int) -> np.ndarray:
    permutation = np.random.RandomState(42).permutation(row_count)
    folds = np.empty(row_count, dtype=np.int8)
    for fold, row_indices in enumerate(np.array_split(permutation, 5)):
        folds[row_indices] = fold
    return folds


def fit_ranker_folds(
    features: np.ndarray,
    targets: np.ndarray,
    sample_rows: np.ndarray,
    slices: Sequence[slice],
    row_folds: np.ndarray,
) -> tuple[list[LGBMRanker], np.ndarray]:
    relevance = np.rint(targets * 100.0).astype(np.int16)
    sample_folds = row_folds[sample_rows]
    oof_quality = np.empty(len(targets), dtype=np.float64)
    models: list[LGBMRanker] = []
    for fold in range(5):
        training_rows = np.flatnonzero(row_folds != fold)
        group_sizes = [
            slices[row].stop - slices[row].start
            for row in training_rows
            if slices[row].stop > slices[row].start
        ]
        train_mask = sample_folds != fold
        validation_mask = ~train_mask
        model = make_ranker(fold)
        model.fit(
            features[train_mask],
            relevance[train_mask],
            group=group_sizes,
        )
        oof_quality[validation_mask] = model.booster_.predict(
            features[validation_mask]
        )
        models.append(model)
    return models, oof_quality


def make_cell_options(
    query: Query,
    neighbors: NeighborSet,
    ranking_quality: np.ndarray,
) -> tuple[CellOptions, np.ndarray | None, np.ndarray | None]:
    start = np.asarray(parse_cell(query.gap["prefix"][-1]), dtype=np.float64)
    end = np.asarray(parse_cell(query.gap["suffix"][0]), dtype=np.float64)
    displacement = end - start
    prefix = cells_to_array(query.gap["prefix"][-PREFIX_CONTEXT:]).astype(
        np.float64
    )
    suffix = cells_to_array(query.gap["suffix"][:SUFFIX_CONTEXT]).astype(
        np.float64
    )
    prefix_velocity = (prefix[-1] - prefix[0]) / (len(prefix) - 1)
    suffix_velocity = (suffix[-1] - suffix[0]) / (len(suffix) - 1)
    public_features = np.asarray(
        [
            query.missing_count,
            displacement[0],
            displacement[1],
            np.max(np.abs(displacement)),
            np.sum(np.abs(displacement)),
            query.missing_count + 1 - np.max(np.abs(displacement)),
            start[0],
            start[1],
            end[0],
            end[1],
            prefix_velocity[0],
            prefix_velocity[1],
            suffix_velocity[0],
            suffix_velocity[1],
            query.point_count,
            query.stroke_count,
            query.gap["stroke_index"],
            len(query.gap["prefix"]),
            len(query.gap["suffix"]),
        ],
        dtype=np.float64,
    )
    hermite_path = cells_to_array(hermite_fallback(query)).astype(np.float64)
    paths = neighbors.paths
    distances = neighbors.distances
    rank_weights = np.exp(
        QUALITY_SCALE * (ranking_quality - np.max(ranking_quality))
        - (distances - distances[0]) / RERANK_DISTANCE_TEMPERATURE
    )

    feature_rows: list[np.ndarray] = []
    cells: list[tuple[int, int]] = []
    steps: list[int] = []
    exact_targets: list[int] | None = [] if query.truth is not None else None
    set_targets: list[int] | None = [] if query.truth is not None else None
    slices: list[slice] = []
    cursor = 0
    truth_coordinates = (
        [parse_cell(cell) for cell in query.truth]
        if query.truth is not None
        else None
    )
    truth_set = set(truth_coordinates) if truth_coordinates is not None else None

    for step in range(query.missing_count):
        unique_cells = list(
            dict.fromkeys(
                (int(cell[0]), int(cell[1])) for cell in paths[:, step]
            )
        )
        mean_cell = np.mean(paths[:, step], axis=0)
        for cell in unique_cells:
            cell_array = np.asarray(cell, dtype=np.float64)
            mask = np.all(paths[:, step] == cell_array, axis=1)
            ranks = np.flatnonzero(mask)
            distance_masses: list[float] = []
            for temperature in (2.0, 4.0, 8.0, 16.0, 32.0):
                weights = np.exp(
                    -(distances - distances[0]) / temperature
                )
                distance_masses.append(
                    float(np.sum(weights[mask]) / np.sum(weights))
                )
            feature = np.concatenate(
                (
                    public_features,
                    np.asarray(
                        [
                            step,
                            step / (query.missing_count + 1),
                            query.missing_count - step,
                        ],
                        dtype=np.float64,
                    ),
                    cell_array,
                    cell_array - start,
                    cell_array - end,
                    cell_array - hermite_path[step],
                    cell_array - mean_cell,
                    np.asarray(
                        [
                            *distance_masses,
                            np.mean(mask),
                            np.sum(rank_weights[mask]) / np.sum(rank_weights),
                            np.sum(mask),
                            np.min(ranks),
                            np.mean(ranks),
                            np.min(distances[mask]),
                            np.mean(distances[mask]),
                            np.max(ranking_quality[mask]),
                            np.mean(ranking_quality[mask]),
                        ],
                        dtype=np.float64,
                    ),
                )
            ).astype(np.float32)
            if len(feature) != 46:
                raise RuntimeError(
                    f"cell option feature width is {len(feature)}, expected 46"
                )
            feature_rows.append(feature)
            cells.append(cell)
            steps.append(step)
            if (
                exact_targets is not None
                and set_targets is not None
                and truth_coordinates is not None
                and truth_set is not None
            ):
                exact_targets.append(int(cell == truth_coordinates[step]))
                set_targets.append(int(cell in truth_set))
        slices.append(slice(cursor, cursor + len(unique_cells)))
        cursor += len(unique_cells)

    options = CellOptions(
        features=np.vstack(feature_rows),
        cells=np.asarray(cells, dtype=np.int16),
        steps=np.asarray(steps, dtype=np.int8),
        slices=slices,
    )
    exact_target_array = (
        np.asarray(exact_targets, dtype=np.int8)
        if exact_targets is not None
        else None
    )
    set_target_array = (
        np.asarray(set_targets, dtype=np.int8)
        if set_targets is not None
        else None
    )
    return options, exact_target_array, set_target_array


def make_cell_dataset(
    queries: Sequence[Query],
    neighbor_sets: Sequence[NeighborSet | None],
    ranking_quality: np.ndarray,
    ranking_slices: Sequence[slice],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[CellOptions | None],
]:
    feature_blocks: list[np.ndarray] = []
    exact_target_blocks: list[np.ndarray] = []
    set_target_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    option_sets: list[CellOptions | None] = []
    for row_index, (query, neighbors) in enumerate(
        zip(queries, neighbor_sets, strict=True)
    ):
        if neighbors is None:
            option_sets.append(None)
            continue
        options, exact_targets, set_targets = make_cell_options(
            query, neighbors, ranking_quality[ranking_slices[row_index]]
        )
        assert exact_targets is not None and set_targets is not None
        option_sets.append(options)
        feature_blocks.append(options.features)
        exact_target_blocks.append(exact_targets)
        set_target_blocks.append(set_targets)
        row_blocks.append(
            np.full(len(exact_targets), row_index, dtype=np.int32)
        )
    return (
        np.vstack(feature_blocks),
        np.concatenate(exact_target_blocks),
        np.concatenate(set_target_blocks),
        np.concatenate(row_blocks),
        option_sets,
    )


def blend_cell_probabilities(
    neighbors: NeighborSet,
    exact_probability: np.ndarray,
    set_probability: np.ndarray,
) -> np.ndarray:
    if neighbors.distances[0] < BLEND_MIN_CONTEXT_DISTANCE:
        return exact_probability
    return np.exp(
        BLEND_EXACT_WEIGHT * np.log(np.clip(exact_probability, 1e-6, 1.0))
        + (1.0 - BLEND_EXACT_WEIGHT)
        * np.log(np.clip(set_probability, 1e-6, 1.0))
    )


def decode_cell_options(
    query: Query, options: CellOptions, probabilities: np.ndarray
) -> list[str]:
    start = np.asarray(parse_cell(query.gap["prefix"][-1]), dtype=np.int16)
    end = np.asarray(parse_cell(query.gap["suffix"][0]), dtype=np.int16)
    unary = [
        np.log(np.clip(probabilities[step_slice], 1e-6, 1.0))
        for step_slice in options.slices
    ]

    first_cells = options.cells[options.slices[0]]
    previous = np.asarray(
        [
            unary[0][index]
            - CONNECTIVITY_PENALTY
            * abs(int(np.max(np.abs(cell - start))) - 1)
            for index, cell in enumerate(first_cells)
        ],
        dtype=np.float64,
    )
    back_pointers: list[np.ndarray] = []

    for step in range(1, query.missing_count):
        prior_cells = options.cells[options.slices[step - 1]]
        current_cells = options.cells[options.slices[step]]
        current = np.full(len(current_cells), -1e9, dtype=np.float64)
        back = np.zeros(len(current_cells), dtype=np.int16)
        for current_index, cell in enumerate(current_cells):
            transition = np.asarray(
                [
                    previous[prior_index]
                    - CONNECTIVITY_PENALTY
                    * abs(int(np.max(np.abs(cell - prior_cell))) - 1)
                    for prior_index, prior_cell in enumerate(prior_cells)
                ]
            )
            best_prior = int(np.argmax(transition))
            current[current_index] = (
                transition[best_prior] + unary[step][current_index]
            )
            back[current_index] = best_prior
        previous = current
        back_pointers.append(back)

    last_cells = options.cells[options.slices[-1]]
    final_scores = np.asarray(
        [
            previous[index]
            - CONNECTIVITY_PENALTY
            * abs(int(np.max(np.abs(end - cell))) - 1)
            for index, cell in enumerate(last_cells)
        ]
    )
    current_index = int(np.argmax(final_scores))
    selected = [current_index]
    for back in reversed(back_pointers):
        current_index = int(back[current_index])
        selected.append(current_index)
    selected.reverse()

    return [
        format_cell(
            int(options.cells[options.slices[step]][selected[step], 0]),
            int(options.cells[options.slices[step]][selected[step], 1]),
        )
        for step in range(query.missing_count)
    ]


def validate_prediction(query: Query, prediction: Sequence[str]) -> None:
    if len(prediction) != query.missing_count:
        raise ValueError(
            f"{query.row_id}: predicted {len(prediction)} cells, expected {query.missing_count}"
        )
    for cell in prediction:
        parse_cell(cell)


def write_submission(path: Path, queries: Sequence[Query], predictions: Sequence[Sequence[str]]) -> None:
    if len(queries) != len(predictions):
        raise ValueError("query/prediction row-count mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer_json"])
        writer.writeheader()
        for query, prediction in zip(queries, predictions, strict=True):
            validate_prediction(query, prediction)
            answer_json = json.dumps({"hidden_cells": list(prediction)}, separators=(",", ":"))
            writer.writerow({"id": query.row_id, "answer_json": answer_json})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_validation(train_rows: Sequence[dict], train_queries: Sequence[Query]) -> None:
    strokes = reconstruct_training_strokes(train_rows, train_queries)
    started = time.perf_counter()
    groups, candidate_count = build_template_library(strokes, train_queries)
    path_features, path_targets, path_rows, path_slices, neighbor_sets = (
        make_reranker_dataset(train_queries, groups)
    )
    row_folds = make_row_folds(len(train_queries))
    _, oof_ranking = fit_ranker_folds(
        path_features, path_targets, path_rows, path_slices, row_folds
    )
    (
        cell_features,
        exact_targets,
        set_targets,
        cell_rows,
        option_sets,
    ) = make_cell_dataset(
        train_queries, neighbor_sets, oof_ranking, path_slices
    )
    print(
        f"Built {candidate_count:,} templates, {len(path_targets):,} path examples, "
        f"and {len(exact_targets):,} cell options in "
        f"{time.perf_counter() - started:.1f}s"
    )
    del groups
    gc.collect()

    cell_folds = row_folds[cell_rows]
    oof_exact_probability = np.empty(len(exact_targets), dtype=np.float64)
    oof_set_probability = np.empty(len(set_targets), dtype=np.float64)
    for fold in range(5):
        train_mask = cell_folds != fold
        validation_mask = ~train_mask
        exact_classifier = make_cell_classifier(fold)
        exact_classifier.fit(
            cell_features[train_mask], exact_targets[train_mask]
        )
        oof_exact_probability[validation_mask] = (
            exact_classifier.booster_.predict(cell_features[validation_mask])
        )
        set_classifier = make_cell_classifier(fold)
        set_classifier.fit(
            cell_features[train_mask], set_targets[train_mask]
        )
        oof_set_probability[validation_mask] = (
            set_classifier.booster_.predict(cell_features[validation_mask])
        )
        print(f"Trained exact/set cell decoders fold {fold + 1}/5")

    totals = np.zeros(5, dtype=np.float64)
    cursor = 0
    for index, query in enumerate(train_queries):
        assert query.truth is not None
        options = option_sets[index]
        if options is None:
            prediction = hermite_fallback(query)
        else:
            count = len(options.features)
            neighbors = neighbor_sets[index]
            assert neighbors is not None
            exact_probability = oof_exact_probability[cursor : cursor + count]
            set_probability = oof_set_probability[cursor : cursor + count]
            probabilities = blend_cell_probabilities(
                neighbors, exact_probability, set_probability
            )
            prediction = decode_cell_options(query, options, probabilities)
            cursor += count
        validate_prediction(query, prediction)
        totals += metric_components(prediction, query.truth)
    if cursor != len(oof_exact_probability):
        raise RuntimeError("cell-option validation cursor mismatch")
    means = totals / len(train_queries)
    print(
        "Distance-gated metric-aware connected decoding: "
        f"score={means[0]:.6f} set_f1={means[1]:.6f} ordered_lcs={means[2]:.6f} "
        f"length={means[3]:.6f} endpoints={means[4]:.6f} rows={len(train_queries)}"
    )


def run_submission(
    base: Path,
    train_rows: Sequence[dict],
    train_queries: Sequence[Query],
    test_queries: Sequence[Query],
) -> None:
    strokes = reconstruct_training_strokes(train_rows, train_queries)

    started = time.perf_counter()
    training_groups, validation_candidate_count = build_template_library(
        strokes, train_queries
    )
    path_features, path_targets, path_rows, path_slices, training_neighbors = (
        make_reranker_dataset(train_queries, training_groups)
    )
    row_folds = make_row_folds(len(train_queries))
    rankers, oof_ranking = fit_ranker_folds(
        path_features, path_targets, path_rows, path_slices, row_folds
    )
    (
        cell_features,
        exact_targets,
        set_targets,
        _,
        _,
    ) = make_cell_dataset(
        train_queries, training_neighbors, oof_ranking, path_slices
    )
    exact_classifier = make_cell_classifier(42)
    exact_classifier.fit(cell_features, exact_targets)
    set_classifier = make_cell_classifier(42)
    set_classifier.fit(cell_features, set_targets)
    print(
        f"Trained five path rankers and the exact/set connected cell decoders on "
        f"{len(path_targets):,} path / {len(exact_targets):,} cell examples from "
        f"{validation_candidate_count:,} templates in "
        f"{time.perf_counter() - started:.1f}s"
    )
    del (
        training_groups,
        path_features,
        path_targets,
        path_rows,
        training_neighbors,
        oof_ranking,
        cell_features,
        exact_targets,
        set_targets,
    )
    gc.collect()

    started = time.perf_counter()
    groups, candidate_count = build_template_library(strokes, test_queries)
    packed_bytes = sum(
        group.features.nbytes + group.paths.nbytes + group.sources.nbytes
        for group in groups.values()
    )
    print(
        f"Built {candidate_count:,} test templates ({packed_bytes / 1_000_000:.1f} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )

    predictions: list[list[str]] = []
    for query in test_queries:
        neighbors = get_neighbors(query, groups)
        if neighbors is None:
            prediction = hermite_fallback(query)
        else:
            features = candidate_quality_features(query, neighbors)
            ranking_quality = np.mean(
                [
                    ranker.booster_.predict(features)
                    for ranker in rankers
                ],
                axis=0,
            )
            options, _, _ = make_cell_options(
                query, neighbors, ranking_quality
            )
            exact_probability = exact_classifier.booster_.predict(
                options.features
            )
            set_probability = set_classifier.booster_.predict(
                options.features
            )
            probabilities = blend_cell_probabilities(
                neighbors, exact_probability, set_probability
            )
            prediction = decode_cell_options(query, options, probabilities)
        predictions.append(prediction)

    output = base / "working" / "submission.csv"
    write_submission(output, test_queries, predictions)
    print(f"Wrote {len(predictions):,} predictions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run leave-one-row-out validation instead of creating a submission",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    public = base / "dataset" / "public"
    train_rows, train_queries = read_dataset(public / "train.csv", require_answers=True)
    if args.validate:
        run_validation(train_rows, train_queries)
        return
    _, test_queries = read_dataset(public / "test.csv", require_answers=False)
    run_submission(base, train_rows, train_queries, test_queries)


if __name__ == "__main__":
    main()
