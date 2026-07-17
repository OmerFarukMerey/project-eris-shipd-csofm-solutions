"""Cross-Locale Repair-Window Propagation — self-contained trained solution.

The script learns two complementary alignment-aware transformers from train.csv.
A train-only route holdout selects training length, ensemble weight, and decoding
calibration. Test rows are only transformed and predicted.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import gc
import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np
np.seterr(divide="ignore", invalid="ignore", over="ignore")
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
N_WIN = 16
N_BUCKET = 512
MAX_COUNT = 10
CAT_COLS = ["anchor_locale", "target_locale", "anchor_error_family"]
SKETCH_COLS = [
    "source_sketch",
    "anchor_draft_sketch",
    "anchor_repaired_sketch",
    "target_draft_sketch",
]
NN_MAX_EPOCHS = 12
MC_SAMPLES = 300
CALIBRATION_SAMPLES = 80

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)



def gaussian_kernel(sigma):
    d = np.abs(np.arange(N_WIN)[:, None] - np.arange(N_WIN)[None, :])
    return np.exp(-(d**2) / (2 * sigma**2)).astype(np.float32)


G1 = gaussian_kernel(1.0)
G2 = gaussian_kernel(2.0)
BAND2 = (
    np.abs(np.arange(N_WIN)[:, None] - np.arange(N_WIN)[None, :]) <= 2
).astype(np.float32)
FEATURE_NAMES = None


# ---------------- parsing and train-fitted evidence weights ----------------
def mask_array(value):
    return np.fromiter((int(c) for c in str(value).zfill(N_WIN)), np.int8, N_WIN)


def sketch_codes(value):
    """Parse one sketch into four bucket ids per window."""
    out = np.empty((N_WIN, 4), dtype=np.int64)
    fill = np.zeros(N_WIN, dtype=np.int8)
    for token in value.split():
        window = int(token[1:3])
        slot = fill[window]
        if slot >= 4:
            raise ValueError("sketch has more than four codes in a window")
        out[window, slot] = int(token[4:])
        fill[window] += 1
    if not np.all(fill == 4):
        raise ValueError("sketch must have exactly four codes per window")
    return out


def code_counts(codes):
    out = np.zeros((N_WIN, N_BUCKET), dtype=np.float32)
    np.add.at(
        out,
        (np.repeat(np.arange(N_WIN), 4), codes.reshape(-1)),
        1.0,
    )
    return out


def code_similarity(left_codes, right_codes):
    """Count equal bucket ids for every pair of windows without dense matmul."""
    return (
        left_codes[:, None, :, None] == right_codes[None, :, None, :]
    ).sum(axis=(2, 3), dtype=np.int16).astype(np.float32)


def weighted_code_similarity(left_codes, right_codes, target_weights):
    matches = left_codes[:, None, :, None] == right_codes[None, :, None, :]
    right_weights = target_weights[
        np.arange(N_WIN)[:, None], right_codes
    ]
    return (matches * right_weights[None, :, None, :]).sum(
        axis=(2, 3), dtype=np.float32
    )


def code_overlap(target_codes, evidence):
    return evidence[target_codes].sum(axis=1, dtype=np.float32)


def weighted_code_overlap(target_codes, evidence, target_weights):
    weights = target_weights[np.arange(N_WIN)[:, None], target_codes]
    return (weights * evidence[target_codes]).sum(axis=1, dtype=np.float32)


def token_pairs(value):
    return {(int(t[1:3]), int(t[4:])) for t in value.split()}


def fit_code_weights(train_df, smoothing=8.0):
    """Learn collision reliability from training sketches only, without labels."""
    reference_cols = SKETCH_COLS[:3]
    occurrence_target = np.zeros((N_WIN, N_BUCKET), dtype=np.float64)
    occurrence_reference = {
        col: np.zeros((N_WIN, N_BUCKET), dtype=np.float64)
        for col in reference_cols
    }
    matches = {
        col: np.zeros((N_WIN, N_BUCKET), dtype=np.float64)
        for col in reference_cols
    }

    cols = reference_cols + ["target_draft_sketch"]
    for row in train_df[cols].itertuples(index=False, name=None):
        target = token_pairs(row[-1])
        for window, bucket in target:
            occurrence_target[window, bucket] += 1.0
        for col, value in zip(reference_cols, row[:-1]):
            reference = token_pairs(value)
            for window, bucket in reference:
                occurrence_reference[col][window, bucket] += 1.0
            for window, bucket in target & reference:
                matches[col][window, bucket] += 1.0

    n_rows = len(train_df)
    weights = {}
    for col, key in zip(reference_cols, ["S", "AD", "AR"]):
        random_match = (occurrence_reference[col] + 1.0) / (n_rows + 2.0)
        base = matches[col].sum() / max(occurrence_target.sum(), 1.0)
        observed = (matches[col] + smoothing * base) / (
            occurrence_target + smoothing
        )
        weights[key] = np.clip(
            np.log(np.maximum(observed, 1e-12) / random_match), 0.0, None
        ).astype(np.float32)
    return weights


# ---------------- row-local alignment features ----------------
def row_inputs(row, code_weights):
    code_stack = np.stack([sketch_codes(row[col]) for col in SKETCH_COLS])
    source, anchor_draft, anchor_repaired, target_draft = [
        code_counts(codes) for codes in code_stack
    ]
    anchor_mask = mask_array(row["anchor_repair_windows"]).astype(np.float32)

    sim_anchor_target = code_similarity(code_stack[1], code_stack[3])
    sim_source_target = code_similarity(code_stack[0], code_stack[3])
    sim_source_anchor = code_similarity(code_stack[0], code_stack[1])
    sim_repaired_target = code_similarity(code_stack[2], code_stack[3])
    eps = 1e-6
    positions = np.arange(N_WIN, dtype=np.float32)

    intersection = np.minimum(anchor_draft, anchor_repaired).sum(1)
    union = np.maximum(anchor_draft, anchor_repaired).sum(1)
    edit_distance = 1.0 - intersection / np.maximum(union, 1.0)
    edited = anchor_mask > 0.5
    edit_count = int(edited.sum())
    edit_positions = positions[edited] if edit_count else np.array([7.5], np.float32)
    nearest_edit = np.abs(positions[:, None] - edit_positions[None, :]).min(1)

    blurred_1 = G1 @ anchor_mask
    blurred_2 = G2 @ anchor_mask
    soft_mask = np.clip(
        anchor_mask
        + 0.35 * (np.roll(anchor_mask, 1) + np.roll(anchor_mask, -1)),
        0.0,
        1.0,
    )
    soft_mask[0] = np.clip(anchor_mask[0] + 0.35 * anchor_mask[1], 0.0, 1.0)
    soft_mask[-1] = np.clip(anchor_mask[-1] + 0.35 * anchor_mask[-2], 0.0, 1.0)

    def propagate(similarity, values):
        column_mass = similarity.sum(0)
        propagated = (similarity * values[:, None]).sum(0) / (column_mass + eps)
        return propagated, column_mass

    prop_anchor, mass_anchor = propagate(sim_anchor_target, anchor_mask)
    prop_anchor_blur, _ = propagate(sim_anchor_target, soft_mask)
    prop_repaired, mass_repaired = propagate(sim_repaired_target, anchor_mask)
    prop_edit_distance, _ = propagate(sim_anchor_target, edit_distance)

    source_mask = (sim_source_anchor * anchor_mask[None, :]).sum(1) / (
        sim_source_anchor.sum(1) + eps
    )
    prop_source, mass_source = propagate(sim_source_target, source_mask)
    prop_source_blur, _ = propagate(sim_source_target, G1 @ source_mask)

    negative = -1.0
    max_anchor_edited = (
        np.where(edited[:, None], sim_anchor_target, negative).max(0)
        if edit_count
        else np.full(N_WIN, negative, np.float32)
    )
    max_anchor_unedited = (
        np.where(~edited[:, None], sim_anchor_target, negative).max(0)
        if edit_count < N_WIN
        else np.full(N_WIN, negative, np.float32)
    )
    max_repaired_edited = (
        np.where(edited[:, None], sim_repaired_target, negative).max(0)
        if edit_count
        else np.full(N_WIN, negative, np.float32)
    )

    best_anchor = sim_anchor_target.argmax(0)
    best_valid = sim_anchor_target.max(0) > 0
    best_distance = np.abs(best_anchor[:, None] - edit_positions[None, :]).min(1)
    best_distance = np.where(best_valid, best_distance, 8.0)
    best_is_edited = np.where(best_valid, anchor_mask[best_anchor], 0.0)

    similarity_mass = sim_anchor_target.sum(0)
    expected_anchor = (sim_anchor_target * positions[:, None]).sum(0) / (
        similarity_mass + eps
    )
    expected_distance = np.where(
        similarity_mass > 0,
        np.abs(expected_anchor[:, None] - edit_positions[None, :]).min(1),
        8.0,
    )

    if edit_count:
        removed = np.clip(anchor_draft - anchor_repaired, 0, None)[edited].sum(0)
        added = np.clip(anchor_repaired - anchor_draft, 0, None)[edited].sum(0)
        edit_draft_codes = anchor_draft[edited].sum(0)
        edit_source_codes = source[edited].sum(0)
    else:
        removed = added = edit_draft_codes = edit_source_codes = np.zeros(
            N_BUCKET, np.float32
        )
    overlap_removed = code_overlap(code_stack[3], removed)
    overlap_added = code_overlap(code_stack[3], added)
    overlap_edit_draft = code_overlap(code_stack[3], edit_draft_codes)
    overlap_edit_source = code_overlap(code_stack[3], edit_source_codes)

    diagonal_anchor = np.diag(sim_anchor_target)
    diagonal_source = np.diag(sim_source_target)
    diagonal_repaired = np.diag(sim_repaired_target)
    target_self = code_similarity(code_stack[3], code_stack[3])
    neighbor_previous = np.concatenate([[0], np.diag(target_self, -1)]).astype(
        np.float32
    )
    neighbor_next = np.concatenate([np.diag(target_self, 1), [0]]).astype(
        np.float32
    )
    duplicate_in_window = (target_draft.max(1) > 1).astype(np.float32)

    def shift(values, amount):
        out = np.zeros(N_WIN, np.float32)
        if amount > 0:
            out[amount:] = values[:-amount]
        elif amount < 0:
            out[:amount] = values[-amount:]
        else:
            out[:] = values
        return out

    row_mass_anchor = sim_anchor_target.sum(1, keepdims=True)
    push_anchor = (
        sim_anchor_target / (row_mass_anchor + eps) * anchor_mask[:, None]
    ).sum(0)
    row_mass_repaired = sim_repaired_target.sum(1, keepdims=True)
    push_repaired = (
        sim_repaired_target / (row_mass_repaired + eps) * anchor_mask[:, None]
    ).sum(0)
    push_anchor_blur = G1 @ push_anchor
    row_mass_source = sim_source_target.sum(1, keepdims=True)
    push_source = (
        sim_source_target / (row_mass_source + eps) * source_mask[:, None]
    ).sum(0)

    band_source_previous = np.concatenate(
        [[0], np.diag(sim_source_target, 1)]
    ).astype(np.float32)
    band_source_next = np.concatenate(
        [np.diag(sim_source_target, -1), [0]]
    ).astype(np.float32)
    combined = (
        prop_anchor_blur + prop_source_blur + push_anchor_blur + 0.5 * prop_repaired
    )

    def rank_feature(values):
        return np.argsort(np.argsort(values)).astype(np.float32)

    def z_feature(values):
        return ((values - values.mean()) / (values.std() + eps)).astype(np.float32)

    features = {
        "position": positions,
        "anchor": anchor_mask,
        "anchor_m1": shift(anchor_mask, 1),
        "anchor_p1": shift(anchor_mask, -1),
        "anchor_m2": shift(anchor_mask, 2),
        "anchor_p2": shift(anchor_mask, -2),
        "nearest_edit": nearest_edit,
        "anchor_blur1": blurred_1,
        "anchor_blur2": blurred_2,
        "edit_distance": edit_distance.astype(np.float32),
        "prop_anchor": prop_anchor,
        "prop_anchor_blur": prop_anchor_blur,
        "prop_repaired": prop_repaired,
        "prop_edit_distance": prop_edit_distance,
        "prop_source": prop_source,
        "prop_source_blur": prop_source_blur,
        "prop_anchor_m1": shift(prop_anchor, 1),
        "prop_anchor_p1": shift(prop_anchor, -1),
        "mass_anchor": mass_anchor,
        "mass_source": mass_source,
        "mass_repaired": mass_repaired,
        "max_anchor_edited": max_anchor_edited,
        "max_anchor_unedited": max_anchor_unedited,
        "max_repaired_edited": max_repaired_edited,
        "best_distance": best_distance,
        "best_is_edited": best_is_edited,
        "expected_distance": expected_distance,
        "similarity_mass": similarity_mass,
        "overlap_removed": overlap_removed,
        "overlap_added": overlap_added,
        "overlap_edit_draft": overlap_edit_draft,
        "overlap_edit_source": overlap_edit_source,
        "diagonal_anchor": diagonal_anchor,
        "diagonal_source": diagonal_source,
        "diagonal_repaired": diagonal_repaired,
        "neighbor_previous": neighbor_previous,
        "neighbor_next": neighbor_next,
        "duplicate_in_window": duplicate_in_window,
        "push_anchor": push_anchor,
        "push_repaired": push_repaired,
        "push_anchor_blur": push_anchor_blur,
        "push_source": push_source,
        "mass_change": mass_anchor - mass_repaired,
        "diagonal_change": diagonal_anchor - diagonal_repaired,
        "prop_change": prop_anchor - prop_repaired,
        "edit_distance_m1": shift(edit_distance.astype(np.float32), 1),
        "edit_distance_p1": shift(edit_distance.astype(np.float32), -1),
        "band_source_previous": band_source_previous,
        "band_source_next": band_source_next,
        "combined": combined,
        "rank_combined": rank_feature(combined),
        "z_combined": z_feature(combined),
        "rank_anchor_blur": rank_feature(blurred_1),
        "rank_source_blur": rank_feature(prop_source_blur),
        "z_prop_anchor": z_feature(prop_anchor_blur),
        "z_push_anchor": z_feature(push_anchor_blur),
        "edit_count": np.full(N_WIN, edit_count, np.float32),
        "edit_mean": np.full(N_WIN, edit_positions.mean(), np.float32),
        "edit_min": np.full(N_WIN, edit_positions.min(), np.float32),
        "edit_max": np.full(N_WIN, edit_positions.max(), np.float32),
        "edit_distance_sum": np.full(N_WIN, edit_distance.sum(), np.float32),
        "diagonal_anchor_mean": np.full(N_WIN, diagonal_anchor.mean(), np.float32),
        "diagonal_source_mean": np.full(N_WIN, diagonal_source.mean(), np.float32),
    }

    # Train-fitted collision reliability; all operations remain within this row.
    weighted_source = weighted_code_similarity(
        code_stack[0], code_stack[3], code_weights["S"]
    )
    weighted_anchor = weighted_code_similarity(
        code_stack[1], code_stack[3], code_weights["AD"]
    )
    weighted_repaired = weighted_code_similarity(
        code_stack[2], code_stack[3], code_weights["AR"]
    )
    band_anchor = weighted_anchor * BAND2
    band_repaired = weighted_repaired * BAND2
    band_source = weighted_source * BAND2
    weighted_prop_anchor = (band_anchor * soft_mask[:, None]).sum(0) / (
        band_anchor.sum(0) + eps
    )
    weighted_prop_repaired = (band_repaired * soft_mask[:, None]).sum(0) / (
        band_repaired.sum(0) + eps
    )
    weighted_push_anchor = (
        band_anchor / (band_anchor.sum(1, keepdims=True) + eps) * anchor_mask[:, None]
    ).sum(0)
    source_anchor_band = sim_source_anchor * BAND2
    source_mask_band = (source_anchor_band * anchor_mask[None, :]).sum(1) / (
        source_anchor_band.sum(1) + eps
    )
    weighted_prop_source = (
        band_source * (G1 @ source_mask_band)[:, None]
    ).sum(0) / (band_source.sum(0) + eps)
    weighted_push_source = (
        band_source / (band_source.sum(1, keepdims=True) + eps)
        * source_mask_band[:, None]
    ).sum(0)
    weighted_combined = (
        weighted_prop_anchor + weighted_prop_source + G1 @ weighted_push_anchor
    )
    features.update(
        {
            "weighted_diagonal_source": np.diag(weighted_source),
            "weighted_diagonal_anchor": np.diag(weighted_anchor),
            "weighted_diagonal_repaired": np.diag(weighted_repaired),
            "weighted_source_m1": np.concatenate(
                [[0], np.diag(weighted_source, 1)]
            ).astype(np.float32),
            "weighted_source_p1": np.concatenate(
                [np.diag(weighted_source, -1), [0]]
            ).astype(np.float32),
            "weighted_prop_anchor": weighted_prop_anchor,
            "weighted_prop_repaired": weighted_prop_repaired,
            "weighted_prop_source": weighted_prop_source,
            "weighted_push_anchor": weighted_push_anchor,
            "weighted_push_source": weighted_push_source,
            "weighted_diagonal_change": np.diag(weighted_anchor)
            - np.diag(weighted_repaired),
            "weighted_overlap_removed": weighted_code_overlap(
                code_stack[3], removed, code_weights["AD"]
            ),
            "weighted_overlap_added": weighted_code_overlap(
                code_stack[3], added, code_weights["AR"]
            ),
            "weighted_overlap_source": weighted_code_overlap(
                code_stack[3], edit_source_codes, code_weights["S"]
            ),
            "weighted_combined": weighted_combined,
            "rank_weighted_combined": rank_feature(weighted_combined),
            "z_weighted_combined": z_feature(weighted_combined),
            "weighted_mass_source": band_source.sum(0),
            "weighted_mass_anchor": band_anchor.sum(0),
            "weighted_source_mean": np.full(
                N_WIN, np.diag(weighted_source).mean(), np.float32
            ),
        }
    )

    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = list(features)
    feature_matrix = np.stack([features[name] for name in FEATURE_NAMES], axis=1)
    similarity_stack = np.stack(
        [
            sim_anchor_target,
            sim_source_target,
            sim_repaired_target,
            sim_source_anchor,
            code_similarity(code_stack[1], code_stack[2]),
        ]
    ).astype(np.float32)
    return feature_matrix, similarity_stack, code_stack


def build_inputs(df, code_weights):
    feature_rows = []
    similarity_rows = []
    code_rows = []
    for _, row in df.iterrows():
        features, similarities, codes = row_inputs(row, code_weights)
        feature_rows.append(features)
        similarity_rows.append(similarities)
        code_rows.append(codes)
    return (
        np.stack(feature_rows).astype(np.float32),
        np.stack(similarity_rows).astype(np.float32),
        np.stack(code_rows).astype(np.int64),
    )


# ---------------- categories, validation, and metric ----------------
def encode_categories(fit_df, apply_df):
    fit_codes = []
    apply_codes = []
    sizes = []
    for col in CAT_COLS:
        categories = sorted(fit_df[col].unique())
        mapping = {value: i for i, value in enumerate(categories)}
        unknown = len(categories)
        fit_codes.append(fit_df[col].map(mapping).fillna(unknown).to_numpy())
        apply_codes.append(apply_df[col].map(mapping).fillna(unknown).to_numpy())
        sizes.append(len(categories) + 1)
    return (
        np.stack(fit_codes, axis=1).astype(np.int64),
        np.stack(apply_codes, axis=1).astype(np.int64),
        sizes,
    )


def locale_weights(df):
    frequencies = df["target_locale"].value_counts()
    return (
        len(df)
        / (len(frequencies) * frequencies[df["target_locale"]].to_numpy())
    ).astype(np.float32)


def route_holdout_mask(train_df):
    """Hold out one directed route per anchor and target, mirroring evaluation."""
    anchors = sorted(train_df["anchor_locale"].unique())
    targets = sorted(train_df["target_locale"].unique())
    routes = set(zip(train_df["anchor_locale"], train_df["target_locale"]))
    route_sizes = train_df.groupby(["anchor_locale", "target_locale"]).size()
    candidates = []
    desired_rows = 0.2 * len(train_df)
    for permutation in itertools.permutations(targets):
        candidate = tuple(zip(anchors, permutation))
        if all(route in routes for route in candidate):
            rows = sum(int(route_sizes.loc[route]) for route in candidate)
            candidates.append((abs(rows - desired_rows), candidate))
    if not candidates:
        raise ValueError("could not construct a route-disjoint validation split")
    validation_routes = set(min(candidates, key=lambda item: (item[0], item[1]))[1])
    mask = np.fromiter(
        (
            (anchor, target) in validation_routes
            for anchor, target in zip(
                train_df["anchor_locale"], train_df["target_locale"]
            )
        ),
        dtype=bool,
        count=len(train_df),
    )
    return mask, sorted(validation_routes)


def adjacent_matches(predicted, truth):
    used = set()
    matches = 0
    for position in sorted(predicted):
        for neighbor in (position - 1, position + 1):
            if neighbor in truth and neighbor not in used:
                used.add(neighbor)
                matches += 1
                break
    return matches


def row_rwu(predicted_mask, true_mask):
    predicted = set(np.flatnonzero(predicted_mask))
    truth = set(np.flatnonzero(true_mask))
    if not predicted or not truth:
        return 0.0
    exact = predicted & truth
    credit = len(exact) + 0.35 * adjacent_matches(predicted - exact, truth - exact)
    weighted_f1 = 2.0 * credit / (len(predicted) + len(truth))
    budget = min(len(predicted), len(truth)) / max(len(predicted), len(truth))
    return 0.8 * weighted_f1 + 0.2 * budget


def locale_balanced_score(predictions, truth, target_locales):
    scores = np.fromiter(
        (row_rwu(pred, true) for pred, true in zip(predictions, truth)),
        dtype=np.float64,
        count=len(truth),
    )
    locales = np.asarray(target_locales)
    return float(np.mean([scores[locales == loc].mean() for loc in np.unique(locales)]))


def topk_decode(probabilities):
    output = np.zeros_like(probabilities, dtype=np.int8)
    for i, row in enumerate(probabilities):
        count = int(np.clip(np.rint(row.sum()), 1, N_WIN))
        output[i, np.argpartition(-row, count - 1)[:count]] = 1
    return output




# ---------------- neural model ----------------
class RepairNet(nn.Module):
    def __init__(self, feature_count, category_sizes, use_codes, width=128):
        super().__init__()
        self.use_codes = use_codes
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_count, width), nn.GELU(), nn.Linear(width, width)
        )
        self.similarity_embedding = nn.Sequential(
            nn.Linear(5 * 32, width), nn.GELU(), nn.Linear(width, width)
        )
        if use_codes:
            self.bucket_embedding = nn.Embedding(N_BUCKET, 32)
            self.code_projection = nn.Sequential(
                nn.Linear(4 * 32, width), nn.GELU(), nn.Linear(width, width)
            )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size, 16) for size in category_sizes]
        )
        self.category_projection = nn.Linear(16 * len(category_sizes), width)
        self.position_embedding = nn.Embedding(N_WIN, width)
        layer = nn.TransformerEncoderLayer(
            width,
            nhead=4,
            dim_feedforward=4 * width,
            dropout=0.15,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.window_head = nn.Linear(width, 1)
        self.count_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, MAX_COUNT)
        )

    def forward(self, features, similarities, codes, categories):
        batch = features.shape[0]
        columns = similarities.permute(0, 3, 1, 2).reshape(batch, N_WIN, 80)
        rows = similarities.permute(0, 2, 1, 3).reshape(batch, N_WIN, 80)
        similarity_state = self.similarity_embedding(torch.cat([columns, rows], dim=-1))
        category_state = self.category_projection(
            torch.cat(
                [
                    embedding(categories[:, i])
                    for i, embedding in enumerate(self.category_embeddings)
                ],
                dim=-1,
            )
        )
        state = (
            self.feature_embedding(features)
            + similarity_state
            + category_state[:, None, :]
            + self.position_embedding.weight[None, :, :]
        )
        if self.use_codes:
            code_state = (
                self.bucket_embedding(codes)
                .mean(dim=3)
                .permute(0, 2, 1, 3)
                .reshape(batch, N_WIN, 128)
            )
            state = state + self.code_projection(code_state)
        state = self.transformer(state)
        return self.window_head(state).squeeze(-1), self.count_head(state.mean(1))


def normalization_stats(features, similarities):
    flat = features.reshape(-1, features.shape[-1])
    feature_mean = flat.mean(0)
    feature_std = flat.std(0) + 1e-6
    return feature_mean, feature_std, similarities.mean(), similarities.std() + 1e-6


def predict_nn(model, stats, features, similarities, codes, categories, batch=1024):
    feature_mean, feature_std, similarity_mean, similarity_std = stats
    probabilities = []
    count_probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch):
            stop = start + batch
            x = torch.as_tensor(
                (features[start:stop] - feature_mean) / feature_std,
                device=DEVICE,
            )
            s = torch.as_tensor(
                (similarities[start:stop] - similarity_mean) / similarity_std,
                device=DEVICE,
            )
            c = torch.as_tensor(codes[start:stop], device=DEVICE)
            k = torch.as_tensor(categories[start:stop], device=DEVICE)
            logits, count_logits = model(x, s, c, k)
            probabilities.append(torch.sigmoid(logits).cpu())
            count_probabilities.append(F.softmax(count_logits, dim=-1).cpu())
    return torch.cat(probabilities).numpy(), torch.cat(count_probabilities).numpy()


def train_nn(
    features,
    similarities,
    codes,
    categories,
    truth,
    row_weights,
    category_sizes,
    epochs,
    use_codes=True,
    validation=None,
):
    stats = normalization_stats(features, similarities)
    feature_mean, feature_std, similarity_mean, similarity_std = stats
    x = torch.as_tensor((features - feature_mean) / feature_std, device=DEVICE)
    s = torch.as_tensor(
        (similarities - similarity_mean) / similarity_std, device=DEVICE
    )
    c = torch.as_tensor(codes, device=DEVICE)
    k = torch.as_tensor(categories, device=DEVICE)
    y = torch.as_tensor(truth, dtype=torch.float32, device=DEVICE)
    count_truth = torch.clamp(y.sum(1), 1, MAX_COUNT).long() - 1
    weights = torch.as_tensor(row_weights, device=DEVICE)

    torch.manual_seed(SEED)
    model = RepairNet(features.shape[-1], category_sizes, use_codes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    batch = 256
    steps_per_epoch = (len(features) + batch - 1) // batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=8e-4,
        total_steps=NN_MAX_EPOCHS * steps_per_epoch,
    )
    generator = torch.Generator().manual_seed(SEED)
    best_score = -1.0
    best_epoch = epochs - 1
    best_state = None
    best_validation = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(features), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(features), batch):
            indices = permutation[start : start + batch].to(DEVICE)
            optimizer.zero_grad()
            logits, count_logits = model(x[indices], s[indices], c[indices], k[indices])
            bce = F.binary_cross_entropy_with_logits(
                logits, y[indices], reduction="none"
            ).mean(1)
            probability = torch.sigmoid(logits)
            dice = 1.0 - (
                2.0 * (probability * y[indices]).sum(1) + 1.0
            ) / (probability.sum(1) + y[indices].sum(1) + 1.0)
            window_loss = ((bce + 0.15 * dice) * weights[indices]).mean()
            count_loss = (
                F.cross_entropy(
                    count_logits, count_truth[indices], reduction="none"
                )
                * weights[indices]
            ).mean()
            loss = window_loss + 0.3 * count_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach().cpu()) * len(indices)
        print(
            f"  nn epoch {epoch + 1}/{epochs} loss {epoch_loss / len(features):.4f}",
            flush=True,
        )

        if validation is not None:
            vx, vs, vc, vk, vy, validation_locales = validation
            val_probability, val_count = predict_nn(model, stats, vx, vs, vc, vk)
            val_score = locale_balanced_score(
                topk_decode(val_probability), vy, validation_locales
            )
            print(f"    validation top-k RWU {val_score:.6f}", flush=True)
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_validation = (val_probability, val_count)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, stats, best_epoch, best_score, best_validation


# ---------------- count-aware expected-utility decoding ----------------
def sampled_candidate_scores(candidate, samples):
    candidate = np.asarray(candidate, dtype=bool)
    samples = np.asarray(samples, dtype=bool)
    exact = (samples & candidate).sum(1).astype(np.float32)
    remaining_truth = samples & ~candidate
    predicted_only = (~samples) & candidate
    used = np.zeros_like(samples, dtype=bool)
    adjacent = np.zeros(len(samples), dtype=np.float32)
    for position in range(N_WIN):
        active = predicted_only[:, position]
        if position > 0:
            hit = active & remaining_truth[:, position - 1] & ~used[:, position - 1]
            used[hit, position - 1] = True
            adjacent += hit
            active = active & ~hit
        if position + 1 < N_WIN:
            hit = active & remaining_truth[:, position + 1] & ~used[:, position + 1]
            used[hit, position + 1] = True
            adjacent += hit
    credit = exact + 0.35 * adjacent
    predicted_count = candidate.sum()
    true_count = samples.sum(1)
    weighted_f1 = 2.0 * credit / (predicted_count + true_count)
    budget = np.minimum(predicted_count, true_count) / np.maximum(
        predicted_count, true_count
    )
    return 0.8 * weighted_f1 + 0.2 * budget


def decode_expected(
    probabilities,
    count_probabilities,
    samples=MC_SAMPLES,
    temperature=1.0,
):
    output = np.zeros_like(probabilities, dtype=np.int8)
    for row_index, row_probability in enumerate(probabilities):
        # Reusing the same fixed draws for every row makes this a true one-row transform.
        rng = np.random.default_rng(SEED)
        count_distribution = count_probabilities[row_index].astype(np.float64)
        count_distribution /= count_distribution.sum()
        uniforms = rng.random(samples)
        sampled_counts = (
            np.searchsorted(np.cumsum(count_distribution), uniforms, side="right") + 1
        )
        probability = np.clip(row_probability, 1e-5, 1.0 - 1e-5)
        log_odds = np.log(probability / (1.0 - probability)) / temperature
        gumbels = log_odds[None, :] + rng.gumbel(size=(samples, N_WIN))
        sampled_masks = np.zeros((samples, N_WIN), dtype=bool)
        for count in np.unique(sampled_counts):
            selected_rows = np.flatnonzero(sampled_counts == count)
            count = min(int(count), N_WIN)
            selected = np.argpartition(
                -gumbels[selected_rows], count - 1, axis=1
            )[:, :count]
            sampled_masks[selected_rows[:, None], selected] = True

        order = np.argsort(-probability)
        best_mask = None
        best_utility = -1.0
        for count in range(1, N_WIN + 1):
            candidate = np.zeros(N_WIN, dtype=bool)
            candidate[order[:count]] = True
            utility = sampled_candidate_scores(candidate, sampled_masks).mean()
            if utility > best_utility:
                best_utility = float(utility)
                best_mask = candidate

        for _ in range(2):
            improved = False
            for position in range(N_WIN):
                if best_mask[position] and best_mask.sum() == 1:
                    continue
                candidate = best_mask.copy()
                candidate[position] = ~candidate[position]
                utility = sampled_candidate_scores(candidate, sampled_masks).mean()
                if utility > best_utility + 1e-9:
                    best_utility = float(utility)
                    best_mask = candidate
                    improved = True
            if not improved:
                break
        output[row_index] = best_mask
    return output


def calibrate_ensemble(aux_probability, code_probability, count_probability, truth, locales):
    best_weight = 0.5
    best_topk = -1.0
    for weight in np.linspace(0.0, 1.0, 11):
        probability = weight * code_probability + (1.0 - weight) * aux_probability
        score = locale_balanced_score(topk_decode(probability), truth, locales)
        if score > best_topk:
            best_topk = score
            best_weight = float(weight)

    probability = best_weight * code_probability + (1.0 - best_weight) * aux_probability
    best_temperature = 1.0
    best_score = -1.0
    for temperature in (0.8, 1.0, 1.2):
        predictions = decode_expected(
            probability,
            count_probability,
            samples=CALIBRATION_SAMPLES,
            temperature=temperature,
        )
        score = locale_balanced_score(predictions, truth, locales)
        if score > best_score:
            best_score = score
            best_temperature = temperature
    return best_weight, best_temperature, best_score


# ---------------- end-to-end run ----------------
def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    train = pd.read_csv(
        public_dir / "train.csv",
        dtype={"anchor_repair_windows": str, "target_repair_windows": str},
    )
    truth = np.stack([mask_array(value) for value in train["target_repair_windows"]])
    validation_mask, validation_routes = route_holdout_mask(train)
    fit_df = train.loc[~validation_mask].reset_index(drop=True)
    validation_df = train.loc[validation_mask].reset_index(drop=True)
    fit_truth = truth[~validation_mask]
    validation_truth = truth[validation_mask]
    print(
        f"train {len(train)}; route holdout {len(validation_df)} rows: {validation_routes}",
        flush=True,
    )

    # Every validation transform/statistic is fit on the fit partition only.
    validation_code_weights = fit_code_weights(fit_df)
    fit_features, fit_similarities, fit_codes = build_inputs(
        fit_df, validation_code_weights
    )
    val_features, val_similarities, val_codes = build_inputs(
        validation_df, validation_code_weights
    )
    fit_categories, val_categories, category_sizes = encode_categories(
        fit_df, validation_df
    )
    fit_row_weights = locale_weights(fit_df)

    validation_args = (
        val_features,
        val_similarities,
        val_codes,
        val_categories,
        validation_truth,
        validation_df["target_locale"].to_numpy(),
    )
    print("validation auxiliary transformer", flush=True)
    validation_aux, _, aux_epoch, _, aux_predictions = train_nn(
        fit_features,
        fit_similarities,
        fit_codes,
        fit_categories,
        fit_truth,
        fit_row_weights,
        category_sizes,
        epochs=NN_MAX_EPOCHS,
        validation=validation_args,
        use_codes=False,
    )
    print("validation bucket transformer", flush=True)
    validation_code, _, code_epoch, _, code_predictions = train_nn(
        fit_features,
        fit_similarities,
        fit_codes,
        fit_categories,
        fit_truth,
        fit_row_weights,
        category_sizes,
        epochs=NN_MAX_EPOCHS,
        validation=validation_args,
        use_codes=True,
    )
    val_aux_probability, _ = aux_predictions
    val_code_probability, val_count_probability = code_predictions
    selected_aux_epochs = max(5, aux_epoch + 1)
    selected_code_epochs = max(5, code_epoch + 1)
    ensemble_weight, decode_temperature, validation_score = calibrate_ensemble(
        val_aux_probability,
        val_code_probability,
        val_count_probability,
        validation_truth,
        validation_df["target_locale"].to_numpy(),
    )
    print(
        "validation RWU "
        f"{validation_score:.6f}; aux_epochs={selected_aux_epochs}; "
        f"code_epochs={selected_code_epochs}; code_weight={ensemble_weight:.1f}; "
        f"temperature={decode_temperature:.1f}",
        flush=True,
    )

    del (
        validation_aux,
        validation_code,
        fit_features,
        fit_similarities,
        fit_codes,
        val_features,
        val_similarities,
        val_codes,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fit every final transform and model using training rows only.
    full_code_weights = fit_code_weights(train)
    train_features, train_similarities, train_codes = build_inputs(
        train, full_code_weights
    )
    train_categories, _, category_sizes = encode_categories(train, train)
    train_row_weights = locale_weights(train)

    print("final auxiliary transformer", flush=True)
    aux_model, aux_stats, _, _, _ = train_nn(
        train_features,
        train_similarities,
        train_codes,
        train_categories,
        truth,
        train_row_weights,
        category_sizes,
        epochs=selected_aux_epochs,
        use_codes=False,
    )
    print("final bucket transformer", flush=True)
    code_model, code_stats, _, _, _ = train_nn(
        train_features,
        train_similarities,
        train_codes,
        train_categories,
        truth,
        train_row_weights,
        category_sizes,
        epochs=selected_code_epochs,
        use_codes=True,
    )
    # Test is inference-only: train-fitted transforms followed by model prediction.
    test = pd.read_csv(
        public_dir / "test.csv", dtype={"anchor_repair_windows": str}
    )
    test_features, test_similarities, test_codes = build_inputs(test, full_code_weights)
    _, test_categories, _ = encode_categories(train, test)
    test_aux_probability, _ = predict_nn(
        aux_model,
        aux_stats,
        test_features,
        test_similarities,
        test_codes,
        test_categories,
    )
    test_code_probability, test_count_probability = predict_nn(
        code_model,
        code_stats,
        test_features,
        test_similarities,
        test_codes,
        test_categories,
    )
    test_probability = (
        ensemble_weight * test_code_probability
        + (1.0 - ensemble_weight) * test_aux_probability
    )
    predictions = decode_expected(
        test_probability,
        test_count_probability,
        samples=MC_SAMPLES,
        temperature=decode_temperature,
    )

    masks = ["".join(str(int(bit)) for bit in row) for row in predictions]
    submission = pd.DataFrame(
        {"id": test["id"], "target_repair_windows": masks}
    )
    submission.to_csv(submission_out, index=False)
    elapsed = time.monotonic() - started
    print(f"wrote {submission_out} ({len(submission)} rows) in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
