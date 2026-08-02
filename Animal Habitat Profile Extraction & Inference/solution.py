#!/usr/bin/env python3
"""Train-only habitat dossier extraction and inference pipeline."""

from __future__ import annotations

import gc
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import FeatureUnion


SEED = 42
HPO_CUTOFF_SECONDS = 2400.0
FINAL_TRAINING_CUTOFF_SECONDS = 3000.0

ID_COLUMN = "id"
RECORD_COLUMN = "record"
TIER_COLUMN = "tier"
SINGLE_FIELDS = [
    "activity",
    "locomotion",
    "social",
    "reproduction",
    "trophic_guild",
]
MULTI_FIELDS = ["habitats", "climate"]
SUBMISSION_COLUMNS = [ID_COLUMN, *SINGLE_FIELDS, *MULTI_FIELDS]

SINGLE_VOCABS = {
    "activity": {"nocturnal", "diurnal", "crepuscular", "cathemeral", "unknown"},
    "locomotion": {
        "terrestrial",
        "arboreal",
        "aquatic",
        "semiaquatic",
        "fossorial",
        "volant",
        "unknown",
    },
    "social": {"solitary", "social", "unknown"},
    "reproduction": {"viviparous", "oviparous", "ovoviviparous", "unknown"},
    "trophic_guild": {
        "carnivore",
        "herbivore",
        "omnivore",
        "insectivore",
        "piscivore",
        "unknown",
    },
}
HABITAT_VOCAB = [
    "Agricultural",
    "Caves",
    "Coastal",
    "Forest",
    "Freshwater",
    "Grassland",
    "Marine",
    "Mountains",
    "Rainforest",
    "Rocky areas",
    "Savanna",
    "Shrubland",
    "Wetlands",
]
CLIMATE_VOCAB = ["tropical", "temperate", "cold", "arid", "polar"]
FIELD_WEIGHTS = {
    "activity": 1.0,
    "locomotion": 1.0,
    "social": 1.0,
    "reproduction": 1.0,
    "trophic_guild": 1.0,
    "habitats": 20.0,
    "climate": 2.0,
}
TOTAL_FIELD_WEIGHT = sum(FIELD_WEIGHTS.values())

EXTRACTION_C_CANDIDATES = (4.0, 16.0, 64.0)
INFERENCE_C_CANDIDATES = (0.25, 0.5, 1.0, 2.0)
BAGGING_SEEDS = (13, 97, 211, 307, 401)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def elapsed(started_at: float) -> float:
    return time.monotonic() - started_at


def clean_records(values: pd.Series) -> list[str]:
    return ["" if pd.isna(value) else str(value) for value in values]


def section_value(record: str, section_name: str) -> str:
    """Generic structured-text parser used only to construct model features/groups."""
    wanted = section_name.casefold()
    for part in str(record).split("|"):
        key, separator, value = part.partition(":")
        if separator and key.strip().casefold() == wanted:
            return value.strip()
    return ""


def geography_views(records: list[str]) -> list[str]:
    return [
        "realms "
        + section_value(record, "Realms")
        + " continents "
        + section_value(record, "Continents")
        for record in records
    ]


def taxonomic_orders(records: list[str]) -> np.ndarray:
    return np.asarray(
        [section_value(record, "Order") or "__missing_order__" for record in records],
        dtype=object,
    )


def write_placeholder(test: pd.DataFrame, submission_out: Path) -> None:
    placeholder = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(str)})
    for field in SINGLE_FIELDS:
        placeholder[field] = "unknown"
    placeholder["habitats"] = ""
    placeholder["climate"] = ""
    placeholder = placeholder[SUBMISSION_COLUMNS]
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    placeholder.to_csv(submission_out, index=False)


def normalize_training_targets(train: pd.DataFrame) -> pd.DataFrame:
    normalized = train.copy()
    for field in SINGLE_FIELDS:
        values = normalized[field].fillna("unknown").astype(str).str.strip().str.lower()
        invalid = ~values.isin(SINGLE_VOCABS[field])
        if invalid.any():
            log(f"WARNING: {int(invalid.sum())} invalid {field} targets changed to unknown")
            values.loc[invalid] = "unknown"
        normalized[field] = values
    normalized[TIER_COLUMN] = (
        pd.to_numeric(normalized[TIER_COLUMN], errors="coerce").fillna(1.0).clip(lower=1.0)
    )
    return normalized


def multilabel_matrix(values: pd.Series, vocabulary: list[str]) -> np.ndarray:
    index = {label: column for column, label in enumerate(vocabulary)}
    matrix = np.zeros((len(values), len(vocabulary)), dtype=np.int8)
    for row, raw_value in enumerate(values.fillna("")):
        for token in str(raw_value).split(";"):
            clean_token = token.strip()
            column = index.get(clean_token)
            if column is not None:
                matrix[row, column] = 1
    return matrix


def dice_per_row(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_bool = np.asarray(truth, dtype=bool)
    pred_bool = np.asarray(prediction, dtype=bool)
    intersections = np.logical_and(truth_bool, pred_bool).sum(axis=1)
    denominators = truth_bool.sum(axis=1) + pred_bool.sum(axis=1)
    return np.where(
        denominators == 0,
        1.0,
        2.0 * intersections / np.maximum(denominators, 1),
    )


def challenge_score(
    single_truth: dict[str, np.ndarray],
    single_prediction: dict[str, np.ndarray],
    habitat_truth: np.ndarray,
    habitat_prediction: np.ndarray,
    climate_truth: np.ndarray,
    climate_prediction: np.ndarray,
    tiers: np.ndarray,
) -> float:
    item_scores = np.zeros(len(tiers), dtype=np.float64)
    for field in SINGLE_FIELDS:
        item_scores += (
            np.asarray(single_truth[field]) == np.asarray(single_prediction[field])
        ).astype(np.float64)
    item_scores += FIELD_WEIGHTS["habitats"] * dice_per_row(
        habitat_truth, habitat_prediction
    )
    item_scores += FIELD_WEIGHTS["climate"] * dice_per_row(
        climate_truth, climate_prediction
    )
    item_scores /= TOTAL_FIELD_WEIGHT
    return float(np.average(item_scores, weights=tiers))


def make_text_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
        ]
    )


def make_geo_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )


def fit_feature_matrices(
    train_records: list[str],
    evaluation_records: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, Any, Any]:
    text_vectorizer = make_text_vectorizer()
    train_text = text_vectorizer.fit_transform(train_records).tocsr()
    evaluation_text = text_vectorizer.transform(evaluation_records).tocsr()

    train_geo = geography_views(train_records)
    evaluation_geo = geography_views(evaluation_records)
    geo_vectorizer = make_geo_vectorizer()
    train_geo_matrix = geo_vectorizer.fit_transform(train_geo).tocsr()
    evaluation_geo_matrix = geo_vectorizer.transform(evaluation_geo).tocsr()

    train_inference = sparse.hstack(
        [train_text, train_geo_matrix], format="csr", dtype=np.float32
    )
    evaluation_inference = sparse.hstack(
        [evaluation_text, evaluation_geo_matrix], format="csr", dtype=np.float32
    )
    return (
        train_text,
        evaluation_text,
        train_inference,
        evaluation_inference,
        text_vectorizer,
        geo_vectorizer,
    )


def transform_inference_features(
    records: list[str],
    text_vectorizer: Any,
    geo_vectorizer: Any,
) -> sparse.csr_matrix:
    text_matrix = text_vectorizer.transform(records).tocsr()
    geo_matrix = geo_vectorizer.transform(geography_views(records)).tocsr()
    return sparse.hstack(
        [text_matrix, geo_matrix], format="csr", dtype=np.float32
    )


def new_logistic_regression(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        max_iter=800,
        solver="liblinear",
        random_state=SEED,
    )


def multiclass_probabilities(
    train_features: sparse.csr_matrix,
    train_target: np.ndarray,
    evaluation_features: sparse.csr_matrix,
    all_classes: list[str],
    c_value: float,
) -> np.ndarray:
    output = np.zeros((evaluation_features.shape[0], len(all_classes)), dtype=np.float32)
    observed = np.unique(train_target)
    if len(observed) == 1:
        output[:, all_classes.index(str(observed[0]))] = 1.0
        return output
    model = new_logistic_regression(c_value)
    model.fit(train_features, train_target)
    local = model.predict_proba(evaluation_features)
    global_index = {label: column for column, label in enumerate(all_classes)}
    for local_column, label in enumerate(model.classes_):
        output[:, global_index[str(label)]] = local[:, local_column]
    return output


def train_binary_bank(
    train_features: sparse.csr_matrix,
    train_targets: np.ndarray,
    c_value: float,
) -> list[Any]:
    bank: list[Any] = []
    for column in range(train_targets.shape[1]):
        target = train_targets[:, column]
        observed = np.unique(target)
        if len(observed) == 1:
            bank.append(float(observed[0]))
            continue
        model = new_logistic_regression(c_value)
        model.fit(train_features, target)
        bank.append(model)
    return bank


def predict_binary_bank(
    bank: list[Any], evaluation_features: sparse.csr_matrix
) -> np.ndarray:
    probabilities = np.zeros(
        (evaluation_features.shape[0], len(bank)), dtype=np.float32
    )
    for column, model_or_constant in enumerate(bank):
        if isinstance(model_or_constant, float):
            probabilities[:, column] = model_or_constant
        else:
            probabilities[:, column] = model_or_constant.predict_proba(
                evaluation_features
            )[:, 1]
    return probabilities


def train_predict_binary(
    train_features: sparse.csr_matrix,
    train_targets: np.ndarray,
    evaluation_features: sparse.csr_matrix,
    c_value: float,
) -> np.ndarray:
    return predict_binary_bank(
        train_binary_bank(train_features, train_targets, c_value),
        evaluation_features,
    )


def tune_threshold_decoder(
    probabilities: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, float]:
    grid = np.unique(np.r_[np.linspace(0.025, 0.8, 32), 1.01])
    shared_scores = [
        float(np.average(dice_per_row(truth, probabilities >= threshold), weights=weights))
        for threshold in grid
    ]
    shared_threshold = float(grid[int(np.argmax(shared_scores))])
    thresholds = np.full(probabilities.shape[1], shared_threshold, dtype=np.float64)
    best_score = float(max(shared_scores))

    for _ in range(4):
        changed = False
        for column in range(probabilities.shape[1]):
            old_threshold = float(thresholds[column])
            local_threshold = old_threshold
            local_score = best_score
            for candidate in grid:
                thresholds[column] = float(candidate)
                prediction = probabilities >= thresholds
                score = float(
                    np.average(dice_per_row(truth, prediction), weights=weights)
                )
                if score > local_score + 1e-12:
                    local_score = score
                    local_threshold = float(candidate)
            thresholds[column] = local_threshold
            if local_score > best_score + 1e-12:
                best_score = local_score
                changed = True
        if not changed:
            break

    prediction = probabilities >= thresholds
    return (
        {"kind": "threshold", "thresholds": thresholds},
        prediction,
        float(np.average(dice_per_row(truth, prediction), weights=weights)),
    )


def soft_dice_decode(
    probabilities: np.ndarray,
    temperature: float,
    bias: float,
    allow_empty: bool,
) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / float(temperature) + float(bias)
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    order = np.argsort(-calibrated, axis=1)
    ordered_probabilities = np.take_along_axis(calibrated, order, axis=1)
    cumulative = np.cumsum(ordered_probabilities, axis=1)
    cardinalities = np.arange(1, probabilities.shape[1] + 1, dtype=np.float64)[None, :]
    approximate_expected_dice = 2.0 * cumulative / (
        cardinalities + calibrated.sum(axis=1, keepdims=True)
    )
    best_cardinality = np.argmax(approximate_expected_dice, axis=1) + 1
    if allow_empty:
        empty_score = np.prod(1.0 - calibrated, axis=1)
        nonempty_score = np.max(approximate_expected_dice, axis=1)
        best_cardinality = np.where(empty_score > nonempty_score, 0, best_cardinality)

    prediction = np.zeros_like(calibrated, dtype=bool)
    for row, cardinality in enumerate(best_cardinality):
        if cardinality:
            prediction[row, order[row, : int(cardinality)]] = True
    return prediction


def tune_soft_decoder(
    probabilities: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, float]:
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    biases = np.linspace(-1.5, 1.5, 13)
    best_score = -1.0
    best_config: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    for allow_empty in (False, True):
        for temperature in temperatures:
            for bias in biases:
                prediction = soft_dice_decode(
                    probabilities, temperature, float(bias), allow_empty
                )
                score = float(
                    np.average(dice_per_row(truth, prediction), weights=weights)
                )
                if score > best_score + 1e-12:
                    best_score = score
                    best_config = {
                        "kind": "soft_dice",
                        "temperature": float(temperature),
                        "bias": float(bias),
                        "allow_empty": bool(allow_empty),
                    }
                    best_prediction = prediction
    if best_config is None or best_prediction is None:
        raise RuntimeError("Decoder search produced no candidate")
    return best_config, best_prediction, best_score


def tune_decoder(
    probabilities: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, float]:
    candidates = [
        tune_threshold_decoder(probabilities, truth, weights),
        tune_soft_decoder(probabilities, truth, weights),
    ]
    return max(candidates, key=lambda result: result[2])


def apply_decoder(probabilities: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if config["kind"] == "threshold":
        return probabilities >= np.asarray(config["thresholds"], dtype=np.float64)
    return soft_dice_decode(
        probabilities,
        float(config["temperature"]),
        float(config["bias"]),
        bool(config["allow_empty"]),
    )


def make_group_splits(
    records: list[str], row_count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = taxonomic_orders(records)
    unique_group_count = len(np.unique(groups))
    if unique_group_count >= 2:
        splitter = GroupKFold(n_splits=min(5, unique_group_count))
        return list(splitter.split(np.arange(row_count), groups=groups))
    n_splits = min(5, row_count)
    if n_splits < 2:
        raise ValueError("At least two training rows are required")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(np.arange(row_count)))


def run_oof_search(
    train: pd.DataFrame,
    records: list[str],
    habitat_targets: np.ndarray,
    climate_targets: np.ndarray,
    started_at: float,
) -> tuple[float, float, dict[str, Any], dict[str, Any], float, float, float]:
    row_count = len(train)
    tiers = train[TIER_COLUMN].to_numpy(dtype=np.float64)
    single_classes = {
        field: sorted(train[field].astype(str).unique().tolist()) for field in SINGLE_FIELDS
    }
    single_truth = {
        field: train[field].astype(str).to_numpy() for field in SINGLE_FIELDS
    }

    single_oof = {
        c_value: {
            field: np.zeros((row_count, len(single_classes[field])), dtype=np.float32)
            for field in SINGLE_FIELDS
        }
        for c_value in EXTRACTION_C_CANDIDATES
    }
    habitat_oof = {
        c_value: np.zeros((row_count, len(HABITAT_VOCAB)), dtype=np.float32)
        for c_value in INFERENCE_C_CANDIDATES
    }
    climate_oof = {
        c_value: np.zeros((row_count, len(CLIMATE_VOCAB)), dtype=np.float32)
        for c_value in INFERENCE_C_CANDIDATES
    }
    valid_mask = np.zeros(row_count, dtype=bool)

    splits = make_group_splits(records, row_count)
    for fold_number, (fit_rows, valid_rows) in enumerate(splits):
        if fold_number and elapsed(started_at) >= HPO_CUTOFF_SECONDS:
            log("Wall-clock guard: stopping additional grouped validation folds")
            break
        fit_records = [records[index] for index in fit_rows]
        valid_records = [records[index] for index in valid_rows]
        (
            fit_text,
            valid_text,
            fit_inference,
            valid_inference,
            _,
            _,
        ) = fit_feature_matrices(fit_records, valid_records)

        for c_value in EXTRACTION_C_CANDIDATES:
            for field in SINGLE_FIELDS:
                single_oof[c_value][field][valid_rows] = multiclass_probabilities(
                    fit_text,
                    single_truth[field][fit_rows],
                    valid_text,
                    single_classes[field],
                    c_value,
                )

        for c_value in INFERENCE_C_CANDIDATES:
            habitat_oof[c_value][valid_rows] = train_predict_binary(
                fit_inference,
                habitat_targets[fit_rows],
                valid_inference,
                c_value,
            )
            climate_oof[c_value][valid_rows] = train_predict_binary(
                fit_inference,
                climate_targets[fit_rows],
                valid_inference,
                c_value,
            )

        valid_mask[valid_rows] = True
        log(
            f"Completed order-held-out fold {fold_number + 1}/{len(splits)} "
            f"({int(valid_mask.sum())}/{row_count} OOF rows, {elapsed(started_at):.1f}s)"
        )

    if not valid_mask.any():
        raise RuntimeError("No grouped validation fold completed")

    selected_extraction_c = EXTRACTION_C_CANDIDATES[0]
    best_extraction_score = -1.0
    selected_single_prediction: dict[str, np.ndarray] = {}
    for c_value in EXTRACTION_C_CANDIDATES:
        predictions = {
            field: np.asarray(single_classes[field], dtype=object)[
                np.argmax(single_oof[c_value][field][valid_mask], axis=1)
            ]
            for field in SINGLE_FIELDS
        }
        weighted_correct = np.zeros(int(valid_mask.sum()), dtype=np.float64)
        for field in SINGLE_FIELDS:
            weighted_correct += (
                predictions[field] == single_truth[field][valid_mask]
            ).astype(np.float64)
        score = float(np.average(weighted_correct / len(SINGLE_FIELDS), weights=tiers[valid_mask]))
        log(f"Extraction HPO C={c_value:g}: weighted five-field accuracy={score:.6f}")
        if score > best_extraction_score + 1e-12:
            best_extraction_score = score
            selected_extraction_c = c_value
            selected_single_prediction = predictions

    masked_single_truth = {
        field: single_truth[field][valid_mask] for field in SINGLE_FIELDS
    }
    selected_inference_c = INFERENCE_C_CANDIDATES[0]
    selected_habitat_decoder: dict[str, Any] | None = None
    selected_climate_decoder: dict[str, Any] | None = None
    selected_habitat_dice = 0.0
    selected_climate_dice = 0.0
    best_metric = -1.0

    for c_value in INFERENCE_C_CANDIDATES:
        habitat_config, habitat_prediction, habitat_dice = tune_decoder(
            habitat_oof[c_value][valid_mask],
            habitat_targets[valid_mask],
            tiers[valid_mask],
        )
        climate_config, climate_prediction, climate_dice = tune_decoder(
            climate_oof[c_value][valid_mask],
            climate_targets[valid_mask],
            tiers[valid_mask],
        )
        metric = challenge_score(
            masked_single_truth,
            selected_single_prediction,
            habitat_targets[valid_mask],
            habitat_prediction,
            climate_targets[valid_mask],
            climate_prediction,
            tiers[valid_mask],
        )
        log(
            f"Inference HPO C={c_value:g}: metric={metric:.6f}, "
            f"habitat Dice={habitat_dice:.6f}, climate Dice={climate_dice:.6f}, "
            f"decoders={habitat_config['kind']}/{climate_config['kind']}"
        )
        if metric > best_metric + 1e-12:
            best_metric = metric
            selected_inference_c = c_value
            selected_habitat_decoder = habitat_config
            selected_climate_decoder = climate_config
            selected_habitat_dice = habitat_dice
            selected_climate_dice = climate_dice

    if selected_habitat_decoder is None or selected_climate_decoder is None:
        raise RuntimeError("Inference search produced no decoder")
    log(
        f"Selected extraction C={selected_extraction_c:g}, inference C={selected_inference_c:g}; "
        f"order-held-out validation metric={best_metric:.6f}"
    )
    return (
        float(selected_extraction_c),
        float(selected_inference_c),
        selected_habitat_decoder,
        selected_climate_decoder,
        float(best_metric),
        float(selected_habitat_dice),
        float(selected_climate_dice),
    )


def run_repeated_habitat_bagging(
    records: list[str],
    test_records: list[str],
    habitat_targets: np.ndarray,
    tiers: np.ndarray,
    c_value: float,
    base_decoder: dict[str, Any],
    base_dice: float,
    started_at: float,
) -> tuple[np.ndarray | None, dict[str, Any], float, int]:
    groups = taxonomic_orders(records)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return None, base_decoder, base_dice, 0

    repeat_oof: list[np.ndarray] = []
    repeat_test: list[np.ndarray] = []
    best_decoder = base_decoder
    best_dice = base_dice
    best_repeat_count = 0
    split_count = min(5, len(unique_groups))

    for repeat_number, seed in enumerate(BAGGING_SEEDS):
        if elapsed(started_at) >= HPO_CUTOFF_SECONDS:
            log("Wall-clock guard: stopping repeated habitat bagging")
            break

        oof_probabilities = np.zeros(
            (len(records), len(HABITAT_VOCAB)), dtype=np.float32
        )
        fold_test_probabilities: list[np.ndarray] = []
        group_splitter = KFold(
            n_splits=split_count, shuffle=True, random_state=seed
        )
        for fold_number, (_, valid_group_rows) in enumerate(
            group_splitter.split(unique_groups)
        ):
            valid_groups = unique_groups[valid_group_rows]
            valid_mask = np.isin(groups, valid_groups)
            valid_rows = np.flatnonzero(valid_mask)
            fit_rows = np.flatnonzero(~valid_mask)
            fit_records = [records[index] for index in fit_rows]
            valid_records = [records[index] for index in valid_rows]
            (
                _,
                _,
                fit_inference,
                valid_inference,
                text_vectorizer,
                geo_vectorizer,
            ) = fit_feature_matrices(fit_records, valid_records)
            test_inference = transform_inference_features(
                test_records, text_vectorizer, geo_vectorizer
            )
            bank = train_binary_bank(
                fit_inference, habitat_targets[fit_rows], c_value
            )
            oof_probabilities[valid_rows] = predict_binary_bank(
                bank, valid_inference
            )
            fold_test_probabilities.append(
                predict_binary_bank(bank, test_inference)
            )
            del (
                fit_inference,
                valid_inference,
                test_inference,
                text_vectorizer,
                geo_vectorizer,
                bank,
            )
            gc.collect()
            log(
                f"Habitat bagging repeat {repeat_number + 1}/{len(BAGGING_SEEDS)}, "
                f"fold {fold_number + 1}/{split_count} complete"
            )

        repeat_oof.append(oof_probabilities)
        repeat_test.append(
            np.mean(np.stack(fold_test_probabilities, axis=0), axis=0)
        )
        candidate_probabilities = np.mean(
            np.stack(repeat_oof, axis=0), axis=0
        )
        decoder, _, candidate_dice = tune_decoder(
            candidate_probabilities, habitat_targets, tiers
        )
        log(
            f"Habitat bagging HPO repeats={repeat_number + 1}: "
            f"Dice={candidate_dice:.6f}, decoder={decoder['kind']}"
        )
        if candidate_dice > best_dice + 1e-12:
            best_dice = candidate_dice
            best_decoder = decoder
            best_repeat_count = repeat_number + 1

    if best_repeat_count == 0:
        return None, base_decoder, base_dice, 0
    selected_test_probabilities = np.mean(
        np.stack(repeat_test[:best_repeat_count], axis=0), axis=0
    )
    return (
        selected_test_probabilities,
        best_decoder,
        float(best_dice),
        best_repeat_count,
    )


def train_final_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_records: list[str],
    test_records: list[str],
    habitat_targets: np.ndarray,
    climate_targets: np.ndarray,
    extraction_c: float,
    inference_c: float,
    habitat_decoder: dict[str, Any],
    climate_decoder: dict[str, Any],
    habitat_probability_override: np.ndarray | None,
    started_at: float,
) -> pd.DataFrame:
    if elapsed(started_at) >= FINAL_TRAINING_CUTOFF_SECONDS:
        log("WARNING: final training began at the wall-clock cutoff; using one selected model only")

    (
        train_text,
        test_text,
        train_inference,
        test_inference,
        _,
        _,
    ) = fit_feature_matrices(train_records, test_records)

    single_predictions: dict[str, np.ndarray] = {}
    for field in SINGLE_FIELDS:
        classes = sorted(train[field].astype(str).unique().tolist())
        probabilities = multiclass_probabilities(
            train_text,
            train[field].astype(str).to_numpy(),
            test_text,
            classes,
            extraction_c,
        )
        single_predictions[field] = np.asarray(classes, dtype=object)[
            np.argmax(probabilities, axis=1)
        ]

    if habitat_probability_override is None:
        habitat_bank = train_binary_bank(
            train_inference, habitat_targets, inference_c
        )
        habitat_probabilities = predict_binary_bank(
            habitat_bank, test_inference
        )
    else:
        habitat_probabilities = habitat_probability_override
    climate_bank = train_binary_bank(train_inference, climate_targets, inference_c)
    climate_probabilities = predict_binary_bank(climate_bank, test_inference)
    habitat_prediction = apply_decoder(habitat_probabilities, habitat_decoder)
    climate_prediction = apply_decoder(climate_probabilities, climate_decoder)

    rows: list[dict[str, str]] = []
    for row_number, test_id in enumerate(test[ID_COLUMN].astype(str)):
        fallback = {
            ID_COLUMN: test_id,
            **{field: "unknown" for field in SINGLE_FIELDS},
            "habitats": "",
            "climate": "",
        }
        try:
            output_row = {ID_COLUMN: test_id}
            for field in SINGLE_FIELDS:
                value = str(single_predictions[field][row_number])
                output_row[field] = value if value in SINGLE_VOCABS[field] else "unknown"
            output_row["habitats"] = ";".join(
                label
                for label, present in zip(
                    HABITAT_VOCAB, habitat_prediction[row_number]
                )
                if present
            )
            output_row["climate"] = ";".join(
                label
                for label, present in zip(
                    CLIMATE_VOCAB, climate_prediction[row_number]
                )
                if present
            )
            rows.append(output_row)
        except Exception as exc:  # one noisy test row must not invalidate all rows
            log(f"WARNING: row {row_number} prediction fallback: {exc}")
            rows.append(fallback)

    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def submission_errors(submission: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if submission.columns.tolist() != SUBMISSION_COLUMNS:
        errors.append("incorrect column names/order")
    if len(submission) != len(test):
        errors.append(f"row count {len(submission)} != {len(test)}")
    expected_ids = test[ID_COLUMN].astype(str).tolist()
    output_ids = submission[ID_COLUMN].astype(str).tolist()
    if output_ids != expected_ids:
        errors.append("ids or row order differ from test.csv")
    if len(output_ids) != len(set(output_ids)):
        errors.append("duplicate output ids")
    for field in SINGLE_FIELDS:
        invalid = ~submission[field].astype(str).isin(SINGLE_VOCABS[field])
        if invalid.any():
            errors.append(f"{int(invalid.sum())} invalid {field} values")
    for field, vocabulary in (
        ("habitats", set(HABITAT_VOCAB)),
        ("climate", set(CLIMATE_VOCAB)),
    ):
        invalid_rows = 0
        for raw_value in submission[field].fillna(""):
            tokens = [token for token in str(raw_value).split(";") if token]
            if len(tokens) != len(set(tokens)) or any(
                token not in vocabulary for token in tokens
            ):
                invalid_rows += 1
        if invalid_rows:
            errors.append(f"{invalid_rows} invalid {field} sets")
    return errors


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 solution.py <public_dir> <submission_out>")

    random.seed(SEED)
    np.random.seed(SEED)
    started_at = time.monotonic()
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input: " + ", ".join(missing))

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required_train_columns = {
        ID_COLUMN,
        RECORD_COLUMN,
        TIER_COLUMN,
        *SINGLE_FIELDS,
        *MULTI_FIELDS,
    }
    required_test_columns = {ID_COLUMN, RECORD_COLUMN}
    if not required_train_columns.issubset(train.columns):
        raise ValueError("train.csv is missing required columns")
    if not required_test_columns.issubset(test.columns):
        raise ValueError("test.csv is missing required columns")

    # Required early safety artifact: this happens before vectorization or model fitting.
    write_placeholder(test, submission_out)
    log(f"Wrote schema-valid placeholder with {len(test)} rows")

    try:
        train = normalize_training_targets(train)
        train_records = clean_records(train[RECORD_COLUMN])
        test_records = clean_records(test[RECORD_COLUMN])
        habitat_targets = multilabel_matrix(train["habitats"], HABITAT_VOCAB)
        climate_targets = multilabel_matrix(train["climate"], CLIMATE_VOCAB)

        (
            extraction_c,
            inference_c,
            habitat_decoder,
            climate_decoder,
            validation_score,
            base_habitat_dice,
            base_climate_dice,
        ) = run_oof_search(
            train,
            train_records,
            habitat_targets,
            climate_targets,
            started_at,
        )
        habitat_probability_override: np.ndarray | None = None
        try:
            (
                habitat_probability_override,
                habitat_decoder,
                bagged_habitat_dice,
                selected_repeat_count,
            ) = run_repeated_habitat_bagging(
                train_records,
                test_records,
                habitat_targets,
                train[TIER_COLUMN].to_numpy(dtype=np.float64),
                inference_c,
                habitat_decoder,
                base_habitat_dice,
                started_at,
            )
            if selected_repeat_count:
                validation_score += (
                    FIELD_WEIGHTS["habitats"]
                    * (bagged_habitat_dice - base_habitat_dice)
                    / TOTAL_FIELD_WEIGHT
                )
                log(
                    f"Selected {selected_repeat_count} habitat bagging repeats; "
                    f"improved validation metric={validation_score:.6f}"
                )
        except Exception as exc:
            log(f"WARNING: habitat bagging failed; using full-data model: {exc}")
        submission = train_final_and_predict(
            train,
            test,
            train_records,
            test_records,
            habitat_targets,
            climate_targets,
            extraction_c,
            inference_c,
            habitat_decoder,
            climate_decoder,
            habitat_probability_override,
            started_at,
        )
        errors = submission_errors(submission, test)
        if errors:
            log("WARNING: final submission rejected by local audit: " + "; ".join(errors))
            return 0
        submission.to_csv(submission_out, index=False)
        log(
            f"Wrote final submission to {submission_out} with {len(submission)} rows; "
            f"validation={validation_score:.6f}; elapsed={elapsed(started_at):.1f}s"
        )
        return 0
    except Exception as exc:
        # The early placeholder remains valid. This path is intentionally non-fatal after
        # heavy work so a transient model/runtime failure does not consume a dead-last run.
        log(f"WARNING: modelling failed; preserving placeholder: {exc}")
        traceback.print_exc(file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
