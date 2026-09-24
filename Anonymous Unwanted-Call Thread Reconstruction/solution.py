#!/usr/bin/env python3
"""Anonymous Unwanted-Call Thread Reconstruction.

CPU-only profile ranking with categorical query boosting and a regularized
linear similarity model. Usage:
    python3 solution.py <public_dir> <submission_out>
"""

import json
import math
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

from catboost import CatBoostRanker, Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 20260922
N_FOLDS = 10
THREADS = 8

CAT_FIELDS = (
    "filing_weekday",
    "filing_hour_bucket",
    "issue_time_bucket",
    "issue_to_filing_lag",
    "method",
    "call_type",
    "service_category",
    "state",
    "advertiser_reported",
)
COMPOSITE_GROUPS = (
    ("method", "call_type"),
    ("method", "call_type", "service_category"),
    ("state", "method", "call_type"),
    ("filing_weekday", "filing_hour_bucket"),
    ("issue_time_bucket", "issue_to_filing_lag"),
)
WEEKDAY = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
QSM_CONFIGS = (
    {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
    },
    {
        "iterations": 800,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.3,
    },
    {
        "iterations": 700,
        "depth": 5,
        "learning_rate": 0.04,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.5,
    },
)
LINEAR_CS = (0.01, 0.03, 0.1)
BLEND_WEIGHTS = tuple(value / 10.0 for value in range(11))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def sorted_card_ids(profile):
    complaints = profile.get("complaints", [])
    ordered = sorted(
        complaints,
        key=lambda card: (
            float(card.get("relative_filing_hour", 0)),
            str(card.get("id", "")),
        ),
    )
    ids = [str(card.get("id", "")) for card in ordered]
    if len(ids) == 3 and len(set(ids)) == 3 and all(ids):
        return ids
    return ["p01_c01", "p01_c02", "p01_c03"]


def parse_case(anchor_text, bank_text):
    anchors = json.loads(anchor_text)
    profiles = json.loads(bank_text)
    if len(anchors) != 2 or len(profiles) != 8:
        raise ValueError("a case must contain two anchors and eight profiles")
    if any(len(profile.get("complaints", [])) != 3 for profile in profiles):
        raise ValueError("each profile must contain three complaints")
    return anchors, profiles


def write_placeholder(test_df, output_path):
    chains = []
    for row in test_df.itertuples(index=False):
        try:
            _, profiles = parse_case(row.anchor_bundle, row.profile_bank)
            chains.append(
                json.dumps(sorted_card_ids(profiles[0]), separators=(",", ":"))
            )
        except Exception:
            chains.append('["p01_c01","p01_c02","p01_c03"]')
    placeholder = pd.DataFrame(
        {"case_id": test_df["case_id"].astype(str), "continuation_chain": chains}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder.to_csv(output_path, index=False)


def load_cases(frame, training=False, label_map=None):
    cases = []
    targets = []
    kept_rows = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        try:
            anchors, profiles = parse_case(row.anchor_bundle, row.profile_bank)
            if training:
                chain = json.loads(label_map[str(row.case_id)])
                target_profile = str(chain[0]).split("_c", 1)[0]
                matches = [i for i, profile in enumerate(profiles)
                           if str(profile.get("profile", "")) == target_profile]
                if len(matches) != 1:
                    raise ValueError("label does not identify exactly one profile")
                targets.append(matches[0])
                kept_rows.append(row_index)
            cases.append((anchors, profiles))
        except Exception as exc:
            if training:
                print(f"warning: skipped malformed training row {row_index}: {exc}")
            else:
                print(f"warning: test row {row_index} uses fallback prediction: {exc}")
                cases.append(None)
    if training:
        return cases, np.asarray(targets, dtype=np.int64), kept_rows
    return cases


def fit_frequencies(cases, indices):
    counts = {field: Counter() for field in CAT_FIELDS}
    totals = {field: 0 for field in CAT_FIELDS}
    for index in indices:
        anchors, profiles = cases[index]
        cards = list(anchors)
        for profile in profiles:
            cards.extend(profile["complaints"])
        for card in cards:
            for field in CAT_FIELDS:
                counts[field][str(card[field])] += 1
                totals[field] += 1
    frequencies = {}
    for field in CAT_FIELDS:
        denominator = max(1, totals[field])
        frequencies[field] = {
            value: count / denominator for value, count in counts[field].items()
        }
        frequencies[field]["__unknown_probability__"] = 1.0 / (denominator + 1.0)
    return frequencies


def relational_features(anchors, cards):
    features = []
    for field in CAT_FIELDS:
        equal = np.asarray(
            [[float(anchor[field] == card[field]) for card in cards]
             for anchor in anchors],
            dtype=np.float32,
        )
        features.extend(equal.ravel())
        features.extend(equal.mean(axis=1))
        features.extend(equal.mean(axis=0))
        features.append(float(equal.mean()))
        features.append(float(anchors[0][field] == anchors[1][field]))
        features.extend(
            float(cards[i][field] == cards[j][field])
            for i, j in ((0, 1), (0, 2), (1, 2))
        )

    anchor_hours = np.asarray(
        [card["relative_filing_hour"] for card in anchors], dtype=np.float64
    )
    profile_hours = np.asarray(
        [card["relative_filing_hour"] for card in cards], dtype=np.float64
    )
    anchor_gap = anchor_hours[1] - anchor_hours[0]
    profile_gaps = np.diff(profile_hours)
    profile_duration = profile_hours[-1] - profile_hours[0]
    raw = np.asarray(
        [
            anchor_gap,
            profile_gaps[0],
            profile_gaps[1],
            profile_duration,
            abs(anchor_gap - profile_gaps[0]),
            abs(anchor_gap - profile_gaps[1]),
            abs(anchor_gap - profile_duration),
            anchor_gap / (1.0 + profile_duration),
            profile_gaps[0] / (1.0 + profile_duration),
            profile_gaps[1] / (1.0 + profile_duration),
        ],
        dtype=np.float64,
    )
    features.extend(np.sign(raw) * np.log1p(np.abs(raw)))
    for period in (24.0, 168.0):
        for value in (anchor_gap, profile_gaps[0], profile_gaps[1], profile_duration):
            features.extend(
                (math.sin(2.0 * math.pi * value / period),
                 math.cos(2.0 * math.pi * value / period))
            )
    return features


def additional_features(anchors, cards, frequencies):
    features = []
    for field in CAT_FIELDS:
        anchor_values = [str(card[field]) for card in anchors]
        profile_values = [str(card[field]) for card in cards]
        anchor_counts = Counter(anchor_values)
        profile_counts = Counter(profile_values)
        keys = set(anchor_counts) | set(profile_counts)
        intersection = sum(
            min(anchor_counts[key], profile_counts[key]) for key in keys
        )
        union = sum(max(anchor_counts[key], profile_counts[key]) for key in keys)
        information_matches = []
        unknown_probability = frequencies[field]["__unknown_probability__"]
        for anchor_value in anchor_values:
            probability = frequencies[field].get(anchor_value, unknown_probability)
            information = -math.log(probability + 1e-12)
            for profile_value in profile_values:
                information_matches.append(
                    information if anchor_value == profile_value else 0.0
                )
        features.extend(
            (
                intersection / 2.0,
                intersection / 3.0,
                intersection / max(1, union),
                len(set(anchor_values) & set(profile_values))
                / max(1, len(set(anchor_values) | set(profile_values))),
                len(set(anchor_values)),
                len(set(profile_values)),
                sum(information_matches),
                max(information_matches),
                sum(
                    abs(anchor_counts[key] / 2.0 - profile_counts[key] / 3.0)
                    for key in keys
                ),
            )
        )

    for anchor in anchors:
        anchor_day = WEEKDAY.get(str(anchor["filing_weekday"]), 0)
        for card in cards:
            profile_day = WEEKDAY.get(str(card["filing_weekday"]), 0)
            distance = abs(anchor_day - profile_day)
            features.append(min(distance, 7 - distance) / 3.0)

    for field in ("relative_filing_day", "relative_filing_hour"):
        anchor_time = np.asarray([float(card[field]) for card in anchors])
        profile_time = np.asarray([float(card[field]) for card in cards])
        anchor_gap = anchor_time[1] - anchor_time[0]
        profile_gaps = np.diff(profile_time)
        duration = profile_time[-1] - profile_time[0]
        values = np.asarray(
            [
                anchor_gap,
                profile_gaps[0],
                profile_gaps[1],
                duration,
                profile_gaps.mean(),
                profile_gaps.min(),
                profile_gaps.max(),
                abs(profile_gaps[0] - profile_gaps[1]),
            ],
            dtype=np.float64,
        )
        features.extend(np.sign(values) * np.log1p(np.abs(values)))
        log_anchor = math.log1p(max(0.0, anchor_gap))
        comparisons = [
            math.log1p(max(0.0, value))
            for value in (profile_gaps[0], profile_gaps[1], duration, duration / 2.0)
        ]
        features.extend(abs(log_anchor - value) for value in comparisons)
        features.extend(
            (
                min(abs(log_anchor - comparisons[0]), abs(log_anchor - comparisons[1])),
                anchor_gap / (1.0 + duration),
                profile_gaps.min() / (1.0 + profile_gaps.max()),
            )
        )
    return features

def joint_card_features(anchors, cards, frequencies):
    """Summarize whole-card and composite matches for a candidate profile."""
    equal = np.zeros((2, 3, len(CAT_FIELDS)), dtype=np.float32)
    weighted = np.zeros_like(equal)
    for anchor_index, anchor in enumerate(anchors):
        for card_index, card in enumerate(cards):
            for field_index, field in enumerate(CAT_FIELDS):
                anchor_value = str(anchor[field])
                if anchor_value == str(card[field]):
                    equal[anchor_index, card_index, field_index] = 1.0
                    probability = frequencies[field].get(
                        anchor_value,
                        frequencies[field]["__unknown_probability__"],
                    )
                    weighted[anchor_index, card_index, field_index] = (
                        -math.log(probability + 1e-12)
                    )

    features = []
    for matrix in (equal.sum(axis=2), weighted.sum(axis=2)):
        values = matrix.ravel()
        features.extend(values)
        features.extend(
            (
                values.mean(),
                values.max(),
                values.min(),
                values.std(),
                np.sort(values)[-2],
            )
        )
        features.extend(matrix.max(axis=1))
        features.extend(matrix.mean(axis=1))
        features.extend(matrix.max(axis=0))
        features.extend(matrix.mean(axis=0))
        features.append(
            max(
                matrix[0, first] + matrix[1, second]
                for first in range(3)
                for second in range(3)
                if first != second
            )
        )

    for group in COMPOSITE_GROUPS:
        matrix = np.asarray(
            [
                [
                    float(all(str(anchor[field]) == str(card[field]) for field in group))
                    for card in cards
                ]
                for anchor in anchors
            ],
            dtype=np.float32,
        )
        values = matrix.ravel()
        features.extend(values)
        features.extend((values.mean(), values.max(), values.sum()))
        features.append(
            max(
                matrix[0, first] + matrix[1, second]
                for first in range(3)
                for second in range(3)
                if first != second
            )
        )
    return features


def build_joint_features(cases, frequencies):
    rows = []
    feature_count = None
    for case in cases:
        if case is None:
            rows.append(None)
            continue
        anchors, profiles = case
        case_features = np.asarray(
            [
                joint_card_features(anchors, profile["complaints"], frequencies)
                for profile in profiles
            ],
            dtype=np.float32,
        )
        feature_count = case_features.shape[1]
        rows.append(case_features)
    if feature_count is None:
        raise ValueError("no parseable cases are available")
    for index, row in enumerate(rows):
        if row is None:
            rows[index] = np.zeros((8, feature_count), dtype=np.float32)
    return np.stack(rows)


def candidate_features(anchors, profile, frequencies):
    cards = profile["complaints"]
    features = []
    for card in list(anchors) + list(cards):
        features.extend(
            (
                math.log1p(max(0.0, float(card["relative_filing_day"]))),
                math.log1p(max(0.0, float(card["relative_filing_hour"]))),
            )
        )
    features.extend(relational_features(anchors, cards))
    features.extend(additional_features(anchors, cards, frequencies))
    return features


def build_base_features(cases, frequencies):
    rows = []
    feature_count = None
    for case in cases:
        if case is None:
            if feature_count is None:
                # The normal release always has valid rows; this is replaced once a
                # valid row establishes the deterministic feature width.
                rows.append(None)
            else:
                rows.append(np.zeros((8, feature_count), dtype=np.float32))
            continue
        anchors, profiles = case
        case_features = np.asarray(
            [candidate_features(anchors, profile, frequencies) for profile in profiles],
            dtype=np.float32,
        )
        feature_count = case_features.shape[1]
        rows.append(case_features)
    if feature_count is None:
        raise ValueError("no parseable cases are available")
    for i, row in enumerate(rows):
        if row is None:
            rows[i] = np.zeros((8, feature_count), dtype=np.float32)
    return np.stack(rows)


def flatten_case_rows(indices):
    indices = np.asarray(indices, dtype=np.int64)
    return (
        indices[:, None] * 8 + np.arange(8, dtype=np.int64)[None, :]
    ).ravel()


def binary_labels(targets, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return (
        np.arange(8, dtype=np.int64)[None, :] == targets[indices, None]
    ).astype(np.float32).ravel()


def build_categorical_table(cases):
    """Expose raw card values to CatBoost without using cross-case state."""
    rows = []
    for case in cases:
        if case is None:
            anchors = [{}, {}]
            profiles = [{"complaints": [{}, {}, {}]} for _ in range(8)]
        else:
            anchors, profiles = case
        anchors = sorted(
            anchors,
            key=lambda card: (
                float(card.get("relative_filing_hour", 0)),
                str(card.get("id", "")),
            ),
        )
        for profile in profiles:
            complaints = sorted(
                profile["complaints"],
                key=lambda card: (
                    float(card.get("relative_filing_hour", 0)),
                    str(card.get("id", "")),
                ),
            )
            row = {}
            for side, cards in (("a", anchors), ("p", complaints)):
                for position, card in enumerate(cards):
                    for field in CAT_FIELDS:
                        row[f"{side}{position}_{field}"] = str(
                            card.get(field, "__missing__")
                        )
                    for group_index, group in enumerate(COMPOSITE_GROUPS):
                        row[f"{side}{position}_composite{group_index}"] = "|".join(
                            str(card.get(field, "__missing__")) for field in group
                        )
                for field in CAT_FIELDS:
                    values = [
                        str(card.get(field, "__missing__")) for card in cards
                    ]
                    row[f"{side}_sequence_{field}"] = "|".join(values)
                    row[f"{side}_set_{field}"] = "|".join(sorted(values))
                for group_index, group in enumerate(COMPOSITE_GROUPS):
                    values = [
                        "|".join(
                            str(card.get(field, "__missing__")) for field in group
                        )
                        for card in cards
                    ]
                    row[f"{side}_sequence_composite{group_index}"] = "#".join(values)
                    row[f"{side}_set_composite{group_index}"] = "#".join(
                        sorted(values)
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def combined_numeric_features(cases, frequencies):
    return np.concatenate(
        (
            build_base_features(cases, frequencies),
            build_joint_features(cases, frequencies),
        ),
        axis=2,
    ).astype(np.float32, copy=False)


def build_model_frame(categorical, numeric):
    columns = [f"numeric_{index}" for index in range(numeric.shape[-1])]
    numeric_frame = pd.DataFrame(
        numeric.reshape(-1, numeric.shape[-1]), columns=columns
    )
    return pd.concat(
        (categorical.reset_index(drop=True), numeric_frame), axis=1
    )


def make_query_pool(frame, categorical_columns, targets, indices):
    rows = flatten_case_rows(indices)
    return Pool(
        frame.iloc[rows],
        binary_labels(targets, indices),
        cat_features=categorical_columns,
        group_id=np.repeat(np.arange(len(indices), dtype=np.int64), 8),
    )


def make_query_ranker(config, seed):
    return CatBoostRanker(
        loss_function="QuerySoftMax",
        verbose=False,
        random_seed=seed,
        thread_count=THREADS,
        allow_writing_files=False,
        max_ctr_complexity=2,
        **config,
    )


def fit_query_ranker(
    frame, categorical_columns, targets, train_indices, config, seed
):
    model = make_query_ranker(config, seed)
    model.fit(
        make_query_pool(
            frame, categorical_columns, targets, np.asarray(train_indices)
        )
    )
    return model


def predict_query_ranker(model, frame, indices=None):
    if indices is None:
        rows = np.arange(len(frame), dtype=np.int64)
        case_count = len(frame) // 8
    else:
        indices = np.asarray(indices, dtype=np.int64)
        rows = flatten_case_rows(indices)
        case_count = len(indices)
    return np.asarray(model.predict(frame.iloc[rows])).reshape(case_count, 8)


def tune_query_ranker(
    frame, categorical_columns, targets, train_indices, valid_indices
):
    results = []
    for config_index, config in enumerate(QSM_CONFIGS):
        started = time.perf_counter()
        model = fit_query_ranker(
            frame,
            categorical_columns,
            targets,
            train_indices,
            config,
            SEED + 1000 + config_index,
        )
        scores = predict_query_ranker(model, frame, valid_indices)
        accuracy = float(
            np.mean(scores.argmax(axis=1) == targets[valid_indices])
        )
        duration = time.perf_counter() - started
        print(
            f"QuerySoftMax HPO {config_index}: score={accuracy:.6f}, "
            f"seconds={duration:.2f}, config={config}"
        )
        results.append((accuracy, -config_index, dict(config)))
    return max(results, key=lambda item: (item[0], item[1]))[2]


def fit_linear_ranker(numeric, targets, train_indices, regularization):
    rows = flatten_case_rows(train_indices)
    flat = numeric.reshape(-1, numeric.shape[-1])
    scaler = StandardScaler()
    train_features = scaler.fit_transform(flat[rows])
    model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(train_features, binary_labels(targets, train_indices))
    return scaler, model


def predict_linear_ranker(fitted, numeric, indices=None):
    scaler, model = fitted
    selected = numeric if indices is None else numeric[indices]
    flat = selected.reshape(-1, selected.shape[-1])
    probabilities = model.predict_proba(scaler.transform(flat))[:, 1]
    return probabilities.reshape(len(selected), 8)


def tune_linear_ranker(numeric, targets, train_indices, valid_indices):
    results = []
    for regularization in LINEAR_CS:
        started = time.perf_counter()
        fitted = fit_linear_ranker(
            numeric, targets, train_indices, regularization
        )
        scores = predict_linear_ranker(fitted, numeric, valid_indices)
        accuracy = float(
            np.mean(scores.argmax(axis=1) == targets[valid_indices])
        )
        duration = time.perf_counter() - started
        print(
            f"linear HPO C={regularization}: score={accuracy:.6f}, "
            f"seconds={duration:.2f}"
        )
        results.append((accuracy, -regularization, regularization))
    return max(results, key=lambda item: (item[0], item[1]))[2]


def row_zscores(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
    return (scores - mean) / std


def tune_blend(query_scores, linear_scores, targets):
    query_normalized = row_zscores(query_scores)
    linear_normalized = row_zscores(linear_scores)
    results = []
    for query_weight in BLEND_WEIGHTS:
        blended = (
            query_weight * query_normalized
            + (1.0 - query_weight) * linear_normalized
        )
        accuracy = float(np.mean(blended.argmax(axis=1) == targets))
        print(
            f"blend HPO query_weight={query_weight:.1f}: "
            f"OOF score={accuracy:.6f}"
        )
        results.append(
            (accuracy, -abs(query_weight - 0.5), query_weight)
        )
    return max(results)[2]


def validate_output(test_df, submission):
    expected_columns = ["case_id", "continuation_chain"]
    problems = []
    if list(submission.columns) != expected_columns:
        problems.append("incorrect column names or order")
    if len(submission) != len(test_df):
        problems.append("row count mismatch")
    if submission["case_id"].duplicated().any():
        problems.append("duplicate case_id")
    if set(submission["case_id"].astype(str)) != set(test_df["case_id"].astype(str)):
        problems.append("case_id set mismatch")
    for row_number, text in enumerate(submission["continuation_chain"]):
        try:
            chain = json.loads(text)
            if len(chain) != 3 or len(set(chain)) != 3:
                raise ValueError("chain must contain three distinct IDs")
            if not all(
                isinstance(card_id, str)
                and len(card_id) == 7
                and card_id[0] == "p"
                and card_id[3:5] == "_c"
                and card_id[1:3].isdigit()
                and 1 <= int(card_id[1:3]) <= 8
                and card_id[5:7].isdigit()
                and 1 <= int(card_id[5:7]) <= 3
                for card_id in chain
            ):
                raise ValueError("unknown card ID")
        except Exception as exc:
            problems.append(f"malformed chain at row {row_number}: {exc}")
            break
    if problems:
        print("warning: final validation failed; retaining early placeholder: " + "; ".join(problems))
        return False
    return True


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    required = (
        public_dir / "train.csv",
        public_dir / "train_labels.csv",
        public_dir / "test.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))

    seed_everything(SEED)
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning
    )

    train_df = pd.read_csv(required[0])
    labels_df = pd.read_csv(required[1])
    test_df = pd.read_csv(required[2])

    # The valid output exists before JSON parsing, feature extraction, or training.
    write_placeholder(test_df, submission_out)

    label_map = dict(
        zip(labels_df["case_id"].astype(str), labels_df["continuation_chain"])
    )
    train_cases, targets, kept_rows = load_cases(
        train_df, training=True, label_map=label_map
    )
    if len(kept_rows) != len(train_df):
        train_df = train_df.iloc[kept_rows].reset_index(drop=True)
    test_cases = load_cases(test_df, training=False)
    all_train = np.arange(len(train_cases), dtype=np.int64)

    print(
        f"plan: {len(QSM_CONFIGS)} QuerySoftMax configs, "
        f"{len(LINEAR_CS)} linear configs, then {N_FOLDS} folds with "
        "one categorical query ranker and one regularized linear ranker; "
        "CPU only; ceiling=5400 seconds"
    )
    total_started = time.perf_counter()

    train_categorical = build_categorical_table(train_cases)
    test_categorical = build_categorical_table(test_cases)
    categorical_columns = list(train_categorical.columns)

    hpo_train, hpo_valid = train_test_split(
        all_train,
        test_size=0.25,
        random_state=SEED,
        stratify=targets,
    )
    hpo_frequencies = fit_frequencies(train_cases, hpo_train)
    hpo_numeric = combined_numeric_features(train_cases, hpo_frequencies)
    hpo_frame = build_model_frame(train_categorical, hpo_numeric)
    query_config = tune_query_ranker(
        hpo_frame,
        categorical_columns,
        targets,
        hpo_train,
        hpo_valid,
    )
    linear_c = tune_linear_ranker(
        hpo_numeric, targets, hpo_train, hpo_valid
    )
    print(f"selected QuerySoftMax config: {query_config}")
    print(f"selected linear C={linear_c}")

    oof_query = np.zeros((len(train_cases), 8), dtype=np.float32)
    oof_linear = np.zeros((len(train_cases), 8), dtype=np.float32)
    test_query = np.zeros((len(test_cases), 8), dtype=np.float32)
    test_linear = np.zeros((len(test_cases), 8), dtype=np.float32)
    fold_seconds = []
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(all_train, targets), start=1
    ):
        fold_started = time.perf_counter()
        frequencies = fit_frequencies(train_cases, train_indices)
        train_numeric = combined_numeric_features(train_cases, frequencies)
        test_numeric = combined_numeric_features(test_cases, frequencies)
        train_frame = build_model_frame(train_categorical, train_numeric)
        test_frame = build_model_frame(test_categorical, test_numeric)

        query_model = fit_query_ranker(
            train_frame,
            categorical_columns,
            targets,
            train_indices,
            query_config,
            SEED + 10000 * fold,
        )
        oof_query[valid_indices] = predict_query_ranker(
            query_model, train_frame, valid_indices
        )
        test_query += (
            predict_query_ranker(query_model, test_frame) / N_FOLDS
        )

        linear_model = fit_linear_ranker(
            train_numeric, targets, train_indices, linear_c
        )
        oof_linear[valid_indices] = predict_linear_ranker(
            linear_model, train_numeric, valid_indices
        )
        test_linear += (
            predict_linear_ranker(linear_model, test_numeric) / N_FOLDS
        )

        duration = time.perf_counter() - fold_started
        fold_seconds.append(duration)
        fold_scores = {
            "query": float(
                np.mean(
                    oof_query[valid_indices].argmax(axis=1)
                    == targets[valid_indices]
                )
            ),
            "linear": float(
                np.mean(
                    oof_linear[valid_indices].argmax(axis=1)
                    == targets[valid_indices]
                )
            ),
        }
        print(
            f"fold {fold}/{N_FOLDS}: seconds={duration:.2f}, "
            f"scores={fold_scores}"
        )

    query_score = float(np.mean(oof_query.argmax(axis=1) == targets))
    linear_score = float(np.mean(oof_linear.argmax(axis=1) == targets))
    print(f"OOF categorical query ranker ProfileAccuracy={query_score:.6f}")
    print(f"OOF linear similarity ranker ProfileAccuracy={linear_score:.6f}")

    query_weight = tune_blend(oof_query, oof_linear, targets)
    oof_scores = (
        query_weight * row_zscores(oof_query)
        + (1.0 - query_weight) * row_zscores(oof_linear)
    )
    official_score = float(np.mean(oof_scores.argmax(axis=1) == targets))
    print(
        f"selected query weight={query_weight:.1f}; "
        f"OOF official score={official_score:.6f}"
    )

    test_scores = (
        query_weight * row_zscores(test_query)
        + (1.0 - query_weight) * row_zscores(test_linear)
    )
    selected_profiles = test_scores.argmax(axis=1)

    chains = []
    for row_index, selected in enumerate(selected_profiles):
        try:
            case = test_cases[row_index]
            if case is None:
                raise ValueError("unparseable case")
            chain = sorted_card_ids(case[1][int(selected)])
            chains.append(json.dumps(chain, separators=(",", ":")))
        except Exception as exc:
            print(f"warning: prediction fallback at test row {row_index}: {exc}")
            try:
                fallback = sorted_card_ids(test_cases[row_index][1][0])
                chains.append(json.dumps(fallback, separators=(",", ":")))
            except Exception:
                chains.append('["p01_c01","p01_c02","p01_c03"]')

    submission = pd.DataFrame(
        {
            "case_id": test_df["case_id"].astype(str),
            "continuation_chain": chains,
        }
    )
    if validate_output(test_df, submission):
        submission.to_csv(submission_out, index=False)

    total_seconds = time.perf_counter() - total_started
    mean_fold = float(np.mean(fold_seconds))
    projected = mean_fold * N_FOLDS + (total_seconds - sum(fold_seconds))
    print(
        f"timing: measured mean={mean_fold:.2f} seconds/fold; "
        f"projected fixed plan={projected:.2f} seconds vs ceiling=5400 seconds; "
        f"actual={total_seconds:.2f} seconds"
    )
    print(
        f"submission: rows={len(submission)}, columns={list(submission.columns)}, "
        f"path={submission_out}"
    )


if __name__ == "__main__":
    main()
