#!/usr/bin/env python3
"""Fine-tune ChemBERTa and graph-aware models for batch provenance prediction."""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from transformers import AutoModel, AutoTokenizer

SEED = 714
LABELS = [
    "coherent_batch",
    "pair_index_swap",
    "three_record_cycle",
    "near_analog_transplant",
    "partial_reference_drift",
]
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "public"
OUTPUT = ROOT / "working" / "submission.csv"
PRETRAINED_CHEMBERTA = "DeepChem/ChemBERTa-5M-MLM"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def claim_features(claim: dict) -> dict[str, float]:
    """Parse the radius-2 graph and radius-3 profile into sparse graph tokens."""
    features: dict[str, float] = {
        "m=" + str(claim["multiplicity"]): 1.0,
        "s=" + str(claim["solvent"]): 1.0,
        "lc=" + str(claim["local_class"]): 1.0,
    }
    environment = str(claim["environment"])
    center, paths_text = environment.split(";paths=", 1)
    root = center[7:]
    features["root=" + root] = 1.0
    for token in root.split("."):
        features["rootattr=" + token] = 1.0

    paths = paths_text.split("|") if paths_text else []
    features["env_npaths"] = float(len(paths))
    for path in paths:
        features["path=" + path] = features.get("path=" + path, 0.0) + 1.0
        continuation = path[len(root) :] if path.startswith(root) else path
        features["cont=" + continuation] = features.get("cont=" + continuation, 0.0) + 1.0
        parts = re.split(r"([\-=#:])", path)
        nodes, bonds = parts[::2], parts[1::2]
        features["pathlen"] = features.get("pathlen", 0.0) + len(bonds)
        for depth, node in enumerate(nodes[1:], 1):
            key = f"n{depth}={node}"
            features[key] = features.get(key, 0.0) + 1.0
            key = f"e{depth}={node.split('.')[0]}"
            features[key] = features.get(key, 0.0) + 1.0
        for depth, bond in enumerate(bonds, 1):
            key = f"b{depth}={bond}"
            features[key] = features.get(key, 0.0) + 1.0
            edge = nodes[depth - 1] + ">" + bond + ">" + nodes[depth]
            key = f"edge{depth}={edge}"
            features[key] = features.get(key, 0.0) + 1.0

    profile = str(claim["environment_profile"])
    for item in profile.split(";"):
        if "=" not in item:
            continue
        key, value = item.rsplit("=", 1)
        try:
            features["p:" + key] = float(value)
        except ValueError:
            features["pcat:" + item] = 1.0
    # Useful only when a lossy public profile recurs; unseen profiles fall back to graph tokens.
    features["profile=" + profile] = 1.0
    return features


def summarize_candidates(rows: list[list[float]], width: int, rank_columns: list[int]) -> list[float]:
    """Permutation-invariant pooling while retaining the strongest candidate rows."""
    if not rows:
        return [999.0] * (width * (4 + 2 * len(rank_columns)))
    values = np.asarray(rows, dtype=float)
    chunks = [
        np.min(values, axis=0),
        np.median(values, axis=0),
        np.max(values, axis=0),
        values[np.argmin(values[:, 0])],
    ]
    for column in rank_columns:
        order = np.argsort(values[:, column])
        chunks.append(values[order[0]])
        chunks.append(values[order[min(1, len(order) - 1)]])
    return np.concatenate(chunks).tolist()


def forensic_features(claims: list[dict], predicted_clean: np.ndarray) -> np.ndarray:
    """Fit all legal swaps/cycles/drift subsets and summarize their residual evidence."""
    observed = np.asarray([claim["shift_ppm"] for claim in claims], dtype=float)
    predicted_clean = np.asarray(predicted_clean, dtype=float)
    residual = observed - predicted_clean
    centered = residual - np.median(residual)
    sorted_abs = np.sort(np.abs(centered))[::-1]
    result = [
        np.mean(centered**2),
        np.mean(np.abs(centered)),
        np.std(residual),
        np.ptp(residual),
        *sorted_abs,
        np.sum(sorted_abs > 2),
        np.sum(sorted_abs > 4),
        np.sum(sorted_abs > 6),
        np.sum(sorted_abs > 10),
    ]

    groups: dict[str, list[int]] = defaultdict(list)
    for index, claim in enumerate(claims):
        groups[str(claim["local_class"])].append(index)
    sizes = sorted((len(indices) for indices in groups.values()), reverse=True)
    result.extend(
        [
            len(groups),
            max(sizes),
            sum(size >= 2 for size in sizes),
            sum(size >= 3 for size in sizes),
            sum(math.comb(size, 2) for size in sizes if size >= 2),
            sum(math.comb(size, 3) for size in sizes if size >= 3),
        ]
    )

    pair_rows: list[list[float]] = []
    triple_rows: list[list[float]] = []
    all_indices = set(range(12))
    for indices in groups.values():
        for first, second in itertools.combinations(indices, 2):
            outside = list(all_indices - {first, second})
            outside_offset = np.median(residual[outside])
            deviations = residual[[first, second]] - outside_offset
            swapped = residual.copy()
            swapped[first] = observed[first] - predicted_clean[second]
            swapped[second] = observed[second] - predicted_clean[first]
            swapped -= np.median(swapped)
            observed_separation = observed[first] - observed[second]
            predicted_separation = predicted_clean[first] - predicted_clean[second]
            outside_centered = residual[outside] - np.median(residual[outside])
            pair_rows.append(
                [
                    np.mean(swapped**2),
                    np.mean(np.abs(swapped)),
                    abs(np.sum(deviations)),
                    abs(deviations[0] - deviations[1]),
                    np.mean(np.abs(deviations)),
                    np.min(np.abs(deviations)),
                    np.max(np.abs(deviations)),
                    abs(abs(observed_separation) - abs(predicted_separation)),
                    abs(observed_separation + predicted_separation),
                    np.std(residual[outside]),
                    np.mean(outside_centered**2),
                ]
            )

        for combination in itertools.combinations(indices, 3):
            first, second, third = combination
            outside = list(all_indices - set(combination))
            outside_offset = np.median(residual[outside])
            deviations = residual[list(combination)] - outside_offset
            cycle_costs, cycle_maes = [], []
            for permutation in ((second, third, first), (third, first, second)):
                cycled = residual.copy()
                for destination, source in zip(combination, permutation):
                    cycled[destination] = observed[destination] - predicted_clean[source]
                cycled -= np.median(cycled)
                cycle_costs.append(np.mean(cycled**2))
                cycle_maes.append(np.mean(np.abs(cycled)))
            outside_centered = residual[outside] - np.median(residual[outside])
            triple_rows.append(
                [
                    min(cycle_costs),
                    min(cycle_maes),
                    abs(np.mean(deviations)),
                    np.std(deviations),
                    np.ptp(deviations),
                    abs(np.sum(deviations)),
                    np.mean(np.abs(deviations)),
                    np.min(np.abs(deviations)),
                    np.max(np.abs(deviations)),
                    np.std(residual[outside]),
                    np.mean(outside_centered**2),
                    np.ptp(deviations),
                    abs(np.sum(np.sign(deviations))),
                ]
            )

    result.extend(summarize_candidates(pair_rows, 11, [0, 2, 3, 7, 10]))
    result.extend(summarize_candidates(triple_rows, 13, [0, 2, 3, 4, 10]))
    return np.asarray(result, dtype=np.float32)


def make_batch_matrix(records: list[dict], predictions: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            forensic_features(record["claims"], predictions[12 * index : 12 * (index + 1)])
            for index, record in enumerate(records)
        ]
    )


def compact_forensics(expanded: np.ndarray) -> np.ndarray:
    """The four aggregate blocks from each expanded 362-feature view."""
    return np.hstack([expanded[:, :70], expanded[:, 180:232]])




class ChemBERTaShiftRegressor(nn.Module):
    """Regression adapter that fine-tunes every layer of pretrained ChemBERTa."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path, add_pooling_layer=False
        ).float()
        hidden_size = self.encoder.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = input_ids.ne(1)
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled /= attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.regression_head(pooled).squeeze(-1)


def chemistry_text(claim: dict) -> str:
    """Serialize public chemistry fields; local_class and IDs are deliberately absent."""
    return " ".join(
        [
            str(claim["multiplicity"]),
            str(claim["solvent"]),
            str(claim["environment_profile"]),
            str(claim["environment"]),
        ]
    )


def predict_chemberta(
    model: ChemBERTaShiftRegressor,
    token_ids: np.ndarray,
    indices: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    batch_size = 512 if device.type != "cpu" else 128
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = torch.from_numpy(
                token_ids[batch_indices].astype(np.int64)
            ).to(device)
            prediction = model(batch).cpu().numpy()
            output[start : start + len(batch_indices)] = (
                prediction * target_std + target_mean
            )
    return output


def crossfit_chemberta(
    flat_train: list[dict],
    flat_test: list[dict],
    clean: np.ndarray,
    profile_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Supervised two-fold adaptation of a chemistry-pretrained encoder."""
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_CHEMBERTA
    )
    train_text = [chemistry_text(claim) for claim in flat_train]
    test_text = [chemistry_text(claim) for claim in flat_test]
    train_ids = tokenizer(
        train_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=False,
        return_tensors="np",
    )["input_ids"].astype(np.uint16)
    test_ids = tokenizer(
        test_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=False,
        return_tensors="np",
    )["input_ids"].astype(np.uint16)
    del train_text, test_text, tokenizer

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_batch_size = 256 if device.type != "cpu" else 64
    oof = np.zeros(len(clean), dtype=np.float32)
    test_predictions = []
    folds = GroupKFold(n_splits=2, shuffle=True, random_state=1701)
    print(
        f"Fine-tuning pretrained ChemBERTa on {device.type} ...",
        flush=True,
    )

    for fold, (train_indices, valid_indices) in enumerate(
        folds.split(train_ids, clean, groups=profile_groups), start=1
    ):
        torch.manual_seed(SEED + fold)
        model = ChemBERTaShiftRegressor(PRETRAINED_CHEMBERTA).to(device)
        target_mean = float(clean[train_indices].mean())
        target_std = float(clean[train_indices].std())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, weight_decay=0.01
        )
        steps_per_epoch = math.ceil(len(train_indices) / train_batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            total_steps=2 * steps_per_epoch,
            pct_start=0.08,
            anneal_strategy="cos",
            div_factor=5,
            final_div_factor=20,
        )
        generator = np.random.default_rng(900 + fold)
        model.train()
        for epoch in range(2):
            order = generator.permutation(train_indices)
            losses = []
            for start in range(0, len(order), train_batch_size):
                indices = order[start : start + train_batch_size]
                inputs = torch.from_numpy(
                    train_ids[indices].astype(np.int64)
                ).to(device)
                targets = torch.from_numpy(
                    ((clean[indices] - target_mean) / target_std).astype(
                        np.float32
                    )
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = nn.functional.mse_loss(prediction, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            print(
                f"  ChemBERTa fold {fold}/2 epoch {epoch + 1}/2 "
                f"loss: {np.mean(losses):.5f}",
                flush=True,
            )

        oof[valid_indices] = predict_chemberta(
            model,
            train_ids,
            valid_indices,
            target_mean,
            target_std,
            device,
        )
        test_predictions.append(
            predict_chemberta(
                model,
                test_ids,
                np.arange(len(test_ids)),
                target_mean,
                target_std,
                device,
            )
        )
        print(
            f"  ChemBERTa fold {fold}/2 held-profile MAE: "
            f"{np.mean(np.abs(oof[valid_indices] - clean[valid_indices])):.4f} ppm",
            flush=True,
        )
        del model, optimizer, scheduler
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    return oof, np.mean(test_predictions, axis=0)


def main() -> None:
    global DATA, OUTPUT
    if len(sys.argv) == 3:
        DATA = Path(sys.argv[1]).expanduser().resolve()
        OUTPUT = Path(sys.argv[2]).expanduser().resolve()
    elif len(sys.argv) != 1:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    np.random.seed(SEED)
    train = load_jsonl(DATA / "train.jsonl")
    test = load_jsonl(DATA / "test.jsonl")
    calibration = pd.read_csv(DATA / "shift_calibration.csv")
    if len(train) != 20_000 or len(test) != 4_000 or len(calibration) != 240_000:
        raise ValueError("Unexpected challenge data dimensions")

    flat_train = [claim for record in train for claim in record["claims"]]
    flat_test = [claim for record in test for claim in record["claims"]]
    expected_batches = np.repeat([record["id"] for record in train], 12)
    expected_claims = [claim["claim_id"] for claim in flat_train]
    if not np.array_equal(calibration["batch_id"].astype(str).to_numpy(), expected_batches):
        raise ValueError("Calibration rows are not aligned with train.jsonl")
    if calibration["claim_id"].astype(str).tolist() != expected_claims:
        raise ValueError("Calibration claim IDs are not aligned with train.jsonl")

    print("Encoding radius-2 graphs and radius-3 profiles ...", flush=True)
    vectorizer = DictVectorizer(dtype=np.float32)
    train_feature_dicts = [claim_features(claim) for claim in flat_train]
    test_feature_dicts = [claim_features(claim) for claim in flat_test]
    claim_train = vectorizer.fit_transform(train_feature_dicts).tocsr()
    claim_test = vectorizer.transform(test_feature_dicts).tocsr()
    del train_feature_dicts, test_feature_dicts

    clean = calibration["clean_shift_ppm"].to_numpy(dtype=np.float32)
    profile_groups = calibration["environment_profile"].astype(str).to_numpy()
    chemberta_oof, chemberta_test = crossfit_chemberta(
        flat_train, flat_test, clean, profile_groups
    )
    folds = GroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    oof_standard = np.zeros_like(clean)
    oof_high_capacity = np.zeros_like(clean)
    test_standard, test_high_capacity = [], []

    standard_params = {
        "objective": "regression_l2",
        "metric": ["l1", "rmse"],
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "num_threads": 8,
        "seed": SEED,
        "force_col_wise": True,
    }
    high_capacity_params = dict(standard_params)
    high_capacity_params.update(
        learning_rate=0.06,
        num_leaves=256,
        min_data_in_leaf=25,
        feature_fraction=0.85,
        bagging_fraction=0.85,
    )

    print("Cross-fitting clean-shift compatibility models ...", flush=True)
    for fold, (train_indices, valid_indices) in enumerate(
        folds.split(claim_train, clean, groups=profile_groups), start=1
    ):
        train_set = lgb.Dataset(claim_train[train_indices], label=clean[train_indices], free_raw_data=False)
        valid_set = lgb.Dataset(claim_train[valid_indices], label=clean[valid_indices], reference=train_set, free_raw_data=False)
        standard = lgb.train(
            standard_params,
            train_set,
            num_boost_round=1000,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
        )
        high_capacity = lgb.train(
            high_capacity_params,
            train_set,
            num_boost_round=800,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        oof_standard[valid_indices] = standard.predict(claim_train[valid_indices], num_iteration=standard.best_iteration)
        oof_high_capacity[valid_indices] = high_capacity.predict(
            claim_train[valid_indices], num_iteration=high_capacity.best_iteration
        )
        test_standard.append(standard.predict(claim_test, num_iteration=standard.best_iteration))
        test_high_capacity.append(
            high_capacity.predict(claim_test, num_iteration=high_capacity.best_iteration)
        )
        fold_mae = np.mean(np.abs((oof_standard[valid_indices] + oof_high_capacity[valid_indices]) / 2 - clean[valid_indices]))
        print(f"  fold {fold}/4 compatibility MAE: {fold_mae:.4f} ppm", flush=True)
        del train_set, valid_set, standard, high_capacity

    test_standard_mean = np.mean(test_standard, axis=0)
    test_high_capacity_mean = np.mean(test_high_capacity, axis=0)
    oof_ensemble = (oof_standard + oof_high_capacity) / 2
    test_ensemble = (test_standard_mean + test_high_capacity_mean) / 2
    print(f"Cross-fitted compatibility MAE: {np.mean(np.abs(oof_ensemble - clean)):.4f} ppm", flush=True)

    print("Fitting joint twelve-claim provenance model ...", flush=True)
    graph_noise_scale = 0.92
    chemistry_noise_scale = 0.96
    train_views = [
        make_batch_matrix(
            train, clean + graph_noise_scale * (oof_standard - clean)
        ),
        make_batch_matrix(
            train, clean + graph_noise_scale * (oof_high_capacity - clean)
        ),
        make_batch_matrix(
            train, clean + graph_noise_scale * (oof_ensemble - clean)
        ),
        make_batch_matrix(
            train, clean + chemistry_noise_scale * (chemberta_oof - clean)
        ),
    ]
    test_views = [
        make_batch_matrix(test, test_standard_mean),
        make_batch_matrix(test, test_high_capacity_mean),
        make_batch_matrix(test, test_ensemble),
        make_batch_matrix(test, chemberta_test),
    ]
    expanded_train = np.hstack(train_views)
    expanded_test = np.hstack(test_views)
    compact_train = np.hstack([compact_forensics(view) for view in train_views])
    compact_test = np.hstack([compact_forensics(view) for view in test_views])
    targets = np.asarray([LABELS.index(record["audit_signature"]) for record in train], dtype=np.int8)

    extra_trees = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=7,
        max_features=0.3,
        n_jobs=8,
        class_weight="balanced",
        random_state=2714,
    )
    extra_trees.fit(expanded_train, targets)
    tree_probabilities = extra_trees.predict_proba(expanded_test)

    batch_booster = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=8,
        reg_alpha=0.2,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=8,
        random_state=SEED,
        tree_method="hist",
    )
    batch_booster.fit(compact_train, targets, verbose=False)
    boosted_probabilities = batch_booster.predict_proba(compact_test)

    probabilities = 0.3 * tree_probabilities + 0.7 * boosted_probabilities
    probabilities = probabilities ** (1 / 0.95)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(probabilities, columns=LABELS)
    submission.insert(0, "id", [record["id"] for record in test])
    submission.to_csv(OUTPUT, index=False, float_format="%.9f")
    print("Hard prediction counts:", dict(zip(LABELS, np.bincount(np.argmax(probabilities, axis=1), minlength=5))))
    print(f"Wrote {OUTPUT} with {len(submission)} rows", flush=True)


if __name__ == "__main__":
    main()
