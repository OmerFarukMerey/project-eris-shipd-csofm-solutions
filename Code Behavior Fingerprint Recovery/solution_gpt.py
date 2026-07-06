#!/usr/bin/env python3
"""
Behavior-fingerprint predictor for the coding-assistant hidden-test challenge.

Reads train.csv/test.csv, learns from train.csv answer_json values, and writes:
    ./working/submission.csv

The script uses only public files: train.csv, test.csv, and optional sample_submission.csv.
No hardcoded id-to-answer mapping is used.
"""

import ast
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This solution requires scikit-learn. In Kaggle notebooks it is normally preinstalled."
    ) from exc


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# File discovery
# -----------------------------

def find_dataset_dir() -> Path:
    """Find a directory containing both train.csv and test.csv."""
    candidate_roots = []

    env_dir = os.environ.get("DATA_DIR") or os.environ.get("INPUT_DIR")
    if env_dir:
        candidate_roots.append(Path(env_dir))

    candidate_roots.extend([
        Path.cwd(),
        Path.cwd() / "data",
        Path("/kaggle/input"),
        Path("/mnt/data"),
    ])

    seen = set()
    roots = []
    for root in candidate_roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen or not root.exists():
            continue
        roots.append(root)
        seen.add(root)

    # First prefer exact directory hits.
    for root in roots:
        if (root / "train.csv").exists() and (root / "test.csv").exists():
            return root

    # Then recursively search common input roots.
    for root in roots:
        try:
            train_paths = list(root.rglob("train.csv"))
        except Exception:
            continue
        for train_path in train_paths:
            parent = train_path.parent
            if (parent / "test.csv").exists():
                return parent

    raise FileNotFoundError(
        "Could not find train.csv and test.csv. Put them in the current directory, "
        "set DATA_DIR, or run inside a Kaggle input environment."
    )


# -----------------------------
# JSON/text helpers
# -----------------------------

def safe_json_loads(value, default=None):
    if default is None:
        default = {}
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return default
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def compact_json(value) -> str:
    parsed = safe_json_loads(value, default=value)
    try:
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value)


def normalize_code(code) -> str:
    """Light normalization that keeps syntax and operator clues."""
    code = "" if pd.isna(code) else str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    # Preserve indentation roughly while reducing very long whitespace runs.
    code = re.sub(r"[ \t]+", " ", code)
    return code.strip()


def row_text(row, mode="full") -> str:
    prompt = str(row.get("prompt", ""))
    problem = str(row.get("problem_statement", ""))
    code = normalize_code(row.get("candidate_code", ""))
    blueprint = compact_json(row.get("test_blueprint_json", ""))
    length = str(row.get("mask_length", ""))

    if mode == "code_blueprint":
        parts = [code, blueprint, f"MASK_LENGTH={length}"]
    elif mode == "problem_code":
        parts = [prompt, problem, code, f"MASK_LENGTH={length}"]
    elif mode == "blueprint":
        parts = [blueprint, f"MASK_LENGTH={length}"]
    else:
        parts = [prompt, problem, code, blueprint, f"MASK_LENGTH={length}"]

    return "\n".join(parts)


def parse_answer_json(value):
    obj = safe_json_loads(value, default={})
    if not isinstance(obj, dict) or "pass_mask" not in obj:
        raise ValueError(f"Bad answer_json: {value!r}")
    return obj


def build_answer_json(pass_mask: str) -> str:
    fail_count = pass_mask.count("0")
    pass_count = pass_mask.count("1")
    obj = {
        "pass_mask": pass_mask,
        "fail_count": int(fail_count),
        "score_bucket": f"s{pass_count:02d}",
    }
    return json.dumps(obj, separators=(",", ":"))


# -----------------------------
# Metric for local validation
# -----------------------------

def failed_f1(true_mask: str, pred_mask: str) -> float:
    true_failed = {i for i, ch in enumerate(true_mask) if ch == "0"}
    pred_failed = {i for i, ch in enumerate(pred_mask[:len(true_mask)]) if ch == "0"}

    # Length mismatches are not expected because we force correct length, but if they occur,
    # extra/missing positions are treated as submitted failures/non-agreements elsewhere.
    if not true_failed and not pred_failed:
        return 1.0
    if not true_failed or not pred_failed:
        return 0.0

    correct = len(true_failed & pred_failed)
    if correct == 0:
        return 0.0
    precision = correct / len(pred_failed)
    recall = correct / len(true_failed)
    return 2 * precision * recall / (precision + recall)


def row_score(true_mask: str, pred_mask: str) -> float:
    L = len(true_mask)
    pred = pred_mask[:L].ljust(L, "?")

    hamming_agreement = sum(1 for a, b in zip(true_mask, pred) if a == b) / max(L, 1)
    f1 = failed_f1(true_mask, pred)
    mask_similarity = 0.55 * hamming_agreement + 0.45 * f1

    true_fail = true_mask.count("0")
    pred_fail = pred_mask.count("0")
    fail_count_score = max(0.0, 1.0 - abs(true_fail - pred_fail) / max(L, 1))

    true_bucket = f"s{true_mask.count('1'):02d}"
    pred_bucket = f"s{pred_mask.count('1'):02d}"
    bucket_exact = 1.0 if true_bucket == pred_bucket else 0.0
    exact_object = 1.0 if true_mask == pred_mask else 0.0

    return (
        0.50 * mask_similarity
        + 0.22 * fail_count_score
        + 0.18 * bucket_exact
        + 0.10 * exact_object
    )


def mean_score(true_masks, pred_masks) -> float:
    return float(np.mean([row_score(t, p) for t, p in zip(true_masks, pred_masks)]))


# -----------------------------
# KNN predictor
# -----------------------------

def weighted_top_indices(similarities, candidate_indices, k):
    if len(candidate_indices) == 0:
        return [], np.array([], dtype=float)

    sims = similarities[candidate_indices]
    k = min(k, len(candidate_indices))
    if k <= 0:
        return [], np.array([], dtype=float)

    # argpartition is faster, then sort the selected candidates.
    if len(sims) > k:
        local = np.argpartition(-sims, k - 1)[:k]
        local = local[np.argsort(-sims[local])]
    else:
        local = np.argsort(-sims)

    idx = np.asarray(candidate_indices, dtype=int)[local]
    top_sims = sims[local]

    # Similarities can occasionally be all zero. Use a stable soft weighting.
    weights = np.maximum(top_sims, 0.0) + 1e-6
    weights = weights / weights.sum()
    return idx.tolist(), weights


def probs_and_count_from_neighbors(train_masks, train_fail_counts, top_idx, weights, L, count_mode):
    if not top_idx:
        return np.ones(L, dtype=float), 0.0

    pass_probs = np.zeros(L, dtype=float)
    fail_counts = []

    for idx, w in zip(top_idx, weights):
        mask = train_masks[idx]
        bits = np.array([1.0 if ch == "1" else 0.0 for ch in mask[:L]], dtype=float)
        if len(bits) < L:
            bits = np.pad(bits, (0, L - len(bits)), constant_values=1.0)
        pass_probs += w * bits
        fail_counts.append(train_fail_counts[idx])

    fail_counts = np.asarray(fail_counts, dtype=float)

    if count_mode == "nearest":
        estimated_fail = float(fail_counts[0])
    elif count_mode == "median":
        estimated_fail = float(np.median(fail_counts))
    elif count_mode == "threshold":
        estimated_fail = float(np.sum(pass_probs < 0.5))
    else:
        estimated_fail = float(np.sum(weights * fail_counts))

    estimated_fail = max(0.0, min(float(L), estimated_fail))
    return pass_probs, estimated_fail


def mask_from_probs(pass_probs, estimated_fail_count, count_rounding="round") -> str:
    L = len(pass_probs)
    if count_rounding == "floor":
        fail_count = int(np.floor(estimated_fail_count))
    elif count_rounding == "ceil":
        fail_count = int(np.ceil(estimated_fail_count))
    else:
        fail_count = int(np.round(estimated_fail_count))

    fail_count = max(0, min(L, fail_count))
    if fail_count == 0:
        return "1" * L
    if fail_count == L:
        return "0" * L

    fail_positions = set(np.argsort(pass_probs)[:fail_count])
    return "".join("0" if i in fail_positions else "1" for i in range(L))


def predict_masks_from_similarity(
    sim_matrix,
    train_masks,
    train_fail_counts,
    train_lengths,
    target_lengths,
    k=8,
    count_mode="weighted",
    exclude_self=False,
    return_probs=False,
):
    """Predict masks for rows represented by sim_matrix rows vs train rows."""
    n_targets = sim_matrix.shape[0]
    predictions = []
    all_probs = []
    all_counts = []

    train_lengths = np.asarray(train_lengths)

    # Precompute same-length train indices.
    by_length = {}
    for L in sorted(set(train_lengths.tolist() + list(map(int, target_lengths)))):
        by_length[int(L)] = np.where(train_lengths == int(L))[0]

    for i in range(n_targets):
        L = int(target_lengths[i])
        candidates = by_length.get(L, np.array([], dtype=int)).copy()

        if exclude_self and i < len(train_masks):
            candidates = candidates[candidates != i]

        if len(candidates) == 0:
            # Fallback: all pass is often a safer default than random failures.
            pass_probs = np.ones(L, dtype=float)
            estimated_fail = 0.0
        else:
            top_idx, weights = weighted_top_indices(sim_matrix[i], candidates, k)
            pass_probs, estimated_fail = probs_and_count_from_neighbors(
                train_masks, train_fail_counts, top_idx, weights, L, count_mode
            )

        pred = mask_from_probs(pass_probs, estimated_fail)
        predictions.append(pred)
        all_probs.append(pass_probs)
        all_counts.append(estimated_fail)

    if return_probs:
        return predictions, all_probs, np.asarray(all_counts, dtype=float)
    return predictions


# -----------------------------
# Optional descriptor-level model
# -----------------------------

def blueprint_items(value, L):
    parsed = safe_json_loads(value, default=[])
    if not isinstance(parsed, list):
        parsed = [parsed]
    items = []
    for j in range(L):
        if j < len(parsed):
            item = parsed[j]
        else:
            item = {"missing_blueprint_item_position": j}
        try:
            items.append(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        except Exception:
            items.append(str(item))
    return items


def make_descriptor_text(row, position, item_text) -> str:
    # Keep this compact: row context + individual hidden-test descriptor + position.
    base = row_text(row, mode="problem_code")
    return "\n".join([
        base,
        "HIDDEN_TEST_DESCRIPTOR:",
        item_text,
        f"POSITION={position}",
        f"MASK_LENGTH={row.get('mask_length', '')}",
    ])


def descriptor_model_probs(train, test):
    """
    Train a lightweight per-hidden-test pass/fail classifier.
    Returns list[np.array] of pass probabilities per test row, or None if unavailable.
    """
    try:
        from sklearn.linear_model import SGDClassifier
    except Exception:
        return None

    x_train = []
    y_train = []

    for _, row in train.iterrows():
        L = int(row["mask_length"])
        mask = str(row["pass_mask"])
        items = blueprint_items(row.get("test_blueprint_json", ""), L)
        for j, item in enumerate(items):
            x_train.append(make_descriptor_text(row, j, item))
            y_train.append(1 if j < len(mask) and mask[j] == "1" else 0)

    if len(set(y_train)) < 2 or len(x_train) < 20:
        return None

    x_test = []
    row_slices = []
    start = 0
    for _, row in test.iterrows():
        L = int(row["mask_length"])
        items = blueprint_items(row.get("test_blueprint_json", ""), L)
        for j, item in enumerate(items):
            x_test.append(make_descriptor_text(row, j, item))
        row_slices.append((start, start + L))
        start += L

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=180_000,
            lowercase=False,
        )
        Xtr = vectorizer.fit_transform(x_train)
        Xte = vectorizer.transform(x_test)

        clf = SGDClassifier(
            loss="log_loss",
            alpha=2e-5,
            max_iter=60,
            tol=1e-4,
            random_state=RANDOM_SEED,
        )
        clf.fit(Xtr, np.asarray(y_train, dtype=int))
        probs_flat = clf.predict_proba(Xte)[:, 1]

        out = []
        for a, b in row_slices:
            out.append(np.asarray(probs_flat[a:b], dtype=float))
        return out
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------

def main():
    data_dir = find_dataset_dir()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    print(f"Using data directory: {data_dir}")

    train = pd.read_csv(train_path).reset_index(drop=True)
    test = pd.read_csv(test_path).reset_index(drop=True)

    train_answers = train["answer_json"].apply(parse_answer_json)
    train["pass_mask"] = train_answers.apply(lambda obj: str(obj["pass_mask"]))
    train["fail_count"] = train["pass_mask"].str.count("0").astype(int)
    train["mask_length"] = train["mask_length"].astype(int)
    test["mask_length"] = test["mask_length"].astype(int)

    train_masks = train["pass_mask"].tolist()
    train_fail_counts = train["fail_count"].to_numpy(dtype=float)
    train_lengths = train["mask_length"].to_numpy(dtype=int)
    test_lengths = test["mask_length"].to_numpy(dtype=int)

    # Candidate vectorization setups. Local leave-one-out validation picks the best ones.
    vectorizer_setups = [
        {
            "name": "full_char_wb_3_6",
            "mode": "full",
            "params": dict(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_features=250_000, lowercase=False),
        },
        {
            "name": "code_blueprint_char_wb_3_6",
            "mode": "code_blueprint",
            "params": dict(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_features=220_000, lowercase=False),
        },
        {
            "name": "problem_code_char_4_7",
            "mode": "problem_code",
            "params": dict(analyzer="char", ngram_range=(4, 7), min_df=2, max_features=220_000, lowercase=False),
        },
        {
            "name": "full_word_1_2",
            "mode": "full",
            "params": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120_000, token_pattern=r"(?u)\b\w+\b|[^\s\w]"),
        },
    ]

    k_values = [1, 3, 5, 8, 12, 20]
    count_modes = ["weighted", "nearest", "median", "threshold"]

    validated_configs = []
    cached_test_outputs = []

    for setup in vectorizer_setups:
        print(f"Fitting vectorizer: {setup['name']}")
        train_texts = train.apply(lambda r: row_text(r, setup["mode"]), axis=1).tolist()
        test_texts = test.apply(lambda r: row_text(r, setup["mode"]), axis=1).tolist()
        all_texts = train_texts + test_texts

        try:
            vectorizer = TfidfVectorizer(**setup["params"])
            X = vectorizer.fit_transform(all_texts)
            X_train = X[:len(train)]
            X_test = X[len(train):]

            sim_train = cosine_similarity(X_train, X_train)
            sim_test = cosine_similarity(X_test, X_train)
        except Exception as exc:
            print(f"  Skipping {setup['name']} because vectorization failed: {exc}")
            continue

        for k in k_values:
            for count_mode in count_modes:
                loo_preds = predict_masks_from_similarity(
                    sim_train,
                    train_masks,
                    train_fail_counts,
                    train_lengths,
                    train_lengths,
                    k=k,
                    count_mode=count_mode,
                    exclude_self=True,
                )
                cv_score = mean_score(train_masks, loo_preds)

                test_preds, test_probs, test_counts = predict_masks_from_similarity(
                    sim_test,
                    train_masks,
                    train_fail_counts,
                    train_lengths,
                    test_lengths,
                    k=k,
                    count_mode=count_mode,
                    exclude_self=False,
                    return_probs=True,
                )

                config_record = {
                    "score": cv_score,
                    "setup": setup["name"],
                    "k": k,
                    "count_mode": count_mode,
                    "test_preds": test_preds,
                    "test_probs": test_probs,
                    "test_counts": test_counts,
                }
                validated_configs.append(config_record)

        best_for_setup = max(
            (c for c in validated_configs if c["setup"] == setup["name"]),
            key=lambda c: c["score"],
        )
        print(
            f"  Best {setup['name']}: score={best_for_setup['score']:.5f}, "
            f"k={best_for_setup['k']}, count={best_for_setup['count_mode']}"
        )

    if not validated_configs:
        print("No vectorizer configs worked. Falling back to all-pass masks.")
        final_masks = ["1" * int(L) for L in test_lengths]
    else:
        validated_configs.sort(key=lambda c: c["score"], reverse=True)
        print("Top local-validation configs:")
        for c in validated_configs[:8]:
            print(f"  {c['score']:.5f} | {c['setup']} | k={c['k']} | count={c['count_mode']}")

        # Ensemble the best few configs. Weight by how much they beat a weak floor.
        top_configs = validated_configs[:7]
        scores = np.array([c["score"] for c in top_configs], dtype=float)
        weights = np.maximum(scores - scores.min() + 1e-4, 1e-4)
        weights = weights / weights.sum()

        desc_probs = descriptor_model_probs(train, test)
        use_desc = desc_probs is not None
        if use_desc:
            print("Descriptor-level classifier enabled.")
        else:
            print("Descriptor-level classifier unavailable; using KNN ensemble only.")

        final_masks = []
        for i, L in enumerate(test_lengths):
            L = int(L)
            combined_probs = np.zeros(L, dtype=float)
            combined_count = 0.0

            for w, c in zip(weights, top_configs):
                probs = c["test_probs"][i]
                if len(probs) != L:
                    probs = np.resize(probs, L)
                combined_probs += w * probs
                combined_count += w * float(c["test_counts"][i])

            if use_desc and len(desc_probs[i]) == L:
                # Conservative blend: KNN is usually better calibrated for row-level fail count,
                # descriptor classifier helps with position-specific failures.
                combined_probs = 0.82 * combined_probs + 0.18 * desc_probs[i]
                desc_fail_est = float(np.sum(1.0 - desc_probs[i]))
                combined_count = 0.88 * combined_count + 0.12 * desc_fail_est

            final_masks.append(mask_from_probs(combined_probs, combined_count))

    submission = pd.DataFrame({
        "id": test["id"].astype(str),
        "answer_json": [build_answer_json(mask) for mask in final_masks],
    })

    out_dir = Path("./working")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    submission.to_csv(out_path, index=False)

    # Also write a convenience copy in the current directory for notebook users.
    try:
        submission.to_csv("submission.csv", index=False)
    except Exception:
        pass

    print(f"Wrote {out_path.resolve()}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
