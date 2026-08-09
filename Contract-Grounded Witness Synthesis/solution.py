#!/usr/bin/env python3
"""Contract-grounded candidate ranking and executable witness synthesis."""

import base64
import difflib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import time
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

SEED = 2026
HASH_DIM = 1 << 20
WALL_GUARD_SECONDS = 3000
MAX_SANDBOX_WORKERS = 10

# Each child receives one released implementation and a released assertion list. It
# executes each assertion in a fresh namespace. CPU, address-space, file-size, and
# descriptor limits supplement the parent's wall timeout.
SANDBOX_WORKER = r'''
import base64
import json
import resource
import sys

try:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 * 1024, 1536 * 1024 * 1024))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
except Exception:
    pass

payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
source = payload["source"]
checks = payload["checks"]
results = []
for check in checks:
    namespace = {}
    try:
        exec(compile(source, "<released_candidate>", "exec"), namespace, namespace)
        exec(compile(check, "<released_witness>", "exec"), namespace, namespace)
        results.append(1)
    except BaseException:
        results.append(0)
print(json.dumps(results, separators=(",", ":")))
'''


def warn(message):
    print("WARNING:", message, file=sys.stderr, flush=True)


def parse_json_array(value, field, episode_id):
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError(f"invalid {field} JSON for {episode_id}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} is not an array for {episode_id}")
    return parsed


def valid_placeholder(bank):
    if not bank:
        raise ValueError("required witness bank is empty")
    return json.dumps([str(bank[0])], ensure_ascii=False)


def write_frame(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def sandbox_checks(source, checks):
    encoded = base64.b64encode(
        json.dumps({"source": source, "checks": checks}, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", SANDBOX_WORKER, encoded],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
        if proc.returncode == 0:
            result = json.loads(proc.stdout.strip())
            if isinstance(result, list) and len(result) == len(checks):
                return [bool(x) for x in result]
    except Exception:
        pass
    return [False] * len(checks)


def parallel_sandbox(jobs):
    if not jobs:
        return []
    workers = min(MAX_SANDBOX_WORKERS, max(1, os.cpu_count() or 1), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: sandbox_checks(job[0], job[1]), jobs))


def clean_code(source):
    return "\n".join(
        line for line in source.splitlines() if not line.startswith("__episode_marker__")
    )


def python_tokens(source):
    values = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(clean_code(source)).readline)
        ignored = {
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        }
        for item in stream:
            if item.type not in ignored:
                values.append(item.string)
    except Exception:
        values = re.findall(r"\w+|[^\w\s]", clean_code(source))
    return values


def normalize_token(value):
    if re.fullmatch(r"local_\d+", value):
        return "VAR"
    if value == "candidate_fn":
        return "FUNC"
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return "NUM:" + value
    if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        return "STR"
    return value.lower()


def ngrams(tokens, low=1, high=2):
    return [
        "~".join(tokens[start : start + width])
        for width in range(low, high + 1)
        for start in range(len(tokens) - width + 1)
    ]


def contract_units(contract):
    tokens = re.findall(r"[a-z]+|\d+|[<>!=]+", contract.lower())
    structural = {"acceptance", "card", "objective", "cues", "data", "and", "constraint"}
    tokens = [token for token in tokens if token not in structural]
    return ngrams(tokens, 1, 2)


def candidate_code_atoms(candidates, candidate_index):
    token_lists = [
        [normalize_token(token) for token in python_tokens(source)] for source in candidates
    ]
    chosen = token_lists[candidate_index]
    atoms = ["code:" + value for value in ngrams(chosen, 1, 3)]

    if len({len(tokens) for tokens in token_lists}) == 1:
        for position, variants in enumerate(zip(*token_lists)):
            if len(set(variants)) == 1:
                continue
            before = "^" if position == 0 else chosen[position - 1]
            after = "$" if position + 1 == len(chosen) else chosen[position + 1]
            alternatives = ",".join(sorted(set(variants) - {chosen[position]}))
            atoms.extend(
                [
                    f"diff:{before}>{chosen[position]}<{after}",
                    f"choice:{chosen[position]}|alts:{alternatives}",
                    f"ctxchoice:{before}|{chosen[position]}|{after}|alts:{alternatives}",
                ]
            )
    else:
        for other_index, other in enumerate(token_lists):
            if other_index == candidate_index:
                continue
            matcher = difflib.SequenceMatcher(a=chosen, b=other, autojunk=False)
            for tag, a0, a1, b0, b1 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                before = "^" if a0 == 0 else chosen[a0 - 1]
                after = "$" if a1 == len(chosen) else chosen[a1]
                selected = "~".join(chosen[a0:a1]) or "<EMPTY>"
                alternative = "~".join(other[b0:b1]) or "<EMPTY>"
                atoms.extend(
                    [
                        f"diff:{before}>{selected}<{after}",
                        f"choice:{selected}|alt:{alternative}",
                        f"ctxchoice:{before}|{selected}|{after}|alt:{alternative}",
                    ]
                )
    return atoms


def pair_feature(contract, candidates, candidate_index):
    cues = contract_units(contract)
    atoms = candidate_code_atoms(candidates, candidate_index)
    features = {}
    for cue in cues:
        features["contract:" + cue] = 1.0
    for atom in atoms:
        features["code_atom:" + atom] = 1.0
    # Learned conjunction weights connect semantic card cues to implementation choices.
    for cue in cues:
        for atom in atoms:
            features["interaction:" + cue + "|" + atom] = 1.0
    return features


def derive_training_examples(train_frame):
    rows = []
    jobs = []
    for row in train_frame.itertuples(index=False):
        episode_id = str(row.episode_id)
        candidates = parse_json_array(
            row.candidate_implementations_json, "candidate_implementations_json", episode_id
        )
        gold = parse_json_array(row.gold_witness_suite, "gold_witness_suite", episode_id)
        if len(candidates) != 4:
            warn(f"skipping {episode_id}: expected four candidates, found {len(candidates)}")
            continue
        rows.append((episode_id, str(row.contract), candidates))
        jobs.extend((source, gold) for source in candidates)

    outcomes = parallel_sandbox(jobs)
    examples = []
    for row_index, (episode_id, contract, candidates) in enumerate(rows):
        candidate_results = outcomes[4 * row_index : 4 * row_index + 4]
        survivors = [index for index, result in enumerate(candidate_results) if all(result)]
        if len(survivors) != 1:
            warn(f"skipping {episode_id}: released gold suite did not isolate one executable candidate")
            continue
        examples.append(
            {
                "episode_id": episode_id,
                "contract": contract,
                "candidates": candidates,
                "label": survivors[0],
            }
        )
    return examples


def vectorize_examples(examples, hasher):
    dictionaries = []
    for example in examples:
        for candidate_index in range(4):
            dictionaries.append(
                pair_feature(example["contract"], example["candidates"], candidate_index)
            )
    return hasher.transform(dictionaries).tocsr()


def pairwise_training_matrix(candidate_matrix, labels, episode_indices):
    rows = []
    targets = []
    for episode_index in episode_indices:
        correct_row = 4 * episode_index + int(labels[episode_index])
        for candidate_index in range(4):
            if candidate_index == labels[episode_index]:
                continue
            wrong_row = 4 * episode_index + candidate_index
            difference = candidate_matrix[correct_row] - candidate_matrix[wrong_row]
            rows.extend((difference, -difference))
            targets.extend((1, 0))
    return vstack(rows).tocsr(), np.asarray(targets, dtype=np.int8)


def challenge_validation_score(labels, predictions):
    accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    identity_skill = max(0.0, (accuracy - 0.25) / 0.75)
    # Executable minimum suites have quality=1 and exactness=1 whenever identity is right.
    return 0.60 * identity_skill + 0.40 * accuracy, accuracy


def train_ranker(examples, candidate_matrix, started_at):
    labels = np.asarray([example["label"] for example in examples], dtype=np.int64)
    count = len(examples)
    if count < 8:
        raise RuntimeError("too few executable labeled episodes to train a ranker")

    groups = np.asarray([example["contract"] for example in examples], dtype=object)
    if len(set(groups)) < 2:
        groups = np.arange(count)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=SEED)
    train_indices, validation_indices = next(splitter.split(np.arange(count), groups=groups))
    train_pairs, train_pair_labels = pairwise_training_matrix(
        candidate_matrix, labels, train_indices
    )

    candidate_cs = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
    best = None
    best_model = None
    for regularization in candidate_cs:
        if time.monotonic() - started_at >= WALL_GUARD_SECONDS:
            warn("wall-clock guard reached during hyperparameter search")
            break
        model = LinearSVC(
            C=regularization,
            dual="auto",
            max_iter=20000,
            random_state=SEED,
        )
        model.fit(train_pairs, train_pair_labels)
        validation_rows = np.concatenate(
            [np.arange(4 * index, 4 * index + 4) for index in validation_indices]
        )
        scores = (candidate_matrix[validation_rows] @ model.coef_.ravel()).reshape(-1, 4)
        predictions = scores.argmax(axis=1)
        metric, accuracy = challenge_validation_score(labels[validation_indices], predictions)
        result = (metric, accuracy, -regularization, regularization)
        if best is None or result[:3] > best[:3]:
            best = result
            best_model = model

    if best is None or best_model is None:
        raise RuntimeError("wall-clock guard prevented ranker selection")
    selected_c = best[3]
    print(
        f"Validation: contract-group holdout n={len(validation_indices)}, "
        f"canonical_accuracy={best[1]:.6f}, challenge_metric={best[0]:.6f}, C={selected_c:g}",
        flush=True,
    )

    if time.monotonic() - started_at >= WALL_GUARD_SECONDS:
        warn("wall-clock guard reached; using the trained holdout model for inference")
        return best_model

    all_pairs, all_pair_labels = pairwise_training_matrix(
        candidate_matrix, labels, np.arange(count)
    )
    final_model = LinearSVC(
        C=selected_c,
        dual="auto",
        max_iter=20000,
        random_state=SEED,
    )
    final_model.fit(all_pairs, all_pair_labels)
    return final_model


def minimum_suite(bank, behavior_matrix, target):
    for size in (1, 2, 3):
        legal = []
        for positions in itertools.combinations(range(len(bank)), size):
            if not all(behavior_matrix[target][position] for position in positions):
                continue
            survivors = [
                candidate
                for candidate in range(4)
                if all(behavior_matrix[candidate][position] for position in positions)
            ]
            if survivors == [target]:
                legal.append(tuple(sorted(bank[position] for position in positions)))
        if legal:
            return list(min(legal))
    return None


def predict_rows(test_frame, model, hasher):
    parsed = []
    feature_dicts = []
    for row in test_frame.itertuples(index=False):
        episode_id = str(row.episode_id)
        bank = parse_json_array(row.witness_bank_json, "witness_bank_json", episode_id)
        try:
            candidates = parse_json_array(
                row.candidate_implementations_json,
                "candidate_implementations_json",
                episode_id,
            )
            if len(candidates) != 4:
                raise ValueError(f"expected four candidates, found {len(candidates)}")
            row_features = [
                pair_feature(str(row.contract), candidates, candidate_index)
                for candidate_index in range(4)
            ]
        except Exception as exc:
            warn(f"{episode_id}: candidate transform failed ({exc}); using fallback")
            parsed.append((episode_id, [], bank, False))
            feature_dicts.extend({} for _ in range(4))
            continue
        parsed.append((episode_id, candidates, bank, True))
        feature_dicts.extend(row_features)

    matrix = hasher.transform(feature_dicts).tocsr()
    flat_scores = np.asarray(matrix @ model.coef_.ravel()).reshape(-1)
    scores = flat_scores.reshape(len(parsed), 4)

    jobs = []
    for _, candidates, bank, valid in parsed:
        if valid:
            jobs.extend((source, bank) for source in candidates)
        else:
            jobs.extend(("", bank) for _ in range(4))
    outcomes = parallel_sandbox(jobs)

    predictions = []
    for row_index, (episode_id, candidates, bank, valid) in enumerate(parsed):
        fallback = valid_placeholder(bank)
        if not valid or not bank:
            predictions.append(fallback)
            continue
        behavior = outcomes[4 * row_index : 4 * row_index + 4]
        chosen_suite = None
        # The trained ranker determines the candidate order; the decoder only enforces
        # the released executable-certificate constraint.
        for target in np.argsort(-scores[row_index], kind="stable"):
            suite = minimum_suite(bank, behavior, int(target))
            if suite is not None:
                chosen_suite = suite
                break
        if chosen_suite is None:
            warn(f"{episode_id}: no 1-to-3 assertion suite survived sandbox evaluation")
            predictions.append(fallback)
        else:
            predictions.append(json.dumps(chosen_suite, ensure_ascii=False))
    return predictions


def validate_final(test_frame, predictions):
    expected_ids = test_frame["episode_id"].astype(str).tolist()
    if len(predictions) != len(expected_ids):
        return False, "prediction row count differs from test"
    if len(set(expected_ids)) != len(expected_ids):
        return False, "test contains duplicate episode IDs"
    for row, prediction in zip(test_frame.itertuples(index=False), predictions):
        bank = parse_json_array(
            row.witness_bank_json, "witness_bank_json", str(row.episode_id)
        )
        try:
            suite = json.loads(prediction)
        except Exception:
            return False, f"invalid prediction JSON for {row.episode_id}"
        if not isinstance(suite, list) or not 1 <= len(suite) <= 3:
            return False, f"invalid suite size for {row.episode_id}"
        if len(set(suite)) != len(suite) or suite != sorted(suite):
            return False, f"suite is duplicate or unsorted for {row.episode_id}"
        if any(item not in bank for item in suite):
            return False, f"assertion absent from bank for {row.episode_id}"
    return True, "ok"


def main():
    started_at = time.monotonic()
    np.random.seed(SEED)
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("required train.csv or test.csv is missing")

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    required_train = {
        "episode_id",
        "contract",
        "candidate_implementations_json",
        "witness_bank_json",
        "gold_witness_suite",
    }
    required_test = {
        "episode_id",
        "contract",
        "candidate_implementations_json",
        "witness_bank_json",
    }
    if not required_train.issubset(train_frame.columns) or not required_test.issubset(
        test_frame.columns
    ):
        raise ValueError("required CSV columns are missing")

    # The placeholder is emitted before sandboxing, feature construction, or training.
    placeholder_predictions = []
    for row in test_frame.itertuples(index=False):
        bank = parse_json_array(row.witness_bank_json, "witness_bank_json", str(row.episode_id))
        placeholder_predictions.append(valid_placeholder(bank))
    placeholder = pd.DataFrame(
        {
            "episode_id": test_frame["episode_id"].astype(str),
            "predicted_witness_suite": placeholder_predictions,
        }
    )
    write_frame(placeholder, submission_out)

    examples = derive_training_examples(train_frame)
    if len(examples) != len(train_frame):
        warn(f"training with {len(examples)} of {len(train_frame)} rows")
    hasher = FeatureHasher(
        n_features=HASH_DIM,
        input_type="dict",
        alternate_sign=False,
    )
    candidate_matrix = vectorize_examples(examples, hasher)
    model = train_ranker(examples, candidate_matrix, started_at)

    predictions = predict_rows(test_frame, model, hasher)
    valid, reason = validate_final(test_frame, predictions)
    if not valid:
        warn("real prediction validation failed; retaining early placeholder: " + reason)
        return
    final_frame = pd.DataFrame(
        {
            "episode_id": test_frame["episode_id"].astype(str),
            "predicted_witness_suite": predictions,
        }
    )
    if list(final_frame.columns) != ["episode_id", "predicted_witness_suite"]:
        warn("final column validation failed; retaining early placeholder")
        return
    write_frame(final_frame, submission_out)
    print(
        f"Wrote {len(final_frame)} complete rows to {submission_out} "
        f"in {time.monotonic() - started_at:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
