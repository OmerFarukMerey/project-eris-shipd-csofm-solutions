#!/usr/bin/env python3
"""End-to-end enzyme/reaction relevance ranker for Project Eris."""

from __future__ import annotations

import gc
import hashlib
import math
import re
import sys
import time
import warnings
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility with older Kaggle images
    StratifiedGroupKFold = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - the classical stack remains a valid fallback
    CatBoostClassifier = None


SEED = 1729
TIME_GUARD_SECONDS = 3000.0
EPS = 1.0e-7
SVC_C_GRID = (0.1, 0.3, 1.0)
LEVELS = (2, 3)
VIEWS = ("char", "token")
DECISION_FEATURES = ("given", "gap", "rank", "probability", "known", "top", "spread")
SMILES_PATTERN = re.compile(
    r"\[[^\]]+\]|Br|Cl|Si|Na|Li|Mg|Ca|Al|[A-Z][a-z]?|[bcnops]|%\d\d|\d|."
)

BASE_FEATURE_NAMES = tuple(
    f"{view}_L{level}_C{c}_{feature}"
    for view in VIEWS
    for level in LEVELS
    for c in SVC_C_GRID
    for feature in DECISION_FEATURES
)
BASE_FEATURE_INDEX = {name: i for i, name in enumerate(BASE_FEATURE_NAMES)}

META_CONFIGS = (
    {"max_leaf_nodes": 7, "l2_regularization": 3.0, "min_samples_leaf": 40, "learning_rate": 0.05, "max_iter": 200},
    {"max_leaf_nodes": 15, "l2_regularization": 5.0, "min_samples_leaf": 40, "learning_rate": 0.05, "max_iter": 200},
    {"max_leaf_nodes": 31, "l2_regularization": 8.0, "min_samples_leaf": 50, "learning_rate": 0.05, "max_iter": 200},
    {"max_leaf_nodes": 15, "l2_regularization": 10.0, "min_samples_leaf": 80, "learning_rate": 0.05, "max_iter": 200},
)
CALIBRATION_SLOPES = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
CALIBRATION_BIASES = (-0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
DIRECT_ALPHA_GRID = (3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4)
NB_ALPHA_GRID = (0.1, 1.0, 5.0)
PROTOTYPE_FEATURES_PER_LEVEL = 12
ADVANCED_FEATURE_WIDTH = (
    len(LEVELS) * len(SVC_C_GRID) * len(DECISION_FEATURES)
    + 3 * 3 * PROTOTYPE_FEATURES_PER_LEVEL
    + len(DIRECT_ALPHA_GRID)
    + 3 * len(NB_ALPHA_GRID) * 2 * len(DECISION_FEATURES)
)
CAT_META_CONFIGS = (
    {"depth": 5, "iterations": 1200, "learning_rate": 0.020, "l2_leaf_reg": 5.0, "random_strength": 0.1, "hard_weight": 1.0, "feature_count": 0},
    {"depth": 5, "iterations": 1200, "learning_rate": 0.020, "l2_leaf_reg": 5.0, "random_strength": 0.1, "hard_weight": 1.5, "feature_count": 0},
    {"depth": 5, "iterations": 1200, "learning_rate": 0.020, "l2_leaf_reg": 5.0, "random_strength": 0.1, "hard_weight": 2.0, "feature_count": 0},
    {"depth": 4, "iterations": 1400, "learning_rate": 0.018, "l2_leaf_reg": 3.0, "random_strength": 0.1, "hard_weight": 1.5, "feature_count": 120},
    {"depth": 5, "iterations": 1400, "learning_rate": 0.018, "l2_leaf_reg": 5.0, "random_strength": 0.1, "hard_weight": 1.5, "feature_count": 120},
    {"depth": 6, "iterations": 1200, "learning_rate": 0.020, "l2_leaf_reg": 8.0, "random_strength": 0.1, "hard_weight": 1.5, "feature_count": 120},
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def log(message: str) -> None:
    print(message, flush=True)


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def safe_token(value: str) -> str:
    return re.sub(r"\s+", "_", value)


def smiles_tokens(text: str) -> list[str]:
    return SMILES_PATTERN.findall(safe_text(text))


def char_ngram_counts(text: str, minimum: int = 1, maximum: int = 4) -> Counter[str]:
    wrapped = "^" + safe_text(text) + "$"
    counts: Counter[str] = Counter()
    for width in range(minimum, maximum + 1):
        counts.update(wrapped[i : i + width] for i in range(len(wrapped) - width + 1))
    return counts


def multiset_dice(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    a, b = Counter(left), Counter(right)
    return 2.0 * sum((a & b).values()) / max(len(left) + len(right), 1)


def reaction_diff_signature(substrates: object, candidate: object) -> str:
    """Generic local edit tokens; no reaction or EC mapping is asserted here."""
    try:
        substrate_text = safe_text(substrates)
        candidate_text = safe_text(candidate)
        components = [part for part in substrate_text.split(".") if part] or [""]
        candidate_chars = Counter(candidate_text)
        best = max(
            components,
            key=lambda part: (
                2.0 * sum((Counter(part) & candidate_chars).values())
                / max(len(part) + len(candidate_text), 1),
                -abs(len(part) - len(candidate_text)),
            ),
        )

        before = char_ngram_counts(best)
        after = char_ngram_counts(candidate_text)
        tokens: list[str] = []
        for ngram, count in (after - before).items():
            tokens.extend(["gain:" + safe_token(ngram)] * min(count, 3))
        for ngram, count in (before - after).items():
            tokens.extend(["loss:" + safe_token(ngram)] * min(count, 3))

        matcher = SequenceMatcher(None, best, candidate_text, autojunk=False)
        for operation, i1, i2, j1, j2 in matcher.get_opcodes():
            if operation == "equal":
                continue
            old, new = best[i1:i2], candidate_text[j1:j2]
            if old:
                for ngram in char_ngram_counts(old, 1, 3):
                    tokens.append("edit_old_" + operation + ":" + safe_token(ngram))
            if new:
                for ngram in char_ngram_counts(new, 1, 3):
                    tokens.append("edit_new_" + operation + ":" + safe_token(ngram))
            left_context = best[max(0, i1 - 3) : i1]
            right_context = best[i2 : min(len(best), i2 + 3)]
            tokens.append("context:" + safe_token(left_context) + ">" + safe_token(right_context))
        return " ".join(tokens) if tokens else "unchanged"
    except Exception:
        return "unparsed"


def tokenize_for_vectorizer(text: str) -> list[str]:
    return smiles_tokens(text)


def fit_alphabet(train: pd.DataFrame) -> tuple[str, ...]:
    observed: set[str] = set()
    for column in ("substrates", "candidate"):
        for value in train[column]:
            observed.update(safe_text(value))
    return tuple(sorted(observed))


def numeric_row_features(substrates: object, candidate: object, alphabet: tuple[str, ...]) -> np.ndarray:
    width = 20 + 2 * len(alphabet)
    try:
        substrate_text = safe_text(substrates)
        candidate_text = safe_text(candidate)
        components = [part for part in substrate_text.split(".") if part] or [""]
        ratios = [SequenceMatcher(None, part, candidate_text, autojunk=False).ratio() for part in components]
        dices = [multiset_dice(part, candidate_text) for part in components]
        best_index = int(np.argmax(ratios))
        best = components[best_index]
        second_ratio = sorted(ratios, reverse=True)[1] if len(ratios) > 1 else 0.0

        cand_counts = Counter(candidate_text)
        best_counts = Counter(best)
        all_counts = Counter(substrate_text.replace(".", ""))
        intersection = sum((cand_counts & best_counts).values())
        union = sum((cand_counts | best_counts).values())

        candidate_token_counts = Counter(smiles_tokens(candidate_text))
        token_dices = []
        for component in components:
            component_counts = Counter(smiles_tokens(component))
            token_dices.append(
                2.0
                * sum((candidate_token_counts & component_counts).values())
                / max(sum(candidate_token_counts.values()) + sum(component_counts.values()), 1)
            )

        core = [
            len(substrate_text) / 240.0,
            len(candidate_text) / 100.0,
            len(best) / 100.0,
            len(components) / 5.0,
            max(dices),
            max(ratios),
            second_ratio,
            intersection / max(len(candidate_text), 1),
            intersection / max(len(best), 1),
            intersection / max(union, 1),
            max(token_dices),
            len(candidate_text) / max(len(best), 1),
            (len(candidate_text) - len(best)) / 100.0,
            float(candidate_text.count("=") - best.count("=")),
            float(candidate_text.count("#") - best.count("#")),
            float(candidate_text.count("(") - best.count("(")),
            float(candidate_text.count("[") - best.count("[")),
            sum(ch.islower() for ch in candidate_text) / max(len(candidate_text), 1),
            float(candidate_text.count("+") - candidate_text.count("-")),
            float(best.count("+") - best.count("-")),
        ]
        deltas_best = [(cand_counts[ch] - best_counts[ch]) / 10.0 for ch in alphabet]
        deltas_all = [(cand_counts[ch] - all_counts[ch]) / 10.0 for ch in alphabet]
        values = np.asarray(core + deltas_best + deltas_all, dtype=np.float32)
        if values.shape[0] != width or not np.isfinite(values).all():
            return np.zeros(width, dtype=np.float32)
        return values
    except Exception:
        return np.zeros(width, dtype=np.float32)


def numeric_features(frame: pd.DataFrame, alphabet: tuple[str, ...]) -> np.ndarray:
    rows = [
        numeric_row_features(substrates, candidate, alphabet)
        for substrates, candidate in frame[["substrates", "candidate"]].itertuples(index=False, name=None)
    ]
    return np.vstack(rows).astype(np.float32, copy=False)

ATOM_TOKEN_PATTERN = re.compile(
    r"^(?:\[[^\]]+\]|Br|Cl|Si|Na|Li|Mg|Ca|Al|[A-Z][a-z]?|[bcnops]|\*)$"
)
BOND_LABELS = {"-": "1", "=": "2", "#": "3", ":": "a", "/": "1", "\\": "1"}


def atom_element(token: str) -> str:
    raw = token[1:-1] if token.startswith("[") else token
    match = re.search(r"([A-Z][a-z]?|[bcnops]|\*)", raw)
    return match.group(1) if match else "*"


def parse_smiles_graph(smiles: object) -> tuple[list[tuple[str, str]], list[tuple[int, int, str]]]:
    atoms: list[tuple[str, str]] = []
    edges: list[tuple[int, int, str]] = []
    current: int | None = None
    branches: list[int | None] = []
    rings: dict[str, tuple[int | None, str]] = {}
    pending_bond = "1"

    for token in smiles_tokens(safe_text(smiles)):
        if ATOM_TOKEN_PATTERN.fullmatch(token):
            atom_index = len(atoms)
            atoms.append((atom_element(token), token))
            if current is not None:
                edges.append((current, atom_index, pending_bond))
            current = atom_index
            pending_bond = "1"
        elif token in BOND_LABELS:
            pending_bond = BOND_LABELS[token]
        elif token == "(":
            branches.append(current)
        elif token == ")":
            if branches:
                current = branches.pop()
        elif token.isdigit() or (token.startswith("%") and token[1:].isdigit()):
            if token in rings:
                other, stored_bond = rings.pop(token)
                if current is not None and other is not None:
                    bond = pending_bond if pending_bond != "1" else stored_bond
                    edges.append((other, current, bond))
            else:
                rings[token] = (current, pending_bond)
            pending_bond = "1"
        elif token == ".":
            current = None
            pending_bond = "1"
    return atoms, edges


def stable_feature_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def wl_fingerprint_counts(smiles: object, max_radius: int = 3) -> Counter[str]:
    atoms, edges = parse_smiles_graph(smiles)
    neighbors: list[list[tuple[int, str]]] = [[] for _ in atoms]
    for left, right, bond in edges:
        neighbors[left].append((right, bond))
        neighbors[right].append((left, bond))

    labels = [stable_feature_hash(f"{element}|{token}") for element, token in atoms]
    counts: Counter[str] = Counter()
    for label in labels:
        counts[f"r0:{label}"] += 1
    for radius in range(1, max_radius + 1):
        next_labels = []
        for atom_index, label in enumerate(labels):
            environment = "|".join(
                sorted(f"{bond}:{labels[neighbor]}" for neighbor, bond in neighbors[atom_index])
            )
            next_labels.append(stable_feature_hash(label + "|" + environment))
        labels = next_labels
        for label in labels:
            counts[f"r{radius}:{label}"] += 1
    return counts


def counter_tanimoto(left: Counter[str], right: Counter[str]) -> float:
    intersection = sum((left & right).values())
    union = sum(left.values()) + sum(right.values()) - intersection
    return intersection / max(union, 1)


def graph_reaction_signature(substrates: object, candidate: object) -> str:
    try:
        components = [part for part in safe_text(substrates).split(".") if part] or [""]
        candidate_fingerprint = wl_fingerprint_counts(candidate)
        substrate_fingerprints = [wl_fingerprint_counts(component) for component in components]
        closest = max(
            substrate_fingerprints,
            key=lambda fingerprint: counter_tanimoto(fingerprint, candidate_fingerprint),
        )
        total: Counter[str] = Counter()
        for fingerprint in substrate_fingerprints:
            total.update(fingerprint)

        tokens: list[str] = []
        for prefix, fingerprint in (("candidate", candidate_fingerprint), ("substrate", total)):
            for feature, count in fingerprint.items():
                tokens.extend([f"{prefix}:{feature}"] * min(count, 3))
        for prefix, difference in (
            ("gain", candidate_fingerprint - closest),
            ("loss", closest - candidate_fingerprint),
            ("gain_all", candidate_fingerprint - total),
            ("loss_all", total - candidate_fingerprint),
        ):
            for feature, count in difference.items():
                tokens.extend([f"{prefix}:{feature}"] * min(count, 3))
        return " ".join(tokens) if tokens else "empty"
    except Exception:
        return "unparsed"


def ec_prefix(values: pd.Series, level: int) -> np.ndarray:
    return (
        values.fillna("")
        .astype(str)
        .str.split(".")
        .str[:level]
        .str.join(".")
        .to_numpy(dtype=str)
    )


def make_folds(train: pd.DataFrame) -> np.ndarray:
    groups = train["candidate"].fillna("").astype(str)
    n_groups = int(groups.nunique())
    n_splits = min(4, n_groups)
    if n_splits < 2:
        return np.zeros(len(train), dtype=np.int16)

    fold_ids = np.full(len(train), -1, dtype=np.int16)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iterator = splitter.split(train, train["label"], groups)
    else:  # pragma: no cover - compatibility fallback
        splitter = GroupKFold(n_splits=n_splits)
        split_iterator = splitter.split(train, train["label"], groups)
    for fold, (_, validation_indices) in enumerate(split_iterator):
        fold_ids[validation_indices] = fold
    return fold_ids


def infer_hard_negatives(train: pd.DataFrame) -> np.ndarray:
    positive_ecs: dict[tuple[str, str], set[str]] = {}
    for substrates, candidate, ec in train.loc[
        train["label"].eq(1), ["substrates", "candidate", "ec"]
    ].itertuples(index=False, name=None):
        positive_ecs.setdefault((substrates, candidate), set()).add(ec)

    hard = []
    for substrates, candidate, ec, label in train[
        ["substrates", "candidate", "ec", "label"]
    ].itertuples(index=False, name=None):
        matching_ecs = positive_ecs.get((substrates, candidate), set())
        hard.append(
            int(label) == 0
            and ec not in matching_ecs
            and any(ec.split(".", 1)[0] == positive_ec.split(".", 1)[0] for positive_ec in matching_ecs)
        )
    return np.asarray(hard, dtype=bool)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def challenge_metric(labels: np.ndarray, scores: np.ndarray, hard_negative: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.clip(np.asarray(scores, dtype=np.float64), 1.0e-9, 1.0 - 1.0e-9)
    rank = float(np.clip(2.0 * safe_auc(labels, probabilities) - 1.0, 0.0, 1.0))
    nll = float(
        -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))
    )
    calibration = float(np.clip(1.0 - nll / math.log(2.0), 0.0, 1.0))
    hard_scope = labels == 1
    hard_scope = np.asarray(hard_scope) | np.asarray(hard_negative, dtype=bool)
    hard_rank = float(
        np.clip(2.0 * safe_auc(labels[hard_scope], probabilities[hard_scope]) - 1.0, 0.0, 1.0)
    )
    quality = 0.45 * rank + 0.20 * calibration + 0.35 * hard_rank
    lower = math.tanh(300.0 * (0.0 - 0.995))
    upper = math.tanh(300.0 * (1.0 - 0.995))
    gate = (math.tanh(300.0 * (quality - 0.995)) - lower) / (upper - lower)
    final_score = 0.10 * quality + 0.25 * quality * quality + 0.65 * gate
    return {
        "auc": (rank + 1.0) / 2.0,
        "rank": rank,
        "nll": nll,
        "cal": calibration,
        "hard_auc": (hard_rank + 1.0) / 2.0,
        "hard_rank": hard_rank,
        "quality": quality,
        "score": final_score,
    }


def build_count_features(reference: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    features = np.zeros((len(query), 8), dtype=np.float32)
    for offset, level in enumerate((1, 2, 3, 4)):
        reference_prefix = pd.Series(ec_prefix(reference["ec"], level), index=reference.index)
        positive_counts = reference_prefix[reference["label"].eq(1)].value_counts()
        negative_counts = reference_prefix[reference["label"].eq(0)].value_counts()
        query_prefix = ec_prefix(query["ec"], level)
        features[:, 2 * offset] = np.log1p([positive_counts.get(value, 0) for value in query_prefix])
        features[:, 2 * offset + 1] = np.log1p(
            [negative_counts.get(value, 0) for value in query_prefix]
        )
    return features


def detailed_decision_features(
    classifier: LinearSVC,
    encoder: LabelEncoder,
    matrix: sparse.csr_matrix,
    given_classes: np.ndarray,
) -> np.ndarray:
    decisions = np.asarray(classifier.decision_function(matrix), dtype=np.float32)
    if decisions.ndim == 1:
        decisions = np.column_stack([-decisions, decisions]).astype(np.float32)
    class_index = {label: index for index, label in enumerate(encoder.classes_)}
    row_count = decisions.shape[0]
    given = np.empty(row_count, dtype=np.float32)
    known = np.zeros(row_count, dtype=np.float32)
    row_minimum = decisions.min(axis=1)
    for row, label in enumerate(given_classes):
        index = class_index.get(label)
        if index is None:
            given[row] = row_minimum[row]
        else:
            given[row] = decisions[row, index]
            known[row] = 1.0

    top = decisions.max(axis=1)
    gap = given - top
    rank = np.mean(decisions <= given[:, None], axis=1, dtype=np.float32)
    shifted = np.clip(decisions - top[:, None], -50.0, 0.0)
    probability = np.exp(np.clip(gap, -50.0, 0.0)) / np.exp(shifted).sum(axis=1)
    spread = decisions.std(axis=1)
    return np.column_stack((given, gap, rank, probability, known, top, spread)).astype(np.float32)


def make_char_matrices(
    train: pd.DataFrame,
    positive_fit_indices: np.ndarray,
    query_frames: dict[str, pd.DataFrame],
    diff_train: sparse.csr_matrix,
    diff_queries: dict[str, sparse.csr_matrix],
) -> tuple[sparse.csr_matrix, dict[str, sparse.csr_matrix]]:
    substrate_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    candidate_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = sparse.hstack(
        (
            substrate_vectorizer.fit_transform(train["substrates"].iloc[positive_fit_indices]),
            candidate_vectorizer.fit_transform(train["candidate"].iloc[positive_fit_indices]),
            diff_train,
        ),
        format="csr",
        dtype=np.float32,
    )
    query_matrices = {
        name: sparse.hstack(
            (
                substrate_vectorizer.transform(frame["substrates"]),
                candidate_vectorizer.transform(frame["candidate"]),
                diff_queries[name],
            ),
            format="csr",
            dtype=np.float32,
        )
        for name, frame in query_frames.items()
    }
    return train_matrix, query_matrices


def make_token_matrices(
    train: pd.DataFrame,
    positive_fit_indices: np.ndarray,
    query_frames: dict[str, pd.DataFrame],
    diff_train: sparse.csr_matrix,
    diff_queries: dict[str, sparse.csr_matrix],
) -> tuple[sparse.csr_matrix, dict[str, sparse.csr_matrix]]:
    substrate_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_vectorizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 5),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    candidate_vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_vectorizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 5),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = sparse.hstack(
        (
            substrate_vectorizer.fit_transform(train["substrates"].iloc[positive_fit_indices]),
            candidate_vectorizer.fit_transform(train["candidate"].iloc[positive_fit_indices]),
            diff_train,
        ),
        format="csr",
        dtype=np.float32,
    )
    query_matrices = {
        name: sparse.hstack(
            (
                substrate_vectorizer.transform(frame["substrates"]),
                candidate_vectorizer.transform(frame["candidate"]),
                diff_queries[name],
            ),
            format="csr",
            dtype=np.float32,
        )
        for name, frame in query_frames.items()
    }
    return train_matrix, query_matrices


def fit_base_models(
    train: pd.DataFrame,
    fit_indices: np.ndarray,
    query_frames: dict[str, pd.DataFrame],
    train_signatures: list[str],
    query_signatures: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    positive_fit_indices = fit_indices[train["label"].iloc[fit_indices].to_numpy() == 1]
    outputs = {
        name: np.zeros((len(frame), len(BASE_FEATURE_NAMES)), dtype=np.float32)
        for name, frame in query_frames.items()
    }
    if len(positive_fit_indices) == 0:
        return outputs

    diff_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    diff_train = diff_vectorizer.fit_transform([train_signatures[i] for i in positive_fit_indices])
    diff_queries = {
        name: diff_vectorizer.transform(query_signatures[name]) for name in query_frames
    }

    for view in VIEWS:
        if view == "char":
            model_matrix, query_matrices = make_char_matrices(
                train,
                positive_fit_indices,
                query_frames,
                diff_train,
                diff_queries,
            )
        else:
            model_matrix, query_matrices = make_token_matrices(
                train,
                positive_fit_indices,
                query_frames,
                diff_train,
                diff_queries,
            )

        for level in LEVELS:
            encoder = LabelEncoder()
            target = encoder.fit_transform(ec_prefix(train["ec"].iloc[positive_fit_indices], level))
            if len(encoder.classes_) < 2:
                continue
            for c_value in SVC_C_GRID:
                try:
                    classifier = LinearSVC(
                        C=c_value,
                        class_weight="balanced",
                        dual=True,
                        max_iter=5000,
                        random_state=SEED + level,
                    )
                    classifier.fit(model_matrix, target)
                    for name, frame in query_frames.items():
                        detailed = detailed_decision_features(
                            classifier,
                            encoder,
                            query_matrices[name],
                            ec_prefix(frame["ec"], level),
                        )
                        for feature_index, feature_name in enumerate(DECISION_FEATURES):
                            column_name = f"{view}_L{level}_C{c_value}_{feature_name}"
                            outputs[name][:, BASE_FEATURE_INDEX[column_name]] = detailed[:, feature_index]
                except Exception as exc:
                    log(f"warning: skipped {view}/L{level}/C{c_value}: {type(exc).__name__}")
                finally:
                    if "classifier" in locals():
                        del classifier
        del model_matrix, query_matrices
        gc.collect()

    del diff_train, diff_queries
    gc.collect()
    return outputs

def detailed_score_features(
    score_matrix: np.ndarray,
    classes: np.ndarray,
    given_classes: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float32)
    class_index = {label: index for index, label in enumerate(classes)}
    row_minimum = scores.min(axis=1)
    given = np.empty(len(given_classes), dtype=np.float32)
    known = np.zeros(len(given_classes), dtype=np.float32)
    for row, label in enumerate(given_classes):
        index = class_index.get(label)
        if index is None:
            given[row] = row_minimum[row]
        else:
            given[row] = scores[row, index]
            known[row] = 1.0
    top = scores.max(axis=1)
    gap = given - top
    rank = np.mean(scores <= given[:, None], axis=1, dtype=np.float32)
    shifted = np.clip(scores - top[:, None], -50.0, 0.0)
    probability = np.exp(np.clip(gap, -50.0, 0.0)) / np.exp(shifted).sum(axis=1)
    spread = scores.std(axis=1)
    return np.column_stack((given, gap, rank, probability, known, top, spread)).astype(np.float32)


def prototype_features_from_similarity(
    similarity: np.ndarray,
    fit_prefixes: dict[int, np.ndarray],
    query: pd.DataFrame,
) -> np.ndarray:
    blocks = []
    for level in (2, 3, 4):
        reference = fit_prefixes[level]
        supplied = ec_prefix(query["ec"], level)
        classes = np.unique(reference)
        class_index = {label: index for index, label in enumerate(classes)}
        group_indices = [np.where(reference == label)[0] for label in classes]
        class_maximum = np.empty((len(query), len(classes)), dtype=np.float32)
        class_mean = np.empty_like(class_maximum)
        for class_column, indices in enumerate(group_indices):
            values = similarity[:, indices]
            class_maximum[:, class_column] = values.max(axis=1)
            class_mean[:, class_column] = values.mean(axis=1)

        maximum_top = class_maximum.max(axis=1)
        mean_top = class_mean.max(axis=1)
        output = np.zeros((len(query), PROTOTYPE_FEATURES_PER_LEVEL), dtype=np.float32)
        for row, label in enumerate(supplied):
            column = class_index.get(label)
            if column is None:
                continue
            values = similarity[row, group_indices[column]]
            nearest = np.sort(values)[-min(3, len(values)) :]
            supplied_maximum = class_maximum[row, column]
            supplied_mean = class_mean[row, column]
            shifted = np.clip(class_maximum[row] - maximum_top[row], -30.0, 0.0)
            output[row] = (
                supplied_maximum,
                supplied_maximum - maximum_top[row],
                np.mean(class_maximum[row] <= supplied_maximum),
                np.exp(np.clip(supplied_maximum - maximum_top[row], -30.0, 0.0))
                / np.exp(shifted).sum(),
                supplied_mean,
                supplied_mean - mean_top[row],
                np.mean(class_mean[row] <= supplied_mean),
                nearest.mean(),
                values.std(),
                np.log1p(len(values)),
                1.0,
                maximum_top[row],
            )
        blocks.append(output)
    return np.hstack(blocks).astype(np.float32, copy=False)


def append_prototype_view(
    outputs: dict[str, list[np.ndarray]],
    fit_matrix: sparse.csr_matrix,
    query_matrices: dict[str, sparse.csr_matrix],
    fit_prefixes: dict[int, np.ndarray],
    query_frames: dict[str, pd.DataFrame],
) -> None:
    for name, matrix in query_matrices.items():
        similarity = (matrix @ fit_matrix.T).toarray().astype(np.float32)
        outputs[name].append(
            prototype_features_from_similarity(similarity, fit_prefixes, query_frames[name])
        )
        del similarity


def sparse_block_interaction(
    matrix: sparse.csr_matrix,
    codes: np.ndarray,
    class_count: int,
) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    repeated_codes = np.repeat(np.asarray(codes, dtype=np.int64), np.diff(matrix.indptr))
    indices = matrix.indices.astype(np.int64) + repeated_codes * matrix.shape[1]
    return sparse.csr_matrix(
        (matrix.data.copy(), indices, matrix.indptr.copy()),
        shape=(matrix.shape[0], matrix.shape[1] * class_count),
        dtype=np.float32,
    )


def fit_advanced_models(
    train: pd.DataFrame,
    fit_indices: np.ndarray,
    query_frames: dict[str, pd.DataFrame],
    train_signatures: list[str],
    query_signatures: dict[str, list[str]],
    train_graph_signatures: list[str],
    query_graph_signatures: dict[str, list[str]],
    hard_negative: np.ndarray,
) -> dict[str, np.ndarray]:
    positive_fit_indices = fit_indices[train["label"].iloc[fit_indices].to_numpy() == 1]
    blocks: dict[str, list[np.ndarray]] = {name: [] for name in query_frames}
    if len(positive_fit_indices) == 0:
        return {
            name: np.zeros((len(frame), ADVANCED_FEATURE_WIDTH), dtype=np.float32)
            for name, frame in query_frames.items()
        }

    fit_prefixes = {
        level: ec_prefix(train["ec"].iloc[positive_fit_indices], level)
        for level in (2, 3, 4)
    }

    graph_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_features=160000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    graph_fit = graph_vectorizer.fit_transform(
        [train_graph_signatures[index] for index in positive_fit_indices]
    )
    graph_queries = {
        name: graph_vectorizer.transform(query_graph_signatures[name])
        for name in query_frames
    }

    for level in LEVELS:
        encoder = LabelEncoder()
        target = encoder.fit_transform(fit_prefixes[level])
        for c_value in SVC_C_GRID:
            try:
                classifier = LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    dual=True,
                    max_iter=5000,
                    random_state=SEED + 100 + level,
                )
                classifier.fit(graph_fit, target)
                for name, frame in query_frames.items():
                    blocks[name].append(
                        detailed_decision_features(
                            classifier,
                            encoder,
                            graph_queries[name],
                            ec_prefix(frame["ec"], level),
                        )
                    )
            except Exception as exc:
                log(f"warning: skipped graph SVC L{level}/C{c_value}: {type(exc).__name__}")
                for name, frame in query_frames.items():
                    blocks[name].append(
                        np.zeros((len(frame), len(DECISION_FEATURES)), dtype=np.float32)
                    )

    append_prototype_view(blocks, graph_fit, graph_queries, fit_prefixes, query_frames)

    reaction_text = train["substrates"] + ">>" + train["candidate"]
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_fit = char_vectorizer.fit_transform(reaction_text.iloc[positive_fit_indices])
    char_queries = {
        name: char_vectorizer.transform(frame["substrates"] + ">>" + frame["candidate"])
        for name, frame in query_frames.items()
    }
    append_prototype_view(blocks, char_fit, char_queries, fit_prefixes, query_frames)
    del char_fit, char_queries, char_vectorizer
    gc.collect()

    diff_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    diff_fit = diff_vectorizer.fit_transform(
        [train_signatures[index] for index in positive_fit_indices]
    )
    diff_queries = {
        name: diff_vectorizer.transform(query_signatures[name]) for name in query_frames
    }
    append_prototype_view(blocks, diff_fit, diff_queries, fit_prefixes, query_frames)
    del diff_fit, diff_queries, diff_vectorizer
    gc.collect()

    direct_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    direct_fit = direct_vectorizer.fit_transform(
        [train_graph_signatures[index] for index in fit_indices]
    )
    direct_queries = {
        name: direct_vectorizer.transform(query_graph_signatures[name])
        for name in query_frames
    }
    fit_parts = [direct_fit]
    query_parts: dict[str, list[sparse.csr_matrix]] = {
        name: [matrix] for name, matrix in direct_queries.items()
    }
    for level in (2, 3, 4):
        fit_values = ec_prefix(train["ec"].iloc[fit_indices], level)
        classes = np.unique(fit_values)
        class_index = {label: index for index, label in enumerate(classes)}
        unknown = len(classes)
        fit_codes = np.asarray([class_index[value] for value in fit_values], dtype=np.int64)
        fit_parts.append(sparse_block_interaction(direct_fit, fit_codes, unknown + 1))
        for name, frame in query_frames.items():
            query_values = ec_prefix(frame["ec"], level)
            query_codes = np.asarray(
                [class_index.get(value, unknown) for value in query_values],
                dtype=np.int64,
            )
            query_parts[name].append(
                sparse_block_interaction(direct_queries[name], query_codes, unknown + 1)
            )
    direct_fit_interactions = sparse.hstack(fit_parts, format="csr", dtype=np.float32)
    direct_query_interactions = {
        name: sparse.hstack(parts, format="csr", dtype=np.float32)
        for name, parts in query_parts.items()
    }
    fit_labels = train["label"].iloc[fit_indices].to_numpy(dtype=np.int8)
    fit_weights = np.where(hard_negative[fit_indices], 2.0, 1.0)
    for alpha in DIRECT_ALPHA_GRID:
        try:
            classifier = SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                max_iter=100,
                tol=1.0e-5,
                average=True,
                random_state=SEED,
            )
            classifier.fit(direct_fit_interactions, fit_labels, sample_weight=fit_weights)
            for name in query_frames:
                blocks[name].append(
                    classifier.decision_function(direct_query_interactions[name])
                    .astype(np.float32)
                    .reshape(-1, 1)
                )
        except Exception as exc:
            log(f"warning: skipped direct graph alpha={alpha}: {type(exc).__name__}")
            for name, frame in query_frames.items():
                blocks[name].append(np.zeros((len(frame), 1), dtype=np.float32))
    del direct_fit_interactions, direct_query_interactions, direct_fit, direct_queries
    gc.collect()

    for level in (2, 3, 4):
        encoder = LabelEncoder()
        target = encoder.fit_transform(fit_prefixes[level])
        for alpha in NB_ALPHA_GRID:
            for classifier_type in (MultinomialNB, ComplementNB):
                try:
                    classifier = classifier_type(alpha=alpha)
                    classifier.fit(graph_fit, target)
                    for name, frame in query_frames.items():
                        blocks[name].append(
                            detailed_score_features(
                                classifier.predict_log_proba(graph_queries[name]),
                                encoder.classes_,
                                ec_prefix(frame["ec"], level),
                            )
                        )
                except Exception as exc:
                    log(
                        f"warning: skipped {classifier_type.__name__} "
                        f"L{level}/alpha={alpha}: {type(exc).__name__}"
                    )
                    for name, frame in query_frames.items():
                        blocks[name].append(
                            np.zeros((len(frame), len(DECISION_FEATURES)), dtype=np.float32)
                        )

    outputs = {
        name: np.hstack(parts).astype(np.float32, copy=False) for name, parts in blocks.items()
    }
    for name, frame in query_frames.items():
        if outputs[name].shape != (len(frame), ADVANCED_FEATURE_WIDTH):
            log(f"warning: invalid advanced feature width for {name}; using neutral features")
            outputs[name] = np.zeros(
                (len(frame), ADVANCED_FEATURE_WIDTH),
                dtype=np.float32,
            )
    del graph_fit, graph_queries
    gc.collect()
    return outputs


def meta_feature_sets(base_width: int, numeric_width: int, count_width: int) -> dict[str, np.ndarray]:
    base = np.arange(base_width)
    numeric = np.arange(base_width, base_width + numeric_width)
    counts = np.arange(base_width + numeric_width, base_width + numeric_width + count_width)
    return {
        "full": np.concatenate((base, numeric, counts)),
        "enzyme": np.concatenate((base, counts)),
        "chemistry": np.concatenate((base, numeric)),
    }


def make_meta_model(configuration: dict[str, float | int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=SEED,
        **configuration,
    )

def catboost_frame(features: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        np.asarray(features, dtype=np.float32),
        columns=[f"x{index}" for index in range(features.shape[1])],
    )
    for level in (1, 2, 3, 4):
        output[f"ec{level}"] = ec_prefix(frame["ec"], level)
    return output


def make_catboost_model(configuration: dict[str, float | int]) -> object:
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is unavailable")
    parameters = {
        key: value
        for key, value in configuration.items()
        if key not in {"hard_weight", "feature_count"}
    }
    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **parameters,
    )


def rank_catboost_numeric_features(
    frame: pd.DataFrame,
    labels: np.ndarray,
    hard_negative: np.ndarray,
) -> list[str]:
    categorical = [f"ec{level}" for level in (1, 2, 3, 4)]
    numeric = [column for column in frame.columns if column not in categorical]
    selector_configuration = {
        "depth": 5,
        "iterations": 800,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.1,
    }
    selector = make_catboost_model(selector_configuration)
    selector.fit(
        frame,
        labels,
        cat_features=categorical,
        sample_weight=np.where(hard_negative, 1.5, 1.0),
    )
    importance = selector.get_feature_importance()
    return sorted(
        numeric,
        key=lambda column: importance[frame.columns.get_loc(column)],
        reverse=True,
    )


def calibrate(probabilities: np.ndarray, slope: float, bias: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    transformed = np.clip(slope * logits + bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-transformed))


def write_submission(path: Path, ids: pd.Series, scores: np.ndarray) -> None:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(ids):
        log("warning: prediction length mismatch; retaining the previous valid submission")
        return
    values = np.where(np.isfinite(values), values, 0.5)
    values = np.clip(values, 0.0, 1.0)
    submission = pd.DataFrame({"id": ids.astype(str).to_numpy(), "score": values})
    if list(submission.columns) != ["id", "score"] or len(submission) != len(ids):
        log("warning: invalid output schema; retaining the previous valid submission")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def tune_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: np.ndarray,
    hard_negative: np.ndarray,
    train_signatures: list[str],
    test_signatures: list[str],
    train_graph_signatures: list[str],
    test_graph_signatures: list[str],
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    available_folds = sorted(int(value) for value in np.unique(folds) if value >= 0)
    if len(available_folds) < 4:
        raise RuntimeError("four product-disjoint folds are required for clean stacked tuning")

    fit_folds = set(available_folds[:2])
    calibration_fold = available_folds[2]
    validation_fold = available_folds[3]
    fit_indices = np.where(np.isin(folds, list(fit_folds)))[0]
    calibration_indices = np.where(folds == calibration_fold)[0]
    validation_indices = np.where(folds == validation_fold)[0]

    query_frames = {
        "calibration": train.iloc[calibration_indices].reset_index(drop=True),
        "validation": train.iloc[validation_indices].reset_index(drop=True),
        "test": test,
    }
    query_signatures = {
        "calibration": [train_signatures[index] for index in calibration_indices],
        "validation": [train_signatures[index] for index in validation_indices],
        "test": test_signatures,
    }
    query_graph_signatures = {
        "calibration": [train_graph_signatures[index] for index in calibration_indices],
        "validation": [train_graph_signatures[index] for index in validation_indices],
        "test": test_graph_signatures,
    }
    base = fit_base_models(
        train,
        fit_indices,
        query_frames,
        train_signatures,
        query_signatures,
    )
    advanced = fit_advanced_models(
        train,
        fit_indices,
        query_frames,
        train_signatures,
        query_signatures,
        train_graph_signatures,
        query_graph_signatures,
        hard_negative,
    )
    reference = train.iloc[fit_indices]
    counts_calibration = build_count_features(reference, query_frames["calibration"])
    counts_validation = build_count_features(reference, query_frames["validation"])
    counts_test = build_count_features(reference, test)

    meta_calibration = np.hstack(
        (
            base["calibration"],
            advanced["calibration"],
            train_numeric[calibration_indices],
            counts_calibration,
        )
    ).astype(np.float32)
    meta_validation = np.hstack(
        (
            base["validation"],
            advanced["validation"],
            train_numeric[validation_indices],
            counts_validation,
        )
    ).astype(np.float32)
    meta_test = np.hstack(
        (base["test"], advanced["test"], test_numeric, counts_test)
    ).astype(np.float32)
    learned_width = len(BASE_FEATURE_NAMES) + ADVANCED_FEATURE_WIDTH
    feature_sets = meta_feature_sets(
        learned_width,
        train_numeric.shape[1],
        counts_calibration.shape[1],
    )

    labels_calibration = train["label"].iloc[calibration_indices].to_numpy(dtype=np.int8)
    labels_validation = train["label"].iloc[validation_indices].to_numpy(dtype=np.int8)
    hard_calibration = hard_negative[calibration_indices]
    hard_validation = hard_negative[validation_indices]
    best: dict[str, object] | None = None
    best_test_scores: np.ndarray | None = None

    def consider(
        raw_validation: np.ndarray,
        raw_test: np.ndarray,
        description: dict[str, object],
    ) -> None:
        nonlocal best, best_test_scores
        for slope in CALIBRATION_SLOPES:
            for bias in CALIBRATION_BIASES:
                validation_scores = calibrate(raw_validation, slope, bias)
                metrics = challenge_metric(labels_validation, validation_scores, hard_validation)
                if best is None or metrics["score"] > float(best["metrics"]["score"]):
                    best = {
                        **description,
                        "slope": slope,
                        "bias": bias,
                        "metrics": metrics,
                        "fit_rows": len(fit_indices),
                        "calibration_rows": len(calibration_indices),
                        "validation_rows": len(validation_indices),
                    }
                    best_test_scores = calibrate(raw_test, slope, bias)

    for feature_set_name, columns in feature_sets.items():
        for config_index, configuration in enumerate(META_CONFIGS):
            model = make_meta_model(configuration)
            model.fit(meta_calibration[:, columns], labels_calibration)
            consider(
                model.predict_proba(meta_validation[:, columns])[:, 1],
                model.predict_proba(meta_test[:, columns])[:, 1],
                {
                    "model_type": "hist",
                    "feature_set": feature_set_name,
                    "config_index": config_index,
                    "configuration": configuration,
                },
            )
            del model

    if CatBoostClassifier is not None:
        categorical = [f"ec{level}" for level in (1, 2, 3, 4)]
        cat_calibration = catboost_frame(meta_calibration, query_frames["calibration"])
        cat_validation = catboost_frame(meta_validation, query_frames["validation"])
        cat_test = catboost_frame(meta_test, test)
        try:
            ranked_numeric = rank_catboost_numeric_features(
                cat_calibration,
                labels_calibration,
                hard_calibration,
            )
            for config_index, configuration in enumerate(CAT_META_CONFIGS):
                feature_count = int(configuration["feature_count"])
                numeric_columns = (
                    ranked_numeric
                    if feature_count <= 0
                    else ranked_numeric[: min(feature_count, len(ranked_numeric))]
                )
                columns = numeric_columns + categorical
                model = make_catboost_model(configuration)
                model.fit(
                    cat_calibration[columns],
                    labels_calibration,
                    cat_features=categorical,
                    sample_weight=np.where(
                        hard_calibration,
                        float(configuration["hard_weight"]),
                        1.0,
                    ),
                )
                consider(
                    model.predict_proba(cat_validation[columns])[:, 1],
                    model.predict_proba(cat_test[columns])[:, 1],
                    {
                        "model_type": "cat",
                        "config_index": config_index,
                        "configuration": configuration,
                        "feature_count": feature_count,
                    },
                )
                del model
        except Exception as exc:
            log(f"warning: CatBoost HPO failed: {type(exc).__name__}: {exc}")

    if best is None or best_test_scores is None:
        raise RuntimeError("no meta-model configuration completed")
    return best, best_test_scores


def production_fit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: np.ndarray,
    hard_negative: np.ndarray,
    train_signatures: list[str],
    test_signatures: list[str],
    train_graph_signatures: list[str],
    test_graph_signatures: list[str],
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
    selection: dict[str, object],
    started_at: float,
) -> tuple[np.ndarray, int]:
    oof_base = np.full((len(train), len(BASE_FEATURE_NAMES)), np.nan, dtype=np.float32)
    oof_advanced = np.full(
        (len(train), ADVANCED_FEATURE_WIDTH),
        np.nan,
        dtype=np.float32,
    )
    oof_counts = np.full((len(train), 8), np.nan, dtype=np.float32)
    test_base_sum = np.zeros((len(test), len(BASE_FEATURE_NAMES)), dtype=np.float64)
    test_advanced_sum = np.zeros((len(test), ADVANCED_FEATURE_WIDTH), dtype=np.float64)
    test_count_sum = np.zeros((len(test), 8), dtype=np.float64)
    completed_folds: list[int] = []

    for fold in sorted(int(value) for value in np.unique(folds) if value >= 0):
        if time.monotonic() - started_at >= TIME_GUARD_SECONDS and completed_folds:
            log("wall-clock guard reached; moving to final stacked inference")
            break
        fit_indices = np.where(folds != fold)[0]
        validation_indices = np.where(folds == fold)[0]
        query_frames = {
            "validation": train.iloc[validation_indices].reset_index(drop=True),
            "test": test,
        }
        query_signatures = {
            "validation": [train_signatures[index] for index in validation_indices],
            "test": test_signatures,
        }
        query_graph_signatures = {
            "validation": [
                train_graph_signatures[index] for index in validation_indices
            ],
            "test": test_graph_signatures,
        }
        fold_base = fit_base_models(
            train,
            fit_indices,
            query_frames,
            train_signatures,
            query_signatures,
        )
        fold_advanced = fit_advanced_models(
            train,
            fit_indices,
            query_frames,
            train_signatures,
            query_signatures,
            train_graph_signatures,
            query_graph_signatures,
            hard_negative,
        )
        oof_base[validation_indices] = fold_base["validation"]
        oof_advanced[validation_indices] = fold_advanced["validation"]
        test_base_sum += fold_base["test"]
        test_advanced_sum += fold_advanced["test"]

        reference = train.iloc[fit_indices]
        oof_counts[validation_indices] = build_count_features(
            reference,
            query_frames["validation"],
        )
        test_count_sum += build_count_features(reference, test)
        completed_folds.append(fold)
        log(f"completed production fold {fold + 1}; elapsed={time.monotonic() - started_at:.1f}s")
        del fold_base, fold_advanced
        gc.collect()

    if not completed_folds:
        raise RuntimeError("no production fold completed")
    valid_rows = (
        np.isfinite(oof_base).all(axis=1)
        & np.isfinite(oof_advanced).all(axis=1)
        & np.isfinite(oof_counts).all(axis=1)
    )
    test_base = (test_base_sum / len(completed_folds)).astype(np.float32)
    test_advanced = (test_advanced_sum / len(completed_folds)).astype(np.float32)
    test_counts = (test_count_sum / len(completed_folds)).astype(np.float32)
    meta_train = np.hstack(
        (
            oof_base[valid_rows],
            oof_advanced[valid_rows],
            train_numeric[valid_rows],
            oof_counts[valid_rows],
        )
    ).astype(np.float32)
    meta_test = np.hstack(
        (test_base, test_advanced, test_numeric, test_counts)
    ).astype(np.float32)

    if selection.get("model_type") == "cat" and CatBoostClassifier is not None:
        train_rows = train.loc[valid_rows].reset_index(drop=True)
        cat_train = catboost_frame(meta_train, train_rows)
        cat_test = catboost_frame(meta_test, test)
        categorical = [f"ec{level}" for level in (1, 2, 3, 4)]
        ranked_numeric = rank_catboost_numeric_features(
            cat_train,
            train["label"].to_numpy(dtype=np.int8)[valid_rows],
            hard_negative[valid_rows],
        )
        feature_count = int(selection.get("feature_count", 0))
        numeric_columns = (
            ranked_numeric
            if feature_count <= 0
            else ranked_numeric[: min(feature_count, len(ranked_numeric))]
        )
        columns = numeric_columns + categorical
        configuration = dict(selection["configuration"])
        model = make_catboost_model(configuration)
        model.fit(
            cat_train[columns],
            train["label"].to_numpy(dtype=np.int8)[valid_rows],
            cat_features=categorical,
            sample_weight=np.where(
                hard_negative[valid_rows],
                float(configuration["hard_weight"]),
                1.0,
            ),
        )
        raw_test = model.predict_proba(cat_test[columns])[:, 1]
    else:
        learned_width = len(BASE_FEATURE_NAMES) + ADVANCED_FEATURE_WIDTH
        feature_sets = meta_feature_sets(learned_width, train_numeric.shape[1], 8)
        selected_columns = feature_sets[str(selection["feature_set"])]
        model = make_meta_model(dict(selection["configuration"]))
        model.fit(
            meta_train[:, selected_columns],
            train["label"].to_numpy(dtype=np.int8)[valid_rows],
        )
        raw_test = model.predict_proba(meta_test[:, selected_columns])[:, 1]

    scores = calibrate(raw_test, float(selection["slope"]), float(selection["bias"]))
    return scores, int(valid_rows.sum())


def validate_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    test_path = public_dir / "test.csv"
    train_path = public_dir / "train.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"missing required input: {test_path}")

    test = pd.read_csv(test_path, dtype=str)
    validate_columns(test, ["id", "substrates", "ec", "candidate"], "test.csv")
    for column in ("id", "substrates", "ec", "candidate"):
        test[column] = test[column].fillna("").astype(str)
    write_submission(submission_out, test["id"], np.full(len(test), 0.5, dtype=np.float64))
    log(f"wrote early schema-safe placeholder with {len(test)} rows")

    if not train_path.is_file():
        raise FileNotFoundError(f"missing required input: {train_path}")
    started_at = time.monotonic()

    try:
        train = pd.read_csv(
            train_path,
            dtype={"id": str, "substrates": str, "ec": str, "candidate": str},
        )
        validate_columns(train, ["id", "substrates", "ec", "candidate", "label"], "train.csv")
        for column in ("id", "substrates", "ec", "candidate"):
            train[column] = train[column].fillna("").astype(str)
        train["label"] = pd.to_numeric(train["label"], errors="coerce").fillna(0).astype(np.int8)
        invalid_labels = ~train["label"].isin([0, 1])
        if invalid_labels.any():
            log(f"warning: {int(invalid_labels.sum())} invalid labels were mapped to zero")
            train.loc[invalid_labels, "label"] = 0

        folds = make_folds(train)
        hard_negative = infer_hard_negatives(train)
        alphabet = fit_alphabet(train)
        train_signatures = [
            reaction_diff_signature(substrates, candidate)
            for substrates, candidate in train[["substrates", "candidate"]].itertuples(
                index=False, name=None
            )
        ]
        test_signatures = [
            reaction_diff_signature(substrates, candidate)
            for substrates, candidate in test[["substrates", "candidate"]].itertuples(
                index=False, name=None
            )
        ]
        train_graph_signatures = [
            graph_reaction_signature(substrates, candidate)
            for substrates, candidate in train[["substrates", "candidate"]].itertuples(
                index=False, name=None
            )
        ]
        test_graph_signatures = [
            graph_reaction_signature(substrates, candidate)
            for substrates, candidate in test[["substrates", "candidate"]].itertuples(
                index=False, name=None
            )
        ]
        train_numeric = numeric_features(train, alphabet)
        test_numeric = numeric_features(test, alphabet)

        selection, interim_scores = tune_pipeline(
            train,
            test,
            folds,
            hard_negative,
            train_signatures,
            test_signatures,
            train_graph_signatures,
            test_graph_signatures,
            train_numeric,
            test_numeric,
        )
        metrics = selection["metrics"]
        log(
            "clean product-disjoint validation: "
            f"score={metrics['score']:.6f} quality={metrics['quality']:.6f} "
            f"auc={metrics['auc']:.6f} hard_auc={metrics['hard_auc']:.6f} "
            f"cal={metrics['cal']:.6f} nll={metrics['nll']:.6f}"
        )
        log(
            f"selected meta={selection['model_type']}/{selection['config_index']} "
            f"slope={selection['slope']} bias={selection['bias']}"
        )
        write_submission(submission_out, test["id"], interim_scores)
        log("wrote trained half-split fallback submission")

        final_scores, meta_rows = production_fit(
            train,
            test,
            folds,
            hard_negative,
            train_signatures,
            test_signatures,
            train_graph_signatures,
            test_graph_signatures,
            train_numeric,
            test_numeric,
            selection,
            started_at,
        )
        write_submission(submission_out, test["id"], final_scores)
        log(
            f"wrote final submission with {len(final_scores)} rows; "
            f"meta_fit_rows={meta_rows}; elapsed={time.monotonic() - started_at:.1f}s"
        )
    except Exception as exc:
        log(f"warning: heavy pipeline stopped with {type(exc).__name__}: {exc}")
        log("the latest schema-valid submission remains on disk")


if __name__ == "__main__":
    main()
