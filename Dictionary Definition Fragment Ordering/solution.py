#!/usr/bin/env python3
"""CPU-only solution for Dictionary Definition Fragment Ordering.

Usage:
    python3 solution.py <public_dir> <submission_out>

With no arguments, paths default to dataset/public and working/submission.csv
next to this file, as required by the local challenge layout.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import itertools
import math
import os
import sys
import time
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
# Bound native libraries before importing NumPy/sklearn. Their unconstrained
# thread pools oversubscribe small CPU runners and make fitting much slower.
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


BOS = "<BOS>"
EOS = "<EOS>"


def read_rows(path: Path, labeled: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            parent = json.loads(raw["parent_context_json"])
            answer = json.loads(raw["answer_json"])["fragment_order"] if labeled else None
            rows.append(
                {
                    "id": raw["id"],
                    "lemmas": json.loads(raw["lemma_tokens_json"]),
                    "pos": raw["part_of_speech"],
                    "parent": parent,
                    "cards": json.loads(raw["fragment_cards_json"]),
                    "order": answer,
                }
            )
    return rows


def ordered_tokens(row: dict[str, Any]) -> list[str]:
    by_id = {card["fragment_id"]: card["tokens"] for card in row["cards"]}
    return [token for fragment_id in row["order"] for token in by_id[fragment_id]]


def synthetic_fragment_rows(
    rows: list[dict[str, Any]], seed: int, copies: int = 1
) -> list[dict[str, Any]]:
    """Re-fragment authentic definitions into synthetic ranker rows."""
    generator = random.Random(seed)
    synthetic: list[dict[str, Any]] = []
    for row in rows:
        tokens = ordered_tokens(row)
        token_count = len(tokens)
        if token_count < 8:
            continue
        for copy_index in range(copies):
            card_count = generator.randint(
                4, min(7, token_count // 2)
            )
            slack = token_count - 2 * card_count
            offsets = sorted(
                generator.choices(
                    range(slack + 1), k=card_count - 1
                )
            )
            bounds = [0]
            bounds.extend(
                offset + 2 * (index + 1)
                for index, offset in enumerate(offsets)
            )
            bounds.append(token_count)
            spans = [
                tokens[bounds[index] : bounds[index + 1]]
                for index in range(card_count)
            ]
            displayed = list(range(card_count))
            generator.shuffle(displayed)
            cards = [
                {
                    "fragment_id": f"s{position:02d}",
                    "tokens": spans[span_index],
                }
                for position, span_index in enumerate(displayed)
            ]
            order = [
                f"s{displayed.index(span_index):02d}"
                for span_index in range(card_count)
            ]
            synthetic.append(
                {
                    "id": f"{row['id']}_syn{copy_index}",
                    "pos": row["pos"],
                    "cards": cards,
                    "order": order,
                    "parent": None,
                }
            )
    return synthetic


def parent_sequences(rows: Iterable[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    return [
        (row["parent"]["part_of_speech"], row["parent"]["definition_tokens"])
        for row in rows
        if row["parent"] is not None
    ]


class TrigramLanguageModel:
    """Interpolated trigram model used only at fragment boundaries."""

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        extra_sequences: list[tuple[str, list[str]]],
        bigram_strength: float = 5.0,
        trigram_strength: float = 30.0,
    ) -> None:
        unigram: Counter[str] = Counter()
        bigram: Counter[tuple[str, str]] = Counter()
        bigram_context: Counter[str] = Counter()
        trigram: Counter[tuple[str, str, str]] = Counter()
        trigram_context: Counter[tuple[str, str]] = Counter()

        sequences = [ordered_tokens(row) for row in train_rows]
        sequences.extend(tokens for _, tokens in extra_sequences)
        for tokens in sequences:
            padded = [BOS, BOS, *tokens, EOS]
            for index in range(2, len(padded)):
                first, second, third = padded[index - 2 : index + 1]
                unigram[third] += 1
                bigram[(second, third)] += 1
                bigram_context[second] += 1
                trigram[(first, second, third)] += 1
                trigram_context[(first, second)] += 1

        self.unigram = unigram
        self.bigram = bigram
        self.bigram_context = bigram_context
        self.trigram = trigram
        self.trigram_context = trigram_context
        self.total = sum(unigram.values())
        self.vocab_size = len(unigram) + 1
        self.bigram_strength = bigram_strength
        self.trigram_strength = trigram_strength

    @lru_cache(maxsize=None)
    def log_probability(self, first: str, second: str, third: str) -> float:
        unigram_probability = (self.unigram[third] + 0.2) / (
            self.total + 0.2 * self.vocab_size
        )
        bigram_probability = (
            self.bigram[(second, third)]
            + self.bigram_strength * unigram_probability
        ) / (self.bigram_context[second] + self.bigram_strength)
        trigram_probability = (
            self.trigram[(first, second, third)]
            + self.trigram_strength * bigram_probability
        ) / (self.trigram_context[(first, second)] + self.trigram_strength)
        return math.log(trigram_probability)

    def start_score(self, tokens: list[str]) -> float:
        return self.log_probability(BOS, BOS, tokens[0]) + self.log_probability(
            BOS, tokens[0], tokens[1]
        )

    def end_score(self, tokens: list[str]) -> float:
        return self.log_probability(tokens[-2], tokens[-1], EOS)

    def edge_score(self, left: list[str], right: list[str]) -> float:
        return self.log_probability(left[-2], left[-1], right[0]) + self.log_probability(
            left[-1], right[0], right[1]
        )


def card_features(row: dict[str, Any], card: dict[str, Any]) -> dict[str, float]:
    tokens = card["tokens"]
    features: dict[str, float] = {
        "bias": 1.0,
        "pos=" + row["pos"]: 1.0,
        "length": float(len(tokens)),
        "log_length": math.log(len(tokens)),
    }
    for token in tokens:
        key = "token=" + token
        features[key] = features.get(key, 0.0) + 1.0
    for offset in range(min(3, len(tokens))):
        features[f"first{offset}={tokens[offset]}"] = 1.0
        features[f"last{offset}={tokens[-1 - offset]}"] = 1.0
    for offset in range(min(2, len(tokens) - 1)):
        features[f"first_bigram{offset}={tokens[offset]}|{tokens[offset + 1]}"] = 1.0
        features[
            f"last_bigram{offset}={tokens[-2 - offset]}|{tokens[-1 - offset]}"
        ] = 1.0

    lemma_tokens = {token for lemma in row["lemmas"] for token in lemma}
    features["lemma_overlap"] = float(sum(token in lemma_tokens for token in tokens))
    if row["parent"] is None:
        features["no_parent"] = 1.0
    else:
        parent_lemmas = {
            token for lemma in row["parent"]["lemma_tokens"] for token in lemma
        }
        parent_definition = set(row["parent"]["definition_tokens"])
        features["parent_lemma_overlap"] = float(
            sum(token in parent_lemmas for token in tokens)
        )
        features["parent_definition_overlap"] = float(
            sum(token in parent_definition for token in tokens)
        )
    return features


class PositionModels:
    """One multiclass fragment-position model for each card count."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.models: dict[int, tuple[DictVectorizer, LogisticRegression]] = {}
        for card_count in range(4, 8):
            feature_rows: list[dict[str, float]] = []
            labels: list[int] = []
            for row in rows:
                if len(row["cards"]) != card_count:
                    continue
                true_position = {
                    fragment_id: index for index, fragment_id in enumerate(row["order"])
                }
                for card in row["cards"]:
                    feature_rows.append(card_features(row, card))
                    labels.append(true_position[card["fragment_id"]])
            vectorizer = DictVectorizer()
            matrix = vectorizer.fit_transform(feature_rows)
            model = LogisticRegression(
                C=0.05,
                max_iter=150,
                solver="lbfgs",
                random_state=20260715,
            )
            model.fit(matrix, labels)
            self.models[card_count] = (vectorizer, model)

    def score_matrix(self, row: dict[str, Any]) -> np.ndarray:
        vectorizer, model = self.models[len(row["cards"])]
        matrix = vectorizer.transform([card_features(row, card) for card in row["cards"]])
        return model.predict_log_proba(matrix)


def boundary_features(
    row: dict[str, Any], left_card: dict[str, Any], right_card: dict[str, Any]
) -> dict[str, float]:
    left = left_card["tokens"]
    right = right_card["tokens"]
    return {
        "bias": 1.0,
        "pos=" + row["pos"]: 1.0,
        "card_count=" + str(len(row["cards"])): 1.0,
        "left_end0=" + left[-1]: 1.0,
        "left_end1=" + left[-2]: 1.0,
        "right_start0=" + right[0]: 1.0,
        "right_start1=" + right[1]: 1.0,
        "cross2=" + left[-1] + "|" + right[0]: 1.0,
        "cross3a=" + left[-2] + "|" + left[-1] + "|" + right[0]: 1.0,
        "cross3b=" + left[-1] + "|" + right[0] + "|" + right[1]: 1.0,
        "left0_right1=" + left[-1] + "|" + right[1]: 1.0,
        "left1_right0=" + left[-2] + "|" + right[0]: 1.0,
        "left_length": float(len(left)),
        "right_length": float(len(right)),
    }


class FragmentAdjacencyModel:
    """Discriminative classifier for adjacent fragment pairs."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        feature_rows: list[dict[str, float]] = []
        labels: list[int] = []
        for row in rows:
            true_position = {
                fragment_id: index for index, fragment_id in enumerate(row["order"])
            }
            for left_index, left_card in enumerate(row["cards"]):
                for right_index, right_card in enumerate(row["cards"]):
                    if left_index == right_index:
                        continue
                    feature_rows.append(boundary_features(row, left_card, right_card))
                    labels.append(
                        int(
                            true_position[right_card["fragment_id"]]
                            == true_position[left_card["fragment_id"]] + 1
                        )
                    )
        self.vectorizer = DictVectorizer()
        matrix = self.vectorizer.fit_transform(feature_rows)
        self.model = LogisticRegression(
            C=10.0,
            max_iter=100,
            solver="liblinear",
            random_state=20260715,
        )
        self.model.fit(matrix, labels)

    def score_matrix(self, row: dict[str, Any]) -> np.ndarray:
        card_count = len(row["cards"])
        features: list[dict[str, float]] = []
        pairs: list[tuple[int, int]] = []
        for left_index, left_card in enumerate(row["cards"]):
            for right_index, right_card in enumerate(row["cards"]):
                if left_index != right_index:
                    features.append(boundary_features(row, left_card, right_card))
                    pairs.append((left_index, right_index))
        scores = self.model.decision_function(self.vectorizer.transform(features))
        result = np.zeros((card_count, card_count), dtype=np.float64)
        for score, (left_index, right_index) in zip(scores, pairs):
            result[left_index, right_index] = score
        return result


def precedence_features(
    row: dict[str, Any], left_card: dict[str, Any], right_card: dict[str, Any]
) -> dict[str, float]:
    left = left_card["tokens"]
    right = right_card["tokens"]
    features: dict[str, float] = {
        "bias": 1.0,
        "pos=" + row["pos"]: 1.0,
        "card_count=" + str(len(row["cards"])): 1.0,
        "left_length": float(len(left)),
        "right_length": float(len(right)),
        "edge=" + left[-1] + "|" + right[0]: 1.0,
    }
    for token in left:
        key = "left_token=" + token
        features[key] = features.get(key, 0.0) + 1.0
    for token in right:
        key = "right_token=" + token
        features[key] = features.get(key, 0.0) + 1.0
    for offset in range(min(3, len(left))):
        features[f"left_first{offset}={left[offset]}"] = 1.0
        features[f"left_last{offset}={left[-1 - offset]}"] = 1.0
    for offset in range(min(3, len(right))):
        features[f"right_first{offset}={right[offset]}"] = 1.0
        features[f"right_last{offset}={right[-1 - offset]}"] = 1.0

    lemma_tokens = {token for lemma in row["lemmas"] for token in lemma}
    features["left_lemma_overlap"] = float(sum(token in lemma_tokens for token in left))
    features["right_lemma_overlap"] = float(sum(token in lemma_tokens for token in right))
    if row["parent"] is not None:
        parent_lemmas = {
            token for lemma in row["parent"]["lemma_tokens"] for token in lemma
        }
        parent_definition = set(row["parent"]["definition_tokens"])
        features["left_parent_lemma"] = float(sum(token in parent_lemmas for token in left))
        features["right_parent_lemma"] = float(sum(token in parent_lemmas for token in right))
        features["left_parent_definition"] = float(
            sum(token in parent_definition for token in left)
        )
        features["right_parent_definition"] = float(
            sum(token in parent_definition for token in right)
        )
    return features


class PrecedenceModel:
    """Pairwise model for whether one fragment occurs anywhere before another."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        feature_rows: list[dict[str, float]] = []
        labels: list[int] = []
        for row in rows:
            true_position = {
                fragment_id: index for index, fragment_id in enumerate(row["order"])
            }
            for left_index, left_card in enumerate(row["cards"]):
                for right_index, right_card in enumerate(row["cards"]):
                    if left_index == right_index:
                        continue
                    feature_rows.append(precedence_features(row, left_card, right_card))
                    labels.append(
                        int(
                            true_position[left_card["fragment_id"]]
                            < true_position[right_card["fragment_id"]]
                        )
                    )
        self.vectorizer = DictVectorizer()
        matrix = self.vectorizer.fit_transform(feature_rows)
        self.model = LogisticRegression(
            C=1.0,
            max_iter=100,
            solver="liblinear",
            random_state=20260715,
        )
        self.model.fit(matrix, labels)

    def score_matrix(self, row: dict[str, Any]) -> np.ndarray:
        card_count = len(row["cards"])
        features: list[dict[str, float]] = []
        pairs: list[tuple[int, int]] = []
        for left_index, left_card in enumerate(row["cards"]):
            for right_index, right_card in enumerate(row["cards"]):
                if left_index != right_index:
                    features.append(precedence_features(row, left_card, right_card))
                    pairs.append((left_index, right_index))
        scores = self.model.decision_function(self.vectorizer.transform(features))
        result = np.zeros((card_count, card_count), dtype=np.float64)
        for score, (left_index, right_index) in zip(scores, pairs):
            result[left_index, right_index] = score
        return result


def local_boundary_features(
    part_of_speech: str, left: tuple[str, str] | list[str], right: tuple[str, str] | list[str]
) -> dict[str, float]:
    return {
        "bias": 1.0,
        "pos=" + part_of_speech: 1.0,
        "left_end0=" + left[-1]: 1.0,
        "left_end1=" + left[-2]: 1.0,
        "right_start0=" + right[0]: 1.0,
        "right_start1=" + right[1]: 1.0,
        "cross2=" + left[-1] + "|" + right[0]: 1.0,
        "cross3a=" + left[-2] + "|" + left[-1] + "|" + right[0]: 1.0,
        "cross3b=" + left[-1] + "|" + right[0] + "|" + right[1]: 1.0,
        "cross4=" + left[-2] + "|" + left[-1] + "|" + right[0] + "|" + right[1]: 1.0,
        "left0_right1=" + left[-1] + "|" + right[1]: 1.0,
        "left1_right0=" + left[-2] + "|" + right[0]: 1.0,
    }


class TokenAdjacencyModel:
    """Learns valid token joins from every known authentic definition."""

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        extra_sequences: list[tuple[str, list[str]]],
    ) -> None:
        windows: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]] = defaultdict(list)
        sequences = [(row["pos"], ordered_tokens(row)) for row in train_rows]
        sequences.extend(extra_sequences)
        for part_of_speech, tokens in sequences:
            for split in range(2, len(tokens) - 1):
                windows[part_of_speech].append(
                    ((tokens[split - 2], tokens[split - 1]), (tokens[split], tokens[split + 1]))
                )

        feature_rows: list[dict[str, float]] = []
        labels: list[int] = []
        negative_shifts = (1009, 3001)
        for part_of_speech, candidates in windows.items():
            candidate_count = len(candidates)
            for index, (left, right) in enumerate(candidates):
                feature_rows.append(local_boundary_features(part_of_speech, left, right))
                labels.append(1)
                for shift in negative_shifts:
                    negative_right = candidates[(index * 37 + shift) % candidate_count][1]
                    feature_rows.append(
                        local_boundary_features(part_of_speech, left, negative_right)
                    )
                    labels.append(0)

        self.vectorizer = DictVectorizer()
        matrix = self.vectorizer.fit_transform(feature_rows)
        self.model = LogisticRegression(
            C=10.0,
            max_iter=100,
            solver="liblinear",
            random_state=20260715,
        )
        self.model.fit(matrix, labels)

    def score_matrix(self, row: dict[str, Any]) -> np.ndarray:
        card_count = len(row["cards"])
        features: list[dict[str, float]] = []
        pairs: list[tuple[int, int]] = []
        for left_index, left_card in enumerate(row["cards"]):
            for right_index, right_card in enumerate(row["cards"]):
                if left_index != right_index:
                    features.append(
                        local_boundary_features(
                            row["pos"], left_card["tokens"], right_card["tokens"]
                        )
                    )
                    pairs.append((left_index, right_index))
        scores = self.model.decision_function(self.vectorizer.transform(features))
        result = np.zeros((card_count, card_count), dtype=np.float64)
        for score, (left_index, right_index) in zip(scores, pairs):
            result[left_index, right_index] = score
        return result


class LatentRoleModels:
    """Transductive syntax models that generalize beyond exact token aliases.

    Stable aliases seen only in evaluation still occur inside correctly ordered
    cards. Their left/right internal contexts therefore reveal grammatical
    behavior without revealing any cross-card answer. We embed those context
    signatures, cluster them into latent roles, and learn class-level joins.
    """

    POS_CODES = ("n", "v", "a", "s", "r")
    CLASS_COUNT = 192
    EMBEDDING_SIZE = 64

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        target_rows: list[dict[str, Any]],
        extra_sequences: list[tuple[str, list[str]]],
    ) -> None:
        self.full_sequences = [
            (row["pos"], ordered_tokens(row)) for row in train_rows
        ] + list(extra_sequences)
        self.fragments = [
            (row["pos"], card["tokens"])
            for row in train_rows + target_rows
            for card in row["cards"]
        ]
        self._fit_role_space()
        self._fit_class_language_model()
        self._fit_class_token_adjacency()
        self._fit_class_fragment_adjacency(train_rows)

    def _fit_role_space(self) -> None:
        term_frequency: Counter[str] = Counter(
            token
            for _, tokens in self.full_sequences + self.fragments
            for token in tokens
        )
        signatures: dict[str, Counter[str]] = defaultdict(Counter)

        def add_contexts(
            records: list[tuple[str, list[str]]], authentic_endpoints: bool
        ) -> None:
            for part_of_speech, tokens in records:
                token_count = len(tokens)
                for index, token in enumerate(tokens):
                    features = signatures[token]
                    features["pos=" + part_of_speech] += 1
                    for distance, prefix in ((1, "left1="), (2, "left2=")):
                        if (
                            index >= distance
                            and term_frequency[tokens[index - distance]] >= 5
                        ):
                            features[prefix + tokens[index - distance]] += 1
                    for distance, prefix in ((1, "right1="), (2, "right2=")):
                        if (
                            index + distance < token_count
                            and term_frequency[tokens[index + distance]] >= 5
                        ):
                            features[prefix + tokens[index + distance]] += 1
                    if authentic_endpoints:
                        if index < 3:
                            features["start_offset=" + str(index)] += 1
                        end_offset = token_count - 1 - index
                        if end_offset < 3:
                            features["end_offset=" + str(end_offset)] += 1
                        position_bin = min(
                            5, int(6 * index / max(1, token_count - 1))
                        )
                        features["position_bin=" + str(position_bin)] += 1

        add_contexts(self.full_sequences, authentic_endpoints=True)
        add_contexts(self.fragments, authentic_endpoints=False)
        self.vocabulary = sorted(signatures)
        feature_documents: Counter[str] = Counter(
            feature
            for signature in signatures.values()
            for feature in signature
        )
        role_embedding = np.zeros(
            (len(self.vocabulary), self.EMBEDDING_SIZE), dtype=np.float64
        )
        document_count = len(self.vocabulary)
        for token_index, token in enumerate(self.vocabulary):
            for feature, count in signatures[token].items():
                digest = hashlib.blake2b(
                    feature.encode("utf-8"), digest_size=8
                ).digest()
                hashed = int.from_bytes(digest, "little")
                bucket = hashed % self.EMBEDDING_SIZE
                sign = 1.0 if hashed >> 63 else -1.0
                inverse_document_frequency = (
                    math.log(
                        (document_count + 1)
                        / (feature_documents[feature] + 1)
                    )
                    + 1.0
                )
                role_embedding[token_index, bucket] += (
                    sign * math.log1p(count) * inverse_document_frequency
                )
        norms = np.sqrt(np.sum(role_embedding * role_embedding, axis=1))
        role_embedding /= np.maximum(norms[:, None], 1e-9)
        clusters = self._stable_clusters(role_embedding)
        self.cluster = {
            token: int(clusters[index])
            for index, token in enumerate(self.vocabulary)
        }

    def _stable_clusters(
        self, role_embedding: np.ndarray, iterations: int = 10
    ) -> np.ndarray:
        """Deterministic Lloyd clustering without version-sensitive BLAS calls."""
        row_count, dimension_count = role_embedding.shape
        centers = np.empty(
            (self.CLASS_COUNT, dimension_count), dtype=np.float64
        )
        centers[0] = role_embedding[0]
        minimum_distance = np.sum(
            (role_embedding - centers[0]) ** 2, axis=1
        )
        for cluster_index in range(1, self.CLASS_COUNT):
            farthest = int(np.argmax(minimum_distance))
            centers[cluster_index] = role_embedding[farthest]
            distance = np.sum(
                (role_embedding - centers[cluster_index]) ** 2, axis=1
            )
            minimum_distance = np.minimum(minimum_distance, distance)

        labels = np.zeros(row_count, dtype=np.int32)
        batch_size = 512
        for _ in range(iterations):
            for start in range(0, row_count, batch_size):
                stop = min(row_count, start + batch_size)
                difference = (
                    role_embedding[start:stop, None, :]
                    - centers[None, :, :]
                )
                labels[start:stop] = np.argmin(
                    np.sum(difference * difference, axis=2), axis=1
                )
            new_centers = np.zeros_like(centers)
            counts = np.bincount(labels, minlength=self.CLASS_COUNT)
            np.add.at(new_centers, labels, role_embedding)
            populated = counts > 0
            new_centers[populated] /= counts[populated, None]
            new_centers[~populated] = centers[~populated]
            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if shift < 1e-8:
                break
        return labels


    def _fit_class_language_model(self) -> None:
        self.class_unigram: Counter[Any] = Counter()
        self.class_bigram: Counter[tuple[Any, Any]] = Counter()
        self.class_bigram_context: Counter[Any] = Counter()
        self.class_trigram: Counter[tuple[Any, Any, Any]] = Counter()
        self.class_trigram_context: Counter[tuple[Any, Any]] = Counter()

        def add(tokens: list[str], padded: bool) -> None:
            classes: list[Any] = [self.cluster[token] for token in tokens]
            if padded:
                classes = ["class_bos", "class_bos", *classes, "class_eos"]
            for token_class in classes[2:] if padded else classes:
                self.class_unigram[token_class] += 1
            for index in range(1, len(classes)):
                self.class_bigram[(classes[index - 1], classes[index])] += 1
                self.class_bigram_context[classes[index - 1]] += 1
            for index in range(2, len(classes)):
                key = (classes[index - 2], classes[index - 1], classes[index])
                self.class_trigram[key] += 1
                self.class_trigram_context[key[:2]] += 1

        for _, tokens in self.full_sequences:
            add(tokens, padded=True)
        for _, tokens in self.fragments:
            add(tokens, padded=False)
        self.class_total = sum(self.class_unigram.values())

    @lru_cache(maxsize=None)
    def _class_log_probability(self, first: Any, second: Any, third: Any) -> float:
        unigram_probability = (self.class_unigram[third] + 0.2) / (
            self.class_total + 0.2 * (self.CLASS_COUNT + 1)
        )
        bigram_probability = (
            self.class_bigram[(second, third)] + 5.0 * unigram_probability
        ) / (self.class_bigram_context[second] + 5.0)
        trigram_probability = (
            self.class_trigram[(first, second, third)]
            + 30.0 * bigram_probability
        ) / (self.class_trigram_context[(first, second)] + 30.0)
        return math.log(trigram_probability)

    @staticmethod
    def _class_token_features(
        part_of_speech: str,
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> dict[str, float]:
        first, second = left
        third, fourth = right
        return {
            "bias": 1.0,
            "pos=" + part_of_speech: 1.0,
            "left1=" + str(first): 1.0,
            "left0=" + str(second): 1.0,
            "right0=" + str(third): 1.0,
            "right1=" + str(fourth): 1.0,
            f"cross2={second}|{third}": 1.0,
            f"cross3a={first}|{second}|{third}": 1.0,
            f"cross3b={second}|{third}|{fourth}": 1.0,
            f"cross4={first}|{second}|{third}|{fourth}": 1.0,
        }

    def _fit_class_token_adjacency(self) -> None:
        windows: dict[
            str, list[tuple[tuple[int, int], tuple[int, int]]]
        ] = defaultdict(list)
        for part_of_speech, tokens in self.full_sequences + self.fragments:
            classes = [self.cluster[token] for token in tokens]
            for split in range(2, len(classes) - 1):
                windows[part_of_speech].append(
                    (
                        (classes[split - 2], classes[split - 1]),
                        (classes[split], classes[split + 1]),
                    )
                )
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for part_of_speech, candidates in windows.items():
            candidate_count = len(candidates)
            for index, (left, right) in enumerate(candidates):
                features.append(
                    self._class_token_features(part_of_speech, left, right)
                )
                labels.append(1)
                for shift in (1009, 3001):
                    mismatched_right = candidates[
                        (index * 37 + shift) % candidate_count
                    ][1]
                    features.append(
                        self._class_token_features(
                            part_of_speech, left, mismatched_right
                        )
                    )
                    labels.append(0)
        self.class_token_vectorizer = DictVectorizer()
        matrix = self.class_token_vectorizer.fit_transform(features)
        self.class_token_model = LogisticRegression(
            C=1.0,
            max_iter=100,
            solver="liblinear",
            random_state=20260715,
        ).fit(matrix, labels)

    def _class_fragment_features(
        self,
        row: dict[str, Any],
        left_card: dict[str, Any],
        right_card: dict[str, Any],
    ) -> dict[str, float]:
        left = [self.cluster[token] for token in left_card["tokens"]]
        right = [self.cluster[token] for token in right_card["tokens"]]
        features: dict[str, float] = {
            "bias": 1.0,
            "pos=" + row["pos"]: 1.0,
            "card_count=" + str(len(row["cards"])): 1.0,
            "left_length": float(len(left)),
            "right_length": float(len(right)),
            f"cross2={left[-1]}|{right[0]}": 1.0,
            f"cross3a={left[-2]}|{left[-1]}|{right[0]}": 1.0,
            f"cross3b={left[-1]}|{right[0]}|{right[1]}": 1.0,
            f"cross4={left[-2]}|{left[-1]}|{right[0]}|{right[1]}": 1.0,
        }
        for offset in range(min(3, len(left))):
            features[f"left_end{offset}={left[-1 - offset]}"] = 1.0
        for offset in range(min(3, len(right))):
            features[f"right_start{offset}={right[offset]}"] = 1.0
        for token_class in left:
            key = "left_class=" + str(token_class)
            features[key] = features.get(key, 0.0) + 1.0
        for token_class in right:
            key = "right_class=" + str(token_class)
            features[key] = features.get(key, 0.0) + 1.0
        return features

    def _fit_class_fragment_adjacency(
        self, train_rows: list[dict[str, Any]]
    ) -> None:
        features: list[dict[str, float]] = []
        labels: list[int] = []
        for row in train_rows:
            true_position = {
                fragment_id: index
                for index, fragment_id in enumerate(row["order"])
            }
            for left_index, left_card in enumerate(row["cards"]):
                for right_index, right_card in enumerate(row["cards"]):
                    if left_index == right_index:
                        continue
                    features.append(
                        self._class_fragment_features(row, left_card, right_card)
                    )
                    labels.append(
                        int(
                            true_position[right_card["fragment_id"]]
                            == true_position[left_card["fragment_id"]] + 1
                        )
                    )
        self.class_fragment_vectorizer = DictVectorizer()
        matrix = self.class_fragment_vectorizer.fit_transform(features)
        self.class_fragment_model = LogisticRegression(
            C=0.03,
            max_iter=100,
            solver="liblinear",
            random_state=20260715,
        ).fit(matrix, labels)


    def score_matrices(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cards = row["cards"]
        card_count = len(cards)
        classes = [
            [self.cluster[token] for token in card["tokens"]]
            for card in cards
        ]
        class_start = np.asarray(
            [
                self._class_log_probability(
                    "class_bos", "class_bos", token_classes[0]
                )
                + self._class_log_probability(
                    "class_bos", token_classes[0], token_classes[1]
                )
                for token_classes in classes
            ]
        )
        class_end = np.asarray(
            [
                self._class_log_probability(
                    token_classes[-2], token_classes[-1], "class_eos"
                )
                for token_classes in classes
            ]
        )
        class_edge = np.full((card_count, card_count), -np.inf)
        class_token_features: list[dict[str, float]] = []
        class_fragment_features: list[dict[str, float]] = []
        pairs: list[tuple[int, int]] = []
        for left_index, left_card in enumerate(cards):
            for right_index, right_card in enumerate(cards):
                if left_index == right_index:
                    continue
                left_classes = classes[left_index]
                right_classes = classes[right_index]
                class_edge[left_index, right_index] = (
                    self._class_log_probability(
                        left_classes[-2], left_classes[-1], right_classes[0]
                    )
                    + self._class_log_probability(
                        left_classes[-1], right_classes[0], right_classes[1]
                    )
                )
                class_token_features.append(
                    self._class_token_features(
                        row["pos"],
                        (left_classes[-2], left_classes[-1]),
                        (right_classes[0], right_classes[1]),
                    )
                )
                class_fragment_features.append(
                    self._class_fragment_features(row, left_card, right_card)
                )
                pairs.append((left_index, right_index))

        class_token_scores = self.class_token_model.decision_function(
            self.class_token_vectorizer.transform(class_token_features)
        )
        class_fragment_scores = self.class_fragment_model.decision_function(
            self.class_fragment_vectorizer.transform(class_fragment_features)
        )
        class_token = np.zeros((card_count, card_count), dtype=np.float64)
        class_fragment = np.zeros((card_count, card_count), dtype=np.float64)
        for index, (left_index, right_index) in enumerate(pairs):
            class_token[left_index, right_index] = class_token_scores[index]
            class_fragment[left_index, right_index] = class_fragment_scores[index]
        return (
            class_start,
            class_end,
            class_edge,
            class_token,
            class_fragment,
        )


class StructuralTokenModels:
    """Longer-range token relations and learned delimiter invariants."""

    DISTANCE_HORIZON = 8

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        target_rows: list[dict[str, Any]],
        extra_sequences: list[tuple[str, list[str]]],
    ) -> None:
        authentic = [
            (row["pos"], ordered_tokens(row)) for row in train_rows
        ]
        fragments = [
            (row["pos"], card["tokens"])
            for row in train_rows + target_rows
            for card in row["cards"]
        ]
        self.all_sequences = authentic + list(extra_sequences) + fragments
        self.delimiter_pairs = self._discover_delimiters(
            [tokens for _, tokens in authentic]
        )
        self.precedence_count: Counter[tuple[str, str]] = Counter()
        self.distance_count: Counter[tuple[str, str, int]] = Counter()
        self.pair_count: Counter[tuple[str, str]] = Counter()
        self.global_distance: Counter[int] = Counter()
        for _, tokens in self.all_sequences:
            for left_index, left_token in enumerate(tokens):
                for right_index in range(left_index + 1, len(tokens)):
                    right_token = tokens[right_index]
                    if left_token != right_token:
                        self.precedence_count[
                            (left_token, right_token)
                        ] += 1
                    distance = right_index - left_index
                    if distance <= self.DISTANCE_HORIZON:
                        self.distance_count[
                            (left_token, right_token, distance)
                        ] += 1
                        self.pair_count[(left_token, right_token)] += 1
                        self.global_distance[distance] += 1
        self.global_distance_total = sum(self.global_distance.values())

    @staticmethod
    def _discover_delimiters(
        definitions: list[list[str]],
    ) -> list[tuple[str, str]]:
        frequency: Counter[str] = Counter(
            token for tokens in definitions for token in tokens
        )
        signatures: dict[
            tuple[tuple[int, int], ...], list[str]
        ] = defaultdict(list)
        for token, count in frequency.items():
            if count < 20:
                continue
            signature = tuple(
                (row_index, tokens.count(token))
                for row_index, tokens in enumerate(definitions)
                if token in tokens
            )
            signatures[signature].append(token)

        result: list[tuple[str, str]] = []
        for signature, aliases in signatures.items():
            support = sum(count for _, count in signature)
            if support < 20 or len(aliases) < 2:
                continue
            for opening in sorted(aliases):
                for closing in sorted(aliases):
                    if opening == closing:
                        continue
                    valid = True
                    for tokens in definitions:
                        balance = 0
                        for token in tokens:
                            balance += int(token == opening)
                            balance -= int(token == closing)
                            if balance < 0:
                                valid = False
                                break
                        if not valid or balance != 0:
                            valid = False
                            break
                    if valid:
                        result.append((opening, closing))
                        break
                if result and result[-1][0] == opening:
                    break
        return result

    def score_matrices(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        cards = row["cards"]
        card_count = len(cards)
        precedence = np.zeros(
            (card_count, card_count), dtype=np.float64
        )
        distance_score = np.zeros(
            (card_count, card_count), dtype=np.float64
        )
        for left_index, left_card in enumerate(cards):
            left_tokens = left_card["tokens"]
            for right_index, right_card in enumerate(cards):
                if left_index == right_index:
                    continue
                right_tokens = right_card["tokens"]
                precedence_value = 0.0
                for left_token in left_tokens:
                    for right_token in right_tokens:
                        if left_token == right_token:
                            continue
                        forward = self.precedence_count[
                            (left_token, right_token)
                        ]
                        backward = self.precedence_count[
                            (right_token, left_token)
                        ]
                        if forward + backward:
                            log_odds = math.log(
                                (forward + 2.0) / (backward + 2.0)
                            )
                            precedence_value += max(
                                -3.0, min(3.0, log_odds)
                            )
                precedence[left_index, right_index] = (
                    precedence_value
                    / math.sqrt(len(left_tokens) * len(right_tokens))
                )

                value = 0.0
                used = 0
                left_length = len(left_tokens)
                for token_index, left_token in enumerate(left_tokens):
                    for right_offset, right_token in enumerate(
                        right_tokens
                    ):
                        distance = left_length - token_index + right_offset
                        pair_total = self.pair_count[
                            (left_token, right_token)
                        ]
                        if (
                            distance > self.DISTANCE_HORIZON
                            or pair_total < 2
                        ):
                            continue
                        observed = self.distance_count[
                            (left_token, right_token, distance)
                        ]
                        conditional = (observed + 0.5) / (
                            pair_total
                            + 0.5 * self.DISTANCE_HORIZON
                        )
                        background = (
                            self.global_distance[distance]
                            / self.global_distance_total
                        )
                        value += math.log(conditional / background)
                        used += 1
                distance_score[left_index, right_index] = (
                    value / math.sqrt(max(1, used))
                )
        return precedence, distance_score

    def delimiter_stats(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        pair_count = len(self.delimiter_pairs)
        card_count = len(row["cards"])
        net = np.zeros((pair_count, card_count), dtype=np.int16)
        minimum = np.zeros((pair_count, card_count), dtype=np.int16)
        for pair_index, (opening, closing) in enumerate(
            self.delimiter_pairs
        ):
            for card_index, card in enumerate(row["cards"]):
                balance = 0
                lowest = 0
                for token in card["tokens"]:
                    balance += int(token == opening)
                    balance -= int(token == closing)
                    lowest = min(lowest, balance)
                net[pair_index, card_index] = balance
                minimum[pair_index, card_index] = lowest
        return net, minimum


class TinyDefinitionLM(torch.nn.Module):
    """Small tied-embedding GRU trained only on supplied token sequences."""

    def __init__(
        self, vocabulary_size: int, dimension: int = 48
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocabulary_size, dimension, padding_idx=0
        )
        self.gru = torch.nn.GRU(
            dimension, dimension, batch_first=True
        )
        self.bias = torch.nn.Parameter(torch.zeros(vocabulary_size))


    def forward(
        self, tokens: torch.Tensor, hidden: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, hidden = self.gru(self.embedding(tokens), hidden)
        logits = torch.nn.functional.linear(
            encoded, self.embedding.weight, self.bias
        )
        return logits, hidden


class CardOrderNetwork(torch.nn.Module):
    """Multitask card encoder for position, adjacency, and precedence."""

    POS_CODES = ("n", "v", "a", "s", "r")

    def __init__(
        self, vocabulary_size: int, initial_embedding: torch.Tensor
    ) -> None:
        super().__init__()
        embedding_dimension = initial_embedding.shape[1]
        self.embedding = torch.nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=0,
        )
        self.embedding.weight.data.copy_(initial_embedding)
        self.gru = torch.nn.GRU(
            embedding_dimension,
            48,
            batch_first=True,
            bidirectional=True,
        )
        self.pos_embedding = torch.nn.Embedding(5, 8)
        self.count_embedding = torch.nn.Embedding(8, 8)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(304, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 96),
            torch.nn.ReLU(),
        )
        self.position_head = torch.nn.Linear(96, 7)
        self.adjacency_head = torch.nn.Sequential(
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 1),
        )
        self.precedence_head = torch.nn.Sequential(
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 1),
        )

    def encode(
        self,
        padded: torch.Tensor,
        lengths: torch.Tensor,
        part_of_speech: torch.Tensor,
        card_count: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _ = self.gru(self.embedding(padded))
        mask = (
            torch.arange(padded.shape[1])[None, :]
            < lengths[:, None]
        )
        mean = (
            encoded * mask[:, :, None]
        ).sum(dim=1) / lengths[:, None]
        maximum = encoded.masked_fill(
            ~mask[:, :, None], -1e9
        ).max(dim=1).values
        last = encoded[
            torch.arange(len(lengths)), lengths - 1
        ]
        return self.projection(
            torch.cat(
                [
                    last,
                    mean,
                    maximum,
                    self.pos_embedding(part_of_speech),
                    self.count_embedding(card_count),
                ],
                dim=1,
            )
        )


    @staticmethod
    def pair_features(
        representation: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        left_representation = representation[left]
        right_representation = representation[right]
        return torch.cat(
            [
                left_representation,
                right_representation,
                left_representation * right_representation,
                left_representation - right_representation,
            ],
            dim=1,
        )


class NeuralSequenceModels:
    """CPU-only neural language and card-order models."""

    PAD = 0
    BEGIN = 1
    END = 2

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        target_rows: list[dict[str, Any]],
        extra_sequences: list[tuple[str, list[str]]],
    ) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        full_sequences = [
            (row["pos"], ordered_tokens(row)) for row in train_rows
        ] + list(extra_sequences)
        fragments = [
            (row["pos"], card["tokens"])
            for row in train_rows + target_rows
            for card in row["cards"]
        ]
        frequency: Counter[str] = Counter(
            token
            for _, tokens in full_sequences + fragments
            for token in tokens
        )
        vocabulary = sorted(
            frequency,
            key=lambda token: (-frequency[token], token),
        )[:2048]
        retained = set(vocabulary)
        self.token_id = {
            token: index + 4
            for index, token in enumerate(vocabulary)
        }
        self.token_id.update(
            {
                token: 3
                for token in frequency
                if token not in retained
            }
        )
        authentic_sequences = [
            [
                self.BEGIN,
                *(self.token_id[token] for token in tokens),
                self.END,
            ]
            for _, tokens in full_sequences
        ]
        fragment_sequences = [
            [self.token_id[token] for token in tokens]
            for _, tokens in fragments
            if len(tokens) >= 2
        ]
        sequences = authentic_sequences * 3 + fragment_sequences
        backward_sequences = [
            [self.BEGIN, *reversed(sequence[1:-1]), self.END]
            for sequence in authentic_sequences * 3
        ]
        backward_sequences.extend(
            list(reversed(sequence))
            for sequence in fragment_sequences
        )
        vocabulary_size = len(vocabulary) + 4
        self.forward_lm = self._train_language_model(
            sequences, vocabulary_size, epochs=8, seed=20260715
        )
        print(
            f"trained forward neural LM in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.backward_lm = self._train_language_model(
            backward_sequences,
            vocabulary_size,
            epochs=6,
            seed=20260716,
        )
        print(
            f"trained backward neural LM in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.card_model = self._train_card_model(
            train_rows
            + synthetic_fragment_rows(train_rows, seed=20260717, copies=2),
            vocabulary_size,
            seed=20260715,
        )
        print(
            f"trained neural card ranker in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.score_cache: dict[str, tuple[np.ndarray, ...]] = {}

    def _train_language_model(
        self,
        sequences: list[list[int]],
        vocabulary_size: int,
        epochs: int,
        seed: int,
    ) -> TinyDefinitionLM:
        torch.manual_seed(seed)
        model = TinyDefinitionLM(vocabulary_size)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.003, weight_decay=1e-5
        )
        loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=self.PAD
        )
        batch_size = 256
        ordered = sorted(
            range(len(sequences)), key=lambda index: len(sequences[index])
        )
        batches = [
            ordered[start : start + batch_size]
            for start in range(0, len(ordered), batch_size)
        ]
        for epoch in range(epochs):
            generator = torch.Generator().manual_seed(seed + epoch)
            batch_order = torch.randperm(
                len(batches), generator=generator
            ).tolist()
            model.train()
            for batch_index in batch_order:
                batch = [
                    torch.tensor(sequences[index], dtype=torch.long)
                    for index in batches[batch_index]
                ]
                padded = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=self.PAD
                )
                inputs = padded[:, :-1]
                targets = padded[:, 1:]
                optimizer.zero_grad()
                logits, _ = model(inputs)
                loss = loss_function(
                    logits.reshape(-1, vocabulary_size),
                    targets.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
        return model

    def _train_card_model(
        self,
        rows: list[dict[str, Any]],
        vocabulary_size: int,
        seed: int,
    ) -> CardOrderNetwork:
        torch.manual_seed(seed)
        model = CardOrderNetwork(
            vocabulary_size,
            self.forward_lm.embedding.weight.detach().clone(),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=1e-4
        )
        position_loss = torch.nn.CrossEntropyLoss()
        adjacency_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(4.5)
        )
        precedence_loss = torch.nn.BCEWithLogitsLoss()
        pos_index = {
            code: index
            for index, code in enumerate(CardOrderNetwork.POS_CODES)
        }
        batch_size = 48
        for epoch in range(8):
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(
                len(rows), generator=generator
            ).tolist()
            model.train()
            for start in range(0, len(rows), batch_size):
                batch_rows = [
                    rows[index]
                    for index in order[start : start + batch_size]
                ]
                sequences: list[torch.Tensor] = []
                part_of_speech: list[int] = []
                card_counts: list[int] = []
                position_targets: list[int] = []
                left_indices: list[int] = []
                right_indices: list[int] = []
                adjacency_targets: list[int] = []
                precedence_targets: list[int] = []
                offset = 0
                for row in batch_rows:
                    true_position = {
                        fragment_id: index
                        for index, fragment_id in enumerate(row["order"])
                    }
                    card_count = len(row["cards"])
                    for card in row["cards"]:
                        sequences.append(
                            torch.tensor(
                                [
                                    self.token_id[token]
                                    for token in card["tokens"]
                                ],
                                dtype=torch.long,
                            )
                        )
                        part_of_speech.append(pos_index[row["pos"]])
                        card_counts.append(card_count)
                        position_targets.append(
                            true_position[card["fragment_id"]]
                        )
                    for left_index, left_card in enumerate(row["cards"]):
                        for right_index, right_card in enumerate(
                            row["cards"]
                        ):
                            if left_index == right_index:
                                continue
                            left_indices.append(offset + left_index)
                            right_indices.append(offset + right_index)
                            adjacency_targets.append(
                                int(
                                    true_position[
                                        right_card["fragment_id"]
                                    ]
                                    == true_position[
                                        left_card["fragment_id"]
                                    ]
                                    + 1
                                )
                            )
                            precedence_targets.append(
                                int(
                                    true_position[
                                        left_card["fragment_id"]
                                    ]
                                    < true_position[
                                        right_card["fragment_id"]
                                    ]
                                )
                            )
                    offset += card_count
                lengths = torch.tensor(
                    [len(sequence) for sequence in sequences]
                )
                padded = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=self.PAD,
                )
                representation = model.encode(
                    padded,
                    lengths,
                    torch.tensor(part_of_speech),
                    torch.tensor(card_counts),
                )
                left = torch.tensor(left_indices)
                right = torch.tensor(right_indices)
                pair_features = model.pair_features(
                    representation, left, right
                )
                position_logits = model.position_head(representation)
                adjacency_logits = model.adjacency_head(
                    pair_features
                ).squeeze(1)
                precedence_logits = model.precedence_head(
                    pair_features
                ).squeeze(1)
                loss = position_loss(
                    position_logits, torch.tensor(position_targets)
                )
                loss += 0.7 * adjacency_loss(
                    adjacency_logits,
                    torch.tensor(
                        adjacency_targets, dtype=torch.float32
                    ),
                )
                loss += 0.4 * precedence_loss(
                    precedence_logits,
                    torch.tensor(
                        precedence_targets, dtype=torch.float32
                    ),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2.0
                )
                optimizer.step()
        return model

    def _language_scores(
        self,
        model: TinyDefinitionLM,
        row: dict[str, Any],
        reverse: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cards = row["cards"]
        sequences = [
            [self.token_id[token] for token in card["tokens"]]
            for card in cards
        ]
        if reverse:
            sequences = [list(reversed(sequence)) for sequence in sequences]
        card_count = len(cards)
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.long,
        )
        padded = torch.nn.utils.rnn.pad_sequence(
            [
                torch.tensor(sequence, dtype=torch.long)
                for sequence in sequences
            ],
            batch_first=True,
            padding_value=self.PAD,
        )
        model.eval()
        with torch.no_grad():
            card_output, _ = model.gru(model.embedding(padded))
            row_indices = torch.arange(card_count)
            final_output = card_output[row_indices, lengths - 1]
            final_logits = torch.nn.functional.linear(
                final_output, model.embedding.weight, model.bias
            )
            final_log_probability = torch.log_softmax(
                final_logits, dim=1
            )
            end = final_log_probability[:, self.END].numpy()

            begin_tokens = torch.full(
                (card_count, 1), self.BEGIN, dtype=torch.long
            )
            begin_output, begin_hidden = model.gru(
                model.embedding(begin_tokens)
            )
            begin_logits = torch.nn.functional.linear(
                begin_output[:, 0],
                model.embedding.weight,
                model.bias,
            )
            begin_log_probability = torch.log_softmax(
                begin_logits, dim=1
            )
            first_tokens = torch.tensor(
                [[sequence[0]] for sequence in sequences],
                dtype=torch.long,
            )
            first_output, _ = model.gru(
                model.embedding(first_tokens), begin_hidden
            )
            first_logits = torch.nn.functional.linear(
                first_output[:, 0],
                model.embedding.weight,
                model.bias,
            )
            first_log_probability = torch.log_softmax(
                first_logits, dim=1
            )
            start = np.asarray(
                [
                    begin_log_probability[
                        card_index, sequence[0]
                    ].item()
                    + (
                        first_log_probability[
                            card_index, sequence[1]
                        ].item()
                        if len(sequence) > 1
                        else 0.0
                    )
                    for card_index, sequence in enumerate(sequences)
                ],
                dtype=np.float64,
            )

            edge = np.zeros(
                (card_count, card_count), dtype=np.float64
            )
            second_left: list[int] = []
            second_right: list[int] = []
            for left_index, left_sequence in enumerate(sequences):
                for right_index, right_sequence in enumerate(sequences):
                    if left_index == right_index:
                        continue
                    edge[left_index, right_index] = (
                        final_log_probability[
                            left_index, right_sequence[0]
                        ].item()
                    )
                    if len(right_sequence) > 1:
                        second_left.append(left_index)
                        second_right.append(right_index)
            if second_left:
                right_first = torch.tensor(
                    [
                        [sequences[right_index][0]]
                        for right_index in second_right
                    ],
                    dtype=torch.long,
                )
                contextual_output, _ = model.gru(
                    model.embedding(right_first),
                    final_output[second_left].unsqueeze(0),
                )
                contextual_logits = torch.nn.functional.linear(
                    contextual_output[:, 0],
                    model.embedding.weight,
                    model.bias,
                )
                contextual_log_probability = torch.log_softmax(
                    contextual_logits, dim=1
                )
                for pair_index, (
                    left_index,
                    right_index,
                ) in enumerate(zip(second_left, second_right)):
                    edge[left_index, right_index] += (
                        contextual_log_probability[
                            pair_index, sequences[right_index][1]
                        ].item()
                    )
        if reverse:
            return end, start, edge.T
        return start, end, edge

    def _card_scores(
        self,
        model: CardOrderNetwork,
        padded: torch.Tensor,
        lengths: torch.Tensor,
        pos_index: int,
        card_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            representation = model.encode(
                padded,
                lengths,
                torch.full((card_count,), pos_index, dtype=torch.long),
                torch.full((card_count,), card_count, dtype=torch.long),
            )
            position = torch.log_softmax(
                model.position_head(representation)[:, :card_count],
                dim=1,
            ).numpy()
            left_indices: list[int] = []
            right_indices: list[int] = []
            for left_index in range(card_count):
                for right_index in range(card_count):
                    if left_index != right_index:
                        left_indices.append(left_index)
                        right_indices.append(right_index)
            pair_features = model.pair_features(
                representation,
                torch.tensor(left_indices),
                torch.tensor(right_indices),
            )
            logits = model.adjacency_head(pair_features).squeeze(1).numpy()
        adjacency = np.zeros(
            (card_count, card_count), dtype=np.float64
        )
        for score, left_index, right_index in zip(
            logits, left_indices, right_indices
        ):
            adjacency[left_index, right_index] = score
        return position, adjacency

    def score_matrices(
        self, row: dict[str, Any]
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        cached = self.score_cache.get(row["id"])
        if cached is not None:
            return cached
        forward_start, forward_end, forward_edge = (
            self._language_scores(self.forward_lm, row, reverse=False)
        )
        backward_start, backward_end, backward_edge = (
            self._language_scores(self.backward_lm, row, reverse=True)
        )

        cards = row["cards"]
        card_count = len(cards)
        sequences = [
            torch.tensor(
                [self.token_id[token] for token in card["tokens"]],
                dtype=torch.long,
            )
            for card in cards
        ]
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences]
        )
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.PAD
        )
        pos_index = CardOrderNetwork.POS_CODES.index(row["pos"])
        position, adjacency = self._card_scores(
            self.card_model, padded, lengths, pos_index, card_count
        )
        result = (
            forward_start,
            forward_end,
            forward_edge,
            backward_start,
            backward_end,
            backward_edge,
            position,
            adjacency,
        )
        self.score_cache[row["id"]] = result
        return result


class OrderingEnsemble:
    """Combines the models and solves the exact constrained permutation."""

    POSITION_WEIGHT = 9.0
    FRAGMENT_ADJACENCY_WEIGHT = 2.0
    PRECEDENCE_WEIGHT = 0.25
    TOKEN_ADJACENCY_WEIGHT = 3.0
    CLASS_LANGUAGE_WEIGHT = 0.25
    CLASS_TOKEN_ADJACENCY_WEIGHT = 1.0
    CLASS_FRAGMENT_ADJACENCY_WEIGHT = 4.0
    NEURAL_POSITION_WEIGHT = 0.4
    NEURAL_FRAGMENT_ADJACENCY_WEIGHT = 0.1
    STRUCTURAL_PRECEDENCE_WEIGHT = 8.0
    STRUCTURAL_DISTANCE_WEIGHT = 4.0
    NEURAL_FORWARD_LANGUAGE_WEIGHT = 2.5
    NEURAL_BACKWARD_LANGUAGE_WEIGHT = 1.5
    EXACT_LANGUAGE_WEIGHT = 1.5

    def __init__(
        self, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
    ) -> None:
        # Parent definitions are authentic, ordered inputs. Using all of them is
        # transductive feature extraction, not lookup or target-answer access.
        extra_sequences = (
            parent_sequences(train_rows) + parent_sequences(test_rows)
        )
        started = time.perf_counter()
        self.language_model = TrigramLanguageModel(train_rows, extra_sequences)
        self.position_models = PositionModels(train_rows)
        augmented_rows = train_rows + synthetic_fragment_rows(
            train_rows, seed=20260720, copies=2
        )
        self.fragment_adjacency = FragmentAdjacencyModel(augmented_rows)
        self.precedence = PrecedenceModel(augmented_rows)
        self.token_adjacency = TokenAdjacencyModel(train_rows, extra_sequences)
        print(
            f"trained sparse models in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.latent_roles = LatentRoleModels(
            train_rows, test_rows, extra_sequences
        )
        print(
            f"trained latent-role models in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.structural_tokens = StructuralTokenModels(
            train_rows, test_rows, extra_sequences
        )
        print(
            f"trained structural models in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        self.neural_sequences = NeuralSequenceModels(
            train_rows, test_rows, extra_sequences
        )
        self.row_score_cache: dict[str, dict[str, np.ndarray]] = {}

    def _score_row(
        self, row: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        cached = self.row_score_cache.get(row["id"])
        if cached is not None:
            return cached
        cards = row["cards"]
        card_count = len(cards)
        position = self.position_models.score_matrix(row)
        fragment_adj = self.fragment_adjacency.score_matrix(row)
        precedence = self.precedence.score_matrix(row)
        token_adj = self.token_adjacency.score_matrix(row)
        structural_precedence, structural_distance = (
            self.structural_tokens.score_matrices(row)
        )
        (
            neural_forward_start,
            neural_forward_end,
            neural_forward_edge,
            neural_backward_start,
            neural_backward_end,
            neural_backward_edge,
            neural_position,
            neural_fragment_adj,
        ) = self.neural_sequences.score_matrices(row)
        (
            class_start,
            class_end,
            class_edge,
            class_token_adj,
            class_fragment_adj,
        ) = self.latent_roles.score_matrices(row)
        start = np.asarray(
            [
                self.language_model.start_score(card["tokens"])
                for card in cards
            ],
            dtype=np.float64,
        )
        end = np.asarray(
            [
                self.language_model.end_score(card["tokens"])
                for card in cards
            ],
            dtype=np.float64,
        )
        language_edge = np.full(
            (card_count, card_count), -np.inf, dtype=np.float64
        )
        for left_index, left_card in enumerate(cards):
            for right_index, right_card in enumerate(cards):
                if left_index == right_index:
                    continue
                language_edge[left_index, right_index] = (
                    self.language_model.edge_score(
                        left_card["tokens"], right_card["tokens"]
                    )
                )
        delimiter_net, delimiter_minimum = (
            self.structural_tokens.delimiter_stats(row)
        )
        result = {
            "position": position,
            "fragment_adj": fragment_adj,
            "precedence": precedence,
            "token_adj": token_adj,
            "structural_precedence": structural_precedence,
            "structural_distance": structural_distance,
            "neural_forward_start": neural_forward_start,
            "neural_forward_end": neural_forward_end,
            "neural_forward_edge": neural_forward_edge,
            "neural_backward_start": neural_backward_start,
            "neural_backward_end": neural_backward_end,
            "neural_backward_edge": neural_backward_edge,
            "neural_position": neural_position,
            "neural_fragment_adj": neural_fragment_adj,
            "class_start": class_start,
            "class_end": class_end,
            "class_edge": class_edge,
            "class_token_adj": class_token_adj,
            "class_fragment_adj": class_fragment_adj,
            "start": start,
            "end": end,
            "language_edge": language_edge,
            "delimiter_net": delimiter_net,
            "delimiter_minimum": delimiter_minimum,
        }
        self.row_score_cache[row["id"]] = result
        return result

    def predict(
        self, row: dict[str, Any], temperature: Any = None
    ) -> list[str]:
        cards = row["cards"]
        card_count = len(cards)
        matrices = self._score_row(row)
        position = (
            matrices["position"]
            + self.NEURAL_POSITION_WEIGHT
            * matrices["neural_position"]
        )
        fragment_adj = (
            matrices["fragment_adj"]
            + self.NEURAL_FRAGMENT_ADJACENCY_WEIGHT
            * matrices["neural_fragment_adj"]
        )
        precedence = (
            matrices["precedence"]
            + self.STRUCTURAL_PRECEDENCE_WEIGHT
            * matrices["structural_precedence"]
        )
        token_adj = matrices["token_adj"]
        class_start = matrices["class_start"]
        class_end = matrices["class_end"]
        class_edge = matrices["class_edge"]
        class_token_adj = matrices["class_token_adj"]
        class_fragment_adj = matrices["class_fragment_adj"]
        start = (
            self.EXACT_LANGUAGE_WEIGHT * matrices["start"]
            + self.NEURAL_FORWARD_LANGUAGE_WEIGHT
            * matrices["neural_forward_start"]
            + self.NEURAL_BACKWARD_LANGUAGE_WEIGHT
            * matrices["neural_backward_start"]
        )
        end = (
            self.EXACT_LANGUAGE_WEIGHT * matrices["end"]
            + self.NEURAL_FORWARD_LANGUAGE_WEIGHT
            * matrices["neural_forward_end"]
            + self.NEURAL_BACKWARD_LANGUAGE_WEIGHT
            * matrices["neural_backward_end"]
        )
        language_edge = (
            self.EXACT_LANGUAGE_WEIGHT * matrices["language_edge"]
            + self.NEURAL_FORWARD_LANGUAGE_WEIGHT
            * matrices["neural_forward_edge"]
            + self.NEURAL_BACKWARD_LANGUAGE_WEIGHT
            * matrices["neural_backward_edge"]
            + self.STRUCTURAL_DISTANCE_WEIGHT
            * matrices["structural_distance"]
        )
        delimiter_net = matrices["delimiter_net"]
        delimiter_minimum = matrices["delimiter_minimum"]
        class_language_weight = self.CLASS_LANGUAGE_WEIGHT
        class_token_weight = self.CLASS_TOKEN_ADJACENCY_WEIGHT
        class_fragment_weight = self.CLASS_FRAGMENT_ADJACENCY_WEIGHT
        active_delimiters = [
            pair_index
            for pair_index in range(len(delimiter_net))
            if int(delimiter_net[pair_index].sum()) == 0
        ]
        delimiter_net = delimiter_net[active_delimiters]
        delimiter_minimum = delimiter_minimum[active_delimiters]
        full_mask = (1 << card_count) - 1
        valid_append = np.ones(
            (full_mask + 1, card_count), dtype=bool
        )
        if len(delimiter_net):
            balances = np.zeros(
                (full_mask + 1, len(delimiter_net)), dtype=np.int16
            )
            for mask in range(1, full_mask + 1):
                least_bit = mask & -mask
                card_index = least_bit.bit_length() - 1
                balances[mask] = (
                    balances[mask ^ least_bit]
                    + delimiter_net[:, card_index]
                )
            for mask in range(full_mask + 1):
                for card_index in range(card_count):
                    valid_append[mask, card_index] = bool(
                        np.all(
                            balances[mask]
                            + delimiter_minimum[:, card_index]
                            >= 0
                        )
                    )

        fragment_adjacency_weight = (
            5.0
            if row["pos"] == "v"
            else self.FRAGMENT_ADJACENCY_WEIGHT
        )
        token_adjacency_weight = (
            4.0
            if row["pos"] == "v"
            else self.TOKEN_ADJACENCY_WEIGHT
        )

        def path_score(permutation: tuple[int, ...]) -> float:
            score = (
                start[permutation[0]]
                + class_language_weight
                * class_start[permutation[0]]
                + end[permutation[-1]]
                + class_language_weight
                * class_end[permutation[-1]]
            )
            score += self.POSITION_WEIGHT * sum(
                position[card_index, output_index]
                for output_index, card_index in enumerate(permutation)
            )
            for output_index in range(card_count - 1):
                left_index = permutation[output_index]
                right_index = permutation[output_index + 1]
                score += language_edge[left_index, right_index]
                score += (
                    fragment_adjacency_weight
                    * fragment_adj[left_index, right_index]
                )
                score += (
                    token_adjacency_weight
                    * token_adj[left_index, right_index]
                )
                score += (
                    class_language_weight
                    * class_edge[left_index, right_index]
                )
                score += (
                    class_token_weight
                    * class_token_adj[left_index, right_index]
                )
                score += (
                    class_fragment_weight
                    * class_fragment_adj[left_index, right_index]
                )
            score += self.PRECEDENCE_WEIGHT * sum(
                precedence[permutation[left], permutation[right]]
                for left in range(card_count)
                for right in range(left + 1, card_count)
            )
            return float(score)

        if temperature is not None:
            permutations = []
            for permutation in itertools.permutations(range(card_count)):
                mask = 0
                valid = True
                for card_index in permutation:
                    if not valid_append[mask, card_index]:
                        valid = False
                        break
                    mask |= 1 << card_index
                if valid:
                    permutations.append(permutation)
            scores = np.asarray(
                [path_score(permutation) for permutation in permutations],
                dtype=np.float64,
            )
            probabilities = np.exp(
                (scores - np.max(scores)) / float(temperature)
            )
            probabilities /= np.sum(probabilities)
            marginals = np.zeros(
                (card_count, card_count), dtype=np.float64
            )
            for probability, permutation in zip(
                probabilities, permutations
            ):
                for output_index, card_index in enumerate(permutation):
                    marginals[card_index, output_index] += probability
            expected_scores = np.empty(
                len(permutations), dtype=np.float64
            )
            for permutation_index, permutation in enumerate(permutations):
                expected_position = sum(
                    marginals[card_index, output_index]
                    for output_index, card_index in enumerate(permutation)
                ) / card_count
                expected_scores[permutation_index] = (
                    0.85 * expected_position
                    + 0.30 * probabilities[permutation_index]
                )
            best_path = permutations[int(np.argmax(expected_scores))]
            return [
                cards[index]["fragment_id"] for index in best_path
            ]

        # Held-Karp subset DP exactly optimizes the complete ensemble score.
        states: dict[
            tuple[int, int], tuple[float, tuple[int, ...]]
        ] = {}
        for card_index in range(card_count):
            if not valid_append[0, card_index]:
                continue
            states[(1 << card_index, card_index)] = (
                start[card_index]
                + class_language_weight * class_start[card_index]
                + self.POSITION_WEIGHT * position[card_index, 0],
                (card_index,),
            )
        for used_count in range(1, card_count):
            for mask in range(1, full_mask + 1):
                if bin(mask).count("1") != used_count:
                    continue
                for last_index in range(card_count):
                    state = states.get((mask, last_index))
                    if state is None:
                        continue
                    old_score, old_path = state
                    for next_index in range(card_count):
                        if (
                            mask & (1 << next_index)
                            or not valid_append[mask, next_index]
                        ):
                            continue
                        score = old_score
                        score += language_edge[last_index, next_index]
                        score += (
                            fragment_adjacency_weight
                            * fragment_adj[last_index, next_index]
                        )
                        score += (
                            token_adjacency_weight
                            * token_adj[last_index, next_index]
                        )
                        score += (
                            class_language_weight
                            * class_edge[last_index, next_index]
                        )
                        score += (
                            class_token_weight
                            * class_token_adj[last_index, next_index]
                        )
                        score += (
                            class_fragment_weight
                            * class_fragment_adj[last_index, next_index]
                        )
                        score += (
                            self.POSITION_WEIGHT
                            * position[next_index, used_count]
                        )
                        score += self.PRECEDENCE_WEIGHT * sum(
                            precedence[earlier_index, next_index]
                            for earlier_index in range(card_count)
                            if mask & (1 << earlier_index)
                        )
                        new_path = old_path + (next_index,)
                        key = (
                            mask | (1 << next_index),
                            next_index,
                        )
                        previous = states.get(key)
                        if (
                            previous is None
                            or score > previous[0] + 1e-12
                            or (
                                abs(score - previous[0]) <= 1e-12
                                and new_path < previous[1]
                            )
                        ):
                            states[key] = (score, new_path)
        candidates = []
        for last_index in range(card_count):
            state = states.get((full_mask, last_index))
            if state is None:
                continue
            candidates.append(
                (
                    state[0]
                    + end[last_index]
                    + class_language_weight
                    * class_end[last_index],
                    state[1],
                )
            )
        best_score = max(score for score, _ in candidates)
        best_path = min(
            path
            for score, path in candidates
            if abs(score - best_score) <= 1e-12
        )
        return [cards[index]["fragment_id"] for index in best_path]


def validate_prediction(row: dict[str, Any], order: list[str]) -> None:
    expected = [card["fragment_id"] for card in row["cards"]]
    if len(order) != len(expected) or set(order) != set(expected):
        raise RuntimeError(f"Invalid fragment permutation for row {row['id']}")


def lexical_family_groups(rows: list[dict[str, Any]]) -> np.ndarray:
    """Build non-overlapping family proxies from observable lexical aliases.

    The CSV does not expose the generator's private family ID. Stable aliases
    shared by target and direct-parent lemmas are the supplied evidence of
    lexical relatedness. Rare aliases are unioned into connected components;
    frequent aliases are excluded because generic words in multiword lemmas
    would incorrectly merge unrelated families.
    """
    lexical_sets: list[set[str]] = []
    document_frequency: Counter[str] = Counter()
    for row in rows:
        aliases = {token for lemma in row["lemmas"] for token in lemma}
        if row["parent"] is not None:
            aliases.update(
                token
                for lemma in row["parent"]["lemma_tokens"]
                for token in lemma
            )
        lexical_sets.append(aliases)
        document_frequency.update(aliases)

    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            lower, higher = sorted((left_root, right_root))
            parents[higher] = lower

    alias_rows: dict[str, list[int]] = defaultdict(list)
    for row_index, aliases in enumerate(lexical_sets):
        for alias in sorted(aliases):
            if document_frequency[alias] <= 2:
                alias_rows[alias].append(row_index)
    for alias in sorted(alias_rows):
        connected_rows = alias_rows[alias]
        for row_index in connected_rows[1:]:
            union(connected_rows[0], row_index)
    return np.asarray([root(index) for index in range(len(rows))])


def challenge_score(
    rows: list[dict[str, Any]], predictions: list[list[str]]
) -> tuple[float, float, float]:
    correct_positions = 0
    total_positions = 0
    complete_orders = 0
    for row, prediction in zip(rows, predictions):
        truth = row["order"]
        correct_positions += sum(
            predicted_id == true_id
            for predicted_id, true_id in zip(prediction, truth)
        )
        total_positions += len(truth)
        complete_orders += int(prediction == truth)
    position_accuracy = correct_positions / total_positions
    complete_accuracy = complete_orders / len(rows)
    score = 0.85 * position_accuracy + 0.15 * complete_accuracy
    return score, position_accuracy, complete_accuracy


def run_grouped_validation(
    rows: list[dict[str, Any]]
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], Any
]:
    """Run an effective-path holdout with lexical components kept intact."""
    groups = lexical_family_groups(rows)
    strata = np.asarray(
        [row["pos"] + str(len(row["cards"])) for row in rows]
    )
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=20260715
    )
    fold_index = int(os.environ.get("DFO_FOLD", "0"))
    fit_indices, validation_indices = list(
        splitter.split(np.zeros(len(rows)), strata, groups)
    )[fold_index]
    fit_rows = [rows[index] for index in fit_indices]
    validation_rows = [rows[index] for index in validation_indices]
    if set(groups[fit_indices]).intersection(groups[validation_indices]):
        raise RuntimeError("Lexical-family group leaked across validation split")

    validation_ensemble = OrderingEnsemble(fit_rows, validation_rows)
    candidates = (
        (0.0, 0.0, 0.0),
        (0.25, 1.0, 4.0),
        (0.375, 1.5, 6.0),
        (0.5, 2.0, 8.0),
        (0.625, 2.5, 10.0),
        (0.75, 3.0, 12.0),
        (1.0, 4.0, 16.0),
        (0.75, 4.0, 16.0),
        (0.5, 1.0, 12.0),
        (0.25, 3.0, 12.0),
        (0.25, 2.0, 6.0),
        (0.5, 1.0, 6.0),
    )
    best_metrics = (-math.inf, 0.0, 0.0)
    best_weights = candidates[0]
    for class_language, class_token, class_fragment in candidates:
        validation_ensemble.CLASS_LANGUAGE_WEIGHT = class_language
        validation_ensemble.CLASS_TOKEN_ADJACENCY_WEIGHT = class_token
        validation_ensemble.CLASS_FRAGMENT_ADJACENCY_WEIGHT = class_fragment
        predictions = [
            validation_ensemble.predict(row) for row in validation_rows
        ]
        metrics = challenge_score(validation_rows, predictions)
        print(
            "grouped candidate "
            f"weights={class_language:g},{class_token:g},{class_fragment:g} "
            f"score={metrics[0]:.6f}"
        )
        if metrics[0] > best_metrics[0]:
            best_metrics = metrics
            best_weights = (class_language, class_token, class_fragment)
    (
        validation_ensemble.CLASS_LANGUAGE_WEIGHT,
        validation_ensemble.CLASS_TOKEN_ADJACENCY_WEIGHT,
        validation_ensemble.CLASS_FRAGMENT_ADJACENCY_WEIGHT,
    ) = best_weights
    component_sweeps: tuple[tuple[str, tuple[float, ...]], ...] = ()
    for attribute, values in component_sweeps:
        component_best = (-math.inf, getattr(validation_ensemble, attribute))
        for value in values:
            setattr(validation_ensemble, attribute, value)
            predictions = [
                validation_ensemble.predict(row)
                for row in validation_rows
            ]
            metrics = challenge_score(validation_rows, predictions)
            print(
                f"grouped component {attribute}={value:g} "
                f"score={metrics[0]:.6f}"
            )
            if metrics[0] > component_best[0]:
                component_best = (metrics[0], value)
        setattr(validation_ensemble, attribute, component_best[1])
        print(
            f"selected component {attribute}={component_best[1]:g} "
            f"score={component_best[0]:.6f}"
        )
    predictions = [
        validation_ensemble.predict(row) for row in validation_rows
    ]
    best_metrics = challenge_score(validation_rows, predictions)
    print(
        "selected grouped validation "
        f"score={best_metrics[0]:.6f} "
        f"position={best_metrics[1]:.6f} "
        f"complete={best_metrics[2]:.6f} "
        f"weights={best_weights}"
    )
    del validation_ensemble
    gc.collect()
    return best_metrics, best_weights, None


def write_submission(
    path: Path, test_rows: list[dict[str, Any]], predictions: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer_json"])
        writer.writeheader()
        for row, order in zip(test_rows, predictions):
            validate_prediction(row, order)
            writer.writerow(
                {
                    "id": row["id"],
                    "answer_json": json.dumps(
                        {"fragment_order": order}, separators=(",", ":")
                    ),
                }
            )


def resolve_paths(arguments: list[str]) -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    if len(arguments) == 1:
        return script_dir / "dataset" / "public", script_dir / "working" / "submission.csv"
    if len(arguments) == 3:
        return Path(arguments[1]), Path(arguments[2])
    raise SystemExit("Usage: python3 solution.py [<public_dir> <submission_out>]")


def main() -> None:
    public_dir, submission_out = resolve_paths(sys.argv)
    train_rows = read_rows(public_dir / "train.csv", labeled=True)
    test_rows = read_rows(public_dir / "test.csv", labeled=False)

    latent_weights = (
        OrderingEnsemble.CLASS_LANGUAGE_WEIGHT,
        OrderingEnsemble.CLASS_TOKEN_ADJACENCY_WEIGHT,
        OrderingEnsemble.CLASS_FRAGMENT_ADJACENCY_WEIGHT,
    )
    temperature = None
    if os.environ.get("DFO_VALIDATE") == "1":
        validation, latent_weights, temperature = (
            run_grouped_validation(train_rows)
        )
        print(
            "selected grouped validation "
            f"score={validation[0]:.6f} "
            f"position={validation[1]:.6f} "
            f"complete={validation[2]:.6f} "
            f"weights={latent_weights} "
            f"temperature={temperature}"
        )

    # The submission path refits every component on all labeled rows.
    ensemble = OrderingEnsemble(train_rows, test_rows)
    (
        ensemble.CLASS_LANGUAGE_WEIGHT,
        ensemble.CLASS_TOKEN_ADJACENCY_WEIGHT,
        ensemble.CLASS_FRAGMENT_ADJACENCY_WEIGHT,
    ) = latent_weights
    predictions = [
        ensemble.predict(row, temperature=temperature)
        for row in test_rows
    ]
    write_submission(submission_out, test_rows, predictions)
    print(f"wrote {len(predictions)} predictions to {submission_out}")


if __name__ == "__main__":
    main()
