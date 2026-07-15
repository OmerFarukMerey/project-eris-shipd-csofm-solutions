from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "public"
OUTPUT = BASE / "working" / "submission.csv"
TEXT_COLUMNS = ["context_tokens", "draft_tokens", "reply_tokens"]
BASE_CATEGORICAL = [
    "prearrival_revision_band",
    "draft_age_band",
    "prearrival_pause_band",
    "draft_snapshot_band",
    "draft_extent_band",
]
INITIAL_REGULARIZATION = (0.07,)
INITIAL_TEST_REGULARIZATION = (0.03, 0.05, 0.07, 0.15, 0.20)
FINAL_REGULARIZATION = (0.05, 0.15, 0.30)
CV_SEEDS = (19, 43, 71, 101, 137)


def token_geometry(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for column in TEXT_COLUMNS:
        prefix = column.split("_")[0]
        tokens = df[column].str.split()
        result[f"{prefix}_ntok"] = tokens.str.len()
        result[f"{prefix}_nw"] = tokens.map(
            lambda values: sum(value.startswith("W_") for value in values)
        )
        result[f"{prefix}_nb"] = tokens.map(
            lambda values: sum(value.startswith("B_") for value in values)
        )
        result[f"{prefix}_uw"] = tokens.map(
            lambda values: len({value for value in values if value.startswith("W_")})
        )
        result[f"{prefix}_ub"] = tokens.map(
            lambda values: len({value for value in values if value.startswith("B_")})
        )
    return result


def add_reply_cluster_features(df: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat(
        [df.reset_index(drop=True), geometry.reset_index(drop=True)], axis=1
    )
    grouped = result.groupby("reply_tokens", sort=False)
    result["reply_group_count"] = grouped["turn_id"].transform("size")
    result["reply_group_draft_nunique"] = grouped["draft_tokens"].transform("nunique")

    for column in ("draft_nw", "context_nw"):
        result[f"{column}_group_min"] = grouped[column].transform("min")
        result[f"{column}_group_max"] = grouped[column].transform("max")
        result[f"{column}_group_mean"] = grouped[column].transform("mean")
        result[f"{column}_group_std"] = grouped[column].transform("std").fillna(0.0)
        result[f"{column}_from_min"] = result[column] - result[f"{column}_group_min"]
        result[f"{column}_from_max"] = result[f"{column}_group_max"] - result[column]
        result[f"{column}_rank_pct"] = grouped[column].rank(
            pct=True, method="average"
        )
    return result


def draft_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compare draft states attached to the same final reply, without labels."""
    result = pd.DataFrame(
        0.0,
        index=df.index,
        columns=[
            "draft_peer_max_jaccard",
            "draft_peer_max_own_contain",
            "draft_peer_max_other_contain",
            "draft_peer_union_share",
            "draft_peer_exact_count",
        ],
    )

    for indices in df.groupby("reply_tokens", sort=False).groups.values():
        row_indices = list(indices)
        if len(row_indices) < 2:
            continue
        word_sets = [
            {
                token
                for token in df.loc[index, "draft_tokens"].split()
                if token.startswith("W_")
            }
            for index in row_indices
        ]
        for position, index in enumerate(row_indices):
            current = word_sets[position]
            peers = [
                words
                for peer_position, words in enumerate(word_sets)
                if peer_position != position
            ]
            intersections = [len(current & peer) for peer in peers]
            result.loc[index, "draft_peer_max_jaccard"] = max(
                intersection / max(len(current | peer), 1)
                for intersection, peer in zip(intersections, peers)
            )
            result.loc[index, "draft_peer_max_own_contain"] = max(
                intersection / max(len(current), 1) for intersection in intersections
            )
            result.loc[index, "draft_peer_max_other_contain"] = max(
                intersection / max(len(peer), 1)
                for intersection, peer in zip(intersections, peers)
            )
            peer_union = set().union(*peers)
            result.loc[index, "draft_peer_union_share"] = (
                len(current & peer_union) / max(len(current), 1)
            )
            result.loc[index, "draft_peer_exact_count"] = sum(
                current == peer for peer in peers
            )
    return result


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.reset_index(drop=True).copy()
    result = add_reply_cluster_features(clean, token_geometry(clean))

    result["reply_minus_draft_w"] = result["reply_nw"] - result["draft_nw"]
    result["reply_draft_absdiff"] = result["reply_minus_draft_w"].abs()
    result["reply_over_draft_w"] = (result["reply_nw"] + 0.5) / (
        result["draft_nw"] + 0.5
    )
    result["draft_over_reply_w"] = (result["draft_nw"] + 0.5) / (
        result["reply_nw"] + 0.5
    )
    result["context_over_draft_w"] = (result["context_nw"] + 0.5) / (
        result["draft_nw"] + 0.5
    )
    result["context_over_reply_w"] = (result["context_nw"] + 0.5) / (
        result["reply_nw"] + 0.5
    )
    result["context_minus_draft_w"] = result["context_nw"] - result["draft_nw"]
    result["reply_shorter"] = (
        (result["reply_nw"] < result["draft_nw"]) & (result["draft_nw"] > 0)
    ).astype(int)
    result["reply_same_or_shorter"] = (
        (result["reply_nw"] <= result["draft_nw"]) & (result["draft_nw"] > 0)
    ).astype(int)
    result["draft_empty"] = (result["draft_nw"] == 0).astype(int)

    for prefix in ("context", "draft", "reply"):
        result[f"{prefix}_word_repeat"] = (
            result[f"{prefix}_nw"] - result[f"{prefix}_uw"]
        )
        result[f"{prefix}_bigram_repeat"] = (
            result[f"{prefix}_nb"] - result[f"{prefix}_ub"]
        )
        result[f"log_{prefix}_nw"] = np.log1p(result[f"{prefix}_nw"])

    result["extent_reply_relation"] = result["draft_extent_band"] + "_" + np.where(
        result["reply_nw"] < result["draft_nw"], "shorter", "not_shorter"
    )
    result["revision_extent"] = (
        result["prearrival_revision_band"] + "_" + result["draft_extent_band"]
    )
    result["age_pause"] = (
        result["draft_age_band"] + "_" + result["prearrival_pause_band"]
    )
    result["snap_extent"] = (
        result["draft_snapshot_band"] + "_" + result["draft_extent_band"]
    )
    result["draft_nw_bin"] = pd.cut(
        result["draft_nw"], [-1, 0, 1, 2, 3, 5, 8, 12, 100], labels=False
    ).astype(str)
    result["reply_nw_bin"] = pd.cut(
        result["reply_nw"], [-1, 0, 1, 2, 3, 5, 8, 12, 16, 100], labels=False
    ).astype(str)
    result["ratio_bin"] = pd.cut(
        result["reply_over_draft_w"],
        [0, 0.5, 0.8, 1, 1.2, 1.5, 2, 3, 5, 200],
        labels=False,
        include_lowest=True,
    ).astype(str)

    result = pd.concat([result, draft_peer_features(clean)], axis=1)
    result["peer_no_overlap"] = (
        (result["reply_group_count"] > 1)
        & (result["draft_peer_max_own_contain"] == 0)
        & (result["reply_nw"] > 2)
    ).astype(int)
    result["peer_has_overlap"] = (
        (result["reply_group_count"] > 1)
        & (result["draft_peer_max_own_contain"] > 0)
        & (result["reply_nw"] > 2)
    ).astype(int)
    result["peer_full_containment"] = (
        (result["reply_group_count"] > 1)
        & (result["draft_peer_max_own_contain"] == 1)
        & (result["reply_nw"] > 2)
    ).astype(int)
    result["peer_similarity_bin"] = pd.cut(
        result["draft_peer_max_own_contain"],
        [-0.1, 0, 0.25, 0.5, 0.75, 0.999, 1.01],
        labels=False,
    ).astype(str)
    return result


def protected_token_matrices(
    train: pd.DataFrame, test: pd.DataFrame, token_kind: str
) -> tuple[sparse.csr_matrix, ...]:
    """Build binary and count matrices in separate draft/reply code spaces."""
    draft_documents = [
        " ".join(
            token
            for token in text.split()
            if token.startswith(f"{token_kind}_")
        )
        for text in pd.concat(
            [train["draft_tokens"], test["draft_tokens"]], ignore_index=True
        )
    ]
    reply_documents = [
        " ".join(
            token
            for token in text.split()
            if token.startswith(f"{token_kind}_")
        )
        for text in pd.concat(
            [train["reply_tokens"], test["reply_tokens"]], ignore_index=True
        )
    ]
    draft_vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        binary=False,
    )
    reply_vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        binary=False,
    )
    draft_counts = draft_vectorizer.fit_transform(draft_documents).tocsr()
    reply_counts = reply_vectorizer.fit_transform(reply_documents).tocsr()
    draft_binary = draft_counts.copy()
    reply_binary = reply_counts.copy()
    draft_binary.data.fill(1)
    reply_binary.data.fill(1)
    n_train = len(train)
    return (
        draft_binary[:n_train],
        reply_binary[:n_train],
        draft_binary[n_train:],
        reply_binary[n_train:],
        draft_counts[:n_train],
        reply_counts[:n_train],
        draft_counts[n_train:],
        reply_counts[n_train:],
    )


def fit_cross_field_association(
    draft_matrix: sparse.csr_matrix,
    reply_matrix: sparse.csr_matrix,
    indices: np.ndarray,
) -> sparse.csr_matrix:
    """Induce an opaque draft-to-reply lexicon using only row co-occurrence."""
    association = (
        draft_matrix[indices].T @ reply_matrix[indices]
    ).tocsr().astype(float)
    draft_frequency = np.asarray(draft_matrix[indices].sum(axis=0)).ravel()
    reply_frequency = np.asarray(reply_matrix[indices].sum(axis=0)).ravel()
    rows, columns = association.nonzero()
    association.data /= np.sqrt(
        draft_frequency[rows] * reply_frequency[columns]
    )
    return association


def score_cross_field_association(
    association: sparse.csr_matrix,
    draft_matrix: sparse.csr_matrix,
    reply_matrix: sparse.csr_matrix,
    indices: np.ndarray,
) -> np.ndarray:
    """Measure how much of each draft has a learned counterpart in its reply."""
    output = np.zeros((len(indices), 11), dtype=float)
    global_best = np.asarray(association.max(axis=1).toarray()).ravel()
    top_one = np.full(association.shape[0], -1, dtype=int)
    top_three: list[set[int]] = []
    for draft_token in range(association.shape[0]):
        start, end = association.indptr[draft_token : draft_token + 2]
        if end == start:
            top_three.append(set())
            continue
        candidates = association.indices[start:end]
        values = association.data[start:end]
        order = np.argsort(values)[::-1]
        top_one[draft_token] = candidates[order[0]]
        top_three.append(set(candidates[order[:3]]))

    for position, row_index in enumerate(indices):
        draft_tokens = draft_matrix[row_index].indices
        reply_tokens = set(reply_matrix[row_index].indices)
        if len(draft_tokens) == 0:
            continue
        present_scores = []
        relative_scores = []
        top_one_hits = []
        top_three_hits = []
        confidences = []
        for draft_token in draft_tokens:
            start, end = association.indptr[draft_token : draft_token + 2]
            present = [
                value
                for candidate, value in zip(
                    association.indices[start:end], association.data[start:end]
                )
                if candidate in reply_tokens
            ]
            best_present = max(present, default=0.0)
            confidence = global_best[draft_token]
            present_scores.append(best_present)
            relative_scores.append(
                best_present / confidence if confidence > 0 else 0.0
            )
            top_one_hits.append(float(top_one[draft_token] in reply_tokens))
            top_three_hits.append(
                float(bool(top_three[draft_token] & reply_tokens))
            )
            confidences.append(confidence)

        present_array = np.asarray(present_scores)
        relative_array = np.asarray(relative_scores)
        top_one_array = np.asarray(top_one_hits)
        top_three_array = np.asarray(top_three_hits)
        confidence_array = np.asarray(confidences)
        covered = confidence_array > 0
        output[position] = [
            present_array.mean(),
            present_array.max(initial=0.0),
            relative_array.mean(),
            relative_array[covered].mean() if covered.any() else 0.0,
            top_one_array.mean(),
            top_one_array[covered].mean() if covered.any() else 0.0,
            top_three_array.mean(),
            top_three_array[covered].mean() if covered.any() else 0.0,
            covered.mean(),
            confidence_array.mean(),
            present_array.sum() / max(confidence_array.sum(), 1e-12),
        ]
    return output

def score_enhanced_association(
    association: sparse.csr_matrix,
    draft_matrix: sparse.csr_matrix,
    reply_matrix: sparse.csr_matrix,
    draft_counts: sparse.csr_matrix,
    reply_counts: sparse.csr_matrix,
    indices: np.ndarray,
    fit_draft_frequency: np.ndarray,
    fit_size: int,
) -> np.ndarray:
    """Summarize confidence-aware missing-content and occurrence-weighted overlap."""
    global_best = np.asarray(association.max(axis=1).toarray()).ravel()
    top_one = np.full(association.shape[0], -1, dtype=int)
    top_three: list[set[int]] = []
    for draft_token in range(association.shape[0]):
        start, end = association.indptr[draft_token : draft_token + 2]
        candidates = association.indices[start:end]
        values = association.data[start:end]
        order = np.argsort(values)[::-1]
        top_one[draft_token] = candidates[order[0]] if len(order) else -1
        top_three.append(set(candidates[order[:3]]))

    inverse_frequency = np.log(
        (fit_size + 1) / (fit_draft_frequency + 1)
    ) + 1
    output = np.zeros((len(indices), 28), dtype=float)
    for position, row_index in enumerate(indices):
        draft_tokens = draft_matrix[row_index].indices
        reply_tokens = set(reply_matrix[row_index].indices)
        if len(draft_tokens) == 0:
            continue
        draft_count_map = dict(
            zip(
                draft_counts[row_index].indices,
                draft_counts[row_index].data,
            )
        )
        reply_count_map = dict(
            zip(
                reply_counts[row_index].indices,
                reply_counts[row_index].data,
            )
        )
        relative_scores = []
        confidences = []
        top_one_hits = []
        top_three_hits = []
        present_scores = []
        matched_counts = []
        for draft_token in draft_tokens:
            start, end = association.indptr[draft_token : draft_token + 2]
            present = [
                value
                for candidate, value in zip(
                    association.indices[start:end], association.data[start:end]
                )
                if candidate in reply_tokens
            ]
            best_present = max(present, default=0.0)
            confidence = global_best[draft_token]
            mapped_reply = top_one[draft_token]
            relative_scores.append(
                best_present / confidence if confidence > 0 else 0.0
            )
            confidences.append(confidence)
            top_one_hits.append(float(mapped_reply in reply_tokens))
            top_three_hits.append(
                float(bool(top_three[draft_token] & reply_tokens))
            )
            present_scores.append(best_present)
            matched_counts.append(
                min(
                    draft_count_map.get(draft_token, 0),
                    reply_count_map.get(mapped_reply, 0),
                )
            )

        relative = np.asarray(relative_scores)
        confidence = np.asarray(confidences)
        top_one_array = np.asarray(top_one_hits)
        top_three_array = np.asarray(top_three_hits)
        present_array = np.asarray(present_scores)
        matched_array = np.asarray(matched_counts, dtype=float)
        counts = np.asarray(
            [draft_count_map[token] for token in draft_tokens], dtype=float
        )
        idf = inverse_frequency[draft_tokens]
        covered = confidence > 0
        relative_quantiles = np.quantile(relative, [0, 0.25, 0.5, 0.75, 1])
        confidence_quantiles = np.quantile(confidence, [0, 0.5, 1])
        output[position] = [
            *relative_quantiles,
            relative.std(),
            *(np.mean(relative <= threshold) for threshold in [0.1, 0.25, 0.5, 0.75]),
            *confidence_quantiles,
            confidence.std(),
            np.sum(top_one_array * confidence) / max(np.sum(confidence), 1e-12),
            np.sum(top_three_array * confidence) / max(np.sum(confidence), 1e-12),
            *(
                np.mean((confidence >= threshold) & (top_one_array == 0))
                for threshold in [0.1, 0.2, 0.3, 0.5]
            ),
            np.sum(top_one_array * idf) / np.sum(idf),
            np.sum(top_one_array * counts) / np.sum(counts),
            np.sum(matched_array) / np.sum(counts),
            np.sum(top_one_array * idf * counts) / np.sum(idf * counts),
            np.sum(matched_array * idf) / np.sum(idf * counts),
            np.sum(present_array) / max(np.sum(confidence), 1e-12),
            covered.mean(),
            top_one_array[covered].mean() if covered.any() else 0.0,
        ]
    return output


def score_bidirectional_association(
    association: sparse.csr_matrix,
    draft_matrix: sparse.csr_matrix,
    reply_matrix: sparse.csr_matrix,
    indices: np.ndarray,
) -> np.ndarray:
    """Add reverse and mutual opaque-code correspondence checks."""
    forward = np.full(association.shape[0], -1, dtype=int)
    for draft_token in range(association.shape[0]):
        start, end = association.indptr[draft_token : draft_token + 2]
        if end > start:
            forward[draft_token] = association.indices[
                start + np.argmax(association.data[start:end])
            ]

    association_csc = association.tocsc()
    reverse = np.full(association.shape[1], -1, dtype=int)
    for reply_token in range(association.shape[1]):
        start, end = association_csc.indptr[reply_token : reply_token + 2]
        if end > start:
            reverse[reply_token] = association_csc.indices[
                start + np.argmax(association_csc.data[start:end])
            ]

    output = np.zeros((len(indices), 7), dtype=float)
    for position, row_index in enumerate(indices):
        draft_tokens = draft_matrix[row_index].indices
        reply_tokens = reply_matrix[row_index].indices
        if len(draft_tokens) == 0:
            continue
        draft_set = set(draft_tokens)
        reply_set = set(reply_tokens)
        forward_hits = np.asarray(
            [forward[token] in reply_set for token in draft_tokens], dtype=float
        )
        reverse_drafts = {
            reverse[token] for token in reply_tokens if reverse[token] >= 0
        }
        reverse_hits = np.asarray(
            [token in reverse_drafts for token in draft_tokens], dtype=float
        )
        reply_hit = (
            np.mean([reverse[token] in draft_set for token in reply_tokens])
            if len(reply_tokens)
            else 0.0
        )
        output[position] = [
            forward_hits.mean(),
            reverse_hits.mean(),
            np.maximum(forward_hits, reverse_hits).mean(),
            np.minimum(forward_hits, reverse_hits).mean(),
            reply_hit,
            np.mean([forward[token] >= 0 for token in draft_tokens]),
            (
                np.mean([reverse[token] >= 0 for token in reply_tokens])
                if len(reply_tokens)
                else 0.0
            ),
        ]
    return output


def cross_fitted_alignment_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate label-free and clean-pair mappings with each train dyad held out."""
    (
        word_draft,
        word_reply,
        word_draft_test,
        word_reply_test,
        word_draft_counts,
        word_reply_counts,
        word_draft_test_counts,
        word_reply_test_counts,
    ) = protected_token_matrices(train, test, "W")
    (
        bigram_draft,
        bigram_reply,
        bigram_draft_test,
        bigram_reply_test,
        _,
        _,
        _,
        _,
    ) = protected_token_matrices(train, test, "B")
    n_train = len(train)
    all_indices = np.arange(n_train)
    targets = train["draft_reframed"].to_numpy()
    groups = train["thread_id"].to_numpy()

    word_scores = np.zeros((n_train, 11))
    bidirectional_scores = np.zeros((n_train, 7))
    bigram_scores = np.zeros((n_train, 11))
    enhanced_scores = np.zeros((n_train, 28))
    negative_word_scores = np.zeros((n_train, 11))
    negative_bidirectional_scores = np.zeros((n_train, 7))
    negative_bigram_scores = np.zeros((n_train, 11))

    for held_out_group in pd.unique(groups):
        validation_indices = np.flatnonzero(groups == held_out_group)
        fit_indices = np.flatnonzero(groups != held_out_group)
        word_association = fit_cross_field_association(
            word_draft, word_reply, fit_indices
        )
        word_scores[validation_indices] = score_cross_field_association(
            word_association, word_draft, word_reply, validation_indices
        )
        bidirectional_scores[validation_indices] = score_bidirectional_association(
            word_association, word_draft, word_reply, validation_indices
        )
        fit_draft_frequency = np.asarray(
            word_draft[fit_indices].sum(axis=0)
        ).ravel()
        enhanced_scores[validation_indices] = score_enhanced_association(
            word_association,
            word_draft,
            word_reply,
            word_draft_counts,
            word_reply_counts,
            validation_indices,
            fit_draft_frequency,
            len(fit_indices),
        )
        bigram_association = fit_cross_field_association(
            bigram_draft, bigram_reply, fit_indices
        )
        bigram_scores[validation_indices] = score_cross_field_association(
            bigram_association,
            bigram_draft,
            bigram_reply,
            validation_indices,
        )

        clean_fit_indices = fit_indices[targets[fit_indices] == 0]
        negative_word_association = fit_cross_field_association(
            word_draft, word_reply, clean_fit_indices
        )
        negative_word_scores[validation_indices] = score_cross_field_association(
            negative_word_association,
            word_draft,
            word_reply,
            validation_indices,
        )
        negative_bidirectional_scores[
            validation_indices
        ] = score_bidirectional_association(
            negative_word_association,
            word_draft,
            word_reply,
            validation_indices,
        )
        negative_bigram_association = fit_cross_field_association(
            bigram_draft, bigram_reply, clean_fit_indices
        )
        negative_bigram_scores[
            validation_indices
        ] = score_cross_field_association(
            negative_bigram_association,
            bigram_draft,
            bigram_reply,
            validation_indices,
        )

    test_indices = np.arange(len(test))
    full_word_association = fit_cross_field_association(
        word_draft, word_reply, all_indices
    )
    test_word_scores = score_cross_field_association(
        full_word_association,
        word_draft_test,
        word_reply_test,
        test_indices,
    )
    test_bidirectional_scores = score_bidirectional_association(
        full_word_association,
        word_draft_test,
        word_reply_test,
        test_indices,
    )
    full_draft_frequency = np.asarray(word_draft.sum(axis=0)).ravel()
    test_enhanced_scores = score_enhanced_association(
        full_word_association,
        word_draft_test,
        word_reply_test,
        word_draft_test_counts,
        word_reply_test_counts,
        test_indices,
        full_draft_frequency,
        n_train,
    )
    full_bigram_association = fit_cross_field_association(
        bigram_draft, bigram_reply, all_indices
    )
    test_bigram_scores = score_cross_field_association(
        full_bigram_association,
        bigram_draft_test,
        bigram_reply_test,
        test_indices,
    )

    clean_indices = all_indices[targets == 0]
    full_negative_word_association = fit_cross_field_association(
        word_draft, word_reply, clean_indices
    )
    test_negative_word_scores = score_cross_field_association(
        full_negative_word_association,
        word_draft_test,
        word_reply_test,
        test_indices,
    )
    test_negative_bidirectional_scores = score_bidirectional_association(
        full_negative_word_association,
        word_draft_test,
        word_reply_test,
        test_indices,
    )
    full_negative_bigram_association = fit_cross_field_association(
        bigram_draft, bigram_reply, clean_indices
    )
    test_negative_bigram_scores = score_cross_field_association(
        full_negative_bigram_association,
        bigram_draft_test,
        bigram_reply_test,
        test_indices,
    )

    word_names = [
        "align_present_mean",
        "align_present_max",
        "align_ratio_mean",
        "align_ratio_covered",
        "align_top1",
        "align_top1_covered",
        "align_top3",
        "align_top3_covered",
        "align_coverage",
        "align_confidence",
        "align_sum_ratio",
    ]
    bidirectional_names = [
        "bi_forward",
        "bi_reverse",
        "bi_union",
        "bi_intersection",
        "bi_replyhit",
        "bi_dcoverage",
        "bi_rcoverage",
    ]
    bigram_names = [
        "bg_present_mean",
        "bg_present_max",
        "bg_ratio_mean",
        "bg_ratio_covered",
        "bg_top1",
        "bg_top1_covered",
        "bg_top3",
        "bg_top3_covered",
        "bg_coverage",
        "bg_confidence",
        "bg_sum_ratio",
    ]
    enhanced_names = [
        "rel_min",
        "rel_q25",
        "rel_median",
        "rel_q75",
        "rel_max",
        "rel_std",
        "rel_le_10",
        "rel_le_25",
        "rel_le_50",
        "rel_le_75",
        "conf_min",
        "conf_median",
        "conf_max",
        "conf_std",
        "conf_weighted_top1",
        "conf_weighted_top3",
        "absent_conf_10",
        "absent_conf_20",
        "absent_conf_30",
        "absent_conf_50",
        "idf_top1",
        "count_top1",
        "count_matched",
        "idf_count_top1",
        "idf_count_matched",
        "present_conf_ratio",
        "enh_coverage",
        "enh_hit_covered",
    ]
    negative_word_names = [f"negw_{name}" for name in word_names]
    negative_bidirectional_names = [
        f"negbi_{name}" for name in bidirectional_names
    ]
    negative_bigram_names = [f"negbg_{name}" for name in bigram_names]
    names = (
        word_names
        + bidirectional_names
        + bigram_names
        + enhanced_names
        + negative_word_names
        + negative_bidirectional_names
        + negative_bigram_names
    )
    train_values = np.hstack(
        [
            word_scores,
            bidirectional_scores,
            bigram_scores,
            enhanced_scores,
            negative_word_scores,
            negative_bidirectional_scores,
            negative_bigram_scores,
        ]
    )
    test_values = np.hstack(
        [
            test_word_scores,
            test_bidirectional_scores,
            test_bigram_scores,
            test_enhanced_scores,
            test_negative_word_scores,
            test_negative_bidirectional_scores,
            test_negative_bigram_scores,
        ]
    )
    return (
        pd.DataFrame(train_values, columns=names),
        pd.DataFrame(test_values, columns=names),
    )


def add_alignment_features(
    features: pd.DataFrame, alignment: pd.DataFrame
) -> pd.DataFrame:
    result = pd.concat(
        [features.reset_index(drop=True), alignment.reset_index(drop=True)], axis=1
    )
    survival_edges = [
        -0.01, 0, 0.1, 0.25, 0.3334, 0.4, 0.5, 0.67, 0.8, 0.999, 1.01
    ]
    result["align_survival_bin"] = pd.cut(
        result["align_top1"], survival_edges, labels=False
    ).astype(str)
    result["align_ratio_score_bin"] = pd.cut(
        result["align_ratio_mean"],
        [-0.01, 0, 0.1, 0.25, 0.4, 0.5, 0.67, 0.8, 0.999, 1.01],
        labels=False,
    ).astype(str)
    result["align_ratio_interaction"] = (
        result["align_survival_bin"] + "_" + result["ratio_bin"]
    )
    result["align_draft_interaction"] = (
        result["align_survival_bin"] + "_" + result["draft_nw_bin"]
    )
    result["mapped_removed_words"] = (
        result["draft_nw"] * (1 - result["align_top1"])
    )
    result["mapped_surviving_words"] = (
        result["draft_nw"] * result["align_top1"]
    )
    result["bi_reverse_bin"] = pd.cut(
        result["bi_reverse"], survival_edges, labels=False
    ).astype(str)
    result["bi_union_bin"] = pd.cut(
        result["bi_union"], survival_edges, labels=False
    ).astype(str)
    result["bi_consensus"] = (
        result["align_survival_bin"] + "_" + result["bi_reverse_bin"]
    )
    result["bg_survival_bin"] = pd.cut(
        result["bg_top1"], survival_edges, labels=False
    ).astype(str)
    result["word_bigram_consensus"] = (
        result["align_survival_bin"] + "_" + result["bg_survival_bin"]
    )

    fraction_edges = [-0.01, 0, 0.1, 0.25, 0.3334, 0.5, 0.67, 0.8, 0.999, 1.01]
    enhanced_bin_columns = [
        "rel_le_50",
        "rel_le_75",
        "absent_conf_10",
        "absent_conf_20",
        "absent_conf_30",
        "absent_conf_50",
    ]
    for column in enhanced_bin_columns:
        result[f"{column}_bin"] = pd.cut(
            result[column], fraction_edges, labels=False
        ).astype(str)
    result["absent_survival_consensus"] = (
        result["absent_conf_20_bin"] + "_" + result["align_survival_bin"]
    )
    result["absent_ratio_consensus"] = (
        result["absent_conf_20_bin"] + "_" + result["ratio_bin"]
    )

    result["negw_surv_bin"] = pd.cut(
        result["negw_align_top1"], survival_edges, labels=False
    ).astype(str)
    result["negw_ratio_bin"] = pd.cut(
        result["negw_align_ratio_mean"],
        [-0.01, 0, 0.1, 0.25, 0.4, 0.5, 0.67, 0.8, 0.999, 1.01],
        labels=False,
    ).astype(str)
    result["negbg_surv_bin"] = pd.cut(
        result["negbg_bg_top1"], survival_edges, labels=False
    ).astype(str)
    result["uns_neg_consensus"] = (
        result["align_survival_bin"] + "_" + result["negw_surv_bin"]
    )
    result["neg_ratio_length"] = (
        result["negw_ratio_bin"] + "_" + result["ratio_bin"]
    )
    result["align_surv_delta"] = (
        result["align_top1"] - result["negw_align_top1"]
    )
    result["align_ratio_delta"] = (
        result["align_ratio_mean"] - result["negw_align_ratio_mean"]
    )
    return result


def make_documents(
    df: pd.DataFrame,
    features: pd.DataFrame,
    categorical: list[str],
) -> list[str]:
    documents: list[str] = []
    thresholds = {
        "context_nw": [1, 2, 3, 5, 8, 12, 20],
        "draft_nw": [0, 1, 2, 3, 5, 8, 12, 20],
        "reply_nw": [0, 1, 2, 3, 5, 8, 12, 16, 25],
        "reply_minus_draft_w": [-10, -5, -2, -1, 0, 1, 2, 5, 10],
        "reply_over_draft_w": [0.5, 0.8, 1, 1.25, 1.5, 2, 3, 5],
    }
    for index, row in df.reset_index(drop=True).iterrows():
        tokens: list[str] = []
        for column, prefix in (
            ("context_tokens", "C"),
            ("draft_tokens", "D"),
            ("reply_tokens", "R"),
        ):
            tokens.extend(f"{prefix}_{token}" for token in row[column].split())
        for column in categorical:
            tokens.append(f"M_{column}={features.iloc[index][column]}")
        for column, edges in thresholds.items():
            value = features.iloc[index][column]
            tokens.extend(f"T_{column}>{edge}" for edge in edges if value > edge)
        documents.append(" ".join(tokens))
    return documents

def make_pair_documents(df: pd.DataFrame) -> list[str]:
    """Create recurring opaque draft/reply code-pair indicators."""
    documents = []
    for _, row in df.iterrows():
        draft_words = sorted(
            {
                token
                for token in row["draft_tokens"].split()
                if token.startswith("W_")
            }
        )
        reply_words = sorted(
            {
                token
                for token in row["reply_tokens"].split()
                if token.startswith("W_")
            }
        )
        documents.append(
            " ".join(
                f"P_{draft_word}_{reply_word}"
                for draft_word in draft_words
                for reply_word in reply_words
            )
        )
    return documents

def score_weighted_leave_one_out(
    draft_matrix: sparse.csr_matrix,
    reply_matrix: sparse.csr_matrix,
    weights: np.ndarray,
) -> np.ndarray:
    """Induce a mapping from pseudo-clean rows while removing each scored row."""
    weighted_reply = reply_matrix.multiply(weights[:, None])
    cooccurrence = (draft_matrix.T @ weighted_reply).tocsr().astype(float)
    draft_frequency = np.asarray(draft_matrix.T @ weights).ravel()
    reply_frequency = np.asarray(reply_matrix.T @ weights).ravel()
    output = np.zeros((draft_matrix.shape[0], 8), dtype=float)

    for row_index in range(draft_matrix.shape[0]):
        draft_tokens = draft_matrix[row_index].indices
        reply_tokens = set(reply_matrix[row_index].indices)
        row_weight = weights[row_index]
        if len(draft_tokens) == 0:
            continue
        top_one_hits = []
        top_two_hits = []
        confidences = []
        for draft_token in draft_tokens:
            start, end = cooccurrence.indptr[draft_token : draft_token + 2]
            candidates = cooccurrence.indices[start:end]
            counts = cooccurrence.data[start:end].copy()
            present = np.fromiter(
                (candidate in reply_tokens for candidate in candidates),
                dtype=bool,
                count=len(candidates),
            )
            adjusted_counts = counts - row_weight * present.astype(float)
            denominator = np.sqrt(
                np.maximum(draft_frequency[draft_token] - row_weight, 0)
                * np.maximum(
                    reply_frequency[candidates]
                    - row_weight * present.astype(float),
                    0,
                )
            )
            scores = np.divide(
                adjusted_counts,
                denominator,
                out=np.zeros_like(adjusted_counts),
                where=(adjusted_counts > 1e-12) & (denominator > 0),
            )
            order = np.argsort(scores)[::-1]
            valid = order[scores[order] > 0]
            mapped = candidates[valid[:3]]
            top_one_hits.append(
                float(len(mapped) > 0 and mapped[0] in reply_tokens)
            )
            top_two_hits.append(
                float(any(token in reply_tokens for token in mapped[:2]))
            )
            confidences.append(scores[valid[0]] if len(valid) else 0.0)

        top_one = np.asarray(top_one_hits)
        top_two = np.asarray(top_two_hits)
        confidence = np.asarray(confidences)
        covered = confidence > 0
        output[row_index] = [
            top_one.mean(),
            top_two.mean(),
            covered.mean(),
            top_one[covered].mean() if covered.any() else 0.0,
            top_two[covered].mean() if covered.any() else 0.0,
            confidence.mean(),
            np.sum(top_one * confidence) / max(np.sum(confidence), 1e-12),
            np.sum(top_two * confidence) / max(np.sum(confidence), 1e-12),
        ]
    return output


def pseudo_clean_alignment_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_risk: np.ndarray,
    test_risk: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt the opaque lexicon with high-confidence unlabeled rows."""
    (
        draft_train,
        reply_train,
        draft_test,
        reply_test,
        _,
        _,
        _,
        _,
    ) = protected_token_matrices(train, test, "W")
    draft_matrix = sparse.vstack([draft_train, draft_test], format="csr")
    reply_matrix = sparse.vstack([reply_train, reply_test], format="csr")
    risk = np.concatenate([train_risk, test_risk])
    all_features = []
    names = []
    for threshold in (0.20, 0.25):
        scores = score_weighted_leave_one_out(
            draft_matrix,
            reply_matrix,
            (risk < threshold).astype(float),
        )
        all_features.append(scores)
        prefix = f"adapt{int(threshold * 100)}"
        names.extend(f"{prefix}_{index}" for index in range(scores.shape[1]))
    values = np.hstack(all_features)
    n_train = len(train)
    return (
        pd.DataFrame(values[:n_train], columns=names),
        pd.DataFrame(values[n_train:], columns=names),
    )


def add_pseudo_clean_features(
    features: pd.DataFrame, pseudo_clean: pd.DataFrame
) -> pd.DataFrame:
    result = pd.concat(
        [features.reset_index(drop=True), pseudo_clean.reset_index(drop=True)],
        axis=1,
    )
    edges = [-0.01, 0, 0.1, 0.25, 0.3334, 0.5, 0.67, 0.8, 0.999, 1.01]
    bin_sources = [
        "adapt20_0",
        "adapt20_1",
        "adapt20_3",
        "adapt20_4",
        "adapt20_7",
        "adapt25_7",
    ]
    for column in bin_sources:
        result[f"{column}_bin"] = pd.cut(
            result[column], edges, labels=False
        ).astype(str)
    result["adapt_consensus"] = (
        result["adapt20_7_bin"] + "_" + result["align_survival_bin"]
    )
    return result


def repeated_grouped_predictions(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    groups: np.ndarray,
    regularization: tuple[float, ...],
) -> np.ndarray:
    prediction_sum = np.zeros(len(target))
    prediction_count = np.zeros(len(target))
    for seed in CV_SEEDS:
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed
        )
        for fit_indices, valid_indices in splitter.split(matrix, target, groups):
            fold_prediction = np.zeros(len(valid_indices))
            for c_value in regularization:
                model = fit_model(matrix[fit_indices], target[fit_indices], c_value)
                fold_prediction += model.predict_proba(matrix[valid_indices])[:, 1]
            prediction_sum[valid_indices] += (
                fold_prediction / len(regularization)
            )
            prediction_count[valid_indices] += 1
    return prediction_sum / prediction_count


def full_ensemble_predictions(
    train_matrix: sparse.csr_matrix,
    test_matrix: sparse.csr_matrix,
    target: np.ndarray,
    regularization: tuple[float, ...],
) -> np.ndarray:
    prediction = np.zeros(test_matrix.shape[0])
    for c_value in regularization:
        model = fit_model(train_matrix, target, c_value)
        prediction += model.predict_proba(test_matrix)[:, 1]
    return prediction / len(regularization)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(copy=True)


def competition_metric(y_true: np.ndarray, prediction: np.ndarray) -> float:
    prevalence = float(np.mean(y_true))
    ap = average_precision_score(y_true, prediction)
    auc = roc_auc_score(y_true, prediction)
    ap_skill = np.clip((ap - prevalence) / (1 - prevalence), 0, 1)
    auc_skill = np.clip(2 * auc - 1, 0, 1)
    return float(0.5 * ap_skill + 0.5 * auc_skill)


def fit_model(
    matrix: sparse.csr_matrix, y: np.ndarray, c_value: float
) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
    )
    model.fit(matrix, y)
    return model


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["draft_reframed"].to_numpy()
    groups = train["thread_id"].to_numpy()
    n_train = len(train)

    train_features = make_features(train)
    test_features = make_features(test)
    train_alignment, test_alignment = cross_fitted_alignment_features(train, test)
    train_features = add_alignment_features(train_features, train_alignment)
    test_features = add_alignment_features(test_features, test_alignment)

    document_categorical = BASE_CATEGORICAL + [
        "extent_reply_relation",
        "revision_extent",
        "age_pause",
        "snap_extent",
        "draft_nw_bin",
        "reply_nw_bin",
        "ratio_bin",
        "peer_similarity_bin",
    ]
    categorical = document_categorical + [
        "align_survival_bin",
        "align_ratio_score_bin",
        "align_ratio_interaction",
        "align_draft_interaction",
        "bi_reverse_bin",
        "bi_union_bin",
        "bi_consensus",
        "bg_survival_bin",
        "word_bigram_consensus",
        "rel_le_50_bin",
        "rel_le_75_bin",
        "absent_conf_10_bin",
        "absent_conf_20_bin",
        "absent_conf_30_bin",
        "absent_conf_50_bin",
        "absent_survival_consensus",
        "absent_ratio_consensus",
        "negw_surv_bin",
        "negw_ratio_bin",
        "negbg_surv_bin",
        "uns_neg_consensus",
        "neg_ratio_length",
    ]
    ignored = set(train.columns) | set(TEXT_COLUMNS) | {
        "thread_id",
        "turn_id",
        "draft_reframed",
    }

    def encode(
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        categorical_columns: list[str],
    ) -> sparse.csr_matrix:
        numeric_columns = [
            column
            for column in train_frame.columns
            if column not in ignored
            and column not in categorical_columns
            and pd.api.types.is_numeric_dtype(train_frame[column])
        ]
        combined = pd.concat(
            [
                train_frame[categorical_columns + numeric_columns],
                test_frame[categorical_columns + numeric_columns],
            ],
            ignore_index=True,
        )
        encoder = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns,
                ),
            ]
        )
        return sparse.csr_matrix(encoder.fit_transform(combined))

    encoded = encode(train_features, test_features, categorical)
    train_documents = make_documents(
        train, train_features, document_categorical
    )
    test_documents = make_documents(test, test_features, document_categorical)
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=3,
        binary=True,
        norm="l2",
    )
    text_matrix = vectorizer.fit_transform(train_documents + test_documents)
    pair_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        binary=True,
        norm="l2",
    )
    pair_matrix = pair_vectorizer.fit_transform(
        make_pair_documents(train) + make_pair_documents(test)
    )

    initial_train_matrix = sparse.hstack(
        [
            encoded[:n_train],
            text_matrix[:n_train] * 4.0,
            pair_matrix[:n_train] * 4.0,
        ],
        format="csr",
    )
    initial_test_matrix = sparse.hstack(
        [
            encoded[n_train:],
            text_matrix[n_train:] * 4.0,
            pair_matrix[n_train:] * 4.0,
        ],
        format="csr",
    )
    initial_oof = repeated_grouped_predictions(
        initial_train_matrix,
        y,
        groups,
        INITIAL_REGULARIZATION,
    )
    initial_prediction = full_ensemble_predictions(
        initial_train_matrix,
        initial_test_matrix,
        y,
        INITIAL_TEST_REGULARIZATION,
    )
    train_empty = train_features["draft_empty"].to_numpy() == 1
    test_empty = test_features["draft_empty"].to_numpy() == 1
    initial_oof[train_empty] = 0.0
    initial_prediction[test_empty] = 0.0
    print(
        f"Initial grouped CV lift: {competition_metric(y, initial_oof):.6f}"
    )

    train_adaptive, test_adaptive = pseudo_clean_alignment_features(
        train,
        test,
        initial_oof,
        initial_prediction,
    )
    train_features = add_pseudo_clean_features(train_features, train_adaptive)
    test_features = add_pseudo_clean_features(test_features, test_adaptive)
    adaptive_categorical = categorical + [
        "adapt20_0_bin",
        "adapt20_1_bin",
        "adapt20_3_bin",
        "adapt20_4_bin",
        "adapt20_7_bin",
        "adapt25_7_bin",
        "adapt_consensus",
    ]
    adaptive_encoded = encode(
        train_features,
        test_features,
        adaptive_categorical,
    )
    final_train_matrix = sparse.hstack(
        [
            adaptive_encoded[:n_train],
            text_matrix[:n_train] * 4.0,
            pair_matrix[:n_train] * 4.0,
        ],
        format="csr",
    )
    final_test_matrix = sparse.hstack(
        [
            adaptive_encoded[n_train:],
            text_matrix[n_train:] * 4.0,
            pair_matrix[n_train:] * 4.0,
        ],
        format="csr",
    )

    final_oof = repeated_grouped_predictions(
        final_train_matrix,
        y,
        groups,
        FINAL_REGULARIZATION,
    )
    final_prediction = full_ensemble_predictions(
        final_train_matrix,
        final_test_matrix,
        y,
        FINAL_REGULARIZATION,
    )
    adaptive_train_rank = 0.5 * (
        percentile_rank(-train_features["adapt20_7"].to_numpy())
        + percentile_rank(-train_features["adapt25_7"].to_numpy())
    )
    adaptive_test_rank = 0.5 * (
        percentile_rank(-test_features["adapt20_7"].to_numpy())
        + percentile_rank(-test_features["adapt25_7"].to_numpy())
    )
    train_cluster = pd.DataFrame(
        {"reply": train["reply_tokens"], "risk": final_oof}
    )
    test_cluster = pd.DataFrame(
        {"reply": test["reply_tokens"], "risk": final_prediction}
    )
    train_grouped = train_cluster.groupby("reply", sort=False)
    test_grouped = test_cluster.groupby("reply", sort=False)
    train_group_size = train_grouped["risk"].transform("size")
    test_group_size = test_grouped["risk"].transform("size")
    train_cluster_bonus = (
        train_grouped["risk"].rank(pct=True) - 0.5
    ) * (train_group_size > 1)
    test_cluster_bonus = (
        test_grouped["risk"].rank(pct=True) - 0.5
    ) * (test_group_size > 1)
    train_safe = (
        (train_features["bi_reverse"].to_numpy() > 0.4)
        | (train_features["adapt20_7"].to_numpy() >= 0.65)
    ).astype(float)
    test_safe = (
        (test_features["bi_reverse"].to_numpy() > 0.4)
        | (test_features["adapt20_7"].to_numpy() >= 0.65)
    ).astype(float)
    final_oof = percentile_rank(
        0.95 * percentile_rank(final_oof)
        + 0.05 * adaptive_train_rank
        - 0.15 * train_safe
        + 0.005 * train_cluster_bonus.to_numpy()
    )
    final_prediction = percentile_rank(
        0.95 * percentile_rank(final_prediction)
        + 0.05 * adaptive_test_rank
        - 0.15 * test_safe
        + 0.005 * test_cluster_bonus.to_numpy()
    )
    final_oof[train_empty] = 0.0
    final_prediction[test_empty] = 0.0
    print(
        f"Two-pass grouped CV lift: {competition_metric(y, final_oof):.6f}"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {"turn_id": test["turn_id"], "reframe_risk": final_prediction}
    )
    submission.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(submission)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
