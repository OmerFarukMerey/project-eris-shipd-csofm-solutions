#!/usr/bin/env python3
"""Rank biomedical evidence candidates.

LEAKAGE DISCIPLINE (see readme.txt "Pre-submit audit"): every vectorizer, IDF,
SVD, PPMI embedding, NMF model, and corpus statistic is FIT ON TRAIN-REFERENCED
DOCUMENTS ONLY. Test documents are only ever transform()-ed. There is no
train+test concatenation anywhere. Slate-relative features are computed strictly
within a single slate (one ranking example), never pooled across slates. The
model is trained on train rows and predict()-ed on test rows.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

SEED = 20260716
N_FOLDS = 5
TOKEN = r"[^ ]+"


def row_dot(left: sparse.csr_matrix, right: sparse.csr_matrix, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(left[a].multiply(right[b]).sum(axis=1)).ravel().astype(np.float32)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def build_doc_space(documents: pd.DataFrame, fit_rows: np.ndarray) -> dict:
    """Fit all transforms on the TRAIN-referenced documents (fit_rows) and return
    per-document representations for EVERY document (train docs fit, all docs
    transformed). Nothing here ever sees test rows of train.csv/test.csv."""
    title = documents["title_tokens"].astype(str).tolist()
    abstract = documents["abstract_tokens"].astype(str).tolist()
    combined = (documents["title_tokens"].astype(str) + " " + documents["abstract_tokens"].astype(str)).tolist()
    fit_combined = [combined[i] for i in fit_rows]
    fit_title = [title[i] for i in fit_rows]
    fit_abstract = [abstract[i] for i in fit_rows]
    n_doc = len(documents)

    # --- TF-IDF (fit vocab + IDF on train docs, transform all) ---
    tfidf_vec = TfidfVectorizer(lowercase=False, token_pattern=TOKEN, sublinear_tf=True, norm="l2", dtype=np.float32)
    tfidf_vec.fit(fit_combined)
    vocab = tfidf_vec.vocabulary_
    idf = tfidf_vec.idf_.astype(np.float32)
    all_tfidf = tfidf_vec.transform(combined).tocsr()
    title_vec = TfidfVectorizer(lowercase=False, token_pattern=TOKEN, sublinear_tf=True, norm="l2",
                                dtype=np.float32, vocabulary=vocab).fit(fit_title)
    abstract_vec = TfidfVectorizer(lowercase=False, token_pattern=TOKEN, sublinear_tf=True, norm="l2",
                                   dtype=np.float32, vocabulary=vocab).fit(fit_abstract)
    title_tfidf = title_vec.transform(title).tocsr()
    abstract_tfidf = abstract_vec.transform(abstract).tocsr()

    count_vec = CountVectorizer(lowercase=False, token_pattern=TOKEN, vocabulary=vocab, dtype=np.float32)
    counts = count_vec.transform(combined).tocsr()
    title_counts = count_vec.transform(title).tocsr()

    binary = (counts > 0).astype(np.float32).tocsr()
    title_binary = (title_counts > 0).astype(np.float32).tocsr()
    abstract_binary = (count_vec.transform(abstract) > 0).astype(np.float32).tocsr()

    # binary weighted by sqrt-idf and idf (train idf)
    sqrt_idf = np.sqrt(idf)
    binary_sqrt = binary.multiply(sqrt_idf).tocsr()
    title_sqrt = title_binary.multiply(sqrt_idf).tocsr()
    abstract_sqrt = abstract_binary.multiply(sqrt_idf).tocsr()

    # --- BM25 (average length from TRAIN docs only) ---
    doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    avg_len = float(doc_len[fit_rows].mean())
    count_rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    denom = counts.data + 1.5 * (0.25 + 0.75 * doc_len[count_rows] / max(avg_len, 1e-6))
    bm25 = counts.copy()
    bm25.data = (counts.data * 2.5 / denom * idf[counts.indices]).astype(np.float32)

    # --- robust title-boosted tf-idf anchors ---
    robust = {}
    for boost in (2.0, 4.0):
        m = (counts + boost * title_counts).tocsr()
        m.data = 1.0 + np.log(m.data)
        m = m.multiply(idf).tocsr()
        robust[boost] = normalize(m, norm="l2", copy=False)

    # --- n-gram tf-idf (fit on train docs) ---
    ngram_vec = TfidfVectorizer(lowercase=False, token_pattern=TOKEN, ngram_range=(2, 3), min_df=2,
                                sublinear_tf=True, norm="l2", dtype=np.float32)
    ngram_vec.fit(fit_combined)
    all_ngram = ngram_vec.transform(combined).tocsr()

    # --- LSA (fit SVD on train-doc tf-idf, transform all) ---
    svd = TruncatedSVD(n_components=192, algorithm="arpack", tol=1e-4, random_state=SEED)
    svd.fit(all_tfidf[fit_rows])
    all_lsa = svd.transform(all_tfidf).astype(np.float32)
    title_lsa = unit_rows(svd.transform(title_tfidf).astype(np.float32))
    abstract_lsa = unit_rows(svd.transform(abstract_tfidf).astype(np.float32))
    lsa_unit = unit_rows(all_lsa)

    svd160 = TruncatedSVD(n_components=160, algorithm="arpack", tol=1e-4, random_state=SEED)
    svd160.fit(all_tfidf[fit_rows])
    lsa160 = unit_rows(svd160.transform(all_tfidf).astype(np.float32))

    # --- PPMI token embeddings (co-occurrence over TRAIN docs only) ---
    token_emb = _ppmi_embeddings(binary[fit_rows], vocab_size=len(vocab), n_doc_fit=len(fit_rows), ndim=128)

    def doc_embed(field_binary: sparse.csr_matrix) -> np.ndarray:
        weighted = field_binary.multiply(idf[None, :]).tocsr()
        return unit_rows(np.nan_to_num((weighted @ token_emb).astype(np.float32)))

    emb_full = doc_embed(binary)
    emb_title = doc_embed(title_binary)
    emb_abstract = doc_embed(abstract_binary)

    # --- NMF topics (fit on train-doc tf-idf, transform all) ---
    nmf = NMF(n_components=64, init="nndsvda", max_iter=200, random_state=SEED, beta_loss="frobenius", solver="cd")
    nmf.fit(all_tfidf[fit_rows])
    topics = nmf.transform(all_tfidf).astype(np.float64)

    # --- per-document scalar stats (train-fitted idf) ---
    unique_count = np.diff(binary.indptr).astype(np.float32)
    title_unique = np.diff(title_binary.indptr).astype(np.float32)
    abstract_unique = np.diff(abstract_binary.indptr).astype(np.float32)
    idf_mass = np.asarray(binary.multiply(idf).sum(axis=1)).ravel().astype(np.float32)
    title_idf_mass = np.asarray(title_binary.multiply(idf).sum(axis=1)).ravel().astype(np.float32)
    abstract_idf_mass = np.asarray(abstract_binary.multiply(idf).sum(axis=1)).ravel().astype(np.float32)
    rare_thr = float(np.quantile(idf, 0.90))
    rare_mask = (idf >= rare_thr).astype(np.float32)
    rare_binary = binary.multiply(rare_mask).tocsr()
    rare_75 = normalize(all_tfidf.multiply((idf >= np.quantile(idf, 0.75)).astype(np.float32)).multiply(idf).tocsr(), norm="l2", copy=False)
    rare_binary_90 = binary.multiply((idf >= np.quantile(idf, 0.90)).astype(np.float32)).tocsr()
    rare_q90 = np.asarray(rare_binary_90.sum(axis=1)).ravel().astype(np.float32)

    # most-discriminative shared token needs per-doc token sets and idf lookup
    doc_tokens = [set(t.split()) for t in combined]

    # top-K highest-idf tokens per field (for soft matching)
    topk = {
        "combined": _topk_tokens(binary, idf, 24, n_doc),
        "title": _topk_tokens(title_binary, idf, 14, n_doc),
        "abstract": _topk_tokens(abstract_binary, idf, 24, n_doc),
    }

    return {
        "idf": idf, "vocab": vocab, "doc_tokens": doc_tokens,
        "all_tfidf": all_tfidf, "title_tfidf": title_tfidf, "abstract_tfidf": abstract_tfidf,
        "binary": binary, "title_binary": title_binary, "abstract_binary": abstract_binary,
        "binary_sqrt": binary_sqrt, "title_sqrt": title_sqrt, "abstract_sqrt": abstract_sqrt,
        "bm25": bm25, "robust": robust, "all_ngram": all_ngram,
        "all_lsa": all_lsa, "lsa_unit": lsa_unit, "title_lsa": title_lsa, "abstract_lsa": abstract_lsa,
        "lsa160": lsa160,
        "token_emb": token_emb, "emb_full": emb_full, "emb_title": emb_title, "emb_abstract": emb_abstract,
        "topics": topics,
        "unique_count": unique_count, "title_unique": title_unique, "abstract_unique": abstract_unique,
        "idf_mass": idf_mass, "title_idf_mass": title_idf_mass, "abstract_idf_mass": abstract_idf_mass,
        "doc_len": doc_len, "rare_binary": rare_binary, "rare_75": rare_75,
        "rare_binary_90": rare_binary_90, "rare_q90": rare_q90, "topk": topk,
    }


def _ppmi_embeddings(binary_fit: sparse.csr_matrix, vocab_size: int, n_doc_fit: int, ndim: int) -> np.ndarray:
    cooc = (binary_fit.T @ binary_fit).tocoo()
    rows, cols, data = cooc.row, cooc.col, cooc.data.astype(np.float64)
    total = data.sum()
    marginal = np.asarray(cooc.sum(axis=1)).ravel().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((data * total) / (marginal[rows] * marginal[cols]))
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)
    positive = np.maximum(pmi, 0.0)
    keep = positive > 0
    ppmi = sparse.csr_matrix((positive[keep], (rows[keep], cols[keep])), shape=(vocab_size, vocab_size), dtype=np.float64)
    svd = TruncatedSVD(n_components=ndim, algorithm="randomized", random_state=SEED)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        us = svd.fit_transform(ppmi)
        singular = np.maximum(svd.singular_values_.astype(np.float64), 1e-8)
        emb = us / np.sqrt(singular)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return normalize(emb, norm="l2", copy=False)


def _topk_tokens(field_binary: sparse.csr_matrix, idf: np.ndarray, k: int, n_doc: int):
    index = np.zeros((n_doc, k), np.int32)
    weight = np.zeros((n_doc, k), np.float32)
    mask = np.zeros((n_doc, k), np.bool_)
    indptr, indices = field_binary.indptr, field_binary.indices
    for r in range(n_doc):
        a, b = indptr[r], indptr[r + 1]
        tokens = indices[a:b]
        if tokens.size == 0:
            continue
        w = idf[tokens]
        if tokens.size > k:
            sel = np.argpartition(-w, k - 1)[:k]
            tokens = tokens[sel]; w = w[sel]
        j = tokens.size
        index[r, :j] = tokens; weight[r, :j] = w; mask[r, :j] = True
    return index, weight, mask


def pair_features(space: dict, qix: np.ndarray, cix: np.ndarray,
                  query_len: np.ndarray, cand_len: np.ndarray) -> pd.DataFrame:
    """Row-wise features for a set of (query, candidate) pairs. Every input array
    indexes into TRAIN-fitted per-document representations, so this is pure
    inference for test pairs (uses only each test document's own transformed
    representation)."""
    idf = space["idf"]
    f: dict[str, np.ndarray] = {}

    # lexical cosines
    f["all_cos"] = row_dot(space["all_tfidf"], space["all_tfidf"], qix, cix)
    f["tt_cos"] = row_dot(space["title_tfidf"], space["title_tfidf"], qix, cix)
    f["ta_cos"] = row_dot(space["title_tfidf"], space["abstract_tfidf"], qix, cix)
    f["at_cos"] = row_dot(space["abstract_tfidf"], space["title_tfidf"], qix, cix)
    f["aa_cos"] = row_dot(space["abstract_tfidf"], space["abstract_tfidf"], qix, cix)
    for boost in (2.0, 4.0):
        f[f"robust_title{int(boost)}"] = row_dot(space["robust"][boost], space["robust"][boost], qix, cix)

    binary = space["binary"]
    overlap = row_dot(binary, binary, qix, cix)
    overlap_idf = row_dot(space["binary_sqrt"], space["binary_sqrt"], qix, cix)
    f["overlap_n"] = overlap
    f["overlap_idf"] = overlap_idf
    f["bm25_qc"] = row_dot(binary, space["bm25"], qix, cix)

    uq, uc = space["unique_count"], space["unique_count"]
    f["q_contain"] = overlap / np.maximum(uq[qix], 1.0)
    f["c_contain"] = overlap / np.maximum(uc[cix], 1.0)
    f["jaccard"] = overlap / np.maximum(uq[qix] + uc[cix] - overlap, 1.0)
    f["idf_q_contain"] = overlap_idf / np.maximum(space["idf_mass"][qix], 1e-6)
    f["idf_c_contain"] = overlap_idf / np.maximum(space["idf_mass"][cix], 1e-6)

    for name, left, right in (("tt", space["title_binary"], space["title_binary"]),
                              ("ta", space["title_binary"], space["abstract_binary"]),
                              ("aa", space["abstract_binary"], space["abstract_binary"])):
        f[f"{name}_n"] = row_dot(left, right, qix, cix)
    for name, left, right in (("tt", space["title_sqrt"], space["title_sqrt"]),
                              ("aa", space["abstract_sqrt"], space["abstract_sqrt"])):
        f[f"{name}_idf"] = row_dot(left, right, qix, cix)
    tu = space["title_unique"]
    f["tt_jaccard"] = f["tt_n"] / np.maximum(tu[qix] + tu[cix] - f["tt_n"], 1.0)

    f["q_len"] = query_len.astype(np.float32)
    f["c_len"] = cand_len.astype(np.float32)
    f["len_ratio"] = np.minimum(query_len, cand_len) / np.maximum(np.maximum(query_len, cand_len), 1.0)
    f["len_diff"] = np.abs(query_len - cand_len).astype(np.float32)

    f["rare_overlap"] = row_dot(space["rare_binary"], space["rare_binary"], qix, cix)
    f["q_rare_frac"] = (np.asarray(space["rare_binary"].sum(axis=1)).ravel()[qix] / np.maximum(uq[qix], 1.0)).astype(np.float32)

    f["ngram_cos"] = row_dot(space["all_ngram"], space["all_ngram"], qix, cix)

    # LSA cosines at several ranks + field-crossed
    for c in (16, 32, 64, 96, 128, 192):
        q = space["all_lsa"][qix, :c]; d = space["all_lsa"][cix, :c]
        f[f"lsa{c}_cos"] = (np.sum(q * d, axis=1) / np.maximum(np.linalg.norm(q, axis=1) * np.linalg.norm(d, axis=1), 1e-8)).astype(np.float32)
    f["lsa_tt"] = np.sum(space["title_lsa"][qix] * space["title_lsa"][cix], axis=1).astype(np.float32)
    f["lsa_ta"] = np.sum(space["title_lsa"][qix] * space["abstract_lsa"][cix], axis=1).astype(np.float32)
    f["lsa_aa"] = np.sum(space["abstract_lsa"][qix] * space["abstract_lsa"][cix], axis=1).astype(np.float32)

    # per-document stats (query & candidate)
    def stat(name, values):
        q = values[qix].astype(np.float32); c = values[cix].astype(np.float32)
        f[f"q_{name}"] = q; f[f"c_{name}"] = c; f[f"diff_{name}"] = np.abs(q - c)
    stat("len", space["doc_len"])
    stat("uniq", space["unique_count"])
    stat("idf_mean", space["idf_mass"] / np.maximum(space["unique_count"], 1.0))
    stat("title_idf_mean", space["title_idf_mass"] / np.maximum(space["title_unique"], 1.0))
    stat("abs_idf_mean", space["abstract_idf_mass"] / np.maximum(space["abstract_unique"], 1.0))

    # semantic-beyond-lexical
    lsa160 = space["lsa160"]
    lsa160_cos = np.sum(lsa160[qix] * lsa160[cix], axis=1).astype(np.float32)
    f["lsa160_cos"] = lsa160_cos
    f["sem_minus_jacc"] = lsa160_cos - f["jaccard"]
    f["sem_minus_qcontain"] = lsa160_cos - f["q_contain"]
    f["allcos_minus_jacc"] = f["all_cos"] - f["jaccard"]
    tt_jacc = f["tt_n"] / np.maximum(tu[qix] + tu[cix] - f["tt_n"], 1.0)
    f["tt_sem_minus_jacc"] = f["tt_cos"] - tt_jacc
    tl, al = space["title_lsa"], space["abstract_lsa"]
    f["lsa_qtitle_cabs"] = np.sum(tl[qix] * al[cix], axis=1).astype(np.float32)
    f["lsa_qabs_ctitle"] = np.sum(al[qix] * tl[cix], axis=1).astype(np.float32)
    f["rare_cos_r75"] = row_dot(space["rare_75"], space["rare_75"], qix, cix)
    rare_inter = row_dot(space["rare_binary_90"], space["rare_binary_90"], qix, cix)
    f["rare_recall_r90"] = (rare_inter / np.maximum(space["rare_q90"][qix], 1.0)).astype(np.float32)

    # most-discriminative shared token
    doc_tokens, vocab = space["doc_tokens"], space["vocab"]
    max_shared = np.zeros(len(qix), np.float32)
    top3_shared = np.zeros(len(qix), np.float32)
    for i in range(len(qix)):
        shared = doc_tokens[qix[i]] & doc_tokens[cix[i]]
        # Skip out-of-vocabulary tokens: the vocabulary is fit on train docs only,
        # so a test document may share a token that was never seen in training.
        vals = [idf[vocab[t]] for t in shared if t in vocab]
        if vals:
            vals = np.asarray(vals, np.float32)
            max_shared[i] = vals.max()
            top3_shared[i] = np.sort(vals)[::-1][:3].mean()
    f["max_shared_idf"] = max_shared
    f["top3_shared_idf"] = top3_shared

    # embeddings
    ef, et, ea = space["emb_full"], space["emb_title"], space["emb_abstract"]
    f["emb_full_cos"] = np.sum(ef[qix] * ef[cix], axis=1).astype(np.float32)
    f["emb_title_cos"] = np.sum(et[qix] * et[cix], axis=1).astype(np.float32)
    f["emb_abs_cos"] = np.sum(ea[qix] * ea[cix], axis=1).astype(np.float32)
    f["emb_qtitle_cabs"] = np.sum(et[qix] * ea[cix], axis=1).astype(np.float32)
    f["emb_qabs_ctitle"] = np.sum(ea[qix] * et[cix], axis=1).astype(np.float32)
    f["emb_minus_jacc"] = f["emb_full_cos"] - f["jaccard"]
    f["emb_minus_qcontain"] = f["emb_full_cos"] - f["q_contain"]
    f["emb_title_minus_jacc"] = f["emb_title_cos"] - f["jaccard"]
    f["emb_cross_minus_jacc"] = f["emb_qtitle_cabs"] - f["jaccard"]

    # NMF topic affinities
    _nmf_pair_features(space["topics"], qix, cix, f, space, idf)

    # soft token matching (MaxSim)
    _softmatch_features(space, qix, cix, f)

    frame = pd.DataFrame(f, dtype=np.float32)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _nmf_pair_features(topics, qix, cix, f, space, idf):
    row_sum = topics.sum(axis=1, keepdims=True)
    prob = topics / np.maximum(row_sum, 1e-12)
    sqrt_prob = np.sqrt(prob)
    unit = topics / np.maximum(np.linalg.norm(topics, axis=1, keepdims=True), 1e-12)
    log_prob = np.log(np.maximum(prob, 1e-12))
    entropy = -(prob * log_prob).sum(axis=1)
    dominant = np.argmax(topics, axis=1)
    dominant_mass = prob[np.arange(len(prob)), dominant]
    pq, pc = prob[qix], prob[cix]
    mixture = 0.5 * (pq + pc)
    log_mix = np.log(np.maximum(mixture, 1e-12))
    topic_cos = (unit[qix] * unit[cix]).sum(axis=1)
    f["nmf_topic_cos"] = topic_cos.astype(np.float32)
    jsd = 0.5 * (pq * (log_prob[qix] - log_mix)).sum(axis=1) + 0.5 * (pc * (log_prob[cix] - log_mix)).sum(axis=1)
    f["nmf_jsd"] = jsd.astype(np.float32)
    f["nmf_jsdist"] = np.sqrt(np.maximum(jsd, 0.0)).astype(np.float32)
    f["nmf_hellinger"] = np.sqrt(np.maximum(0.5 * ((sqrt_prob[qix] - sqrt_prob[cix]) ** 2).sum(axis=1), 0.0)).astype(np.float32)
    f["nmf_bhatt"] = (sqrt_prob[qix] * sqrt_prob[cix]).sum(axis=1).astype(np.float32)
    f["nmf_ent_q"] = entropy[qix].astype(np.float32)
    f["nmf_ent_c"] = entropy[cix].astype(np.float32)
    f["nmf_ent_absdiff"] = np.abs(entropy[qix] - entropy[cix]).astype(np.float32)
    f["nmf_ent_min"] = np.minimum(entropy[qix], entropy[cix]).astype(np.float32)
    f["nmf_dommass_q"] = dominant_mass[qix].astype(np.float32)
    f["nmf_dommass_c"] = dominant_mass[cix].astype(np.float32)
    same = (dominant[qix] == dominant[cix]).astype(np.float64)
    f["nmf_same_dom"] = same.astype(np.float32)
    f["nmf_same_dom_mass"] = (same * dominant_mass[qix] * dominant_mass[cix]).astype(np.float32)
    f["nmf_cmass_on_qdom"] = prob[cix, dominant[qix]].astype(np.float32)
    f["nmf_qmass_on_cdom"] = prob[qix, dominant[cix]].astype(np.float32)
    f["nmf_cos_minus_jacc"] = (topic_cos - f["jaccard"]).astype(np.float32)
    f["nmf_bhatt_minus_jacc"] = (f["nmf_bhatt"] - f["jaccard"]).astype(np.float32)


def _softmatch_features(space, qix, cix, f):
    emb = space["token_emb"]
    thr = 0.5

    def maxsim(field, src, dst, softonly=False):
        index, token_idf, mask = space["topk"][field]
        n = src.shape[0]
        idf_mean = np.zeros(n, np.float32); cover = np.zeros(n, np.float32)
        soft_mean = np.zeros(n, np.float32); soft_frac = np.zeros(n, np.float32)
        for s in range(0, n, 4000):
            e = min(s + 4000, n)
            si, di = src[s:e], dst[s:e]
            qi, qm, qw = index[si], mask[si], token_idf[si]
            ci, cm = index[di], mask[di]
            with np.errstate(over="ignore", invalid="ignore"):
                sim = np.matmul(emb[qi].astype(np.float64), np.transpose(emb[ci], (0, 2, 1)).astype(np.float64))
            sim = np.nan_to_num(sim)
            sim = np.where(cm[:, None, :], sim, -1e9)
            best = sim.max(axis=2)
            best = np.where(best < -1.0, 0.0, best)
            best = np.clip(np.where(qm, best, 0.0), -1.0, 1.0).astype(np.float32)
            w = np.where(qm, qw, 0.0)
            idf_mean[s:e] = (best * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)
            dn = qm.sum(axis=1).astype(np.float32)
            cover[s:e] = (np.where(qm, best, -1.0) > thr).sum(axis=1) / np.maximum(dn, 1e-6)
            if softonly:
                exact = ((qi[:, :, None] == ci[:, None, :]) & cm[:, None, :]).any(axis=2) & qm
                soft = qm & ~exact
                sd = soft.sum(axis=1).astype(np.float32)
                soft_mean[s:e] = np.where(soft, best, 0.0).sum(axis=1) / np.maximum(sd, 1e-6)
                soft_frac[s:e] = (np.where(soft, best, -1.0) > thr).sum(axis=1) / np.maximum(dn, 1e-6)
        return idf_mean, cover, soft_mean, soft_frac

    im, cov, som, sof = maxsim("combined", qix, cix, softonly=True)
    f["sm_q2c_idf"] = im; f["sm_q2c_cov50"] = cov
    f["sm_q2c_softmean"] = som; f["sm_q2c_softfrac"] = sof
    rim, rcov, _, _ = maxsim("combined", cix, qix)
    f["sm_c2q_idf"] = rim; f["sm_c2q_cov50"] = rcov
    tim, tcov, _, _ = maxsim("title", qix, cix)
    f["sm_title_idf"] = tim; f["sm_title_cov50"] = tcov
    aim, _, _, _ = maxsim("abstract", qix, cix)
    f["sm_abs_idf"] = aim
    f["sm_q2c_idf_minus_jacc"] = f["sm_q2c_idf"] - f["jaccard"]
    f["sm_q2c_cov_minus_qcontain"] = f["sm_q2c_cov50"] - f["q_contain"]


def add_slate_features(frame: pd.DataFrame, slate_ids: np.ndarray) -> pd.DataFrame:
    """Slate-relative distractor discriminators. Ranks/z are computed WITHIN each
    slate only (one ranking example); never pooled across different slates, so this
    is valid per-example inference for test slates."""
    def rank_z(col):
        v = frame[col].to_numpy()
        g = pd.DataFrame({"s": slate_ids, "v": v}).groupby("s", sort=False)["v"]
        rank = g.rank(method="average").to_numpy() - 1.0
        z = (v - g.transform("mean").to_numpy()) / np.maximum(g.transform("std").to_numpy(), 1e-8)
        return rank.astype(np.float32), z.astype(np.float32)

    sem = [c for c in ("emb_full_cos", "nmf_topic_cos", "all_cos", "lsa160_cos") if c in frame.columns]
    lex = [c for c in ("jaccard", "bm25_qc", "overlap_n", "overlap_idf") if c in frame.columns]
    sem_rank = np.mean([rank_z(c)[0] for c in sem], axis=0)
    lex_rank = np.mean([rank_z(c)[0] for c in lex], axis=0)
    out = {"ds_lex_minus_sem_rank": (lex_rank - sem_rank).astype(np.float32)}
    emb_rank, emb_z = rank_z("emb_full_cos")
    for c in lex:
        lr, _ = rank_z(c)
        out[f"ds_emb_minus_{c}_rank"] = (lr - emb_rank).astype(np.float32)
    out["ds_emb_rank"] = emb_rank
    out["ds_emb_z"] = emb_z
    nr, _ = rank_z("nmf_topic_cos"); jr, _ = rank_z("jaccard")
    out["ds_nmf_minus_jacc_rank"] = (jr - nr).astype(np.float32)
    slate = pd.DataFrame(out, dtype=np.float32)
    combined = pd.concat([frame.reset_index(drop=True), slate], axis=1)
    return combined.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def zscore_by_slate(scores: np.ndarray, slate_ids: np.ndarray) -> np.ndarray:
    g = pd.DataFrame({"s": slate_ids, "v": scores}).groupby("s", sort=False)["v"]
    return ((scores - g.transform("mean").to_numpy()) / np.maximum(g.transform("std").to_numpy(), 1e-8)).astype(np.float32)


def track_flags(train: pd.DataFrame, train_features: pd.DataFrame) -> dict:
    slates = train["slate_id"].astype(str).to_numpy()
    labels = train["relevance_gain"].to_numpy()
    overlap = train_features["overlap_n"].to_numpy()
    qrare = train_features["q_rare_frac"].to_numpy()
    qtok = train["query_token_count"].to_numpy()
    frame = pd.DataFrame({"s": slates, "g": labels, "ov": overlap, "qr": qrare, "qt": qtok})
    distract, longq, rare = {}, {}, {}
    for s, gp in frame.groupby("s", sort=False):
        lab = gp["g"].to_numpy(); ov = gp["ov"].to_numpy()
        g2, g1 = ov[lab == 2], ov[lab == 1]
        distract[s] = len(g2) > 0 and len(g1) > 0 and g1.max() > g2.max()
        longq[s] = gp["qt"].iloc[0] >= 125
        rare[s] = gp["qr"].iloc[0]
    threshold = np.quantile(np.fromiter(rare.values(), dtype=np.float64), 0.75)
    order = pd.unique(slates)
    return {"order": order,
            "distract": np.array([distract[s] for s in order]),
            "long": np.array([longq[s] for s in order]),
            "rare": np.array([rare[s] >= threshold for s in order])}


def composite_metric(labels, scores, slate_ids, flags):
    frame = pd.DataFrame({"s": slate_ids, "l": labels.astype(np.float64), "sc": scores})
    disc = 1.0 / np.log2(np.arange(2, 5))
    ndcg, hit = {}, {}
    for s, g in frame.groupby("s", sort=False):
        gains = np.power(2.0, g["l"].to_numpy()) - 1.0
        rank = np.argsort(-g["sc"].to_numpy(), kind="stable")
        top = gains[rank][:3]; ideal = np.sort(gains)[::-1][:3]
        idcg = float(np.sum(ideal * disc[:len(ideal)]))
        ndcg[s] = float(np.sum(top * disc[:len(top)])) / idcg if idcg > 0 else 0.0
        hit[s] = 1.0 if g["l"].to_numpy()[rank[0]] == 2 else 0.0
    order = flags["order"]
    nd = np.array([ndcg[s] for s in order])
    base = float(nd.mean()); tophit = float(np.mean([hit[s] for s in order]))
    rare = float(nd[flags["rare"]].mean()) if flags["rare"].any() else base
    dis = float(nd[flags["distract"]].mean()) if flags["distract"].any() else base
    lon = float(nd[flags["long"]].mean()) if flags["long"].any() else base
    worst = min(rare, dis, lon)
    comp = 0.5 * base + 0.2 * tophit + 0.1 * rare + 0.1 * dis + 0.05 * lon + 0.05 * worst
    return comp, base, tophit


def train_and_predict(train_features, test_features, train, folds):
    labels = train["relevance_gain"].to_numpy(dtype=np.int64)
    slates = train["slate_id"].astype(str).to_numpy()
    rank_test = np.zeros(len(test_features), dtype=np.float32)
    rank_oof = np.full(len(train_features), np.nan, dtype=np.float32)
    for k, (tr_i, va_i) in enumerate(folds):
        to = tr_i[np.argsort(slates[tr_i], kind="stable")]
        vo = va_i[np.argsort(slates[va_i], kind="stable")]
        model = CatBoostRanker(iterations=600, depth=6, learning_rate=0.04, loss_function="YetiRank",
                               eval_metric="NDCG:top=3", l2_leaf_reg=5.0, random_strength=0.5,
                               random_seed=SEED + k, thread_count=-1, verbose=False, allow_writing_files=False)
        model.fit(Pool(train_features.iloc[to], labels[to], group_id=slates[to]),
                  eval_set=Pool(train_features.iloc[vo], labels[vo], group_id=slates[vo]),
                  early_stopping_rounds=70, verbose=False)
        rank_oof[va_i] = model.predict(train_features.iloc[va_i]).astype(np.float32)
        rank_test += model.predict(test_features).astype(np.float32) / len(folds)
        print(f"fold {k + 1}/{len(folds)} trained", flush=True)
    flags = track_flags(train, train_features)
    comp, base, hit = composite_metric(labels, zscore_by_slate(rank_oof, slates), slates, flags)
    print(f"OOF composite {comp:.4f} (ndcg3 {base:.4f}, tophit {hit:.4f}) on {train_features.shape[1]} features", flush=True)
    return rank_test


def make_submission(test, scores):
    ranked = test[["slate_id", "candidate_doc_id"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(["slate_id", "score", "candidate_doc_id"], ascending=[True, False, True], kind="stable")
    return (ranked.groupby("slate_id", sort=False)["candidate_doc_id"].agg(" ".join)
            .rename("ranked_candidate_doc_ids").reset_index())


def validate_submission(test, submission):
    if set(test["slate_id"].astype(str)) != set(submission["slate_id"].astype(str)) or len(submission) != test["slate_id"].nunique():
        raise ValueError("Submission slate IDs do not match test.csv")
    expected = test.groupby("slate_id", sort=False)["candidate_doc_id"].agg(set)
    submitted = submission.set_index("slate_id")["ranked_candidate_doc_ids"]
    for slate_id, cands in expected.items():
        ranking = submitted.loc[slate_id].split()
        if len(ranking) != 12 or len(set(ranking)) != 12 or set(ranking) != cands:
            raise ValueError(f"Slate {slate_id} invalid")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])

    documents = pd.read_csv(public_dir / "documents.csv")
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")

    doc_ids = documents["doc_id"].astype(str).tolist()
    doc_to_row = {d: i for i, d in enumerate(doc_ids)}

    # Documents referenced by TRAIN only define the fitting corpus. Test-only
    # documents are never used to fit anything (see leakage statement in readme).
    train_doc_ids = set(train["query_doc_id"].astype(str)) | set(train["candidate_doc_id"].astype(str))
    fit_rows = np.array(sorted(doc_to_row[d] for d in train_doc_ids), dtype=np.int64)
    print(f"fitting document space on {len(fit_rows)} train-referenced documents", flush=True)
    space = build_doc_space(documents, fit_rows)

    def features_for(pairs: pd.DataFrame) -> pd.DataFrame:
        qix = pairs["query_doc_id"].astype(str).map(doc_to_row).to_numpy(np.int64)
        cix = pairs["candidate_doc_id"].astype(str).map(doc_to_row).to_numpy(np.int64)
        feats = pair_features(space, qix, cix,
                              pairs["query_token_count"].to_numpy(np.float32),
                              pairs["candidate_token_count"].to_numpy(np.float32))
        return add_slate_features(feats, pairs["slate_id"].astype(str).to_numpy())

    print("computing train pair features", flush=True)
    train_features = features_for(train)
    print("computing test pair features (inference only)", flush=True)
    test_features = features_for(test)

    folds = list(GroupKFold(n_splits=N_FOLDS).split(train, train["relevance_gain"], groups=train["query_doc_id"]))
    print(f"training {N_FOLDS} grouped rankers on {train_features.shape[1]} features", flush=True)
    scores = train_and_predict(train_features, test_features, train, folds)

    submission = make_submission(test, scores)
    validate_submission(test, submission)
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    print(f"wrote {len(submission)} ranked slates to {submission_out}", flush=True)


if __name__ == "__main__":
    main()
