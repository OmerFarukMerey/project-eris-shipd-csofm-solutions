#!/usr/bin/env python3
"""Swadesh Phoneme Cipher Decoding.

Unsupervised decipherment of a hidden Uralic language whose IPA has been
enciphered by a single global one-to-one substitution (each opaque token
x{N} -> exactly one true IPA segment, shared across every test row).

Approach (all learning happens inside this script, from the provided data only):

  1. Cross-lingual cognate alignment (EM-style refinement).  The enciphered
     lexicon is aligned, concept by concept, against the true-IPA wordlists of
     the target's Uralic relatives.  Each alignment position casts a weighted
     vote token -> segment.  The token->segment map, the per-relative reliability
     weights, and the alignments are re-estimated jointly for several iterations
     (iterative / EM-style refinement under the global one-to-one constraint --
     exactly the class of method the challenge asks for).  This yields, per
     token, an evidence matrix of votes.

     The per-relative weighting is sharp (mean-similarity ** 6): a divergent target
     -- one whose own sounds match only its single nearest relative -- needs the
     closest relative trusted decisively, otherwise distant families contaminate
     the frequent tokens.

  2. A from-scratch neural ranking network (trained ML component).  It is trained
     by CIPHER SIMULATION self-supervision on the train languages only: every
     train Uralic language is, in turn, enciphered with a random substitution and
     decoded exactly as the real target is; because we know that language's true
     IPA, the true token->segment answer is known -- a fully supervised signal
     with NO test labels.  The network scores each candidate segment for a token
     from its alignment evidence.

  3. Rare-token fill-in.  The frequent, well-attested tokens -- which drive the
     score -- keep the alignment's one-to-one assignment.  Only the RARE tokens
     (few cognate occurrences, where the alignment is essentially guessing) are
     re-assigned by the trained network, over the segments the frequent tokens did
     not take.  A model that overrides the frequent tokens hurts a divergent target
     (it is trained on the average cross-Uralic reflex and pulls the target's own
     sounds toward its relatives'), so it is deliberately confined to the tail.

  4. Decode every test row with the recovered global map.

Compliance: everything (vocabularies, alignment, relative weights, the neural
network) is fit on train.csv only.  The test cipher is used only to run the same
decipherment/inference procedure on it -- no statistic, count, scaler, encoder
or hyper-parameter is fit on it, and there is no train+test concatenation.
Recovering the single global substitution IS the task the challenge defines.
"""

import sys
import time
import random
import collections
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_EM_ITERS = 8          # EM-style alignment refinement iterations
WEIGHT_EXP = 4.0        # sharpness of per-relative reliability weighting.  The target keeps ALL
                        # its Finnic relatives, so aggregating them (exponent 4) beats over-
                        # focusing on a single one (measured: sharper weighting regressed).
CONF_TH = 0.5           # tokens with alignment confidence below this are decided by the neural model
SIM_SEED_BASE = 1000    # rng offset for cipher-simulation training tasks
TIME_BUDGET_S = 3000.0  # stop-and-submit guard (guidebook: ~50 min)
START = time.time()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def similarity(a, b):
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def nw_align(a, b, subcost):
    """Needleman-Wunsch; returns list of aligned index pairs (substitutions)."""
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        dpi, dpi1, bti = dp[i], dp[i - 1], bt[i]
        for j in range(1, m + 1):
            c_sub = dpi1[j - 1] + subcost(ai, b[j - 1])
            c_del = dpi1[j] + 1.0
            c_ins = dpi[j - 1] + 1.0
            best = c_sub
            d = 0
            if c_del < best:
                best = c_del
                d = 1
            if c_ins < best:
                best = c_ins
                d = 2
            dpi[j] = best
            bti[j] = d
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        d = bt[i][j]
        if d == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# --------------------------------------------------------------------------- #
# EM-style cognate-alignment decipherment
# --------------------------------------------------------------------------- #
def decipher_votes(cipher_rows, support, n_seg, seg_index, n_iter=N_EM_ITERS,
                   wexp=WEIGHT_EXP):
    """Recover token->segment evidence by iterative cognate alignment.

    cipher_rows : list of (concept, [tokens])
    support     : dict concept -> list of (lang, [segments])
    Returns tokens list, provisional map, and vote matrices (overall + tiers).
    """
    tokens = sorted({t for _, toks in cipher_rows for t in toks})
    tok_index = {t: i for i, t in enumerate(tokens)}
    n_tok = len(tokens)
    cur_map = {}                                   # token -> segment (provisional)
    lang_weight = collections.defaultdict(lambda: 1.0)
    langs = sorted({l for c in support for l, _ in support[c]})

    def subcost(tok, seg):
        if tok in cur_map:
            return 0.0 if cur_map[tok] == seg else 1.0
        return 0.9

    from scipy.optimize import linear_sum_assignment

    votes = np.zeros((n_tok, n_seg))
    for it in range(n_iter):
        votes = np.zeros((n_tok, n_seg))
        for concept, toks in cipher_rows:
            forms = support.get(concept)
            if not forms:
                continue
            for lang, segs in forms:
                w = lang_weight[lang]
                pairs = nw_align(toks, segs, subcost)
                if cur_map:
                    agree = sum(1 for (i, j) in pairs
                                if toks[i] in cur_map and cur_map[toks[i]] == segs[j])
                    conf = agree / max(len(toks), len(segs))
                else:
                    conf = 0.3
                wc = w * (0.2 + conf)
                for (i, j) in pairs:
                    votes[tok_index[toks[i]], seg_index[segs[j]]] += wc
        # global one-to-one assignment (maximise votes)
        rows, cols = linear_sum_assignment(-votes)
        cur_map = {}
        for ri, ci in zip(rows, cols):
            if votes[ri, ci] > 0:
                cur_map[tokens[ri]] = seg_index.inv[ci]
        for k, t in enumerate(tokens):
            if t not in cur_map:
                cur_map[t] = seg_index.inv[int(np.argmax(votes[k]))]
        # re-estimate per-relative reliability from current decode
        acc = collections.defaultdict(list)
        for concept, toks in cipher_rows:
            forms = support.get(concept)
            if not forms:
                continue
            dec = [cur_map.get(t, "?") for t in toks]
            for lang, segs in forms:
                acc[lang].append(similarity(dec, segs))
        for lang in acc:
            lang_weight[lang] = max(0.02, float(np.mean(acc[lang]))) ** wexp

    # final pass: record votes split by relative-closeness tier (features for the ranker)
    ranked = sorted(langs, key=lambda l: -lang_weight[l])
    top1 = set(ranked[:1])
    top3 = set(ranked[:3])
    top6 = set(ranked[:6])
    V = np.zeros((n_tok, n_seg))
    V1 = np.zeros((n_tok, n_seg))
    V3 = np.zeros((n_tok, n_seg))
    V6 = np.zeros((n_tok, n_seg))
    for concept, toks in cipher_rows:
        forms = support.get(concept)
        if not forms:
            continue
        for lang, segs in forms:
            w = lang_weight[lang]
            pairs = nw_align(toks, segs, subcost)
            agree = sum(1 for (i, j) in pairs
                        if toks[i] in cur_map and cur_map[toks[i]] == segs[j])
            conf = agree / max(len(toks), len(segs))
            wc = w * (0.2 + conf)
            for (i, j) in pairs:
                a = tok_index[toks[i]]
                b = seg_index[segs[j]]
                V[a, b] += wc
                if lang in top1:
                    V1[a, b] += 1.0
                if lang in top3:
                    V3[a, b] += 1.0
                if lang in top6:
                    V6[a, b] += 1.0
    freq = collections.Counter(t for _, toks in cipher_rows for t in toks)
    return tokens, cur_map, {"V": V, "V1": V1, "V3": V3, "V6": V6}, dict(freq)


# --------------------------------------------------------------------------- #
# feature extraction for the neural ranker
# --------------------------------------------------------------------------- #
N_FEAT = 14
TOPK = 80


def token_candidates(vote, seg_prior, freq_tok, ti):
    """Return (candidate_seg_indices, feature_matrix, confidence) for one token."""
    V = vote["V"][ti]
    V1 = vote["V1"][ti]
    V3 = vote["V3"][ti]
    V6 = vote["V6"][ti]
    sa = V.sum()
    s1 = V1.sum()
    s3 = V3.sum()
    s6 = V6.sum()
    order = np.argsort(-V)
    nz = int((V > 0).sum())
    cands = list(order[:max(TOPK, min(nz, 120))])
    for arr in (V1, V3, V6):
        am = int(np.argmax(arr))
        if arr[am] > 0 and am not in cands:
            cands.append(am)
    conf = V[order[0]] / sa if sa > 0 else 0.0
    p = V / sa if sa > 0 else V
    ent = -float(np.sum(p * np.log(p + 1e-12))) if sa > 0 else 0.0
    am_all = int(order[0])
    am1 = int(np.argmax(V1))
    am3 = int(np.argmax(V3))
    am6 = int(np.argmax(V6))
    lf = np.log1p(freq_tok) / 7.0
    feats = []
    for rank, si in enumerate(cands):
        va = V[si] / sa if sa > 0 else 0.0
        v1 = V1[si] / s1 if s1 > 0 else 0.0
        feats.append([
            va, rank / TOPK,
            v1, V3[si] / s3 if s3 > 0 else 0.0, V6[si] / s6 if s6 > 0 else 0.0,
            1.0 if si == am_all else 0.0,
            1.0 if si == am1 else 0.0,
            1.0 if si == am3 else 0.0,
            1.0 if si == am6 else 0.0,
            conf, min(ent, 4.0) / 4.0, lf,
            seg_prior[si] * 20.0,
            va - v1,
        ])
    return cands, np.asarray(feats, dtype=np.float32), conf


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
class InvertibleIndex(dict):
    """dict segment->idx that also exposes .inv (idx->segment)."""
    def __init__(self, items):
        super().__init__({s: i for i, s in enumerate(items)})
        self.inv = list(items)


def main():
    if len(sys.argv) >= 3:
        public_dir = Path(sys.argv[1])
        submission_out = Path(sys.argv[2])
    else:
        public_dir = Path("dataset/public")
        submission_out = Path("working/submission.csv")
    seed_everything(SEED)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")

    # ----- write a schema-valid placeholder immediately (the platform may probe the
    # output path before a long run finishes); it is overwritten with real predictions.
    def write_submission(mapping_rows):
        submission_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mapping_rows, columns=["id", "ipa"]).to_csv(submission_out, index=False)

    placeholder = [(rid, "a") for rid in test["id"].tolist()]
    write_submission(placeholder)

    # ----- restrict crib material to the stated family (Uralic) -----
    uralic = train.loc[train["family"].eq("Uralic")].copy()
    uralic["segments"] = uralic["ipa"].str.split()
    uralic = uralic[uralic["segments"].map(lambda s: isinstance(s, list) and len(s) > 0)]

    global_segments = sorted({s for segs in uralic["segments"] for s in segs})
    seg_index = InvertibleIndex(global_segments)
    n_seg = len(global_segments)

    ur_langs = sorted(uralic["language"].unique())
    # concept -> list of (lang, segments)
    forms_by_concept = collections.defaultdict(list)
    lang_forms = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in uralic[["language", "concept", "segments"]].itertuples(index=False):
        forms_by_concept[row.concept].append((row.language, row.segments))
        lang_forms[row.language][row.concept].append(row.segments)

    # segment prior (train-only marginal, used as a mild ranker feature)
    seg_prior = np.zeros(n_seg)
    for segs in uralic["segments"]:
        for s in segs:
            seg_prior[seg_index[s]] += 1.0
    seg_prior = seg_prior / max(seg_prior.sum(), 1.0)

    # ===================================================================== #
    # (A) Run the decipherment on the REAL enciphered lexicon.
    # ===================================================================== #
    def split_cipher(c):
        return str(c).split() if isinstance(c, str) and c.strip() else []

    test_rows = [(r.id, split_cipher(r.cipher))
                 for r in test[["id", "cipher"]].itertuples(index=False)]

    def decode_and_write(mapping):
        rows = []
        for rid, toks in test_rows:
            decoded = [d for d in (mapping.get(t) for t in toks) if d]
            rows.append((rid, " ".join(decoded) if decoded else "a"))
        write_submission(rows)

    test_cipher_rows = [(r.concept, split_cipher(r.cipher))
                        for r in test[["concept", "cipher"]].itertuples(index=False)]
    tokens, em_map, vote, freq = decipher_votes(
        test_cipher_rows, forms_by_concept, n_seg, seg_index)

    # A real (alignment-only) decode is written now, so a valid, non-placeholder
    # submission always survives even if training or the assignment below fails.
    decode_and_write(em_map)

    # ===================================================================== #
    # (B) Train the neural ranker by cipher simulation on train languages.
    # (C) Confidence-gated one-to-one assignment: the confident tokens keep the
    #     alignment map; the trained model decides the low-confidence tokens.
    # (D) Decode with the resulting map.
    # ===================================================================== #
    final_map = em_map
    try:
        trained_model = train_ranker(
            ur_langs, lang_forms, forms_by_concept, n_seg, seg_index, seg_prior)
        if trained_model is not None and (time.time() - START) < TIME_BUDGET_S:
            final_map = gated_assign(tokens, vote, seg_prior, freq,
                                     seg_index, n_seg, trained_model)
    except Exception as exc:   # keep the already-written EM decode on any failure
        sys.stderr.write("neural refinement skipped, EM decode kept: %r\n" % (exc,))

    decode_and_write(final_map)


# --------------------------------------------------------------------------- #
# neural ranker: training via cipher simulation + inference assignment
# --------------------------------------------------------------------------- #
def _make_cipher(lang_forms_lang, concept_set, rng):
    """Encipher one train language's wordlist with a random global substitution."""
    inv = sorted({s for concept in lang_forms_lang for segs in lang_forms_lang[concept]
                  for s in segs})
    perm = list(range(len(inv)))
    rng.shuffle(perm)
    seg2tok = {s: "x%d" % perm[i] for i, s in enumerate(inv)}
    tok2seg = {v: k for k, v in seg2tok.items()}
    cipher_rows = []
    for concept in lang_forms_lang:
        if concept not in concept_set:
            continue
        for segs in lang_forms_lang[concept]:
            cipher_rows.append((concept, [seg2tok[s] for s in segs]))
    return cipher_rows, tok2seg


def train_ranker(ur_langs, lang_forms, forms_by_concept, n_seg, seg_index, seg_prior):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    concept_set = set(forms_by_concept.keys())

    # ---- build training examples by simulating the cipher on each train language ----
    # Train on the LOW-CONFIDENCE regime, since that is where the model is used at
    # inference (the confident tokens are handled by the alignment assignment).
    low_examples = []
    all_examples = []
    for li, lang in enumerate(ur_langs):
        if (time.time() - START) > TIME_BUDGET_S * 0.7:
            break
        rng = random.Random(SIM_SEED_BASE + li)
        cipher_rows, tok2seg = _make_cipher(lang_forms[lang], concept_set, rng)
        # support = the OTHER train Uralic languages (mirrors the real setup)
        support = collections.defaultdict(list)
        for concept, forms in forms_by_concept.items():
            for l2, segs in forms:
                if l2 != lang:
                    support[concept].append((l2, segs))
        toks, _map, vote, freq = decipher_votes(cipher_rows, support, n_seg, seg_index)
        for ti, t in enumerate(toks):
            cands, feats, conf = token_candidates(vote, seg_prior, freq.get(t, 0), ti)
            true_seg = tok2seg.get(t)
            true_i = seg_index.get(true_seg, -1)
            y = np.asarray([1.0 if si == true_i else 0.0 for si in cands], dtype=np.float32)
            if y.sum() == 0:
                continue    # true segment not among candidates -> no learnable signal
            all_examples.append((feats, y))
            if conf < CONF_TH:
                low_examples.append((feats, y))

    examples = low_examples if len(low_examples) >= 200 else all_examples
    if len(examples) < 50:
        return None

    class Ranker(nn.Module):
        def __init__(self, din=N_FEAT, h=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(din, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
                nn.Linear(h, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)   # raw ranking logit (softmax over candidates)

    torch.manual_seed(SEED)
    model = Ranker()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    groups = [(torch.from_numpy(X), torch.from_numpy(y)) for X, y in examples]
    n = len(groups)
    EPOCHS = 25
    for ep in range(EPOCHS):
        if (time.time() - START) > TIME_BUDGET_S * 0.85:
            break
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, 64):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            cnt = 0
            for gi in perm[start:start + 64]:
                X, y = groups[gi]
                logit = model(X)                       # listwise ranking over candidate segments
                target = y / y.sum()
                loss = loss - (target * F.log_softmax(logit, 0)).sum()
                cnt += 1
            (loss / max(cnt, 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    model.eval()
    return model


def gated_assign(tokens, vote, seg_prior, freq, seg_index, n_seg, model):
    """Confidence-gated one-to-one assignment.

    High-confidence tokens keep the alignment's global Hungarian assignment.
    The trained neural model decides the low-confidence tokens, re-solving their
    assignment over the segments not already taken by the confident tokens.
    """
    import torch
    from scipy.optimize import linear_sum_assignment
    V = vote["V"]
    n_tok = len(tokens)

    # base alignment assignment (global one-to-one on votes)
    rows, cols = linear_sum_assignment(-V)
    base = {ri: ci for ri, ci in zip(rows, cols)}
    for k in range(n_tok):
        if k not in base:
            base[k] = int(np.argmax(V[k]))

    # per-token confidence + neural scores over candidate segments
    confs = np.zeros(n_tok)
    nn_scores = {}
    for ti, t in enumerate(tokens):
        cands, feats, conf = token_candidates(vote, seg_prior, freq.get(t, 0), ti)
        confs[ti] = conf
        with torch.no_grad():
            nn_scores[ti] = (cands, model(torch.from_numpy(feats)).numpy())

    high = [ti for ti in range(n_tok) if confs[ti] >= CONF_TH]
    low = [ti for ti in range(n_tok) if confs[ti] < CONF_TH]

    mapping = {}
    used = set()
    for ti in high:
        mapping[tokens[ti]] = seg_index.inv[base[ti]]
        used.add(base[ti])

    if low:
        avail = [s for s in range(n_seg) if s not in used]
        avail_pos = {s: j for j, s in enumerate(avail)}
        score_mat = np.full((len(low), len(avail)), -1e9)
        for r, ti in enumerate(low):
            cands, sc = nn_scores[ti]
            for si, v in zip(cands, sc):
                if si in avail_pos:
                    score_mat[r, avail_pos[si]] = v
        rr, cc = linear_sum_assignment(-score_mat)
        for r, c in zip(rr, cc):
            mapping[tokens[low[r]]] = seg_index.inv[avail[c]]
        for ti in low:                     # safety: keep alignment choice if unassigned
            if tokens[ti] not in mapping:
                mapping[tokens[ti]] = seg_index.inv[base[ti]]
    return mapping


if __name__ == "__main__":
    main()
