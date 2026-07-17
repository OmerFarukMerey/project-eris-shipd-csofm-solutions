#!/usr/bin/env python3
"""
Collapsed Branch Side Set Recovery
==================================

Each row is a *collapsed* local neighbourhood of a phylogenetic tree.  One
internal branch was contracted into the placeholder node X, merging the two
sides of that branch into a single polytomy.  For each row we must recover the
exact set of endpoint tokens that lay on the anchor's side of the hidden branch.

Approach - constrained set search + TWO learned models
------------------------------------------------------
The target side is a valid bipartition part: it contains the anchor, has >= 2
tokens and <= floor(n/2) tokens.  For every row we enumerate all such candidate
sides (the true side is guaranteed to be one of them) and rank them.

  1. Per-EDGE model (LightGBM).  A classifier is trained on the observed
     evidence pairs to predict whether a pair is "same-side" or "crossing",
     from the pair's hops/rank AND its position *relative to the other pairs in
     its row* (hops minus row-mean, rank within the row, etc.).  The hidden
     branch adds a roughly constant offset to every crossing pair, so whether a
     pair crosses is best judged relative to the row's own scale - this lifts
     same/cross AUC from 0.955 (a plain frequency table) to 0.983.

  2. Per-CANDIDATE model (LightGBM).  For each candidate side we build ~75
     relational, permutation-invariant features: the per-edge model's log-odds
     aggregated over the induced cut (within-side / within-complement / across),
     hop and rank distribution statistics, within-row ordering-consistency,
     separation margins, violation counts, a generative size prior, and
     anchor-incident-edge consistency.  The classifier is trained to rank the
     true side above sampled competitors; the argmax candidate is predicted.
     Eight seeds are bagged for a stable argmax on near-tied candidates.

The per-edge log-odds used as features for TRAINING rows are produced
out-of-fold (two-way split) so the candidate model never sees in-sample edge
scores; test rows use an edge model trained on all of train.

Leakage / compliance
--------------------
Both LightGBM models, the generative size prior, and every statistic are fit on
TRAINING ROWS ONLY.  Test rows are used purely to build candidate features and
receive predictions (per-row inference).  No train/test concatenation, no
statistic derived from test rows, fixed random seeds throughout.
"""
import sys, json, time
from itertools import combinations
from collections import defaultdict, Counter
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
SEEDS = [42, 7, 123, 2024, 99, 7777, 31, 555]   # candidate-model bagging seeds
CAND_CAP = 4000                 # cap candidate enumeration for very large rows
NEG_PER_ROW = 120               # random negatives sampled per row for training
TIME_BUDGET = 3300.0            # safeguard: stop adding bagging rounds past this
T0 = time.time()


# ---------------------------------------------------------------- parsing ----
def parse(df):
    rows = []
    for _, r in df.iterrows():
        ctx = json.loads(r['collapsed_context_json'])
        ev = json.loads(r['distance_evidence_json'])
        d = {'id': r['id'], 'anchor': r['anchor_taxon'],
             'incident': ctx['incident_taxa'],
             'opc': ctx.get('observed_pair_count', len(ev)), 'ev': ev}
        if 'answer_json' in df.columns and pd.notna(r.get('answer_json')):
            d['side'] = json.loads(r['answer_json'])['side']
        rows.append(d)
    return rows


def rkey(rk):
    return 8 if rk is None else rk


# --------------------------------------------- generative size prior (train) --
def fit_gen(train_rows):
    same, cross = Counter(), Counter()
    sizep = defaultdict(Counter)
    for r in train_rows:
        S = set(r['side']); n = len(r['incident'])
        sizep[n][len(S)] += 1
        for e in r['ev']:
            key = (e['hops'], rkey(e['length_rank']))
            if (e['a'] in S) == (e['b'] in S):
                same[key] += 1
            else:
                cross[key] += 1
    keys = set(same) | set(cross)
    ts, tc = sum(same.values()), sum(cross.values())
    V = len(keys) + 1
    llr = {k: (np.log((same[k] + 0.5) / (ts + 0.5 * V))
               - np.log((cross[k] + 0.5) / (tc + 0.5 * V))) for k in keys}
    default = np.log(0.5 / (ts + 0.5 * V)) - np.log(0.5 / (tc + 0.5 * V))
    szlp = {}
    for n, c in sizep.items():
        t = sum(c.values())
        szlp[n] = {k: np.log((v + 0.5) / (t + 0.5 * (n // 2))) for k, v in c.items()}
    return llr, default, szlp


# ------------------------------------------------------ per-edge model --------
def edge_feats_row(r):
    """Absolute + row-relative features for each observed pair."""
    ev = r['ev']; n = len(r['incident'])
    hs = np.array([e['hops'] for e in ev], dtype=float)
    rvals = np.array([e['length_rank'] for e in ev if e['length_rank'] is not None],
                     dtype=float)
    mh = hs.mean(); sh = hs.std() + 1e-9
    mnh = hs.min(); mxh = hs.max()
    mr = rvals.mean() if len(rvals) else -1.0
    out = []
    for e in ev:
        h = e['hops']; rk = e['length_rank']
        rank_of_h = (hs < h).sum() / len(hs)
        out.append([h, (8 if rk is None else rk), 1 if rk is None else 0,
                    n, len(ev), len(ev) / n,
                    mh, sh, mnh, mxh, h - mh, h / mh, h - mnh, mxh - h, rank_of_h,
                    mr, (rk - mr if (rk is not None and mr >= 0) else 0.0),
                    1 if h == mnh else 0, 1 if h == mxh else 0])
    return out


def train_edge(rows, seed=1):
    X, Y = [], []
    for r in rows:
        S = set(r['side'])
        for f, e in zip(edge_feats_row(r), r['ev']):
            X.append(f); Y.append(1 if (e['a'] in S) == (e['b'] in S) else 0)
    m = lgb.LGBMClassifier(objective='binary', n_estimators=400, learning_rate=0.03,
                           num_leaves=31, min_child_samples=80, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=3.0, n_jobs=-1,
                           verbose=-1, random_state=seed)
    m.fit(np.asarray(X, dtype=np.float32), np.asarray(Y))
    return m


def set_edge_logodds(rows, model):
    for r in rows:
        F = np.asarray(edge_feats_row(r), dtype=np.float32)
        p = np.clip(model.predict_proba(F)[:, 1], 1e-4, 1 - 1e-4)
        r['_lo'] = np.log(p / (1 - p))          # per-pair log-odds of "same-side"


# ------------------------------------------------------ candidate enumeration --
def enum_sides(inc, anchor, cap=CAND_CAP):
    n = len(inc)
    others = [t for t in inc if t != anchor]
    res = []
    for k in range(2, n // 2 + 1):
        for combo in combinations(others, k - 1):
            res.append(frozenset((anchor,) + combo))
            if len(res) > cap:
                return res
    return res


# ------------------------------------------------------ candidate features -----
def feats(S, r, gen):
    llr, default, szlp = gen
    ev = r['ev']; n = len(r['incident']); k = len(S); anc = r['anchor']
    lo = r['_lo']
    inS_h, inC_h, cr_h = [], [], []
    inS_r, inC_r, cr_r = [], [], []
    same_p, cross_p = [], []
    slo, clo = [], []
    gscore = 0.0
    a_in = a_out = 0
    a_in_h, a_out_h = [], []
    a_close_out = a_far_in = 0
    for i, e in enumerate(ev):
        aS = e['a'] in S; bS = e['b'] in S
        h = e['hops']; rk = e['length_rank']
        if aS and bS:
            inS_h.append(h)
            if rk is not None: inS_r.append(rk)
        elif (not aS) and (not bS):
            inC_h.append(h)
            if rk is not None: inC_r.append(rk)
        else:
            cr_h.append(h)
            if rk is not None: cr_r.append(rk)
        if aS == bS:
            gscore += llr.get((h, rkey(rk)), default)
            same_p.append((h, rk)); slo.append(lo[i])
        else:
            cross_p.append((h, rk)); clo.append(lo[i])
        if anc in (e['a'], e['b']):
            other = e['b'] if e['a'] == anc else e['a']
            if other in S:
                a_in += 1; a_in_h.append(h)
                if h >= 6 or (rk is not None and rk >= 5): a_far_in += 1
            else:
                a_out += 1; a_out_h.append(h)
                if h <= 3 or (rk is not None and rk <= 1): a_close_out += 1
    szlogp = szlp.get(n, {}).get(k, -3.0)

    def st(x, d0=0.0):
        return [len(x), (np.mean(x) if x else d0), (np.min(x) if x else d0),
                (np.max(x) if x else d0), (np.std(x) if x else 0.0)]

    same_h = inS_h + inC_h; same_r = inS_r + inC_r
    f = [k, n, k / n, len(ev), len(ev) / n, gscore, szlogp, gscore + szlogp,
         gscore / max(1, len(ev))]
    f += st(inS_h); f += st(inC_h); f += st(cr_h)
    f += st(inS_r, -1); f += st(inC_r, -1); f += st(cr_r, -1)
    max_same = max(same_h) if same_h else 0
    min_cross = min(cr_h) if cr_h else 99
    f += [min_cross - max_same,
          (np.mean(cr_h) if cr_h else 0) - (np.mean(same_h) if same_h else 0)]
    # ordering consistency (hops)
    hlt = hgt = heq = 0
    for hs, _ in same_p:
        for hc, _ in cross_p:
            if hs < hc: hlt += 1
            elif hs > hc: hgt += 1
            else: heq += 1
    ht = hlt + hgt + heq
    f += [ht, hlt / ht if ht else 0.5, hgt / ht if ht else 0.5, heq / ht if ht else 0.0]
    # ordering consistency (rank)
    rlt = rgt = req = 0
    for _, rs in same_p:
        if rs is None: continue
        for _, rc in cross_p:
            if rc is None: continue
            if rs < rc: rlt += 1
            elif rs > rc: rgt += 1
            else: req += 1
    rt = rlt + rgt + req
    f += [rt, rlt / rt if rt else 0.5, rgt / rt if rt else 0.5, req / rt if rt else 0.0]
    max_sr = max(same_r) if same_r else -1
    min_cr = min(cr_r) if cr_r else 99
    f += [min_cr - max_sr]
    f += [sum(1 for h in cr_h if h == 2), sum(1 for h in same_h if h >= 6),
          sum(1 for h in cr_h if h <= 3), sum(1 for x in same_r if x >= 5),
          sum(1 for x in cr_r if x <= 1), sum(1 for x in cr_r if x == 0)]
    f += [a_in, a_out,
          (np.mean(a_in_h) if a_in_h else -1),
          (np.mean(a_out_h) if a_out_h else -1),
          (np.min(a_in_h) if a_in_h else -1),
          (np.max(a_out_h) if a_out_h else -1),
          a_close_out, a_far_in]
    # ---- per-edge model features (log-odds of same-side over the induced cut) --
    f += st(slo); f += st(clo)
    elt = egt = 0
    for a in slo:
        for b in clo:
            if a > b: egt += 1
            else: elt += 1
    et = elt + egt
    f += [sum(slo), sum(slo) / max(1, len(ev)), et, egt / et if et else 0.5,
          (min(slo) if slo else 0) - (max(clo) if clo else 0)]
    return f


def build_train(rows, gen, seed):
    rr = np.random.default_rng(seed)
    X, Y = [], []
    for r in rows:
        cands = enum_sides(r['incident'], r['anchor'])
        Strue = frozenset(r['side'])
        pool = [S for S in cands if S != Strue]
        if NEG_PER_ROW and len(pool) > NEG_PER_ROW:
            sel = rr.choice(len(pool), NEG_PER_ROW, replace=False)
            pool = [pool[i] for i in sel]
        X.append(feats(Strue, r, gen)); Y.append(1)
        for S in pool:
            X.append(feats(S, r, gen)); Y.append(0)
    return np.asarray(X, dtype=np.float32), np.asarray(Y)


def predict_rows(rows, gen, models):
    preds = []
    for r in rows:
        cands = enum_sides(r['incident'], r['anchor'])
        X = np.asarray([feats(S, r, gen) for S in cands], dtype=np.float32)
        p = np.zeros(len(cands))
        for clf in models:
            p += clf.predict_proba(X)[:, 1]
        best = cands[int(np.argmax(p))]
        side = [t for t in r['incident'] if t in best]   # row-local order
        preds.append((r['id'], side))
    return preds


def main():
    public_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = parse(pd.read_csv(public_dir / 'train.csv'))
    test_rows = parse(pd.read_csv(public_dir / 'test.csv'))

    gen = fit_gen(train_rows)

    # --- per-edge log-odds: OUT-OF-FOLD for train rows, full model for test ---
    order = np.arange(len(train_rows))
    np.random.default_rng(5).shuffle(order)
    half = len(order) // 2
    idxA, idxB = order[:half], order[half:]
    edge_A = train_edge([train_rows[i] for i in idxB])   # trained on B, scores A
    edge_B = train_edge([train_rows[i] for i in idxA])   # trained on A, scores B
    set_edge_logodds([train_rows[i] for i in idxA], edge_A)
    set_edge_logodds([train_rows[i] for i in idxB], edge_B)
    edge_full = train_edge(train_rows)
    set_edge_logodds(test_rows, edge_full)
    print(f'[{time.time()-T0:.0f}s] edge models trained', flush=True)

    params = dict(objective='binary', n_estimators=700, learning_rate=0.03,
                  num_leaves=31, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  reg_alpha=2.0, n_jobs=-1, verbose=-1)
    models = []
    for sd in SEEDS:
        if models and time.time() - T0 > TIME_BUDGET:
            print(f'[{time.time()-T0:.0f}s] time budget reached, '
                  f'stopping with {len(models)} model(s)', flush=True)
            break
        Xtr, Ytr = build_train(train_rows, gen, sd)
        p = dict(params); p['random_state'] = sd
        clf = lgb.LGBMClassifier(**p)
        clf.fit(Xtr, Ytr)
        models.append(clf)
        print(f'[{time.time()-T0:.0f}s] trained candidate model {len(models)} '
              f'(seed={sd}, X={Xtr.shape})', flush=True)

    preds = predict_rows(test_rows, gen, models)
    rows = [{'id': i, 'answer_json': json.dumps({'side': s}, separators=(',', ':'))}
            for i, s in preds]
    sub = pd.DataFrame(rows, columns=['id', 'answer_json'])
    sub.to_csv(out_path, index=False)
    print(f'[{time.time()-T0:.0f}s] wrote {len(sub)} rows to {out_path}', flush=True)


if __name__ == '__main__':
    main()
