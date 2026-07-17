"""Anonymized Vocal Fragment Routing — stacked route solver.

Pipeline (all statistics fit on train only; test used for inference only):
  1. GlobalStats: melodic transition statistics from train sequences.
  2. Length model (LightGBM): predicts route cardinality from row-local
     duration features and duration-compatible subset counts.
  3. Step model (LightGBM): teacher-forced next-fragment classifier with
     engineered melodic context features.
  4. Pointer network (small transformer): encodes prefix/suffix/bank/span and
     decodes the route autoregressively with pointer attention over the bank.
  5. Event-distinct candidate generation: exact-duration beam search with both
     sequence models and subset-sum feasibility pruning.
  6. Route-level LightGBM reranker trained against the composite row metric on
     overlap-blocked out-of-fold candidate pools.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import sys
import os
import json
import math
import time
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
T_START = time.time()


def log(msg):
    print(f'[{time.time()-T_START:7.1f}s] {msg}', flush=True)


# ------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------

def parse_ev(tok):
    if tok.startswith('R'):
        return (None, int(tok.split(':')[1]))
    p, d = tok[1:].split(':')
    return (int(p), int(d))


def parse_row(r):
    bank = json.loads(r['fragment_bank'])
    for f in bank:
        f['pitch'], _ = parse_ev(f['event'])
    pre_toks = r['prefix_events'].split()
    suf_toks = r['suffix_events'].split()
    return {
        'id': r['id'],
        'pre_toks': pre_toks,
        'suf_toks': suf_toks,
        'pre': [parse_ev(t) for t in pre_toks],
        'suf': [parse_ev(t) for t in suf_toks],
        'bank': bank,
        'span': int(r['span_units']),
    }


def overlap_group_folds(rows, n_folds=2, seed=SEED, overlap=12):
    """Block validation leakage from long train-sequence overlaps.

    The grouping is used only for train/validation splits.  It never becomes a
    model feature and is never computed from test rows.
    """
    parent = list(range(len(rows)))
    size = [1] * len(rows)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    owners = {}
    for i, pr in enumerate(rows):
        bank_by = {f['fragment']: f['event'] for f in pr['bank']}
        middle = [bank_by[a] for a in pr['route']]
        seq = pr['pre_toks'] + middle + pr['suf_toks']
        for j in range(len(seq) - overlap + 1):
            events = [parse_ev(t) for t in seq[j:j + overlap]]
            origin = next((p for p, _ in events if p is not None), 0)
            signature = tuple(
                (None if p is None else p - origin, d) for p, d in events)
            previous = owners.get(signature)
            if previous is None:
                owners[signature] = i
            else:
                union(i, previous)

    roots = [find(i) for i in range(len(rows))]
    counts = collections.Counter(roots)
    items = list(counts.items())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    items.sort(key=lambda item: item[1], reverse=True)
    loads = [0] * n_folds
    root_fold = {}
    for root, count in items:
        fold = int(np.argmin(loads))
        root_fold[root] = fold
        loads[fold] += count
    return np.asarray([root_fold[root] for root in roots]), len(counts)


# ------------------------------------------------------------------
# local metric (labels for reranker training on train rows only)
# ------------------------------------------------------------------

def _f1(a, b):
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    if inter == 0:
        return 0.0
    prec = inter / sum(a.values())
    rec = inter / sum(b.values())
    return 2 * prec * rec / (prec + rec)


def row_score(pred_route, true_route, bank, span):
    bank_by = {f['fragment']: f for f in bank}
    if not pred_route or len(set(pred_route)) != len(pred_route):
        return 0.0
    if any(a not in bank_by for a in pred_route):
        return 0.0
    pe = [(bank_by[a]['pitch'], bank_by[a]['duration']) for a in pred_route]
    te = [(bank_by[a]['pitch'], bank_by[a]['duration']) for a in true_route]
    alias_f1 = _f1(collections.Counter(pred_route),
                   collections.Counter(true_route))
    m = sum(1 for i in range(min(len(pred_route), len(true_route)))
            if pred_route[i] == true_route[i])
    pos_alias = m / max(len(pred_route), len(true_route))

    def onsets(evs):
        out, t = [], 0
        for p, d in evs:
            out.append((t, p, d))
            t += d
        return out

    po, to = onsets(pe), onsets(te)
    ot = _f1(collections.Counter((t, p is None) for t, p, d in po),
             collections.Counter((t, p is None) for t, p, d in to))
    ee = _f1(collections.Counter(po), collections.Counter(to))

    def frames(evs, total):
        arr, t = [None] * total, 0
        for p, d in evs:
            for k in range(t, min(t + d, total)):
                arr[k] = p if p is not None else 'R'
            t += d
        return arr

    tf, pf = frames(te, span), frames(pe, span)
    s = 0.0
    for a, b in zip(pf, tf):
        if a is None or b is None:
            continue
        if a == b:
            s += 1.0
        elif a != 'R' and b != 'R' and abs(a - b) <= 2:
            s += 0.4
    fr = s / span if span else 1.0

    def contour(evs):
        ps = [p for p, d in evs if p is not None]
        return collections.Counter(
            (0 if b == a else (1 if b > a else -1))
            for a, b in zip(ps, ps[1:]))

    cf = _f1(contour(pe), contour(te))
    pt, tt = sum(d for _, d in pe), sum(d for _, d in te)
    du = max(0.0, 1 - abs(pt - tt) / tt) if tt else 1.0
    return (0.18 * alias_f1 + 0.17 * pos_alias + 0.15 * ot + 0.22 * ee
            + 0.15 * fr + 0.08 * cf + 0.05 * du)


# ------------------------------------------------------------------
# global melodic statistics (train only)
# ------------------------------------------------------------------

class GlobalStats:
    def __init__(self):
        self.interval = collections.Counter()
        self.dur_trans = collections.Counter()
        self.dur_marg = collections.Counter()
        self.kind_trans = collections.Counter()
        self.tok_bigram = collections.Counter()
        self.tok_marg = collections.Counter()
        self.n_int = 0

    def add_seq(self, toks):
        evs = [parse_ev(t) for t in toks]
        prev_p = None
        for (p1, d1), (p2, d2) in zip(evs, evs[1:]):
            self.dur_trans[(d1, d2)] += 1
            self.kind_trans[(p1 is None, p2 is None)] += 1
        for p, d in evs:
            self.dur_marg[d] += 1
            if p is not None:
                if prev_p is not None:
                    self.interval[p - prev_p] += 1
                    self.n_int += 1
                prev_p = p
        for t1, t2 in zip(toks, toks[1:]):
            self.tok_bigram[(t1, t2)] += 1
            self.tok_marg[t1] += 1

    def fit(self, rows):
        for pr in rows:
            bank_by = {f['fragment']: f for f in pr['bank']}
            mid = [bank_by[a]['event'] for a in pr['route']]
            self.add_seq(pr['pre_toks'] + mid + pr['suf_toks'])

    def lp_interval(self, delta):
        return math.log((self.interval.get(delta, 0) + 0.5)
                        / (self.n_int + 25))

    def lp_dur(self, d1, d2):
        return math.log((self.dur_trans.get((d1, d2), 0) + 0.5)
                        / (self.dur_marg.get(d1, 0) + 8))

    def lp_kind(self, k1, k2):
        num = self.kind_trans.get((k1, k2), 0) + 0.5
        den = (self.kind_trans.get((k1, True), 0)
               + self.kind_trans.get((k1, False), 0) + 1)
        return math.log(num / den)

    def lp_tok(self, t1, t2):
        return math.log((self.tok_bigram.get((t1, t2), 0) + 0.1)
                        / (self.tok_marg.get(t1, 0) + 10))


# ------------------------------------------------------------------
# learned route-length prior (train only)
# ------------------------------------------------------------------

LENGTH_DURS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
LENGTH_FEATURE_NAMES = [
    'span', 'bank_dur_mean', 'bank_dur_std', 'bank_dur_min',
    'bank_dur_max', 'bank_dur_sum', 'ctx_rest_count', 'ctx_dur_mean',
    'ctx_dur_std', 'bank_rest_count',
] + [f'bank_dur_{d}' for d in LENGTH_DURS
] + [f'ctx_dur_{d}' for d in LENGTH_DURS
] + [f'log_duration_subsets_len_{k}' for k in range(1, 17)]


def length_features(pr):
    bank_durs = [f['duration'] for f in pr['bank']]
    context = pr['pre'] + pr['suf']
    context_durs = [d for _, d in context]
    out = [
        float(pr['span']), float(np.mean(bank_durs)),
        float(np.std(bank_durs)), float(min(bank_durs)),
        float(max(bank_durs)), float(sum(bank_durs)),
        float(sum(p is None for p, _ in context)),
        float(np.mean(context_durs)), float(np.std(context_durs)),
        float(sum(f['pitch'] is None for f in pr['bank'])),
    ]
    out.extend(float(sum(d == value for d in bank_durs))
               for value in LENGTH_DURS)
    out.extend(float(sum(d == value for d in context_durs))
               for value in LENGTH_DURS)

    # Counts are row-local inference features, not corpus statistics.
    subset_counts = {(0, 0): 1}
    for duration in bank_durs:
        updated = dict(subset_counts)
        for (total, count), ways in subset_counts.items():
            key = (total + duration, count + 1)
            updated[key] = updated.get(key, 0) + ways
        subset_counts = updated
    out.extend(math.log1p(subset_counts.get((pr['span'], k), 0))
               for k in range(1, 17))
    return out


class LengthWrap:
    def __init__(self, booster, min_len, max_len):
        self.b = booster
        self.min_len = min_len
        self.max_len = max_len

    def predict_logp(self, pr):
        x = np.asarray([length_features(pr)], np.float32)
        probabilities = np.asarray(self.b.predict(x))[0]
        return np.log(np.maximum(probabilities, 1e-12))


def train_length_gbm(rows, seed=SEED):
    import lightgbm as lgb
    x = np.asarray([length_features(pr) for pr in rows], np.float32)
    lengths = np.asarray([len(pr['route']) for pr in rows])
    min_len, max_len = int(lengths.min()), int(lengths.max())
    y = lengths - min_len
    n_classes = max_len - min_len + 1
    row_folds, _ = overlap_group_folds(rows, n_folds=5, seed=seed)
    train_mask = row_folds != 0
    params = dict(
        objective='multiclass', num_class=n_classes, learning_rate=0.04,
        num_leaves=31, min_data_in_leaf=30, feature_fraction=0.9,
        bagging_fraction=0.8, bagging_freq=1, verbose=-1,
        num_threads=max(1, os.cpu_count() or 1), seed=seed,
        deterministic=True, force_row_wise=True)
    dtrain = lgb.Dataset(x[train_mask], label=y[train_mask],
                         feature_name=LENGTH_FEATURE_NAMES)
    dvalid = lgb.Dataset(x[~train_mask], label=y[~train_mask],
                         feature_name=LENGTH_FEATURE_NAMES)
    tuned = lgb.train(
        params, dtrain, 1000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(60, verbose=False)])
    dall = lgb.Dataset(x, label=y, feature_name=LENGTH_FEATURE_NAMES)
    booster = lgb.train(
        params, dall, num_boost_round=max(1, tuned.best_iteration))
    log(f'length gbm: {tuned.best_iteration} iters, classes '
        f'{min_len}-{max_len}')
    return LengthWrap(booster, min_len, max_len)


# ------------------------------------------------------------------
# per-row context + step features
# ------------------------------------------------------------------

FEATURE_NAMES = [
    'kind_rest', 'dur', 'pitch', 'step_idx', 'span', 'dur_so_far_frac',
    'remaining_after', 'completes', 'n_unused_after',
    'interval', 'abs_interval', 'interval2', 'abs_interval2',
    'same_dur', 'dur_ratio_log',
    'bigram_in_ctx', 'trigram_in_ctx', 'tok_in_ctx_cnt',
    'pitch_m_ctxmean', 'abs_pitch_m_ctxmean', 'pitch_m_prelast',
    'pitch_m_suf0', 'min_abs_pitch_ctx',
    'gl_lp_int', 'gl_lp_dur', 'gl_lp_kind', 'gl_lp_tok',
    'prev_rest', 'both_rest', 'dup_count',
    'complete_suf_lp_int', 'complete_suf_bigram_ctx', 'complete_suf_lp_tok',
    'feasible_after', 'span_m_dur',
]


class RowCtx:
    def __init__(self, pr, gs):
        self.pr = pr
        self.gs = gs
        ctx_toks = pr['pre_toks'] + pr['suf_toks']
        self.ctx_bigrams = set(zip(ctx_toks, ctx_toks[1:]))
        self.ctx_trigrams = set(zip(ctx_toks, ctx_toks[1:], ctx_toks[2:]))
        self.ctx_tok_cnt = collections.Counter(ctx_toks)
        ctx_p = [p for p, d in pr['pre'] + pr['suf'] if p is not None]
        self.ctx_pitch_mean = float(np.mean(ctx_p)) if ctx_p else 0.0
        self.ctx_pitches = ctx_p
        pre_p = [p for p, d in pr['pre'] if p is not None]
        self.pre_last_pitch = pre_p[-1] if pre_p else None
        suf_p = [p for p, d in pr['suf'] if p is not None]
        self.suf_first_pitch = suf_p[0] if suf_p else None
        self.suf0_tok = pr['suf_toks'][0]
        self.suf0 = pr['suf'][0]
        self.bank = pr['bank']
        self.durs = [f['duration'] for f in self.bank]
        self.span = pr['span']
        cnt = collections.Counter(f['event'] for f in self.bank)
        self.dup = {f['fragment']: cnt[f['event']] for f in self.bank}
        self._feas_cache = {}

    def feasible(self, unused_idx_tuple, target):
        if target == 0:
            return True
        if target < 0:
            return False
        got = self._feas_cache.get(unused_idx_tuple)
        if got is None:
            bits = 1
            for i in unused_idx_tuple:
                bits |= bits << self.durs[i]
            self._feas_cache[unused_idx_tuple] = got = bits
        return (got >> target) & 1 == 1


def build_features(rc, prev2, prev1, hist_toks, dur_so_far, step_idx,
                   cand_frag, unused_after_idx):
    gs = rc.gs
    p2, d2 = cand_frag['pitch'], cand_frag['duration']
    tok2 = cand_frag['event']
    p1, d1 = prev1
    tok0, tok1 = hist_toks
    kind_rest = 1.0 if p2 is None else 0.0
    prev_rest = 1.0 if p1 is None else 0.0
    has_int = p2 is not None and p1 is not None
    interval = float(p2 - p1) if has_int else 0.0
    interval2 = 0.0
    if p2 is not None and prev2 is not None and prev2[0] is not None:
        interval2 = float(p2 - prev2[0])
    remaining = rc.span - dur_so_far - d2
    completes = 1.0 if remaining == 0 else 0.0
    feas = 1.0 if rc.feasible(unused_after_idx, remaining) else 0.0
    pitch_val = float(p2) if p2 is not None else 0.0
    pm_ctx = pitch_val - rc.ctx_pitch_mean if p2 is not None else 0.0
    pm_pre = (float(p2 - rc.pre_last_pitch)
              if (p2 is not None and rc.pre_last_pitch is not None) else 0.0)
    pm_suf = (float(p2 - rc.suf_first_pitch)
              if (p2 is not None and rc.suf_first_pitch is not None) else 0.0)
    min_abs = (min(abs(p2 - q) for q in rc.ctx_pitches)
               if (p2 is not None and rc.ctx_pitches) else 0.0)
    c_lp_int = c_big = c_lp_tok = 0.0
    if remaining == 0:
        sp, sd = rc.suf0
        if p2 is not None and sp is not None:
            c_lp_int = gs.lp_interval(sp - p2)
        c_big = 1.0 if (tok2, rc.suf0_tok) in rc.ctx_bigrams else 0.0
        c_lp_tok = gs.lp_tok(tok2, rc.suf0_tok)
    tri = 1.0 if (tok0 is not None
                  and (tok0, tok1, tok2) in rc.ctx_trigrams) else 0.0
    return [
        kind_rest, float(d2), pitch_val, float(step_idx), float(rc.span),
        dur_so_far / rc.span,
        float(remaining), completes, float(len(unused_after_idx)),
        interval, abs(interval), interval2, abs(interval2),
        1.0 if d2 == d1 else 0.0, math.log(d2 / d1),
        1.0 if (tok1, tok2) in rc.ctx_bigrams else 0.0,
        tri, float(rc.ctx_tok_cnt.get(tok2, 0)),
        pm_ctx, abs(pm_ctx), pm_pre, pm_suf, float(min_abs),
        gs.lp_interval(int(interval)) if has_int else 0.0,
        gs.lp_dur(d1, d2), gs.lp_kind(p1 is None, p2 is None),
        gs.lp_tok(tok1, tok2),
        prev_rest, kind_rest * prev_rest,
        float(rc.dup[cand_frag['fragment']]),
        c_lp_int, c_big, c_lp_tok, feas, float(rc.span - d2),
    ]


# ------------------------------------------------------------------
# step-model (LightGBM) + GBM beam search
# ------------------------------------------------------------------

def gen_training_examples(rows, gs):
    X, y, groups = [], [], []
    for gi, pr in enumerate(rows):
        rc = RowCtx(pr, gs)
        bank = pr['bank']
        alias2idx = {f['fragment']: i for i, f in enumerate(bank)}
        route_idx = [alias2idx[a] for a in pr['route']]
        prev1, prev2 = pr['pre'][-1], pr['pre'][-2]
        tok1, tok0 = pr['pre_toks'][-1], pr['pre_toks'][-2]
        used = set()
        dur_so_far = 0
        for step, true_i in enumerate(route_idx):
            unused = [i for i in range(len(bank)) if i not in used]
            for i in unused:
                f = bank[i]
                if dur_so_far + f['duration'] > pr['span']:
                    continue
                after = tuple(j for j in unused if j != i)
                X.append(build_features(rc, prev2, prev1, (tok0, tok1),
                                        dur_so_far, step, f, after))
                y.append(1 if i == true_i else 0)
                groups.append(gi)
            tf = bank[true_i]
            prev2, prev1 = prev1, (tf['pitch'], tf['duration'])
            tok0, tok1 = tok1, tf['event']
            used.add(true_i)
            dur_so_far += tf['duration']
    return (np.array(X, np.float32), np.array(y, np.int8), np.array(groups))


class GBMWrap:
    def __init__(self, booster):
        self.b = booster

    def predict_logp(self, X):
        return -np.logaddexp(0.0, -self.b.predict(X, raw_score=True))


def train_step_gbm(rows, gs, seed=SEED):
    import lightgbm as lgb
    X, y, g = gen_training_examples(rows, gs)
    row_folds, _ = overlap_group_folds(rows, n_folds=5, seed=seed)
    m = row_folds[g] != 0
    dtr = lgb.Dataset(X[m], label=y[m], feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(X[~m], label=y[~m], feature_name=FEATURE_NAMES)
    params = dict(objective='binary', learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                  num_threads=max(1, os.cpu_count() or 1), seed=seed,
                  deterministic=True, force_row_wise=True)
    tuned = lgb.train(
        params, dtr, 2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)])
    dall = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    booster = lgb.train(
        params, dall, num_boost_round=max(1, tuned.best_iteration))
    log(f'step gbm: {tuned.best_iteration} iters on {X.shape[0]} examples')
    return GBMWrap(booster)


def _select_diverse_routes(scored_routes, bank, topk, length_bounds):
    """Keep high-scoring, event-distinct routes and cover plausible lengths."""
    bank_by = {f['fragment']: f for f in bank}
    ranked = sorted(scored_routes, key=lambda item: -item[0])
    selected, seen = [], set()

    def add_routes(candidates, limit):
        added = 0
        for score, route in candidates:
            event_key = tuple(bank_by[a]['event'] for a in route)
            if event_key in seen:
                continue
            seen.add(event_key)
            selected.append((score, route))
            added += 1
            if added >= limit:
                break

    add_routes(ranked, topk)
    if length_bounds is not None:
        for length in range(length_bounds[0], length_bounds[1] + 1):
            present = sum(len(route) == length for _, route in selected)
            needed = max(0, 6 - present)
            if needed:
                add_routes(
                    (item for item in ranked if len(item[1]) == length),
                    needed)
    selected.sort(key=lambda item: -item[0])
    return selected


def gbm_beam_search(pr, rc, gs, model, beam_width=128, step_bonus=0.75,
                    topk=40, length_bounds=None):
    bank = pr['bank']
    alias_event = {f['fragment']: f['event'] for f in bank}
    n = len(bank)
    span = pr['span']
    beams = [(0.0, 0, 0, pr['pre'][-2], pr['pre'][-1],
              pr['pre_toks'][-2], pr['pre_toks'][-1], ())]
    completed = []
    for step in range(16):
        feats, meta = [], []
        for si, st in enumerate(beams):
            score, dur, mask, prev2, prev1, tok0, tok1, route = st
            unused = [i for i in range(n) if not (mask >> i) & 1]
            for i in unused:
                f = bank[i]
                d2 = f['duration']
                if dur + d2 > span:
                    continue
                after = tuple(j for j in unused if j != i)
                rem = span - dur - d2
                if rem > 0 and not rc.feasible(after, rem):
                    continue
                feats.append(build_features(rc, prev2, prev1, (tok0, tok1),
                                            dur, step, f, after))
                meta.append((si, i, rem))
        if not feats:
            break
        lp = model.predict_logp(np.asarray(feats, np.float32))
        cands = []
        for (si, i, rem), value in zip(meta, lp):
            score, dur, mask, prev2, prev1, tok0, tok1, route = beams[si]
            f = bank[i]
            state = (score + float(value) + step_bonus,
                     dur + f['duration'], mask | (1 << i), prev1,
                     (f['pitch'], f['duration']), tok1, f['event'],
                     route + (f['fragment'],))
            if rem == 0:
                completed.append(state)
            else:
                cands.append(state)
        cands.sort(key=lambda state: -state[0])
        seen = set()
        beams = []
        for state in cands:
            event_key = tuple(alias_event[a] for a in state[7])
            if event_key in seen:
                continue
            seen.add(event_key)
            beams.append(state)
            if len(beams) >= beam_width:
                break
        if not beams:
            break
    scored = [(state[0], list(state[7])) for state in completed]
    return _select_diverse_routes(scored, bank, topk, length_bounds)


def fallback_route(pr):
    """Always produce a unique-alias route matching span when one exists."""
    states = {0: ()}
    for fragment in pr['bank']:
        updated = dict(states)
        for total, route in states.items():
            new_total = total + fragment['duration']
            if new_total <= pr['span'] and new_total not in updated:
                updated[new_total] = route + (fragment['fragment'],)
        states = updated
    if pr['span'] in states:
        return list(states[pr['span']])
    return list(states[max(states)])


# ------------------------------------------------------------------
# pointer network
# ------------------------------------------------------------------

PITCH_MIN, PITCH_MAX = -36, 36
REST_IDX = PITCH_MAX - PITCH_MIN + 1
N_PITCH = REST_IDX + 1
MAX_DUR = 24
MAX_STEPS = 12
REM_BUCKETS = 96
BANDS = {'low': 0, 'mid': 1, 'high': 2, 'rest': 3}
N_BANK = 16
OFF_PRE, OFF_SUF, OFF_BANK, OFF_DEC = 0, 10, 20, 36
T_TOTAL = OFF_DEC + MAX_STEPS


def pitch_to_idx(p, shift=0):
    if p is None:
        return REST_IDX
    return int(np.clip(p + shift, PITCH_MIN, PITCH_MAX)) - PITCH_MIN


class PointerNet(nn.Module):
    def __init__(self, d=192, n_layers=4, n_heads=4, ff=384, dropout=0.1):
        super().__init__()
        self.pitch_emb = nn.Embedding(N_PITCH, d)
        self.dur_emb = nn.Embedding(MAX_DUR, d)
        self.seg_emb = nn.Embedding(4, d)
        self.pos_emb = nn.Embedding(24, d)
        self.band_emb = nn.Embedding(4, d)
        self.rem_emb = nn.Embedding(REM_BUCKETS, d)
        self.step_emb = nn.Embedding(MAX_STEPS + 1, d)
        self.start_tok = nn.Parameter(torch.randn(d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, batch):
        event_x = (self.pitch_emb(batch['pitch'])
                   + self.dur_emb(batch['dur'])
                   + self.band_emb(batch['band']))
        is_start = batch['is_start'].unsqueeze(-1).bool()
        event_x = torch.where(is_start, self.start_tok.expand_as(event_x),
                              event_x)
        x = (event_x + self.seg_emb(batch['seg'])
             + self.pos_emb(batch['pos']) + self.rem_emb(batch['rem'])
             + self.step_emb(batch['step']))
        h = self.encoder(x, mask=batch['attn_mask'])
        bank_h = h[:, OFF_BANK:OFF_BANK + N_BANK, :]
        dec_h = h[:, OFF_DEC:OFF_DEC + MAX_STEPS, :]
        q = self.q_proj(dec_h)
        k = self.k_proj(bank_h)
        return torch.einsum('bsd,bnd->bsn', q, k) * self.scale


def build_attn_mask():
    m = torch.zeros(T_TOTAL, T_TOTAL)
    NEG = float('-inf')
    m[:OFF_DEC, OFF_DEC:] = NEG
    for i in range(OFF_DEC, T_TOTAL):
        m[i, i + 1:] = NEG
    return m


ATTN_MASK = build_attn_mask()
NET_KEYS = ['pitch', 'dur', 'seg', 'pos', 'band', 'rem', 'step', 'is_start']


def encode_row_net(pr, route_idx=None, shift=0):
    pitch = np.zeros(T_TOTAL, np.int64)
    dur = np.zeros(T_TOTAL, np.int64)
    seg = np.zeros(T_TOTAL, np.int64)
    pos = np.zeros(T_TOTAL, np.int64)
    band = np.zeros(T_TOTAL, np.int64)
    rem = np.zeros(T_TOTAL, np.int64)
    step = np.zeros(T_TOTAL, np.int64)
    is_start = np.zeros(T_TOTAL, np.int64)
    for i, (p, d) in enumerate(pr['pre']):
        j = OFF_PRE + i
        pitch[j] = pitch_to_idx(p, shift)
        dur[j] = min(d, MAX_DUR) - 1
        pos[j] = i
    for i, (p, d) in enumerate(pr['suf']):
        j = OFF_SUF + i
        pitch[j] = pitch_to_idx(p, shift)
        dur[j] = min(d, MAX_DUR) - 1
        seg[j] = 1
        pos[j] = i
    for i, f in enumerate(pr['bank']):
        j = OFF_BANK + i
        pitch[j] = pitch_to_idx(f['pitch'], shift)
        dur[j] = min(f['duration'], MAX_DUR) - 1
        seg[j] = 2
        band[j] = BANDS[f['pitch_band']]
    span = pr['span']
    seg[OFF_DEC:] = 3
    is_start[OFF_DEC] = 1
    rem[OFF_DEC] = min(span, REM_BUCKETS - 1)
    if route_idx is not None:
        dsum = 0
        for s, bi in enumerate(route_idx[:MAX_STEPS - 1]):
            f = pr['bank'][bi]
            j = OFF_DEC + 1 + s
            pitch[j] = pitch_to_idx(f['pitch'], shift)
            dur[j] = min(f['duration'], MAX_DUR) - 1
            pos[j] = s
            band[j] = BANDS[f['pitch_band']]
            dsum += f['duration']
            rem[j] = min(span - dsum, REM_BUCKETS - 1)
            step[j] = min(s + 1, MAX_STEPS)
    return dict(pitch=pitch, dur=dur, seg=seg, pos=pos, band=band, rem=rem,
                step=step, is_start=is_start)


def feas_masks_for_route(pr, rc, route_idx):
    n = len(pr['bank'])
    span = pr['span']
    L = len(route_idx)
    masks = np.zeros((L, N_BANK), bool)
    used = set()
    dsum = 0
    for s in range(L):
        unused = [i for i in range(n) if i not in used]
        for i in unused:
            d = pr['bank'][i]['duration']
            if dsum + d > span:
                continue
            after = tuple(j for j in unused if j != i)
            r = span - dsum - d
            if r == 0 or rc.feasible(after, r):
                masks[s, i] = True
        used.add(route_idx[s])
        dsum += pr['bank'][route_idx[s]]['duration']
    return masks


def train_pointer_net(rows, gs, device, epochs=110, bs=64, lr=3e-4, d=192,
                      n_layers=4, seed=SEED):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    net = PointerNet(d=d, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    enc_all = []
    for pr in rows:
        rc = RowCtx(pr, gs)
        alias2idx = {f['fragment']: i for i, f in enumerate(pr['bank'])}
        ridx = [alias2idx[a] for a in pr['route']]
        fmask = feas_masks_for_route(pr, rc, ridx)
        L = len(ridx)
        tgt = np.full(MAX_STEPS, -100, np.int64)
        tgt[:min(L, MAX_STEPS)] = ridx[:MAX_STEPS]
        smask = np.zeros((MAX_STEPS, N_BANK), bool)
        smask[:min(L, MAX_STEPS)] = fmask[:MAX_STEPS]
        encs = {sh: encode_row_net(pr, ridx, shift=sh)
                for sh in range(-3, 4)}
        enc_all.append((encs, tgt, smask))
    n = len(enc_all)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs))
    rng = np.random.default_rng(seed)
    attn = ATTN_MASK.to(device)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(n)
        for b0 in range(0, n, bs):
            sel = perm[b0:b0 + bs]
            encs = [enc_all[i] for i in sel]
            shifts = rng.integers(-3, 4, len(sel))
            batch = {k: torch.from_numpy(
                np.stack([e[0][sh][k] for e, sh in zip(encs, shifts)])
                ).to(device) for k in NET_KEYS}
            batch['attn_mask'] = attn
            tgt = torch.from_numpy(
                np.stack([e[1] for e in encs])).to(device)
            fmask = torch.from_numpy(
                np.stack([e[2] for e in encs])).to(device)
            logits = net(batch).masked_fill(~fmask, float('-inf'))
            valid = tgt != -100
            safe = torch.where(valid.unsqueeze(-1), logits,
                               torch.zeros_like(logits))
            loss = F.cross_entropy(safe.reshape(-1, N_BANK),
                                   tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    net.eval()
    return net


@torch.no_grad()
def net_beam_search(pr, rc, net, device, beam_width=64, topk=40,
                    length_bounds=None):
    bank = pr['bank']
    n = len(bank)
    span = pr['span']
    beams = [(0.0, 0, 0, ())]
    completed = []
    attn = ATTN_MASK.to(device)
    for step_i in range(MAX_STEPS - 1):
        if not beams:
            break
        encs = [encode_row_net(pr, list(route)) for _, _, _, route in beams]
        batch = {k: torch.from_numpy(
            np.stack([e[k] for e in encs])).to(device) for k in NET_KEYS}
        batch['attn_mask'] = attn
        logits_all = net(batch)[:, step_i, :].cpu().numpy()
        new = []
        for bi_state, (score, dcur, mask, route) in enumerate(beams):
            unused = [i for i in range(n) if not (mask >> i) & 1]
            feasible = []
            for i in unused:
                d2 = bank[i]['duration']
                if dcur + d2 > span:
                    continue
                after = tuple(j for j in unused if j != i)
                remaining = span - dcur - d2
                if remaining > 0 and not rc.feasible(after, remaining):
                    continue
                feasible.append((i, remaining))
            if not feasible:
                continue
            values = logits_all[bi_state][[i for i, _ in feasible]]
            maximum = values.max()
            logp = values - (maximum + np.log(np.exp(values - maximum).sum()))
            for (i, remaining), value in zip(feasible, logp):
                state = (score + float(value), dcur + bank[i]['duration'],
                         mask | (1 << i), route + (i,))
                if remaining == 0:
                    completed.append(state)
                else:
                    new.append(state)
        new.sort(key=lambda state: -state[0])
        seen = set()
        beams = []
        for state in new:
            event_key = tuple(bank[i]['event'] for i in state[3])
            if event_key in seen:
                continue
            seen.add(event_key)
            beams.append(state)
            if len(beams) >= beam_width:
                break
    scored = [
        (state[0], [bank[i]['fragment'] for i in state[3]])
        for state in completed
    ]
    return _select_diverse_routes(scored, bank, topk, length_bounds)


@torch.no_grad()
def score_routes_net(pr, rc, net, device, routes):
    bank = pr['bank']
    alias2idx = {f['fragment']: i for i, f in enumerate(bank)}
    span = pr['span']
    encs, all_ridx = [], []
    for route in routes:
        ridx = [alias2idx[a] for a in route]
        all_ridx.append(ridx)
        encs.append(encode_row_net(pr, ridx))
    batch = {k: torch.from_numpy(
        np.stack([e[k] for e in encs])).to(device) for k in NET_KEYS}
    batch['attn_mask'] = ATTN_MASK.to(device)
    logits = net(batch).cpu().numpy()
    out = []
    for b, ridx in enumerate(all_ridx):
        used = set()
        dsum = 0
        lps = []
        for s, i in enumerate(ridx[:MAX_STEPS]):
            unused = [j for j in range(len(bank)) if j not in used]
            feas = []
            for j in unused:
                d2 = bank[j]['duration']
                if dsum + d2 > span:
                    continue
                after = tuple(t for t in unused if t != j)
                r = span - dsum - d2
                if r == 0 or rc.feasible(after, r):
                    feas.append(j)
            if i not in feas:
                feas.append(i)
            lg = logits[b, s, feas]
            mx = lg.max()
            lp = lg - (mx + np.log(np.exp(lg - mx).sum()))
            lps.append(float(lp[feas.index(i)]))
            used.add(i)
            dsum += bank[i]['duration']
        out.append(np.array(lps))
    return out


# ------------------------------------------------------------------
# route-level features + candidate pool
# ------------------------------------------------------------------

ROUTE_FEATS = [
    'beam_score', 'beam_rank', 'route_len', 'span', 'mean_dur',
    'std_dur', 'route_len_sq',
    'bigram_ctx_frac', 'trigram_ctx_frac', 'fourgram_ctx_cnt',
    'tok_ctx_frac',
    'mean_abs_int', 'max_abs_int', 'n_big_leap', 'n_zero_int',
    'mean_lp_int', 'min_lp_int', 'mean_lp_dur', 'mean_lp_tok',
    'bnd_pre_lp_int', 'bnd_suf_lp_int', 'bnd_pre_big', 'bnd_suf_big',
    'pitch_mean_m_ctx', 'abs_pitch_mean_m_ctx', 'pitch_range',
    'rest_cnt', 'rest_dur_frac', 'ctx_rest_frac',
    'n_dup_used', 'beam_score_per_step',
    'first_dur', 'last_dur', 'contour_updown_balance',
]

STACK_FEATS = ROUTE_FEATS + [
    'gbm_score', 'gbm_score_per_step', 'gbm_min_step', 'gbm_rank',
    'net_score', 'net_score_per_step', 'net_min_step', 'net_rank',
    'in_gbm', 'in_net', 'in_both',
    'gbm_gap_to_best', 'net_gap_to_best',
    'length_logp', 'length_is_mode', 'length_distance_mode',
]


def route_features(pr, rc, gs, route, beam_score, rank):
    bank_by = {f['fragment']: f for f in pr['bank']}
    evs = [(bank_by[a]['pitch'], bank_by[a]['duration']) for a in route]
    toks = [bank_by[a]['event'] for a in route]
    L = len(route)
    span = pr['span']
    full = [pr['pre_toks'][-1]] + toks + [pr['suf_toks'][0]]
    ctx_toks = pr['pre_toks'] + pr['suf_toks']
    ctx_four = set(zip(ctx_toks, ctx_toks[1:], ctx_toks[2:], ctx_toks[3:]))
    bgs = list(zip(full, full[1:]))
    tgs = list(zip(full, full[1:], full[2:]))
    fgs = list(zip(full, full[1:], full[2:], full[3:]))
    big_frac = float(np.mean([b in rc.ctx_bigrams for b in bgs]))
    tri_frac = (float(np.mean([t in rc.ctx_trigrams for t in tgs]))
                if tgs else 0.0)
    four_cnt = sum(f in ctx_four for f in fgs)
    tok_frac = float(np.mean([t in rc.ctx_tok_cnt for t in toks]))
    ps = [p for p, d in evs if p is not None]
    ints = [b - a for a, b in zip(ps, ps[1:])]
    lp_ints = [gs.lp_interval(i) for i in ints]
    durs = [d for p, d in evs]
    lp_durs = [gs.lp_dur(a, b) for a, b in zip(durs, durs[1:])]
    lp_toks = [gs.lp_tok(a, b) for a, b in zip(full, full[1:])]
    bnd_pre = (gs.lp_interval(ps[0] - rc.pre_last_pitch)
               if (ps and rc.pre_last_pitch is not None) else 0.0)
    bnd_suf = (gs.lp_interval(rc.suf_first_pitch - ps[-1])
               if (ps and rc.suf_first_pitch is not None) else 0.0)
    bnd_pre_big = (1.0 if (pr['pre_toks'][-1], toks[0]) in rc.ctx_bigrams
                   else 0.0)
    bnd_suf_big = (1.0 if (toks[-1], pr['suf_toks'][0]) in rc.ctx_bigrams
                   else 0.0)
    pmean = float(np.mean(ps)) if ps else 0.0
    prange = float(max(ps) - min(ps)) if ps else 0.0
    rest_cnt = sum(1 for p, d in evs if p is None)
    rest_dur = sum(d for p, d in evs if p is None)
    ctx_rest = float(np.mean([1.0 if p is None else 0.0
                              for p, d in pr['pre'] + pr['suf']]))
    n_dup = sum(1 for a in route if rc.dup[a] > 1)
    ups = sum(1 for i in ints if i > 0)
    downs = sum(1 for i in ints if i < 0)
    bal = (ups - downs) / max(1, len(ints))
    return [
        beam_score, float(rank), float(L), float(span), span / L,
        float(np.std(durs)), float(L * L),
        big_frac, tri_frac, float(four_cnt), tok_frac,
        float(np.mean([abs(i) for i in ints])) if ints else 0.0,
        float(max([abs(i) for i in ints], default=0)),
        float(sum(abs(i) > 7 for i in ints)),
        float(sum(i == 0 for i in ints)),
        float(np.mean(lp_ints)) if lp_ints else 0.0,
        float(min(lp_ints)) if lp_ints else 0.0,
        float(np.mean(lp_durs)) if lp_durs else 0.0,
        float(np.mean(lp_toks)),
        bnd_pre, bnd_suf, bnd_pre_big, bnd_suf_big,
        pmean - rc.ctx_pitch_mean, abs(pmean - rc.ctx_pitch_mean), prange,
        float(rest_cnt), rest_dur / span, ctx_rest,
        float(n_dup), beam_score / L,
        float(durs[0]), float(durs[-1]), bal,
    ]


def score_route_gbm(pr, rc, gs, model, route_aliases):
    bank = pr['bank']
    alias2idx = {f['fragment']: i for i, f in enumerate(bank)}
    ridx = [alias2idx[a] for a in route_aliases]
    prev1, prev2 = pr['pre'][-1], pr['pre'][-2]
    tok1, tok0 = pr['pre_toks'][-1], pr['pre_toks'][-2]
    used = set()
    dur_so_far = 0
    feats = []
    for step, i in enumerate(ridx):
        unused = [j for j in range(len(bank)) if j not in used]
        after = tuple(j for j in unused if j != i)
        f = bank[i]
        feats.append(build_features(rc, prev2, prev1, (tok0, tok1),
                                    dur_so_far, step, f, after))
        prev2, prev1 = prev1, (f['pitch'], f['duration'])
        tok0, tok1 = tok1, f['event']
        used.add(i)
        dur_so_far += f['duration']
    return model.predict_logp(np.asarray(feats, np.float32))


def gen_pool(pr, gs, gbm, net, length_model, device, topk=40):
    rc = RowCtx(pr, gs)
    length_bounds = (length_model.min_len, length_model.max_len)
    length_logp = length_model.predict_logp(pr)
    length_mode = length_model.min_len + int(np.argmax(length_logp))
    g_cands = gbm_beam_search(
        pr, rc, gs, gbm, topk=topk, length_bounds=length_bounds)
    n_cands = net_beam_search(
        pr, rc, net, device, topk=topk, length_bounds=length_bounds)
    alias_event = {f['fragment']: f['event'] for f in pr['bank']}
    pool = {}
    for rank, (beam_score, route) in enumerate(g_cands):
        event_key = tuple(alias_event[a] for a in route)
        pool[event_key] = dict(
            route=route, gbm_rank=rank, net_rank=99,
            in_gbm=1.0, in_net=0.0, beam_score=beam_score,
            beam_rank=rank)
    for rank, (beam_score, route) in enumerate(n_cands):
        event_key = tuple(alias_event[a] for a in route)
        if event_key in pool:
            pool[event_key]['net_rank'] = rank
            pool[event_key]['in_net'] = 1.0
        else:
            pool[event_key] = dict(
                route=route, gbm_rank=99, net_rank=rank,
                in_gbm=0.0, in_net=1.0, beam_score=beam_score,
                beam_rank=rank)
    cands = list(pool.values())
    if not cands:
        return [dict(route=fallback_route(pr), feats=None)]
    routes = [c['route'] for c in cands]
    net_lps = score_routes_net(pr, rc, net, device, routes)
    for c, net_lp in zip(cands, net_lps):
        c['net_lp'] = net_lp
        c['gbm_lp'] = score_route_gbm(pr, rc, gs, gbm, c['route'])
    gbm_best = max(float(np.sum(c['gbm_lp'])) for c in cands)
    net_best = max(float(np.sum(c['net_lp'])) for c in cands)
    for c in cands:
        gbm_sum = float(np.sum(c['gbm_lp']))
        net_sum = float(np.sum(c['net_lp']))
        route_len = len(c['route'])
        length_index = route_len - length_model.min_len
        if 0 <= length_index < len(length_logp):
            route_length_logp = float(length_logp[length_index])
        else:
            route_length_logp = math.log(1e-12)
        stack = [
            gbm_sum, gbm_sum / route_len, float(np.min(c['gbm_lp'])),
            float(c['gbm_rank']),
            net_sum, net_sum / route_len, float(np.min(c['net_lp'])),
            float(c['net_rank']),
            c['in_gbm'], c['in_net'], c['in_gbm'] * c['in_net'],
            gbm_sum - gbm_best, net_sum - net_best,
            route_length_logp, float(route_len == length_mode),
            float(abs(route_len - length_mode)),
        ]
        c['feats'] = route_features(
            pr, rc, gs, c['route'], c['beam_score'], c['beam_rank']) + stack
    return cands


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def selected_candidate_score(predictions, labels, groups):
    selected = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        best = indices[int(np.argmax(predictions[indices]))]
        selected.append(labels[best])
    return float(np.mean(selected))


def groupwise_normalize(values, groups):
    normalized = np.zeros_like(values, dtype=np.float64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        group_values = values[indices]
        scale = float(np.std(group_values))
        if scale > 1e-8:
            normalized[indices] = (
                group_values - float(np.mean(group_values))) / scale
    return normalized


BLEND_OPTIONS = [
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0, 1.0),
    (3.0, 1.0, 1.0, 1.0),
]


def blend_model_predictions(models, features, groups, weights):
    blended = np.zeros(len(features), dtype=np.float64)
    for model, weight in zip(models, weights):
        if weight:
            values = np.asarray(model.predict(features), dtype=np.float64)
            blended += weight * groupwise_normalize(values, groups)
    return blended


def main():
    import lightgbm as lgb
    if len(sys.argv) != 3:
        raise SystemExit('usage: python3 solution.py <public_dir> '
                         '<submission_out>')
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = 'cuda'
    elif (getattr(torch.backends, 'mps', None) is not None
          and torch.backends.mps.is_available()):
        device = 'mps'
    else:
        device = 'cpu'
    log(f'device={device}')

    train_df = pd.read_csv(public_dir / 'train.csv')
    train_rows = []
    for _, row in train_df.iterrows():
        parsed = parse_row(row)
        parsed['route'] = row['target_route'].split()
        train_rows.append(parsed)
    # Test data is deliberately not read until every model has been fitted.
    log(f'train rows={len(train_rows)}')

    # ---- overlap-blocked OOF candidate pools for reranker training ----
    n_folds = 2
    fold_id, n_overlap_groups = overlap_group_folds(
        train_rows, n_folds=n_folds, seed=SEED)
    folds = [
        [row for row, row_fold in zip(train_rows, fold_id)
         if row_fold == fold]
        for fold in range(n_folds)
    ]
    fold_sizes = '/'.join(str(len(fold)) for fold in folds)
    log(f'validation groups={n_overlap_groups}; fold sizes={fold_sizes}')
    oof = []
    for fold in range(n_folds):
        fit_rows = [
            row for other_fold in range(n_folds) if other_fold != fold
            for row in folds[other_fold]
        ]
        stats_fold = GlobalStats()
        stats_fold.fit(fit_rows)
        length_fold = train_length_gbm(fit_rows, seed=SEED + fold)
        gbm_fold = train_step_gbm(
            fit_rows, stats_fold, seed=SEED + fold)
        net_fold = train_pointer_net(
            fit_rows, stats_fold, device, epochs=90, seed=SEED + fold)
        log(f'fold {fold}: models trained')
        for k, row in enumerate(folds[fold]):
            candidates = gen_pool(
                row, stats_fold, gbm_fold, net_fold, length_fold, device)
            oof.append((fold, row, candidates))
            if (k + 1) % 600 == 0:
                log(f'fold {fold}: generated {k + 1} pools')
        log(f'fold {fold}: pools done')
        del length_fold, gbm_fold, net_fold

    X, y, groups, candidate_folds = [], [], [], []
    gbm_validation, net_validation, oracle_validation = [], [], []
    for group, (fold, row, candidates) in enumerate(oof):
        valid_candidates = [c for c in candidates if c['feats'] is not None]
        if not valid_candidates:
            score = row_score(
                candidates[0]['route'], row['route'], row['bank'], row['span'])
            gbm_validation.append(score)
            net_validation.append(score)
            oracle_validation.append(score)
            continue
        row_labels = [
            row_score(c['route'], row['route'], row['bank'], row['span'])
            for c in valid_candidates
        ]
        gbm_choice = int(np.argmin(
            [c['gbm_rank'] for c in valid_candidates]))
        net_choice = int(np.argmin(
            [c['net_rank'] for c in valid_candidates]))
        gbm_validation.append(row_labels[gbm_choice])
        net_validation.append(row_labels[net_choice])
        oracle_validation.append(max(row_labels))
        for candidate, label in zip(valid_candidates, row_labels):
            X.append(candidate['feats'])
            y.append(label)
            groups.append(group)
            candidate_folds.append(fold)
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    candidate_folds = np.asarray(candidate_folds)
    log('overlap-blocked candidates: '
        f'gbm={np.mean(gbm_validation):.5f} '
        f'net={np.mean(net_validation):.5f} '
        f'oracle={np.mean(oracle_validation):.5f}')
    log(f'reranker data: {X.shape}')

    validation_fold = n_folds - 1
    rerank_train = candidate_folds != validation_fold
    rerank_valid = ~rerank_train
    dtrain = lgb.Dataset(
        X[rerank_train], label=y[rerank_train], feature_name=STACK_FEATS)
    dvalid = lgb.Dataset(
        X[rerank_valid], label=y[rerank_valid], feature_name=STACK_FEATS)
    common_params = dict(
        learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
        verbose=-1, num_threads=max(1, os.cpu_count() or 1), seed=SEED,
        deterministic=True, force_row_wise=True)
    regression_objectives = ('regression', 'regression_l1')
    tuned_regressors, regression_iterations = [], []
    for objective in regression_objectives:
        regression_params = dict(objective=objective, **common_params)
        tuned = lgb.train(
            regression_params, dtrain, 2000, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False)])
        tuned_regressors.append(tuned)
        regression_iterations.append(max(1, tuned.best_iteration))

    relevance = np.clip(np.rint(y * 20), 0, 20).astype(np.int32)
    train_group_sizes = np.unique(
        groups[rerank_train], return_counts=True)[1].tolist()
    valid_group_sizes = np.unique(
        groups[rerank_valid], return_counts=True)[1].tolist()
    ranking_objectives = ('lambdarank', 'rank_xendcg')
    tuned_rankers, ranking_iterations = [], []
    for objective in ranking_objectives:
        rank_train = lgb.Dataset(
            X[rerank_train], label=relevance[rerank_train],
            group=train_group_sizes, feature_name=STACK_FEATS)
        rank_valid = lgb.Dataset(
            X[rerank_valid], label=relevance[rerank_valid],
            group=valid_group_sizes, feature_name=STACK_FEATS)
        rank_params = dict(
            objective=objective, metric='ndcg', eval_at=[1],
            label_gain=[float(i) for i in range(21)], **common_params)
        tuned = lgb.train(
            rank_params, rank_train, 1500, valid_sets=[rank_valid],
            callbacks=[lgb.early_stopping(100, verbose=False)])
        tuned_rankers.append(tuned)
        ranking_iterations.append(max(1, tuned.best_iteration))

    validation_models = tuned_regressors + tuned_rankers
    validation_groups = groups[rerank_valid]
    best_weights, best_validation_score = None, -1.0
    component_scores = []
    for model_index, model in enumerate(validation_models):
        weights = tuple(
            1.0 if i == model_index else 0.0
            for i in range(len(validation_models)))
        predictions = blend_model_predictions(
            validation_models, X[rerank_valid], validation_groups, weights)
        component_scores.append(selected_candidate_score(
            predictions, y[rerank_valid], validation_groups))
    for weights in BLEND_OPTIONS:
        predictions = blend_model_predictions(
            validation_models, X[rerank_valid], validation_groups, weights)
        score = selected_candidate_score(
            predictions, y[rerank_valid], validation_groups)
        if score > best_validation_score:
            best_validation_score = score
            best_weights = weights

    rerank_all = lgb.Dataset(X, label=y, feature_name=STACK_FEATS)
    regressors = []
    for objective, iterations in zip(
            regression_objectives, regression_iterations):
        regression_params = dict(objective=objective, **common_params)
        regressors.append(lgb.train(
            regression_params, rerank_all, num_boost_round=iterations))
    all_group_sizes = np.unique(groups, return_counts=True)[1].tolist()
    rankers = []
    for objective, iterations in zip(
            ranking_objectives, ranking_iterations):
        rank_all = lgb.Dataset(
            X, label=relevance, group=all_group_sizes,
            feature_name=STACK_FEATS)
        rank_params = dict(
            objective=objective, metric='ndcg', eval_at=[1],
            label_gain=[float(i) for i in range(21)], **common_params)
        rankers.append(lgb.train(
            rank_params, rank_all, num_boost_round=iterations))
    rerank_models = regressors + rankers
    log('rerank validation: '
        f'l2={component_scores[0]:.5f} '
        f'l1={component_scores[1]:.5f} '
        f'lambdarank={component_scores[2]:.5f} '
        f'xendcg={component_scores[3]:.5f} '
        f'blend={best_validation_score:.5f} weights={best_weights}')

    # ---- final models on all training rows ----
    stats = GlobalStats()
    stats.fit(train_rows)
    length_model = train_length_gbm(train_rows, seed=SEED)
    gbm = train_step_gbm(train_rows, stats, seed=SEED)
    net = train_pointer_net(
        train_rows, stats, device, epochs=110, seed=SEED)
    log('final models trained')

    # Test is read only after fitting, then transformed and predicted one row
    # at a time.  No information passes between test-row predictions.
    test_df = pd.read_csv(public_dir / 'test.csv')
    test_ids, predictions = [], []
    for k, (_, test_row) in enumerate(test_df.iterrows()):
        row = parse_row(test_row)
        test_ids.append(test_row['id'])
        candidates = gen_pool(
            row, stats, gbm, net, length_model, device)
        if candidates[0].get('feats') is None:
            best_route = candidates[0]['route']
        else:
            features = np.asarray(
                [candidate['feats'] for candidate in candidates], np.float32)
            inference_groups = np.zeros(len(features), dtype=np.int8)
            scores = blend_model_predictions(
                rerank_models, features, inference_groups, best_weights)
            best = int(np.argmax(scores))
            best_route = candidates[best]['route']
        if not best_route:
            best_route = fallback_route(row)
        predictions.append(' '.join(best_route))
        if (k + 1) % 200 == 0:
            log(f'inference rows={k + 1}')

    submission = pd.DataFrame(
        {'id': test_ids, 'predicted_route': predictions})
    submission.to_csv(submission_out, index=False)
    log(f'wrote {submission_out}')


if __name__ == '__main__':
    main()
