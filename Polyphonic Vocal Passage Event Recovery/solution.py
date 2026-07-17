"""Polyphonic Vocal Passage Event Recovery.

Recovers the missing 4-7 vocal events of a gap using:
  - a k-best constrained rhythm DP trained on train-only duration transitions;
  - a learned CatBoost rhythm candidate reranker;
  - independent and conditional CatBoost pitch/rest rankers;
  - Viterbi decoding of the learned conditional transition-emission scores;
  - deterministic tie handling from boundary tie constraints.

All statistics are fit on train.csv only. Test rows are used strictly for
per-row inference (transform + predict). Fully deterministic.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import json
import gc
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker

DUR_VOCAB = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
DUR_IDX = {d: i for i, d in enumerate(DUR_VOCAB)}
MAX_INT = 24
N_CLS = 5  # interval classes: leap-, step-, repeat, step+, leap+ (5 = after-rest)

W = {
    "phase": 0.1, "grid_on": 1.8, "grid_off": 3.0, "ctxdur": 0.4, "pc": 0.7,
    "chord": 4.0, "chord12u": 2.5, "chordm": 0.3, "reuse": 0.1, "range": 1.0,
    "iv": 0.3, "rest": 0.5, "out": 1.4, "rhy": 0.3, "strength": 0.0,
    "lag": 1.5, "lagL": 12, "deg": 0.0, "degbig": 0.4, "interp": 0.1,
    "offbeat": 0.4, "top": 3.0, "top12": 1.0, "lagp": 0.6, "iv2": 0.0,
    "posdur": 0.15, "endbar": 0.2,
}

KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def icls(d):
    if d <= -3:
        return 0
    if d < 0:
        return 1
    if d == 0:
        return 2
    if d <= 2:
        return 3
    return 4


def dbucket(d):
    if d in DUR_IDX:
        return d
    return min(DUR_VOCAB, key=lambda v: abs(v - d))


def parse_rows(df, with_answer):
    rows = []
    for _, r in df.iterrows():
        ctx = json.loads(r.score_context_json)
        row = {
            'id': r.id,
            'before': ctx['target_before'],
            'after': ctx['target_after'],
            'chords': sorted(ctx['accompaniment_chords'], key=lambda c: c['onset_tick']),
            'n': int(r.missing_event_count),
            'voice': ctx['target_voice_token'],
        }
        if with_answer:
            row['answer'] = json.loads(r.answer_json)['events']
        rows.append(row)
    return rows


def estimate_key(row):
    w = np.zeros(12)
    for c in row['chords']:
        for p in c['pitches']:
            w[p % 12] += 1.0
    for e in row['before'] + row['after']:
        if e['pitch'] != -1:
            w[e['pitch'] % 12] += 2.0
    if w.sum() == 0 or w.std() == 0:
        return 0, 0
    best = (-2.0, 0, 0)
    for mode, prof in ((0, KS_MAJ), (1, KS_MIN)):
        for root in range(12):
            pr = np.roll(prof, root)
            c = np.corrcoef(w, pr)[0, 1]
            if c > best[0]:
                best = (c, root, mode)
    return best[1], best[2]


def strength_map(row):
    out = {}
    prev_bass = None
    for c in row['chords']:
        n = len(c['pitches'])
        bass = min(c['pitches']) if c['pitches'] else None
        b = 1 if n == 1 else (2 if n <= 3 else 3)
        if bass is not None and prev_bass is not None and abs(bass - prev_bass) >= 3 and b < 3:
            b += 1
        prev_bass = bass
        out[c['onset_tick']] = max(out.get(c['onset_tick'], 0), b)
    return out


def est_meter(row):
    marks = dict(strength_map(row))
    for e in row['before'] + row['after']:
        if e['duration_tick'] >= 6:
            marks[e['onset_tick']] = marks.get(e['onset_tick'], 0) + 2
    if not marks:
        return 8, 0
    ticks = sorted(marks)
    w = np.array([marks[t] for t in ticks], dtype=float)
    ta = np.array(ticks)
    tot_mean = w.mean()
    best = (-1e9, 8, 0)
    for L in (6, 8, 12, 16, 24):
        for phi in range(L):
            sel = (ta - phi) % L == 0
            ns = int(sel.sum())
            if ns < 2:
                continue
            score = (w[sel].mean() - tot_mean) * math.sqrt(ns)
            if score > best[0]:
                best = (score, L, phi)
    return best[1], best[2]


def pos_class(t, L, phi):
    pos = (t - phi) % L
    if pos == 0:
        return 0
    if L % 2 == 0 and pos == L // 2:
        return 1
    return 2 if pos % 2 == 0 else 3


class Model:
    def fit(self, rows):
        nd = len(DUR_VOCAB)
        big = np.ones((nd, nd)) * 0.5
        pha = np.ones((4, nd)) * 0.5
        stg = np.ones((4, nd)) * 0.5
        posdur = np.ones((4, nd)) * 0.5
        big2 = np.zeros((4, nd, nd))
        iv = np.ones(2 * MAX_INT + 1) * 0.5
        iv2 = np.ones((6, 2 * MAX_INT + 1)) * 0.5
        rest_after = np.array([1.0, 1.0])
        tot_after = np.array([2.0, 2.0])
        deg = np.ones((2, 12)) * 0.5
        degbig = np.ones((2, 12, 12)) * 0.5
        restc = np.ones((2, 8, 2))
        firstlong = np.ones((2, 2))
        lastafter = np.ones((2, 2))
        for r in rows:
            sb = strength_map(r)
            root, kmode = estimate_key(r)
            L_, phi_ = est_meter(r)
            seq = r['before'] + r['answer'] + r['after']
            prev_d = None
            prev_p = None
            prev_rest = False
            prev_cls = None
            for e in seq:
                d = dbucket(e['duration_tick'])
                if prev_d is not None:
                    big[DUR_IDX[prev_d], DUR_IDX[d]] += 1
                    big2[pos_class(e['onset_tick'], L_, phi_), DUR_IDX[prev_d], DUR_IDX[d]] += 1
                pha[e['onset_tick'] % 4, DUR_IDX[d]] += 1
                stg[sb.get(e['onset_tick'], 0), DUR_IDX[d]] += 1
                posdur[pos_class(e['onset_tick'], L_, phi_), DUR_IDX[d]] += 1
                prev_d = d
                if e['pitch'] == -1:
                    rest_after[1 if prev_rest else 0] += 1
                    tot_after[1 if prev_rest else 0] += 1
                    prev_rest = True
                else:
                    tot_after[1 if prev_rest else 0] += 1
                    dg = (e['pitch'] - root) % 12
                    deg[kmode, dg] += 1
                    if prev_p is not None:
                        delta = max(-MAX_INT, min(MAX_INT, e['pitch'] - prev_p))
                        iv[delta + MAX_INT] += 1
                        cls_now = 5 if prev_rest else prev_cls
                        if cls_now is not None:
                            iv2[cls_now, delta + MAX_INT] += 1
                        prev_cls = 5 if prev_rest else icls(delta)
                        degbig[kmode, (prev_p - root) % 12, dg] += 1
                    prev_p = e['pitch']
                    prev_rest = False
        for r in rows:
            prev_rest = r['before'][-1]['pitch'] == -1 if r['before'] else False
            before_long = r['before'][-1]['duration_tick'] >= 8 if r['before'] else False
            after_rest = r['after'][0]['pitch'] == -1 if r['after'] else False
            ans = r['answer']
            for i, e in enumerate(ans):
                isrest = 1 if e['pitch'] == -1 else 0
                db = min(e['duration_tick'], 8) - 1
                restc[1 if prev_rest else 0, db, isrest] += 1
                if i == 0:
                    firstlong[1 if before_long else 0, isrest] += 1
                if i == len(ans) - 1:
                    lastafter[1 if after_rest else 0, isrest] += 1
                prev_rest = isrest == 1
        P1 = big / big.sum(axis=1, keepdims=True)
        self.log_big = np.log(P1)
        big2s = big2 + 5.0 * P1[None, :, :]
        self.log_big2 = np.log(big2s / big2s.sum(axis=2, keepdims=True))
        self.log_pha = np.log(pha / pha.sum(axis=1, keepdims=True))
        self.log_stg = np.log(stg / stg.sum(axis=1, keepdims=True))
        self.log_posdur = np.log(posdur / posdur.sum(axis=1, keepdims=True))
        self.log_iv = np.log(iv / iv.sum())
        self.log_iv2 = np.log(iv2 / iv2.sum(axis=1, keepdims=True))
        self.p_rest = rest_after / tot_after
        self.rest_logit = np.log(restc[:, :, 1] / restc[:, :, 0])
        fl = np.log(firstlong[:, 1] / firstlong[:, 0])
        self.first_long_delta = fl[1] - fl[0]
        la = np.log(lastafter[:, 1] / lastafter[:, 0])
        self.last_after_delta = la[1] - la[0]
        self.log_deg = np.log(deg / deg.sum(axis=1, keepdims=True))
        self.log_degbig = np.log(degbig / degbig.sum(axis=2, keepdims=True))
        return self

    def rhythm_candidates(self, row, topk=10, beam=24):
        after = row['after']
        G = after[0]['onset_tick']
        N = row['n']
        if G < N or G <= 0:
            return None
        sb = strength_map(row)
        Lm, phi = est_meter(row)
        acc_on = {c['onset_tick'] for c in row['chords']}
        ctx_durs = {e['duration_tick'] for e in row['before'] + row['after']}
        L = W['lagL']
        ctx_lag = {(e['onset_tick'] % L, e['duration_tick'])
                   for e in row['before'] + row['after']}
        prev0 = dbucket(row['before'][-1]['duration_tick']) if row['before'] else 2
        after_d = dbucket(after[0]['duration_tick'])
        durs = [d for d in DUR_VOCAB if d <= G - (N - 1)]
        states = {(0, 0): [(0.0, ())]}
        for k in range(N):
            for t in range(G):
                entries = states.get((k, t))
                if not entries:
                    continue
                for sc, seq in entries:
                    pi = DUR_IDX[dbucket(seq[-1])] if seq else DUR_IDX[prev0]
                    for d in durs:
                        t2 = t + d
                        if t2 > G or t2 + (N - k - 2) > G:
                            continue
                        if k == N - 1 and t2 != G:
                            continue
                        di = DUR_IDX[d]
                        s = sc + self.log_big2[pos_class(t, Lm, phi), pi, di]
                        s += W['phase'] * self.log_pha[t % 4, di]
                        s += W['strength'] * self.log_stg[sb.get(t, 0), di]
                        s += W['posdur'] * self.log_posdur[pos_class(t, Lm, phi), di]
                        if (t2 - phi) % Lm == 0:
                            s += W['endbar']
                        if t in acc_on:
                            s += W['grid_on']
                        if t2 in acc_on or t2 == G:
                            s += W['grid_off']
                        if d in ctx_durs:
                            s += W['ctxdur']
                        if (t % L, d) in ctx_lag:
                            s += W['lag']
                        if k == N - 1:
                            s += self.log_big[di, DUR_IDX[after_d]]
                        states.setdefault((k + 1, t2), []).append((s, seq + (d,)))
            for t in range(G + 1):
                key = (k + 1, t)
                if key in states and len(states[key]) > beam:
                    states[key].sort(key=lambda x: -x[0])
                    del states[key][beam:]
        final = sorted(states.get((N, G), []), key=lambda x: -x[0])
        return [(sc, list(seq)) for sc, seq in final[:topk]]

    def pitch_decode(self, row, onsets, durs):
        before, after = row['before'], row['after']
        N = len(onsets)
        ctx_p = [e['pitch'] for e in before + after if e['pitch'] != -1]
        if not ctx_p:
            ctx_p = [72]
        lo, hi = min(ctx_p) - 4, max(ctx_p) + 4
        cands = list(range(lo, hi + 1))
        P = len(cands)
        chords = row['chords']
        ch_on = [c['onset_tick'] for c in chords]
        pc = np.ones(12) * 0.5
        for c in chords:
            for p in c['pitches']:
                pc[p % 12] += 1
        for p in ctx_p:
            pc[p % 12] += 2
        log_pc = np.log(pc / pc.sum())
        ctx_set = set(ctx_p)
        lo0, hi0 = min(ctx_p), max(ctx_p)

        def chord_at(t):
            best = None
            for i, o in enumerate(ch_on):
                if o <= t:
                    best = i
                else:
                    break
            return set(chords[best]['pitches']) if best is not None else set()

        root, kmode = estimate_key(row)
        acc_on = {c['onset_tick'] for c in chords}
        ctx_pitch_at = {e['onset_tick']: e['pitch'] for e in before + after if e['pitch'] != -1}
        pB = next((e['pitch'] for e in reversed(before) if e['pitch'] != -1), None)
        pA = next((e['pitch'] for e in after if e['pitch'] != -1), None)
        G = after[0]['onset_tick'] if after else (onsets[-1] + durs[-1])
        em = np.zeros((N, P))
        carr = np.array(cands)
        for i in range(N):
            t = onsets[i]
            ch = chord_at(t)
            ch12 = {p % 12 for p in ch}
            on_beat = t in acc_on
            w_ch = W['chord'] if on_beat else W['chord'] * W['offbeat']
            w_ch12 = W['chord12u'] if on_beat else W['chord12u'] * W['offbeat']
            w_chm = W['chordm'] if on_beat else W['chordm'] * W['offbeat']
            if pB is not None and pA is not None and G > 0:
                interp = pB + (pA - pB) * (t / G)
            else:
                interp = None
            top = max(ch) if ch else None
            L = W['lagL']
            lag_p = {ctx_pitch_at.get(t - L), ctx_pitch_at.get(t + L)}
            lag_p.discard(None)
            for j, p in enumerate(cands):
                s = W['pc'] * log_pc[p % 12]
                s += W['deg'] * self.log_deg[kmode, (p - root) % 12]
                if p in ch:
                    s += w_ch
                elif p - 12 in ch:
                    s += w_ch12
                elif p % 12 in ch12:
                    s += w_chm
                if top is not None:
                    if p == top:
                        s += W['top']
                    elif p == top + 12:
                        s += W['top12']
                if p in lag_p:
                    s += W['lagp']
                if p in ctx_set:
                    s += W['reuse']
                if p < lo0:
                    s -= W['range'] * (lo0 - p)
                elif p > hi0:
                    s -= W['range'] * (p - hi0)
                if interp is not None:
                    s -= W['interp'] * abs(p - interp)
                em[i, j] = s
        deltas = np.clip(carr[None, :] - carr[:, None], -MAX_INT, MAX_INT)
        dg = (carr - root) % 12
        base_deg = self.log_degbig[kmode][dg[:, None], dg[None, :]] * W['degbig']
        base_iv = self.log_iv[deltas + MAX_INT] * W['iv']
        B = np.empty((6, P, P))
        for c in range(6):
            B[c] = base_iv + self.log_iv2[c][deltas + MAX_INT] * W['iv2'] + base_deg
        raw_delta = carr[None, :] - carr[:, None]
        cls_of = np.select(
            [raw_delta <= -3, raw_delta < 0, raw_delta == 0, raw_delta <= 2],
            [0, 1, 2, 3], default=4)
        cls_mask = [np.where(cls_of == c, 0.0, -1e18) for c in range(N_CLS)]

        before_long = before[-1]['duration_tick'] >= 8 if before else False
        after_rest0 = after[0]['pitch'] == -1 if after else False
        rest_lp = []
        for i in range(N):
            db = min(durs[i], 8) - 1
            pair = []
            for prev in (0, 1):
                logit = self.rest_logit[prev, db] + W['rest']
                if i == 0 and before_long:
                    logit += self.first_long_delta
                if i == N - 1 and after_rest0:
                    logit += self.last_after_delta
                lr = -math.log1p(math.exp(-logit))
                lnr = -math.log1p(math.exp(logit))
                pair.append((lr, lnr))
            rest_lp.append(pair)

        NEG = -1e18
        S = 6 * P
        last_sound = next((e['pitch'] for e in reversed(before) if e['pitch'] != -1), None)
        prev_is_rest = before[-1]['pitch'] == -1 if before else False
        v = np.full(S, NEG)
        if last_sound is None:
            last_sound = (lo0 + hi0) // 2
        j0 = min(range(P), key=lambda j: abs(cands[j] - last_sound))
        if prev_is_rest:
            v[5 * P + j0] = 0.0
        else:
            bp_seq = [e['pitch'] for e in before if e['pitch'] != -1]
            if len(bp_seq) >= 2:
                c0 = icls(bp_seq[-1] - bp_seq[-2])
                v[c0 * P + j0] = 0.0
            else:
                for c in range(N_CLS):
                    v[c * P + j0] = 0.0
        bp = np.zeros((N, S), dtype=np.int32)

        force_first = force_last = None
        if before and before[-1]['tie'] in ('start', 'continue') and before[-1]['pitch'] != -1:
            force_first = before[-1]['pitch']
        if after and after[0]['tie'] in ('stop', 'continue') and after[0]['pitch'] != -1:
            force_last = after[0]['pitch']

        ar = np.arange(P)
        for i in range(N):
            (lr_n, lnr_n), (lr_r, lnr_r) = rest_lp[i]
            Mn = B + em[i][None, None, :]
            mx = Mn.max(axis=2, keepdims=True)
            Mn = Mn - (mx + np.log(np.exp(Mn - mx).sum(axis=2, keepdims=True)))
            vs = v[:5 * P].reshape(5, P)
            vr = v[5 * P:]
            src_sound = vs[:, :, None] + Mn[:5] + lnr_n
            best_c = np.argmax(src_sound, axis=0)
            A_sound = np.take_along_axis(src_sound, best_c[None], axis=0)[0]
            A_rest = vr[:, None] + Mn[5] + lnr_r
            from_rest = A_rest > A_sound
            A = np.where(from_rest, A_rest, A_sound)
            src_state = np.where(from_rest, 5 * P + ar[:, None], best_c * P + ar[:, None])
            new_v = np.full(S, NEG)
            new_bp = np.zeros(S, dtype=np.int32)
            for c in range(N_CLS):
                Ac = A + cls_mask[c]
                pbest = np.argmax(Ac, axis=0)
                new_v[c * P: (c + 1) * P] = Ac[pbest, ar]
                new_bp[c * P: (c + 1) * P] = src_state[pbest, ar]
            stay_note = vs + lr_n
            cbest = np.argmax(stay_note, axis=0)
            best_note = stay_note[cbest, ar]
            rest_stay = vr + lr_r
            use_rest = rest_stay > best_note
            new_v[5 * P:] = np.where(use_rest, rest_stay, best_note)
            new_bp[5 * P:] = np.where(use_rest, 5 * P + ar, cbest * P + ar)
            if i == 0 and force_first is not None and lo <= force_first <= hi:
                mask = np.full(S, NEG)
                jj = force_first - lo
                for c in range(N_CLS):
                    mask[c * P + jj] = 0.0
                new_v = new_v + mask
            if i == N - 1:
                if force_last is not None and lo <= force_last <= hi:
                    mask = np.full(S, NEG)
                    jj = force_last - lo
                    for c in range(N_CLS):
                        mask[c * P + jj] = 0.0
                    new_v = new_v + mask
                else:
                    nxt = next((e['pitch'] for e in after if e['pitch'] != -1), None)
                    if nxt is not None:
                        dnx = np.clip(nxt - carr, -MAX_INT, MAX_INT) + MAX_INT
                        for c in range(N_CLS):
                            out_iv = (self.log_iv[dnx] * W['iv']
                                      + self.log_iv2[c][dnx] * W['iv2']) * W['out']
                            new_v[c * P: (c + 1) * P] += out_iv
                        new_v[5 * P:] += (self.log_iv[dnx] * W['iv']
                                          + self.log_iv2[5][dnx] * W['iv2']) * W['out']
            bp[i] = new_bp
            v = new_v
        j = int(np.argmax(v))
        best_score = float(v[j])
        out = []
        for i in range(N - 1, -1, -1):
            if j >= 5 * P:
                out.append(-1)
            else:
                out.append(cands[j % P])
            j = int(bp[i, j])
        out.reverse()
        return best_score, out

    def predict(self, row, rhythm_topk=10, mode_lookup=None):
        after = row['after']
        N = row['n']
        if not after:
            d = row['before'][-1]['duration_tick'] if row['before'] else 2
            ons = [i * d for i in range(N)]
            drs = [d] * N
            _, pitches = self.pitch_decode(row, ons, drs)
            return self._emit(row, ons, drs, pitches)
        G = after[0]['onset_tick']
        rc = self.rhythm_candidates(row, rhythm_topk)
        if not rc:
            if mode_lookup and (G, N) in mode_lookup:
                ons, drs = mode_lookup[(G, N)]
                ons, drs = list(ons), list(drs)
            else:
                ons = [min(i, max(G - 1, 0)) for i in range(N)]
                drs = [1] * N
            _, pitches = self.pitch_decode(row, ons, drs)
            return self._emit(row, ons, drs, pitches)
        best = None
        for rs, dseq in rc:
            ons = [0]
            for d in dseq[:-1]:
                ons.append(ons[-1] + d)
            ps, pitches = self.pitch_decode(row, ons, dseq)
            tot = W['rhy'] * rs + ps
            if best is None or tot > best[0]:
                best = (tot, ons, dseq, pitches)
        _, ons, drs, pitches = best
        return self._emit(row, ons, drs, pitches)

    def _emit(self, row, ons, drs, pitches):
        N = row['n']
        before, after = row['before'], row['after']
        ties = ['none'] * N
        pitches = list(pitches)
        if before and before[-1]['tie'] in ('start', 'continue') and before[-1]['pitch'] != -1:
            pitches[0] = before[-1]['pitch']
            ties[0] = 'stop'
        if after and after[0]['tie'] in ('stop', 'continue') and after[0]['pitch'] != -1:
            pitches[-1] = after[0]['pitch']
            ties[-1] = 'start'
            if N == 1 and ties[0] == 'stop':
                ties[0] = 'continue'
        return [{'onset_tick': int(o), 'duration_tick': int(d), 'pitch': int(p),
                 'tie': ties[i], 'voice_token': row['voice']}
                for i, (o, d, p) in enumerate(zip(ons, drs, pitches))]


def fit_mode_lookup(rows):
    m = {}
    for r in rows:
        if not r['after']:
            continue
        G = r['after'][0]['onset_tick']
        durs = tuple(e['duration_tick'] for e in r['answer'])
        ons = tuple(e['onset_tick'] for e in r['answer'])
        m.setdefault((G, r['n']), Counter())[(ons, durs)] += 1
    return {k: c.most_common(1)[0][0] for k, c in m.items()}


class LearnedPitchRanker:
    """Train-only candidate ranker used as the pitch emission model."""

    def _cache(self, row):
        after = row['after']
        gap = after[0]['onset_tick'] if after else max(row['n'], 1) * 2
        context = row['before'] + after
        pitches = [e['pitch'] for e in context if e['pitch'] != -1] or [72]
        return {
            'gap': gap,
            'pitches': pitches,
            'chord_onsets': {c['onset_tick'] for c in row['chords']},
            'context_events': {e['onset_tick']: e for e in context},
            'context_pitches': {
                e['onset_tick']: e['pitch'] for e in context if e['pitch'] != -1
            },
            'root_mode': estimate_key(row),
        }

    @staticmethod
    def _context_fields(events, gap, after):
        out = []
        for i in range(6):
            if i < len(events):
                e = events[i]
                onset = e['onset_tick'] - gap if after else e['onset_tick']
                out.extend((onset, e['duration_tick'], e['pitch'],
                            int(e['pitch'] == -1),
                            {'none': 0, 'start': 1, 'stop': 2,
                             'continue': 3}.get(e['tie'], 0)))
            else:
                out.extend((0, 0, -1, 1, 0))
        return out

    def _event_base(self, row, i, onset, duration, cache):
        gap = cache['gap']
        n_events = row['n']
        out = [
            n_events, gap, i, i / max(n_events - 1, 1), onset, gap - onset,
            (gap - onset) / max(n_events - i, 1), duration,
        ]
        out.extend(self._context_fields(row['before'], gap, False))
        out.extend(self._context_fields(row['after'], gap, True))

        chord_onsets = cache['chord_onsets']
        future = sorted(o - onset for o in chord_onsets if o >= onset)[:6]
        past = sorted(onset - o for o in chord_onsets if o < onset)[:3]
        out.extend((int(onset in chord_onsets),
                    int(onset + duration in chord_onsets),
                    int(onset + duration == gap)))
        out.extend(future)
        out.extend([99] * (6 - len(future)))
        out.extend(past)
        out.extend([99] * (3 - len(past)))
        out.extend(int(onset + d in chord_onsets or onset + d == gap)
                   for d in DUR_VOCAB)

        context_events = cache['context_events']
        for lag in (2, 3, 4, 6, 8, 12, 16, 24):
            for tick in (onset - lag, onset + lag):
                event = context_events.get(tick)
                out.extend((event['duration_tick'] if event else 0,
                            int(event is not None)))
        out.extend(onset % modulus for modulus in (2, 3, 4, 6, 8, 12, 16, 24))
        return out

    @staticmethod
    def _chord_at(row, tick):
        current = None
        for chord in row['chords']:
            if chord['onset_tick'] > tick:
                break
            current = chord
        return current

    def _event_info(self, row, i, onset, duration, cache):
        context_pitches = cache['pitches']
        pitch_before = next(
            (e['pitch'] for e in reversed(row['before']) if e['pitch'] != -1),
            round(float(np.mean(context_pitches))),
        )
        pitch_after = next(
            (e['pitch'] for e in row['after'] if e['pitch'] != -1),
            round(float(np.mean(context_pitches))),
        )
        gap = cache['gap']
        interpolation = pitch_before + (pitch_after - pitch_before) * (
            onset / max(gap, 1)
        )
        chord_pitches = []
        for tick in (onset, onset + duration - 1, onset + duration):
            chord = self._chord_at(row, tick)
            chord_pitches.append(chord['pitches'] if chord else [])
        return {
            'base': self._event_base(row, i, onset, duration, cache),
            'context_pitches': context_pitches,
            'pitch_before': pitch_before,
            'pitch_after': pitch_after,
            'interpolation': interpolation,
            'chord_pitches': chord_pitches,
            'root_mode': cache['root_mode'],
            'context_pitch_at': cache['context_pitches'],
            'onset': onset,
        }

    @staticmethod
    def _features(info, pitch):
        out = list(info['base'])
        context_pitches = info['context_pitches']
        is_rest = int(pitch == -1)
        numeric_pitch = float(np.mean(context_pitches)) if is_rest else pitch
        pitch_before = info['pitch_before']
        pitch_after = info['pitch_after']
        interpolation = info['interpolation']
        root, mode = info['root_mode']
        out.extend((
            is_rest, numeric_pitch, numeric_pitch - pitch_before,
            numeric_pitch - pitch_after, numeric_pitch - interpolation,
            abs(numeric_pitch - interpolation), min(context_pitches),
            max(context_pitches), float(np.mean(context_pitches)),
            float(np.std(context_pitches)),
            int(not is_rest and pitch in context_pitches),
            min(abs(numeric_pitch - p) for p in context_pitches),
            (numeric_pitch - root) % 12, mode,
        ))
        for event_pitch in context_pitches:
            out.extend((numeric_pitch - event_pitch if not is_rest else 99,
                        int(not is_rest and pitch == event_pitch)))

        # The problem promises six events on either side. Pad the sounded-only
        # view so feature width remains constant when some context events rest.
        sounded_count = len(context_pitches)
        for _ in range(12 - sounded_count):
            out.extend((99, 0))

        for chord_pitches in info['chord_pitches']:
            if chord_pitches and not is_rest:
                pitch_classes = {p % 12 for p in chord_pitches}
                top = max(chord_pitches)
                bass = min(chord_pitches)
                out.extend((
                    len(chord_pitches), int(pitch in chord_pitches),
                    int(pitch % 12 in pitch_classes), pitch - top, pitch - bass,
                    min(abs(pitch - p) for p in chord_pitches),
                    int(pitch == top), int(pitch == top + 12),
                ))
            else:
                out.extend((len(chord_pitches), 0, 0, 99, 99, 99, 0, 0))

        context_pitch_at = info['context_pitch_at']
        onset = info['onset']
        for lag in (2, 3, 4, 6, 8, 12, 16, 24):
            nearby = (context_pitch_at.get(onset - lag),
                      context_pitch_at.get(onset + lag))
            distances = [
                abs(pitch - p) for p in nearby
                if p is not None and not is_rest
            ]
            out.extend((int(not is_rest and pitch in nearby),
                        min(distances) if distances else 99))
        return out

    @staticmethod
    def _conditional_tail(info, current_pitch, previous_pitch):
        previous_rest = int(previous_pitch == -1)
        current_rest = int(current_pitch == -1)
        previous_numeric = (
            info['pitch_before'] if previous_rest else previous_pitch
        )
        current_numeric = (
            info['interpolation'] if current_rest else current_pitch
        )
        delta = (
            99 if previous_rest or current_rest
            else current_pitch - previous_pitch
        )
        return [
            previous_rest, previous_numeric, current_rest, current_numeric,
            delta, abs(delta) if delta != 99 else 99, int(delta == 0),
            int(delta in (-2, -1, 1, 2)),
            int(delta != 99 and abs(delta) >= 3),
            int(not previous_rest and not current_rest
                and previous_pitch % 12 == current_pitch % 12),
        ]

    @staticmethod
    def _candidate_pitches(cache, true_pitch=None):
        pitches = cache['pitches']
        candidates = list(range(min(pitches) - 8, max(pitches) + 9))
        candidates.append(-1)
        if true_pitch is not None and true_pitch not in candidates:
            candidates.append(true_pitch)
        return candidates

    def fit(self, rows):
        n_samples = 0
        for row in rows:
            cache = self._cache(row)
            base_count = len(self._candidate_pitches(cache))
            for event in row['answer']:
                n_samples += base_count + int(
                    event['pitch'] not in self._candidate_pitches(cache)
                )

        first_row = rows[0]
        first_cache = self._cache(first_row)
        first_event = first_row['answer'][0]
        first_info = self._event_info(
            first_row, 0, first_event['onset_tick'],
            first_event['duration_tick'], first_cache,
        )
        self.feature_count = len(
            self._features(first_info, first_event['pitch'])
        )
        conditional_count = self.feature_count + len(
            self._conditional_tail(
                first_info, first_event['pitch'],
                first_row['before'][-1]['pitch'],
            )
        )
        features = np.empty(
            (n_samples, conditional_count), dtype=np.float32
        )
        labels = np.zeros(n_samples, dtype=np.float32)
        groups = np.empty(n_samples, dtype=np.int32)

        position = 0
        group = 0
        for row in rows:
            cache = self._cache(row)
            previous_pitch = (
                row['before'][-1]['pitch'] if row['before'] else -1
            )
            for i, event in enumerate(row['answer']):
                info = self._event_info(
                    row, i, event['onset_tick'], event['duration_tick'], cache,
                )
                for pitch in self._candidate_pitches(cache, event['pitch']):
                    features[position, :self.feature_count] = self._features(
                        info, pitch
                    )
                    features[position, self.feature_count:] = (
                        self._conditional_tail(
                            info, pitch, previous_pitch
                        )
                    )
                    labels[position] = float(pitch == event['pitch'])
                    groups[position] = group
                    position += 1
                previous_pitch = event['pitch']
                group += 1
        assert position == n_samples

        common = {
            'loss_function': 'QuerySoftMax',
            'l2_leaf_reg': 5.0,
            'thread_count': 8,
            'verbose': False,
            'allow_writing_files': False,
        }
        self.ranker = CatBoostRanker(
            iterations=550,
            depth=8,
            learning_rate=0.07,
            random_strength=0.3,
            random_seed=127,
            **common,
        )
        independent_features = np.ascontiguousarray(
            features[:, :self.feature_count]
        )
        self.ranker.fit(
            independent_features, labels, group_id=groups
        )
        del independent_features
        gc.collect()
        self.conditional_ranker = CatBoostRanker(
            iterations=450,
            depth=8,
            learning_rate=0.07,
            random_strength=0.3,
            random_seed=223,
            **common,
        )
        self.conditional_ranker.fit(features, labels, group_id=groups)
        del features, labels, groups
        gc.collect()

        def reverse_candidates(cache, true_pitch):
            candidates = self._candidate_pitches(cache, true_pitch)
            return [
                pitch for i, pitch in enumerate(candidates)
                if i % 3 == 0 or pitch in (-1, true_pitch)
            ]

        reverse_samples = 0
        for row in rows:
            cache = self._cache(row)
            reverse_samples += sum(
                len(reverse_candidates(cache, event['pitch']))
                for event in row['answer']
            )
        reverse_features = np.empty(
            (reverse_samples, self.feature_count + 10), dtype=np.float32
        )
        reverse_labels = np.zeros(reverse_samples, dtype=np.float32)
        reverse_groups = np.empty(reverse_samples, dtype=np.int32)
        position = 0
        group = 0
        for row in rows:
            cache = self._cache(row)
            for i, event in enumerate(row['answer']):
                next_pitch = (
                    row['answer'][i + 1]['pitch']
                    if i + 1 < len(row['answer'])
                    else (row['after'][0]['pitch'] if row['after'] else -1)
                )
                info = self._event_info(
                    row, i, event['onset_tick'],
                    event['duration_tick'], cache,
                )
                for pitch in reverse_candidates(cache, event['pitch']):
                    reverse_features[
                        position, :self.feature_count
                    ] = self._features(info, pitch)
                    reverse_features[
                        position, self.feature_count:
                    ] = self._conditional_tail(
                        info, pitch, next_pitch
                    )
                    reverse_labels[position] = float(
                        pitch == event['pitch']
                    )
                    reverse_groups[position] = group
                    position += 1
                group += 1
        assert position == reverse_samples
        self.reverse_ranker = CatBoostRanker(
            iterations=150,
            depth=7,
            learning_rate=0.08,
            loss_function='QuerySoftMax',
            l2_leaf_reg=5.0,
            random_strength=0.3,
            random_seed=337,
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
        self.reverse_ranker.fit(
            reverse_features, reverse_labels, group_id=reverse_groups
        )
        del reverse_features, reverse_labels, reverse_groups
        gc.collect()
        return self

    @staticmethod
    def _forced_pitches(row):
        first = last = None
        if (row['before']
                and row['before'][-1]['tie'] in ('start', 'continue')
                and row['before'][-1]['pitch'] != -1):
            first = row['before'][-1]['pitch']
        if (row['after']
                and row['after'][0]['tie'] in ('stop', 'continue')
                and row['after'][0]['pitch'] != -1):
            last = row['after'][0]['pitch']
        return first, last

    def _decode_independent(
        self, row, candidates, emissions, base_model
    ):
        context_pitches = [
            e['pitch'] for e in row['before'] + row['after']
            if e['pitch'] != -1
        ] or [72]
        pitch_before = next(
            (e['pitch'] for e in reversed(row['before'])
             if e['pitch'] != -1),
            round(float(np.mean(context_pitches))),
        )
        pitch_after = next(
            (e['pitch'] for e in row['after'] if e['pitch'] != -1),
            round(float(np.mean(context_pitches))),
        )
        previous_rest = int(bool(row['before'])
                            and row['before'][-1]['pitch'] == -1)
        rest_probability = np.clip(base_model.p_rest, 1e-6, 1 - 1e-6)
        rest_logs = np.log(rest_probability)
        sound_logs = np.log1p(-rest_probability)
        rest_weight = 0.2
        transition_weight = 0.1
        sounding = candidates != -1

        current = emissions[0].copy()
        current[sounding] += (
            rest_weight * sound_logs[previous_rest]
            + transition_weight * base_model.log_iv[
                np.clip(candidates[sounding] - pitch_before,
                        -MAX_INT, MAX_INT) + MAX_INT
            ]
        )
        current[~sounding] += rest_weight * rest_logs[previous_rest]
        force_first, force_last = self._forced_pitches(row)
        if force_first is not None:
            current[candidates != force_first] = -1e18

        source = candidates[:, None]
        target = candidates[None, :]
        source_rest = source == -1
        target_rest = target == -1
        transition = np.empty(
            (len(candidates), len(candidates)), dtype=float
        )
        both_sound = ~source_rest & ~target_rest
        transition[both_sound] = (
            rest_weight * sound_logs[0]
            + transition_weight * base_model.log_iv[
                np.clip((target - source)[both_sound],
                        -MAX_INT, MAX_INT) + MAX_INT
            ]
        )
        to_rest = np.broadcast_to(target_rest, transition.shape)
        from_rest = np.broadcast_to(source_rest, transition.shape)
        transition[to_rest & ~from_rest] = rest_weight * rest_logs[0]
        transition[to_rest & from_rest] = rest_weight * rest_logs[1]
        transition[~to_rest & from_rest] = rest_weight * sound_logs[1]

        backpointers = []
        for i in range(1, len(emissions)):
            scores = current[:, None] + transition + emissions[i][None, :]
            backpointer = np.argmax(scores, axis=0)
            current = scores[backpointer, np.arange(len(candidates))]
            backpointers.append(backpointer)
        if force_last is not None:
            current[candidates != force_last] = -1e18
        else:
            current[sounding] += transition_weight * base_model.log_iv[
                np.clip(pitch_after - candidates[sounding],
                        -MAX_INT, MAX_INT) + MAX_INT
            ]

        state = int(np.argmax(current))
        decoded = [int(candidates[state])]
        for backpointer in reversed(backpointers):
            state = int(backpointer[state])
            decoded.append(int(candidates[state]))
        decoded.reverse()
        return float(np.max(current)), decoded

    def score_rhythms(
        self, row, base_model, topk=24, rhythm_candidates=None
    ):
        if rhythm_candidates is None:
            rhythm_candidates = base_model.rhythm_candidates(row, topk)
        if not rhythm_candidates:
            return []
        cache = self._cache(row)
        candidates = np.asarray(
            self._candidate_pitches(cache), dtype=np.int32
        )
        n_candidates = len(candidates)
        feature_count = self.feature_count
        total = sum(len(durations) * n_candidates
                    for _, durations in rhythm_candidates)
        features = np.empty((total, feature_count), dtype=np.float32)
        slices = []
        decoded_rhythms = []
        position = 0
        for rhythm_score, durations in rhythm_candidates:
            onsets = [0]
            for duration in durations[:-1]:
                onsets.append(onsets[-1] + duration)
            start = position
            for i, (onset, duration) in enumerate(zip(onsets, durations)):
                info = self._event_info(
                    row, i, onset, duration, cache
                )
                for pitch in candidates:
                    features[position] = self._features(info, int(pitch))
                    position += 1
            slices.append((start, position))
            decoded_rhythms.append(
                (rhythm_score, onsets, durations)
            )
        raw = self.ranker.predict(features)
        output = []
        for (rhythm_score, onsets, durations), (start, end) in zip(
                decoded_rhythms, slices):
            emissions = raw[start:end].reshape(
                len(durations), n_candidates
            )
            pitch_score, _ = self._decode_independent(
                row, candidates, emissions, base_model
            )
            output.append((
                rhythm_score, pitch_score, onsets, durations, emissions
            ))
        return output

    def predict_conditional(
        self, row, onsets, durations, independent_emissions=None
    ):
        cache = self._cache(row)
        candidates = np.asarray(
            self._candidate_pitches(cache), dtype=np.int32
        )
        n_candidates = len(candidates)
        n_events = len(onsets)
        infos = [
            self._event_info(row, i, onset, duration, cache)
            for i, (onset, duration) in enumerate(zip(onsets, durations))
        ]
        base_features = np.empty(
            (n_events, n_candidates, self.feature_count), dtype=np.float32
        )
        for i, info in enumerate(infos):
            for j, pitch in enumerate(candidates):
                base_features[i, j] = self._features(info, int(pitch))
        if independent_emissions is None:
            independent_emissions = self.ranker.predict(
                base_features.reshape(
                    n_events * n_candidates, self.feature_count
                )
            ).reshape(n_events, n_candidates)

        conditional_count = (
            n_candidates + (n_events - 1) * n_candidates * n_candidates
        )
        conditional_features = np.empty(
            (conditional_count, self.feature_count + 10), dtype=np.float32
        )
        position = 0
        previous = (
            row['before'][-1]['pitch'] if row['before'] else -1
        )
        for j, pitch in enumerate(candidates):
            conditional_features[position, :self.feature_count] = (
                base_features[0, j]
            )
            conditional_features[position, self.feature_count:] = (
                self._conditional_tail(infos[0], int(pitch), previous)
            )
            position += 1
        for i in range(1, n_events):
            for previous in candidates:
                for j, pitch in enumerate(candidates):
                    conditional_features[
                        position, :self.feature_count
                    ] = base_features[i, j]
                    conditional_features[
                        position, self.feature_count:
                    ] = self._conditional_tail(
                        infos[i], int(pitch), int(previous)
                    )
                    position += 1
        conditional = self.conditional_ranker.predict(
            conditional_features
        )
        position = 0
        for i in range(n_events - 1):
            for j, pitch in enumerate(candidates):
                for next_pitch in candidates:
                    conditional_features[
                        position, :self.feature_count
                    ] = base_features[i, j]
                    conditional_features[
                        position, self.feature_count:
                    ] = self._conditional_tail(
                        infos[i], int(pitch), int(next_pitch)
                    )
                    position += 1
        next_pitch = (
            row['after'][0]['pitch'] if row['after'] else -1
        )
        for j, pitch in enumerate(candidates):
            conditional_features[position, :self.feature_count] = (
                base_features[-1, j]
            )
            conditional_features[position, self.feature_count:] = (
                self._conditional_tail(
                    infos[-1], int(pitch), next_pitch
                )
            )
            position += 1
        assert position == conditional_count
        reverse = self.reverse_ranker.predict(conditional_features)

        current = (
            conditional[:n_candidates]
            + 2.0 * independent_emissions[0]
        )
        force_first, force_last = self._forced_pitches(row)
        if force_first is not None:
            current[candidates != force_first] = -1e18

        backpointers = []
        forward_position = n_candidates
        reverse_position = 0
        reverse_weight = 0.18
        for i in range(1, n_events):
            count = n_candidates * n_candidates
            transition = conditional[
                forward_position:forward_position + count
            ].reshape(n_candidates, n_candidates)
            reverse_transition = reverse[
                reverse_position:reverse_position + count
            ].reshape(n_candidates, n_candidates)
            forward_position += count
            reverse_position += count
            scores = (
                current[:, None] + transition
                + reverse_weight * reverse_transition
                + 2.0 * independent_emissions[i][None, :]
            )
            backpointer = np.argmax(scores, axis=0)
            current = scores[
                backpointer, np.arange(n_candidates)
            ]
            backpointers.append(backpointer)
        current += reverse_weight * reverse[-n_candidates:]
        if force_last is not None:
            current[candidates != force_last] = -1e18
        state = int(np.argmax(current))
        decoded = [int(candidates[state])]
        for backpointer in reversed(backpointers):
            state = int(backpointer[state])
            decoded.append(int(candidates[state]))
        decoded.reverse()
        return decoded

    def predict(self, row, onsets, durations, base_model):
        cache = self._cache(row)
        candidates = np.asarray(
            self._candidate_pitches(cache), dtype=np.int32
        )
        features = []
        for i, (onset, duration) in enumerate(zip(onsets, durations)):
            info = self._event_info(
                row, i, onset, duration, cache
            )
            for pitch in candidates:
                features.append(self._features(info, int(pitch)))
        emissions = self.ranker.predict(
            np.asarray(features, dtype=np.float32)
        ).reshape(len(onsets), len(candidates))
        return self._decode_independent(
            row, candidates, emissions, base_model
        )[1]


class LearnedDurationModel:
    """Adds complementary rhythm candidates from learned duration emissions."""

    def fit(self, rows, feature_builder):
        features = []
        labels = []
        for row in rows:
            cache = feature_builder._cache(row)
            for i, event in enumerate(row['answer']):
                features.append(feature_builder._event_base(
                    row, i, event['onset_tick'], 0, cache
                ))
                labels.append(DUR_IDX[event['duration_tick']])
        self.model = CatBoostClassifier(
            iterations=350,
            depth=8,
            learning_rate=0.08,
            loss_function='MultiClass',
            l2_leaf_reg=4.0,
            random_strength=0.3,
            random_seed=401,
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
        self.model.fit(
            np.asarray(features, dtype=np.float32), labels
        )
        return self

    def _learned_candidates(
        self, row, base_model, feature_builder, topk
    ):
        cache = feature_builder._cache(row)
        gap = cache['gap']
        n_events = row['n']
        minimum = min(DUR_VOCAB)
        if gap <= 0 or gap < n_events * minimum:
            return []

        features = []
        keys = []
        for i in range(n_events):
            upper = gap - (n_events - i) * minimum
            for onset in range(i * minimum, upper + 1):
                features.append(feature_builder._event_base(
                    row, i, onset, 0, cache
                ))
                keys.append((i, onset))
        raw = self.model.predict_proba(
            np.asarray(features, dtype=np.float32)
        )
        log_probabilities = np.full(
            (len(features), len(DUR_VOCAB)), -27.631, dtype=float
        )
        log_probabilities[
            :, np.asarray(self.model.classes_, dtype=int)
        ] = np.log(np.maximum(raw, 1e-12))
        emissions = dict(zip(keys, log_probabilities))

        previous = DUR_IDX[dbucket(
            row['before'][-1]['duration_tick']
            if row['before'] else 2
        )]
        beam = {(0, 0): [(0.0, (), previous)]}
        for i in range(n_events):
            next_beam = defaultdict(list)
            for (_, onset), hypotheses in beam.items():
                for score, sequence, previous_index in hypotheses:
                    for duration_index, duration in enumerate(DUR_VOCAB):
                        end = onset + duration
                        if (end > gap
                                or end + (n_events - i - 1) * minimum > gap
                                or (i == n_events - 1 and end != gap)):
                            continue
                        candidate_score = (
                            score
                            + emissions[(i, onset)][duration_index]
                            + 0.15 * base_model.log_big[
                                previous_index, duration_index
                            ]
                        )
                        next_beam[(i + 1, end)].append((
                            candidate_score,
                            sequence + (duration,),
                            duration_index,
                        ))
            beam = {
                key: sorted(values, reverse=True)[:topk]
                for key, values in next_beam.items()
            }
        return [
            (score, list(sequence))
            for score, sequence, _ in sorted(
                beam.get((n_events, gap), []), reverse=True
            )[:topk]
        ]

    @staticmethod
    def _base_score(row, durations, base_model):
        after = row['after']
        gap = after[0]['onset_tick']
        strengths = strength_map(row)
        meter, phase = est_meter(row)
        chord_onsets = {
            chord['onset_tick'] for chord in row['chords']
        }
        context_durations = {
            event['duration_tick']
            for event in row['before'] + after
        }
        lag = W['lagL']
        lagged_context = {
            (event['onset_tick'] % lag, event['duration_tick'])
            for event in row['before'] + after
        }
        previous = dbucket(
            row['before'][-1]['duration_tick']
            if row['before'] else 2
        )
        onset = 0
        score = 0.0
        for duration in durations:
            previous_index = DUR_IDX[dbucket(previous)]
            duration_index = DUR_IDX[duration]
            end = onset + duration
            position = pos_class(onset, meter, phase)
            score += base_model.log_big2[
                position, previous_index, duration_index
            ]
            score += W['phase'] * base_model.log_pha[
                onset % 4, duration_index
            ]
            score += W['strength'] * base_model.log_stg[
                strengths.get(onset, 0), duration_index
            ]
            score += W['posdur'] * base_model.log_posdur[
                position, duration_index
            ]
            if (end - phase) % meter == 0:
                score += W['endbar']
            if onset in chord_onsets:
                score += W['grid_on']
            if end in chord_onsets or end == gap:
                score += W['grid_off']
            if duration in context_durations:
                score += W['ctxdur']
            if (onset % lag, duration) in lagged_context:
                score += W['lag']
            previous = duration
            onset = end
        score += base_model.log_big[
            DUR_IDX[dbucket(durations[-1])],
            DUR_IDX[dbucket(after[0]['duration_tick'])],
        ]
        return float(score)

    def candidates(
        self, row, base_model, feature_builder, topk=10
    ):
        symbolic = base_model.rhythm_candidates(row, topk) or []
        if not row['after']:
            return symbolic
        learned = self._learned_candidates(
            row, base_model, feature_builder, topk
        )
        output = []
        seen = set()
        for score, durations in symbolic:
            key = tuple(durations)
            if key not in seen:
                seen.add(key)
                output.append((float(score), durations))
        for _, durations in learned:
            key = tuple(durations)
            if key not in seen:
                seen.add(key)
                output.append((
                    self._base_score(row, durations, base_model),
                    durations,
                ))
        return output


class LearnedRhythmRanker:
    """Reranks the symbolic DP's rhythm candidates with a trained model."""

    @staticmethod
    def _features(row, durations, pitch_model):
        cache = pitch_model._cache(row)
        gap = cache['gap']
        n_events = row['n']
        onsets = [0]
        for duration in durations[:-1]:
            onsets.append(onsets[-1] + duration)
        output = pitch_model._event_base(row, 0, 0, 0, cache)
        context = row['before'] + row['after']
        context_durations = [e['duration_tick'] for e in context]
        context_at = cache['context_events']
        chord_onsets = cache['chord_onsets']
        output.extend((
            float(np.mean(durations)), float(np.std(durations)),
            min(durations), max(durations), len(set(durations)),
        ))
        for i in range(7):
            if i >= n_events:
                output.extend([0] * 22)
                continue
            onset = onsets[i]
            duration = durations[i]
            end = onset + duration
            next_offsets = sorted(
                o - onset for o in chord_onsets if o > onset
            )
            values = [
                onset, duration, end, gap - end,
                int(onset in chord_onsets),
                int(end in chord_onsets or end == gap),
                min([abs(onset - o) for o in chord_onsets] + [99]),
                min([abs(end - o) for o in chord_onsets] + [99]),
                next_offsets[0] if next_offsets else 99,
                int(duration in context_durations),
                onset % 2, onset % 3, onset % 4, onset % 6,
                onset % 8, onset % 12,
            ]
            for lag in (4, 6, 8, 12, 16, 24):
                left = context_at.get(onset - lag)
                right = context_at.get(onset + lag)
                values.append(int(bool(
                    (left and left['duration_tick'] == duration)
                    or (right and right['duration_tick'] == duration)
                )))
            output.extend(values)
        for i in range(6):
            if i + 1 < n_events:
                output.extend((
                    durations[i + 1] - durations[i],
                    int(durations[i + 1] == durations[i]),
                ))
            else:
                output.extend((0, 0))

        before_durations = [
            e['duration_tick'] for e in row['before']
        ]
        after_durations = [
            e['duration_tick'] for e in row['after']
        ]
        for sequence in (
                before_durations[-n_events:],
                after_durations[:n_events]):
            output.extend((
                sum(a == b for a, b in zip(durations, sequence)),
                sum(abs(a - b) for a, b in zip(durations, sequence)),
                int(len(sequence) == n_events
                    and list(durations) == list(sequence)),
            ))
        output.extend((
            sum(onset in chord_onsets for onset in onsets),
            sum(onset + duration in chord_onsets
                or onset + duration == gap
                for onset, duration in zip(onsets, durations)),
        ))
        return output

    def fit(
        self, rows, base_model, pitch_model, duration_model
    ):
        groups_data = []
        for row in rows:
            if not row['after']:
                continue
            gap = row['after'][0]['onset_tick']
            true = tuple(
                event['duration_tick'] for event in row['answer']
            )
            contiguous = (
                gap > 0
                and all(
                    event['onset_tick'] == sum(
                        prior['duration_tick']
                        for prior in row['answer'][:i]
                    )
                    for i, event in enumerate(row['answer'])
                )
                and sum(true) == gap
            )
            if not contiguous:
                continue
            candidates = duration_model.candidates(
                row, base_model, pitch_model, 10
            )
            sequences = [true]
            sequences.extend(
                tuple(candidate[1]) for candidate in candidates
                if tuple(candidate[1]) != true
            )
            groups_data.append((row, sequences[:25]))

        n_samples = sum(
            len(sequences) for _, sequences in groups_data
        )
        feature_count = len(self._features(
            groups_data[0][0], groups_data[0][1][0], pitch_model
        ))
        features = np.empty(
            (n_samples, feature_count), dtype=np.float32
        )
        labels = np.zeros(n_samples, dtype=np.float32)
        groups = np.empty(n_samples, dtype=np.int32)
        position = 0
        for group, (row, sequences) in enumerate(groups_data):
            for i, durations in enumerate(sequences):
                features[position] = self._features(
                    row, durations, pitch_model
                )
                labels[position] = float(i == 0)
                groups[position] = group
                position += 1

        self.ranker = CatBoostRanker(
            iterations=450,
            depth=8,
            learning_rate=0.07,
            loss_function='QuerySoftMax',
            l2_leaf_reg=5.0,
            random_strength=0.3,
            random_seed=219,
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
        self.ranker.fit(features, labels, group_id=groups)
        del features, labels, groups, groups_data
        gc.collect()
        return self

    def predict(self, row, scored_rhythms, pitch_model):
        features = np.asarray([
            self._features(row, record[3], pitch_model)
            for record in scored_rhythms
        ], dtype=np.float32)
        return self.ranker.predict(features)


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train = pd.read_csv(public_dir / 'train.csv')
    test = pd.read_csv(public_dir / 'test.csv')
    train_rows = parse_rows(train, with_answer=True)
    test_rows = parse_rows(test, with_answer=False)
    base_model = Model().fit(train_rows)
    pitch_ranker = LearnedPitchRanker().fit(train_rows)
    duration_model = LearnedDurationModel().fit(
        train_rows, pitch_ranker
    )
    rhythm_ranker = LearnedRhythmRanker().fit(
        train_rows, base_model, pitch_ranker, duration_model
    )
    mode_lookup = fit_mode_lookup(train_rows)
    fusion_weights = {
        4: (0.20, 0.50),
        5: (0.40, 0.75),
        6: (0.30, 0.40),
        7: (0.75, 0.75),
    }
    ids, answers = [], []
    for row in test_rows:
        rhythm_candidates = duration_model.candidates(
            row, base_model, pitch_ranker, 10
        )
        scored = pitch_ranker.score_rhythms(
            row, base_model, rhythm_candidates=rhythm_candidates
        )
        if scored:
            rhythm_scores = rhythm_ranker.predict(
                row, scored, pitch_ranker
            )
            rhythm_weight, learned_weight = fusion_weights[row['n']]
            scores = [
                rhythm_weight * record[0] + record[1]
                + learned_weight * rhythm_scores[i]
                for i, record in enumerate(scored)
            ]
            selected = scored[int(np.argmax(scores))]
            onsets, durations, emissions = (
                selected[2], selected[3], selected[4]
            )
        else:
            baseline_events = base_model.predict(row, 10, mode_lookup)
            onsets = [
                event['onset_tick'] for event in baseline_events
            ]
            durations = [
                event['duration_tick'] for event in baseline_events
            ]
            emissions = None
        pitches = pitch_ranker.predict_conditional(
            row, onsets, durations, emissions
        )
        events = base_model._emit(row, onsets, durations, pitches)
        assert len(events) == row['n']
        ids.append(row['id'])
        answers.append(json.dumps(
            {'events': events}, sort_keys=True, separators=(',', ':')
        ))
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': ids, 'answer_json': answers}).to_csv(
        submission_out, index=False
    )
    print(f'wrote {len(ids)} rows to {submission_out}')


if __name__ == '__main__':
    main()
