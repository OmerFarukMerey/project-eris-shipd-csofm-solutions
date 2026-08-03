#!/usr/bin/env python3
"""
Middle Polish Archive Variant Lattice - editorial spelling -> source spelling lattices.

Approach (NLP / seq-to-seq, guidebook 5.1):
  A character-level neural transducer, trained from scratch and entirely in-script on the
  provided training rows.  Each query token is read character by character and the network
  emits one output "piece" per source character; the piece inventory is *induced* from
  Levenshtein alignments of the training (query_token, observed_token) pairs, and the
  source spellings plus their probabilities come out of beam search over the trained
  model.  No lookup table, template, or hand-written spelling rule ever produces an
  answer.

  Few-shot transfer to an unseen source profile is done with a learned memory: the row's
  three calibration examples are aligned, encoded, and cross-attended to by every query
  character, so the network learns *when* to imitate that profile's own habits.  A compact
  style summary of the same calibration evidence is fed in as an extra input vector
  (support features into the trained model, never an answer by itself).

  The lattice decoder is decision-theoretic: it maximises the expected PROBLEM.md row
  score under the model's own predictive distribution, and every free constant it uses is
  searched in-script by coordinate ascent on a profile-disjoint train holdout against the
  exact competition metric.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import sys, os, json, math, time, random, collections, warnings

T_START = time.time()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config
SEED = 1234
TRAIN_DEADLINE = 2350.0     # do not launch a new training run after this
EPOCH_DEADLINE = 2850.0     # break out of a running training loop after this
HARD_DEADLINE = 3200.0      # skip remaining optional work after this
HOLDOUT_FRAC = 0.12
MAX_MODELS = 4
MAXQ, MAXM = 512, 896

WORD = '\x01'               # word-start marker character
COPY = '\x00COPY'           # label meaning "emit the source character unchanged"
PAD, UNK, BOS = 0, 1, 2


def log(*a):
    print('[%7.1fs]' % (time.time() - T_START), *a, flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ alignment / pieces
def align_pieces(a, b):
    """Monotonic Levenshtein alignment of query token -> observed token.

    Returns len(a)+1 output pieces: slot 0 is the prefix-insertion slot (carried by the
    word-start marker), slot k is the output emitted for a[k-1].  Concatenating the
    pieces reproduces b exactly, so the transduction is lossless by construction.
    """
    if a == b:
        return [''] + list(a)
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]; Di = D[i]; Dm = D[i - 1]
        for j in range(1, m + 1):
            v = Dm[j - 1] + (0 if ai == b[j - 1] else 1)
            x = Dm[j] + 1
            if x < v: v = x
            x = Di[j - 1] + 1
            if x < v: v = x
            Di[j] = v
    i, j = n, m
    pieces = [''] * (n + 1)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            pieces[i] = b[j - 1] + pieces[i]; i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            pieces[i] = b[j - 1] + pieces[i]; j -= 1
    return pieces


def lab_of(src_char, piece):
    return COPY if piece == src_char else piece


class Vocab:
    def __init__(self):
        self.chars = {'<pad>': 0, '<unk>': 1}
        self.pieces = {'<pad>': 0, '<unk>': 1, '<bos>': 2}
        self.codes = {'<pad>': 0, '<unk>': 1}

    def build(self, cc, pc, gc, min_piece=2):
        for c, _ in cc.most_common():
            if c not in self.chars: self.chars[c] = len(self.chars)
        for p, n in pc.most_common():
            if n >= min_piece and p not in self.pieces: self.pieces[p] = len(self.pieces)
        for special in ('', COPY):
            if special not in self.pieces: self.pieces[special] = len(self.pieces)
        for g, _ in gc.most_common():
            if g not in self.codes: self.codes[g] = len(self.codes)
        self.ipieces = {v: k for k, v in self.pieces.items()}

    def ci(self, c): return self.chars.get(c, UNK)
    def pi(self, p): return self.pieces.get(p, UNK)
    def gi(self, g): return self.codes.get(g, UNK)


def enc_sentence(toks, v):
    ch, tk, iw, spans = [], [], [], []
    for ti, t in enumerate(toks):
        s = len(ch)
        ch.append(v.ci(WORD)); tk.append(ti); iw.append(0)
        for k, c in enumerate(t):
            ch.append(v.ci(c)); tk.append(ti); iw.append(min(k + 1, 31))
        spans.append((s, len(ch)))
    return ch, tk, iw, spans


def enc_labels(toks, obs, v):
    """One label per character slot of enc_sentence(toks).  A token with no aligned
    observed form falls back to "unchanged", so the label stream can never end up a
    different length from the character stream."""
    lab = []
    for i, t in enumerate(toks):
        o = obs[i] if i < len(obs) else t
        pcs = align_pieces(t, o)
        lab.append(v.pi(pcs[0]))
        for k, c in enumerate(t):
            lab.append(v.pi(lab_of(c, pcs[k + 1])))
    return lab


def style_vec(cal, v, skip=None):
    """Compact style summary of the row's OWN calibration evidence (a per-row input).

    A histogram of which edit pieces this profile used and how often each character was
    left untouched.  Support features consumed by the network - on their own they emit
    nothing.
    """
    npiece, nchar = len(v.pieces), len(v.chars)
    a = np.zeros(npiece, dtype=np.float32)
    chg = np.zeros(nchar, dtype=np.float32)
    cnt = np.zeros(nchar, dtype=np.float32)
    n = 0
    for i, c in enumerate(cal or []):
        if skip is not None and i == skip:
            continue
        for t, o in zip(c.get('query_tokens') or [], c.get('observed_tokens') or []):
            pcs = align_pieces(t, o)
            a[v.pi(pcs[0])] += 1; n += 1
            for k, ch in enumerate(t):
                lab = lab_of(ch, pcs[k + 1])
                a[v.pi(lab)] += 1; n += 1
                ci = v.ci(ch); cnt[ci] += 1
                if lab != COPY: chg[ci] += 1
    a = np.sqrt(a / max(n, 1))
    rate = chg / np.maximum(cnt, 1.0)
    seen = (cnt > 0).astype(np.float32)
    return np.concatenate([a, rate, seen, np.array([math.log1p(n) / 8.0], dtype=np.float32)])


# ------------------------------------------------------------------ model
class EncBlock(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.do = nn.Dropout(p)

    def forward(self, x, xpad):
        h = self.ln1(x)
        x = x + self.do(self.sa(h, h, h, key_padding_mask=xpad, need_weights=False)[0])
        x = x + self.do(self.ff(self.ln2(x)))
        return x


class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.do = nn.Dropout(p)

    def forward(self, x, mem, xpad, mpad):
        h = self.ln1(x)
        x = x + self.do(self.sa(h, h, h, key_padding_mask=xpad, need_weights=False)[0])
        h = self.ln2(x)
        x = x + self.do(self.ca(h, mem, mem, key_padding_mask=mpad, need_weights=False)[0])
        x = x + self.do(self.ff(self.ln3(x)))
        return x


class Transducer(nn.Module):
    def __init__(self, nchar, npiece, ncode, nstyle, d=320, nlay=5, nmem=2, nh=8, p=0.1):
        super().__init__()
        self.d = d
        self.ce = nn.Embedding(nchar, d, padding_idx=0)
        self.pe = nn.Embedding(npiece, d, padding_idx=0)
        self.ge = nn.Embedding(ncode, d, padding_idx=0)
        self.qpos = nn.Embedding(MAXQ, d)
        self.mpos = nn.Embedding(MAXM, d)
        self.iw = nn.Embedding(33, d)
        self.sty = nn.Sequential(nn.Linear(nstyle, d), nn.GELU(), nn.Linear(d, d))
        self.pool = nn.Linear(d, d)
        self.mem_blocks = nn.ModuleList([EncBlock(d, nh, p) for _ in range(nmem)])
        self.blocks = nn.ModuleList([Block(d, nh, p) for _ in range(nlay)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, npiece))
        self.do = nn.Dropout(p)

    def encode_mem(self, mc, ml, mi, mp, st):
        L = mc.shape[1]
        pos = torch.arange(L, device=mc.device).clamp(max=MAXM - 1)
        x = self.ce(mc) + self.pe(ml) + self.iw(mi) + self.mpos(pos)[None] + self.sty(st)[:, None, :]
        x = self.do(x)
        for b in self.mem_blocks:
            x = b(x, mp)
        return x

    def encode_q(self, qc, qi, qt, cd, qp, mem, mp, st):
        B, L = qc.shape
        pos = torch.arange(L, device=qc.device).clamp(max=MAXQ - 1)
        cm = (cd > 0).float().unsqueeze(-1)
        g = (self.ge(cd) * cm).sum(2) / cm.sum(2).clamp(min=1.0)
        g = torch.gather(g, 1, qt.clamp(min=0).unsqueeze(-1).expand(B, L, self.d))
        mk = (~mp).float().unsqueeze(-1)
        x = (self.ce(qc) + self.iw(qi) + self.qpos(pos)[None] + g
             + self.pool((mem * mk).sum(1) / mk.sum(1).clamp(min=1.0))[:, None, :]
             + self.sty(st)[:, None, :])
        x = self.do(x)
        for b in self.blocks:
            x = b(x, mem, qp, mp)
        return self.lnf(x)


# ------------------------------------------------------------------ batching
class Builder:
    def __init__(self, v):
        self.v = v

    def mem_of(self, cal, skip=None):
        toks, obs = [], []
        for i, c in enumerate(cal or []):
            if skip is not None and i == skip:
                continue
            qt = list(c.get('query_tokens') or [])
            ot = list(c.get('observed_tokens') or [])
            n = min(len(qt), len(ot))       # never trust the two arrays to line up
            toks.extend(qt[:n]); obs.extend(ot[:n])
        ch, tk, iw, _ = enc_sentence(toks, self.v)
        lab = enc_labels(toks, obs, self.v)
        if len(lab) != len(ch):             # belt and braces: the two streams must match
            lab = (lab + [PAD] * len(ch))[:len(ch)]
        return ch[:MAXM], lab[:MAXM], iw[:MAXM]

    def row(self, toks, codes, cal, obs=None, skip=None):
        ch, tk, iw, spans = enc_sentence(toks, self.v)
        cd = [[self.v.gi(g) for g in cs] for cs in codes]
        lab = enc_labels(toks, obs, self.v) if obs is not None else None
        mch, mlab, miw = self.mem_of(cal, skip)
        return dict(ch=ch[:MAXQ], iw=iw[:MAXQ], tk=tk[:MAXQ], cd=cd,
                    lab=(lab[:MAXQ] if lab is not None else None),
                    mch=mch, mlab=mlab, miw=miw, spans=spans, toks=toks,
                    sty=style_vec(cal, self.v, skip))


def _rnd(v, m):
    return int(((v + m - 1) // m) * m)


def collate(items, device):
    B = len(items)
    L = min(MAXQ, max(64, _rnd(max(len(x['ch']) for x in items), 64)))
    M = max(128, min(MAXM, _rnd(max(len(x['mch']) for x in items), 128)))
    T = max(8, _rnd(max(len(x['cd']) for x in items), 8))
    C = _rnd(max(1, max(max((len(c) for c in x['cd']), default=1) for x in items)), 4)
    qc = torch.zeros(B, L, dtype=torch.long); qi = torch.zeros(B, L, dtype=torch.long)
    qt = torch.zeros(B, L, dtype=torch.long); qp = torch.ones(B, L, dtype=torch.bool)
    mc = torch.zeros(B, M, dtype=torch.long); ml = torch.zeros(B, M, dtype=torch.long)
    mi = torch.zeros(B, M, dtype=torch.long); mp = torch.ones(B, M, dtype=torch.bool)
    cd = torch.zeros(B, T, C, dtype=torch.long)
    st = torch.zeros(B, len(items[0]['sty']), dtype=torch.float)
    lb = torch.zeros(B, L, dtype=torch.long)
    pv = torch.zeros(B, L, dtype=torch.long)
    for b, x in enumerate(items):
        n = min(len(x['ch']), len(x['iw']), len(x['tk']), L)
        if n:
            qc[b, :n] = torch.tensor(x['ch'][:n]); qi[b, :n] = torch.tensor(x['iw'][:n])
            qt[b, :n] = torch.tensor(x['tk'][:n]); qp[b, :n] = False
        m = min(len(x['mch']), len(x['mlab']), len(x['miw']), M)
        if m:
            mc[b, :m] = torch.tensor(x['mch'][:m]); ml[b, :m] = torch.tensor(x['mlab'][:m])
            mi[b, :m] = torch.tensor(x['miw'][:m]); mp[b, :m] = False
        else:
            mp[b, 0] = False
        for t, cs in enumerate(x['cd'][:T]):
            if cs: cd[b, t, :min(len(cs), C)] = torch.tensor(cs[:C])
        st[b] = torch.from_numpy(x['sty'])
        if x['lab'] is not None and n:
            nl = min(n, len(x['lab']))
            lb[b, :nl] = torch.tensor(x['lab'][:nl])
            p = [BOS] * nl
            for k in range(1, nl):
                if x['iw'][k] != 0: p[k] = x['lab'][k - 1]
            pv[b, :nl] = torch.tensor(p)
    d = lambda t: t.to(device, non_blocking=True)
    return dict(qc=d(qc), qi=d(qi), qt=d(qt), qp=d(qp), mc=d(mc), ml=d(ml), mi=d(mi),
                mp=d(mp), cd=d(cd), lb=d(lb), prev=d(pv), st=d(st))


# ------------------------------------------------------------------ ensemble beam search
@torch.no_grad()
def predict(models, items, device, vocab, beam=8, topk=4, bs=24, amp=False):
    """Beam search over the trained transducer(s).  Returns, per row and token, the list
    of (source-spelling string, model probability) plus the probability of the unchanged
    (identity) spelling, computed exactly by forced decoding."""
    for m in models:
        m.eval()
    ip = vocab.ipieces; V = len(vocab.pieces)
    copy_id = vocab.pieces[COPY]; empty_id = vocab.pieces['']
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]['ch']))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = collate([items[i] for i in idx], device)
        Hs = []
        with (torch.autocast('cuda', dtype=torch.bfloat16) if amp
              else torch.autocast('cpu', enabled=False)):
            for m in models:
                mem = m.encode_mem(batch['mc'], batch['ml'], batch['mi'], batch['mp'], batch['st'])
                Hs.append(m.encode_q(batch['qc'], batch['qi'], batch['qt'], batch['cd'],
                                     batch['qp'], mem, batch['mp'], batch['st']).float())
        L = Hs[0].shape[1]
        toks = []
        for bi, ii in enumerate(idx):
            for ti, (s0, s1) in enumerate(items[ii]['spans']):
                if s0 >= L: continue
                toks.append((bi, ii, ti, s0, min(s1, L) - s0))
        for ii in idx:
            if out[ii] is None:
                out[ii] = [None] * len(items[ii]['spans'])
        if not toks:
            continue
        NT = len(toks); Lm = max(t[4] for t in toks); d = Hs[0].shape[-1]
        HH = []
        for h in Hs:
            H = torch.zeros(NT, Lm, d, device=device)
            for k, (bi, ii, ti, s0, ln) in enumerate(toks):
                H[k, :ln] = h[bi, s0:s0 + ln]
            HH.append(H)
        M = torch.zeros(NT, Lm, dtype=torch.bool, device=device)
        for k, (bi, ii, ti, s0, ln) in enumerate(toks):
            M[k, :ln] = True
        K = beam
        logp = torch.full((NT, K), -1e9, device=device); logp[:, 0] = 0.0
        prev = torch.full((NT, K), BOS, dtype=torch.long, device=device)
        BP, PID = [], []
        for t in range(Lm):
            lsm = None
            for m, H in zip(models, HH):
                lg = m.head(torch.cat([H[:, t][:, None, :].expand(NT, K, d), m.pe(prev)], -1))
                q = F.log_softmax(lg.float(), -1)
                lsm = q if lsm is None else lsm + q
            lsm = torch.nan_to_num(lsm / len(models), nan=-1e9, neginf=-1e9)
            lsm[:, :, :BOS + 1] = -1e9
            if t == 0:
                lsm[:, :, copy_id] = -1e9
            fin = ~M[:, t]
            if bool(fin.any()):
                lsm[fin] = -1e9
                lsm[fin, :, PAD] = 0.0
            tot = (logp[:, :, None] + lsm).view(NT, -1)
            v, ix = tot.topk(min(K, tot.shape[-1]), -1)
            BP.append(torch.div(ix, V, rounding_mode='floor')); PID.append(ix % V)
            logp = v; prev = PID[-1]
        # exact log-probability of the identity (unchanged) spelling
        lab_id = torch.zeros(NT, Lm, dtype=torch.long, device=device)
        lab_id[:, 0] = empty_id
        if Lm > 1: lab_id[:, 1:] = copy_id
        pvv = torch.full((NT, Lm), BOS, dtype=torch.long, device=device)
        if Lm > 1: pvv[:, 1:] = lab_id[:, :-1]
        lsi = None
        for m, H in zip(models, HH):
            q = F.log_softmax(m.head(torch.cat([H, m.pe(pvv)], -1)).float(), -1)
            lsi = q if lsi is None else lsi + q
        lsi = lsi / len(models)
        idl = (lsi.gather(-1, lab_id[:, :, None]).squeeze(-1) * M).sum(1)

        BPn = torch.stack(BP, 0).cpu().numpy(); PIDn = torch.stack(PID, 0).cpu().numpy()
        lpn = logp.cpu().numpy(); idln = idl.cpu().numpy()
        for k, (bi, ii, ti, s0, ln) in enumerate(toks):
            tok = items[ii]['toks'][ti]
            agg = {}
            for j in range(lpn.shape[1]):
                if not np.isfinite(lpn[k, j]) or lpn[k, j] < -1e8:
                    continue
                seq = []; cur = j
                for t in range(Lm - 1, -1, -1):
                    seq.append(int(PIDn[t, k, cur])); cur = int(BPn[t, k, cur])
                seq = seq[::-1][:ln]
                parts = []
                for q, pid in enumerate(seq):
                    parts.append(ip.get(pid, '') if q == 0 else
                                 (tok[q - 1] if pid == copy_id else ip.get(pid, '')))
                stg = ''.join(parts)
                agg[stg] = float(np.logaddexp(agg[stg], lpn[k, j])) if stg in agg else float(lpn[k, j])
            pq = float(math.exp(min(0.0, float(idln[k]))))
            if tok not in agg:
                agg[tok] = float(idln[k])
            cands = sorted(agg.items(), key=lambda z: -z[1])[:topk]
            out[ii][ti] = ([(t, math.exp(min(0.0, lp))) for t, lp in cands], pq)
    for ii in range(len(items)):
        if out[ii] is None:
            out[ii] = []
        while len(out[ii]) < len(items[ii]['spans']):
            out[ii].append(None)
        for ti in range(len(out[ii])):
            if out[ii][ti] is None:
                out[ii][ti] = ([(items[ii]['toks'][ti], 1.0)], 1.0)
    return out


# ------------------------------------------------------------------ exact PROBLEM.md metric
def _lev(a, b):
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb; ca = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
        prev = cur
    return prev[lb]


def char_sim(v, t):
    return max(0.0, 1.0 - _lev(v, t) / max(len(v), len(t), 1))


def row_score(cands, query, truth, novel):
    cp, c1, cc, npb, n1, up = [], [], [], [], [], []
    cov = 1
    for i in range(len(truth)):
        cl = cands[i]
        S = sum(p * p for _, p in cl)
        pt = sum(p for v, p in cl if v == truth[i])
        pc = max(0.0, 2 * pt - S)
        ec = sum(p * char_sim(v, truth[i]) for v, p in cl)
        t1 = 1.0 if cl[0][0] == truth[i] else 0.0
        if query[i] != truth[i]:
            cp.append(pc); c1.append(t1); cc.append(ec)
            if truth[i] not in [v for v, _ in cl]:
                cov = 0
            if novel[i]:
                npb.append(pc); n1.append(t1)
        else:
            up.append(pc)
    mn = lambda x: float(np.mean(x)) if x else 0.0
    return (0.20 * mn(cp) + 0.10 * mn(c1) + 0.10 * mn(cc) + 0.25 * mn(npb)
            + 0.10 * mn(n1) + 0.15 * cov + 0.10 * mn(up))


# ------------------------------------------------------------------ lattice decoder
def sanitize(t):
    t = ''.join(ch for ch in str(t) if ord(ch) >= 32 and not (0xD800 <= ord(ch) <= 0xDFFF))
    return t.strip()[:160]


def make_lattice(pred_row, toks, budget, pmin=0.01, temp=1.0, min_gain=0.0,
                 max_c=3, mode='sharp'):
    """Choose, per position, how many source-spelling candidates to submit and with what
    probabilities, maximising the expected PROBLEM.md row score under the model's own
    predictive distribution.

    probability_credit = max(0, 2*p_true - sum_j p_j^2) is maximised by concentrating
    mass on the model's argmax, so an extra candidate can only pay for itself through the
    full_changed_coverage term.  The allocation of the row's expansion budget is therefore
    an exact knapsack that maximises the log-probability that *every* changed position is
    covered.  pmin / temp / min_gain / max_c / mode are searched in-script on a
    profile-disjoint train holdout.
    """
    n = len(toks)
    P = []
    for i in range(n):
        cands, pq = pred_row[i] if i < len(pred_row) else ([(toks[i], 1.0)], 1.0)
        c = [(t, max(float(p), 1e-12) ** (1.0 / temp)) for t, p in cands]
        z = sum(p for _, p in c)
        if z <= 0:
            c = [(toks[i], 1.0)]; z = 1.0
        c = [(t, p / z) for t, p in c]
        d = dict(c)
        pqv = d.get(toks[i], min(max(float(pq), 1e-12) ** (1.0 / temp) / z, 1.0))
        P.append((c, pqv))

    def cov(i, k):
        c, pq = P[i]
        s = pq
        for t, p in c[:k]:
            if t != toks[i]: s += p
        return min(1.0, max(s, 1e-9))

    gains = []
    for i in range(n):
        c0 = math.log(cov(i, 1)); g = []
        for k in (2, 3):
            if k <= len(P[i][0]) and k <= max_c and P[i][0][k - 1][1] > 1e-9:
                g.append(math.log(cov(i, k)) - c0)
            else:
                g.append(g[-1] if g else 0.0)
        gains.append(g)

    B = int(budget); NEG = -1e18
    dp = [[NEG] * (B + 1) for _ in range(n + 1)]
    par = [[0] * (B + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n):
        for b in range(B + 1):
            if dp[i][b] == NEG: continue
            for e in (0, 1, 2):
                if b + e > B: continue
                gv = 0.0 if e == 0 else gains[i][e - 1]
                if e > 0 and gv < min_gain: continue
                if dp[i][b] + gv > dp[i + 1][b + e]:
                    dp[i + 1][b + e] = dp[i][b] + gv
                    par[i + 1][b + e] = e
    best_b = max(range(B + 1), key=lambda b: dp[n][b])
    ks = [1] * n; b = best_b
    for i in range(n, 0, -1):
        e = par[i][b]; ks[i - 1] = 1 + e; b -= e

    lat = []
    for i in range(n):
        seen = set(); texts = []; raw = []
        for t, p in P[i][0]:
            t2 = sanitize(t)
            if not t2 or t2 in seen: continue
            seen.add(t2); texts.append(t2); raw.append(p)
            if len(texts) >= ks[i]: break
        if not texts:
            texts = [sanitize(toks[i]) or 'x']; raw = [1.0]
        k = len(texts)
        if k == 1:
            pr = [1.0]
        elif mode == 'renorm':
            z = sum(raw) or 1.0
            pr = sorted([max(pmin, r / z) for r in raw], reverse=True)
            z2 = sum(pr)
            pr = [round(max(pmin, p / z2), 6) for p in pr]
            pr[0] = round(1.0 - sum(pr[1:]), 6)
            if not (all(pr[j] >= pr[j + 1] - 1e-12 for j in range(k - 1))
                    and 0.01 <= pr[0] <= 1.0):
                pr = [round(1.0 - pmin * (k - 1), 6)] + [pmin] * (k - 1)
        else:
            pr = [round(1.0 - pmin * (k - 1), 6)] + [pmin] * (k - 1)
        lat.append(list(zip(texts, pr)))
    return lat


def validate_lattice(lat, toks, budget):
    """Final safety net: enforce every submission constraint from PROBLEM.md.
    Returns a list (one entry per query token) of [(text, prob), ...] tuples."""
    if not isinstance(lat, list) or len(lat) != len(toks):
        lat = [[(t, 1.0)] for t in toks]
    used = 0
    out = []
    for i, cl in enumerate(lat):
        seen = set(); cc = []; pp = []
        for t, p in cl:
            t = sanitize(t)
            if not t or t in seen: continue
            seen.add(t); cc.append(t); pp.append(float(p))
        if not cc:
            cc = [sanitize(toks[i]) or 'x']; pp = [1.0]
        extra = max(0, min(len(cc) - 1, int(budget) - used, 2))
        cc = cc[:1 + extra]; pp = pp[:1 + extra]; used += extra
        k = len(cc)
        if k == 1:
            pr = [1.0]
        else:
            pr = [round(min(1.0, max(0.01, p)), 6) for p in pp]
            pr[0] = round(1.0 - sum(pr[1:]), 6)
            ok = (all(0.01 - 1e-12 <= p <= 1.0 for p in pr)
                  and all(pr[j] >= pr[j + 1] - 1e-12 for j in range(k - 1))
                  and abs(sum(pr) - 1.0) < 1e-9)
            if not ok:
                pr = [round(1.0 - 0.01 * (k - 1), 6)] + [0.01] * (k - 1)
        out.append(list(zip(cc, [float(p) for p in pr])))
    return out


def to_json_lattice(lat):
    return [[{"text": t, "prob": p} for t, p in cl] for cl in lat]


def final_lattice(pred_row, toks, budget, **knobs):
    """The exact object that gets submitted - also what the knob search scores, so the
    holdout number and the submission can never drift apart."""
    return validate_lattice(make_lattice(pred_row, toks, budget, **knobs), toks, budget)


# ------------------------------------------------------------------ main
def main():
    if len(sys.argv) < 3:
        print('usage: solution.py <public_dir> <submission_out>'); sys.exit(1)
    pub, out_path = sys.argv[1], sys.argv[2]
    outdir = os.path.dirname(os.path.abspath(out_path))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    set_seed(SEED)
    J = json.loads
    tr = pd.read_csv(os.path.join(pub, 'train.csv'))
    te = pd.read_csv(os.path.join(pub, 'test.csv'))
    log('train', tr.shape, 'test', te.shape)

    for df in (tr, te):
        df['q'] = df['query_tokens'].map(J)
        df['cal'] = df['calibration_examples'].map(J)
        df['ac'] = df['analysis_codes'].map(J)
    tr['o'] = tr['observed_tokens'].map(J)

    te_bud = []
    for v in te['expansion_budget'].tolist():
        try:
            te_bud.append(max(0, int(v)))
        except Exception:
            te_bud.append(0)

    cids = te['case_id'].tolist()

    def write_sub(lattices):
        """Single serialisation choke point.  Coerces whatever it is handed into the
        exact {"text":..,"prob":..} object form and refuses to emit a short file."""
        lattices = list(lattices)
        if len(lattices) != len(cids):
            raise ValueError('lattice count %d != test rows %d' % (len(lattices), len(cids)))
        rows = []
        for cid, lat in zip(cids, lattices):
            js = []
            for cl in lat:
                pos = []
                for c in cl:
                    if isinstance(c, dict):
                        pos.append({"text": str(c["text"]), "prob": float(c["prob"])})
                    else:
                        pos.append({"text": str(c[0]), "prob": float(c[1])})
                js.append(pos)
            rows.append((cid, json.dumps(js, ensure_ascii=False, separators=(',', ':'))))
        pd.DataFrame(rows, columns=['case_id', 'variant_lattice']).to_csv(out_path, index=False)

    base = [to_json_lattice(validate_lattice([[(t, 1.0)] for t in q], q, b))
            for q, b in zip(te['q'], te_bud)]
    write_sub(base)
    log('placeholder submission written ->', out_path)

    try:
        run(tr, te, te_bud, write_sub, base)
    except Exception:
        import traceback
        traceback.print_exc()
        log('FALLBACK: keeping the last valid submission on disk')


def run(tr, te, te_bud, write_sub, base):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log('device', device)

    # ---- profile-disjoint holdout: decoder-knob search + CV reporting ----
    rng = np.random.RandomState(SEED)
    profs = sorted(tr['profile_id'].unique())
    rng.shuffle(profs)
    nho = max(10, int(len(profs) * HOLDOUT_FRAC))
    hp = set(profs[:nho])
    va = tr[tr['profile_id'].isin(hp)].reset_index(drop=True)
    trn = tr[~tr['profile_id'].isin(hp)].reset_index(drop=True)
    if len(trn) == 0:
        trn, va = tr.reset_index(drop=True), tr.reset_index(drop=True)
    log('fit rows %d | holdout rows %d over %d profiles' % (len(trn), len(va), len(hp)))

    # ---- vocabularies: fitted on the training split only ----
    cc, pc, gc = collections.Counter(), collections.Counter(), collections.Counter()
    cc[WORD] = 10 ** 9

    def scan(qs, os_):
        for q, o in zip(qs, os_):
            for t in q: cc.update(t)
            for t in o: cc.update(t)
            for t, ob in zip(q, o):
                pcs = align_pieces(t, ob)
                pc[pcs[0]] += 1
                for k, ch in enumerate(t):
                    pc[lab_of(ch, pcs[k + 1])] += 1

    scan(trn['q'], trn['o'])
    for ac in trn['ac']:
        for cs in ac:
            gc.update(cs)
    for cal in trn['cal']:
        scan([c['query_tokens'] for c in cal], [c['observed_tokens'] for c in cal])
    V = Vocab(); V.build(cc, pc, gc, min_piece=2)
    log('vocab: chars %d, pieces %d, codes %d'
        % (len(V.chars), len(V.pieces), len(V.codes)))

    B = Builder(V)
    t0 = time.time()
    fit_items = [B.row(r['q'], r['ac'], r['cal'], r['o']) for _, r in trn.iterrows()]
    seenp = set()
    for _, r in trn.iterrows():
        if r['profile_id'] in seenp: continue
        seenp.add(r['profile_id'])
        for k, c in enumerate(r['cal']):
            if len(c['query_tokens']) == 0: continue
            fit_items.append(B.row(c['query_tokens'], [[] for _ in c['query_tokens']],
                                   r['cal'], c['observed_tokens'], skip=k))
    def safe_row(r):
        """A malformed row must degrade to "no calibration evidence", never crash."""
        try:
            return B.row(r['q'], r['ac'], r['cal'])
        except Exception:
            try:
                return B.row(list(r['q']), [[] for _ in r['q']], [])
            except Exception:
                return B.row(['x'], [[]], [])

    va_items = [safe_row(r) for _, r in va.iterrows()]
    te_items = [safe_row(r) for _, r in te.iterrows()]
    log('%d training items (rows + leave-one-out calibration sentences), %.1fs'
        % (len(fit_items), time.time() - t0))

    nstyle = len(fit_items[0]['sty'])
    BS = 16 if device.type == 'cuda' else 8
    MAX_EPOCHS, MIN_EPOCHS = 24, 5
    LR = 4e-4
    nbatch = max(1, (len(fit_items) + BS - 1) // BS)

    def new_model():
        return Transducer(len(V.chars), len(V.pieces), len(V.codes), nstyle,
                          d=320, nlay=5, nmem=2, p=0.1).to(device)

    # How many models and how many epochs each: decided from the measured cost of the
    # first epoch on whatever machine this actually runs on.
    sched = {'n_models': MAX_MODELS, 'epochs': None}

    def plan_epochs(per_epoch):
        left_now = TRAIN_DEADLINE - (time.time() - T_START)
        if sched['epochs'] is not None:
            return sched['epochs']
        n = int(left_now // max(per_epoch * 12.0, 1e-6))
        sched['n_models'] = max(1, min(MAX_MODELS, n))
        share = left_now / sched['n_models']
        sched['epochs'] = int(max(MIN_EPOCHS, min(MAX_EPOCHS, share // max(per_epoch, 1e-6))))
        return sched['epochs']

    def train_one(seed, n_epochs):
        set_seed(seed)
        m = new_model()
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
        sd = m.state_dict()
        ekeys = [k for k, v in sd.items() if v.is_floating_point()]
        eref = [sd[k] for k in ekeys]
        ema = [v.detach().clone().float() for v in eref]
        plan = {'total': max(2, nbatch * n_epochs)}
        step = 0
        t_start = time.time()
        order = list(range(len(fit_items)))
        ep = 0
        while ep < n_epochs:
            if time.time() - T_START > EPOCH_DEADLINE:
                log('  hard epoch deadline reached at epoch %d' % ep); break
            m.train(); random.shuffle(order)
            order.sort(key=lambda i: len(fit_items[i]['ch']) // 48)
            batches = [order[i:i + BS] for i in range(0, len(order), BS)]
            random.shuffle(batches)
            tl = 0.0; nb = 0
            for bidx in batches:
                frac = min(1.0, step / plan['total'])
                if frac < 0.10:
                    f = frac / 0.10
                else:
                    f = 0.5 * (1.0 + math.cos(math.pi * (frac - 0.10) / 0.90))
                for gp in opt.param_groups:
                    gp['lr'] = LR * (0.01 + 0.99 * f)
                bt = collate([fit_items[i] for i in bidx], device)
                with (torch.autocast('cuda', dtype=torch.bfloat16) if amp
                      else torch.autocast('cpu', enabled=False)):
                    mem = m.encode_mem(bt['mc'], bt['ml'], bt['mi'], bt['mp'], bt['st'])
                    h = m.encode_q(bt['qc'], bt['qi'], bt['qt'], bt['cd'], bt['qp'],
                                   mem, bt['mp'], bt['st'])
                    prev = bt['prev']
                    msk = torch.rand_like(prev, dtype=torch.float) < 0.15
                    prev = torch.where(msk, torch.full_like(prev, UNK), prev)
                    lg = m.head(torch.cat([h, m.pe(prev)], -1))
                lb = bt['lb']
                sel = (~bt['qp']) & (lb != PAD) & (lb != UNK)
                if not bool(sel.any()):
                    continue
                loss = F.cross_entropy(lg.float()[sel], lb[sel], label_smoothing=0.02)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                step += 1
                with torch.no_grad():
                    dcy = 0.999 if frac > 0.3 else 0.95
                    torch._foreach_mul_(ema, dcy)
                    torch._foreach_add_(ema, [v.detach().float() for v in eref],
                                        alpha=1.0 - dcy)
                tl += float(loss.detach()); nb += 1
            ep += 1
            if ep == 1:
                per = time.time() - t_start
                fit_ep = plan_epochs(per)
                if fit_ep != n_epochs:
                    log('  re-planned: %d -> %d epochs, %d model(s) (%.0fs/epoch)'
                        % (n_epochs, fit_ep, sched['n_models'], per))
                    n_epochs = fit_ep
                    plan['total'] = max(2, nbatch * n_epochs)
            if ep % 5 == 1 or ep == n_epochs:
                log('  seed %d epoch %d/%d loss %.4f' % (seed, ep, n_epochs, tl / max(nb, 1)))
        me = new_model()
        esd = me.state_dict()
        for k_, v_ in zip(ekeys, ema):
            esd[k_].copy_(v_.to(dtype=esd[k_].dtype))
        me.load_state_dict(esd)
        return m, me

    models, models_ema = [], []
    for k in range(MAX_MODELS):
        if k >= sched['n_models']:
            log('training-time budget: stopping at %d model(s)' % len(models))
            break
        if time.time() - T_START > TRAIN_DEADLINE:
            log('train deadline reached: stopping at %d model(s)' % len(models))
            break
        s0 = time.time()
        mm, me = train_one(SEED + 17 * k, sched['epochs'] or MAX_EPOCHS)
        models.append(mm); models_ema.append(me)
        log('model %d/%d trained (%.0fs)' % (k + 1, sched['n_models'], time.time() - s0))
    if not models:
        return

    # ---- (query, observed) pairs already seen publicly, for the novel-variant flag.
    # Built from the TRAIN split only: nothing is ever counted over test rows, not even
    # for validation bookkeeping.  This makes the local novel flag slightly stricter than
    # the grader's, which is the conservative direction. ----
    pub_pairs = set()
    for q, o in zip(trn['q'], trn['o']):
        for a, b in zip(q, o): pub_pairs.add((a, b))
    for cal in trn['cal']:
        for c in cal:
            for a, b in zip(c['query_tokens'], c['observed_tokens']):
                pub_pairs.add((a, b))

    IBS = 24 if device.type == 'cuda' else 8
    best = dict(pmin=0.01, temp=1.0, min_gain=0.0, max_c=3, mode='sharp')
    use = models
    try:
        va_bud = [max(0, int(b)) for b in va['expansion_budget']]
        novel = []
        for _, r in va.iterrows():
            own = set()
            for c in r['cal']:
                for a, b in zip(c['query_tokens'], c['observed_tokens']):
                    own.add((a, b))
            novel.append([(a != b) and ((a, b) not in pub_pairs) and ((a, b) not in own)
                          for a, b in zip(r['q'], r['o'])])
        prof = list(va['profile_id'])
        vq = list(va['q']); vo = list(va['o'])

        def score_preds(pp, **kw):
            p = dict(best); p.update(kw)
            per = collections.defaultdict(list)
            for i in range(len(vq)):
                lat = final_lattice(pp[i], vq[i], va_bud[i], **p)
                per[prof[i]].append(row_score(lat, vq[i], vo[i], novel[i]))
            return 100.0 * float(np.mean([np.mean(v) for v in per.values()]))

        # reference point: the trained model must beat "copy the query token everywhere"
        ident = [[([(t, 1.0)], 1.0) for t in q] for q in vq]
        s_id = score_preds(ident)

        # raw weights vs exponential-moving-average weights: pick on the holdout
        vp = predict(models, va_items, device, V, beam=8, topk=4, bs=IBS, amp=amp)
        s_raw = score_preds(vp)
        vpe = predict(models_ema, va_items, device, V, beam=8, topk=4, bs=IBS, amp=amp)
        s_ema = score_preds(vpe)
        log('holdout: identity %.3f | raw weights %.3f | EMA weights %.3f'
            % (s_id, s_raw, s_ema))
        if s_ema > s_raw:
            use, vp = models_ema, vpe
        if max(s_raw, s_ema) <= s_id:
            log('SANITY GATE: trained models do not beat the identity baseline on the '
                'holdout - keeping the identity submission')
            return

        def sc(**kw):
            return score_preds(vp, **kw)

        cur = sc()
        log('holdout score with default decode knobs: %.3f' % cur)
        grid = [('mode', ['sharp', 'renorm']),
                ('pmin', [0.01, 0.02, 0.05, 0.10, 0.20]),
                ('temp', [0.6, 0.8, 1.0, 1.25, 1.6]),
                ('min_gain', [0.0, 0.002, 0.01, 0.05, 0.15]),
                ('max_c', [2, 3])]
        for _ in range(2):
            for name, vals in grid:
                if time.time() - T_START > HARD_DEADLINE:
                    break
                bv, bsc = best[name], cur
                for v in vals:
                    if v == best[name]: continue
                    s = sc(**{name: v})
                    if s > bsc + 1e-9:
                        bsc, bv = s, v
                if bv != best[name]:
                    log('  knob %s: %r -> %r  (%.3f -> %.3f)' % (name, best[name], bv, cur, bsc))
                    best[name] = bv; cur = bsc
        log('HOLDOUT CV SCORE %.3f | knobs %s' % (cur, best))

        c1, n1, cvf, up = [], [], [], []
        for i in range(len(vq)):
            lat = final_lattice(vp[i], vq[i], va_bud[i], **best)
            ok = 1
            for j, (a, b) in enumerate(zip(vq[i], vo[i])):
                if a != b:
                    c1.append(lat[j][0][0] == b)
                    if b not in [t for t, _ in lat[j]]: ok = 0
                    if novel[i][j]: n1.append(lat[j][0][0] == b)
                else:
                    up.append(lat[j][0][0] == b)
            cvf.append(ok)
        f = lambda x: float(np.mean(x)) if len(x) else 0.0
        log('holdout: changed_top1 %.3f | novel_top1 %.3f | unchanged_top1 %.3f | full_coverage %.3f'
            % (f(c1), f(n1), f(up), f(cvf)))
    except Exception:
        import traceback; traceback.print_exc()
        log('knob search failed - falling back to default decode knobs')

    # ---- test inference (strictly per-row; no statistic is ever taken across test rows) ----
    def ident_preds():
        return [[([(t, 1.0)], 1.0) for t in q] for q in te['q']]

    try:
        tp = predict(use, te_items, device, V, beam=8, topk=4, bs=IBS, amp=amp)
    except Exception:
        import traceback; traceback.print_exc()
        log('batched inference failed - retrying one row at a time')
        tp = []
        for i, it in enumerate(te_items):
            try:
                tp.append(predict(use, [it], device, V, beam=8, topk=4, bs=1, amp=amp)[0])
            except Exception:
                tp.append([([(t, 1.0)], 1.0) for t in te['q'][i]])
    lat = []
    for i in range(len(te)):
        try:
            l = to_json_lattice(final_lattice(tp[i], te['q'][i], te_bud[i], **best))
        except Exception:
            l = base[i]
        lat.append(l)
    write_sub(lat)
    log('final submission written: %d rows' % len(lat))


if __name__ == '__main__':
    main()
