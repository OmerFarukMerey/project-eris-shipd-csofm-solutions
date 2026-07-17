"""Lean Proof Patch Recovery — solution.

Approach (everything fit on train only; test rows used for inference only):

1. Start-line model: gradient-boosted classifier over the non-empty lines of
   each broken proof. Features: position relative to line_hint / end of proof /
   `:= by` line, indentation, first token, alias overlap with compiler
   feedback and state hint, neighboring-line context, feedback category.
   delete_line_count is always predicted as 1 (99.4% of train rows).

2. Insert-line candidates, generated per (row, candidate start line):
   - copy of the line being replaced (the true fix is often a small edit of it)
   - generic tactics at matching indentation ("simp", "rfl", "simp [new]")
   - k-NN retrieval: char-TFIDF over alias-normalized (deleted line +
     compiler feedback + state hint); neighbors are used raw, x-remapped, and
     fully x/T-remapped with stable handling of repeated aliases
   - most frequent "slotted templates" mined from train (aliases encoded as
     deleted-line-position / context / new-sequential slots), instantiated
     with the target row's aliases; both global and per-feedback-category.

3. A train-fitted logistic model predicts the repaired line's tactic family
   from the retrieval document. Out-of-fold probabilities are used while
   training the final ranker, preventing target leakage into ranker features.

4. Joint ranker blend: a CatBoost regressor plus a lightweight histogram
   gradient-boosted regressor predict each (start line, insert candidate)
   pair's expected row score
   (0.18*loc + 0.58*text + 0.14*line-lcs + 0.10*exact). At inference the
   argmax pair over the top-2 predicted start lines is submitted.

Key data insight: anonymized aliases (x###/T###) are numbered by first
appearance, so identifiers that the true fix introduces are always the next
sequential ids after the row's max context alias; insert indentation always
equals the replaced line's indentation.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

SEED = 42
K_NN = 12
N_TPL = 50
N_TPL_CAT = 8
N_START = 2
N_FB_CATS = 40
N_RANK_ITER = 800
RANK_HGB_BLEND = 0.15

PAT_X = re.compile(r'\bx(\d{3})\b')
PAT_T = re.compile(r'\bT(\d{3})\b')

FIRST_TOKENS = ['exact', 'rw', 'simp', 'rfl', 'have', 'ext', '·', 'intro', 'apply',
                'simpa', 'by_cases', 'constructor', 'cases', 'obtain', 'refine',
                'unfold', 'calc', 'norm_num', 'ring', 'field_simp', 'linarith',
                'omega', 'decide', 'aesop', 'induction', 'use', 'show', 'rcases',
                'nlinarith', 'positivity', 'gcongr', 'convert', 'push_neg']
TOK2ID = {t: i + 1 for i, t in enumerate(FIRST_TOKENS)}

try:
    from rapidfuzz.distance import Levenshtein as _RfLev

    def lev(a, b):
        return _RfLev.distance(a, b)
except ImportError:
    def lev(a, b):
        """Levenshtein distance, vectorized row DP."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return max(la, lb)
        if la < lb:
            a, b, la, lb = b, a, lb, la
        bb = np.frombuffer(b.encode('utf-32-le'), dtype=np.uint32)
        ar = np.arange(lb + 1)
        prev = ar.copy()
        cur = np.empty(lb + 1, dtype=np.int64)
        for i, ca in enumerate(a, 1):
            cur[0] = i
            np.minimum(prev[:-1] + (bb != ord(ca)), prev[1:] + 1, out=cur[1:])
            np.minimum.accumulate(cur - ar, out=cur)
            cur += ar
            prev, cur = cur, prev
        return int(prev[-1])


def sim(a, b):
    return 1.0 - lev(a, b) / max(len(a), len(b), 1)


def first_token(s):
    parts = s.strip().split()
    return parts[0] if parts else ''



def tok_id(s):
    t = first_token(s)
    if t in TOK2ID:
        return TOK2ID[t]
    if PAT_X.fullmatch(t) or PAT_T.fullmatch(t):
        return len(FIRST_TOKENS) + 1
    return 0


def indent_of(s):
    return len(s) - len(s.lstrip())


def norm_alias(s):
    return PAT_T.sub('T', PAT_X.sub('X', s))


def fb_shape(fb):
    return norm_alias(str(fb).split('\n')[0])[:60]


def row_context(r):
    lines = r.broken_proof.split('\n')
    fb = str(r.compiler_feedback)
    sh = str(r.state_hint)
    ctx_text = r.broken_proof + '\n' + fb + '\n' + sh
    ctx_x = set(int(m) for m in PAT_X.findall(ctx_text))
    fb_x = set(int(m) for m in PAT_X.findall(fb))
    sh_x = set(int(m) for m in PAT_X.findall(sh))
    ctx_T = set(int(m) for m in PAT_T.findall(ctx_text))
    last_ne = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_ne = i + 1
            break
    by_line = 0
    for i, l in enumerate(lines):
        ls = l.rstrip()
        if ls.endswith(':= by') or ls.endswith(' by') or ls.endswith(':=by'):
            by_line = i + 1
            break
    return {
        'lines': lines, 'fb': fb, 'sh': sh,
        'ctx_x': ctx_x, 'ctx_T': ctx_T, 'fb_x': fb_x, 'sh_x': sh_x,
        'max_x': max(ctx_x) if ctx_x else -1,
        'max_T': max(ctx_T) if ctx_T else -1,
        'last_ne': last_ne, 'nlines': len(lines), 'by_line': by_line,
        'line_hint': int(r.line_hint),
    }


# ---------------- start-line model ----------------

def line_features(ctx, i, fbc):
    lines = ctx['lines']
    line = lines[i - 1]
    ls = line.strip()
    lx = set(int(m) for m in PAT_X.findall(line))
    prev = lines[i - 2] if i >= 2 else ''
    nxt = lines[i] if i < len(lines) else ''
    return [
        i - ctx['line_hint'],
        ctx['last_ne'] - i,
        ctx['nlines'] - i,
        i / max(1, ctx['nlines']),
        1.0 if i == ctx['last_ne'] else 0.0,
        1.0 if i > ctx['by_line'] and ctx['by_line'] > 0 else 0.0,
        i - ctx['by_line'],
        indent_of(line) if ls else -1,
        len(ls),
        tok_id(line),
        len(lx & ctx['fb_x']) / max(1, len(lx)),
        len(lx & ctx['sh_x']) / max(1, len(lx)),
        len(lx),
        1.0 if ls.endswith('by') else 0.0,
        sum(1 for j in range(i, len(lines)) if lines[j].strip()),
        fbc,
        tok_id(prev), tok_id(nxt),
        indent_of(prev) if prev.strip() else -1,
        indent_of(nxt) if nxt.strip() else -1,
        1.0 if '·' in line else 0.0,
        len(lx & ctx['fb_x']), len(lx & ctx['sh_x']),
        1.0 if ':=' in line else 0.0,
        1.0 if prev.strip().endswith('by') else 0.0,
        len(ctx['fb']), ctx['fb'].count('case '),
        sum(1 for j in range(i, len(lines))
            if lines[j].strip() and indent_of(lines[j]) == indent_of(line)),
    ]


def start_candidates(ctx):
    return [i for i in range(1, ctx['nlines'] + 1) if ctx['lines'][i - 1].strip()]


def build_start_matrix(ctxs, fb_cats):
    X, grp, lineno = [], [], []
    for gi, c in enumerate(ctxs):
        fbc = fb_cats.get(fb_shape(c['fb']), 0)
        for i in start_candidates(c):
            X.append(line_features(c, i, fbc))
            grp.append(gi)
            lineno.append(i)
    return np.array(X, dtype=np.float32), np.array(grp), np.array(lineno)


def fit_start_model(ctxs, starts, fb_cats):
    X, grp, lineno = build_start_matrix(ctxs, fb_cats)
    y = (lineno == np.array([starts[g] for g in grp])).astype(int)
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06,
                                       max_leaf_nodes=63, min_samples_leaf=20,
                                       random_state=SEED)
    m.fit(X, y)
    return m


def predict_start_topk(model, ctxs, fb_cats, topk):
    X, grp, lineno = build_start_matrix(ctxs, fb_cats)
    p = model.predict_proba(X)[:, 1]
    n = max(grp) + 1
    per_group = [[] for _ in range(n)]
    for pi, gi, li in zip(p, grp, lineno):
        per_group[gi].append((float(pi), int(li)))
    tops = []
    for g in range(n):
        per_group[g].sort(reverse=True)
        tops.append(per_group[g][:topk])
    return tops


# ---------------- alias slotting / templates ----------------

def slotted_template(ins, del_line, ctx):
    """Encode an insert line with alias slots relative to (deleted line, row context)."""
    del_x = []
    for m in PAT_X.findall(del_line):
        v = int(m)
        if v not in del_x:
            del_x.append(v)
    out = []
    last = 0
    new_seen = {}
    for m in PAT_X.finditer(ins):
        out.append(ins[last:m.start()])
        v = int(m.group(1))
        if v in del_x:
            out.append('\x00D%d\x00' % del_x.index(v))
        elif v not in ctx['ctx_x']:
            if v not in new_seen:
                new_seen[v] = len(new_seen)
            out.append('\x00N%d\x00' % new_seen[v])
        else:
            out.append('\x00C\x00')
        last = m.end()
    out.append(ins[last:])
    return PAT_T.sub('T', ''.join(out))


SLOT_RE = re.compile(r'\x00(D(\d+)|N(\d+)|C)\x00')


def instantiate_template(tpl, del_line, ctx):
    del_x = []
    for m in PAT_X.findall(del_line):
        v = int(m)
        if v not in del_x:
            del_x.append(v)
    max_x = ctx['max_x']
    ctx_sorted = sorted(ctx['ctx_x'])

    def sub(m):
        g = m.group(1)
        if g == 'C':
            return 'x%03d' % (ctx_sorted[-1] if ctx_sorted else 0)
        if g.startswith('D'):
            k = int(m.group(2))
            if k < len(del_x):
                return 'x%03d' % del_x[k]
            return 'x%03d' % (max_x + 1)
        return 'x%03d' % (max_x + 1 + int(m.group(3)))

    s = SLOT_RE.sub(sub, tpl)
    if '\x00' in s:
        return None
    s = re.sub(r'(?<![A-Za-z0-9_])T(?![0-9A-Za-z_])',
               'T%03d' % max(ctx['max_T'], 0), s)
    return s


def remap_insert(src_ins, src_del, src_ctx, tgt_del, tgt_ctx):
    """Adapt a neighbor's insert line into the target row's alias space."""
    src_del_x = [int(m) for m in PAT_X.findall(src_del)]
    tgt_del_x = [int(m) for m in PAT_X.findall(tgt_del)]
    pos_map = {}
    for k, v in enumerate(src_del_x):
        if v not in pos_map and k < len(tgt_del_x):
            pos_map[v] = tgt_del_x[k]
    res = []
    last = 0
    new_counter = 0
    for m in PAT_X.finditer(src_ins):
        res.append(src_ins[last:m.start()])
        v = int(m.group(1))
        if v in pos_map:
            res.append('x%03d' % pos_map[v])
        elif v not in src_ctx['ctx_x']:
            new_counter += 1
            res.append('x%03d' % (tgt_ctx['max_x'] + new_counter))
        elif v in tgt_ctx['ctx_x']:
            res.append('x%03d' % v)
        else:
            new_counter += 1
            res.append('x%03d' % (tgt_ctx['max_x'] + new_counter))
        last = m.end()
    res.append(src_ins[last:])
    return ''.join(res)


def remap_alias_kind(text, src_del, src_context, tgt_del, tgt_context,
                     tgt_max, pattern, prefix):
    """Remap one anonymized alias kind, keeping repeated aliases stable."""
    src_deleted = []
    for value in pattern.findall(src_del):
        value = int(value)
        if value not in src_deleted:
            src_deleted.append(value)
    tgt_deleted = []
    for value in pattern.findall(tgt_del):
        value = int(value)
        if value not in tgt_deleted:
            tgt_deleted.append(value)
    positional = {value: tgt_deleted[i] for i, value in enumerate(src_deleted)
                  if i < len(tgt_deleted)}
    extra = {}
    next_value = tgt_max + 1
    chunks = []
    last = 0
    for match in pattern.finditer(text):
        chunks.append(text[last:match.start()])
        value = int(match.group(1))
        if value in positional:
            mapped = positional[value]
        elif value in src_context and value in tgt_context:
            mapped = value
        else:
            if value not in extra:
                extra[value] = next_value
                next_value += 1
            mapped = extra[value]
        chunks.append(f'{prefix}{mapped:03d}')
        last = match.end()
    chunks.append(text[last:])
    return ''.join(chunks)


def remap_all_aliases(src_ins, src_del, src_ctx, tgt_del, tgt_ctx):
    """Remap x### and T### aliases from a neighbor into the target row."""
    remapped = remap_alias_kind(
        src_ins, src_del, src_ctx['ctx_x'], tgt_del, tgt_ctx['ctx_x'],
        tgt_ctx['max_x'], PAT_X, 'x')
    return remap_alias_kind(
        remapped, src_del, src_ctx['ctx_T'], tgt_del, tgt_ctx['ctx_T'],
        tgt_ctx['max_T'], PAT_T, 'T')



def make_doc(ctx, del_line):
    return (norm_alias(del_line.strip()) + ' || ' + norm_alias(ctx['fb'])[:200]
            + ' || ' + norm_alias(ctx['sh'])[:200])


# ---------------- candidates + ranker ----------------

SRC_LIST = ['copy', 'simp', 'rfl', 'simp_new', 'nn_raw', 'nn_remap',
            'nn_remap_all', 'template', 'template_cat']


def trigram_consensus(strs):
    M = len(strs)
    if M <= 1:
        return np.zeros(M, dtype=np.float32)
    vocab = {}
    grams_per = []
    for s in strs:
        grams = Counter(s[i:i + 3] for i in range(max(1, len(s) - 2)))
        grams_per.append(grams)
        for g in grams:
            if g not in vocab:
                vocab[g] = len(vocab)
    A = np.zeros((M, len(vocab)), dtype=np.float32)
    for r, grams in enumerate(grams_per):
        for g, cnt in grams.items():
            A[r, vocab[g]] = cnt
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1
    A /= norms
    with np.errstate(all='ignore'):  # Accelerate BLAS raises spurious FP flags
        S = A @ A.T
    return ((S.sum(axis=1) - 1) / (M - 1)).astype(np.float32)


class PatchModel:
    """Container for all train-fitted state."""

    def __init__(self):
        self.fb_cats = None
        self.start_model = None
        self.vec = None
        self.nn = None
        self.src_rows = None
        self.top_templates = None
        self.top_by_cat = None
        self.ins_freq = None
        self.token_classes = None
        self.token_to_col = None
        self.ranker = None
        self.ranker_hgb = None

    # ----- candidate generation -----
    def gen_candidates(self, c, dl, nn_idx, nn_dist):
        ind = ' ' * indent_of(dl)
        cands = {}

        def add(s, src, **kw):
            if s not in cands:
                cands[s] = {'srcs': set(), 'nn_rank': 99, 'nn_dist': 1.0, 'votes': 0,
                            'tpl_freq': 0, 'tpl_cat_freq': 0, 'nb_del_sim': 0.0, 'wvote': 0.0}
            d = cands[s]
            d['srcs'].add(src)
            d['votes'] += 1
            d['wvote'] += kw.get('w', 0.0)
            for key in ('nn_rank', 'nn_dist'):
                if key in kw:
                    d[key] = min(d[key], kw[key])
            for key in ('tpl_freq', 'tpl_cat_freq', 'nb_del_sim'):
                if key in kw:
                    d[key] = max(d[key], kw[key])

        add(dl, 'copy')
        add(ind + 'simp', 'simp')
        add(ind + 'rfl', 'rfl')
        add(ind + 'simp [x%03d]' % (c['max_x'] + 1), 'simp_new')
        dl_norm = norm_alias(dl.strip())
        for k in range(len(nn_idx)):
            s_ins, s_del, s_ctx = self.src_rows[nn_idx[k]]
            nb_sim = 1.0 - lev(norm_alias(s_del.strip()), dl_norm) / max(len(dl_norm), 1)
            w = 1.0 - nn_dist[k]
            add(ind + s_ins.strip(), 'nn_raw', nn_rank=k, nn_dist=nn_dist[k],
                nb_del_sim=nb_sim, w=w)
            rm = remap_insert(s_ins.strip(), s_del, s_ctx, dl, c)
            add(ind + rm, 'nn_remap', nn_rank=k, nn_dist=nn_dist[k],
                nb_del_sim=nb_sim, w=w)
            rm_all = remap_all_aliases(s_ins.strip(), s_del, s_ctx, dl, c)
            add(ind + rm_all, 'nn_remap_all', nn_rank=k, nn_dist=nn_dist[k],
                nb_del_sim=nb_sim, w=w)
        for tpl, freq in self.top_templates:
            inst = instantiate_template(tpl, dl, c)
            if inst:
                add(ind + inst, 'template', tpl_freq=freq)
        fbc = self.fb_cats.get(fb_shape(c['fb']), 0)
        for tpl, freq in self.top_by_cat.get(fbc, []):
            inst = instantiate_template(tpl, dl, c)
            if inst:
                add(ind + inst, 'template_cat', tpl_cat_freq=freq)
        return cands

    # ----- features -----
    def featurize(self, cands, dl, c, sp, sp2, srank, is_lastne, token_probs):
        strs = list(cands.keys())
        anchor = None
        bestw = -1.0
        for s_, d_ in cands.items():
            if 'nn_remap' in d_['srcs'] and d_['wvote'] > bestw:
                bestw = d_['wvote']
                anchor = s_
        rows = []
        dl_tok = tok_id(dl)
        fbc = self.fb_cats.get(fb_shape(c['fb']), 0)
        n_case = c['fb'].count('case ')
        sh_x = c['sh_x']
        for s in strs:
            d = cands[s]
            st = s.strip()
            sim_dl = sim(s, dl)
            aliases = PAT_X.findall(s)
            n_new = sum(1 for m in aliases if int(m) > c['max_x'])
            n_in_sh = sum(1 for m in aliases if int(m) in sh_x)
            token_col = self.token_to_col.get(tok_id(s))
            token_prob = float(token_probs[token_col]) if token_col is not None else 0.0
            token_rank = float(np.sum(token_probs > token_prob))
            f = [1.0 if src in d['srcs'] else 0.0 for src in SRC_LIST]
            f += [
                d['nn_rank'], d['nn_dist'], d['votes'], d['wvote'],
                d['tpl_freq'], d['tpl_cat_freq'], d['nb_del_sim'],
                self.ins_freq.get(st, 0),
                len(st), len(dl.strip()), len(st) / max(1, len(dl.strip())),
                tok_id(s), dl_tok, 1.0 if tok_id(s) == dl_tok else 0.0,
                sim_dl, len(aliases), n_new, n_in_sh / max(1, len(aliases)),
                1.0 if '←' in s else 0.0,
                fbc, n_case, len(c['sh']),
                sp, sp2, sp - sp2, srank, is_lastne,
                sim(s, anchor) if anchor else 0.0,
                token_prob, token_rank, float(np.max(token_probs)),
            ]
            rows.append(f)
        F = np.array(rows, dtype=np.float32)
        F = np.hstack([F, trigram_consensus(strs)[:, None]])
        return strs, F

    def build_row_entries(self, c, tops, d_nn, i_nn, token_probs, exclude=None):
        """All (start_line, candidate, features) for one row.

        d_nn/i_nn and token_probs are aligned per-start-rank arrays.
        exclude is a train row global index to drop from its own neighbor list.
        """
        entries = []
        probs = [p for p, _ in tops]
        p1 = probs[0]
        p2 = probs[1] if len(probs) > 1 else 0.0
        for rank, (sp, sl) in enumerate(tops):
            dl = c['lines'][sl - 1]
            nbr = list(zip(i_nn[rank], d_nn[rank]))
            if exclude is not None:
                nbr = [(j, dd) for j, dd in nbr if j != exclude]
            nbr = nbr[:K_NN]
            cands = self.gen_candidates(c, dl, [j for j, _ in nbr], [dd for _, dd in nbr])
            strs, F = self.featurize(cands, dl, c, sp, p2 if rank == 0 else p1,
                                     rank, 1.0 if sl == c['last_ne'] else 0.0,
                                     token_probs[rank])
            for s, frow in zip(strs, F):
                entries.append((sl, s, frow))
        return entries


def run_pipeline(train, test):
    """Fit on train and return predicted test patches."""
    ans = train['answer_json'].map(json.loads)
    train_start = ans.map(lambda a: a['replace_start_line']).to_numpy()
    train_ins = [a['insert_lines'][0] for a in ans]

    tr_ctxs = [row_context(r) for r in train.itertuples()]
    te_ctxs = [row_context(r) for r in test.itertuples()]
    n_tr = len(train)

    M = PatchModel()
    M.fb_cats = {s: i + 1 for i, (s, _) in
                 enumerate(Counter(fb_shape(c['fb']) for c in tr_ctxs).most_common(N_FB_CATS))}

    # ---- start model ----
    M.start_model = fit_start_model(tr_ctxs, train_start, M.fb_cats)
    tops_tr = predict_start_topk(M.start_model, tr_ctxs, M.fb_cats, N_START)
    tops_te = predict_start_topk(M.start_model, te_ctxs, M.fb_cats, N_START)

    tr_dels = [tr_ctxs[i]['lines'][train_start[i] - 1] for i in range(n_tr)]

    # ---- retrieval index over train rows (true deleted lines) ----
    docs_tr = [make_doc(tr_ctxs[i], tr_dels[i]) for i in range(n_tr)]
    M.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2,
                            max_features=200000)
    Xtr = M.vec.fit_transform(docs_tr)
    M.nn = NearestNeighbors(n_neighbors=K_NN + 1, metric='cosine').fit(Xtr)
    M.src_rows = {i: (train_ins[i], tr_dels[i], tr_ctxs[i]) for i in range(n_tr)}
    token_y = np.array([tok_id(s) for s in train_ins])
    M.token_classes = np.unique(token_y)
    M.token_to_col = {value: col for col, value in enumerate(M.token_classes)}
    def aligned_token_probs(model, X):
        probs = np.zeros((X.shape[0], len(M.token_classes)), dtype=np.float32)
        raw = model.predict_proba(X)
        for source_col, value in enumerate(model.classes_):
            probs[:, M.token_to_col[value]] = raw[:, source_col]
        return probs

    # ---- templates / frequencies ----
    tpl_counter = Counter()
    tpl_by_cat = defaultdict(Counter)
    for i in range(n_tr):
        tpl = slotted_template(train_ins[i].strip(), tr_dels[i], tr_ctxs[i])
        tpl_counter[tpl] += 1
        tpl_by_cat[M.fb_cats.get(fb_shape(tr_ctxs[i]['fb']), 0)][tpl] += 1
    M.top_templates = tpl_counter.most_common(N_TPL)
    M.top_by_cat = {cat: cnt.most_common(N_TPL_CAT) for cat, cnt in tpl_by_cat.items()}
    M.ins_freq = Counter(s.strip() for s in train_ins)

    # ---- ranker training data (leave-one-out neighbors, top-N_START starts) ----
    docs_all = []
    doc_owners = []
    for g in range(n_tr):
        for sp, sl in tops_tr[g]:
            docs_all.append(make_doc(tr_ctxs[g], tr_ctxs[g]['lines'][sl - 1]))
            doc_owners.append(g)
    Xall = M.vec.transform(docs_all)
    d_all, i_all = M.nn.kneighbors(Xall)
    doc_owners = np.asarray(doc_owners)
    token_probs_all = np.zeros((len(docs_all), len(M.token_classes)), dtype=np.float32)
    token_folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_rows, holdout_rows in token_folds.split(Xtr):
        token_model = LogisticRegression(C=1.0, max_iter=300)
        token_model.fit(Xtr[fit_rows], token_y[fit_rows])
        mask = np.isin(doc_owners, holdout_rows)
        token_probs_all[mask] = aligned_token_probs(token_model, Xall[mask])

    Xr, yr = [], []
    ptr = 0
    for g in range(n_tr):
        k = len(tops_tr[g])
        d_nn = d_all[ptr:ptr + k]
        i_nn = i_all[ptr:ptr + k]
        token_probs = token_probs_all[ptr:ptr + k]
        ptr += k
        entries = M.build_row_entries(tr_ctxs[g], tops_tr[g], d_nn, i_nn,
                                      token_probs, exclude=g)
        ts_true = int(train_start[g])
        for sl, s, frow in entries:
            ts = sim(s, train_ins[g])
            ex = 1.0 if s == train_ins[g] else 0.0
            loc = 0.5 * math.exp(-abs(sl - ts_true) / 2) + 0.5
            y = 0.18 * loc + 0.58 * max(0.0, ts) + 0.14 * ex \
                + 0.10 * ex * (1.0 if sl == ts_true else 0.0)
            Xr.append(frow)
            yr.append(y)
    Xr = np.asarray(Xr, dtype=np.float32)
    yr = np.asarray(yr, dtype=np.float32)
    M.ranker = CatBoostRegressor(iterations=N_RANK_ITER, depth=8, learning_rate=0.05,
                                 loss_function='RMSE', random_seed=SEED,
                                 verbose=False, allow_writing_files=False,
                                 task_type='CPU')
    M.ranker.fit(Xr, yr)
    if RANK_HGB_BLEND > 0:
        M.ranker_hgb = HistGradientBoostingRegressor(
            max_iter=50, learning_rate=0.08, max_leaf_nodes=63,
            random_state=SEED)
        M.ranker_hgb.fit(Xr, yr)

    # ---- inference on test ----
    docs_te = []
    for g in range(len(test)):
        for sp, sl in tops_te[g]:
            docs_te.append(make_doc(te_ctxs[g], te_ctxs[g]['lines'][sl - 1]))
    Xte = M.vec.transform(docs_te)
    d_te, i_te = M.nn.kneighbors(Xte)
    token_model = LogisticRegression(C=1.0, max_iter=300)
    token_model.fit(Xtr, token_y)
    token_probs_te = aligned_token_probs(token_model, Xte)

    preds = []
    ptr = 0
    for g in range(len(test)):
        k = len(tops_te[g])
        d_nn = d_te[ptr:ptr + k]
        i_nn = i_te[ptr:ptr + k]
        token_probs = token_probs_te[ptr:ptr + k]
        ptr += k
        entries = M.build_row_entries(te_ctxs[g], tops_te[g], d_nn, i_nn, token_probs)
        F = np.array([e[2] for e in entries])
        scores = M.ranker.predict(F)
        if M.ranker_hgb is not None:
            scores = ((1.0 - RANK_HGB_BLEND) * scores +
                      RANK_HGB_BLEND * M.ranker_hgb.predict(F))
        bi = int(np.argmax(scores))
        preds.append({'replace_start_line': int(entries[bi][0]),
                      'delete_line_count': 1,
                      'insert_lines': [entries[bi][1]]})
    return preds


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(public_dir / 'train.csv')
    test = pd.read_csv(public_dir / 'test.csv')

    preds = run_pipeline(train, test)

    out = [json.dumps(p, separators=(',', ':'), ensure_ascii=False) for p in preds]
    sub = pd.DataFrame({'id': test['id'], 'answer_json': out})
    sub.to_csv(submission_out, index=False)
    print(f'wrote {submission_out} ({len(sub)} rows)')


if __name__ == '__main__':
    main()
