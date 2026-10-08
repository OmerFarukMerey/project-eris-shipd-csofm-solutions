"""Henkin Dependency Recovery - end-to-end solution.

python3 solution.py <public_dir> <submission_out>

Pipeline (fixed, static plan):
  1. Parse every formula; compute hint-independent structure per formula (degree / polarity / clause-shape
     counts, Tseitin gate detection, BFS distances from every universal, co-occurrence, polarity-aware
     Weisfeiler-Lehman colours, one/two-hop clause-shape signatures, random-walk profiles).
  2. Train formulas: simulate the documented damage model (each line survives w.p. 8%, queries = up to 400
     strict + about half as many full) twice per formula and build per-query / per-(query, universal) /
     per-(query, hint) feature rows. Test formulas: the real test_hints.csv lines and test.csv queries.
  3. Stage S  : LightGBM "same dependency set" model on (query, hinted existential) pairs; 5-fold grouped
                out-of-fold predictions on train, final model on all train for test.
  4. Stage G/P: LightGBM gate model P(depends on all universals) and LightGBM pair model
                P(u in D | strict) using the stage-S similarity votes as extra features.
                5-fold grouped OOF -> validation score + in-script search of one decode knob.
  5. Decode   : per query, expected-metric maximisation over the model's own top-k sets
                (Monte-Carlo over the pair-model Bernoulli posterior) against the gate probability.
"""
import os
# fixed thread counts for the numeric backends (same on every machine; set before numpy/scipy load)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
import sys, gzip, math, zlib, collections
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import lightgbm as lgb

SEED = 0
N_DRAWS = 4            # simulated hint/query draws per train formula
N_FOLDS = 5            # grouped CV folds (OOF stacking + validation)
N_SAMPLES = 256        # Monte-Carlo samples per query in the decoder
N_SEEDS = 3            # LightGBM seed ensemble size for the gate and pair models
NUM_THREADS = 8
FAMILIES = ["bloem_synthesis", "bounded_synthesis", "cnf_lifted", "partial_equivalence", "ramsey",
            "random_dqbf", "scholl_henkin", "succinct_graph", "tentrup_synthesis"]
FAM_CODE = {f: i for i, f in enumerate(FAMILIES)}
BIG = 99.0
CLIP = 30.0
MAX_HINT_SIM = 3000
MAX_SH = 160
MAX_FH = 40
# damage / query-sampling model as documented in PROBLEM.md (used only to simulate training episodes)
P_HINT = 0.08
MAX_STRICT_Q = 400


def log(*a):
    print(*a, flush=True)


# =============================================================================== parsing
def parse_formula(path):
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")
    hdr = lines[0].split()
    V = int(hdr[2])
    uni = np.array(lines[1].split()[1:-1], dtype=np.int64)
    exi = np.array(lines[2].split()[1:-1], dtype=np.int64)
    lits = np.array(b" ".join(lines[3:]).split(), dtype=np.int64)
    ends = np.flatnonzero(lits == 0)
    lens = np.diff(np.concatenate([[-1], ends])) - 1
    lits = lits[lits != 0]
    lens = lens[lens > 0]  # drop empty clauses if any
    ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    if len(lits):
        V = max(V, int(np.abs(lits).max()))
    if len(uni):
        V = max(V, int(uni.max()))
    if len(exi):
        V = max(V, int(exi.max()))
    return dict(V=V, uni=uni, exi=exi, lits=lits, ptr=ptr)


def parse_deps(s):
    s = str(s).strip()
    if s in ("{}", "none", "", "nan"):
        return frozenset()
    return frozenset(int(x) for x in s.replace(",", " ").replace(";", " ").split())


# =============================================================================== per-formula structure
def gate_features(lits, ptr, clen, V, maxlen=40):
    off = V + 1
    bmask = clen == 2
    b0 = lits[ptr[:-1][bmask]]; b1 = lits[ptr[:-1][bmask] + 1]
    keys = np.unique(np.concatenate([(b0 + off) * (2 * off) + (b1 + off), (b1 + off) * (2 * off) + (b0 + off)]))
    gate_out = np.zeros(V, np.float32); gate_in = np.zeros(V, np.float32)
    eqk = (-b0 + off) * (2 * off) + (-b1 + off)
    has_eq = np.isin(eqk, keys)
    n_eq = (np.bincount(np.abs(b0[has_eq]) - 1, minlength=V) + np.bincount(np.abs(b1[has_eq]) - 1, minlength=V)).astype(np.float32)

    def member(q):
        if len(keys) == 0:
            return np.zeros(q.shape, bool)
        pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return keys[pos] == q
    sel = np.flatnonzero((clen >= 3) & (clen <= maxlen))
    for k in np.unique(clen[sel]):
        cs = sel[clen[sel] == k]
        II, JJ = np.nonzero(~np.eye(k, dtype=bool))
        chunk = max(1, 4_000_000 // (k * (k - 1)))
        oks = []
        for s in range(0, len(cs), chunk):
            M = lits[ptr[cs[s:s + chunk]][:, None] + np.arange(k)[None, :]]
            qk = (-M[:, II] + off) * (2 * off) + (-M[:, JJ] + off)
            oks.append(member(qk).reshape(len(M), k, k - 1).all(2))
        ok = np.concatenate(oks)
        M = lits[ptr[cs][:, None] + np.arange(k)[None, :]]
        rows, cols = np.nonzero(ok)
        np.add.at(gate_out, np.abs(M[rows, cols]) - 1, 1)
        if len(rows):
            Min = np.abs(M[rows]) - 1
            mask = np.ones(Min.shape, bool); mask[np.arange(len(rows)), cols] = False
            np.add.at(gate_in, Min[mask], 1)
    return gate_out, gate_in, n_eq


_M1 = np.uint64(0xbf58476d1ce4e5b9); _M2 = np.uint64(0x94d049bb133111eb)


def mix64(x):
    x = x.astype(np.uint64, copy=True)
    x ^= x >> np.uint64(30); x *= _M1
    x ^= x >> np.uint64(27); x *= _M2
    x ^= x >> np.uint64(31)
    return x


def wl_colors(f, iters=4):
    """Polarity-aware Weisfeiler-Lehman refinement (multiset hashing) on the literal/clause incidence."""
    lits = f["lits"]; ptr = f["ptr"]; V = f["V"]
    var = np.abs(lits) - 1
    sgn = (lits > 0).astype(np.uint64)
    order = np.argsort(var, kind="stable")
    vs = var[order]
    vstarts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
    vids = vs[vstarts]
    clen = np.diff(ptr)
    col = np.zeros(V, np.uint64); col[f["uni"] - 1] = np.uint64(1)
    out = []
    with np.errstate(over="ignore"):
        for it in range(iters):
            contrib = mix64(col[var] * np.uint64(2) + sgn + np.uint64(1000 + it))
            ccol = np.add.reduceat(contrib, ptr[:-1])
            msg = mix64(np.repeat(ccol, clen) - contrib + sgn * np.uint64(7919) + np.uint64(77 + it))
            acc = np.zeros(V, np.uint64)
            acc[vids] = np.add.reduceat(msg[order], vstarts)
            col = mix64(col * np.uint64(31) + acc + np.uint64(5))
            _, inv = np.unique(col, return_inverse=True)
            out.append(inv.astype(np.int64))
    return np.stack(out, 0)


def formula_cache(f):
    V = f["V"]; lits = f["lits"]; ptr = f["ptr"]; uni = f["uni"]
    C = len(ptr) - 1; nU = len(uni)
    clen = np.diff(ptr)
    var = np.abs(lits) - 1
    cl = np.repeat(np.arange(C), clen)
    pos = lits > 0
    is_uni = np.zeros(V, bool); is_uni[uni - 1] = True
    luni = is_uni[var]
    c_nu = np.bincount(cl, weights=luni, minlength=C)
    deg = np.bincount(var, minlength=V).astype(np.float32)
    npos = np.bincount(var, weights=pos, minlength=V).astype(np.float32)
    lclen = clen[cl]
    nbin = np.bincount(var, weights=(lclen == 2), minlength=V).astype(np.float32)
    ntern = np.bincount(var, weights=(lclen == 3), minlength=V).astype(np.float32)
    nlong = np.bincount(var, weights=(lclen >= 4), minlength=V).astype(np.float32)
    nunit = np.bincount(var, weights=(lclen == 1), minlength=V).astype(np.float32)
    sumlen = np.bincount(var, weights=lclen, minlength=V).astype(np.float32)
    uni_in_cl = c_nu[cl] - luni
    n_uni_cl = np.bincount(var, weights=(uni_in_cl > 0), minlength=V).astype(np.float32)
    sum_uni_cl = np.bincount(var, weights=uni_in_cl, minlength=V).astype(np.float32)
    gate_out, gate_in, n_eq = gate_features(lits, ptr, clen, V)
    vf = np.stack([deg, npos, deg - npos, nbin, ntern, nlong, nunit,
                   np.where(deg > 0, sumlen / np.maximum(deg, 1), 0), n_uni_cl, sum_uni_cl,
                   gate_out, gate_in, n_eq], 1).astype(np.float32)
    A = sp.csr_matrix((np.ones(len(var), np.float32), (var, cl)), shape=(V, C))
    Ap = sp.csr_matrix((np.ones(pos.sum(), np.float32), (var[pos], cl[pos])), shape=(V, C))
    An = sp.csr_matrix((np.ones((~pos).sum(), np.float32), (var[~pos], cl[~pos])), shape=(V, C))
    # BFS distances from universals (all paths / paths avoiding other universals)
    B = sp.bmat([[None, A], [A.T, None]], format="csr")
    d_all = shortest_path(B, unweighted=True, indices=uni - 1)[:, :V] / 2.0
    A_e = sp.diags((~is_uni).astype(np.float32)) @ A
    Au = A[uni - 1]
    B2 = sp.bmat([[None, A_e, None], [A_e.T, None, Au.T], [None, Au, None]], format="csr")
    d_nou = shortest_path(B2, unweighted=True, indices=V + C + np.arange(nU))[:, :V] / 2.0
    d_all[np.isinf(d_all)] = BIG; d_nou[np.isinf(d_nou)] = BIG
    co = (A @ Au.T).toarray().astype(np.float32)
    wl = wl_colors(f, 4)
    # clause-shape signatures
    s = pos.astype(np.int64)
    npos_c = np.bincount(cl, weights=s, minlength=C)
    Lb = np.minimum(clen[cl], 4) - 1
    pb = np.minimum(npos_c[cl] - s, 2).astype(np.int64)
    ub = ((c_nu[cl] - luni) > 0).astype(np.int64)
    b = s * 24 + Lb * 6 + pb * 2 + ub
    sig1 = sp.csr_matrix((np.ones(len(var), np.float32), (var, b)), shape=(V, 48)).toarray()
    P = Ap.T @ sig1; N = An.T @ sig1
    same = Ap @ P + An @ N - deg[:, None] * sig1
    opp = Ap @ N + An @ P
    sig2 = np.concatenate([same, opp], 1).astype(np.float32)
    # random-walk profiles from universals
    dv = np.asarray(A.sum(1)).ravel(); dc = np.asarray(A.sum(0)).ravel()
    Xu = np.zeros((V, nU), np.float32); Xu[uni - 1, np.arange(nU)] = 1.0
    walks = []
    for _ in range(4):
        Xu = (A @ ((A.T @ Xu) / np.maximum(dc, 1)[:, None])) / np.maximum(dv, 1)[:, None]
        walks.append(Xu.astype(np.float32))
    return dict(vf=vf, d_all=d_all.astype(np.float32), d_nou=d_nou.astype(np.float32), co=co, wl=wl,
                uni=uni, exi=f["exi"], C=C, V=V, A=A, Ap=Ap, An=An, sig1=sig1, sig2=sig2, walk=np.stack(walks, 0))


# =============================================================================== query feature rows
def _log(x):
    return np.log1p(np.maximum(x, 0))


VF_NAMES = ["deg", "npos", "nneg", "nbin", "ntern", "nlong", "nunit", "mlen", "n_uni_cl", "sum_uni_cl", "gate_out", "gate_in", "n_eq"]


def build_rows(cache, family, hints, queries, rng_seed=0):
    """Per-query gate rows and per-(query, universal) pair rows. Uses only the formula, its hint lines
    and the query itself (each query's row is independent of which other queries exist)."""
    vf = cache["vf"]; d_all = cache["d_all"]; d_nou = cache["d_nou"]; co = cache["co"]
    uni = cache["uni"]; exi = cache["exi"]
    nU = len(uni); nE = max(1, len(exi)); V = vf.shape[0]; C = float(max(1, cache["C"]))
    uidx = {int(u): j for j, u in enumerate(uni)}
    q = np.asarray(queries, dtype=np.int64) - 1
    nq = len(q)
    hv = np.array(sorted(hints.keys()), dtype=np.int64)
    hfull = np.array([len(hints[v]) == nU for v in hv], bool)
    hmat = np.zeros((len(hv), nU), np.float32)
    for i, v in enumerate(hv):
        for u in hints[v]:
            hmat[i, uidx[u]] = 1
    n_h = len(hv); n_hs = int((~hfull).sum())
    frac_hfull = float(hfull.mean()) if n_h else 0.5
    if n_h > MAX_HINT_SIM:
        sub = np.sort(np.random.default_rng(rng_seed).choice(n_h, MAX_HINT_SIM, replace=False))
    else:
        sub = np.arange(n_h)
    hs_v = hv[sub] - 1; hs_full = hfull[sub]; hs_mat = hmat[sub]

    Pq = np.minimum(d_all[:, q].T, CLIP); Pq2 = np.minimum(d_nou[:, q].T, CLIP)
    Ph = np.minimum(d_all[:, hs_v].T, CLIP)
    Pqn = Pq - Pq.min(1, keepdims=True); Phn = Ph - Ph.min(1, keepdims=True)
    lvf = _log(vf)
    Fq = lvf[q]; Fh = lvf[hs_v]

    def mad(Aa, Bb):
        out = np.empty((len(Aa), len(Bb)), np.float32)
        if len(Bb) == 0:
            return out
        step = max(1, 2_000_000 // max(1, len(Bb) * Aa.shape[1]))
        for s in range(0, len(Aa), step):
            out[s:s + step] = np.abs(Aa[s:s + step, None, :] - Bb[None, :, :]).mean(2)
        return out
    dprof = mad(Pqn, Phn); dfeat = mad(Fq, Fh); dcomb = dprof + dfeat

    def nearest(dm, mask):
        if mask.sum() == 0:
            return np.full(nq, 50.0, np.float32)
        dd = np.where(mask[None, :], dm, np.inf)
        return dd[np.arange(nq), dd.argmin(1)]

    g = {}
    for name, dm in (("prof", dprof), ("feat", dfeat), ("comb", dcomb)):
        nf = nearest(dm, hs_full); ns = nearest(dm, ~hs_full)
        g[f"nn_full_{name}"] = nf; g[f"nn_strict_{name}"] = ns
        g[f"nn_diff_{name}"] = np.minimum(nf, 50) - np.minimum(ns, 50)
        for tau in (0.25, 1.0):
            w = np.exp(-dm / tau); sw = w.sum(1)
            g[f"knn_full_{name}_{tau}"] = np.where(sw > 1e-12, (w * hs_full[None, :]).sum(1) / np.maximum(sw, 1e-12), frac_hfull)
            g[f"knn_mass_{name}_{tau}"] = sw
    for k, nm in enumerate(VF_NAMES):
        g["e_" + nm] = vf[q, k]
    for nm, P in (("all", Pq), ("nou", Pq2)):
        g[f"p{nm}_min"] = P.min(1); g[f"p{nm}_mean"] = P.mean(1); g[f"p{nm}_max"] = P.max(1)
        g[f"p{nm}_std"] = P.std(1); g[f"p{nm}_nmin"] = (P == P.min(1, keepdims=True)).mean(1)
        g[f"p{nm}_reach"] = (P < CLIP).mean(1)
    coq = co[q]
    g["co_n"] = (coq > 0).sum(1); g["co_frac"] = (coq > 0).mean(1); g["co_sum"] = coq.sum(1)
    g["f_nU"] = np.full(nq, nU, np.float32); g["f_lnE"] = np.full(nq, np.log(nE), np.float32)
    g["f_lC"] = np.full(nq, np.log(C), np.float32); g["f_EU"] = np.full(nq, np.log(nE / max(1, nU)), np.float32)
    g["f_fam"] = np.full(nq, FAM_CODE[family], np.float32)
    g["h_n"] = np.full(nq, np.log1p(n_h), np.float32); g["h_frac_full"] = np.full(nq, frac_hfull, np.float32)
    g["h_n_strict"] = np.full(nq, n_hs, np.float32)
    wl = cache["wl"]
    wl_votes = []
    for k in range(wl.shape[0]):
        colk = wl[k]; cq = colk[q]; ch = colk[hv - 1]
        g[f"wl{k}_csize"] = np.log1p(np.bincount(colk)[cq])
        order = np.argsort(ch, kind="stable"); chs = ch[order]
        lo = np.searchsorted(chs, cq, "left"); hi = np.searchsorted(chs, cq, "right")
        nsame = (hi - lo).astype(np.float32)
        g[f"wl{k}_nsame"] = np.log1p(nsame)
        cumf = np.concatenate([[0], np.cumsum(hfull[order])])
        nfull_same = cumf[hi] - cumf[lo]
        g[f"wl{k}_ffull"] = np.where(nsame > 0, nfull_same / np.maximum(nsame, 1), -1)
        hm_s = hmat[order] * (~hfull[order])[:, None]
        cum = np.concatenate([np.zeros((1, nU), np.float32), np.cumsum(hm_s, 0)])
        nstr = nsame - nfull_same
        v = (cum[hi] - cum[lo]) / np.maximum(nstr, 1)[:, None]
        v[nstr == 0] = -1
        wl_votes.append(v)

    pair = {}
    UU = np.tile(np.arange(nU), nq)
    pair["d_all"] = Pq.reshape(-1); pair["d_nou"] = Pq2.reshape(-1)
    pair["d_all_n"] = (Pq - Pq.min(1, keepdims=True)).reshape(-1)
    pair["d_nou_n"] = (Pq2 - Pq2.min(1, keepdims=True)).reshape(-1)
    pair["d_all_rlt"] = (Pq[:, :, None] > Pq[:, None, :]).mean(2).reshape(-1)
    pair["d_all_rle"] = (Pq[:, :, None] >= Pq[:, None, :]).mean(2).reshape(-1)
    pair["d_nou_rlt"] = (Pq2[:, :, None] > Pq2[:, None, :]).mean(2).reshape(-1)
    pair["d_all_z"] = ((Pq - Pq.mean(1, keepdims=True)) / (Pq.std(1, keepdims=True) + 1e-3)).reshape(-1)
    pair["co"] = coq.reshape(-1)
    uvf = lvf[uni - 1]
    for k, nm in enumerate(VF_NAMES):
        pair["u_" + nm] = uvf[UU, k]
    dE = np.minimum(d_all[:, exi - 1], CLIP) if len(exi) else np.zeros((nU, 1), np.float32)
    u_mean = dE.mean(1)
    pair["u_meanE"] = u_mean[UU]; pair["u_meanE_rel"] = (u_mean - u_mean.mean())[UU]
    pair["u_frac_near"] = (dE <= 2).mean(1)[UU]
    hp = (hs_mat[~hs_full].sum(0) + 0.5) / (n_hs + 1.0) if n_hs else np.full(nU, 0.5, np.float32)
    pair["h_prev"] = hp[UU]; pair["h_prev_rel"] = (hp - hp.mean())[UU]
    pair["h_n_strict"] = np.full(nq * nU, n_hs, np.float32)
    smask = ~hs_full; Hs = hs_mat[smask]
    for name, dm in (("prof", dprof), ("feat", dfeat), ("comb", dcomb)):
        dms = dm[:, smask]
        if dms.shape[1] == 0:
            for tau in (0.25, 1.0):
                pair[f"vote_{name}_{tau}"] = np.full(nq * nU, -1, np.float32)
            pair[f"nn_mem_{name}"] = np.full(nq * nU, -1, np.float32)
            pair[f"nn_d_{name}"] = np.full(nq * nU, 50, np.float32)
            continue
        for tau in (0.25, 1.0):
            w = np.exp(-(dms - dms.min(1, keepdims=True)) / tau)
            pair[f"vote_{name}_{tau}"] = ((w @ Hs) / w.sum(1, keepdims=True)).reshape(-1)
        a = dms.argmin(1)
        pair[f"nn_mem_{name}"] = Hs[a].reshape(-1)
        pair[f"nn_d_{name}"] = np.repeat(dms[np.arange(nq), a], nU)
    for k, v in enumerate(wl_votes):
        pair[f"wl{k}_vote"] = v.reshape(-1)
    A = cache["A"]; Ap = cache["Ap"]; An = cache["An"]
    is_u = np.zeros(V, bool); is_u[uni - 1] = True
    Ae = sp.diags((~is_u).astype(np.float32)) @ A

    def step(M, X):
        dv = np.asarray(M.sum(1)).ravel(); dc = np.asarray(M.sum(0)).ravel()
        return (M @ ((M.T @ X) / np.maximum(dc, 1)[:, None])) / np.maximum(dv, 1)[:, None]
    S = np.zeros((V, nU + 2), np.float32)
    S[hv[~hfull] - 1, :nU] = hmat[~hfull]; S[hv[~hfull] - 1, nU] = 1.0; S[hv[hfull] - 1, nU + 1] = 1.0
    X = S
    for k in range(1, 4):
        X = step(Ae, X)
        Xq = X[q]; den = Xq[:, nU]
        pair[f"rw{k}_vote"] = np.where(den[:, None] > 1e-9, Xq[:, :nU] / np.maximum(den, 1e-12)[:, None], -1).reshape(-1)
        pair[f"rw{k}_mass"] = np.repeat(np.log10(den + 1e-9), nU)
        fm = Xq[:, nU + 1]
        g[f"rw{k}_ffull"] = np.where(den + fm > 1e-9, fm / np.maximum(den + fm, 1e-12), -1)
        g[f"rw{k}_mass"] = np.log10(den + fm + 1e-9)
    same = (Ap @ (Ap.T @ S) + An @ (An.T @ S))[q]
    opp = (Ap @ (An.T @ S) + An @ (Ap.T @ S))[q]
    for nm, M in (("same", same), ("opp", opp)):
        den = M[:, nU]
        pair[f"co_{nm}_vote"] = np.where(den[:, None] > 0, M[:, :nU] / np.maximum(den, 1e-12)[:, None], -1).reshape(-1)
        pair[f"co_{nm}_mass"] = np.repeat(np.log1p(den), nU)
        g[f"co_{nm}_ffull"] = np.where(den + M[:, nU + 1] > 0, M[:, nU + 1] / np.maximum(den + M[:, nU + 1], 1e-12), -1)
    for k in range(4):
        R = cache["walk"][k][q]
        tot = R.sum(1, keepdims=True)
        pair[f"wk{k + 1}_rel"] = (np.where(tot > 0, R / np.maximum(tot, 1e-30), 0) * nU).reshape(-1)
        pair[f"wk{k + 1}_log"] = np.log10(R + 1e-12).reshape(-1)
    for nm in ["e_deg", "e_gate_out", "e_gate_in", "e_n_eq", "e_n_uni_cl", "pall_min", "pall_mean", "pnou_min",
               "co_n", "f_nU", "f_lnE", "f_fam", "f_EU", "nn_diff_comb", "knn_full_comb_0.25",
               "wl0_nsame", "wl1_nsame", "wl2_nsame", "wl3_nsame", "wl0_csize", "wl2_csize"]:
        pair["q_" + nm] = np.repeat(np.asarray(g[nm], np.float32), nU)
    gn = list(g.keys()); pn = list(pair.keys())
    gX = np.stack([np.asarray(g[k], np.float32) for k in gn], 1)
    pX = np.stack([np.broadcast_to(np.asarray(pair[k], np.float32), (nq * nU,)) for k in pn], 1)
    return gX, gn, pX, pn


def _rownorm(X):
    return X / np.maximum(X.sum(1, keepdims=True), 1e-9)


def build_sim(cache, family, hints, queries, rng_seed=0):
    """(query, hinted existential) pair rows for the learned 'same dependency set' model."""
    vf = cache["vf"]; d_all = cache["d_all"]; d_nou = cache["d_nou"]; co = cache["co"]
    uni = cache["uni"]; nU = len(uni)
    uidx = {int(u): j for j, u in enumerate(uni)}
    q = np.asarray(queries, np.int64) - 1; nq = len(q)
    hv = np.array(sorted(hints.keys()), np.int64)
    hfull = np.array([len(hints[v]) == nU for v in hv], bool)
    rng = np.random.default_rng(rng_seed + 12345)
    si = np.flatnonzero(~hfull); fi = np.flatnonzero(hfull)
    if len(si) > MAX_SH:
        si = np.sort(rng.choice(si, MAX_SH, replace=False))
    if len(fi) > MAX_FH:
        fi = np.sort(rng.choice(fi, MAX_FH, replace=False))
    sel = np.concatenate([si, fi]).astype(np.int64)
    hsel = hv[sel]; h = hsel - 1; nh = len(h)
    hmat = np.zeros((nh, nU), np.float32)
    for i, v in enumerate(hsel):
        for u in hints[int(v)]:
            hmat[i, uidx[u]] = 1
    if nh == 0 or nq == 0:
        return np.zeros((nq, 0, 0), np.float16), [], hsel, hmat
    keys_all = collections.Counter(hints[int(v)] for v in hv)
    hcnt = np.array([keys_all[hints[int(v)]] for v in hsel], np.float32)
    f = {}
    wl = cache["wl"]
    for k in range(wl.shape[0]):
        f[f"wl_eq{k}"] = (wl[k][q][:, None] == wl[k][h][None, :]).astype(np.float32)
    lvf = _log(vf); Fq, Fh = lvf[q], lvf[h]
    f["vf_l1"] = np.abs(Fq[:, None, :] - Fh[None, :, :]).mean(2)
    for j, nm in [(0, "deg"), (10, "gate_out"), (11, "gate_in"), (12, "n_eq"), (8, "n_uni_cl")]:
        f[f"d_{nm}"] = np.abs(Fq[:, j][:, None] - Fh[:, j][None, :])
    for nm, Sg in (("s1", cache["sig1"]), ("s2", cache["sig2"])):
        A_ = _rownorm(Sg[q]); B_ = _rownorm(Sg[h])
        f[f"{nm}_l1"] = np.abs(A_[:, None, :] - B_[None, :, :]).sum(2)
        An_ = A_ / np.maximum(np.linalg.norm(A_, axis=1, keepdims=True), 1e-9)
        Bn_ = B_ / np.maximum(np.linalg.norm(B_, axis=1, keepdims=True), 1e-9)
        f[f"{nm}_cos"] = An_ @ Bn_.T
    for nm, d in (("pa", d_all), ("pn", d_nou)):
        Pq = np.minimum(d[:, q].T, CLIP); Ph = np.minimum(d[:, h].T, CLIP)
        diff = np.abs((Pq - Pq.min(1, keepdims=True))[:, None, :] - (Ph - Ph.min(1, keepdims=True))[None, :, :])
        f[f"{nm}_mad"] = diff.mean(2); f[f"{nm}_max"] = diff.max(2); f[f"{nm}_eqf"] = (diff == 0).mean(2)
        f[f"{nm}_shift"] = Pq.min(1)[:, None] - Ph.min(1)[None, :]
    Cq = (co[q] > 0).astype(np.float32); Ch = (co[h] > 0).astype(np.float32)
    inter = Cq @ Ch.T
    un = Cq.sum(1)[:, None] + Ch.sum(1)[None, :] - inter
    f["co_jac"] = np.where(un > 0, inter / np.maximum(un, 1), -1)
    A = cache["A"]; Ap = cache["Ap"]; An = cache["An"]
    f["shared"] = np.log1p((A[q] @ A[h].T).toarray())
    f["shared_same"] = np.log1p((Ap[q] @ Ap[h].T + An[q] @ An[h].T).toarray())
    f["shared_opp"] = np.log1p((Ap[q] @ An[h].T + An[q] @ Ap[h].T).toarray())
    nb_q = A[q] @ A.T; nb_h = A[h] @ A.T
    nb_q.data[:] = 1; nb_h.data[:] = 1
    f["nb_common"] = np.log1p((nb_q @ nb_h.T).toarray())
    for k in range(cache["walk"].shape[0]):
        a = _rownorm(cache["walk"][k][q]); b = _rownorm(cache["walk"][k][h])
        f[f"walk{k}_l1"] = np.abs(a[:, None, :] - b[None, :, :]).sum(2)
    f["h_full"] = np.broadcast_to(hfull[sel][None, :].astype(np.float32), (nq, nh))
    f["h_size"] = np.broadcast_to((hmat.sum(1) / max(1, nU))[None, :], (nq, nh))
    f["h_cnt"] = np.broadcast_to(np.log1p(hcnt)[None, :], (nq, nh))
    f["q_gate_out"] = np.broadcast_to(np.minimum(vf[q, 10], 5)[:, None], (nq, nh))
    f["q_deg"] = np.broadcast_to(Fq[:, 0][:, None], (nq, nh))
    f["f_fam"] = np.full((nq, nh), FAM_CODE[family], np.float32)
    f["f_nU"] = np.full((nq, nh), nU, np.float32)
    names = list(f.keys())
    X = np.stack([np.broadcast_to(np.asarray(f[n], np.float32), (nq, nh)) for n in names], 2)
    return X.astype(np.float16), names, hsel, hmat


def make_item(cache, family, fid, hints, qv, rng_seed):
    gX, gn, pX, pn = build_rows(cache, family, hints, qv, rng_seed=rng_seed)
    sX, sn, hsel, hmat_s = build_sim(cache, family, hints, qv, rng_seed=rng_seed)
    return dict(fid=fid, family=family, qv=np.asarray(qv), nU=len(cache["uni"]), uni=cache["uni"], gX=gX, gn=gn,
                pX=pX, pn=pn, sX=sX, sn=sn, hmat_s=hmat_s, deps_h=[hints[int(v)] for v in hsel])


# =============================================================================== models
def lgb_params(objective, leaves, min_leaf, rounds):
    return dict(objective=objective, learning_rate=0.05, num_leaves=leaves, min_data_in_leaf=min_leaf, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=NUM_THREADS, seed=SEED,
                bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED,
                deterministic=True, force_row_wise=True, n_rounds=rounds)


PG = lgb_params("binary", 31, 20, 300)    # gate
PP = lgb_params("binary", 63, 50, 400)    # pair membership
PS = lgb_params("binary", 63, 40, 300)    # same-set similarity


class SeedEnsemble:
    """Average of LightGBM models that differ only in their random seeds."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], 0)


def fit_ens(params, X, y, w, cat):
    ms = []
    for sd in range(N_SEEDS):
        p = dict(params, seed=SEED + sd, bagging_seed=SEED + sd, feature_fraction_seed=SEED + sd, data_random_seed=SEED + sd)
        ms.append(fit(p, X, y, w, cat))
    return SeedEnsemble(ms)


def fit(params, X, y, w, cat):
    p = {k: v for k, v in params.items() if k != "n_rounds"}
    return lgb.train(p, lgb.Dataset(X, y, weight=w * len(w) / w.sum(), categorical_feature=cat), num_boost_round=params["n_rounds"])


def sim_train_rows(items, rng, max_q=60):
    Xs, ys, ws = [], [], []
    for it in items:
        if it["sX"].shape[1] == 0:
            continue
        nq = it["sX"].shape[0]
        qs = np.sort(rng.choice(nq, min(nq, max_q), replace=False))
        X = it["sX"][qs].reshape(-1, it["sX"].shape[2]).astype(np.float32)
        y = np.array([[it["deps_q"][a] == b for b in it["deps_h"]] for a in qs], np.float32).reshape(-1)
        Xs.append(X); ys.append(y); ws.append(np.full(len(y), 1.0 / len(y)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def sim_predict(m, it):
    nq, nh = it["sX"].shape[:2]
    if nh == 0:
        return np.zeros((nq, 0), np.float32)
    return m.predict(it["sX"].reshape(-1, it["sX"].shape[2]).astype(np.float32)).reshape(nq, nh)


def augment(it, s):
    """Votes from the learned same-set similarity -> extra gate and pair features."""
    nq = len(it["qv"]); nU = it["nU"]; nh = s.shape[1]
    hm = it["hmat_s"]
    g = np.full((nq, 6), -1, np.float32)
    p = np.full((nq, nU, 6), -1, np.float32)
    if nh:
        hfull = hm.sum(1) == nU
        st = np.where(~hfull[None, :], s, 0); sf = np.where(hfull[None, :], s, 0)
        g[:, 0] = s.max(1)
        g[:, 1] = st.max(1) if (~hfull).any() else -1
        g[:, 2] = sf.max(1) if hfull.any() else -1
        g[:, 3] = np.where(s.sum(1) > 0, sf.sum(1) / np.maximum(s.sum(1), 1e-9), -1)
        g[:, 4] = np.log10(st.sum(1) + 1e-6)
        g[:, 5] = g[:, 1] - g[:, 2]
        if (~hfull).any():
            hs = hm[~hfull]; ss = s[:, ~hfull]
            p[:, :, 0] = (ss @ hs) / np.maximum(ss.sum(1, keepdims=True), 1e-9)
            p[:, :, 1] = hs[ss.argmax(1)]
            p[:, :, 2] = ss.max(1)[:, None]
            w4 = ss ** 4
            p[:, :, 3] = (w4 @ hs) / np.maximum(w4.sum(1, keepdims=True), 1e-12)
            l1 = np.log(np.clip(1 - ss, 1e-6, 1))
            p[:, :, 4] = 1 - np.exp(l1 @ hs)
            p[:, :, 5] = 1 - np.exp(l1 @ (1 - hs))
    return g, p.reshape(nq * nU, 6)


def score_matrix(Z, M, nU):
    Zf = Z.astype(np.float32); Mf = M.astype(np.float32)
    inter = Mf @ Zf.T
    sp_ = Mf.sum(1)[:, None]; sd = Zf.sum(1)[None, :]
    un = sp_ + sd - inter
    j1 = np.where(un > 0, inter / np.maximum(un, 1e-9), 1.0)
    inter_c = (1 - Mf) @ (1 - Zf).T
    un_c = (nU - sp_) + (nU - sd) - inter_c
    j2 = np.where(un_c > 0, inter_c / np.maximum(un_c, 1e-9), 1.0)
    return np.sqrt(j1 * j2)


def best_strict_set(p, seed):
    """Expected-metric maximiser over the model's top-k sets (k < nU) under its Bernoulli posterior."""
    nU = len(p)
    rng = np.random.default_rng(seed)
    order = np.argsort(-p, kind="stable")
    M = np.zeros((nU, nU), bool)
    for k in range(1, nU):
        M[k, order[:k]] = True
    Z = rng.random((N_SAMPLES, nU)) < p[None, :]
    E = score_matrix(Z, M, nU).mean(1)
    b = int(E.argmax())
    return M[b], float(E[b])


def qseed(fid, v):
    return zlib.crc32(f"{fid}:{int(v)}".encode())


def true_score(pred, ymask):
    nU = len(ymask)
    P = set(np.flatnonzero(pred)); Dd = set(np.flatnonzero(ymask)); U = set(range(nU))
    def j(a, b):
        u = len(a | b)
        return 1.0 if u == 0 else len(a & b) / u
    return math.sqrt(j(P, Dd) * j(U - P, U - Dd))


def weighted(df):
    fam = {}
    for fm, gg in df.groupby("family"):
        if fm == "cnf_lifted":
            continue
        fam[fm] = gg.groupby("formula_id").score.mean().mean()
    return (float(np.mean(list(fam.values()))) if fam else float("nan")), fam


# =============================================================================== main
def main():
    pub = Path(sys.argv[1]); out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(pub / "formulas.csv")
    test = pd.read_csv(pub / "test.csv")
    hints_df = pd.read_csv(pub / "test_hints.csv")
    labels = pd.read_csv(pub / "train_labels.csv.gz")
    fam_of = dict(zip(meta.formula_id, meta.family))
    log(f"plan: {N_DRAWS} simulated draws/train formula, {N_FOLDS}-fold grouped OOF stacking, {N_SEEDS}-seed gate/pair ensembles, "
        f"LightGBM gate {PG['n_rounds']} / pair {PP['n_rounds']} / sim {PS['n_rounds']} rounds, {N_SAMPLES} MC samples")

    # ---- parse test formulas first and write a schema-valid placeholder (all universals)
    formulas = {}
    test_fids = sorted(test.formula_id.unique())
    for fid in test_fids:
        formulas[fid] = parse_formula(pub / "formulas" / f"{fid}.dqx.gz")
    def all_uni(fid):
        u = formulas[fid]["uni"]
        return " ".join(map(str, u.tolist())) if len(u) else "{}"
    placeholder = pd.DataFrame({"id": test.id, "deps": [all_uni(f) for f in test.formula_id]})
    placeholder.to_csv(out_path, index=False)
    log("placeholder written", len(placeholder))

    # ---- train episodes
    train_fids = meta[meta.split == "train"].formula_id.tolist()
    lab_g = dict(tuple(labels.groupby("formula_id")))
    items = []
    for fid in train_fids:
        f = parse_formula(pub / "formulas" / f"{fid}.dqx.gz")
        cache = formula_cache(f)
        g = lab_g[fid]
        nU = len(cache["uni"])
        uidx = {int(u): j for j, u in enumerate(cache["uni"])}
        var = g.variable.values.astype(np.int64)
        deps = [parse_deps(s) for s in g.deps.values]
        full = np.array([len(d) == nU for d in deps])
        for ds in range(N_DRAWS):
            rng = np.random.default_rng(zlib.crc32(f"{fid}_{ds}".encode()))
            n = len(var)
            keep = rng.random(n) < P_HINT
            if not keep.any():
                keep[rng.integers(n)] = True
            lost = np.flatnonzero(~keep)
            ls = lost[~full[lost]]; lf = lost[full[lost]]
            ns = min(MAX_STRICT_Q, len(ls))
            qs = rng.choice(ls, ns, replace=False) if ns else np.array([], np.int64)
            nf = min(len(lf), max(3, int(round(ns / 2))))
            qf = rng.choice(lf, nf, replace=False) if nf else np.array([], np.int64)
            qi = np.concatenate([qs, qf]).astype(np.int64)
            if len(qi) == 0:
                continue
            hints = {int(var[i]): deps[i] for i in np.flatnonzero(keep)}
            it = make_item(cache, fam_of[fid], fid, hints, var[qi], rng_seed=ds)
            ymat = np.zeros((len(qi), nU), np.float32)
            for a, i in enumerate(qi):
                for u in deps[i]:
                    ymat[a, uidx[u]] = 1
            it.update(draw=ds, qfull=full[qi], ymat=ymat, deps_q=[deps[i] for i in qi])
            items.append(it)
        del cache
    log(f"train episodes built: {len(items)}")
    GN = items[0]["gn"]; PN = items[0]["pn"]; SN = next(it["sn"] for it in items if it["sn"])
    cat_g = [GN.index("f_fam")]; cat_p = [PN.index("q_f_fam")]; cat_s = [SN.index("f_fam")]

    # ---- grouped folds (family + #universals; random_dqbf formulas individually)
    keyf = {r.formula_id: (r.formula_id if r.family == "random_dqbf" else f"{r.family}_{r.n_universal}") for r in meta.itertuples()}
    keys = sorted(set(keyf[f] for f in train_fids))
    perm = np.random.default_rng(SEED).permutation(len(keys))
    fold_of_key = {k: int(p % N_FOLDS) for k, p in zip(keys, perm)}
    fold = {f: fold_of_key[keyf[f]] for f in train_fids}

    # ---- stage S: same-set similarity, OOF on train
    S_oof = {}
    for k in range(N_FOLDS):
        tr = [it for it in items if fold[it["fid"]] != k]
        X, y, w = sim_train_rows(tr, np.random.default_rng(SEED + k))
        m = fit(PS, X, y, w, cat_s)
        for it in items:
            if fold[it["fid"]] == k:
                S_oof[id(it)] = sim_predict(m, it)
    X, y, w = sim_train_rows(items, np.random.default_rng(SEED + 99))
    sim_final = fit(PS, X, y, w, cat_s)
    del X, y, w
    log("stage S done")
    for it in items:
        ga, pa = augment(it, S_oof[id(it)])
        it["gXa"] = np.concatenate([it["gX"], ga], 1)
        it["pXa"] = np.concatenate([it["pX"], pa], 1)
        it["pstrict"] = np.repeat(~it["qfull"], it["nU"])
        it["py"] = it["ymat"].reshape(-1)
        del it["sX"]

    def train_gp(tr):
        gX = np.concatenate([it["gXa"] for it in tr]); gy = np.concatenate([it["qfull"] for it in tr]).astype(np.float32)
        gw = np.concatenate([np.full(len(it["qfull"]), 1.0 / len(it["qfull"])) for it in tr])
        gm = fit_ens(PG, gX, gy, gw, cat_g)
        del gX
        pX = np.concatenate([it["pXa"][it["pstrict"]] for it in tr]); py = np.concatenate([it["py"][it["pstrict"]] for it in tr])
        pw = np.concatenate([np.full(int(it["pstrict"].sum()), 1.0 / max(1, it["pstrict"].sum())) for it in tr])
        pm = fit_ens(PP, pX, py, pw, cat_p)
        return gm, pm

    # ---- stage G/P: OOF validation + decode-knob search
    oof_rows = []
    for k in range(N_FOLDS):
        tr = [it for it in items if fold[it["fid"]] != k]
        gm, pm = train_gp(tr)
        for it in items:
            if fold[it["fid"]] != k:
                continue
            pf = gm.predict(it["gXa"])
            pp = pm.predict(it["pXa"]).reshape(len(it["qv"]), it["nU"])
            for i in range(len(it["qv"])):
                mask, E = best_strict_set(pp[i], qseed(it["fid"], it["qv"][i]))
                yv = it["ymat"][i] > 0.5
                oof_rows.append(dict(family=it["family"], formula_id=f"{it['fid']}_{it['draw']}", pf=pf[i], E=E,
                                     s_full=1.0 if it["qfull"][i] else 0.0,
                                     s_strict=0.0 if it["qfull"][i] else true_score(mask, yv)))
    oof = pd.DataFrame(oof_rows)
    log(f"stage G/P OOF done ({len(oof)} queries)")
    grid = np.round(np.arange(-0.30, 0.301, 0.025), 3)
    res = []
    for b in grid:
        choose_full = oof.pf + b > (1 - oof.pf) * oof.E
        df = oof.assign(score=np.where(choose_full, oof.s_full, oof.s_strict))
        tot, fam = weighted(df)
        res.append((tot, -abs(b), b, fam))
    res.sort(key=lambda r: (r[0], r[1]), reverse=True)
    best_tot, _, FULL_BIAS, best_fam = res[0]
    base = [r for r in res if r[2] == 0.0][0]
    log(f"OOF validation (grouped {N_FOLDS}-fold, exact metric): bias=0 -> {base[0]:.4f}; searched full_bias={FULL_BIAS} -> {best_tot:.4f}")
    log("OOF per family:", {k: round(v, 4) for k, v in best_fam.items()})

    # ---- final gate / pair models on all train episodes
    gm, pm = train_gp(items)
    del items
    log("final models trained")

    # ---- test inference (per formula; each query uses only its formula, the formula's hint lines and itself)
    hints_by_f = {fid: {int(v): parse_deps(s) for v, s in zip(g.variable, g.deps)} for fid, g in hints_df.groupby("formula_id")}
    pred = {}
    for fid in test_fids:
        tq = test[test.formula_id == fid]
        cache = formula_cache(formulas[fid])
        uni = cache["uni"]; nU = len(uni)
        uset = set(int(u) for u in uni)
        hints = {v: frozenset(u for u in d if u in uset) for v, d in hints_by_f[fid].items()}
        it = make_item(cache, fam_of[fid], fid, hints, tq.variable.values.astype(np.int64), rng_seed=0)
        s = sim_predict(sim_final, it)
        ga, pa = augment(it, s)
        pf = gm.predict(np.concatenate([it["gX"], ga], 1))
        pp = pm.predict(np.concatenate([it["pX"], pa], 1)).reshape(len(tq), nU)
        for i, (qid, v) in enumerate(zip(tq.id.values, tq.variable.values)):
            mask, E = best_strict_set(pp[i], qseed(fid, v))
            if pf[i] + FULL_BIAS > (1 - pf[i]) * E:
                mask = np.ones(nU, bool)
            ids = uni[mask]
            pred[qid] = " ".join(map(str, ids.tolist())) if len(ids) else "{}"
        del cache
    log("test inference done")

    sub = pd.DataFrame({"id": test.id, "deps": [pred[i] for i in test.id]})
    sub.to_csv(out_path, index=False)
    nfull = np.mean([d == all_uni(f) for d, f in zip(sub.deps, test.formula_id)])
    log(f"wrote {out_path} rows={len(sub)} unique_ids={sub.id.nunique()} frac_predicted_full={nfull:.3f}")


if __name__ == "__main__":
    main()
