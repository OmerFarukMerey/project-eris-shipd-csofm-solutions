"""Vector Fragment Route Program Repair — self-contained solution (v2).

Approach: structured route decoding driven by learned models, all trained inside
this script on the provided train.csv only.

  0. outer_inner split model  P(margin-sorted component partition is correct) —
     supervised labels derived from train routes; picks the component partition
     for outer_inner rows (out-of-fold on train, full-train model on test).
  1. orientation model   P(shown point order == route direction) per fragment.
     Out-of-fold predictions on train feed oriented-geometry features downstream;
     a full-train model predicts test rows.
  2. pairwise comparator P(fragment a precedes fragment b) for same-component
     pairs, trained symmetrically (both directions) and ensembled
     (2x LightGBM + XGBoost). Out-of-fold predictions define a per-component
     max-likelihood total order (Held-Karp DP) whose ranks become features; the
     averaged bidirectional pair log-probabilities are also an additive term in
     the beam score.
  3. start model         P(fragment is the route start).
  4. successor policy    P(candidate is next | current fragment + decode state),
     trained by imitation of the true routes; ensembled (2x LightGBM + XGBoost).

Per-row preprocessing (pure per-row geometry, identical for train and test):
fragments are clustered into component_count spatial components (tiny built-in
KMeans on layout-appropriate features; model-chosen margin split for
outer_inner), a per-component hub (densest point) is estimated, and hub-relative
geometric features are computed under several distance metrics (isotropic,
anisotropic w-weighted, component-extent-normalized) so the models can learn the
per-layout ordering convention.

Decoding: beam search (width 32) over Hamiltonian paths scoring start +
successor + pairwise terms.

Test rows are used strictly per-row for transform + predict: no statistic,
vocabulary, or fitted state is derived from test data.
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
T_START = time.time()
TIME_SOFT = 3000.0  # seconds; degrade decode beyond this
SEED = 0


def elapsed():
    return time.time() - T_START


import json
import numpy as np
from collections import defaultdict

CANVAS = 128
TAGS = {"horizontal": 0, "vertical": 1, "sweep": 2, "fall": 3, "dot": 4, "bent": 5}
LAYOUTS = {"left_right": 0, "top_bottom": 1, "outer_inner": 2, "center_sides": 3, "diagonal": 4}


def _kmeans(X, k, iters=60, seeds=5):
    n = X.shape[0]
    best_in, best_lab = None, None
    rng = np.random.RandomState(0)
    for s in range(seeds):
        if s == 0:
            idx = [int(np.argmax(((X - X.mean(0)) ** 2).sum(1)))]
            while len(idx) < k:
                d = np.min(((X[:, None, :] - X[None, idx, :]) ** 2).sum(-1), axis=1)
                idx.append(int(np.argmax(d)))
            C = X[idx].copy()
        else:
            C = X[rng.choice(n, k, replace=False)].copy()
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best_in is None or inertia < best_in - 1e-9:
            best_in, best_lab = inertia, lab.copy()
    return np.asarray(best_lab, int)


def route_from_prog(prog):
    succ = dict(map(tuple, prog["links"]))
    order = [prog["start"]]
    while order[-1] in succ:
        order.append(succ[order[-1]])
    return order


# ---------------- base per-row preprocessing (clustering-independent) ----------------
def prep_base(cards_json, layout, k):
    cards = json.loads(cards_json)
    n = len(cards)
    R = {"n": n, "k": k, "layout": LAYOUTS[layout], "layout_name": layout,
         "names": [c["fragment"] for c in cards]}
    pts = [np.asarray(c["points"], float) for c in cards]
    R["pts"] = pts
    F = np.zeros((n, 26))
    tag_ids = np.zeros(n, int)
    anchors = np.zeros((n, 2)); firsts = np.zeros((n, 2)); lasts = np.zeros((n, 2)); cents = np.zeros((n, 2))
    margins = np.zeros(n)
    for i, c in enumerate(cards):
        p = pts[i]
        b = c["bbox"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        w, h = b[2] - b[0], b[3] - b[1]
        seg = np.diff(p, axis=0)
        sl = np.hypot(seg[:, 0], seg[:, 1])
        plen = sl.sum()
        imax = int(np.argmax(sl)) if len(sl) else 0
        dm = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        kk = max(2, len(p) // 2)
        ds = np.sort(dm, axis=1)[:, 1:kk + 1].mean(1)
        anc = p[np.argmin(ds)]
        anchors[i] = anc; firsts[i] = p[0]; lasts[i] = p[-1]; cents[i] = (cx, cy)
        turn = 0.0
        if len(seg) >= 2:
            a = np.arctan2(seg[:, 1], seg[:, 0])
            da = np.diff(a); da = (da + np.pi) % (2 * np.pi) - np.pi
            turn = float(da.sum())
        margin = min(b[0], b[1], CANVAS - b[2], CANVAS - b[3])
        margins[i] = margin
        reach = max(abs(b[0] - 64), abs(b[2] - 64), abs(b[1] - 64), abs(b[3] - 64))
        F[i] = [cx, cy, w, h, np.hypot(w, h), margin, reach, plen, len(p),
                np.hypot(*(p[-1] - p[0])), turn, abs(turn),
                p[0][0], p[0][1], p[-1][0], p[-1][1],
                sl[imax] if len(sl) else 0, (imax + 1) / max(len(sl), 1),
                np.hypot(*(p[0] - anc)), np.hypot(*(p[-1] - anc)),
                (np.hypot(*(p - p[0]).T) < 12).mean(), (np.hypot(*(p - p[-1]).T) < 12).mean(),
                anc[0], anc[1], p[-1][0] - p[0][0], p[-1][1] - p[0][1]]
        tag_ids[i] = TAGS[c["shape_tag"]]
    R["F"] = F; R["tag"] = tag_ids; R["anchors"] = anchors
    R["firsts"] = firsts; R["lasts"] = lasts; R["cents"] = cents; R["margins"] = margins
    Dmin = np.zeros((n, n)); Danc = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dmm = np.hypot(pts[i][:, None, 0] - pts[j][None, :, 0], pts[i][:, None, 1] - pts[j][None, :, 1])
            Dmin[i, j] = Dmin[j, i] = dmm.min()
            Danc[i, j] = Danc[j, i] = np.hypot(*(anchors[i] - anchors[j]))
    R["Dmin"] = Dmin; R["Danc"] = Danc
    R["psign"] = np.ones(n)  # predicted P(+); to be filled after orientation model
    return R


def primary_labels(R):
    if "lab_override" in R:
        return R["lab_override"]
    n, k = R["n"], R["k"]
    if k >= n:
        return np.arange(n)
    lay = R["layout_name"]
    if lay == "outer_inner":
        return _kmeans(R["margins"][:, None].copy(), k)
    if lay == "center_sides":
        X = np.hstack([R["cents"][:, :1], R["cents"][:, 1:] * 0.3])
        return _kmeans(X, k)
    return _kmeans(R["cents"].copy(), k)


def candidate_labelings(R):
    """primary + alternates (for outer_inner: margin-sorted split shifts)."""
    if "lab_override" in R:
        return [R["lab_override"]]
    lab0 = primary_labels(R)
    out = [lab0]
    n, k = R["n"], R["k"]
    if R["layout_name"] == "outer_inner" and k == 2 and n > k:
        srt = np.argsort(R["margins"])
        lab_sorted = lab0[srt]
        change = np.where(np.diff(lab_sorted) != 0)[0]
        km_split = int(change[0]) + 1 if len(change) else n // 2
        for split in range(max(1, km_split - 2), min(n - 1, km_split + 2) + 1):
            lab = np.zeros(n, int)
            lab[srt[split:]] = 1
            if not np.array_equal(lab, lab0) and not np.array_equal(1 - lab, lab0):
                out.append(lab)
    return out


class ProbaEnsemble:
    """Average predict_proba over several fitted binary classifiers."""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        import numpy as _np
        ps = [m.predict_proba(X)[:, 1] for m in self.models]
        p = _np.mean(ps, axis=0)
        return _np.stack([1 - p, p], axis=1)


# ---------------- outer_inner split model ----------------
def rr_deal_sim(seq):
    """Simulate round-robin dealing over the label sequence's first-appearance cycle.
    Used ONLY to derive supervised training labels for the split classifier from
    train routes; never applied to test rows."""
    first_order = []
    for l in seq:
        if l not in first_order:
            first_order.append(l)
    remaining = {}
    for l in seq:
        remaining[l] = remaining.get(l, 0) + 1
    rr = []; idx = 0; cyc = first_order[:]
    while any(v > 0 for v in remaining.values()):
        c = cyc[idx % len(cyc)]
        if remaining[c] > 0:
            rr.append(c); remaining[c] -= 1
        else:
            pos = idx % len(cyc); cyc = [x for x in cyc if x != c]; idx = pos; continue
        idx += 1
    return rr


def _hub_of(pts_list):
    allp = np.vstack(pts_list)
    d = np.hypot(allp[:, None, 0] - allp[None, :, 0], allp[:, None, 1] - allp[None, :, 1])
    return allp[(d < 7).sum(1).argmax()]


def split_feats(R, srt, s):
    n = R["n"]
    margins = R["margins"][srt]
    g0 = srt[:s]; g1 = srt[s:]
    reach = R["F"][:, 6]
    h0 = _hub_of([R["pts"][j] for j in g0]); h1 = _hub_of([R["pts"][j] for j in g1])
    return [
        n, s, n - s, abs(2 * s - n) / n,
        margins[s] - margins[s - 1],
        margins[:s].mean() - margins[s:].mean(),
        margins[s - 1], margins[s] if s < n else 128.0,
        reach[g0].max() - reach[g1].max(),
        reach[g0].mean() - reach[g1].mean(),
        np.hypot(*(h0 - h1)),
        R["F"][g0, 4].mean() - R["F"][g1, 4].mean(),
        R["F"][g0, 7].mean() - R["F"][g1, 7].mean(),
    ]


def split_candidates(R):
    srt = np.argsort(R["margins"])
    n = R["n"]
    out = []
    for s in range(1, n):
        lab = np.zeros(n, int); lab[srt[s:]] = 1
        out.append((s, lab, split_feats(R, srt, s)))
    return out


def gen_split_xy(Rs, progs):
    X, y = [], []
    for R, prog in zip(Rs, progs):
        if R["layout_name"] != "outer_inner" or R["k"] != 2 or R["n"] <= 2:
            continue
        idx = {f: j for j, f in enumerate(R["names"])}
        order = [idx[f] for f in route_from_prog(prog)]
        for s, lab, feats in split_candidates(R):
            seq = [lab[j] for j in order]
            X.append(feats); y.append(1 if seq == rr_deal_sim(seq) else 0)
    return np.array(X), np.array(y)


def apply_split_model(R, m_split):
    """Set R["lab_override"] to the classifier's best split (outer_inner k=2 only)."""
    if R["layout_name"] != "outer_inner" or R["k"] != 2 or R["n"] <= 2:
        return R
    cands = split_candidates(R)
    pv = m_split.predict_proba(np.array([f for _, _, f in cands]))[:, 1]
    R["lab_override"] = cands[int(np.argmax(pv))][1]
    return R


# ---------------- clustering-dependent view ----------------
def with_clusters(R, lab):
    V = dict(R)
    n, k = R["n"], R["k"]
    lab = np.asarray(lab, int)
    V["lab"] = lab
    pts = R["pts"]
    hubs = np.zeros((k, 2)); csize = np.zeros(k); ccent = np.zeros((k, 2))
    cmargin = np.zeros(k); creach = np.zeros(k)
    for l in range(k):
        mem = [i for i in range(n) if lab[i] == l]
        if not mem:
            hubs[l] = (64, 64); ccent[l] = (64, 64); continue
        allp = np.vstack([pts[i] for i in mem])
        dm = np.hypot(allp[:, None, 0] - allp[None, :, 0], allp[:, None, 1] - allp[None, :, 1])
        hubs[l] = allp[(dm < 7).sum(1).argmax()]
        csize[l] = len(mem)
        ccent[l] = R["cents"][mem].mean(0)
        cmargin[l] = R["F"][mem, 5].min()
        creach[l] = R["F"][mem, 6].max()
    cw = np.ones(k); ch = np.ones(k); cvr = np.ones(k)
    for l in range(k):
        mem = [i for i in range(n) if lab[i] == l]
        if not mem: continue
        allp = np.vstack([pts[i] for i in mem])
        cw[l] = max(allp[:, 0].max() - allp[:, 0].min(), 1e-6)
        ch[l] = max(allp[:, 1].max() - allp[:, 1].min(), 1e-6)
        cvr[l] = (allp[:, 1].std() + 1e-6) / (allp[:, 0].std() + 1e-6)
    V["cw"] = cw; V["ch"] = ch; V["cvr"] = cvr
    V["hubs"] = hubs; V["csize"] = csize; V["ccent"] = ccent

    def rank_of(vals):
        order = np.argsort(vals)
        rk = np.empty(k); rk[order] = np.arange(k)
        return rk
    V["crank_x"] = rank_of(ccent[:, 0]); V["crank_y"] = rank_of(ccent[:, 1])
    V["crank_xy"] = rank_of(ccent[:, 0] + ccent[:, 1]); V["crank_xmy"] = rank_of(ccent[:, 0] - ccent[:, 1])
    V["crank_center"] = rank_of(np.abs(ccent[:, 0] - 64) + np.abs(ccent[:, 1] - 64))
    V["crank_margin"] = rank_of(-cmargin); V["crank_reach"] = rank_of(-creach)

    # refined hubs: density peak over oriented tail points (post-max-segment under
    # predicted orientation) of cluster members — excludes distal heads and jumps
    psign = R["psign"]
    hubs2 = hubs.copy()
    for l in range(k):
        mem = [i for i in range(n) if lab[i] == l]
        if not mem: continue
        tails = []
        for i in mem:
            p = pts[i] if psign[i] >= 0.5 else pts[i][::-1]
            seg = np.diff(p, axis=0)
            sl = np.hypot(seg[:, 0], seg[:, 1])
            imax = int(np.argmax(sl)) if len(sl) else 0
            tails.append(p[imax + 1:])
        allt = np.vstack([t for t in tails if len(t)]) if any(len(t) for t in tails) else None
        if allt is not None and len(allt) >= 3:
            dm = np.hypot(allt[:, None, 0] - allt[None, :, 0], allt[:, None, 1] - allt[None, :, 1])
            hubs2[l] = allt[(dm < 7).sum(1).argmax()]
    V["hubs2"] = hubs2

    # hub-relative block, incl. predicted-orientation features
    HB = np.zeros((n, 26))
    for i in range(n):
        hub = hubs[lab[i]]
        p = pts[i]
        d_f = np.hypot(*(R["firsts"][i] - hub)); d_l = np.hypot(*(R["lasts"][i] - hub))
        d_anc = np.hypot(*(R["anchors"][i] - hub))
        distal = max(d_f, d_l); prox = min(d_f, d_l)
        d_all = np.hypot(*(p - hub).T)
        dp = R["firsts"][i] if d_f >= d_l else R["lasts"][i]
        ang = np.arctan2(dp[1] - hub[1], dp[0] - hub[0])
        # predicted-orientation distal = first point under predicted sign
        pfirst = R["firsts"][i] if psign[i] >= 0.5 else R["lasts"][i]
        plast = R["lasts"][i] if psign[i] >= 0.5 else R["firsts"][i]
        d_pf = np.hypot(*(pfirst - hub)); d_pl = np.hypot(*(plast - hub))
        pang = np.arctan2(pfirst[1] - hub[1], pfirst[0] - hub[0])
        hub2 = hubs2[lab[i]]
        d_pf2 = np.hypot(*(pfirst - hub2)); d_pl2 = np.hypot(*(plast - hub2))
        pang2 = np.arctan2(pfirst[1] - hub2[1], pfirst[0] - hub2[0])
        # oriented spoke landing point distance to refined hub
        po = p if psign[i] >= 0.5 else p[::-1]
        seg_o = np.diff(po, axis=0); sl_o = np.hypot(seg_o[:, 0], seg_o[:, 1])
        io = int(np.argmax(sl_o)) if len(sl_o) else 0
        d_land2 = np.hypot(*(po[min(io + 1, len(po) - 1)] - hub2))
        dxh = pfirst[0] - hub[0]; dyh = pfirst[1] - hub[1]
        l_ = lab[i]
        m_aspect = np.sqrt((dxh / V["cw"][l_]) ** 2 + (dyh / V["ch"][l_]) ** 2)
        m_cov = np.sqrt((dxh * V["cvr"][l_]) ** 2 + dyh ** 2)
        m_w015 = np.sqrt(0.15 * dxh ** 2 + dyh ** 2)
        m_w03 = np.sqrt(0.3 * dxh ** 2 + dyh ** 2)
        m_w2 = np.sqrt(2.0 * dxh ** 2 + dyh ** 2)
        m_l1 = abs(dxh) + abs(dyh)
        HB[i] = [d_f, d_l, distal, prox, d_anc, d_all.max(), d_all.min(), d_all.mean(),
                 (d_all < 10).mean(), d_f - d_l, ang, np.hypot(*(dp - hub)),
                 d_pf, d_pl, pang, d_pf - d_pl,
                 d_pf2, d_pl2, pang2, d_land2,
                 m_aspect, m_cov, m_w015, m_w03, m_w2, m_l1]
    V["HB"] = HB
    # within-cluster ranks among all members: distal(max-end), pred-distal(refined hub), spoke, plen, diag
    WR = np.zeros((n, 7))
    for l in range(k):
        mem = [i for i in range(n) if lab[i] == l]
        if not mem: continue
        m = len(mem)
        for col, vals in enumerate([HB[mem, 2], HB[mem, 16], R["F"][mem, 16], R["F"][mem, 7], R["F"][mem, 4],
                                    HB[mem, 20], HB[mem, 21]]):
            o = np.argsort(vals)
            rk = np.empty(m); rk[o] = np.arange(m)
            for jj, i in enumerate(mem):
                WR[i, col] = rk[jj] / max(m - 1, 1)
    V["WR"] = WR
    return V


# ---------------- orientation features (clustering-independent + primary-cluster hub) ----------------
def orient_feats(V, i):
    p = V["pts"][i]
    hub = V["hubs"][V["lab"][i]]
    d_f = np.hypot(*(p[0] - hub)); d_l = np.hypot(*(p[-1] - hub))
    seg = np.diff(p, axis=0); sl = np.hypot(seg[:, 0], seg[:, 1])
    imax = int(np.argmax(sl)) if len(sl) else 0
    anc = V["anchors"][i]
    d_f_anc = np.hypot(*(p[0] - anc)); d_l_anc = np.hypot(*(p[-1] - anc))
    prefix_len = sl[:imax].sum() if len(sl) else 0
    suffix_len = sl[imax + 1:].sum() if len(sl) else 0
    near_f = (np.hypot(*(p - p[0]).T) < 12).mean(); near_l = (np.hypot(*(p - p[-1]).T) < 12).mean()
    turn = 0.0
    if len(seg) >= 2:
        a = np.arctan2(seg[:, 1], seg[:, 0])
        da = np.diff(a); da = (da + np.pi) % (2 * np.pi) - np.pi
        turn = float(da.sum())
    return np.array([
        V["layout"], V["tag"][i], len(p),
        p[0][0], p[0][1], p[-1][0], p[-1][1],
        p[-1][0] - p[0][0], p[-1][1] - p[0][1],
        d_f, d_l, d_f - d_l, d_f_anc, d_l_anc, d_f_anc - d_l_anc,
        (imax + 1) / max(len(sl), 1), sl[imax] if len(sl) else 0,
        prefix_len, suffix_len, prefix_len - suffix_len,
        near_f, near_l, near_f - near_l, turn,
        V["HB"][i, 8], V["F"][i, 5], V["F"][i, 6],
        hub[0], hub[1], p[0][0] - hub[0], p[0][1] - hub[1], p[-1][0] - hub[0], p[-1][1] - hub[1],
    ])


# ---------------- start features ----------------
def start_feats(V, i):
    l = V["lab"][i]
    return np.concatenate([
        [V["layout"], V["n"], V["k"]],
        V["F"][i], [V["tag"][i]],
        V["HB"][i, [2, 3, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]],
        V["WR"][i],
        [V["csize"][l], V["crank_x"][l], V["crank_y"][l], V["crank_xy"][l], V["crank_xmy"][l],
         V["crank_center"][l], V["crank_margin"][l], V["crank_reach"][l]],
        [V["dpr"][i], float(V["dp_pred"][i] == -1)],
    ])


# ---------------- pairwise comparator features (same-cluster pairs) ----------------
def pair_feats(V, a, b):
    ha, hb_ = V["HB"][a], V["HB"][b]
    fa, fb = V["F"][a], V["F"][b]
    dang = (ha[14] - hb_[14] + np.pi) % (2 * np.pi) - np.pi
    dang2 = (ha[18] - hb_[18] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([
        [V["layout"], V["n"], V["k"], V["csize"][V["lab"][a]]],
        ha[[2, 3, 5, 7, 8, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25]],
        hb_[[2, 3, 5, 7, 8, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25]],
        ha[[2, 3, 5, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25]] - hb_[[2, 3, 5, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25]],
        [dang, abs(dang), dang2, abs(dang2)],
        [fa[16] - fb[16], fa[7] - fb[7], fa[4] - fb[4], fa[9] - fb[9]],
        [fa[16], fb[16], fa[7], fb[7], fa[9], fb[9], fa[8], fb[8]],
        [V["tag"][a], V["tag"][b]],
        [V["Dmin"][a, b], V["Danc"][a, b]],
        [V["cents"][a, 0] - V["cents"][b, 0], V["cents"][a, 1] - V["cents"][b, 1]],
        [V["WR"][a, 0] - V["WR"][b, 0], V["WR"][a, 1] - V["WR"][b, 1], V["WR"][a, 2] - V["WR"][b, 2],
         V["WR"][a, 5] - V["WR"][b, 5], V["WR"][a, 6] - V["WR"][b, 6]],
    ])


# ---------------- pairwise-DP order (max-likelihood total order per cluster) ----------------
def dp_total_order(members, lp):
    """members: list of frag indices; lp[a][b] = log P(a before b) (global index).
    Returns list ordering members to maximize sum of pairwise logprobs. Held-Karp."""
    m = len(members)
    if m <= 1:
        return list(members)
    # local logprob matrix
    L = np.zeros((m, m))
    for x in range(m):
        for y in range(m):
            if x != y:
                L[x, y] = lp[members[x]][members[y]]
    full = 1 << m
    dp = np.full(full, -1e18); dp[0] = 0.0
    par = np.full(full, -1, dtype=np.int8)
    for S in range(full):
        if dp[S] <= -1e17: continue
        for j in range(m):
            if S & (1 << j): continue
            # score of appending j after set S: all of S before j
            add = 0.0
            for u in range(m):
                if S & (1 << u):
                    add += L[u, j]
            nS = S | (1 << j)
            v = dp[S] + add
            if v > dp[nS]:
                dp[nS] = v; par[nS] = j
    order = []
    S = full - 1
    while S:
        j = int(par[S])
        order.append(members[j])
        S &= ~(1 << j)
    return order[::-1]


def add_dp_ranks(V, m_pair):
    """Compute per-cluster max-likelihood total order from the pairwise comparator
    and attach normalized ranks. V['dpr'][i] = rank of i within its cluster order."""
    n, k = V["n"], V["k"]
    lp = np.zeros((n, n))
    prs, idxs = [], []
    for a in range(n):
        for b in range(a + 1, n):
            if V["lab"][a] == V["lab"][b]:
                prs.append(pair_feats(V, a, b)); prs.append(pair_feats(V, b, a))
                idxs.append((a, b))
    if prs:
        raw = m_pair.predict_proba(np.array(prs))[:, 1]
        for t, (a, b) in enumerate(idxs):
            p = np.clip((raw[2 * t] + 1 - raw[2 * t + 1]) / 2, 1e-6, 1 - 1e-6)
            lp[a, b] = np.log(p); lp[b, a] = np.log(1 - p)
    dpr = np.zeros(n); dpo = -np.ones(n, int)  # dpo[i] = predecessor of i in cluster dp order
    for l in range(k):
        mem = [i for i in range(n) if V["lab"][i] == l]
        if not mem: continue
        order = dp_total_order(mem, lp)
        for r, i in enumerate(order):
            dpr[i] = r / max(len(mem) - 1, 1)
            dpo[i] = order[r - 1] if r > 0 else -1
    V["dpr"] = dpr; V["dp_pred"] = dpo
    V["pair_lp"] = lp
    return V


# ---------------- successor policy features ----------------
def policy_feats(V, cur, cand, taken_cnt, last_in_cluster, t, since_visit, remaining, last2_in_cluster):
    k = V["k"]; n = V["n"]
    lc = V["lab"][cand]; lu = V["lab"][cur]
    rem_c = V["csize"][lc] - taken_cnt[lc]
    rem_u = V["csize"][lu] - taken_cnt[lu]
    n_rem_clusters = sum(1 for l in range(k) if V["csize"][l] - taken_cnt[l] > 0)
    sv = since_visit[lc]
    lrv = -1; best = -2
    for l in range(k):
        if V["csize"][l] - taken_cnt[l] > 0:
            s = since_visit[l] if since_visit[l] >= 0 else 10 ** 6
            if s > best:
                best = s; lrv = l
    peers = [j for j in remaining if V["lab"][j] == lc and j != cand]
    cd = V["HB"][cand, 16]; ca = V["HB"][cand, 18]; cs = V["F"][cand, 16]
    cda = V["HB"][cand, 20]; cdc = V["HB"][cand, 21]
    if peers:
        pd_ = np.array([V["HB"][j, 16] for j in peers])
        pa = np.array([V["HB"][j, 18] for j in peers])
        ps = np.array([V["F"][j, 16] for j in peers])
        n_smaller_d = float((pd_ < cd).sum()); n_smaller_s = float((ps < cs).sum())
        is_min_d = float((pd_ >= cd).all()); is_min_s = float((ps >= cs).all())
        pda = np.array([V["HB"][j, 20] for j in peers]); pdc = np.array([V["HB"][j, 21] for j in peers])
        n_smaller_a = float((pda < cda).sum()); is_min_a = float((pda >= cda).all())
        n_smaller_c = float((pdc < cdc).sum()); is_min_c = float((pdc >= cdc).all())
        dang = np.abs((pa - ca + np.pi) % (2 * np.pi) - np.pi)
        n_similar_ang = float((dang < 0.20).sum())
        n_simang_smaller = float(((dang < 0.20) & (pd_ < cd)).sum())
        min_gap_d = float(np.min(np.abs(pd_ - cd)))
    else:
        n_smaller_d = n_smaller_s = 0.0; is_min_d = is_min_s = 1.0
        n_smaller_a = n_smaller_c = 0.0; is_min_a = is_min_c = 1.0
        n_similar_ang = n_simang_smaller = 0.0; min_gap_d = np.nan
    lic = last_in_cluster[lc]
    if lic >= 0:
        la = V["HB"][lic, 18]
        dang_lic = abs((ca - la + np.pi) % (2 * np.pi) - np.pi)
        d_lic = [V["HB"][cand, 16] - V["HB"][lic, 16],
                 V["HB"][cand, 16] / max(V["HB"][lic, 16], 1e-6),
                 V["HB"][cand, 20] - V["HB"][lic, 20],
                 V["HB"][cand, 21] - V["HB"][lic, 21],
                 V["F"][cand, 16] - V["F"][lic, 16],
                 V["F"][cand, 7] - V["F"][lic, 7],
                 V["Dmin"][cand, lic], V["Danc"][cand, lic], dang_lic,
                 V["cents"][cand, 0] - V["cents"][lic, 0],
                 V["cents"][cand, 1] - V["cents"][lic, 1]]
    else:
        d_lic = [np.nan] * 11
    l2 = last2_in_cluster[lc]
    if l2 >= 0 and lic >= 0:
        d_l2 = [V["HB"][cand, 16] - V["HB"][l2, 16],
                (V["HB"][cand, 16] - V["HB"][lic, 16]) - (V["HB"][lic, 16] - V["HB"][l2, 16]),
                V["Dmin"][cand, l2]]
    else:
        d_l2 = [np.nan] * 3
    # pairwise-DP order features
    dpr_c = V["dpr"][cand]
    if peers:
        dp_is_next = float(all(V["dpr"][j] > dpr_c for j in peers))
    else:
        dp_is_next = 1.0
    dp_pred_ok = float(V["dp_pred"][cand] == lic) if lic >= 0 else float(V["dp_pred"][cand] == -1)
    dp_delta_lic = (dpr_c - V["dpr"][lic]) if lic >= 0 else np.nan
    dp_feats = [dpr_c, dp_is_next, dp_pred_ok, dp_delta_lic]
    return np.concatenate([
        [V["layout"], n, k, t / n],
        V["F"][cand], [V["tag"][cand]],
        V["HB"][cand, [2, 3, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], V["WR"][cand],
        [V["csize"][lc], V["crank_x"][lc], V["crank_y"][lc], V["crank_xy"][lc], V["crank_xmy"][lc],
         V["crank_center"][lc], V["crank_margin"][lc], V["crank_reach"][lc]],
        [float(lc == lu), rem_c, rem_u, n_rem_clusters,
         float(lc == lrv), sv if sv >= 0 else np.nan, taken_cnt[lc]],
        [n_smaller_d, n_smaller_s, is_min_d, is_min_s, n_similar_ang, n_simang_smaller,
         min_gap_d, len(peers), n_smaller_a, is_min_a, n_smaller_c, is_min_c],
        [V["Dmin"][cand, cur], V["Danc"][cand, cur],
         V["cents"][cand, 0] - V["cents"][cur, 0], V["cents"][cand, 1] - V["cents"][cur, 1],
         V["ccent"][lc, 0] - V["ccent"][lu, 0], V["ccent"][lc, 1] - V["ccent"][lu, 1]],
        d_lic, d_l2, dp_feats,
    ])


# ---------------- training sample generation ----------------
def gen_pair(views, progs):
    Xq, yq = [], []
    for V, prog in zip(views, progs):
        idx = {f: i for i, f in enumerate(V["names"])}
        order = [idx[f] for f in route_from_prog(prog)]
        pos = {j: t for t, j in enumerate(order)}
        n = V["n"]; k = V["k"]
        for l in range(k):
            mem = [i for i in range(n) if V["lab"][i] == l]
            for ai in range(len(mem)):
                for bi in range(ai + 1, len(mem)):
                    a, b = mem[ai], mem[bi]
                    Xq.append(pair_feats(V, a, b)); yq.append(1 if pos[a] < pos[b] else 0)
                    Xq.append(pair_feats(V, b, a)); yq.append(1 if pos[b] < pos[a] else 0)
    return np.array(Xq), np.array(yq)


def gen_training(views, progs):
    """start + policy samples; views must already have dp ranks attached."""
    Xs, ys = [], []
    Xp, yp = [], []
    for V, prog in zip(views, progs):
        names = V["names"]
        idx = {f: i for i, f in enumerate(names)}
        order = [idx[f] for f in route_from_prog(prog)]
        n = V["n"]; k = V["k"]
        for i in range(n):
            Xs.append(start_feats(V, i)); ys.append(1 if i == order[0] else 0)
        taken_cnt = np.zeros(k); lic = -np.ones(k, int); l2c = -np.ones(k, int); sv = -np.ones(k, int)
        remaining = set(range(n))
        cur = order[0]
        lcur = V["lab"][cur]
        taken_cnt[lcur] += 1; lic[lcur] = cur; sv[lcur] = 0
        remaining.discard(cur)
        for t in range(1, n):
            true_next = order[t]
            for cand in remaining:
                Xp.append(policy_feats(V, cur, cand, taken_cnt, lic, t, sv, remaining, l2c))
                yp.append(1 if cand == true_next else 0)
            for l in range(k):
                if sv[l] >= 0: sv[l] += 1
            lu = V["lab"][true_next]
            taken_cnt[lu] += 1; l2c[lu] = lic[lu]; lic[lu] = true_next; sv[lu] = 0
            remaining.discard(true_next)
            cur = true_next
    return (np.array(Xs), np.array(ys)), (np.array(Xp), np.array(yp))


# ---------------- beam decode ----------------
def decode_view(V, m_start, m_policy, beam_width=24, alpha=0.6):
    n = V["n"]; k = V["k"]
    xs = np.array([start_feats(V, i) for i in range(n)])
    s_scores = np.clip(m_start.predict_proba(xs)[:, 1], 1e-9, 1)
    slog = np.log(s_scores) - np.log(s_scores.sum())
    pair_lp = V["pair_lp"]
    order0 = np.argsort(-slog)[:beam_width]
    beams = []
    for i in order0:
        taken_cnt = np.zeros(k); lic = -np.ones(k, int); l2c = -np.ones(k, int); sv = -np.ones(k, int)
        lcur = V["lab"][i]
        taken_cnt[lcur] += 1; lic[lcur] = i; sv[lcur] = 0
        beams.append((float(slog[i]), [i], frozenset(range(n)) - {i}, taken_cnt, lic, sv, l2c, float(slog[i])))
    for t in range(1, n):
        cand_rows = []; meta = []
        for bi, (sc, seq, rem, taken_cnt, lic, sv, l2c, snp) in enumerate(beams):
            for cand in rem:
                cand_rows.append(policy_feats(V, seq[-1], cand, taken_cnt, lic, t, sv, rem, l2c))
                meta.append((bi, cand))
        pr = np.clip(m_policy.predict_proba(np.array(cand_rows))[:, 1], 1e-9, 1)
        sums = defaultdict(float)
        for (bi, cand), p in zip(meta, pr):
            sums[bi] += p
        scored = []
        for (bi, cand), p in zip(meta, pr):
            sc = beams[bi][0]; seq = beams[bi][1]
            lp_step = np.log(p / sums[bi])
            add = alpha * sum(pair_lp[u, cand] for u in seq if V["lab"][u] == V["lab"][cand])
            scored.append((sc + lp_step + add, lp_step, bi, cand))
        scored.sort(key=lambda x: -x[0])
        new_beams = []
        seen = set()
        for scv, lp_step, bi, cand in scored:
            if (bi, cand) in seen: continue
            seen.add((bi, cand))
            sc, seq, rem, taken_cnt, lic, sv, l2c, snp = beams[bi]
            tc = taken_cnt.copy(); li = lic.copy(); s2 = sv.copy(); l22 = l2c.copy()
            for l in range(k):
                if s2[l] >= 0: s2[l] += 1
            lu = V["lab"][cand]
            tc[lu] += 1; l22[lu] = li[lu]; li[lu] = cand; s2[lu] = 0
            new_beams.append((scv, seq + [cand], rem - {cand}, tc, li, s2, l22, snp + lp_step))
            if len(new_beams) >= beam_width: break
        beams = new_beams
    best = max(beams, key=lambda b: b[0])
    return best[0], best[1], best[7]


def decode_row(R, m_start, m_policy, m_pair, m_orient, beam_width=24, alpha=0.6):
    """R must already have psign filled. Tries candidate labelings, keeps best by score."""
    best_score, best_seq, best_V = -1e18, None, None
    for lab in candidate_labelings(R):
        V = add_dp_ranks(with_clusters(R, lab), m_pair)
        sc, seq, sc_nopair = decode_view(V, m_start, m_policy, beam_width, alpha)
        if sc_nopair > best_score:
            best_score, best_seq, best_V = sc_nopair, seq, V
    n = R["n"]; names = R["names"]
    xo = np.array([orient_feats(best_V, i) for i in range(n)])
    po = m_orient.predict_proba(xo)[:, 1]
    signs = ["+" if p >= 0.5 else "-" for p in po]
    return {
        "start": names[best_seq[0]],
        "links": [[names[a], names[b]] for a, b in zip(best_seq, best_seq[1:])],
        "orientations": {names[i]: signs[i] for i in range(n)},
    }




def fallback_prog(names):
    return {
        "start": names[0],
        "links": [[names[a], names[a + 1]] for a in range(len(names) - 1)],
        "orientations": {f: "+" for f in names},
    }


def main():
    import xgboost as xgb
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    print(f"[{elapsed():.0f}s] loaded train {train.shape} test {test.shape}", flush=True)

    tr_R, tr_prog = [], []
    for i in range(len(train)):
        row = train.iloc[i]
        tr_R.append(prep_base(row.fragment_cards, row.layout_hint, row.component_count))
        tr_prog.append(json.loads(row.target_program))
    print(f"[{elapsed():.0f}s] train prep done", flush=True)

    params = dict(objective="binary", learning_rate=0.04, num_leaves=63, min_child_samples=20,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, verbose=-1,
                  n_estimators=900, random_state=SEED, n_jobs=-1)
    all_idx = list(range(len(tr_R)))
    folds = [all_idx[f::4] for f in range(4)]

    # ---- Stage 0: outer_inner split model (OOF partition choice on train; full model for test) ----
    split_params = dict(objective="binary", n_estimators=500, learning_rate=0.05,
                        num_leaves=31, verbose=-1, random_state=SEED, n_jobs=-1)
    for f in range(4):
        tr_f = [i for ff in range(4) if ff != f for i in folds[ff]]
        Xsp, ysp = gen_split_xy([tr_R[i] for i in tr_f], [tr_prog[i] for i in tr_f])
        m = lgb.LGBMClassifier(**split_params).fit(Xsp, ysp)
        for i in folds[f]:
            apply_split_model(tr_R[i], m)
    Xsp, ysp = gen_split_xy(tr_R, tr_prog)
    m_split = lgb.LGBMClassifier(**split_params).fit(Xsp, ysp)
    print(f"[{elapsed():.0f}s] split stage done", flush=True)

    # ---- Stage 1: orientation model (OOF on train rows; full model for test) ----
    views = [with_clusters(R, primary_labels(R)) for R in tr_R]

    def orient_xy(indices):
        X, y, owners = [], [], []
        for i in indices:
            V = views[i]
            for j in range(V["n"]):
                X.append(orient_feats(V, j))
                y.append(1 if tr_prog[i]["orientations"][V["names"][j]] == "+" else 0)
                owners.append((i, j))
        return np.array(X), np.array(y), owners

    for f in range(4):
        tr_f = [i for ff in range(4) if ff != f for i in folds[ff]]
        Xf, yf, _ = orient_xy(tr_f)
        m = lgb.LGBMClassifier(**params).fit(Xf, yf)
        Xv, _, owners = orient_xy(folds[f])
        pv = m.predict_proba(Xv)[:, 1]
        for (i, j), p in zip(owners, pv):
            tr_R[i]["psign"][j] = p
        print(f"[{elapsed():.0f}s] orient fold {f} done", flush=True)
    Xall, yall, _ = orient_xy(all_idx)
    m_orient = lgb.LGBMClassifier(**params).fit(Xall, yall)
    print(f"[{elapsed():.0f}s] orientation stage done", flush=True)

    # rebuild views now that psign is filled
    views = [with_clusters(R, primary_labels(R)) for R in tr_R]

    # ---- Stage 2: pairwise comparator (OOF dp-ranks on train; ensemble for decode) ----
    pair_params = {**params, "n_estimators": 1200}
    for f in range(4):
        tr_f = [i for ff in range(4) if ff != f for i in folds[ff]]
        Xq, yq = gen_pair([views[i] for i in tr_f], [tr_prog[i] for i in tr_f])
        m = lgb.LGBMClassifier(**pair_params).fit(Xq, yq)
        for i in folds[f]:
            add_dp_ranks(views[i], m)
        print(f"[{elapsed():.0f}s] pair fold {f} done", flush=True)
    Xq, yq = gen_pair(views, tr_prog)
    m_pair = ProbaEnsemble([
        lgb.LGBMClassifier(**pair_params).fit(Xq, yq),
        lgb.LGBMClassifier(**{**pair_params, "random_state": 1, "feature_fraction": 0.8}).fit(Xq, yq),
        xgb.XGBClassifier(n_estimators=900, learning_rate=0.05, max_depth=7, subsample=0.9,
                          colsample_bytree=0.9, tree_method="hist", random_state=SEED,
                          n_jobs=-1, verbosity=0).fit(Xq, yq),
    ])
    print(f"[{elapsed():.0f}s] pair stage done", flush=True)

    # ---- Stage 3: start + successor policy ----
    (Xs, ys), (Xp, yp) = gen_training(views, tr_prog)
    print(f"[{elapsed():.0f}s] samples: start {Xs.shape} policy {Xp.shape}", flush=True)
    m_start = lgb.LGBMClassifier(**params).fit(Xs, ys)
    pol_params = {**params, "n_estimators": 1500}
    m_policy = ProbaEnsemble([
        lgb.LGBMClassifier(**pol_params).fit(Xp, yp),
        lgb.LGBMClassifier(**{**pol_params, "random_state": 1, "feature_fraction": 0.8}).fit(Xp, yp),
        xgb.XGBClassifier(n_estimators=1100, learning_rate=0.05, max_depth=7, subsample=0.9,
                          colsample_bytree=0.9, tree_method="hist", random_state=SEED,
                          n_jobs=-1, verbosity=0).fit(Xp, yp),
    ])
    print(f"[{elapsed():.0f}s] models trained", flush=True)

    # ---- test inference (strictly per-row transform + predict) ----
    out_ids, out_progs = [], []
    for i in range(len(test)):
        row = test.iloc[i]
        R = prep_base(row.fragment_cards, row.layout_hint, row.component_count)
        try:
            apply_split_model(R, m_split)
            V0 = with_clusters(R, primary_labels(R))
            xo = np.array([orient_feats(V0, j) for j in range(R["n"])])
            R["psign"] = m_orient.predict_proba(xo)[:, 1]
            bw = 32 if elapsed() < TIME_SOFT else 6
            prog = decode_row(R, m_start, m_policy, m_pair, m_orient, beam_width=bw, alpha=0.6)
        except Exception as e:
            print(f"row {row.id} failed ({type(e).__name__}: {e}); using fallback", flush=True)
            prog = fallback_prog(R["names"])
        out_ids.append(row.id)
        out_progs.append(json.dumps(prog, separators=(",", ":")))
        if (i + 1) % 300 == 0:
            print(f"[{elapsed():.0f}s] decoded {i + 1}/{len(test)}", flush=True)

    sub = pd.DataFrame({"id": out_ids, "predicted_program": out_progs})
    sub.to_csv(submission_out, index=False)
    print(f"[{elapsed():.0f}s] wrote {submission_out} ({len(sub)} rows)", flush=True)


if __name__ == "__main__":
    main()
