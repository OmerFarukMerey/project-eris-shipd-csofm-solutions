"""Recovering Denied Echo Structure From A First-Return Point Cloud.

Pipeline
--------
1. For every item (train and test), compute the published ground reference g and,
   for each query pulse, (a) two blocks of hand-crafted neighbourhood statistics
   and (b) the raw local point neighbourhood (K nearest first returns in the
   horizontal plane) around the pulse.
2. Train, in-script and from scratch, a PointNet-style deep network on the raw
   neighbourhoods (with the hand-crafted statistics fed into the head as extra
   signal), with two classification heads (echo_class, depth_class), under a
   grouped K-fold split by item_id.  The epoch checkpoint used for prediction is
   selected in-script on each fold's validation score.  Once the grouped roster
   has produced out-of-fold predictions, the remaining budget trains additional
   models on 100% of the training rows (at the epoch count the cross-validation
   selected) whose test predictions join the same ensemble average.
3. Train a LightGBM model per fold on the hand-crafted statistics as a secondary
   ensemble member (its hyperparameters are searched in-script on fold 0).
4. Search the NN/LGBM blend weight per head in-script on out-of-fold predictions
   against the challenge's macro-F1-based metric, then search per-class decode
   offsets on the blended OOF -- keeping them only if they transfer across a
   held-out half of the training items -- and take the per-row argmax.

All fitted state (normalisation statistics, models, blend weights, decode
offsets) comes from training data only; test items are used strictly for
per-item transform + per-row inference.
"""
import os
import sys
import time
import warnings
import traceback
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

T0 = time.time()
# Soft budgets (guidebook 3.5).  The grouped roster stops first so that the
# full-data refits always get a slice of the budget.
ROSTER_BUDGET_S = float(os.environ.get("ROSTER_BUDGET_S", 2700))
TRAIN_BUDGET_S = float(os.environ.get("TRAIN_BUDGET_S", 3250))
HARD_STOP_S = float(os.environ.get("HARD_STOP_S", 3900))
EPOCHS_ENV = int(os.environ.get("EPOCHS", 0))
SEED = 20260807

K_NEI = 448                      # fine neighbourhood size for the point encoder
K_WIDE = 1792                    # wide pool; every 4th point -> coarse branch
C_STRIDE = 4
RADII = (1.5, 3.0, 6.0, 12.0)    # multi-scale cylinder radii for statistics
KNN = (10, 25, 50)               # eigen-feature neighbourhood sizes
R2 = (1.0, 2.0, 4.0, 8.0, 16.0)  # radii for the structural feature block
KN2 = (8, 32, 128)               # k-NN sizes for the structural feature block
TTA = 8                          # rotations averaged per test row at inference


def elapsed():
    return time.time() - T0


def log(*a):
    print(f"[{elapsed():7.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------
# Ground reference exactly as published in the problem statement.
# --------------------------------------------------------------------------
def ground_reference(x, y, z):
    ix = np.floor((x - x.min()) / 5.0).astype(np.int64)
    iy = np.floor((y - y.min()) / 5.0).astype(np.int64)
    key = iy * (ix.max() + 1) + ix
    order = np.argsort(key, kind="stable")
    key_s, z_s = key[order], z[order]
    starts = np.flatnonzero(np.append(True, key_s[1:] != key_s[:-1]))
    ends = np.append(starts[1:], len(key_s))
    g = np.empty(len(z), dtype=np.float64)
    for s, e in zip(starts, ends):
        g[order[s:e]] = np.percentile(z_s[s:e], 5.0)
    return g


# --------------------------------------------------------------------------
# Per-item extraction: hand-crafted per-query statistics + raw neighbourhoods.
# --------------------------------------------------------------------------
def item_features(x, y, z, it, g, hg, tree2, tree3, qidx):
    """Block A: multi-scale cylinder statistics and k-NN eigen-features."""
    N = len(x)
    out = []
    zmax_item = np.percentile(hg, 99)
    for qi in qidx:
        z0 = z[qi]
        h0 = z0 - g[qi]
        i0 = it[qi]
        f = [h0, i0, np.log1p(N), zmax_item, h0 / max(zmax_item, 1.0)]
        for r in RADII:
            idx = tree2.query_ball_point([x[qi], y[qi]], r)
            idx = np.asarray(idx, dtype=np.int64)
            n = len(idx)
            f.append(np.log1p(n))
            f.append(n / (np.pi * r * r))
            if n < 3:
                f.extend([0.0] * 23)
                continue
            zn = z[idx]
            hn = hg[idx]
            inn = it[idx]
            dz = zn - z0
            f.append((dz > 0.5).mean())
            f.append((dz < -0.5).mean())
            f.append((dz < -2.0).mean())
            f.append((dz < -0.5 * h0).mean())
            f.append(dz.mean())
            f.append(dz.std())
            f.extend(list(np.percentile(hn, [5, 25, 50, 75, 95])))
            f.append(hn.max())
            f.append(h0 - hn.max())
            f.append(h0 / max(hn.max(), 1.0))
            f.append(hn.std())
            f.append(hn.mean())
            f.append(inn.mean())
            f.append(inn.std())
            f.append(i0 - inn.mean())
            below = hn[hn < h0]
            f.append(len(below) / n)
            if len(below) >= 2:
                f.append(below.std())
                f.append(below.mean() / max(h0, 1.0))
            else:
                f.extend([0.0, 0.0])
            nb = max(int(np.ceil(h0)), 1)
            occ = np.histogram(hn[(hn >= 0) & (hn < h0)],
                               bins=min(nb, 30), range=(0, max(h0, 1.0)))[0]
            f.append((occ > 0).mean())
        for k in KNN:
            kk = min(k, N)
            dd, ii = tree3.query([x[qi], y[qi], z[qi]], k=kk)
            dd = np.atleast_1d(dd)
            ii = np.atleast_1d(ii)
            P = np.c_[x[ii], y[ii], z[ii]]
            P = P - P.mean(0)
            C = P.T @ P / max(len(P) - 1, 1)
            w, V = np.linalg.eigh(C)
            w = np.clip(w[::-1], 1e-9, None)
            V = V[:, ::-1]
            s = w.sum()
            f.extend([w[0] / s, w[1] / s, w[2] / s,
                      (w[0] - w[1]) / w[0], (w[1] - w[2]) / w[0], w[2] / w[0],
                      abs(V[2, 2]), abs(V[2, 0]),
                      dd.mean(), dd.max(), np.log1p(w[2])])
        out.append(f)
    return np.asarray(out, dtype=np.float32)


NPR2, NPK2, NCHM2 = 18, 9, 14
N_FEAT_A = 5 + len(RADII) * 25 + len(KNN) * 11
N_FEAT_B = len(R2) * NPR2 + len(KN2) * NPK2 + NCHM2
N_FEAT = N_FEAT_A + N_FEAT_B


def chm_raster(x, y, hg, cell=0.5):
    """Canopy-height raster of the item (max height above ground per cell)."""
    x0, y0 = x.min(), y.min()
    nx = int(np.floor((x.max() - x0) / cell)) + 1
    ny = int(np.floor((y.max() - y0) / cell)) + 1
    ix = np.clip(((x - x0) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell).astype(np.int64), 0, ny - 1)
    flat = iy * nx + ix
    chm = np.full(nx * ny, np.nan)
    np.maximum.at(chm, flat, hg)
    return chm.reshape(ny, nx), x0, y0, nx, ny, cell


def _win(a, cy, cx, h):
    y0 = max(cy - h, 0)
    y1 = min(cy + h + 1, a.shape[0])
    x0 = max(cx - h, 0)
    x1 = min(cx + h + 1, a.shape[1])
    return a[y0:y1, x0:x1]


def item_features_b(x, y, z, it, g, hg, tree2, qidx):
    """Block B: scale-free height ranks, gap / vertical-layer structure,
    crown-edge geometry and canopy-height-raster roughness."""
    out = np.zeros((len(qidx), N_FEAT_B), np.float32)
    chm, gx0, gy0, gnx, gny, cell = chm_raster(x, y, hg)
    neigh = tree2.query_ball_point(np.c_[x[qidx], y[qidx]], max(R2))
    dd_all, ii_all = tree2.query(np.c_[x[qidx], y[qidx]],
                                 k=min(max(KN2), len(x)))
    dd_all = np.atleast_2d(dd_all)
    ii_all = np.atleast_2d(ii_all)
    for j, qi in enumerate(qidx):
        f = []
        h0 = max(hg[qi], 1.0)
        z0 = z[qi]
        i0 = it[qi]
        big = np.asarray(neigh[j], dtype=np.int64)
        rbig = np.hypot(x[big] - x[qi], y[big] - y[qi])
        for r in R2:
            idx = big[rbig <= r]
            n = len(idx)
            if n < 4:
                f.extend([0.0] * NPR2)
                continue
            hn = hg[idx]
            inn = it[idx]
            f.append((hn < h0).mean())
            f.append((hn < h0 - 1.0).mean())
            f.append((hn > h0 + 1.0).mean())
            q10, q50, q90 = np.percentile(hn, [10, 50, 90])
            f.append(h0 - q10)
            f.append(h0 - q50)
            f.append(h0 - q90)
            f.append(h0 / max(np.percentile(hn, 95), 0.5))
            f.append((hn < 1.0).mean())
            f.append((hn < 2.0).mean())
            f.append((np.abs(hn - h0) < 0.5).mean())
            nb = max(int(np.ceil(h0)), 1)
            occ = np.histogram(hn[(hn >= 0) & (hn < h0)], bins=nb,
                               range=(0, h0))[0]
            f.append((occ > 0).sum() / nb)
            run = best = 0
            for v in occ:
                run = run + 1 if v == 0 else 0
                best = max(best, run)
            f.append(best * h0 / nb)
            f.append(best / nb)
            f.append(hn.std())
            m = hn.mean()
            s = hn.std() + 1e-6
            f.append(float(((hn - m) ** 3).mean() / s ** 3))
            f.append(inn.mean() - i0)
            near = np.abs(hn - h0) < 1.0
            f.append(inn[near].mean() - i0 if near.sum() > 2 else 0.0)
            f.append(np.log1p(n) / np.log1p(np.pi * r * r))
        for k in KN2:
            kk = min(k, dd_all.shape[1])
            ii = ii_all[j, :kk]
            dd = dd_all[j, :kk]
            dh = hg[ii] - h0
            f.append(dh.mean())
            f.append(dh.max())
            f.append(dh.min())
            f.append(dh.std())
            hi = np.flatnonzero(dh > 1.0)
            f.append(dd[hi[0]] if len(hi) else 30.0)
            lo = np.flatnonzero(dh < -2.0)
            f.append(dd[lo[0]] if len(lo) else 30.0)
            f.append(dd[-1])
            A = np.c_[x[ii] - x[qi], y[ii] - y[qi], np.ones(kk)]
            try:
                c, *_ = np.linalg.lstsq(A, z[ii], rcond=None)
                res = z[ii] - A @ c
                f.append(float(np.sqrt((res ** 2).mean())))
                f.append(float(z0 - c[2]))
            except Exception:
                f.extend([0.0, 0.0])
        cx = int(np.clip((x[qi] - gx0) / cell, 0, gnx - 1))
        cy = int(np.clip((y[qi] - gy0) / cell, 0, gny - 1))
        for h in (3, 7, 15):
            w = _win(chm, cy, cx, h)
            v = w[np.isfinite(w)]
            if len(v) < 4:
                f.extend([0.0] * 4)
                continue
            f.append(v.std())
            f.append(h0 - v.max())
            f.append((v < h0 - 2.0).mean())
            f.append(np.isnan(w).mean())
        w1 = _win(chm, cy, cx, 1)
        w5 = _win(chm, cy, cx, 5)
        v5 = w5[np.isfinite(w5)]
        f.append(float(np.nanmean(w1) - h0) if np.isfinite(w1).any() else 0.0)
        f.append(float(h0 - np.median(v5)) if len(v5) else 0.0)
        out[j] = np.asarray(f[:N_FEAT_B], np.float32)
    return out


def process_item(args):
    """Extract (features, neighbourhoods) for one item's query pulses."""
    npz_path, qidx = args
    nq = len(qidx)
    try:
        from scipy.spatial import cKDTree
        zz = np.load(npz_path)
        pts = zz["points"]
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        z = pts[:, 2].astype(np.float64)
        it = pts[:, 3].astype(np.float64)
        g = ground_reference(x, y, z)
        hg = z - g
        tree2 = cKDTree(np.c_[x, y])
        tree3 = cKDTree(np.c_[x, y, z])
        FA = item_features(x, y, z, it, g, hg, tree2, tree3, qidx)
        FB = item_features_b(x, y, z, it, g, hg, tree2, qidx)
        F = np.concatenate([FA, FB], 1)
        kw = min(K_WIDE, len(x))
        _, ii = tree2.query(np.c_[x[qidx], y[qidx]], k=kw)
        ii = np.atleast_2d(ii)
        NB = np.zeros((nq, K_NEI, 6), dtype=np.float16)
        NBc = np.zeros((nq, K_NEI, 6), dtype=np.float16)

        def fill(dst, j, idx, qi):
            kk = len(idx)
            dx = x[idx] - x[qi]
            dy = y[idx] - y[qi]
            dz = z[idx] - z[qi]
            dst[j, :kk, 0] = dx
            dst[j, :kk, 1] = dy
            dst[j, :kk, 2] = dz
            dst[j, :kk, 3] = hg[idx]
            dst[j, :kk, 4] = it[idx]
            dst[j, :kk, 5] = np.sqrt(dx * dx + dy * dy)

        for j, qi in enumerate(qidx):
            fill(NB, j, ii[j][:K_NEI], qi)
            fill(NBc, j, ii[j][::C_STRIDE][:K_NEI], qi)
        return F, NB, NBc
    except Exception:
        traceback.print_exc()
        return (np.zeros((nq, N_FEAT), dtype=np.float32),
                np.zeros((nq, K_NEI, 6), dtype=np.float16),
                np.zeros((nq, K_NEI, 6), dtype=np.float16))


def extract_split(df, folder):
    """Extract features/neighbourhoods for every row of df (item-grouped)."""
    tasks = []
    order = []
    for iid, gdf in df.groupby("item_id", sort=False):
        tasks.append((os.path.join(folder, str(iid) + ".npz"),
                      gdf.query_index.values.astype(np.int64)))
        order.append(gdf.index.values)
    F = np.zeros((len(df), N_FEAT), dtype=np.float32)
    NB = np.zeros((len(df), K_NEI, 6), dtype=np.float16)
    NBc = np.zeros((len(df), K_NEI, 6), dtype=np.float16)

    def consume(idxs, res):
        Fi, NBi, NBci = res
        F[idxs] = Fi
        NB[idxs] = NBi
        NBc[idxs] = NBci

    try:
        from concurrent.futures import ProcessPoolExecutor
        nw = min(12, max(2, (os.cpu_count() or 4)))
        with ProcessPoolExecutor(max_workers=nw) as ex:
            # stream results into the preallocated arrays to keep peak RAM low
            for idxs, res in zip(order, ex.map(process_item, tasks,
                                               chunksize=8)):
                consume(idxs, res)
    except Exception:
        traceback.print_exc()
        log("parallel extraction failed; falling back to serial")
        for idxs, t in zip(order, tasks):
            consume(idxs, process_item(t))
    return F, NB, NBc


# --------------------------------------------------------------------------
# Metric helpers.
# --------------------------------------------------------------------------
def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro")


def head_score(f1, K):
    return float(np.clip((f1 - 1.0 / K) / (1.0 - 1.0 / K), 0.01, 1.0))


def combined(f1e, f1d):
    return 0.6 * head_score(f1e, 3) + 0.4 * head_score(f1d, 4)


def search_class_bias(P, y, K):
    """Coordinate-descent search (on training OOF only) for per-class
    log-probability offsets that maximise macro-F1.  Applied per-row at
    decode time; no test data is involved in the search."""
    logP = np.log(P + 1e-9)
    best = macro_f1(y, logP.argmax(1))
    b = np.zeros(K)
    grid = np.arange(-0.8, 0.801, 0.05)
    for _ in range(3):
        for c in range(K):
            for g in grid:
                bt = b.copy()
                bt[c] = g
                f1 = macro_f1(y, (logP + bt).argmax(1))
                if f1 > best:
                    best, b = f1, bt
    return b, best


def robust_class_bias(P, y, groups, K, rng):
    """Same search, but only adopted if the offsets transfer to held-out
    training items.  The two halves are disjoint item groups, mirroring the
    structural train/test separation described in the problem statement."""
    logP = np.log(P + 1e-9)
    gains, n = 0.0, 0
    for _ in range(2):                      # two independent group splits
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        inA = np.isin(groups, uniq[:len(uniq) // 2])
        for tr_mask, te_mask in ((inA, ~inA), (~inA, inA)):
            if tr_mask.sum() < 100 or te_mask.sum() < 100:
                continue
            bh, _ = search_class_bias(P[tr_mask], y[tr_mask], K)
            base = macro_f1(y[te_mask], logP[te_mask].argmax(1))
            gains += macro_f1(y[te_mask], (logP[te_mask] + bh).argmax(1)) - base
            n += 1
    gain = gains / max(n, 1)
    b, f1 = search_class_bias(P, y, K)
    if n == 0 or gain <= 0:
        log(f"class bias K={K} did not transfer (held-out gain "
            f"{gain:+.4f}); using zero offsets")
        return np.zeros(K), macro_f1(y, logP.argmax(1))
    log(f"class bias K={K} transfers (held-out gain {gain:+.4f})")
    return b, f1


# --------------------------------------------------------------------------
# Deep model.
# --------------------------------------------------------------------------
def build_torch():
    import torch
    import torch.nn as nn

    class PointNet(nn.Module):
        def __init__(self, fdim, two_scale=False, use_feat=True, attn=False):
            super().__init__()
            self.two_scale = two_scale
            self.use_feat = use_feat
            self.attn_on = attn

            def enc():
                return nn.Sequential(
                    nn.Conv1d(8, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                )
            self.enc_f = enc()
            self.enc_c = enc() if two_scale else None
            pdim = 512 * (2 if two_scale else 1)
            if attn:
                self.attn = nn.Conv1d(256, 4, 1)  # 4 attention heads
                pdim += 4 * 256
            if use_feat:
                self.fmlp = nn.Sequential(
                    nn.Linear(fdim, 128), nn.BatchNorm1d(128), nn.ReLU())
                pdim += 128
            self.head = nn.Sequential(
                nn.Linear(pdim, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.he = nn.Linear(128, 3)
            self.hd = nn.Linear(128, 4)

        def forward(self, nb, f, nbc=None):
            e = self.enc_f(nb.transpose(1, 2))
            parts = [e.max(2).values, e.mean(2)]
            if self.attn_on:
                w = torch.softmax(self.attn(e), dim=2)       # [B,4,K]
                parts.append(torch.einsum("bhk,bck->bhc", w, e).flatten(1))
            if self.two_scale:
                ec = self.enc_c(nbc.transpose(1, 2))
                parts += [ec.max(2).values, ec.mean(2)]
            if self.use_feat:
                parts.append(self.fmlp(f))
            h = self.head(torch.cat(parts, 1))
            return self.he(h), self.hd(h)

    return torch, nn, PointNet


# Fixed unit scalings for the raw point channels (metres -> O(1)); the first
# BatchNorm layer learns the actual normalisation from training data.
CH_SCALE = np.array([5.0, 5.0, 10.0, 20.0, 1.0, 5.0], dtype=np.float32)
CH_SCALE_C = np.array([15.0, 15.0, 10.0, 20.0, 1.0, 15.0], dtype=np.float32)


def make_point_batch(NB, H0, idx, sc=CH_SCALE):
    nb = NB[idx].astype(np.float32)
    h0 = H0[idx][:, None]
    c6 = np.clip(nb[:, :, 2] / h0, -3, 3)
    c7 = np.clip(nb[:, :, 3] / h0, 0, 3)
    nb = nb / sc
    return np.concatenate([nb, c6[:, :, None], c7[:, :, None]], 2)


def train_nn_fold(NB, NBc, Fn, H0, ye, yd, trn, val,
                  NB_te, NBc_te, Fn_te, H0_te,
                  seed, epochs, dev, two_scale=False, flip=False,
                  use_feat=True, attn=False):
    """Train one network.  ``val`` may be None, in which case the model is a
    full-data refit: no epoch selection, train for exactly ``epochs`` epochs and
    return test probabilities only."""
    torch, nn, PointNet = build_torch()
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    use_cuda = dev == "cuda"
    net = PointNet(Fn.shape[1], two_scale=two_scale,
                   use_feat=use_feat, attn=attn).to(dev)
    bs = 512
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = ((len(trn) + bs - 1) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3,
                                                total_steps=steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    except Exception:                       # older torch API
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    Ye = torch.from_numpy(ye).long()
    Yd = torch.from_numpy(yd).long()

    def rotate(nb, ca, sa):
        dx = nb[:, :, 0] * ca - nb[:, :, 1] * sa
        dy = nb[:, :, 0] * sa + nb[:, :, 1] * ca
        nb[:, :, 0] = dx
        nb[:, :, 1] = dy

    def predict(indices, NBa, NBca, Fa, Ha, tta=1):
        net.eval()
        pes, pds = [], []
        with torch.no_grad():
            for i in range(0, len(indices), 4096):
                bi = indices[i:i + 4096]
                nb0 = torch.from_numpy(make_point_batch(NBa, Ha, bi)).to(dev)
                nbc0 = None
                if two_scale:
                    nbc0 = torch.from_numpy(
                        make_point_batch(NBca, Ha, bi, CH_SCALE_C)).to(dev)
                f = torch.from_numpy(Fa[bi]).to(dev)
                pe_acc = 0
                pd_acc = 0
                for t in range(tta):
                    nb, nbc = nb0, nbc0
                    if t > 0:
                        ang = torch.tensor(2 * np.pi * t / tta, device=dev)
                        ca, sa = torch.cos(ang), torch.sin(ang)
                        nb = nb0.clone()
                        rotate(nb, ca, sa)
                        if two_scale:
                            nbc = nbc0.clone()
                            rotate(nbc, ca, sa)
                    with torch.amp.autocast("cuda", enabled=use_cuda):
                        le, ld = net(nb, f, nbc)
                    pe_acc = pe_acc + le.float().softmax(1)
                    pd_acc = pd_acc + ld.float().softmax(1)
                pes.append((pe_acc / tta).cpu().numpy())
                pds.append((pd_acc / tta).cpu().numpy())
        return np.concatenate(pes), np.concatenate(pds)

    best = (-1.0, None, -1)  # (score, state_dict, epoch)
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(trn)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 8:
                continue
            nb = torch.from_numpy(make_point_batch(NB, H0, idx)).to(dev)
            nbc = None
            if two_scale:
                nbc = torch.from_numpy(
                    make_point_batch(NBc, H0, idx, CH_SCALE_C)).to(dev)
            f = torch.from_numpy(Fn[idx]).to(dev)
            B = len(idx)
            ang = torch.rand(B, device=dev) * 2 * np.pi
            ca, sa = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
            rotate(nb, ca, sa)
            if two_scale:
                rotate(nbc, ca, sa)
            if flip:
                fl = ((torch.rand(B, 1, device=dev) < 0.5).float() * (-2) + 1)
                nb[:, :, 0] = nb[:, :, 0] * fl
                if two_scale:
                    nbc[:, :, 0] = nbc[:, :, 0] * fl
            with torch.amp.autocast("cuda", enabled=use_cuda):
                le, ld = net(nb, f, nbc)
                loss = ce(le, Ye[idx].to(dev)) + ce(ld, Yd[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        # in-script epoch selection on the fold's validation split
        if val is not None and ep >= max(6, epochs // 3):
            pe, pd_ = predict(val, NB, NBc, Fn, H0)
            sc = combined(macro_f1(ye[val], pe.argmax(1)),
                          macro_f1(yd[val], pd_.argmax(1)))
            if sc > best[0]:
                best = (sc, {k: v.detach().cpu().clone()
                             for k, v in net.state_dict().items()}, ep)
        if elapsed() > HARD_STOP_S - 240:
            log("nn training truncated by hard stop guard at epoch", ep)
            break
    if best[1] is not None:
        net.load_state_dict(best[1])
    pe_v = pd_v = None
    if val is not None:
        pe_v, pd_v = predict(val, NB, NBc, Fn, H0)
    te_idx = np.arange(len(Fn_te))
    pe_t, pd_t = predict(te_idx, NB_te, NBc_te, Fn_te, H0_te, tta=TTA)
    return pe_v, pd_v, pe_t, pd_t, {"best_ep": best[2], "val_score": best[0]}


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def main():
    from pathlib import Path

    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(public_dir / "test.csv")

    # ---- schema-valid placeholder immediately (crash insurance) ----
    placeholder = pd.DataFrame({
        "id": test["id"],
        "echo_class": np.ones(len(test), dtype=np.int64),
        "depth_class": np.ones(len(test), dtype=np.int64),
    })
    placeholder.to_csv(submission_out, index=False)
    log("placeholder submission written")

    train = pd.read_csv(public_dir / "train.csv")
    log("train", train.shape, "test", test.shape)

    try:
        run_pipeline(public_dir, submission_out, train, test, placeholder)
    except Exception:
        # any unexpected failure must still leave a valid submission on disk
        traceback.print_exc()
        log("pipeline crashed; last written submission is kept")
        try:
            sub = pd.read_csv(submission_out)
            ok = (list(sub.columns) == ["id", "echo_class", "depth_class"]
                  and len(sub) == len(test))
        except Exception:
            ok = False
        if not ok:
            placeholder.to_csv(submission_out, index=False)
            log("placeholder restored after crash")


def run_pipeline(public_dir, submission_out, train, test, placeholder):

    def write_submission(pe, pdp):
        """Per-row argmax of blended probabilities -> CSV."""
        pe = np.asarray(pe)
        pdp = np.asarray(pdp)
        e = np.where(np.isfinite(pe.sum(1)), pe.argmax(1), 1)
        d = np.where(np.isfinite(pdp.sum(1)), pdp.argmax(1), 1)
        out = pd.DataFrame({
            "id": test["id"],
            "echo_class": np.clip(e, 0, 2).astype(np.int64),
            "depth_class": np.clip(d, 0, 3).astype(np.int64),
        })
        out.to_csv(submission_out, index=False)

    # ---- extraction ----
    F_tr, NB_tr, NBc_tr = extract_split(train, str(public_dir / "train"))
    log("train extracted", F_tr.shape, NB_tr.shape)
    F_te, NB_te, NBc_te = extract_split(test, str(public_dir / "test"))
    log("test extracted", F_te.shape, NB_te.shape)

    ye = train["echo_class"].values.astype(np.int64)
    yd = train["depth_class"].values.astype(np.int64)
    groups = train["item_id"].values

    from sklearn.model_selection import GroupKFold
    NF = 5
    folds = list(GroupKFold(n_splits=NF).split(F_tr, ye, groups))

    # ---- safety-net LightGBM (fast; guarantees a trained-model submission),
    #      with a small in-script hyperparameter search on fold 0.
    #      On any failure fall back to uniform probabilities so the later
    #      blend search simply drives its weight to the NN. ----
    oof_lgb_e = np.full((len(train), 3), 1.0 / 3)
    oof_lgb_d = np.full((len(train), 4), 1.0 / 4)
    te_lgb_e = np.full((len(test), 3), 1.0 / 3)
    te_lgb_d = np.full((len(test), 4), 1.0 / 4)
    try:
        import lightgbm as lgb
        trn0, val0 = folds[0]
        best_hp, best_sc = (31, 600), -1.0
        for nl, ne in ((31, 600), (63, 350)):
            f1s = []
            for tgt in (ye, yd):
                m = lgb.LGBMClassifier(
                    n_estimators=ne, learning_rate=0.05, num_leaves=nl,
                    subsample=0.8, colsample_bytree=0.7,
                    random_state=SEED % 100000, verbose=-1)
                m.fit(F_tr[trn0], tgt[trn0])
                f1s.append(macro_f1(tgt[val0],
                                    m.predict_proba(F_tr[val0]).argmax(1)))
            sc = combined(f1s[0], f1s[1])
            if sc > best_sc:
                best_sc, best_hp = sc, (nl, ne)
            if elapsed() > ROSTER_BUDGET_S * 0.35:
                break
        log("lgb HP search:", best_hp, "fold0 score", round(best_sc, 4))

        def fit_lgb(trn, tgt):
            nl, ne = best_hp
            m = lgb.LGBMClassifier(
                n_estimators=ne, learning_rate=0.05, num_leaves=nl,
                subsample=0.8, colsample_bytree=0.7,
                random_state=SEED % 100000, verbose=-1)
            m.fit(F_tr[trn], tgt[trn])
            return m

        te_lgb_e[:] = 0.0
        te_lgb_d[:] = 0.0
        for fi, (trn, val) in enumerate(folds):
            me = fit_lgb(trn, ye)
            md = fit_lgb(trn, yd)
            oof_lgb_e[val] = me.predict_proba(F_tr[val])
            oof_lgb_d[val] = md.predict_proba(F_tr[val])
            te_lgb_e += me.predict_proba(F_te) / NF
            te_lgb_d += md.predict_proba(F_te) / NF
            if fi == 0:
                # intermediate real-model submission as an early safety net
                write_submission(te_lgb_e * NF, te_lgb_d * NF)
                log("lgb fold0 submission written")
            if elapsed() > ROSTER_BUDGET_S:
                log("budget: stopping lgb folds after fold", fi)
                # scale partial test accumulations back to a mean
                te_lgb_e *= NF / (fi + 1)
                te_lgb_d *= NF / (fi + 1)
                break
        f1e_l = macro_f1(ye, oof_lgb_e.argmax(1))
        f1d_l = macro_f1(yd, oof_lgb_d.argmax(1))
        log("lgb OOF: echoF1", round(f1e_l, 4), "depthF1", round(f1d_l, 4),
            "score", round(combined(f1e_l, f1d_l), 4))
        write_submission(te_lgb_e, te_lgb_d)
        log("lgb full submission written")
    except Exception:
        traceback.print_exc()
        # reset to uniform so a partial failure cannot corrupt the blend
        oof_lgb_e = np.full((len(train), 3), 1.0 / 3)
        oof_lgb_d = np.full((len(train), 4), 1.0 / 4)
        te_lgb_e = np.full((len(test), 3), 1.0 / 3)
        te_lgb_d = np.full((len(test), 4), 1.0 / 4)
        log("lightgbm stage failed; continuing with NN only")

    # ---- deep model: grouped folds x configs, time-guarded ----
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    except Exception:
        dev = None
        log("torch unavailable; keeping LightGBM submission")
    H0_tr = np.maximum(F_tr[:, 0], 1.0).astype(np.float32)
    H0_te = np.maximum(F_te[:, 0], 1.0).astype(np.float32)

    oof_nn_e = np.zeros((len(train), 3))
    oof_nn_d = np.zeros((len(train), 4))
    oof_nn_cnt = np.zeros(len(train))
    te_nn_e = np.zeros((len(test), 3))
    te_nn_d = np.zeros((len(test), 4))
    n_models = 0
    best_eps = []
    # full-train feature normalisation, reused by the refit stage
    mu_all = F_tr.mean(0)
    sd_all = F_tr.std(0) + 1e-6
    Fn_all = np.clip((F_tr - mu_all) / sd_all, -8, 8).astype(np.float32)
    Fn_all_te = np.clip((F_te - mu_all) / sd_all, -8, 8).astype(np.float32)

    if dev is not None:
        EPOCHS = EPOCHS_ENV or (26 if dev in ("cuda", "mps") else 8)
        # diverse model roster (config-level ensemble diversity beats extra
        # same-config seeds on grouped validation):
        # (name, two_scale, flip, use_feat, attn)
        ROSTER = [
            ("ss", False, False, True, False),
            ("ts", True, False, True, False),      # + coarse context branch
            ("at", False, False, True, True),      # attention pooling
            ("ssf", False, True, True, False),     # mirror-flip augmented
            ("nf", False, False, False, False),    # points-only
            ("tsf", True, True, True, False),
        ]
        cost = {False: None, True: None}   # per-model cost by two_scale
        stop = False
        for ci, (cname, two_scale, flip, use_feat, attn) in enumerate(ROSTER):
            if stop:
                break
            for fi, (trn, val) in enumerate(folds):
                if n_models > 0:
                    proj = elapsed() + (cost[two_scale] or
                                        1.9 * (cost[False] or 0))
                    if proj > ROSTER_BUDGET_S:
                        log("budget: stop roster before", cname, "fold", fi)
                        stop = True
                        break
                t_m = time.time()
                # fold-local feature normalisation (train-fold statistics)
                mu_f = F_tr[trn].mean(0)
                sd_f = F_tr[trn].std(0) + 1e-6
                Fn = np.clip((F_tr - mu_f) / sd_f, -8, 8).astype(np.float32)
                Fn_te = np.clip((F_te - mu_f) / sd_f, -8, 8).astype(np.float32)
                try:
                    pe_v, pd_v, pe_t, pd_t, info = train_nn_fold(
                        NB_tr, NBc_tr, Fn, H0_tr, ye, yd, trn, val,
                        NB_te, NBc_te, Fn_te, H0_te,
                        seed=SEED + 1000 * ci + fi, epochs=EPOCHS, dev=dev,
                        two_scale=two_scale, flip=flip,
                        use_feat=use_feat, attn=attn)
                except Exception:
                    traceback.print_exc()
                    log("nn fold failed; continuing")
                    continue
                oof_nn_e[val] += pe_v
                oof_nn_d[val] += pd_v
                oof_nn_cnt[val] += 1
                te_nn_e += pe_t
                te_nn_d += pd_t
                n_models += 1
                if info["best_ep"] >= 0:
                    best_eps.append(info["best_ep"])
                cost[two_scale] = time.time() - t_m
                log(f"nn {cname} fold{fi} best_ep {info['best_ep']} "
                    f"val {info['val_score']:.4f} cost {cost[two_scale]:.0f}s")

        # ---- full-data refits: same architecture, trained on 100% of the
        #      training rows for the number of epochs the cross-validation
        #      selected.  These predict test only (they have no held-out rows),
        #      and join the same ensemble average. ----
        n_refit = 0
        if n_models > 0:
            ep_refit = int(np.clip(int(np.median(best_eps)) + 1, 6, EPOCHS)) \
                if best_eps else EPOCHS
            log(f"refit epochs from CV best-epoch median: {ep_refit}")
            all_idx = np.arange(len(train))
            REFITS = [("ss", False, False, True, False),
                      ("ts", True, False, True, False),
                      ("ssf", False, True, True, False),
                      ("at", False, False, True, True)]
            for ri, (cname, two_scale, flip, use_feat, attn) in \
                    enumerate(REFITS):
                unit = cost[two_scale] or (1.9 * (cost[False] or 0))
                # a refit sees 1/(1-1/NF) more rows per epoch but fewer epochs
                proj = elapsed() + unit * (ep_refit / max(EPOCHS, 1)) * 1.3
                if proj > TRAIN_BUDGET_S:
                    log("budget: stop refits before", cname)
                    break
                t_m = time.time()
                try:
                    _, _, pe_t, pd_t, _ = train_nn_fold(
                        NB_tr, NBc_tr, Fn_all, H0_tr, ye, yd, all_idx, None,
                        NB_te, NBc_te, Fn_all_te, H0_te,
                        seed=SEED + 777 + ri, epochs=ep_refit, dev=dev,
                        two_scale=two_scale, flip=flip,
                        use_feat=use_feat, attn=attn)
                except Exception:
                    traceback.print_exc()
                    log("nn refit failed; continuing")
                    continue
                te_nn_e += pe_t
                te_nn_d += pd_t
                n_models += 1
                n_refit += 1
                log(f"nn refit {cname} cost {time.time() - t_m:.0f}s")
        log(f"nn models: {n_models} ({n_refit} full-data refits)")

    # ---- blending: weight searched in-script on OOF vs the real metric ----
    if n_models == 0 or oof_nn_cnt.max() == 0:
        log("no NN models trained; keeping the LightGBM submission")
    else:
        cov = oof_nn_cnt > 0
        oof_e = np.full((len(train), 3), 1.0 / 3)
        oof_d = np.full((len(train), 4), 1.0 / 4)
        oof_e[cov] = oof_nn_e[cov] / oof_nn_cnt[cov, None]
        oof_d[cov] = oof_nn_d[cov] / oof_nn_cnt[cov, None]
        te_nn_e_n = te_nn_e / n_models
        te_nn_d_n = te_nn_d / n_models

        best_we, best_f1e = 1.0, -1.0
        best_wd, best_f1d = 1.0, -1.0
        for w in np.arange(0.0, 1.0001, 0.05):
            f1e = macro_f1(ye[cov],
                           (w * oof_e[cov] + (1 - w) * oof_lgb_e[cov]).argmax(1))
            f1d = macro_f1(yd[cov],
                           (w * oof_d[cov] + (1 - w) * oof_lgb_d[cov]).argmax(1))
            if f1e > best_f1e:
                best_we, best_f1e = w, f1e
            if f1d > best_f1d:
                best_wd, best_f1d = w, f1d
        log(f"blend: we={best_we:.2f} echoF1 {best_f1e:.4f} | "
            f"wd={best_wd:.2f} depthF1 {best_f1d:.4f} | "
            f"OOF score {combined(best_f1e, best_f1d):.4f} "
            f"({n_models} nn models, {int(cov.sum())} covered rows)")

        # metric-aware decode: per-class offsets searched on the blended OOF,
        # adopted only if they transfer across held-out training item groups
        rng = np.random.RandomState(SEED % 100000)
        be, f1e_b = robust_class_bias(
            best_we * oof_e[cov] + (1 - best_we) * oof_lgb_e[cov],
            ye[cov], groups[cov], 3, rng)
        bd, f1d_b = robust_class_bias(
            best_wd * oof_d[cov] + (1 - best_wd) * oof_lgb_d[cov],
            yd[cov], groups[cov], 4, rng)
        log(f"class bias: echo {np.round(be, 2)} F1 {f1e_b:.4f} | "
            f"depth {np.round(bd, 2)} F1 {f1d_b:.4f} | "
            f"OOF score {combined(f1e_b, f1d_b):.4f}")
        final_e = best_we * te_nn_e_n + (1 - best_we) * te_lgb_e
        final_d = best_wd * te_nn_d_n + (1 - best_wd) * te_lgb_d
        write_submission(np.log(final_e + 1e-9) + be,
                         np.log(final_d + 1e-9) + bd)
        log("final blended submission written")

    # ---- final verification ----
    sub = pd.read_csv(submission_out)
    ok = (list(sub.columns) == ["id", "echo_class", "depth_class"]
          and len(sub) == len(test)
          and set(sub["id"]) == set(test["id"])
          and sub["id"].is_unique
          and sub["echo_class"].between(0, 2).all()
          and sub["depth_class"].between(0, 3).all()
          and not sub.isna().any().any())
    log("submission verified:", ok, "rows", len(sub))
    if not ok:
        placeholder.to_csv(submission_out, index=False)
        log("verification failed -> placeholder restored")


if __name__ == "__main__":
    main()
    log("done")
    sys.stdout.flush()
    sys.exit(0)
