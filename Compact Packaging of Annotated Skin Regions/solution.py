"""Compact Packaging of Annotated Skin Regions -- end-to-end solution.

usage: python3 solution.py <public_dir> <submission_out>

Pipeline (fixed, static plan):
  1. Dense one-class detector (CenterNet-style heatmap + FCOS-style ltrb box regression + per-edge
     Laplace scale head) on a timm ImageNet backbone with an FPN neck, trained in-script.
  2. K-fold training on train.csv only; every fold model predicts its held-out fold (OOF) and the
     test images.  Test maps are the average over fold models and flip TTA (per image only).
  3. Calibration models fitted on the OOF detections (train only): logistic P(detection is a distinct
     annotated region) at IoU 0.5 and 0.75 from score / predicted edge uncertainty / size.
  4. Region list: per image, the number of top detections that maximises the expected RegionF1.
  5. Crop plan: per image, Monte-Carlo expected-metric decoding -- worlds sample which detections are
     real (calibrated probability) and where their true edges are (predicted Laplace scale); candidate
     plans are exact minimum-area <= 3-rectangle partitions of subsets of expanded detections; the plan
     with the highest expected 0.2*coverage + 0.45*coverage*min(1, D_ref/D) is delivered.
  6. Every decode knob is searched in-script on the OOF predictions against the PROBLEM.md metric.
"""
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numba
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Deterministic cuBLAS workspace; must be set before the first CUDA call creates a cuBLAS handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Fixed plan (static; never changed by runtime, hardware or timers)
# ----------------------------------------------------------------------------------------------
BACKBONE = "convnext_tiny.fb_in22k"
FOLDS = 3
EPOCHS = 14
BATCH = 8
LR = 3e-4
WD = 0.05
EMA_DECAY = 0.999
SEED = 2026
IN_W, IN_H = 768, 640          # photographs are 768x640; resized to this if a file differs
STRIDE = 4
MAXB = 24                      # max boxes per training image kept in the target tensor
TTA = ("none", "h")
INFER_BATCH = 8                # inference always runs on fixed-shape zero-padded batches
TOPK = 100

SEARCH_NMS = [1.0, 0.6]
SEARCH_RMHAT = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
SEARCH_W75 = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Bayes crop decoder
CAND_TMIN = 0.05               # crop candidates: heat peaks above this (calibrated probability decides the rest)
BAYES_JMAX = 16                # top-j inclusion levels tried per image
PLAN_EXACT = 8                 # exact partition DP size (larger candidate sets are greedily pre-merged to this)
KMAX = 20                      # crop candidates per image considered by the Bayes decoder
LS_ROUNDS = 2                  # add/drop local-search rounds after the top-j scan
MC_RS = np.random.RandomState(SEED + 2)
MC_WORLDS = 128
MC_U = MC_RS.rand(MC_WORLDS, KMAX)     # fixed Monte-Carlo draws shared by every image
MC_N = MC_RS.laplace(size=(MC_WORLDS, KMAX, 4))
SEARCH_CNMS = [1.0, 0.6]
SEARCH_KAPPA = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]
SEARCH_M0 = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
SEARCH_MHAT = [0.0, 0.5, 1.0, 2.0, 3.0]
SEARCH_PSCALE = [0.8, 0.9, 1.0, 1.1, 1.25]
SEARCH_ALEVELS = [(0.0, 1.0, 2.0, 3.0), (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), (1.0, 2.0), (0.0,)]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------------------------
def box_json(boxes):
    return json.dumps([[int(v) for v in b] for b in boxes], separators=(",", ":"))


def write_submission(path, ids, regions, crops):
    df = pd.DataFrame({"case_id": ids, "regions": [box_json(r) for r in regions], "crop_plan": [box_json(c) for c in crops]})
    df.to_csv(path, index=False)


def load_img(path):
    try:
        im = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        im = None
    if im is None:
        log(f"WARNING: could not read {path}; using a blank image")
        return np.zeros((IN_H, IN_W, 3), np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[0] != IN_H or im.shape[1] != IN_W:
        im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return im


def load_images(paths):
    with ThreadPoolExecutor(8) as ex:
        ims = list(ex.map(load_img, paths))
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous()


def parse_boxes(s):
    try:
        b = np.array(json.loads(s), dtype=np.float64).reshape(-1, 4)
    except Exception:
        b = np.zeros((0, 4))
    return b


def boxes_to_tensor(box_lists):
    out = torch.zeros(len(box_lists), MAXB, 4)
    valid = torch.zeros(len(box_lists), MAXB, dtype=torch.bool)
    for i, bl in enumerate(box_lists):
        b = torch.tensor(bl, dtype=torch.float32).reshape(-1, 4)[:MAXB]
        b[:, [0, 2]] *= IN_W / 1024.0
        b[:, [1, 3]] *= IN_H / 1024.0
        out[i, :len(b)] = b
        valid[i, :len(b)] = True
    return out, valid


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class Det(nn.Module):
    def __init__(self, name, ch=128):
        super().__init__()
        self.bb = timm.create_model(name, pretrained=True, features_only=True)
        red = self.bb.feature_info.reduction()
        self.keep = [i for i, r in enumerate(red) if r in (4, 8, 16, 32)]
        chs = [self.bb.feature_info.channels()[i] for i in self.keep]
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in chs])
        self.td = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
                                 for _ in chs[:-1]])

        def head(nout, bias):
            m = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                              nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                              nn.Conv2d(ch, nout, 1))
            nn.init.constant_(m[-1].bias, bias)
            nn.init.normal_(m[-1].weight, std=0.01)
            return m
        self.hm = head(1, -4.6)
        self.reg = head(8, 0.0)  # 4 ltrb distances (cells) + 4 log Laplace scales

    def forward(self, x):
        feats = self.bb(x)
        feats = [feats[i] for i in self.keep]
        p = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode="nearest") + self.lat[i](feats[i])
            p = self.td[i](p)
        r = self.reg(p)
        return self.hm(p), F.softplus(r[:, :4] + 1.0), r[:, 4:].clamp(-6, 4)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gpu_augment(img, boxes, valid, gen):
    """Random horizontal flip, isotropic scale (0.75-1.33) + translation, photometric jitter; boxes follow.
    No vertical flip: the annotation boxes are not symmetric under it (see readme)."""
    B, _, H, W = img.shape
    dev = img.device
    s = torch.exp(torch.empty(B, device=dev).uniform_(math.log(0.75), math.log(1.33), generator=gen))
    fx = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    ax, ay = s * fx, s
    bx = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - s).abs()
    by = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - s).abs()
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = 1 / ax
    theta[:, 0, 2] = -bx / ax
    theta[:, 1, 1] = 1 / ay
    theta[:, 1, 2] = -by / ay
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    X0 = ax[:, None] * (2 * boxes[..., 0] / W - 1) + bx[:, None]
    X1 = ax[:, None] * (2 * boxes[..., 2] / W - 1) + bx[:, None]
    Y0 = ay[:, None] * (2 * boxes[..., 1] / H - 1) + by[:, None]
    Y1 = ay[:, None] * (2 * boxes[..., 3] / H - 1) + by[:, None]
    nb = torch.stack([torch.minimum(X0, X1), torch.minimum(Y0, Y1), torch.maximum(X0, X1), torch.maximum(Y0, Y1)], -1)
    nb[..., [0, 2]] = (nb[..., [0, 2]] + 1) * W / 2
    nb[..., [1, 3]] = (nb[..., [1, 3]] + 1) * H / 2
    full = (nb[..., 2] - nb[..., 0]) * (nb[..., 3] - nb[..., 1])
    nb[..., [0, 2]] = nb[..., [0, 2]].clamp(0, W)
    nb[..., [1, 3]] = nb[..., [1, 3]].clamp(0, H)
    kept = (nb[..., 2] - nb[..., 0]) * (nb[..., 3] - nb[..., 1])
    valid = valid & (kept > 0.45 * full) & (nb[..., 2] - nb[..., 0] > 1) & (nb[..., 3] - nb[..., 1] > 1)
    br = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2, generator=gen)
    ct = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2, generator=gen)
    cg = torch.empty(B, 3, 1, 1, device=dev).uniform_(0.92, 1.08, generator=gen)
    m = img.mean((1, 2, 3), keepdim=True)
    img = ((img - m) * ct + m) * br * cg
    return img.clamp(0, 1), nb, valid


def grid_centers(Hs, Ws, dev):
    cx = (torch.arange(Ws, device=dev) + 0.5) * STRIDE
    cy = (torch.arange(Hs, device=dev) + 0.5) * STRIDE
    return cx.view(1, 1, Ws).expand(1, Hs, Ws), cy.view(1, Hs, 1).expand(1, Hs, Ws)


def build_targets(boxes, valid, Hs, Ws, alpha=0.54):
    """Elliptic Gaussian heatmap targets (exact 1 at the centre cell) and per-cell assigned box."""
    B = boxes.shape[0]
    dev = boxes.device
    cx = (torch.arange(Ws, device=dev) + 0.5) * STRIDE
    cy = (torch.arange(Hs, device=dev) + 0.5) * STRIDE
    bcx = (boxes[..., 0] + boxes[..., 2]) / 2
    bcy = (boxes[..., 1] + boxes[..., 3]) / 2
    sx = (alpha * (boxes[..., 2] - boxes[..., 0]).clamp(min=1) / 6).clamp(min=STRIDE * 0.5)
    sy = (alpha * (boxes[..., 3] - boxes[..., 1]).clamp(min=1) / 6).clamp(min=STRIDE * 0.5)
    gx = torch.exp(-((cx[None, None, :] - bcx[..., None]) ** 2) / (2 * sx[..., None] ** 2))
    gy = torch.exp(-((cy[None, None, :] - bcy[..., None]) ** 2) / (2 * sy[..., None] ** 2))
    g = gy[..., :, None] * gx[..., None, :] * valid[..., None, None]
    pi = (bcy / STRIDE).floor().long().clamp(0, Hs - 1)
    pj = (bcx / STRIDE).floor().long().clamp(0, Ws - 1)
    bi, ni = torch.nonzero(valid, as_tuple=True)
    g[bi, ni, pi[bi, ni], pj[bi, ni]] = 1.0
    hm = g.max(1).values
    inside = ((cx[None, None, None, :] >= boxes[..., 0, None, None]) & (cx[None, None, None, :] <= boxes[..., 2, None, None])
              & (cy[None, None, :, None] >= boxes[..., 1, None, None]) & (cy[None, None, :, None] <= boxes[..., 3, None, None]))
    inside = inside | (g >= 1.0)
    w, idx = torch.where(inside, g, torch.zeros_like(g)).max(1)
    tb = torch.gather(boxes, 1, idx.view(B, -1, 1).expand(-1, -1, 4)).view(B, Hs, Ws, 4)
    return hm, w, tb, valid.sum()


def ltrb_to_boxes(ltrb):
    _, _, Hs, Ws = ltrb.shape
    cx, cy = grid_centers(Hs, Ws, ltrb.device)
    d = ltrb * STRIDE
    return torch.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], -1)


def compute_loss(model, x, boxes, valid):
    hm_logit, ltrb, logb = model(x)
    _, _, Hs, Ws = hm_logit.shape
    hm_t, w, tb, npos = build_targets(boxes, valid, Hs, Ws)
    # penalty-reduced focal loss (CenterNet)
    p = torch.sigmoid(hm_logit[:, 0].float()).clamp(1e-4, 1 - 1e-4)
    pos = hm_t.eq(1).float()
    l_hm = -((torch.log(p) * (1 - p) ** 2 * pos).sum()
             + (torch.log(1 - p) * p ** 2 * (1 - hm_t) ** 4 * (1 - pos)).sum()) / npos.clamp(min=1)
    # GIoU on cells inside a box, weighted by the Gaussian
    pb = ltrb_to_boxes(ltrb.float())
    x0 = torch.maximum(pb[..., 0], tb[..., 0]); y0 = torch.maximum(pb[..., 1], tb[..., 1])
    x1 = torch.minimum(pb[..., 2], tb[..., 2]); y1 = torch.minimum(pb[..., 3], tb[..., 3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    union = (pb[..., 2] - pb[..., 0]) * (pb[..., 3] - pb[..., 1]) + (tb[..., 2] - tb[..., 0]) * (tb[..., 3] - tb[..., 1]) - inter + 1e-6
    enc = ((torch.maximum(pb[..., 2], tb[..., 2]) - torch.minimum(pb[..., 0], tb[..., 0]))
           * (torch.maximum(pb[..., 3], tb[..., 3]) - torch.minimum(pb[..., 1], tb[..., 1])) + 1e-6)
    giou = inter / union - (enc - union) / enc
    wsum = w.sum().clamp(min=1e-6)
    l_reg = ((1 - giou) * w).sum() / wsum
    # Laplace NLL of each edge error (cells); the distance prediction itself is detached
    cx, cy = grid_centers(Hs, Ws, x.device)
    gt_d = torch.stack([cx - tb[..., 0], cy - tb[..., 1], tb[..., 2] - cx, tb[..., 3] - cy], 1) / STRIDE
    err = (ltrb.float().detach() - gt_d).abs()
    lb = logb.float()
    l_unc = ((err * torch.exp(-lb) + lb).sum(1) * w).sum() / wsum
    return l_hm + 2.0 * l_reg + 0.1 * l_unc, torch.stack([l_hm.detach(), l_reg.detach(), l_unc.detach()])


class EMA:
    def __init__(self, model):
        sd = model.state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in sd.items()}
        self.fk = [k for k, v in sd.items() if v.dtype.is_floating_point]
        self.ik = [k for k, v in sd.items() if not v.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model, decay):
        sd = model.state_dict()
        sh = [self.shadow[k] for k in self.fk]
        torch._foreach_mul_(sh, decay)
        torch._foreach_add_(sh, [sd[k].detach().float() for k in self.fk], alpha=1 - decay)
        for k in self.ik:
            self.shadow[k].copy_(sd[k])

    def load_into(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: v.to(sd[k].dtype) for k, v in self.shadow.items()})


def train_model(imgs, rows, boxes, valid, dev, seed):
    """Train one detector on the image rows `rows` (images are indexed per batch, never copied)."""
    seed_all(seed)
    model = Det(BACKBONE).to(dev)
    model.train()
    bb_ids = {id(p) for p in model.bb.parameters()}
    opt = torch.optim.AdamW([{"params": list(model.bb.parameters())},
                             {"params": [p for p in model.parameters() if id(p) not in bb_ids]}], lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    n = len(rows)
    steps_per_epoch = n // BATCH
    total = steps_per_epoch * EPOCHS
    warm = max(1, int(0.03 * total))
    ema = EMA(model)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    rng = np.random.RandomState(seed)
    mean, std = MEAN.to(dev), STD.to(dev)
    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        agg = torch.zeros(3, device=dev)
        for it in range(steps_per_epoch):
            idx = torch.from_numpy(rows[perm[it * BATCH:(it + 1) * BATCH]])
            x = imgs[idx].to(dev, non_blocking=True).float() / 255.0
            nb = max(1, int(valid[idx].sum(1).max()))
            b = boxes[idx, :nb].to(dev)
            v = valid[idx, :nb].to(dev)
            x, b, v = gpu_augment(x, b, v, gen)
            x = (x - mean) / std
            lr_f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = LR * lr_f
            with torch.autocast("cuda", dtype=torch.float16):
                loss, parts = compute_loss(model, x, b, v)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model, min(EMA_DECAY, (1 + step) / (10 + step)))
            agg += parts
            step += 1
        a = (agg / steps_per_epoch).tolist()
        log(f"    epoch {ep + 1}/{EPOCHS} focal {a[0]:.4f} giou {a[1]:.4f} lapl {a[2]:.4f} "
            f"({time.time() - t0:.0f}s, {(ep + 1) * steps_per_epoch * BATCH / (time.time() - t0):.1f} img/s)")
    ema.load_into(model)
    model.eval()
    return model


@torch.no_grad()
def predict_maps(model, imgs, dev):
    """Per-image dense outputs averaged over flip TTA.  Every forward pass uses a fixed-shape batch of
    INFER_BATCH images (zero padded), so an image's output never depends on which images share its batch."""
    mean, std = MEAN.to(dev), STD.to(dev)
    H_out, L_out, B_out = [], [], []
    for i in range(0, imgs.shape[0], INFER_BATCH):
        xb = imgs[i:i + INFER_BATCH]
        nreal = xb.shape[0]
        if nreal < INFER_BATCH:
            xb = torch.cat([xb, torch.zeros((INFER_BATCH - nreal,) + tuple(xb.shape[1:]), dtype=xb.dtype)])
        x = (xb.to(dev).float() / 255.0 - mean) / std
        hs = ls = bs = 0
        for t in TTA:
            xi = x.flip(-1) if t == "h" else x
            with torch.autocast("cuda", dtype=torch.float16):
                hm, ltrb, logb = model(xi)
            hm = torch.sigmoid(hm.float())[:, 0]
            ltrb = ltrb.float()
            logb = logb.float()
            if t == "h":
                hm = hm.flip(-1)
                ltrb = ltrb.flip(-1)[:, [2, 1, 0, 3]]
                logb = logb.flip(-1)[:, [2, 1, 0, 3]]
            hs = hs + hm
            ls = ls + ltrb
            bs = bs + logb
        k = float(len(TTA))
        H_out.append((hs / k)[:nreal].cpu())
        L_out.append((ls / k)[:nreal].cpu())
        B_out.append((bs / k)[:nreal].cpu())
    return torch.cat(H_out), torch.cat(L_out), torch.cat(B_out)


def extract_peaks(heat, ltrb, logb):
    """Local maxima of the heatmap (3x3) -> score, peak box, heat-weighted 3x3 voted box, edge scales.
    Boxes are returned in the 0..1024 normalized coordinate system.  Purely per image."""
    N, Hs, Ws = heat.shape
    hm = heat[:, None]
    keep = (hm == F.max_pool2d(hm, 3, 1, 1)).float() * hm
    sc, ind = keep.view(N, -1).topk(TOPK, dim=1)
    boxes = ltrb_to_boxes(ltrb)
    pb = torch.gather(boxes.view(N, -1, 4), 1, ind[..., None].expand(-1, -1, 4))
    pad_b = F.pad(boxes.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
    pad_h = F.pad(heat[:, None], (1, 1, 1, 1))
    iy, ix = ind // Ws, ind % Ws
    num = torch.zeros(N, TOPK, 4)
    den = torch.zeros(N, TOPK)
    ar = torch.arange(N)[:, None]
    for dy in range(3):
        for dx in range(3):
            hv = pad_h[ar, 0, iy + dy, ix + dx]
            num += hv[..., None] * pad_b[ar, :, iy + dy, ix + dx]
            den += hv
    vb = num / den.clamp(min=1e-9)[..., None]
    lb = torch.gather(logb.reshape(N, 4, -1), 2, ind[:, None, :].expand(-1, 4, -1)).permute(0, 2, 1)
    sxy = torch.tensor([1024.0 / IN_W, 1024.0 / IN_H, 1024.0 / IN_W, 1024.0 / IN_H])
    return {"score": sc.numpy().astype(np.float64), "box": (pb * sxy).numpy().astype(np.float64),
            "vbox": (vb * sxy).numpy().astype(np.float64), "eb": (torch.exp(lb) * STRIDE * sxy).numpy().astype(np.float64)}


# ----------------------------------------------------------------------------------------------
# Metric (PROBLEM.md) and crop optimisation
# ----------------------------------------------------------------------------------------------
@numba.njit(cache=False)
def greedy_merge(b, target):
    """Repeatedly merge the pair of groups with the smallest bounding-area increase until `target` groups remain."""
    n = b.shape[0]
    g = b.copy()
    alive = np.ones(n, np.bool_)
    cnt = n
    while cnt > target:
        best = 0
        bi = -1
        bj = -1
        for i in range(n):
            if not alive[i]:
                continue
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                ux0 = min(g[i, 0], g[j, 0]); uy0 = min(g[i, 1], g[j, 1])
                ux1 = max(g[i, 2], g[j, 2]); uy1 = max(g[i, 3], g[j, 3])
                c = ((ux1 - ux0) * (uy1 - uy0) - (g[i, 2] - g[i, 0]) * (g[i, 3] - g[i, 1])
                     - (g[j, 2] - g[j, 0]) * (g[j, 3] - g[j, 1]))
                if bi < 0 or c < best:
                    best = c; bi = i; bj = j
        g[bi, 0] = min(g[bi, 0], g[bj, 0]); g[bi, 1] = min(g[bi, 1], g[bj, 1])
        g[bi, 2] = max(g[bi, 2], g[bj, 2]); g[bi, 3] = max(g[bi, 3], g[bj, 3])
        alive[bj] = False
        cnt -= 1
    return g[alive]


def reference_cost(boxes):
    """D_ref of the metric: documented reference plan built from a box list (used for scoring and as a cost estimate)."""
    if len(boxes) == 0:
        return 0
    b = np.array(sorted(map(tuple, np.asarray(boxes, dtype=np.int64).tolist())), dtype=np.int64)
    g = greedy_merge(b, 3)
    return int(((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).sum())


@numba.njit(cache=False)
def best_partition(b):
    """Exact minimum total bounding-rectangle area partition of n (<=~14) boxes into at most 3 groups."""
    n = b.shape[0]
    N = 1 << n
    ar = np.zeros(N, np.int64)
    x0 = np.empty(N, np.int64); y0 = np.empty(N, np.int64); x1 = np.empty(N, np.int64); y1 = np.empty(N, np.int64)
    x0[0] = 1 << 40; y0[0] = 1 << 40; x1[0] = -(1 << 40); y1[0] = -(1 << 40)
    for m in range(1, N):
        low = m & (-m)
        i = 0
        while (1 << i) != low:
            i += 1
        p = m ^ low
        x0[m] = min(x0[p], b[i, 0]); y0[m] = min(y0[p], b[i, 1])
        x1[m] = max(x1[p], b[i, 2]); y1[m] = max(y1[p], b[i, 3])
        ar[m] = (x1[m] - x0[m]) * (y1[m] - y0[m])
    best2 = np.empty(N, np.int64); arg2 = np.empty(N, np.int64)
    best2[0] = 0; arg2[0] = 0
    for m in range(1, N):
        best2[m] = ar[m]; arg2[m] = m
        low = m & (-m)
        rest = m ^ low
        s = rest
        while True:
            a = s | low
            c = ar[a] + ar[m ^ a]
            if c < best2[m]:
                best2[m] = c; arg2[m] = a
            if s == 0:
                break
            s = (s - 1) & rest
    full = N - 1
    best = best2[full]; ga = 0
    rest = full ^ 1
    s = rest
    while True:
        a = s | 1
        if a != full:
            c = ar[a] + best2[full ^ a]
            if c < best:
                best = c; ga = a
        if s == 0:
            break
        s = (s - 1) & rest
    if ga == 0:
        g1 = arg2[full]
        groups = np.array([g1, full ^ g1])
    else:
        r = full ^ ga
        g2 = arg2[r]
        groups = np.array([ga, g2, r ^ g2])
    out = np.zeros((3, 4), np.int64)
    k = 0
    for gi in range(groups.shape[0]):
        g = groups[gi]
        if g == 0:
            continue
        out[k, 0] = x0[g]; out[k, 1] = y0[g]; out[k, 2] = x1[g]; out[k, 3] = y1[g]
        k += 1
    return out[:k]


def plan_crops(boxes, max_exact=PLAN_EXACT):
    if len(boxes) == 0:
        return np.zeros((0, 4), np.int64)
    b = np.ascontiguousarray(boxes, dtype=np.int64)
    if len(b) > max_exact:
        b = greedy_merge(b, max_exact)
    return best_partition(np.ascontiguousarray(b))


def to_int_boxes(b):
    b = np.round(np.asarray(b, dtype=np.float64)).astype(np.int64).reshape(-1, 4)
    b[:, 0] = np.clip(b[:, 0], 0, 1023)
    b[:, 1] = np.clip(b[:, 1], 0, 1023)
    b[:, 2] = np.clip(b[:, 2], b[:, 0] + 1, 1024)
    b[:, 3] = np.clip(b[:, 3], b[:, 1] + 1, 1024)
    return b


def expand_boxes(b, marg):
    e = np.empty((len(b), 4))
    e[:, 0] = np.floor(b[:, 0] - marg[:, 0]); e[:, 1] = np.floor(b[:, 1] - marg[:, 1])
    e[:, 2] = np.ceil(b[:, 2] + marg[:, 2]); e[:, 3] = np.ceil(b[:, 3] + marg[:, 3])
    e = np.clip(e, 0, 1024).astype(np.int64)
    e[:, 2] = np.maximum(e[:, 2], np.minimum(e[:, 0] + 1, 1024))
    e[:, 0] = np.minimum(e[:, 0], e[:, 2] - 1)
    e[:, 3] = np.maximum(e[:, 3], np.minimum(e[:, 1] + 1, 1024))
    e[:, 1] = np.minimum(e[:, 1], e[:, 3] - 1)
    return e


def nms(b, s, thr):
    order = np.argsort(-s, kind="stable")
    bo = b[order]
    ar = (bo[:, 2] - bo[:, 0]) * (bo[:, 3] - bo[:, 1])
    supp = np.zeros(len(bo), bool)
    keep = []
    for i in range(len(bo)):
        if supp[i]:
            continue
        keep.append(order[i])
        ix = np.clip(np.minimum(bo[i, 2], bo[:, 2]) - np.maximum(bo[i, 0], bo[:, 0]), 0, None)
        iy = np.clip(np.minimum(bo[i, 3], bo[:, 3]) - np.maximum(bo[i, 1], bo[:, 1]), 0, None)
        inter = ix * iy
        supp |= inter / np.maximum(ar[i] + ar - inter, 1e-9) >= thr
    return np.array(keep, dtype=np.int64)


def iou_mat(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :] - inter
    return inter / np.maximum(ua, 1e-9)


def cover_util(crops, ref, dref):
    if len(crops) == 0 or len(ref) == 0:
        return 0.0, 0.0
    inside = ((crops[None, :, 0] <= ref[:, None, 0]) & (crops[None, :, 1] <= ref[:, None, 1])
              & (crops[None, :, 2] >= ref[:, None, 2]) & (crops[None, :, 3] >= ref[:, None, 3])).any(1)
    cov = float(inside.mean())
    d = float(((crops[:, 2] - crops[:, 0]) * (crops[:, 3] - crops[:, 1])).sum())
    return cov, (cov * min(1.0, dref / d) if d > 0 else 0.0)


# ----------------------------------------------------------------------------------------------
# Decoding (all per image; parameters come from the OOF search)
# ----------------------------------------------------------------------------------------------
def candidates(pk, i, src, nm, tmin):
    """Peaks of one image above tmin, NMS'd at IoU nm (1.0 = off), sorted by descending score.
    Greedy NMS only lets higher-scored boxes suppress lower ones, so raising the score threshold
    afterwards keeps a prefix of this list; all thresholds are evaluated from one candidate list."""
    s = pk["score"][i]
    m = s >= tmin
    b, e, ss = pk[src][i][m], pk["eb"][i][m], s[m]
    if nm < 1.0 and len(b):
        k = nms(b, ss, nm)
        b, e, ss = b[k], e[k], ss[k]
    o = np.argsort(-ss, kind="stable")
    return b[o], e[o], ss[o]


@numba.njit(cache=False)
def _greedy_cost3(g, n):
    """Reference-plan cost of the first n rows of g (already in lexicographic order); g is modified."""
    alive = np.ones(n, np.bool_)
    cnt = n
    while cnt > 3:
        best = 0
        bi = -1
        bj = -1
        for i in range(n):
            if not alive[i]:
                continue
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                c = ((max(g[i, 2], g[j, 2]) - min(g[i, 0], g[j, 0])) * (max(g[i, 3], g[j, 3]) - min(g[i, 1], g[j, 1]))
                     - (g[i, 2] - g[i, 0]) * (g[i, 3] - g[i, 1]) - (g[j, 2] - g[j, 0]) * (g[j, 3] - g[j, 1]))
                if bi < 0 or c < best:
                    best = c; bi = i; bj = j
        g[bi, 0] = min(g[bi, 0], g[bj, 0]); g[bi, 1] = min(g[bi, 1], g[bj, 1])
        g[bi, 2] = max(g[bi, 2], g[bj, 2]); g[bi, 3] = max(g[bi, 3], g[bj, 3])
        alive[bj] = False
        cnt -= 1
    s = 0
    for i in range(n):
        if alive[i]:
            s += (g[i, 2] - g[i, 0]) * (g[i, 3] - g[i, 1])
    return s


@numba.njit(cache=False)
def sample_worlds(b, e, p, U, N, kappa):
    """Monte-Carlo worlds for one image.  World s: candidate k is a real region iff U[s,k] < p[k]; its true box
    is the predicted box plus kappa * (predicted Laplace edge scale) * N[s,k,:].  Returns the real mask, the true
    boxes, the number of real regions and the metric's reference cost D_ref of each world."""
    K = b.shape[0]
    S = U.shape[0]
    Z = np.zeros((S, K), np.bool_)
    TB = np.zeros((S, K, 4), np.float64)
    cnt = np.zeros(S, np.int64)
    dref = np.zeros(S, np.float64)
    g = np.empty((K, 4), np.int64)
    idx = np.empty(K, np.int64)
    for s in range(S):
        n = 0
        for k in range(K):
            x0 = b[k, 0] + kappa * e[k, 0] * N[s, k, 0]
            y0 = b[k, 1] + kappa * e[k, 1] * N[s, k, 1]
            x1 = b[k, 2] + kappa * e[k, 2] * N[s, k, 2]
            y1 = b[k, 3] + kappa * e[k, 3] * N[s, k, 3]
            TB[s, k, 0] = min(x0, x1 - 1.0); TB[s, k, 1] = min(y0, y1 - 1.0)
            TB[s, k, 2] = max(x1, x0 + 1.0); TB[s, k, 3] = max(y1, y0 + 1.0)
            if U[s, k] < p[k]:
                Z[s, k] = True
                idx[n] = k
                n += 1
        cnt[s] = n
        if n > 0:
            for a in range(n):
                k = idx[a]
                g[a, 0] = int(np.floor(TB[s, k, 0] + 0.5)); g[a, 1] = int(np.floor(TB[s, k, 1] + 0.5))
                g[a, 2] = int(np.floor(TB[s, k, 2] + 0.5)); g[a, 3] = int(np.floor(TB[s, k, 3] + 0.5))
            for a in range(1, n):  # insertion sort into lexicographic order (reference plan convention)
                r0 = g[a, 0]; r1 = g[a, 1]; r2 = g[a, 2]; r3 = g[a, 3]
                c = a - 1
                while c >= 0:
                    if r0 != g[c, 0]:
                        less = r0 < g[c, 0]
                    elif r1 != g[c, 1]:
                        less = r1 < g[c, 1]
                    elif r2 != g[c, 2]:
                        less = r2 < g[c, 2]
                    else:
                        less = r3 < g[c, 3]
                    if not less:
                        break
                    g[c + 1] = g[c]
                    c -= 1
                g[c + 1, 0] = r0; g[c + 1, 1] = r1; g[c + 1, 2] = r2; g[c + 1, 3] = r3
            dref[s] = _greedy_cost3(g, n)
    return Z, TB, cnt, dref


@numba.njit(cache=False)
def expected_crop_score(crops, Z, TB, cnt, dref, mhat):
    """Monte-Carlo estimate of E[0.2*cov + 0.45*cov*min(1, D_ref/D)] for a crop plan; mhat = expected number of
    real regions the detector missed (they count in T but are assumed uncovered)."""
    S, K = Z.shape
    D = 0.0
    for c in range(crops.shape[0]):
        D += (crops[c, 2] - crops[c, 0]) * (crops[c, 3] - crops[c, 1])
    tot = 0.0
    for s in range(S):
        T = cnt[s] + mhat
        if T <= 0:
            continue
        m = 0
        for k in range(K):
            if Z[s, k]:
                for c in range(crops.shape[0]):
                    if (crops[c, 0] <= TB[s, k, 0] and crops[c, 1] <= TB[s, k, 1]
                            and crops[c, 2] >= TB[s, k, 2] and crops[c, 3] >= TB[s, k, 3]):
                        m += 1
                        break
        cv = m / T
        tot += cv * (0.2 + 0.45 * min(1.0, dref[s] / D))
    return tot / S


@numba.njit(cache=False)
def _expand_nb(b, e, m0, a):
    """Outward-rounded boxes grown by m0 + a * (edge scale), clipped to the 0..1024 frame."""
    K = b.shape[0]
    out = np.empty((K, 4), np.int64)
    for k in range(K):
        x0 = min(max(np.floor(b[k, 0] - (m0 + a * e[k, 0])), 0.0), 1023.0)
        y0 = min(max(np.floor(b[k, 1] - (m0 + a * e[k, 1])), 0.0), 1023.0)
        x1 = min(max(np.ceil(b[k, 2] + (m0 + a * e[k, 2])), x0 + 1.0), 1024.0)
        y1 = min(max(np.ceil(b[k, 3] + (m0 + a * e[k, 3])), y0 + 1.0), 1024.0)
        out[k, 0] = int(x0); out[k, 1] = int(y0); out[k, 2] = int(x1); out[k, 3] = int(y1)
    return out


@numba.njit(cache=False)
def _plan_subset(eb, mask):
    """Optimal <=3 crop partition of the expanded boxes selected by mask."""
    n = 0
    for k in range(mask.shape[0]):
        if mask[k]:
            n += 1
    b = np.empty((n, 4), np.int64)
    c = 0
    for k in range(mask.shape[0]):
        if mask[k]:
            b[c] = eb[k]
            c += 1
    if n > PLAN_EXACT:
        b = greedy_merge(b, PLAN_EXACT)
    return best_partition(b)


@numba.njit(cache=False)
def bayes_best_plan(b, e, p, U, N, kappa, m0, alevels, mhat):
    """Plan of maximum Monte-Carlo expected crop score.  Stage 1: every {top-j detections} x {margin level};
    stage 2: LS_ROUNDS rounds of single-detection toggles (add / drop) around the best stage-1 plan."""
    K = b.shape[0]
    Z, TB, cnt, dref = sample_worlds(b, e, p, U, N, kappa)
    best = -1.0
    bestc = np.zeros((1, 4), np.int64)
    bestc[0, 2] = 1024
    bestc[0, 3] = 1024
    besta = 0
    bestj = 1
    for ai in range(alevels.shape[0]):
        eb = _expand_nb(b, e, m0, alevels[ai])
        for j in range(1, min(K, BAYES_JMAX) + 1):
            mask = np.zeros(K, np.bool_)
            mask[:j] = True
            crops = _plan_subset(eb, mask)
            v = expected_crop_score(crops, Z, TB, cnt, dref, mhat)
            if v > best + 1e-12:
                best = v
                bestc = crops
                besta = ai
                bestj = j
    eb = _expand_nb(b, e, m0, alevels[besta])
    mask = np.zeros(K, np.bool_)
    mask[:bestj] = True
    for r in range(LS_ROUNDS):
        improved = False
        for k in range(K):
            mask[k] = not mask[k]
            nsel = 0
            for q in range(K):
                if mask[q]:
                    nsel += 1
            if nsel > 0:
                crops = _plan_subset(eb, mask)
                v = expected_crop_score(crops, Z, TB, cnt, dref, mhat)
                if v > best + 1e-12:
                    best = v
                    bestc = crops
                    improved = True
                    continue
            mask[k] = not mask[k]
        if not improved:
            break
    return bestc


@numba.njit(parallel=True, cache=False)
def eval_crop_rows(B, E, P, Kn, R, Tn, Dref, U, N, kappa, m0, alevels, mhat, pscale):
    """Coverage and utility of the Bayes plan for every row (rows are independent; parallel over rows)."""
    n = B.shape[0]
    cov = np.zeros(n)
    util = np.zeros(n)
    for r in numba.prange(n):
        K = min(Kn[r], U.shape[1])
        if K == 0:
            crops = np.zeros((1, 4), np.int64)
            crops[0, 2] = 1024
            crops[0, 3] = 1024
        else:
            p = np.minimum(1.0, P[r, :K] * pscale)
            crops = bayes_best_plan(B[r, :K], E[r, :K], p, U, N, kappa, m0, alevels, mhat)
        T = Tn[r]
        m = 0
        for t in range(T):
            for c in range(crops.shape[0]):
                if (crops[c, 0] <= R[r, t, 0] and crops[c, 1] <= R[r, t, 1]
                        and crops[c, 2] >= R[r, t, 2] and crops[c, 3] >= R[r, t, 3]):
                    m += 1
                    break
        D = 0.0
        for c in range(crops.shape[0]):
            D += (crops[c, 2] - crops[c, 0]) * (crops[c, 3] - crops[c, 1])
        if T > 0:
            cov[r] = m / T
            util[r] = cov[r] * min(1.0, Dref[r] / D)
    return cov, util


def fit_calibrator(cands, rows, refs):
    """Logistic model P(peak is a distinct annotated region, IoU>=0.5) from score, edge uncertainty and size,
    fitted on OOF crop candidates of `rows`."""
    X, y = [], []
    for i in rows:
        b, e, ss = cands[i]
        if len(b):
            X.append(region_features(b, e, ss))
            y.append(match_labels(to_int_boxes(b), refs[i], 0.5))
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(np.concatenate(X), np.concatenate(y))


def calibrated(model, b, e, ss):
    if len(b) == 0:
        return np.zeros(0)
    return model.predict_proba(region_features(b, e, ss))[:, 1]  # features use the whole candidate list


def crop_plan(b, e, p, prm):
    """Bayes crop plan for one image (candidates sorted by descending probability)."""
    K = min(len(b), KMAX)
    if K == 0:
        return np.array([[0, 0, 1024, 1024]], np.int64)
    pp = np.minimum(1.0, np.asarray(p[:K], dtype=np.float64) * prm["pscale"])
    plan = bayes_best_plan(np.ascontiguousarray(b[:K], dtype=np.float64), np.ascontiguousarray(e[:K], dtype=np.float64),
                           pp, MC_U, MC_N, float(prm["kappa"]), float(prm["m0"]),
                           np.array(prm["alevels"], dtype=np.float64), float(prm["mhat"]))
    _, u = np.unique(plan, axis=0, return_index=True)
    return plan[np.sort(u)][:3]


def f1_from_iou(iou, P, T):
    if P == 0 or T == 0:
        return 0.0
    tot = 0.0
    for t in (0.5, 0.75):
        adj = (iou >= t).astype(np.float64)
        r, c = linear_sum_assignment(-adj)
        tot += 2 * adj[r, c].sum() / (P + T)
    return tot / 2


def region_candidates(pk, i, src, nm):
    """Region candidates of one image: NMS'd peaks sorted by score, integer boxes, duplicates removed (higher kept)."""
    b, e, ss = candidates(pk, i, src, nm, CAND_TMIN)
    bi = to_int_boxes(b)
    if len(bi):
        _, u = np.unique(bi, axis=0, return_index=True)
        u = np.sort(u)
        b, e, ss, bi = b[u], e[u], ss[u], bi[u]
    return b, e, ss, bi


def region_features(b, e, ss):
    """Features of every detection of ONE image, computed from that image's own candidate list (sorted by
    score): score logit, edge uncertainty, size, relative uncertainty, rank, number of confident detections,
    top score of the image and distance to the nearest other detection."""
    s = np.clip(ss, 1e-4, 1 - 1e-4)
    area = (b[:, 2] - b[:, 0]).clip(min=1) * (b[:, 3] - b[:, 1]).clip(min=1)
    eu = e.mean(1) + 1e-3
    K = len(b)
    rank = np.log1p(np.arange(K))
    nstrong = np.full(K, np.log1p((ss >= 0.3).sum()))
    top = np.full(K, ss.max() if K else 0.0)
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    if K > 1:
        dd = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) + np.eye(K) * 1e9
        nn = np.log1p(dd.min(1))
    else:
        nn = np.full(K, np.log1p(1e3))
    return np.stack([np.log(s / (1 - s)), np.log(eu), np.log(area), np.log(eu / np.sqrt(area)), rank, nstrong, top, nn], 1)


def match_labels(bi, ref, thr):
    """1 if a candidate (in score order) claims a not-yet-claimed reference region with IoU >= thr."""
    y = np.zeros(len(bi))
    if len(bi) == 0 or len(ref) == 0:
        return y
    used = np.zeros(len(ref), bool)
    iou = iou_mat(bi, ref)
    for k in range(len(bi)):
        c = np.where((iou[k] >= thr) & ~used)[0]
        if len(c):
            used[c[np.argmax(iou[k, c])]] = True
            y[k] = 1
    return y


def fit_region_models(rc, rows, refs):
    """Two logistic models on OOF candidates: P(distinct match at IoU>=0.5) and P(distinct match at IoU>=0.75)."""
    X, y5, y75 = [], [], []
    for i in rows:
        b, e, ss, bi = rc[i]
        if len(b):
            X.append(region_features(b, e, ss))
            y5.append(match_labels(bi, refs[i], 0.5))
            y75.append(match_labels(bi, refs[i], 0.75))
    X = np.concatenate(X)
    m5 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, np.concatenate(y5))
    m75 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, np.concatenate(y75))
    return m5, m75


def region_probs(rc, models, rows):
    out = {}
    for i in rows:
        b, e, ss, _ = rc[i]
        if len(b):
            F = region_features(b, e, ss)
            out[i] = (models[0].predict_proba(F)[:, 1], models[1].predict_proba(F)[:, 1])
        else:
            out[i] = (np.zeros(0), np.zeros(0))
    return out


def region_count(q5, q75, prm):
    """Expected-F1 decode: number j of top candidates maximising
    (sum_{k<=j} q5_k + w75 * sum_{k<=j} q75_k) / (j + sum_k q5_k + mhat)."""
    if len(q5) == 0:
        return 0
    J = min(len(q5), 64)
    That = q5.sum() + prm["rmhat"]
    j = np.arange(1, J + 1)
    v = (np.cumsum(q5)[:J] + prm["w75"] * np.cumsum(q75)[:J]) / (j + That)
    return int(np.argmax(v)) + 1


def eval_region(rcs, rps, rows, refs, prm):
    tot = 0.0
    for i in rows:
        _, _, _, bi = rcs[(prm["src"], prm["nms"])][i]
        q5, q75 = rps[(prm["src"], prm["nms"])][i]
        k = region_count(q5, q75, prm)
        tot += f1_from_iou(iou_mat(bi[:k], refs[i]), k, len(refs[i]))
    return tot / len(rows)


def search_region(rcs, rps, rows, refs):
    """Grid search over candidate source / NMS and the two expected-F1 knobs, maximising mean RegionF1."""
    best = None
    for key in rcs:
        ious = {i: iou_mat(rcs[key][i][3][:64], refs[i]) for i in rows}
        for mh in SEARCH_RMHAT:
            for w in SEARCH_W75:
                prm = {"src": key[0], "nms": key[1], "rmhat": mh, "w75": w}
                tot = 0.0
                for i in rows:
                    q5, q75 = rps[key][i]
                    k = region_count(q5, q75, prm)
                    tot += f1_from_iou(ious[i][:k], k, len(refs[i]))
                v = tot / len(rows)
                if best is None or v > best[0] + 1e-12:
                    best = (v, prm)
    return best


def all_region_candidates(pk, rows):
    return {(src, nm): {i: region_candidates(pk, i, src, nm) for i in rows} for src in ("box", "vbox") for nm in SEARCH_NMS}


def all_region_probs(rcs, fit_rows, rows, refs):
    return {key: region_probs(rc, fit_region_models(rc, fit_rows, refs), rows) for key, rc in rcs.items()}


def all_candidates(pk, rows):
    return {nm: {i: candidates(pk, i, "vbox", nm, CAND_TMIN) for i in rows} for nm in SEARCH_CNMS}


def pack_candidates(cands, n):
    """Fixed-size arrays (n, KMAX, ...) of the crop candidates for the parallel evaluator."""
    out = {}
    for nm, cd in cands.items():
        B = np.zeros((n, KMAX, 4)); E = np.zeros((n, KMAX, 4)); S = np.zeros((n, KMAX)); Kn = np.zeros(n, np.int64)
        for i in range(n):
            b, e, ss = cd[i]
            k = min(len(b), KMAX)
            B[i, :k], E[i, :k], S[i, :k], Kn[i] = b[:k], e[:k], ss[:k], k
        out[nm] = (B, E, S, Kn)
    return out


def crop_probs(cands, packs, cal_rows, refs):
    """Calibrated probabilities for all packed candidates; calibrators are fitted on `cal_rows` only."""
    out = {}
    for nm, cd in cands.items():
        model = fit_calibrator(cd, cal_rows, refs)
        P = np.zeros(packs[nm][2].shape)
        for i in range(P.shape[0]):
            b, e, ss = cd[i]
            k = min(len(b), KMAX)
            if k:
                P[i, :k] = calibrated(model, b, e, ss)[:k]
        out[nm] = P
    return out


def pack_refs(refs):
    T = max(len(r) for r in refs)
    R = np.zeros((len(refs), max(T, 1), 4), np.int64)
    Tn = np.zeros(len(refs), np.int64)
    for i, r in enumerate(refs):
        R[i, :len(r)] = r
        Tn[i] = len(r)
    return R, Tn


def eval_crop(packs, probs, rows, R, Tn, drefs, prm):
    B, E, _, Kn = packs[prm["cnms"]]
    P = probs[prm["cnms"]]
    cov, util = eval_crop_rows(B[rows], E[rows], P[rows], Kn[rows], R[rows], Tn[rows], drefs[rows], MC_U, MC_N,
                               float(prm["kappa"]), float(prm["m0"]), np.array(prm["alevels"], dtype=np.float64),
                               float(prm["mhat"]), float(prm["pscale"]))
    return float(0.2 * cov.mean() + 0.45 * util.mean())


def search_crop(packs, probs, rows, R, Tn, drefs, sweeps=2):
    """Coordinate ascent over the Bayes crop-decoder knobs, scored by 0.2*coverage + 0.45*utility."""
    prm = {"cnms": 1.0, "kappa": 1.0, "m0": 0.0, "mhat": 1.0, "pscale": 1.0, "alevels": (0.0, 1.0, 2.0, 3.0)}
    grids = {"kappa": SEARCH_KAPPA, "m0": SEARCH_M0, "mhat": SEARCH_MHAT, "pscale": SEARCH_PSCALE,
             "alevels": SEARCH_ALEVELS, "cnms": SEARCH_CNMS}
    cur = eval_crop(packs, probs, rows, R, Tn, drefs, prm)
    for sweep in range(sweeps):
        for k, grid in grids.items():
            for v in grid:
                if v == prm[k]:
                    continue
                q = dict(prm)
                q[k] = v
                val = eval_crop(packs, probs, rows, R, Tn, drefs, q)
                if val > cur + 1e-12:
                    cur, prm = val, q
        log(f"  crop search sweep {sweep + 1}: {cur:.4f} {prm}")
    return cur, prm


# ----------------------------------------------------------------------------------------------
def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    test_ids = test["case_id"].astype(str).tolist()
    # schema-valid placeholder before any heavy work
    write_submission(submission_out, test_ids, [[] for _ in test_ids], [[] for _ in test_ids])
    log(f"train {len(train)} rows, test {len(test)} rows; placeholder written to {submission_out}")
    log(f"plan: {BACKBONE} | {FOLDS} folds x 1 seed x {EPOCHS} epochs | batch {BATCH} | input {IN_W}x{IN_H} | TTA {TTA}")

    seed_all(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device placement only
    # deterministic kernels: no cuDNN autotuning, deterministic cuDNN / cuBLAS / torch algorithms
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if dev.type == "cuda":
        log(f"device: {torch.cuda.get_device_name(0)}")

    refs = [to_int_boxes(parse_boxes(s)) for s in train["regions"]]
    drefs = np.array([reference_cost(r) for r in refs], dtype=np.float64)
    tr_imgs = load_images([public_dir / p for p in train["image_path"]])
    te_imgs = load_images([public_dir / p for p in test["image_path"]])
    boxes, valid = boxes_to_tensor([r.astype(np.float32) for r in refs])
    log(f"images loaded: train {tuple(tr_imgs.shape)} test {tuple(te_imgs.shape)}")

    n = len(train)
    fold_of = np.random.RandomState(SEED).permutation(n) % FOLDS
    oof = {k: np.zeros((n, TOPK) + s, np.float64) for k, s in (("score", ()), ("box", (4,)), ("vbox", (4,)), ("eb", (4,)))}
    te_heat = te_ltrb = te_logb = None
    for f in range(FOLDS):
        tri = np.where(fold_of != f)[0]
        vai = np.where(fold_of == f)[0]
        log(f"fold {f + 1}/{FOLDS}: train {len(tri)} / held-out {len(vai)}")
        t0 = time.time()
        model = train_model(tr_imgs, tri, boxes, valid, dev, SEED + 17 * f)
        log(f"  fold {f + 1} trained in {time.time() - t0:.0f}s (observational only)")
        pk = extract_peaks(*predict_maps(model, tr_imgs[vai], dev))
        for k in oof:
            oof[k][vai] = pk[k]
        h, l, b = predict_maps(model, te_imgs, dev)
        te_heat = h if te_heat is None else te_heat + h
        te_ltrb = l if te_ltrb is None else te_ltrb + l
        te_logb = b if te_logb is None else te_logb + b
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log(f"  fold {f + 1} done")
    te_pk = extract_peaks(te_heat / FOLDS, te_ltrb / FOLDS, te_logb / FOLDS)

    # ---------------- decode search on OOF (train only) ----------------
    rows = np.arange(n)
    cands = all_candidates(oof, rows)
    packs = pack_candidates(cands, n)
    R, Tn = pack_refs(refs)
    rcs = all_region_candidates(oof, rows)
    halves = [rows[np.random.RandomState(SEED + 1).permutation(n) % 2 == h] for h in (0, 1)]
    held = []
    for h in (0, 1):
        a, bh = halves[h], halves[1 - h]
        probs = crop_probs(cands, packs, a, refs)         # calibrators see half a only
        rps = all_region_probs(rcs, a, rows, refs)
        rv, rp = search_region(rcs, rps, a, refs)
        cv, cp = search_crop(packs, probs, a, R, Tn, drefs, sweeps=1)
        hr = eval_region(rcs, rps, bh, refs, rp)
        hc = eval_crop(packs, probs, bh, R, Tn, drefs, cp)
        held.append(0.35 * hr + hc)
        log(f"  half {h}: searched-on {0.35 * rv + cv:.4f} -> held-out {0.35 * hr + hc:.4f} (RegionF1 {hr:.4f}, crop part {hc:.4f})")
    log(f"OOF held-out estimate (fit+search on one half, score the other) mean: {np.mean(held):.4f}")
    probs = crop_probs(cands, packs, rows, refs)
    rps = all_region_probs(rcs, rows, rows, refs)
    rv, rp = search_region(rcs, rps, rows, refs)
    cv, cp = search_crop(packs, probs, rows, R, Tn, drefs)
    log(f"final decode params region={rp} crop={cp}")
    log(f"OOF in-sample score with final params: {0.35 * rv + cv:.4f} (RegionF1 {rv:.4f}, crop part {cv:.4f})")
    calib = fit_calibrator(cands[cp["cnms"]], rows, refs)
    rkey = (rp["src"], rp["nms"])
    rmodels = fit_region_models(rcs[rkey], rows, refs)

    # ---------------- test decode (per image) ----------------
    regions, crops = [], []
    for i in range(len(test)):
        try:
            b, e, ss, bi = region_candidates(te_pk, i, rp["src"], rp["nms"])
            if len(b):
                F = region_features(b, e, ss)
                k = region_count(rmodels[0].predict_proba(F)[:, 1], rmodels[1].predict_proba(F)[:, 1], rp)
            else:
                k = 0
            r = bi[:k]
            b, e, ss = candidates(te_pk, i, "vbox", cp["cnms"], CAND_TMIN)
            c = crop_plan(b, e, calibrated(calib, b, e, ss), cp)
        except Exception as ex:  # never let one row kill the run
            log(f"WARNING: decode failed for row {i}: {ex}")
            r, c = np.zeros((0, 4), np.int64), np.array([[0, 0, 1024, 1024]], np.int64)
        regions.append(r.tolist())
        crops.append(c.tolist())
    write_submission(submission_out, test_ids, regions, crops)

    sub = pd.read_csv(submission_out)
    ok = (list(sub.columns) == ["case_id", "regions", "crop_plan"] and len(sub) == len(test)
          and sub["case_id"].astype(str).nunique() == len(test) and set(sub["case_id"].astype(str)) == set(test_ids))
    nreg = [len(json.loads(s)) for s in sub["regions"]]
    ncrop = [len(json.loads(s)) for s in sub["crop_plan"]]
    log(f"submission written: {len(sub)} rows, schema ok={ok}, regions/img mean {np.mean(nreg):.2f}, "
        f"crops/img mean {np.mean(ncrop):.2f}, max crops {max(ncrop)}")
    log("done")


if __name__ == "__main__":
    main()
