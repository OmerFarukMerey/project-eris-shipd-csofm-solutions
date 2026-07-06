#!/usr/bin/env python3
"""
solution.py — Microscopy Bacilli Localization & Counting  (single self-contained script)
=========================================================================================
Trains a domain-robust detector on the 700 labeled images and writes ONE `submission.csv`
(`image_id, pred_boxes, pred_count`) for the 540 test images, optimizing the composite
metric (re-implemented below in the LOCAL GRADER section). Designed for a GPU+internet
"train-each-run" environment; heavy deps auto-install. Pure-PIL/numpy helpers (data prep,
color transfer, box format, grader) run without torch/cv2 so they can be tested locally.

Pipeline (see README): synthetic-blue + rod-paste + vignette offline augmentation → YOLO11
detector (wall-clock guarded, per-epoch checkpoints) → optional independent count regressor
→ domain-routed TTA + Weighted-Box-Fusion inference → count blend + conservative two-signal
negative gate → format-safe submission with a ground-truth round-trip self-check.

Run:
    python solution.py               # full run: train, predict, write submission.csv
    SMOKE=1 python solution.py       # tiny fast end-to-end check (not competitive quality)
    python solution.py --selfcheck   # validate the built-in grader against known anchors
Env overrides: EPOCHS, IMGSZ, MODEL, WALL_BUDGET_H, VAL_FRAC, USE_COUNT_REGRESSOR,
               N_SYNTH_BLUE, N_RODPASTE, N_VIGNETTE, DEVICE
"""
from __future__ import annotations

import os
import sys
import time
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ======================================================================================
# CONFIG
# ======================================================================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "public"
WORK = ROOT / "_work"                 # scratch: yolo dataset, runs, checkpoints
# Shipd harness expects the predictions at ./working/submission.csv. We write there
# (relative to CWD *and* to the script dir) plus a root copy, to be safe either way.
SUBMISSION = ROOT / "working" / "submission.csv"

T0 = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"


def _envf(name, default):
    v = os.environ.get(name)
    return type(default)(v) if v is not None else default


class CFG:
    # --- model / training ---
    MODEL = os.environ.get("MODEL", "yolo11l.pt")          # fallback handled in train_detector
    IMGSZ = _envf("IMGSZ", 1280)
    EPOCHS = _envf("EPOCHS", 100)
    BATCH = _envf("BATCH", 16)
    VAL_FRAC = _envf("VAL_FRAC", 0.10)
    WALL_BUDGET_H = _envf("WALL_BUDGET_H", 8.0)            # total wall-clock budget (hours)
    DEVICE = os.environ.get("DEVICE", "")                  # "" -> auto (cuda/mps/cpu)

    # --- offline augmentation volumes ---
    N_SYNTH_BLUE = _envf("N_SYNTH_BLUE", 700)              # synthetic-blue copies of train imgs
    N_RODPASTE = _envf("N_RODPASTE", 200)                 # extra crowded images
    N_VIGNETTE = _envf("N_VIGNETTE", 100)                 # circular-FOV (png domain) sims

    # --- inference ---
    INFER_IMGSZ = _envf("INFER_IMGSZ", 1600)              # warm 1632/1440 domain
    BLUE_IMGSZ = _envf("BLUE_IMGSZ", 1536)                # upscale tiny blue rods
    PNG_IMGSZ = _envf("PNG_IMGSZ", 1440)
    PRED_CONF_FLOOR = 0.001                                # keep recall for AP; threshold later
    NMS_IOU = 0.60
    MAX_DET = 300
    WBF_IOU = 0.55

    # --- thresholds (tuned offline via the grader on leave-group-out folds) ---
    TAU_BOX = _envf("TAU_BOX", 0.05)                      # min conf to include a box in pred_boxes
    TAU_COUNT = _envf("TAU_COUNT", 0.20)                  # min conf to count a box
    TAU_NEG = _envf("TAU_NEG", 0.15)                      # below this max-conf -> candidate empty
    COUNT_PRIOR = 4.0                                      # verified argmax for uncertain count
    HIGH_BURDEN = 8.0                                      # anti-undercount kicks in above this
    BLEND_DET = 0.6                                        # pred_count = 0.6*det + 0.4*reg
    USE_COUNT_REGRESSOR = os.environ.get("USE_COUNT_REGRESSOR", "1") == "1"

    # --- box format (SWAP IN ONE PLACE if the grader wants a different convention) ---
    FMT_NORMALIZED = True          # coords in [0,1] (True) vs pixels (False)
    FMT_ORDER = "xyxy"             # "xyxy" or "xywh" (xywh = x1 y1 w h)
    FMT_CONF_SLOT = "last"         # "last", "first", or "none"
    FMT_SEP_IN_BOX = " "
    FMT_SEP_BETWEEN = ";"
    FMT_DECIMALS = 6

    if SMOKE:                       # tiny fast config for local end-to-end validation
        MODEL = "yolo11n.pt"
        IMGSZ = 320
        EPOCHS = 2
        BATCH = 4
        N_SYNTH_BLUE = 12
        N_RODPASTE = 6
        N_VIGNETTE = 4
        INFER_IMGSZ = 640
        BLUE_IMGSZ = 640
        PNG_IMGSZ = 640
        USE_COUNT_REGRESSOR = False


def log(msg):
    dt = time.time() - T0
    print(f"[{dt/60:6.1f}m] {msg}", flush=True)


def elapsed_h():
    return (time.time() - T0) / 3600.0


# ======================================================================================
# BOX FORMAT  (all submission formatting funnels through here — one-line swap if wrong)
# ======================================================================================
def format_box(x1, y1, x2, y2, conf, W=1, H=1):
    """Format one predicted box into the submission token string, honoring CFG flags."""
    if not CFG.FMT_NORMALIZED:
        x1, x2 = x1 * W, x2 * W
        y1, y2 = y1 * H, y2 * H
    if CFG.FMT_ORDER == "xywh":
        a, b, c, d = x1, y1, (x2 - x1), (y2 - y1)
    else:  # xyxy
        a, b, c, d = x1, y1, x2, y2
    nd = CFG.FMT_DECIMALS
    coords = [f"{a:.{nd}f}", f"{b:.{nd}f}", f"{c:.{nd}f}", f"{d:.{nd}f}"]
    cstr = f"{conf:.{nd}f}"
    if CFG.FMT_CONF_SLOT == "last":
        toks = coords + [cstr]
    elif CFG.FMT_CONF_SLOT == "first":
        toks = [cstr] + coords
    else:
        toks = coords
    return CFG.FMT_SEP_IN_BOX.join(toks)


def format_boxes(boxes, confs, W=1, H=1):
    """boxes: (N,4) normalized xyxy; confs: (N,). -> submission pred_boxes string."""
    if len(boxes) == 0:
        return ""
    return CFG.FMT_SEP_BETWEEN.join(
        format_box(b[0], b[1], b[2], b[3], c, W, H) for b, c in zip(boxes, confs)
    )


# ======================================================================================
# PARSING
# ======================================================================================
def parse_gt_boxes(s):
    """Parse a 'boxes' string -> (N,4) array of normalized [x1,y1,x2,y2]."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return np.zeros((0, 4), dtype=np.float64)
    out = [list(map(float, b.split())) for b in str(s).split(";") if b.strip()]
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def parse_pred_boxes(s):
    """Parse a predicted 'pred_boxes' string 'x1 y1 x2 y2 conf;...' -> ((N,4), (N,) conf)."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    boxes, confs = [], []
    for b in str(s).split(";"):
        b = b.strip()
        if not b:
            continue
        vals = list(map(float, b.split()))
        boxes.append(vals[:4])
        confs.append(vals[4] if len(vals) >= 5 else 1.0)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(confs, dtype=np.float64)


# ======================================================================================
# LOCAL GRADER  (faithful re-implementation of the composite metric; pure numpy)
# --------------------------------------------------------------------------------------
# Used offline to (a) validate the submission format via a GT round-trip, and (b) tune the
# global thresholds on leave-group-out proxy folds of the *training* data. Composite:
#   0.22*AP50 + 0.13*AP25 + 0.12*CountScore + 0.10*BurdenBinMacroF1
#   + 0.14*AppearanceShift + 0.14*AcquisitionShift + 0.10*BurdenExtreme + 0.05*NegSpecificity
# ======================================================================================
WEIGHTS = {
    "AP50": 0.22, "AP25": 0.13, "CountScore": 0.12, "BurdenBinMacroF1": 0.10,
    "AppearanceShiftTrackScore": 0.14, "AcquisitionShiftTrackScore": 0.14,
    "BurdenExtremeTrackScore": 0.10, "NegativeImageSpecificity": 0.05,
}


def iou_matrix(a, b):
    """IoU between boxes a:(N,4) and b:(M,4), all [x1,y1,x2,y2]. Returns (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _voc_ap(rec, prec):
    """VOC all-points AP: area under the precision envelope."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _tp_fp_for_image(boxes, confs, gts, iou_thr):
    """Greedy per-image matching. Returns (tp, fp) aligned to conf-sorted preds, sorted
    confs, and #GTs. Each GT matched at most once."""
    order = np.argsort(-confs)
    boxes = boxes[order]
    tp = np.zeros(len(boxes), dtype=np.float64)
    fp = np.zeros(len(boxes), dtype=np.float64)
    if len(gts) == 0:
        fp[:] = 1.0
        return tp, fp, confs[order], 0
    ious = iou_matrix(boxes, gts)
    matched = np.zeros(len(gts), dtype=bool)
    for i in range(len(boxes)):
        j = int(np.argmax(ious[i])) if ious.shape[1] else -1
        if j >= 0 and ious[i, j] >= iou_thr and not matched[j]:
            tp[i] = 1.0
            matched[j] = True
        else:
            fp[i] = 1.0
    return tp, fp, confs[order], len(gts)


def average_precision(per_image, iou_thr):
    """Pooled (dataset-level) VOC AP; matching is per-image, PR curve over all preds by conf."""
    all_conf, all_tp, all_fp, npos = [], [], [], 0
    for im in per_image:
        tp, fp, conf, m = _tp_fp_for_image(im["boxes"], im["confs"], im["gts"], iou_thr)
        all_conf.append(conf); all_tp.append(tp); all_fp.append(fp); npos += m
    if npos == 0:
        return 1.0 if sum(len(c) for c in all_conf) == 0 else 0.0
    conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0)
    fp = np.concatenate(all_fp) if all_fp else np.zeros(0)
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp_cum, fp_cum = np.cumsum(tp[order]), np.cumsum(fp[order])
    rec = tp_cum / npos
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return _voc_ap(rec, prec)


def per_image_ap(boxes, confs, gts, iou_thr, empty_gt_value=1.0):
    """Per-image VOC AP (used inside RowScore). Empty-GT conventions:
    no GT & no preds -> empty_gt_value; no GT & preds -> 0.0; GT present -> standard AP."""
    if len(gts) == 0:
        return empty_gt_value if len(boxes) == 0 else 0.0
    if len(boxes) == 0:
        return 0.0
    tp, fp, _, npos = _tp_fp_for_image(boxes, confs, gts, iou_thr)
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rec = tp_cum / npos
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return _voc_ap(rec, prec)


def row_count_score(pred_count, true_count):
    err = abs(float(pred_count) - float(true_count))
    denom = max(float(true_count), 3.0)
    return 1.0 - min(err / denom, 1.0)


def count_score(pred_counts, true_counts):
    pred_counts = np.asarray(pred_counts, dtype=np.float64)
    true_counts = np.asarray(true_counts, dtype=np.float64)
    err = np.abs(pred_counts - true_counts)
    denom = np.maximum(true_counts, 3.0)
    return float(np.mean(1.0 - np.minimum(err / denom, 1.0)))


def burden_bin(counts):
    """0=negative(0), 1=low(1-3), 2=medium(4-10), 3=high(>=11). Counts are rounded first."""
    x = np.round(np.asarray(counts, dtype=np.float64)).astype(int)
    return np.where(x <= 0, 0, np.where(x <= 3, 1, np.where(x <= 10, 2, 3)))


def burden_bin_macro_f1(pred_counts, true_counts):
    """Macro-F1 over the 4 burden bins (pure numpy; == sklearn macro, zero_division=0)."""
    yt, yp = burden_bin(true_counts), burden_bin(pred_counts)
    f1s = []
    for cl in (0, 1, 2, 3):
        tp = np.sum((yp == cl) & (yt == cl))
        fp = np.sum((yp == cl) & (yt != cl))
        fn = np.sum((yp != cl) & (yt == cl))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s))


def row_score(boxes, confs, gts, pred_count, true_count, iou_thr=0.25, empty_gt_value=1.0):
    """RowScore = 0.55*AP25_image + 0.35*row_count_score + 0.10*negative_clean."""
    ap = per_image_ap(boxes, confs, gts, iou_thr, empty_gt_value=empty_gt_value)
    rcs = row_count_score(pred_count, true_count)
    is_neg = float(true_count) == 0.0
    negative_clean = 1.0 if (not is_neg) or (is_neg and len(boxes) == 0) else 0.0
    return 0.55 * ap + 0.35 * rcs + 0.10 * negative_clean


def evaluate(df, preds_by_image, empty_gt_value=1.0):
    """Full composite + breakdown. df needs image_id, target_count, boxes (and optionally
    _is_appearance_shift / _is_acquisition_shift bool columns for the shift-track proxies).
    preds_by_image: {image_id: {'boxes':(N,4), 'confs':(N,), 'count':float}}."""
    per_image, rows = [], []
    for _, r in df.iterrows():
        iid = r["image_id"]
        gts = parse_gt_boxes(r.get("boxes"))
        p = preds_by_image.get(iid, {"boxes": np.zeros((0, 4)), "confs": np.zeros((0,)), "count": 0.0})
        boxes = np.asarray(p["boxes"], float).reshape(-1, 4)
        confs = np.asarray(p["confs"], float).reshape(-1)
        per_image.append({"boxes": boxes, "confs": confs, "gts": gts})
        rows.append({
            "image_id": iid, "true_count": float(r["target_count"]),
            "pred_count": float(p.get("count", len(boxes))), "n_pred": len(boxes),
            "row_score": row_score(boxes, confs, gts, p.get("count", len(boxes)),
                                   r["target_count"], 0.25, empty_gt_value),
            "app_shift": bool(r.get("_is_appearance_shift", False)),
            "acq_shift": bool(r.get("_is_acquisition_shift", False)),
            "is_neg": float(r["target_count"]) == 0.0, "is_high": float(r["target_count"]) >= 11.0,
        })
    rd = pd.DataFrame(rows)
    ap50 = average_precision(per_image, 0.50)
    ap25 = average_precision(per_image, 0.25)
    cs = count_score(rd["pred_count"].values, rd["true_count"].values)
    mf1 = burden_bin_macro_f1(rd["pred_count"].values, rd["true_count"].values)
    app = rd.loc[rd["app_shift"], "row_score"].mean() if rd["app_shift"].any() else rd["row_score"].mean()
    acq = rd.loc[rd["acq_shift"], "row_score"].mean() if rd["acq_shift"].any() else rd["row_score"].mean()
    ext_mask = rd["is_neg"] | rd["is_high"]
    ext = rd.loc[ext_mask, "row_score"].mean() if ext_mask.any() else rd["row_score"].mean()
    neg = rd[rd["is_neg"]]
    neg_spec = float((neg["n_pred"] == 0).mean()) if len(neg) else 1.0
    comp = {
        "AP50": ap50, "AP25": ap25, "CountScore": cs, "BurdenBinMacroF1": mf1,
        "AppearanceShiftTrackScore": float(app), "AcquisitionShiftTrackScore": float(acq),
        "BurdenExtremeTrackScore": float(ext), "NegativeImageSpecificity": neg_spec,
    }
    comp["COMPOSITE"] = float(sum(WEIGHTS[k] * comp[k] for k in WEIGHTS))
    return comp


def selfcheck(train_csv=None):
    """Validate the grader against known anchors (run: python solution.py --selfcheck)."""
    df = pd.read_csv(train_csv or (DATA / "train.csv"))
    t = df["target_count"].values
    print("== CountScore for constant predictions ==")
    for c in [3, 4, 5, 6, 7, 7.464, 8]:
        print(f"  const {c:>6}: CountScore={count_score(np.full_like(t, c, float), t):.4f}"
              f"  MacroF1={burden_bin_macro_f1(np.full_like(t, c, float), t):.4f}")
    grid = np.arange(0, 20, 0.1)
    best = max(grid, key=lambda c: count_score(np.full_like(t, c, float), t))
    print(f"  argmax constant CountScore = {best:.1f} (expected ~4.0)")

    print("\n== Perfect prediction (GT as preds, conf=1) ==")
    preds = {r["image_id"]: {"boxes": parse_gt_boxes(r.get("boxes")),
                             "confs": np.ones(len(parse_gt_boxes(r.get("boxes")))),
                             "count": float(r["target_count"])} for _, r in df.iterrows()}
    comp = evaluate(df, preds)
    for k, v in comp.items():
        print(f"  {k:28} {v:.4f}")
    assert comp["COMPOSITE"] > 0.999, "perfect composite should be ~1.0"

    print("\n== Empty boxes + constant count 4 (a smart do-nothing floor) ==")
    empty = {r["image_id"]: {"boxes": np.zeros((0, 4)), "confs": np.zeros((0,)), "count": 4.0}
             for _, r in df.iterrows()}
    for k, v in evaluate(df, empty).items():
        print(f"  {k:28} {v:.4f}")
    print("\nAnchor checks passed.")


# ======================================================================================
# DATA PREP  (PIL/numpy only)
# ======================================================================================
def xyxy_to_yolo(boxes):
    """normalized [x1,y1,x2,y2] -> normalized [cx,cy,w,h] (YOLO label format)."""
    if len(boxes) == 0:
        return np.zeros((0, 4))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=1)


def write_label(path, boxes):
    yb = xyxy_to_yolo(np.clip(boxes, 0, 1))
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in yb if w > 0 and h > 0]
    Path(path).write_text("\n".join(lines))


def stratified_split(df, val_frac, seed=42):
    """Stratify val by appearance x acquisition x burden bin; keep all negatives in train."""
    rng = random.Random(seed)
    df = df.copy()
    df["_bin"] = pd.cut(df["target_count"], [-1, 0, 3, 10, 10_000], labels=[0, 1, 2, 3])
    by_id = df.set_index("image_id")
    val_ids = []
    for _, grp in df.groupby(["appearance_group", "acquisition_group", "_bin"], observed=True):
        ids = [i for i in grp["image_id"].tolist() if float(by_id.loc[i, "target_count"]) > 0]
        rng.shuffle(ids)
        val_ids.extend(ids[: int(round(len(ids) * val_frac))])
    return set(val_ids)


# ======================================================================================
# COLOR TRANSFER + OFFLINE AUGMENTATION  (PIL/numpy only)
# ======================================================================================
def img_channel_stats(path, max_side=256):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    a = np.asarray(im, dtype=np.float64).reshape(-1, 3)
    return a.mean(0), a.std(0)


def compute_blue_reference(test_df, sample=40):
    """Per-channel mean/std over blue (R-B<0) test images -> Reinhard target."""
    blue, paths = [], [DATA / p for p in test_df["file_name"].tolist()]
    random.Random(0).shuffle(paths)
    for p in paths:
        if not p.exists():
            continue
        try:
            m, _ = img_channel_stats(p)
        except Exception:
            continue
        if m[0] - m[2] < 0:
            blue.append(p)
        if len(blue) >= sample:
            break
    if not blue:
        return np.array([120.0, 150.0, 185.0]), np.array([45.0, 45.0, 45.0])
    means, stds = zip(*[img_channel_stats(p) for p in blue])
    return np.mean(means, axis=0), np.mean(stds, axis=0)


def reinhard_to_ref(arr, ref_mean, ref_std):
    """Per-channel mean/std color transfer of an image toward the reference palette."""
    flat = arr.reshape(-1, 3).astype(np.float64)
    sm, ss = flat.mean(0), flat.std(0) + 1e-6
    out = (arr.astype(np.float64) - sm) / ss * ref_std + ref_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def make_synthetic_blue(df, ref_mean, ref_std, out_img_dir, out_lbl_dir, n):
    """Color-transfer n train images toward the blue domain, keeping their real labels."""
    ids = df["image_id"].tolist()
    random.Random(1).shuffle(ids)
    idx = df.set_index("image_id")
    made = 0
    for iid in ids:
        if made >= n:
            break
        r = idx.loc[iid]
        src = DATA / r["file_name"]
        if not src.exists():
            continue
        try:
            arr = np.asarray(Image.open(src).convert("RGB"))
        except Exception:
            continue
        name = f"synblue_{iid}"
        Image.fromarray(reinhard_to_ref(arr, ref_mean, ref_std)).save(out_img_dir / f"{name}.jpg", quality=92)
        write_label(out_lbl_dir / f"{name}.txt", parse_gt_boxes(r.get("boxes")))
        made += 1
    return made


def _crop_rods(df, pool=400):
    """Collect a pool of GT rod crops (arrays) for copy-paste."""
    idx = df.set_index("image_id")
    ids = df["image_id"].tolist()
    random.Random(2).shuffle(ids)
    crops = []
    for iid in ids:
        r = idx.loc[iid]
        src, boxes = DATA / r["file_name"], parse_gt_boxes(r.get("boxes"))
        if not src.exists() or len(boxes) == 0:
            continue
        try:
            im = Image.open(src).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        for x1, y1, x2, y2 in boxes:
            px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            if px2 - px1 >= 4 and py2 - py1 >= 4:
                crops.append(np.asarray(im.crop((px1, py1, px2, py2))))
                if len(crops) >= pool:
                    return crops
    return crops


def make_rod_paste(df, out_img_dir, out_lbl_dir, n):
    """Paste extra rod crops onto base images to synthesize crowded (high-burden) fields."""
    crops = _crop_rods(df)
    if not crops:
        return 0
    idx = df.set_index("image_id")
    ids = df["image_id"].tolist()
    rng = random.Random(3)
    made = 0
    for _ in range(n * 3):
        if made >= n:
            break
        r = idx.loc[rng.choice(ids)]
        src = DATA / r["file_name"]
        if not src.exists():
            continue
        try:
            im = Image.open(src).convert("RGB").copy()
        except Exception:
            continue
        W, H = im.size
        boxes = list(parse_gt_boxes(r.get("boxes")))
        for _ in range(rng.randint(3, 12)):
            c = Image.fromarray(crops[rng.randrange(len(crops))])
            if rng.random() < 0.5:
                c = c.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                c = c.transpose(Image.FLIP_TOP_BOTTOM)
            c = c.rotate(rng.choice([0, 90, 180, 270]), expand=True)
            cw, ch = c.size
            if cw >= W or ch >= H:
                continue
            px, py = rng.randint(0, W - cw), rng.randint(0, H - ch)
            im.paste(c, (px, py))
            boxes.append([px / W, py / H, (px + cw) / W, (py + ch) / H])
        name = f"rodpaste_{made:05d}"
        im.save(out_img_dir / f"{name}.jpg", quality=92)
        write_label(out_lbl_dir / f"{name}.txt", np.asarray(boxes).reshape(-1, 4))
        made += 1
    return made


def make_vignette(df, out_img_dir, out_lbl_dir, n):
    """Circular field-of-view (dark corners) sims for the .png green-vignette test domain."""
    idx = df.set_index("image_id")
    ids = df["image_id"].tolist()
    rng = random.Random(4)
    made = 0
    for _ in range(n * 3):
        if made >= n:
            break
        r = idx.loc[rng.choice(ids)]
        src = DATA / r["file_name"]
        if not src.exists():
            continue
        try:
            im = Image.open(src).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        mask = Image.new("L", (W, H), 0)
        rad = int(min(W, H) * rng.uniform(0.48, 0.55))
        cx, cy = W // 2, H // 2
        ImageDraw.Draw(mask).ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
        out = Image.composite(im, Image.new("RGB", (W, H), (0, 0, 0)), mask)
        keep = []
        for x1, y1, x2, y2 in parse_gt_boxes(r.get("boxes")):
            bx, by = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H
            if (bx - cx) ** 2 + (by - cy) ** 2 <= (rad * 0.95) ** 2:
                keep.append([x1, y1, x2, y2])
        name = f"vignette_{made:05d}"
        out.save(out_img_dir / f"{name}.jpg", quality=92)
        write_label(out_lbl_dir / f"{name}.txt", np.asarray(keep).reshape(-1, 4))
        made += 1
    return made


def build_dataset(train_df, test_df):
    """Materialize a YOLO dataset (real + synthetic) and write data.yaml. Returns yaml path."""
    ds = WORK / "yolo_ds"
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (ds / sub).mkdir(parents=True, exist_ok=True)
    img_tr, img_va = ds / "images/train", ds / "images/val"
    lbl_tr, lbl_va = ds / "labels/train", ds / "labels/val"

    val_set = stratified_split(train_df, CFG.VAL_FRAC)
    idx = train_df.set_index("image_id")
    n_tr = n_va = 0
    for iid in train_df["image_id"]:
        r = idx.loc[iid]
        src = DATA / r["file_name"]
        if not src.exists():
            continue
        stem = Path(r["file_name"]).stem
        dst_img_dir, dst_lbl_dir = (img_va, lbl_va) if iid in val_set else (img_tr, lbl_tr)
        link = dst_img_dir / src.name
        if not link.exists():
            try:
                os.symlink(src, link)
            except Exception:
                shutil.copy(src, link)
        write_label(dst_lbl_dir / f"{stem}.txt", parse_gt_boxes(r.get("boxes")))
        n_va += iid in val_set
        n_tr += iid not in val_set

    log("Computing blue reference palette from test images ...")
    ref_mean, ref_std = compute_blue_reference(test_df)
    log(f"  blue ref mean={np.round(ref_mean,1)} std={np.round(ref_std,1)}")
    tr_only = train_df[~train_df.image_id.isin(val_set)]
    nb = make_synthetic_blue(tr_only, ref_mean, ref_std, img_tr, lbl_tr, CFG.N_SYNTH_BLUE)
    nr = make_rod_paste(tr_only, img_tr, lbl_tr, CFG.N_RODPASTE)
    nv = make_vignette(tr_only, img_tr, lbl_tr, CFG.N_VIGNETTE)
    log(f"Dataset: {n_tr} real train + {nb} synth-blue + {nr} rod-paste + {nv} vignette; {n_va} val")

    yaml_path = ds / "data.yaml"
    yaml_path.write_text(f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n  0: bacillus\n")
    return yaml_path


# ======================================================================================
# HEAVY DEPS
# ======================================================================================
def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def ensure_deps():
    need = []
    for mod, pkg in [("cv2", "opencv-python-headless"), ("torch", "torch"),
                     ("torchvision", "torchvision"), ("ultralytics", "ultralytics"),
                     ("ensemble_boxes", "ensemble-boxes"), ("albumentations", "albumentations")]:
        try:
            __import__(mod)
        except Exception:
            need.append(pkg)
    if CFG.USE_COUNT_REGRESSOR:
        try:
            __import__("timm")
        except Exception:
            need.append("timm")
    if need:
        log(f"Installing: {need}")
        pip_install(need)


def get_device():
    if CFG.DEVICE:
        return CFG.DEVICE
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ======================================================================================
# TRAINING
# ======================================================================================
def train_detector(yaml_path):
    from ultralytics import YOLO

    dev = get_device()
    log(f"Training detector {CFG.MODEL} on device={dev} imgsz={CFG.IMGSZ} epochs={CFG.EPOCHS}")
    try:
        model = YOLO(CFG.MODEL)
    except Exception as e:
        log(f"  {CFG.MODEL} unavailable ({e}); falling back to yolo11m.pt")
        model = YOLO("yolo11m.pt")

    train_deadline_h = CFG.WALL_BUDGET_H * 0.80   # reserve ~20% of budget for inference

    def _guard(trainer):
        if elapsed_h() > train_deadline_h:
            log("  wall-clock guard: stopping training to reserve time for inference")
            trainer.stop = True

    try:
        model.add_callback("on_train_epoch_end", _guard)
    except Exception:
        pass

    args = dict(
        data=str(yaml_path), epochs=CFG.EPOCHS, imgsz=CFG.IMGSZ, batch=CFG.BATCH,
        device=dev, project=str(WORK / "runs"), name="detector", exist_ok=True,
        optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=5e-4,
        warmup_epochs=3, patience=max(10, CFG.EPOCHS // 5), close_mosaic=10,
        box=8.0, cls=0.5, dfl=1.5,               # up-weight box loss -> tighter boxes (AP50)
        hsv_h=0.15, hsv_s=0.7, hsv_v=0.4,        # keep the stable magenta cue (not full gray)
        degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.5,
        mosaic=1.0, mixup=0.1, save=True, plots=False, verbose=False, amp=True,
    )
    try:
        model.train(**args)
    except Exception as e:                       # OOM / other -> smaller batch + imgsz
        log(f"  train failed ({e}); retrying smaller")
        args.update(batch=max(4, CFG.BATCH // 2), imgsz=min(CFG.IMGSZ, 1024))
        model.train(**args)

    best = WORK / "runs" / "detector" / "weights" / "best.pt"
    return YOLO(str(best)) if best.exists() else model


def train_count_regressor(train_df, val_set):
    """Independent count regressor (efficientnet_b0, log1p target). Time-gated; returns
    a predict(pil)->float or None if skipped/failed."""
    if not CFG.USE_COUNT_REGRESSOR or elapsed_h() > CFG.WALL_BUDGET_H * 0.85:
        return None
    try:
        import torch
        import torch.nn as nn
        import timm
        from torch.utils.data import Dataset, DataLoader
        import torchvision.transforms as T
    except Exception as e:
        log(f"  count regressor deps missing ({e}); skipping")
        return None

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    idx = train_df.set_index("image_id")
    ids = [i for i in train_df["image_id"] if (DATA / idx.loc[i, "file_name"]).exists()]
    tr_ids = [i for i in ids if i not in val_set]
    RES = 384
    tf = T.Compose([T.Resize((RES, RES)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class DS(Dataset):
        def __init__(self, ids):
            self.ids = ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            r = idx.loc[self.ids[i]]
            im = Image.open(DATA / r["file_name"]).convert("RGB")
            return tf(im), torch.tensor([np.log1p(float(r["target_count"]))], dtype=torch.float32)

    try:
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.HuberLoss()
        dl = DataLoader(DS(tr_ids), batch_size=16, shuffle=True, num_workers=2, drop_last=False)
        epochs = 1 if SMOKE else 12
        log(f"Training count regressor ({len(tr_ids)} imgs, {epochs} ep) on {dev}")
        model.train()
        for ep in range(epochs):
            if elapsed_h() > CFG.WALL_BUDGET_H * 0.88:
                log("  regressor wall-clock guard: stopping")
                break
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                lossf(model(x), y).backward()
                opt.step()
        model.eval()
    except Exception as e:
        log(f"  count regressor training failed ({e}); continuing detector-only")
        return None

    @torch.no_grad()
    def predict(pil):
        x = tf(pil.convert("RGB")).unsqueeze(0).to(dev)
        return max(0.0, float(np.expm1(model(x).cpu().numpy().ravel()[0])))

    return predict


# ======================================================================================
# INFERENCE  (domain router + TTA + WBF)
# ======================================================================================
def _predict_variant(model, pil, imgsz, dev):
    """Run the detector once, return (boxes_xyxy_norm (N,4), confs (N,))."""
    res = model.predict(source=pil, imgsz=imgsz, conf=CFG.PRED_CONF_FLOOR, iou=CFG.NMS_IOU,
                        max_det=CFG.MAX_DET, augment=False, verbose=False, device=dev)
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    return r.boxes.xyxyn.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy().astype(np.float64)


def _nms(boxes, confs, iou_thr):
    if len(boxes) == 0:
        return boxes, confs
    order = np.argsort(-confs)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return boxes[keep], confs[keep]


def _wbf(box_lists, conf_lists, iou_thr):
    """Weighted Box Fusion across TTA variants; graceful fallback to score-sorted NMS."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
        if sum(len(c) for c in conf_lists) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        bl = [np.clip(b, 0, 1).tolist() for b in box_lists]
        sl = [c.tolist() for c in conf_lists]
        ll = [[0] * len(c) for c in conf_lists]
        boxes, scores, _ = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thr, skip_box_thr=0.0)
        return np.asarray(boxes).reshape(-1, 4), np.asarray(scores).reshape(-1)
    except Exception:
        allb = np.concatenate([b for b in box_lists if len(b)], axis=0) if any(len(b) for b in box_lists) else np.zeros((0, 4))
        allc = np.concatenate([c for c in conf_lists if len(c)], axis=0) if any(len(c) for c in conf_lists) else np.zeros((0,))
        return _nms(allb, allc, iou_thr)


def predict_image(model, path):
    """Domain-routed TTA inference for one test image. Returns (boxes_xyxy_norm, confs, pil)."""
    dev = get_device()
    pil = Image.open(path).convert("RGB")
    rmb = float(np.asarray(pil.resize((128, 128)), dtype=np.float64).reshape(-1, 3).mean(0)[[0, 2]] @ [1, -1])
    is_png = str(path).lower().endswith(".png")
    imgsz = CFG.BLUE_IMGSZ if rmb < 0 else (CFG.PNG_IMGSZ if is_png else CFG.INFER_IMGSZ)

    variants = [pil, pil.transpose(Image.FLIP_LEFT_RIGHT), pil.transpose(Image.FLIP_TOP_BOTTOM)]
    box_lists, conf_lists = [], []
    for k, v in enumerate(variants):
        b, c = _predict_variant(model, v, imgsz, dev)
        if len(b):
            if k == 1:                       # undo hflip
                b = b.copy(); b[:, [0, 2]] = 1.0 - b[:, [2, 0]]
            elif k == 2:                     # undo vflip
                b = b.copy(); b[:, [1, 3]] = 1.0 - b[:, [3, 1]]
        box_lists.append(np.clip(b, 0, 1))
        conf_lists.append(c)
    boxes, confs = _wbf(box_lists, conf_lists, CFG.WBF_IOU)
    if len(boxes):
        order = np.argsort(-confs)
        boxes, confs = boxes[order], confs[order]
    return boxes, confs, pil


# ======================================================================================
# POSTPROCESS  (count blend + conservative negative gate + prior)
# ======================================================================================
def postprocess(boxes, confs, reg_count):
    """Return (final_boxes, final_confs, pred_count) for one image."""
    max_conf = float(confs.max()) if len(confs) else 0.0
    det_count = int((confs >= CFG.TAU_COUNT).sum())

    if reg_count is not None and np.isfinite(reg_count):
        base = CFG.BLEND_DET * det_count + (1 - CFG.BLEND_DET) * reg_count
        if base >= CFG.HIGH_BURDEN:                 # anti-undercount on crowded/blue fields
            base = max(base, det_count, reg_count)
    else:
        base = float(det_count)

    # conservative two-signal negative gate: declare empty only when the detector is
    # unconfident AND the independent count signal is ~0 (asymmetry favors keeping boxes).
    reg_says_empty = (reg_count is None) or (reg_count < 0.5)
    if (max_conf < CFG.TAU_NEG) and (det_count == 0) and reg_says_empty:
        return np.zeros((0, 4)), np.zeros((0,)), 0.0

    keep = confs >= CFG.TAU_BOX
    fb, fc = boxes[keep], confs[keep]
    count = max(base, 1.0)                           # not gated empty -> at least one bacillus
    for edge in (3.5, 10.5):                         # nudge off burden-bin rounding boundaries
        if abs(count - edge) < 0.15:
            count = edge + (0.2 if count >= edge else -0.2)
    return fb, fc, float(count)


# ======================================================================================
# SUBMISSION
# ======================================================================================
def _parse_pred_string(s):
    """Inverse of format_boxes for the round-trip check (respects CFG flags)."""
    if not s:
        return np.zeros((0, 4)), np.zeros((0,))
    boxes, confs = [], []
    for tok in s.split(CFG.FMT_SEP_BETWEEN):
        vals = list(map(float, tok.split(CFG.FMT_SEP_IN_BOX)))
        if CFG.FMT_CONF_SLOT == "first":
            conf, coords = vals[0], vals[1:5]
        elif CFG.FMT_CONF_SLOT == "last":
            coords, conf = vals[0:4], (vals[4] if len(vals) >= 5 else 1.0)
        else:
            coords, conf = vals[0:4], 1.0
        if CFG.FMT_ORDER == "xywh":
            x1, y1, w, h = coords
            coords = [x1, y1, x1 + w, y1 + h]
        boxes.append(coords)
        confs.append(conf)
    return np.asarray(boxes).reshape(-1, 4), np.asarray(confs).reshape(-1)


def gt_round_trip_check(train_df, n=50):
    """Format GT as predictions and score via the grader; must reproduce ~1.0. Guards
    against a wrong box-format convention silently zeroing the detection score."""
    sub = train_df.head(n)
    preds = {}
    for _, r in sub.iterrows():
        g = parse_gt_boxes(r.get("boxes"))
        bb, cc = _parse_pred_string(format_boxes(g, np.ones(len(g))))
        preds[r["image_id"]] = {"boxes": bb, "confs": cc, "count": float(r["target_count"])}
    comp = evaluate(sub, preds)
    ok = comp["AP50"] > 0.999 and comp["AP25"] > 0.999
    log(f"GT round-trip self-check: AP50={comp['AP50']:.4f} AP25={comp['AP25']:.4f} "
        f"-> {'OK' if ok else 'FORMAT MISMATCH!'}")
    return ok


def write_submission(rows):
    df = pd.DataFrame(rows, columns=["image_id", "pred_boxes", "pred_count"])
    targets = []
    for p in (Path.cwd() / "working" / "submission.csv",   # ./working/ relative to CWD (Shipd)
              ROOT / "working" / "submission.csv",          # ./working/ next to the script
              ROOT / "submission.csv"):                      # root copy (backup / convenience)
        if p not in targets:
            targets.append(p)
    for p in targets:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            log(f"Wrote {p} ({len(df)} rows)")
        except Exception as e:
            log(f"  could not write {p}: {e}")


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    WORK.mkdir(exist_ok=True)
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    log(f"train={len(train_df)} test={len(test_df)}  SMOKE={SMOKE}")

    if not gt_round_trip_check(train_df):
        log("WARNING: box format round-trip failed — check CFG.FMT_* flags before trusting AP.")

    ensure_deps()
    val_set = stratified_split(train_df, CFG.VAL_FRAC)
    yaml_path = build_dataset(train_df, test_df)

    model = train_detector(yaml_path)
    regressor = train_count_regressor(train_df, val_set)

    log("Predicting on test set ...")
    rows = []
    for n, (_, r) in enumerate(test_df.iterrows(), 1):
        path = DATA / r["file_name"]
        try:
            boxes, confs, pil = predict_image(model, path)
            reg_count = None
            if regressor is not None:
                try:
                    reg_count = regressor(pil)
                except Exception:
                    reg_count = None
            fb, fc, count = postprocess(boxes, confs, reg_count)
            pred_boxes = format_boxes(fb, fc)
        except Exception as e:
            log(f"  inference failed for {r['image_id']} ({e}); using count prior")
            pred_boxes, count = "", CFG.COUNT_PRIOR
        rows.append([r["image_id"], pred_boxes, round(float(count), 3)])
        if n % 100 == 0:
            log(f"  {n}/{len(test_df)} predicted")

    write_submission(rows)
    log("DONE.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lstrip("-") in ("selfcheck", "grade", "test"):
        selfcheck()
    else:
        main()
