#!/usr/bin/env python3
"""
infer_only.py — build submission.csv from an ALREADY-TRAINED checkpoint (no training)
=====================================================================================
Use this when you stopped training early (Ctrl-C) and just want predictions from the
best checkpoint saved by solution.py. It reuses solution.py's exact inference path
(domain router + TTA + Weighted Box Fusion + conservative negative gate + format-safe
submission), so results are identical to a full run's inference stage.

Requirements: run it in the SAME folder as solution.py (it imports it), with a checkpoint
present at _work/runs/detector/weights/best.pt (solution.py saves this automatically).

Run (on the pod):
    DEVICE=0 python infer_only.py
    # optional: point at a specific checkpoint
    DEVICE=0 python infer_only.py _work/runs/detector/weights/best.pt
    # optional: override inference resolution (higher = better + slower)
    INFER_IMGSZ=1600 BLUE_IMGSZ=1920 PNG_IMGSZ=1440 DEVICE=0 python infer_only.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

import solution as S   # reuse the whole pipeline (must be in the same directory)

# --- inference resolution: fast-but-good defaults; override via env if you have time ---
S.CFG.INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ", 1536))   # warm 1632px domain (~80% of test)
S.CFG.BLUE_IMGSZ = int(os.environ.get("BLUE_IMGSZ", 1536))     # blue 640px domain: upscale tiny rods
S.CFG.PNG_IMGSZ = int(os.environ.get("PNG_IMGSZ", 1280))       # green-vignette .png domain


def find_checkpoint():
    # 1) explicit path as argv[1] or env CKPT
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
    env = os.environ.get("CKPT")
    if env and Path(env).exists():
        return Path(env)
    # 2) the standard location solution.py writes to
    wdir = S.WORK / "runs" / "detector" / "weights"
    for name in ("best.pt", "last.pt"):
        if (wdir / name).exists():
            return wdir / name
    # 3) last resort: any best.pt under _work/runs
    cands = sorted((S.WORK / "runs").glob("**/weights/best.pt"))
    if cands:
        return cands[-1]
    cands = sorted((S.WORK / "runs").glob("**/weights/last.pt"))
    if cands:
        return cands[-1]
    return None


def main():
    S.ensure_deps()          # make sure ultralytics/ensemble-boxes/etc are present
    from ultralytics import YOLO

    ckpt = find_checkpoint()
    if ckpt is None:
        print("ERROR: no checkpoint found. Expected _work/runs/detector/weights/best.pt")
        print("       (run solution.py first, or pass a path: python infer_only.py <best.pt>)")
        sys.exit(1)
    S.log(f"Loading checkpoint: {ckpt}")
    S.log(f"Inference imgsz -> warm={S.CFG.INFER_IMGSZ} blue={S.CFG.BLUE_IMGSZ} png={S.CFG.PNG_IMGSZ}")
    model = YOLO(str(ckpt))

    test_df = pd.read_csv(S.DATA / "test.csv")
    rows = []
    for n, (_, r) in enumerate(test_df.iterrows(), 1):
        path = S.DATA / r["file_name"]
        try:
            boxes, confs, _pil = S.predict_image(model, path)
            fb, fc, count = S.postprocess(boxes, confs, None)   # detector-only count (no regressor)
            pred_boxes = S.format_boxes(fb, fc)
        except Exception as e:
            S.log(f"  inference failed for {r['image_id']} ({e}); using count prior")
            pred_boxes, count = "", S.CFG.COUNT_PRIOR
        rows.append([r["image_id"], pred_boxes, round(float(count), 3)])
        if n % 100 == 0:
            S.log(f"  {n}/{len(test_df)} predicted")

    S.write_submission(rows)

    # quick sanity summary
    sub = pd.DataFrame(rows, columns=["image_id", "pred_boxes", "pred_count"])
    n_empty = int((sub["pred_boxes"] == "").sum())
    n_boxes = int(sub["pred_boxes"].apply(lambda s: 0 if s == "" else s.count(";") + 1).sum())
    S.log(f"Summary: {len(sub)} images | mean pred_count={sub['pred_count'].mean():.2f} "
          f"| total boxes={n_boxes} | images with no boxes={n_empty}")
    S.log("DONE.")


if __name__ == "__main__":
    main()
