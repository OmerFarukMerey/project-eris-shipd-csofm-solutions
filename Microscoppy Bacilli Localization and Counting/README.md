# Microscopy Bacilli — Localization & Counting

## The problem

These are **microscope images of sputum smears** used to diagnose **tuberculosis (TB)**. TB
bacteria — *bacilli* — appear as small **magenta/pink rod-shaped** objects against a stained
background. For each test image the task is to output:

- **`pred_boxes`** — where the bacilli are (bounding boxes, each with a confidence), and
- **`pred_count`** — how many bacilli are in the image.

You are given 700 labeled training images and must predict on 540 unlabeled test images.

### How it's scored (composite ∈ [0,1], higher is better)

| Weight | Component | Rewards |
|--------|-----------|---------|
| 0.22 | AP50 | precise boxes (IoU ≥ 0.50) |
| 0.13 | AP25 | approximate localization (IoU ≥ 0.25) |
| 0.12 | CountScore | `mean(1 − min(|pred−true|/max(true,3), 1))` |
| 0.10 | BurdenBinMacroF1 | macro-F1 over count bins {0}, {1–3}, {4–10}, {11+} |
| 0.14 | AppearanceShiftTrack | RowScore on **unseen staining/appearance** images |
| 0.14 | AcquisitionShiftTrack | RowScore on **unseen microscope/acquisition** images |
| 0.10 | BurdenExtremeTrack | RowScore on **negative + high-burden** images |
| 0.05 | NegativeImageSpecificity | fraction of empty images predicted with **no boxes** |

`RowScore = 0.55·AP25_image + 0.35·row_count_score + 0.10·negative_clean`.

**What actually drives the score (from reading the weights + the data):**

1. **AP25 quality controls ~0.34 of the total** (0.13 direct + 0.55 of the three RowScore tracks).
   → optimize for **recall and approximate localization**, not just pixel-perfect boxes.
2. **Generalizing to unseen imaging domains ≈ 0.28** — *more than raw AP50*. We verified this is real:
   ~**18% of the test set is a blue Ziehl-Neelsen stain** (mean R−B ≈ −42) that is **absent from
   training** (the small training images are warm/olive, R−B ≈ +44), with tiny ~10–15px rods; and ~2%
   are warm green-vignette `.png` images. → **domain robustness is the central design goal.**
3. **Negatives punish you** across several components, but there are **only 4 negative training images**,
   so "predict empty" must be handled carefully. A false box on an empty image is expensive.
4. **Asymmetry:** wrongly zeroing a *positive* shift image costs `0.55·AP25 + 0.35·count` inside a
   0.28-weighted track; NegativeSpecificity alone is only 0.05. → when unsure, **keep boxes**.
5. **Verified:** a constant count of **4** maximizes the count sub-score (0.506 vs 0.462 for the mean
   7.46), so the uncertain-count prior is 4, not the mean.

## The solution

A **single, robustness-first YOLO11 detector**, plus four safety systems built around the metric:

1. **Format safety (highest ROI).** The exact `pred_boxes` string can't be confirmed from the data, and
   a wrong convention silently zeros ~56% of the score. All formatting funnels through one
   `format_box()` with config flags (normalized/pixels, xyxy/xywh, conf slot, separators) — a wrong
   guess is a one-line change — plus a **GT round-trip self-check** that re-scores ground truth through
   the local grader (must reproduce AP = 1.0). Default = train convention `x1 y1 x2 y2 conf`, normalized.
2. **Synthesize the unseen blue domain.** Offline, we color-transfer training images toward the measured
   blue test palette (per-channel Reinhard) so the model learns "magenta rod on blue background" **with
   real labels** — stronger than blind color jitter. We also paste real rod crops to synthesize crowded
   (high-burden) fields, and add circular-FOV vignette sims for the `.png` domain.
3. **Independent count regressor + prior 4.** An EfficientNet-B0 predicts count directly (not from
   detector boxes), giving a count **floor** when detection under-fires on hard domains. Counts blend
   `0.6·detector + 0.4·regressor` (max on crowded fields). Catastrophic per-image failures fall back to 4.
4. **Conservative two-signal negative gate.** An image is declared empty (no boxes, count 0) only when
   **both** the detector is unconfident **and** the independent count signal is ~0 — protecting
   NegativeSpecificity without wrongly zeroing faint positives.

**Inference** routes each image by measured color/resolution (blue → upscale to 1536 for tiny rods;
png → vignette-aware; warm → full 1600), runs **TTA** (original + h/v flips) merged with **Weighted Box
Fusion** (tightens boxes → helps AP50), then thresholds and counts.

**Reliability:** one model, per-epoch checkpoints, and a wall-clock guard that reserves time for inference
and degrades gracefully — a valid `submission.csv` always gets written.

**Tuning:** the built-in grader (`python solution.py --selfcheck`) re-implements all 8 metric components exactly (validated against anchors:
const-4 → 0.506, perfect → 1.0). Thresholds are chosen on **leave-group-out** folds (leave-appearance-out,
leave-acquisition-out, synthetic-blue) — global values only, never fit on the tiny real groups.

## Files

Everything is in **one self-contained script** plus this explanation.

| File | Purpose |
|------|---------|
| `solution.py` | Full pipeline (data prep → offline aug → train → infer → submit) **and** the exact metric grader, in one file. |
| `README.md` | This problem + solution explanation. |
| `submission.csv` | The single output: `image_id, pred_boxes, pred_count`. |

## How to run

```bash
python solution.py                 # full run (GPU + internet; trains, predicts, writes submission.csv)
SMOKE=1 python solution.py         # tiny fast end-to-end check (not competitive quality)
python solution.py --selfcheck     # validate the built-in grader against known anchors
```

Key env overrides: `EPOCHS, IMGSZ, MODEL, WALL_BUDGET_H, VAL_FRAC, USE_COUNT_REGRESSOR,
N_SYNTH_BLUE, N_RODPASTE, N_VIGNETTE, DEVICE`. Heavy deps (torch, ultralytics, timm,
ensemble-boxes, opencv, albumentations) auto-install on first run.

## What was validated locally (no GPU here)

- Grader anchors reproduce exactly (const-4 → 0.5062; perfect → 1.0; empty+count-4 floor → 0.227).
- Format round-trip → AP 1.0 (submission convention is self-consistent).
- Data prep: stratified split, YOLO labels, negatives → empty-label background images.
- Color transfer: synthetic-blue images land at R−B ≈ −41 (matches the real blue test domain).
- Postprocess + grader integration: a simulated detector scores composite **0.872**, negative gate blanks
  low-confidence images, confident boxes are retained.

The detector/regressor **training and inference** require the GPU + internet scoring environment; that code
is written against stable Ultralytics/timm APIs and reviewed, but was not executed on this machine.
