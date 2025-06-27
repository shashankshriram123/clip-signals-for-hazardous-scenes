#!/usr/bin/env python3
"""
eval_prompt_usefulness.py
--------------------------------------------------------------------
For every scene:
• Treat the last prompt row (Row -1) as the **ground-truth prompt**.
• Compute its Temporal IoU (threshold + optional smoothing).
• Among all *other* rows, find the one with the highest IoU
  → call it the **CLIP top prediction**.
• Record: scene, GT prompt, top prompt, match flag, IoU_GT, IoU_top.
Outputs:
• CSV  : results/prompt_usefulness.csv
• Stats: Top-1 accuracy, mean IoU(GT), mean IoU(top), ΔIoU.
"""

# ─── configuration ──────────────────────────────────────────────────
BASE          = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
THRESH        = 0.20         # similarity threshold
SMOOTH_WIN    = 5            # frames for smoothing (0 → off)
GT_ROWS       = 1            # how many GT rows at bottom (1 → only target_hazard)
# -------------------------------------------------------------------

import csv, json, numpy as np
from pathlib import Path

SCORE_DIR = Path(BASE) / "clip_scores"
ANNOT_DIR = Path(BASE) / "annotations"
OUT_DIR   = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH  = OUT_DIR / "prompt_usefulness.csv"

# ─── helpers ────────────────────────────────────────────────────────
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1] + 1] for g in groups]

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/union if union else 0.0

def row_iou(scores_row, gt_int):
    mask = scores_row > THRESH
    if SMOOTH_WIN:
        mask = np.convolve(mask.astype(int),
                           np.ones(SMOOTH_WIN, int), "same") > 0
    preds = mask_to_intervals(mask)
    if not preds: return 0.0
    ious = [max(iou_1d(g, p) for p in preds) for g in gt_int]
    return float(np.mean(ious))

# ─── process every scene ────────────────────────────────────────────
records = []
for npy_path in sorted(SCORE_DIR.glob("scene_*.npy")):
    stem = npy_path.stem
    ann  = ANNOT_DIR / f"{stem}.mp4.json"
    prom_json = SCORE_DIR / f"{stem}_prompts.json"
    if not ann.exists() or not prom_json.exists(): continue

    scores  = np.load(npy_path)                 # [P,F]
    prompts = json.load(open(prom_json))["prompts"]
    gt_int  = [[e["start"], e["end"]] for e in json.load(open(ann))["events"]]

    P, _ = (scores[:, None] if scores.ndim == 1 else scores).shape

    # ---------------- Ground-truth row (last)
    gt_row_idx = -1
    iou_gt     = row_iou(scores[gt_row_idx], gt_int)
    gt_prompt  = prompts[gt_row_idx]

    # ---------------- Best non-GT row
    best_iou, best_idx = 0.0, None
    for r in range(P - GT_ROWS):
        iou = row_iou(scores[r], gt_int)
        if iou > best_iou:
            best_iou, best_idx = iou, r
    top_prompt = prompts[best_idx] if best_idx is not None else "—"
    match      = top_prompt == gt_prompt

    records.append([
        stem, gt_prompt, top_prompt, match, round(iou_gt, 3), round(best_iou, 3)
    ])

# ─── write CSV ──────────────────────────────────────────────────────
header = ["scene", "gt_prompt", "top_prompt", "match", "IoU_gt", "IoU_top"]
with open(CSV_PATH, "w", newline="") as f:
    csv.writer(f).writerow(header)
    csv.writer(f).writerows(records)
print(f"✅  saved {CSV_PATH}")

# ─── summary metrics ────────────────────────────────────────────────
matches      = sum(r[3] for r in records)
total        = len(records)
mean_iou_gt  = np.mean([r[4] for r in records])
mean_iou_top = np.mean([r[5] for r in records])
delta        = mean_iou_top - mean_iou_gt

print("\nSummary")
print("-------")
print(f"Scenes evaluated      : {total}")
print(f"Top-1 prompt accuracy : {matches/total:.2%}  ({matches}/{total})")
print(f"Mean IoU (GT prompt)  : {mean_iou_gt:.3f}")
print(f"Mean IoU (Top prompt) : {mean_iou_top:.3f}")
print(f"Δ IoU (Top − GT)      : {delta:+.3f}")

