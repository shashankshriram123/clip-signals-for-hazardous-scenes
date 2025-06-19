#!/usr/bin/env python3
"""
Threshold sweep for CLIP score rows 0,1,2.
Outputs:
  • threshold_sweep_rows012.png  (IoU vs. threshold)
  • console printout of best t for each row
"""

import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE       = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
ANNOT_DIR  = Path(BASE) / "annotations"
SCORE_DIR  = Path(BASE) / "clip_scores"

ROWS       = [0, 1, 2]                     # which rows to analyse
T_GRID     = np.arange(0.00, 1.01, 0.01)   # thresholds 0 → 1 step 0.01

# ─── helpers ───────────────────────────────────────────────────────────────
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    gaps = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, gaps)
    return [[g[0], g[-1] + 1] for g in groups]   # half-open intervals [s,e)

def iou_1d(a, b):
    """a, b are [start,end) lists."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0

# ─── preload ground-truth and scores ───────────────────────────────────────
videos = []          # list of (gt_intervals, score_matrix)

for ann_path in sorted(ANNOT_DIR.glob("video_*.json")):
    stem = ann_path.stem.split(".")[0]            # video_0024
    score_path = SCORE_DIR / f"{stem}.npy"
    if not score_path.exists():
        continue

    # ground-truth intervals as [start,end) lists
    gt_json = json.load(open(ann_path))["events"]
    gt = [[ev["start"], ev["end"]] for ev in gt_json]

    # score matrix rows×frames, orient correctly
    sc = np.load(score_path)
    if sc.ndim == 1:
        sc = sc[None, :]                          # shape (1, F)
    elif sc.shape[0] > sc.shape[1]:
        sc = sc.T                                 # ensure (rows, F)

    videos.append((gt, sc))

total_events = sum(len(gt) for gt, _ in videos)

# ─── sweep thresholds ──────────────────────────────────────────────────────
iou_curves = {r: [] for r in ROWS}

for t in T_GRID:
    # mean IoU per row across all videos & events
    for r in ROWS:
        acc = 0.0
        for gt, sc in videos:
            mask  = sc[r] > t
            preds = mask_to_intervals(mask)
            for g in gt:
                best = max((iou_1d(g, p) for p in preds), default=0.0)
                acc += best
        iou_curves[r].append(acc / total_events)

# ─── best threshold per row ────────────────────────────────────────────────
best = {}
for r in ROWS:
    idx = int(np.argmax(iou_curves[r]))
    best[r] = (T_GRID[idx], iou_curves[r][idx])

# ─── plot ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
colors = ["tab:blue", "tab:orange", "tab:green"]
labels = [f"Row {r}" for r in ROWS]

for r, c, lab in zip(ROWS, colors, labels):
    plt.plot(T_GRID, iou_curves[r], color=c,
             label=f"{lab}  (best t={best[r][0]:.2f})")
    plt.axvline(best[r][0], color=c, linestyle="--", alpha=0.4)

plt.xlabel("threshold t")
plt.ylabel("Mean IoU")
plt.title("Threshold sweep (rows 0,1,2)")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_sweep_rows012.png", dpi=150)
plt.close()

# ─── console summary ───────────────────────────────────────────────────────
print("Best thresholds:")
for r in ROWS:
    print(f"  Row {r}:  t = {best[r][0]:.2f}   mean IoU = {best[r][1]:.3f}")
print("Saved plot → threshold_sweep_rows012.png")
