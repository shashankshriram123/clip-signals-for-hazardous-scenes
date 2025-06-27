#!/usr/bin/env python3
"""
threshold_sweep.py  –  sweep T, maximise mean IoU, pretty-print table.

Identical math to the original:
  • union-row mask  (scores > T).any(axis=0)
  • optional temporal smoothing (--smooth N)
  • mean IoU over all scenes

Extras:
  • boxed table in terminal, best row coloured green
  • CSV still saved to results/threshold_sweep.csv
"""

import argparse, csv, json, numpy as np
from pathlib import Path

# ─── CLI ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--base",
                default="/home/sshriram2/mi3Testing/hazard_detection_CLIP")
ap.add_argument("--min",    type=float, default=0.10)
ap.add_argument("--max",    type=float, default=0.40)
ap.add_argument("--step",   type=float, default=0.01)
ap.add_argument("--smooth", type=int,   default=5,
                help="temporal smoothing window (frames, 0 = off)")
args = ap.parse_args()

SCORE_DIR = Path(args.base) / "clip_scores"
ANNOT_DIR = Path(args.base) / "annotations"
OUT_CSV   = Path("results/threshold_sweep.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

# ─── helpers (unchanged math) ────────────────────────────────────
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1] + 1] for g in groups]

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.

def mean_iou_any(scores, gt, T):
    mask = (scores > T).any(axis=0)
    if args.smooth:
        mask = np.convolve(mask.astype(int),
                           np.ones(args.smooth, int), "same") > 0
    preds = mask_to_intervals(mask)
    ious  = [max(iou_1d(g, p) for p in preds) if preds else 0. for g in gt]
    return np.mean(ious) if ious else 0.

# ─── sweep ───────────────────────────────────────────────────────
rows = []
for T in np.arange(args.min, args.max + 1e-9, args.step):
    vals = []
    for npy in SCORE_DIR.glob("scene_*.npy"):
        stem = npy.stem
        ann  = ANNOT_DIR / f"{stem}.mp4.json"
        if not ann.exists(): continue
        scores = np.load(npy)
        if scores.ndim == 1: scores = scores[None, :]
        gt = [[e["start"], e["end"]]
              for e in json.load(open(ann))["events"]]
        vals.append(mean_iou_any(scores, gt, T))
    rows.append([round(T, 3), float(np.mean(vals))])

# ─── save CSV (same as before) ───────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["threshold", "mean_IoU"])
    csv.writer(f).writerows(rows)
print(f"✅  wrote {OUT_CSV}")

# ─── pretty table print ──────────────────────────────────────────
best_T, best_IoU = max(rows, key=lambda r: r[1])

class ANSI:
    GRN = "\033[32m"
    BLD = "\033[1m"
    END = "\033[0m"

col_w = [max(len(str(c)) for c in col) for col in zip(*rows, ("Thresh", "Mean IoU"))]
top = "┌" + "┬".join("─" * (w + 2) for w in col_w) + "┐"
sep = "├" + "┼".join("─" * (w + 2) for w in col_w) + "┤"
bot = "└" + "┴".join("─" * (w + 2) for w in col_w) + "┘"

print("\nThreshold Sweep (scene_*)")
print(top)
print("│ " + " │ ".join(f"{h:^{w}}" for h, w in zip(("Thresh", "Mean IoU"), col_w)) + " │")
print(sep)
for T, IoU in rows:
    txt = f"{T:.2f}", f"{IoU:.3f}"
    line = "│ " + " │ ".join(f"{t:>{w}}" for t, w in zip(txt, col_w)) + " │"
    if T == best_T:
        line = ANSI.GRN + ANSI.BLD + line + ANSI.END
    print(line)
print(bot)
print(f"\nBest threshold by mean IoU → {ANSI.BLD}{best_T:.2f}{ANSI.END} "
      f"(IoU = {best_IoU:.3f})\n")
