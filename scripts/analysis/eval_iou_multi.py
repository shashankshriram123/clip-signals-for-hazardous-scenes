#!/usr/bin/env python3
import argparse, json, numpy as np
from pathlib import Path

# ─── paths & defaults ──────────────────────────────────────────────────────
BASE       = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
ANNOT_DIR  = Path(BASE) / "annotations"
SCORE_DIR  = Path(BASE) / "clip_scores"
THRESH_DEF = 0.35

# ─── helpers ───────────────────────────────────────────────────────────────
def mask_to_intervals(mask):
    """Convert boolean frame mask -> list[[start, end)) intervals."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1] + 1] for g in groups]

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0

# ─── CLI ───────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="each",
               choices=["single", "each", "best", "any"],
               help="IoU aggregation mode")
ap.add_argument("--row", type=int, default=0,
               help="Row index (used only with --mode single)")
ap.add_argument("--thresh", type=float, default=THRESH_DEF,
               help="Similarity > thresh ⇒ hazard frame")
ap.add_argument("--pattern", default="scene_*.npy",
               help="Glob pattern for score files inside clip_scores/")
args = ap.parse_args()

# ─── evaluation loop ───────────────────────────────────────────────────────
results, prompt_lookup = {}, {}

score_files = sorted(SCORE_DIR.glob(args.pattern))
if not score_files:
    print(f"No .npy files matched pattern {args.pattern!r} in {SCORE_DIR}")
    exit()

for npy_path in score_files:
    stem   = npy_path.stem                       # e.g. scene_003
    ann_js = ANNOT_DIR / f"{stem}.mp4.json"
    if not ann_js.exists():
        print(f"⚠️  Missing annotation for {stem}, skipping.")
        continue

    scores = np.load(npy_path)                   # [P, F]
    if scores.ndim == 1:
        scores = scores[None, :]
    P, F = scores.shape

    # prompt labels (for nice printing)
    prom_path = npy_path.with_name(f"{stem}_prompts.json")
    if prom_path.exists():
        prompt_lookup[stem] = json.load(open(prom_path))["prompts"]
    else:
        prompt_lookup[stem] = [f"row{i}" for i in range(P)]

    # ground-truth intervals
    gt = [[e["start"], e["end"]] for e in json.load(open(ann_js))["events"]]

    # ── evaluation modes ────────────────────────────────────────────────
    def row_iou(row_mask):
        preds = mask_to_intervals(row_mask)
        vals  = [max(iou_1d(g, p) for p in preds) if preds else 0.0 for g in gt]
        return np.mean(vals) if vals else 0.0 

    if args.mode == "single":
        if not (0 <= args.row < P):
            raise ValueError(f"--row {args.row} out of range (0-{P-1})")
        results[stem] = [row_iou(scores[args.row] > args.thresh)]

    elif args.mode == "each":
        results[stem] = [
            row_iou(scores[r] > args.thresh) for r in range(P)
        ]

    elif args.mode == "best":
        best_per_gt = []
        for g in gt:
            best = 0.0
            for r in range(P):
                preds = mask_to_intervals(scores[r] > args.thresh)
                best = max(best, *(iou_1d(g, p) for p in preds) if preds else [0])
            best_per_gt.append(best)
        results[stem] = [np.mean(best_per_gt)]

    elif args.mode == "any":
        union_mask = (scores > args.thresh).any(axis=0)
        results[stem] = [row_iou(union_mask)]

# ─── summary ───────────────────────────────────────────────────────────────
print("\n=== IoU results ===")
all_vals = []
for vid, vals in results.items():
    if args.mode == "each":
        for idx, val in enumerate(vals):
            name = prompt_lookup[vid][idx]
            print(f"{vid}  {name:40}  IoU={val:.3f}")
    else:
        print(f"{vid}  IoU={vals[0]:.3f}")
    all_vals.extend([v for v in vals if not np.isnan(v)])

mean_iou = np.mean(all_vals) if all_vals else float("nan")
print(f"\nOverall mean IoU: {mean_iou:.3f}")
