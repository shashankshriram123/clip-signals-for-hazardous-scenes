#!/usr/bin/env python3
"""
threshold_by_type.py  –  find best T per hazard category
"""

import argparse, json, numpy as np
from pathlib import Path

# ─── CLI ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--base",
                default="/home/sshriram2/mi3Testing/hazard_detection_CLIP")
ap.add_argument("--min",    type=float, default=0.15)
ap.add_argument("--max",    type=float, default=0.35)
ap.add_argument("--step",   type=float, default=0.01)
ap.add_argument("--smooth", type=int,   default=0)
args = ap.parse_args()

SCORE_DIR = Path(args.base) / "clip_scores"
ANNOT_DIR = Path(args.base) / "annotations"

# ─── IoU helpers (unchanged) ──────────────────────────────────────
def mask_to_intervals(m):
    idx = np.where(m)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1]+1] for g in groups]

def iou_1d(a,b):
    inter = max(0, min(a[1],b[1]) - max(a[0],b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/union if union else 0.

def mean_iou(scores, gt, T):
    mask = (scores > T).any(axis=0)
    if args.smooth:
        mask = np.convolve(mask.astype(int),
                           np.ones(args.smooth,int),"same") > 0
    preds = mask_to_intervals(mask)
    return np.mean([
        max(iou_1d(g,p) for p in preds) if preds else 0.
        for g in gt
    ]) if gt else 0.

# ─── collect scenes by type ───────────────────────────────────────
groups = {}   # type → list[(scores, gt)]
for npy in SCORE_DIR.glob("scene_*.npy"):
    stem = npy.stem
    ann  = ANNOT_DIR / f"{stem}.mp4.json"
    if not ann.exists(): continue
    info = json.load(open(ann))

    typ  = info.get("type","unknown").split()[0].lower()
    # skip "normal-no-hazard" scenes in optimisation
    if typ.startswith("normal"): 
        continue

    sc   = np.load(npy)
    if sc.ndim == 1: sc = sc[None,:]
    gt   = [[e["start"], e["end"]] for e in info["events"]]

    groups.setdefault(typ, []).append((sc, gt))

# ─── sweep per type ───────────────────────────────────────────────
print("\nOptimal threshold per hazard category")
for typ, items in sorted(groups.items()):
    best_T, best_IoU = None, -1
    for T in np.arange(args.min, args.max+1e-9, args.step):
        IoU = np.mean([mean_iou(sc, gt, T) for sc, gt in items])
        if IoU > best_IoU:
            best_IoU, best_T = IoU, T
    print(f"  {typ:<12s}  T = {best_T:.2f}   mean IoU = {best_IoU:.3f}   "
          f"(n={len(items)})")

