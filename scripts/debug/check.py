#!/usr/bin/env python3
"""
Evaluate pre-computed CLIP scores (5 prompts per frame) against temporal
hazard annotations.  Uses the hazard-specific column if present; otherwise,
uses the best (max) of the three generic animal prompts.
"""

import os, glob, json, csv, numpy as np

ROOT             = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
CLIP_SCORES_DIR  = f"{ROOT}/clip_scores"
ANNOTATIONS_DIR  = f"{ROOT}/annotations"
THRESHOLD        = 0.25        # start here; adjust after a quick run

GENERIC_COLS     = slice(0, 3) # first three columns are always generic

def load_prompts(path: str) -> list[str]:
    return json.load(open(path))["prompts"]

def load_gt_frames(json_path: str):
    with open(json_path) as f:
        ann = json.load(f)
    frames = set()
    for ev in ann["events"]:
        frames.update(range(ev["start"], ev["end"] + 1))
    return frames, ann["target_hazard"]

def evaluate(per_frame: np.ndarray, gt_frames: set[int], thr: float):
    pred = {i for i, s in enumerate(per_frame) if s > thr}

    inter = len(pred & gt_frames)
    union = len(pred | gt_frames) or 1
    iou   = inter / union

    tp, fp, fn = inter, len(pred) - inter, len(gt_frames) - inter
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return iou, prec, rec, f1

rows = []
for npy_path in sorted(glob.glob(f"{CLIP_SCORES_DIR}/video_*.npy")):
    vid       = os.path.splitext(os.path.basename(npy_path))[0]
    prom_path = f"{CLIP_SCORES_DIR}/{vid}_prompts.json"
    ann_path  = f"{ANNOTATIONS_DIR}/{vid}.mp4.json"

    if not os.path.exists(prom_path) or not os.path.exists(ann_path):
        print(f"⚠️  Missing prompts or annotation for {vid}, skipping"); continue

    prompts      = load_prompts(prom_path)          # len == 5
    scores_2d    = np.load(npy_path).astype(float).reshape(-1, len(prompts))
    gt_frames, hazard = load_gt_frames(ann_path)

    # ── choose column(s) ────────────────────────────────────────────────────
    if hazard in prompts:                 # e.g. "dog" is col-3 or col-4
        col_idx   = prompts.index(hazard)
        per_frame = scores_2d[:, col_idx]
    else:                                 # no hazard-specific column → use generic best
        per_frame = scores_2d[:, GENERIC_COLS].max(axis=1)
    # ───────────────────────────────────────────────────────────────────────

    iou, p, r, f1 = evaluate(per_frame, gt_frames, THRESHOLD)
    print(f"{vid}: IoU={iou:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    rows.append(dict(video=vid, iou=round(iou,3),
                     precision=round(p,3), recall=round(r,3), f1=round(f1,3)))

# save summary
out_csv = f"{CLIP_SCORES_DIR}/clip_eval.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print(f"\n✅ Results saved to {out_csv}")
