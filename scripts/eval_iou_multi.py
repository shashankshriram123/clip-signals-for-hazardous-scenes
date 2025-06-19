import os, glob, json, argparse, numpy as np
from pathlib import Path

ANNOT_DIR  = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/annotations"
SCORE_DIR  = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/clip_scores"
THRESH     = 0.35        # similarity > THRESH ⇒ hazard frame

# ---------- helpers ---------------------------------------------------------
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1]+1] for g in groups]

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter / union if union else 0.

# ---------- CLI -------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="each",
        choices=["single", "each", "best", "any"],
        help="IoU aggregation mode")
ap.add_argument("--row", type=int, default=0,
        help="Row index if --mode single")
ap.add_argument("--thresh", type=float, default=THRESH)
args = ap.parse_args()

# ---------- evaluation loop --------------------------------------------------
results = {}              # video → list[IoU] (per-row or aggregate)
prompt_lookup = {}        # video → list[str]

for npy_path in sorted(Path(SCORE_DIR).glob("video_*.npy")):
    stem   = npy_path.stem                 # video_0024
    ann_js = Path(ANNOT_DIR) / f"{stem}.mp4.json"
    if not ann_js.exists(): continue

    scores = np.load(npy_path)             # shape [P,F]
    if scores.ndim == 1: scores = scores[None, :]   # fallback single row
    P, F = scores.shape

    # load prompts for nice printing
    prom_path = npy_path.with_name(f"{npy_path.stem}_prompts.json")
    if prom_path.exists():
        prompt_lookup[stem] = json.load(open(prom_path))["prompts"]
    else:
        prompt_lookup[stem] = [f"row{i}" for i in range(P)]

    # ground-truth intervals
    gt = [[e["start"], e["end"]] for e in json.load(open(ann_js))["events"]]

    # prepare masks per mode
    if args.mode == "single":
        mask = scores[args.row] > args.thresh
        preds = mask_to_intervals(mask)
        ious  = [max(iou_1d(g, p) for p in preds) if preds else 0. for g in gt]
        results.setdefault(stem, []).append(np.mean(ious))

    elif args.mode == "each":
        for r in range(P):
            preds = mask_to_intervals(scores[r] > args.thresh)
            ious  = [max(iou_1d(g, p) for p in preds) if preds else 0. for g in gt]
            results.setdefault(stem, []).append(np.mean(ious))

    elif args.mode == "best":
        row_best = []
        for g in gt:
            best = 0
            for r in range(P):
                preds = mask_to_intervals(scores[r] > args.thresh)
                best = max(best, *(iou_1d(g, p) for p in preds) if preds else [0])
            row_best.append(best)
        results[stem] = [np.mean(row_best)]

    elif args.mode == "any":
        mask_any = (scores > args.thresh).any(axis=0)
        preds = mask_to_intervals(mask_any)
        ious  = [max(iou_1d(g, p) for p in preds) if preds else 0. for g in gt]
        results[stem] = [np.mean(ious)]

# ---------- print summary ----------------------------------------------------
print("\n=== IoU results ===")
mean_all = []
for vid, vals in results.items():
    if args.mode == "each":
        for idx, val in enumerate(vals):
            name = prompt_lookup[vid][idx]
            print(f"{vid}  {name:40s}  IoU={val:.3f}")
    else:
        print(f"{vid}  IoU={vals[0]:.3f}")
    mean_all.extend(vals)
print(f"\nOverall mean IoU: {np.mean(mean_all):.3f}")
