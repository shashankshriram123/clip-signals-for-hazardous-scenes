#!/usr/bin/env python3
"""
eval_iou_table_dynamic_csv.py
↳ Computes IoU for *all* prompt rows and dumps a wide CSV instead of a box table.
"""
import csv, json, numpy as np
from pathlib import Path

BASE      = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
ANNOT_DIR = Path(BASE) / "annotations"
SCORE_DIR = Path(BASE) / "clip_scores"
OUT_DIR   = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH  = OUT_DIR / "iou_table_dynamic.csv"
THRESH    = 0.20

# ─── helpers ─────────────────────────────────────────────────────────────
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1] + 1] for g in groups]

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0

# ─── determine max #rows & master prompt list ───────────────────────────
prompt_master = []
for pjson in SCORE_DIR.glob("*_prompts.json"):
    prompts = json.load(open(pjson))["prompts"]
    if len(prompts) > len(prompt_master):
        prompt_master = prompts
P_MAX = len(prompt_master)

# ─── compute IoUs & collect rows ────────────────────────────────────────
csv_rows = []
for npy_path in sorted(SCORE_DIR.glob("scene_*.npy")):
    stem = npy_path.stem
    ann  = ANNOT_DIR / f"{stem}.mp4.json"
    if not ann.exists():
        continue

    scores = np.load(npy_path)
    if scores.ndim == 1:
        scores = scores[None, :]
    P, _ = scores.shape

    gt = [[e["start"], e["end"]] for e in json.load(open(ann))["events"]]

    def row_iou(mask):
        preds = mask_to_intervals(mask)
        vals  = [max(iou_1d(g, p) for p in preds) if preds else 0. for g in gt]
        return np.mean(vals) if vals else 0.0

    row_vals = [
        row_iou(scores[r] > THRESH) if r < P else 0.0
        for r in range(P_MAX)
    ]
    csv_rows.append([stem] + row_vals)

# ─── write CSV ──────────────────────────────────────────────────────────
header = ["video"] + [f"Row{i+1}" for i in range(P_MAX)]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_rows)

print(f"✅  Saved IoU table to {CSV_PATH}")

# ─── optional: save legend ──────────────────────────────────────────────
LEGEND_PATH = OUT_DIR / "iou_table_legend.txt"
with open(LEGEND_PATH, "w") as f:
    f.write(f"THRESH = {THRESH}\n")
    for i, prompt in enumerate(prompt_master):
        f.write(f"Row {i+1:2}: {prompt}\n")
print(f"ℹ️  Legend saved to {LEGEND_PATH}")
