#!/usr/bin/env python3
import os, json, numpy as np
from pathlib import Path

BASE        = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
ANNOT_DIR   = f"{BASE}/annotations"
SCORE_DIR   = f"{BASE}/clip_scores"
THRESH      = 0.22      # global threshold

# ─── helpers ──────────────────────────────────────────────────────────
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1] + 1] for g in groups]  # [start, end)

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0

# ─── gather IoUs per video & row ─────────────────────────────────────
rows = []              # will become table body
for npy_path in sorted(Path(SCORE_DIR).glob("video_*.npy")):
    stem     = npy_path.stem                      # video_0024
    ann_json = Path(ANNOT_DIR) / f"{stem}.mp4.json"
    prom_json= Path(SCORE_DIR) / f"{stem}_prompts.json"
    if not ann_json.exists() or not prom_json.exists():
        continue

    # load score matrix  [rows, frames]
    sc = np.load(npy_path)
    if sc.ndim == 1:
        sc = sc[None, :]
    elif sc.shape[0] > sc.shape[1]:   # transposed
        sc = sc.T

    # ground-truth intervals
    gt = [[e["start"], e["end"]] for e in json.load(open(ann_json))["events"]]

    # IoU for each of the first 5 rows
    def row_iou(r):
        preds = mask_to_intervals(sc[r] > THRESH)
        ious  = [max(iou_1d(g, p) for p in preds) if preds else 0. for g in gt]
        return float(np.mean(ious))

    vals = [row_iou(r) for r in range(5)]          # rows 0…4
    rows.append([stem] + vals)

# ─── pretty table printer ────────────────────────────────────────────
def box_table(headers, body, digits=3):
    body_fmt = [
        [f"{c:.{digits}f}" if isinstance(c, float) else str(c) for c in row]
        for row in body
    ]
    col_w = [
        max(len(h), *(len(r[i]) for r in body_fmt))
        for i, h in enumerate(headers)
    ]
    def pad(txt, w, align="right"):
        if align == "left":   return txt.ljust(w)
        if align == "center": return txt.center(w)
        return txt.rjust(w)

    top    = "┌" + "┬".join("─" * (w + 2) for w in col_w) + "┐"
    sep    = "├" + "┼".join("─" * (w + 2) for w in col_w) + "┤"
    bottom = "└" + "┴".join("─" * (w + 2) for w in col_w) + "┘"

    # header
    header_line = "│ " + " │ ".join(
        pad(h, w, "center") for h, w in zip(headers, col_w)
    ) + " │"

    # data
    data_lines = []
    for row in body_fmt:
        cells = [
            pad(c, w, "left" if i == 0 else "right")
            for i, (c, w) in enumerate(zip(row, col_w))
        ]
        data_lines.append("│ " + " │ ".join(cells) + " │")

    return "\n".join([top, header_line, sep, *data_lines, bottom])

# ─── assemble table & print ─────────────────────────────────────────
headers = ["video", "Row 1", "Row 2", "Row 3", "Row 4", "Row 5"]

# add mean row
col_sum = np.zeros(5)
for r in rows:
    col_sum += r[1:]
mean_row = ["**mean**"] + [round(v, 3) for v in col_sum / len(rows)]

table_output = box_table(headers, rows + [mean_row])
print(table_output)

legend = [
    ("THRESH", THRESH ),
    ("Row 1 ", "animal"),
    ("Row 2 ", "animal crossing the road"),
    ("Row 3 ", "animal crossing the road unexpectedly"),
    ("Row 4 ", "description  (video-specific)"),
    ("Row 5 ", "target_hazard (video-specific)"),
]


for tag, meaning in legend:
    print(f"  {tag:<5} : {meaning}")
