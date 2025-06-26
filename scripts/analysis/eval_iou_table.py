#!/usr/bin/env python3
# eval_iou_table_5.py  – fixed 5-row summary
import json, numpy as np
from pathlib import Path

BASE       = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
ANNOT_DIR  = Path(BASE) / "annotations"
SCORE_DIR  = Path(BASE) / "clip_scores"
THRESH     = 0.20

# ---------- helpers -------------------------------------------------
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1]+1] for g in groups]

def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter / union if union else 0.0

# ---------- gather IoUs --------------------------------------------
rows = []
for npy in SCORE_DIR.glob("scene_*.npy"):
    stem = npy.stem
    ann  = ANNOT_DIR / f"{stem}.mp4.json"
    if not ann.exists(): continue

    sc = np.load(npy)
    if sc.ndim == 1: sc = sc[None, :]
    sc = sc[:5]                       # keep first 5 rows

    gt = [[e["start"], e["end"]] for e in json.load(open(ann))["events"]]
    vals = []
    for r in range(sc.shape[0]):
        preds = mask_to_intervals(sc[r] > THRESH)
        ious  = [max(iou_1d(g,p) for p in preds) if preds else 0. for g in gt]
        vals.append(float(np.mean(ious)))
    rows.append([stem] + vals)

# ---------- table printer ------------------------------------------
def box_table(headers, body):
    bodyf = [[f"{c:.3f}" if isinstance(c,float) else str(c) for c in row]
             for row in body]
    col_w = [max(len(h),*(len(r[i]) for r in bodyf)) for i,h in enumerate(headers)]
    pad   = lambda t,w,a="right": t.ljust(w) if a=="left" else t.rjust(w)
    border = lambda c: c + c.join("─"*(w+2) for w in col_w) + c
    lines  = [border("┌"), "│ "+ " │ ".join(pad(h,w,"center") for h,w in zip(headers,col_w))+ " │", border("├")]
    for row in bodyf:
        lines.append("│ "+ " │ ".join(pad(c,w,"left" if i==0 else "right") for i,(c,w) in enumerate(zip(row,col_w))) +" │")
    lines.append(border("└"))
    return "\n".join(lines)

headers = ["video","Row1","Row2","Row3","Row4","Row5"]
table   = box_table(headers, rows)
print(table)

legend = [
    ("THRESH", THRESH),
    ("Row1", "animal"),
    ("Row2", "animal crossing the road"),
    ("Row3", "animal crossing the road unexpectedly"),
    ("Row4", "description (video-specific)"),
    ("Row5", "target_hazard (video-specific)")
]
print("\nLegend:")
for k,v in legend:
    print(f"  {k:<6}: {v}")
