#!/usr/bin/env python3
"""
viz_timeline_all.py
• Generates timeline_<scene>.png for EVERY scene in clip_scores/
• Generates prompt_mean_IoU.png for all non-answer prompts
Images are written to results/plots/.
"""

# ─── configuration ────────────────────────────────────────────────────
BASE          = "/home/sshriram2/mi3Testing/hazard_detection_CLIP"
THRESH        = 0.20           # similarity threshold
WINDOW        = 5              # smoothing window (frames)
EXCLUDE_LAST  = 2              # bottom-most rows to skip (scene-specific “answers”)
TOP_CURVES    = 3              # curves to show in each timeline
# ---------------------------------------------------------------------

import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

SCORE_DIR = Path(BASE) / "clip_scores"
ANNOT_DIR = Path(BASE) / "annotations"
CSV_PATH  = Path("results/iou_table_dynamic.csv")
OUT_DIR   = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────
def mask_to_intervals(mask):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [[g[0], g[-1] + 1] for g in groups]

def load_scene(stem):
    scores  = np.load(SCORE_DIR / f"{stem}.npy")
    prompts = json.load(open(SCORE_DIR / f"{stem}_prompts.json"))["prompts"]
    gtfile  = ANNOT_DIR / f"{stem}.mp4.json"
    gt      = []
    if gtfile.exists():
        ann = json.load(open(gtfile))
        gt  = [[e["start"], e["end"]] for e in ann["events"]]
    return scores, prompts, gt

def timeline_plot(stem):
    scores_all, prompts_all, gt = load_scene(stem)

    # drop answer rows
    if EXCLUDE_LAST:
        scores  = scores_all[:-EXCLUDE_LAST]
        prompts = prompts_all[:-EXCLUDE_LAST]
    else:
        scores, prompts = scores_all, prompts_all

    mask   = (scores > THRESH).any(axis=0)
    smooth = np.convolve(mask.astype(int), np.ones(WINDOW, int), "same") > 0
    t      = np.arange(scores.shape[1])

    top_idx = scores.mean(axis=1).argsort()[-TOP_CURVES:][::-1]

    plt.figure(figsize=(14, 4))
    for idx in top_idx:
        plt.plot(t, scores[idx], label=prompts[idx])
    plt.axhline(THRESH, ls="--", label=f"T={THRESH}")
    plt.fill_between(
        t, 0, 1, where=smooth, alpha=0.2,
        transform=plt.gca().get_xaxis_transform(), label="Predicted mask"
    )
    for i, (s, e) in enumerate(gt):
        plt.axvspan(s, e, alpha=0.3, hatch="///",
                    label="GT hazard" if i == 0 else None)

    plt.title(f"CLIP similarity timeline — {stem}")
    plt.xlabel("Frame"); plt.ylabel("Similarity")
    plt.legend(loc="upper right"); plt.tight_layout()

    out = OUT_DIR / f"timeline_{stem}.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  • saved {out}")

# ─── 1. timeline PNG for each scene ───────────────────────────────────
print("Generating timeline plots …")
for npy in sorted(SCORE_DIR.glob("scene_*.npy")):
    timeline_plot(npy.stem)

# ─── 2. global prompt-mean bar chart ──────────────────────────────────
print("Generating global prompt bar chart …")
df = pd.read_csv(CSV_PATH)
prompt_cols = df.columns[1:]                     # drop 'video'
if EXCLUDE_LAST:
    prompt_cols = prompt_cols[:-EXCLUDE_LAST]

means = df[prompt_cols].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(means.index, means.values)
plt.gca().invert_yaxis()
plt.title("Mean IoU by prompt row (answers excluded)")
plt.xlabel("Mean IoU")
plt.tight_layout()

bar_path = OUT_DIR / "prompt_mean_IoU.png"
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"  • saved {bar_path}")

print("✅  All plots generated in results/plots/")
