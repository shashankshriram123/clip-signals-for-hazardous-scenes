import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────
npy_file = Path("/home/sshriram2/mi3Testing/hazard_detection_CLIP/clip_scores/video_0031.npy")
out_png  = npy_file.with_name(f"{npy_file.stem}_scores.png")   # video_0024_scores.png
thresh   = 0.35

# ─── LOAD & ORIENT ──────────────────────────────────────────────────
scores = np.load(npy_file)
if scores.ndim == 1:
    scores = scores[None, :]
elif scores.shape[0] > scores.shape[1]:
    scores = scores.T         # make it [prompts, frames]

print("array shape =", scores.shape, "min/max =", scores.min(), scores.max())

# ─── PLOT ────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
for r in range(scores.shape[0]):
    plt.plot(scores[r], label=f"row {r}")
plt.axhline(thresh, color='r', linestyle='--', label=f"threshold {thresh}")
plt.xlabel("frame index")
plt.ylabel("CLIP similarity")
plt.title(npy_file.stem)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
plt.close()

print(f"✅  saved {out_png}")
