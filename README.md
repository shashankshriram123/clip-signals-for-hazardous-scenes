# CLIP Signals for Hazardous Scenes

> **Goal:** Detect and localise hazardous events in driving videos using CLIP similarity scores.
>
> *Current dataset:* 4 short clips (`video_0024–0031`).
> *Prompts per frame:* 5 (3 generic animal prompts + 2 video‑specific prompts)

---

\## Repository Structure

```
clip-signals-for-hazardous-scenes/
├── annotations/            # ground‑truth JSONs (video_xxxx.mp4.json)
├── clip_scores/            # ⋯.npy scores + prompt lists + evaluation CSV
├── dataset/                # raw .mp4 videos (source only)
├── scripts/                # inference + evaluation utilities
└── graphs/                 # per‑video similarity plots
```

---

\## Pipeline Overview

1. **Frame Scoring** (`run_clip_similarity.py`)

   * For each video, compute CLIP similarity for **five prompts**:

     1. `animal`
     2. `animal crossing the road`
     3. `animal crossing the road unexpectedly`
     4. *video‑specific prompt 1*  (e.g. `dog crossing road`)
     5. *video‑specific prompt 2*  (e.g. `dog`)
   * Save a `(frames × 5)` array → `clip_scores/video_xxxx.npy`
   * Save the prompt list → `video_xxxx_prompts.json`

2. **Evaluation** (`evaluate_clip_scores.py`)

   * Reshape each `.npy` to `(frames, 5)`.
   * **Column choice**
     \* If ground‑truth hazard prompt appears in columns 4–5 → use that column.
     \* Else → take `max(axis=1)` of the first three generic columns.
   * Threshold scores at **T = 0.25** → predicted hazard frames.
   * Compare to ground‑truth hazard interval → compute metrics below.

---

\## What the Metrics Mean

| Metric            | Formula         | Intuition                                            |     |           |   |                                                       |
| ----------------- | --------------- | ---------------------------------------------------- | --- | --------- | - | ----------------------------------------------------- |
| **Precision (P)** |  TP / (TP + FP) | "When I predict a hazard, how often am I right?"     |     |           |   |                                                       |
| **Recall (R)**    |  TP / (TP + FN) | "When a hazard exists, how often do I catch it?"     |     |           |   |                                                       |
| **F1 Score**      |  2PR / (P + R)  | Harmonic mean → high **only** if both P & R are high |     |           |   |                                                       |
| **Temporal IoU**  |                 | pred ∩ gt                                            |  /  | pred ∪ gt |   | Overlap between predicted and true time windows (0–1) |
| **Threshold T**   | –               | Similarity cutoff (0.25). ↓T ⇒ ↑Recall, ↓Precision   |     |           |   |                                                       |

**TP** = true‑positive frames    **FP** = false‑positive frames    **FN** = false‑negative frames

---

\## Current Results (T = 0.25)

| Video    | Temporal IoU | Precision | Recall    | F1 Score  | Notes                                           |
| -------- | ------------ | --------- | --------- | --------- | ----------------------------------------------- |
| **0025** | **0.678**    | 0.757     | **0.867** | **0.808** | Best overall clip                               |
| 0030     | 0.630        | **0.970** | 0.643     | 0.773     | Very high precision; raise recall by lowering T |
| 0031     | 0.484        | 0.949     | 0.497     | 0.652     | Good precision, missed half GT frames           |
| 0024     | 0.444        | 0.829     | 0.488     | 0.615     | Similar to 0031                                 |

CSV version: `clip_scores/clip_eval.csv`.

---

\## Next Steps

1. **Threshold Sweep**

   * Evaluate T ∈ \[0.15 … 0.30] and plot IoU/F1 vs T.
2. **Prompt Refinement**

   * Try using *only* the hazard‑specific column when available.
3. **False‑Negative Inspection**

   * Visualise frames missed in 0024/0031 → adjust prompts or pre‑processing.
4. **Extend Dataset**

   * Add more hazard classes (pedestrians, construction, debris) and videos.

---

\## Quick Usage

```bash
# 1. Create & activate env
conda create -n clip_scores python=3.10 -y
conda activate clip_scores
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git opencv-python matplotlib tqdm numpy pillow

# 2. Generate scores & plots\python scripts/run_clip_similarity.py   # loops over dataset/*.mp4

# 3. Evaluate
python scripts/evaluate_clip_scores.py  # writes clip_scores/clip_eval.csv
```

---

*Maintainer: @shashankshriram123 – feel free to open issues or PRs with improvements!*
