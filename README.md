# CLIP Signals for Hazardous Scenes

> **Goal:** Detect and localise hazardous events in driving videos using CLIP similarity scores.
> *Original dataset:* 4 short clips (`video_0024–0031`)
> *Prompts per frame:* 5 (3 generic animal prompts + 2 video‑specific prompts)

---

## Repository Structure

```
clip-signals-for-hazardous-scenes/
├── annotations/              # Ground-truth JSONs (scene_xxx.mp4.json)
├── clip_scores/              # .npy similarity scores, prompt lists, eval CSVs
├── dataset/                  # Raw videos (.mp4) – ignored by git (email for access)
├── graphs/                   # Similarity plots & threshold sweep results
├── scripts/
│   ├── run/                  # Main pipeline scripts
│   ├── debug/                # Dev-time utilities (e.g., video splitting)
│   └── setup/                # Setup and dependencies (e.g., CLIP wrappers)
└── hpc_jobs/                 # SLURM job scripts for remote execution
```

---

## Pipeline Overview

1. **Frame Scoring** (`run_clip_similarity.py`)

   * For each video, compute CLIP similarity for **five prompts**:

     1. `animal`
     2. `animal crossing the road`
     3. `animal crossing the road unexpectedly`
     4. *video-specific prompt 1* (e.g. `dog crossing road`)
     5. *video-specific prompt 2* (e.g. `dog`)
   * Save results as:

     * `.npy` array of shape `(frames × 5)`
     * associated `prompts.json`

2. **Evaluation** (`evaluate_clip_scores.py`)

   * Select the most relevant column:

     * If ground-truth hazard uses prompts 4–5, choose that column.
     * Else, use `max(axis=1)` over generic prompts (1–3).
   * Apply threshold `T = 0.25` to predict hazard frames.
   * Compare to ground-truth intervals via **Temporal IoU** and other metrics.

---

## What the Metrics Mean

| Metric            | Formula         | Intuition                                            |
| ----------------- | --------------- | ---------------------------------------------------- |
| **Precision (P)** |  TP / (TP + FP) | "When I predict a hazard, how often am I right?"     |
| **Recall (R)**    |  TP / (TP + FN) | "When a hazard exists, how often do I catch it?"     |
| **F1 Score**      |  2PR / (P + R)  | High only if both precision and recall are high      |
| **Temporal IoU**  |                 | Overlap between predicted and ground-truth intervals |
| **Threshold (T)** | –               | CLIP similarity cutoff (default: 0.25)               |

**TP** = true‑positive frames     **FP** = false‑positive     **FN** = false‑negative

---

## Current Results (T = 0.25)

| Video    | Temporal IoU | Precision | Recall    | F1 Score  | Notes                                           |
| -------- | ------------ | --------- | --------- | --------- | ----------------------------------------------- |
| **0025** | **0.678**    | 0.757     | **0.867** | **0.808** | Best overall clip                               |
| 0030     | 0.630        | **0.970** | 0.643     | 0.773     | Very high precision; raise recall by lowering T |
| 0031     | 0.484        | 0.949     | 0.497     | 0.652     | Good precision, missed half GT frames           |
| 0024     | 0.444        | 0.829     | 0.488     | 0.615     | Similar to 0031                                 |

📄 CSV version: `clip_scores/clip_eval.csv`

---

## Next Steps

1. **Threshold Sweep**

   * Try `T ∈ [0.15 … 0.30]`, visualize IoU and F1

2. **Prompt Refinement**

   * Experiment with better hazard-specific phrasing

3. **False-Negative Inspection**

   * Visualize low-scoring but GT-labeled frames in videos 0024/0031

4. **Dataset Expansion**

   * Add more hazard categories (pedestrian, debris, construction)

---

## Quick Usage

```bash
# 1. Create environment
conda create -n clip_scores python=3.10 -y
conda activate clip_scores

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git opencv-python matplotlib tqdm numpy pillow

# 3. Run scoring pipeline
python scripts/run/run_clip_similarity.py

# 4. Run evaluation
python scripts/evaluate_clip_scores.py
```

---

📩 **Note:** For access to the full dataset (14+ driving clips), email `sshrir2@ucsc.edu`.
