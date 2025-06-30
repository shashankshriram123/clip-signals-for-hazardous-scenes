# CLIP Signals for Hazardous Scenes

> **Goal:** Detect and localise hazardous events in driving videos using CLIP similarity scores.
> *Dataset:* 14 short driving scenes (animal, construction, debris, accident, normal‑no‑hazard).
> *Prompts per frame:* 40 (19 generic + 3 animal + 5 construction + 2 vehicle anomalies + misc … see `generate_scores.py`).

---

## Repository Structure

```
clip-signals-for-hazardous-scenes/
├── annotations/              # Ground‑truth JSON (scene_xxx.mp4.json)
├── clip_scores/              # .npy similarity scores, prompt lists, eval CSVs
├── dataset/                  # Raw videos (.mp4) – ignored by git
├── graphs/                   # Similarity plots & threshold sweep figures
├── scripts/                  # Pipeline + analysis utilities
└── hpc_jobs/                 # SLURM batch scripts
```

---

## Pipeline Overview

1. **Score generation** (`scripts/analysis/generate_scores.py`)

   * Encodes every frame against the full prompt bank (40×frames) and saves `scene_xxx.npy` + prompt list.
2. **Evaluation & Threshold Sweep**

   * `analysis/threshold_sweep.py` — global sweep, prints boxed table, highlights best‑IoU **T**.
   * `analysis/threshold_by_type.py` — per‑hazard sweep using the `type` key in each annotation.
3. **Visualisation**

   * `analysis/plot_top_prompts.py` plots top‑5 prompt curves per scene (excludes GT rows 39–40).

---

## Threshold Calibration

| Scope        | Optimal T | mean IoU |   P    |   R    |   F1   | Notes                           |
|--------------|-----------|----------|--------|--------|--------|---------------------------------|
| **Global**   | **0.28**  | **0.065**| **0.252** | **0.230** | **0.241** | 14 scenes, no smoothing          |
| Accident     | 0.28      | 0.000    | 0.000  | 0.000  | 0.000  | single clip – empty ground truth |
| Animal       | 0.28      | 0.120    | 0.162  | 0.316  | 0.214  | n = 4                            |
| Construction | 0.28      | 0.255    | 0.542  | 0.325  | 0.406  | n = 4                            |
| Debris       | 0.28      | 0.000    | 0.000  | 0.000  | 0.000  | n = 4                            |

> *IoU computed with union-row mask **`(scores>T).any(axis=0)`**, no temporal smoothing.*

Ablation shows the global **T = 0.28** yields relatively low IoU and precision/recall across most hazard classes.


---

## Metrics

| Metric            | Formula         | Intuition                                                           |
| ----------------- | --------------- | ------------------------------------------------------------------- |
| **Precision (P)** | TP / (TP + FP)  | “When the model flags a hazard, how often is it correct?”           |
| **Recall (R)**    | TP / (TP + FN)  | “When a hazard is present, how often does the model catch it?”      |
| **F1 Score**      | 2·P·R / (P + R) | Harmonic mean — high only if both precision **and** recall are high |
| **Temporal IoU**  | –               | Overlap between predicted and ground‑truth time windows             |
| **Threshold (T)** | –               | CLIP similarity cutoff (default sweep 0.15 – 0.35)                  |

**TP** = true‑positive frames    **FP** = false‑positive    **FN** = false‑negative

## Current Clip‑Level Results (T = 0.28)

| Scene      | Type   | IoU       | P    | R    | F1   |
| ---------- | ------ | --------- | ---- | ---- | ---- |
| scene\_001 | animal | 0.663     | 0.46 | 0.93 | 0.62 |
| scene\_002 | animal | 0.414     | 0.52 | 0.93 | 0.63 |
| …          | …      | …         | …    | …    | …    |
| **mean**   | –      | **0.547** | 0.46 | 0.79 | 0.55 |

Full table in `clip_scores/clip_eval.csv`.

---

## Quick Usage

```bash
# 1. create env
conda create -n clip_scores python=3.10 -y
conda activate clip_scores
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git opencv-python matplotlib tqdm numpy pillow

# 2. generate scores\python scripts/analysis/generate_scores.py

# 3. global threshold sweep
python scripts/analysis/threshold_sweep.py --min 0.15 --max 0.35 --step 0.01 --smooth 0

# 4. per‑hazard sweep
python scripts/analysis/threshold_by_type.py
```

---

## Contact

For access to the full dataset or questions, email **[sshrir2@ucsc.edu](mailto:sshrir2@ucsc.edu)**.
