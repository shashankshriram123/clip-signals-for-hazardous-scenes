# clip-signals-for-hazardous-scenes

A research playground for rapid experiments on **hazard detection** in dash‑cam footage.
The repo combines

* **CLIP‑based similarity scoring** for scene retrieval
* **Instance & Panoptic segmentation** (Detectron 2)
* **Simple HPC batch jobs** so you can scale runs on UC Merced’s cluster without `sudo`

> **Goal**  Feed pixel‑precise object/"stuff" masks into downstream reasoning modules (e.g. VLMs) so that higher‑level hazard logic doesn’t have to re‑discover where the road, cars, or pedestrians are.

---

## 1  Folder layout

| Path                    | Contents                                                     |
| ----------------------- | ------------------------------------------------------------ |
| `dataset/`              | Raw videos for testing (`video_0024.mp4`, …)                 |
| `scripts/`              | All segmentation + plotting scripts (see below)              |
| `segmented_everything/` | Instance‑segmented videos written by `segment_everything.py` |
| `segmented_panoptic/`   | Panoptic‑segmented videos from `segment_panoptic.py`         |
| `graphs/`               | Similarity‑score plots produced by earlier CLIP runs         |
| `hpc_jobs/`             | SLURM batch scripts & log stub (`logs/`)                     |
| `requirements.txt`      | Conda/pip dependency pin (mirrors commands below)            |

---

## 2  Quick start (local)

```bash
# 1. Clone
$ git clone https://github.com/shashankshriram123/clip-signals-for-hazardous-scenes.git
$ cd clip-signals-for-hazardous-scenes

# 2. Create env (CPU‑only example)
$ conda create -n clip_scores python=3.10 -y
$ conda activate clip_scores

# 3. Install vision stack
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
$ pip install numpy==1.24.4 opencv-python cython pyyaml iopath tqdm

# 4. Detectron2 build (source, no sudo)
$ git clone https://github.com/facebookresearch/detectron2.git
$ pip install -e detectron2 --no-build-isolation --no-deps

# 5. Run a quick test on one video
$ python scripts/segment_panoptic.py
```

> If you **have CUDA 11.7+ and a driver** just replace the CPU wheels with the matching `+cu117` wheels and Detectron2 will auto‑run on GPU.

---

## 3  Segmentation scripts

| Script                  | Model                                 | What it does                                         | Typical output               |
| ----------------------- | ------------------------------------- | ---------------------------------------------------- | ---------------------------- |
| `segment_frames.py`     | Mask R‑CNN (`mask_rcnn_R_50_FPN_3x`)  | Instance segmentation on *images*                    | `segmented_frames/*.jpg`     |
| `segment_everything.py` | Mask R‑CNN                            | Instance segmentation on images **and** videos       | `segmented_everything/*.mp4` |
| `segment_panoptic.py`   | Panoptic FPN (`panoptic_fpn_R_50_3x`) | Panoptic segmentation ("things" + "stuff") on videos | `segmented_panoptic/*.mp4`   |

All three

1. Auto‑detect GPU vs CPU (`torch.cuda.is_available()`).
2. Download weights on first run.
3. Use random but consistent RGB palettes.
4. Log progress every 50 frames.

### Example‑usage

```bash
# Instance masks only
python scripts/segment_everything.py \
  --input_dir dataset/ --out_dir segmented_everything/

# Panoptic masks (recommended for hazard work)
python scripts/segment_panoptic.py \
  --input_dir dataset/ --out_dir segmented_panoptic/
```

*(Both scripts fall back to the default paths if flags are omitted.)*

---

## 4  Batch runs on UC Merced HPC

1. Edit `hpc_jobs/run_segmentation.slurm` – point `--input_dir` to your scratch space.
2. Submit with `sbatch hpc_jobs/run_segmentation.slurm`.
3. Logs land in `hpc_jobs/logs/*.qlog` (as wired in the SLURM header).

The SLURM script loads GCC 11 + CMake 3.21 modules and activates the `clip_scores` Conda env before invoking `python segment_panoptic.py`.

---

## 5  Troubleshooting

| Symptom                                                    | Fix                                                                           |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **`RuntimeError: Found no NVIDIA driver`**                 | GPU build of PyTorch but no driver → reinstall CPU wheels or load a GPU node. |
| **`ModuleNotFoundError: torch` during Detectron2 install** | Add `--no-build-isolation` so the build sees your existing PyTorch.           |
| **NumPy 2.x mismatch** (`Numpy is not available`)          | `pip install numpy==1.24.4` (Detectron2 wheels are still on NumPy 1.x ABI).   |
| Empty-mask reshape crash                                   | Guarded in latest scripts (`if masks.shape[0] == 0: return frame`).           |

---

## 6  Roadmap / Ideas

* Plug panoptic masks into the CLIP similarity pipeline → richer hazard scoring.
* Train a lightweight **Cityscapes** model for even finer road classes (lane‑marks, poles…).
* Export masks as **COCO‑JSON** to feed into downstream VLM prompts.
* Package SLURM runs as **Snakemake** workflow for full reproducibility.

---

## 7  License & citation

Code is Apache‑2.0.  Models are from the official Detectron 2 Model Zoo (Apache 2.0).
If you use this repo in academic work, please cite **Detectron2** and **Mask R‑CNN / Panoptic FPN** original papers.

---

Made with ☕ in Merced. Pull requests are welcome!
