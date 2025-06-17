#!/usr/bin/env python3
import os
import cv2
import random
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# ─── CONFIGURE DEVICE ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on: {DEVICE}")

# ─── LOAD & CONFIGURE MODEL ───────────────────────────────────────────────────────
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.DEVICE = DEVICE
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
predictor = DefaultPredictor(cfg)

# ─── DEFINE COLORS & CLASSES ──────────────────────────────────────────────────────
CLASS_NAMES = predictor.metadata.get("thing_classes")
# pick only these COCO classes; you can add/remove as needed
TARGET_CLASSES = {
    "person": 1,
    "car": 3,
    "truck": 8,
    "bus": 6,
    "traffic light": 10,
    "stop sign": 12,
}

# assign a random color for each
palette = {name: [random.randint(0, 255) for _ in range(3)] 
           for name in TARGET_CLASSES}

def overlay_instances(image, outputs, alpha=0.5):
    """
    Overlay each detected mask in its class color onto the image.
    Smaller instances are drawn last so they appear on top.
    """
    masks  = outputs["instances"].pred_masks.cpu().numpy()
    labels = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    # sort by mask area (desc) so small objects end up on top
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)

    vis = image.copy()
    for i in order:
        cls_id = labels[i]
        name   = CLASS_NAMES[cls_id]
        if name not in TARGET_CLASSES:
            continue

        mask  = masks[i]
        color = palette[name]

        # create a 3-channel colored mask
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]

        # blend
        vis = np.where(
            mask[:, :, None],
            cv2.addWeighted(vis, 1 - alpha, colored_mask, alpha, 0),
            vis
        )
    return vis

# ─── PROCESS IMAGES ───────────────────────────────────────────────────────────────
INPUT_DIR  = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset/"
OUTPUT_DIR = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/segmented_frames/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".jpg", ".png", ".bmp")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Failed to read {fname}, skipping.")
        continue

    outputs = predictor(img)
    vis = overlay_instances(img, outputs)
    
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, vis)
    print(f"✅ Saved: {out_path}")
