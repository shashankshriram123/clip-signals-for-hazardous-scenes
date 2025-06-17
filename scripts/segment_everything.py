#!/usr/bin/env python3

import os
import cv2
import random
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# ─── DEVICE SETUP ───────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on: {DEVICE}")

# ─── LOAD & CONFIGURE MODEL ─────────────────────────────────────────────────────
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

# ─── DEFINE CLASSES & COLORS ────────────────────────────────────────────────────
CLASS_NAMES = predictor.metadata.get("thing_classes")
TARGET_CLASSES = {
    "person": 1,
    "car": 3,
    "truck": 8,
    "bus": 6,
    "traffic light": 10,
    "stop sign": 12,
}
palette = {
    name: [random.randint(0, 255) for _ in range(3)]
    for name in TARGET_CLASSES
}

def overlay_instances(frame, outputs, alpha=0.5):
    """
    Overlay each detected mask in its class color onto the frame.
    If no instances are detected (empty masks), returns the original frame.
    """
    masks  = outputs["instances"].pred_masks.cpu().numpy()  # shape: [N, H, W]
    if masks.shape[0] == 0:
        return frame.copy()

    labels = outputs["instances"].pred_classes.cpu().numpy()

    # sort by mask area (desc) so small objects overlay on top
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)

    vis = frame.copy()
    for i in order:
        name = CLASS_NAMES[labels[i]]
        if name not in TARGET_CLASSES:
            continue

        mask  = masks[i]
        color = palette[name]
        # build a 3-channel colored mask
        colored = np.zeros_like(frame, dtype=np.uint8)
        for c in range(3):
            colored[:, :, c] = mask * color[c]

        # blend mask into image
        vis = np.where(
            mask[:, :, None],
            cv2.addWeighted(vis, 1 - alpha, colored, alpha, 0),
            vis
        )
    return vis

# ─── INPUT/OUTPUT PATHS ─────────────────────────────────────────────────────────
INPUT_DIR  = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset/"
OUTPUT_DIR = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/segmented_everything/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── PROCESS ALL FILES ──────────────────────────────────────────────────────────
for f in sorted(os.listdir(INPUT_DIR)):
    name, ext = os.path.splitext(f.lower())
    inp = os.path.join(INPUT_DIR, f)

    # ── IMAGE FILES ─────────────────────────────────────────────────────────────
    if ext in (".jpg", ".png", ".bmp"):
        img = cv2.imread(inp)
        if img is None:
            print(f"⚠️ Could not read {f}, skipping.")
            continue
        outputs = predictor(img)
        vis = overlay_instances(img, outputs)
        out_path = os.path.join(OUTPUT_DIR, f)
        cv2.imwrite(out_path, vis)
        print("✅ Saved image:", out_path)

    # ── VIDEO FILES ─────────────────────────────────────────────────────────────
    elif ext in (".avi", ".mp4"):
        cap = cv2.VideoCapture(inp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(OUTPUT_DIR, f"{name}_segmented.mp4")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            outputs = predictor(frame)
            vis = overlay_instances(frame, outputs)
            writer.write(vis)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames of {f}…")

        cap.release()
        writer.release()
        print("✅ Saved video:", out_path)

    # ── SKIP OTHER FILES ────────────────────────────────────────────────────────
    else:
        continue
