#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ─── DEVICE ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running panoptic inference on: {DEVICE}")

# ─── LOAD PANOPTIC MODEL ─────────────────────────────────────────────────────────
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
))
cfg.MODEL.DEVICE = DEVICE
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

predictor = DefaultPredictor(cfg)

# ─── CORRECTLY GET PANOPTIC METADATA ────────────────────────────────────────────
# cfg.DATASETS.TEST is a tuple like ("coco_2017_val_panoptic",)
panoptic_dataset = cfg.DATASETS.TEST[0]
meta = MetadataCatalog.get(panoptic_dataset)

# ─── I/O PATHS ───────────────────────────────────────────────────────────────────
INPUT_DIR  = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset/"
OUTPUT_DIR = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/segmented_panoptic/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── PROCESS VIDEOS ──────────────────────────────────────────────────────────────
for vid in sorted(os.listdir(INPUT_DIR)):
    name, ext = os.path.splitext(vid.lower())
    if ext not in (".mp4", ".avi"):
        continue

    inp_path = os.path.join(INPUT_DIR, vid)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_panoptic.mp4")

    cap = cv2.VideoCapture(inp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run panoptic segmentation
        outputs = predictor(frame)
        pan_seg, seg_info = outputs["panoptic_seg"]

        # Visualize
        vis = Visualizer(frame[:, :, ::-1], meta, instance_mode=ColorMode.IMAGE)
        vis_out = vis.draw_panoptic_seg(pan_seg.to("cpu"), seg_info)
        vis_bgr = vis_out.get_image()[:, :, ::-1]

        writer.write(vis_bgr)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames of {vid}…")

    cap.release()
    writer.release()
    print(f"✅ Saved panoptic video: {out_path}")
