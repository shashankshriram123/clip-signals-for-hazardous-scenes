#!/usr/bin/env python3
import os
import glob
import cv2
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ====== CONFIG ======
DATASET_DIR  = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset"
OUTPUT_DIR   = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/graphs"
PROMPTS = [
    # --- concise catch-all / single-word hazards ------------------------------
    "animal",
    "debris",
    "obstacle",
    "hazard",
    "blockage",
    "construction",
    "barricade",
    "roadwork",
    "cone",
    "worker",
    "pedestrian",
    "emergency",
    "crash",
    "collision",
    "accident",
    "stall",
    "breakdown",
    "object",
    "obstruction",

    # --- animals --------------------------------------------------------------
    "animal in the road",
    "animal crossing the road",
    "animal crossing the road unexpectedly",

    # --- construction & traffic control --------------------------------------
    "construction zone ahead",
    "construction workers in the roadway",
    "construction blocking the lane",
    "traffic cones on the road",
    "road rerouted by workers",

    # --- vehicle anomalies ----------------------------------------------------
    "stalled vehicle in the lane",
    "vehicle swerving out of control",

    # --- debris / loose objects ----------------------------------------------
    "object debris on the road",
    "unexpected debris on the road",
    "dangerous debris on the road",
    "tumbleweed rolling across the road",
    "trash on the road",
    "household object lying on the road",

    # --- generic fallback phrases --------------------------------------------
    "road blocked ahead",
    "emergency vehicle or blocked road",
    "road hazard ahead"
]
FRAME_STRIDE = 1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# =====================

# Prepare output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CLIP once
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
# Tokenize & encode all prompts once
text_tokens = clip.tokenize(PROMPTS).to(DEVICE)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Find all videos
video_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mp4")))

for VIDEO_PATH in video_paths:
    video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    print(f"\n▶ Processing `{video_name}`")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  ✖ Cannot open {VIDEO_PATH}, skipping.")
        continue

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = frame_count / fps

    # Storage containers
    timestamps     = []
    frame_indices  = []
    similarity_dict = defaultdict(list)

    frame_number = 0
    with torch.no_grad():
        pbar = tqdm(total=frame_count, desc=f"{video_name}", leave=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % FRAME_STRIDE == 0:
                # Preprocess frame
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                inp     = preprocess(pil_img).unsqueeze(0).to(DEVICE)

                # Encode & normalize
                img_feat = model.encode_image(inp)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

                # Compute similarities
                sims = (img_feat @ text_features.T).squeeze().tolist()

                # Record
                t = frame_number / fps
                timestamps.append(t)
                frame_indices.append(frame_number)
                for i, prompt in enumerate(PROMPTS):
                    similarity_dict[prompt].append(sims[i])

            frame_number += 1
            pbar.update(1)
        pbar.close()
    cap.release()

    # Plot
    plt.figure(figsize=(14, 7))
    for prompt in PROMPTS:
        plt.plot(timestamps, similarity_dict[prompt], label=f"'{prompt}'")

    plt.xlabel("Time (s)")
    plt.ylabel("CLIP Similarity Score")
    plt.ylim(0, 1)
    plt.title(f"CLIP Similarity Over Time — {video_name}")
    plt.grid(True)
    plt.legend()

    # Secondary x-axis for frame numbers
    ax    = plt.gca()
    secax = ax.secondary_xaxis(
        'top',
        functions=(lambda t: t * fps, lambda f: f / fps)
    )
    secax.set_xlabel("Frame Number")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"{video_name}_CLIP_graph.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"  ✔ Saved plot to: {out_png}")
