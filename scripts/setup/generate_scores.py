import os, json, argparse, numpy as np, torch, cv2, tqdm, clip
from pathlib import Path
from PIL import Image

DATASET_DIR   = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset"
ANNOT_DIR     = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/annotations"
OUT_DIR       = "/home/sshriram2/mi3Testing/hazard_detection_CLIP/clip_scores"
BASE_PROMPTS  = [
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

def build_prompt_list(annot_path):
    with open(annot_path) as f:
        ann = json.load(f)

    prompts = BASE_PROMPTS.copy()

    desc = ann.get("description", "")
    if isinstance(desc, str) and desc.strip():
        prompts.append(desc.strip())

    hazard = ann.get("target_hazard", "")
    if isinstance(hazard, str) and hazard.strip():
        prompts.append(hazard.strip())

    return prompts

def main(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    for vid_path in sorted(Path(DATASET_DIR).glob("*.mp4")):
        stem  = vid_path.stem                      # video_0024
        ann_path = Path(ANNOT_DIR) / f"{stem}.mp4.json"
        if not ann_path.exists():
            print(f"⚠️  No annotation found for {stem}, skipping.")
            continue

        prompts = build_prompt_list(ann_path)
        print(f"{stem}: {len(prompts)} prompts ->", prompts)

        # encode all prompts once (rows = prompts)
        with torch.no_grad():
            text_tok   = clip.tokenize(prompts).to(device)
            text_feats = model.encode_text(text_tok).float()
            text_feats /= text_feats.norm(dim=-1, keepdim=True)  # [P,512]

        # open video
        cap = cv2.VideoCapture(str(vid_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scores   = np.zeros((len(prompts), n_frames), dtype=np.float32)

        pbar = tqdm.tqdm(range(n_frames), desc=stem)
        for idx in pbar:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(img_tensor).float()
                img_feat /= img_feat.norm(dim=-1, keepdim=True)      # [1,512]
                sim = (img_feat @ text_feats.T).cpu().numpy()[0]      # [P]
            scores[:, idx] = sim
            if idx % 50 == 0:
                pbar.set_postfix(sim=f"{sim.mean():.3f}")

        cap.release()

        np.save(f"{OUT_DIR}/{stem}.npy", scores)
        json.dump({"prompts": prompts},
                  open(f"{OUT_DIR}/{stem}_prompts.json", "w"), indent=2)
        print(f"✅  saved {stem}.npy  shape={scores.shape}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device)