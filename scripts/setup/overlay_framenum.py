#!/usr/bin/env python3
"""
Overlay zero-padded frame numbers on every video in
/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset/
and write the results to
/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset_framed/

Dependencies: opencv-python (cv2)  •  Python ≥3.7
"""

import cv2
import pathlib
import os

SRC_DIR = pathlib.Path("/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset")
DST_DIR = pathlib.Path("/home/sshriram2/mi3Testing/hazard_detection_CLIP/dataset_framed")
DST_DIR.mkdir(parents=True, exist_ok=True)            # make output folder if missing

def overlay_video(src_path: pathlib.Path, dst_path: pathlib.Path) -> None:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {src_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out   = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame,
                    f"{frame_idx:06d}",            # 000001, 000002, …
                    (15, 35),                      # (x, y) in pixels
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[OK] {src_path.name}  →  {dst_path.name}")

def main() -> None:
    vids = sorted(SRC_DIR.glob("**/*.mp4"))  # recurse & match *.mp4
    if not vids:
        print("No MP4 files found.")
        return
    for vid in vids:
        # keep sub-folder structure, mirror it under dataset_framed
        rel_path = vid.relative_to(SRC_DIR)
        out_path = DST_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_path.with_stem(out_path.stem + "_framenum")
        overlay_video(vid, out_path)

if __name__ == "__main__":
    main()
