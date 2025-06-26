#!/usr/bin/env python3
"""
npy_to_mp4.py
-------------
Convert every *.npy file in a directory (or its sub-directories) to an .mp4.

Each .npy is expected to contain a 4-D uint8 tensor shaped
   (num_frames, height, width, 3)   in RGB order.

Usage
-----
    python npy_to_mp4.py /absolute/or/relative/path [--fps 10] [--ext mp4]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

def write_video(array: np.ndarray, out_path: Path, fps: int) -> None:
    """Save a uint8 RGB tensor to MP4."""
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{out_path.stem}: expected shape (N,H,W,3), got {array.shape}")

    n, h, w, _ = array.shape
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # OpenCV needs BGR
    for i in range(n):
        frame_bgr = cv2.cvtColor(array[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()

def convert_folder(root: Path, fps: int, ext: str) -> None:
    npy_files = sorted(root.rglob("*.npy"))
    if not npy_files:
        print("No .npy files found under", root, file=sys.stderr)
        return

    print(f"Found {len(npy_files)} .npy files. Converting…")
    for npy in npy_files:
        out_path = npy.with_suffix(f".{ext}")
        print(f"  {npy.name}  →  {out_path.name}")
        arr = np.load(npy)              # load into RAM
        write_video(arr, out_path, fps)
    print("Done!")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path,
                        help="Directory containing *.npy files (searched recursively)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the output video (default 10)")
    parser.add_argument("--ext", default="mp4",
                        help="Video file extension/format (default mp4)")
    args = parser.parse_args()

    if not args.folder.exists():
        parser.error(f"Folder {args.folder} does not exist.")
    convert_folder(args.folder, args.fps, args.ext)

if __name__ == "__main__":
    main()

