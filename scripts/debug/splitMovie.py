#!/usr/bin/env python
"""
split_video_ffmpeg.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Split a long video into named clips using FFmpeg.

Usage (inside the edit_movie env):
    python split_video_ffmpeg.py \
        /path/to/full_video.mp4 \
        ~/extracted_clips
"""

import os, subprocess, sys, textwrap

# --------------------- EDIT HERE ONLY IF NEEDED ------------- #
SNIPPETS = {
    "snippet_1": ("00:21:50", "00:22:03"),
    "snippet_2": ("00:22:38", "00:22:51"),
    "snippet_3": ("00:27:07", "00:27:25"),
}
# ----------------------------------------------------------- #

def run(cmd):
    """Run a shell command and raise if it fails."""
    print("➜", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main(video_path, out_dir):
    if not os.path.exists(video_path):
        sys.exit(f"[ERROR] Input video not found: {video_path}")

    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for name, (start, end) in SNIPPETS.items():
        out_file = os.path.join(out_dir, f"{name}.mp4")

        # -ss <start> -to <end> BEFORE -i gives fast, accurate seeking
        cmd = [
            "ffmpeg",
            "-y",                   # overwrite output if it exists
            "-ss", start,
            "-to", end,
            "-i", video_path,
            "-c:v", "libx264",      # re-encode for max compatibility
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            out_file,
        ]
        try:
            run(cmd)
            print(f"✅  Created {out_file}\n")
        except subprocess.CalledProcessError:
            print(f"❌  Failed on {name}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(textwrap.dedent(f"""
            usage:
                python {sys.argv[0]} /path/to/full_video.mp4 ~/extracted_clips
        """).strip())
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
