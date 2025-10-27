#!/usr/bin/env python3
import argparse
import subprocess

import subprocess
import shlex

def convert_mts_to_mp4(input_file, output_file, reduce_quality=False, duration=None):
    """
    Convert MTS to MP4.

    - If reduce_quality is True, re-encode to H.264/AAC and normalise FPS to 30.
    - Otherwise, copy video and re-encode audio to AAC (MP4-compatible).
    - duration (float or int): optional, trim to given seconds.
    """
    cmd = ["ffmpeg", "-y", "-i", input_file]

    if reduce_quality:
        # Normalize frame rate and compress
        cmd += [
            "-r", "30",
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k"
        ]
    else:
        # Copy video, re-encode audio for MP4 compatibility
        cmd += [
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k"
        ]

    # Optional trim
    if duration:
        cmd += ["-t", str(duration)]

    cmd.append(output_file)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        print("Command:", " ".join(shlex.quote(x) for x in cmd))
        raise


def main():
    parser = argparse.ArgumentParser(description="Convert MTS to MP4")
    parser.add_argument("input", help="Input MTS file")
    parser.add_argument("output", help="Output MP4 file")
    parser.add_argument("--reduce-quality", action="store_true",
                        help="Re-encode to standard high-quality MP4 and normalise FPS to 30")
    parser.add_argument("--duration", type=int, help="Duration in seconds for trimming")
    args = parser.parse_args()

    output_file = args.output or args.input.rsplit(".", 1)[0] + ".mp4"
    convert_mts_to_mp4(args.input, output_file, args.reduce_quality, args.duration)
    print(f"Converted {args.input} → {output_file}")

if __name__ == "__main__":
    main()
