#!/usr/bin/env python3
import argparse
import subprocess

def convert_mts_to_mp4(input_file, output_file, reduce_quality=False, duration=60):
    """
    Convert MTS to MP4.
    - If reduce_quality is True, re-encode to H.264/AAC and normalise FPS to 30.
    - Otherwise, attempt a fast stream copy.
    - duration is in seconds (default 60s).
    """
    if reduce_quality:
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-r", "30",                 # normalise frame rate
            "-t", str(duration),        # trim duration
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            output_file
        ]
    else:
        # For fast copy, we still need to trim duration
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-t", str(duration),
            "-c", "copy",
            output_file
        ]

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Convert MTS to MP4")
    parser.add_argument("input", help="Input MTS file")
    parser.add_argument("output", nargs='?', help="Output MP4 file (optional)")
    parser.add_argument("--reduce-quality", action="store_true",
                        help="Re-encode to standard high-quality MP4 and normalise FPS to 30")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds for trimming (default: 60)")
    args = parser.parse_args()

    output_file = args.output or args.input.rsplit('.', 1)[0] + ".mp4"
    convert_mts_to_mp4(args.input, output_file, args.reduce_quality, args.duration)
    print(f"Converted {args.input} → {output_file}")

if __name__ == "__main__":
    main()
