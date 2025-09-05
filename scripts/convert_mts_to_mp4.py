#!/usr/bin/env python3
import argparse
import subprocess

def convert_mts_to_mp4(input_file, output_file, reduce_quality=False):
    if reduce_quality:
        # Re-encode to standard high-quality MP4 and normalise FPS to 30
        cmd = [
            "ffmpeg", "-i", input_file,
            "-r", "30",  # normalise frame rate
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            output_file
        ]
    else:
        # Fast copy if possible
        cmd = ["ffmpeg", "-i", input_file, "-c", "copy", output_file]

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Convert MTS to MP4")
    parser.add_argument("input", help="Input MTS file")
    parser.add_argument("output", nargs='?', help="Output MP4 file (optional)")
    parser.add_argument("--reduce-quality", action="store_true", help="Re-encode to standard high-quality MP4 and normalise FPS to 30")
    args = parser.parse_args()

    output_file = args.output or args.input.rsplit('.', 1)[0] + ".mp4"
    convert_mts_to_mp4(args.input, output_file, args.reduce_quality)
    print(f"Converted {args.input} → {output_file}")

if __name__ == "__main__":
    main()
