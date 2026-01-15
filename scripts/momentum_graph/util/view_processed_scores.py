import argparse

import pandas as pd
from matplotlib import pyplot as plt

from scripts.momentum_graph.process_score_lights import (
    OUTPUT_CSV_NAME as PROCESSED_SCORE_LIGHTS_CSV_NAME,
)
from scripts.momentum_graph.process_scores import densify_frames
from src.util.file_names import ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and smooth score predictions."
    )
    parser.add_argument("folder", type=str, help="Path to the working folder.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    folder = args.folder
    cap, _, _, _, total_length = setup_input_video_io(f"{folder}/{ORIGINAL_VIDEO_NAME}")
    cap.release()
    # --- 1. Load and smooth ---
    df = pd.read_csv(f"{folder}/{PROCESSED_SCORE_LIGHTS_CSV_NAME}")
    # rename left_light to left_score and right_light to right_score
    df.rename(
        columns={"left_light": "left_score", "right_light": "right_score"}, inplace=True
    )
    df = densify_frames(df, total_length=total_length)

    df.rename(
        columns={"left_score": "left_light", "right_score": "right_light"}, inplace=True
    )

    # --- 2. Plot ---
    plt.figure("Left Light", figsize=(8, 3))
    plt.plot(df["frame_id"], df["left_light"], label="Left light")
    plt.xlabel("Frame ID")
    plt.ylabel("Light state (0=off, 1=on)")
    plt.legend()
    plt.tight_layout()

    plt.figure("Right Light", figsize=(8, 3))
    plt.plot(df["frame_id"], df["right_light"], label="Right light", color="orange")
    plt.xlabel("Frame ID")
    plt.ylabel("Light state (0=off, 1=on)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
