"""
This script compares predicted score light activations against ground truth.
It computes frame differences for each light-on event and plots predictions vs GT.
"""

import argparse
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.momentum_graph.plot_momentum import densify_lights_data
from scripts.momentum_graph.process_scores import densify_frames
from scripts.momentum_graph.util.file_names import LIGHTS_GT_CSV, PROCESSED_LIGHTS_CSV
from src.util.file_names import CROPPED_SCORE_LIGHTS_VIDEO_NAME
from src.util.io import setup_input_video_io


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare predicted score lights against ground truth."
    )
    parser.add_argument("folder", type=str, help="Path to the working folder.")
    return parser.parse_args()


def light_frame_diffs(gt, pred):
    """
    gt and pred: np arrays of shape (n_frames, 2), values in {0,1}
    Returns: dict with keys 'left' and 'right', each containing frame diffs
    """
    diffs = {"left": [], "right": []}

    for side, idx in [("left", 0), ("right", 1)]:
        gt_on = np.where(np.diff(gt[:, idx], prepend=0) == 1)[0]
        pred_on = np.where(np.diff(pred[:, idx], prepend=0) == 1)[0]

        # pair events in order
        n = min(len(gt_on), len(pred_on))
        diffs[side] = (pred_on[:n] - gt_on[:n]).tolist()

    return diffs


def main():
    args = parse_arguments()
    folder = args.folder

    video_path = path.join(folder, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
    pred_path = path.join(folder, PROCESSED_LIGHTS_CSV)

    # load CSVs
    pred_df = pd.read_csv(pred_path)[["frame_id", "left_light", "right_light"]]
    gt_df = pd.read_csv(path.join(folder, LIGHTS_GT_CSV))[["frame_id", "left", "right"]]
    gt_df.rename(columns={"left": "left_light", "right": "right_light"}, inplace=True)

    cap, fps, _, _, total_length = setup_input_video_io(video_path)
    cap.release()

    pred = densify_lights_data(pred_df, total_length)
    gt = densify_lights_data(gt_df, total_length)

    # --- Left light ---
    plt.figure("Left Light Activations", figsize=(12, 4))
    plt.plot(pred[:, 0], label="Pred Left", alpha=0.8)
    plt.plot(gt[:, 0], "--", label="GT Left")
    plt.title("Predicted vs Ground Truth – Left Score Light")
    plt.xlabel("Frame ID")
    plt.ylabel("Light State")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Right light ---
    plt.figure("Right Light Activations", figsize=(12, 4))
    plt.plot(pred[:, 1], label="Pred Right", alpha=0.8)
    plt.plot(gt[:, 1], "--", label="GT Right")
    plt.title("Predicted vs Ground Truth – Right Score Light")
    plt.xlabel("Frame ID")
    plt.ylabel("Light State")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
