"""
This script compares the predicted scores against the ground truth scores.
It requires a ground truth CSV file and a folder containing the processed scores and lights data.
It computes the frame differences for each score increase and plots both predicted and ground truth scores for visual
comparison.
"""

import argparse
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.momentum_graph.process_scores import densify_frames
from scripts.momentum_graph.util.file_names import (
    PROCESSED_SCORES_CSV as SCORES_CSV_NAME,
)
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME
from src.util.io import setup_input_video_io


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and smooth score predictions."
    )
    # parser.add_argument("gt", type=str, help="Path to the ground truth CSV file.")
    parser.add_argument("folder", type=str, help="Path to the working folder.")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten temporary score increases due to incorrect point awarding.",
    )
    return parser.parse_args()


def frame_diffs(gt, pred):
    """
    gt and pred: np arrays of shape (n_frames, 2)
    Returns: dict with keys 'left' and 'right', each containing a list of frame differences per score increase
    """
    diffs = {"left": [0], "right": [0]}
    for fencer in [0, 1]:
        gt_scores = gt[:, fencer]
        pred_scores = pred[:, fencer]
        max_score = max(gt_scores.max(), pred_scores.max())

        for score in range(1, int(max_score) + 1):
            # first frame where score is reached
            gt_frame = np.argmax(gt_scores >= score)
            pred_frame = np.argmax(pred_scores >= score)
            diffs["left" if fencer == 0 else "right"].append(pred_frame - gt_frame)

    return diffs


def main():
    args = parse_arguments()
    folder = args.folder
    gt = path.join(folder, "scores_gt.csv")
    flatten = args.flatten

    scoreboard_video_path = path.join(folder, CROPPED_SCOREBOARD_VIDEO_NAME)
    processed_scores_path = path.join(folder, SCORES_CSV_NAME)

    # Load both CSVs
    pred = pd.read_csv(processed_scores_path)[["left_score", "right_score"]].to_numpy()
    gt = pd.read_csv(gt)

    cap, fps, _, _, total_length = setup_input_video_io(scoreboard_video_path)
    cap.release()

    # get the left_score and right_score columns as numpy arrays
    gt = densify_frames(gt, total_length)[["left_score", "right_score"]].to_numpy()

    print(pred.shape, gt.shape)
    # Optionally flatten both predictions and ground truth
    # if flatten:
    #     # check if pred is monotonic non-decreasing
    #     if not np.all(np.diff(pred[:, 0]) >= 0):
    #         print("Warning: Predictions are not monotonic non-decreasing on the left side.")
    #     if not np.all(np.diff(pred[:, 1]) >= 0):
    #         print("Warning: Predictions are not monotonic non-decreasing on the right side.")

    #     smoothed_pred = pd.DataFrame({'frame_id': np.arange(len(pred)), 'left_score': _retroactive_flatten(pred[:, 0].tolist()), 'right_score': _retroactive_flatten(pred[:, 1].tolist())
    #     })
    #     smoothed_gt = pd.DataFrame({'frame_id': np.arange(len(gt)), 'left_score': _retroactive_flatten(gt[:, 0].tolist()), 'right_score': _retroactive_flatten(gt[:, 1].tolist())
    #     })
    #     pred = densify_frames(smoothed_pred, len(pred)).to_numpy()
    #     gt = densify_frames(smoothed_gt, len(gt)).to_numpy()

    diffs = frame_diffs(gt, pred)
    diffs["left"] = [d / fps for d in diffs["left"]]  # convert to seconds
    diffs["right"] = [d / fps for d in diffs["right"]]

    # combined statistics
    total = diffs["left"] + diffs["right"]
    abs_total = [abs(d) for d in total]
    print("MAE Total Score Increases (seconds):", np.mean(abs_total))
    print(
        "RMSE Total Score Increases (seconds):",
        np.sqrt(np.mean(np.array(abs_total) ** 2)),
    )
    print("Max deviation (seconds):", np.max(abs_total))

    # plot diffs bar charts on separate figures, divide by fps to get seconds
    plt.figure("Differences (Left)", figsize=(12, 6))
    plt.bar(
        range(len(diffs["left"])),
        np.array(diffs["left"]),
        label="Left Score Increases",
        color="blue",
        alpha=0.8,
    )
    plt.title("Differences for Left Score Increases")
    plt.xlabel("Point ID")
    plt.ylabel("Difference (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure("Differences (Right)", figsize=(12, 6))
    plt.bar(
        range(len(diffs["right"])),
        np.array(diffs["right"]),
        label="Right Score Increases",
        color="red",
        alpha=0.8,
    )
    plt.title("Differences for Right Score Increases")
    plt.xlabel("Point ID")
    plt.ylabel("Difference (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # pred = pred[50000:]
    # gt = gt[50000:]

    # ---- Step 4: Plot both predictions and ground truth ----
    # --- Left ---
    plt.figure("Left", figsize=(12, 6))
    plt.plot(pred[:, 0], label="Pred Left (Smoothed)", color="blue", alpha=0.8)
    plt.plot(gt[:, 0], "--", color="red", label="GT Left")
    plt.title("Smoothed Predicted vs Ground Truth Scores (Left)")
    plt.xlabel("Frame ID")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Right ---
    plt.figure("Right", figsize=(12, 6))
    plt.plot(pred[:, 1], label="Pred Right (Smoothed)", color="blue", alpha=0.8)
    plt.plot(gt[:, 1], "--", color="red", label="GT Right")
    plt.title("Smoothed Predicted vs Ground Truth Scores (Right)")
    plt.xlabel("Frame ID")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show both figures
    plt.show()


if __name__ == "__main__":
    main()
