"""
This script compares the predicted momentum graph against the ground truth momentum graph.
It requires a ground truth CSV file and a folder containing the processed scores and lights data.
It plots both momentum graphs for visual comparison and shows the frame differences for each score increase.
"""

import argparse
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.momentum_graph.plot_momentum import (
    densify_lights_data,
    get_momentum_data_points,
    plot_momentum,
)
from scripts.momentum_graph.process_score_lights import (
    OUTPUT_CSV_NAME as LIGHTS_CSV_NAME,
)
from scripts.momentum_graph.util.evaluate_score_events import (
    extract_score_increases,
    refine_score_frames_with_lights,
)
from scripts.momentum_graph.util.file_names import (
    CROPPED_SCOREBOARD_VIDEO_NAME as CROPPED_SCOREBOARD_VIDEO_NAME,
)
from scripts.momentum_graph.util.file_names import (
    PROCESSED_SCORES_CSV as SCORES_CSV_NAME,
)
from src.util.io import setup_input_video_io

MOMENTUM_GT_CSV_NAME = "momentum_gt.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and smooth score predictions."
    )
    parser.add_argument("folder", type=str, help="Path to the working folder.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="last_activation",
        help="Refinement algorithm to use",
    )
    return parser.parse_args()


def get_gt_momentum_data_points(gt_path: str, fps: int):
    gt_df = pd.read_csv(gt_path, usecols=["frame_id", "left_score", "right_score"])
    momenta = gt_df["left_score"] - gt_df["right_score"]
    return gt_df["frame_id"].to_numpy(), momenta.to_numpy()


def main():
    args = parse_arguments()
    folder = args.folder
    algorithm = args.algorithm

    video_path = path.join(folder, CROPPED_SCOREBOARD_VIDEO_NAME)
    scores_path = path.join(folder, SCORES_CSV_NAME)
    lights_path = path.join(folder, LIGHTS_CSV_NAME)

    gt_momentum_path = path.join(folder, MOMENTUM_GT_CSV_NAME)

    # load video for fps
    cap, fps, _, _, n_frames = setup_input_video_io(video_path)
    cap.release()

    # Load processed scores
    scores_df = pd.read_csv(
        scores_path, usecols=["frame_id", "left_score", "right_score"]
    )
    lights_df = pd.read_csv(lights_path)

    # Overlay score increases on light activation plot
    refined_scores = extract_score_increases(scores_df)
    lights = densify_lights_data(lights_df, total_length=n_frames)
    # plot_score_light_progression(refined_scores, lights, fps, frame_ids=scores_df["frame_id"].to_numpy())

    # Refine score occurrences using lights data
    score_occurrences = refine_score_frames_with_lights(
        lights, refined_scores, fps, algorithm=algorithm
    )
    print("Refined Score Occurrences:", score_occurrences)

    # Plot momentum graph
    frames, momenta = get_momentum_data_points(score_occurrences, fps)
    plot_momentum(frames, momenta, fps, label="Predicted")

    # Plot ground truth momentum graph
    gt_frames, gt_momenta = get_gt_momentum_data_points(gt_momentum_path, fps)
    plot_momentum(gt_frames, gt_momenta, fps, color="tab:orange", label="Ground Truth")

    # print(frames, gt_frames)
    # print(len(frames), len(gt_frames))

    # plot time differences between predicted and ground truth frame ids for all score increases if they match
    if len(frames) != len(gt_frames):
        print(
            "Warning: Number of score increases in prediction and ground truth do not match."
        )
        print("Predicted increases:", len(frames))
        print("Ground truth increases:", len(gt_frames))
    else:
        diffs = frames - gt_frames

        plt.figure("Frame Differences", figsize=(10, 5))
        plt.plot(diffs, label="Frame Differences", color="tab:red")
        plt.title(
            "Frame Differences Between Predicted and Ground Truth Score Increases"
        )
        plt.xlabel("Score Increase Index")
        plt.ylabel("Frame Difference")
        plt.axhline(
            np.mean(diffs), color="gray", linestyle="--", label="Mean Difference"
        )
        plt.legend()

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
