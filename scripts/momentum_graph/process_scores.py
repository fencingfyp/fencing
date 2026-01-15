import argparse
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.momentum_graph.perform_ocr as perform_ocr
from scripts.momentum_graph.util.file_names import (
    CROPPED_SCOREBOARD_VIDEO_NAME,
    PROCESSED_SCORES_CSV,
)
from src.util.io import setup_input_video_io


def remove_isolated_spikes(series):
    """
    Removes isolated spikes surrounded by NaNs or with large jumps vs nearest valid neighbors.
    - diff_threshold: optional numeric limit; if abs difference > threshold, mark as NaN.
    """
    s = series.copy()
    isnan = s.isna()
    n = len(s)

    # Remove points surrounded by NaNs (both neighbors NaN)
    mask_isolated = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if not isnan[i] and isnan[i - 1] and isnan[i + 1]:
            mask_isolated[i] = True

    s[mask_isolated] = np.nan
    return s


def densify_frames(pred: pd.DataFrame, total_length: int) -> pd.DataFrame:
    """
    Given a DataFrame with possibly sparse frame_ids, forward-fill to produce a dense sequence.

    returns frame id, left_score, right_score DataFrame of length total_length
    """
    # Ensure numeric
    pred["left_score"] = pd.to_numeric(pred["left_score"], errors="coerce")
    pred["right_score"] = pd.to_numeric(pred["right_score"], errors="coerce")

    # Build a complete frame_id range and forward-fill
    full_index = pd.RangeIndex(0, total_length)
    pred = (
        pred.set_index("frame_id")
        .reindex(full_index)
        .ffill()
        .reset_index(names="frame_id")
    )

    return pred[["frame_id", "left_score", "right_score"]]


def apply_rolling_mode(series: pd.Series, window_size: int) -> pd.Series:
    """
    Apply rolling mode smoothing to a pandas Series.
    """
    return series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1]
    )


def process_scores(
    pred: pd.DataFrame, total_length=None, window_median=350, confidence_threshold=0.5
) -> np.ndarray:
    """
    Loads, cleans, and smooths prediction CSVs.
    Handles sparse frame_ids by forward-filling to produce a dense sequence.

    Returns:
        np.ndarray (n×2): [[left_score_smooth, right_score_smooth], ...]
    """

    # Ensure numeric
    pred["left_score"] = pd.to_numeric(pred["left_score"], errors="coerce")
    pred["right_score"] = pd.to_numeric(pred["right_score"], errors="coerce")

    # # Remove unrealistic outliers early
    pred.loc[pred["left_score"] > 15, "left_score"] = np.nan
    pred.loc[pred["right_score"] > 15, "right_score"] = np.nan

    # Remove low-confidence predictions
    if "left_confidence" in pred.columns:
        pred.loc[pred["left_confidence"] < confidence_threshold, "left_score"] = np.nan
    if "right_confidence" in pred.columns:
        pred.loc[pred["right_confidence"] < confidence_threshold, "right_score"] = (
            np.nan
        )

    # --- Remove isolated spikes ---
    pred["left_score"] = remove_isolated_spikes(pred["left_score"])
    pred["right_score"] = remove_isolated_spikes(pred["right_score"])

    # forward fill to densify frames
    pred = (
        densify_frames(pred, total_length=total_length)
        if total_length is not None
        else pred
    )

    # for side in ['left_score', 'right_score']:
    #     diffs = pred[side].diff()
    #     pred.loc[diffs > 1, side] = np.nan

    # --- Interpolate missing values (smoothly) ---
    for col in ["left_score", "right_score"]:
        pred[col] = (
            pred[col]
            .interpolate(method="values", x=pred["frame_id"], limit_direction="both")
            .round()
        )
        pred[col] = apply_rolling_mode(pred[col], window_median).round()

    # --- Final cleanup: prohibit jumps >1 between frames ---
    for col in ["left_score", "right_score"]:
        values = pred[col].to_numpy()
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]

            # If the jump between consecutive frames exceeds 1, fix it
            if abs(diff) > 1:
                # ----------------------------------------
                # Option 1: Snap to previous (maintain stability)
                # ----------------------------------------
                values[i] = values[i - 1]

                # ----------------------------------------
                # Option 2: Ignore (leave as is)
                # ----------------------------------------
                # pass

        pred[col] = values

    # --- Return as NumPy array ---
    return pred[["left_score", "right_score"]].to_numpy()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and smooth score predictions."
    )
    parser.add_argument("folder", type=str, help="Path to the working folder.")
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output anything"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    folder = args.folder
    demo_mode = args.demo
    # Load both CSVs
    df = pd.read_csv(path.join(folder, perform_ocr.OUTPUT_CSV_NAME))
    cap, fps, _, _, total_length = setup_input_video_io(
        path.join(folder, CROPPED_SCOREBOARD_VIDEO_NAME)
    )
    cap.release()
    pred = process_scores(df, total_length=total_length, window_median=int(fps * 6))

    # Rewrite the predictions CSV with cleaned data in this format: frame_id,left_score,right_score,left_confidence,right_confidence. set confidence to 1.0
    frame_ids = np.arange(len(pred))
    pred_df = pd.DataFrame(
        {"frame_id": frame_ids, "left_score": pred[:, 0], "right_score": pred[:, 1]}
    )
    if not demo_mode:
        pred_df.to_csv(path.join(folder, PROCESSED_SCORES_CSV), index=False)

    # ---- Step 4: Plot both predictions ----
    plt.figure("Left", figsize=(12, 6))
    plt.plot(pred[:, 0], label="Pred Left (Smoothed)", color="red", alpha=0.8)
    # plt.plot(pred[:, 1], label='Pred Right (Smoothed)', color='green', alpha=0.8)
    plt.title("Smoothed Scores")
    plt.xlabel("Frame ID")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure("Right", figsize=(12, 6))
    # plt.plot(pred[:, 0], label='Pred Left (Smoothed)', color='red', alpha=0.8)
    plt.plot(pred[:, 1], label="Pred Right (Smoothed)", color="green", alpha=0.8)
    plt.title("Smoothed Scores")
    plt.xlabel("Frame ID")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show both figures
    plt.show()


if __name__ == "__main__":
    main()
