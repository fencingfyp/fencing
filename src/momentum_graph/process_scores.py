import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.util import setup_input_video_io

def remove_isolated_spikes(series):
    """
    Removes isolated spikes surrounded by NaNs or with large jumps vs nearest valid neighbors.
    - diff_threshold: optional numeric limit; if abs difference > threshold, mark as NaN.
    """
    # return series
    s = series.copy()
    isnan = s.isna()
    n = len(s)

    # Remove points surrounded by NaNs (both neighbors NaN)
    mask_isolated = np.zeros(n, dtype=bool)
    for i in range(1, n-1):
        if not isnan[i] and isnan[i-1] and isnan[i+1]:
            mask_isolated[i] = True

    s[mask_isolated] = np.nan
    return s

def process_scores(pred: pd.DataFrame, smoothen=True, total_length=None, window_median=350):
    """
    Loads, cleans, and smooths prediction CSVs.
    Handles sparse frame_ids by forward-filling to produce a dense sequence.

    Returns:
        np.ndarray (n×2): [[left_score_smooth, right_score_smooth], ...]
    """

    # Ensure numeric
    pred['left_score'] = pd.to_numeric(pred['left_score'], errors='coerce')
    pred['right_score'] = pd.to_numeric(pred['right_score'], errors='coerce')

    # Remove unrealistic outliers early
    if smoothen:
        pred.loc[pred['left_score'] > 15, 'left_score'] = np.nan
        pred.loc[pred['right_score'] > 15, 'right_score'] = np.nan

        # --- Remove isolated spikes ---
        pred['left_score'] = remove_isolated_spikes(pred['left_score'])
        pred['right_score'] = remove_isolated_spikes(pred['right_score'])

    # --- Forward-fill missing frames (sparse CSVs) ---
    # Build a complete frame_id range from min to max, join, and ffill
    full_index = pd.RangeIndex(pred['frame_id'].min(), pred['frame_id'].max() + 1)
    pred = (
        pred.set_index('frame_id')
        .reindex(full_index)
        .ffill()
        .reset_index()
        .rename(columns={'index': 'frame_id'})
    )

    if smoothen:
        # --- Interpolate missing values (smoothly) ---
        for col in ['left_score', 'right_score']:
            pred[col] = (
                pred[col]
                .interpolate(method='values', x=pred['frame_id'], limit_direction='both')
                .round()
            )

        # --- Median smoothing (rolling) ---
        # Use a fixed-size centered rolling window
        pred['left_score'] = (
            pred['left_score']
            .rolling(window=window_median, center=True, min_periods=1)
            .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1])
        )
        pred['right_score'] = (
            pred['right_score']
            .rolling(window=window_median, center=True, min_periods=1)
            .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1])
        )

        # --- Final cleanup: prohibit jumps >1 between frames ---
        for col in ['left_score', 'right_score']:
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

    # If total_length is specified, truncate or pad to that length
    if total_length is not None:
        n = len(pred)
        if total_length < n:
            # Truncate to desired length
            pred = pred.iloc[:total_length]
        elif total_length > n:
            # Extend by repeating the last row
            last_row = pred.iloc[-1:]
            pad = pd.concat([last_row] * (total_length - n), ignore_index=True)
            pred = pd.concat([pred, pad], ignore_index=True)


    # --- Return as NumPy array ---
    return pred[['left_score', 'right_score']].to_numpy()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and smooth score predictions.")
    parser.add_argument('folder', type=str, help='Path to the working folder.')
    args = parser.parse_args()
    return args.folder

def main():
    folder = parse_arguments()
    # Load both CSVs
    df = pd.read_csv(f'{folder}/raw_scores.csv')
    cap, fps, _, _, _ = setup_input_video_io(f"{folder}/cropped_scoreboard.mp4")
    cap.release()
    pred = process_scores(df, window_median=int(fps * 7))

    # Rewrite the predictions CSV with cleaned data in this format: frame_id,left_score,right_score,left_confidence,right_confidence. set confidence to 1.0
    frame_ids = np.arange(len(pred))
    pred_df = pd.DataFrame({'frame_id': frame_ids, 'left_score': pred[:, 0], 'right_score': pred[:, 1]})
    pred_df.to_csv(f'{folder}/processed_scores.csv', index=False)

    # ---- Step 4: Plot both predictions ----
    plt.figure("Left", figsize=(12, 6))
    plt.plot(pred[:, 0], label='Pred Left (Smoothed)', color='red', alpha=0.8)
    plt.plot(pred[:, 1], label='Pred Right (Smoothed)', color='green', alpha=0.8)
    plt.title('Smoothed Scores')
    plt.xlabel('Frame ID')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show both figures
    plt.show()

if __name__ == "__main__":
    main()