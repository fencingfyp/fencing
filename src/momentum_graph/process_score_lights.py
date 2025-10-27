import pandas as pd
import matplotlib.pyplot as plt
import argparse
  
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and visualize light state changes from CSV.")
    parser.add_argument('folder', type=str, help='Path to the folder containing the input CSV file.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video.')
    args = parser.parse_args()
    return args.folder, args.fps

def remove_false_negatives(series, min_off_len=5):
    """
    Remove false negatives in a binary Series (0=inactive, 1=active).
    If there's a run of 0s shorter than min_off_len between 1s,
    set those 0s to 1s.
    """
    s = series.to_numpy().copy()
    n = len(s)
    if n == 0:
        return series.copy()

    i = 0
    while i < n:
        if s[i] == 0:
            start = i
            while i < n and s[i] == 0:
                i += 1
            end = i  # first non-zero after the run
            # Check if the 0-run is between two 1s
            if start > 0 and end < n and (end - start) < min_off_len:
                s[start:end] = 1
        else:
            i += 1

    return pd.Series(s, index=series.index)


def remove_false_positives(series, min_on_len=5):
    """
    Remove false positives in a binary Series (0=inactive, 1=active).
    If there's a run of 1s shorter than min_on_len between 0s,
    set those 1s to 0s.
    """
    s = series.to_numpy().copy()
    n = len(s)
    if n == 0:
        return series.copy()

    i = 0
    while i < n:
        if s[i] == 1:
            start = i
            while i < n and s[i] == 1:
                i += 1
            end = i  # first zero after the run
            # Check if the 1-run is between two 0s
            if start > 0 and end < n and (end - start) < min_on_len:
                s[start:end] = 0
        else:
            i += 1

    return pd.Series(s, index=series.index)

def main():
    folder, fps = parse_arguments()
    # --- 1. Load and smooth ---
    df = pd.read_csv(f"{folder}/raw_lights.csv")

    df["left_light_smooth"] = remove_false_negatives(df["left_light"], min_off_len=fps//2)
    df["right_light_smooth"] = remove_false_negatives(df["right_light"], min_off_len=fps//2)

    df["left_light_smooth"] = remove_false_positives(df["left_light_smooth"], min_on_len=fps//2)
    df["right_light_smooth"] = remove_false_positives(df["right_light_smooth"], min_on_len=fps//2)

    # --- 2. Plot ---
    plt.figure("Left Light", figsize=(8, 3))
    plt.plot(df["frame_id"], df["left_light_smooth"], label="Left light")
    plt.xlabel("Frame ID")
    plt.ylabel("Light state (0=off, 1=on)")
    plt.legend()
    plt.tight_layout()

    plt.figure("Right Light", figsize=(8, 3))
    plt.plot(df["frame_id"], df["right_light_smooth"], label="Right light", color='orange')
    plt.xlabel("Frame ID")
    plt.ylabel("Light state (0=off, 1=on)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 3. Extract state changes ---
    changes = df.loc[
        (df["left_light_smooth"].diff() != 0) | (df["right_light_smooth"].diff() != 0),
        ["frame_id", "left_light_smooth", "right_light_smooth"]
    ]
    changes.rename(
        columns={"left_light_smooth": "left_light", "right_light_smooth": "right_light"},
        inplace=True,
    )

    print(changes)
    # Optional: save to CSV
    changes.to_csv(f"{folder}/processed_lights.csv", index=False)

if __name__ == "__main__":
    main()