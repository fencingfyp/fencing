import pandas as pd
import matplotlib.pyplot as plt
import argparse
  
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and visualize light state changes from CSV.")
    parser.add_argument('folder', type=str, help='Path to the folder containing the input CSV file.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video.')
    args = parser.parse_args()
    return args.folder, args.fps

def remove_false_positives(series, min_on_len=5):
    s = series.copy().to_numpy()
    start = 0
    while start < len(s):
        val = s[start]
        end = start
        while end < len(s) and s[end] == val:
            end += 1
        run_length = end - start
        # If a short 1-run inside 0s, remove it
        if val == 1 and run_length < min_on_len:
            s[start:end] = 0
        start = end
    return pd.Series(s, index=series.index)

def main():
    folder, fps = parse_arguments()
    # --- 1. Load and smooth ---
    df = pd.read_csv(f"{folder}/raw_lights.csv")

    df["left_light_smooth"] = remove_false_positives(df["left_light"], min_on_len=fps//2)
    df["right_light_smooth"] = remove_false_positives(df["right_light"], min_on_len=fps//2)

    # --- 2. Plot ---
    plt.figure(figsize=(8, 3))
    plt.plot(df["frame_id"], df["left_light_smooth"], label="Left light")
    plt.plot(df["frame_id"], df["right_light_smooth"], label="Right light")
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