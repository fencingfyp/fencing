

import argparse
import pandas as pd
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from src.util import setup_input_video_io

from src.momentum_graph.process_scores import process_scores

def _retroactive_flatten(scores: list[int]) -> list[int]:
    """
    Flatten transient bumps by retroactively reverting any segment whose value is
    later decreased below that value.
    Example: [1,2,2,2,1,2,3] -> [1,1,1,1,1,2,3]
    """
    n = len(scores)
    if n == 0:
        return []

    cleaned = scores[:]  # will be modified
    # stack of (value, start_index)
    stack = [(scores[0], 0)]

    for i in range(1, n):
        curr = scores[i]
        top_val, _ = stack[-1]

        if curr > top_val:
            # new higher segment starts here
            stack.append((curr, i))
        elif curr == top_val:
            # continues current top segment
            continue
        else:  # curr < top_val: need to pop any segments above curr and revert them
            while stack and stack[-1][0] > curr:
                popped_val, popped_start = stack.pop()
                # the value to revert to is the new top's value (if any), else curr
                revert_to = stack[-1][0] if stack else curr
                # revert the frames belonging to the popped segment up to i-1
                for k in range(popped_start, i):
                    cleaned[k] = revert_to
            # now top == curr (or stack empty), if top != curr we need to push curr start
            if not stack or stack[-1][0] != curr:
                # Either we matched an existing lower value, or we start a new segment at i
                stack.append((curr, i))

    return cleaned

def extract_score_increases(df: pd.DataFrame) -> Dict[str, Dict[int, int | None]]:
    """
    Reads per-frame scores and returns the first frame_id where each valid score increase occurs,
    ignoring temporary bumps (e.g. 1->2->1 treated as continuous 1).

    Args:
        csv_path: CSV path with columns [frame_id, left_score, right_score, ...]

    Returns:
        {
            "left": {score: first_frame_id or None},
            "right": {score: first_frame_id or None}
        }
    """
    frames = df["frame_id"].to_list()
    result = {"left": {}, "right": {}}

    for side in ("left", "right"):
        raw_scores = df[f"{side}_score"].ffill().astype(int).to_list()
        cleaned = _retroactive_flatten(raw_scores)

        seen = {}
        last_recorded = cleaned[0] if cleaned else None
        if last_recorded is not None:
            seen[last_recorded] = int(frames[0])

        for i, s in enumerate(cleaned[1:], start=1):
            if s > last_recorded:
                # a true, non-reverted increase
                last_recorded = s
                # record first frame where this cleaned score appears (if not already)
                if s not in seen:
                    seen[s] = int(frames[i])
            else:
                # same or flattened value: do nothing
                pass

        result[side] = seen

    return result

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract score increases from a smoothened graph.")
    parser.add_argument('folder', type=str, help='Path to the working folder.')
    args = parser.parse_args()
    return args.folder

def main():
    folder = parse_arguments()

    # Load both CSVs
    pred = pd.read_csv(f'{folder}/raw_scores.csv')

    cap, fps, _, _, _ = setup_input_video_io(f"{folder}/cropped_scoreboard.mp4")
    cap.release()

    pred = process_scores(pred, window_median=int(fps * 7))
    
    # retroactively flatten the prediction by columns and convert back to DataFrame
    smoothed_df = pd.DataFrame({'frame_id': np.arange(len(pred)), 'left_score': _retroactive_flatten(pred[:, 0].tolist()), 'right_score': _retroactive_flatten(pred[:, 1].tolist())
    })
    smoothed = process_scores(smoothed_df, smoothen=False)

    # --- Left ---
    plt.figure("Left", figsize=(12, 6))
    plt.plot(pred[:, 0], label='Pred Left (Flattened)', color='blue', alpha=0.8)
    plt.plot(smoothed[:, 0], '--', color='red', label='Pred Left')
    plt.title('Flattened Predicted vs Normal Predicted (Left)')
    plt.xlabel('Frame ID')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Right ---
    plt.figure("Right", figsize=(12, 6))
    plt.plot(pred[:, 1], label='Pred Right (Flattened)', color='blue', alpha=0.8)
    plt.plot(smoothed[:, 1], '--', color='red', label='Pred Right')
    plt.title('Flattened Predicted vs Normal Predicted (Right)')
    plt.xlabel('Frame ID')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show both figures
    plt.show()

if __name__ == "__main__":
    main()