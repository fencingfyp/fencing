import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from src.momentum_graph.extract_score_increases import extract_score_increases
from src.momentum_graph.evaluate_score_events import refine_score_frames_with_lights
from src.momentum_graph.process_scores import process_scores

def plot_momentum(refined_scores: dict[str, dict[int, int]], n_frames: int):
    """
    Plots the momentum graph across all frames.

    Args:
        refined_scores: dict like {"left": {1: frame_id, 2: frame_id, ...},
                                   "right": {...}}
        n_frames: total number of frames in the video
    """
    momentum = np.zeros(n_frames, dtype=int)

    # Convert to event list: [(frame_id, delta)]
    events = []
    for score, frame_id in refined_scores["left"].items():
        events.append((frame_id, +1))
    for score, frame_id in refined_scores["right"].items():
        events.append((frame_id, -1))

    # Sort by frame in case not ordered
    events.sort(key=lambda x: x[0])

    current = 0
    idx = 0
    for f in range(n_frames):
        # Apply all events that happen at this frame
        while idx < len(events) and events[idx][0] == f:
            current += events[idx][1]
            idx += 1
        momentum[f] = current

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n_frames), momentum, label="Momentum", linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Momentum Over Time")
    plt.xlabel("Frame ID")
    plt.ylabel("Momentum (+ left / - right)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_arguments():
    parser = argparse.ArgumentParser(description="Plot momentum graph from refined score increases.")
    parser.add_argument('folder', type=str, help='Path to the working folder.')
    args = parser.parse_args()
    return args.folder

def main():
    folder = get_arguments()

    # Load processed scores
    scores_df = pd.read_csv(f'{folder}/processed_scores.csv')
    n_frames = len(scores_df)

    
    refined_scores = extract_score_increases(f'{folder}/processed_scores.csv')

    lights_df = pd.read_csv(f'{folder}/processed_lights.csv')
    # rename columns to match row_mapper
    lights_df.rename(columns={"left_light": "left_score", "right_light": "right_score"}, inplace=True)
    lights = process_scores(lights_df, smoothen=False, total_length=n_frames)


    score_occurrences = refine_score_frames_with_lights(lights, refined_scores)


    # Plot momentum graph
    plot_momentum(score_occurrences, n_frames)

if __name__ == "__main__":
    main()