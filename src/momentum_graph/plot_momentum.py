import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from src.momentum_graph.util.extract_score_increases import extract_score_increases
from src.momentum_graph.util.evaluate_score_events import refine_score_frames_with_lights
from src.momentum_graph.process_scores import process_scores
from src.util import setup_input_video_io

def plot_momentum(refined_scores: dict[str, dict[int, int]], n_frames: int, fps: int):
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
    print(refined_scores)

    # Sort by frame in case not ordered
    events.sort(key=lambda x: x[0])
    # Build sparse event-only series: one point per frame where momentum changes
    event_frames = [0]
    event_momenta = [0]
    current = 0
    idx = 0
    while idx < len(events):
        frame = events[idx][0]
        delta = 0
        # accumulate all deltas that occur at the same frame
        while idx < len(events) and events[idx][0] == frame:
            delta += events[idx][1]
            idx += 1
        current += delta
        event_frames.append(frame)
        event_momenta.append(current)

    # convert to numpy arrays for downstream plotting (frames can be converted to seconds with / fps)
    event_frames = np.array(event_frames, dtype=int)
    event_momenta = np.array(event_momenta, dtype=int)
    print(event_frames)
    print(event_momenta)

    current = 0
    idx = 0
    for f in range(n_frames):
        # Apply all events that happen at this frame
        while idx < len(events) and events[idx][0] == f:
            current += events[idx][1]
            idx += 1
        momentum[f] = current


    # Plot line graph and convert frames to seconds on x-axis
    plt.figure(figsize=(10, 5))
    plt.plot(event_frames / fps, event_momenta, label="Momentum", linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Momentum Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Momentum (+ left / - right)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.show()

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
    parser.add_argument('--algorithm', type=str, default="first_increase", help="Refinement algorithm to use")
    args = parser.parse_args()
    return args.folder, args.algorithm

def main():
    folder, algorithm = get_arguments()

    # load video for fps
    video_path = f'{folder}/cropped_scoreboard.mp4'
    cap, fps, _, _, n_frames = setup_input_video_io(video_path)
    cap.release()


    # Load processed scores
    refined_scores = extract_score_increases(f'{folder}/processed_scores.csv')

    lights_df = pd.read_csv(f'{folder}/processed_lights.csv')
    # rename columns to match row_mapper
    lights_df.rename(columns={"left_light": "left_score", "right_light": "right_score"}, inplace=True)
    lights = process_scores(lights_df, smoothen=False, total_length=n_frames)


    score_occurrences = refine_score_frames_with_lights(lights, refined_scores, algorithm=algorithm)


    # Plot momentum graph
    plot_momentum(score_occurrences, n_frames, fps)

if __name__ == "__main__":
    main()