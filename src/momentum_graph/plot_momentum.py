import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from src.momentum_graph.util.extract_score_increases import extract_score_increases
from src.momentum_graph.util.evaluate_score_events import refine_score_frames_with_lights
from src.momentum_graph.process_scores import process_scores
from src.util import setup_input_video_io

def plot_momentum(refined_scores: dict[str, dict[int, int]], fps: int):
    """
    Plots the momentum graph across all frames.
    Flattens spikes where both fencers score within 1 second of each other
    by keeping momentum constant over that region.
    """
    # Combine and sort events: (frame, delta)
    events = [(f, +1) for f in refined_scores["left"].values()] + \
             [(f, -1) for f in refined_scores["right"].values()]
    events.sort(key=lambda x: x[0])
    print(events)

    frame_thresh = fps  # within 1 second
    frames, momenta = [0], [0]
    current = 0
    i = 0

    while i < len(events):
        frame_i, delta_i = events[i]

        # Check next event for near-simultaneous opposite score
        if i + 1 < len(events):
            frame_j, delta_j = events[i + 1]
            if abs(frame_i - frame_j) <= frame_thresh and delta_i != delta_j:
                # Flatten: insert both frames with same momentum (no change)
                frames.extend([frame_i, frame_j])
                momenta.extend([current, current])
                i += 2
                continue

        # Normal momentum change
        current += delta_i
        frames.append(frame_i)
        momenta.append(current)
        i += 1

    return np.array(frames), np.array(momenta)

def densify_lights_data(lights: pd.DataFrame, total_length: int) -> np.ndarray:
    lights.rename(columns={"left_light": "left_score", "right_light": "right_score"}, inplace=True)
    lights = process_scores(lights, smoothen=False, total_length=total_length)
    return lights

def get_arguments():
    parser = argparse.ArgumentParser(description="Plot momentum graph from refined score increases.")
    parser.add_argument('folder', type=str, help='Path to the working folder.')
    parser.add_argument('--algorithm', type=str, default="first_increase", help="Refinement algorithm to use")
    return parser.parse_args()

def main():
    args = get_arguments()
    folder = args.folder
    algorithm = args.algorithm

    # load video for fps
    video_path = f'{folder}/cropped_scoreboard.mp4'
    cap, fps, _, _, n_frames = setup_input_video_io(video_path)
    cap.release()

    # Load processed scores
    scores_df = pd.read_csv(f'{folder}/processed_scores.csv', usecols=["frame_id", "left_score", "right_score"])
    lights_df = pd.read_csv(f'{folder}/processed_lights.csv')

    refined_scores = extract_score_increases(scores_df)
    lights = densify_lights_data(lights_df, total_length=n_frames)

    score_occurrences = refine_score_frames_with_lights(lights, refined_scores, fps, algorithm=algorithm)

    print("Refined Score Occurrences:", score_occurrences)

    ## Plot fencer progression

    # --- Load CSV for total frame count ---
    frame_ids = scores_df["frame_id"].to_numpy()
    time_s = frame_ids / fps

    # --- Prepare figure ---
    _, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fencers = ["left", "right"]
    colours = ["tab:blue", "tab:red"]

    for i, fencer in enumerate(fencers):
        ax = axes[i]
        light_curve = lights[:, i]

        # Plot lights activation (continuous)
        ax.plot(time_s, light_curve, color=colours[i], label=f"{fencer.title()} Light", linewidth=1.5)

        # Overlay score progression markers
        scores = refined_scores[fencer]
        for score, frame in scores.items():
            if frame is not None:
                t = frame / fps
                ax.axvline(t, color=colours[i], linestyle="--", alpha=0.6)
                ax.text(t, np.max(light_curve)*0.9, str(score),
                        rotation=90, ha="right", va="center",
                        fontsize=8, color=colours[i])

        ax.set_title(f"{fencer.title()} Fencer Progression")
        ax.set_ylabel("Light Activation")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # print only from seconds 600 to 700
    # for ax in axes:
    #     ax.set_xlim(600, 700)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()

    ## Plot momentum graph
    frames, momenta = plot_momentum(score_occurrences, fps)
    # Plot
    plt.figure("Momentum", figsize=(10, 5))
    plt.plot(frames / fps, momenta, label="Momentum", linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Momentum Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Momentum (+ left / - right)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()