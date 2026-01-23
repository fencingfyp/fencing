import argparse

import numpy as np
import pandas as pd

from scripts.momentum_graph.process_scores import densify_frames
from scripts.momentum_graph.util.extract_score_increases import extract_score_increases
from src.model.OpenCvUi import OpenCvUi, UiCodes
from src.util.io import setup_input_video_io, setup_output_video_io

DEFAULT_FPS = 50
FULL_DELAY = int(1000 / DEFAULT_FPS)  # milliseconds
HALF_DELAY = FULL_DELAY // 16  # milliseconds

SCORE_TIMEOUT = 5  # seconds to wait before re-trying fencer score detection
# FALSE_ALARM_TIMEOUT = 3  # seconds to wait before re-trying fencer score detection


def perform_last_activation_algorithm(
    lights: np.ndarray, score_occ: dict[str, dict[int, int]]
) -> dict[str, dict[int, int]]:
    """Process raw_scores.csv to extract last score light activations."""
    """
    For each fencer and score, find the first frame of the latest light activation
    near the score increase frame.

    Args:
        lights: (n,2) binary array where 1 = light on, 0 = off.
        score_occ: dict from extract_last_score_occurrences, e.g.
                   {"left": {1: 120, 2: 450}, "right": {1: 200, 2: 480}}

    Returns:
        Same structure but with frame_ids adjusted to the first frame
        of the detected light activation.
    """
    refined = {"left": {}, "right": {}}
    n_frames = len(lights)
    sides = {"left": 0, "right": 1}

    for side, col in sides.items():
        for score, idx in score_occ[side].items():
            i = int(idx)
            if i < 0 or i >= n_frames:
                refined[side][score] = idx
                print(
                    f"Score {side} {score} at frame {idx} is out of bounds, skipping refinement"
                )
                continue

            # --- Step 1: Walk backward over zeros until we find a 1 ---
            while i >= 0 and lights[i, col] == 0:
                i -= 1
            if i < 0:
                refined[side][score] = idx  # no activation found
                continue

            # --- Step 2: Walk backward until the light turns off (start of activation) ---
            start = i
            while start > 0 and lights[start - 1, col] == 1:
                start -= 1

            # --- Step 3: Optionally, find the end if needed ---
            # end = i
            # while end + 1 < n_frames and lights[end + 1, col] == 1:
            #     end += 1
            # print(
            #     f"Score {side} {score} at frame {idx} adjusted to light activation starting at frame {start}"
            # )

            refined[side][score] = start

    return refined


def perform_first_increase_algorithm(
    lights: np.ndarray, score_occ: dict[str, dict[int, int]], fps: float
) -> dict[str, dict[int, int]]:
    """For each light activation, search forward up to 7 seconds for the
    first score increase and record the activation frame instead of the score frame."""

    max_forward = int(7 * fps)  # allowed forward window in frames
    side_to_col = {"left": 0, "right": 1}
    result: dict[str, dict[int, int]] = {"left": {}, "right": {}}

    # Precompute activation start frames (rising edges)
    activations = {"left": [], "right": []}
    for side, col in side_to_col.items():
        light_col = lights[:, col]
        rising_edges = np.where((light_col[:-1] == 0) & (light_col[1:] == 1))[0] + 1
        activations[side] = rising_edges

    # For each activation, find the first score increase that occurs within 7 seconds *after* it
    for side in ["left", "right"]:
        score_items = list(score_occ[side].items())  # list of (increase_idx, frame)
        score_items.sort(key=lambda x: x[1])  # sort by frame

        for start_frame in activations[side]:
            # find first score frame >= start_frame and <= start_frame + max_forward
            for inc_idx, inc_frame in score_items:
                # skip increases already assigned earlier (prevents overwriting)
                if inc_idx in result[side]:
                    continue
                if inc_frame >= start_frame and inc_frame - start_frame <= max_forward:
                    result[side][inc_idx] = start_frame
                    break  # move to next activation

    return result


def refine_score_frames_with_lights(
    lights: np.ndarray,
    score_occ: dict[str, dict[int, int]],
    fps: float,
    algorithm: str = "last_activation",
) -> dict[str, dict[int, int]]:
    if algorithm == "last_activation":
        return perform_last_activation_algorithm(lights, score_occ)
    elif algorithm == "first_increase":
        return perform_first_increase_algorithm(lights, score_occ, fps)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def row_mapper(row: list[str]) -> dict[str, any]:
    left = row[1]
    right = row[2]
    return {
        "id": "id",  # required by FrameInfoManager
        "left_light_on": int(left),
        "right_light_on": int(right),
    }


# id column is required for FrameInfoManager
def get_header_row() -> list[str]:
    return ["frame_id", "left_light", "right_light"]


def get_arguments():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument(
        "input_folder", help="Path to input folder containing CSV files"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="first_increase",
        help="Refinement algorithm to use",
    )
    parser.add_argument(
        "--output_video", help="Path to output video file (optional)", default=None
    )
    args = parser.parse_args()
    return args.input_folder, args.input_video, args.output_video, args.algorithm


def main():
    input_folder, input_video_path, output_video_path, algorithm = get_arguments()
    csv_path = f"{input_folder}/processed_lights.csv"
    scores_csv_path = f"{input_folder}/processed_scores.csv"

    writer = None
    cap, fps, width, height, total_frames = setup_input_video_io(input_video_path)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = FULL_DELAY // 16
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    ui = OpenCvUi("Fencing Analysis", width=int(width), height=int(height))
    if output_video_path:
        writer = setup_output_video_io(
            output_video_path, fps, ui.get_output_dimensions()
        )

    scores_df = pd.read_csv(
        f"{input_folder}/processed_scores.csv",
        usecols=["frame_id", "left_score", "right_score"],
    )
    scores_map = extract_score_increases(scores_df)

    # extract lights info into a np array
    lights_df = pd.read_csv(csv_path)
    # rename columns to match row_mapper
    lights_df.rename(
        columns={"left_light": "left_score", "right_light": "right_score"}, inplace=True
    )
    lights = densify_frames(lights, total_frames).to_numpy()

    score_occurrences = refine_score_frames_with_lights(
        lights, scores_map, fps, algorithm=algorithm
    )

    left_last_known_score = 0
    right_last_known_score = 0

    left_score_timeout = -1
    right_score_timeout = -1

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ui.set_fresh_frame(frame)
        ui.refresh_frame()
        left_has_score = (
            score_occurrences["left"].get(left_last_known_score + 1, None) == frame_id
        )
        is_left_timeout = frame_id <= left_score_timeout
        if not is_left_timeout and left_has_score:
            left_score_timeout = frame_id + int(fps * SCORE_TIMEOUT)
            left_last_known_score += 1
        left_string = (
            f"Left scored: {left_last_known_score-1} -> {left_last_known_score}"
            if is_left_timeout
            else f"Left score: {left_last_known_score}"
        )

        right_has_score = (
            score_occurrences["right"].get(right_last_known_score + 1, None) == frame_id
        )
        is_right_timeout = frame_id <= right_score_timeout
        if not is_right_timeout and right_has_score:
            right_score_timeout = frame_id + int(fps * SCORE_TIMEOUT)
            right_last_known_score += 1
        right_string = (
            f"Right scored: {right_last_known_score-1} -> {right_last_known_score}"
            if is_right_timeout
            else f"Right score: {right_last_known_score}"
        )

        # ui.write_to_ui(f"{left_string}, {right_string}, press 'p' to pause, 'q' to quit")
        ui.write_to_ui(f"{left_string}, {right_string}")

        ui.show_frame()

        if writer:
            writer.write(ui.current_frame)

        delay: int = FULL_DELAY if slow else FAST_FORWARD
        action = ui.take_user_input(
            delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW, UiCodes.PAUSE]
        )
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:  # q or Esc to quit
            break
        elif action == UiCodes.PAUSE:
            early_exit = ui.handle_pause()

        if early_exit:
            break
        frame_id += 1

    if writer:
        writer.release()

    cap.release()
    ui.close()


def check_for_score(
    fencer: str,
    frame_id: int,
    scores_map: dict[str, dict[int, int]],
    fps: float,
    seconds_ahead: int = 5,
) -> bool:
    fencer_scores = scores_map[fencer]
    if not fencer_scores:
        raise ValueError(f"No scores found for fencer: {fencer}")
    # Check if there are any scores within the specified time frame
    for score, score_frame_timing in scores_map[fencer].items():
        if score_frame_timing is None:
            continue
        if frame_id < score_frame_timing <= frame_id + int(fps * seconds_ahead):
            return score
    return None


if __name__ == "__main__":
    main()
