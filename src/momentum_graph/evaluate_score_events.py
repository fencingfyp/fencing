import argparse
import pandas as pd
import cv2
import numpy as np
from src.model.Ui import Ui
from archive.verify_scores_5 import handle_pause
from src.momentum_graph.extract_score_increases import extract_score_increases
from src.util import UiCodes, convert_to_opencv_format, convert_from_opencv_format
from src.model.FrameInfoManager import FrameInfoManager
from src.model.PatchLightDetector import PatchLightDetector
from src.momentum_graph.extract_first_score_occurences import extract_first_score_occurrences
from src.momentum_graph.crop_scoreboard_tracked import setup_input_video_io, setup_output_video_io


DEFAULT_FPS = 50
FULL_DELAY = int(1000 / DEFAULT_FPS)  # milliseconds
HALF_DELAY = FULL_DELAY // 16  # milliseconds

SCORE_TIMEOUT = 5  # seconds to wait before re-trying fencer score detection
# FALSE_ALARM_TIMEOUT = 3  # seconds to wait before re-trying fencer score detection

def refine_score_frames_with_lights(lights: np.ndarray,
                                    score_occ: dict[str, dict[int, int]]) -> dict[str, dict[int, int]]:
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

            refined[side][score] = start

    return refined

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
    parser.add_argument("input_folder", help="Path to input folder containing CSV files")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args()
    return args.input_folder, args.input_video, args.output_video

def main():
    input_folder, input_video_path, output_video_path = get_arguments()
    csv_path = f"{input_folder}/processed_lights.csv"
    scores_csv_path = f"{input_folder}/processed_scores.csv"

    writer = None
    cap = setup_input_video_io(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = FULL_DELAY // 16
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = Ui("Fencing Analysis", width=int(width), height=int(height))
    if output_video_path:
        writer = setup_output_video_io(output_video_path, fps, (width, height + ui.text_box_height))
    frame_info_manager = FrameInfoManager(csv_path, fps, get_header_row(), row_mapper)

    scores_map = extract_score_increases(scores_csv_path)

    left_score_timeout = 0
    right_score_timeout = 0

    left_last_known_score = 0
    right_last_known_score = 0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = frame_info_manager.get_frame_info_at(frame_id).get("id")
        ui.set_fresh_frame(frame)
        ui.refresh_frame()

        is_left_red = detections.get("left_light_on", False)
        is_right_green = detections.get("right_light_on", False)
        
        left_has_score = None
        is_left_timeout = frame_id <= left_score_timeout
        if is_left_red and not is_left_timeout:
            left_has_score = check_for_score("left", frame_id, scores_map, fps, SCORE_TIMEOUT)
            if left_has_score is not None:
                left_score_timeout = frame_id + int(fps * SCORE_TIMEOUT)
                left_last_known_score = left_has_score
        left_string = f"Left scored: {left_last_known_score-1} -> {left_last_known_score}" if is_left_timeout else f"Left score: {left_last_known_score}"


        right_has_score = None
        is_right_timeout = frame_id <= right_score_timeout
        if is_right_green and not is_right_timeout:
            right_has_score = check_for_score("right", frame_id, scores_map, fps, SCORE_TIMEOUT)
            if right_has_score is not None: 
                right_score_timeout = frame_id + int(fps * SCORE_TIMEOUT)
                right_last_known_score = right_has_score
        right_string = f"Right scored: {right_last_known_score-1} -> {right_last_known_score}" if is_right_timeout else f"Right score: {right_last_known_score}"


        # ui.write_to_ui(f"{left_string}, {right_string}, press 'p' to pause, 'q' to quit")
        ui.write_to_ui(f"{left_string}, {right_string}")

        ui.show_frame()

        if writer:
            writer.write(ui.current_frame)

        delay: int = FULL_DELAY if slow else FAST_FORWARD
        action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW, UiCodes.PAUSE])
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:  # q or Esc to quit
            break
        elif action == UiCodes.PAUSE:
            early_exit = handle_pause(ui)

        if early_exit:
            break
        frame_id += 1

    if writer:
        writer.release()

    cap.release()
    ui.close()

def check_for_score(fencer: str, frame_id: int, scores_map: dict[str, dict[int, int]], fps: float, seconds_ahead: int = 5) -> bool:
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