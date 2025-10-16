
import os
from pathlib import Path
import cv2
import argparse

from src.model.Ui import Ui
from src.util import UiCodes

import pandas as pd
from collections import Counter

def extract_first_score_occurrences(csv_path, window=50, conf_threshold=0.7, majority_fraction=0.6):
    """
    Reads a CSV and returns the first frame_id where each score is confidently displayed for both players.
    Uses a sliding window and requires a score to appear in >= majority_fraction of the window to count.

    Args:
        csv_path: CSV path with columns [frame_id, left_score, right_score, left_confidence, right_confidence]
        window: number of frames in sliding window
        conf_threshold: minimum confidence to consider OCR reading
        majority_fraction: fraction of frames in window that must match to accept the score

    Returns:
        {
            "left": {score: first_frame_id or None},
            "right": {score: first_frame_id or None}
        }
    """
    df = pd.read_csv(csv_path)
    frame_ids = df["frame_id"].tolist()
    left_scores = df["left_score"].astype(str).tolist()
    right_scores = df["right_score"].astype(str).tolist()
    left_conf = df["left_confidence"].tolist()
    right_conf = df["right_confidence"].tolist()

    first_occurrences = {"left": {}, "right": {}}
    seen_scores = {"left": set(), "right": set()}

    for i in range(len(frame_ids)):
        window_end = min(i + window, len(frame_ids))

        for side, scores, confs in [
            ("left", left_scores, left_conf),
            ("right", right_scores, right_conf),
        ]:
            chunk_scores = scores[i:window_end]
            chunk_confs = confs[i:window_end]
            chunk_frame_ids = frame_ids[i:window_end]

            # filter confident scores
            confident_scores = [
                s for s, c in zip(chunk_scores, chunk_confs)
                if c >= conf_threshold and s != "nan"
            ]
            if not confident_scores:
                continue

            counts = Counter(confident_scores)
            most_common, count = counts.most_common(1)[0]

            try:
                score_val = int(float(most_common))
            except ValueError:
                continue

            # accept only if majority of window matches
            if count / len(chunk_scores) >= majority_fraction:
                if score_val not in seen_scores[side]:
                    first_occurrences[side][score_val] = chunk_frame_ids[0]
                    seen_scores[side].add(score_val)

    # fill missing scores with None
    for side in ["left", "right"]:
        all_scores = set()
        try:
            all_scores |= set(int(float(s)) for s in left_scores if s != "nan")
            all_scores |= set(int(float(s)) for s in right_scores if s != "nan")
        except Exception:
            pass
        for s in all_scores:
            if s not in first_occurrences[side]:
                first_occurrences[side][s] = None
    
    # delete scores > 15
    for side in ["left", "right"]:
        to_delete = [s for s in first_occurrences[side] if s > 15]
        for s in to_delete:
            del first_occurrences[side][s]

    return first_occurrences

def get_header_row() -> list[str]:
    return [
        "fencer_direction",  # "left" or "right"
        *[f"score_{i+1}" for i in range(15)]
    ]

def main():
    parser = argparse.ArgumentParser(description="Analyse video to obtain scoring chart")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_folder", help="Path to output CSV folder")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args() 

    csv_path = args.input_csv
    input_video_path = args.input_video
    output_video_path = args.output_video
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    path_object = Path(args.input_video)
    filename_without_extension = path_object.stem
    output_csv_path = os.path.join(output_folder, f"scoring_events_{filename_without_extension}.csv")

    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video {input_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = max(1, FULL_DELAY // 32)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms, Fast-forward delay: {FAST_FORWARD} ms")

    # UI
    slow = False
    early_exit = False

    video_writer = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = Ui("Score correction", width=int(width), height=int(height))
    if output_video_path:
        print(f"Output video will be saved to: {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height + ui.text_box_height))
        print(width, height + ui.text_box_height)
        if not video_writer.isOpened():
            print(f"Failed to open video writer for {output_video_path}. Check the path and codec.")
            return

    score_map = extract_first_score_occurrences(csv_path)
    current_left_score = 0
    current_right_score = 0
    updated_score_map = {"left": {}, "right": {}}

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        delay: int = FULL_DELAY if slow else FAST_FORWARD
        
        ui.set_fresh_frame(frame)
        ui.refresh_frame()


        string_map = {}
        for side in ["left", "right"]:
            current_score = current_left_score if side == "left" else current_right_score
            next_frame = score_map[side].get(current_score + 1)

            # build string
            if next_frame is not None:
                frames_to_next = (next_frame - frame_id) / fps
                string_map[side] = f"{side.capitalize()} fencer score: {current_score}, next score in {int(frames_to_next)} s"
            else:
                string_map[side] = f"{side.capitalize()} fencer score: {current_score}, next timing unknown"

            # confirmation
            if next_frame == frame_id:
                confirmation = ui.get_confirmation(f"{side.capitalize()} fencer scored! Confirm? (w to confirm, q to cancel)")
                if confirmation:
                    if side == "left":
                        current_left_score += 1
                        updated_score_map[side][current_left_score] = frame_id
                    else:
                        current_right_score += 1
                        updated_score_map[side][current_right_score] = frame_id
                else:
                    # avoid repeated prompts
                    score_map[side][current_score + 1] = None

        ui.write_to_ui(f"{string_map["left"]}, {string_map["right"]}, press 'w' if a score was missed, 'p' to pause, 'q' to quit")

        ui.show_frame()

        if video_writer:
            video_writer.write(ui.current_frame)

        action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW, UiCodes.PAUSE, UiCodes.CONFIRM_INPUT])
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:  # q or Esc to quit
            break
        elif action == UiCodes.PAUSE:
            early_exit = handle_pause(ui)
        elif action == UiCodes.CONFIRM_INPUT:
            # a fencer scored differently from predicted
            # ask which fencer scored
            confirmation = ui.get_fencer("Which fencer scored? n for left, m for right, q to cancel")
            if confirmation == "left":
                current_left_score += 1
                updated_score_map["left"][current_left_score] = frame_id
            elif confirmation == "right":
                current_right_score += 1
                updated_score_map["right"][current_right_score] = frame_id
            
        if early_exit:
            break
        frame_id += 1

    if video_writer:
        video_writer.release()

    cap.release()
    ui.close()

    with open(output_csv_path, "w") as f:
        f.write(",".join(get_header_row()) + "\n")
        for side in ["left", "right"]:
            row = [side] + [str(updated_score_map[side].get(i + 1, "")) for i in range(15)]
            f.write(",".join(row) + "\n")

def handle_pause(ui):
    while True:
        action = ui.take_user_input(100, [UiCodes.PAUSE, UiCodes.QUIT])
        if action == UiCodes.PAUSE:
            return False
        elif action == UiCodes.QUIT:
            return True
            

if __name__ == "__main__":
    main()