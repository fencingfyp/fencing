import argparse
import csv
import os

import cv2
import pandas as pd

from model.OpenCvUi import NORMAL_UI_FUNCTIONS, OpenCvUi, UiCodes
from scripts.momentum_graph.perform_ocr import validate_input_video
from scripts.momentum_graph.process_scores import densify_frames
from scripts.momentum_graph.util.file_names import (
    LAB_VALUES_CSV,
    LAB_VALUES_VIDEO_NAME,
    LIGHTS_GT_CSV,
)
from src.model.PatchLightDetector import PatchLightDetector
from src.util.file_names import CROPPED_SCORE_LIGHTS_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io

MIN_WINDOW_HEIGHT = 780


def get_output_header_row() -> list[str]:
    headers = ["frame_id", "label", "l", "a", "b"]
    return headers


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="Obtain the LAB values of score lights"
    )
    parser.add_argument(
        "output_folder", help="Path to output folder for intermediate/final products"
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="If set, outputs video with LAB value results",
    )
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output anything"
    )
    parser.add_argument(
        "--debug", action="store_true", help="If set, enables debug prints"
    )
    return parser.parse_args()


def main():
    args = get_parse_args()
    output_video = args.output_video
    output_folder = args.output_folder
    demo_mode = args.demo

    output_csv_path = setup_output_file(output_folder, LAB_VALUES_CSV)
    input_video_path = os.path.join(output_folder, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
    original_video_path = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    lights_gt_csv_path = os.path.join(output_folder, LIGHTS_GT_CSV)

    validate_input_video(original_video_path, input_video_path)

    cap, fps, original_width, original_height, total_length = setup_input_video_io(
        input_video_path
    )
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    lights_gt = pd.read_csv(lights_gt_csv_path)
    lights_gt.rename(
        columns={"left": "left_score", "right": "right_score"}, inplace=True
    )
    lights_gt = densify_frames(lights_gt, total_length=total_length)
    lights_gt.rename(
        columns={"left_score": "left", "right_score": "right"}, inplace=True
    )

    # UI
    slow = False
    early_exit = False

    ui = OpenCvUi(
        "LAB values collection",
        width=original_width,
        height=original_height,
        display_height=MIN_WINDOW_HEIGHT,
    )
    video_writer: cv2.VideoWriter = None
    if output_video:
        output_video_path = os.path.join(output_folder, LAB_VALUES_VIDEO_NAME)
        video_writer = setup_output_video_io(
            output_video_path, fps, ui.get_output_dimensions()
        )

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    left_colour_detector = PatchLightDetector("red")
    right_colour_detector = PatchLightDetector("green")

    left_score_positions = ui.get_quadrilateral(frame, "left fencer score light")
    right_score_positions = ui.get_quadrilateral(frame, "right fencer score light")

    frame_id = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to beginning
    mode = "a" if demo_mode else "w"
    with open(output_csv_path, mode, newline="") as f:
        csv_writer = csv.writer(f)
        header_row = get_output_header_row()
        if not demo_mode:
            csv_writer.writerow(header_row)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ui.set_fresh_frame(frame)
            ui.refresh_frame()

            left_lab_values = left_colour_detector.get_lab_values(
                ui.get_current_frame(), left_score_positions
            )
            right_lab_values = right_colour_detector.get_lab_values(
                ui.get_current_frame(), right_score_positions
            )

            ui.draw_quadrilateral(left_score_positions, color=(0, 0, 255))
            ui.draw_quadrilateral(right_score_positions, color=(0, 255, 0))

            if not demo_mode:
                if int(lights_gt.at[frame_id, "left"]) == 1:
                    csv_writer.writerow([frame_id, "red", *left_lab_values])
                if int(lights_gt.at[frame_id, "right"]) == 1:
                    csv_writer.writerow([frame_id, "green", *right_lab_values])

            ui.write_to_ui(
                f"Frame: {frame_id} | Left LAB: {left_lab_values} | Right LAB: {right_lab_values}"
            )
            ui.show_frame()

            if video_writer:
                video_writer.write(ui.current_frame)

            delay: int = FULL_DELAY if slow else FAST_FORWARD
            action = ui.get_user_input(delay, [*NORMAL_UI_FUNCTIONS])
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

        if video_writer:
            video_writer.release()

        cap.release()
        ui.close_additional_windows()


if __name__ == "__main__":
    main()
