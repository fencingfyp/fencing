import argparse
import csv
import os

import cv2

from scripts.momentum_graph.perform_ocr import validate_input_video
from scripts.momentum_graph.util.file_names import SCORE_LIGHTS_VIDEO_NAME
from src.model.OpenCvUi import NORMAL_UI_FUNCTIONS, OpenCvUi, UiCodes
from src.model.PatchLightDetector import Colour, PatchLightDetector
from src.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
    ORIGINAL_VIDEO_NAME,
)
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io

MIN_WINDOW_HEIGHT = 780


def get_output_header_row(is_debug: bool = False) -> list[str]:
    headers = ["frame_id", "left_light", "right_light"]
    if is_debug:
        headers.extend(["left_debug_info", "right_debug_info"])
    return headers


def get_parse_args():
    parser = argparse.ArgumentParser(description="Detect score lights")
    parser.add_argument(
        "output_folder", help="Path to output folder for intermediate/final products"
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="If set, outputs video with OCR results",
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
    debug_mode = args.debug

    output_csv_path = setup_output_file(output_folder, DETECT_LIGHTS_OUTPUT_CSV_NAME)
    input_video_path = os.path.join(output_folder, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
    original_video_path = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)

    validate_input_video(original_video_path, input_video_path)

    cap, fps, original_width, original_height, _ = setup_input_video_io(
        input_video_path
    )
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    ui = OpenCvUi(
        "Score Light Detection",
        width=original_width,
        height=original_height,
        display_height=MIN_WINDOW_HEIGHT,
    )
    video_writer: cv2.VideoWriter = None
    if output_video:
        output_video_path = os.path.join(output_folder, SCORE_LIGHTS_VIDEO_NAME)
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
        header_row = get_output_header_row(is_debug=debug_mode)
        if not demo_mode:
            csv_writer.writerow(header_row)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ui.set_fresh_frame(frame)
            ui.refresh_frame()

            is_left_red = (
                left_colour_detector.classify(frame, left_score_positions) == Colour.RED
            )
            is_right_green = (
                right_colour_detector.classify(frame, right_score_positions)
                == Colour.GREEN
            )

            ui.draw_quadrilateral(left_score_positions, color=(0, 0, 255))
            ui.draw_quadrilateral(right_score_positions, color=(0, 255, 0))

            if not demo_mode:
                row = [frame_id, 1 if is_left_red else 0, 1 if is_right_green else 0]
                if debug_mode:
                    debug_info = [
                        left_colour_detector.get_debug_info(),
                        right_colour_detector.get_debug_info(),
                    ]
                    row.extend(debug_info)
                csv_writer.writerow(row)

            ui.write_to_ui(
                f"left light on: {is_left_red}, right light on: {is_right_green}"
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
