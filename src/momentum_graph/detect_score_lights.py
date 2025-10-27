import argparse
import csv
import os
import cv2
import numpy as np
from src.model.PatchLightDetector import PatchLightDetector
from src.model.Ui import Ui
from src.util import UiCodes, convert_to_opencv_format, convert_from_opencv_format, generate_select_quadrilateral_instructions, \
    setup_input_video_io, setup_output_video_io, setup_output_file, NORMAL_UI_FUNCTIONS
from src.momentum_graph.perform_ocr import convert_from_box_to_rect, convert_from_rect_to_box

MIN_WINDOW_HEIGHT = 780

def get_output_header_row() -> list[str]:
    return ["frame_id", "left_light", "right_light"]

def get_parse_args():
    parser = argparse.ArgumentParser(description="Use OCR to read scoreboard")
    parser.add_argument("output_folder", help="Path to output folder for intermediate/final products")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    return parser.parse_args()

def main():
    args = get_parse_args()
    output_video_path = args.output_video
    output_folder = args.output_folder

    output_csv_path = setup_output_file(output_folder, "raw_lights.csv")
    input_video_path = os.path.join(output_folder, "cropped_score_lights.mp4")

    cap, fps, original_width, original_height, _ = setup_input_video_io(input_video_path)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    aspect_ratio = original_width / original_height
    width = original_width if original_height >= MIN_WINDOW_HEIGHT else int(MIN_WINDOW_HEIGHT * aspect_ratio)
    height = original_height if original_height >= MIN_WINDOW_HEIGHT else MIN_WINDOW_HEIGHT

    ui = Ui("Score Light Detection", width=width, height=height)
    video_writer = None
    if output_video_path:
        video_writer = setup_output_video_io(output_video_path, fps, (width, height + ui.text_box_height))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    # autoscale frame to fit window if needed
    if original_height < MIN_WINDOW_HEIGHT:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    ui.set_fresh_frame(frame)

    left_colour_detector = PatchLightDetector('red')
    right_colour_detector = PatchLightDetector('green')

    left_score_positions = ui.get_n_points(generate_select_quadrilateral_instructions("left fencer score light"))
    right_score_positions = ui.get_n_points(generate_select_quadrilateral_instructions("right fencer score light"))

    frame_id = 1
    with open(output_csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        header_row = get_output_header_row()
        csv_writer.writerow(header_row)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # autoscale frame to fit window if needed
            if original_height < MIN_WINDOW_HEIGHT:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            ui.set_fresh_frame(frame)
            ui.refresh_frame()

            ui.draw_polygon(convert_to_opencv_format(left_score_positions), color=(0, 0, 255)) # cv2 uses BGR
            ui.draw_polygon(convert_to_opencv_format(right_score_positions), color=(0, 255, 0))

            is_left_red = left_colour_detector.check_light(frame, left_score_positions)
            is_right_green = right_colour_detector.check_light(frame, right_score_positions)

            csv_writer.writerow([frame_id, 1 if is_left_red else 0, 1 if is_right_green else 0])

            ui.write_to_ui(
                f"Is left light on: {is_left_red}, Is right light on: {is_right_green}"
            )
            ui.show_frame()

            if video_writer:
                video_writer.write(ui.current_frame)

            delay: int = FULL_DELAY if slow else FAST_FORWARD
            action = ui.take_user_input(delay, [*NORMAL_UI_FUNCTIONS])
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
        ui.close()
 

if __name__ == "__main__":
    main()