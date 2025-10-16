import argparse
import csv
import os
import cv2
import numpy as np
from src.model.PatchLightDetector import PatchLightDetector
from src.model.Ui import Ui
from src.util import UiCodes, convert_to_opencv_format, convert_from_opencv_format, generate_select_quadrilateral_instructions
from src.momentum_graph.perform_ocr import convert_from_box_to_rect, convert_from_rect_to_box

MIN_WINDOW_HEIGHT = 780

def get_output_header_row() -> list[str]:
    return ["frame_id", "left_light", "right_light"]

def main():
    parser = argparse.ArgumentParser(description="Use OCR to read scoreboard")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to output folder for intermediate/final products")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args() 

    input_video_path = args.input_video
    output_video_path = args.output_video
    output_folder = args.output_folder

    output_csv_path = os.path.join(output_folder, "raw_lights.csv")
    print(f"Output CSV will be saved to: {output_csv_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height
    width = original_width if original_height >= MIN_WINDOW_HEIGHT else int(MIN_WINDOW_HEIGHT * aspect_ratio)
    height = original_height if original_height >= MIN_WINDOW_HEIGHT else MIN_WINDOW_HEIGHT

    ui = Ui("Score Light Detection", width=width, height=height)
    video_writer = None
    if output_video_path:
        print(f"Output video will be saved to: {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height + ui.text_box_height))
        print(width, height + ui.text_box_height)
        if not video_writer.isOpened():
            print(f"Failed to open video writer for {output_video_path}. Check the path and codec.")
            return

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

    frame_id = 0
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
            action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW])
            if action == UiCodes.TOGGLE_SLOW:
                slow = not slow
                print(f"Slow mode {'enabled' if slow else 'disabled'}.")
            elif action == UiCodes.QUIT:  # q or Esc to quit
                break

            if early_exit:
                break
            frame_id += 1

        if video_writer:
            video_writer.release()
        
        cap.release()
        ui.close()
 

if __name__ == "__main__":
    main()