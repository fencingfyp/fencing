import argparse
import csv
import os
from pathlib import Path
import cv2
import numpy as np
from src.model.Ui import Ui
from src.util import UiCodes, LEFT_FENCER_SCORE_LIGHT_INSTRUCTIONS, RIGHT_FENCER_SCORE_LIGHT_INSTRUCTIONS, \
LEFT_FENCER_SCORE_INSTRUCTIONS, RIGHT_FENCER_SCORE_INSTRUCTIONS, convert_to_opencv_format, convert_from_opencv_format
from src.model.FrameInfoManager import FrameInfoManager
from src.model.PlanarTracker import PlanarTracker
from src.model.StaticTracker import StaticTracker



DEFAULT_FPS = 50
FULL_DELAY = int(1000 / DEFAULT_FPS)  # milliseconds
HALF_DELAY = FULL_DELAY // 2  # milliseconds


def main():
    parser = argparse.ArgumentParser(description="Perform planar tracking and output csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to output folder for CSV files")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args() 

    output_folder = args.output_folder
    video_path = args.input_video

    os.makedirs(output_folder, exist_ok=True)
    path_object = Path(args.input_video)
    filename_without_extension = path_object.stem
    csv_path = os.path.join(output_folder, f"tracked_planes_{filename_without_extension}.csv")
    print(f"Output CSV will be saved to: {csv_path}")

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = FULL_DELAY // 8
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = Ui("Planar tracking", width=int(width), height=int(height))
    writer = None
    if args.output_video:
        print(f"Output video will be saved to: {args.output_video}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height + ui.text_box_height))
        print(width, height + ui.text_box_height)
        if not writer.isOpened():
            print(f"Failed to open video writer for {args.output_video}. Check the path and codec.")
            return

    # Setup video analysis
    # very roughly load the first frame of the video
    temp_cap = cv2.VideoCapture(video_path)
    ret, frame = temp_cap.read()
    temp_cap.release()
    if not ret:
        print(f"Error: Could not read frame from video {video_path}")
        return

    ui.set_fresh_frame(frame)
    piste_positions = ui.get_piste_positions()
    if len(piste_positions) != 4:
        print("Piste positions not fully selected, exiting.")
        return

    left_fencer_score_light_positions = ui.get_n_points(LEFT_FENCER_SCORE_LIGHT_INSTRUCTIONS)
    if len(left_fencer_score_light_positions) != 4:
        print("Left fencer score light positions not fully selected, exiting.")
        return
    
    right_fencer_score_light_positions = ui.get_n_points(RIGHT_FENCER_SCORE_LIGHT_INSTRUCTIONS)
    if len(right_fencer_score_light_positions) != 4:
        print("Right fencer score light positions not fully selected, exiting.")
        return

    left_fencer_score_display_positions = ui.get_n_points(LEFT_FENCER_SCORE_INSTRUCTIONS)
    if len(left_fencer_score_display_positions) != 4:
        print("Left fencer score display positions not fully selected, exiting.")
        return

    right_fencer_score_display_positions = ui.get_n_points(RIGHT_FENCER_SCORE_INSTRUCTIONS)
    if len(right_fencer_score_display_positions) != 4:
        print("Right fencer score display positions not fully selected, exiting.")
        return

    planar_tracker = StaticTracker()
    # planar_tracker.add_target("piste", frame, convert_to_opencv_format(piste_positions))
    # planar_tracker.add_target("left_fencer_score_light", frame, convert_to_opencv_format(left_fencer_score_light_positions))
    # planar_tracker.add_target("right_fencer_score_light", frame, convert_to_opencv_format(right_fencer_score_light_positions))
    planar_tracker.add_target("left_fencer_score_display", frame, convert_to_opencv_format(left_fencer_score_display_positions))
    planar_tracker.add_target("right_fencer_score_display", frame, convert_to_opencv_format(right_fencer_score_display_positions))

    frame_id = 0

    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        header_row = get_header_row()
        csv_writer.writerow(header_row)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ui.set_fresh_frame(frame)

            tracked_positions = planar_tracker.update_all(frame)
            piste_positions_tracked = tracked_positions.get("piste", None)
            left_fencer_score_light_tracked = tracked_positions.get("left_fencer_score_light", None)
            right_fencer_score_light_tracked = tracked_positions.get("right_fencer_score_light", None)
            left_fencer_score_display_tracked = tracked_positions.get("left_fencer_score_display", None)
            right_fencer_score_display_tracked = tracked_positions.get("right_fencer_score_display", None)

            ui.draw_polygon(piste_positions_tracked, color=(0, 255, 0))
            ui.draw_polygon(left_fencer_score_light_tracked, color=(0, 0, 255)) # cv2 uses BGR
            ui.draw_polygon(right_fencer_score_light_tracked, color=(0, 255, 0))
            ui.draw_polygon(left_fencer_score_display_tracked, color=(255, 255, 255))
            ui.draw_polygon(right_fencer_score_display_tracked, color=(255, 255, 255))

            ui.show_frame()
            
            if writer:
                writer.write(ui.current_frame)

            write_to_csv(csv_writer, frame_id, tracked_positions)

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

    if writer:
        writer.release()

    cap.release()
    ui.close()

def write_to_csv(writer, frame_id: int, tracked_positions: dict[str, np.ndarray]) -> None:
    for target_id, pts in tracked_positions.items():
        if pts is None or len(pts) == 0:
            continue
        if pts.shape != (4, 1, 2):
            raise ValueError(f"Unexpected shape for points: {pts.shape}")
        # map the 4 points to a flat list
        flat_pts = pts.flatten().tolist()  # [x1, y1, x2, y2, x3, y3, x4, y4]
        row = [frame_id, target_id] + flat_pts
        writer.writerow(row)
    

def get_header_row() -> list[str]:
    return ["frame_id", "target_id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]

if __name__ == "__main__":
    main()