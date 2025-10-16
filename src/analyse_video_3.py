import argparse
import cv2
import numpy as np
from model.Ui import Ui
from util import UiCodes, LEFT_FENCER_SCORE_LIGHT_INSTRUCTIONS, RIGHT_FENCER_SCORE_LIGHT_INSTRUCTIONS, convert_to_opencv_format, convert_from_opencv_format
from model.FrameInfoManager import FrameInfoManager
from model.PlanarTracker import PlanarTarget, PlanarTracker

CSV_COLS = 58  # 7 + 17 * 3
NUM_KEYPOINTS = 17

LEFT_FENCER_ID = 0
RIGHT_FENCER_ID = 1

DEFAULT_FPS = 50
FULL_DELAY = int(1000 / DEFAULT_FPS)  # milliseconds
HALF_DELAY = FULL_DELAY // 2  # milliseconds

# planar tracking parameters
MIN_INLIERS = 15

def get_piste_centre_line(positions: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(positions) != 4:
        raise ValueError("Need exactly 4 positions to define the piste corners.")
    
    # Take the average of the top and bottom lines to get center line
    left_x = (positions[0][0] + positions[3][0]) // 2
    left_y = (positions[0][1] + positions[3][1]) // 2
    right_x = (positions[1][0] + positions[2][0]) // 2
    right_y = (positions[1][1] + positions[2][1]) // 2
    return (left_x, left_y), (right_x, right_y)

def main():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args() 

    csv_path = args.input_csv
    video_path = args.input_video

    writer = None
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    HALF_DELAY = FULL_DELAY // 2
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = Ui("Fencing Analysis", width=int(width), height=int(height))
    if args.output_video:
        print(f"Output video will be saved to: {args.output_video}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height + ui.text_box_height))
        print(width, height + ui.text_box_height)
        if not writer.isOpened():
            print(f"Failed to open video writer for {args.output_video}. Check the path and codec.")
            return
    frame_manager = FrameInfoManager(csv_path, fps)

    # Setup video analysis
    # very roughly load the first frame of the video
    temp_cap = cv2.VideoCapture(video_path)
    ret, frame = temp_cap.read()
    temp_cap.release()
    if not ret:
        print(f"Error: Could not read frame from video {video_path}")
        return
    if frame is None:
        print("No frames found in CSV, exiting.")
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


    planar_tracker = PlanarTracker()
    planar_tracker.add_target("piste", frame, convert_to_opencv_format(piste_positions))
    planar_tracker.add_target("left_fencer_score_light", frame, convert_to_opencv_format(left_fencer_score_light_positions))
    planar_tracker.add_target("right_fencer_score_light", frame, convert_to_opencv_format(right_fencer_score_light_positions))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = frame_manager.get_frame_info_at(frame_id)
        ui.set_fresh_frame(frame)

        # track fencers
        left_fencer_position = detections.get(LEFT_FENCER_ID, None)
        right_fencer_position = detections.get(RIGHT_FENCER_ID, None)

        tracked_positions = planar_tracker.update_all(frame)
        piste_positions_tracked = tracked_positions.get("piste", piste_positions)
        left_fencer_score_light_tracked = tracked_positions.get("left_fencer_score_light", None)
        right_fencer_score_light_tracked = tracked_positions.get("right_fencer_score_light", None)
        adapted_positions = convert_from_opencv_format(piste_positions_tracked)
        piste_centre_line = get_piste_centre_line(adapted_positions)

        ui.draw_polygon(left_fencer_score_light_tracked, color=(255, 255, 255))
        ui.draw_polygon(right_fencer_score_light_tracked, color=(255, 255, 255))

        ui.draw_piste_centre_line(piste_centre_line)
        if left_fencer_position is not None:
            ui.draw_fencer_centrepoint(left_fencer_position, is_left=True)
        if right_fencer_position is not None:
            ui.draw_fencer_centrepoint(right_fencer_position, is_left=False)
        ui.draw_fencer_positions_on_piste(left_fencer_position, right_fencer_position, piste_centre_line)
        ui.show_frame()
        
        if writer:
            writer.write(ui.current_frame)

        delay: int = FULL_DELAY if slow else HALF_DELAY
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


if __name__ == "__main__":
    main()