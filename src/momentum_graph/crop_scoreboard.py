import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from src.model.Ui import Ui
from src.model.PlanarTracker import PlanarTracker
from src.util import UiCodes, convert_to_opencv_format, setup_input_video_io, setup_output_video_io, setup_output_file, \
    NORMAL_UI_FUNCTIONS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Crop and rectify scoreboard region from video (with tracking)")
    parser.add_argument("output_folder", help="Path to folder for intermediate/final products")
    parser.add_argument("--demo", action="store_true", help="If set, doesn't output anything")
    args = parser.parse_args()
    return args.output_folder, args.demo

def get_planar_dimensions(scoreboard_positions):
    scoreboard_corners = np.array(scoreboard_positions, dtype=np.float32)

    # Compute initial scoreboard dimensions
    scoreboard_width = int(max(
        np.linalg.norm(scoreboard_corners[0] - scoreboard_corners[1]),
        np.linalg.norm(scoreboard_corners[2] - scoreboard_corners[3])
    ))
    scoreboard_height = int(max(
        np.linalg.norm(scoreboard_corners[0] - scoreboard_corners[3]),
        np.linalg.norm(scoreboard_corners[1] - scoreboard_corners[2])
    ))
    return scoreboard_width, scoreboard_height
    

def main():
    output_folder, demo_mode = parse_arguments()
    input_video = os.path.join(output_folder, "original.mp4")
    output_path = setup_output_file(output_folder, "cropped_scoreboard.mp4") if not demo_mode else None
    crop_region(input_video, output_path, "scoreboard", "Scoreboard Cropping")

def crop_region(input_video: str, output_path: str, plane_id: str, window_name: str = "Crop Planar Region"):
    write_output = output_path is not None

    cap, fps, width, height, _ = setup_input_video_io(input_video)

    full_delay = int(1000 / fps)
    fast_forward = min(full_delay // 8, 1)
    print(f"Video FPS: {fps:.2f}")

    ui = Ui(window_name, width=width, height=height)

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Get initial scoreboard corners from user
    ui.set_fresh_frame(frame)
    positions = ui.get_n_points([
        "Select top left", "Select top right", "Select bottom right", "Select bottom left"
    ])

    if len(positions) != 4:
        print("Error: Need to select 4 points for the planar region.")
        return
    
    dimensions = get_planar_dimensions(positions)
    width, height = dimensions
    # Destination points (canonical rectangle)
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Initialise planar tracker
    planar_tracker = PlanarTracker()
    planar_tracker.add_target(plane_id, frame, convert_to_opencv_format(positions))

    if write_output:
        writer = setup_output_video_io(output_path, fps, dimensions)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    slow = False
    early_exit = False

    print("Tracking and warping in progress...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ui.set_fresh_frame(frame)
        tracked_positions = planar_tracker.update_all(frame)
        scoreboard_positions_tracked = tracked_positions.get(plane_id, None)
        ui.show_frame()

        if scoreboard_positions_tracked is not None and len(scoreboard_positions_tracked) == 4:
            tracked_corners = np.array(scoreboard_positions_tracked, dtype=np.float32)
            # Compute a new perspective transform for the current frame
            transform_matrix = cv2.getPerspectiveTransform(tracked_corners, dst_corners)
            rectified = cv2.warpPerspective(frame, transform_matrix, dimensions)
        else:
            # Fallback if tracking failed
            rectified = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(rectified, "Tracking lost", (10, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, rectified)
        if write_output:
            writer.write(rectified)

        delay = full_delay if slow else fast_forward
        action = ui.take_user_input(delay, [*NORMAL_UI_FUNCTIONS])
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:
            break
        elif action == UiCodes.PAUSE:
            early_exit = ui.handle_pause()
        
        if early_exit:
            break

    cv2.destroyWindow(window_name)
    if write_output:
        writer.release()
    cap.release()
    ui.close()
    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    main()
