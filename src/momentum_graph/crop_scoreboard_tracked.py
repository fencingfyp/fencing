import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from src.model.Ui import Ui
from src.model.PlanarTracker import PlanarTracker
from src.util import UiCodes, convert_to_opencv_format


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crop and rectify scoreboard region from video (with tracking)")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to folder for intermediate/final products")
    args = parser.parse_args()
    return args.input_video, args.output_folder

def setup_input_video_io(video_path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit(1)

    return cap

def setup_output_video_io(output_path, fps, frame_size) -> cv2.VideoWriter | None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print(f"Error: Could not open video writer {output_path}")
        return exit(1)
    return writer

def setup_output_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    video_path = os.path.join(folder_path, "cropped_scoreboard.mp4")
    print(f"Output video will be saved to: {video_path}")
    return video_path

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
    input_video, output_folder = parse_arguments()
    output_path = setup_output_folder(output_folder)
    crop_region(input_video, output_path, "scoreboard", "Scoreboard Cropping")

def crop_region(input_video: str, output_path: str, plane_id: str, window_name: str = "Crop Planar Region"):

    cap = setup_input_video_io(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    writer = setup_output_video_io(output_path, fps, dimensions)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    slow = False

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
        writer.write(rectified)

        delay = full_delay if slow else fast_forward
        action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW])
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:
            break
    cv2.destroyWindow(window_name)
    writer.release()
    cap.release()
    ui.close()
    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    main()
