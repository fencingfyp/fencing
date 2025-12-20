"""Crop and rectify score lights region from video (with tracking)"""

import argparse
import os

import cv2
import numpy as np

from scripts.momentum_graph.crop_scoreboard import (
    get_planar_dimensions,
    get_rectified_target,
    get_scoreboard_initial_corners,
)
from scripts.momentum_graph.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    ORIGINAL_VIDEO_NAME,
)
from src.model.tracker import SiftTracker
from src.model.Ui import NORMAL_UI_FUNCTIONS, Ui, UiCodes
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Crop and rectify score lights region from video (with tracking)"
    )
    parser.add_argument(
        "output_folder", help="Path to folder for intermediate/final products"
    )
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output anything"
    )
    args = parser.parse_args()
    return args.output_folder, args.demo


def main():
    output_folder, demo_mode = parse_arguments()
    video_path = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    output_path = (
        setup_output_file(output_folder, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
        if not demo_mode
        else None
    )
    window_name = "Score Lights Cropping"
    plane_id = "score_lights"
    write_output = output_path is not None

    cap, fps, width, height, _ = setup_input_video_io(video_path)

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
    positions = get_scoreboard_initial_corners(ui, frame)

    dimensions = get_planar_dimensions(positions)
    width, height = dimensions
    # Destination points (canonical rectangle)
    dst_corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    # Initialise planar tracker
    planar_tracker = SiftTracker()
    planar_tracker.add_target(plane_id, frame, positions)

    if write_output:
        writer = setup_output_video_io(output_path, fps, dimensions)

    cv2.namedWindow("cropped_view", cv2.WINDOW_NORMAL)
    slow = False
    early_exit = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to beginning
    print("Tracking and warping in progress...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ui.set_fresh_frame(frame)
        tracked_positions = planar_tracker.update_all(frame)
        scoreboard_positions_tracked = tracked_positions.get(plane_id, None)

        if scoreboard_positions_tracked is not None:
            rectified = get_rectified_target(
                frame,
                scoreboard_positions_tracked,
                dst_corners,
                dimensions,
            )
        else:
            # Fallback if tracking failed
            rectified = np.zeros((height, width, 3), dtype=np.uint8)
            print("Tracking lost for frame:", int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

        cv2.imshow("cropped_view", rectified)
        pts = planar_tracker.targets[plane_id].get_points()
        ui.plot_points(pts, (0, 255, 0))
        ui.show_frame()
        if write_output:
            writer.write(rectified)

        delay = full_delay if slow else fast_forward
        action = ui.take_user_input(delay)
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:
            early_exit = True
            break
        elif action == UiCodes.PAUSE:
            early_exit = ui.handle_pause()

        if early_exit:
            break

    cv2.destroyWindow("cropped_view")
    if write_output:
        writer.release()
        print(f"Saved video to: {output_path}")
    cap.release()
    ui.close()


if __name__ == "__main__":
    main()
