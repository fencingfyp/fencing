"Script to crop and rectify scoreboard region from video using planar tracking."

import argparse
import os

import cv2
import numpy as np

from src.model import (
    OpenCvUi,
    OrbTracker,
    PipelineUiDriver,
    Quadrilateral,
    SiftTracker,
    UiCodes,
)
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io


def parse_arguments():
    """Parse command line arguments for cropping scoreboard region."""
    parser = argparse.ArgumentParser(
        description="Crop and rectify scoreboard region from video (with tracking)"
    )
    parser.add_argument(
        "output_folder", help="Path to folder for intermediate/final products"
    )
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output anything"
    )
    args = parser.parse_args()
    return args.output_folder, args.demo


def get_planar_dimensions(scoreboard_positions: Quadrilateral) -> tuple[int, int]:
    scoreboard_corners = scoreboard_positions.numpy().astype(np.float32)

    # Compute initial scoreboard dimensions
    scoreboard_width = int(
        max(
            np.linalg.norm(scoreboard_corners[0] - scoreboard_corners[1]),
            np.linalg.norm(scoreboard_corners[2] - scoreboard_corners[3]),
        )
    )
    scoreboard_height = int(
        max(
            np.linalg.norm(scoreboard_corners[0] - scoreboard_corners[3]),
            np.linalg.norm(scoreboard_corners[1] - scoreboard_corners[2]),
        )
    )
    return scoreboard_width, scoreboard_height


def main():
    """Main function to crop and rectify scoreboard region from video."""
    output_folder, demo_mode = parse_arguments()
    input_video = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    output_path = (
        setup_output_file(output_folder, CROPPED_SCOREBOARD_VIDEO_NAME)
        if not demo_mode
        else None
    )
    cap, _, width, height, _ = setup_input_video_io(input_video)
    ui = OpenCvUi("Scoreboard Cropping", width=width, height=height)
    crop_region(cap, output_path, ui)


def get_rectified_target(
    frame,
    tracked_corners: Quadrilateral,
    dst_corners: np.ndarray,
    dimensions: tuple[int, int],
) -> np.ndarray:
    """Get the frame rectified to the tracked planar region."""
    width, height = dimensions
    transform_matrix = cv2.getPerspectiveTransform(tracked_corners.numpy(), dst_corners)
    rectified = cv2.warpPerspective(frame, transform_matrix, (width, height))
    return rectified


def get_scoreboard_initial_corners(
    ui: PipelineUiDriver, frame: np.ndarray
) -> Quadrilateral:
    """Get initial scoreboard corners from user input."""
    ui.set_fresh_frame(frame)
    positions = ui.get_n_points(
        [
            "Select top left",
            "Select top right",
            "Select bottom right",
            "Select bottom left",
        ]
    )
    if len(positions) != 4:
        raise ValueError("Need to select 4 points for the planar region.")
    return Quadrilateral(positions)


def get_exclude_regions(
    frame: np.ndarray,
    positions: Quadrilateral,
    dst_corners: np.ndarray,
    dimensions: tuple[int, int],
) -> list[Quadrilateral]:
    width, height = dimensions
    # calculate exclude regions (score area) to improve tracking
    obtain_exclude_regions_ui = OpenCvUi(
        "Exclude Regions for Tracking", width=width, height=height, display_width=1280
    )
    rectified_for_exclude = get_rectified_target(
        frame,
        positions,
        dst_corners,
        dimensions,
    )
    exclude_regions: list[Quadrilateral] = []
    for i in range(3):  # 3 score regions to exclude, left, right, timer
        obtain_exclude_regions_ui.set_fresh_frame(rectified_for_exclude)
        exclude_region = obtain_exclude_regions_ui.get_quadrilateral(
            f"exclude region {i+1} for tracking",
            return_original_size=True,
        )
        exclude_regions.append(exclude_region)

    H_inv = cv2.getPerspectiveTransform(
        dst_corners.astype(np.float32),
        positions.opencv_format(),
    )

    transformed_exclude_regions: list[Quadrilateral] = []

    for region in exclude_regions:
        # rectified plane corners
        rect_corners = region.opencv_format()

        # map back to original image
        orig_corners = cv2.perspectiveTransform(rect_corners, H_inv).reshape(-1, 2)

        transformed_exclude_regions.append(Quadrilateral(orig_corners))

    obtain_exclude_regions_ui.close()
    return transformed_exclude_regions


def crop_region(
    cap: cv2.VideoCapture,
    output_path: str,
    ui: OpenCvUi,
):
    """Crop and rectify a planar region from the input video using tracking."""
    write_output = output_path is not None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    full_delay = int(1000 / fps)
    fast_forward = 1
    print(f"Video FPS: {fps:.2f}")

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

    # transformed_exclude_regions = get_exclude_regions(
    #     frame,
    #     positions,
    #     dst_corners,
    #     dimensions,
    # )
    ui.set_fresh_frame(frame)
    # for region in transformed_exclude_regions:
    #     ui.draw_polygon(np.array(region.to_drawable()), (0, 0, 255))

    # Initialise planar tracker
    planar_tracker = OrbTracker()
    plane_id = "tracked_plane"
    planar_tracker.add_target(
        plane_id,
        frame,
        positions,
    )

    if write_output:
        print(output_path, fps, dimensions)
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

        if scoreboard_positions_tracked is None:
            print("Tracking lost for frame:", int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            last_known_positions = planar_tracker.get_previous_quad(plane_id)
            scoreboard_positions_tracked = last_known_positions

        rectified = get_rectified_target(
            frame,
            scoreboard_positions_tracked,
            dst_corners,
            dimensions,
        )

        # visualise tracking points
        pts = planar_tracker.get_target_pts(plane_id)
        pts = [(pt[0][0], pt[0][1]) for pt in pts]
        ui.plot_points(pts, (0, 255, 0))
        ui.show_frame()

        cv2.imshow("cropped_view", rectified)
        if write_output:
            writer.write(rectified)

        delay = full_delay if slow else fast_forward
        action = ui.get_user_input(delay)
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
