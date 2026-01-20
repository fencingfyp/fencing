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
    TargetTracker,
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


class CropRegionPipeline:
    def __init__(
        self,
        tracker: TargetTracker,
        plane_id: str,
        dst_corners: np.ndarray,
        dimensions: tuple[int, int],
    ):
        self.tracker = tracker
        self.plane_id = plane_id
        self.dst_corners = dst_corners
        self.dimensions = dimensions

    def process(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        tracked = self.tracker.update_all(frame)
        quad = tracked.get(self.plane_id)

        if quad is None:
            quad = self.tracker.get_previous_quad(self.plane_id)

        rectified = get_rectified_target(frame, quad, self.dst_corners, self.dimensions)

        pts = self.tracker.get_target_pts(self.plane_id)

        return rectified, pts

    def get_quad(self):
        return self.tracker.get_previous_quad(self.plane_id)


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


def get_target_initial_corners(
    ui: PipelineUiDriver, frame: np.ndarray
) -> Quadrilateral:
    """Get initial target corners from user input."""
    positions = ui.get_n_points(
        frame,
        [
            "Select top left",
            "Select top right",
            "Select bottom right",
            "Select bottom left",
        ],
    )
    if len(positions) != 4:
        raise ValueError("Need to select 4 points for the planar region.")
    return Quadrilateral(positions)


def make_destination_corners(dimensions: tuple[int, int]) -> np.ndarray:
    """Creates destination corners for rectification. They can technically
    be any size, but we use the dimensions of the planar region for simplicity."""
    width, height = dimensions
    dst_corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    return dst_corners


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


def crop_region(
    cap: cv2.VideoCapture,
    output_path: str,
    ui: PipelineUiDriver,
):
    ret, frame = cap.read()
    if not ret:
        return

    positions = get_target_initial_corners(ui, frame)
    dimensions = get_planar_dimensions(positions)
    dst_corners = make_destination_corners(dimensions)

    tracker = OrbTracker()
    plane_id = "tracked_plane"
    tracker.add_target(plane_id, frame, positions)

    pipeline = CropRegionPipeline(
        tracker=tracker,
        plane_id=plane_id,
        dst_corners=dst_corners,
        dimensions=dimensions,
    )

    writer = (
        setup_output_video_io(output_path, cap.get(cv2.CAP_PROP_FPS), dimensions)
        if output_path
        else None
    )

    ui.process_crop_region_loop(
        cap=cap,
        pipeline=pipeline,
        writer=writer,
    )


if __name__ == "__main__":
    main()
