import argparse
import os

import cv2
import numpy as np

from src.model import (
    OpenCvUi,
    OrbTracker,
    Quadrilateral,
    SiftTracker,
    TargetTracker,
    Ui,
    UiCodes,
)
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io
from src.util.utils import generate_select_quadrilateral_instructions


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


class CropRegionController:
    def __init__(self, cap, output_path: str | None, ui: Ui, region: str):
        self.cap = cap
        self.output_path = output_path
        self.ui = ui
        self.region = region

        # Persistent state
        self.first_frame: np.ndarray | None = None
        self.positions: Quadrilateral | None = None
        self.dimensions: tuple[int, int] | None = None
        self.dst_corners: np.ndarray | None = None
        self.tracker: TargetTracker | None = None
        self.writer = None

        self.plane_id = "tracked_plane"

    # ---- ENTRY POINT ----

    def start(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.first_frame = frame

        # UI step 1: select corners (returns via callback)
        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions(self.region),
            callback=self.on_initial_corners_selected,
        )

    # ---- CALLBACK 1 ----

    def on_initial_corners_selected(self, positions: list[tuple[int, int]]):
        if len(positions) != 4:
            raise ValueError("Need to select 4 points for the planar region.")

        self.positions = Quadrilateral(positions)

        # Geometry
        self.dimensions = get_planar_dimensions(self.positions)
        self.dst_corners = make_destination_corners(self.dimensions)

        # Tracker
        self.tracker = OrbTracker()
        self.tracker.add_target(self.plane_id, self.first_frame, self.positions)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Writer
        if self.output_path:
            self.writer = setup_output_video_io(
                self.output_path,
                self.cap.get(cv2.CAP_PROP_FPS),
                self.dimensions,
            )

        # Start processing loop (UI-owned)
        self.ui.process_crop_region_loop(
            cap=self.cap,
            frame_callback=self.process_frame,
            writer=self.writer,
        )

    # ---- PER-FRAME CALLBACK ----

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """
        Called by the UI loop on every frame.
        Returns rectified frame + tracked points for overlay.
        """
        tracked = self.tracker.update_all(frame)
        quad = tracked.get(self.plane_id)

        if quad is None:
            quad = self.tracker.get_previous_quad(self.plane_id)

        rectified = self.get_rectified(frame, quad)
        pts = self.tracker.get_target_pts(self.plane_id)

        return rectified, pts

    # ---- HELPERS ----

    def get_rectified(self, frame, quad: Quadrilateral) -> np.ndarray:
        width, height = self.dimensions
        transform = cv2.getPerspectiveTransform(quad.numpy(), self.dst_corners)
        return cv2.warpPerspective(frame, transform, (width, height))


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


def get_target_initial_corners(ui: Ui, frame: np.ndarray) -> Quadrilateral:
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
    output_folder, demo_mode = parse_arguments()
    input_video = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    output_path = (
        setup_output_file(output_folder, CROPPED_SCOREBOARD_VIDEO_NAME)
        if not demo_mode
        else None
    )

    cap, _, width, height, _ = setup_input_video_io(input_video)
    ui = OpenCvUi("Scoreboard Cropping", width=width, height=height)

    controller = CropRegionController(cap, output_path, ui, region="scoreboard")
    controller.start()


if __name__ == "__main__":
    main()
