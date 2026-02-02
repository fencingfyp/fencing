import os

import cv2
import numpy as np

from src.model import Quadrilateral, Ui, UiCodes
from src.model.PysideUi2 import PysideUi2
from src.model.tracker import OrbTracker
from src.util.io import setup_output_video_io
from src.util.utils import generate_select_quadrilateral_instructions


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


class CropRegionPipeline:
    def __init__(
        self,
        cap,
        output_path: str | None,
        ui: PysideUi2,
        region: str = "region",
    ):

        self.cap = cap
        self.output_path = output_path
        self.ui = ui
        self.region = region

        # Persistent state
        self.first_frame = None
        self.positions = None
        self.dimensions = None
        self.dst_corners = None
        self.tracker = None
        self.writer = None

        self.plane_id = "tracked_plane"

        self.ui.initialise(cap.get(cv2.CAP_PROP_FPS))
        self.cancelled = False

    # ---- ENTRY POINT ----

    def start(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cancel()
            return

        self.first_frame = frame

        # Async UI point selection
        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions(self.region),
            callback=self._on_initial_corners_selected,
        )

    # ---- CALLBACK 1: corners selected ----

    def _on_initial_corners_selected(self, positions):
        if len(positions) != 4:
            self.cancel()
            raise ValueError("Need to select 4 points for the planar region.")

        self.positions = Quadrilateral(positions)

        # Geometry
        self.dimensions = get_planar_dimensions(self.positions)
        self.dst_corners = make_destination_corners(self.dimensions)

        # Tracker
        self.tracker = OrbTracker()
        self.tracker.add_target(self.plane_id, self.first_frame, self.positions)

        # Writer
        if self.output_path:
            self.writer = setup_output_video_io(
                self.output_path,
                self.cap.get(cv2.CAP_PROP_FPS),
                self.dimensions,
            )

        # Align number of frames with original video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.ui.schedule(self.advance)

    # ---- CALLBACK 2: Loop ----

    def advance(self):
        if self.cancelled:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        rectified, pts = self._process_frame(frame)

        # UI rendering
        self.ui.set_fresh_frame(frame)
        self.ui.plot_points(pts, (0, 255, 0))
        self.ui.show_frame()
        self.ui.show_additional("cropped_view", rectified)

        if self.writer:
            self.writer.write(rectified)

        # Schedule next frame
        self._schedule(self.advance, 0)

    # ---- PROCESSING ----

    def _process_frame(self, frame):
        tracked = self.tracker.update_all(frame)
        quad = tracked.get(self.plane_id)

        if quad is None:
            quad = self.tracker.get_previous_quad(self.plane_id)

        rectified = self._get_rectified(frame, quad)
        pts = self.tracker.get_target_pts(self.plane_id)
        return rectified, pts

    def _get_rectified(self, frame, quad: Quadrilateral) -> np.ndarray:
        width, height = self.dimensions
        transform = cv2.getPerspectiveTransform(quad.numpy(), self.dst_corners)
        return cv2.warpPerspective(frame, transform, (width, height))

    # ---- UTILITIES ----
    def _schedule(self, fn, delay_ms):
        if not self.cancelled:
            self.ui.schedule(fn, delay_ms)

    # ---- CLEANUP ----

    def cancel(self):
        self.cancelled = True
        self.cleanup()

        # delete video because it's incomplete
        if self.output_path and os.path.exists(self.output_path):
            os.remove(self.output_path)

    def cleanup(self):
        self.ui.close_additional_windows()
        if self.writer:
            self.writer.release()
        self.cap.release()

    def stop(self):
        self.cleanup()
        self.on_finished()

    def set_on_finished(self, fn=None):
        self._on_finished = fn

    def on_finished(self):
        if self._on_finished:
            self._on_finished()
