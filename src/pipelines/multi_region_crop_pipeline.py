import os

import cv2
import numpy as np

from src.model import Quadrilateral, Ui, UiCodes
from src.model.PysideUi import PysideUi
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


class MultiRegionCropPipeline:
    def __init__(self, cap, ui: PysideUi, output_paths: dict[str, str]):
        self.cap = cap
        self.ui = ui
        self.regions = list(output_paths.keys())
        self.output_paths = output_paths

        # State per region
        self.region_index = 0
        self.region_data = {}

        # Tracker and video-level state
        self.tracker = OrbTracker()
        self.first_frame = None
        self.cancelled = False

        self.ui.initialise(cap.get(cv2.CAP_PROP_FPS))
        self._on_finished = None

    # ---- ENTRY POINT ----
    def start(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cancel()
            return
        self.first_frame = frame
        self._ask_next_region()

    # ---- REGION SELECTION ----
    def _ask_next_region(self):
        if self.region_index >= len(self.regions):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.advance()
            return

        label = self.regions[self.region_index]
        self.ui.get_n_points_async(
            self.first_frame,
            generate_select_quadrilateral_instructions(label),
            callback=self._on_corners_selected,
        )

    def _on_corners_selected(self, positions):
        if len(positions) != 4:
            self.cancel()
            raise ValueError("Need to select 4 points for each planar region.")

        label = self.regions[self.region_index]
        quad = Quadrilateral(positions)
        dims = get_planar_dimensions(quad)
        dst_corners = make_destination_corners(dims)

        # Add target to the single tracker
        self.tracker.add_target(label, self.first_frame, quad)

        writer = None
        output_path = self.output_paths.get(label)
        if output_path:
            writer = setup_output_video_io(
                output_path, self.cap.get(cv2.CAP_PROP_FPS), dims
            )

        self.region_data[label] = {
            "dimensions": dims,
            "dst_corners": dst_corners,
            "writer": writer,
        }

        self.region_index += 1
        self._ask_next_region()

    # ---- MAIN LOOP ----
    def advance(self):
        if self.cancelled:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        updated_quads = self.tracker.update_all(frame)

        for label, data in self.region_data.items():
            quad = updated_quads.get(label) or self.tracker.get_previous_quad(label)
            rectified = cv2.warpPerspective(
                frame,
                cv2.getPerspectiveTransform(quad.numpy(), data["dst_corners"]),
                data["dimensions"],
            )
            pts = self.tracker.get_target_pts(label)

            self.ui.show_additional(label, rectified)
            self.ui.plot_points(pts, (0, 255, 0))
            if data["writer"]:
                data["writer"].write(rectified)

        self.ui.set_fresh_frame(frame)
        self.ui.show_frame()
        self._schedule(self.advance, 0)

    # ---- UTILITIES ----
    def _schedule(self, fn, delay_ms):
        if not self.cancelled:
            self.ui.schedule(fn, delay_ms)

    # ---- CLEANUP ----
    def cancel(self):
        self.cancelled = True
        self.cleanup()
        # for path in self.output_paths.values():
        #     if path and os.path.exists(path):
        #         os.remove(path)

    def cleanup(self):
        self.ui.close_additional_windows()
        for data in self.region_data.values():
            if data["writer"]:
                data["writer"].release()
        self.cap.release()

    def stop(self):
        self.cancelled = True
        self.cleanup()
        if self._on_finished:
            self._on_finished()

    def set_on_finished(self, fn):
        self._on_finished = fn
