from dataclasses import dataclass
from typing import Callable, List

import cv2
import numpy as np

from src.model import Quadrilateral
from src.model.tracker import OrbTracker
from src.pyside.PysideUi import PysideUi

from .defined_region import DefinedRegion
from .output.region_output import RegionOutput


@dataclass
class RegionState:
    label: str
    outputs: List[RegionOutput]

    def process(self, frame: np.ndarray, quad_np: np.ndarray, frame_id: int):
        for output in self.outputs:
            output.process(frame, quad_np, frame_id)

    def close(self):
        for output in self.outputs:
            output.close()

    def delete(self):
        for output in self.outputs:
            output.delete()


class MultiRegionProcessingPipeline:
    """
    UI-optional pipeline.
    Consumes DefinedRegion objects, tracks ROIs, and forwards frames/quads to outputs.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        defined_regions: list[DefinedRegion],
        ui: PysideUi | None = None,
        on_finished: Callable | None = None,
    ):
        self.cap = cap
        self.defined_regions = defined_regions
        self.ui = ui
        self.on_finished = on_finished

        self.tracker = OrbTracker()
        self.frame_id = 0
        self.cancelled = False

        first_frame = self.cap.read()[1]
        if first_frame is None:
            raise ValueError("Failed to read first frame from video.")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

        # Build typed region states
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.region_states: list[RegionState] = []
        for region in defined_regions:
            # Register with tracker
            self.tracker.add_target(
                region.label, first_frame, Quadrilateral(region.quad_np)
            )
            # Build outputs
            outputs = region.output_factory(region.quad_np, fps)
            self.region_states.append(RegionState(label=region.label, outputs=outputs))

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def start(self):
        if self.ui:
            self._schedule(self._advance)
        else:
            self._run_headless()

    def _run_headless(self):
        while not self.cancelled:
            ret, frame = self.cap.read()
            if not ret:
                break
            self._process_frame(frame)
        self._finish()

    def _advance(self):
        if self.cancelled:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._finish()
            return

        self._process_frame(frame)

        if self.ui:
            points = []
            for state in self.region_states:
                pts = getattr(self.tracker, "get_target_pts", lambda l: None)(
                    state.label
                )
                if pts is not None:
                    points.extend(pts.tolist())
            self.ui.set_fresh_frame(frame)
            self.ui.plot_points(points, (0, 255, 0))

            self._schedule(self._advance)

    # -------------------------
    # Frame processing
    # -------------------------
    def _process_frame(self, frame: np.ndarray):
        updated_quads = self.tracker.update_all(frame)

        for state in self.region_states:
            quad = updated_quads.get(state.label) or self.tracker.get_previous_quad(
                state.label
            )
            quad_np = quad.numpy()
            state.process(frame, quad_np, self.frame_id)

        self.frame_id += 1

    # -------------------------
    # Utilities
    # -------------------------
    def _schedule(self, fn: Callable):
        if self.ui and not self.cancelled:
            self.ui.schedule(fn)

    # -------------------------
    # Cleanup
    # -------------------------
    def cleanup(self):
        self.cancelled = True
        for state in self.region_states:
            state.close()

        self.cap.release()

    def cancel(self):
        self.cleanup()
        # delete all outputs since they're not complete
        # for state in self.region_states:
        #     for output in state.outputs:
        #         output.delete()

    def _finish(self):
        self.cleanup()

        if self.on_finished:
            self.on_finished()
