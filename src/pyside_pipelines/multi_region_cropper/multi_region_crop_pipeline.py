# src/pyside_pipelines/multi_region_cropper/multi_region_crop_pipeline.py

from dataclasses import dataclass
from typing import Callable, List

import cv2
import numpy as np
from PySide6.QtCore import QTimer

from src.model.tracker.DefinedRegion import DefinedRegion
from src.model.tracker.TargetTracker import TargetTracker
from src.model.tracker.tracker_factory import build_tracker
from src.pyside.PysideUi import PysideUi

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
    UI-optional, tracker-agnostic processing pipeline.

    Accepts DefinedRegion objects and delegates tracker construction to
    build_tracker(), which selects the appropriate concrete tracker(s)
    based on each region's TrackingStrategy. The pipeline itself has no
    knowledge of ORB vs AKAZE — swapping or mixing strategies requires
    only changes to LabelConfig in the widget.

    An optional tracker parameter allows injection for testing.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        defined_regions: list[DefinedRegion],
        ui: PysideUi | None = None,
        on_finished: Callable | None = None,
        tracker: TargetTracker | None = None,
    ):
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.defined_regions = defined_regions
        self.ui = ui
        self.on_finished = on_finished
        self.frame_id = 0
        self.cancelled = False

        ret, first_frame = self.cap.read()
        if not ret or first_frame is None:
            raise ValueError("Failed to read first frame from video.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.tracker: TargetTracker = tracker or build_tracker(
            defined_regions, first_frame
        )

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.region_states: list[RegionState] = [
            RegionState(
                label=region.label,
                outputs=region.output_factory(region.quad_np, fps),
            )
            for region in defined_regions
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self):
        self._schedule(self._advance)

    def _advance(self):
        if self.cancelled:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._finish()
            return

        self._process_frame(frame)

        if self.ui:
            percent = (self.frame_id / self.total_frames) * 100
            self.ui.write(f"Processing ({percent:.1f}%)", silent=True)

        self._schedule(self._advance)

    def _process_frame(self, frame: np.ndarray):
        updated_quads = self.tracker.update_all(frame)

        for state in self.region_states:
            quad = updated_quads.get(state.label) or self.tracker.get_previous_quad(
                state.label
            )
            state.process(frame, quad.numpy(), self.frame_id)

        self.frame_id += 1

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule(self, fn: Callable):
        if not self.cancelled:
            if self.ui:
                self.ui.schedule(fn)
            else:
                QTimer.singleShot(0, fn)

    # ------------------------------------------------------------------
    # Lifecycle — three distinct concerns:
    #   cleanup()  — always runs, releases resources
    #   cancel()   — aborted run, no completion signal
    #   _finish()  — natural end, emits completion
    # ------------------------------------------------------------------

    def _cleanup(self):
        for state in self.region_states:
            state.close()
        self.cap.release()

    def cancel(self):
        self.cancelled = True
        self._cleanup()

    def _finish(self):
        self._cleanup()
        if self.on_finished:
            self.on_finished()
