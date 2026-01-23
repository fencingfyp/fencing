import os
from typing import override

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication, QWidget

from scripts.momentum_graph.crop_scoreboard import (
    get_planar_dimensions,
    make_destination_corners,
)
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import PysideUi, Quadrilateral, UiCodes
from src.model.tracker import OrbTracker
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_output_video_io
from src.util.utils import generate_select_quadrilateral_instructions

from .base_task_widget import BaseTaskWidget


class CropScoreboardWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        self.ui.videoLabel.setFixedSize(
            *self.get_new_video_label_size(
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        )
        self.interactive_ui.show_single_frame(self.cap)
        self.interactive_ui.write("Press 'Run' to start cropping the scoreboard.")

    @override
    @Slot()
    def on_runButton_clicked(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.CROP_SCOREBOARD)

        output_path = os.path.join(self.working_dir, CROPPED_SCOREBOARD_VIDEO_NAME)

        # Create controller
        self.controller = CropRegionPysideController(
            cap=self.cap,
            output_path=output_path,
            ui=self.interactive_ui,
            parent=self,
            region="scoreboard",
        )

        # When finished → emit completion
        self.controller.finished.connect(self.on_finished)

        # Start async pipeline
        self.controller.start()

    def on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.CROP_SCOREBOARD)
        self.interactive_ui.write("Cropping scoreboard completed.")


class CropRegionPysideController(QObject):
    finished = Signal()

    def __init__(
        self,
        cap,
        output_path: str | None,
        ui: PysideUi,
        parent=None,
        region: str = "region",
    ):
        super().__init__(parent)

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

        self.timer = QTimer(self)
        QApplication.instance().aboutToQuit.connect(self.cancel)

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
            callback=self.on_initial_corners_selected,
        )

    # ---- CALLBACK 1: corners selected ----

    def on_initial_corners_selected(self, positions):
        if len(positions) != 4:
            self.cancel()
            return

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

        # reset cap
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start Qt-driven frame loop
        self.timer.timeout.connect(self.step)
        self.timer.start(0)  # as fast as event loop allows

    # ---- FRAME STEP (Qt event loop driven) ----

    def step(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        rectified, pts = self.process_frame(frame)

        # UI rendering
        self.ui.set_fresh_frame(frame)
        self.ui.plot_points(pts, (0, 255, 0))
        self.ui.show_frame()

        self.ui.show_additional("cropped_view", rectified)

        if self.writer:
            self.writer.write(rectified)

        # Input handling
        action = self.ui.take_user_input()
        if action == UiCodes.QUIT:
            self.cancel()

    # ---- PROCESSING ----

    def process_frame(self, frame):
        tracked = self.tracker.update_all(frame)
        quad = tracked.get(self.plane_id)

        if quad is None:
            quad = self.tracker.get_previous_quad(self.plane_id)

        rectified = self.get_rectified(frame, quad)
        pts = self.tracker.get_target_pts(self.plane_id)
        return rectified, pts

    def get_rectified(self, frame, quad: Quadrilateral) -> np.ndarray:
        width, height = self.dimensions
        transform = cv2.getPerspectiveTransform(quad.numpy(), self.dst_corners)
        return cv2.warpPerspective(frame, transform, (width, height))

    # ---- CLEANUP ----

    def cancel(self):
        self.timer.stop()

        if self.writer:
            self.writer.release()

    def stop(self):
        self.cancel()
        self.finished.emit()


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = CropScoreboardWidget()
    widget.set_working_directory("matches_data/sabre_1")
    widget.show()
    sys.exit(app.exec())
