import csv
import os
import sys
from typing import Optional, override

import cv2
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from scripts.momentum_graph.perform_ocr import validate_input_video
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import NORMAL_UI_FUNCTIONS, PysideUi, Quadrilateral, UiCodes
from src.model.PatchLightDetector import Colour, PatchLightDetector
from src.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
    ORIGINAL_VIDEO_NAME,
)
from src.util.io import setup_input_video_io, setup_output_file
from src.util.utils import generate_select_quadrilateral_instructions

from .base_task_widget import BaseTaskWidget


class DetectScoreLightsWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        video_path = os.path.join(self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        self.ui.videoLabel.setFixedSize(
            *self.get_new_video_label_size(
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        )
        self.interactive_ui.show_single_frame(self.cap)
        self.interactive_ui.write("Press 'Run' to start detecting the score lights.")

    def _on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS)
        self.interactive_ui.write("Score lights detection completed.")

    def on_runButton_clicked(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.PERFORM_OCR)

        # Create controller
        self.controller = ScoreLightsController(
            ui=self.interactive_ui,
            working_dir=self.working_dir,
        )

        # When finished → emit completion
        self.controller.finished.connect(self._on_finished)

        # Start async pipeline
        self.controller.start()


def get_output_header_row(is_debug: bool = False) -> list[str]:
    headers = ["frame_id", "left_light", "right_light"]
    if is_debug:
        headers.extend(["left_debug_info", "right_debug_info"])
    return headers


class ScoreLightsController(QObject):
    """Widget-friendly score light detection controller."""

    finished = Signal()

    def __init__(
        self,
        ui: PysideUi,
        working_dir: str,
        demo_mode: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.ui = ui
        self.working_dir = working_dir
        self.demo_mode = demo_mode
        self.debug_mode = debug_mode

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.fps: float = 0
        self.current_frame_id: int = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        self.waiting_for_user = False

        self.csv_file = None
        self.csv_writer = None

        self.left_score_positions = None
        self.right_score_positions = None
        self.left_detector = None
        self.right_detector = None

        self.FULL_DELAY = 0
        self.FAST_FORWARD = 1
        self.slow = False
        self.early_exit = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def start(self):
        """Initialize video, UI, detectors, and positions asynchronously."""
        input_video_path = os.path.join(
            self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME
        )
        original_video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        validate_input_video(original_video_path, input_video_path)

        self.cap, self.fps, _, _, self.frame_count = setup_input_video_io(
            input_video_path
        )
        self.FULL_DELAY = int(1000 / self.fps)
        self.FAST_FORWARD = min(self.FULL_DELAY // 16, 1)

        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video.")

        # Initialise detectors
        self.left_detector = PatchLightDetector("red")
        self.right_detector = PatchLightDetector("green")

        # Ask user to define quadrilaterals asynchronously
        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions("left fencer score light"),
            lambda pts: self._on_left_score_positions(frame, pts),
        )

    def _on_left_score_positions(self, frame, pts):
        self.left_score_positions = Quadrilateral(pts)
        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions("right fencer score light"),
            lambda pts2: self._on_right_score_positions(frame, pts2),
        )

    def _on_right_score_positions(self, frame, pts):
        self.right_score_positions = Quadrilateral(pts)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_id = 0

        # Setup CSV
        output_csv_path = setup_output_file(
            self.working_dir, DETECT_LIGHTS_OUTPUT_CSV_NAME
        )
        mode = "a" if self.demo_mode else "w"
        self.csv_file = open(output_csv_path, mode, newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not self.demo_mode:
            self.csv_writer.writerow(get_output_header_row(self.debug_mode))

        # Start timer loop
        self.timer.start(0)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(self):
        if self.waiting_for_user:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.ui.set_fresh_frame(frame)

        # Detect lights
        is_left_red = (
            self.left_detector.classify(frame, self.left_score_positions) == Colour.RED
        )
        is_right_green = (
            self.right_detector.classify(frame, self.right_score_positions)
            == Colour.GREEN
        )

        # Draw quadrilaterals
        self.ui.draw_quadrilateral(self.left_score_positions, color=(255, 0, 0))
        self.ui.draw_quadrilateral(self.right_score_positions, color=(0, 255, 0))

        # Write CSV
        if not self.demo_mode:
            row = [self.current_frame_id, int(is_left_red), int(is_right_green)]
            if self.debug_mode:
                row.extend(
                    [
                        self.left_detector.get_debug_info(),
                        self.right_detector.get_debug_info(),
                    ]
                )
            self.csv_writer.writerow(row)

        # Update UI
        self.ui.write(f"left light on: {is_left_red}, right light on: {is_right_green}")
        self.ui.show_frame()

        self.current_frame_id += 1

        # Handle interactive actions
        action = self.ui.get_user_input()
        if action == UiCodes.QUIT:
            self.cancel()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cancel(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def stop(self):
        self.cancel()
        self.finished.emit()


if __name__ == "__main__":
    app = QApplication([])

    widget = DetectScoreLightsWidget()
    widget.set_working_directory("matches_data/sabre_1")
    widget.show()
    sys.exit(app.exec())
