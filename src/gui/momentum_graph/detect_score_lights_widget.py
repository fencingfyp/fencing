import csv
import os
import sys
from typing import Optional, override

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.gui.video_player_widget import VideoPlayerWidget
from src.model import Quadrilateral
from src.model.AutoPatchLightDetector import SinglePatchAutoDetector
from src.model.drawable import QuadrilateralDrawable
from src.model.PatchLightDetector import Colour
from src.model.PysideUi import PysideUi
from src.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
)
from src.util.utils import generate_select_quadrilateral_instructions

from .base_task_widget import BaseTaskWidget


class DetectScoreLightsWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Temporarily hide run button until we implement run options
        self.ui.runButton.hide()

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
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read video from {video_path}")
        self.interactive_ui.set_fresh_frame(frame)
        self.interactive_ui.show_frame()
        self.cap.release()
        self.cap = None
        self.interactive_ui.write("Press 'Run' to start detecting the score lights.")

        self.run_task()

    def _on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS)
        self.interactive_ui.write("Score lights detection completed.")
        self.is_running = False

    def on_runButton_clicked(self):
        if not self.working_dir:
            return
        self.run_task()

    def run_task(self):
        self.is_running = True
        self.run_started.emit(MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS)

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
        self.debug_mode = True

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.fps: float = 0
        self.current_frame_id: int = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        # self.timer.setInterval(30)  # as fast as possible
        self.waiting_for_user = False

        self.csv_file = None
        self.csv_writer = None

        self.left_score_positions = None
        self.right_score_positions = None
        self.left_detector: Optional[SinglePatchAutoDetector] = None
        self.right_detector: Optional[SinglePatchAutoDetector] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def start(self):
        # 1. ROI selection
        self.cap = cv2.VideoCapture(
            os.path.join(self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
        )
        self.csv_file = open(
            os.path.join(self.working_dir, DETECT_LIGHTS_OUTPUT_CSV_NAME),
            "w",
            newline="",
        )
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(get_output_header_row(is_debug=self.debug_mode))
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.roi_selector = ScoreLightROISelector(ui=self.ui, frame=frame)
        self.roi_selector.finished.connect(self._on_roi_selected)
        self.roi_selector.start()

    def _on_roi_selected(self, left_quad, right_quad):
        self.ui.video_label.hide()
        self.left_score_positions = left_quad
        self.right_score_positions = right_quad

        # 2. Time selection
        video_path = os.path.join(self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
        self.time_selector = ScoreLightQuadSelector(
            video_path,
            container=self.ui.parent,
        )
        self.time_selector.finished.connect(self._on_timestamps_selected)

    def _on_timestamps_selected(self, timestamps):
        self.timestamps = timestamps
        self.time_selector.deactivate()
        self.time_selector = None

        # Now run autopatch detection on all timestamps
        self._run_autopatch_detection()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _build_frame_map(self):
        self.cap = cv2.VideoCapture(
            os.path.join(self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
        )
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video for autopatch detection.")
        out = {}
        for key in self.timestamps:
            frame_id = self.timestamps[key]
            if frame_id < 0 or frame_id >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                raise ValueError(f"Invalid frame id {frame_id} for label {key}.")
            frame = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read frame {frame_id} for label {key}.")
            out[key] = frame
        return out

    def _run_autopatch_detection(self):
        # obtain patches of interest
        frame_map = self._build_frame_map()
        self.left_detector = SinglePatchAutoDetector(
            frame_map["left_pos"],
            self.left_score_positions,
            frame_map["left_neg"],
            self.left_score_positions,
        )
        self.right_detector = SinglePatchAutoDetector(
            frame_map["right_pos"],
            self.right_score_positions,
            frame_map["right_neg"],
            self.right_score_positions,
        )
        self.ui.video_label.show()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.ui.schedule(self._process_frame)

    def _process_frame(self):
        if self.waiting_for_user:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.ui.set_fresh_frame(frame)

        # Detect lights
        is_left_red = self.left_detector.classify(frame, self.left_score_positions)
        is_right_green = self.right_detector.classify(frame, self.right_score_positions)

        # Draw quadrilaterals
        self.ui.draw_objects(
            [
                QuadrilateralDrawable(self.left_score_positions, color=(255, 0, 0)),
                QuadrilateralDrawable(self.right_score_positions, color=(0, 255, 0)),
            ]
        )

        # Write CSV
        if not self.demo_mode:
            row = [self.current_frame_id, int(is_left_red), int(is_right_green)]
            if self.debug_mode:
                left_debug = self.left_detector.get_debug_info()
                right_debug = self.right_detector.get_debug_info()
                row.extend([left_debug, right_debug])
            self.csv_writer.writerow(row)

        # Update UI
        self.ui.write(f"left light on: {is_left_red}, right light on: {is_right_green}")
        self.ui.show_frame()

        self.current_frame_id += 1
        self.ui.schedule(self._process_frame)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cancel(self):
        self.cleanup()
        if not self.demo_mode:
            output_csv_path = os.path.join(
                self.working_dir, DETECT_LIGHTS_OUTPUT_CSV_NAME
            )
            # if os.path.exists(output_csv_path):
            #     os.remove(output_csv_path)

    def cleanup(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        if hasattr(self, "time_selector") and self.time_selector:
            self.time_selector.deactivate()
            self.time_selector = None

    def stop(self):
        self.cleanup()
        self.finished.emit()


class ScoreLightROISelector(QObject):
    finished = Signal(Quadrilateral, Quadrilateral)

    def __init__(self, ui: PysideUi, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.frame = frame

    def start(self):
        self.ui.get_n_points_async(
            self.frame,
            generate_select_quadrilateral_instructions("left fencer score light"),
            self._on_left_done,
        )

    def _on_left_done(self, left_pts):
        self.left_quad = Quadrilateral(left_pts)
        self.ui.get_n_points_async(
            self.frame,
            generate_select_quadrilateral_instructions("right fencer score light"),
            self._on_right_done,
        )

    def _on_right_done(self, right_pts):
        self.right_quad = Quadrilateral(right_pts)
        self.finished.emit(self.left_quad, self.right_quad)


class ScoreLightQuadSelector(QObject):
    """
    Allows user to select four timestamps in order:
    left_neg, left_pos, right_neg, right_pos
    """

    finished = Signal(dict)  # Map of label -> frame number

    def __init__(self, video_path: str, container: QWidget | None = None):
        super().__init__(container)
        self.container = container or QWidget()
        self.layout = QVBoxLayout(self.container)

        # Video player
        self.player = VideoPlayerWidget(self.container)
        self.player.set_video_source(video_path)
        self.layout.addWidget(self.player)

        # Info label
        self.info = QLabel(self.container)
        self.layout.addWidget(self.info)

        # Finish button
        self.finish_button = QPushButton("Finish selection", self.container)
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self.finish)
        self.layout.addWidget(self.finish_button)

        self.container.setLayout(self.layout)
        self.activate()

    def activate(self):
        self.player.activate()
        self.labels = ["left_neg", "left_pos", "right_neg", "right_pos"]
        self.current_index = 0
        self.timestamps: dict = {}

        # Use one key to advance through the labels
        self.player.register_shortcut(QKeySequence(Qt.Key.Key_E), self.mark_next_frame)
        self.update_info()

    def update_info(self):
        if self.current_index < len(self.labels):
            self.info.setText(
                f"Press 'E' to mark {self.labels[self.current_index]} (frame {self.player.video_frame.get_current_frame_number()})"
            )
        else:
            self.info.setText("All frames marked. Click Finish.")

    def deactivate(self):
        self.player.deactivate()
        self.player.hide()
        self.finish_button.hide()
        self.info.hide()

    def mark_next_frame(self):
        if self.current_index >= len(self.labels):
            return

        frame = self.player.video_frame.get_current_frame_number()
        label = self.labels[self.current_index]
        self.timestamps[label] = frame
        self.current_index += 1
        self.finish_button.setEnabled(self.current_index == len(self.labels))
        self.update_info()

    def finish(self):
        self.deactivate()
        self.finished.emit(self.timestamps)


if __name__ == "__main__":
    app = QApplication([])

    widget = DetectScoreLightsWidget()
    widget.set_working_directory("matches_data/foil_4")
    widget.show()
    sys.exit(app.exec())
