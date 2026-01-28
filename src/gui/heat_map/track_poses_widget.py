import csv
import os
from typing import override

import cv2
from ultralytics import YOLO

from scripts.estimate_poses import extract_rows, get_header_row
from src.gui.momentum_graph.base_task_widget import BaseTaskWidget
from src.gui.util.task_graph import HeatMapTasksToIds
from src.model import Ui
from src.util.file_names import ORIGINAL_VIDEO_NAME, RAW_POSE_DATA_CSV_NAME
from src.util.io import setup_output_video_io


class TrackPosesWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        self.interactive_ui.write("Press 'Run' to run pose tracking.")

    @override
    def on_runButton_clicked(self):
        if not self.working_dir:
            return

        self.run_started.emit(HeatMapTasksToIds.TRACK_POSES)

        input_video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        model_path = os.path.join("models", "yolo", "yolo11l-pose.pt")

        # Create controller
        self.controller = PoseToCsvController(
            ui=self.interactive_ui,
            input_path=input_video_path,
            output_folder=self.working_dir,
            model_path=model_path,
        )

        # When finished → emit completion
        self.controller.set_on_finished(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.interactive_ui.write("Pose tracking completed.")
        self.run_completed.emit(HeatMapTasksToIds.TRACK_POSES)


class PoseToCsvController:
    """
    Controller for processing a video with YOLO pose+tracking and writing to CSV.

    Drives processing via ui.run_loop(step_callback).
    """

    def set_on_finished(self, callback):
        self._on_finished = callback

    def on_finished(self):
        if self._on_finished:
            self._on_finished()

    def __init__(
        self,
        ui,
        input_path: str,
        output_folder: str,
        model_path: str,
    ):
        self.ui: Ui = ui
        self.input_path = input_path
        self.output_folder = output_folder
        self.model_path = model_path

        self.cap: cv2.VideoCapture | None = None
        self.model: YOLO | None = None
        self.writer = None
        self.csv_file = None

        self.frame_idx = 0

        self._on_finished = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Initialise resources and start processing loop."""

        self.ui.write("Loading YOLO model...")
        self.model = YOLO(self.model_path)

        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            self.ui.write(f"Error opening video file: {self.input_path}")
            return

        csv_path = os.path.join(self.output_folder, RAW_POSE_DATA_CSV_NAME)
        self.csv_file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        # Write header
        self.writer.writerow(get_header_row())

        self.ui.write("Processing video...")
        self.ui.run_loop(self._step, self.on_finished)

    # ------------------------------------------------------------------
    # Loop step
    # ------------------------------------------------------------------

    def _step(self) -> bool:
        """
        Returns:
            True  -> continue loop
            False -> stop loop
        """

        ret, frame = self.cap.read()
        if not ret:
            self.cancel()
            return False

        results = self.model.track(frame, persist=True, verbose=False)

        # Write CSV rows
        rows = extract_rows(results, self.frame_idx)
        if rows:
            self.writer.writerows(rows)

        annotated_frame = results[0].plot()
        self.ui.set_fresh_frame(annotated_frame)

        self.frame_idx += 1
        return True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cancel(self):
        if self.cap is not None:
            self.cap.release()

        if self.csv_file is not None:
            self.csv_file.close()


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = TrackPosesWidget()
    widget.set_working_directory("matches_data/sabre_2")
    widget.show()
    sys.exit(app.exec())
