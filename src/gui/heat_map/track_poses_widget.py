import csv
import os
from typing import override

import cv2
from ultralytics import YOLO

from scripts.estimate_poses import extract_rows, get_header_row
from src.gui.momentum_graph.base_task_widget import BaseTaskWidget
from src.gui.util.task_graph import HeatMapTasksToIds
from src.model import Ui
from src.model.FileManager import FileRole
from src.pyside.MatchContext import MatchContext
from src.pyside.PysideUi import PysideUi
from src.util.gpu import get_device


class TrackPosesWidget(BaseTaskWidget):
    def __init__(self, match_context, parent=None):
        super().__init__(match_context, parent)

    @override
    def setup(self):
        self.ui.write("Press 'Run' to run pose tracking.")
        self.run_task()

    @override
    def on_runButton_clicked(self):
        self.run_task()

    def run_task(self):
        if not self.match_context.file_manager:
            return

        self.is_running = True

        self.run_started.emit(HeatMapTasksToIds.TRACK_POSES)

        input_video_path = self.match_context.file_manager.get_original_video()
        model_path = os.path.join("models", "yolo", "yolo11l-pose.pt")

        # Create controller
        self.controller = PoseToCsvController(
            ui=self.ui,
            input_path=input_video_path,
            output_path=self.match_context.file_manager.get_path(FileRole.RAW_POSE),
            model_path=model_path,
        )

        # When finished → emit completion
        self.controller.set_on_finished(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.is_running = False
        self.ui.write("Pose tracking completed.")
        self.run_completed.emit(HeatMapTasksToIds.TRACK_POSES)

    def cancel(self):
        if hasattr(self, "controller"):
            self.controller.cancel()


class PoseToCsvController:
    """
    Controller for processing a video with YOLO pose+tracking and writing to CSV.

    Drives processing via ui.run_loop(step_callback).
    """

    def set_on_finished(self, callback):
        self._on_finished = callback

    def on_finished(self):
        self.cancel()
        if self._on_finished:
            self._on_finished()

    def __init__(
        self,
        ui: PysideUi,
        input_path: str,
        output_path: str,
        model_path: str,
    ):
        self.ui: PysideUi = ui
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path

        self.cap: cv2.VideoCapture | None = None
        self.model: YOLO | None = None
        self.device = get_device()
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

        self.csv_file = open(self.output_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        # Write header
        self.writer.writerow(get_header_row())

        self.ui.write("Processing video...")
        self.ui.schedule(self._step)

    # ------------------------------------------------------------------
    # Loop step
    # ------------------------------------------------------------------

    def _step(self):
        """
        Returns:
            True  -> continue loop
            False -> stop loop
        """

        ret, frame = self.cap.read()
        if not ret:
            self.on_finished()
            return

        results = self.model.track(frame, persist=True, verbose=False)

        # Write CSV rows
        rows = extract_rows(results, self.frame_idx)
        if rows:
            self.writer.writerows(rows)

        annotated_frame = results[0].plot()
        self.ui.set_fresh_frame(annotated_frame)

        self.frame_idx += 1
        self.ui.schedule(self._step)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cancel(self):
        if self.cap is not None:
            self.cap.release()

        if self.csv_file is not None:
            self.csv_file.close()

        if hasattr(self, "model") and self.model is not None:
            del self.model


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = TrackPosesWidget(match_context)
        match_context.set_file("matches_data/foil_2.mp4")
        widget.show()
        sys.exit(app.exec())

    # Run the profiler and save stats to a file
    cProfile.run("main()", "profile.stats")

    # Load stats
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()  # remove extraneous path info
    stats.sort_stats("tottime")  # sort by total time

    # Print only top 10 functions
    stats.print_stats(10)
