import csv
import os
from typing import Optional, override

import cv2
from PySide6.QtCore import QObject, QTimer, Signal

from scripts.momentum_graph.perform_ocr import (
    DO_OCR_EVERY_N_FRAMES,
    extract_score_frame_from_frame,
    process_image,
    regularise_rectangle,
)
from src.gui.momentum_graph.base_task_widget import BaseTaskWidget
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import Quadrilateral
from src.model.drawable.quadrilateral_drawable import QuadrilateralDrawable
from src.model.EasyOcrReader import EasyOcrReader
from src.model.FileManager import FileRole
from src.pyside.MatchContext import MatchContext
from src.pyside.PysideUi import PysideUi
from src.util.gpu import get_device
from src.util.io import setup_input_video_io, setup_output_file
from src.util.utils import generate_select_quadrilateral_instructions


class PerformOcrWidget(BaseTaskWidget):
    def __init__(self, match_context, parent=None):
        super().__init__(match_context, parent)

        self.uiTextLabel.setText("Press 'Run' to start OCR processing.")

    @override
    def setup(self):
        video_path = self.match_context.file_manager.get_path(
            FileRole.CROPPED_SCOREBOARD
        )
        self.cap = cv2.VideoCapture(video_path)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        self.ui.set_fresh_frame(frame)
        self.ui.write("Press 'Run' to start OCR processing.")
        self.run_task()

    @override
    def on_runButton_clicked(self):
        self.run_task()

    @override
    def run_task(self):
        if self.is_running or not self.match_context.file_manager:
            return
        self.run_started.emit(MomentumGraphTasksToIds.PERFORM_OCR)
        self.is_running = True

        if self.controller:
            self.controller.cancel()
            self.controller.deleteLater()
        self.controller = OcrController(
            ui=self.ui,
            file_paths={
                "video": self.match_context.file_manager.get_original_video(),
                "cropped_video": self.match_context.file_manager.get_path(
                    FileRole.CROPPED_SCOREBOARD
                ),
                "output_csv": self.match_context.file_manager.get_path(
                    FileRole.RAW_SCORES
                ),
            },
            threshold_boundary=120,
            use_seven_segment=False,
        )
        self.controller.finished.connect(self._on_finished)
        self.controller.start()

    def _on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.PERFORM_OCR)
        self.ui.write("OCR processing completed.")
        self.is_running = False


class OcrController(QObject):
    finished = Signal()

    def __init__(
        self,
        ui: PysideUi,
        file_paths: str,
        threshold_boundary: int = 120,
        use_seven_segment: bool = False,
    ):
        super().__init__()
        self.ui = ui
        self.file_paths = file_paths
        self.threshold_boundary = threshold_boundary
        self.seven_segment = use_seven_segment

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_id: int = 0
        self.frame_count: int = 0
        self.fps: float = 0

        self.left_score_positions = None
        self.right_score_positions = None
        self.ocr_reader: Optional[EasyOcrReader] = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)

        self.csv_writer = None
        self.csv_file = None

    def start(self):
        """Initialize video and request score positions."""
        input_video_path = self.file_paths["cropped_video"]
        original_video_path = self.file_paths["video"]
        self._validate_input_video(original_video_path, input_video_path)

        self.cap, self.fps, _, _, self.frame_count = setup_input_video_io(
            input_video_path
        )

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video.")

        # Ask for left score positions
        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions("left fencer score display"),
            lambda pts: self._on_left_score_positions(frame, pts),
        )

    def _on_left_score_positions(self, frame, pts):
        self.left_score_positions = regularise_rectangle(pts)

        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions("right fencer score display"),
            lambda pts2: self._on_right_score_positions(frame, pts2),
        )

    def _on_right_score_positions(self, frame, pts):
        self.right_score_positions = regularise_rectangle(pts)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_id = 0

        self.ocr_reader = EasyOcrReader(get_device(), seven_segment=self.seven_segment)

        self.csv_file = open(self.file_paths["output_csv"], "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "frame_id",
                "left_score",
                "right_score",
                "left_confidence",
                "right_confidence",
            ]
        )

        self.timer.start(0)

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.ui.set_fresh_frame(frame)

        l_frame = extract_score_frame_from_frame(frame, self.left_score_positions)
        r_frame = extract_score_frame_from_frame(frame, self.right_score_positions)

        l_score = r_score = l_conf = r_conf = None
        if self.current_frame_id % DO_OCR_EVERY_N_FRAMES == 0:
            l_score, l_conf = self.ocr_reader.read(
                process_image(l_frame, self.threshold_boundary, self.seven_segment)
            )
            r_score, r_conf = self.ocr_reader.read(
                process_image(r_frame, self.threshold_boundary, self.seven_segment)
            )
            self.csv_writer.writerow(
                [self.current_frame_id, l_score, r_score, l_conf, r_conf]
            )
            self.ui.write(f"Frame {self.current_frame_id}: L={l_score} R={r_score}")

        self.ui.draw_objects(
            [
                QuadrilateralDrawable(Quadrilateral(self.left_score_positions)),
                QuadrilateralDrawable(Quadrilateral(self.right_score_positions)),
            ]
        )

        self.current_frame_id += 1

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def cancel(self):
        self.cleanup()

    def cleanup(self):
        if self.timer.isActive():
            self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def stop(self):
        self.cleanup()
        self.finished.emit()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _validate_input_video(self, original_path: str, cropped_path: str):
        if not os.path.exists(cropped_path):
            raise IOError(f"Cropped video not found at {cropped_path}")

        cap = cv2.VideoCapture(cropped_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open cropped video {cropped_path}")
        cropped_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open original video {original_path}")
        original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if cropped_frames != original_frames:
            raise ValueError(
                "Cropped video frame count does not match original video frame count."
            )


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = PerformOcrWidget(match_context)
        match_context.set_file("matches_data/sabre_2.mp4")
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
