import csv
import os
from typing import Optional, override

import cv2
from PySide6.QtCore import QObject, QTimer, Signal

from scripts.momentum_graph.perform_ocr import (
    DO_OCR_EVERY_N_FRAMES,
    OUTPUT_OCR_L_WINDOW,
    OUTPUT_OCR_R_WINDOW,
    OUTPUT_VIDEO_NAME,
    extract_score_frame_from_frame,
    process_image,
    regularise_rectangle,
)
from src.gui.momentum_graph.base_task_widget import BaseTaskWidget
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import PysideUi
from src.model.EasyOcrReader import EasyOcrReader
from src.util.file_names import (
    CROPPED_SCOREBOARD_VIDEO_NAME,
    OCR_OUTPUT_CSV_NAME,
    ORIGINAL_VIDEO_NAME,
)
from src.util.gpu import get_device
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io
from src.util.utils import (
    convert_from_box_to_rect,
    generate_select_quadrilateral_instructions,
)


class PerformOcrWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        video_path = os.path.join(self.working_dir, CROPPED_SCOREBOARD_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        self.ui.videoLabel.setFixedSize(
            *self.get_new_video_label_size(
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        )
        self.interactive_ui.show_single_frame(self.cap)
        self.interactive_ui.write("Press 'Run' to start OCR processing.")

    def on_runButton_clicked(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.PERFORM_OCR)

        # Create controller
        self.controller = OcrController(
            ui=self.interactive_ui,
            working_dir=self.working_dir,
            threshold_boundary=120,
            use_seven_segment=False,
            output_video=False,
        )

        # When finished → emit completion
        self.controller.finished.connect(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.PERFORM_OCR)
        self.interactive_ui.write("OCR processing completed.")


class OcrController(QObject):
    """Widget-friendly OCR controller with non-blocking callbacks"""

    finished = Signal()

    def __init__(
        self,
        ui: PysideUi,
        working_dir: str,
        threshold_boundary: int = 120,
        use_seven_segment: bool = False,
        output_video: bool = True,
    ):
        super().__init__()
        self.ui = ui
        self.working_dir = working_dir
        self.threshold_boundary = threshold_boundary
        self.seven_segment = use_seven_segment
        self.output_video = output_video

        # Video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.fps: float = 0
        self.current_frame_id: int = 0

        # OCR state
        self.left_score_positions = None
        self.right_score_positions = None
        self.ocr_reader: Optional[EasyOcrReader] = None

        # Timer-driven loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        self.waiting_for_user = False

        # Video writers
        self.video_writer = None
        self.ocr_window_l_writer = None
        self.ocr_window_r_writer = None

        # CSV output
        self.csv_writer = None
        self.csv_file = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def start(self):
        """Initialize video, OCR, UI, and score positions asynchronously."""
        # Open cropped video
        input_video_path = os.path.join(self.working_dir, CROPPED_SCOREBOARD_VIDEO_NAME)
        original_video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        self._validate_input_video(original_video_path, input_video_path)

        self.cap, self.fps, _, _, self.frame_count = setup_input_video_io(
            input_video_path
        )

        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video.")

        # Ask for score positions asynchronously
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        self.current_frame_id = 0

        # Initialize OCR
        device = get_device()
        self.ocr_reader = EasyOcrReader(device, seven_segment=self.seven_segment)

        # Initialize video writers if needed
        if self.output_video:
            output_video_path = os.path.join(self.working_dir, OUTPUT_VIDEO_NAME)
            self.video_writer = setup_output_video_io(
                output_video_path, self.fps, frame.shape[1::-1]
            )

            # Separate windows
            _, _, w1, h1 = convert_from_box_to_rect(self.left_score_positions)
            _, _, w2, h2 = convert_from_box_to_rect(self.right_score_positions)
            ocr_window_l_path = os.path.join(self.working_dir, OUTPUT_OCR_L_WINDOW)
            ocr_window_r_path = os.path.join(self.working_dir, OUTPUT_OCR_R_WINDOW)
            self.ocr_window_l_writer = setup_output_video_io(
                ocr_window_l_path, self.fps, (w1, h1)
            )
            self.ocr_window_r_writer = setup_output_video_io(
                ocr_window_r_path, self.fps, (w2, h2)
            )

        # Open CSV
        output_csv_path = setup_output_file(self.working_dir, OCR_OUTPUT_CSV_NAME)
        self.csv_file = open(output_csv_path, "w", newline="")
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

        # Start timer loop
        self.timer.start(0)

    # ------------------------------------------------------------------
    # Frame processing loop
    # ------------------------------------------------------------------

    def _process_frame(self):
        if self.waiting_for_user:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.ui.set_fresh_frame(frame)

        l_frame = extract_score_frame_from_frame(frame, self.left_score_positions)
        r_frame = extract_score_frame_from_frame(frame, self.right_score_positions)

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
        else:
            l_score = r_score = l_conf = r_conf = None

        # Write video frames
        if self.output_video:
            self.ocr_window_l_writer.write(
                process_image(l_frame, self.threshold_boundary, self.seven_segment)
            )
            self.ocr_window_r_writer.write(
                process_image(r_frame, self.threshold_boundary, self.seven_segment)
            )
            self.video_writer.write(frame)

        self.current_frame_id += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cancel(self):
        if self.timer.isActive():
            self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        for writer in [
            self.video_writer,
            self.ocr_window_l_writer,
            self.ocr_window_r_writer,
        ]:
            if writer:
                writer.release()

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def stop(self):
        """Stop the timer and release resources."""
        self.cancel()
        self.finished.emit()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
                f"Cropped video frames ({cropped_frames}) do not match original ({original_frames})"
            )


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = PerformOcrWidget()
    widget.set_working_directory("matches_data/sabre_1")
    widget.show()
    sys.exit(app.exec())
