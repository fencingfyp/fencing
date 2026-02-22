"""
perform_ocr_widget.py
---------------------
PySide6 widget for batched OCR processing of fencing scoreboard videos.

Processing is split into two phases driven by a single QTimer to avoid
threading and lock contention:

  Phase 1 — COLLECTING: iterate all frames, extract and preprocess ROIs,
             accumulate into a preallocated results array alongside frame IDs.
  Phase 2 — INFERRING: drain the accumulated crops in batches through
             EasyOCR (single GPU call per batch), write results to CSV
             in strict frame order.
"""

import csv
import os
from enum import Enum, auto
from typing import Optional, override

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from scripts.momentum_graph.perform_ocr import (
    DO_OCR_EVERY_N_FRAMES,
    extract_roi,
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

MAX_FRAMES = 200_000
TARGET_CROPS_PER_BATCH = 16


class Phase(Enum):
    COLLECTING = auto()
    INFERRING = auto()


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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
            use_seven_segment=True,
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
        file_paths: dict,
        use_seven_segment: bool = False,
    ):
        super().__init__()
        self.ui = ui
        self.file_paths = file_paths
        self.seven_segment = use_seven_segment

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.fps: float = 0
        self.current_frame_id: int = 0

        self.left_score_positions = None
        self.right_score_positions = None
        self.ocr_reader: Optional[EasyOcrReader] = None

        # --- Collection state ---
        # Preallocated storage: one row per OCR frame, columns:
        #   [frame_id, l_score, r_score, l_conf, r_conf]
        # Scores/confs filled in during inference phase.
        self._ocr_frame_ids: list[int] = []
        self._pending_rois: list[np.ndarray] = []  # interleaved: [l0, r0, l1, r1, ...]

        # --- Inference state ---
        self._batch_size: int = 0
        self._batch_index: int = 0  # index into _pending_rois
        self._results: Optional[np.ndarray] = (
            None  # shape (n_ocr_frames, 4): l_score, r_score, l_conf, r_conf
        )

        self._phase = Phase.COLLECTING
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.csv_file = None
        self.csv_writer = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self):
        """Validate videos, open capture, and request ROI selection from user."""
        input_video_path = self.file_paths["cropped_video"]
        original_video_path = self.file_paths["video"]
        self._validate_input_video(original_video_path, input_video_path)

        self.cap, self.fps, _, _, self.frame_count = setup_input_video_io(
            input_video_path
        )

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video.")

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
        self._batch_size = TARGET_CROPS_PER_BATCH

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

        self._phase = Phase.COLLECTING
        self.timer.start(0)

    # ------------------------------------------------------------------
    # Timer tick — dispatches to current phase
    # ------------------------------------------------------------------

    def _tick(self):
        if self._phase == Phase.COLLECTING:
            self._collect_frame()
        else:
            self._infer_batch()

    # ------------------------------------------------------------------
    # Phase 1: collecting
    # ------------------------------------------------------------------

    def _collect_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self._begin_inference()
            return

        # self.ui.set_fresh_frame(frame)
        # self.ui.draw_objects(
        #     [
        #         QuadrilateralDrawable(Quadrilateral(self.left_score_positions)),
        #         QuadrilateralDrawable(Quadrilateral(self.right_score_positions)),
        #     ]
        # )
        self.ui.write(f"Collecting frame {self.current_frame_id}/{self.frame_count}...")

        if self.current_frame_id % DO_OCR_EVERY_N_FRAMES == 0:
            l_roi = extract_roi(frame, self.left_score_positions)
            r_roi = extract_roi(frame, self.right_score_positions)
            # Preprocess on CPU now, store for batched GPU inference later
            self._pending_rois.append(self.ocr_reader.preprocessor(l_roi))
            self._pending_rois.append(self.ocr_reader.preprocessor(r_roi))
            self._ocr_frame_ids.append(self.current_frame_id)

        self.current_frame_id += 1

    def _begin_inference(self):
        """Switch to inference phase once all frames are collected."""
        n_ocr_frames = len(self._ocr_frame_ids)
        # results array: rows = ocr frames, cols = [l_score, r_score, l_conf, r_conf]
        # stored as object dtype to hold string scores and float confidences
        self._results = np.empty((n_ocr_frames, 4), dtype=object)
        self._batch_index = 0
        self._phase = Phase.INFERRING
        self.ui.write(
            f"Collection complete. Running inference on {len(self._pending_rois)} crops "
            f"in batches of {self._batch_size}..."
        )

    # ------------------------------------------------------------------
    # Phase 2: inferring
    # ------------------------------------------------------------------

    def _infer_batch(self):
        """Process one batch of crops per tick, then write CSV when done."""
        start = self._batch_index
        end = min(start + self._batch_size, len(self._pending_rois))
        batch = self._pending_rois[start:end]

        results = self.ocr_reader.read_batch(batch)

        # results is flat [l0, r0, l1, r1, ...] — pair them back up
        for i, pair_start in enumerate(range(0, len(results), 2)):
            ocr_frame_idx = (start // 2) + i
            l_score, l_conf = results[pair_start]
            r_score, r_conf = results[pair_start + 1]
            self._results[ocr_frame_idx] = [l_score, r_score, l_conf, r_conf]

        self.ui.write(f"Inference: {end}/{len(self._pending_rois)} crops processed...")

        self._batch_index = end
        if self._batch_index >= len(self._pending_rois):
            self._write_csv()
            self.stop()

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _write_csv(self):
        """Write results in strict frame ID order."""
        # _ocr_frame_ids is already in ascending order since we collected
        # frames sequentially, but sort defensively
        order = np.argsort(self._ocr_frame_ids)
        for idx in order:
            frame_id = self._ocr_frame_ids[idx]
            l_score, r_score, l_conf, r_conf = self._results[idx]
            self.csv_writer.writerow([frame_id, l_score, r_score, l_conf, r_conf])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

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
        self._pending_rois.clear()
        self._ocr_frame_ids.clear()
        self._results = None

    def stop(self):
        self.cleanup()
        self.finished.emit()

    # ------------------------------------------------------------------
    # Validation
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

    cProfile.run("main()", "profile.stats")
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(10)
