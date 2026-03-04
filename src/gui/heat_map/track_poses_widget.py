import csv
import os
import time
from types import SimpleNamespace
from typing import override

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

from scripts.estimate_poses import get_header_row
from src.gui.base_task_widget.base_task_widget import BaseTaskWidget
from src.gui.base_task_widget.ui.PysideUi import PysideUi
from src.gui.MatchContext import MatchContext
from src.gui.task_graph.task_graph import TasksToIds
from src.model.FileManager import FileRole
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

        self.run_started.emit(TasksToIds.TRACK_POSES.value)

        input_video_path = self.match_context.file_manager.get_original_video()
        model_path = os.path.join("models", "yolo", "yolo26l-pose.pt")

        self.t0 = time.time()

        # Create controller
        self.controller = Roller(
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
        self.ui.write(
            "Pose tracking completed. Elapsed time: {:.2f} seconds.".format(
                time.time() - self.t0
            )
        )
        self.ui.hide_loading()
        self.run_completed.emit(TasksToIds.TRACK_POSES.value)

    def cancel(self):
        if hasattr(self, "controller"):
            self.controller.cancel()


BATCH_SIZE = 8

BOTSORT_ARGS = SimpleNamespace(
    tracker_type="botsort",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
    gmc_method="sparseOptFlow",
    proximity_thresh=0.5,
    appearance_thresh=0.8,
    with_reid=False,
    model="auto",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def extract_rows(frame_idx: int, result, track_ids: dict[int, int]) -> list:
    """
    result: single YOLO Results object for one frame
    track_ids: det_ind -> track_id from BOTSORT output
    """
    output = []
    if not hasattr(result, "boxes") or result.boxes is None:
        return output
    for i, (box, kps) in enumerate(zip(result.boxes, result.keypoints)):
        if int(box.cls.item()) != 0:
            continue
        track_id = track_ids.get(i, -1)
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        row = [frame_idx, track_id, float(box.conf.item()), x1, y1, x2, y2]
        for (x, y), v in zip(kps.xy[0].tolist(), kps.conf[0].tolist()):
            row.extend([x, y, v])
        output.append(row)
    return output


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------


class _RollerWorker(QThread):
    progress = Signal(float)
    finished = Signal()
    error = Signal(str)

    def __init__(self, input_path, output_path, model_path, device):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.device = device
        self._cancelled = False

    def run(self):
        try:
            self._process()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _process(self):
        model = YOLO(self.model_path, task="pose")

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.error.emit(f"Error opening video file: {self.input_path}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, BATCH_SIZE * 2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        tracker = BOTSORT(BOTSORT_ARGS, frame_rate=int(fps))

        csv_file = open(self.output_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(get_header_row())

        frame_idx = 0
        try:
            while not self._cancelled:
                # Read a batch of frames
                frames = []
                frame_indices = []
                for _ in range(BATCH_SIZE):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                    frame_indices.append(frame_idx)
                    frame_idx += 1

                if not frames:
                    break

                # Batch inference
                batch_results = model.predict(
                    frames,
                    verbose=False,
                    device=self.device,
                    half=True,
                    batch=BATCH_SIZE,
                )

                # Feed each frame into BOTSORT sequentially
                for result, frame, fidx in zip(batch_results, frames, frame_indices):
                    result.boxes = (
                        result.boxes.cpu() if result.boxes is not None else None
                    )
                    result.keypoints = (
                        result.keypoints.cpu() if result.keypoints is not None else None
                    )
                    tracks = tracker.update(result.boxes, frame)

                    track_ids = {}
                    if len(tracks) > 0:
                        for track in tracks:
                            det_ind = int(track[7])
                            track_ids[det_ind] = int(track[4])

                    rows = extract_rows(fidx, result, track_ids)
                    if rows:
                        writer.writerows(rows)

                pct = frame_idx / total_frames if total_frames > 0 else 0.0
                self.progress.emit(pct)

        finally:
            cap.release()
            csv_file.close()

    def cancel(self):
        self._cancelled = True


# ------------------------------------------------------------------
# Roller
# ------------------------------------------------------------------


class Roller:
    """
    Controller for processing a video with YOLO pose + offline BOTSORT tracking,
    writing results to CSV.

    Runs in a QThread to keep the Qt event loop responsive during inference.
    Frames are read in batches of BATCH_SIZE for native YOLO batch inference.
    BOTSORT processes each frame sequentially to maintain correct track state.
    """

    def __init__(
        self,
        ui: PysideUi,
        input_path: str,
        output_path: str,
        model_path: str,
    ):
        self.ui = ui
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.device = get_device()
        self._on_finished = None
        self._worker: _RollerWorker | None = None

    def set_on_finished(self, callback):
        self._on_finished = callback

    def on_finished(self):
        self.cancel()
        if self._on_finished:
            self._on_finished()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self.ui.write("Loading YOLO model...")
        self._worker = _RollerWorker(
            self.input_path,
            self.output_path,
            self.model_path,
            self.device,
        )
        self._worker.progress.connect(
            lambda pct: self.ui.update_loading(pct, "Processing video...")
        )
        self._worker.finished.connect(self.on_finished)
        self._worker.error.connect(lambda e: self.ui.write(f"Error: {e}"))
        self._worker.start()
        self.ui.write("", silent=True)  # clear previous messages
        self.ui.show_loading("Processing video...")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait()  # block until thread exits cleanly before releasing resources
            self._worker = None


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = TrackPosesWidget(match_context)
        match_context.set_file("matches_data/sabre_7.mp4")
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
