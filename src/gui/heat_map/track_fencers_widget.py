import os
from typing import override

from scripts.manual_track_fencers import (
    NUM_FRAMES_TO_SKIP,
    appears_in_future_detections,
    get_header_row,
    reprocess_csv,
    row_mapper,
)
from src.gui.util.task_graph import HeatMapTasksToIds
from src.model import FrameInfoManager
from src.pyside.PysideUi import PysideUi
from src.util.file_names import (
    ORIGINAL_VIDEO_NAME,
    PROCESSED_POSE_DATA_CSV_NAME,
    RAW_POSE_DATA_CSV_NAME,
)
from src.util.io import setup_input_video_io
from src.util.lru_frame_reader import LruFrameReader

from ..momentum_graph.base_task_widget import BaseTaskWidget


class TrackFencersWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        self.ui.write("Press 'Run' to start tracking fencers.")
        self.run_task()

    @override
    def on_runButton_clicked(self):
        self.run_task()

    def run_task(self):
        if not self.working_dir:
            return

        self.run_started.emit(HeatMapTasksToIds.TRACK_FENCERS)

        input_video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        input_csv_path = os.path.join(
            self.working_dir,
            RAW_POSE_DATA_CSV_NAME,
        )
        output_csv_path = os.path.join(self.working_dir, PROCESSED_POSE_DATA_CSV_NAME)

        # Create controller
        self.controller = FencerAssignmentController(
            video_path=input_video_path,
            input_csv_path=input_csv_path,
            ui=self.ui,
            output_csv_path=output_csv_path,
        )

        # When finished → emit completion
        self.controller.set_on_finished(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.ui.write("Fencer tracking completed.")
        self.run_completed.emit(HeatMapTasksToIds.TRACK_FENCERS)

    def cancel(self):
        self.ui.cancel_running_subtasks()
        return super().cancel()


class FencerAssignmentController:
    def __init__(self, video_path, input_csv_path, ui: PysideUi, output_csv_path):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path

        cap, fps, _, _, _ = setup_input_video_io(video_path)
        self.cap = cap
        self.frame_reader = LruFrameReader(
            cap, max_cache_bytes=1 * 1024 * 1024 * 1024
        )  # 1 GB cache
        self.frame_manager = FrameInfoManager(
            csv_path=input_csv_path,
            row_mapper=row_mapper,
            fps=fps,
            header_format=get_header_row(),
        )

        self.ui = ui
        self.frame_idx = 0
        self.internal_clock = 0
        self.ms_per_frame = int(1000 / fps)

        # Fencer tracking
        self.current_left_id = None
        self.current_right_id = None
        self.left_ids = set()
        self.right_ids = set()
        self.not_left_ids = set()
        self.not_right_ids = set()

        self.left_timer = -1e9
        self.right_timer = -1e9

        self.cancelled = False

    def start(self):
        self._schedule(self.advance)

    def advance(self):
        if self.cancelled:
            return

        ret, frame = self.frame_reader.read_at(self.frame_idx)
        if not ret:
            self.on_finish()
            return

        detections = self.frame_manager.get_frame_and_advance(self.frame_idx)

        self.ui.set_fresh_frame(frame)
        self.ui.video_renderer.render_detections(detections)

        self._invalidate_lost_fencers(detections)

        if self._needs_left_selection():
            self._request_selection(left=True, detections=detections)
            return

        if self._needs_right_selection():
            self._request_selection(left=False, detections=detections)
            return

        self._advance_frame()
        self._schedule(self.advance)

    # --------------------------------------------------------
    # State handlers
    # --------------------------------------------------------
    def _invalidate_lost_fencers(self, detections: dict[int, dict]):
        if (
            self.current_left_id is not None
            and self.current_left_id not in detections
            and not appears_in_future_detections(
                self.frame_manager, self.frame_idx, self.current_left_id
            )
        ):
            self.current_left_id = None

        if (
            self.current_right_id is not None
            and self.current_right_id not in detections
            and not appears_in_future_detections(
                self.frame_manager, self.frame_idx, self.current_right_id
            )
        ):
            self.current_right_id = None

    def _request_selection(self, left: bool, detections: dict):
        candidates = {
            det_id: det
            for det_id, det in detections.items()
            if det_id not in (self.not_left_ids if left else self.not_right_ids)
        }

        if not candidates:
            self._apply_skip(left)
            self._advance_frame()
            self._schedule(self.advance)
            return

        self.ui.get_fencer_id(
            candidates,
            left,
            on_done=lambda result: self._on_selection_done(left, result),
        )

    def _on_selection_done(self, left: bool, result):
        if result is None:
            self._apply_skip(left)
        else:
            if left:
                self.current_left_id = result
                self.left_ids.add(result)
                self.not_right_ids.add(result)
            else:
                self.current_right_id = result
                self.right_ids.add(result)
                self.not_left_ids.add(result)

        self._schedule(self.advance)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _needs_left_selection(self):
        return self.current_left_id is None and self.left_timer < self.internal_clock

    def _needs_right_selection(self):
        return self.current_right_id is None and self.right_timer < self.internal_clock

    def _apply_skip(self, left: bool):
        timer = self.internal_clock + NUM_FRAMES_TO_SKIP * self.ms_per_frame
        if left:
            self.left_timer = timer
        else:
            self.right_timer = timer

    def _advance_frame(self):
        self.frame_idx += 1
        self.internal_clock += self.ms_per_frame

    def _schedule(self, callback, delay_ms=0):
        if self.cancelled:
            return
        self.ui.schedule(callback, delay_ms)

    # ------------------------------------------------------------------
    # Finish / quit
    # ------------------------------------------------------------------

    def on_finish(self):
        self.cancel()

        if self._on_finished_callback:
            self._on_finished_callback()

        # write to output
        reprocess_csv(
            input_csv=self.input_csv_path,
            left_fencer_ids=self.left_ids,
            right_fencer_ids=self.right_ids,
            output_csv_path=self.output_csv_path,
        )

    def set_on_finished(self, callback):
        self._on_finished_callback = callback

    def cancel(self):
        self.cancelled = True
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    import cProfile
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        widget = TrackFencersWidget()
        widget.set_working_directory("matches_data/sabre_2")
        widget.show()
        sys.exit(app.exec())

    cProfile.run("main()", sort="tottime")
