import os
from typing import override

import cv2

from scripts.manual_track_fencers import (
    NUM_FRAMES_TO_SKIP,
    appears_in_future_detections,
    get_header_row,
    row_mapper,
)
from src.gui.util.task_graph import HeatMapTasksToIds
from src.model import FrameInfoManager, PysideUi, Ui
from src.model.InputController import InputController
from src.util.file_names import ORIGINAL_VIDEO_NAME, RAW_POSE_DATA_CSV_NAME

from ..momentum_graph.base_task_widget import BaseTaskWidget


class TrackFencersWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        self.interactive_ui.write("Press 'Run' to start tracking fencers.")

    @override
    def on_runButton_clicked(self):
        if not self.working_dir:
            return

        self.run_started.emit(HeatMapTasksToIds.TRACK_FENCERS)

        input_video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)

        # Create controller
        self.controller = ObtainFencerIdsController(
            ui=self.interactive_ui,
            video_path=input_video_path,
            csv_path=os.path.join(self.working_dir, RAW_POSE_DATA_CSV_NAME),
        )

        # When finished → emit completion
        self.controller.set_on_finished(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.interactive_ui.write("Fencer tracking completed.")
        self.run_completed.emit(HeatMapTasksToIds.TRACK_FENCERS)


class ObtainFencerIdsTask:
    def __init__(
        self,
        ui: Ui,
        input: InputController,
        csv_path: str,
        video_path: str,
    ):
        self.ui = ui
        self.input = input

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ms_per_frame = int(1000 / self.fps)

        self.frame_manager = FrameInfoManager(
            csv_path, self.fps, get_header_row(), row_mapper
        )

        # State
        self.frame_idx = 0
        self.internal_clock = 0
        self.selection_in_progress = False

        self.current_left = None
        self.current_right = None

        self.left_ids = set()
        self.right_ids = set()
        self.not_left = set()
        self.not_right = set()

        self.left_timer = -1e9
        self.right_timer = -1e9

    # ---------------- Loop step ----------------

    def step(self) -> bool:
        if self.selection_in_progress:
            return True  # pause progression

        ret, frame = self.cap.read()
        if not ret:
            return False

        detections = self.frame_manager.get_frame_and_advance(self.frame_idx)
        self.frame_idx += 1

        self.ui.set_frame(frame)
        self.ui.draw_candidates(detections)

        self._invalidate_lost_fencers(detections)

        if self._needs_left_selection():
            self._request_selection(left=True, detections=detections)
            return True

        if self._needs_right_selection():
            self._request_selection(left=False, detections=detections)
            return True

        self.internal_clock += self.ms_per_frame
        return True

    def _request_selection(self, *, left: bool, detections):
        self.selection_in_progress = True

        known = self.left_ids if left else self.right_ids
        exclude = self.not_left if left else self.not_right

        candidates = {i: d for i, d in detections.items() if i not in exclude}
        if not candidates:
            self.selection_in_progress = False
            return

        self.input.request_fencer_selection(
            candidates=candidates,
            left=left,
            callback=lambda result: self._on_selection(left, result),
        )

    def _on_selection(self, left: bool, result: int | None):
        self.selection_in_progress = False

        if result is None:
            delay = NUM_FRAMES_TO_SKIP * self.ms_per_frame
            if left:
                self.left_timer = self.internal_clock + delay
            else:
                self.right_timer = self.internal_clock + delay
            return

        if left:
            self.current_left = result
            self.left_ids.add(result)
            self.not_right.add(result)
        else:
            self.current_right = result
            self.right_ids.add(result)
            self.not_left.add(result)

    def _needs_left_selection(self) -> bool:
        return self.current_left is None and self.left_timer < self.internal_clock

    def _needs_right_selection(self) -> bool:
        return self.current_right is None and self.right_timer < self.internal_clock

    def _invalidate_lost_fencers(self, detections: dict[int, dict]):
        if (
            self.current_left is not None
            and self.current_left not in detections
            and not appears_in_future_detections(
                self.frame_manager, self.frame_idx, self.current_left
            )
        ):
            self.current_left = None

        if (
            self.current_right is not None
            and self.current_right not in detections
            and not appears_in_future_detections(
                self.frame_manager, self.frame_idx, self.current_right
            )
        ):
            self.current_right = None

    def cleanup(self):
        self.input.cancel()
        if self.cap:
            self.cap.release()
        self.ui.close()


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = TrackFencersWidget()
    widget.set_working_directory("matches_data/sabre_2")
    widget.show()
    sys.exit(app.exec())
