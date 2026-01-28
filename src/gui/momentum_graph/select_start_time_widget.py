import json
import os
from typing import override

import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QKeySequence, QShortcut

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import Ui
from src.util.file_names import ORIGINAL_VIDEO_NAME, START_TIME_JSON_NAME

from .base_task_widget import BaseTaskWidget


class SelectStartTimeWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Widget initialization code here
        self.save_path = None

    @override
    def setup(self):
        video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        self.ui.videoLabel.setFixedSize(
            *self.get_new_video_label_size(
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        )
        self.interactive_ui.show_single_frame(self.cap)
        self.interactive_ui.write("Press 'Run' to start selecting the start time.")
        self.save_path = os.path.join(self.working_dir, START_TIME_JSON_NAME)

    @override
    @Slot()
    def on_runButton_clicked(self):
        if not self.cap or not self.working_dir:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.run_started.emit(MomentumGraphTasksToIds.GET_START_TIME)

        output_path = os.path.join(self.working_dir, START_TIME_JSON_NAME)

        # Create controller
        self.controller = GetStartTimeController(
            cap=self.cap,
            save_path=output_path,
            ui=self.interactive_ui,
        )

        # When finished → emit completion
        self.controller.set_on_finished(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.interactive_ui.write("Selecting start time completed.")
        self.run_completed.emit(MomentumGraphTasksToIds.GET_START_TIME)


class GetStartTimeController:
    """
    Controller for interactively selecting a start timestamp (frame index).

    Controls:
    ← / → : step backward / forward one frame
    P     : pause / play
    W     : confirm timestamp
    Q     : quit
    """

    def __init__(self, ui: Ui, cap: cv2.VideoCapture, save_path: str):
        self.ui = ui
        self.cap = cap
        self.save_path = save_path

        self.paused = False
        self.selected_frame: int | None = None

        # Install shortcuts via UI parent
        parent = self.ui.parent

        self._left = QShortcut(QKeySequence(Qt.Key_Left), parent)
        self._right = QShortcut(QKeySequence(Qt.Key_Right), parent)
        self._pause = QShortcut(QKeySequence("P"), parent)
        self._confirm = QShortcut(QKeySequence("W"), parent)

        self._left.activated.connect(self.step_backward)
        self._right.activated.connect(self.step_forward)
        self._pause.activated.connect(self.toggle_pause)
        self._confirm.activated.connect(self.confirm)

        self._on_finished_callback = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_finished(self, callback):
        """Set callback to be called when finished."""
        self._on_finished_callback = callback

    def start(self):
        """Start interactive timestamp selection."""
        self.ui.write(
            "Select start time:\n"
            "← / → : step frames\n"
            "P     : pause / play\n"
            "W     : confirm timestamp\n"
            "Q     : quit"
        )

        # Show initial frame
        self.ui.show_single_frame(self.cap)

        # Start playback loop
        self.ui.run_loop(self._on_step, self.cancel)

    def get_result(self) -> int | None:
        """Returns the selected frame index after completion."""
        return self.selected_frame

    # ------------------------------------------------------------------
    # Loop + controls
    # ------------------------------------------------------------------

    def _on_step(self):
        if self.paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.ui.set_fresh_frame(frame)

    def step_forward(self):
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos + fps)
        self.ui.show_single_frame(self.cap)

    def step_backward(self):
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - fps))
        self.ui.show_single_frame(self.cap)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.ui.write("Paused. ← / → to step, W to confirm.")
        else:
            self.ui.write("Playing. P to pause, W to confirm.")

    def confirm(self):
        self.selected_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Selected start frame: {self.selected_frame}")
        with open(self.save_path, "w") as f:
            json.dump({"start_frame": self.selected_frame}, f)
        self.cancel()
        self.on_finished()

    def on_finished(self):
        if self._on_finished_callback:
            self._on_finished_callback()

    def cancel(self):
        self.selected_frame = None
        for sc in [self._left, self._right, self._pause, self._confirm]:
            sc.setParent(None)
        self.cap.release()


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = SelectStartTimeWidget()
    widget.set_working_directory("matches_data/sabre_1")
    widget.show()
    sys.exit(app.exec())
