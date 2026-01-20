import os

import cv2
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from scripts.momentum_graph.crop_scoreboard import crop_region
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import PysideUi
from src.util.file_names import CROPPED_SCORE_LIGHTS_VIDEO_NAME, ORIGINAL_VIDEO_NAME

from .ui_crop_score_lights_widget import Ui_CropScoreLightsWidget


class CropScoreLightsWidget(QWidget):
    back_button_clicked = Signal()
    run_started = Signal(object)
    run_completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_CropScoreLightsWidget()
        self.ui.setupUi(self)

        self.cap = None
        self.working_dir = None

        self.ui.backButton.clicked.connect(self.back_button_clicked.emit)

        # UI adapter (composition)
        self.interactive_ui = PysideUi(
            video_label=self.ui.videoLabel,
            text_label=self.ui.uiTextLabel,
            parent=self,
        )

        self.ui.runButton.clicked.connect(self.on_run_button_clicked)
        self.interactive_ui.task_completed.connect(
            lambda: self.run_completed.emit(MomentumGraphTasksToIds.CROP_SCORE_LIGHTS)
        )

    def set_working_directory(self, working_dir: str):
        self.working_dir = working_dir
        video_path = os.path.join(working_dir, ORIGINAL_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        self.interactive_ui.show_single_frame(self.cap)

    def on_run_button_clicked(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.CROP_SCORE_LIGHTS)
        crop_region(
            cap=self.cap,
            output_path=os.path.join(self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME),
            ui=self.interactive_ui,
        )


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = CropScoreLightsWidget()
    widget.set_working_directory("matches_data/sabre_1/original.mp4")
    widget.show()
    sys.exit(app.exec())
