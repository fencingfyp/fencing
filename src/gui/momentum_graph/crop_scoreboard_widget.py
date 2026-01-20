import os

import cv2
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QWidget

from scripts.momentum_graph.crop_scoreboard import crop_region
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import PysideUi
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME

from .ui_crop_scoreboard_widget import Ui_CropScoreboardWidget


class CropScoreboardWidget(QWidget):
    back_button_clicked = Signal()
    run_started = Signal(object)
    run_completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_CropScoreboardWidget()
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

        self.interactive_ui.task_completed.connect(
            lambda: self.run_completed.emit(MomentumGraphTasksToIds.CROP_SCOREBOARD)
        )

    @Slot(str)
    def set_working_directory(self, working_dir: str):
        self.working_dir = working_dir
        video_path = os.path.join(working_dir, ORIGINAL_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        self.interactive_ui.show_single_frame(self.cap)

    @Slot()
    def on_runButton_clicked(self):
        """Partial camel case naming so Pyside6 can auto-connect the slot."""
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.CROP_SCOREBOARD)
        crop_region(
            cap=self.cap,
            output_path=os.path.join(self.working_dir, CROPPED_SCOREBOARD_VIDEO_NAME),
            ui=self.interactive_ui,
        )


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = CropScoreboardWidget()
    widget.set_working_directory("matches_data/sabre_1")
    widget.show()
    sys.exit(app.exec())
