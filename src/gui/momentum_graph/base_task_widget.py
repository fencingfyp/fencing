import os

import cv2
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QWidget

from src.model import PysideUi
from src.util.file_names import ORIGINAL_VIDEO_NAME

from .ui_base_task_widget import Ui_BaseTaskWidget


class BaseTaskWidget(QWidget):
    back_button_clicked = Signal()
    run_started = Signal(object)
    run_completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_BaseTaskWidget()
        self.ui.setupUi(self)
        self.video_label_original_size = self.ui.videoLabel.size()

        self.cap = None
        self.working_dir = None
        self.controller = None

        self.ui.backButton.clicked.connect(self.on_back_button_clicked)

        self.interactive_ui = PysideUi(
            video_label=self.ui.videoLabel,
            text_label=self.ui.uiTextLabel,
            parent=self,
        )

    def on_back_button_clicked(self):
        if self.controller:
            self.controller.cancel()
        self.interactive_ui.close()
        self.back_button_clicked.emit()

    @Slot(str)
    def set_working_directory(self, working_dir: str):
        self.working_dir = working_dir
        self.setup()

    def setup(self):
        pass

    def get_new_video_label_size(
        self, frame_width: int, frame_height: int
    ) -> tuple[int, int]:
        if not self.cap:
            return self.ui.videoLabel.size()
        label_w, label_h = (
            self.video_label_original_size.width(),
            self.video_label_original_size.height(),
        )
        frame_aspect = frame_width / frame_height
        label_aspect = label_w / label_h
        if frame_aspect > label_aspect:
            new_w = label_w
            new_h = int(label_w / frame_aspect)
        else:
            new_h = label_h
            new_w = int(label_h * frame_aspect)
        return new_w, new_h

    @Slot()
    def on_runButton_clicked(self):
        pass
