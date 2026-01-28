import os
from typing import override

import cv2
from PySide6.QtCore import Slot

from src.gui.momentum_graph.crop_scoreboard_widget import CropRegionPipeline
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.util.file_names import CROPPED_SCORE_LIGHTS_VIDEO_NAME, ORIGINAL_VIDEO_NAME

from .base_task_widget import BaseTaskWidget


class CropScoreLightsWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

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
        self.interactive_ui.write("Press 'Run' to start cropping the score lights.")

    @override
    @Slot()
    def on_runButton_clicked(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.CROP_SCORE_LIGHTS)

        output_path = os.path.join(self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME)

        # Create controller
        self.controller = CropRegionPipeline(
            cap=self.cap,
            output_path=output_path,
            ui=self.interactive_ui,
            parent=self,
            region="score lights",
        )
        self.controller.set_on_finished(self._on_finished)

        # Start async pipeline
        self.controller.start()

    def _on_finished(self):
        self.interactive_ui.write("Cropping score lights completed.")
        self.run_completed.emit(MomentumGraphTasksToIds.CROP_SCORE_LIGHTS)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = CropScoreLightsWidget()
    widget.set_working_directory("matches_data/sabre_1")
    widget.show()
    sys.exit(app.exec())
