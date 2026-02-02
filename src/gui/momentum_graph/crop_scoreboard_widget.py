import os
from typing import override

import cv2
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model.PysideUi2 import PysideUi2
from src.pipelines.crop_region_pipeline import CropRegionPipeline
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME

from .base_task_widget import BaseTaskWidget


class CropScoreboardWidget(BaseTaskWidget):
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
        self.interactive_ui.write("Press 'Run' to start cropping the scoreboard.")

    def run_task(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.CROP_SCOREBOARD)

        output_path = os.path.join(self.working_dir, CROPPED_SCOREBOARD_VIDEO_NAME)

        # Create controller
        self.controller = CropRegionPipeline(
            cap=self.cap,
            output_path=output_path,
            ui=PysideUi2(
                video_label=self.interactive_ui.video_label,
                text_label=self.interactive_ui.text_label,
                parent=self,
            ),
            region="scoreboard",
        )
        self.controller.set_on_finished(self.on_finished)

        self.controller.start()

    @override
    @Slot()
    def on_runButton_clicked(self):
        self.run_task()

    def on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.CROP_SCOREBOARD)
        self.interactive_ui.write("Cropping scoreboard completed.")


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        widget = CropScoreboardWidget()
        widget.set_working_directory("matches_data/sabre_1")
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
