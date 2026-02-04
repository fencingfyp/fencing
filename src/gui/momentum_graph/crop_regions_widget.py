import os
from typing import override

import cv2
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model.PysideUi import PysideUi
from src.pipelines.multi_region_crop_pipeline import MultiRegionCropPipeline
from src.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    CROPPED_SCOREBOARD_VIDEO_NAME,
    CROPPED_TIMER_VIDEO_NAME,
    ORIGINAL_VIDEO_NAME,
)

from .base_task_widget import BaseTaskWidget


class CropRegionsWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        self.cap = cv2.VideoCapture(video_path)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read video from {video_path}")
        self.ui.videoLabel.setFixedSize(
            *self.get_new_video_label_size(
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        )
        self.interactive_ui.set_fresh_frame(frame)
        self.interactive_ui.show_frame()
        self.interactive_ui.write(
            "Press 'Run' to start cropping the regions of interest."
        )
        self.ui.runButton.setEnabled(True)

        # Temporarily hide run button until we implement run options
        self.ui.runButton.hide()
        self.run_task()

    def run_task(self):
        if not self.cap or not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.CROP_REGIONS)
        self.is_running = True

        labels_to_output_paths = {
            "scoreboard": os.path.join(self.working_dir, CROPPED_SCOREBOARD_VIDEO_NAME),
            "score lights": os.path.join(
                self.working_dir, CROPPED_SCORE_LIGHTS_VIDEO_NAME
            ),
            "timer": os.path.join(self.working_dir, CROPPED_TIMER_VIDEO_NAME),
        }

        # Create controller
        self.controller = MultiRegionCropPipeline(
            cap=self.cap,
            output_paths=labels_to_output_paths,
            ui=self.interactive_ui,
        )
        self.controller.set_on_finished(self.on_finished)

        self.controller.start()
        self.ui.runButton.setEnabled(False)

    @override
    @Slot()
    def on_runButton_clicked(self):
        self.run_task()

    def on_finished(self):
        self.run_completed.emit(MomentumGraphTasksToIds.CROP_REGIONS)
        self.interactive_ui.write("Cropping regions completed.")
        self.is_running = False


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication, QWidget

    def main():
        app = QApplication(sys.argv)
        widget = CropRegionsWidget()
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
