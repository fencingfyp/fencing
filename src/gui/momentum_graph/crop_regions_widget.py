from typing import override

import cv2
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model.FileManager import FileRole
from src.pipelines.multi_region_crop_pipeline import MultiRegionCropPipeline
from src.pyside.MatchContext import MatchContext

from .base_task_widget import BaseTaskWidget


class CropRegionsWidget(BaseTaskWidget):
    def __init__(self, match_context: MatchContext, parent=None):
        super().__init__(match_context, parent)

    @override
    def setup(self):
        if not self.match_context.file_manager.get_working_directory():
            return

        video_path = self.match_context.file_manager.get_original_video()
        self.cap = cv2.VideoCapture(str(video_path))

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read video from {video_path}")

        self.ui.set_fresh_frame(frame)
        self.ui.write("Press 'Run' to start cropping the regions of interest.")

        self.run_task()

    def run_task(self):
        if not self.cap:
            return
        self.is_running = True

        self.run_started.emit(MomentumGraphTasksToIds.CROP_REGIONS)

        labels_to_output_paths = {
            "scoreboard": self.match_context.file_manager.get_path(
                FileRole.CROPPED_SCOREBOARD
            ),
            "score lights": self.match_context.file_manager.get_path(
                FileRole.CROPPED_SCORE_LIGHTS
            ),
            "timer": self.match_context.file_manager.get_path(FileRole.CROPPED_TIMER),
        }

        self.controller = MultiRegionCropPipeline(
            cap=self.cap,
            output_paths=labels_to_output_paths,
            ui=self.ui,
        )
        self.controller.set_on_finished(self.on_finished)
        self.controller.start()

    @override
    @Slot()
    def on_runButton_clicked(self):
        if self.is_running:
            return
        self.run_task()

    def on_finished(self):
        self.is_running = False
        self.run_completed.emit(MomentumGraphTasksToIds.CROP_REGIONS)
        self.ui.write("Cropping regions completed.")


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication, QWidget

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = CropRegionsWidget(match_context)
        match_context.set_file("matches_data/sabre_3.mp4")
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
