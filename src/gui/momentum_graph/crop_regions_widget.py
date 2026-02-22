import time
from typing import override

import cv2
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model.FileManager import FileRole
from src.model.tracker.TrackingStrategy import TrackingStrategy
from src.pyside.MatchContext import MatchContext
from src.pyside_pipelines.multi_region_cropper.multi_region_crop_pipeline import (
    MultiRegionProcessingPipeline,
)
from src.pyside_pipelines.multi_region_cropper.output.csv_oneshot_quad_output import (
    CsvOneShotQuadOutput,
)
from src.pyside_pipelines.multi_region_cropper.output.rectified_video_output import (
    RectifiedVideoOutput,
)
from src.pyside_pipelines.multi_region_cropper.roi_selection_controller import (
    LabelConfig,
    ROISelectionPipeline,
)
from src.pyside_pipelines.multi_region_cropper.superpoint_multi_region_pipeline import (
    SuperPointBatchPipeline,
)

from .base_task_widget import BaseTaskWidget


class CropRegionsWidget(BaseTaskWidget):
    def __init__(self, match_context: MatchContext, parent=None):
        super().__init__(match_context, parent)
        self.cap: cv2.VideoCapture | None = None
        self.is_running = False
        self.roi_pipeline: ROISelectionPipeline | None = None
        self.processing_pipeline: (
            MultiRegionProcessingPipeline | SuperPointBatchPipeline | None
        ) = None

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
        self.ui.write("Press 'Run' to start defining and cropping regions of interest.")

        self.run_task()

    def run_task(self):
        if not self.cap or self.is_running:
            return
        self.is_running = True

        self.run_started.emit(MomentumGraphTasksToIds.CROP_REGIONS)

        label_configs = {
            "scoreboard": LabelConfig(
                output_factory=lambda quad, fps: [
                    RectifiedVideoOutput(
                        self.match_context.file_manager.get_path(
                            FileRole.CROPPED_SCOREBOARD
                        ),
                        fps,
                        quad,
                    ),
                ],
                tracking_strategy=TrackingStrategy.ORB,
                mask_margin=0.3,
            ),
            "score lights": LabelConfig(
                output_factory=lambda quad, fps: [
                    RectifiedVideoOutput(
                        self.match_context.file_manager.get_path(
                            FileRole.CROPPED_SCORE_LIGHTS
                        ),
                        fps,
                        quad,
                    )
                ],
                mask_margin=0.3,
            ),
            # "timer": LabelConfig(
            #     output_factory=lambda quad, fps: [
            #         RectifiedVideoOutput(
            #             self.match_context.file_manager.get_path(
            #                 FileRole.CROPPED_TIMER
            #             ),
            #             fps,
            #             quad,
            #         )
            #     ],
            # ),
            "piste": LabelConfig(
                output_factory=lambda quad, fps: [
                    CsvOneShotQuadOutput(
                        self.match_context.file_manager.get_path(
                            FileRole.RAW_PISTE_QUADS
                        ),
                        quad,
                    ),
                    RectifiedVideoOutput(
                        self.match_context.file_manager.get_path(
                            FileRole.CROPPED_PISTE
                        ),
                        fps,
                        quad,
                    ),
                ],
                tracking_strategy=TrackingStrategy.ORB,
                mask_margin=1,
            ),
        }

        # Start ROI selection UI
        ret, first_frame = self.cap.read()
        if not ret:
            self.ui.write("Failed to read first frame for ROI selection.")
            self.is_running = False
            return

        self.roi_pipeline = ROISelectionPipeline(
            first_frame=first_frame,
            ui=self.ui,
            label_configs=label_configs,
            on_finished=self._on_rois_defined,
        )
        self.roi_pipeline.start()

    # -------------------------
    # Callback from ROISelectionPipeline
    # -------------------------
    def _on_rois_defined(self, defined_regions):
        self.t0 = time.time()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        self.processing_pipeline = MultiRegionProcessingPipeline(
            cap=self.cap,
            defined_regions=defined_regions,
            ui=self.ui,
            on_finished=self.on_finished,
        )
        self.processing_pipeline.start()

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
        print(f"CropRegionsWidget finished in {time.time() - self.t0:.2f} seconds.")

    @override
    def cancel(self):
        self.is_running = False
        if self.roi_pipeline:
            self.roi_pipeline = (
                None  # No cleanup needed since outputs aren't created yet
            )
        if self.processing_pipeline:
            self.processing_pipeline.cancel()
            self.processing_pipeline = None
        return super().cancel()


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = CropRegionsWidget(match_context)
        match_context.set_file("matches_data/sabre_2.mp4")
        widget.show()
        sys.exit(app.exec())

    cProfile.run("main()", "profile.stats")
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(20)
