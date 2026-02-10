import csv
from typing import override

import cv2
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from src.gui.video_player_widget import VideoPlayerWidget
from src.model import Quadrilateral
from src.model.AutoPatchLightDetector import SinglePatchAutoDetector
from src.model.drawable import QuadrilateralDrawable
from src.model.FileManager import FileRole
from src.pyside import MatchContext
from src.pyside.PysideUi import PysideUi
from src.util.utils import generate_select_quadrilateral_instructions

from .base_task_widget import BaseTaskWidget


class DetectScoreLightsWidget(BaseTaskWidget):
    def __init__(self, match_context, parent=None):
        super().__init__(match_context, parent)
        self.video_with_instructions = VideoWithInstructions(self)
        self.register_widget(self.video_with_instructions)

        self.roi_stage = RoiSelectionStage(self.ui)
        self.time_stage = TimeSelectionStage(self.video_with_instructions)
        self.processing_stage = ProcessingStage(self.ui)

        # Stage wiring
        self.roi_stage.roi_selected_callback = self._on_roi_selected
        self.time_stage.timestamps_selected_callback = self._on_timestamps_selected
        self.processing_stage.completed_callback = self._on_processing_completed

    @override
    def setup(self):
        # Show base UI first
        self.show_default_ui()
        self.processing_stage.set_paths(
            self.match_context.file_manager.get_path(FileRole.CROPPED_SCORE_LIGHTS),
            self.match_context.file_manager.get_path(FileRole.RAW_LIGHTS),
        )
        video_path = self.match_context.file_manager.get_path(
            FileRole.CROPPED_SCORE_LIGHTS
        )

        # Start ROI selection
        self.is_running = True
        self.roi_stage.start(frame=cv2.VideoCapture(video_path).read()[1])

    def _on_roi_selected(self, left_quad, right_quad):
        self.processing_stage.set_positions(left_quad, right_quad)
        video_path = self.match_context.file_manager.get_path(
            FileRole.CROPPED_SCORE_LIGHTS
        )
        self.time_stage.activate(video_path)
        self.show_widget(self.video_with_instructions)

    def _on_timestamps_selected(self, timestamps):
        self.time_stage.deactivate()
        self.processing_stage.init_detectors(timestamps)
        self.show_default_ui()
        self.processing_stage.start_processing()

    def _on_processing_completed(self):
        self.ui.write("Score lights detection completed.")
        self.is_running = False

    def cancel(self):
        self.time_stage.deactivate()
        self.processing_stage.cancel()


class VideoWithInstructions(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = VideoPlayerWidget(self)
        self.label = QLabel(self)
        self.label.setStyleSheet(
            "color: white; background: rgba(0,0,0,128); padding: 4px;"
        )
        self.label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.player)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def set_instructions(self, text: str):
        self.label.setText(text)


class RoiSelectionStage:
    def __init__(self, ui: PysideUi):
        self.ui = ui
        self.left_quad = None
        self.right_quad = None
        self.roi_selected_callback = None
        self.frame = None

    def start(self, frame):
        # Use base UI to ask for points
        self.frame = frame
        self.ui.get_n_points_async(
            frame,
            generate_select_quadrilateral_instructions("left fencer score light"),
            self._on_left_done,
        )

    def _on_left_done(self, left_pts):
        self.left_quad = Quadrilateral(left_pts)
        self.ui.get_n_points_async(
            self.frame,
            generate_select_quadrilateral_instructions("right fencer score light"),
            self._on_right_done,
        )

    def _on_right_done(self, right_pts):
        self.right_quad = Quadrilateral(right_pts)
        if self.roi_selected_callback:
            self.roi_selected_callback(self.left_quad, self.right_quad)


class TimeSelectionStage:
    def __init__(self, video_with_instructions: VideoWithInstructions):
        self.player = video_with_instructions.player
        self.video_with_instructions = video_with_instructions
        self.timestamps_selected_callback = None
        self.labels = ["left_neg", "left_pos", "right_neg", "right_pos"]
        self.index = 0
        self.timestamps = {}

        self.instructions = [
            "Press E to select the OFF frame for the left fencer score light.",
            "Press E to select the ON frame for the left fencer score light.",
            "Press E to select the OFF frame for the right fencer score light.",
            "Press E to select the ON frame for the right fencer score light.",
        ]

    def activate(self, video_path: str):
        self.player.set_video_source(video_path)
        self.player.register_shortcut("E", self.mark_frame)
        self.index = 0
        self.video_with_instructions.set_instructions(self.instructions[self.index])
        self.timestamps.clear()

    def deactivate(self):
        pass

    def mark_frame(self):
        label = self.labels[self.index]
        frame = self.player.video_frame.get_current_frame_number()
        self.timestamps[label] = frame
        self.index += 1
        if self.index >= len(self.labels):
            if self.timestamps_selected_callback:
                self.timestamps_selected_callback(self.timestamps)
                return
        self.video_with_instructions.set_instructions(self.instructions[self.index])


class ProcessingStage:
    def __init__(self, ui: PysideUi):
        self.ui = ui
        self.cropped_lights_video_path = None
        self.output_csv_path = None
        self.cap = None
        self.left_detector = None
        self.left_positions = None
        self.right_detector = None
        self.right_positions = None
        self.csv_file = None
        self.csv_writer = None
        self.current_frame_id = 0
        self.completed_callback = None

    def set_paths(self, cropped_lights_video_path: str, output_csv_path: str):
        self.cropped_lights_video_path = cropped_lights_video_path
        self.output_csv_path = output_csv_path

    def init_detectors(self, timestamps):
        cap = cv2.VideoCapture(self.cropped_lights_video_path)

        def grab(fid):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, f = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read frame {fid}")
            return f

        frames = {k: grab(v) for k, v in timestamps.items()}
        cap.release()

        self.left_detector = SinglePatchAutoDetector(
            frames["left_pos"],
            self.left_positions,
            frames["left_neg"],
            self.left_positions,
        )
        self.right_detector = SinglePatchAutoDetector(
            frames["right_pos"],
            self.right_positions,
            frames["right_neg"],
            self.right_positions,
        )

    def set_positions(self, left_quad: Quadrilateral, right_quad: Quadrilateral):
        self.left_positions = left_quad
        self.right_positions = right_quad

    def start_processing(self):
        self.cap = cv2.VideoCapture(self.cropped_lights_video_path)
        self.csv_file = open(
            self.output_csv_path,
            "w",
            newline="",
        )
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame_id", "left_light", "right_light"])
        self.ui.schedule(self._process_frame)

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.finish()
            return

        self.ui.set_fresh_frame(frame)
        left_on = self.left_detector.classify(frame, self.left_positions)
        right_on = self.right_detector.classify(frame, self.right_positions)

        self.ui.draw_objects(
            [
                QuadrilateralDrawable(self.left_positions, (255, 0, 0)),
                QuadrilateralDrawable(self.right_positions, (0, 255, 0)),
            ]
        )
        self.csv_writer.writerow([self.current_frame_id, int(left_on), int(right_on)])
        self.ui.show_frame()
        self.current_frame_id += 1
        self.ui.schedule(self._process_frame)

    def finish(self):
        self.cancel()
        self.completed_callback()

    def cancel(self):
        if self.cap:
            self.cap.release()
        if self.csv_file:
            self.csv_file.close()


if __name__ == "__main__":
    app = QApplication([])
    match_context = MatchContext()
    match_context.set_file("matches_data/sabre_2/sabre_2.mp4")
    widget = DetectScoreLightsWidget()
    widget.show()
    app.exec()
