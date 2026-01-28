import os
from typing import override

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from scripts.momentum_graph.plot_momentum import (
    densify_lights_data,
    get_momentum_data_points,
    plot_score_light_progression,
)
from scripts.momentum_graph.process_score_lights import process_score_lights
from scripts.momentum_graph.process_scores import process_scores
from scripts.momentum_graph.util.evaluate_score_events import (
    refine_score_frames_with_lights,
)
from scripts.momentum_graph.util.extract_score_increases import extract_score_increases
from src.gui.util.conversion import pixmap_to_np
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model import PysideUi
from src.util.file_names import (
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
    MOMENTUM_DATA_CSV_NAME,
    OCR_OUTPUT_CSV_NAME,
    ORIGINAL_VIDEO_NAME,
)
from src.util.io import setup_input_video_io

from .base_task_widget import BaseTaskWidget


class GenerateMomentumGraphWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    @override
    def setup(self):
        self.interactive_ui.write("Press 'Run' to start generating the momentum graph.")

    def _on_finished(self):
        self.interactive_ui.write("Momentum graph generated.")
        self.run_completed.emit(MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH)

    @override
    @Slot()
    def on_runButton_clicked(self):
        if not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH)

        # Create controller
        self.controller = MomentumGraphController(
            working_dir=self.working_dir,
            ui=self.interactive_ui,
            parent=self,
        )

        # When finished → emit completion
        self.controller.finished.connect(self._on_finished)

        # Start async pipeline
        self.controller.start()


class MomentumGraphController(QObject):
    finished = Signal()

    def cancel(self):
        """Required to match BaseTaskWidget; no-op here."""
        pass

    def __init__(self, working_dir, ui: PysideUi, parent=None):
        super().__init__(parent)
        self.working_dir = working_dir
        self.ui = ui
        self.parent = parent
        self.algorithm = "last_activation"

    def _obtain_score_increases(self, fps: float, total_length: int):
        scores_df = pd.read_csv(os.path.join(self.working_dir, OCR_OUTPUT_CSV_NAME))
        pred = process_scores(
            scores_df, total_length=total_length, window_median=int(fps * 6)
        )

        frame_ids = np.arange(len(pred))
        processed_scores_df = pd.DataFrame(
            {"frame_id": frame_ids, "left_score": pred[:, 0], "right_score": pred[:, 1]}
        )
        return extract_score_increases(processed_scores_df)

    def _obtain_score_lights_occurrences(self, fps: float, total_length: int):
        score_lights_df = pd.read_csv(
            os.path.join(self.working_dir, DETECT_LIGHTS_OUTPUT_CSV_NAME)
        )
        score_lights_df = process_score_lights(score_lights_df, fps=fps)
        changes = score_lights_df.loc[
            (score_lights_df["left_light"].diff() != 0)
            | (score_lights_df["right_light"].diff() != 0),
            ["frame_id", "left_light", "right_light"],
        ]

        return densify_lights_data(changes, total_length=total_length)

    def start(self):
        self.ui.write("Generating momentum graph... (this may take a while)")

        cap, fps, _, _, total_length = setup_input_video_io(
            os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        )
        cap.release()

        score_increases = self._obtain_score_increases(
            fps=fps, total_length=total_length
        )
        lights = self._obtain_score_lights_occurrences(
            fps=fps, total_length=total_length
        )
        score_occurrences = refine_score_frames_with_lights(
            lights, score_increases, fps, algorithm=self.algorithm
        )

        frames, momenta = get_momentum_data_points(score_occurrences, fps)

        momentum_df = pd.DataFrame({"frame_id": frames, "momentum": momenta})
        momentum_df.to_csv(
            os.path.join(self.working_dir, MOMENTUM_DATA_CSV_NAME), index=False
        )

        self.plot_momentum_on_label(frames, momenta, fps)

    def plot_momentum_on_label(self, frames, momenta, fps):
        pixmap = get_momentum_graph_pixmap(frames, momenta, fps)

        # Paint onto the label
        self.ui.set_fresh_frame(pixmap_to_np(pixmap))
        self.finished.emit()


def get_momentum_graph_pixmap(frames, momenta, fps):
    fig = Figure(figsize=(10, 5), dpi=300)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.plot(
        frames / fps,
        momenta,
        label="Momentum",
        linewidth=2,
        color="tab:green",
        marker="o",
        markersize=6,
        markeredgecolor="k",
        markerfacecolor="tab:green",
    )
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_title("Momentum Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Momentum (+ left / - right)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    # Render to RGBA array
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(int(height), int(width), 4)
    # Convert ARGB → RGBA
    img = img[:, :, [1, 2, 3, 0]]
    img = np.ascontiguousarray(img)

    # Convert to QPixmap
    pixmap = QPixmap.fromImage(
        QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGBA8888)
    )
    return pixmap


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = GenerateMomentumGraphWidget()
    widget.set_working_directory("matches_data/foil_3")
    widget.show()
    sys.exit(app.exec())
