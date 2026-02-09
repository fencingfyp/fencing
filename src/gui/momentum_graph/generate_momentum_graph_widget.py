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
from scripts.momentum_graph.process_score_lights import (
    process_score_lights,
    process_score_lights_np,
)
from scripts.momentum_graph.process_scores import densify_frames_np, process_scores
from scripts.momentum_graph.util.evaluate_score_events import (
    refine_score_frames_with_lights,
)
from scripts.momentum_graph.util.extract_score_increases import (
    extract_score_increases,
    extract_score_increases_np,
)
from src.gui.util.conversion import pixmap_to_np
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.pyside.PysideUi import PysideUi
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
        self.ui.write("Press 'Run' to start generating the momentum graph.")

        self.run_task()

    def _on_finished(self):
        self.ui.write("Momentum graph generated.")
        self.run_completed.emit(MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH)

    @override
    @Slot()
    def on_runButton_clicked(self):
        self.run_task()

    def run_task(self):
        if not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH)

        # Create controller
        self.controller = MomentumGraphController(
            working_dir=self.working_dir,
            ui=self.ui,
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
        self.algorithm = "last_activation"

    def _obtain_score_increases(self, fps: float, total_length: int):
        scores_df = pd.read_csv(os.path.join(self.working_dir, OCR_OUTPUT_CSV_NAME))
        pred = process_scores(
            scores_df, total_length=total_length, window_median=int(fps * 6)
        )

        frame_ids = np.arange(len(pred))
        return extract_score_increases_np(frame_ids, pred[:, 0], pred[:, 1])

    def _obtain_score_lights_occurrences(self, fps: float, total_length: int):
        df = pd.read_csv(os.path.join(self.working_dir, DETECT_LIGHTS_OUTPUT_CSV_NAME))

        frame_ids = df["frame_id"].to_numpy(dtype=np.int32)
        lights = np.column_stack(
            (
                df["left_light"].to_numpy(),
                df["right_light"].to_numpy(),
            )
        )

        # clean lights
        lights = process_score_lights_np(lights, fps=int(fps))

        # detect changes (vectorised)
        diffs = np.any(lights[1:] != lights[:-1], axis=1)
        change_idx = np.flatnonzero(diffs) + 1  # +1 because diff is shifted

        change_frames = frame_ids[change_idx]
        change_lights = lights[change_idx]

        # densify
        return densify_frames_np(
            change_frames,
            change_lights[:, 0],
            change_lights[:, 1],
            total_length=total_length,
        )

    def start(self):
        self.ui.write("Generating momentum graph... (this may take a while)")

        cap, fps, _, _, total_length = setup_input_video_io(
            os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        )
        cap.release()

        score_increases = self._obtain_score_increases(
            fps=fps, total_length=total_length
        )
        frame_ids, left_lights, right_lights = self._obtain_score_lights_occurrences(
            fps=fps, total_length=total_length
        )
        stacked_lights = np.column_stack((left_lights, right_lights))
        score_occurrences = refine_score_frames_with_lights(
            stacked_lights, score_increases, fps, algorithm=self.algorithm
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


def get_momentum_graph_pixmap(frames, momenta, fps, periods=None):
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
    # Overlay periods (in data coordinates)
    if periods:
        ymin, ymax = ax.get_ylim()
        for i, p in enumerate(periods):
            t_start = p["start_ms"] / 1000
            t_end = p["end_ms"] / 1000
            ax.axvspan(
                t_start,
                t_end,
                color="red",
                alpha=0.3,
                zorder=0,
            )

            ax.text(
                t_start,
                ymax,
                f"Period {i+1}",
                va="top",
                ha="left",
                color="white",
                fontsize=9,
                zorder=1,
            )

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
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication, QWidget

    def main():
        app = QApplication(sys.argv)
        widget = GenerateMomentumGraphWidget()
        widget.set_working_directory("matches_data/sabre_2")
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
