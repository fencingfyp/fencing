from typing import override

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from scripts.momentum_graph.plot_momentum import get_momentum_data_points
from scripts.momentum_graph.process_score_lights import process_score_lights_np
from scripts.momentum_graph.process_scores import densify_frames_np, process_scores
from scripts.momentum_graph.util.evaluate_score_events import (
    refine_score_frames_with_lights,
)
from scripts.momentum_graph.util.extract_score_increases import (
    extract_score_increases_np,
)
from src.gui.util.conversion import pixmap_to_np
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model.FileManager import FileRole
from src.pyside.MatchContext import MatchContext
from src.pyside.PysideUi import PysideUi
from src.util.io import setup_input_video_io

from .base_task_widget import BaseTaskWidget


class GenerateMomentumGraphWidget(BaseTaskWidget):
    def __init__(self, match_context, parent=None):
        super().__init__(match_context, parent)

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
        if not self.match_context.file_manager:
            return

        file_manager = self.match_context.file_manager

        self.run_started.emit(MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH)

        # Create controller
        self.controller = MomentumGraphController(
            file_paths={
                "ocr_csv": file_manager.get_path(FileRole.RAW_SCORES),
                "score_lights_csv": file_manager.get_path(FileRole.RAW_LIGHTS),
                "video": file_manager.get_original_video(),
                "output_csv": file_manager.get_path(FileRole.MOMENTUM_DATA),
            },
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

    def __init__(self, file_paths, ui: PysideUi, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.ui = ui
        self.algorithm = "last_activation"

    def _obtain_score_increases(self, fps: float, total_length: int):
        scores_df = pd.read_csv(self.file_paths["ocr_csv"])
        pred = process_scores(
            scores_df, total_length=total_length, window_median=int(fps * 6)
        )

        frame_ids = np.arange(len(pred))
        return extract_score_increases_np(frame_ids, pred[:, 0], pred[:, 1])

    def _obtain_score_lights_occurrences(self, fps: float, total_length: int):
        df = pd.read_csv(self.file_paths["score_lights_csv"])

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

        cap, fps, _, _, total_length = setup_input_video_io(self.file_paths["video"])
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
        momentum_df.to_csv(self.file_paths["output_csv"], index=False)

        self.plot_momentum_on_label(frames, momenta, fps)

    def plot_momentum_on_label(self, frames, momenta, fps):
        seconds = frames / fps
        pixmap = get_momentum_graph_pixmap(
            overlays=[
                {
                    "seconds": seconds,
                    "momenta": momenta,
                    "label": "Momentum",
                    "color": "tab:green",
                }
            ],
        )

        self.ui.set_fresh_frame(pixmap_to_np(pixmap))
        self.finished.emit()


def draw_period_markers(ax, periods, color="tab:red"):
    if not periods:
        return

    ylim_top = ax.get_ylim()[1]

    for i, p in enumerate(periods):
        ax.axvline(p["start_sec"], linestyle="--", color=color, alpha=0.7)
        ax.axvline(p["end_sec"], linestyle="--", color=color, alpha=0.7)

        ax.text(
            p["start_sec"],
            ylim_top,
            f"P{i+1} Start",
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            color=color,
        )

        ax.text(
            p["end_sec"],
            ylim_top,
            f"P{i+1} End",
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            color=color,
        )


def draw_momentum_curve(
    ax,
    frames,
    momenta,
    fps,
    *,
    label,
    color,
    linewidth=2,
    marker="o",
):
    ax.plot(
        frames / fps,
        momenta,
        label=label,
        linewidth=linewidth,
        color=color,
        marker=marker,
        markersize=5,
        markeredgecolor="k",
        markerfacecolor=color,
        zorder=2,
    )


def realign_periods(
    seconds: np.ndarray,
    periods: list[dict] | None,
) -> tuple[np.ndarray, list[dict] | None]:
    """
    Returns:
        aligned_time_sec
        aligned_periods (in seconds)
    """
    seconds = seconds.copy()

    if not periods or len(periods) == 0:
        return seconds, None  # first point is always at time 0

    first_start_second = periods[0]["start_ms"] / 1000.0
    seconds[0] = first_start_second
    seconds -= first_start_second

    aligned_periods = []
    for p in periods:
        start = p["start_ms"] / 1000.0
        end = p["end_ms"] / 1000.0
        shift = periods[0]["start_ms"] / 1000.0
        aligned_periods.append(
            {
                "start_sec": start - shift,
                "end_sec": end - shift,
            }
        )

    return seconds, aligned_periods


def get_momentum_graph_pixmap(overlays):
    """
    Draw multiple momentum overlays on the same graph.

    Each overlay must provide:
        - seconds (np.ndarray)
        - momenta (np.ndarray)

    Optional:
        - periods (list[dict])
        - label (str)
        - color (str)
        - period_color (str)
    """

    fig = Figure(figsize=(10, 5), dpi=300)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    for overlay in overlays:
        seconds = overlay["seconds"]
        momenta = overlay["momenta"]
        periods = overlay.get("periods")

        # --- Align frames ---
        seconds, aligned_periods = realign_periods(
            seconds,
            periods,
        )

        # --- Draw momentum curve (with dots preserved) ---
        ax.plot(
            seconds,
            momenta,
            marker="o",  # restores the dots
            markersize=5,
            linewidth=1.5,
            label=overlay.get("label", "Momentum"),
            color=overlay.get("color", "tab:green"),
        )
        # --- Draw period markers ---
        draw_period_markers(
            ax,
            aligned_periods,
            color=overlay.get("period_color", overlay.get("color", "tab:red")),
        )

    # --- Shared styling ---
    ax.axhline(0, linestyle="--", linewidth=1, color="gray", zorder=0)

    ax.set_title("Momentum Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Momentum (+ left / - right)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    # Render canvas
    canvas.draw()

    # Convert to QPixmap safely (ARGB path)
    width, height = fig.canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = QImage(
        buf,
        width,
        height,
        QImage.Format_RGBA8888,
    )
    return QPixmap.fromImage(img.copy())


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication, QWidget

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = GenerateMomentumGraphWidget(match_context)
        match_context.set_file("matches_data/sabre_2.mp4")
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
