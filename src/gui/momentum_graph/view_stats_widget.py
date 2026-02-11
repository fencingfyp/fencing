import io
import json
import os
from typing import override

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtGui import QPixmap

from src.gui.util.conversion import pixmap_to_np
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.model.FileManager import FileRole
from src.pyside.MatchContext import MatchContext

from .base_task_widget import BaseTaskWidget
from .generate_momentum_graph_widget import get_momentum_graph_pixmap


def dataframe_to_pixmap(df, dpi=100) -> QPixmap:
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.read(), "PNG")
    return pixmap


def load_momentum_df(path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("Momentum data file not found.")
    return pd.read_csv(path)


def load_periods(path) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError("Periods file not found.")
    with open(path, "r") as f:
        return json.load(f)


def compute_time_between_points(
    df: pd.DataFrame, start_frame: int, fps: float
) -> pd.DataFrame:
    df = df.copy()
    df["adjusted_frame_id"] = df["frame_id"] - start_frame
    df.loc[0, "adjusted_frame_id"] = 0
    df["delta_time_sec"] = (df["adjusted_frame_id"].diff() / fps).round(2)
    df = df.iloc[1:].reset_index(drop=True)[["momentum", "delta_time_sec"]]
    return df.rename(
        columns={"momentum": "Momentum", "delta_time_sec": "Delta Time (sec)"}
    )


def compute_average_delta_time(df: pd.DataFrame) -> float:
    return float(df["Delta Time (sec)"].mean())


def render_momentum_graph_with_periods(
    momentum_df: pd.DataFrame, fps: float, periods: list[dict]
) -> np.ndarray:
    pixmap = get_momentum_graph_pixmap(
        overlays=[
            {
                "seconds": momentum_df["frame_id"].to_numpy() / fps,
                "momenta": momentum_df["momentum"].to_numpy(),
                "label": "Momentum",
                "periods": periods,
                "color": "tab:green",
                "fps": fps,
            }
        ],
    )
    return pixmap_to_np(pixmap)


class ViewStatsWidget(BaseTaskWidget):
    def __init__(self, match_context, parent=None):
        super().__init__(match_context, parent)
        self.fps = None

    @override
    def setup(self):
        self.ui.write("Press 'Run' to view the momentum graph statistics.")
        self.run_task()

    @override
    def on_runButton_clicked(self):
        self.run_task()

    def get_fps(self):
        if not self.match_context.file_manager:
            return None

        cap = cv2.VideoCapture(self.match_context.file_manager.get_original_video())
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def run_task(self):
        if not self.match_context.file_manager:
            return

        self.fps = self.get_fps()
        if self.fps is None:
            self.ui.write("Failed to retrieve FPS from video.")
            return

        self.run_started.emit(MomentumGraphTasksToIds.VIEW_STATS)
        self.is_running = True

        try:
            momentum_df = load_momentum_df(
                self.match_context.file_manager.get_path(FileRole.MOMENTUM_DATA)
            )
            periods = load_periods(
                self.match_context.file_manager.get_path(FileRole.PERIODS)
            )
        except FileNotFoundError as e:
            self.ui.write(str(e))
            self.is_running = False
            return

        # Show momentum graph with period overlays
        graph_np = render_momentum_graph_with_periods(momentum_df, self.fps, periods)
        self.ui.set_fresh_frame(graph_np)

        # Compute timing table
        start_frame = int((periods[0]["start_ms"] / 1000) * self.fps)
        timing_df = compute_time_between_points(momentum_df, start_frame, self.fps)
        avg_time = compute_average_delta_time(timing_df)
        self.ui.write(
            f"Average time between momentum data points: {avg_time:.2f} seconds"
        )

        # Show table as image
        pixmap = dataframe_to_pixmap(timing_df)
        self.ui.show_additional("data", pixmap_to_np(pixmap))

        # Save to file
        output_path = self.match_context.file_manager.get_path(FileRole.MOMENTUM_GRAPH)
        pixmap.save(str(output_path))
        self.is_running = False
        self.run_completed.emit(MomentumGraphTasksToIds.VIEW_STATS)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    match_context = MatchContext()
    widget = ViewStatsWidget(match_context)
    match_context.set_file("matches_data/sabre_2.mp4")
    widget.show()
    sys.exit(app.exec())
