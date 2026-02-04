import io
import json
import os
from typing import override

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QLabel, QPushButton, QTableView, QVBoxLayout

from src.gui.util.conversion import pixmap_to_np
from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.util.file_names import (
    MOMENTUM_DATA_CSV_NAME,
    MOMENTUM_GRAPH_IMAGE_NAME,
    ORIGINAL_VIDEO_NAME,
    PERIODS_JSON_NAME,
)

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


def load_momentum_df(working_dir: str) -> pd.DataFrame:
    path = os.path.join(working_dir, MOMENTUM_DATA_CSV_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError("Momentum data file not found.")
    return pd.read_csv(path)


def load_periods(working_dir: str) -> list[dict]:
    path = os.path.join(working_dir, PERIODS_JSON_NAME)
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
    # Base graph
    pixmap = get_momentum_graph_pixmap(
        momentum_df["frame_id"].to_numpy(), momentum_df["momentum"].to_numpy(), fps
    )
    graph_np = pixmap_to_np(pixmap)

    # Overlay periods
    overlay = graph_np.copy()
    h, w, _ = overlay.shape

    max_frame = momentum_df["frame_id"].max()
    for i, p in enumerate(periods):
        start_frac = p["start_ms"] / 1000 * fps / max_frame
        end_frac = p["end_ms"] / 1000 * fps / max_frame
        x_start = int(start_frac * w)
        x_end = int(end_frac * w)

        # Draw semi-transparent rectangle
        cv2.rectangle(
            overlay,
            (x_start, 0),
            (x_end, h),
            color=(0, 0, 255),
            thickness=-1,
        )

        # Label period
        cv2.putText(
            overlay,
            f"Period {i+1}",
            (x_start + 5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    # Blend with original graph
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, graph_np, 1 - alpha, 0)
    return blended


class ViewStatsWidget(BaseTaskWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps = None

    @override
    def setup(self):
        self.interactive_ui.write("Press 'Run' to view the momentum graph statistics.")
        cap = cv2.VideoCapture(os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    @override
    def on_runButton_clicked(self):
        if not self.working_dir:
            return

        self.run_started.emit(MomentumGraphTasksToIds.VIEW_STATS)
        self.is_running = True

        try:
            momentum_df = load_momentum_df(self.working_dir)
            periods = load_periods(self.working_dir)
        except FileNotFoundError as e:
            self.interactive_ui.write(str(e))
            self.is_running = False
            return

        # Show momentum graph with period overlays
        graph_np = render_momentum_graph_with_periods(momentum_df, self.fps, periods)
        self.interactive_ui.set_fresh_frame(graph_np)

        # Compute timing table
        start_frame = int((periods[0]["start_ms"] / 1000) * self.fps)
        timing_df = compute_time_between_points(momentum_df, start_frame, self.fps)
        avg_time = compute_average_delta_time(timing_df)
        self.interactive_ui.write(
            f"Average time between momentum data points: {avg_time:.2f} seconds"
        )

        # Show table as image
        pixmap = dataframe_to_pixmap(timing_df)
        self.interactive_ui.show_additional("data", pixmap_to_np(pixmap))

        # Save to file
        output_path = os.path.join(self.working_dir, MOMENTUM_GRAPH_IMAGE_NAME)
        pixmap.save(output_path)
        self.is_running = False


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = ViewStatsWidget()
    widget.set_working_directory("matches_data/sabre_2")
    widget.show()
    sys.exit(app.exec())
