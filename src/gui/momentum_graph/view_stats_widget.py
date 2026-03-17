import io
import json
import os
from typing import override

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics, QImage, QPainter, QPixmap

from src.gui.base_task_widget.base_task_widget import BaseTaskWidget
from src.gui.MatchContext import MatchContext
from src.gui.task_graph.task_graph import TasksToIds
from src.gui.util.conversion import pixmap_to_np
from src.model.FileManager import FileRole

from .generate_momentum_graph_widget import get_momentum_graph_pixmap


def dataframe_to_pixmap(df, dpi=100) -> QPixmap:
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=0.5, top=0.5, bottom=0)
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


def text_to_pixmap(text: str, fontsize=30) -> QPixmap:
    font = QFont()
    font.setPointSize(fontsize)
    metrics = QFontMetrics(font)
    rect = metrics.boundingRect(text)

    pixmap = QPixmap(rect.width() + 20, rect.height() + 10)
    pixmap.fill(Qt.GlobalColor.white)

    painter = QPainter(pixmap)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return pixmap


def stack_pixmaps_vertically(pixmaps: list[QPixmap]) -> QPixmap:
    """Stack a list of QPixmaps vertically into a single QPixmap."""
    arrays = [pixmap_to_np(p) for p in pixmaps]
    max_width = max(a.shape[1] for a in arrays)

    padded = []
    for a in arrays:
        h, w = a.shape[:2]
        if w < max_width:
            pad_left = (max_width - w) // 2
            pad_right = max_width - w - pad_left
            a = np.pad(
                a,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        padded.append(a)

    combined = np.vstack(padded)
    height, width, channels = combined.shape
    bytes_per_line = channels * width
    fmt = (
        QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
    )
    image = QImage(combined.tobytes(), width, height, bytes_per_line, fmt)
    return QPixmap.fromImage(image)


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
        # self.ui.write("Press 'Run' to view the momentum graph statistics.")
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

        self.run_started.emit(TasksToIds.VIEW_STATS.value)
        self.is_running = True

        try:
            momentum_df = load_momentum_df(
                self.match_context.file_manager.get_path(FileRole.RAW_MOMENTUM_DATA)
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

        # Compute stats
        start_frame = int((periods[0]["start_ms"] / 1000) * self.fps)
        timing_df = compute_time_between_points(momentum_df, start_frame, self.fps)
        avg_time = compute_average_delta_time(timing_df)
        final_score = compute_final_score(momentum_df["momentum"].to_numpy())

        # Build stacked display: score label → table → avg label
        score_pixmap = text_to_pixmap(
            f"Final Score: {final_score[0]} - {final_score[1]}", fontsize=16
        )
        table_pixmap = dataframe_to_pixmap(timing_df)
        avg_pixmap = text_to_pixmap(
            f"Average time between points: {avg_time:.2f} sec", fontsize=14
        )

        combined_pixmap = stack_pixmaps_vertically(
            [score_pixmap, table_pixmap, avg_pixmap]
        )
        self.ui.show_additional("data", pixmap_to_np(combined_pixmap))

        # Save to file
        csv_output_path = self.match_context.file_manager.get_path(
            FileRole.PROCESSED_MOMENTUM_DATA
        )
        timing_df.to_csv(csv_output_path, index=False)
        image_output_path = self.match_context.file_manager.get_path(
            FileRole.MOMENTUM_GRAPH
        )
        combined_pixmap.save(str(image_output_path))
        self.is_running = False
        self.run_completed.emit(TasksToIds.VIEW_STATS.value)


def compute_final_score(momentum_values: np.ndarray) -> tuple[int, int]:
    """Given a graph of momentum values, compute the final score.
    The score is determined by counting how many times the momentum increases (left) vs decreases (right).
    If momentum change is 0, both fencers scored."""
    if len(momentum_values) == 0:
        return 0, 0
    score_changes = np.diff(momentum_values)
    left = np.count_nonzero(score_changes >= 0)
    right = np.count_nonzero(score_changes <= 0)

    return left, right


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    match_context = MatchContext()
    widget = ViewStatsWidget(match_context)
    match_context.set_file("matches_data/bar.mp4")
    widget.show()
    sys.exit(app.exec())
    sys.exit(app.exec())
