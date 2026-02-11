from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.gui.momentum_graph.generate_momentum_graph_widget import (
    get_momentum_graph_pixmap,
)
from src.gui.navbar.navigation_controller import NavigationController, View
from src.gui.util.conversion import pixmap_to_np
from src.model.FileManager import FileRole
from src.pyside.MatchContext import MatchContext
from src.pyside.PysideUi import PysideUi


def navigation(nav: NavigationController, match_ctx: MatchContext):
    widget = MultiMomentumGraphWidget()
    nav.register(
        view=View.COMPARE_MOMENTUM,
        title="Multi Momentum Graph",
        widget=widget,
        parent=View.HOME,
    )


class MultiMomentumGraphWidget(QWidget):
    """
    Standalone widget that renders multiple momentum graphs
    onto a single pixmap.

    Usage:
        widget.set_matches([match_ctx1, match_ctx2])
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.matches: List[MatchContext] = []
        self.fps: float | None = None

        self.display_label = QLabel(self)
        self.display_label.setMinimumSize(1, 1)
        self.display_label.setStyleSheet("background: black;")
        # self.display_label.setScaledContents(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.display_label)

        # Use PysideUi wrapper for scaling/rendering
        self.ui = PysideUi(
            video_label=self.display_label,
            text_label=None,
            parent=self,
        )

    # ============================================================
    # Public API
    # ============================================================

    def set_matches(self, matches: List[MatchContext]):
        """
        Provide match contexts to render.
        """
        self.matches = matches
        self.render()

    # ============================================================
    # Core rendering
    # ============================================================

    def render(self):
        if not self.matches:
            return

        overlays = []

        for idx, match in enumerate(self.matches):
            if not match.file_manager:
                continue

            fps = self._get_fps(match)
            if fps is None:
                continue

            momentum_df = self._load_momentum(match)
            periods = self._load_periods(match)

            if momentum_df is None or periods is None:  # TODO: Move this to validation
                raise ValueError(
                    f"Failed to load momentum data or periods for match {idx + 1}."
                )

            overlays.append(
                {
                    "seconds": momentum_df["frame_id"].to_numpy() / fps,
                    "momenta": momentum_df["momentum"].to_numpy(),
                    "label": match.match_name,
                    "periods": periods,
                    "fps": fps,
                    "color": f"C{idx}",  # cycle through colors
                }
            )

        if not overlays:
            return

        pixmap = get_momentum_graph_pixmap(
            overlays=overlays,
        )
        self.ui.set_fresh_frame(pixmap_to_np(pixmap))

    # ============================================================
    # Helpers
    # ============================================================

    def _get_fps(self, match: MatchContext) -> float | None:
        video_path = match.file_manager.get_original_video()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def _load_momentum(self, match: MatchContext) -> pd.DataFrame | None:
        try:
            path = match.file_manager.get_path(FileRole.MOMENTUM_DATA)
            return pd.read_csv(path)
        except Exception:
            return None

    def _load_periods(self, match: MatchContext) -> list[dict]:
        try:
            path = match.file_manager.get_path(FileRole.PERIODS)
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.ui and self.ui.video_renderer:
            self.ui.video_renderer._redraw()


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Example usage with dummy match contexts
    match1 = MatchContext()
    match1.set_file("matches_data/sabre_2.mp4")

    match2 = MatchContext()
    match2.set_file("matches_data/epee_4.mp4")

    widget = MultiMomentumGraphWidget()

    widget.set_matches([match1, match2])
    widget.show()

    sys.exit(app.exec())
