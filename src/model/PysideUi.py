from abc import ABC
from typing import override

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.gui.inline_point_picker import InlinePointPicker
from src.gui.util.conversion import np_to_pixmap, qlabel_to_np

from .Quadrilateral import Quadrilateral
from .Ui import Ui


class ABCQObjectMeta(type(QObject), type(ABC)):
    """This allows using PySide6 as the implementation for the UI abstraction defined in Ui.py."""

    pass


class PysideUi(QObject, Ui, metaclass=ABCQObjectMeta):
    task_completed = Signal()
    quit_requested = Signal()

    def __init__(self, video_label: QLabel, text_label: QLabel | None, parent: QObject):
        super().__init__(parent)
        self.video_label = video_label
        self.text_label = text_label
        self.parent = parent

        self._frame: np.ndarray | None = None
        self._points: list[tuple[float, float]] = []

        # quit shortcut
        self._quit_shortcut = QShortcut(QKeySequence("Q"), parent)
        self._quit_shortcut.activated.connect(self.quit_requested.emit)

        self._additional_windows: dict[int | str, QWidget] = {}

    # ------------------------------------------------------------------
    # Rendering API (matches OpenCvUi intent)
    # ------------------------------------------------------------------

    def write(self, text: str):
        if self.text_label:
            self.text_label.setText(text)

    def set_fresh_frame(self, frame: np.ndarray) -> QPixmap:
        self._frame = frame
        self._redraw()
        return self.video_label.pixmap()

    def plot_points(self, pts, color=None):
        """
        pts: list of (x, y) in frame coordinates
        Scales points to match the displayed pixmap size.
        """
        if self._frame is None:
            return

        frame_h, frame_w = self._frame.shape[:2]
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return

        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()

        # Compute aspect-ratio fit offsets
        scale_w = pixmap_w / frame_w
        scale_h = pixmap_h / frame_h
        scale = min(scale_w, scale_h)

        offset_x = (pixmap_w - frame_w * scale) / 2
        offset_y = (pixmap_h - frame_h * scale) / 2

        # Scale points from frame → pixmap coordinates
        scaled_points = []
        for p in pts:
            if len(p) == 2:
                x, y = p
            else:
                x, y = p[0]

            scaled_x = x * scale + offset_x
            scaled_y = y * scale + offset_y
            scaled_points.append((scaled_x, scaled_y))
        self._points = scaled_points

        self._redraw()

    def show_text(self, text: str):
        if self.text_label:
            self.text_label.setText(text)

    def show_frame(self):
        pass

    def show_additional(self, key: int | str, frame: np.ndarray):
        """
        Show the given frame in a separate top-level window.
        Each index corresponds to a separate window.
        """

        pixmap = np_to_pixmap(frame)

        # Create window if it doesn't exist
        if key not in self._additional_windows:
            win = QWidget(None)  # top-level window
            win.setWindowTitle(f"Additional View {key}")

            # Set a reasonable size (or match frame)
            win.resize(frame.shape[1], frame.shape[0])

            label = QLabel(win)
            label.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout(win)
            layout.addWidget(label)
            win.setLayout(layout)

            self._additional_windows[key] = win
            win.show()
        else:
            win = self._additional_windows[key]
            label = win.findChild(QLabel)

        # Scale pixmap to label size while preserving aspect ratio
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        label.update()

    def take_user_input(self):
        pass  # todo: implement

    def get_n_points_async(self, frame, prompts: list[str], callback):
        if frame is None:
            return

        def on_done(points):
            callback(points)

        InlinePointPicker(
            video_label=self.video_label,
            text_label=self.text_label,
            frame=frame,
            prompts=prompts,
            on_done=on_done,
        )

    def draw_quadrilateral(self, quad: Quadrilateral, color=(0, 255, 0)):
        """
        Draws a quadrilateral on the current pixmap.

        positions: list of 4 (x, y) tuples in frame coordinates
        color: RGB tuple
        """
        if self._frame is None:
            return

        # Convert frame coordinates to pixmap coordinates
        pixmap = self.video_label.pixmap().copy()
        pw, ph = pixmap.width(), pixmap.height()
        fh, fw = self._frame.shape[:2]

        sx, sy = pw / fw, ph / fh
        mapped_pts = [(x * sx, y * sy) for x, y in quad.numpy()]

        painter = QPainter(pixmap)
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        painter.setPen(pen)

        # Draw quadrilateral lines
        for i in range(4):
            x1, y1 = mapped_pts[i]
            x2, y2 = mapped_pts[(i + 1) % 4]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.end()
        self.video_label.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        for win in self._additional_windows.values():
            win.close()
        self._additional_windows.clear()

    def show_single_frame(self, cap):
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if ret:
            self.set_fresh_frame(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _redraw(self):
        if self._frame is None:
            return

        pixmap = self._frame_to_pixmap(self._frame)

        if self._points:
            pixmap = self._draw_points(pixmap, self._points)

        self.video_label.setPixmap(pixmap)

    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def _draw_points(
        self, pixmap: QPixmap, points: list[tuple[float, float]]
    ) -> QPixmap:
        painter = QPainter(pixmap)
        pen = QPen(Qt.green)
        pen.setWidth(2)
        painter.setPen(pen)

        for x, y in points:
            painter.drawPoint(int(x), int(y))

        painter.end()
        return pixmap
