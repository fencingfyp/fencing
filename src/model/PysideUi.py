from abc import ABC, ABCMeta
from typing import override

import cv2
import numpy as np
from PySide6.QtCore import QMetaObject, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QLabel, QWidget

from src.gui.point_picker_dialog import PointPickerDialog
from src.gui.util.conversion import np_to_pixmap, qlabel_to_np

from .OpenCvUi import UiCodes
from .Quadrilateral import Quadrilateral
from .Ui import PipelineUiDriver


class ABCQObjectMeta(type(QObject), type(ABC)):
    """This allows using PySide6 as the implementation for the UI abstraction defined in Ui.py."""

    pass


class PysideUi(QObject, PipelineUiDriver, metaclass=ABCQObjectMeta):
    task_completed = Signal()

    def __init__(self, video_label: QLabel, text_label: QLabel | None, parent: QObject):
        super().__init__(parent)
        self.video_label = video_label
        self.text_label = text_label
        self.parent = parent

        # loop state
        self.cap = None
        self.pipeline = None
        self.writer = None
        self.timer: QTimer | None = None

        # visual state
        self._frame: np.ndarray | None = None
        self._points: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Rendering API (matches OpenCvUi intent)
    # ------------------------------------------------------------------

    def set_fresh_frame(self, frame: np.ndarray) -> QPixmap:
        self._frame = frame
        self._redraw()
        return self.video_label.pixmap()

    def plot_quadrilateral(self, quad: Quadrilateral, color=None):
        self.plot_points(quad.points, color)

    def plot_points(self, pts, color=None):
        # pts expected: [(x, y), ...] or OpenCV-style [[x, y]]
        self._points = [
            (
                (float(p[0]), float(p[1]))
                if len(p) == 2
                else (float(p[0][0]), float(p[0][1]))
            )
            for p in pts
        ]
        self._redraw()

    def show_text(self, text: str):
        if self.text_label:
            self.text_label.setText(text)

    def show_frame(self):
        pass  # Qt repaints automatically

    # ------------------------------------------------------------------
    # Loop control (Qt replaces while True)
    # ------------------------------------------------------------------

    def process_crop_region_loop(self, cap, pipeline, writer=None):
        self.cap = cap
        self.pipeline = pipeline
        self.writer = writer

        self.timer = QTimer(self.parent)
        self.timer.timeout.connect(self._on_frame)
        self.timer.start(0)

    def on_task_completed(self):
        self.close()
        self.text_label.setText("Finished processing video")
        self.task_completed.emit()

    def _on_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.on_task_completed()
            return

        rectified, pts = self.pipeline.process(frame)

        # 1. Render frame first (creates scaled pixmap)
        pixmap = self.set_fresh_frame(frame)

        # 2. Map frame-space points → pixmap-space
        fh, fw = frame.shape[:2]
        pw, ph = pixmap.width(), pixmap.height()

        sx = pw / fw
        sy = ph / fh

        pts = [pts[0] for pts in pts]  # flatten OpenCV-style [[x, y]] → (x, y)
        mapped_pts = [(x * sx, y * sy) for x, y in pts]

        # 3. Draw mapped points
        self.plot_points(mapped_pts)

        self.show_text("Tracking region")

        if self.writer:
            self.writer.write(rectified)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self.timer:
            self.timer.stop()
            self.timer = None
        if self.writer:
            self.writer.release()
            self.writer = None

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

    @override
    def get_n_points(self, frame, prompts: list[str]) -> list[tuple[float, float]]:
        points = []
        arr = qlabel_to_np(self.video_label)
        dlg = PointPickerDialog(prompts, frame)
        dlg.picked_positions.connect(lambda positions: points.extend(positions))
        dlg.exec()
        frame_h, frame_w = frame.shape[:2] if frame is not None else arr.shape[:2]
        # resize points to original frame size
        for i in range(len(points)):
            x, y = points[i]
            points[i] = (
                x * frame_w,
                y * frame_h,
            )
        return points
