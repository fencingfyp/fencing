# src/pyside/video_renderer.py
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from src.model.drawable import Drawable
from src.model.drawable.box_drawable import BoxDrawable
from src.model.drawable.detections_drawable import DetectionsDrawable
from src.model.drawable.points_drawable import PointsDrawable
from src.model.drawable.quadrilateral_drawable import QuadrilateralDrawable


class VideoRenderer:
    def __init__(self, video_label: QLabel):
        self.video_label = video_label
        self._frame = None
        self._drawables: list[Drawable] = []
        self._sx = 1.0
        self._sy = 1.0

    # ---------- Public API ----------

    def set_frame(self, frame: np.ndarray) -> QPixmap:
        """Set the current frame and redraw."""
        self._frame = frame
        self._redraw()
        return self.video_label.pixmap()

    def render_points(self, points: list[tuple[float, float]], color=(0, 255, 0)):
        self.render([PointsDrawable(points, color=color)])

    def render_quadrilateral(self, quad, color=(0, 255, 0)):
        self.render([QuadrilateralDrawable(quad, color=color)])

    def render_detections(self, detections: dict, highlight_id=None):
        self.render([DetectionsDrawable(detections, highlight_id)])

    def render(self, drawables: list[Drawable]):
        self._drawables = drawables
        self._redraw()

    def get_current_frame(self) -> np.ndarray | None:
        return self._frame.copy() if self._frame is not None else None

    # ---------- Drawing ----------

    def _draw_drawable(self, painter: QPainter, drawable: "Drawable"):
        sub_drawables = drawable.get_sub_drawables()
        if sub_drawables:
            for sub in sub_drawables:
                self._draw_drawable(painter, sub)
            return

        style = drawable.style()
        for primitive_type, data in drawable.primitives():
            self._render_primitive(painter, primitive_type, data, style)

    def _render_primitive(
        self, painter: QPainter, primitive_type: str, data, style: dict
    ):
        if primitive_type == "points":
            painter.setPen(QPen(QColor(*style["color"]), style.get("size", 2)))
            for x, y in data:
                painter.drawPoint(int(x), int(y))
        elif primitive_type in ("lines", "polygon"):
            painter.setPen(QPen(QColor(*style["color"]), style.get("thickness", 2)))
            for i in range(len(data)):
                x1, y1 = data[i]
                x2, y2 = data[(i + 1) % len(data)]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        elif primitive_type == "box":
            x1, y1 = data[0], data[1]
            x2, y2 = data[2], data[3]
            pen = QPen(QColor(*style["color"]))
            pen.setWidth(style.get("thickness", 2))
            painter.setPen(pen)
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        elif primitive_type == "text":
            (x, y), text = data[0], data[1]
            painter.setPen(QColor(*style["color"]))
            font = QFont()
            font.setPointSize(style.get("font_size", 12))
            painter.setFont(font)
            painter.drawText(int(x), int(y) - 2, text)

    def _redraw(self):
        if self._frame is None:
            return

        pixmap = self._frame_to_pixmap(self._frame)

        painter = QPainter(pixmap)
        for drawable in self._drawables:
            self._draw_drawable(painter, drawable)
        painter.end()

        # update label
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _frame_to_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
