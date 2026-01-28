# src/pyside/video_renderer.py
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from src.model.drawable import Drawable


class VideoRenderer:
    def __init__(self, video_label: QLabel):
        self.video_label = video_label
        self._frame = None
        self._drawables: list[Drawable] = []

    def set_frame(self, frame):
        self._frame = frame
        self._redraw()
        return self.video_label.pixmap()

    def render(self, drawables: list[Drawable]):
        self._drawables = drawables
        self._redraw()

    def _draw_drawable(self, painter: QPainter, drawable: Drawable):
        if drawable.get_sub_drawables():
            for sub in drawable.get_sub_drawables():
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
            x1, y1, x2, y2 = data
            color = style["color"]
            pen = QPen(QColor(*color))
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
        self.video_label.setPixmap(pixmap)

    def _frame_to_pixmap(self, frame):
        import cv2
        from PySide6.QtGui import QImage, QPixmap

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
