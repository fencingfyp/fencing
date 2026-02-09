# src/pyside/video_renderer.py
import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from src.model.drawable.box_drawable import BoxDrawable
from src.model.drawable.detections_drawable import DetectionsDrawable
from src.model.drawable.drawable import Drawable
from src.model.drawable.points_drawable import PointsDrawable
from src.model.drawable.quadrilateral_drawable import QuadrilateralDrawable


class VideoRenderer(QObject):
    mouse_clicked = Signal(float, float)  # emits frame coordinates

    def __init__(self, video_label: QLabel, parent=None):
        super().__init__(parent)
        self.video_label = video_label
        self.video_label.setAlignment(Qt.AlignCenter)

        self._frame: np.ndarray | None = None
        self._drawables: list[Drawable] = []

        # Scaling factors for mapping primitives and mouse clicks
        self._draw_scale = 1.0  # scale applied when drawing on scaled frame
        self._display_scale = 1.0  # final scale from original frame → label
        self._offset_x = 0
        self._offset_y = 0

        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._on_mouse_press

    def _on_mouse_press(self, event):
        if self._frame is None:
            return

        fx, fy = self.map_label_coords_to_frame(
            event.position().x(), event.position().y()
        )
        self.mouse_clicked.emit(fx, fy)  # emit frame coordinates

    # ---------- Public API ----------

    def set_frame(self, frame: np.ndarray) -> QPixmap:
        """Set the current frame and redraw."""
        self._frame = frame
        self._drawables.clear()
        self._redraw()
        return self.video_label.pixmap()

    def render_points(
        self, points: list[tuple[float, float]], color=(0, 255, 0), size=5
    ):
        self.render([PointsDrawable(points, color=color, size=size)])

    def render_quadrilateral(self, quad, color=(0, 255, 0)):
        self.render([QuadrilateralDrawable(quad, color=color)])

    def render_detections(self, detections: dict, highlight_id=None):
        self.render([DetectionsDrawable(detections, highlight_id)])

    def render(self, drawables: list[Drawable]):
        """Set the drawables to render on the current frame."""
        self._drawables = drawables
        self._redraw()

    def get_current_frame(self) -> np.ndarray | None:
        return self._frame.copy() if self._frame is not None else None

    def map_label_coords_to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Convert a QLabel click (x, y) to coordinates on the original frame."""
        if self._frame is None:
            return 0, 0

        # remove offsets from label letterboxing
        rel_x = x - self._offset_x
        rel_y = y - self._offset_y

        # clamp to displayed frame
        frame_w, frame_h = self._frame.shape[1], self._frame.shape[0]
        rel_x = max(0, min(rel_x, frame_w * self._display_scale))
        rel_y = max(0, min(rel_y, frame_h * self._display_scale))

        # map back to original frame coordinates
        fx = rel_x / self._display_scale
        fy = rel_y / self._display_scale
        return fx, fy

    # ---------- Drawing ----------

    def _draw_drawable_scaled(self, painter: QPainter, drawable: Drawable):
        """Draw a drawable on the frame pixmap using the draw scale."""
        for sub in drawable.get_sub_drawables():
            self._draw_drawable_scaled(painter, sub)
            return

        style = drawable.style()
        for primitive_type, data in drawable.primitives():
            if primitive_type == "points":
                radius = style.get("size", 2) // 2
                painter.setBrush(QColor(*style["color"]))
                painter.setPen(Qt.NoPen)
                for x, y in data:
                    cx = int(x * self._draw_scale)
                    cy = int(y * self._draw_scale)
                    painter.drawEllipse(
                        cx - radius, cy - radius, radius * 2, radius * 2
                    )

            elif primitive_type in ("lines", "polygon"):
                painter.setPen(QPen(QColor(*style["color"]), style.get("thickness", 2)))
                scaled = [(x * self._draw_scale, y * self._draw_scale) for x, y in data]
                for i in range(len(scaled)):
                    x1, y1 = scaled[i]
                    x2, y2 = scaled[(i + 1) % len(scaled)]
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            elif primitive_type == "box":
                x1, y1, x2, y2 = data
                pen = QPen(QColor(*style["color"]))
                pen.setWidth(style.get("thickness", 2))
                painter.setPen(pen)
                painter.drawRect(
                    int(x1 * self._draw_scale),
                    int(y1 * self._draw_scale),
                    int((x2 - x1) * self._draw_scale),
                    int((y2 - y1) * self._draw_scale),
                )

            elif primitive_type == "text":
                (x, y), text = data
                painter.setPen(QColor(*style["color"]))
                font = QFont()
                font.setPointSize(style.get("font_size", 12))
                painter.setFont(font)
                painter.drawText(
                    int(x * self._draw_scale), int(y * self._draw_scale) - 2, text
                )

    def _redraw(self):
        if self._frame is None:
            return

        frame_h, frame_w = self._frame.shape[:2]
        label_w, label_h = self.video_label.width(), self.video_label.height()

        # 1. Scale frame to fit label while keeping aspect ratio
        scale_w = label_w / frame_w
        scale_h = label_h / frame_h
        self._draw_scale = min(scale_w, scale_h)
        disp_w = int(frame_w * self._draw_scale)
        disp_h = int(frame_h * self._draw_scale)

        # 2. Convert frame to pixmap at scaled size
        scaled_frame = cv2.resize(
            self._frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
        )
        pixmap = self._frame_to_pixmap(scaled_frame)

        # 3. Draw drawables on scaled pixmap
        painter = QPainter(pixmap)
        for drawable in self._drawables:
            self._draw_drawable_scaled(painter, drawable)
        painter.end()

        # 4. Scale pixmap to label size with KeepAspectRatio (Qt handles letterboxing)
        final_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # 5. Compute offsets and final display scale for mouse mapping
        self._display_scale = final_pixmap.width() / frame_w
        self._offset_x = (label_w - final_pixmap.width()) // 2
        self._offset_y = (label_h - final_pixmap.height()) // 2

        # 6. Set pixmap in label
        self.video_label.setPixmap(final_pixmap)

    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """Convert a BGR frame (or grayscale) to QPixmap."""
        if len(frame.shape) == 2:  # grayscale
            qimg = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.strides[0],
                QImage.Format_Grayscale8,
            )
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            )
        return QPixmap.fromImage(qimg)
