import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class RawVideoWidget(QWidget):
    """
    Dumb display widget. Accepts frames and renders them, nothing else.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image_label = QLabel(self)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._image_label.setMinimumSize(1, 1)

        self._current_pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._image_label, 1)
        layout.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------------ internal

    def _render(self):
        if self._current_pixmap is None:
            return
        self._image_label.setPixmap(
            self._current_pixmap.scaled(
                self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # ------------------------------------------------------------------ events

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ------------------------------------------------------------------ public

    def display_frame(self, frame: np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._render()

    def clear(self):
        self._current_pixmap = None
        self._image_label.clear()
