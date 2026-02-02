import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QLabel

from src.gui.util.conversion import np_to_pixmap


class NPointPicker(QObject):
    """
    Click → confirm pattern for picking multiple points on a QLabel.
    """

    def __init__(
        self,
        video_label: QLabel,
        text_label: QLabel,
        frame: np.ndarray,
        prompts: list[str],
        on_done,
        w_shortcut: QShortcut | None = None,
    ):
        super().__init__(video_label)

        self.video_label = video_label
        self.text_label = text_label
        self.frame = frame
        self.prompts = prompts
        self.on_done = on_done

        self.current_idx = 0
        self.picked_points: list[tuple[float, float]] = []
        self.current_click: tuple[float, float] | None = None

        # save original mouse handler
        self._orig_mouse_handler = video_label.mousePressEvent
        video_label.setMouseTracking(True)
        video_label.mousePressEvent = self._on_mouse_press

        # setup confirm shortcut (W key)
        if w_shortcut is not None:
            self._shortcut = w_shortcut
        else:
            self._shortcut = QShortcut(QKeySequence(Qt.Key.Key_W), video_label)
        self._shortcut.activated.connect(self.confirm_point)

        # show first prompt
        self._update_prompt()

    # ----------------------- mouse -----------------------
    def _on_mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return

        # map click to frame coordinates
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        label_w, label_h = self.video_label.width(), self.video_label.height()
        sx, sy = fw / label_w, fh / label_h

        x = int(event.position().x() * sx)
        y = int(event.position().y() * sy)

        self.current_click = (x, y)
        self._redraw_preview()

    # ----------------------- confirm -----------------------
    def confirm_point(self):
        if self.current_click is None:
            return

        self.picked_points.append(self.current_click)
        self.current_click = None
        self.current_idx += 1

        if self.current_idx >= len(self.prompts):
            self.finish()
        else:
            self._update_prompt()
            self._redraw_preview()

    # ----------------------- helpers -----------------------
    def _update_prompt(self):
        self.text_label.setText(self.prompts[self.current_idx])

    def _redraw_preview(self):
        # Convert frame → QPixmap
        pixmap = np_to_pixmap(self.frame)

        # Scale to label size (keep aspect ratio)
        pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        painter = QPainter(pixmap)
        pen = QPen(Qt.red)
        pen.setWidth(5)
        painter.setPen(pen)

        pw, ph = pixmap.width(), pixmap.height()
        fh, fw = self.frame.shape[:2]

        # draw confirmed points
        for nx, ny in self.picked_points:
            x = int(nx * pw / fw)
            y = int(ny * ph / fh)
            painter.drawPoint(x, y)

        # draw current candidate
        if self.current_click:
            x, y = self.current_click
            x = int(x * pw / fw)
            y = int(y * ph / fh)
            painter.drawPoint(x, y)

        painter.end()
        self.video_label.setPixmap(pixmap)

    def finish(self):
        self.cancel()
        self.on_done(self.picked_points)

    def cancel(self):
        self.video_label.mousePressEvent = self._orig_mouse_handler
        self._shortcut.deleteLater()

        self.text_label.setText("")
        self._shortcut.activated.disconnect(self.confirm_point)
