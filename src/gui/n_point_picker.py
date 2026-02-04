import numpy as np
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QLabel

from src.gui.util.conversion import np_to_pixmap


class NPointPicker(QObject):
    """
    Click → confirm pattern for picking multiple points on a QLabel.
    Short-lived: activate when picking, deactivate when done.
    """

    def __init__(
        self,
        video_label: QLabel,
        text_label: QLabel,
        frame: np.ndarray,
        prompts: list[str],
        on_done,
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

        self._shortcut: QShortcut | None = None
        self._orig_mouse_handler = video_label.mousePressEvent
        self._active = False
        self.activate()

    # ----------------------- lifecycle -----------------------
    def activate(self):
        if self._active:
            return
        self._active = True

        # replace mouse handler
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._on_mouse_press

        # setup confirm shortcut (W key)
        self._shortcut = QShortcut(QKeySequence(Qt.Key.Key_W), self.video_label)
        self._shortcut.activated.connect(self.confirm_point)

        self.current_idx = 0
        self.picked_points.clear()
        self.current_click = None
        self._update_prompt()
        self._redraw_preview()

    def deactivate(self):
        if not self._active:
            return
        self._active = False

        # restore mouse handler
        self.video_label.mousePressEvent = self._orig_mouse_handler

        # remove shortcut
        if self._shortcut:
            self._shortcut.activated.disconnect(self.confirm_point)
            self._shortcut.setParent(None)
            self._shortcut = None

        self.text_label.setText("")
        self.video_label.setPixmap(np_to_pixmap(self.frame))  # reset to original frame

    # ----------------------- mouse -----------------------
    def _on_mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return

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
        pixmap = np_to_pixmap(self.frame).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        painter = QPainter(pixmap)
        pen = QPen(Qt.red)
        pen.setWidth(5)
        painter.setPen(pen)

        fh, fw = self.frame.shape[:2]
        label_w, label_h = self.video_label.width(), self.video_label.height()
        ratio = min(label_w / fw, label_h / fh)
        x_offset = (label_w - fw * ratio) / 2
        y_offset = (label_h - fh * ratio) / 2

        for nx, ny in self.picked_points:
            x = int(nx * ratio + x_offset)
            y = int(ny * ratio + y_offset)
            painter.drawPoint(x, y)

        if self.current_click:
            x, y = self.current_click
            x = int(x * ratio + x_offset)
            y = int(y * ratio + y_offset)
            painter.drawPoint(x, y)

        painter.end()
        self.video_label.setPixmap(pixmap)

    def finish(self):
        self.deactivate()
        self.on_done(self.picked_points)
