from typing import override

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPen, QPixmap, QShortcut, Qt

from src.gui.raw_video_widget import RawVideoWidget


class DrawableVideoWidget(RawVideoWidget):
    key_pressed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points: list[tuple[float, float]] = []
        self.overlay = QPixmap(self.size())
        self.draw_dots = False
        # self.is_paused = False

    def setup_shortcuts(self):
        # QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_W), self, lambda: self.key_pressed.emit("w"))

    # def play_pause(self):
    #     self.is_paused = not self.is_paused

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.draw_dots:
            print("Clicked at:", event.position().toPoint())
            self.points.append(
                (
                    event.position().x() / self.image_label.width(),
                    event.position().y() / self.image_label.height(),
                )
            )
            self.repeat_frame()

    def set_draw_dots(self, draw: bool):
        if not draw:
            self.points = []
            self.draw_dots = False
            self.repeat_frame()
        if draw:
            self.draw_dots = True

    def draw_dots(self, points: list[tuple[float, float]]):
        self.points = points
        self.repeat_frame()

    def draw_on_overlay(self):
        self.overlay.fill(Qt.transparent)
        painter = QPainter(self.overlay)
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(5)
        painter.setPen(pen)
        for point in self.points:
            point = QPoint(
                int(point[0] * self.image_label.width()),
                int(point[1] * self.image_label.height()),
            )
            painter.drawPoint(point)
        painter.end()

    @override
    def next_frame(self) -> bool:
        base = self.get_next_pixmap()
        if base is None:
            return False

        # ensure overlay matches
        if self.overlay is None or self.overlay.size() != base.size():
            self.overlay = QPixmap(base.size())

        # draw points onto overlay
        self.draw_on_overlay()

        # compose
        composed = QPixmap(base)
        p = QPainter(composed)
        p.drawPixmap(0, 0, self.overlay)
        p.end()

        self.setPixmap(composed)
        return True


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = DrawableVideoWidget()
    widget.set_video_source("matches_data/epee_2/original.mp4")
    widget.show()
    sys.exit(app.exec())
