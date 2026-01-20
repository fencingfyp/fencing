import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout


class PointPickerDialog(QDialog):
    picked_positions = Signal(list)  # List of (x, y) tuples

    def __init__(self, instructions: list[str], image: np.ndarray, parent=None):
        super().__init__(parent)
        self.instructions = instructions
        self.positions = [(0, 0)]
        self.current_idx = 0

        self.setWindowTitle(f"Select {len(self.instructions)} Points")
        # set size to 80% of screen size
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.resize(int(screen_size.width() * 0.8), int(screen_size.height() * 0.8))

        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setPixmap(self.create_image_pixmap(image))
        self.image.installEventFilter(self)

        self.original_pixmap = self.image.pixmap()

        # self.image.setStyleSheet("border: 1px solid red;")

        self.instruction_label = QLabel(self)
        self.instruction_label.setText(self.instructions[self.current_idx])
        # self.instruction_label.setStyleSheet("border: 1px solid blue;")
        # self.setStyleSheet("border: 1px solid green;")

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.image)
        self.vboxlayout.addWidget(self.instruction_label)
        self.setLayout(self.vboxlayout)

        self.set_shortcuts()

    def eventFilter(self, obj, event):
        if obj is self.image and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                pos = event.position().toPoint()

                pixmap_rect = self._pixmap_rect()
                if pixmap_rect.contains(pos):
                    self.positions.pop()  # remove last dummy
                    x = pos.x() - pixmap_rect.x()
                    y = pos.y() - pixmap_rect.y()
                    self._store_point(x / pixmap_rect.width(), y / pixmap_rect.height())
                    return True
        return super().eventFilter(obj, event)

    def _pixmap_rect(self) -> QRect:
        pm = self.image.pixmap()
        if not pm:
            return QRect()

        label_size = self.image.size()
        pm_size = pm.size()

        x = (label_size.width() - pm_size.width()) // 2
        y = (label_size.height() - pm_size.height()) // 2
        return QRect(x, y, pm_size.width(), pm_size.height())

    def _store_point(self, x, y):
        self.positions.append((x, y))
        self.redraw_points()

    def set_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_W), self, self.confirm)

    def create_image_pixmap(self, image: np.ndarray) -> QPixmap:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bits_per_line = ch * w
        qimg = QImage(frame.data, w, h, bits_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return scaled

    def redraw_points(self):
        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)

        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)

        w = pixmap.width()
        h = pixmap.height()

        for nx, ny in self.positions:
            x = int(nx * w)
            y = int(ny * h)
            painter.drawPoint(x, y)

        painter.end()
        self.image.setPixmap(pixmap)

    def confirm(self):
        if len(self.instructions) > self.current_idx + 1:
            self.positions.append((0, 0))
            self.current_idx += 1
            self.instruction_label.setText(self.instructions[self.current_idx])
        else:
            self.picked_positions.emit(self.positions)
            self.accept()


if __name__ == "__main__":
    import sys

    import cv2
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    video_path = "matches_data/epee_2/original.mp4"
    # get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    dialog = PointPickerDialog(
        ["Select point 1", "Select point 2", "Select point 3", "Select point 4"], frame
    )
    dialog.show()
    sys.exit(app.exec())
