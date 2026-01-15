"""
This class is built with heavy reference from:
https://github.com/BBC-Esq/Pyside6_PyQt6_video_audio_player/blob/main/media_player_pyside6.py
"""

import cv2
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            value = (
                QSlider.minimum(self)
                + (
                    (QSlider.maximum(self) - QSlider.minimum(self))
                    * event.position().x()
                )
                / self.width()
            )
            self.setValue(int(value))
            self.sliderPressed.emit()
            self.sliderMoved.emit(int(value))
            self.sliderReleased.emit()
        super().mousePressEvent(event)


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.is_paused = False
        self.is_dragging = False

        # QLabel to display frames
        self.video_frame = QLabel(self)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.delay = None

        self.positionslider = ClickableSlider(Qt.Orientation.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.sliderMoved.connect(self.set_position)
        self.positionslider.sliderPressed.connect(self.slider_pressed)
        self.positionslider.sliderReleased.connect(self.slider_released)

        self.timelabel = QLabel("00:00 / 00:00")
        self.timelabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hbuttonbox = QHBoxLayout()

        self.playbutton = QPushButton("Pause")
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(self.play_pause)

        self.stopbutton = QPushButton("Stop")
        self.hbuttonbox.addWidget(self.stopbutton)
        self.stopbutton.clicked.connect(self.stop)

        self.hbuttonbox.addStretch(1)

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.video_frame, 1)
        self.vboxlayout.addWidget(self.positionslider)
        self.vboxlayout.addWidget(self.timelabel)
        self.vboxlayout.addLayout(self.hbuttonbox)

        self.setLayout(self.vboxlayout)

        self.setup_shortcuts()

    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.skip(-5000))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.skip(5000))

    def play_pause(self):
        if self.cap is None:
            return
        if self.is_paused:
            self.playbutton.setText("Pause")
            self.is_paused = False
            self.timer.start(self.delay)
        else:
            self.playbutton.setText("Play")
            self.is_paused = True
            self.timer.stop()

    def slider_pressed(self):
        self.is_dragging = True
        self.timer.stop()

    def slider_released(self):
        self.is_dragging = False
        self.set_position()
        if not self.is_paused:
            self.timer.start(self.delay)

    def set_position(self):
        pos = self.positionslider.value()
        if self.cap:
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_number = int((pos / 1000) * total_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.next_frame()

    def set_video_source(self, video_path: str, geometry: QRect = None):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.delay = (
            self.cap.get(cv2.CAP_PROP_FPS)
            and int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
            or 30
        )
        self.timer.start(self.delay)

    def closeEvent(self, event):
        self.cleanup()
        self.instance.release()
        event.accept()

    def cleanup(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.playbutton.setText("Play")
        self.is_paused = True
        self.timer.stop()
        self.positionslider.setValue(0)
        self.timelabel.setText("00:00 / 00:00")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.next_frame()

    def skip(self, milliseconds: int):
        if self.cap:
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            total_time = (
                self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                / self.cap.get(cv2.CAP_PROP_FPS)
                * 1000
            )
            new_time = max(0, min(total_time, current_time + milliseconds))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, int(new_time))
            self.next_frame()

    def update_ui(self):
        if self.cap:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0:
                pos = int((current_frame / total_frames) * 1000)
                self.positionslider.setValue(pos)

            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            total_time = total_frames / self.cap.get(cv2.CAP_PROP_FPS)
            self.timelabel.setText(
                f"{int(current_time // 60):02}:{int(current_time % 60):02} / "
                f"{int(total_time // 60):02}:{int(total_time % 60):02}"
            )
            self.next_frame()

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bits_per_line = ch * w
        qimg = QImage(frame.data, w, h, bits_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_frame.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_frame.setPixmap(scaled)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    player = VideoPlayerWidget()
    player.set_video_source(
        "matches_data/epee_2/original.mp4"
    )  # Replace with your video path
    player.show()
    sys.exit(app.exec())
