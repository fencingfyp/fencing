"""
This class is built with heavy reference from:
https://github.com/BBC-Esq/Pyside6_PyQt6_video_audio_player/blob/main/media_player_pyside6.py
"""

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .raw_video_widget import RawVideoWidget


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
        self.is_paused = False
        self.is_dragging = False

        # QLabel to display frames
        self.video_frame = RawVideoWidget(self)

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
        self.video_frame.set_relative_position(pos)
        self.set_time_label()

    def set_time_label(self):
        current_time = self.video_frame.get_current_time_msec() / 1000
        total_time = self.video_frame.get_total_time_msec() / 1000
        self.timelabel.setText(
            f"{int(current_time // 60):02}:{int(current_time % 60):02} / "
            f"{int(total_time // 60):02}:{int(total_time % 60):02}"
        )

    def set_video_source(self, video_path: str):
        self.video_frame.set_video_source(video_path)
        self.delay = self.video_frame.get_delay()
        self.timer.start(self.delay)

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def cleanup(self):
        if self.timer.isActive():
            self.timer.stop()

    def stop(self):
        self.playbutton.setText("Play")
        self.is_paused = True
        self.timer.stop()
        self.positionslider.setValue(0)
        self.timelabel.setText("00:00 / 00:00")
        self.video_frame.set_seconds_position(0)

    def skip(self, milliseconds: int):
        if self.video_frame.has_video():
            self.video_frame.skip_milliseconds(milliseconds)

    def update_ui(self):
        if self.video_frame.has_video():
            current_frame = self.video_frame.get_current_frame_number()
            total_frames = self.video_frame.get_max_frame_number()
            if total_frames > 0:
                pos = int((current_frame / total_frames) * 1000)
                self.positionslider.setValue(pos)

            self.set_time_label()
            self.next_frame()

    def next_frame(self):
        has_next = self.video_frame.next_frame()
        if not has_next:
            self.stop()


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    player = VideoPlayerWidget()
    player.set_video_source(
        "matches_data/epee_2/original.mp4",
    )  # Replace with your video path
    player.show()
    sys.exit(app.exec())
