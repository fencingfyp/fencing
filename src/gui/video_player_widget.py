"""
This class is built with heavy reference from:
https://github.com/BBC-Esq/Pyside6_PyQt6_video_audio_player/blob/main/media_player_pyside6.py
"""

from PySide6.QtCore import Qt, QTimer
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
        self.is_paused = True
        self.is_dragging = False
        self._active = False
        self._shortcuts: dict[str, QShortcut] = {}

        # Video display
        self.video_frame = RawVideoWidget(parent)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.delay: int | None = None

        # Slider
        self.positionslider = ClickableSlider(Qt.Orientation.Horizontal, self)
        self.positionslider.setMaximum(1000)
        self.positionslider.sliderMoved.connect(self.set_position)
        self.positionslider.sliderPressed.connect(self.slider_pressed)
        self.positionslider.sliderReleased.connect(self.slider_released)

        # Time label
        self.timelabel = QLabel("00:00 / 00:00")
        self.timelabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Playback buttons
        self.playbutton = QPushButton("Play")
        self.playbutton.clicked.connect(self.play_pause)
        # self.stopbutton = QPushButton("Stop")
        # self.stopbutton.clicked.connect(self.stop)
        hbox = QHBoxLayout()
        hbox.addWidget(self.playbutton)
        # hbox.addWidget(self.stopbutton)
        hbox.addStretch(1)

        # Layout
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.video_frame, 1)
        vbox.addWidget(self.positionslider)
        vbox.addWidget(self.timelabel)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

    # ----------------------------- Lifecycle -----------------------------
    def activate(self):
        """Start timer if a video is loaded."""
        if not self._active and self.video_frame.has_video():
            self._active = True
            if self.delay is not None and not self.timer.isActive():
                self.play()
        self.register_shortcut(Qt.Key.Key_Space, self.toggle_pause)
        self.register_shortcut(Qt.Key.Key_Left, self.step_backward)
        self.register_shortcut(Qt.Key.Key_Right, self.step_forward)

    def deactivate(self):
        """Stop timer and optionally remove shortcuts."""
        self._active = False
        if self.timer.isActive():
            self.timer.stop()
        # Disconnect and remove any task-specific shortcuts
        for sc in self._shortcuts.values():
            sc.activated.disconnect()
            sc.setParent(None)
        self._shortcuts.clear()

    def register_shortcut(self, key_sequence: str, callback):
        """Add a temporary shortcut for the lifetime of this player activation."""
        sc = QShortcut(QKeySequence(key_sequence), self)
        sc.activated.connect(callback)
        self._shortcuts[key_sequence] = sc
        return sc

    # ----------------------------- Controls -----------------------------
    def play(self):
        self.playbutton.setText("Pause")
        self.is_paused = False
        if self._active and self.delay:
            self.timer.start(self.delay)

    def play_pause(self):
        if self.is_paused:
            self.play()
        else:
            self.playbutton.setText("Play")
            self.is_paused = True
            self.timer.stop()

    def stop(self):
        self.playbutton.setText("Play")
        self.is_paused = True
        self.timer.stop()
        self.positionslider.setValue(0)
        self.timelabel.setText("00:00 / 00:00")
        self.video_frame.set_seconds_position(0)

    def slider_pressed(self):
        self.is_dragging = True
        self.timer.stop()

    def slider_released(self):
        self.is_dragging = False
        self.set_position()
        if not self.is_paused and self._active and self.delay:
            self.timer.start(self.delay)

    def step_backward(self):
        self.skip(-2000)

    def step_forward(self):
        self.skip(2000)

    def toggle_pause(self):
        self.play_pause()

    # ----------------------------- Video Handling -----------------------------
    def set_video_source(self, video_path: str):
        """Reset player state and load a new video."""
        self.stop()
        self.video_frame.set_video_source(video_path)
        self.delay = self.video_frame.get_delay()
        self.positionslider.setValue(0)
        self.timelabel.setText("00:00 / 00:00")

    def skip(self, milliseconds: int):
        if self.video_frame.has_video():
            self.video_frame.skip_milliseconds(milliseconds)

    # ----------------------------- UI Updates -----------------------------
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

    def update_ui(self):
        if not self._active or not self.video_frame.has_video():
            return

        current_frame = self.video_frame.get_current_frame_number()
        total_frames = self.video_frame.get_max_frame_number()
        if total_frames and total_frames > 0:
            self.positionslider.setValue(int((current_frame / total_frames) * 1000))

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
