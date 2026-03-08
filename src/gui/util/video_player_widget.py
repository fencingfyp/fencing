"""
This class is built with heavy reference from:
https://github.com/BBC-Esq/Pyside6_PyQt6_video_audio_player/blob/main/media_player_pyside6.py
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .raw_video_widget import RawVideoWidget
from .video_state import VideoState


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
    """
    Orchestrates VideoState (data) and RawVideoWidget (display).
    Extend by connecting to signals or subclassing and overriding _build_controls().
    """

    # Emitted on every frame advance — useful for extensions (e.g. tag overlay)
    frame_changed = Signal(int, float)  # (frame_idx, time_msec)
    # Emitted when the user clicks/drags the slider to a new position
    position_seeked = Signal(int, float)  # (frame_idx, time_msec)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = VideoState()
        self._is_paused = True
        self._is_dragging = False
        self._active = False
        self._playback_rate = 1.0
        self._shortcuts: dict[str, QShortcut] = {}

        self._build_ui()

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self):
        self._video_frame = RawVideoWidget(self)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._slider = ClickableSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMaximum(1000)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._play_button = QPushButton("Play")
        self._play_button.clicked.connect(self.play_pause)

        self._speed_box = QComboBox(self)
        self._speed_box.addItems(["0.5x", "1x", "2x"])
        self._speed_box.setCurrentText("1x")
        self._speed_box.currentTextChanged.connect(self._set_playback_speed)

        controls = self._build_controls()

        vbox = QVBoxLayout(self)
        vbox.addWidget(self._video_frame, 1)
        vbox.addWidget(self._slider)
        vbox.addWidget(self._time_label)
        vbox.addLayout(controls)

    def _build_controls(self) -> QHBoxLayout:
        """
        Override in subclasses to inject extra controls into the bottom bar,
        or add widgets before/after by calling super() and modifying the layout.
        """
        hbox = QHBoxLayout()
        hbox.addWidget(self._play_button)
        hbox.addStretch(1)
        hbox.addWidget(QLabel("Speed"))
        hbox.addWidget(self._speed_box)
        return hbox

    # ------------------------------------------------------------------ lifecycle

    def activate(self):
        if not self._active and self._state.is_loaded:
            self._active = True
            if not self._is_paused:
                self._start_timer()
        self.register_shortcut(Qt.Key.Key_Space, self.toggle_pause)
        self.register_shortcut(Qt.Key.Key_Left, self.step_backward)
        self.register_shortcut(Qt.Key.Key_Right, self.step_forward)
        self._play_button.setEnabled(True)

    def deactivate(self):
        self._active = False
        self._timer.stop()
        self._play_button.setEnabled(False)
        for sc in self._shortcuts.values():
            sc.activated.disconnect()
            sc.setParent(None)
        self._shortcuts.clear()

    def showEvent(self, event):
        self.activate()
        return super().showEvent(event)

    def hideEvent(self, event):
        self.deactivate()
        return super().hideEvent(event)

    def closeEvent(self, event):
        self.deactivate()
        self._state.release()
        return super().closeEvent(event)

    def register_shortcut(self, key_sequence, callback):
        sc = QShortcut(QKeySequence(key_sequence), self)
        sc.activated.connect(callback)
        self._shortcuts[key_sequence] = sc
        return sc

    # ------------------------------------------------------------------ public controls

    def play(self):
        if not self._is_paused:
            return
        self._play_button.setText("Pause")
        self._is_paused = False
        if self._active:
            self._start_timer()

    def pause(self):
        if self._is_paused:
            return
        self._play_button.setText("Play")
        self._is_paused = True
        self._timer.stop()

    def play_pause(self):
        self.play() if self._is_paused else self.pause()

    def toggle_pause(self):
        self.play_pause()

    def stop(self):
        self.pause()
        self._slider.setValue(0)
        self._time_label.setText("00:00 / 00:00")
        ok, frame = self._state.seek_to_seconds(0)
        if ok:
            self._video_frame.display_frame(frame)

    def step_backward(self):
        self._seek_msec(-2000)

    def step_forward(self):
        self._seek_msec(2000)

    # ------------------------------------------------------------------ public API

    def set_video_source(self, video_path: str, cache_bytes: int = 512 * 1024 * 1024):
        self.stop()
        self._state.load(video_path, cache_bytes)
        ok, frame = self._state.read_current()
        if ok:
            self._video_frame.display_frame(frame)
        self._update_timer_delay()
        self._update_time_label()

    def skip_milliseconds(self, milliseconds: float):
        self._seek_msec(milliseconds)

    def set_frame_position(self, frame_idx: int):
        ok, frame = self._state.seek_frame(frame_idx)
        if ok:
            self._video_frame.display_frame(frame)
            self._sync_ui()

    def set_msec_position(self, milliseconds: float):
        ok, frame = self._state.seek_to_milliseconds(milliseconds)
        if ok:
            self._video_frame.display_frame(frame)
            self._sync_ui()

    def set_seconds_position(self, seconds: float):
        ok, frame = self._state.seek_to_seconds(seconds)
        if ok:
            self._video_frame.display_frame(frame)
            self._sync_ui()

    def set_relative_position(self, pos: float):
        ok, frame = self._state.seek_to_relative(pos)
        if ok:
            self._video_frame.display_frame(frame)
            self._sync_ui()

    def has_video(self) -> bool:
        return self._state.is_loaded

    def get_current_frame_number(self) -> int | None:
        return self._state.current_frame_idx if self._state.is_loaded else None

    def get_current_time_msec(self) -> float | None:
        return self._state.current_time_msec

    def get_total_time_msec(self) -> float | None:
        return self._state.total_time_msec

    def get_video_capture(self):
        return self._state.get_video_capture()

    # ------------------------------------------------------------------ internal

    def _seek_msec(self, milliseconds: float):
        ok, frame = self._state.seek_milliseconds(milliseconds)
        if ok:
            self._video_frame.display_frame(frame)
            self._sync_ui()
            self.position_seeked.emit(
                self._state.current_frame_idx, self._state.current_time_msec
            )

    def _start_timer(self):
        delay = self._current_delay()
        if delay:
            self._timer.start(delay)

    def _current_delay(self) -> int | None:
        base = self._state.delay_msec
        if base is None:
            return None
        return int(base / self._playback_rate)

    def _set_playback_speed(self, text: str):
        self._playback_rate = float(text.replace("x", ""))
        self._update_timer_delay()

    def _update_timer_delay(self):
        delay = self._current_delay()
        if delay and not self._is_paused and self._active:
            self._timer.start(delay)

    def _on_timer(self):
        if not self._active or not self._state.is_loaded:
            return

        ok, frame = self._state.read_next()
        if not ok:
            self.stop()
            return

        self._video_frame.display_frame(frame)
        self._sync_ui()
        self.frame_changed.emit(
            self._state.current_frame_idx, self._state.current_time_msec
        )

    def _on_slider_pressed(self):
        self._is_dragging = True
        self._timer.stop()

    def _on_slider_moved(self, value: int):
        ok, frame = self._state.seek_to_relative(value)
        if ok:
            self._video_frame.display_frame(frame)
            self._update_time_label()

    def _on_slider_released(self):
        self._is_dragging = False
        if not self._is_paused and self._active:
            self._start_timer()
        self.position_seeked.emit(
            self._state.current_frame_idx, self._state.current_time_msec
        )

    def _sync_ui(self):
        total = self._state.total_frames
        if total and total > 0:
            self._slider.setValue(int((self._state.current_frame_idx / total) * 1000))
        self._update_time_label()

    def _update_time_label(self):
        current = self._state.current_time_msec
        total = self._state.total_time_msec
        if current is None or total is None:
            self._time_label.setText("00:00 / 00:00")
            return
        c, t = current / 1000, total / 1000
        self._time_label.setText(
            f"{int(c // 60):02}:{int(c % 60):02} / {int(t // 60):02}:{int(t % 60):02}"
        )


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    player = VideoPlayerWidget()
    player.set_video_source("matches_data/epee_3.mp4")
    player.show()
    sys.exit(app.exec())
