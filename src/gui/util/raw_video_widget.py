import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.util.lru_frame_reader import LruFrameReader


class RawVideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Video state
        self.cap: cv2.VideoCapture | None = None
        self.frame_reader: LruFrameReader | None = None
        self.current_frame_idx: int = 0

    # ------------------------------------------------------------------ internal

    def _display_frame(self, frame: np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _read_and_display(self, frame_idx: int) -> bool:
        if not self.frame_reader:
            return False

        ok, frame = self.frame_reader.read_at(frame_idx)
        if not ok:
            return False

        self.current_frame_idx = frame_idx
        self._display_frame(frame)
        return True

    # ------------------------------------------------------------------ public controls

    def next_frame(self) -> bool:
        return self._read_and_display(self.current_frame_idx + 1)

    def repeat_frame(self) -> bool:
        return self._read_and_display(self.current_frame_idx)

    def skip_frames(self, number_of_frames: int):
        if not self.cap:
            return

        max_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        target = max(0, min(self.current_frame_idx + number_of_frames, max_frame))
        self._read_and_display(target)

    def skip_milliseconds(self, milliseconds: float):
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.skip_frames(int((milliseconds / 1000) * fps))

    def set_seconds_position(self, seconds: float):
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._read_and_display(int(seconds * fps))

    def set_relative_position(self, pos: float):
        if not self.cap:
            return
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._read_and_display(int((pos / 1000) * total_frames))

    # ------------------------------------------------------------------ lifecycle

    def set_video_source(
        self, video_path: str, *, cache_bytes: int = 512 * 1024 * 1024
    ):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        self.frame_reader = LruFrameReader(self.cap, cache_bytes)
        self.current_frame_idx = 0
        self.repeat_frame()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.cap = None
        if self.frame_reader:
            self.frame_reader.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------ public API (unchanged)

    def setPixmap(self, pixmap: QPixmap):
        self.image_label.setPixmap(pixmap)

    def get_current_frame_number(self) -> np.ndarray | None:
        if not self.cap:
            return None
        return self.current_frame_idx

    def get_max_frame_number(self) -> np.ndarray | None:
        if not self.cap:
            return None
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_current_time_msec(self) -> np.ndarray | None:
        if not self.cap:
            return None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return None
        return (self.current_frame_idx / fps) * 1000

    def get_total_time_msec(self) -> np.ndarray | None:
        if not self.cap:
            return None
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return None
        return (total_frames / fps) * 1000

    def has_video(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_delay(self) -> int | None:
        if not self.cap:
            return None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return None

        return int(1000 / fps)

    def set_size(self, new_size):
        if not self.cap:
            self.image_label.setFixedSize(new_size)
            return

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        aspect_ratio = width / height

        if new_size.width() / new_size.height() > aspect_ratio:
            scaled_width = int(new_size.height() * aspect_ratio)
            scaled_height = new_size.height()
        else:
            scaled_width = new_size.width()
            scaled_height = int(new_size.width() / aspect_ratio)

        self.image_label.setFixedSize(scaled_width, scaled_height)

    def get_video_capture(self) -> cv2.VideoCapture | None:
        return self.cap


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = RawVideoWidget()
    widget.set_video_source(
        "matches_data/epee_2/original.mp4"
    )  # Replace with your video path
    widget.set_relative_position(0)
    widget.show()
    sys.exit(app.exec())
