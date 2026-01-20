import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class RawVideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.cap = None
        layout = QVBoxLayout()
        layout.addWidget(self.image_label, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def next_frame(self) -> bool:
        """Update the displayed frame. Returns True if successful, False otherwise."""
        scaled = self.get_next_pixmap()
        if scaled is None:
            return False
        self.image_label.setPixmap(scaled)
        return True

    def setPixmap(self, pixmap: QPixmap):
        """Sets the displayed pixmap directly."""
        self.image_label.setPixmap(pixmap)

    def get_next_pixmap(self) -> QPixmap | None:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    def repeat_frame(self) -> bool:
        """Re-displays the current frame without advancing."""
        if self.cap is None:
            return False
        current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number - 1)
        return self.next_frame()

    def skip_frames(self, number_of_frames: int):
        """Skips the specified number of frames (no-op for raw display). If negative, goes back."""
        if not self.cap:
            return
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_number = min(
            max(0, current_frame + number_of_frames),
            int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
        )
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def skip_milliseconds(self, milliseconds: float):
        """Skips the specified number of milliseconds (no-op for raw display). If negative, goes back."""
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int((milliseconds / 1000) * fps)
        self.skip_frames(frames_to_skip)
        self.next_frame()

    def set_seconds_position(self, seconds: float):
        """Sets the video position to the specified time in seconds."""
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(seconds * fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.next_frame()

    def set_relative_position(self, pos: float):
        """Sets the video position to the specified relative position (0-1000)."""
        if not self.cap:
            return
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_number = int((pos / 1000) * total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.next_frame()

    def set_video_source(self, video_path: str):
        """Sets the video source."""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.repeat_frame()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.cap = None
        return super().closeEvent(event)

    def get_current_frame_number(self) -> np.ndarray | None:
        """Returns the current frame as a numpy array, or None if not available."""
        if not self.cap:
            return None
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    def get_max_frame_number(self) -> np.ndarray | None:
        """Returns the maximum frame number, or None if not available."""
        if not self.cap:
            return None
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_current_time_msec(self) -> np.ndarray | None:
        """Returns the current time in milliseconds, or None if not available."""
        if not self.cap:
            return None
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_total_time_msec(self) -> np.ndarray | None:
        """Returns the total time in milliseconds, or None if not available."""
        if not self.cap:
            return None
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return None
        return (total_frames / fps) * 1000

    def has_video(self) -> bool:
        """Returns True if a video source is set, False otherwise."""
        return self.cap is not None and self.cap.isOpened()

    def get_delay(self) -> int | None:
        """Returns the delay between frames in milliseconds."""
        if not self.cap:
            return None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return None
        return int(1000 / fps)

    def set_size(self, new_size):
        """Sets the size of the video display area, scaled to maintain aspect ratio."""
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
        """Returns the underlying cv2.VideoCapture object."""
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
