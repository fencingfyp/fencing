import cv2
import numpy as np

from src.util.lru_frame_reader import LruFrameReader


class VideoState:
    """
    Owns video capture and frame navigation. No Qt, fully testable.
    """

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._frame_reader: LruFrameReader | None = None
        self._current_frame_idx: int = 0

    # ------------------------------------------------------------------ lifecycle

    def load(self, video_path: str, cache_bytes: int = 512 * 1024 * 1024):
        self.release()
        self._cap = cv2.VideoCapture(video_path)
        self._frame_reader = LruFrameReader(self._cap, cache_bytes)
        self._current_frame_idx = 0

    def release(self):
        if self._cap:
            self._cap.release()
        self._cap = None
        if self._frame_reader:
            self._frame_reader.close()
        self._frame_reader = None

    # ------------------------------------------------------------------ properties

    @property
    def is_loaded(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def current_frame_idx(self) -> int:
        return self._current_frame_idx

    @property
    def total_frames(self) -> int | None:
        if not self.is_loaded:
            return None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float | None:
        if not self.is_loaded:
            return None
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else None

    @property
    def delay_msec(self) -> int | None:
        """Frame interval in milliseconds at 1x speed."""
        if self.fps is None:
            return None
        return int(1000 / self.fps)

    @property
    def current_time_msec(self) -> float | None:
        if self.fps is None:
            return None
        return (self._current_frame_idx / self.fps) * 1000

    @property
    def total_time_msec(self) -> float | None:
        if self.fps is None or self.total_frames is None:
            return None
        return (self.total_frames / self.fps) * 1000

    # ------------------------------------------------------------------ navigation

    def read_at(self, frame_idx: int) -> tuple[bool, np.ndarray | None]:
        if not self._frame_reader:
            return False, None
        ok, frame = self._frame_reader.read_at(frame_idx)
        if ok:
            self._current_frame_idx = frame_idx
        return ok, frame if ok else None

    def read_current(self) -> tuple[bool, np.ndarray | None]:
        return self.read_at(self._current_frame_idx)

    def read_next(self) -> tuple[bool, np.ndarray | None]:
        return self.read_at(self._current_frame_idx + 1)

    def seek_frame(self, frame_idx: int) -> tuple[bool, np.ndarray | None]:
        if not self.is_loaded:
            return False, None
        max_frame = self.total_frames - 1
        target = max(0, min(frame_idx, max_frame))
        return self.read_at(target)

    def seek_milliseconds(self, milliseconds: float) -> tuple[bool, np.ndarray | None]:
        if self.fps is None:
            return False, None
        delta_frames = int((milliseconds / 1000) * self.fps)
        return self.seek_frame(self._current_frame_idx + delta_frames)

    def seek_to_milliseconds(
        self, milliseconds: float
    ) -> tuple[bool, np.ndarray | None]:
        if self.fps is None:
            return False, None
        target_frame = int((milliseconds / 1000) * self.fps)
        return self.seek_frame(target_frame)

    def seek_to_seconds(self, seconds: float) -> tuple[bool, np.ndarray | None]:
        if self.fps is None:
            return False, None
        return self.seek_frame(int(seconds * self.fps))

    def seek_to_relative(self, pos: float) -> tuple[bool, np.ndarray | None]:
        """pos in range [0, 1000]."""
        if self.total_frames is None:
            return False, None
        return self.seek_frame(int((pos / 1000) * self.total_frames))

    def get_video_capture(self) -> cv2.VideoCapture | None:
        return self._cap
