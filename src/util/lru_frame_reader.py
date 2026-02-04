from collections import OrderedDict

import cv2
import numpy as np


class LruFrameReader:
    def __init__(self, cap: cv2.VideoCapture, max_cache_bytes: int):
        self.cap = cap
        self.max_cache_bytes = max_cache_bytes

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._decoder_pos: int | None = None

    def _touch(self, idx: int) -> np.ndarray:
        frame = self._cache.pop(idx)
        self._cache[idx] = frame
        return frame

    def _evict_if_needed(self, incoming_bytes: int):
        while self._cache and self._cache_bytes + incoming_bytes > self.max_cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes

    def _insert(self, idx: int, frame: np.ndarray):
        frame_bytes = frame.nbytes

        # Frame larger than cache → don't store
        if frame_bytes > self.max_cache_bytes:
            return

        self._evict_if_needed(frame_bytes)
        self._cache[idx] = frame
        self._cache_bytes += frame_bytes

    def _ensure_decoder_at(self, frame_idx: int) -> bool:
        """
        Ensures the next read() will return frame_idx.
        Returns False on failure.
        """
        if self._decoder_pos is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._decoder_pos = frame_idx
            return True

        if frame_idx != self._decoder_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._decoder_pos = frame_idx
            return True
        return True

    def _read_frame(self, frame_idx: int):
        if not self._ensure_decoder_at(frame_idx):
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self._decoder_pos = frame_idx + 1
        return True, frame

    def read_at(self, frame_idx: int):
        # Cache hit
        if frame_idx in self._cache:
            return True, self._touch(frame_idx)

        ok, frame = self._read_frame(frame_idx)
        if not ok:
            return False, None

        self._insert(frame_idx, frame)
        return True, frame

    def close(self):
        self.cap.release()
        self._cache.clear()
        self._cache_bytes = 0
        self._decoder_pos = None
