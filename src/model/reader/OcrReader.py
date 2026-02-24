# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────
import abc

import numpy as np


class OcrReader(abc.ABC):
    """
    Minimal interface all OCR readers must implement.
    Subclasses own their own preprocessing and inference logic.
    """

    @abc.abstractmethod
    def read(self, raw_image: np.ndarray, debug: bool = False) -> tuple[str, float]:
        """Preprocess and read a single raw BGR ROI crop."""

    @abc.abstractmethod
    def read_batch(
        self, raw_images: list[np.ndarray], debug: bool = False
    ) -> list[tuple[str, float]]:
        """Preprocess and read a batch of raw BGR ROI crops."""
        """Preprocess and read a batch of raw BGR ROI crops."""
        """Preprocess and read a batch of raw BGR ROI crops."""
        """Preprocess and read a batch of raw BGR ROI crops."""
        """Preprocess and read a batch of raw BGR ROI crops."""
