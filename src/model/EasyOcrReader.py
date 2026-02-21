"""
EasyOcrReader.py
----------------
Encapsulates image preprocessing and OCR reading for fencing scoreboard digits.

Responsibilities:
  - ScorePreprocessor: converts a raw BGR ROI crop into a clean binarized image
    ready for OCR. All image processing lives here.
  - EasyOcrReader: runs EasyOCR on a preprocessed image. Knows nothing about
    image processing beyond what format EasyOCR expects.
"""

from __future__ import annotations

from typing import Optional

import cv2
import easyocr
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_HEIGHT = 64  # Canonical height all crops are scaled to
BORDER_PAD = 20  # White border added around binarized image for OCR
EDGE_MARGIN = max(
    1, int(TARGET_HEIGHT * 0.08)
)  # Pixels to ignore at ROI edges when computing histogram
# 8% is arbitary.
HIST_SMOOTH_KERNEL = 25  # GaussianBlur kernel size for histogram smoothing

THRESHOLD_RATIO = 0.7  # Threshold at this fraction of the way from Otsu to bright peak


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class ScorePreprocessor:
    """
    Converts a raw BGR score ROI into a clean binary BGR image for OCR.

    Pipeline:
      1. Channel selection    — pick the channel with highest contrast
      2. Edge margin crop     — exclude noisy ROI edges from histogram analysis
      3. Resize               — scale to fixed canonical height
      4. Histogram thresholding — Otsu to find background/foreground split,
                                  then FWHM-based offset to cut below digit peak
      5. Polarity normalisation — ensure dark digits on white background
      6. Border padding        — white border so OCR doesn't clip edge digits
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        gray = self._select_channel(image)
        gray = self._resize(gray)
        binary = self._threshold(gray)
        binary = self._normalise_polarity(binary)
        binary = self._add_border(binary)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _select_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Pick the single BGR channel with the highest standard deviation.
        This handles variable digit colour (red/white/amber) across videos
        without any hardcoded channel assumption.
        """
        channels = cv2.split(image)
        return max(channels, key=lambda c: c.std())

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        """Scale to TARGET_HEIGHT, preserving aspect ratio."""
        h, w = gray.shape
        new_w = max(int(w * TARGET_HEIGHT / h), 1)
        return cv2.resize(gray, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Threshold using the bright peak of the histogram.

        The ROI histogram has (at least) two modes: digit cores (bright) and
        background (dark), with a halo/glow tail between them. Otsu finds the
        valley between background and foreground, anchoring our search to the
        upper half. We then find the bright peak above Otsu and threshold at
        one FWHM below it, cutting the halo tail while keeping the digit core.

        Edge pixels are excluded from histogram computation to avoid ROI
        boundary noise skewing the bright peak.
        """
        # Compute histogram on inner region only (exclude noisy edges)
        inner = gray[EDGE_MARGIN:-EDGE_MARGIN, EDGE_MARGIN:-EDGE_MARGIN]
        hist = cv2.calcHist([inner], [0], None, [256], [0, 256]).flatten()
        hist_smooth = cv2.GaussianBlur(
            hist[None, :], (1, HIST_SMOOTH_KERNEL), 0
        ).flatten()

        # Otsu gives us the background/foreground boundary
        otsu_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_thresh = int(otsu_thresh)

        # Find the bright peak above Otsu
        upper = hist_smooth[otsu_thresh:]
        bright_peak = int(np.argmax(upper)) + otsu_thresh

        # Threshold at a fixed fraction of the way from Otsu up to the bright peak

        threshold = int(otsu_thresh + (bright_peak - otsu_thresh) * THRESHOLD_RATIO)

        # draw histogram and thresholds for debugging
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(8, 4))
        # plt.plot(hist_smooth, label="Smoothed histogram")
        # plt.axvline(otsu_thresh, color="orange", linestyle="--", label="Otsu threshold")
        # plt.axvline(bright_peak, color="green", linestyle="--", label="Bright peak")
        # plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
        # plt.legend()
        # plt.title("Histogram and thresholds")
        # plt.xlabel("Pixel intensity")
        # plt.ylabel("Count")
        # plt.show()

        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _normalise_polarity(self, binary: np.ndarray) -> np.ndarray:
        """
        Ensure dark digits on white background.
        EasyOCR expects this convention; scoreboards may be either polarity.
        Majority vote: if more than half the pixels are dark, the background
        is dark and we invert.
        """
        if np.mean(binary) < 127:
            return cv2.bitwise_not(binary)
        return binary

    def _add_border(self, binary: np.ndarray) -> np.ndarray:
        """Add white border so OCR doesn't clip digits at image edges."""
        return cv2.copyMakeBorder(
            binary,
            BORDER_PAD,
            BORDER_PAD,
            BORDER_PAD,
            BORDER_PAD,
            cv2.BORDER_CONSTANT,
            value=255,
        )


# ---------------------------------------------------------------------------
# OCR Reader
# ---------------------------------------------------------------------------


class EasyOcrReader:
    """
    Runs EasyOCR on a preprocessed score ROI.

    Two modes:
      - Normal: pass the full image to EasyOCR and take the highest-confidence
        detection closest to the image centre.
      - Seven-segment: segment individual digits by contour, run OCR on each,
        then concatenate results left-to-right.
    """

    def __init__(self, device: str, seven_segment: bool = False):
        self.reader = easyocr.Reader(["ch_sim"], gpu=device)
        self.seven_segment = seven_segment
        self.preprocessor = ScorePreprocessor()

    def read(self, raw_image: np.ndarray) -> tuple[str, float]:
        """
        Preprocess and read a score from a raw BGR ROI crop.

        Returns:
            (score, confidence): score as string, confidence in [0, 1].
            Returns ("", 0.0) if nothing is detected.
        """
        processed = self.preprocessor(raw_image)
        if self.seven_segment:
            return self._read_seven_segment(processed)
        return self._read_normal(processed)

    # ------------------------------------------------------------------
    # Reading modes
    # ------------------------------------------------------------------

    def _read_normal(self, image: np.ndarray) -> tuple[str, float]:
        """
        Run EasyOCR on the full image.
        Among all detections, pick the one with the highest confidence
        whose bounding box centre is closest to the image centre.
        Proximity to centre breaks ties and avoids picking up background noise
        at image edges.
        """
        results = self.reader.recognize(image, allowlist="0123456789")
        if not results:
            return "", 0.0

        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        def centre_distance(bbox):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return ((np.mean(xs) - cx) ** 2 + (np.mean(ys) - cy) ** 2) ** 0.5

        # Filter out malformed results
        valid = [
            (bbox, text, prob)
            for bbox, text, prob in results
            if bbox is not None and len(bbox) == 4 and text
        ]
        if not valid:
            return "", 0.0

        # Primary sort: confidence (desc). Secondary: proximity to centre (asc).
        best = min(valid, key=lambda r: (-r[2], centre_distance(r[0])))
        _, text, prob = best
        return text, float(prob)

    def _read_seven_segment(self, image: np.ndarray) -> tuple[str, float]:
        """
        Segment individual digit blobs by contour, run OCR on each crop,
        then concatenate left-to-right.
        """
        boxes = self._segment_digits(image)
        if not boxes:
            return "", 0.0

        digits, confidences = [], []
        for i, (x, y, w, h) in enumerate(boxes):
            crop = image[y : y + h, x : x + w]
            crop = cv2.copyMakeBorder(
                crop,
                BORDER_PAD,
                BORDER_PAD,
                BORDER_PAD,
                BORDER_PAD,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            # cv2.imshow(f"crop_{i}", crop)
            results = self.reader.recognize(crop, allowlist="0123456789")
            if results:
                _, text, prob = max(results, key=lambda r: r[2])
                digits.append(text)
                confidences.append(prob)

        if not digits:
            return "", 0.0
        return "".join(digits), float(np.mean(confidences))

    # ------------------------------------------------------------------
    # Seven-segment helpers
    # ------------------------------------------------------------------
    def _segment_digits(
        self,
        binary: np.ndarray,
        max_digits: int = 2,
        open_kernel: tuple[int, int] = (
            3,
            1,
        ),  # wide enough to break digit bridges, short enough to preserve horizontal segments
        min_area_fraction: float = 0.05,
        height_tol: float = 0.6,
    ) -> list[tuple[int, int, int, int]]:
        """
        Segment digits using morphological opening followed by contour detection.
        Opening with a horizontal kernel breaks thin pixel bridges between touching
        digits without destroying tall vertical bars or horizontal segments.

        Parameters
        ----------
        open_kernel : tuple[int, int]
            (width, height) of the opening kernel. Wider breaks more connections;
            taller risks destroying horizontal digit segments.
        min_area_fraction : float
            Blobs smaller than this fraction of the largest blob are treated as noise.
        height_tol : float
            Blobs shorter than this fraction of the tallest blob are dropped.
        """
        gray = (
            cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
            if len(binary.shape) == 3
            else binary
        )
        inv = cv2.bitwise_not(gray)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
        inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        boxes = [cv2.boundingRect(c) for c in contours]
        max_area = max(b[2] * b[3] for b in boxes)
        tallest_h = max(b[3] for b in boxes)

        filtered = [
            b
            for b in boxes
            if b[2] * b[3] >= min_area_fraction * max_area
            and b[3] >= height_tol * tallest_h
        ]

        return sorted(filtered, key=lambda b: b[0])[:max_digits]
