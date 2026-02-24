"""
EasyOcrReader.py
----------------
Encapsulates image preprocessing and OCR reading for fencing scoreboard digits.

Responsibilities:
  - ScorePreprocessor: converts a raw BGR ROI crop into a clean binarized image
    ready for OCR. All image processing lives here.
  - EasyOcrReader: orchestrates preprocessing and batched inference via EasyOCR's
    recognize() with pre-known bounding boxes, skipping CRAFT detection entirely.
    Both normal and seven-segment modes batch all crops into a single GPU call.

Fallback strategy
-----------------
ch_sim is the primary model. Any crop that returns confidence == 0.0 is
collected and passed to a second english-model pass in one batched GPU call.
This handles ch_sim's known failure on seven-segment '3' without affecting
throughput on well-recognised digits.

1/7 disambiguation
------------------
Both models confidently misread seven-segment '7' as '1' across some videos
because the vertical stroke dominates and neither model learned the blocky
seven-segment top bar as a distinguishing feature. After inference, any crop
read as '1' is checked for a significant horizontal dark run in the top band
of the image. If one is found, the result is corrected to '7'.
"""

from __future__ import annotations

import cv2
import easyocr
import numpy as np

TARGET_HEIGHT = 64  # Canonical height all crops are scaled to
BORDER_PAD = 20  # White border added around binarized image for OCR
HIST_SMOOTH_KERNEL = 25  # GaussianBlur kernel size for histogram smoothing
THRESHOLD_RATIO = 0.6  # Fraction from Otsu up to bright peak for threshold
# 0.0 = Otsu only (includes halo), 1.0 = peak only (core only)
EDGE_MARGIN = max(1, int(TARGET_HEIGHT * 0.08))


class EasyOcrScorePreprocessor:
    """
    Converts a raw BGR score ROI into a clean binary grayscale image for OCR.

    Pipeline:
      1. Channel selection     — pick the channel with highest contrast
      2. Resize                — scale to fixed canonical height
      3. Histogram thresholding — Otsu to find background/foreground split,
                                  then ratio-based offset to cut below digit peak
      4. Polarity normalisation — ensure dark digits on white background
      5. Tight cropping         — crop to content bounding box, removing excess whitespace
      6. Re-enforce height      — resize back to TARGET_HEIGHT after crop
      7. Border padding         — white border so OCR doesn't clip edge digits
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        gray = self._select_channel(image)
        gray = self._resize(gray)
        binary = self._threshold(gray)
        binary = self._normalise_polarity(binary)
        binary = self._tight_crop(binary)
        binary = self._resize_height(binary)
        binary = self._add_border(binary)
        return binary

    def _select_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Pick the single BGR channel with the highest standard deviation.
        Handles variable digit colour (red/white/amber) across videos without
        any hardcoded channel assumption.
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

        Otsu finds the valley between background and foreground modes. We find
        the bright peak above Otsu and threshold at THRESHOLD_RATIO of the way
        from Otsu to the peak, cutting the halo tail while keeping digit core.
        Edge pixels are excluded from histogram computation to avoid ROI
        boundary noise skewing the bright peak.
        """
        inner = gray[EDGE_MARGIN:-EDGE_MARGIN, EDGE_MARGIN:-EDGE_MARGIN]
        hist = cv2.calcHist([inner], [0], None, [256], [0, 256]).flatten()
        hist_smooth = cv2.GaussianBlur(
            hist[None, :], (1, HIST_SMOOTH_KERNEL), 0
        ).flatten()

        otsu_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_thresh = int(otsu_thresh)

        upper = hist_smooth[otsu_thresh:]
        bright_peak = int(np.argmax(upper)) + otsu_thresh

        threshold = int(otsu_thresh + (bright_peak - otsu_thresh) * THRESHOLD_RATIO)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _normalise_polarity(self, binary: np.ndarray) -> np.ndarray:
        """
        Ensure dark digits on white background.
        If more than half the pixels are dark, background is dark — invert.
        """
        if np.mean(binary) < 127:
            return cv2.bitwise_not(binary)
        return binary

    def _tight_crop(self, binary: np.ndarray) -> np.ndarray:
        """
        Crop to the bounding box of actual content pixels.
        Removes excess whitespace so the CRNN doesn't waste sequence steps
        on empty columns. Border is added after, so padding stays consistent.
        """
        dark = binary < 127
        rows = np.where(dark.any(axis=1))[0]
        cols = np.where(dark.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return binary
        return binary[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    def _resize_height(self, binary: np.ndarray) -> np.ndarray:
        """Re-enforce TARGET_HEIGHT after tight crop may have changed it."""
        h, w = binary.shape
        if h == TARGET_HEIGHT:
            return binary
        new_w = max(int(w * TARGET_HEIGHT / h), 1)
        return cv2.resize(
            binary, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST
        )

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
# Constants
# ---------------------------------------------------------------------------


FALLBACK_CONFIDENCE_THRESHOLD = 0.00  # crops below this go to english fallback


# ---------------------------------------------------------------------------
# OCR Reader
# ---------------------------------------------------------------------------


class EasyOcrReader:
    """
    Orchestrates preprocessing and batched inference for fencing scoreboard OCR.

    Two modes:
      - Normal: preprocess each ROI, tile onto a canvas, pass all boxes to
        EasyOCR recognize() in one GPU call.
      - Seven-segment: preprocess each ROI, segment digits via contour detection,
        tile ALL digit crops from ALL ROIs onto a single canvas, then pass all
        boxes to recognize() in one GPU call. Results are reassembled per-image
        using the precomputed digit counts.

    In both modes, CRAFT detection is skipped entirely — bounding boxes are
    supplied directly to recognize().

    Fallback
    --------
    After the primary ch_sim pass, any crop whose confidence is below
    FALLBACK_CONFIDENCE_THRESHOLD is collected and sent to a single batched
    english-model pass. Results are merged back into the output list before
    returning. The english reader is instantiated lazily on first use.
    """

    def __init__(self, device: str, seven_segment: bool = False):
        self.seven_segment = seven_segment
        self.preprocessor = EasyOcrScorePreprocessor()
        self.device = device
        self.reader = easyocr.Reader(["ch_sim"], gpu=(device != "cpu"), verbose=True)
        self._en_reader: easyocr.Reader | None = None  # lazy init

    @property
    def en_reader(self) -> easyocr.Reader:
        """English reader, instantiated on first fallback use."""
        if self._en_reader is None:
            self._en_reader = easyocr.Reader(
                ["en"], gpu=(self.device != "cpu"), verbose=True
            )
        return self._en_reader

    def read(self, raw_image: np.ndarray, debug: bool = False) -> tuple[str, float]:
        """Preprocess and read a single raw BGR ROI crop."""
        return self.read_batch([raw_image], debug=debug)[0]

    def read_batch(
        self, raw_images: list[np.ndarray], debug: bool = False
    ) -> list[tuple[str, float]]:
        """
        Preprocess and read a batch of raw BGR ROI crops in a single GPU call.

        Normal mode: one crop per image, one result per image.
        Seven-segment mode: segment each preprocessed image into digit crops,
        batch all digit crops together, then reassemble results per image.
        """
        processed = [self.preprocessor(img) for img in raw_images]

        if self.seven_segment:
            return self._read_batch_seven_segment(processed, debug=debug)
        return self._read_batch_normal(processed, debug=debug)

    # ------------------------------------------------------------------
    # Normal mode
    # ------------------------------------------------------------------

    def _read_batch_normal(
        self, images: list[np.ndarray], debug: bool = False
    ) -> list[tuple[str, float]]:
        canvas, boxes = self._make_canvas(images, debug=debug)
        results = self._recognize_with_boxes(
            canvas, boxes, n_expected=len(images), reader=self.reader
        )
        results = self._apply_fallback(results, images)
        return results

    # ------------------------------------------------------------------
    # Seven-segment mode
    # ------------------------------------------------------------------

    def _read_batch_seven_segment(
        self, images: list[np.ndarray], debug: bool = False
    ) -> list[tuple[str, float]]:
        """
        Segment all images into digit crops, batch all crops in one GPU call,
        then reassemble results back into per-image (score, confidence) pairs.

        Precomputing digit counts per image lets us slice the flat results list
        back into per-image groups after inference.
        """
        all_crops: list[np.ndarray] = []
        digit_counts: list[int] = []

        for image in images:
            boxes = self._segment_digits(image)
            crops = self._extract_digit_crops(image, boxes)
            all_crops.extend(crops)
            digit_counts.append(len(crops))

        if not all_crops:
            return [("", 0.0)] * len(images)

        canvas, canvas_boxes = self._make_canvas(all_crops, debug=debug)
        flat_results = self._recognize_with_boxes(
            canvas, canvas_boxes, n_expected=len(all_crops), reader=self.reader
        )

        flat_results = self._apply_fallback(flat_results, all_crops)

        # Reassemble flat results into per-image (score, confidence) pairs
        output = []
        idx = 0
        for count in digit_counts:
            image_results = flat_results[idx : idx + count]
            idx += count

            digits = [text for text, _ in image_results if text]
            confidences = [conf for _, conf in image_results if conf > 0]

            if not digits:
                output.append(("", 0.0))
            else:
                output.append(
                    (
                        "".join(digits),
                        float(np.mean(confidences)),
                    )
                )
        return output

    def _extract_digit_crops(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """Extract, normalise height, and pad individual digit crops."""
        crops = []
        for x, y, w, h in boxes:
            crop = image[y : y + h, x : x + w]
            crop = self.preprocessor._resize_height(crop)
            crop = cv2.copyMakeBorder(
                crop,
                BORDER_PAD,
                BORDER_PAD,
                BORDER_PAD,
                BORDER_PAD,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            crops.append(crop)
        return crops

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _apply_fallback(
        self,
        results: list[tuple[str, float]],
        crops: list[np.ndarray],
    ) -> list[tuple[str, float]]:
        """
        Collect indices where ch_sim returned zero/near-zero confidence,
        run all failed crops through the english model in one batched call,
        then merge the english results back into the output list.
        """
        failed_indices = [
            i
            for i, (_, conf) in enumerate(results)
            if conf <= FALLBACK_CONFIDENCE_THRESHOLD
        ]

        if not failed_indices:
            return results

        failed_crops = [crops[i] for i in failed_indices]
        canvas, boxes = self._make_canvas(failed_crops)
        fallback_results = self._recognize_with_boxes(
            canvas, boxes, n_expected=len(failed_crops), reader=self.en_reader
        )

        merged = list(results)
        for idx, fallback in zip(failed_indices, fallback_results):
            merged[idx] = fallback

        return merged

    # ------------------------------------------------------------------
    # Canvas builder
    # ------------------------------------------------------------------

    def _make_canvas(
        self, images: list[np.ndarray], debug: bool = False
    ) -> tuple[np.ndarray, list[list[int]]]:
        """
        Tile preprocessed grayscale images horizontally onto a white canvas.
        All images must be the same height (TARGET_HEIGHT + 2 * BORDER_PAD).
        Returns the canvas and bounding boxes as [x_min, x_max, y_min, y_max].
        """
        h = images[0].shape[0]
        total_w = sum(img.shape[1] for img in images)
        canvas = np.full((h, total_w), 255, dtype=np.uint8)

        boxes = []
        x = 0
        for img in images:
            w = img.shape[1]
            canvas[:, x : x + w] = img
            boxes.append([x, x + w, 0, h])
            x += w
        if debug:
            cv2.imshow("Debug Canvas", debug_canvas(canvas, boxes))
        return canvas, boxes

    # ------------------------------------------------------------------
    # Recognizer
    # ------------------------------------------------------------------

    def _recognize_with_boxes(
        self,
        canvas: np.ndarray,
        boxes: list[list[int]],
        n_expected: int,
        reader: easyocr.Reader,
    ) -> list[tuple[str, float]]:
        """
        Run EasyOCR recognition on a canvas with pre-known bounding boxes.
        Skips CRAFT detection; batches CRNN over all boxes in one GPU call.
        Results are matched back to input order by x_min position.
        """
        results = reader.recognize(
            canvas,
            horizontal_list=boxes,
            free_list=[],
            allowlist="0123456789",
            batch_size=n_expected,
        )

        if results:
            results = sorted(results, key=lambda r: r[0][0][0])

        output = []
        for i in range(n_expected):
            if i < len(results) and results[i]:
                _, text, conf = results[i]
                output.append((text, float(conf)))
            else:
                output.append(("", 0.0))
        return output

    # ------------------------------------------------------------------
    # Seven-segment helpers
    # ------------------------------------------------------------------

    def _segment_digits(
        self,
        binary: np.ndarray,
        max_digits: int = 2,
        open_kernel: tuple[int, int] = (3, 1),
        min_area_fraction: float = 0.05,
        height_tol: float = 0.6,
    ) -> list[tuple[int, int, int, int]]:
        """
        Segment digits using morphological opening followed by contour detection.
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


# ---------------------------------------------------------------------------
# Debug utilities
# ---------------------------------------------------------------------------


def debug_canvas(canvas: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """
    Annotate a canvas with bounding boxes and indices for debugging.
    Returns a BGR image suitable for cv2.imshow.
    """
    vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for i, (x_min, x_max, y_min, y_max) in enumerate(boxes):
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(
            vis,
            str(i),
            (x_min + 2, y_min + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
    return vis
