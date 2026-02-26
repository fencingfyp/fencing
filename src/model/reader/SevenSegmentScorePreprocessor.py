from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np


@dataclass
class PreprocessorConfig:
    otsu_ratio: float = (
        0.8  # interpolation between otsu threshold and bright peak: 0=otsu, 1=bright peak
    )
    output_size: tuple = (64, 100)  # (H, W) fed to the model
    padding: int = 4  # pixels of padding around tight crop
    min_component_area: int = 20  # ignore tiny noise blobs when tight-cropping
    min_foreground_fraction: float = (
        0.07  # at least these percent of the crop must be kept
    )


class SevenSegmentScorePreprocessor:
    """
    Binarises a cropped seven-segment ROI and prepares it for model inference/training.

    Threshold = otsu_threshold + ratio * (bright_peak - otsu_threshold)
      - ratio=0.0 -> pure Otsu (wider/fatter digits, blob risk)
      - ratio=1.0 -> bright peak (very thin, segment-gap risk)
      - ratio~0.3-0.5 is a reasonable starting point
    """

    def __init__(self, config: PreprocessorConfig = None):
        self.cfg = config or PreprocessorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)

    def process(
        self,
        images: Union[np.ndarray, list],
        otsu_ratio: float = None,
    ) -> Union[np.ndarray, list]:
        """
        Process one image or a list of images.

        Args:
            images:     Single HxW or HxWxC ndarray, or a list of them.
            otsu_ratio: Override config value for this call (useful for augmentation).

        Returns:
            Single processed image (HxW uint8) or list thereof, resized to output_size.
        """
        ratio = otsu_ratio if otsu_ratio is not None else self.cfg.otsu_ratio
        single = isinstance(images, np.ndarray)
        imgs = [images] if single else images
        results = [self._process_one(img, ratio) for img in imgs]
        return results[0] if single else results

    def process_batch_ratios(
        self,
        image: np.ndarray,
        ratios: list,
    ) -> list:
        """
        Process a single image at multiple thresholds.
        Useful for threshold augmentation during training: call with e.g.
        [0.3, 0.5, 0.7] to get three variants without re-running Otsu.
        """
        return [self._process_one(image, r) for r in ratios]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_one_steps(
        self, image: np.ndarray, ratio: float
    ) -> list[tuple[str, np.ndarray]]:
        """
        Runs the full pipeline and returns each intermediate step as a
        (step_name, image) tuple. Used by both _process_one and _process_one_debug
        to avoid duplicating pipeline logic.
        """
        steps = [("original", image.copy())]

        subtracted = self._subtract_background(image)
        steps.append(("bg_subtracted", subtracted))

        contrasted = self._enhance_contrast_bgr(subtracted)
        steps.append(("contrasted", contrasted))

        gray = self._to_gray(contrasted)
        steps.append(("gray", gray))

        resized = self._pad_to_output_size(gray)
        steps.append(("output", resized))

        return steps

    def _process_one(self, image: np.ndarray, ratio: float) -> np.ndarray:
        return self._process_one_steps(image, ratio)[-1][1]

    def _process_one_debug(
        self, image: np.ndarray, ratio: float
    ) -> list[tuple[str, np.ndarray]]:
        return self._process_one_steps(image, ratio)

    def _subtract_background(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate the dark background level and subtract it from all channels.
        Assumes the darkest pixels are background — valid for seven-segment
        displays where segments are bright against a dark surround.
        Subtracting the floor increases effective contrast before binarisation
        without rescaling, preserving relative intensity differences between segments.
        """
        result = np.zeros_like(image)
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            floor = np.percentile(
                channel, 5
            )  # dark floor: bottom 5% assumed background
            result[:, :, c] = np.clip(channel - floor, 0, 255).astype(np.uint8)
        return result

    def _enhance_contrast_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply global contrast stretching independently to each BGR channel.
        Linearly remaps each channel so its meaningful intensity range fills
        0-255, improving separation between dim digit strokes and background
        without CLAHE's local amplification which can worsen bloomed regions.
        The percentile clip discards outlier pixels (noise, hot pixels) before
        stretching so they don't compress the useful range.
        """
        result = np.zeros_like(image)
        for c in range(3):
            channel = image[:, :, c]
            lo = np.percentile(channel, 1)  # clip bottom 1% as noise floor
            hi = np.percentile(channel, 99)  # clip top 1% to avoid bloom dominating
            if hi > lo:
                stretched = (channel.astype(np.float32) - lo) / (hi - lo) * 255
                result[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
            else:
                result[:, :, c] = channel
        return result

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to boost local contrast before binarisation.
        Particularly helps recover dim strokes like the leading '1' in two-digit
        numbers, which global Otsu tends to lose when the adjacent digit is brighter.
        clipLimit controls how aggressively local contrast is boosted — too high
        and noise gets amplified; tileGridSize should be small relative to crop size.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)

    def _detect_dominant_channel(self, image: np.ndarray) -> int:
        """
        Return the channel index with the highest contrast in the central region,
        excluding the channel with the highest mean — which is likely the blooming
        channel (e.g. red flare) rather than the most informative one.
        """
        h, w = image.shape[:2]
        cy1, cy2 = h // 4, 3 * h // 4
        cx1, cx2 = w // 4, 3 * w // 4
        center = image[cy1:cy2, cx1:cx2]

        channel_means = [center[:, :, c].mean() for c in range(3)]
        channel_stds = [center[:, :, c].std() for c in range(3)]

        # Exclude the brightest channel (most likely to be the bloom source)
        # then pick the highest std among the remaining two
        bloomed_channel = int(np.argmax(channel_means))
        candidates = [i for i in range(3) if i != bloomed_channel]
        return max(candidates, key=lambda i: channel_stds[i])

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
        dominant = self._detect_dominant_channel(image)
        return image[:, :, dominant]

    def _visualise_histogram(
        self,
        hist: np.ndarray,
        otsu: int,
        bright_peak: int,
        left_edge: int,
        threshold: int,
    ) -> None:
        canvas_h, canvas_w = 300, 512
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Normalise histogram to fit canvas height, excluding the zero bin which
        # often dwarfs everything else and makes the digit-region detail invisible
        hist_display = hist.copy()
        hist_display[0] = 0
        max_val = hist_display.max()
        if max_val > 0:
            hist_display = (hist_display / max_val * (canvas_h - 20)).astype(np.int32)

        # Draw histogram bars
        for i in range(256):
            x = int(i * canvas_w / 256)
            x_next = int((i + 1) * canvas_w / 256)
            bar_h = hist_display[i]
            cv2.rectangle(
                canvas,
                (x, canvas_h - bar_h),
                (x_next, canvas_h),
                (180, 180, 180),
                -1,
            )

        def draw_vline(val, colour, label):
            x = int(val * canvas_w / 256)
            cv2.line(canvas, (x, 0), (x, canvas_h), colour, 1)
            cv2.putText(
                canvas,
                f"{label}:{val}",
                (x + 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                colour,
                1,
                cv2.LINE_AA,
            )

        draw_vline(otsu, (255, 100, 0), "O")  # blue  — Otsu
        draw_vline(bright_peak, (0, 100, 255), "P")  # red   — bright peak
        draw_vline(left_edge, (0, 165, 255), "L")  # orange — left edge
        draw_vline(threshold, (0, 255, 0), "T")  # green — threshold

        # Shade ratio range
        x0 = int(otsu * canvas_w / 256)
        x1 = int(left_edge * canvas_w / 256)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, 0), (x1, canvas_h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, canvas, 0.9, 0, canvas)

        cv2.imshow("Histogram  O=Otsu  P=Peak  L=LeftEdge  T=Threshold", canvas)
        cv2.moveWindow("Histogram  O=Otsu  P=Peak  L=LeftEdge  T=Threshold", 100, 100)

    def _tight_crop(self, binary: np.ndarray) -> np.ndarray:
        """
        Crop to the bounding box of all foreground pixels,
        ignoring tiny noise components, then add padding.
        """
        # Filter out small noise blobs before computing bounding box,
        # so a stray pixel far from the digit doesn't inflate the crop region.
        # clean = self._remove_small_components(binary)

        coords = cv2.findNonZero(binary)
        if coords is None:
            # Fallback: nothing survived filtering, use original
            coords = cv2.findNonZero(binary)
        if coords is None:
            # Totally blank image — return as-is
            return binary

        x, y, w, h = cv2.boundingRect(coords)
        p = self.cfg.padding
        H, W = binary.shape
        x1 = max(x - p, 0)
        y1 = max(y - p, 0)
        x2 = min(x + w + p, W)
        y2 = min(y + h + p, H)
        return binary[y1:y2, x1:x2]

    def _pad_to_output_size(self, binary: np.ndarray) -> np.ndarray:
        """
        Scale the image to fit within output_size while preserving aspect ratio,
        then pad with black to reach exactly output_size. This replaces the
        separate _pad_to_aspect and resize steps — do not call cv2.resize after this.
        """
        target_h, target_w = self.cfg.output_size
        h, w = binary.shape

        # Scale to fit within the target canvas, preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        scaled = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad symmetrically to reach exactly output_size
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        return cv2.copyMakeBorder(
            scaled,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    def _remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        clean = np.zeros_like(binary)
        for label in range(1, num_labels):  # 0 is background
            if stats[label, cv2.CC_STAT_AREA] >= self.cfg.min_component_area:
                clean[labels == label] = 255
        return clean
