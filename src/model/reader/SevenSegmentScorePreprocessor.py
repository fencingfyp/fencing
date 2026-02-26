from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np


@dataclass
class PreprocessorConfig:
    output_size: tuple = (64, 100)  # (H, W) fed to the model


class SevenSegmentScorePreprocessor:

    def __init__(self, config: PreprocessorConfig = None):
        self.cfg = config or PreprocessorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)

    def process(self, images: Union[np.ndarray, list]) -> Union[np.ndarray, list]:
        """
        Process one image or a list of images.

        Args:
            images:     Single HxW or HxWxC ndarray, or a list of them.
            otsu_ratio: Override config value for this call (useful for augmentation).

        Returns:
            Single processed image (HxW uint8) or list thereof, resized to output_size.
        """

        single = isinstance(images, np.ndarray)
        imgs = [images] if single else images
        results = [self._process_one(img) for img in imgs]
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
        self,
        image: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Runs the full pipeline and returns each intermediate step as a
        (step_name, image) tuple. Used by both _process_one and _process_one_debug
        to avoid duplicating pipeline logic.
        """
        steps = [("original", image.copy())]

        resized = self._pad_to_output_size(image)
        steps.append(("output", resized))

        return steps

    def _process_one(self, image: np.ndarray) -> np.ndarray:
        return self._process_one_steps(image)[-1][1]

    def _process_one_debug(self, image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        return self._process_one_steps(image)

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

    def _pad_to_output_size(self, img: np.ndarray) -> np.ndarray:
        """
        Scale to fit within output_size preserving aspect ratio,
        then pad with black to reach exactly output_size.
        """
        if img.ndim == 2:
            return self._pad_to_output_size_binary(img)

        target_h, target_w = self.cfg.output_size
        h, w = img.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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
            value=(0, 0, 0),
        )

    def _pad_to_output_size_binary(self, img: np.ndarray) -> np.ndarray:
        """
        Scale the image to fit within output_size while preserving aspect ratio,
        then pad with black to reach exactly output_size. This replaces the
        separate _pad_to_aspect and resize steps — do not call cv2.resize after this.
        """
        target_h, target_w = self.cfg.output_size
        h, w = img.shape

        # Scale to fit within the target canvas, preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

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
