import random
from dataclasses import dataclass

import cv2
import numpy as np

from src.model.reader.SevenSegmentScorePreprocessor import SevenSegmentScorePreprocessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    # Rotation
    rotation_max_deg: float = 5.0
    rotation_p: float = 0.3616461765929567

    # Perspective warp
    perspective_max_shift: float = 0.1  # fraction of image dimension
    perspective_p: float = 0.43

    # Crop and border padding
    # Random crop simulates overcropping; border padding simulates loose ROI
    crop_max_fraction: float = 0.05  # max fraction to crop from any edge
    pad_max_fraction: float = 0.10  # max fraction of image size to pad
    crop_p: float = 0.44
    pad_p: float = 0.64

    # Brightness and contrast:  out = alpha * in + beta
    brightness_max_delta: float = 43.0  # beta range: [-delta, +delta]
    contrast_range: tuple = (0.7, 1.4)  # alpha range
    brightness_contrast_p: float = 0.45

    # Gamma:  out = in ^ (1/gamma) — values >1 brighten midtones, <1 darken
    gamma_range: tuple = (0.76, 1.37)
    gamma_p: float = 0.468

    # Gaussian blur — kernel must be odd
    blur_kernel_size: int = 3
    blur_p: float = 0.3

    # Gaussian noise — simulates sensor noise and binarisation instability
    noise_std_range: tuple = (1.39, 18.44)
    noise_max_value: float = (
        180  # clip noise to this value to avoid false bright pixels that could skew binarisation
    )
    noise_p: float = 0.229

    # Colour jitter (hue/saturation shift in HSV space)
    # Covers different segment colours across display types
    hue_max_shift: int = 9  # degrees in [0, 180] OpenCV hue space
    saturation_range: tuple = (0.446, 1.58)
    colour_jitter_p: float = 0.4


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------


class SevenSegmentAugmenter:
    """
    Augmentation pipeline for seven-segment score crops stored as BGR uint8.

    Call augment(image) to get a single augmented copy.
    Each augmentation is applied independently with its configured probability,
    so the model also sees relatively clean samples rather than always receiving
    maximally distorted inputs.

    The preprocessor (binarisation, tight crop, aspect-ratio padding, resize)
    should be applied AFTER augmentation so that augmented variants are
    normalised to the same canvas before being fed to the model.
    """

    def __init__(self, config: AugmentationConfig = None):
        self.cfg = config or AugmentationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a random subset of augmentations to a single BGR uint8 image.
        Returns an augmented BGR uint8 image of the same spatial size.
        """
        img = image.copy()

        if self._roll(self.cfg.rotation_p):
            img = self._rotate(img)

        if self._roll(self.cfg.perspective_p):
            img = self._perspective_warp(img)

        if self._roll(self.cfg.crop_p):
            img = self._random_crop(img)

        if self._roll(self.cfg.pad_p):
            img = self._border_pad(img)

        if self._roll(self.cfg.colour_jitter_p):
            img = self._colour_jitter(img)

        if self._roll(self.cfg.brightness_contrast_p):
            img = self._brightness_contrast(img)

        if self._roll(self.cfg.gamma_p):
            img = self._gamma(img)

        if self._roll(self.cfg.blur_p):
            img = self._blur(img)

        if self._roll(self.cfg.noise_p):
            img = self._gaussian_noise(img)

        return img

    def augment_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Convenience wrapper to augment a list of images."""
        return [self.augment(img) for img in images]

    # ------------------------------------------------------------------
    # Individual augmentations
    # ------------------------------------------------------------------

    def _rotate(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        angle = random.uniform(-self.cfg.rotation_max_deg, self.cfg.rotation_max_deg)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # Border reflect avoids hard black edges that could confuse binarisation
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _perspective_warp(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        shift = self.cfg.perspective_max_shift

        def rand_shift():
            return random.uniform(-shift, shift)

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32(
            [
                [w * rand_shift(), h * rand_shift()],
                [w + w * rand_shift(), h * rand_shift()],
                [w + w * rand_shift(), h + h * rand_shift()],
                [w * rand_shift(), h + h * rand_shift()],
            ]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly remove a small fraction from each edge, then resize back to
        original dimensions. Simulates a tighter-than-intended ROI crop that
        may clip digit edges.
        """
        h, w = img.shape[:2]
        f = self.cfg.crop_max_fraction
        top = int(random.uniform(0, f) * h)
        bottom = int(random.uniform(0, f) * h)
        left = int(random.uniform(0, f) * w)
        right = int(random.uniform(0, f) * w)

        # Guard against degenerate crop
        y1, y2 = top, max(h - bottom, top + 1)
        x1, x2 = left, max(w - right, left + 1)
        cropped = img[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _border_pad(self, img: np.ndarray) -> np.ndarray:
        """
        Add random padding on each edge filled with the mean border colour,
        then resize back. Simulates a looser-than-intended ROI that includes
        background pixels around the digit.
        """
        h, w = img.shape[:2]
        f = self.cfg.pad_max_fraction
        top = int(random.uniform(0, f) * h)
        bottom = int(random.uniform(0, f) * h)
        left = int(random.uniform(0, f) * w)
        right = int(random.uniform(0, f) * w)

        # Use mean border colour rather than black so padding does not produce
        # hard edges that artificially skew the binarisation threshold
        border_colour = [int(img[:, :, c].mean()) for c in range(img.shape[2])]
        padded = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=border_colour,
        )
        return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)

    def _colour_jitter(self, img: np.ndarray) -> np.ndarray:
        """Shift hue and scale saturation in HSV space."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (
            hsv[:, :, 0]
            + random.randint(-self.cfg.hue_max_shift, self.cfg.hue_max_shift)
        ) % 180
        sat_scale = random.uniform(*self.cfg.saturation_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply alpha * img + beta, clipped to [0, 255]."""
        alpha = random.uniform(*self.cfg.contrast_range)
        beta = random.uniform(
            -self.cfg.brightness_max_delta, self.cfg.brightness_max_delta
        )
        return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    def _gamma(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction via a lookup table.
        gamma > 1 brightens midtones; gamma < 1 darkens them.
        Simulates different display brightness settings and camera exposure curves.
        """
        gamma = random.uniform(*self.cfg.gamma_range)
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(img, lut)

    def _blur(self, img: np.ndarray) -> np.ndarray:
        k = self.cfg.blur_kernel_size
        assert k % 2 == 1, "Blur kernel size must be odd"
        return cv2.GaussianBlur(img, (k, k), 0)

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        std = random.uniform(*self.cfg.noise_std_range)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(
            img.astype(np.float32) + noise, 0, self.cfg.noise_max_value
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _roll(p: float) -> bool:
        return random.random() < p


import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="Augment and process seven-segment score crops"
    )
    parser.add_argument(
        "folder",
        help="Path to main folder containing video subfolders (each with 0-15 label subfolders)",
    )
    parser.add_argument(
        "--class",
        dest="label",
        type=int,
        default=None,
        help="Fix to a specific class (0-15). If omitted, samples randomly.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Fix to a specific video subfolder name. If omitted, samples across all videos.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Upscale factor for display (default: 4.0)",
    )
    return parser.parse_args()


def collect_images(
    root: Path,
    label: int | None,
    video: str | None,
) -> list[tuple[Path, int]]:
    """
    Return a list of (image_path, class_id) tuples from root/video/class/*.png.
    Filters by label and/or video name if specified.
    """
    samples = []
    video_dirs = [root / video] if video else [d for d in root.iterdir() if d.is_dir()]

    for video_dir in video_dirs:
        if not video_dir.is_dir():
            raise ValueError(f"Video folder not found: {video_dir}")
        label_range = [label] if label is not None else range(16)
        for cls in label_range:
            label_dir = video_dir / str(cls)
            if label_dir.is_dir():
                for p in label_dir.glob("*.png"):
                    samples.append((p, cls))

    if not samples:
        raise ValueError(
            f"No images found in {root} "
            f"(video={video or 'any'}, class={label if label is not None else 'any'})"
        )
    return samples


def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h = img.shape[0]
    if h == target_h:
        return img
    return cv2.copyMakeBorder(
        img, 0, target_h - h, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    w = img.shape[1]
    if w == target_w:
        return img
    return cv2.copyMakeBorder(
        img, 0, 0, 0, target_w - w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def upscale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST
    )


def to_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


if __name__ == "__main__":
    args = get_parse_args()
    root = Path(args.folder)
    samples = collect_images(root, args.label, args.video)

    aug = SevenSegmentAugmenter()
    pre = SevenSegmentScorePreprocessor()

    while True:
        img_path, cls = random.choice(samples)
        img = cv2.imread(str(img_path))

        augmented = aug.augment(img)
        preprocessed_list = pre._process_one_debug(augmented, 0.8)

        # Ensure all panels are BGR for stacking
        img_bgr = to_bgr(img)
        # aug_bgr = to_bgr(augmented)
        pre_bgr_list = [to_bgr(p) for n, p in preprocessed_list]
        augmented = pre_bgr_list[0]

        # Pad to common height and width before stacking
        max_h = max(img_bgr.shape[0], *[p.shape[0] for p in pre_bgr_list])
        max_w = max(img_bgr.shape[1], *[p.shape[1] for p in pre_bgr_list])

        panels = [
            pad_to_width(pad_to_height(p, max_h), max_w)
            for p in (img_bgr, *pre_bgr_list)
        ]

        combined = upscale(np.hstack(panels), args.scale)

        # Encode class and filename in the window title
        window_title = f"class={cls}  file={img_path.name}  | any key=next  q=quit"
        cv2.imshow(window_title, combined)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            break
