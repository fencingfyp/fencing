import cv2
import numpy as np

from src.model import Quadrilateral


class SinglePatchAutoDetector:
    """
    Detects a single patch colour based on a positive sample and
    a negative sample for baseline brightness.
    Robust against lighting changes and noise.
    """

    def __init__(
        self,
        pos_frame: np.ndarray,
        pos_patch: Quadrilateral,
        neg_frame: np.ndarray,
        neg_patch: Quadrilateral,
        L_thresh_scale: float = 0.5,
        chroma_ratio_thresh: float = 0.015,
        dist_thresh: float = 10.0,
    ):
        # learn colour model from positive patch (PIXELS, not mean)
        L_pos, ab_pixels, chroma_pos = self.lab_stats_pixels(pos_frame, pos_patch)
        self.mu_ab = np.mean(ab_pixels, axis=0)

        cov = np.cov(ab_pixels.T) + np.eye(2) * 1e-3
        self.cov_inv = np.linalg.inv(cov)

        self.baseline_chroma = chroma_pos

        # learn baseline luminosity from negative patch
        L_neg, _, _ = self.lab_stats_pixels(neg_frame, neg_patch)
        self.baseline_L = L_neg

        # thresholds
        self.L_thresh = max(L_thresh_scale * abs(L_pos - L_neg), 5.0)
        self.chroma_ratio_thresh = chroma_ratio_thresh
        self.dist_thresh = dist_thresh

        self.debug_info = None

    def lab_stats_pixels(self, frame: np.ndarray, patch_pts: Quadrilateral):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [patch_pts.numpy().astype(np.int32)], 255)

        if not np.any(mask):
            raise ValueError("Empty ROI")

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = cv2.split(lab)
        region = mask.astype(bool)

        if not np.any(region):
            raise ValueError("No valid pixels in ROI")

        chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)

        # robust stats (avoid means)
        L_stat = np.percentile(L[region], 75)
        chroma_stat = np.percentile(chroma[region], 75)

        ab_pixels = np.stack([a[region], b[region]], axis=1)

        return L_stat, ab_pixels, chroma_stat

    @staticmethod
    def mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
        d = x - mu
        return float(d.T @ cov_inv @ d)

    def get_debug_info(self):
        return self.debug_info

    def classify(self, frame: np.ndarray, patch_pts: Quadrilateral) -> bool:
        L, ab_pixels, chroma = self.lab_stats_pixels(frame, patch_pts)
        ab_mean = np.mean(ab_pixels, axis=0)

        # brightness gate (symmetric & forgiving)
        if abs(L - self.baseline_L) < self.L_thresh:
            self.debug_info = (
                f"Brightness gate failed: L={L:.1f}, "
                f"baseline_L={self.baseline_L:.1f}, thresh={self.L_thresh:.1f}"
            )
            return False

        # chroma relative to luminance (lighting invariant)
        chroma_ratio = chroma / (L + 1e-3)
        if chroma_ratio < self.chroma_ratio_thresh:
            self.debug_info = (
                f"Chroma ratio failed: ratio={chroma_ratio:.4f}, "
                f"thresh={self.chroma_ratio_thresh:.4f}"
            )
            return False

        # colour distance
        dist = self.mahalanobis(ab_mean, self.mu_ab, self.cov_inv)
        if dist > self.dist_thresh:
            self.debug_info = f"Colour mismatch: dist={dist:.2f}, " f"ab={ab_mean}"
            return False

        self.debug_info = (
            f"Patch detected: L={L:.1f}, "
            f"ab={ab_mean}, chroma={chroma:.1f}, "
            f"ratio={chroma_ratio:.4f}, dist={dist:.2f}"
        )
        return True

    def get_lab_values(self, frame: np.ndarray, patch_pts: Quadrilateral):
        L, ab_pixels, _ = self.lab_stats_pixels(frame, patch_pts)
        ab_mean = np.mean(ab_pixels, axis=0)
        return int(L), int(ab_mean[0]), int(ab_mean[1])
