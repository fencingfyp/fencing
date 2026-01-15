import enum

import cv2
import numpy as np

from src.model import Quadrilateral


class Colour(enum.Enum):
    RED = "red"
    GREEN = "green"
    WHITE = "white"


COLOURS_TO_LAB_DISTRIBUTIONS = {  # color_name -> (mu_ab, cov_inv)
    Colour.RED.value: (
        np.array([165, 150]),
        np.linalg.inv(np.diag([18**2, 8**2])),
    ),
    Colour.GREEN.value: (
        np.array([80, 150]),
        np.linalg.inv(np.diag([18**2, 15**2])),
    ),
    Colour.WHITE.value: (
        np.array([128, 128]),
        np.linalg.inv(np.diag([8**2, 8**2])),
    ),
}


class PatchLightDetector:
    @staticmethod
    def _convert_target_colors(
        target_colors: list[str] | str | Colour | list[Colour],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if isinstance(target_colors, str):
            target_colors = [target_colors]
        elif isinstance(target_colors, Colour):
            target_colors = [target_colors.value]
        elif all(isinstance(c, Colour) for c in target_colors):
            target_colors = [c.value for c in target_colors]

        return {color: COLOURS_TO_LAB_DISTRIBUTIONS[color] for color in target_colors}

    def __init__(
        self,
        target_colors: list[str] | str | Colour | list[Colour],
        L_thresh=15.0,
        dist_thresh=3.5,
        chroma_thresh=8.0,
        baseline_L=None,
    ):
        """
        target_colors: dict of color_name -> (mu_ab, cov_inv)
        Example:
        {
            "red":   (np.array([160, 150]), np.linalg.inv(np.diag([15**2, 15**2]))),
            "green": (np.array([90, 150]),  np.linalg.inv(np.diag([15**2, 15**2]))),
            "white": (np.array([128, 128]), np.linalg.inv(np.diag([8**2, 8**2]))),
        }
        """
        self.target_colors = self._convert_target_colors(target_colors)
        self.L_thresh = L_thresh
        self.dist_thresh = dist_thresh
        self.baseline_L = baseline_L  # can be set dynamically at init or first frame
        self.baseline_chroma = None
        self.debug_info = None
        self.chroma_thresh = chroma_thresh

    def get_debug_info(self) -> str | None:
        return self.debug_info

    def mean_lab_ab(self, frame: np.ndarray, patch_pts: Quadrilateral):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [patch_pts.numpy().astype(np.int32)], 255)

        if not np.any(mask):
            raise ValueError("Empty ROI")

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = cv2.split(lab)
        region = mask.astype(bool)

        # chroma per pixel (distance from neutral)
        chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)

        # reject saturated / unreliable pixels
        valid = region

        if not np.any(valid):
            raise ValueError("No valid pixels after filtering")

        Lm = np.mean(L[valid])
        ab_mean = np.array([np.mean(a[valid]), np.mean(b[valid])])

        chroma_mean = np.mean(chroma[valid])

        return Lm, ab_mean, chroma_mean

    @staticmethod
    def mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
        d = x - mu
        return float(d.T @ cov_inv @ d)

    def classify(self, frame: np.ndarray, patch_pts: Quadrilateral) -> Colour | None:
        L, ab, chroma = self.mean_lab_ab(frame, patch_pts)

        if self.baseline_L is None:
            self.baseline_L = L
            self.baseline_chroma = chroma
            return None

        # brightness gate
        if L - self.baseline_L < self.L_thresh:
            self.debug_info = f"L={L}, baseline_L={self.baseline_L} -> below threshold"
            print(self.debug_info)
            return None

        # chroma gate (reject skin / beige / weak colour)
        # if chroma - self.baseline_chroma < self.chroma_thresh:
        #     self.debug_info = (
        #         f"Chroma gate failed: chroma={chroma:.1f}, "
        #         f"baseline={self.baseline_chroma:.1f}"
        #     )
        #     print(self.debug_info)
        #     return None

        # distance to each colour model
        dists = {
            name: self.mahalanobis(ab, mu, cov_inv)
            for name, (mu, cov_inv) in self.target_colors.items()
        }

        label, dmin = min(dists.items(), key=lambda x: x[1])
        if dmin > self.dist_thresh:
            self.debug_info = f"No colour match found: {dists}, ab={ab}"
            print(self.debug_info)
            return None

        self.debug_info = f"L={L}, baseline_L={self.baseline_L}, ab={ab}, dists={dists}, chroma={chroma}"
        print(self.debug_info)

        return Colour(label)

    def get_lab_values(
        self, frame: np.ndarray, patch_pts: Quadrilateral
    ) -> tuple[float, float, float]:
        L, ab, chroma = self.mean_lab_ab(frame, patch_pts)
        return int(L), int(ab[0]), int(ab[1])
