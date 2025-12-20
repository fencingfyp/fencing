from typing import Dict, Tuple

import cv2
import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.KeypointTarget import KeypointTarget
from src.model.tracker.TargetTracker import TargetTracker

MIN_INLIERS = 20  # lower threshold; tune to your use case


class SiftTarget(KeypointTarget):
    def __init__(
        self,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        sift=None,
        matcher=None,
        exclude_regions: list[Quadrilateral] | None = None,
    ):
        """
        exclude_regions: list of (x, y, w, h) rectangles to ignore
        """
        self.sift = sift or cv2.SIFT_create()

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = matcher or cv2.FlannBasedMatcher(index_params, search_params)

        frame = self._gaussian_preblur(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = KeypointTarget.build_mask(gray.shape, initial_positions, exclude_regions)

        # --- compute keypoints only in allowed areas ---
        self.kp_ref, self.des_ref = self.sift.detectAndCompute(gray, mask)

        self.points = KeypointTarget.get_reference_src_pts(self.kp_ref)
        self.initial_positions = initial_positions
        self.last_quad = self.initial_positions.copy()

    def get_points(self) -> np.ndarray:
        """Returns the currently tracked planar keypoints in the latest frame.
        Shape: (N, 1, 2)"""
        return self.points

    def _gaussian_preblur(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int] = (3, 3),
        sigma: float = 1.5,
    ) -> np.ndarray:
        # return image
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def get_previous_quad(self) -> Quadrilateral:
        return self.last_quad

    def update(self, frame: np.ndarray) -> Quadrilateral | None:
        frame = self._gaussian_preblur(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        if des_frame is None or len(kp_frame) == 0:
            print("No descriptors found in frame.")
            return None

        matches = self.matcher.knnMatch(self.des_ref, des_frame, k=2)
        good = []
        # use Lowe's ratio test
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < MIN_INLIERS:
            print("Not enough good matches:", len(good))
            return None

        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            print("Homography could not be computed or not enough inliers.")
            return None

        inlier_mask = mask.ravel().astype(bool)
        planar_dst_pts = dst_pts[inlier_mask]
        self.points = planar_dst_pts

        inliers = int(mask.sum())
        if inliers < MIN_INLIERS:
            print("Not enough inliers after RANSAC:", inliers)
            return None
        next_quad = cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        self.last_quad = Quadrilateral.from_opencv_format(next_quad)
        return self.last_quad


class SiftTracker(TargetTracker):
    def __init__(self) -> None:
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.targets: Dict[str, SiftTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] = None,
    ) -> None:
        self.targets[name] = SiftTarget(
            frame,
            initial_positions,
            self.sift,
            self.matcher,
            exclude_regions,
        )

    def update_all(self, frame: np.ndarray) -> Dict[str, Quadrilateral]:
        outputs: Dict[str, Quadrilateral] = {}
        for name, tgt in self.targets.items():
            outputs[name] = tgt.update(frame)
        return outputs

    def get_target_pts(self, name: str) -> np.ndarray:
        target = self.targets.get(name, None)
        if target is None:
            return None
        return target.get_points()

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        target = self.targets.get(name, None)
        if target is None:
            return None
        return target.get_previous_quad()
