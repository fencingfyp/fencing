from typing import Tuple

import cv2
import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.KeypointTarget import KeypointTarget
from src.model.tracker.TargetTracker import TargetTracker

MIN_INLIERS = 20  # tune as needed


class OrbTarget(KeypointTarget):
    def __init__(
        self,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
        orb=None,
        bf=None,
    ):
        # ORB + matcher
        self.orb = orb or cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        frame = self._gaussian_preblur(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- build mask for excluded regions ---
        mask = KeypointTarget.build_mask(gray.shape, initial_positions, exclude_regions)

        # detect using mask
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(gray, mask)

        self.points = KeypointTarget.get_reference_src_pts(self.kp_ref)
        self.initial_positions = initial_positions
        self.last_quad = initial_positions.copy()

    def get_previous_quad(self) -> Quadrilateral:
        return self.last_quad

    def get_points(self) -> np.ndarray:
        return self.points

    def _gaussian_preblur(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int] = (7, 7),
        sigma: float = 3,
    ) -> np.ndarray:
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def update(self, frame: np.ndarray) -> Quadrilateral:
        frame = self._gaussian_preblur(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) == 0:
            print("No descriptors found in frame.")
            return None

        matches = self.bf.knnMatch(self.des_ref, des_frame, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < MIN_INLIERS:
            return None

        # optional but recommended
        good = sorted(good, key=lambda m: m.distance)

        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )

        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("Homography could not be computed.")
            return None
        inliers = mask.sum()
        if inliers < MIN_INLIERS:
            print("Not enough inliers after RANSAC:", inliers)
            return None

        inlier_mask = mask.ravel().astype(bool)
        planar_dst_pts = dst_pts[inlier_mask]
        self.points = planar_dst_pts

        next_quad = cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        self.last_quad = Quadrilateral.from_opencv_format(next_quad)

        return self.last_quad


class OrbTracker(TargetTracker):
    def __init__(self):
        # share one ORB + BFMatcher across all targets
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.targets: dict[str, OrbTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
    ):
        target = OrbTarget(frame, initial_positions, exclude_regions, self.orb, self.bf)
        self.targets[name] = target

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral]:
        outputs = {}
        for name, target in self.targets.items():
            outputs[name] = target.update(frame)
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
