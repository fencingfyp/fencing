from typing import Optional

import cv2
import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.KeypointTarget import KeypointTarget
from src.model.tracker.TargetTracker import TargetTracker

MIN_INLIERS = 20


class OrbTarget(KeypointTarget):
    def __init__(
        self,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None,
        orb: cv2.ORB,
        bf: cv2.BFMatcher,
    ):
        self.orb = orb
        self.bf = bf

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # build mask for excluded regions
        mask = KeypointTarget.build_mask(gray.shape, initial_positions, exclude_regions)

        self.kp_ref, self.des_ref = self.orb.detectAndCompute(gray, mask)
        if self.des_ref is None or len(self.kp_ref) == 0:
            raise RuntimeError("No reference descriptors found for target")

        self.points = KeypointTarget.get_reference_src_pts(self.kp_ref)
        self.initial_positions = initial_positions
        self.last_quad = initial_positions.copy()
        self.last_inliers: int | None = None

        # limit number of reference keypoints for performance
        # MAX_REF = 500
        # if len(self.kp_ref) > MAX_REF:
        #     idx = np.random.choice(len(self.kp_ref), MAX_REF, replace=False)
        #     self.kp_ref = [self.kp_ref[i] for i in idx]
        #     self.des_ref = self.des_ref[idx]

    def get_previous_quad(self) -> Quadrilateral:
        return self.last_quad

    def get_points(self) -> np.ndarray:
        return self.points

    # ---- spatial filtering helpers ----
    def _compute_roi(self) -> Quadrilateral | None:
        """
        Decide ROI size based on last tracking confidence.
        Return None to indicate global search.
        """
        if self.last_inliers is None:
            return None

        if self.last_inliers > 120:
            expand_px = 40
        elif self.last_inliers > 60:
            expand_px = 120
        else:
            return None  # recovery mode → global search

        return self.last_quad.expand(expand_px, expand_px)

    @staticmethod
    def _filter_keypoints_by_roi(kp, des, roi: Quadrilateral):
        if roi is None:
            return kp, des

        x, y, w, h = roi.to_xywh()
        kp_filt = []
        des_filt = []
        for i, k in enumerate(kp):
            px, py = k.pt
            if x <= px <= x + w and y <= py <= y + h:
                kp_filt.append(k)
                des_filt.append(des[i])
        if not kp_filt:
            return None, None

        return kp_filt, np.asarray(des_filt)

    # ---- update using shared frame features ----
    def update_with_features(
        self,
        kp_frame,
        des_frame,
    ) -> Optional[Quadrilateral]:

        if des_frame is None or len(kp_frame) == 0:
            return None

        # ---- adaptive spatial filtering ----
        roi = self._compute_roi()
        kp_use, des_use = self._filter_keypoints_by_roi(kp_frame, des_frame, roi)

        # fallback to global search if ROI filtering failed
        if kp_use is None or des_use is None:
            kp_use, des_use = kp_frame, des_frame

        matches = self.bf.knnMatch(self.des_ref, des_use, k=2)

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < MIN_INLIERS:
            return None

        # Optional cap for RANSAC stability
        if len(good) > 300:
            good = good[:300]

        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )

        dst_pts = np.float32([kp_use[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inliers = int(mask.sum())
        self.last_inliers = inliers
        if inliers < MIN_INLIERS:
            return None

        inlier_mask = mask.ravel().astype(bool)
        self.points = dst_pts[inlier_mask]

        next_quad = cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        self.last_quad = Quadrilateral.from_opencv_format(next_quad)

        return self.last_quad


class OrbTracker(TargetTracker):
    def __init__(self):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.targets: dict[str, OrbTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
    ):
        target = OrbTarget(
            frame=frame,
            initial_positions=initial_positions,
            exclude_regions=exclude_regions,
            orb=self.orb,
            bf=self.bf,
        )
        self.targets[name] = target

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        # preprocess once per frame
        # frame = cv2.GaussianBlur(frame, (7, 7), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)

        outputs = {}
        for name, target in self.targets.items():
            outputs[name] = target.update_with_features(kp_frame, des_frame)

        return outputs

    def get_target_pts(self, name: str) -> np.ndarray | None:
        target = self.targets.get(name)
        if target is None:
            return None
        return target.get_points()

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        target = self.targets.get(name)
        if target is None:
            return None
        return target.get_previous_quad()
