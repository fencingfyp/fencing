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
        orb: cv2.ORB,
        bf: cv2.BFMatcher,
        mask_margin: float = 0.2,
        detection_scale: float = 0.5,
    ):
        self.orb = orb
        self.bf = bf
        self.mask_margin = mask_margin
        self.detection_scale = detection_scale
        self.frame_size = (frame.shape[1], frame.shape[0])  # full-res (w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        small = self._downscale(gray)

        # Build mask at detection resolution so it aligns with downscaled keypoints
        small_quad = self._scale_quad(initial_positions, detection_scale)
        mask = KeypointTarget.build_mask(small.shape, small_quad, None, mask_margin)

        self.kp_ref, self.des_ref = self.orb.detectAndCompute(small, mask)
        if self.des_ref is None or len(self.kp_ref) == 0:
            raise RuntimeError("No reference descriptors found for target")

        # Scale reference keypoints back to full-frame space so they are
        # comparable to frame keypoints (which are also scaled up after detection)
        self._upscale_keypoints(self.kp_ref)

        # Cap reference keypoints by response strength to bound knnMatch cost
        MAX_REF = 500
        if len(self.kp_ref) > MAX_REF:
            responses = np.array([k.response for k in self.kp_ref])
            idx = np.argsort(responses)[::-1][:MAX_REF]
            self.kp_ref = [self.kp_ref[i] for i in idx]
            self.des_ref = self.des_ref[idx]

        self.points = KeypointTarget.get_reference_src_pts(self.kp_ref)
        self.initial_positions = initial_positions
        self.last_quad = initial_positions.copy()
        self.last_inliers: int | None = None

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def _downscale(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        return cv2.resize(
            gray,
            (int(w * self.detection_scale), int(h * self.detection_scale)),
        )

    def _upscale_keypoints(self, kp: list) -> None:
        """Translate keypoint coordinates in-place from detection space to full-frame space."""
        inv = 1.0 / self.detection_scale
        for k in kp:
            k.pt = (k.pt[0] * inv, k.pt[1] * inv)

    @staticmethod
    def _scale_quad(quad: Quadrilateral, scale: float) -> Quadrilateral:
        return Quadrilateral((quad.numpy() * scale).astype(np.float32))

    # ------------------------------------------------------------------
    # Spatial filtering
    # ------------------------------------------------------------------

    def get_previous_quad(self) -> Quadrilateral:
        return self.last_quad

    def get_points(self) -> np.ndarray:
        return self.points

    def _compute_roi(self) -> Quadrilateral | None:
        if self.last_inliers is None:
            return None

        w, h = self.frame_size
        full_expand = int(min(w, h) * self.mask_margin)

        if self.last_inliers > 120:
            return self.last_quad.expand(full_expand, full_expand)
        if self.last_inliers > 60:
            return self.last_quad.expand(full_expand * 2, full_expand * 2)
        return None  # recovery — global search

    @staticmethod
    def _filter_keypoints_by_roi(
        kp: list, des: np.ndarray, roi: Quadrilateral
    ) -> tuple[list, np.ndarray] | tuple[None, None]:
        if roi is None or not kp:
            return kp, des

        x, y, w, h = roi.to_xywh()
        pts = np.array([k.pt for k in kp])
        mask = (
            (pts[:, 0] >= x)
            & (pts[:, 0] <= x + w)
            & (pts[:, 1] >= y)
            & (pts[:, 1] <= y + h)
        )
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None, None

        return [kp[i] for i in idx], des[idx]

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update_with_features(
        self,
        kp_frame: list,
        des_frame: np.ndarray,
    ) -> Optional[Quadrilateral]:
        if des_frame is None or not kp_frame:
            return None

        roi = self._compute_roi()
        kp_use, des_use = self._filter_keypoints_by_roi(kp_frame, des_frame, roi)
        if kp_use is None:
            kp_use, des_use = kp_frame, des_frame

        matches = self.bf.knnMatch(self.des_ref, des_use, k=2)

        good = [
            m
            for pair in matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.70 * n.distance
        ]

        if len(good) < MIN_INLIERS:
            return None

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

        self.points = dst_pts[mask.ravel().astype(bool)]
        self.last_quad = Quadrilateral.from_opencv_format(
            cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        )
        return self.last_quad


class OrbTracker(TargetTracker):
    def __init__(self, detection_scale: float = 0.5):
        self.orb = cv2.ORB_create(2000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.detection_scale = detection_scale
        self.targets: dict[str, OrbTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
        mask_margin: float = 0.2,
    ) -> None:
        self.targets[name] = OrbTarget(
            frame=frame,
            initial_positions=initial_positions,
            orb=self.orb,
            bf=self.bf,
            mask_margin=mask_margin,
            detection_scale=self.detection_scale,
        )

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        # Downscale for detection — cuts detectAndCompute cost by ~4x at scale=0.5
        h, w = gray.shape
        small = cv2.resize(
            gray,
            (int(w * self.detection_scale), int(h * self.detection_scale)),
        )
        kp_frame, des_frame = self.orb.detectAndCompute(small, None)

        # Scale keypoints back to full-frame space — reference keypoints are
        # also stored in full-frame space so coordinates are directly comparable
        inv = 1.0 / self.detection_scale
        for kp in kp_frame:
            kp.pt = (kp.pt[0] * inv, kp.pt[1] * inv)

        return {
            name: target.update_with_features(kp_frame, des_frame)
            for name, target in self.targets.items()
        }

    def get_target_pts(self, name: str) -> np.ndarray | None:
        t = self.targets.get(name)
        return t.get_points() if t else None

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        t = self.targets.get(name)
        return t.get_previous_quad() if t else None
