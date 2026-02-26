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
        orb: cv2.ORB,  # always full ORB for reference
        bf: cv2.BFMatcher,
        clahe: cv2.CLAHE,
        mask_margin: float = 0.2,
        detection_scale: float = 0.5,
    ):
        self.orb = orb
        self.bf = bf
        self.mask_margin = mask_margin
        self.detection_scale = detection_scale
        self.frame_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        small = self._downscale(gray)

        small_quad = self._scale_quad(initial_positions, detection_scale)
        mask = KeypointTarget.build_mask(small.shape, small_quad, None, mask_margin)

        self.kp_ref, self.des_ref = self.orb.detectAndCompute(small, mask)
        if self.des_ref is None or len(self.kp_ref) == 0:
            raise RuntimeError("No reference descriptors found for target")

        self._upscale_keypoints(self.kp_ref)

        MAX_REF = 500
        if len(self.kp_ref) > MAX_REF:
            responses = np.array([k.response for k in self.kp_ref])
            idx = np.argsort(responses)[::-1][:MAX_REF]
            self.kp_ref = [self.kp_ref[i] for i in idx]
            self.des_ref = self.des_ref[idx]

        self.ref_pts = np.float32([k.pt for k in self.kp_ref])
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
    def _filter_keypoints_by_roi(kp, des, kp_pts, roi):
        """kp_pts: pre-extracted (N,2) float32 array."""
        if roi is None or not kp:
            return kp, des, kp_pts

        x, y, w, h = roi.to_xywh()
        mask = (
            (kp_pts[:, 0] >= x)
            & (kp_pts[:, 0] <= x + w)
            & (kp_pts[:, 1] >= y)
            & (kp_pts[:, 1] <= y + h)
        )
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None, None, None

        return [kp[i] for i in idx], des[idx], kp_pts[idx]

    def update_with_features(self, kp_frame, des_frame, kp_pts):
        if des_frame is None or not kp_frame:
            return None

        roi = self._compute_roi()
        kp_use, des_use, kp_pts_use = self._filter_keypoints_by_roi(
            kp_frame, des_frame, kp_pts, roi
        )
        if kp_use is None:
            kp_use, des_use, kp_pts_use = kp_frame, des_frame, kp_pts

        matches = self.bf.knnMatch(self.des_ref, des_use, k=2)

        pairs = [p for p in matches if len(p) == 2]
        if not pairs:
            return None

        d = np.array([(p[0].distance, p[1].distance) for p in pairs], dtype=np.float32)
        q = np.array([p[0].queryIdx for p in pairs], dtype=np.int32)
        t = np.array([p[0].trainIdx for p in pairs], dtype=np.int32)

        ratio_mask = d[:, 0] < 0.70 * d[:, 1]
        n_good = int(ratio_mask.sum())
        if n_good < MIN_INLIERS:
            return None

        q = q[ratio_mask][:300]
        t = t[ratio_mask][:300]

        src_pts = self.ref_pts[q].reshape(-1, 1, 2)
        dst_pts = kp_pts_use[t].reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        mask_flat = mask.ravel().astype(bool)
        inliers = int(mask_flat.sum())
        self.last_inliers = inliers
        if inliers < MIN_INLIERS:
            return None

        self.points = dst_pts[mask_flat]
        self.last_quad = Quadrilateral.from_opencv_format(
            cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        )
        return self.last_quad

    def is_tracking_well(self) -> bool:
        return self.last_inliers is not None and self.last_inliers > 80

    def is_struggling(self) -> bool:
        return self.last_inliers is None or self.last_inliers < 50


class OrbTracker(TargetTracker):
    # Hysteresis thresholds for switching between full/lite detection
    _LITE_INLIERS_THRESHOLD = 80  # all targets must exceed this to switch to lite
    _FULL_INLIERS_THRESHOLD = 50  # any target below this forces switch back to full

    def __init__(self, detection_scale: float = 0.5):
        self.orb_full = cv2.ORB_create(
            2000, scaleFactor=1.2, nlevels=8, fastThreshold=10
        )
        self.orb_lite = cv2.ORB_create(
            1000,
            scaleFactor=1.2,
            nlevels=6,
            fastThreshold=15,
            scoreType=cv2.ORB_FAST_SCORE,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.detection_scale = detection_scale
        self.targets: dict[str, OrbTarget] = {}
        self._using_lite = False

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
        mask_margin: float = 0.2,
    ) -> None:
        # Always use full ORB for reference frame — best possible descriptors
        self.targets[name] = OrbTarget(
            frame=frame,
            initial_positions=initial_positions,
            orb=self.orb_full,
            bf=self.bf,
            clahe=self.clahe,
            mask_margin=mask_margin,
            detection_scale=self.detection_scale,
        )

    def _update_orb_mode(self) -> None:
        """Hysteresis switch — only flip mode when all targets agree."""
        if not self.targets:
            return
        targets = self.targets.values()
        if all(t.is_tracking_well() for t in targets):
            self._using_lite = True
        elif any(t.is_struggling() for t in targets):
            self._using_lite = False

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        self._update_orb_mode()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        h, w = gray.shape
        small = cv2.resize(
            gray, (int(w * self.detection_scale), int(h * self.detection_scale))
        )

        kp_frame, des_frame = self._detect(small)
        kp_frame, kp_pts = self._upscale_and_extract_pts(kp_frame)

        results = {}
        for name, target in self.targets.items():
            results[name] = target.update_with_features(kp_frame, des_frame, kp_pts)

        if self._using_lite and self._any_target_struggling(results):
            self._retry_struggling_targets(small, results)

        return results

    def _any_target_struggling(self, results: dict) -> bool:
        return any(
            r is None or self.targets[name].is_struggling()
            for name, r in results.items()
        )

    def _retry_struggling_targets(self, small: np.ndarray, results: dict) -> None:
        kp_full, des_full = self.orb_full.detectAndCompute(small, None)
        kp_full, kp_pts_full = self._upscale_and_extract_pts(kp_full)
        self._using_lite = False

        for name, target in self.targets.items():
            if results[name] is None or target.is_struggling():
                results[name] = target.update_with_features(
                    kp_full, des_full, kp_pts_full
                )

    def _upscale_and_extract_pts(self, kp_frame: list) -> tuple[list, np.ndarray]:
        inv = 1.0 / self.detection_scale
        for kp in kp_frame:
            kp.pt = (kp.pt[0] * inv, kp.pt[1] * inv)
        return kp_frame, np.float32([kp.pt for kp in kp_frame])

    def _detect(self, small: np.ndarray) -> tuple:
        orb = self.orb_lite if self._using_lite else self.orb_full
        return orb.detectAndCompute(small, None)

    def get_target_pts(self, name: str) -> np.ndarray | None:
        t = self.targets.get(name)
        return t.get_points() if t else None

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        t = self.targets.get(name)
        return t.get_previous_quad() if t else None
