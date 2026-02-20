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
        use_whole_frame: bool = False,
        use_akaze: bool = False,  # better for low-feature objects
    ):
        self.orb = orb
        self.bf = bf
        self.use_akaze = use_akaze

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # For low-feature objects, CLAHE improves contrast before detection
        if use_whole_frame or use_akaze:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        mask = (
            KeypointTarget.build_mask(gray.shape, initial_positions, exclude_regions)
            if not use_whole_frame
            else None
        )

        if use_akaze:
            # AKAZE handles low-texture regions much better than ORB.
            # It uses a non-linear scale space and produces binary descriptors
            # compatible with NORM_HAMMING, so we can reuse BFMatcher.
            self._detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                threshold=0.001,  # lower = more keypoints on flat regions
            )
        else:
            self._detector = orb

        self.kp_ref, self.des_ref = self._detector.detectAndCompute(gray, mask)

        if self.des_ref is None or len(self.kp_ref) == 0:
            raise RuntimeError("No reference descriptors found for target")

        # Cap reference keypoints: large sets massively slow down knnMatch
        MAX_REF = 500
        if len(self.kp_ref) > MAX_REF:
            # Keep highest-response keypoints rather than random sampling
            responses = np.array([k.response for k in self.kp_ref])
            idx = np.argsort(responses)[::-1][:MAX_REF]
            self.kp_ref = [self.kp_ref[i] for i in idx]
            self.des_ref = self.des_ref[idx]

        self.points = KeypointTarget.get_reference_src_pts(self.kp_ref)
        self.initial_positions = initial_positions
        self.last_quad = initial_positions.copy()
        self.last_inliers: int | None = None
        self.use_whole_frame = use_whole_frame

    def get_previous_quad(self) -> Quadrilateral:
        return self.last_quad

    def get_points(self) -> np.ndarray:
        return self.points

    def _compute_roi(self) -> Quadrilateral | None:
        if self.last_inliers is None:
            return None
        if self.last_inliers > 120:
            expand_px = 40
        elif self.last_inliers > 60:
            expand_px = 120
        else:
            return None
        return self.last_quad.expand(expand_px, expand_px)

    @staticmethod
    def _filter_keypoints_by_roi(
        kp: list, des: np.ndarray, roi: Quadrilateral
    ) -> tuple[list, np.ndarray]:
        """Vectorised ROI filter — avoids Python-level per-keypoint loop."""
        if roi is None or not kp:
            return kp, des

        x, y, w, h = roi.to_xywh()
        pts = np.array([k.pt for k in kp])  # (N, 2)
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

    def update_with_features(
        self,
        kp_frame: list,
        des_frame: np.ndarray,
        gray_frame: np.ndarray | None = None,  # needed for AKAZE targets
    ) -> Optional[Quadrilateral]:

        # AKAZE targets re-detect on the frame themselves because the shared
        # ORB features computed in update_all won't match AKAZE descriptors.
        if self.use_akaze and gray_frame is not None:
            roi = self._compute_roi()
            if roi is not None:
                x, y, w, h = roi.to_xywh()
                x, y = max(0, int(x)), max(0, int(y))
                w, h = int(w), int(h)
                roi_gray = gray_frame[y : y + h, x : x + w]
                kp_local, des_local = self._detector.detectAndCompute(roi_gray, None)
                if kp_local and des_local is not None:
                    # shift keypoints back to full-frame coordinates
                    for k in kp_local:
                        k.pt = (k.pt[0] + x, k.pt[1] + y)
                    kp_use, des_use = kp_local, des_local
                else:
                    kp_use, des_use = kp_frame, des_frame  # fallback
            else:
                kp_use, des_use = kp_frame, des_frame
        else:
            if des_frame is None or len(kp_frame) == 0:
                return None

            roi = self._compute_roi()
            kp_use, des_use = self._filter_keypoints_by_roi(kp_frame, des_frame, roi)
            if kp_use is None or des_use is None:
                kp_use, des_use = kp_frame, des_frame

        if des_use is None or len(kp_use) == 0:
            return None

        matches = self.bf.knnMatch(self.des_ref, des_use, k=2)

        good = [
            m
            for pair in matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]

        if len(good) < MIN_INLIERS:
            return None

        if len(good) > 300:
            good = sorted(good, key=lambda m: m.distance)[:300]

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
        next_quad = cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        self.last_quad = Quadrilateral.from_opencv_format(next_quad)
        return self.last_quad


class OrbTracker(TargetTracker):
    def __init__(self):
        # Reduce nfeatures: 3000 is excessive and is the root cause of the
        # knnMatch bottleneck. 1500 cuts match time roughly in half with
        # minimal tracking quality loss for normal objects.
        self.orb = cv2.ORB_create(
            nfeatures=1500,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=10,  # lower = more keypoints on low-contrast frames
        )
        # NORM_HAMMING2 is faster than NORM_HAMMING for ORB with WTA_K=3 or 4,
        # but for default ORB (WTA_K=2) NORM_HAMMING is correct — keep it.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Shared AKAZE matcher with NORM_HAMMING (MLDB descriptors are binary)
        self.bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.targets: dict[str, OrbTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
        use_whole_frame: bool = False,
        use_akaze: bool = False,
    ):
        bf = self.bf_akaze if use_akaze else self.bf
        target = OrbTarget(
            frame=frame,
            initial_positions=initial_positions,
            exclude_regions=exclude_regions,
            orb=self.orb,
            bf=bf,
            use_whole_frame=use_whole_frame,
            use_akaze=use_akaze,
        )
        self.targets[name] = target

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only compute ORB features if at least one non-AKAZE target exists.
        # This avoids a wasted detectAndCompute when all targets use AKAZE.
        has_orb_targets = any(not t.use_akaze for t in self.targets.values())
        if has_orb_targets:
            kp_frame, des_frame = self.orb.detectAndCompute(gray, None)
        else:
            kp_frame, des_frame = [], None

        outputs = {}
        for name, target in self.targets.items():
            outputs[name] = target.update_with_features(
                kp_frame, des_frame, gray_frame=gray
            )

        return outputs

    def get_target_pts(self, name: str) -> np.ndarray | None:
        t = self.targets.get(name)
        return t.get_points() if t else None

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        t = self.targets.get(name)
        return t.get_previous_quad() if t else None
