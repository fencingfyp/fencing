from typing import Optional

import cv2
import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.KeypointTarget import KeypointTarget
from src.model.tracker.TargetTracker import TargetTracker

MIN_INLIERS = 20


class AkazeTarget(KeypointTarget):
    """
    Single AKAZE-tracked region.

    Unlike OrbTarget, each AkazeTarget re-detects features on the current
    frame independently rather than consuming shared frame features. This
    is necessary because AKAZE and ORB descriptors are incompatible, and
    computing a full-frame AKAZE detection to share across targets would
    be prohibitively slow (~50-100ms vs ~5-15ms for ORB).

    Instead, detection is restricted to the predicted ROI when tracking
    confidence is high, falling back to the full frame in recovery mode.
    CLAHE preprocessing is applied before detection to improve local
    contrast on low-texture surfaces (the primary use case for AKAZE).
    """

    def __init__(
        self,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None,
        akaze: cv2.AKAZE,
        bf: cv2.BFMatcher,
    ):
        self.akaze = akaze
        self.bf = bf
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        gray = self._preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        mask = self._build_exclusion_mask(gray.shape, initial_positions)
        self.kp_ref, self.des_ref = self.akaze.detectAndCompute(gray, mask=mask)
        if self.des_ref is None or len(self.kp_ref) == 0:
            raise RuntimeError("No reference descriptors found for AKAZE target")

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

    @staticmethod
    def _build_exclusion_mask(image_shape: tuple, quad: Quadrilateral) -> np.ndarray:
        mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
        cv2.fillPoly(mask, [quad.opencv_format().astype(np.int32)], 0)
        return mask

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        return self.clahe.apply(gray)

    def get_previous_quad(self) -> Quadrilateral:
        return self.last_quad

    def get_points(self) -> np.ndarray:
        return self.points

    def _compute_roi(self) -> Quadrilateral | None:
        if self.last_inliers is None:
            return None
        if self.last_inliers > 120:
            return self.last_quad.expand(40, 40)
        if self.last_inliers > 60:
            return self.last_quad.expand(120, 120)
        return None  # low confidence — full frame search

    def update(self, frame: np.ndarray) -> Optional[Quadrilateral]:
        gray = self._preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        roi = self._compute_roi()

        if roi is not None:
            kp, des = self._detect_in_roi(gray, roi)
            if kp is None:
                # ROI detection failed, fall back to full frame
                kp, des = self.akaze.detectAndCompute(gray, None)
        else:
            kp, des = self.akaze.detectAndCompute(gray, None)

        if des is None or not kp:
            return None

        print(f"[AKAZE] kp={len(kp)}, inliers={self.last_inliers}")
        return self._match_and_estimate(kp, des)

    def _detect_in_roi(
        self, gray: np.ndarray, roi: Quadrilateral
    ) -> tuple[list, np.ndarray] | tuple[None, None]:
        x, y, w, h = roi.to_xywh()
        x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)

        patch = gray[y : y + h, x : x + w]
        kp, des = self.akaze.detectAndCompute(patch, None)

        if not kp or des is None:
            return None, None

        # Translate patch-local coordinates back to full-frame space
        for k in kp:
            k.pt = (k.pt[0] + x, k.pt[1] + y)

        return kp, des

    def _match_and_estimate(
        self, kp_use: list, des_use: np.ndarray
    ) -> Optional[Quadrilateral]:
        matches = self.bf.knnMatch(self.des_ref, des_use, k=2)

        good = [
            m
            for pair in matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]
        all_distances = [
            (pair[0].distance, pair[1].distance) for pair in matches if len(pair) == 2
        ]
        if all_distances:
            ratios = [a / b for a, b in all_distances]
            print(
                f"[AKAZE] total pairs={len(all_distances)}, median_ratio={np.median(ratios):.2f}, passing_0.75={sum(r < 0.75 for r in ratios)}, passing_0.85={sum(r < 0.85 for r in ratios)}"
            )

        if len(good) < MIN_INLIERS:
            print(f"[AKAZE] Not enough good matches: {len(good)} < {MIN_INLIERS}")
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
            print(f"[AKAZE] Not enough inliers after RANSAC: {inliers} < {MIN_INLIERS}")
            return None

        self.points = dst_pts[mask.ravel().astype(bool)]
        self.last_quad = Quadrilateral.from_opencv_format(
            cv2.perspectiveTransform(self.initial_positions.opencv_format(), H)
        )
        return self.last_quad


class AkazeTracker(TargetTracker):
    """
    Tracks one or more regions using AKAZE features.
    Each target detects independently per frame — there are no shared
    frame features since AKAZE detection is too slow to run globally.
    Prefer this over OrbTracker for low-texture targets.
    """

    def __init__(self):
        self.akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            threshold=0.0003,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.targets: dict[str, AkazeTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
    ) -> None:
        # exclude_regions not applied for AKAZE — whole-frame search is
        # intentional since low-texture targets need surrounding context
        self.targets[name] = AkazeTarget(
            frame=frame,
            initial_positions=initial_positions,
            exclude_regions=exclude_regions,
            akaze=self.akaze,
            bf=self.bf,
        )

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        return {name: target.update(frame) for name, target in self.targets.items()}

    def get_target_pts(self, name: str) -> np.ndarray | None:
        t = self.targets.get(name)
        return t.get_points() if t else None

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        t = self.targets.get(name)
        return t.get_previous_quad() if t else None
