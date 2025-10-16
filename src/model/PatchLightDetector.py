import cv2
import numpy as np

class PatchLightDetector:
    def __init__(self, target, alpha=1.05):
        if target not in ['red', 'green']:
            raise ValueError("target must be 'red' or 'green'")
        self.target = target  # 'red', 'green'
        self.alpha = alpha
        self.baseline = None

    def _mean_rgb(self, frame, patch_pts):
        patch_pts = np.array(patch_pts)
        rect = cv2.boundingRect(patch_pts)
        x,y,w,h = rect
        roi = frame[y:y+h, x:x+w]
        mask = np.zeros((h,w), np.uint8)
        shifted = patch_pts - [x,y]
        cv2.fillPoly(mask, [shifted], 255)
        return cv2.mean(roi, mask=mask)[:3]  # B,G,R

    def check_light(
        self,
        frame,
        patch_pts,
        v_thresh=40,
        rel_thresh=0.25,
        sat_thresh=0.05,
        dominance_thresh=1.02,
        baseline_alpha=0.05,  # for slow baseline drift correction
    ):
        b, g, r = self._mean_rgb(frame, patch_pts)
        R, G, B = map(float, (r, g, b))
        Y = R + G + B + 1e-6  # total brightness

        # chromaticity (normalised RGB)
        r_norm, g_norm, b_norm = R / Y, G / Y, B / Y
        chroma = max(r_norm, g_norm, b_norm) - min(r_norm, g_norm, b_norm)

        # initialise or update baseline slowly
        if self.baseline is None:
            self.baseline = (R, G, B, Y)
            return False

        R0, G0, B0, Y0 = self.baseline
        dY = Y - Y0
        relY = dY / (Y0 + 1e-6)

        # optionally update baseline (slow drift handling)
        # self.baseline = tuple(
        #     (1 - baseline_alpha) * b0 + baseline_alpha * b1
        #     for b0, b1 in zip((R0, G0, B0, Y0), (R, G, B, Y))
        # )

        # brightness jump check
        if dY < v_thresh or relY < rel_thresh:
            return False

        # reject low-saturation / white blobs
        if chroma < sat_thresh:
            return False

        # colour dominance check
        if self.target == "red" and R > G * dominance_thresh and R > B * dominance_thresh:
            return True
        if self.target == "green" and G > R * dominance_thresh and G > B * dominance_thresh:
            return True

        return False


