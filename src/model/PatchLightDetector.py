import cv2
import numpy as np

class PatchLightDetector:
    def __init__(self, target, alpha=1.05):
        self.target = target  # 'red', 'green', 'white'
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

    def update(self, frame, patch_pts, v_thresh=40, rel_thresh=0.25, c_thresh=0.005, dominance_thresh=1.0):
        b,g,r = self._mean_rgb(frame, patch_pts)
        R,G,B = float(r), float(g), float(b)
        Y = R + G + B

        if self.baseline is None:
            self.baseline = (R,G,B,Y)  # store first frame baseline
            return False  # assume OFF

        R0,G0,B0,Y0 = self.baseline

        # brightness increase relative to baseline
        dY = Y - Y0
        relY = dY / (Y0 + 1e-6)
        if dY < v_thresh or relY < rel_thresh:
            return False  # patch not significantly brighter

        # compute chroma for white detection
        mx, mn = max(R,G,B), min(R,G,B)
        chroma_ratio = (mx - mn) / (Y + 1e-6)

        if self.target == 'white':
            return chroma_ratio < c_thresh  # low chroma = white

        # coloured detection: target channel must dominate others
        if self.target == 'red':
            # red must be dominant over green and blue
            if R > G * dominance_thresh and R > B * dominance_thresh:
                return True
        if self.target == 'green':
            if G > R * dominance_thresh and G > B * dominance_thresh:
                return True

        return False

