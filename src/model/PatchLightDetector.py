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

    def _extract_patch(self, frame, patch_pts):
        """
        Extract rectangular ROI covering patch_pts and mask out pixels
        outside the polygon. Pixels outside the polygon are set to 255
        (saturated) so downstream 'unsaturated' masking will ignore them.
        Returns a 3-channel array (B,G,R) matching the cropped ROI size.
        """
        patch_pts = np.asarray(patch_pts, dtype=np.int32)

        # bounding rect (clamped to frame)
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = cv2.boundingRect(patch_pts)
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(0, min(w, w_frame - x))
        h = max(0, min(h, h_frame - y))

        if w == 0 or h == 0:
            # return an empty safe patch to avoid downstream crashes
            return np.full((1, 1, 3), 255, dtype=frame.dtype)

        roi = frame[y:y + h, x:x + w].copy()

        # ensure 3 channels
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # polygon mask (shifted into ROI coords)
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = patch_pts - np.array([x, y], dtype=np.int32)
        cv2.fillPoly(mask, [shifted], 255)

        # set outside-polygon pixels to saturated value so they get excluded
        # by the "> 250" saturation test in downstream code
        masked_roi = roi.copy()
        # mask==0 -> outside polygon
        masked_roi[mask == 0] = 255

        return masked_roi


    def check_light(
        self,
        frame,
        patch_pts,
        v_thresh=35,
        rel_thresh=0.15,
        sat_thresh=0.09,
        chroma_thresh=0.04,
        baseline_alpha=0.001,  # for slow baseline drift correction
    ):
        # 1. Extract patch and split into channels
        patch = self._extract_patch(frame, patch_pts).astype(np.float32)
        if patch.size == 0:
            print(patch.shape)
            print("Warning: empty patch extracted")
            return False
        Bc, Gc, Rc = cv2.split(patch)

        # 2. Brightness from all pixels (saturated included)
        Y_total = np.mean(Rc + Gc + Bc) + 1e-6

        # 3. Usable subset (unsaturated)
        mask = (Rc < 250) & (Gc < 250) & (Bc < 250)
        if np.any(mask):
            Rm, Gm, Bm = Rc[mask], Gc[mask], Bc[mask]
        else:
            Rm, Gm, Bm = Rc, Gc, Bc

        # 4. Mean RGB over usable region
        R, G, B = float(np.mean(Rm)), float(np.mean(Gm)), float(np.mean(Bm))
        Y = R + G + B + 1e-6

        # 5. Chromaticity
        r_norm, g_norm, b_norm = R / Y, G / Y, B / Y

        # 6. Directional strengths
        chroma = max(r_norm, g_norm, b_norm) - min(r_norm, g_norm, b_norm)
        red_strength   = max(0.0, r_norm - max(g_norm, b_norm))
        green_strength = max(0.0, ((g_norm + 0.5 * b_norm) if ((g_norm + 20 > b_norm) and (r_norm < g_norm)) else 0) - r_norm)

        # 7. Saturation warning
        sat_ratio = np.mean(mask)
        if sat_ratio < 0.2:
            print(f"Warning: heavily saturated patch ({sat_ratio*100:.1f}% usable)")

        # 8. Initialise baseline
        if self.baseline is None:
            self.baseline = (R, G, B, Y_total)
            return False

        _, _, _, Y0 = self.baseline

        # 9. Brightness change
        dY = Y_total - Y0
        relY = dY / (Y0 + 1e-6)

        print(f"Detecting: {self.target}")

        # 10. Brightness check
        if dY < v_thresh or relY < rel_thresh:
            print(f"Failed brightness check: dY={dY:.1f}, relY={relY:.3f}")
            return False
        
        # chroma check
        if chroma < chroma_thresh:
            print(f"Failed chroma check: chroma={chroma:.3f}")
            return False

        # 11. Target colour strength check
        if self.target == "red":
            if red_strength < sat_thresh:
                print(f"Failed red strength: {red_strength:.3f}")
                return False
        elif self.target == "green":
            if green_strength < sat_thresh:
                print(f"Failed green strength: {green_strength:.3f}")
                return False
        else:
            print("Unknown target color")
            return False

        # 12. Success log
        print(
            f"Passed all checks: dY={dY:.1f}, relY={relY:.3f}, "
            f"red_strength={red_strength:.3f}, green_strength={green_strength:.3f}, "
            f"R={R:.1f}, G={G:.1f}, B={B:.1f}"
        )
        return True
