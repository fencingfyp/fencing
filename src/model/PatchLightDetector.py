import cv2
import numpy as np

from src.model import Quadrilateral


class PatchLightDetector:
    def __init__(self, target, alpha=1.05):
        if target not in ["red", "green"]:
            raise ValueError("target must be 'red' or 'green'")
        self.target = target  # 'red', 'green'
        self.alpha = alpha
        self.baseline = None

    def check_light(
        self,
        frame: np.ndarray,
        patch_pts: Quadrilateral,
        v_thresh=35,
        rel_thresh=0.25,
        sat_thresh=0.09,
        chroma_thresh=0.04,
        baseline_alpha=0.001,
    ):
        """
        Check if the light patch defined by `patch_pts` in `frame` is lit.
        Uses colour and brightness analysis against a baseline.
        """
        h, w = frame.shape[:2]

        # 1. Polygon mask
        poly = patch_pts.numpy().astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        if not np.any(mask):
            print("Warning: empty mask")
            return False

        frame_f = frame.astype(np.float32)
        B, G, R = cv2.split(frame_f)

        region = mask.astype(bool)

        # 2. Brightness (all pixels)
        Y_total = np.mean((R + G + B)[region]) + 1e-6

        # 3. Unsaturated subset
        usable = region & (R < 250) & (G < 250) & (B < 250)
        if not np.any(usable):
            usable = region

        # 4. Mean RGB
        Rm = np.mean(R[usable])
        Gm = np.mean(G[usable])
        Bm = np.mean(B[usable])
        Y = Rm + Gm + Bm + 1e-6

        # 5. Chromaticity
        r_norm, g_norm, b_norm = Rm / Y, Gm / Y, Bm / Y
        chroma = max(r_norm, g_norm, b_norm) - min(r_norm, g_norm, b_norm)

        # 6. Directional strengths
        red_strength = max(0.0, r_norm - max(g_norm, b_norm) - 0.5 * g_norm)
        green_strength = max(
            0.0,
            (
                (g_norm + 0.5 * b_norm)
                if (g_norm + 20 > b_norm and r_norm < g_norm)
                else 0
            )
            - r_norm,
        )

        # 7. Saturation warning
        sat_ratio = usable.sum() / region.sum()
        if sat_ratio < 0.2:
            print(f"Warning: heavily saturated patch ({sat_ratio*100:.1f}% usable)")

        # 8. Baseline init
        if self.baseline is None:
            self.baseline = (Rm, Gm, Bm, Y_total)
            return False

        _, _, _, Y0 = self.baseline

        # 9. Brightness change
        dY = Y_total - Y0
        relY = dY / (Y0 + 1e-6)

        # 10. Brightness checks
        if dY < v_thresh or relY < rel_thresh:
            print(f"Failed brightness check: dY={dY:.1f}, relY={relY:.3f}")
            return False

        if chroma < chroma_thresh:
            print(f"Failed chroma check: chroma={chroma:.3f}")
            return False

        # 11. Target colour
        if self.target == "red" and red_strength < sat_thresh:
            print(f"Failed red strength: {red_strength:.3f}")
            return False
        if self.target == "green" and green_strength < sat_thresh:
            print(f"Failed green strength: {green_strength:.3f}")
            return False

        print(
            f"Passed all checks: dY={dY:.1f}, relY={relY:.3f}, "
            f"red_strength={red_strength:.3f}, green_strength={green_strength:.3f}, "
            f"R={Rm:.1f}, G={Gm:.1f}, B={Bm:.1f}"
        )

        return True
