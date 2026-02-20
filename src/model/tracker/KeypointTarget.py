import abc
from typing import Tuple

import cv2
import numpy as np

from src.model.Quadrilateral import Quadrilateral


class KeypointTarget(abc.ABC):
    @staticmethod
    def get_reference_src_pts(kp_ref: list[cv2.KeyPoint]) -> np.ndarray:
        if not kp_ref:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.float32([kp.pt for kp in kp_ref]).reshape(-1, 1, 2)

    @staticmethod
    def build_mask(
        image_shape: Tuple[int, int],
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
        mask_margin: float = 0.2,  # fraction of full-frame dimensions
    ) -> np.ndarray:
        """
        Build a binary mask that restricts feature detection to a neighbourhood
        around the target quad.

        mask_margin controls how much of the surrounding scene is included —
        a larger value anchors to more stable context outside the target surface,
        which helps when the target itself is low-feature (e.g. a scoreboard screen).
        Expressed as a fraction of the full frame dimensions so it's
        resolution-independent.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        margin_x = int(w * mask_margin)
        margin_y = int(h * mask_margin)
        expanded = initial_positions.expand(margin_x, margin_y)

        cv2.fillPoly(mask, [expanded.opencv_format().astype(np.int32)], 255)

        if exclude_regions:
            for region in exclude_regions:
                x, y, rw, rh = region.to_xywh()
                mask[y : y + rh, x : x + rw] = 0

        return mask
