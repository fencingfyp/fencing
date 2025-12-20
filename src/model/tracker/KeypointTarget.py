import abc
from typing import Tuple

import cv2
import numpy as np

from src.model.Quadrilateral import Quadrilateral


class KeypointTarget(abc.ABC):
    @staticmethod
    def get_reference_src_pts(kp_ref: list[cv2.KeyPoint]) -> np.ndarray:
        """Calculates and returns the reference source points used for matching.
        These are the planar keypoints detected in the reference frame.
        Shape: (N, 1, 2)"""
        if not kp_ref:
            return np.empty((0, 1, 2), dtype=np.float32)
        # Convert keypoints to np array
        src_pts = np.float32([kp.pt for kp in kp_ref]).reshape(-1, 1, 2)
        return src_pts

    @staticmethod
    def build_mask(
        image_shape: Tuple[int, int],
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
    ) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # set margin to 1/5 of image dimensions
        margin_x = w // 5
        margin_y = h // 5
        expanded_quad = initial_positions.expand(margin_x, margin_y)

        # include only the initial planar region + some margin for stability
        cv2.fillPoly(
            mask,
            [expanded_quad.opencv_format().astype(np.int32)],
            255,
        )
        if exclude_regions:
            for region in exclude_regions:
                x, y, w_, h_ = region.to_xywh()
                mask[y : y + h_, x : x + w_] = 0  # block score area
        return mask
