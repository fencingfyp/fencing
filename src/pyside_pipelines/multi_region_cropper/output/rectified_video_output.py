import cv2
import numpy as np

from .region_output import RegionOutput


class RectifiedVideoOutput(RegionOutput):
    def __init__(
        self,
        video_path: str,
        fps: float,
        quad_initial: np.ndarray,
    ):
        """
        quad_initial: (4,2) float32 array from first frame
        """

        self._dimensions = self._compute_planar_dimensions(quad_initial)
        self._dst_corners = self._make_destination_corners(self._dimensions)
        self.video_path = video_path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            self._dimensions,
        )

    # ---------- Internal helpers ----------

    @staticmethod
    def _compute_planar_dimensions(quad: np.ndarray) -> tuple[int, int]:
        width = int(
            max(
                np.linalg.norm(quad[0] - quad[1]),
                np.linalg.norm(quad[2] - quad[3]),
            )
        )

        height = int(
            max(
                np.linalg.norm(quad[0] - quad[3]),
                np.linalg.norm(quad[1] - quad[2]),
            )
        )

        return width, height

    @staticmethod
    def _make_destination_corners(dimensions: tuple[int, int]) -> np.ndarray:
        w, h = dimensions
        return np.array(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ],
            dtype=np.float32,
        )

    # ---------- Main processing ----------

    def process(
        self,
        frame: np.ndarray,
        quad_np: np.ndarray,
        frame_id: int,
    ):
        H = cv2.getPerspectiveTransform(
            quad_np.astype(np.float32),
            self._dst_corners,
        )

        rectified = cv2.warpPerspective(
            frame,
            H,
            self._dimensions,
        )

        self._writer.write(rectified)

    def close(self):
        self._writer.release()

    def delete(self):
        self.close()
        import os

        os.remove(self.video_path)
