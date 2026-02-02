from typing import Iterable, Tuple, Union

import numpy as np

from .drawable import Color, Drawable


class PointsDrawable(Drawable):
    def __init__(
        self,
        points: Union[
            Iterable[Tuple[float, float]],
            np.ndarray,
        ],
        color: Color = (0, 255, 0),
    ):
        self._points = self._normalise_points(points)
        self._color = color

    @staticmethod
    def _normalise_points(points) -> np.ndarray:
        arr = np.asarray(points, dtype=float)

        if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
            arr = arr.reshape(-1, 2)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"points must be array-like with shape (N, 2) or (N, 1, 2), got {arr.shape}"
            )

        return arr

    def primitives(self):
        yield ("points", self._points)

    def style(self):
        return {"color": self._color, "size": 2}
