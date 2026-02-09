from typing import Iterable, Tuple, Union

import numpy as np

from .drawable import Color, Drawable


class LabelledPointsDrawable(Drawable):
    def __init__(
        self,
        points: Union[
            Iterable[Tuple[float, float]],
            np.ndarray,
        ],
        labels: list[str],
        color: Color = (0, 255, 0),
        size: int = 2,
    ):
        self._points = self._normalise_points(points)
        self._labels = labels
        self._color = color
        self._size = size

    @staticmethod
    def _normalise_points(points) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)
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
        for i in range(len(self._points)):
            yield (
                "text",
                ((self._points[i][0] + 5, self._points[i][1] + 5), self._labels[i]),
            )

    def style(self):
        return {"color": self._color, "size": self._size}
