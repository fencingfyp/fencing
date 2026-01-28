from typing import Tuple

from .drawable import Color, Drawable


class PointsDrawable(Drawable):
    def __init__(self, points: list[Tuple[float, float]], color: Color = (0, 255, 0)):
        self._points = points
        self._color = color

    def primitives(self):
        yield ("points", self._points)

    def style(self):
        return {"color": self._color, "size": 2}
