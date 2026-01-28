from typing import Tuple

from .drawable import Color, Drawable


class BoxDrawable(Drawable):
    def __init__(
        self, box: Tuple[float, float, float, float], color: Color = (0, 255, 0)
    ):
        self._box = box
        self._color = color

    def primitives(self):
        x1, y1, x2, y2 = self._box
        # represent as polygon
        yield ("polygon", [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def style(self):
        return {"color": self._color, "thickness": 2}
