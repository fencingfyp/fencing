from typing import Tuple

from src.model.Quadrilateral import Quadrilateral

from .drawable import Color, Drawable


class QuadrilateralDrawable(Drawable):
    """Draws a quadrilateral object."""

    def __init__(self, quad: Quadrilateral, color: Color = (0, 255, 0)):
        self.quad = quad
        self._color = color

    def primitives(self):
        # Return polygon points
        yield ("polygon", self.quad.to_drawable_list())

    def style(self):
        return {"color": self._color, "thickness": 2}
