from typing import override

from .box_drawable import BoxDrawable
from .drawable import Drawable
from .text_drawable import TextDrawable


class DetectionsDrawable(Drawable):
    def __init__(self, detections, highlight_id=None):
        self.sub_drawables: list[Drawable] = []
        for det in detections.values():
            color = (255, 0, 0) if det["id"] == highlight_id else (0, 255, 0)
            self.sub_drawables.append(BoxDrawable(det["box"], color=color))
            top_left_x, top_left_y = det["box"][0], det["box"][1]
            self.sub_drawables.append(
                TextDrawable(str(det["id"]), top_left_x, top_left_y, color=color)
            )

    def primitives(self):
        return []

    def style(self):
        # optional if individual drawables manage style
        return {}

    @override
    def get_sub_drawables(self):
        return self.sub_drawables
