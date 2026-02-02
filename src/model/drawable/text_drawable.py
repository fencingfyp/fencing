from .drawable import Drawable


class TextDrawable(Drawable):
    def __init__(self, text, x, y, color=(0, 255, 0), font_size=15):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.font_size = font_size

    def primitives(self):
        yield ("text", ((self.x, self.y), self.text))

    def style(self):
        return {"color": self.color, "font_size": self.font_size}
