# src/model/drawables.py
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

Color = Tuple[int, int, int]  # RGB


class Drawable(ABC):
    """Abstract base for anything that can be drawn."""

    @abstractmethod
    def primitives(self) -> Iterable[Tuple[str, Any]]:
        """
        Returns iterable of (type, data) pairs.
        Type is 'points', 'lines', 'polygon', 'box', etc.
        Data is primitive-specific:
        - 'points': list of (x, y)
        - 'lines': list of ((x1, y1), (x2, y2))
        - 'polygon': list of (x, y)
        - 'box': (x1, y1, x2, y2)
        """
        pass

    @abstractmethod
    def style(self) -> dict:
        """Return styling hints: color, thickness, label, etc."""
        pass

    def get_sub_drawables(self) -> list["Drawable"]:
        """Return list of sub-drawables if any."""
        return []
