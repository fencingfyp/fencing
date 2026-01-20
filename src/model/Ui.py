from abc import ABC, abstractmethod

import numpy as np

from .Quadrilateral import Quadrilateral


class Ui(ABC):
    @abstractmethod
    def set_fresh_frame(self, frame):
        """Set a fresh frame for user interaction."""
        pass

    @abstractmethod
    def get_n_points(self, frame, prompts: list[str]) -> list[tuple[float, float]]:
        """Get n points from user input with given prompts. Returns list of (x, y) tuples scaled to the original frame size."""
        pass

    @abstractmethod
    def close(self):
        """Close the UI."""
        pass

    @abstractmethod
    def plot_points(
        self, points: list[tuple[float, float]], color: tuple[int, int, int]
    ):
        """Plot points on the current frame."""
        pass

    @abstractmethod
    def show_frame(self):
        """Show the current frame."""
        pass

    @abstractmethod
    def process_crop_region_loop(self, cap, pipeline, writer=None):
        """Process the crop region loop."""
        pass
