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
    def run_loop(self, step_fn):
        """
        UI-owned main loop.
        step_fn() is called every frame.
        step_fn should return True to continue, False to stop.
        """
        pass

    @abstractmethod
    def initialise(self, fps: float) -> None:
        """Initialise the UI with given FPS."""
        pass

    @abstractmethod
    def get_n_points_async(self, frame, prompts: list[str], callback):
        """Get n points asynchronously from user input with given prompts. Calls callback with list of (x, y) tuples scaled to the original frame size."""
        pass

    @abstractmethod
    def show_additional(self, key: int | str, frame: np.ndarray):
        """Show the given frame in a separate top-level window."""
        pass

    @abstractmethod
    def take_user_input(self):
        """Take user input and return an action code."""
        pass
