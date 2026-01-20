from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .ui_request import UiRequest


class PipelineTask(ABC):
    """
    UI-agnostic task interface.
    One task processes exactly one video (for now).
    """

    # ---- lifecycle -------------------------------------------------

    @abstractmethod
    def start(self, first_frame: np.ndarray) -> UiRequest:
        """
        Called once when the task begins.
        Should initialise state and return the first UI request (if any).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Called when the task finishes or is aborted.
        Must release all resources (writers, trackers, etc).
        """
        pass

    # ---- driving --------------------------------------------------

    @abstractmethod
    def on_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, UiRequest]:
        """
        Called for every frame while running.

        Returns:
            - output_frame (or None if nothing to display / task idle)
            - next UiRequest (or NoRequest)
        """
        pass

    @abstractmethod
    def on_user_input(self, data: Any) -> UiRequest:
        """
        Called when the UI finishes a requested user interaction.
        `data` depends on the previous UiRequest (e.g. list of points).

        Returns next UiRequest (or NoRequest).
        """
        pass

    @abstractmethod
    def on_action(self, code: Any) -> UiRequest:
        """
        Called when user presses a control key / button (quit, toggle slow, etc).
        Should update state or request abort.

        Returns next UiRequest (or NoRequest).
        """
        pass

    # ---- status ---------------------------------------------------

    @abstractmethod
    def is_finished(self) -> bool:
        """Return True when the task is complete."""
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """Optional final result (paths, metadata, etc)."""
        pass
