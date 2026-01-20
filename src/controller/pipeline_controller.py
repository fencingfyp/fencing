from typing import Any

import numpy as np

from .pipeline_task import PipelineTask
from .ui_request import NoRequest, UiRequest


class PipelineController:
    """
    UI-agnostic pipeline runner for exactly one task.
    """

    def __init__(self, task: PipelineTask):
        self.task = task
        self._finished = False
        self._aborted = False
        self._waiting_for_input = False
        self._pending_request: UiRequest | None = None

    # ---- lifecycle -----------------------------------------------

    def start(self, first_frame: np.ndarray) -> UiRequest:
        """
        Start the pipeline with the first frame.
        """
        self._pending_request = self.task.start(first_frame)
        self._waiting_for_input = not isinstance(self._pending_request, NoRequest)
        return self._pending_request

    def abort(self) -> None:
        """
        Abort the pipeline completely.
        """
        self._aborted = True
        self.task.close()

    # ---- driving --------------------------------------------------

    def feed_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, UiRequest | None]:
        """
        Feed one frame into the pipeline.

        Returns:
            (output_frame, ui_request or None)
        """
        if self._aborted or self.task.is_finished():
            return None, None

        # If waiting for user input, we must not advance frames
        if self._waiting_for_input:
            return None, self._pending_request

        output, request = self.task.on_frame(frame)

        if not isinstance(request, NoRequest):
            self._pending_request = request
            self._waiting_for_input = True
            return output, request

        if self.task.is_finished():
            self.task.close()
            self._finished = True

        return output, None

    def feed_user_input(self, data: Any) -> UiRequest | None:
        """
        Deliver user input back to the task.
        """
        if not self._waiting_for_input:
            return None

        request = self.task.on_user_input(data)
        self._waiting_for_input = not isinstance(request, NoRequest)
        self._pending_request = request if self._waiting_for_input else None
        return request

    def feed_action(self, code: Any) -> UiRequest | None:
        """
        Deliver control action (quit, toggle slow, etc).
        """
        if code == "QUIT":  # or UiCodes.QUIT
            self.abort()
            return None

        request = self.task.on_action(code)
        self._waiting_for_input = not isinstance(request, NoRequest)
        self._pending_request = request if self._waiting_for_input else None
        return request

    # ---- status ---------------------------------------------------

    def is_finished(self) -> bool:
        return self._finished or self._aborted
