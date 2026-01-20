from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class UiRequest(ABC):
    """Marker base class for all UI requests."""

    pass


class RequestPoints(UiRequest):
    def __init__(self, frame: np.ndarray, prompts: list[str]):
        self.frame = frame  # frame to display
        self.prompts = prompts  # ordered instructions


class RequestMessage(UiRequest):
    def __init__(self, text: str):
        self.text = text


class NoRequest(UiRequest):
    pass
