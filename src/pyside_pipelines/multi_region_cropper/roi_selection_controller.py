from typing import Callable

import cv2
import numpy as np

from src.model import Quadrilateral
from src.pyside.PysideUi import PysideUi
from src.util.utils import generate_select_quadrilateral_instructions

from .defined_region import DefinedRegion


class ROISelectionPipeline:
    """
    UI-only pipeline.
    Responsible ONLY for:
        - Asking user for 4 points per label
        - Producing DefinedRegion objects
    """

    def __init__(
        self,
        first_frame: np.ndarray,
        ui: PysideUi,
        region_output_factories: dict[str, Callable],
        on_finished: Callable | None = None,
    ):
        self.first_frame = first_frame.copy()
        self.ui = ui
        self.region_output_factories = region_output_factories

        self.labels = list(region_output_factories.keys())
        self.region_index = 0

        self.defined_regions: list[DefinedRegion] = []

        self._on_finished = on_finished

    def start(self):
        self._ask_next_region()

    def _ask_next_region(self):
        if self.region_index >= len(self.labels):
            self._finish()
            return

        label = self.labels[self.region_index]

        self.ui.get_n_points_async(
            self.first_frame,
            generate_select_quadrilateral_instructions(label),
            callback=self._on_corners_selected,
        )

    def _on_corners_selected(self, positions):
        if self.region_index >= len(self.labels):
            return
        if len(positions) != 4:
            raise ValueError(
                "Exactly 4 points required."
            )  # This should never happen since the UI enforces it (currently, but UI might change behaviour), but just in case...

        label = self.labels[self.region_index]
        factory = self.region_output_factories[label]

        quad = Quadrilateral(positions)
        quad_np = quad.numpy().astype(np.float32)

        self.defined_regions.append(
            DefinedRegion(
                label=label,
                quad_np=quad_np,
                output_factory=factory,
            )
        )

        self.region_index += 1
        self._ask_next_region()

    # ============================================================
    # FINISH
    # ============================================================

    def _finish(self):
        if self._on_finished:
            self._on_finished(self.defined_regions)
