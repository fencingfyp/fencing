from typing import Callable

import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.DefinedRegion import DefinedRegion
from src.pyside.PysideUi import PysideUi
from src.util.utils import generate_select_quadrilateral_instructions

from .label_config import LabelConfig


class ROISelectionPipeline:
    """
    UI-only pipeline. Prompts the user to select 4 corners per label in
    sequence and produces DefinedRegion objects. Knows nothing about
    tracking internals — strategy and tuning live in LabelConfig.
    """

    def __init__(
        self,
        first_frame: np.ndarray,
        ui: PysideUi,
        label_configs: dict[str, LabelConfig],
        on_finished: Callable[[list[DefinedRegion]], None] | None = None,
    ):
        self.first_frame = first_frame.copy()
        self.ui = ui
        self.label_configs = label_configs
        self.labels = list(label_configs.keys())
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

    def _on_corners_selected(self, positions: list):
        if len(positions) != 4:
            raise ValueError(
                f"Expected exactly 4 points for quad selection, got {len(positions)}"
            )

        label = self.labels[self.region_index]
        config = self.label_configs[label]

        self.defined_regions.append(
            DefinedRegion(
                label=label,
                quad_np=Quadrilateral(positions).numpy(),
                output_factory=config.output_factory,
                tracking_strategy=config.tracking_strategy,
                mask_margin=config.mask_margin,
            )
        )

        self.region_index += 1
        self._ask_next_region()

    def _finish(self):
        if self._on_finished:
            self._on_finished(self.defined_regions)
