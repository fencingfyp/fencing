from dataclasses import dataclass
from typing import List

import numpy as np

from .output.region_output import RegionOutput


@dataclass
class RegionState:
    label: str
    outputs: List[RegionOutput]

    def process(self, frame: np.ndarray, quad_np: np.ndarray, frame_id: int):
        for output in self.outputs:
            output.process(frame, quad_np, frame_id)

    def close(self):
        for output in self.outputs:
            output.close()

    def delete(self):
        for output in self.outputs:
            output.delete()
