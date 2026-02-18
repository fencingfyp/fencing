from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DefinedRegion:
    label: str
    quad_np: np.ndarray
    output_factory: Callable[[np.ndarray, float], list]
