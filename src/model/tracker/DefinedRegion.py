from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .TrackingStrategy import TrackingStrategy


@dataclass
class DefinedRegion:
    label: str
    quad_np: np.ndarray
    output_factory: Callable  # (quad_np, fps) -> list[RegionOutput]
    tracking_strategy: TrackingStrategy = TrackingStrategy.ORB
    # mask_margin only applies to ORB targets — controls how much surrounding
    # scene context is included when building the reference feature mask.
    # Increase this if the target surface itself is low-feature (e.g. a screen).
    mask_margin: float = field(default=0.2)
