# src/pyside_pipelines/multi_region_cropper/label_config.py

from dataclasses import dataclass, field
from typing import Callable

from src.model.tracker.TrackingStrategy import TrackingStrategy


@dataclass
class LabelConfig:
    output_factory: Callable
    tracking_strategy: TrackingStrategy = TrackingStrategy.ORB
    mask_margin: float = field(default=0.2)
