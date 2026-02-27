from dataclasses import dataclass, field
from typing import Callable

from src.model.tracker.TrackingStrategy import TrackingStrategy


@dataclass
class TrackingConfig:
    tracking_strategy: TrackingStrategy = TrackingStrategy.ORB
    mask_margin: float = field(default=0.2)

    def to_dict(self) -> dict:
        return {
            "tracking_strategy": self.tracking_strategy.value,
            "mask_margin": self.mask_margin,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrackingConfig":
        return TrackingConfig(
            tracking_strategy=TrackingStrategy(d["tracking_strategy"]),
            mask_margin=d["mask_margin"],
        )
