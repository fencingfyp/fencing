from dataclasses import dataclass, field

from src.pyside_pipelines.multi_region_cropper.output.output_config import OutputConfig

from .tracking_config import TrackingConfig


@dataclass
class LabelConfig:
    output_configs: list[OutputConfig]
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
