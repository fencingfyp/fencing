from dataclasses import dataclass, field


@dataclass
class Tag:
    frame_idx: int
    time_msec: int
    description: str
    category: str
    subcategory: str
    id: int | None = field(default=None)
