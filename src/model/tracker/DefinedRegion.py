from dataclasses import dataclass

import numpy as np


@dataclass
class DefinedRegion:
    label: str
    quad_np: np.ndarray

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "quad_np": self.quad_np.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DefinedRegion":
        return cls(
            label=d["label"],
            quad_np=np.array(d["quad_np"], dtype=np.float32),
        )
