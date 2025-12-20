import numpy as np

from src.model.Quadrilateral import Quadrilateral


class StaticTracker:
    def __init__(self):
        self.targets = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] = None,
    ) -> None:
        self.targets[name] = initial_positions

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral]:
        return self.targets.copy()

    def get_target_reference_pts(self, name: str) -> np.ndarray:
        return np.empty((0, 1, 2), dtype=np.float32)

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        target = self.targets.get(name, None)
        if target is None:
            return None
        return target
