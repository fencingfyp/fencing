import abc

import numpy as np

from src.model.Quadrilateral import Quadrilateral


class TargetTracker(abc.ABC):
    @abc.abstractmethod
    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        pass

    @abc.abstractmethod
    def get_target_pts(self, name: str) -> np.ndarray | None:
        pass

    @abc.abstractmethod
    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        pass
