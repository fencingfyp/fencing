# src/model/tracker/CompositeTracker.py

import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.AkazeTracker import AkazeTracker
from src.model.tracker.OrbTracker import OrbTracker
from src.model.tracker.TargetTracker import TargetTracker


class CompositeTracker(TargetTracker):
    """
    Delegates each target to either OrbTracker or AkazeTracker based on
    which tracker it was registered with. Presents a unified TargetTracker
    interface to callers — the internal split is an implementation detail.
    """

    def __init__(self, orb: OrbTracker, akaze: AkazeTracker):
        self._orb = orb
        self._akaze = akaze
        # Track which sub-tracker owns each name for O(1) delegation
        self._owner: dict[str, TargetTracker] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        exclude_regions: list[Quadrilateral] | None = None,
        tracker: TargetTracker | None = None,  # pass orb or akaze instance explicitly
        **kwargs,
    ) -> None:
        # CompositeTracker.add_target is not called directly by external code —
        # use build_tracker() which routes correctly. This exists to satisfy
        # the ABC and for testing.
        raise NotImplementedError(
            "Use build_tracker() to construct a CompositeTracker — "
            "targets are registered via the sub-trackers directly."
        )

    def _register(
        self,
        name: str,
        frame: np.ndarray,
        initial_positions: Quadrilateral,
        sub_tracker: TargetTracker,
        **kwargs,
    ) -> None:
        """Called by build_tracker to register a target with its chosen sub-tracker."""
        sub_tracker.add_target(name, frame, initial_positions, **kwargs)
        self._owner[name] = sub_tracker

    def update_all(self, frame: np.ndarray) -> dict[str, Quadrilateral | None]:
        results = {}
        # Only call update_all on trackers that have targets — avoids a
        # wasted ORB detectAndCompute if all targets are AKAZE and vice versa
        if self._orb.targets:
            results.update(self._orb.update_all(frame))
        if self._akaze.targets:
            results.update(self._akaze.update_all(frame))
        return results

    def get_target_pts(self, name: str) -> np.ndarray | None:
        owner = self._owner.get(name)
        return owner.get_target_pts(name) if owner else None

    def get_previous_quad(self, name: str) -> Quadrilateral | None:
        owner = self._owner.get(name)
        return owner.get_previous_quad(name) if owner else None
