# src/model/tracker/tracker_factory.py

import numpy as np

from src.model.Quadrilateral import Quadrilateral
from src.model.tracker.AkazeTracker import AkazeTracker
from src.model.tracker.CompositeTracker import CompositeTracker
from src.model.tracker.OrbTracker import OrbTracker
from src.model.tracker.TargetTracker import TargetTracker
from src.model.tracker.TrackingStrategy import TrackingStrategy

from .DefinedRegion import DefinedRegion


def build_tracker(
    defined_regions: list[DefinedRegion],
    first_frame: np.ndarray,
) -> TargetTracker:
    """
    Inspects the region list, instantiates only the tracker(s) required,
    and returns a unified TargetTracker.

    - All ORB  → returns OrbTracker directly
    - All AKAZE → returns AkazeTracker directly
    - Mixed     → returns CompositeTracker delegating to both

    Callers (e.g. MultiRegionProcessingPipeline) don't need to know which
    concrete type they receive — they interact solely via TargetTracker.
    """
    strategies = {r.tracking_strategy for r in defined_regions}
    needs_orb = TrackingStrategy.ORB in strategies
    needs_akaze = TrackingStrategy.AKAZE in strategies

    orb = OrbTracker() if needs_orb else None
    akaze = AkazeTracker() if needs_akaze else None

    if needs_orb and needs_akaze:
        composite = CompositeTracker(orb, akaze)
        for region in defined_regions:
            sub = orb if region.tracking_strategy is TrackingStrategy.ORB else akaze
            kwargs = (
                {"mask_margin": region.mask_margin}
                if region.tracking_strategy is TrackingStrategy.ORB
                else {}
            )
            composite._register(
                name=region.label,
                frame=first_frame,
                initial_positions=Quadrilateral(region.quad_np),
                sub_tracker=sub,
                **kwargs,
            )
        return composite

    # Single-strategy fast path — no composite overhead
    tracker = orb if needs_orb else akaze
    for region in defined_regions:
        kwargs = (
            {"mask_margin": region.mask_margin}
            if region.tracking_strategy is TrackingStrategy.ORB
            else {}
        )
        tracker.add_target(
            name=region.label,
            frame=first_frame,
            initial_positions=Quadrilateral(region.quad_np),
            **kwargs,
        )
    return tracker
