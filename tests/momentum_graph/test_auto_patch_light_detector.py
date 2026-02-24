"""
Differential tests: classify() vs classify_batch()
Every test asserts that classify_batch returns the same result as
calling classify() individually on each (frame, quad) pair.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.model import Quadrilateral
from src.model.AutoPatchLightDetector import SinglePatchAutoDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_quad(x0, y0, x1, y1) -> Quadrilateral:
    """Axis-aligned rectangle expressed as a Quadrilateral."""
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return Quadrilateral(pts)


def solid_bgr(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def build_detector(
    pos_bgr=(0, 200, 50),  # vivid green-ish
    neg_bgr=(128, 128, 128),  # neutral grey
    h=100,
    w=100,
) -> SinglePatchAutoDetector:
    pos_frame = solid_bgr(h, w, pos_bgr)
    neg_frame = solid_bgr(h, w, neg_bgr)
    quad = make_quad(10, 10, 90, 90)
    return SinglePatchAutoDetector(pos_frame, quad, neg_frame, quad)


def run_differential(detector, pairs):
    """Core helper: compare element-wise."""
    individual = [detector.classify(f, q) for f, q in pairs]
    batched = detector.classify_batch(pairs)
    return individual, batched


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector():
    return build_detector()


@pytest.fixture
def quad():
    return make_quad(10, 10, 90, 90)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDifferentialSinglePair:
    """classify_batch([pair]) must match classify(pair) for one element."""

    def test_positive_match(self, detector, quad):
        frame = solid_bgr(100, 100, (0, 200, 50))
        individual, batched = run_differential(detector, [(frame, quad)])
        assert batched == individual

    def test_negative_brightness(self, detector, quad):
        # Same hue as positive but dark — brightness gate should reject
        frame = solid_bgr(100, 100, (0, 30, 7))
        individual, batched = run_differential(detector, [(frame, quad)])
        assert batched == individual

    def test_low_chroma(self, detector, quad):
        # Achromatic grey — chroma gate should reject
        frame = solid_bgr(100, 100, (180, 180, 180))
        individual, batched = run_differential(detector, [(frame, quad)])
        assert batched == individual

    def test_wrong_colour(self, detector, quad):
        # Bright but wrong hue (vivid red)
        frame = solid_bgr(100, 100, (0, 0, 240))
        individual, batched = run_differential(detector, [(frame, quad)])
        assert batched == individual


class TestDifferentialBatch:
    """Multi-pair batches — all elements must agree."""

    def test_all_positive(self, detector, quad):
        pairs = [(solid_bgr(100, 100, (0, 200, 50)), quad)] * 5
        individual, batched = run_differential(detector, pairs)
        assert batched == individual

    def test_all_negative(self, detector, quad):
        pairs = [(solid_bgr(100, 100, (180, 180, 180)), quad)] * 5
        individual, batched = run_differential(detector, pairs)
        assert batched == individual

    def test_mixed(self, detector, quad):
        frames = [
            solid_bgr(100, 100, (0, 200, 50)),  # should pass
            solid_bgr(100, 100, (180, 180, 180)),  # chroma fail
            solid_bgr(100, 100, (0, 200, 50)),  # should pass
            solid_bgr(100, 100, (0, 0, 240)),  # colour fail
            solid_bgr(100, 100, (0, 30, 7)),  # brightness fail
        ]
        pairs = [(f, quad) for f in frames]
        individual, batched = run_differential(detector, pairs)
        assert batched == individual

    def test_varying_quads(self, detector):
        """Different quad positions on the same frame."""
        frame = solid_bgr(100, 100, (0, 200, 50))
        quads = [
            make_quad(0, 0, 50, 50),
            make_quad(50, 50, 100, 100),
            make_quad(10, 10, 90, 90),
        ]
        pairs = [(frame, q) for q in quads]
        individual, batched = run_differential(detector, pairs)
        assert batched == individual

    def test_large_batch(self, detector, quad):
        """Stress test — 50 pairs, half positive half negative."""
        pos = solid_bgr(100, 100, (0, 200, 50))
        neg = solid_bgr(100, 100, (180, 180, 180))
        pairs = [(pos if i % 2 == 0 else neg, quad) for i in range(50)]
        individual, batched = run_differential(detector, pairs)
        assert batched == individual


class TestDifferentialEdgeCases:

    def test_single_pixel_quad(self, detector):
        """Degenerate tiny patch — both paths should handle or both should raise."""
        frame = solid_bgr(100, 100, (0, 200, 50))
        tiny_quad = make_quad(50, 50, 51, 51)
        try:
            individual = [detector.classify(frame, tiny_quad)]
            batched = detector.classify_batch([(frame, tiny_quad)])
            assert batched == individual
        except ValueError:
            # Both should raise — if classify raises, batch should too
            with pytest.raises(ValueError):
                detector.classify_batch([(frame, tiny_quad)])

    def test_quad_at_frame_boundary(self, detector):
        frame = solid_bgr(100, 100, (0, 200, 50))
        edge_quad = make_quad(0, 0, 100, 100)
        individual, batched = run_differential(detector, [(frame, edge_quad)])
        assert batched == individual

    def test_noisy_frame(self, detector, quad):
        """Random noise — both should agree even if result is non-deterministic."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        individual, batched = run_differential(detector, [(frame, quad)])
        assert batched == individual

    def test_batch_of_one_matches_single(self, detector, quad):
        """Explicit check: classify_batch([x]) == [classify(x)]."""
        frame = solid_bgr(100, 100, (0, 200, 50))
        assert detector.classify_batch([(frame, quad)]) == [
            detector.classify(frame, quad)
        ]

    def test_result_types_are_bool(self, detector, quad):
        frame = solid_bgr(100, 100, (0, 200, 50))
        results = detector.classify_batch([(frame, quad)])
        assert all(isinstance(r, bool) for r in results)
        results = detector.classify_batch([(frame, quad)])
        assert all(isinstance(r, bool) for r in results)
