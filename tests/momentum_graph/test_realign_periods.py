import numpy as np
import pytest

from src.gui.momentum_graph.generate_momentum_graph_widget import realign_periods


def test_no_periods_returns_original_and_none():
    seconds = np.array([0.0, 1.0, 2.0])
    aligned, periods = realign_periods(seconds, None)

    assert np.array_equal(aligned, seconds)
    assert periods is None


def test_empty_period_list_returns_original_and_none():
    seconds = np.array([0.0, 1.0, 2.0])
    aligned, periods = realign_periods(seconds, [])

    assert np.array_equal(aligned, seconds)
    assert periods is None


def test_single_period_alignment():
    seconds = np.array([0.0, 2.0, 4.0])
    periods = [{"start_ms": 2000, "end_ms": 6000}]

    aligned, aligned_periods = realign_periods(seconds, periods)

    # First period start = 2s
    # seconds[0] forced to 2, then subtract 2
    expected_seconds = np.array([0.0, 0.0, 2.0])
    assert np.allclose(aligned, expected_seconds)

    assert aligned_periods == [{"start_sec": 0.0, "end_sec": 4.0}]


def test_multiple_periods_alignment():
    seconds = np.array([0.0, 5.0, 10.0])
    periods = [
        {"start_ms": 3000, "end_ms": 7000},
        {"start_ms": 8000, "end_ms": 12000},
    ]

    aligned, aligned_periods = realign_periods(seconds, periods)

    # shift = 3s
    expected_seconds = np.array([0.0, 2.0, 7.0])
    assert np.allclose(aligned, expected_seconds)

    assert aligned_periods == [
        {"start_sec": 0.0, "end_sec": 4.0},
        {"start_sec": 5.0, "end_sec": 9.0},
    ]


def test_input_array_not_mutated():
    seconds = np.array([0.0, 1.0, 2.0])
    original_copy = seconds.copy()

    periods = [{"start_ms": 1000, "end_ms": 2000}]
    _ = realign_periods(seconds, periods)

    # Ensure original unchanged
    assert np.array_equal(seconds, original_copy)
