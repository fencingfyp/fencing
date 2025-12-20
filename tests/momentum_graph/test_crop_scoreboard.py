# tests/momentum_graph/test_crop_scoreboard.py

import numpy as np

from src.momentum_graph.crop_scoreboard import get_output_dimensions


def test_square_axis_aligned():
    pts = [
        [0, 0],  # TL
        [10, 0],  # TR
        [10, 10],  # BR
        [0, 10],  # BL
    ]
    w, h = get_output_dimensions(pts)
    assert w == 10
    assert h == 10


def test_rectangle_axis_aligned():
    pts = [
        [2, 5],  # TL
        [12, 5],  # TR
        [12, 25],  # BR
        [2, 25],  # BL
    ]
    w, h = get_output_dimensions(pts)
    assert w == 10
    assert h == 20


def test_rotated_rectangle():
    # 10×5.1 rectangle rotated 45 degrees
    pts = np.array(
        [
            [0, 0],  # TL
            [7.07, 7.07],  # TR
            [4.24, 11.31],  # BR
            [-2.83, 4.24],  # BL
        ]
    )
    w, h = get_output_dimensions(pts)
    assert w == 10  # long edge
    assert h == 5  # short edge


def test_skewed_quadrilateral():
    pts = [
        [0, 0],
        [10, 1],
        [9, 8],
        [1, 7],
    ]
    w, h = get_output_dimensions(pts)

    # manual distances
    top = np.linalg.norm(np.array([0, 0]) - np.array([10, 1]))
    bottom = np.linalg.norm(np.array([9, 8]) - np.array([1, 7]))
    left = np.linalg.norm(np.array([0, 0]) - np.array([1, 7]))
    right = np.linalg.norm(np.array([10, 1]) - np.array([9, 8]))

    assert w == int(max(top, bottom))
    assert h == int(max(left, right))


def test_float_inputs():
    pts = [
        [0.5, 1.5],
        [4.5, 1.5],
        [4.5, 6.5],
        [0.5, 6.5],
    ]
    w, h = get_output_dimensions(pts)
    assert w == 4
    assert h == 5
