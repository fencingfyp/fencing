import pytest

from scripts.momentum_graph.util.extract_score_increases import _retroactive_flatten


@pytest.mark.parametrize(
    "scores, expected",
    [
        # trivial / edge cases
        ([], []),
        ([1], [1]),
        ([0, 0, 0], [0, 0, 0]),
        # monotonic cases
        ([1, 2, 3], [1, 2, 3]),
        ([2, 1, 1], [1, 1, 1]),
        ([1, 1, 2, 2], [1, 1, 2, 2]),
        # simple transient bumps
        ([1, 2, 1], [1, 1, 1]),
        ([1, 2, 2, 1], [1, 1, 1, 1]),
        ([1, 2, 3, 2], [1, 2, 2, 2]),
        # nested / cascading reversions
        ([1, 2, 3, 1], [1, 1, 1, 1]),
        ([1, 3, 5, 3, 1], [1, 1, 1, 1, 1]),
        ([1, 2, 3, 2, 3], [1, 2, 2, 2, 3]),
        # example from docstring
        ([1, 2, 2, 2, 1, 2, 3], [1, 1, 1, 1, 1, 2, 3]),
        # plateaus with later drops
        ([1, 2, 2, 3, 3, 2], [1, 2, 2, 2, 2, 2]),
        # multiple recoveries
        ([1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1, 2]),
    ],
)
def test_retroactive_flatten(scores, expected):
    assert _retroactive_flatten(scores) == expected
