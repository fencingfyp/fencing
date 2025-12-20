from src.util.utils import project_point_on_line


def test_project_point_on_line():
    tests = [
        # Point directly above the midpoint
        (((0, 0), (4, 0)), (2, 3), (2.0, 0.0)),
        # Point lies already on the line
        (((0, 0), (4, 0)), (3, 0), (3.0, 0.0)),
        # Vertical line
        (((1, 1), (1, 5)), (4, 3), (1.0, 3.0)),
        # Diagonal line
        (((0, 0), (2, 2)), (1, 0), (0.5, 0.5)),
        # Degenerate line (single point)
        (((2, 2), (2, 2)), (5, 5), (2.0, 2.0)),
    ]

    for i, (line, point, expected) in enumerate(tests, 1):
        result = project_point_on_line(line, point)
        print(f"Test {i}: {result} (expected {expected})")
        assert all(abs(a - b) < 1e-9 for a, b in zip(result, expected))
