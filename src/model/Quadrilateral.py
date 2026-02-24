import numpy as np


class Quadrilateral:
    """Represents a quadrilateral defined by four points. Each point is a tuple (x, y)."""

    def __init__(self, points):
        """
        points: list[tuple[int, int]] OR np.ndarray of shape (4, 2)
        """
        if isinstance(points, np.ndarray):
            if points.shape != (4, 2):
                raise ValueError("NumPy array must be shape (4,2)")
            self.points = points.astype(np.float32)
        else:
            if len(points) != 4 or not all(len(pt) == 2 for pt in points):
                raise ValueError("A quadrilateral must have exactly 4 points (x,y).")
            self.points = np.array(points, dtype=np.float32)

    def is_valid(self) -> bool:
        """Check if the quadrilateral has 4 distinct points."""
        return len(self.points) == 4 and len(set(map(tuple, self.points))) == 4

    def numpy(self) -> np.ndarray:
        return self.points.copy()

    def to_drawable_list(self, type: str = "float") -> list[tuple[float, float]]:
        ordered_points = self._order_points()
        if type == "int":
            return [(int(pt[0]), int(pt[1])) for pt in ordered_points]
        else:
            return [(float(pt[0]), float(pt[1])) for pt in ordered_points]

    @staticmethod
    def from_opencv_format(pts: np.ndarray) -> "Quadrilateral":
        """Create a Quadrilateral from OpenCV format (Nx1x2 array)."""
        if pts.shape != (4, 1, 2):
            raise ValueError("Input must be of shape (4, 1, 2).")
        points = pts.reshape((4, 2))
        return Quadrilateral([tuple(pt) for pt in points])

    def opencv_format(self) -> np.ndarray:
        """Return points in OpenCV format (Nx1x2 array)."""
        return self.points.reshape((-1, 1, 2)).copy()

    @staticmethod
    def from_xywh(xywh: tuple[int, int, int, int]) -> "Quadrilateral":
        """Create a Quadrilateral from bounding box format (x, y, width, height)."""
        x, y, w, h = xywh
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return Quadrilateral(points)

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Convert quadrilateral to bounding box format (x, y, width, height)."""
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        x_min = np.min(x_coords)
        y_min = np.min(y_coords)
        width = np.max(x_coords) - x_min
        height = np.max(y_coords) - y_min
        return (int(x_min), int(y_min), int(width), int(height))

    def to_drawable(self) -> np.ndarray:
        """Return points in a format suitable for drawing functions."""
        ordered_points = self._order_points()
        return ordered_points.reshape((-1, 1, 2)).copy()

    def _order_points(self) -> np.ndarray:
        """Ensure points are ordered as: top-left, top-right, bottom-right, bottom-left."""
        points = self.points.copy()
        # Sort by y-coordinate, then x-coordinate
        sorted_by_y = points[np.argsort(points[:, 1])]

        # Top two points and bottom two points
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        # Sort top two by x-coordinate (left to right)
        top_left, top_right = top_two[np.argsort(top_two[:, 0])]

        # Sort bottom two by x-coordinate (left to right)
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def copy(self) -> "Quadrilateral":
        """Return a copy of the Quadrilateral."""
        return Quadrilateral([tuple(pt) for pt in self.points])

    def expand(self, margin_x: int, margin_y: int) -> "Quadrilateral":
        """Expand the quadrilateral by given margins in x and y directions."""
        center_x = np.mean(self.points[:, 0])
        center_y = np.mean(self.points[:, 1])

        expanded_points = []
        for x, y in self.points:
            if x < center_x:
                new_x = x - margin_x
            else:
                new_x = x + margin_x

            if y < center_y:
                new_y = y - margin_y
            else:
                new_y = y + margin_y

            expanded_points.append((new_x, new_y))

        return Quadrilateral(expanded_points)

    def __repr__(self) -> str:
        return f"Quadrilateral({[tuple(pt) for pt in self.points]})"

    def __str__(self):
        return f"Quadrilateral with points: {[tuple(pt) for pt in self.points]}"
