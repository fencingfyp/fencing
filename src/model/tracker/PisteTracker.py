from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Line2D:
    p1: tuple[float, float]
    p2: tuple[float, float]

    def as_array(self, type="float") -> np.ndarray:
        return np.array(
            [self.p1, self.p2], dtype=np.float32 if type == "float" else np.int32
        )

    def direction(self) -> np.ndarray:
        v = np.array(self.p2) - np.array(self.p1)
        return v / np.linalg.norm(v)

    def to_tuple(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(map(int, self.p1)), tuple(map(int, self.p2))


@dataclass
class PisteGeometry:
    left_boundary: Line2D
    right_boundary: Line2D
    centre_line: Line2D
    engarde_lines: list[Line2D]
    longitudinal_axis: Line2D


class PisteTracker:
    def __init__(self):
        self.targets: dict[str, PisteTarget] = {}

    def add_target(
        self,
        name: str,
        frame: np.ndarray,
        left_boundary: Line2D,
        right_boundary: Line2D,
        centre_line: Line2D,
        engarde_lines: list[Line2D],
    ):
        target = PisteTarget(
            frame=frame,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            centre_line=centre_line,
            engarde_lines=engarde_lines,
        )
        self.targets[name] = target

    def update_all(self, frame: np.ndarray) -> dict[str, PisteGeometry | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        outputs = {}
        for name, target in self.targets.items():
            outputs[name] = target.update_with_flow(gray)

        return outputs

    def get_geometry(self, name: str) -> PisteGeometry | None:
        target = self.targets.get(name)
        if target is None:
            return None
        return target.get_geometry()


class PisteTarget:
    def __init__(
        self,
        frame: np.ndarray,
        left_boundary: Line2D,
        right_boundary: Line2D,
        centre_line: Line2D,
        engarde_lines: list[Line2D],
        samples_per_line: int = 30,
    ):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.samples_per_line = samples_per_line

        # Store semantic lines
        self.semantic_lines = {
            "left": left_boundary,
            "right": right_boundary,
            "centre": centre_line,
        }

        for i, l in enumerate(engarde_lines):
            self.semantic_lines[f"engarde_{i}"] = l

        # Tracking state
        self.track_points, self.line_slices = self._initialise_tracking_points()

        self.current_geometry = self._compute_geometry(self.semantic_lines)

        self.min_points_per_line = 6
        self.max_error = 20.0

    def _initialise_tracking_points(self):
        all_pts = []
        line_slices = {}
        start = 0

        for name, line in self.semantic_lines.items():
            pts = self._sample_line(line)  # (N,2)
            end = start + len(pts)

            line_slices[name] = slice(start, end)
            all_pts.append(pts)
            start = end

        all_pts = np.vstack(all_pts).astype(np.float32)
        all_pts = all_pts.reshape(-1, 1, 2)

        return all_pts, line_slices

    def _sample_line(self, line: Line2D) -> np.ndarray:
        p1 = np.array(line.p1, dtype=np.float32)
        p2 = np.array(line.p2, dtype=np.float32)

        ts = np.linspace(0.0, 1.0, self.samples_per_line)
        pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        return pts

    def update_with_flow(self, gray: np.ndarray) -> PisteGeometry | None:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.track_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )

        if next_pts is None:
            return None

        status = status.flatten()
        err = err.flatten()

        good_mask = (status == 1) & (err < self.max_error)

        new_lines = {}

        for name, sl in self.line_slices.items():
            line_pts = next_pts[sl].reshape(-1, 2)
            line_good = good_mask[sl]

            good_pts = line_pts[line_good]

            if len(good_pts) < self.min_points_per_line:
                # critical failure if side lines are lost
                if name in ("left", "right"):
                    return None
                continue

            new_lines[name] = self._fit_line(good_pts)

        # Must have both side lines
        if "left" not in new_lines or "right" not in new_lines:
            return None

        self.semantic_lines.update(new_lines)

        self.current_geometry = self._compute_geometry(self.semantic_lines)

        # Re-sample fresh points from updated lines
        self.track_points, self.line_slices = self._initialise_tracking_points()

        self.prev_gray = gray

        return self.current_geometry

    def _fit_line(self, points: np.ndarray) -> Line2D:
        mean = points.mean(axis=0)
        cov = np.cov(points - mean, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        direction = direction / np.linalg.norm(direction)

        p1 = mean - direction * 1000
        p2 = mean + direction * 1000

        return Line2D(tuple(p1), tuple(p2))

    def _compute_geometry(self, lines: dict[str, Line2D]) -> PisteGeometry:
        left = lines["left"]
        right = lines["right"]

        axis = self._compute_axis(left, right)

        centre = lines.get("centre")
        engarde = [v for k, v in lines.items() if k.startswith("engarde")]

        return PisteGeometry(
            left_boundary=left,
            right_boundary=right,
            centre_line=centre,
            engarde_lines=engarde,
            longitudinal_axis=axis,
        )

    def _compute_axis(self, left: Line2D, right: Line2D) -> Line2D:
        v1 = left.direction()
        v2 = right.direction()

        v = (v1 + v2) / 2
        v = v / np.linalg.norm(v)

        mid = (np.array(left.p1) + np.array(right.p1)) / 2

        return Line2D(
            tuple(mid - v * 1000),
            tuple(mid + v * 1000),
        )

    def get_geometry(self) -> PisteGeometry | None:
        return self.current_geometry
