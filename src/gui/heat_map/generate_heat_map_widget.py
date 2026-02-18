import csv
import os
from typing import override

import cv2
import numpy as np
import pandas as pd

from scripts.manual_track_fencers import (
    get_header_row as get_header_row_for_fencer_poses_csv,
)
from scripts.manual_track_fencers import row_mapper as fencer_poses_row_mapper
from scripts.visualise_fencers_on_piste import LEFT_FENCER_ID, RIGHT_FENCER_ID
from src.gui.util.conversion import pixmap_to_np
from src.gui.util.task_graph import HeatMapTasksToIds
from src.model import FrameInfoManager
from src.model.FileManager import FileRole
from src.model.FrameInfoManager import FrameInfoManager
from src.model.Quadrilateral import Quadrilateral
from src.pyside.MatchContext import MatchContext
from src.pyside_pipelines.multi_region_cropper.output.csv_oneshot_quad_output import (
    row_mapper as piste_csv_row_mapper,
)

from ..momentum_graph.base_task_widget import BaseTaskWidget


def load_momentum_diffs(path) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("Momentum data file not found.")
    df = pd.read_csv(path)
    df_diff = (
        df.assign(momentum=df["momentum"].diff())
        .iloc[1:]
        .reset_index(drop=True)
        .astype({"frame_id": int, "momentum": int})
    )

    return df_diff


def generate_heat_map_data(
    fencer_thirds: np.ndarray, momentum_frames: pd.DataFrame
) -> dict[str, dict[str, int]]:
    heat_map = {
        "left": {"left": 0, "right": 0},
        "centre": {"left": 0, "right": 0},
        "right": {"left": 0, "right": 0},
    }

    # scoring_order = []  # accumulate scorer order

    for i in range(len(momentum_frames)):
        frame_id = momentum_frames.iloc[i]["frame_id"]
        momentum_change = momentum_frames.iloc[i]["momentum"]

        if momentum_change == 1:
            left_third, _ = fencer_thirds[frame_id]
            if left_third:
                heat_map[left_third]["left"] += 1
                # scoring_order.append((frame_id, "left", left_third))

        elif momentum_change == -1:
            _, right_third = fencer_thirds[frame_id]
            if right_third:
                heat_map[right_third]["right"] += 1
                # scoring_order.append((frame_id, "right", right_third))

        elif momentum_change == 0:
            left_third, right_third = fencer_thirds[frame_id]

            # left first for simultaneous
            if left_third:
                heat_map[left_third]["left"] += 1
                # scoring_order.append((frame_id, "left", left_third))

            if right_third:
                heat_map[right_third]["right"] += 1
                # scoring_order.append((frame_id, "right", right_third))
    # print("Scoring order:")
    # for idx, (frame_id, fencer, third) in enumerate(scoring_order, 1):
    #     print(f"{idx}. {int(frame_id/25)} ({fencer}) - {third}")

    return heat_map


def load_piste_data(piste_csv_path) -> Quadrilateral:
    with open(piste_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        row = next(reader)  # assume only one row of data
        return piste_csv_row_mapper(row)
    raise ValueError(f"No piste data found in {piste_csv_path}")


class GenerateHeatMapWidget(BaseTaskWidget):
    def __init__(self, match_context, parent=None):
        super().__init__(match_context, parent)

    @override
    def setup(self):
        self.ui.write("Press 'Run' to start tracking fencers.")
        self.run_task()

    def run_task(self):
        if not self.match_context.file_manager:
            return

        self.run_started.emit(HeatMapTasksToIds.GENERATE_HEAT_MAP)

        fencer_thirds = obtain_fencer_thirds_array(
            input_video=self.match_context.file_manager.get_original_video(),
            input_csv=self.match_context.file_manager.get_path(FileRole.PROCESSED_POSE),
            engarde_quad=load_piste_data(
                self.match_context.file_manager.get_path(FileRole.RAW_PISTE_QUADS)
            ),
        )
        momentum_csv_path = self.match_context.file_manager.get_path(
            FileRole.MOMENTUM_DATA
        )
        momentum_frames = load_momentum_diffs(momentum_csv_path)

        heat_map_data = generate_heat_map_data(fencer_thirds, momentum_frames)
        pixmap = heatmap_to_pixmap(heat_map_data)
        self.ui.set_fresh_frame(pixmap_to_np(pixmap))

        self._on_finished()

    def _on_finished(self):
        self.ui.write("Heat map generation completed.")
        self.run_completed.emit(HeatMapTasksToIds.GENERATE_HEAT_MAP)

    def cancel(self):
        pass


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PySide6.QtGui import QImage, QPixmap


def heatmap_to_pixmap(heat_map, figsize=(6, 4), title="Heat Map"):
    categories = list(heat_map.keys())
    subcats = list(next(iter(heat_map.values())).keys())

    values = {sub: [heat_map[c][sub] for c in categories] for sub in subcats}

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    canvas = FigureCanvas(fig)

    colors = ["steelblue", "indianred"]

    for i, sub in enumerate(subcats):
        bars = ax.bar(
            x - width / 2 + i * width,
            values[sub],
            width,
            label=sub,
            color=colors[i],
        )

        # Faster text placement (no annotate)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(height),
                ha="center",
                va="bottom",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    fig.subplots_adjust(bottom=0.2)

    canvas.draw()
    width, height = canvas.get_width_height()
    image = QImage(canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
    pixmap = QPixmap.fromImage(image)

    plt.close(fig)

    return pixmap


# =========================
# Mapping Logic
# =========================


class PisteMapper:
    PISTE_LENGTH_M = 14.0
    ENGARDE_BOX_LENGTH_M = 4.0
    ENGARDE_LEFT_M = (PISTE_LENGTH_M - ENGARDE_BOX_LENGTH_M) / 2

    def __init__(self, engarde_quad: Quadrilateral):
        self.engarde_quad = engarde_quad
        self.rect_w, self.rect_h = 400, 100
        self.H = self._compute_homography()

    def _compute_homography(self):
        src_pts = np.array(self.engarde_quad.points, dtype=np.float32)
        dst_pts = np.array(
            [
                [0, 0],
                [self.rect_w - 1, 0],
                [self.rect_w - 1, self.rect_h - 1],
                [0, self.rect_h - 1],
            ],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def warp_point(self, pt: tuple[int, int]) -> float:
        pts = np.array([[pt]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return float(warped[0, 0, 0])  # return warped x only

    def warped_x_to_piste_m(self, warped_x: float) -> float:
        frac = warped_x / self.rect_w
        return self.ENGARDE_LEFT_M + frac * self.ENGARDE_BOX_LENGTH_M

    def get_third(self, piste_x_m: float) -> str:
        third_length = self.PISTE_LENGTH_M / 3
        if piste_x_m < third_length:
            return "left"
        elif piste_x_m < 2 * third_length:
            return "centre"
        else:
            return "right"


class FencerClassifier:
    """Keeps track of last known third per fencer."""

    def __init__(self):
        self.prev_left_third = None
        self.prev_right_third = None

    def classify(
        self,
        mapper: PisteMapper,
        left_coords: tuple[int, int] | None,
        right_coords: tuple[int, int] | None,
    ) -> tuple[str | None, str | None]:

        # ---- LEFT ----
        if left_coords is not None:
            warped = mapper.warp_point(left_coords)
            piste_x = mapper.warped_x_to_piste_m(warped)
            left_third = mapper.get_third(piste_x)
            self.prev_left_third = left_third
        else:
            left_third = self.prev_left_third

        # ---- RIGHT ----
        if right_coords is not None:
            warped = mapper.warp_point(right_coords)
            piste_x = mapper.warped_x_to_piste_m(warped)
            right_third = mapper.get_third(piste_x)
            self.prev_right_third = right_third
        else:
            right_third = self.prev_right_third

        return left_third, right_third


# =========================
# Fencer Validation
# =========================


def get_valid_fencer_coords(
    fencer_position: dict, confidence_thresh: float = 0.5
) -> tuple[int, int] | None:
    if fencer_position is None:
        return None

    keypoints = fencer_position.get("keypoints", [])
    if len(keypoints) <= 16:
        return None

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if (
        left_ankle[2] > confidence_thresh
        and right_ankle[2] > confidence_thresh
        and fencer_position.get("confidence", 0) > confidence_thresh
    ):
        cx = int((left_ankle[0] + right_ankle[0]) / 2)
        cy = int((left_ankle[1] + right_ankle[1]) / 2)
        return cx, cy

    return None


# =========================
# Core Processing Logic
# =========================


def classify_fencers(
    detections: dict,
    mapper: PisteMapper,
    prev_left=None,
    prev_right=None,
):
    left_dict = detections.get(LEFT_FENCER_ID, prev_left)
    right_dict = detections.get(RIGHT_FENCER_ID, prev_right)

    left_coords = get_valid_fencer_coords(left_dict)
    right_coords = get_valid_fencer_coords(right_dict)

    left_third = None
    right_third = None

    if left_coords:
        warped_x = mapper.warp_point(left_coords)
        piste_x = mapper.warped_x_to_piste_m(warped_x)
        left_third = mapper.get_third(piste_x)

    if right_coords:
        warped_x = mapper.warp_point(right_coords)
        piste_x = mapper.warped_x_to_piste_m(warped_x)
        right_third = mapper.get_third(piste_x)

    return left_third, right_third


def get_fps_and_total_frames(video_path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def obtain_fencer_thirds_array(
    input_video: str,
    input_csv: str,
    engarde_quad: Quadrilateral,
) -> np.ndarray:
    fps, total_frames = get_fps_and_total_frames(input_video)
    frame_manager = FrameInfoManager(
        input_csv, fps, get_header_row_for_fencer_poses_csv(), fencer_poses_row_mapper
    )

    mapper = PisteMapper(engarde_quad)
    classifier = FencerClassifier()

    output = np.zeros(
        (total_frames, 2), dtype=object
    )  # store (left_third, right_third)

    for frame_id in range(total_frames):
        detections = frame_manager.get_frame_and_advance(frame_id)
        if detections is None:
            output[frame_id] = (classifier.prev_left_third, classifier.prev_right_third)
            continue

        left_third, right_third = classifier.classify(
            mapper,
            get_valid_fencer_coords(detections.get(LEFT_FENCER_ID)),
            get_valid_fencer_coords(detections.get(RIGHT_FENCER_ID)),
        )
        output[frame_id] = (left_third, right_third)

    # write to file for debugging
    # with open("matches_data/epee_3.data/fencer_thirds_debug.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["frame_id", "left_third", "right_third"])
    #     for frame_id in range(total_frames):
    #         writer.writerow([frame_id, *output[frame_id]])

    return output


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = GenerateHeatMapWidget(match_context)
        match_context.set_file("matches_data/epee_3.mp4")
        widget.show()
        sys.exit(app.exec())

    # Run the profiler and save stats to a file
    cProfile.run("main()", "profile.stats")

    # Load stats
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()  # remove extraneous path info
    stats.sort_stats("tottime")  # sort by time

    # Print only top 10 functions
    stats.print_stats(10)
