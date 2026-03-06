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
from src.gui.base_task_widget.base_task_widget import BaseTaskWidget
from src.gui.MatchContext import MatchContext
from src.gui.task_graph.task_graph import TasksToIds
from src.gui.util.conversion import pixmap_to_np
from src.model import FrameInfoManager
from src.model.FileManager import FileRole
from src.model.FrameInfoManager import FrameInfoManager
from src.model.Quadrilateral import Quadrilateral
from src.pyside_pipelines.multi_region_cropper.output.csv_quad_output import (
    get_header_row as quad_get_header_row,
)
from src.pyside_pipelines.multi_region_cropper.output.csv_quad_output import (
    row_mapper as quad_row_mapper,
)


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


from dataclasses import dataclass
from enum import Enum


class AttackType(Enum):
    SIMPLE = "simple"
    COUNTER = "counter"
    PARRY = "parry"
    # Future types can be added here, e.g.:
    # COMPOUND = "compound"
    # COUNTER = "counter"


@dataclass
class ScoringEvent:
    frame_id: int
    fencer: str  # "left" | "right"
    third: str  # "left" | "centre" | "right"
    attack_type: AttackType


def generate_heat_map_data(
    fencer_thirds: np.ndarray, momentum_frames: pd.DataFrame
) -> list[ScoringEvent]:
    scoring_events: list[ScoringEvent] = []

    for i in range(len(momentum_frames)):
        frame_id = momentum_frames.iloc[i]["frame_id"]
        momentum_change = momentum_frames.iloc[i]["momentum"]

        if momentum_change == 1:
            left_third, _ = fencer_thirds[frame_id]
            if left_third:
                scoring_events.append(
                    ScoringEvent(frame_id, "left", left_third, AttackType.SIMPLE)
                )

        elif momentum_change == -1:
            _, right_third = fencer_thirds[frame_id]
            if right_third:
                scoring_events.append(
                    ScoringEvent(frame_id, "right", right_third, AttackType.SIMPLE)
                )

        elif momentum_change == 0:
            if len(fencer_thirds[frame_id]) != 2:
                print(
                    f"Warning: fencer_thirds for frame {frame_id}: "
                    f"{fencer_thirds[frame_id]} does not have length 2"
                )
            left_third, right_third = fencer_thirds[frame_id]
            if left_third:
                scoring_events.append(
                    ScoringEvent(frame_id, "left", left_third, AttackType.SIMPLE)
                )
            if right_third:
                scoring_events.append(
                    ScoringEvent(frame_id, "right", right_third, AttackType.SIMPLE)
                )

    return scoring_events


def load_piste_data(piste_csv_path, fps) -> FrameInfoManager:
    return FrameInfoManager(
        piste_csv_path,
        fps=fps,
        header_format=quad_get_header_row(),
        row_mapper=quad_row_mapper,
    )


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

        self.run_started.emit(TasksToIds.GENERATE_HEAT_MAP.value)

        fps, total_frames = get_fps_and_total_frames(
            self.match_context.file_manager.get_original_video()
        )

        fencer_thirds = obtain_fencer_thirds_array(
            input_csv=self.match_context.file_manager.get_path(FileRole.PROCESSED_POSE),
            engarde_quad_frame_manager=load_piste_data(
                self.match_context.file_manager.get_path(FileRole.RAW_PISTE_QUADS),
                fps,
            ),
            fps=fps,
            total_frames=total_frames,
        )

        momentum_csv_path = self.match_context.file_manager.get_path(
            FileRole.RAW_MOMENTUM_DATA
        )
        momentum_frames = load_momentum_diffs(momentum_csv_path)

        heat_map_data = generate_heat_map_data(fencer_thirds, momentum_frames)
        pixmap = heatmap_to_pixmap(heat_map_data)
        self.ui.set_fresh_frame(pixmap_to_np(pixmap))

        self._on_finished()

    def _on_finished(self):
        self.ui.write("Heat map generation completed.")
        self.run_completed.emit(TasksToIds.GENERATE_HEAT_MAP.value)

    def cancel(self):
        pass


import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from PySide6.QtGui import QImage, QPixmap


def heatmap_to_pixmap(
    scoring_events: list[ScoringEvent], figsize=(6, 4), title="Heat Map"
):
    thirds = ["left", "centre", "right"]
    fencers = ["left", "right"]
    fencer_colors = {"left": "steelblue", "right": "indianred"}

    attack_markers = {
        AttackType.SIMPLE: ("x", "Attack", 80),
        AttackType.COUNTER: ("o", "Counter", 60),
        AttackType.PARRY: ("*", "Parry", 100),
    }

    fig, ax = plt.subplots(figsize=figsize)
    canvas = FigureCanvas(fig)

    third_positions = {t: i for i, t in enumerate(thirds)}
    fencer_offsets = {"left": -0.2, "right": 0.2}

    # Group events by (third, fencer, attack_type) → count
    counts: dict[tuple, int] = {}
    for event in scoring_events:
        key = (event.third, event.fencer, event.attack_type)
        counts[key] = counts.get(key, 0) + 1

    # Stack events vertically per (third, fencer) column
    column_heights: dict[tuple, float] = {}  # tracks next y position per column

    for (third, fencer, attack_type), count in counts.items():
        marker, _, marker_size = attack_markers[attack_type]
        color = fencer_colors[fencer]
        x = third_positions[third] + fencer_offsets[fencer]
        y_base = column_heights.get((third, fencer), 0)

        for j in range(count):
            y = y_base + j + 0.5
            ax.scatter(x, y, marker=marker, color=color, s=marker_size, zorder=3)

        ax.text(
            x, y_base + count + 0.1, str(count), ha="center", va="bottom", fontsize=8
        )
        column_heights[(third, fencer)] = y_base + count

    # --- Legends ---
    # Fencer color legend
    fencer_handles = [
        mlines.Line2D(
            [],
            [],
            color=c,
            marker="s",
            linestyle="None",
            markersize=8,
            label=f"{f.capitalize()} fencer",
        )
        for f, c in fencer_colors.items()
    ]
    # Attack type shape legend
    shape_handles = [
        mlines.Line2D(
            [], [], color="grey", marker=m, linestyle="None", markersize=8, label=label
        )
        for _, (m, label, _) in attack_markers.items()
    ]

    legend1 = ax.legend(handles=fencer_handles, loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=shape_handles, loc="upper right", fontsize=8)

    ax.set_xticks(range(len(thirds)))
    ax.set_xticklabels([t.capitalize() for t in thirds])
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(-0.6, len(thirds) - 0.4)
    max_height = max(column_heights.values(), default=1)
    ax.set_ylim(0, max_height + 1.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.subplots_adjust(bottom=0.15)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = QImage(canvas.buffer_rgba(), w, h, QImage.Format_ARGB32)
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
    input_csv: str,
    engarde_quad_frame_manager: FrameInfoManager,
    fps: float,
    total_frames: int,
) -> np.ndarray:
    fencer_pose_frame_manager = FrameInfoManager(
        input_csv, fps, get_header_row_for_fencer_poses_csv(), fencer_poses_row_mapper
    )

    classifier = FencerClassifier()

    output = np.zeros(
        (total_frames, 2), dtype=object
    )  # store (left_third, right_third)

    for frame_id in range(total_frames):
        detections = fencer_pose_frame_manager.get_frame_and_advance(frame_id)
        engarde_quad = (
            engarde_quad_frame_manager.get_frame_and_advance(frame_id)
            .get("quad")
            .get("quad")
        )  # extract Quadrilateral from dict

        if engarde_quad is None or detections is None:
            output[frame_id] = (classifier.prev_left_third, classifier.prev_right_third)
            continue

        mapper = PisteMapper(engarde_quad=engarde_quad)

        left_third, right_third = classifier.classify(
            mapper,
            get_valid_fencer_coords(detections.get(LEFT_FENCER_ID)),
            get_valid_fencer_coords(detections.get(RIGHT_FENCER_ID)),
        )
        output[frame_id] = (left_third, right_third)

    # write to file for debugging
    # with open(
    #     "matches_data/sabre_6.data/fencer_thirds_debug.csv", "w", newline=""
    # ) as f:
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
        match_context.set_file("matches_data/sabre_6.mp4")
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
