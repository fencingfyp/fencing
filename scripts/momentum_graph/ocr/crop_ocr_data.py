import argparse
import bisect
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from src.model import OpenCvUi, UiCodes
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.gpu import get_device
from src.util.io import setup_input_video_io, setup_output_file
from src.util.utils import (
    convert_from_box_to_rect,
    convert_from_rect_to_box,
    generate_select_quadrilateral_instructions,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTRACT_EVERY_N_FRAMES = 5  # temporal subsampling — adjacent frames are near-identical
TRANSITION_EXCLUSION_FRAMES = 15  # frames to skip either side of a score change;
# displays may flicker or show partial updates mid-transition
MIN_WINDOW_HEIGHT = 780
METADATA_FILENAME = "collection_metadata.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LabelWindow:
    frame_start: int
    frame_end: int  # inclusive
    score_left: int
    score_right: int


@dataclass
class CollectionMetadata:
    """Written to the output root after a successful run.
    Re-running will detect this file and abort to prevent duplicate data."""

    source_video: str
    label_csv: str
    extract_every_n_frames: int
    transition_exclusion_frames: int
    total_crops_written: int


# ---------------------------------------------------------------------------
# Label CSV loading
# ---------------------------------------------------------------------------


def load_label_windows(csv_path: str) -> list[LabelWindow]:
    """
    Load the label CSV. Expected format (header required):

        frame_start,frame_end,score_left,score_right
        0,1832,0,0
        1833,2104,1,0

    Rows must be non-overlapping. Frames outside all windows are skipped,
    which is useful for excluding non-playing periods.
    """
    windows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            windows.append(
                LabelWindow(
                    frame_start=int(row["frame_start"]),
                    frame_end=int(row["frame_end"]),
                    score_left=int(row["score_left"]),
                    score_right=int(row["score_right"]),
                )
            )
    windows.sort(key=lambda w: w.frame_start)
    return windows


def get_label_for_frame(
    frame_id: int,
    windows: list[LabelWindow],
    exclusion: int,
) -> Optional[tuple[int, int]]:
    """
    Return (score_left, score_right) for a frame, or None if it should be skipped.
    Frames within `exclusion` frames of any window boundary are excluded to avoid
    capturing mid-transition display states.
    """
    # Binary search to find candidate window
    starts = [w.frame_start for w in windows]
    idx = bisect.bisect_right(starts, frame_id) - 1
    if idx < 0:
        return None

    window = windows[idx]
    if frame_id > window.frame_end:
        return None

    # Exclude frames near the start or end of the window (transition zone)
    if frame_id < window.frame_start + exclusion:
        return None
    if frame_id > window.frame_end - exclusion:
        return None

    return (window.score_left, window.score_right)


# ---------------------------------------------------------------------------
# ROI helpers (unchanged from perform_ocr.py)
# ---------------------------------------------------------------------------


def regularise_rectangle(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return convert_from_rect_to_box(convert_from_box_to_rect(pts))


def extract_roi(frame: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    x, y, w, h = convert_from_box_to_rect(positions)
    return frame[int(y) : int(y + h), int(x) : int(x + w)]


def select_roi(ui: OpenCvUi, frame: np.ndarray, label: str) -> list[tuple[int, int]]:
    return regularise_rectangle(
        ui.get_n_points(frame, generate_select_quadrilateral_instructions(label))
    )


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------


def setup_output_dir(output_dir: str) -> None:
    """
    Create the ImageFolder-compatible class directories (0–15).
    Raises if a metadata file already exists — re-running would produce
    duplicate images in the class folders and corrupt class balance.
    """
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    if os.path.exists(metadata_path):
        raise RuntimeError(
            f"Output directory '{output_dir}' already contains '{METADATA_FILENAME}'. "
            "This directory has already been populated. Use a new output directory "
            "or delete the metadata file if you intentionally want to overwrite."
        )
    for class_id in range(16):
        os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)


def write_metadata(
    output_dir: str,
    source_video: str,
    label_csv: str,
    total_crops: int,
) -> None:
    metadata = CollectionMetadata(
        source_video=os.path.abspath(source_video),
        label_csv=os.path.abspath(label_csv),
        extract_every_n_frames=EXTRACT_EVERY_N_FRAMES,
        transition_exclusion_frames=TRANSITION_EXCLUSION_FRAMES,
        total_crops_written=total_crops,
    )
    path = os.path.join(output_dir, METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    print(f"Metadata written to {path}")


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def save_crop(
    roi: np.ndarray,
    output_dir: str,
    class_id: int,
    video_name: str,
    frame_id: int,
    side: str,  # 'L' or 'R'
) -> None:
    """
    Save a crop into the appropriate class subfolder.
    Filename encodes provenance (video, frame, side) for later debugging.
    e.g. v01_f004820_L.png
    """
    filename = f"{video_name}_f{frame_id:07d}_{side}.png"
    out_path = os.path.join(output_dir, str(class_id), filename)
    cv2.imwrite(out_path, roi)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_videos(original_path: str, cropped_path: str) -> None:
    for path, label in [(cropped_path, "Cropped"), (original_path, "Original")]:
        if not os.path.exists(path):
            raise IOError(f"{label} video not found: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {label} video: {path}")
        cap.release()

    def frame_count(p):
        c = cv2.VideoCapture(p)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.release()
        return n

    n_orig = frame_count(original_path)
    n_crop = frame_count(cropped_path)
    if n_orig != n_crop:
        raise ValueError(f"Frame count mismatch: original={n_orig}, cropped={n_crop}.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labelled score crops for training dataset"
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing the source videos (original + cropped scoreboard)",
    )
    parser.add_argument(
        "output_dir", help="Root output directory for the ImageFolder dataset"
    )
    parser.add_argument(
        "label_csv", help="CSV file mapping frame ranges to score labels"
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="Short identifier embedded in saved filenames, e.g. 'v01'. "
        "Defaults to the input folder basename.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cropped_path = os.path.join(args.input_folder, CROPPED_SCOREBOARD_VIDEO_NAME)
    original_path = os.path.join(args.input_folder, ORIGINAL_VIDEO_NAME)
    validate_videos(original_path, cropped_path)

    label_windows = load_label_windows(args.label_csv)
    print(f"Loaded {len(label_windows)} label windows.")

    setup_output_dir(args.output_dir)

    video_name = args.video_name or os.path.basename(os.path.abspath(args.input_folder))

    cap, fps, width, height, frame_count = setup_input_video_io(cropped_path)

    ui = OpenCvUi(
        "Collecting Dataset",
        width=int(width),
        height=int(height),
        display_height=MIN_WINDOW_HEIGHT,
    )

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    left_positions = select_roi(ui, first_frame, "left fencer score")
    right_positions = select_roi(ui, first_frame, "right fencer score")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0
    total_crops = 0
    slow = False

    FULL_DELAY = int(1000 / fps)
    FAST_DELAY = max(FULL_DELAY // 16, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames on the sampling cadence
            if frame_id % EXTRACT_EVERY_N_FRAMES == 0:
                labels = get_label_for_frame(
                    frame_id, label_windows, TRANSITION_EXCLUSION_FRAMES
                )

                if labels is not None:
                    score_left, score_right = labels
                    l_roi = extract_roi(frame, left_positions)
                    r_roi = extract_roi(frame, right_positions)

                    save_crop(
                        l_roi, args.output_dir, score_left, video_name, frame_id, "L"
                    )
                    save_crop(
                        r_roi, args.output_dir, score_right, video_name, frame_id, "R"
                    )
                    total_crops += 2

            ui.set_fresh_frame(frame)
            ui.refresh_frame()
            ui.write_to_ui(
                f"Crops saved: {total_crops}  |  frame {frame_id}/{frame_count}"
            )
            ui.show_frame()

            action = ui.get_user_input(FULL_DELAY if slow else FAST_DELAY)
            if action == UiCodes.TOGGLE_SLOW:
                slow = not slow
            elif action == UiCodes.QUIT:
                break
            elif action == UiCodes.PAUSE:
                if ui.handle_pause():
                    break

            frame_id += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    write_metadata(
        args.output_dir,
        source_video=cropped_path,
        label_csv=args.label_csv,
        total_crops=total_crops,
    )
    print(f"Done. {total_crops} crops saved to {args.output_dir}")


if __name__ == "__main__":
    main()
