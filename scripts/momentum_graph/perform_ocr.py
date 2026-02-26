import argparse
import csv
import os
from typing import Optional

import cv2
import numpy as np

from src.model import OpenCvUi, UiCodes
from src.model.reader.EasyOcrReader import EasyOcrReader
from src.model.reader.SevenSegmentReader import SevenSegmentReader
from src.util.file_names import (
    CROPPED_SCOREBOARD_VIDEO_NAME,
    OCR_OUTPUT_CSV_NAME,
    ORIGINAL_VIDEO_NAME,
)
from src.util.gpu import get_device
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io
from src.util.utils import (
    convert_from_box_to_rect,
    convert_from_rect_to_box,
    generate_select_quadrilateral_instructions,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DO_OCR_EVERY_N_FRAMES = 5
MIN_WINDOW_HEIGHT = 780

OUTPUT_VIDEO_NAME = "perform_ocr_output.mp4"
OUTPUT_OCR_L_NAME = "ocr_left.mp4"
OUTPUT_OCR_R_NAME = "ocr_right.mp4"

# CALIBRATION_N_SAMPLES = 10
# CALIBRATION_MIN_GAP_SECONDS = 7
# CALIBRATION_ACCURACY_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------


def regularise_rectangle(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert an arbitrary quadrilateral selection to an axis-aligned bounding box."""
    return convert_from_rect_to_box(convert_from_box_to_rect(pts))


def extract_roi(frame: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    """Crop the score ROI from a frame given its bounding box positions."""
    x, y, w, h = convert_from_box_to_rect(positions)
    return frame[int(y) : int(y + h), int(x) : int(x + w)]


def select_roi(ui: OpenCvUi, frame: np.ndarray, label: str) -> list[tuple[int, int]]:
    """Ask the user to select a quadrilateral ROI for a named score display."""
    return regularise_rectangle(
        ui.get_n_points(frame, generate_select_quadrilateral_instructions(label))
    )


# ---------------------------------------------------------------------------
# Video writers
# ---------------------------------------------------------------------------


def create_video_writers(
    output_folder: str,
    fps: float,
    ui: OpenCvUi,
    left_positions: list,
    right_positions: list,
) -> tuple[cv2.VideoWriter, cv2.VideoWriter, cv2.VideoWriter]:
    """Instantiate video writers for the main view and both OCR preview windows."""
    main_writer = setup_output_video_io(
        os.path.join(output_folder, OUTPUT_VIDEO_NAME),
        fps,
        ui.get_output_dimensions(),
    )
    l_x, l_y, l_w, l_h = convert_from_box_to_rect(left_positions)
    r_x, r_y, r_w, r_h = convert_from_box_to_rect(right_positions)
    l_writer = setup_output_video_io(
        os.path.join(output_folder, OUTPUT_OCR_L_NAME), fps, (int(l_w), int(l_h))
    )
    r_writer = setup_output_video_io(
        os.path.join(output_folder, OUTPUT_OCR_R_NAME), fps, (int(r_w), int(r_h))
    )
    return main_writer, l_writer, r_writer


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


# def _sample_frame_indices(total_frames: int, fps: float) -> list[int]:
#     """Pick up to CALIBRATION_N_SAMPLES frames spaced at least CALIBRATION_MIN_GAP_SECONDS apart."""
#     min_gap = int(fps * CALIBRATION_MIN_GAP_SECONDS)
#     selected = []
#     attempts = 0
#     while len(selected) < CALIBRATION_N_SAMPLES and attempts < 10_000:
#         candidate = np.random.randint(0, total_frames)
#         if all(abs(candidate - s) >= min_gap for s in selected):
#             selected.append(candidate)
#         attempts += 1
#     return sorted(selected)


# def calibrate_ocr(
#     ui: OpenCvUi,
#     cap: cv2.VideoCapture,
#     ocr_reader: EasyOcrReader,
#     total_frames: int,
#     fps: float,
#     left_positions: list,
#     right_positions: list,
# ) -> float:
#     """
#     Show the user a sample of OCR results on random frames and ask for confirmation.
#     Returns the observed accuracy. Warns if below threshold.
#     """
#     print("Running OCR calibration check on random frames...")
#     indices = _sample_frame_indices(total_frames, fps)
#     n_correct = n_total = 0

#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         positions = left_positions if idx % 2 == 0 else right_positions
#         roi = extract_roi(frame, positions)
#         score, conf = ocr_reader.read(roi)

#         ui.clear_frame()
#         ui.set_fresh_frame(ocr_reader.preprocessor(roi))
#         ui.write_to_ui(
#             f"Score: {score}  Conf: {conf:.2f} | "
#             "1=correct  2=wrong  3=skip  Q=quit calibration"
#         )
#         ui.show_frame()

#         action = ui.get_user_input(
#             0,
#             [UiCodes.CUSTOM_1, UiCodes.CUSTOM_2, UiCodes.CUSTOM_3, UiCodes.QUIT],
#             must_be_valid=True,
#         )
#         if action == UiCodes.QUIT:
#             break
#         elif action == UiCodes.CUSTOM_3:
#             continue
#         n_total += 1
#         if action == UiCodes.CUSTOM_1:
#             n_correct += 1

#     accuracy = n_correct / n_total if n_total > 0 else 0.0
#     print(f"Calibration: {accuracy*100:.1f}% ({n_correct}/{n_total})")
#     if n_total > 0 and accuracy < CALIBRATION_ACCURACY_THRESHOLD:
#         print(
#             f"Warning: accuracy {accuracy*100:.1f}% is below "
#             f"{CALIBRATION_ACCURACY_THRESHOLD*100:.0f}%. "
#             "Consider adjusting the ROI selection or lighting."
#         )
#     return accuracy


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_videos(original_path: str, cropped_path: str) -> None:
    """Ensure both videos exist and have the same frame count."""
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
        raise ValueError(
            f"Frame count mismatch: original={n_orig}, cropped={n_crop}. "
            "Please re-crop the video."
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR fencing scoreboard scores")
    parser.add_argument(
        "output_folder", help="Folder containing input videos and for outputs"
    )
    parser.add_argument(
        "--output-video", action="store_true", help="Write OCR preview videos"
    )
    parser.add_argument(
        "--seven-segment", action="store_true", help="Seven-segment digit mode"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run without writing CSV output"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cropped_path = os.path.join(args.output_folder, CROPPED_SCOREBOARD_VIDEO_NAME)
    original_path = os.path.join(args.output_folder, ORIGINAL_VIDEO_NAME)
    validate_videos(original_path, cropped_path)

    output_csv_path = setup_output_file(args.output_folder, OCR_OUTPUT_CSV_NAME)
    cap, fps, width, height, frame_count = setup_input_video_io(cropped_path)

    ui = OpenCvUi(
        "Performing OCR",
        width=int(width),
        height=int(height),
        display_height=MIN_WINDOW_HEIGHT,
    )

    # --- ROI selection ---
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    left_positions = select_roi(ui, first_frame, "left fencer score")
    right_positions = select_roi(ui, first_frame, "right fencer score")

    if args.seven_segment:
        ocr_reader = SevenSegmentReader()
    else:
        ocr_reader = EasyOcrReader(get_device(), seven_segment=args.seven_segment)

    # --- Preview windows ---
    cv2.namedWindow("OCR Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("OCR Right", cv2.WINDOW_NORMAL)

    # --- Optional video writers ---
    video_writer = l_writer = r_writer = None
    if args.output_video:
        video_writer, l_writer, r_writer = create_video_writers(
            args.output_folder, fps, ui, left_positions, right_positions
        )

    # --- Main loop ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0
    l_score = r_score = None
    l_conf = r_conf = 0.0
    slow = False

    FULL_DELAY = int(1000 / fps)
    FAST_DELAY = max(FULL_DELAY // 16, 1)

    csv_file = open(output_csv_path, "w", newline="") if not args.demo else None
    csv_writer = None
    if csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame_id",
                "left_score",
                "right_score",
                "left_confidence",
                "right_confidence",
            ]
        )

    try:
        while True:
            ret, frame = cap.read()
            if frame_id < 7000:
                frame_id += 1
                continue
            if not ret:
                break

            l_roi = extract_roi(frame, left_positions)
            r_roi = extract_roi(frame, right_positions)

            if frame_id % DO_OCR_EVERY_N_FRAMES == 0:
                l_score, l_conf = ocr_reader.read(l_roi)
                r_score, r_conf = ocr_reader.read(r_roi)
                cv2.imshow("OCR Left", ocr_reader.preprocessor(l_roi))
                cv2.imshow("OCR Right", ocr_reader.preprocessor(r_roi))
                if csv_writer:
                    csv_writer.writerow([frame_id, l_score, r_score, l_conf, r_conf])

            ui.set_fresh_frame(frame)
            ui.refresh_frame()
            ui.write_to_ui(
                f"L: {l_score} ({l_conf:.2f})  R: {r_score} ({r_conf:.2f})  "
                f"| frame {frame_id}/{frame_count}"
            )
            ui.show_frame()

            if args.output_video:
                video_writer.write(ui.current_frame)
                l_writer.write(ocr_reader.preprocessor(l_roi))
                r_writer.write(ocr_reader.preprocessor(r_roi))

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
        if csv_file:
            csv_file.close()
        if args.output_video:
            for w in (video_writer, l_writer, r_writer):
                if w:
                    w.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
