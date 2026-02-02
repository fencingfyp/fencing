import argparse
import csv
import os

import cv2
import numpy as np

from src.model import OpenCvUi, UiCodes
from src.model.EasyOcrReader import EasyOcrReader
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

DO_OCR_EVERY_N_FRAMES = (
    5  # Set >1 to skip frames for speed (but less frequent score updates)
)
MIN_WINDOW_HEIGHT = 780

OUTPUT_VIDEO_NAME = "perform_ocr_output.mp4"
OUTPUT_OCR_L_WINDOW = "ocr_left.mp4"
OUTPUT_OCR_R_WINDOW = "ocr_right.mp4"


def fontify_7segment(binary):
    # assume binary: 0/255, white digits on black background

    # --- Step 1: vertical erosion to thin right bars (helps distinguish 9 from 4)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    thinned = cv2.erode(binary, kernel_v, iterations=1)

    # --- Step 2: horizontal dilation to strengthen crossbars (makes 4 more distinct)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    enhanced = cv2.dilate(thinned, kernel_h, iterations=1)

    # --- Step 3: gentle closing to fill small segment gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close)

    # --- Step 4: optional small opening to remove faint inactive segments
    # kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # --- Step 5: re-binarise to ensure crisp edges
    final = cv2.threshold(closed, 128, 255, cv2.THRESH_BINARY)[1]

    return final


def row_mapper(row: list[str]) -> dict[str, any]:
    # Convert row to dict
    id = row[1]
    box = list(map(int, list(map(float, row[2:10]))))
    # map to tuple of 4 points
    box = [(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])]

    return {
        "id": id,
        "box": box,  # [x1, y1, x2, y2, x3, y3, x4, y4]
    }


def get_output_header_row() -> list[str]:
    return [
        "frame_id",
        "left_score",
        "right_score",
        "left_confidence",
        "right_confidence",
    ]


def validate_input_video(original_path: str, cropped_path: str) -> bool:
    # Check if cropped video exists and is valid
    if not os.path.exists(cropped_path):
        raise IOError(
            f"Error: Cropped video not found at {cropped_path}. Please run cropping first."
        )

    cap = cv2.VideoCapture(cropped_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open cropped video at {cropped_path}.")

    cropped_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    orig_cap = cv2.VideoCapture(original_path)
    if not orig_cap.isOpened():
        raise IOError(f"Error: Could not open original video at {original_path}.")

    original_total_frames = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_cap.release()

    if cropped_total_frames != original_total_frames:
        raise ValueError(
            f"Error: Cropped video frame count ({cropped_total_frames}) does not match original video frame count ({original_total_frames}). Please re-crop the video."
        )


def process_image(image, threshold_boundary, is_seven_segment=False):
    gray_up = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_up = cv2.copyMakeBorder(
        gray_up, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Adaptive threshold (handles varying lighting)
    gray_up = cv2.threshold(gray_up, threshold_boundary, 255, cv2.THRESH_BINARY)[1]
    # if is_seven_segment:
    #     gray_up = fontify_7segment(gray_up)

    return cv2.cvtColor(gray_up, cv2.COLOR_GRAY2BGR)


def get_parse_args():
    parser = argparse.ArgumentParser(description="Use OCR to read scoreboard")
    parser.add_argument(
        "output_folder", help="Path to output folder for intermediate/final products"
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="If set, outputs video with OCR results",
    )
    parser.add_argument(
        "--threshold-boundary",
        type=int,
        help="Threshold for binary segmentation",
        default=120,
    )
    parser.add_argument(
        "--seven-segment",
        action="store_true",
        help="Use seven-segment digit recognition mode",
    )
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output any csv"
    )
    return parser.parse_args()


def random_with_min_gap(total_frames, n, min_gap):
    selected = []
    attempts = 0
    while len(selected) < n and attempts < 10_000:
        candidate = np.random.randint(0, total_frames)
        if all(abs(candidate - s) >= min_gap for s in selected):
            selected.append(candidate)
        attempts += 1
    return np.sort(selected)


def ask_user_confirmation(
    ui: OpenCvUi, frame, threshold_boundary, ocr_reader, n_correct, n_total
) -> tuple[int, int, bool]:
    score, conf = extract_score_from_frame(frame, threshold_boundary, ocr_reader)
    processed = process_image(frame, threshold_boundary, ocr_reader.seven_segment)
    ui.clear_frame()
    ui.set_fresh_frame(processed)
    ui.write_to_ui(
        f"OCR Left Score: {score} (Conf: {conf:.2f}), press 1 if it's correct, 2 if not, 3 to skip"
    )
    ui.show_frame()
    action = ui.get_user_input(
        0,
        [UiCodes.CUSTOM_1, UiCodes.CUSTOM_2, UiCodes.CUSTOM_3, UiCodes.QUIT],
        must_be_valid=True,
    )
    if action == UiCodes.CUSTOM_1:
        return n_correct + 1, n_total + 1, False
    elif action == UiCodes.CUSTOM_2:
        return n_correct, n_total + 1, False
    elif action == UiCodes.CUSTOM_3:
        return n_correct, n_total, False
    return n_correct, n_total, True


def calibrate_ocr(
    ui: OpenCvUi,
    ocr_reader,
    cap,
    threshold_boundary,
    total_frames,
    left_score_positions,
    right_score_positions,
    threshold_confidence=0.9,
):
    print("Performing OCR precheck on random frames...")
    sample_frame_indices = random_with_min_gap(
        total_frames, 15, cap.get(cv2.CAP_PROP_FPS) * 7
    )  # 15 frames, at least 7 seconds apart
    n_correct = 0
    n_total = 0
    for frame_idx in sample_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(
            frame, (ui.display_width, ui.display_height), interpolation=cv2.INTER_CUBIC
        )
        if frame_idx % 2 == 0:
            frame = extract_score_frame_from_frame(frame, left_score_positions)
        else:
            frame = extract_score_frame_from_frame(frame, right_score_positions)
        n_correct, n_total, early_break = ask_user_confirmation(
            ui, frame, threshold_boundary, ocr_reader, n_correct, n_total
        )
        if early_break:
            break
    accuracy = (n_correct / n_total) if n_total > 0 else 0.0
    print(
        f"OCR Precheck complete. Accuracy: {accuracy*100:.2f}% ({n_correct}/{n_total})"
    )
    if accuracy < threshold_confidence:
        print(
            "Warning: OCR accuracy below threshold. Consider recalibrating or adjusting settings."
        )


def regularise_rectangle(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return convert_from_rect_to_box(convert_from_box_to_rect(pts))


def main():
    args = get_parse_args()
    output_video = args.output_video
    output_folder = args.output_folder
    threshold_boundary = args.threshold_boundary
    use_seven_segment = args.seven_segment
    demo_mode = args.demo

    input_video_path = os.path.join(
        output_folder,
        CROPPED_SCOREBOARD_VIDEO_NAME,
    )
    original_video_path = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    validate_input_video(original_video_path, input_video_path)

    output_csv_path = setup_output_file(output_folder, OCR_OUTPUT_CSV_NAME)

    cap, fps, original_width, original_height, frame_count = setup_input_video_io(
        input_video_path
    )
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False
    ui = OpenCvUi(
        "Performing OCR",
        width=int(original_width),
        height=int(original_height),
        display_height=MIN_WINDOW_HEIGHT,
    )

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    left_score_positions = regularise_rectangle(
        ui.get_n_points(
            frame,
            generate_select_quadrilateral_instructions("left fencer score display"),
        )
    )
    right_score_positions = regularise_rectangle(
        ui.get_n_points(
            frame,
            generate_select_quadrilateral_instructions("right fencer score display"),
        )
    )

    # Initialise OCR
    device = get_device()
    print(f"Using device: {device}")
    ocr_reader = EasyOcrReader(device, seven_segment=use_seven_segment)

    calibrate_ocr(
        ui,
        ocr_reader,
        cap,
        threshold_boundary,
        frame_count,
        left_score_positions,
        right_score_positions,
    )

    ocr_window_l = "OCR Preview L"
    cv2.namedWindow(ocr_window_l, cv2.WINDOW_NORMAL)
    ocr_window_r = "OCR Preview R"
    cv2.namedWindow(ocr_window_r, cv2.WINDOW_NORMAL)

    video_writer = None
    if output_video:
        output_video_path = os.path.join(output_folder, OUTPUT_VIDEO_NAME)
        ocr_window_l_path = os.path.join(output_folder, OUTPUT_OCR_L_WINDOW)
        ocr_window_r_path = os.path.join(output_folder, OUTPUT_OCR_R_WINDOW)
        video_writer = setup_output_video_io(
            output_video_path, fps, ui.get_output_dimensions()
        )
        _, _, w1, h1 = convert_from_box_to_rect(left_score_positions)
        _, _, w2, h2 = convert_from_box_to_rect(right_score_positions)
        ocr_window_l_writer = setup_output_video_io(ocr_window_l_path, fps, (w1, h1))
        ocr_window_r_writer = setup_output_video_io(ocr_window_r_path, fps, (w2, h2))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    frame_id = 0
    mode = "a" if demo_mode else "w"
    with open(output_csv_path, mode, newline="") as f:
        csv_writer = csv.writer(f)
        header_row = get_output_header_row()
        if not demo_mode:
            csv_writer.writerow(header_row)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ui.set_fresh_frame(frame)
            ui.refresh_frame()

            l_frame = extract_score_frame_from_frame(
                ui.get_current_frame(), left_score_positions
            )
            r_frame = extract_score_frame_from_frame(
                ui.get_current_frame(), right_score_positions
            )

            if frame_id % DO_OCR_EVERY_N_FRAMES == 0:
                l_score, l_conf = extract_score_from_frame(
                    l_frame, threshold_boundary, ocr_reader, ocr_window_l
                )
                r_score, r_conf = extract_score_from_frame(
                    r_frame, threshold_boundary, ocr_reader, ocr_window_r
                )

                if not demo_mode:
                    csv_writer.writerow([frame_id, l_score, r_score, l_conf, r_conf])

            ui.write_to_ui(
                f"Left score: {l_score} Right score: {r_score} | "
                f"OCR confidence L: {l_conf:.2f} R: {r_conf:.2f}"
            )
            ui.show_frame()

            if output_video:
                ocr_window_l_writer.write(
                    process_image(l_frame, threshold_boundary, use_seven_segment)
                )
                ocr_window_r_writer.write(
                    process_image(r_frame, threshold_boundary, use_seven_segment)
                )
                video_writer.write(ui.current_frame)

            delay: int = FULL_DELAY if slow else FAST_FORWARD
            action = ui.get_user_input(delay)
            if action == UiCodes.TOGGLE_SLOW:
                slow = not slow
                print(f"Slow mode {'enabled' if slow else 'disabled'}.")
            elif action == UiCodes.QUIT:  # q or Esc to quit
                break
            elif action == UiCodes.PAUSE:
                early_exit = ui.handle_pause()

            if early_exit:
                break
            frame_id += 1

        if output_video:
            video_writer.release()
            ocr_window_l_writer.release()
            ocr_window_r_writer.release()

        cap.release()
        ui.close_additional_windows()
        cv2.destroyWindow(ocr_window_l)
        cv2.destroyWindow(ocr_window_r)


def extract_score_frame_from_frame(frame, score_positions):
    x, y, w, h = convert_from_box_to_rect(score_positions)
    out = frame[y : y + h, x : x + w]
    return out


def extract_score_from_frame(
    frame, threshold_boundary, ocr_reader: EasyOcrReader, window_name=None
):
    processed = process_image(frame, threshold_boundary, ocr_reader.seven_segment)
    if window_name:
        cv2.imshow(window_name, processed)
    return ocr_reader.read(processed)  # returns score, confidence


if __name__ == "__main__":
    main()
