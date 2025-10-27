import argparse
import csv
import os
import cv2
import numpy as np
import easyocr
import torch
from src.model.Ui import Ui
from src.util import UiCodes, convert_to_opencv_format, convert_from_opencv_format,\
    generate_select_quadrilateral_instructions, setup_input_video_io, setup_output_video_io, \
    setup_output_file
from src.model.EasyOcrReader import EasyOcrReader

DO_OCR_EVERY_N_FRAMES = 5  # Set >1 to skip frames for speed (but less frequent score updates)
OUTPUT_CSV_NAME = "raw_scores.csv"
MIN_WINDOW_HEIGHT = 780

def fontify_7segment(binary):
    # assume binary: 0/255, white digits on black background
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,7))
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    smoothed = cv2.dilate(smoothed, kernel_close, iterations=2)
    smoothed = cv2.GaussianBlur(smoothed, (5,5), 0)
    smoothed = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)[1]
    padded = cv2.copyMakeBorder(smoothed, 8,8,8,8, cv2.BORDER_CONSTANT, value=0)
    # compressed = cv2.resize(padded, None, fx=0.8, fy=1.0, interpolation=cv2.INTER_NEAREST)
    return smoothed

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

def convert_from_box_to_rect(box: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Convert 4-point box to x,y,w,h rectangle"""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return (x, y, w, h)

def convert_from_rect_to_box(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Convert x,y,w,h rectangle to 4-point box"""
    x, y, w, h = rect
    return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

def get_output_header_row() -> list[str]:
    return ["frame_id", "left_score", "right_score", "left_confidence", "right_confidence"]

def process_image(image, threshold_boundary, is_7_segment=False):
    # Scale up to help OCR (makes thin strokes thicker)
    gray_up = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 10
    # gray_up = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    # print(gray_up.shape)
    # Light denoising
    # gray_up = cv2.medianBlur(gray_up, 3)

    # apply unsharp masking
    # gaussian = cv2.GaussianBlur(gray_up, (7, 7), 2.0)
    # gray_up = cv2.addWeighted(gray_up, 2, gaussian, -1, 0)
    

    # Adaptive threshold (handles varying lighting)
    gray_up = cv2.threshold(gray_up, threshold_boundary, 255, cv2.THRESH_BINARY)[1]
    # gray_up = fontify_7segment(gray_up)
    
    # print(thresh)
    # thresh = cv2.adaptiveThreshold(
    #     gray_up, 255, 
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
    #     11, 2
    # )[1]
    
    # sharpen image
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5,-1],
    #                    [0, -1, 0]])
    # thresh = cv2.filter2D(gray_up, -1, kernel)

    # Small closing to connect broken strokes
    # kernel = np.ones((3,3), np.uint8)
    # gray_up = cv2.morphologyEx(gray_up, cv2.MORPH_CLOSE, kernel, iterations=5)

    # erode to thin lines
    # kernel = np.ones((3,3), np.uint8)
    # gray_up = cv2.erode(gray_up, kernel, iterations=3)

    # Optional: remove tiny specks with contour filtering
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros_like(thresh)
    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 5:  # keep only larger blobs
    #         cv2.drawContours(mask, [cnt], -1, 255, -1)
    # clean = mask
    return gray_up

def get_device():
    # Select best available device: CUDA -> MPS (Apple) -> CPU
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            mps = getattr(torch.backends, "mps", None)
            mps_available = False
            if mps is not None and hasattr(mps, "is_available"):
                try:
                    mps_available = mps.is_available()
                except Exception:
                    mps_available = False
            return torch.device("mps" if mps_available else "cpu")
    except Exception:
        return torch.device("cpu")
    
def get_parse_args():
    parser = argparse.ArgumentParser(description="Use OCR to read scoreboard")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to output folder for intermediate/final products")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    parser.add_argument("--threshold-boundary", type=int, help="Threshold for binary segmentation", default=120)
    parser.add_argument("--seven-segment", action="store_true", help="Use seven-segment digit recognition mode")
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

def ask_user_confirmation(ui: Ui, frame, threshold_boundary, ocr_reader, n_correct, n_total) -> tuple[int, int, bool]:
    score, conf = extract_score_from_frame(frame, threshold_boundary, ocr_reader)
    processed = process_image(frame, threshold_boundary)
    # rearrange grayscale to 3-channel BGR for UI display
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    ui.clear_frame()
    ui.set_fresh_frame(processed)
    ui.write_to_ui(f"OCR Left Score: {score} (Conf: {conf:.2f}), press 1 if it's correct, 2 if not, 3 to skip")
    ui.show_frame()
    action = ui.take_user_input(0, [UiCodes.CUSTOM_1, UiCodes.CUSTOM_2, UiCodes.CUSTOM_3, UiCodes.QUIT], must_be_valid=True)
    if action == UiCodes.CUSTOM_1:
        return n_correct + 1, n_total + 1, False
    elif action == UiCodes.CUSTOM_2:
        return n_correct, n_total + 1, False
    elif action == UiCodes.CUSTOM_3:
        return n_correct, n_total, False
    return n_correct, n_total, True

def calibrate_ocr(ui: Ui, ocr_reader, cap, threshold_boundary, total_frames, left_score_positions, right_score_positions, threshold_confidence=0.9):
    print("Performing OCR precheck on random frames...")
    sample_frame_indices = random_with_min_gap(total_frames, 15, cap.get(cv2.CAP_PROP_FPS) * 7)  # 15 frames, at least 7 seconds apart
    n_correct = 0
    n_total = 0
    for frame_idx in sample_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (ui.width, ui.height), interpolation=cv2.INTER_CUBIC)
        if frame_idx % 2 == 0:
            frame = extract_score_frame_from_frame(frame, left_score_positions)
        else:
            frame = extract_score_frame_from_frame(frame, right_score_positions)
        n_correct, n_total, early_break = ask_user_confirmation(ui, frame, threshold_boundary, ocr_reader, n_correct, n_total)
        if early_break:
            break
    accuracy = (n_correct / n_total) if n_total > 0 else 0.0
    print(f"OCR Precheck complete. Accuracy: {accuracy*100:.2f}% ({n_correct}/{n_total})")
    if accuracy < threshold_confidence:
        print("Warning: OCR accuracy below threshold. Consider recalibrating or adjusting settings.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

def main():
    args = get_parse_args()
    input_video_path = args.input_video
    output_video_path = args.output_video
    output_folder = args.output_folder
    threshold_boundary = args.threshold_boundary
    use_seven_segment = args.seven_segment

    output_csv_path = setup_output_file(output_folder, OUTPUT_CSV_NAME)

    cap, fps, original_width, original_height, frame_count = setup_input_video_io(input_video_path)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    aspect_ratio = original_width / original_height
    width = original_width if original_height >= MIN_WINDOW_HEIGHT else int(MIN_WINDOW_HEIGHT * aspect_ratio)
    height = original_height if original_height >= MIN_WINDOW_HEIGHT else MIN_WINDOW_HEIGHT

    # UI
    slow = False
    early_exit = False
    ui = Ui("Fencing Analysis", width=int(width), height=int(height))

    video_writer = None
    if output_video_path:
        video_writer = setup_output_video_io(output_video_path, fps, (width, height + ui.text_box_height))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # autoscale frame to fit window if needed
    if original_height < MIN_WINDOW_HEIGHT:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

    ui.set_fresh_frame(frame)

    left_score_positions = ui.get_n_points(generate_select_quadrilateral_instructions("left fencer score display"))
    right_score_positions = ui.get_n_points(generate_select_quadrilateral_instructions("right fencer score display"))

    # Initialise OCR
    device = get_device()
    print(f"Using device: {device}")
    ocr_reader = EasyOcrReader(device, seven_segment=use_seven_segment)

    calibrate_ocr(ui, ocr_reader, cap, threshold_boundary, frame_count, left_score_positions, right_score_positions)

    ocr_window_l = "OCR Preview L"
    cv2.namedWindow(ocr_window_l, cv2.WINDOW_NORMAL)
    ocr_window_r = "OCR Preview R"
    cv2.namedWindow(ocr_window_r, cv2.WINDOW_NORMAL)

    frame_id = 0
    with open(output_csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        header_row = get_output_header_row()
        csv_writer.writerow(header_row)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # autoscale frame to fit window if needed
            if original_height < MIN_WINDOW_HEIGHT:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            ui.set_fresh_frame(frame)
            ui.refresh_frame()

            l_frame = extract_score_frame_from_frame(frame, left_score_positions)
            r_frame = extract_score_frame_from_frame(frame, right_score_positions)

            if frame_id % DO_OCR_EVERY_N_FRAMES == 0:
                l_score, l_conf = extract_score_from_frame(l_frame, threshold_boundary, ocr_reader, ocr_window_l)
                r_score, r_conf = extract_score_from_frame(r_frame, threshold_boundary, ocr_reader, ocr_window_r)

                csv_writer.writerow([frame_id, l_score, r_score, l_conf, r_conf])

            ui.write_to_ui(
                f"Left score: {l_score} Right score: {r_score} | "
                f"OCR confidence L: {l_conf:.2f} R: {r_conf:.2f}"
            )
            ui.show_frame()

            if video_writer:
                video_writer.write(ui.current_frame)

            delay: int = FULL_DELAY if slow else FAST_FORWARD
            action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW])
            if action == UiCodes.TOGGLE_SLOW:
                slow = not slow
                print(f"Slow mode {'enabled' if slow else 'disabled'}.")
            elif action == UiCodes.QUIT:  # q or Esc to quit
                break

            if early_exit:
                break
            frame_id += 1

        if video_writer:
            video_writer.release()
        
        cap.release()
        ui.close()
        cv2.destroyWindow(ocr_window_l)
        cv2.destroyWindow(ocr_window_r)

def extract_score_frame_from_frame(frame, score_positions):
    rect = convert_from_box_to_rect(score_positions)
    x, y, w, h = rect
    out = frame[y:y+h, x:x+w]
    return out

def extract_score_from_frame(frame, threshold_boundary, ocr_reader: EasyOcrReader, window_name=None):
    processed = process_image(frame, threshold_boundary)
    if window_name:
        cv2.imshow(window_name, processed)
    return ocr_reader.read(processed)  # returns score, confidence
 

if __name__ == "__main__":
    main()