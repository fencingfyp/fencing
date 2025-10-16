import argparse
import csv
import os
import cv2
import numpy as np
import easyocr
import torch
from src.model.Ui import Ui
from src.util import UiCodes, convert_to_opencv_format, convert_from_opencv_format, generate_select_quadrilateral_instructions

DO_OCR_EVERY_N_FRAMES = 5  # Set >1 to skip frames for speed (but less frequent score updates)

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

def process_image(image, threshold_boundary):
    # Scale up to help OCR (makes thin strokes thicker)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 10
    gray_up = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    # print(gray_up.shape)
    # Light denoising
    # gray_up = cv2.medianBlur(gray_up, 3)

    # apply unsharp masking
    # gaussian = cv2.GaussianBlur(gray_up, (7, 7), 2.0)
    # gray_up = cv2.addWeighted(gray_up, 2, gaussian, -1, 0)
    

    # Adaptive threshold (handles varying lighting)
    gray_up = cv2.threshold(gray_up, threshold_boundary, 255, cv2.THRESH_BINARY)[1]
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

def main():
    parser = argparse.ArgumentParser(description="Use OCR to read scoreboard")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to output folder for intermediate/final products")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    parser.add_argument("--threshold-boundary", type=int, help="Threshold for binary segmentation", default=120)
    args = parser.parse_args() 

    input_video_path = args.input_video
    output_video_path = args.output_video
    output_folder = args.output_folder
    threshold_boundary = args.threshold_boundary


    output_csv_path = os.path.join(output_folder, "raw_scores.csv")
    print(f"Output CSV will be saved to: {output_csv_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = min(FULL_DELAY // 16, 1)
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = Ui("Fencing Analysis", width=int(width), height=int(height))
    video_writer = None
    if output_video_path:
        print(f"Output video will be saved to: {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height + ui.text_box_height))
        print(width, height + ui.text_box_height)
        if not video_writer.isOpened():
            print(f"Failed to open video writer for {output_video_path}. Check the path and codec.")
            return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    ui.set_fresh_frame(frame)

    left_score_positions = ui.get_n_points(generate_select_quadrilateral_instructions("left fencer score display"))
    right_score_positions = ui.get_n_points(generate_select_quadrilateral_instructions("right fencer score display"))

    # Initialise OCR
    device = torch.device("mps") # TODO: generalise for non-Mac systems
    ocr_reader = easyocr.Reader(['ch_sim'], gpu=device)

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
            ui.set_fresh_frame(frame)
            ui.refresh_frame()

            left_rect = convert_from_box_to_rect(left_score_positions)
            right_rect = convert_from_box_to_rect(right_score_positions)

            l_x, l_y, l_w, l_h = left_rect
            r_x, r_y, r_w, r_h = right_rect

            l_frame = frame[l_y:l_y+l_h, l_x:l_x+l_w]
            r_frame = frame[r_y:r_y+r_h, r_x:r_x+r_w]

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

def extract_score_from_frame(frame, threshold_boundary, ocr_reader, window_name=None):
    processed = process_image(frame, threshold_boundary)
    if window_name:
        cv2.imshow(window_name, processed)

    results = ocr_reader.recognize(processed, allowlist="0123456789")
    score, conf = "", 0

    if results:
        for bbox, text, prob in sorted(results, key=lambda x: x[2], reverse=True):
            if bbox is None or len(bbox) != 4 or not text:
                continue
            score = text
            conf = prob
            break
    return score, conf
 

if __name__ == "__main__":
    main()