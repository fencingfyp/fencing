#!/usr/bin/env python3
import argparse
import os
import csv
import cv2
from pathlib import Path
from ultralytics import YOLO

def load_model(local_path: str, fallback: str = "yolov8n-pose.pt") -> YOLO:
    """Try to load a local YOLO model, else fallback to a default model."""
    try:
        model = YOLO(local_path)
        print(f"Loaded local model: {local_path}")
    except Exception as e:
        print(f"Failed to load local model: {local_path}. Error: {e}")
        print(f"Downloading fallback model: {fallback}")
        model = YOLO(fallback)  # will download automatically if not found
    return model

def main():
    """WARNING: this needs a pose model"""
    parser = argparse.ArgumentParser(description="Process video/image with YOLO pose+tracking to CSV")
    parser.add_argument("input", help="Path to input video or image file")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Path to local YOLO model")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (informational)")
    parser.add_argument("--output", default=".", help="Output folder for CSV")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    path_object = Path(args.input)
    filename_without_extension = path_object.stem
    csv_path = os.path.join(args.output, f"pose_results_{filename_without_extension}.csv")

    # Load YOLO model with fallback
    model = load_model(args.model)

    # Decide if input is video or image
    is_video = cv2.VideoCapture(args.input).isOpened() and args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    frame_idx = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Assume COCO keypoints (17)
        kp_len = 17
        header = ["frame_id", "class", "confidence", "x1", "y1", "x2", "y2"]
        x_header = [f"kp{i}_x" for i in range(kp_len)]
        y_header = [f"kp{i}_y" for i in range(kp_len)]
        vis_header = [f"kp{i}_visible" for i in range(kp_len)]
        for x, y, v in zip(x_header, y_header, vis_header):
            header.extend([x, y, v])
        writer.writerow(header)

        if is_video:
            cap = cv2.VideoCapture(args.input)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(frame, persist=True, verbose=False)
                rows = extract_rows(results, frame_idx)
                writer.writerows(rows)

                frame_idx += 1
            cap.release()
        else:
            # Single image case
            frame = cv2.imread(args.input)
            results = model.predict(frame, verbose=False)
            rows = extract_rows(results, 0)
            writer.writerows(rows)
            

    print(f"CSV saved to {csv_path}")

def extract_rows(results, frame_idx):
    output = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box, kps in zip(r.boxes, r.keypoints):
            if int(box.cls.cpu().item()) != 0:  # 0 = person
                continue

            row = [
                frame_idx,
                int(box.cls.cpu().item()),
                float(box.conf.cpu().item()),
                float(box.xyxy[0][0]),
                float(box.xyxy[0][1]),
                float(box.xyxy[0][2]),
                float(box.xyxy[0][3])
            ]
            # print(row)
            # print(kps)
            xy = kps.xyn[0].tolist()     # [[x, y], ...] normalised
            conf = kps.conf[0].tolist()  # [visibilities]
            for (x, y), v in zip(xy, conf):
                row.extend([x, y, v])
            output.append(row)
    return output


if __name__ == "__main__":
    main()
