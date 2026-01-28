#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.util.io import setup_output_video_io

CSV_COLS = 58  # 7 + 17 * 3
NUM_KEYPOINTS = 17

"""
Input: A video (check accepted formats below) or an image, a yolo pose model
Output: A csv marking the poses
"""


def main():
    parser = argparse.ArgumentParser(
        description="Process video/image with YOLO pose+tracking to CSV"
    )
    parser.add_argument("folder", help="Path to working folder")
    parser.add_argument(
        "--model", default="models/yolo11l-pose.pt", help="Path to local YOLO model"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show processed video/image"
    )
    args = parser.parse_args()
    output_folder = args.folder
    model_path = args.model

    csv_path = os.path.join(output_folder, "raw_pose_results.csv")
    model = YOLO(model_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = get_header_row()
        writer.writerow(header_row)

        process_video(args.input, model, writer, args.show, args.output_folder)

    print(f"CSV saved to {csv_path}")


def process_video(
    input_path: str, model: YOLO, writer: csv.writer, show: bool, output_folder: str
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    if show:
        cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
        video_writer = setup_output_video_io(
            os.path.join(output_folder, "output_video.mp4"),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)
        if show:
            annotated_frame = results[0].plot()
            cv2.imshow("Processed Video", annotated_frame)
            video_writer.write(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        rows = extract_rows(results, frame_idx)
        writer.writerows(rows)
        frame_idx += 1
    cap.release()
    if show:
        cv2.destroyAllWindows()
        video_writer.release()


def get_header_row() -> list[str]:
    kp_len = 17
    header = ["frame_id", "id", "confidence", "x1", "y1", "x2", "y2"]
    x_header = [f"kp{i}_x" for i in range(kp_len)]
    y_header = [f"kp{i}_y" for i in range(kp_len)]
    vis_header = [f"kp{i}_visible" for i in range(kp_len)]
    for x, y, v in zip(x_header, y_header, vis_header):
        header.extend([x, y, v])
    return header


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
                int(box.id.cpu().item()) if box.id is not None else -1,
                float(box.conf.cpu().item()),
                float(box.xyxy[0][0]),
                float(box.xyxy[0][1]),
                float(box.xyxy[0][2]),
                float(box.xyxy[0][3]),
            ]
            xy = kps.xy[0].tolist()  # [[x, y], ...]
            conf = kps.conf[0].tolist()  # [visibilities]
            for (x, y), v in zip(xy, conf):
                row.extend([x, y, v])
            output.append(row)
    return output


if __name__ == "__main__":
    main()
