# I/O stuff
import os

import cv2


def setup_input_video_io(video_path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit(1)

    return (
        cap,
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def setup_output_video_io(output_path, fps, frame_size) -> cv2.VideoWriter | None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print(f"Error: Could not open video writer {output_path}")
        return exit(1)
    print(f"Output video will be saved to: {output_path}")
    return writer


def setup_output_file(folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    print(f"Output file will be saved to: {file_path}")
    return file_path
