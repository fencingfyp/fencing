"""
Video Cropper - Click two points (top-left, bottom-right) to crop a video.
Usage: python crop_video.py <input_video> [output_video]
"""

import sys

import cv2

points = []
clone = None


def click_handler(event, x, y, flags, param):
    global points, clone

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))

        # Draw feedback on the preview
        frame = clone.copy()
        for p in points:
            cv2.circle(frame, p, 5, (0, 255, 0), -1)
        if len(points) == 2:
            cv2.rectangle(frame, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Select crop region", frame)


def select_crop_region(video_path):
    global clone

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot read first frame.")
        sys.exit(1)

    clone = frame.copy()
    h, w = frame.shape[:2]
    print(f"Video resolution: {w}x{h}")
    print("Click TOP-LEFT then BOTTOM-RIGHT to define crop region.")
    print("Press R to reset, Enter/Space to confirm, Q to quit.")

    cv2.namedWindow("Select crop region")
    cv2.setMouseCallback("Select crop region", click_handler)
    cv2.imshow("Select crop region", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):  # Reset
            points.clear()
            cv2.imshow("Select crop region", clone.copy())
            print("Reset — click two points again.")

        elif key in (13, 32):  # Enter or Space to confirm
            if len(points) == 2:
                break
            print("Please select both points first.")

        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("Quit.")
            sys.exit(0)

    cv2.destroyAllWindows()
    return points[0], points[1]


def crop_video(input_path, output_path, pt1, pt2):
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (x2 - x1, y2 - y1))

    print(f"Cropping region: ({x1},{y1}) → ({x2},{y2})")
    print(f"Output size: {x2-x1}x{y2-y1} | FPS: {fps} | Frames: {total}")
    print("Processing", end="", flush=True)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame[y1:y2, x1:x2])
        frame_num += 1
        if frame_num % 30 == 0:
            print(".", end="", flush=True)

    cap.release()
    out.release()
    print(f"\nDone! Saved to '{output_path}' ({frame_num} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crop_video.py <input_video> [output_video]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cropped_output.mp4"

    pt1, pt2 = select_crop_region(input_path)
    crop_video(input_path, output_path, pt1, pt2)
