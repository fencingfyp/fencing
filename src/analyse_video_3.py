import argparse
import cv2
import numpy as np
from process_video_2 import read_csv_by_frame, HALF_DELAY, FULL_DELAY, TEXT_COLOR

CSV_COLS = 58  # 7 + 17 * 3
NUM_KEYPOINTS = 17

LEFT_FENCER_ID = 0
RIGHT_FENCER_ID = 1

def draw_fencer_centrepoints(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        if det["id"] == LEFT_FENCER_ID:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # draw only the centerpoint of shoulder points (6 and 7) https://docs.ultralytics.com/tasks/pose/
        left_shoulder = det["keypoints"][6]
        right_shoulder = det["keypoints"][7]
        if left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1:
            cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
            cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
            cv2.circle(frame, (cx, cy), 3, color, -1)
    return frame

def draw_piste_lines(frame: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    if len(positions) != 4:
        return frame

    # Take the average of the top and bottom lines to get center line
    left_x = (positions[0][0] + positions[2][0]) // 2
    left_y = (positions[0][1] + positions[2][1]) // 2
    right_x = (positions[1][0] + positions[3][0]) // 2
    right_y = (positions[1][1] + positions[3][1]) // 2
    cv2.line(frame, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)  # Center line

    return frame

def get_piste_positions(frame: np.ndarray) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    instructions = [
        "Click TOP LEFT corner, press Enter to confirm",
        "Click TOP RIGHT corner, press Enter to confirm",
        "Click BOTTOM LEFT corner, press Enter to confirm",
        "Click BOTTOM RIGHT corner, press Enter to confirm"
    ]

    clone = frame.copy()
    current_idx = 0
    window_name = "Select Piste Corners"

    def mouse_callback(event, x, y, flags, param):
        nonlocal positions, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(positions) <= current_idx:
                positions.append((x, y))
            else:
                positions[current_idx] = (x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while current_idx < len(instructions):
        disp = clone.copy()

        # Draw existing points
        for i, (px, py) in enumerate(positions):
            cv2.circle(disp, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i+1), (px+10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show instruction text
        cv2.putText(disp, instructions[current_idx], (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter
            if len(positions) > current_idx:
                current_idx += 1
        elif key in (27, ord('q'), ord('Q')):  # ESC or q to quit early
            positions = []
            break

    cv2.destroyWindow(window_name)
    return positions


def main():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    args = parser.parse_args() 

    cap = cv2.VideoCapture(args.input_video)
    slow = False
    for frame_id, detections in read_csv_by_frame(args.input_csv):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == 0:
            piste_positions = get_piste_positions(frame)
            if len(piste_positions) != 4:
                print("Piste positions not fully selected, exiting.")
                early_exit = True
                break
        frame = draw_fencer_centrepoints(frame, detections)
        cv2.imshow("Detections", frame)

        delay: int = HALF_DELAY if slow else FULL_DELAY
        key: int = cv2.waitKey(delay) & 0xFF
        if key == ord(" "):          # toggle on space
            slow = not slow
        elif key in (ord("q"), ord("Q"), 27):  # q or Esc to quit
            break

        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()