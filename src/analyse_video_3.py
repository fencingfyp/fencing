import argparse
import cv2
import numpy as np
from process_video_2 import read_csv_by_frame, HALF_DELAY, FULL_DELAY, TEXT_COLOR, draw_text_box

CSV_COLS = 58  # 7 + 17 * 3
NUM_KEYPOINTS = 17

LEFT_FENCER_ID = 0
RIGHT_FENCER_ID = 1
LEFT_FENCER_COLOUR = (255, 0, 0)  # Blue
RIGHT_FENCER_COLOUR = (0, 0, 255)  # Red

def project_point_on_line(line: tuple[tuple[int, int], tuple[int, int]], 
                          point: tuple[int, int]) -> tuple[int, int]:
    (x1, y1), (x2, y2) = line
    x, y = point

    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:  # line is just a point
        return x1, y1

    px, py = x - x1, y - y1
    t = (px * dx + py * dy) / (dx * dx + dy * dy)

    x_out = x1 + t * dx
    y_out = y1 + t * dy
    return int(x_out), int(y_out)

def calculate_centrepoint(det):
    left_shoulder = det["keypoints"][6]
    right_shoulder = det["keypoints"][7]
    cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
    cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
    return cx, cy

def draw_fencer_centrepoints(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        if det["id"] == LEFT_FENCER_ID:
            color = LEFT_FENCER_COLOUR
        else:
            color = RIGHT_FENCER_COLOUR
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

def get_piste_centre_line(frame: np.ndarray, positions: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(positions) != 4:
        raise ValueError("Need exactly 4 positions to define the piste corners.")
    
    # Take the average of the top and bottom lines to get center line
    left_x = (positions[0][0] + positions[2][0]) // 2
    left_y = (positions[0][1] + positions[2][1]) // 2
    right_x = (positions[1][0] + positions[3][0]) // 2
    right_y = (positions[1][1] + positions[3][1]) // 2
    return (left_x, left_y), (right_x, right_y)

def draw_piste_centre_line(frame: np.ndarray, line: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    if not line:
        return frame
    cv2.line(frame, *line, (0, 255, 0), 2)  # Center line
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
        draw_text_box(frame)
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

def draw_fencer_positions(frame: np.ndarray, detections: list, piste_centre_line: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    left_proj = None
    right_proj = None

    # Calculate fencer projections on line
    left_cands = [det for det in detections if det["id"] == 0]
    if len(left_cands) > 0:
        centrept = calculate_centrepoint(left_cands[0])
        left_proj = project_point_on_line(piste_centre_line, centrept)
        cv2.circle(frame, left_proj, 3, LEFT_FENCER_COLOUR, -1)
        
    right_cands = [det for det in detections if det["id"] == 1]
    if len(right_cands) > 0:
        centrept = calculate_centrepoint(right_cands[0])
        right_proj = project_point_on_line(piste_centre_line, centrept)
        cv2.circle(frame, right_proj, 3, RIGHT_FENCER_COLOUR, -1)

    frame = draw_text_box(frame)
    if left_proj and right_proj:
        cv2.putText(frame, f"Distance: {int(np.hypot(right_proj[0]-left_proj[0], right_proj[1]-left_proj[1]))} px", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
    else:
        cv2.putText(frame, "Distance: N/A", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    args = parser.parse_args() 

    cap = cv2.VideoCapture(args.input_video)
    slow = False
    piste_positions = []
    for frame_id, detections in read_csv_by_frame(args.input_csv):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == 0:
            piste_positions = get_piste_positions(frame)
            if len(piste_positions) != 4:
                print("Piste positions not fully selected, exiting.")
                break
        frame = draw_fencer_centrepoints(frame, detections)
        centre_line = get_piste_centre_line(frame, piste_positions)

        frame = draw_piste_centre_line(frame, centre_line)
        frame = draw_fencer_positions(frame, detections, centre_line)
        
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