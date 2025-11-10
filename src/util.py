import enum
import os
import numpy as np
import cv2

class UiCodes(enum.Enum):
    QUIT = 0
    TOGGLE_SLOW = 1
    SKIP_INPUT = 2
    CONFIRM_INPUT = 3
    PAUSE = 4
    PICK_LEFT_FENCER = 5
    PICK_RIGHT_FENCER = 6
    CUSTOM_1 = 7
    CUSTOM_2 = 8
    CUSTOM_3 = 9
    CUSTOM_4 = 10

NORMAL_UI_FUNCTIONS = [UiCodes.QUIT, UiCodes.TOGGLE_SLOW, UiCodes.PAUSE]

PISTE_LENGTH_M = 14  # Standard piste length in meters

def generate_select_quadrilateral_instructions(target_name: str, confirm_key = 'w') -> list[str]:
    return [
        f"Select {target_name} top left corner, press '{confirm_key}' to confirm",
        f"Select {target_name} top right corner, press '{confirm_key}' to confirm",
        f"Select {target_name} bottom right corner, press '{confirm_key}' to confirm",
        f"Select {target_name} bottom left corner, press '{confirm_key}' to confirm",
    ]

PISTE_INSTRUCTIONS = generate_select_quadrilateral_instructions("piste")
LEFT_FENCER_WHITE_LIGHT_INSTRUCTIONS = generate_select_quadrilateral_instructions("left fencer's white light")
LEFT_FENCER_SCORE_LIGHT_INSTRUCTIONS = generate_select_quadrilateral_instructions("left fencer's score light")
RIGHT_FENCER_WHITE_LIGHT_INSTRUCTIONS = generate_select_quadrilateral_instructions("right fencer's white light")
RIGHT_FENCER_SCORE_LIGHT_INSTRUCTIONS = generate_select_quadrilateral_instructions("right fencer's score light")
LEFT_FENCER_SCORE_INSTRUCTIONS = generate_select_quadrilateral_instructions("left fencer's score display")
RIGHT_FENCER_SCORE_INSTRUCTIONS = generate_select_quadrilateral_instructions("right fencer's score display")


def convert_to_opencv_format(list_of_points: list[tuple[int, int]]) -> np.ndarray:
    return np.array(list_of_points, dtype=np.float32).reshape((-1, 1, 2))

def convert_from_opencv_format(pts: np.ndarray) -> list[tuple[int, int]]:
    return [(int(pt[0][0]), int(pt[0][1])) for pt in pts]

def calculate_centrepoint(det):
    left_shoulder = det["keypoints"][6]
    right_shoulder = det["keypoints"][7]
    cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
    cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
    return cx, cy

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

# I/O stuff
def setup_input_video_io(video_path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit(1)

    return cap, cap.get(cv2.CAP_PROP_FPS), \
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def setup_output_video_io(output_path, fps, frame_size) -> cv2.VideoWriter | None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
