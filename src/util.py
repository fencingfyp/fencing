import enum
import numpy as np

class UiCodes(enum.Enum):
    QUIT = 0
    TOGGLE_SLOW = 1
    SKIP_INPUT = 2
    CONFIRM_INPUT = 3

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