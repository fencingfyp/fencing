import enum

class UiCodes(enum.Enum):
    QUIT = 0
    TOGGLE_SLOW = 1
    SKIP_INPUT = 2
    CONFIRM_INPUT = 3
    
    

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