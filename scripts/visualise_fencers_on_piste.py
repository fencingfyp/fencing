import argparse

import cv2
import numpy as np

from src.model import Quadrilateral
from src.model.FrameInfoManager import FrameInfoManager
from src.model.OpenCvUi import OpenCvUi, UiCodes
from src.util.io import setup_output_video_io
from src.util.utils import generate_select_quadrilateral_instructions

from .manual_track_fencers import get_header_row, row_mapper

LEFT_FENCER_ID = 0
RIGHT_FENCER_ID = 1


class PisteMapper:
    """Maps fencer positions from the engarde box to a top-down plane and calculates thirds."""

    PISTE_LENGTH_M = 14.0
    ENGARDE_BOX_LENGTH_M = 4.0
    ENGARDE_LEFT_M = (PISTE_LENGTH_M - ENGARDE_BOX_LENGTH_M) / 2

    def __init__(self, engarde_quad: Quadrilateral):
        self.engarde_quad = engarde_quad
        # Rectangle size for planar mapping (arbitrary)
        self.rect_w, self.rect_h = 400, 100
        self.H = self.compute_homography()

    def compute_homography(self):
        src_pts = np.array(self.engarde_quad.points, dtype=np.float32)
        dst_pts = np.array(
            [
                [0, 0],
                [self.rect_w - 1, 0],
                [self.rect_w - 1, self.rect_h - 1],
                [0, self.rect_h - 1],
            ],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def warp_point(self, pt: tuple[int, int]) -> tuple[float, float]:
        pts = np.array([[pt]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])

    def map_to_piste(self, warped_x: float) -> float:
        """Map x-coordinate in warped plane to full piste meters."""
        frac = warped_x / self.rect_w
        return self.ENGARDE_LEFT_M + frac * self.ENGARDE_BOX_LENGTH_M

    def get_third(self, piste_x_m: float) -> str:
        third_length = self.PISTE_LENGTH_M / 3
        if piste_x_m < third_length:
            return "left"
        elif piste_x_m < 2 * third_length:
            return "centre"
        else:
            return "right"


def get_valid_fencer_coords(
    fencer_position: dict, confidence_thresh: float = 0.1
) -> tuple[int, int] | None:
    """
    Validate fencer position dict and return ankle midpoint if confidence is sufficient.
    Returns None if data is missing or below confidence threshold.
    """
    if fencer_position is None:
        return None

    keypoints = fencer_position.get("keypoints", [])
    if len(keypoints) <= 16:
        return None

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if left_ankle[2] > confidence_thresh and right_ankle[2] > confidence_thresh:
        cx = int((left_ankle[0] + right_ankle[0]) / 2)
        cy = int((left_ankle[1] + right_ankle[1]) / 2)
        return cx, cy

    return None


def draw_fencer(
    ui: OpenCvUi, fencer_position: dict, color: tuple[int, int, int]
) -> tuple[int, int] | None:
    """
    Draws fencer box and ankle midpoint on the UI.
    Returns the midpoint coordinates if valid, else None.
    """
    if fencer_position is None:
        return

    # Draw bounding box
    x1, y1, x2, y2 = map(int, fencer_position["box"])
    ui.draw_quadrilateral(
        Quadrilateral([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), color=color
    )

    # Validate and get ankle midpoint
    coords = get_valid_fencer_coords(fencer_position)
    if coords:
        cx, cy = coords
        ui.plot_points(np.array([[cx, cy]]), color=color)


def main():
    parser = argparse.ArgumentParser(description="Analyse video with CSV data")
    parser.add_argument("input_video")
    parser.add_argument("input_csv")
    parser.add_argument("--output_video", default=None)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    full_delay = int(1000 / fps)
    half_delay = full_delay // 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = OpenCvUi("Fencing Analysis", width, height)
    writer = (
        setup_output_video_io(args.output_video, fps, width, height)
        if args.output_video
        else None
    )
    frame_manager = FrameInfoManager(args.input_csv, fps, get_header_row(), row_mapper)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from video {args.input_video}")
        return

    # User selects engarde box
    points = ui.get_n_points(
        frame, generate_select_quadrilateral_instructions("piste engarde box")
    )
    if len(points) != 4:
        print("Error: Exactly 4 points required for engarde box")
        return
    engarde_quad = Quadrilateral(points)
    mapper = PisteMapper(engarde_quad)

    frame_id = 0
    slow = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_left = None
    prev_right = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = frame_manager.get_frame_and_advance(frame_id)
        ui.set_fresh_frame(frame)
        ui.draw_quadrilateral(engarde_quad, color=(0, 255, 0))

        left_dict = detections.get(LEFT_FENCER_ID, prev_left)
        right_dict = detections.get(RIGHT_FENCER_ID, prev_right)

        left_coords = get_valid_fencer_coords(left_dict)
        right_coords = get_valid_fencer_coords(right_dict)

        if left_coords:
            draw_fencer(ui, left_dict, ui.left_fencer_colour)
        if right_coords:
            draw_fencer(ui, right_dict, ui.right_fencer_colour)

        # Warp fencer positions to planar view
        left_warp = mapper.warp_point(left_coords) if left_coords else None
        right_warp = mapper.warp_point(right_coords) if right_coords else None

        # Compute thirds using warped x-coordinate
        left_third = (
            mapper.get_third(mapper.map_to_piste(left_warp[0])) if left_warp else None
        )
        right_third = (
            mapper.get_third(mapper.map_to_piste(right_warp[0])) if right_warp else None
        )

        ui.write_to_ui(
            f"Left fencer in {left_third} third, Right fencer in {right_third} third"
        )

        ui.show_frame()
        if writer:
            writer.write(ui.get_current_frame())

        delay = full_delay if slow else half_delay
        action = ui.get_user_input(delay)
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print("Toggled slow mode:", slow)
        elif action == UiCodes.QUIT:
            break
        elif action == UiCodes.PAUSE:
            if ui.handle_pause():
                break  # exit if user chose to quit during pause

        frame_id += 1

    if writer:
        writer.release()
    cap.release()
    ui.close_additional_windows()


if __name__ == "__main__":
    main()
