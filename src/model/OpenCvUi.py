import enum
import textwrap

import cv2
import numpy as np

from src.util.utils import (
    PISTE_INSTRUCTIONS,
    PISTE_LENGTH_M,
    generate_select_quadrilateral_instructions,
    project_point_on_line,
)

from .Quadrilateral import Quadrilateral
from .Ui import Ui


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
    CUSTOM_5 = 11
    CUSTOM_6 = 12


NORMAL_UI_FUNCTIONS = [UiCodes.QUIT, UiCodes.TOGGLE_SLOW, UiCodes.PAUSE]

QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q or Esc to quit

ALLOWED_ACTIONS_TO_KEYBINDS = {
    UiCodes.QUIT: QUIT_KEYS,
    UiCodes.TOGGLE_SLOW: {ord(" ")},
    UiCodes.SKIP_INPUT: {ord("1")},
    UiCodes.CONFIRM_INPUT: {ord("w")},
    UiCodes.PAUSE: {ord("p"), ord("P")},
    UiCodes.PICK_LEFT_FENCER: {ord("n"), ord("N")},
    UiCodes.PICK_RIGHT_FENCER: {ord("m"), ord("M")},
    UiCodes.CUSTOM_1: {ord("1")},
    UiCodes.CUSTOM_2: {ord("2")},
    UiCodes.CUSTOM_3: {ord("3")},
    UiCodes.CUSTOM_4: {ord("4")},
    UiCodes.CUSTOM_5: {ord("5")},
    UiCodes.CUSTOM_6: {ord("6")},
}


def calculate_centrepoint(det):
    left_shoulder = det["keypoints"][6]
    right_shoulder = det["keypoints"][7]
    cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
    cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
    return cx, cy


class OpenCvUi(Ui):
    @staticmethod
    def calculate_display_dimensions(
        width: int, height: int, display_width: int = None, display_height: int = None
    ) -> tuple[int, int]:
        if display_width is not None and display_height is not None:
            raise ValueError(
                "Specify either display_width and display_height or neither."
            )
        aspect_ratio = width / height
        if display_width is None and display_height is None:
            display_width = width
            display_height = height
        elif display_width is not None:
            display_height = int(display_width / aspect_ratio)
        elif display_height is not None:
            display_width = int(display_height * aspect_ratio)
        return display_width, display_height

    def __init__(
        self,
        window_name: str,
        width: int = 1280,
        height: int = 720,
        display_width: int = None,
        display_height: int = None,
        text_box_height: int = 100,
    ) -> None:
        self.window_name = window_name
        self.width = width
        self.height = height
        self.display_width, self.display_height = self.calculate_display_dimensions(
            width, height, display_width, display_height
        )
        self.text_box_height = text_box_height
        self.current_frame = np.zeros(
            (self.display_height + text_box_height, self.display_width, 3),
            dtype=np.uint8,
        )
        self.fresh_frame = np.zeros(
            (self.display_height, self.display_width, 3), dtype=np.uint8
        )  # just the video frame
        self.text_color = (0, 0, 0)  # Black
        self.background_color = (255, 255, 255)  # White
        self.left_fencer_colour = (255, 0, 0)  # Blue
        self.right_fencer_colour = (0, 0, 255)  # Red
        self.piste_centre_line_colour = (0, 255, 0)  # Green
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def plot_points(self, points: np.ndarray, color: tuple[int, int, int]) -> None:
        for pt in points:
            x, y = map(int, pt.ravel())
            x, y = self.apply_offset_point(x, y)
            cv2.circle(self.current_frame, (x, y), 5, color, -1)

    def get_output_dimensions(self) -> tuple[int, int]:
        return (self.display_width, self.display_height + self.text_box_height)

    def show_candidates(self, detections) -> None:
        self._set_current_frame(self.fresh_frame)
        self.draw_text_box()
        self.draw_candidates(detections)
        self.show_frame()

    def set_fresh_frame(self, frame) -> None:
        self._set_current_frame(frame)
        self.fresh_frame = self.get_current_frame()

    def draw_text_box(self) -> np.ndarray:
        self.current_frame[: self.text_box_height, :, :] = self.background_color
        return self.current_frame

    def apply_offset(self, x1, y1, x2, y2) -> None:
        return *self.apply_offset_point(x1, y1), *self.apply_offset_point(x2, y2)

    def apply_offset_point(self, x, y) -> None:
        y += self.text_box_height
        return x, y

    def draw_candidates(self, detections) -> np.ndarray:
        for det in detections.values():
            x1, y1, x2, y2 = map(int, det["box"])
            x1, y1, x2, y2 = self.apply_offset(x1, y1, x2, y2)
            cv2.rectangle(
                self.current_frame, (x1, y1), (x2, y2), self.text_color, 2
            )  # draw bounding box
            cx, cy = self.apply_offset_point(*calculate_centrepoint(det))
            cv2.circle(self.current_frame, (cx, cy), 3, self.text_color, -1)
            cv2.putText(
                self.current_frame,
                str(det["id"]),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                self.text_color,
                2,
            )
        return self.current_frame

    def _set_current_frame(self, frame) -> None:
        # Check aspect ratio and pad if needed
        frame_h, frame_w = frame.shape[:2]
        target_aspect = self.display_width / self.display_height
        frame_aspect = frame_w / frame_h
        if frame_aspect > target_aspect:
            # frame is wider than target, pad height
            new_height = int(frame_w / target_aspect)
            pad_vert = (new_height - frame_h) // 2
            padded = cv2.copyMakeBorder(
                frame, pad_vert, pad_vert, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            target = cv2.resize(
                padded,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_CUBIC,
            )
        elif frame_aspect < target_aspect:
            # frame is taller than target, pad width
            new_width = int(frame_h * target_aspect)
            pad_horiz = (new_width - frame_w) // 2
            padded = cv2.copyMakeBorder(
                frame, 0, 0, pad_horiz, pad_horiz, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            target = cv2.resize(
                padded,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            # same aspect ratio
            target = cv2.resize(
                frame,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_CUBIC,
            )
        if len(target.shape) == 2 or target.shape[2] == 1:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

        # Write back into current_frame
        self.current_frame[self.text_box_height :, :, :] = target

    def get_current_frame(self) -> np.ndarray:
        return self.current_frame[self.text_box_height :, :, :].copy()

    def setMouseCallback(self, callback) -> None:
        # wrap callback to add offset
        def wrapped_callback(event, x, y, flags, param):
            if event in [
                cv2.EVENT_LBUTTONDOWN,
                cv2.EVENT_LBUTTONUP,
                cv2.EVENT_MOUSEMOVE,
            ]:
                y -= self.text_box_height
            callback(event, x, y, flags, param)

        cv2.setMouseCallback(self.window_name, wrapped_callback)

    def unsetMouseCallback(self) -> None:
        cv2.setMouseCallback(self.window_name, lambda *_: None)

    def show_updated_fencer_selection_frame(
        self, candidates: dict[int, dict], fencer_dir: str, selected_id: int | None
    ) -> None:
        self._set_current_frame(self.fresh_frame)
        self.draw_candidates(candidates)
        self.write_to_ui(
            f"Click on the {fencer_dir} Fencer if their centrepoint is "
            f"present and press 'w' to confirm. If not, press '1'.\n"
            f"Selected ID: "
            + (str(selected_id) if selected_id is not None else "No Fencer Selected")
        )
        self.show_frame()

    def write_to_ui(
        self,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.0,
        thickness=2,
        color=(0, 0, 0),
        line_spacing=1.2,
    ) -> None:
        """
        Draw text inside a bounding box (x1,y1,x2,y2), auto-scaling font to fit.
        """
        self.draw_text_box()
        box_w, box_h = self.display_width, self.text_box_height
        x1, y1 = 0, 0

        scale = font_scale
        while scale > 0.1:
            # Estimate max chars per line (rough, since proportional font)
            char_w, char_h = cv2.getTextSize("M", font, scale, thickness)[0]
            if char_w == 0:
                break
            max_chars = max(1, box_w // char_w)

            # Wrap text into lines
            lines = textwrap.wrap(text, width=max_chars)

            # Measure total height
            line_h = char_h + thickness
            total_h = int(len(lines) * line_h * line_spacing)

            # Check fit
            if total_h <= box_h:
                # Draw centered
                y_start = y1 + (box_h - total_h) // 2 + char_h
                for i, line in enumerate(lines):
                    (text_w, _), _ = cv2.getTextSize(line, font, scale, thickness)
                    x_pos = x1 + (box_w - text_w) // 2
                    y_pos = int(y_start + i * line_h * line_spacing)
                    cv2.putText(
                        self.current_frame,
                        line,
                        (x_pos, y_pos),
                        font,
                        scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                return self.current_frame

            # If doesn’t fit, shrink
            scale -= 0.1

        # If still doesn’t fit, just put smallest text
        cv2.putText(
            self.current_frame,
            text,
            (x1, y1 + box_h // 2),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return self.current_frame

    def show_frame(self) -> None:
        cv2.imshow(self.window_name, self.current_frame)

    def draw_polygon(
        self,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        # input format: np.ndarray of shape (n, 1, 2)
        if len(points) < 2:
            return
        # TODO: optimize by avoiding creating new list
        pts = [self.apply_offset_point(int(pt[0][0]), int(pt[0][1])) for pt in points]
        cv2.polylines(
            self.current_frame,
            [np.array(pts)],
            isClosed=True,
            color=color,
            thickness=thickness,
        )

    def draw_quadrilateral(
        self,
        quad: Quadrilateral,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        # quick hack: scale to current frame size from original frame size
        quad = quad.opencv_format()
        # using self.width and self.height as original frame size
        scale_x = self.display_width / self.width
        scale_y = self.display_height / self.height
        for i in range(4):
            quad[i][0][0] = int(quad[i][0][0] * scale_x)
            quad[i][0][1] = int(quad[i][0][1] * scale_y)
        self.draw_polygon(quad, color=color, thickness=thickness)

    def draw_line(
        self, positions: list[tuple[int, int]], color: tuple[int, int, int] = None
    ) -> None:
        if len(positions) != 2:
            raise ValueError("Lines require exactly 2 points.")
        if color is None:
            color = self.piste_centre_line_colour
        x1, y1, x2, y2 = self.apply_offset(
            positions[0][0], positions[0][1], positions[1][0], positions[1][1]
        )
        cv2.line(self.current_frame, (x1, y1), (x2, y2), color, 2)

    def get_quadrilateral(self, frame: np.ndarray, item_name: str) -> Quadrilateral:
        return Quadrilateral(
            self.get_n_points(
                frame,
                generate_select_quadrilateral_instructions(item_name),
            )
        )

    def get_n_points_async(self, frame, instructions, callback) -> None:
        callback(self.get_n_points(frame, instructions))

    def get_n_points(
        self, frame: np.ndarray, instructions: list[str]
    ) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        current_idx = 0

        # assume these are set when displaying the frame
        self.height, self.width = frame.shape[:2]

        def mouse_callback(event, x, y, flags, param):
            nonlocal positions
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(positions) <= current_idx:
                    positions.append((x, y))
                else:
                    positions[current_idx] = (x, y)

                self.show_updated_piste_selection_frame(
                    positions, instructions, current_idx
                )

        self.setMouseCallback(mouse_callback)

        self.set_fresh_frame(frame)
        while current_idx < len(instructions):
            self.show_updated_piste_selection_frame(
                positions, instructions, current_idx
            )

            action = self.get_user_input(0, [UiCodes.QUIT, UiCodes.CONFIRM_INPUT])

            if action == UiCodes.CONFIRM_INPUT:
                if len(positions) > current_idx:
                    current_idx += 1
            elif action == UiCodes.QUIT:
                self.unsetMouseCallback()
                return []

        self.unsetMouseCallback()

        # scale back to original frame size
        scaled_positions: list[tuple[int, int]] = []
        for px, py in positions:
            orig_x = int(px * self.width / self.display_width)
            orig_y = int(py * self.height / self.display_height)
            scaled_positions.append((orig_x, orig_y))

        return scaled_positions

    def get_piste_positions(self, frame: np.ndarray) -> list[tuple[int, int]]:
        return self.get_n_points(frame, PISTE_INSTRUCTIONS)

    def show_updated_piste_selection_frame(
        self, positions, instructions, current_idx
    ) -> None:
        self.refresh_frame()
        # Draw existing points
        for i, (px, py) in enumerate(positions):
            px, py = self.apply_offset_point(px, py)
            cv2.circle(self.current_frame, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                self.current_frame,
                str(i + 1),
                (
                    px + 10 if i % 2 == 0 else px - 30,
                    py,
                ),  # adjust text position for right side
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Show instruction text
        self.write_to_ui(instructions[current_idx])
        self.show_frame()

    def refresh_frame(self) -> None:
        self._set_current_frame(self.fresh_frame)
        self.draw_text_box()

    def clear_frame(self) -> None:
        self.current_frame = np.zeros(
            (self.display_height + self.text_box_height, self.display_width, 3),
            dtype=np.uint8,
        )

    def get_fencer_id(self, candidates: dict[int, dict], left: bool) -> int | None:
        if not candidates:
            return None
        fencer_dir = "Left" if left else "Right"
        selected_id = None
        self.show_updated_fencer_selection_frame(candidates, fencer_dir, selected_id)

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_id
            if event == cv2.EVENT_LBUTTONDOWN:
                closest_det = min(
                    candidates.values(),
                    key=lambda c: (calculate_centrepoint(c)[0] - x) ** 2
                    + (calculate_centrepoint(c)[1] - y) ** 2,
                    default=None,
                )
                if closest_det:
                    selected_id = closest_det["id"]

                self.show_updated_fencer_selection_frame(
                    candidates, fencer_dir, selected_id
                )

        self.setMouseCallback(mouse_callback)

        while True:
            action = self.get_user_input(
                0, [UiCodes.QUIT, UiCodes.CONFIRM_INPUT, UiCodes.SKIP_INPUT]
            )
            if action == UiCodes.CONFIRM_INPUT and selected_id is not None:
                break
            elif action == UiCodes.SKIP_INPUT:
                selected_id = None
                break
            elif action == UiCodes.QUIT:
                selected_id = -1
                break

        self.unsetMouseCallback()
        return selected_id

    def get_user_input(
        self,
        delay: int,
        allowed_actions: list[UiCodes] = [],
        must_be_valid=False,
    ) -> UiCodes | None:
        """Takes user input for a given delay in milliseconds.
        If allowed_actions is provided, only those actions and the normal UI functions are considered valid.
        If must_be_valid is True, the function will keep waiting until a valid action is received
        """
        allowed_actions_unique = set(allowed_actions)
        allowed_actions_unique.update(NORMAL_UI_FUNCTIONS)
        while True:
            key = cv2.waitKey(delay) & 0xFF
            for action in allowed_actions_unique:
                if key in ALLOWED_ACTIONS_TO_KEYBINDS[action]:
                    return action
            if not must_be_valid:
                return None

    def close_additional_windows(self) -> None:
        cv2.destroyWindow(self.window_name)

    def get_confirmation(self, prompt: str) -> bool:
        self.refresh_frame()
        self.write_to_ui(prompt)
        self.show_frame()
        while True:
            action = self.get_user_input(
                0, [UiCodes.CONFIRM_INPUT, UiCodes.SKIP_INPUT, UiCodes.QUIT]
            )
            if action in [UiCodes.CONFIRM_INPUT, UiCodes.SKIP_INPUT, UiCodes.QUIT]:
                return action == UiCodes.CONFIRM_INPUT

    def handle_pause(self):
        print("Paused. Press 'p' to resume or 'q' to quit.")
        while True:
            action = self.get_user_input(100, [UiCodes.PAUSE, UiCodes.QUIT])
            if action == UiCodes.PAUSE:
                return False
            elif action == UiCodes.QUIT:
                return True

    def process_crop_region_loop(
        self,
        cap: cv2.VideoCapture,
        frame_callback,
        writer: cv2.VideoWriter | None,
    ):
        """
        UI-owned main loop.
        frame_callback(frame) -> (rectified, pts)
        """

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Delegate all processing
            rectified, pts = frame_callback(frame)

            # UI rendering
            self.set_fresh_frame(frame)
            self.plot_points(pts, (0, 255, 0))
            self.show_frame()

            cv2.imshow("cropped_view", rectified)
            if writer:
                writer.write(rectified)

            # Input handling (this is the real event boundary)
            action = self.get_user_input(1)
            if action == UiCodes.QUIT:
                break

    def run_loop(self, step_fn):
        """
        UI-owned main loop.
        step_fn() is called every frame.
        """

        while True:
            if step_fn():
                break

    def initialise(self, fps):
        self.fps = fps

    def show_additional(self, key, frame):
        cv2.imshow(key, frame)

    def take_user_input(self):
        pass
