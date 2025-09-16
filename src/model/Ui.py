import cv2
import numpy as np
import textwrap
from util import UiCodes, calculate_centrepoint, project_point_on_line, PISTE_LENGTH_M

QUIT_KEYS = {ord('q'), ord('Q'), 27}  # q or Esc to quit

ALLOWED_ACTIONS_TO_KEYBINDS = {
  UiCodes.QUIT: QUIT_KEYS,
  UiCodes.TOGGLE_SLOW: {ord(' ')},
  UiCodes.SKIP_INPUT: {ord('1')},
  UiCodes.CONFIRM_INPUT: {13}  # Enter
}

class Ui:
  def __init__(self, window_name: str, width: int = 1280, height: int = 720, text_box_height: int = 100) -> None:
    self.window_name = window_name
    self.width = width
    self.height = height
    self.text_box_height = text_box_height
    self.current_frame = np.zeros((height + text_box_height, width, 3), dtype=np.uint8)
    self.fresh_frame = np.zeros((height, width, 3), dtype=np.uint8) # just the video frame
    self.text_color = (0, 0, 0) # Black
    self.background_color = (255, 255, 255) # White
    self.left_fencer_colour = (255, 0, 0)  # Blue
    self.right_fencer_colour = (0, 0, 255)  # Red
    self.piste_centre_line_colour = (0, 255, 0)  # Green
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

  def show_candidates(self, detections) -> None:
    self.set_current_frame(self.fresh_frame)
    self.draw_text_box()
    self.draw_candidates(detections)
    self.show_frame()

  def set_fresh_frame(self, frame) -> None:
    self.fresh_frame = frame.copy()
    self.current_frame[self.text_box_height:, :, :] = self.fresh_frame

  def draw_frame(self, frame) -> np.ndarray:
    self.current_frame[self.text_box_height:, :, :] = frame
    return self.current_frame

  def draw_text_box(self) -> np.ndarray:
    self.current_frame[:self.text_box_height, :, :] = self.background_color
    return self.current_frame

  def apply_offset(self, x1, y1, x2, y2) -> None:
    y1 += self.text_box_height
    y2 += self.text_box_height
    return x1, y1, x2, y2
  
  def apply_offset_point(self, x, y) -> None:
    y += self.text_box_height
    return x, y
  
  def draw_fresh_frame(self) -> np.ndarray:
    self.current_frame[self.text_box_height:, :, :] = self.fresh_frame
    return self.current_frame

  def draw_candidates(self, detections) -> np.ndarray:
    for det in detections.values():
      x1, y1, x2, y2 = map(int, det["box"])
      x1, y1, x2, y2 = self.apply_offset(x1, y1, x2, y2)
      # draw only the centerpoint of shoulder points (6 and 7) https://docs.ultralytics.com/tasks/pose/
      left_shoulder = det["keypoints"][6]
      right_shoulder = det["keypoints"][7]
      cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), self.text_color, 2)
      cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
      cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
      cx, cy = self.apply_offset_point(cx, cy)
      cv2.circle(self.current_frame, (cx, cy), 3, self.text_color, -1)
      cv2.putText(self.current_frame, str(det["id"]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
    return self.current_frame

  def set_current_frame(self, frame) -> None:
    self.current_frame[self.text_box_height:, :, :] = frame

  def get_current_frame(self) -> np.ndarray:
    return self.current_frame[self.text_box_height:, :, :].copy()

  def setMouseCallback(self, callback) -> None:
    # wrap callback to add offset
    def wrapped_callback(event, x, y, flags, param):
      if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE]:
        y -= self.text_box_height
      callback(event, x, y, flags, param)
    cv2.setMouseCallback(self.window_name, wrapped_callback)

  def unsetMouseCallback(self) -> None:
    cv2.setMouseCallback(self.window_name, lambda *_ : None)

  def show_updated_fencer_selection_frame(self, candidates: dict[int, dict], fencer_dir: str, selected_id: int | None) -> None:
    self.set_current_frame(self.fresh_frame)
    self.draw_candidates(candidates)
    self.write_to_ui(f"Click on the {fencer_dir} Fencer if their centrepoint is "
                     f"present and press enter to confirm. If not, press '1'.\n"
                     f"Selected ID: "+ (str(selected_id) if selected_id is not None else "No Fencer Selected"))
    self.show_frame()

  def write_to_ui(self, text, font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=1.0, thickness=2, color=(0, 0, 0), line_spacing=1.2) -> None:
    """
    Draw text inside a bounding box (x1,y1,x2,y2), auto-scaling font to fit.
    """
    self.draw_text_box()
    box_w, box_h = self.width, self.text_box_height
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
          cv2.putText(self.current_frame, line, (x_pos, y_pos), font, scale, color, thickness, cv2.LINE_AA)
        return self.current_frame

      # If doesn’t fit, shrink
      scale -= 0.1

    # If still doesn’t fit, just put smallest text
    cv2.putText(self.current_frame, text, (x1, y1 + box_h//2), font, scale, color, thickness, cv2.LINE_AA)
    return self.current_frame

  def show_frame(self) -> None:
    cv2.imshow(self.window_name, self.current_frame)

  def show_analysis(self, left_fencer_position : dict | None, right_fencer_position: dict | None, piste_centre_line: tuple[tuple[int, int], tuple[int, int]]) -> None:
    self.refresh_frame()
    if left_fencer_position is not None:
      self.draw_fencer_centrepoint(left_fencer_position, is_left=True)
    if right_fencer_position is not None:
      self.draw_fencer_centrepoint(right_fencer_position, is_left=False)
    self.draw_piste_centre_line(piste_centre_line)
    self.draw_fencer_positions_on_piste(left_fencer_position, right_fencer_position, piste_centre_line)
    self.show_frame()

  def draw_piste_centre_line(self, piste_centre_line: tuple[tuple[int, int], tuple[int, int]]) -> None:
    (x1, y1), (x2, y2) = piste_centre_line
    x1, y1, x2, y2 = self.apply_offset(x1, y1, x2, y2)
    cv2.line(self.current_frame, (x1, y1), (x2, y2), self.piste_centre_line_colour, 2)

  def draw_fencer_centrepoint(self, det: dict, is_left: bool) -> None:
    color = self.left_fencer_colour if is_left else self.right_fencer_colour
    x1, y1, x2, y2 = map(int, det["box"])
    x1, y1, x2, y2 = self.apply_offset(x1, y1, x2, y2)
    cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), color, 2)
    # draw only the centerpoint of shoulder points (6 and 7) https://docs.ultralytics.com/tasks/pose/
    left_shoulder = det["keypoints"][6]
    right_shoulder = det["keypoints"][7]
    if left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1:
      cx = int((left_shoulder[0] + right_shoulder[0]) / 2)
      cy = int((left_shoulder[1] + right_shoulder[1]) / 2)
      cx, cy = self.apply_offset_point(cx, cy)
      cv2.circle(self.current_frame, (cx, cy), 3, color, -1)

  def draw_fencer_positions_on_piste(self, left_fencer_position: dict, right_fencer_position: dict, piste_centre_line: tuple[tuple[int, int], tuple[int, int]]) -> None:
    left_proj = None
    right_proj = None

    # Calculate fencer projections on line
    if left_fencer_position is not None:
      centrept = calculate_centrepoint(left_fencer_position)
      left_proj = project_point_on_line(piste_centre_line, centrept)
      left_proj_adjusted = self.apply_offset_point(left_proj[0], left_proj[1])
      cv2.circle(self.current_frame, left_proj_adjusted, 3, self.left_fencer_colour, -1)

    if right_fencer_position is not None:
      centrept = calculate_centrepoint(right_fencer_position)
      right_proj = project_point_on_line(piste_centre_line, centrept)
      right_proj_adjusted = self.apply_offset_point(right_proj[0], right_proj[1])
      cv2.circle(self.current_frame, right_proj_adjusted, 3, self.right_fencer_colour, -1)

    piste_x1, piste_y1, piste_x2, piste_y2 = self.apply_offset(piste_centre_line[0][0], piste_centre_line[0][1], piste_centre_line[1][0], piste_centre_line[1][1])
    piste_pixel_distance = int(np.hypot(piste_x2 - piste_x1, piste_y2 - piste_y1))
    self.draw_text_box()
    if left_proj is not None and right_proj is not None:
      fencer_pixel_distance = int(np.hypot(right_proj_adjusted[0]-left_proj_adjusted[0], right_proj_adjusted[1]-left_proj_adjusted[1]))
      fencer_real_distance = (fencer_pixel_distance / piste_pixel_distance) * PISTE_LENGTH_M
      self.write_to_ui(f"Distance: {fencer_real_distance} m")
    else:
      self.write_to_ui("Distance: N/A")
    
  def get_piste_positions(self) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    instructions = [
      "Click TOP LEFT corner, press Enter to confirm",
      "Click TOP RIGHT corner, press Enter to confirm",
      "Click BOTTOM LEFT corner, press Enter to confirm",
      "Click BOTTOM RIGHT corner, press Enter to confirm"
    ]
    current_idx = 0
    def mouse_callback(event, x, y, flags, param):
      nonlocal positions
      if event == cv2.EVENT_LBUTTONDOWN:
        if len(positions) <= current_idx:
          positions.append((x, y))
        else:
          positions[current_idx] = (x, y)
        self.show_updated_piste_selection_frame(positions, instructions, current_idx)
    self.setMouseCallback(mouse_callback)

    while current_idx < len(instructions):
      self.show_updated_piste_selection_frame(positions, instructions, current_idx)
      action = self.take_user_input(0, [UiCodes.QUIT, UiCodes.CONFIRM_INPUT])
      if action == UiCodes.CONFIRM_INPUT: 
        if len(positions) > current_idx:
          current_idx += 1
      elif action == UiCodes.QUIT: 
          positions = []
          break

    self.unsetMouseCallback()
    return positions
     
  def show_updated_piste_selection_frame(self, positions, instructions, current_idx) -> None:
    self.refresh_frame()
    # Draw existing points
    for i, (px, py) in enumerate(positions):
      px, py = self.apply_offset_point(px, py)
      cv2.circle(self.current_frame, (px, py), 5, (0, 0, 255), -1)
      cv2.putText(self.current_frame, str(i+1), (px+10 if i % 2 == 0 else px-30, py), # adjust text position for right side
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show instruction text
    self.write_to_ui(instructions[current_idx])
    self.show_frame()

  def refresh_frame(self) -> None:
    self.set_current_frame(self.fresh_frame)
    self.draw_text_box()

  def get_fencer_id(self, candidates: dict[int, dict], left: bool) -> int | None:
    if not candidates:
      return None
    fencer_dir = "Left" if left else "Right"
    selected_id = None
    self.show_updated_fencer_selection_frame(candidates, fencer_dir, selected_id)

    def mouse_callback(event, x, y, flags, param):
      nonlocal selected_id
      if event == cv2.EVENT_LBUTTONDOWN:
        closest_det = None
        closest_dist = float('inf')
        for candidate in candidates.values():
          left_shoulder = candidate["keypoints"][6]
          right_shoulder = candidate["keypoints"][7]
          midpt = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
          cx, cy = midpt

          dist = (cx - x) ** 2 + (cy - y) ** 2
          if dist < closest_dist:
            closest_dist = dist
            closest_det = candidate
        if closest_det:
          selected_id = closest_det["id"]
          # print(f"Selected ID: {selected_id}")
        self.show_updated_fencer_selection_frame(candidates, fencer_dir, selected_id)
    self.setMouseCallback(mouse_callback)

    while True:
      action = self.take_user_input(0, [UiCodes.QUIT, UiCodes.CONFIRM_INPUT, UiCodes.SKIP_INPUT])
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

  def take_user_input(self, delay: int, allowed_actions: list[UiCodes]) -> UiCodes | None:
    key = cv2.waitKey(delay) & 0xFF
    for action in allowed_actions:
      if key in ALLOWED_ACTIONS_TO_KEYBINDS[action]:
        return action
    return None

  def close(self) -> None:
    cv2.destroyAllWindows()