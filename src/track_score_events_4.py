import argparse
import cv2
import numpy as np
from model.Ui import Ui
from util import UiCodes, convert_to_opencv_format, convert_from_opencv_format
from model.FrameInfoManager import FrameInfoManager
from model.PatchLightDetector import PatchLightDetector
from track_planes_3 import get_header_row


DEFAULT_FPS = 50
FULL_DELAY = int(1000 / DEFAULT_FPS)  # milliseconds
HALF_DELAY = FULL_DELAY // 16  # milliseconds

def row_mapper(row: list[str]) -> dict[str, any]:
    # Convert row to dict
    id = row[1]
    box = list(map(int, list(map(float, row[2:10]))))
    # map to tuple of 4 points
    box = [(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])]

    return {
      "id": id,
      "box": box,  # [x1, y1, x2, y2, x3, y3, x4, y4]
    }

# def is_target_colour(frame, patch_pts, target, c_thresh=0.12, v_thresh=40, alpha=1.25):
#     """
#     frame: input BGR frame (uint8).
#     patch_pts: 4 points (x,y) defining patch polygon.
#     target: 'red' | 'green' | 'white'.
#     """
#     # get bounding box of polygon
#     rect = cv2.boundingRect(np.array(patch_pts))
#     x,y,w,h = rect
#     roi = frame[y:y+h, x:x+w].copy()

#     # mask polygon within roi
#     mask = np.zeros((h,w), np.uint8)
#     pts = np.array(patch_pts) - [x,y]
#     cv2.fillPoly(mask, [pts], 255)
#     mean_bgr = cv2.mean(roi, mask=mask)[:3]
#     b,g,r = mean_bgr
#     R,G,B = float(r), float(g), float(b)
#     Y = R+G+B

#     if Y < v_thresh:
#         return False

#     # colourfulness (chroma)
#     mx, mn = max(R,G,B), min(R,G,B)
#     chroma = mx - mn
#     chroma_ratio = chroma / (Y+1e-6)

#     if target == 'white':
#         return chroma_ratio < c_thresh

#     # normalised fractions
#     r_frac, g_frac = R/(Y+1e-6), G/(Y+1e-6)

#     if target == 'red':
#         return r_frac > g_frac * alpha
#     if target == 'green':
#         return g_frac > r_frac * alpha

#     return False


def main():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args() 

    csv_path = args.input_csv
    input_video_path = args.input_video
    output_video_path = args.output_video

    writer = None
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    FULL_DELAY = int(1000 / fps)
    FAST_FORWARD = FULL_DELAY // 16
    print(f"Video FPS: {fps}, Frame delay: {FULL_DELAY} ms")

    # UI
    slow = False
    early_exit = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = Ui("Fencing Analysis", width=int(width), height=int(height))
    if output_video_path:
        print(f"Output video will be saved to: {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height + ui.text_box_height))
        print(width, height + ui.text_box_height)
        if not writer.isOpened():
            print(f"Failed to open video writer for {output_video_path}. Check the path and codec.")
            return
    frame_info_manager = FrameInfoManager(csv_path, fps, get_header_row(), row_mapper)

    left_colour_detector = PatchLightDetector('red')
    right_colour_detector = PatchLightDetector('green')
    left_white_detector = PatchLightDetector('white')
    right_white_detector = PatchLightDetector('white')

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = frame_info_manager.get_frame_info_at(frame_id)
        ui.set_fresh_frame(frame)
        ui.refresh_frame()

        # draw piste
        piste_positions_tracked = detections.get("piste")
        left_fencer_score_light_tracked = detections.get("left_fencer_score_light")
        right_fencer_score_light_tracked = detections.get("right_fencer_score_light")
        left_fencer_white_light_tracked = detections.get("left_fencer_white_light")
        right_fencer_white_light_tracked = detections.get("right_fencer_white_light")

        # ui.draw_polygon(convert_to_opencv_format(left_fencer_score_light_tracked["box"]), color=(255, 0, 0))

        # ui.draw_polygon(convert_to_opencv_format(piste_positions_tracked["box"]), color=(0, 255, 0))
        # piste_centre_line = get_piste_centre_line(piste_positions_tracked["box"])
        # ui.draw_piste_centre_line(piste_centre_line)

        is_left_red = left_colour_detector.update(frame, left_fencer_score_light_tracked["box"])
        is_right_green = right_colour_detector.update(frame, right_fencer_score_light_tracked["box"])
        is_left_white = left_white_detector.update(frame, left_fencer_white_light_tracked["box"])
        is_right_white = right_white_detector.update(frame, right_fencer_white_light_tracked["box"])
        ui.write_to_ui(f"Left fencer score light is {'ON' if is_left_red else 'OFF'}.\n"
                      f"Right fencer score light is {'ON' if is_right_green else 'OFF'}.\n"
                      f"Left fencer white light is {'ON' if is_left_white else 'OFF'}.\n"
                      f"Right fencer white light is {'ON' if is_right_white else 'OFF'}.")

        ui.show_frame()

        if writer:
            writer.write(ui.current_frame)

        delay: int = FULL_DELAY if slow else FAST_FORWARD
        action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW])
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:  # q or Esc to quit
            break

        if early_exit:
            break
        frame_id += 1

    if writer:
        writer.release()

    cap.release()
    ui.close()

if __name__ == "__main__":
    main()