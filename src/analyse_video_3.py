import argparse
import cv2
import numpy as np
from process_video_2 import read_csv_by_frame
from model.Ui import Ui
from util import UiCodes

CSV_COLS = 58  # 7 + 17 * 3
NUM_KEYPOINTS = 17

LEFT_FENCER_ID = 0
RIGHT_FENCER_ID = 1

DEFAULT_FPS = 50
FULL_DELAY = int(1000 / DEFAULT_FPS)  # milliseconds
HALF_DELAY = FULL_DELAY // 2  # milliseconds

def get_piste_centre_line(positions: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(positions) != 4:
        raise ValueError("Need exactly 4 positions to define the piste corners.")
    
    # Take the average of the top and bottom lines to get center line
    left_x = (positions[0][0] + positions[2][0]) // 2
    left_y = (positions[0][1] + positions[2][1]) // 2
    right_x = (positions[1][0] + positions[3][0]) // 2
    right_y = (positions[1][1] + positions[3][1]) // 2
    return (left_x, left_y), (right_x, right_y)

def main():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output_video", help="Path to output video file (optional)", default=None)
    args = parser.parse_args() 

    writer = None
    cap = cv2.VideoCapture(args.input_video)
    slow = False
    piste_positions = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.output_video:
        print(f"Output video will be saved to: {args.output_video}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output_video, fourcc, DEFAULT_FPS, (width, height))
    ui = Ui("Fencing Analysis", width=int(width), height=int(height))
    for frame_id, detections in read_csv_by_frame(args.input_csv):
        ret, frame = cap.read()
        if not ret:
            break
        ui.set_fresh_frame(frame)
        if frame_id == 0:
            piste_positions = ui.get_piste_positions()
            if len(piste_positions) != 4:
                print("Piste positions not fully selected, exiting.")
                break
        left_fencer_position = [det for det in detections if det["id"] == LEFT_FENCER_ID]
        if len(left_fencer_position) > 0:
            left_fencer_position = left_fencer_position[0]
        else:
            left_fencer_position = None
        right_fencer_position = [det for det in detections if det["id"] == RIGHT_FENCER_ID]
        if len(right_fencer_position) > 0:
            right_fencer_position = right_fencer_position[0]
        else:
            right_fencer_position = None

        ui.show_analysis(left_fencer_position, right_fencer_position, get_piste_centre_line(piste_positions))

        if writer:
            writer.write(frame)

        delay: int = HALF_DELAY if slow else FULL_DELAY
        action = ui.take_user_input(delay, [UiCodes.QUIT, UiCodes.TOGGLE_SLOW])
        if action == UiCodes.TOGGLE_SLOW:
            slow = not slow
            print(f"Slow mode {'enabled' if slow else 'disabled'}.")
        elif action == UiCodes.QUIT:  # q or Esc to quit
            break

    if writer:
        writer.release()

    cap.release()
    ui.close()

if __name__ == "__main__":
    main()