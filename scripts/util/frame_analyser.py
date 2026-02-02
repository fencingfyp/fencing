import argparse

import cv2

from src.model import OpenCvUi, UiCodes
from src.util.io import setup_input_video_io


def get_arguments():
    parser = argparse.ArgumentParser(description="Analyse video with csv data")
    parser.add_argument("input_video", help="Path to input video file")
    args = parser.parse_args()
    return args.input_video


def main():
    input_video = get_arguments()

    cap, fps, width, height, _ = setup_input_video_io(input_video)

    full_delay = int(1000 / fps)
    fast_forward = min(full_delay // 8, 1)
    print(f"Video FPS: {fps:.2f}")

    ui = OpenCvUi("Window", width=width, height=height)

    current_frame_index = 0
    early_exit = False
    slow = False
    is_paused = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ui.set_fresh_frame(frame)
        ui.write_to_ui(f"Frame: {current_frame_index}")
        ui.show_frame()

        delay = full_delay if slow else fast_forward
        action = ui.get_user_input(
            delay,
            [
                UiCodes.CUSTOM_1,
                UiCodes.CUSTOM_2,
                UiCodes.CUSTOM_3,
                UiCodes.CUSTOM_4,
                UiCodes.CUSTOM_5,
                UiCodes.CUSTOM_6,
            ],
            must_be_valid=is_paused,
        )
        match action:
            case UiCodes.TOGGLE_SLOW:
                slow = not slow
                print(f"Slow mode {'enabled' if slow else 'disabled'}.")
            case UiCodes.QUIT:
                early_exit = True
            case UiCodes.PAUSE:
                is_paused = not is_paused
            case UiCodes.CUSTOM_1:
                # set frame-1 to current frame
                current_frame_index = max(0, current_frame_index - 1)
            case UiCodes.CUSTOM_2:
                # set frame+1 to current frame
                current_frame_index = min(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame_index + 1
                )
            case UiCodes.CUSTOM_3:
                current_frame_index = max(
                    0, current_frame_index - int(fps)
                )  # skip back 1 second
            case UiCodes.CUSTOM_4:
                current_frame_index = min(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1,
                    current_frame_index + int(fps),
                )  # skip forward 1 second
            case UiCodes.CUSTOM_5:
                current_frame_index = int(
                    max(0, current_frame_index - fps * 10)
                )  # skip back 10 seconds
            case UiCodes.CUSTOM_6:
                current_frame_index += int(fps * 10)  # skip forward 10 seconds
            case _:
                if not is_paused:
                    current_frame_index += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_index))
        if early_exit:
            break

    cap.release()
    ui.close_additional_windows()


if __name__ == "__main__":
    main()
