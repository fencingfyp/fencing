import cv2
import numpy as np
import pandas as pd

from scripts.manual_track_fencers import (
    get_header_row as get_header_row_for_fencer_poses_csv,
)
from scripts.manual_track_fencers import row_mapper as fencer_poses_row_mapper
from src.gui.heat_map.generate_heat_map_widget import (
    LEFT_FENCER_ID,
    RIGHT_FENCER_ID,
    FencerClassifier,
    PisteMapper,
    get_valid_fencer_coords,
    load_piste_data,
)
from src.model.FrameInfoManager import FrameInfoManager

# -----------------------------
# DEBUG DRAW SCRIPT
# -----------------------------


def debug_visualise_fencers(
    video_path: str,
    frame_manager,
    engarde_quad,
    momentum_data_path,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    mapper = PisteMapper(engarde_quad)
    classifier = FencerClassifier()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df = pd.read_csv(momentum_data_path)
    score_times = set(df["frame_id"].tolist())

    for frame_id in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        detections = frame_manager.get_frame_and_advance(frame_id)
        if frame_id < 21000:
            continue

        # ---- Draw Piste Quad ----
        quad_pts = np.array(engarde_quad.points, dtype=np.int32)
        cv2.polylines(frame, [quad_pts], True, (0, 255, 255), 2)

        # ---- Draw Third Boundaries (in warped space projected back) ----
        third_w = mapper.rect_w / 3

        # for i in [1, 2]:
        #     x_warp = third_w * i
        #     pts = np.array([[[x_warp, 0]]], dtype=np.float32)
        #     invH = np.linalg.inv(mapper.H)
        #     original = cv2.perspectiveTransform(pts, invH)
        #     x_orig = int(original[0, 0, 0])

        #     cv2.line(frame, (x_orig, 0), (x_orig, frame.shape[0]), (255, 0, 255), 1)

        # ---- Process Fencers ----
        left_coords = None
        right_coords = None

        if detections:
            left_coords = get_valid_fencer_coords(detections.get(LEFT_FENCER_ID))
            right_coords = get_valid_fencer_coords(detections.get(RIGHT_FENCER_ID))

        left_third, right_third = classifier.classify(
            mapper,
            left_coords,
            right_coords,
        )

        # ---- Draw LEFT ----
        if left_coords:
            cv2.circle(frame, left_coords, 6, (255, 0, 0), -1)
            cv2.putText(
                frame,
                f"L: {left_third}",
                (left_coords[0] + 10, left_coords[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        # ---- Draw RIGHT ----
        if right_coords:
            cv2.circle(frame, right_coords, 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"R: {right_third}",
                (right_coords[0] + 10, right_coords[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # ---- Frame Label ----
        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Fencer Third Debug", frame)
        if frame_id in score_times:
            cv2.waitKey(0)  # Pause on score frames

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = "matches_data/epee_3.mp4"
    engarde_quad = load_piste_data("matches_data/epee_3.data/raw_piste_quads.csv")

    frame_manager = FrameInfoManager(
        "matches_data/epee_3.data/processed_pose_data.csv",
        25,
        get_header_row_for_fencer_poses_csv(),
        fencer_poses_row_mapper,
    )

    debug_visualise_fencers(
        video_path,
        frame_manager,
        engarde_quad,
        "matches_data/epee_3.data/momentum_data.csv",
    )


if __name__ == "__main__":
    main()
