import cv2
import os
import argparse

def video_to_frames(video_path, output_folder, img_format="jpg"):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # No more frames
            break

        # Save frame as image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.{img_format}")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Done! Extracted {frame_count} frames to {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Convert a video into a folder of images.")
    parser.add_argument("video_path", help="Path to the input video (e.g., input.mp4)")
    parser.add_argument("output_folder", nargs="?", help="Output folder")
    parser.add_argument("format", nargs="?", choices=["jpg", "png"], default="jpg", help="Image format (default: jpg)")

    args = parser.parse_args()
    video_to_frames(args.video_path, args.output_folder, args.format)

if __name__ == "__main__":
    main()