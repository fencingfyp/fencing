import cv2
import os
import argparse

def images_to_video(input_folder, output_video, fps=30, img_format="jpg"):
    # Get sorted list of image files (ignore non-images like .json)
    files = sorted(f for f in os.listdir(input_folder) 
                   if f.lower().endswith(img_format.lower()))

    if not files:
        print("No images found in the folder.")
        return

    # Read first image to get dimensions
    first_frame = cv2.imread(os.path.join(input_folder, files[0]))
    height, width, layers = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for file in files:
        img_path = os.path.join(input_folder, file)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Convert a folder of images into a video.")
    parser.add_argument("input_folder", help="Folder containing images")
    parser.add_argument("output_video", help="Output video file path (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Image format to use (default: jpg)")

    args = parser.parse_args()
    images_to_video(args.input_folder, args.output_video, args.fps, args.format)

if __name__ == "__main__":
    main()
