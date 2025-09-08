import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 object tracking on a video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to save annotated video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to use (n/s/m/l/x)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load YOLOv8 object detection model
    model = YOLO(args.model)

    # Run tracking inference and save annotated video
    results = model.track(
        source=args.video,
        show=False,
        save=True,
        project="tmp_track",
        name="annotated",
        exist_ok=True
    )

    # The annotated video is saved in tmp_track/annotated/<input_filename>
    input_filename = os.path.basename(args.video)
    annotated_path = os.path.join("tmp_track", "annotated", input_filename)
    
    # Move to desired output
    os.rename(annotated_path, args.output)
    print(f"✅ Annotated video saved to {args.output}")

if __name__ == "__main__":
    main()

