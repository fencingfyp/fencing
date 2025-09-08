import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 pose estimation or object tracking on a video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out_folder", type=str, default=None, help="Folder to save annotated video (default: current folder)")
    parser.add_argument("--out_name", type=str, default=None, help="Custom output filename (optional)")
    parser.add_argument("--mode", type=str, choices=["pose", "track"], default="pose", help="Mode: 'pose' or 'track'")
    parser.add_argument("--model", type=str, default=None, help="YOLOv8 model file to use (defaults: pose->yolov8n-pose.pt, track->yolov8n.pt)")
    args = parser.parse_args()

    # Set output folder
    out_folder = args.out_folder or os.getcwd()
    os.makedirs(out_folder, exist_ok=True)

    # Set default model
    if args.model is None:
        args.model = "yolov8n-pose.pt" if args.mode == "pose" else "yolov8n.pt"

    # Determine output filename
    if args.out_name:
        out_file = os.path.join(out_folder, args.out_name)
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        out_file = os.path.join(out_folder, f"{base}_{args.mode}_{model_name}.mp4")

    # Load model
    model = YOLO(args.model)

    # Run inference
    if args.mode == "pose":
        model.predict(
            source=args.video,
            save=True,
            project=out_folder,
            exist_ok=True
        )
    else:  # track
        model.track(
            source=args.video,
            show=False,
            save=True,
            project=out_folder,
            exist_ok=True
        )

    print(f"✅ Annotated video saved to {out_file}")

if __name__ == "__main__":
    main()

