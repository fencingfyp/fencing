# YOLO Pose + Tracking Video Processor

This project provides utilities to process videos or images with a YOLO pose and tracking model, export detections to CSV, and visualise them back on the original media.

## Features

- Run YOLO pose+track inference on:
  - Videos (frame by frame)
  - Single images
- Save results in a structured **CSV** format:
- Each row = one detected person in one frame  
- 17 keypoints per detection, each with `(x, y, visibility)`  
- Stream results directly to CSV per frame (avoids OOM on long videos)
- Replay results by loading CSV + original video
- Adjustable playback FPS for visualisation
- Interactive piste marking:
- On the first frame, user clicks **Top Left, Top Right, Bottom Left, Bottom Right** corners of the piste
- Each click is confirmed by pressing **Enter**
- Points are drawn and indexed on screen

## Usage

### 1. Process a video or image
```bash
python process_video.py input.mp4 --model yolov8n.pt --fps 30 --output out_folder
```

- `--model` : YOLO model to use
- `--fps` : Target FPS for processing output (default: 30)
- `--output`: Folder to store CSV (default: script folder)

### 2. Mark piste and fencer IDs
```bash
python mark_video.py csv_path video_path --save-video --output-path outputs/output.mp4
```
- `--save-video` : Save video to output path
- `--output-path`: File to store video (default: script folder)


## Dependencies
### Mamba/Conda
python 3.13
pyside6

### Pip
???
