import argparse
import os
from src.momentum_graph.crop_scoreboard_tracked import crop_region

def parse_arguments():
    parser = argparse.ArgumentParser(description="Crop and rectify score lights region from video (with tracking)")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to folder for intermediate/final products")
    args = parser.parse_args()
    return args.input_video, args.output_folder

def setup_output_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    video_path = os.path.join(folder_path, "cropped_score_lights.mp4")
    print(f"Output video will be saved to: {video_path}")
    return video_path

def main():
    video_path, output_folder = parse_arguments()
    output_path = setup_output_folder(output_folder)
    crop_region(video_path, output_path, "score_lights", "Score Lights Cropping")

if __name__ == "__main__":
    main()
