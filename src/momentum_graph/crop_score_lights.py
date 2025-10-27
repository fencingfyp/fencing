import argparse
import os
from src.momentum_graph.crop_scoreboard import crop_region
from src.util import setup_output_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Crop and rectify score lights region from video (with tracking)")
    # parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_folder", help="Path to folder for intermediate/final products")
    args = parser.parse_args()
    return args.output_folder

def main():
    output_folder = parse_arguments()
    video_path = os.path.join(output_folder, "original.mp4")
    output_path = setup_output_file(output_folder, "cropped_score_lights.mp4")
    crop_region(video_path, output_path, "score_lights", "Score Lights Cropping")

if __name__ == "__main__":
    main()
