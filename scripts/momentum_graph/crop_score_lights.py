"""Crop and rectify score lights region from video (with tracking)"""

import argparse
import os

from pipelines.multi_region_crop_pipeline import MultiRegionCropPipeline
from src.model import OpenCvUiV2
from src.util.file_names import CROPPED_SCORE_LIGHTS_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io, setup_output_file


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Crop and rectify score lights region from video (with tracking)"
    )
    parser.add_argument(
        "output_folder", help="Path to folder for intermediate/final products"
    )
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output anything"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_folder = args.output_folder
    demo_mode = args.demo
    input_video_path = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    output_video_path = (
        setup_output_file(output_folder, CROPPED_SCORE_LIGHTS_VIDEO_NAME)
        if not demo_mode
        else None
    )
    cap, _, width, height, _ = setup_input_video_io(input_video_path)
    ui = OpenCvUiV2("Score Lights Cropping", width=width, height=height)
    controller = MultiRegionCropPipeline(
        cap,
        ui,
        output_paths={"score lights": output_video_path},
    )
    controller.start()


if __name__ == "__main__":
    main()
