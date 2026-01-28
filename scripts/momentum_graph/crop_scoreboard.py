import argparse
import os

import cv2
import numpy as np

from src.model import (
    OpenCvUiV2,
    OrbTracker,
    Quadrilateral,
    SiftTracker,
    TargetTracker,
    Ui,
    UiCodes,
)
from src.pipelines.crop_region_pipeline import CropRegionPipeline
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io, setup_output_file, setup_output_video_io
from src.util.utils import generate_select_quadrilateral_instructions


def parse_arguments():
    """Parse command line arguments for cropping scoreboard region."""
    parser = argparse.ArgumentParser(
        description="Crop and rectify scoreboard region from video (with tracking)"
    )
    parser.add_argument(
        "output_folder", help="Path to folder for intermediate/final products"
    )
    parser.add_argument(
        "--demo", action="store_true", help="If set, doesn't output anything"
    )
    args = parser.parse_args()
    return args.output_folder, args.demo


def main():
    output_folder, demo_mode = parse_arguments()
    input_video = os.path.join(output_folder, ORIGINAL_VIDEO_NAME)
    output_path = (
        setup_output_file(output_folder, CROPPED_SCOREBOARD_VIDEO_NAME)
        if not demo_mode
        else None
    )

    cap, _, width, height, _ = setup_input_video_io(input_video)
    ui = OpenCvUiV2("Scoreboard Cropping", width=width, height=height)

    controller = CropRegionPipeline(cap, output_path, ui, region="scoreboard")
    controller.start()


if __name__ == "__main__":
    main()
