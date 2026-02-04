import argparse
import os

from src.model import OpenCvUiV2
from src.pipelines.multi_region_crop_pipeline import MultiRegionCropPipeline
from src.util.file_names import CROPPED_SCOREBOARD_VIDEO_NAME, ORIGINAL_VIDEO_NAME
from src.util.io import setup_input_video_io, setup_output_file


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

    controller = MultiRegionCropPipeline(
        cap, ui, output_paths={"scoreboard": output_path}
    )
    controller.start()


if __name__ == "__main__":
    import cProfile
    import pstats

    # Run the profiler and save stats to a file
    cProfile.run("main()", "profile.stats")

    # Load stats
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()  # remove extraneous path info
    stats.sort_stats("tottime")  # sort by total time

    # Print only top 10 functions
    stats.print_stats(10)
