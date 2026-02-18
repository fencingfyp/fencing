import json
from enum import Enum, StrEnum
from pathlib import Path

# Example file names
from src.util.file_names import (
    CROPPED_PISTE_VIDEO_NAME,
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    CROPPED_SCOREBOARD_VIDEO_NAME,
    CROPPED_TIMER_VIDEO_NAME,
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
    MOMENTUM_DATA_CSV_NAME,
    MOMENTUM_GRAPH_IMAGE_NAME,
    OCR_OUTPUT_CSV_NAME,
    ORIGINAL_VIDEO_NAME,
    PERIODS_JSON_NAME,
    PROCESSED_POSE_DATA_CSV_NAME,
    RAW_PISTE_QUADS_CSV_NAME,
    RAW_POSE_DATA_CSV_NAME,
)

# Sidecar metadata file
METADATA_FILE_NAME = "metadata.json"
METADATA_VERSION = 1


class FileRole(StrEnum):
    # Momentum graph
    CROPPED_SCOREBOARD = CROPPED_SCOREBOARD_VIDEO_NAME
    CROPPED_SCORE_LIGHTS = CROPPED_SCORE_LIGHTS_VIDEO_NAME
    CROPPED_TIMER = CROPPED_TIMER_VIDEO_NAME
    RAW_SCORES = OCR_OUTPUT_CSV_NAME
    RAW_LIGHTS = DETECT_LIGHTS_OUTPUT_CSV_NAME
    MOMENTUM_DATA = MOMENTUM_DATA_CSV_NAME
    MOMENTUM_GRAPH = MOMENTUM_GRAPH_IMAGE_NAME
    PERIODS = PERIODS_JSON_NAME

    # Heat map
    RAW_POSE = RAW_POSE_DATA_CSV_NAME
    PROCESSED_POSE = PROCESSED_POSE_DATA_CSV_NAME
    RAW_PISTE_QUADS = RAW_PISTE_QUADS_CSV_NAME
    CROPPED_PISTE = CROPPED_PISTE_VIDEO_NAME


class FileManager:
    """
    Resolves paths for a given video file inside its enforced sidecar folder.
    Enforces that the metadata file exists for minimal integrity.
    """

    SIDECAR_SUFFIX = ".data"

    def __init__(self, video_file_path: str):
        self.video_file = Path(video_file_path)
        self.video_name = self.video_file.stem
        self.workspace_dir = (
            self.video_file.parent / f"{self.video_name}{self.SIDECAR_SUFFIX}"
        )

        if not self.workspace_dir.exists():
            raise FileNotFoundError(
                f"Required sidecar folder '{self.workspace_dir}' does not exist."
            )

        # minimal integrity check: metadata file must exist
        if not (self.workspace_dir / METADATA_FILE_NAME).exists():
            raise FileNotFoundError(
                f"Metadata file '{METADATA_FILE_NAME}' is missing in sidecar folder '{self.workspace_dir}'."
            )

    def get_original_video(self) -> Path:
        """Return the original video file path."""
        return self.video_file

    def get_working_directory(self) -> Path:
        """Return the sidecar folder path."""
        return self.workspace_dir

    def get_match_name(self) -> str:
        """Return the match name derived from the video file name."""
        return self.video_name

    def get_path(self, role: FileRole) -> Path:
        """Return the path for a given file role in the sidecar folder."""
        return self.workspace_dir / role.value

    def file_exists(self, path_str: str) -> bool:
        """Check if the file for a given role exists."""
        return (self.workspace_dir / path_str).exists()

    @staticmethod
    def create_sidecar(video_file_path: str, overwrite_metadata=False) -> Path:
        """
        Create the sidecar folder for a video if it doesn't exist.
        Also creates a default metadata file.
        """
        video_file = Path(video_file_path)
        sidecar = video_file.parent / f"{video_file.stem}{FileManager.SIDECAR_SUFFIX}"
        sidecar.mkdir(exist_ok=True)

        metadata_path = sidecar / METADATA_FILE_NAME
        if overwrite_metadata or not metadata_path.exists():
            metadata = {
                "version": METADATA_VERSION,
                "original_video": video_file.name,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        return sidecar

    def read_metadata(self) -> dict:
        """Return the metadata contents as a dictionary."""
        metadata_path = self.workspace_dir / METADATA_FILE_NAME
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_integrity(self) -> bool:
        """
        Minimal integrity check: metadata file exists.
        Returns True if valid, False otherwise.
        """
        return (self.workspace_dir / METADATA_FILE_NAME).exists()

    @staticmethod
    def has_valid_sidecar(video_file_path: str) -> bool:
        """
        Check if the given video file has a valid sidecar folder with metadata.
        Returns True if valid, False otherwise.
        """
        video_file = Path(video_file_path)
        sidecar = video_file.parent / f"{video_file.stem}{FileManager.SIDECAR_SUFFIX}"
        metadata_path = sidecar / METADATA_FILE_NAME
        return sidecar.exists() and metadata_path.exists()
