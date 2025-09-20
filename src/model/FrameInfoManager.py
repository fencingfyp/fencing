import csv
from collections.abc import Callable
from typing import Any

"""
This class manages loading and caching frame data from a CSV file.

The CSV file is expected to have a header row, with the first column being the frame index (integer),
and subsequent columns representing data for detected objects in that frame. The row_mapper function
is expected to provide an object with an "id" field to uniquely identify each object in a frame.

It is used to preload a certain number of frames ahead to allow "looking into the future",
to 1: check if a fencer id reappears in the future frames, and 2: check the scoreboard
in future frames if a score event is detected.
"""

class FrameInfoManager:
    def __init__(
        self,
        csv_path: str,
        fps: int,
        header_format: list[str],
        row_mapper: Callable[[list[str]], dict[str, Any]],
        num_ms_ahead: int = 10000,
    ):
        self.num_ms_ahead = num_ms_ahead
        self.frame_records = {}
        self.current_frame_index = 0
        self.loaded_up_to_frame = -1
        self.header_format = header_format
        self.row_mapper = row_mapper
        self.frame_reader = self.read_csv_by_frame(csv_path)
        self.fps = fps
        self.num_frames_ahead = max(1, int(self.fps * (self.num_ms_ahead / 1000)))
        print(
            f"FrameInfoManager initialized to load {self.num_frames_ahead} frames ahead at {self.fps} fps"
        )

    def read_csv_by_frame(self, path):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Ensure correct number of columns
            if len(header) != len(self.header_format):
                raise ValueError(
                    f"CSV has {len(header)} columns, expected {len(self.header_format)}"
                )

            current_frame = None
            batch = {}

            for row in reader:
                frame_id = int(row[0])  # assumes first col = frame_id

                if current_frame is None:
                    current_frame = frame_id

                # Fill in skipped frames
                while frame_id > current_frame:
                    yield current_frame, batch
                    batch = {}
                    current_frame += 1

                # Use user-supplied row mapping
                obj = self.row_mapper(row)
                batch[obj["id"]] = obj

            if current_frame is not None:
                yield current_frame, batch

    def preload_info_at_frame(self, frame_index: int):
        target_index = frame_index + self.num_frames_ahead
        while self.loaded_up_to_frame < target_index:
            frame_num, batch = next(self.frame_reader, (None, None))
            if frame_num is None or batch is None:
                break
            self.loaded_up_to_frame += 1
            self.frame_records[frame_num] = batch
        return self.frame_records.get(frame_index)

    def get_frame_info_at(self, frame_index: int) -> dict[int, dict] | None:
        self.frame_records.pop(frame_index - 1, None)
        if frame_index < self.current_frame_index:
            raise ValueError("Can only get frames in increasing order")
        if frame_index == self.current_frame_index:
            self.current_frame_index += 1
            return self.preload_info_at_frame(frame_index)
        return self.search_frame(frame_index)

    def search_frame(self, frame_index: int) -> dict[int, dict] | None:
        if frame_index in self.frame_records:
            return self.frame_records[frame_index]
        return None

    def iter_from_frame(self, start_frame: int):
        frame = start_frame
        while True:
            info = self.search_frame(frame)
            if info is None:
                break
            yield info
            frame += 1
        