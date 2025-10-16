import csv
from collections.abc import Callable
from typing import Any
import copy

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

    def read_csv_by_frame(self, path, ffill: bool = True):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Ensure correct number of columns
            if len(header) != len(self.header_format):
                raise ValueError(
                    f"CSV has {len(header)} columns, expected {len(self.header_format)}"
                )

            last_seen = {}
            batch = {}
            current_frame = None

            try:
                current_row = next(reader)
            except StopIteration:
                return  # empty CSV

            while True:
                frame_id = int(current_row[0])
                obj = self.row_mapper(current_row)
                obj_id = obj["id"]

                # Initialize frame tracking
                if current_frame is None:
                    current_frame = frame_id

                # If we moved to a new frame, yield the previous one
                if frame_id != current_frame:
                    # yield previous frame info (with ffill if enabled)
                    # print(f"Yielding new frame {current_frame} with {len(batch)} objects")
                    yield current_frame, copy.deepcopy(last_seen if ffill else batch)

                    # fill missing frames if needed
                    if ffill:
                        for missing_frame in range(current_frame + 1, frame_id):
                            # print(f"Filling missing frame {missing_frame}")
                            yield missing_frame, copy.deepcopy(last_seen)

                    # reset for next frame
                    batch = {}
                    current_frame = frame_id

                # Update current frame info
                batch[obj_id] = obj
                last_seen[obj_id] = obj

                # Advance to next row
                try:
                    current_row = next(reader)
                except StopIteration:
                    # End of file — yield last known data once
                    yield current_frame, copy.deepcopy(last_seen if ffill else batch)
                    break
            # Continue yielding last known state indefinitely
            if current_frame is not None and ffill:
                frozen_state = copy.deepcopy(last_seen)
                while True:
                    # print(f"Filling post-end frame {current_frame}")
                    current_frame += 1
                    yield current_frame, frozen_state
                    

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
        