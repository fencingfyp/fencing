import copy
import csv
from collections.abc import Callable
from typing import Any, Dict, Iterator


class FrameInfoManager:
    """
    Loads frame-indexed object data from a CSV and maintains a sliding cache.

    Guarantees:
    - Frames may be requested multiple times.
    - A frame `k` is evicted only when frame `k + 2` is requested.
    - Frames are loaded lazily, up to a configurable lookahead window.
    """

    def __init__(
        self,
        csv_path: str,
        fps: int,
        header_format: list[str],
        row_mapper: Callable[[list[str]], dict[str, Any]],
        num_ms_ahead: int = 10_000,
    ):
        self.fps = fps
        self.header_format = header_format
        self.row_mapper = row_mapper

        self.num_ms_ahead = num_ms_ahead
        self.num_frames_ahead = max(1, int(self.fps * (self.num_ms_ahead / 1000)))

        # Sliding cache: frame_index -> frame_data
        self._cache: Dict[int, Dict[int, dict]] = {}

        # Generator state
        self._frame_reader = self._read_csv_by_frame(csv_path)
        self._loaded_up_to_frame: int = -1

        # Highest frame index ever requested (monotonic)
        self._max_requested_frame: int = -1

        print(
            f"FrameInfoManager: preloading {self.num_frames_ahead} frames ahead at {self.fps} fps"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame_and_advance(self, frame_index: int) -> Dict[int, dict] | None:
        """
        Main access method.

        Semantics:
        - Frame may be requested multiple times.
        - When frame_index + 2 is requested, frame_index is evicted.
        """
        if frame_index < self._max_requested_frame:
            # backward requests are allowed, but must already be cached
            return self._cache.get(frame_index)

        self._max_requested_frame = frame_index

        # Preload ahead
        self._preload_up_to(frame_index + self.num_frames_ahead)

        # Evict frames that are now too old
        self._evict_before(frame_index - 1)

        return self._cache.get(frame_index)

    def find_cached_forward(
        self, start_frame: int
    ) -> Iterator[tuple[int, Dict[int, dict]]]:
        """
        Iterate forward over already-cached frames starting at start_frame.
        Stops when a frame is missing from cache.
        """
        frame = start_frame
        while frame in self._cache:
            yield frame, self._cache[frame]
            frame += 1

    def get_cached(self, frame_index: int) -> Dict[int, dict] | None:
        """Pure lookup: does not preload, does not evict."""
        return self._cache.get(frame_index)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _preload_up_to(self, target_frame: int) -> None:
        """
        Load frames from the CSV generator until we reach target_frame
        or exhaust the generator.
        """
        while self._loaded_up_to_frame < target_frame:
            frame_num, batch = next(self._frame_reader, (None, None))
            if frame_num is None:
                break

            self._cache[frame_num] = batch
            self._loaded_up_to_frame = frame_num

    def _evict_before(self, min_frame_to_keep: int) -> None:
        """
        Remove all cached frames < min_frame_to_keep.
        With the new rule, when requesting k+2, we evict k.
        """
        to_delete = [f for f in self._cache if f < min_frame_to_keep]
        for f in to_delete:
            del self._cache[f]

    # ------------------------------------------------------------------
    # CSV reader
    # ------------------------------------------------------------------

    def _read_csv_by_frame(self, path: str, ffill: bool = True):
        """
        Yields (frame_index, frame_data_dict) indefinitely, forward-filling
        the last known state if ffill=True.
        """
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            if len(header) != len(self.header_format):
                raise ValueError(
                    f"CSV has {len(header)} columns, expected {len(self.header_format)}"
                )

            batch: Dict[int, dict] = {}
            current_frame: int | None = None

            try:
                current_row = next(reader)
            except StopIteration:
                return

            while True:
                frame_id = int(current_row[0])
                obj = self.row_mapper(current_row)
                obj_id = obj["id"]

                if current_frame is None:
                    current_frame = frame_id

                # New frame encountered → flush previous frame
                if frame_id != current_frame:
                    yield current_frame, copy.deepcopy(batch)

                    # Forward-fill missing frames
                    if ffill:
                        for missing in range(current_frame + 1, frame_id):
                            yield missing, {}

                    batch = {}
                    current_frame = frame_id

                # Update current frame
                batch[obj_id] = obj

                try:
                    current_row = next(reader)
                except StopIteration:
                    yield current_frame, copy.deepcopy(batch)
                    break

            # After EOF, forward-fill forever
            if current_frame is not None and ffill:
                frozen = copy.deepcopy(batch)
                while True:
                    current_frame += 1
                    yield current_frame, frozen
