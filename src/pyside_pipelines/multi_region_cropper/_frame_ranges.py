# src/pyside_pipelines/multi_region_cropper/_frame_ranges.py


def get_frame_ranges(total_frames: int, n_workers: int) -> list[tuple[int, int]]:
    chunk = total_frames // n_workers
    ranges = [(i * chunk, (i + 1) * chunk) for i in range(n_workers)]
    ranges[-1] = (ranges[-1][0], total_frames)  # last worker gets remainder
    return ranges
