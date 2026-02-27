# src/pyside_pipelines/multi_region_cropper/_worker.py

import multiprocessing as mp
import os

import cv2
import numpy as np

from src.model.tracker.DefinedRegion import DefinedRegion
from src.model.tracker.TargetTracker import TargetTracker
from src.model.tracker.tracker_factory import build_tracker
from src.pyside_pipelines.multi_region_cropper.output.output_config import OutputConfig
from src.pyside_pipelines.multi_region_cropper.output.region_output import RegionOutput

from .tracking_config import TrackingConfig


def build_tracker_from_configs(
    region_configs: list[dict],
    video_path: str,
) -> TargetTracker:
    """
    Reconstruct a tracker in a worker process.
    Always uses frame 0 — consistent with the main process reference frame.
    region_configs: list of dicts with keys: label, quad_np, tracking_strategy, mask_margin
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        raise RuntimeError(f"Worker could not read frame 0 from {video_path}")

    defined_regions = [
        DefinedRegion(
            label=c["label"],
            quad_np=np.array(c["quad_np"], dtype=np.float32),
        )
        for c in region_configs
    ]

    tracking_configs = {
        c["label"]: TrackingConfig.from_dict(
            {
                "tracking_strategy": c["tracking_strategy"],
                "mask_margin": c["mask_margin"],
            }
        )
        for c in region_configs
    }

    return build_tracker(defined_regions, tracking_configs, first_frame)


def run_worker(
    video_path: str,
    region_configs: list[dict],  # label + quad + tracking params
    output_configs: list[dict],  # serialised OutputConfig per label
    start_frame: int,
    end_frame: int,
    warmup_frames: int,
    chunk_dir: str,
    cancel_flag,
    progress_queue: mp.Queue,
    worker_idx: int,
    n_workers: int,
    fps: float,
):
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()

    run_worker_inner(
        video_path,
        region_configs,
        output_configs,
        start_frame,
        end_frame,
        warmup_frames,
        chunk_dir,
        cancel_flag,
        progress_queue,
        worker_idx,
        n_workers,
        fps,
    )

    profiler.disable()
    profiler.dump_stats(f"worker_{worker_idx}_profile.stats")


def run_worker_inner(
    video_path: str,
    region_configs: list[dict],  # label + quad + tracking params
    output_configs: list[dict],  # serialised OutputConfig per label
    start_frame: int,
    end_frame: int,
    warmup_frames: int,
    chunk_dir: str,
    cancel_flag,
    progress_queue: mp.Queue,
    worker_idx: int,
    n_workers: int,
    fps: float,
) -> None:
    thread_count = max(2, os.cpu_count() // n_workers)
    cv2.setNumThreads(thread_count)

    tracker = build_tracker_from_configs(region_configs, video_path)

    # Build chunk-scoped outputs — each worker writes to its own temp files
    region_outputs: dict[str, list[RegionOutput]] = {}

    for cfg in output_configs:
        label = cfg["label"]
        quad_np = np.array(
            next(r for r in region_configs if r["label"] == label)["quad_np"],
            dtype=np.float32,
        )
        output_cfg = OutputConfig.from_dict(cfg)
        chunk_path = output_cfg.chunk_path(chunk_dir, worker_idx)
        if chunk_path is None:
            continue  # e.g. CsvOutput on non-zero worker
        if label not in region_outputs:
            region_outputs[label] = []
        region_outputs[label].append(output_cfg.build_at_path(chunk_path, quad_np, fps))

    cap = cv2.VideoCapture(video_path)
    actual_start = max(0, start_frame - warmup_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

    frame_idx = actual_start
    while frame_idx < end_frame:
        if cancel_flag.value:
            break

        ret, frame = cap.read()
        if not ret:
            break

        quads = tracker.update_all(frame)

        if frame_idx >= start_frame:
            for label, outputs in region_outputs.items():
                quad = quads.get(label) or tracker.get_previous_quad(label)
                if quad is not None:
                    for output in outputs:
                        output.process(frame, quad.numpy(), frame_idx)
            progress_queue.put(1)

        frame_idx += 1

    cap.release()
    for outputs in region_outputs.values():
        for output in outputs:
            output.close()
