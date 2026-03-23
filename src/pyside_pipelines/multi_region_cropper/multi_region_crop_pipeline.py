import glob
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Callable

import cv2
from PySide6.QtCore import QTimer

from src.gui.base_task_widget.ui.PysideUi import PysideUi
from src.model.drawable.points_drawable import PointsDrawable
from src.model.drawable.quadrilateral_drawable import QuadrilateralDrawable
from src.model.tracker.DefinedRegion import DefinedRegion
from src.model.tracker.TargetTracker import TargetTracker
from src.pyside_pipelines.multi_region_cropper._concatenate import concatenate_files
from src.pyside_pipelines.multi_region_cropper._frame_ranges import get_frame_ranges
from src.pyside_pipelines.multi_region_cropper._worker import (
    build_tracker_from_configs,
    run_worker,
)
from src.pyside_pipelines.multi_region_cropper.label_config import LabelConfig
from src.pyside_pipelines.multi_region_cropper.region_state import RegionState

log = logging.getLogger(__name__)


class MultiRegionProcessingPipeline:
    WARMUP_FRAMES_S = 2
    N_WORKERS = max(2, (os.cpu_count() // 2))

    def __init__(
        self,
        video_path: str,
        defined_regions: list[DefinedRegion],
        label_configs: dict[str, LabelConfig],
        ui: PysideUi | None = None,
        on_finished: Callable | None = None,
        tracker: TargetTracker | None = None,
        n_workers: int | None = None,
        sequential: bool = False,
    ):
        self.video_path = video_path
        self.defined_regions = defined_regions
        self.label_configs = label_configs
        self.ui = ui
        self.on_finished = on_finished
        self.cancelled = False
        self.n_workers = n_workers or self.N_WORKERS

        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        ret, self._first_frame = cap.read()
        cap.release()
        if not ret or self._first_frame is None:
            raise ValueError("Failed to read first frame from video.")

        self._verify_frame_count(video_path)

        self._region_configs: list[dict] = [
            {
                **r.to_dict(),
                **label_configs[r.label].tracking.to_dict(),
            }
            for r in defined_regions
        ]

        self._output_configs: list[dict] = [
            cfg.to_dict()
            for r in defined_regions
            for cfg in label_configs[r.label].output_configs
        ]

        self.chunk_dir = tempfile.mkdtemp(prefix="mrcp_chunks_")
        self._cancel_flag = mp.Value("b", 0)
        self._progress_queue: mp.Queue = mp.Queue()
        self._processes: list[mp.Process] = []
        self._frames_completed = 0
        self._test_tracker = tracker
        self._run_sequentially = sequential

    def _verify_frame_count(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)

        # seek to near the end and read forward to find true EOF
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            reported
            - 10,  # 10 is arbitrary buffer to ensure we can read the last few frames
        )

        real_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        while cap.read()[0]:
            real_count += 1
        cap.release()

        if real_count != reported:
            log.warning(
                "FRAME COUNT MISMATCH: reported=%d actual=%d", reported, real_count
            )

        self.total_frames = real_count  # always use seek-verified count

    def start(self):
        self.ui.show_loading("Starting processing...")

        if self._test_tracker or self._run_sequentially:
            self.tracker = self._test_tracker or build_tracker_from_configs(
                self._region_configs, self.video_path
            )
            self._run_sequential()
            return

        ranges = list(get_frame_ranges(self.total_frames, self.n_workers))
        self._log_ranges(ranges)

        for i, (start, end) in enumerate(ranges):
            p = mp.Process(
                target=run_worker,
                args=(
                    self.video_path,
                    self._region_configs,
                    self._output_configs,
                    start,
                    end,
                    int(self.WARMUP_FRAMES_S * self.fps),
                    self.chunk_dir,
                    self._cancel_flag,
                    self._progress_queue,
                    i,
                    self.n_workers,
                    self.fps,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        self._poll_progress()

    @staticmethod
    def _log_ranges(ranges: list[tuple[int, int]]) -> None:
        total = 0
        log.debug("Worker ranges:")
        for i, (s, e) in enumerate(ranges):
            n = e - s
            log.debug("  worker %d: frames [%d, %d)  count=%d", i, s, e, n)
            total += n
        # This tells you immediately if ranges are disjoint and exhaustive
        log.debug("  Total frames across all ranges: %d", total)
        # Check for gaps/overlaps
        for i in range(1, len(ranges)):
            prev_end = ranges[i - 1][1]
            cur_start = ranges[i][0]
            if cur_start != prev_end:
                log.warning(
                    "GAP/OVERLAP between worker %d (end=%d) and worker %d (start=%d): delta=%d",
                    i - 1,
                    prev_end,
                    i,
                    cur_start,
                    cur_start - prev_end,
                )

    def _poll_progress(self):
        if self.cancelled:
            return

        try:
            while True:
                self._progress_queue.get_nowait()
                self._frames_completed += 1
        except Exception:
            pass

        if self.ui:
            pct = (
                (self._frames_completed / self.total_frames)
                if self.total_frames > 0
                else 0.0
            )
            self.ui.update_loading(
                pct,
                f"Processing... ({self._frames_completed}/{self.total_frames} frames)",
            )

        all_dead = all(not p.is_alive() for p in self._processes)

        if all_dead:
            # ── DIAGNOSTIC: log exit codes before touching output files ───────
            for i, p in enumerate(self._processes):
                log.debug("Worker %d exitcode=%s", i, p.exitcode)
                if p.exitcode not in (0, None):
                    log.warning("Worker %d exited with non-zero code %s", i, p.exitcode)
            self._concatenate_and_finish()
        else:
            QTimer.singleShot(100, self._poll_progress)

    def _validate_chunks(self):
        ranges = list(get_frame_ranges(self.total_frames, self.n_workers))

        for r in self.defined_regions:
            for cfg in self.label_configs[r.label].output_configs:
                sample_chunk = cfg.chunk_path(self.chunk_dir, 0)
                if sample_chunk is None:
                    continue
                ext = sample_chunk.suffix
                if ext not in (".mp4", ".avi", ".mov"):
                    continue

                total_written = 0
                for i, (start, end) in enumerate(ranges):
                    chunk_path = cfg.chunk_path(self.chunk_dir, i)
                    if chunk_path is None or not chunk_path.exists():
                        log.warning("MISSING chunk %d for %s", i, cfg.label)
                        continue

                    cap = cv2.VideoCapture(str(chunk_path))
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    expected = end - start
                    cap.release()

                    status = (
                        "OK"
                        if n_frames == expected
                        else f"MISMATCH — expected {expected}, got {n_frames} (delta {n_frames - expected})"
                    )
                    log.debug(
                        "chunk %d [%d-%d]: %d frames — %s",
                        i,
                        start,
                        end,
                        n_frames,
                        status,
                    )
                    total_written += n_frames

                log.debug(
                    "%s: %d/%d frames total (%s)",
                    cfg.label,
                    total_written,
                    self.total_frames,
                    "OK" if total_written == self.total_frames else "MISMATCH",
                )

    def _concatenate_and_finish(self):
        if self.cancelled:
            self._cleanup()
            return

        for p in self._processes:
            p.join()

        import glob

        for diag_file in glob.glob(os.path.join(self.chunk_dir, "worker_*_diag.txt")):
            with open(diag_file) as f:
                print(f.read())

        self._validate_chunks()

        for r in self.defined_regions:
            for cfg in self.label_configs[r.label].output_configs:
                sample_chunk = cfg.chunk_path(self.chunk_dir, 0)
                if sample_chunk is None:
                    continue

                ext = sample_chunk.suffix
                chunk_files = sorted(
                    glob.glob(os.path.join(self.chunk_dir, f"*_{cfg.label}{ext}"))
                )
                if chunk_files:
                    concatenate_files(chunk_files, str(cfg.path))

        self._finish()

    def _run_sequential(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.region_states = [
            RegionState(
                label=r.label,
                outputs=[
                    cfg.build(r.quad_np, self.fps)
                    for cfg in self.label_configs[r.label].output_configs
                ],
            )
            for r in self.defined_regions
        ]

        self.frame_id = 0
        QTimer.singleShot(0, self._step)

    def _step(self):
        if self.cancelled:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            for state in self.region_states:
                state.close()
            self._finish()
            return

        updated_quads = self.tracker.update_all(frame)
        draw_quads = []
        pts = []
        for state in self.region_states:
            quad = updated_quads.get(state.label) or self.tracker.get_previous_quad(
                state.label
            )
            if quad is not None:
                state.process(frame, quad.numpy(), self.frame_id)
                if state.label == "scoreboard":
                    draw_quads.append(QuadrilateralDrawable(quad, color=(0, 255, 0)))
                    pts.append(
                        PointsDrawable(
                            self.tracker.get_target_pts(state.label),
                            color=(255, 0, 0),
                            size=5,
                        )
                    )
        self.ui.update_loading(
            self.frame_id / self.total_frames if self.total_frames > 0 else 0.0,
            f"Processing frame {self.frame_id}/{self.total_frames}",
        )
        self.ui.set_fresh_frame(frame)
        self.ui.draw_objects(draw_quads + pts)
        self.frame_id += 1
        QTimer.singleShot(0, self._step)

    def _cleanup(self):
        if os.path.exists(self.chunk_dir):
            shutil.rmtree(self.chunk_dir, ignore_errors=True)
        self.cancelled = True
        self._cancel_flag.value = 1
        for p in self._processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

    def cancel(self):
        self._cleanup()
        files = [cfg["path"] for cfg in self._output_configs if "path" in cfg]
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

    def _finish(self):
        self._cleanup()
        self.ui.hide_loading()
        if self.on_finished:
            self.on_finished()
            self.on_finished()
