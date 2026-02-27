# src/pyside_pipelines/multi_region_cropper/output/output_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .csv_oneshot_quad_output import CsvOneShotQuadOutput
from .csv_quad_output import CsvQuadOutput
from .rectified_video_output import RectifiedVideoOutput
from .region_output import RegionOutput


@dataclass
class OutputConfig:
    label: str
    path: Path

    def build(self, quad: np.ndarray, fps: float) -> RegionOutput:
        raise NotImplementedError

    def build_at_path(self, path: Path, quad: np.ndarray, fps: float) -> RegionOutput:
        raise NotImplementedError

    def chunk_path(self, chunk_dir: str, worker_idx: int) -> Path:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict) -> OutputConfig:
        match d["output_type"]:
            case "rectified_video":
                return RectifiedOutputConfig(label=d["label"], path=Path(d["path"]))
            case "csv_oneshot_quad":
                return CsvOneShotQuadOutputConfig(
                    label=d["label"], path=Path(d["path"])
                )
            case "csv_quad":
                return CsvQuadOutputConfig(label=d["label"], path=Path(d["path"]))
            case _:
                raise ValueError(f"Unknown output type: {d['output_type']}")


@dataclass
class RectifiedOutputConfig(OutputConfig):
    def build(self, quad: np.ndarray, fps: float) -> RegionOutput:
        return RectifiedVideoOutput(self.path, fps, quad)

    def build_at_path(self, path: Path, quad: np.ndarray, fps: float) -> RegionOutput:
        return RectifiedVideoOutput(path, fps, quad)

    def chunk_path(self, chunk_dir: str, worker_idx: int) -> Path:
        return Path(chunk_dir) / f"{worker_idx:03d}_{self.label}.mp4"

    def to_dict(self) -> dict:
        return {
            "output_type": "rectified_video",
            "label": self.label,
            "path": str(self.path),
        }


@dataclass
class CsvOneShotQuadOutputConfig(OutputConfig):
    def build(self, quad: np.ndarray, fps: float) -> RegionOutput:
        return CsvOneShotQuadOutput(self.path, quad)

    def build_at_path(self, path: Path, quad: np.ndarray, fps: float) -> RegionOutput:
        return CsvOneShotQuadOutput(path, quad)

    def chunk_path(self, chunk_dir: str, worker_idx: int) -> Path:
        # CSV is a one-shot output — only worker 0 writes it, others skip
        if worker_idx == 0:
            return Path(chunk_dir) / f"{worker_idx:03d}_{self.label}.csv"
        return None

    def to_dict(self) -> dict:
        return {
            "output_type": "csv_oneshot_quad",
            "label": self.label,
            "path": str(self.path),
        }


@dataclass
class CsvQuadOutputConfig(OutputConfig):
    def build(self, quad: np.ndarray, fps: float) -> RegionOutput:
        return CsvQuadOutput(self.path)

    def build_at_path(self, path: Path, quad: np.ndarray, fps: float) -> RegionOutput:
        return CsvQuadOutput(path)

    def chunk_path(self, chunk_dir: str, worker_idx: int) -> Path:
        return Path(chunk_dir) / f"{worker_idx:03d}_{self.label}.csv"

    def to_dict(self) -> dict:
        return {
            "output_type": "csv_quad",
            "label": self.label,
            "path": str(self.path),
        }
