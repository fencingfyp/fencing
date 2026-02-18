import csv

import numpy as np

from .region_output import RegionOutput


class CsvQuadOutput(RegionOutput):
    def __init__(self, csv_path: str):
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)

        self._writer.writerow(
            [
                "frame_id",
                "x0",
                "y0",
                "x1",
                "y1",
                "x2",
                "y2",
                "x3",
                "y3",
            ]
        )

    def process(
        self,
        frame: np.ndarray,
        quad_np: np.ndarray,
        frame_id: int,
    ):
        # quad_np shape: (4, 2)
        self._writer.writerow([frame_id, *quad_np.flatten()])

    def close(self):
        self._file.close()

    def delete(self):
        import os

        self._file.close()
        os.remove(self._file.name)
