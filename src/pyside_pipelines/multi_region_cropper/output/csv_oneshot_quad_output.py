import csv

import numpy as np

from src.model.Quadrilateral import Quadrilateral

from .region_output import RegionOutput


def get_header_row():
    return [
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


def row_mapper(row) -> Quadrilateral:
    # Assumes header is in format: frame_id, x0, y0, x1, y1, x2, y2, x3, y3
    return Quadrilateral(
        [
            (float(row[1]), float(row[2])),
            (float(row[3]), float(row[4])),
            (float(row[5]), float(row[6])),
            (float(row[7]), float(row[8])),
        ]
    )


class CsvOneShotQuadOutput(RegionOutput):
    def __init__(self, csv_path: str, quad_np: np.ndarray):
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._file)

        self._writer.writerows(
            [
                get_header_row(),
                [0, *quad_np.flatten()],
            ]
        )

    def process(
        self,
        frame: np.ndarray,
        quad_np: np.ndarray,
        frame_id: int,
    ):
        pass  # No processing needed, as we write the quad in the constructor

    def close(self):
        self._file.close()

    def delete(self):
        import os

        self._file.close()
        os.remove(self._file.name)
