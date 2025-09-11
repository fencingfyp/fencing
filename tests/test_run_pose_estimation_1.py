# tests/test_extract_rows.py

from typing import Any, List
import torch
import pytest

from src.run_pose_estimation_1 import extract_rows


class DummyBox:
    def __init__(self):
        self.cls = torch.tensor(0)  # person
        self.id = torch.tensor(1)
        self.conf = torch.tensor(0.4126)
        self.xyxy = torch.tensor([[0.0, 485.2184, 71.9604, 893.2097]])


class DummyKeypoints:
    def __init__(self):
        self.xyn = torch.tensor([[
            [0.0000, 0.5592],
            [0.0000, 0.5529],
            [0.0000, 0.5533],
            [0.0000, 0.5488],
            [0.0000, 0.5496],
            [0.0044, 0.5711],
            [0.0038, 0.5720],
            [0.0276, 0.6219],
            [0.0268, 0.6281],
            [0.0507, 0.6708],
            [0.0493, 0.6732],
            [0.0058, 0.6615],
            [0.0068, 0.6622],
            [0.0141, 0.7095],
            [0.0156, 0.7105],
            [0.0108, 0.7841],
            [0.0099, 0.7855],
        ]])
        self.conf = torch.tensor([[
            0.0150, 0.0116, 0.0076, 0.0448, 0.0246, 0.1564,
            0.0814, 0.3230, 0.1354, 0.2901, 0.1455, 0.3322,
            0.2588, 0.2427, 0.1918, 0.1332, 0.1128
        ]])
        self.xy = torch.tensor([[
            [0.0, 603.9612],
            [0.0, 597.1505],
            [0.0, 597.5877],
            [0.0, 592.6752],
            [0.0, 593.5822],
            [3.5961, 616.7649],
            [3.1113, 617.8072],
            [22.3863, 671.6356],
            [21.7002, 678.3873],
            [41.1069, 724.4854],
            [39.9496, 727.1018],
            [4.6953, 714.3774],
            [5.4998, 715.1501],
            [11.4606, 766.2814],
            [12.6142, 767.3644],
            [8.7498, 846.7787],
            [7.9856, 848.2888],
        ]])


class DummyResult:
    def __init__(self):
        self.boxes = [DummyBox()]
        self.keypoints = [DummyKeypoints()]


def test_extract_rows() -> None:
    results: List[Any] = [DummyResult()]
    frame_idx: int = 0

    rows = extract_rows(results, frame_idx)

    # ensure one row is extracted
    assert len(rows) == 1
    row = rows[0]

    # length
    assert len(row) == 58 # 7 + 17 * 3
    # frame index
    assert row[0] == 0
    # id
    assert row[1] == 1
    # confidence float close
    assert pytest.approx(row[2], rel=1e-4) == 0.4126
    # bounding box coords
    assert pytest.approx(row[3:7], rel=1e-4) == [0.0, 485.2184, 71.9604, 893.2097]
    # check some keypoints normalised
    assert pytest.approx(row[7], rel=1e-4) == 0.0000  # first kp x
    assert pytest.approx(row[8], rel=1e-4) == 603.9612  # first kp y
    assert pytest.approx(row[9], rel=1e-4) == 0.0150  # first kp visibility
    assert pytest.approx(row[55], rel=1e-4) == 7.9856  # last kp x
    assert pytest.approx(row[56], rel=1e-4) == 848.2888  # last kp y
    assert pytest.approx(row[57], rel=1e-4) == 0.1128  # last kp visibility
