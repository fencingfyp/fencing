# src/model/tracker/TrackingStrategy.py

from enum import Enum, auto


class TrackingStrategy(Enum):
    ORB = auto()
    AKAZE = auto()
