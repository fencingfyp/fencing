from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QObject, Signal


class TaskState(Enum):
    LOCKED = 0
    READY = 1
    DONE = 2


@dataclass
class GraphLayout:
    layers: list[list[str]]  # left → right columns
    edges: list[tuple[str, str]]  # (from, to)


class AbstractTaskGraph(QObject):
    task_changed = Signal(str)
    graph_changed = Signal()

    def state(self, tid: str) -> TaskState: ...
    def get_task_states(self) -> dict[str, str]: ...
    def get_index_map(self) -> dict[str, int]: ...
    def snapshot(self) -> GraphLayout: ...
    def topological_order(self) -> list[str]: ...
