from PySide6.QtCore import QObject, Signal

from .abstract_task_graph import AbstractTaskGraph, GraphLayout, TaskState
from .task_graph import TaskGraph


class TaskGraphView(AbstractTaskGraph):
    task_changed = Signal(str)
    graph_changed = Signal()

    def __init__(
        self,
        core: TaskGraph,
        visible_ids: set[str],
        parent=None,
    ):
        super().__init__(parent)

        self.core = core
        self.visible_ids = visible_ids

        self.core.task_changed.connect(self._forward_task_changed)
        self.core.graph_changed.connect(self.graph_changed.emit)

    # ===============================
    # Public API (same as TaskGraph)
    # ===============================

    def state(self, tid: str) -> TaskState:
        return self.core.state(tid)

    def rerun(self, tid: str):
        self.core.rerun(tid)

    def mark_finished(self, tid: str):
        self.core.mark_finished(tid)

    def set_working_directory(self, working_dir: str):
        self.core.set_working_directory(working_dir)

    def get_task_states(self) -> dict[str, str]:
        return {
            tid: state
            for tid, state in self.core.get_task_states().items()
            if tid in self.visible_ids
        }

    def get_index_map(self) -> dict[str, int]:
        index_map = self.core.get_index_map()
        return {tid: idx for tid, idx in index_map.items() if tid in self.visible_ids}

    def snapshot(self) -> GraphLayout:
        full = self.core.snapshot()

        layers = [
            [tid for tid in layer if tid in self.visible_ids] for layer in full.layers
        ]
        layers = [layer for layer in layers if layer]

        edges = [
            (a, b)
            for (a, b) in full.edges
            if a in self.visible_ids and b in self.visible_ids
        ]

        return GraphLayout(layers, edges)

    def topological_order(self) -> list[str]:
        """
        Return topological order filtered to visible tasks,
        preserving the core graph's ordering.
        """
        full_order = self.core.topological_order()
        return [tid for tid in full_order if tid in self.visible_ids]

    # ===============================
    # Internal
    # ===============================

    def _forward_task_changed(self, tid: str):
        if tid in self.visible_ids:
            self.task_changed.emit(tid)
        if tid in self.visible_ids:
            self.task_changed.emit(tid)
