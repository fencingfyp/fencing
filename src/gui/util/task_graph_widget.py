from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView, QVBoxLayout, QWidget

from src.gui.util.task_graph_view import TaskGraphScene

from .task_graph import GraphLayout, TaskGraph, TaskState


class TaskGraphWidget(QWidget):
    node_clicked = Signal(str)  # emits task_id

    def __init__(self, task_graph: TaskGraph, parent=None):
        super().__init__(parent)
        self.task_graph = task_graph
        self.task_graph.graph_changed.connect(self.update_states)

        self.scene = TaskGraphScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.NoDrag)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        # build static layout
        self.graph_layout = self._build_graph_layout(task_graph)
        self.states = {tid: task_graph.state(tid).name for tid in task_graph.tasks}

        # render scene
        self.scene.render(
            self.graph_layout, self.states, self.task_graph.get_index_map()
        )

        # wire clicks
        for tid, node_item in self.scene.node_items.items():
            widget = node_item.widget()
            if widget:
                widget.mousePressEvent = self._make_node_click_handler(tid)

    def _make_node_click_handler(self, tid):
        def handler(event):
            self.node_clicked.emit(tid)

        return handler

    @Slot()
    def update_states(self):
        self.states = self.task_graph.get_task_states()
        self.scene.update_nodes(self.states)

    # optional: private helpers to build layout
    def _build_graph_layout(self, task_graph):
        from collections import defaultdict

        layers_dict = defaultdict(list)
        depth_cache = {}

        def compute_depth(tid):
            if tid in depth_cache:
                return depth_cache[tid]
            task = task_graph.tasks[tid]
            if not task.deps:
                depth_cache[tid] = 0
            else:
                depth_cache[tid] = 1 + max(compute_depth(d) for d in task.deps)
            return depth_cache[tid]

        for tid in task_graph.tasks:
            compute_depth(tid)

        for tid, depth in depth_cache.items():
            layers_dict[depth].append(tid)

        layers = [layers_dict[i] for i in sorted(layers_dict)]
        edges = [(t.id, c) for t in task_graph.tasks.values() for c in t.children]

        return GraphLayout(layers=layers, edges=edges)
