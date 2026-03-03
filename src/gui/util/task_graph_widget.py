from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView, QVBoxLayout, QWidget

from src.gui.util.task_graph_scene import TaskGraphScene

from .abstract_task_graph import AbstractTaskGraph


class TaskGraphWidget(QWidget):
    node_clicked = Signal(str)

    def __init__(self, task_graph: AbstractTaskGraph, parent=None):
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

        # Build from public API only
        self.graph_layout = self.task_graph.snapshot()
        self.states = self.task_graph.get_task_states()

        self.scene.render(
            self.graph_layout,
            self.states,
            self.task_graph.get_index_map(),
        )

        self._wire_clicks()

    # ----------------------------

    def _wire_clicks(self):
        for tid, node_item in self.scene.node_items.items():
            widget = node_item.widget()
            if widget:
                widget.mousePressEvent = self._make_node_click_handler(tid)

    def _make_node_click_handler(self, tid):
        def handler(event):
            self.node_clicked.emit(tid)

        return handler

    # ----------------------------

    @Slot()
    def update_states(self):
        self.states = self.task_graph.get_task_states()
        self.scene.update_nodes(self.states)
