from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

from src.gui.util.task_graph import TaskGraph, TaskState
from src.gui.util.task_graph_widget import TaskGraphWidget


class MomentumGraphOverviewWidget(QWidget):
    task_selected = Signal(str)

    def __init__(self, task_graph: TaskGraph, parent=None):
        super().__init__(parent)
        self.task_graph = task_graph

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graph_widget = TaskGraphWidget(task_graph)
        layout.addWidget(self.graph_widget)

        self.graph_widget.node_clicked.connect(self.on_task_button_clicked)
        self.task_graph.graph_changed.connect(self.graph_widget.update_states)

    @Slot(str)
    def on_task_button_clicked(self, task_id: str):
        self.task_selected.emit(task_id)
