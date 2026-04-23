from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from src.gui.task_graph.task_graph import TaskGraph


# This file is for the "overview" page of the heat map section. It currently doesn't do much.
class HeatMapOverviewWidget(QWidget):
    task_selected = Signal(str)

    def __init__(self, task_graph: TaskGraph, parent=None):
        super().__init__(parent)
        self.task_graph = task_graph
        self.task_graph = task_graph
