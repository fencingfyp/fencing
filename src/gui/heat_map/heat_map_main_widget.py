import os
from enum import Enum

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.select_match_widget import MATCH_LIST_FOLDER
from src.gui.util.task_graph import HeatMapTasksToIds, Task, TaskGraph, TaskState
from src.gui.util.task_graph_navbar import TaskGraphNavbar
from src.util.file_names import RAW_POSE_DATA_CSV_NAME

from .heat_map_overview_widget import HeatMapOverviewWidget
from .track_fencers_widget import TrackFencersWidget
from .track_poses_widget import TrackPosesWidget

TASK_DEPENDENCIES = [
    Task(
        HeatMapTasksToIds.TRACK_POSES.value,
        [RAW_POSE_DATA_CSV_NAME],
        deps=[],
    ),
    Task(
        HeatMapTasksToIds.TRACK_FENCERS.value,
        [],
        deps=[HeatMapTasksToIds.TRACK_POSES.value],
    ),
]


class HeatMapMainWidget(QWidget):
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_directory = None

        self.task_graph: TaskGraph = TaskGraph(TASK_DEPENDENCIES)
        self.task_graph.graph_changed.connect(self._update_navbar_states)

        self.navbar = self.initialise_navbar()

        self.stack = QStackedWidget()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.navbar)
        root.addWidget(self.stack, 1)
        self.setLayout(root)

        self.overview_widget = HeatMapOverviewWidget(self.task_graph)
        self.stack.addWidget(self.overview_widget)
        self.overview_widget.task_selected.connect(self._open_task)

        self.initialise_task_widgets()
        self._update_navbar_states()

    def initialise_navbar(self):
        ordered_tasks = self.task_graph.topological_order()
        navbar = TaskGraphNavbar(ordered_tasks)
        navbar.task_clicked.connect(self._open_task)
        navbar.overview_clicked.connect(lambda: self._switch_to(self.overview_widget))
        navbar.back_clicked.connect(self._on_exit_requested)
        return navbar

    def initialise_task_widgets(self):
        self.tasks_to_widgets = {}
        self.tasks_to_widgets[HeatMapTasksToIds.TRACK_POSES.value] = TrackPosesWidget()
        self.tasks_to_widgets[HeatMapTasksToIds.TRACK_FENCERS.value] = (
            TrackFencersWidget()
        )

        for task_id, widget in self.tasks_to_widgets.items():
            self.stack.addWidget(widget)
            widget.run_completed.connect(
                lambda tid=task_id: self.task_graph.mark_finished(tid.value)
            )
            widget.run_started.connect(
                lambda tid=task_id: self.task_graph.rerun(tid.value)
            )

    def _update_navbar_states(self):
        for tid in self.tasks_to_widgets:
            state = self.task_graph.state(tid)
            self.navbar.update_task_state(tid, state)

    def _on_exit_requested(self):
        current_widget = self.stack.currentWidget()
        if (
            current_widget
            and hasattr(current_widget, "cancel")
            and hasattr(current_widget, "is_running")
            and current_widget.is_running
        ):
            current_widget.cancel()
        self.exit_requested.emit()

    def set_match(self, match_name: str):
        self.match_name = match_name
        self.working_directory = os.path.join(MATCH_LIST_FOLDER, match_name)
        self.task_graph.set_working_directory(self.working_directory)

    def _switch_to(self, widget: QWidget):
        current = self.stack.currentWidget()
        if (
            current
            and hasattr(current, "cancel")
            and hasattr(current, "is_running")
            and current.is_running
        ):
            current.cancel()

        self.stack.setCurrentWidget(widget)

    @Slot(str)
    def _open_task(self, task_id: str):
        if self.task_graph.state(task_id) != TaskState.LOCKED:
            self._switch_to(self.tasks_to_widgets[task_id])
            self.tasks_to_widgets[task_id].set_working_directory(self.working_directory)

    def closeEvent(self, event):
        for widget in self.tasks_to_widgets.values():
            if hasattr(widget, "cancel"):
                widget.cancel()
        return super().closeEvent(event)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = HeatMapMainWidget()
    widget.set_match("sabre_1")
    widget.show()
    sys.exit(app.exec())
