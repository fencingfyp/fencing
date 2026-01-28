import os
from enum import Enum

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from src.gui.select_match_widget import MATCH_LIST_FOLDER
from src.gui.util.task_graph import HeatMapTasksToIds, Task, TaskGraph
from src.util.file_names import RAW_POSE_DATA_CSV_NAME

from .heat_map_menu_widget import HeatMapMenuWidget
from .track_poses_widget import TrackPosesWidget

TASK_DEPENDENCIES = [
    Task(
        HeatMapTasksToIds.TRACK_POSES.value,
        [RAW_POSE_DATA_CSV_NAME],
        deps=[],
    )
]


class HeatMapMainWidget(QWidget):
    navigate_to_manage_match = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_directory = None

        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self.setLayout(layout)

        self.task_graph: TaskGraph = TaskGraph(TASK_DEPENDENCIES)

        self.menu_widget = HeatMapMenuWidget(self.task_graph)
        self.stack.addWidget(self.menu_widget)
        self.menu_widget.navigate_to_manage_match.connect(
            lambda: self.navigate_to_manage_match.emit()
        )
        self.menu_widget.navigate_to_task.connect(self._on_task_button_clicked)

        self.tasks_to_widgets = {}
        # Add task widgets here, e.g.:
        # self.tasks_to_widgets[MomentumGraphTasksToIds.SOME_TASK.value] = SomeTaskWidget(self)
        self.tasks_to_widgets[HeatMapTasksToIds.TRACK_POSES.value] = TrackPosesWidget()

        for task_id, widget in self.tasks_to_widgets.items():
            self.stack.addWidget(widget)
            widget.back_button_clicked.connect(
                lambda: self.stack.setCurrentWidget(self.menu_widget)
            )
            widget.run_completed.connect(
                lambda tid=task_id: self.task_graph.mark_finished(tid.value)
            )
            widget.run_started.connect(
                lambda tid=task_id: self.task_graph.rerun(tid.value)
            )

    def set_match(self, match_name: str):
        self.match_name = match_name
        self.working_directory = os.path.join(MATCH_LIST_FOLDER, match_name)
        self.task_graph.set_working_directory(self.working_directory)

    @Slot(str)
    def _on_task_button_clicked(self, task_id: str):
        self.stack.setCurrentWidget(self.tasks_to_widgets[task_id])
        self.tasks_to_widgets[task_id].set_working_directory(self.working_directory)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = HeatMapMainWidget()
    widget.set_match("sabre_1")
    widget.show()
    sys.exit(app.exec())
