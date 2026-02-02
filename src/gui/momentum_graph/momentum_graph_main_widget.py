import os
from enum import Enum

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from src.gui.select_match_widget import MATCH_LIST_FOLDER
from src.gui.util.task_graph import MomentumGraphTasksToIds, Task, TaskGraph, TaskState
from src.gui.util.task_graph_navbar import TaskGraphNavbar
from src.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    CROPPED_SCOREBOARD_VIDEO_NAME,
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
    MOMENTUM_DATA_CSV_NAME,
    MOMENTUM_GRAPH_IMAGE_NAME,
    OCR_OUTPUT_CSV_NAME,
    START_TIME_JSON_NAME,
)

from .crop_score_lights_widget import CropScoreLightsWidget
from .crop_scoreboard_widget import CropScoreboardWidget
from .detect_score_lights_widget import DetectScoreLightsWidget
from .generate_momentum_graph_widget import GenerateMomentumGraphWidget
from .momentum_graph_overview_widget import MomentumGraphOverviewWidget
from .perform_ocr_widget import PerformOcrWidget
from .select_start_time_widget import SelectStartTimeWidget
from .view_stats_widget import ViewStatsWidget

TASK_DEPENDENCIES = [
    Task(
        MomentumGraphTasksToIds.CROP_SCOREBOARD.value, [CROPPED_SCOREBOARD_VIDEO_NAME]
    ),
    Task(
        MomentumGraphTasksToIds.CROP_SCORE_LIGHTS.value,
        [CROPPED_SCORE_LIGHTS_VIDEO_NAME],
    ),
    Task(
        MomentumGraphTasksToIds.PERFORM_OCR.value,
        [OCR_OUTPUT_CSV_NAME],
        deps=[MomentumGraphTasksToIds.CROP_SCOREBOARD.value],
    ),
    Task(
        MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS.value,
        [DETECT_LIGHTS_OUTPUT_CSV_NAME],
        deps=[MomentumGraphTasksToIds.CROP_SCORE_LIGHTS.value],
    ),
    Task(
        MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH.value,
        [MOMENTUM_DATA_CSV_NAME],
        deps=[
            MomentumGraphTasksToIds.PERFORM_OCR.value,
            MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS.value,
        ],
    ),
    Task(
        MomentumGraphTasksToIds.GET_START_TIME.value,
        [START_TIME_JSON_NAME],
        deps=[],
    ),
    Task(
        MomentumGraphTasksToIds.VIEW_STATS.value,
        [MOMENTUM_GRAPH_IMAGE_NAME],
        deps=[
            MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH.value,
            MomentumGraphTasksToIds.GET_START_TIME.value,
        ],
    ),
]


class MomentumGraphMainWidget(QWidget):
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

        self.overview_widget = MomentumGraphOverviewWidget(self.task_graph)
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
        self.tasks_to_widgets[MomentumGraphTasksToIds.CROP_SCOREBOARD.value] = (
            CropScoreboardWidget()
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.CROP_SCORE_LIGHTS.value] = (
            CropScoreLightsWidget()
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.PERFORM_OCR.value] = (
            PerformOcrWidget()
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS.value] = (
            DetectScoreLightsWidget()
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH.value] = (
            GenerateMomentumGraphWidget()
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.GET_START_TIME.value] = (
            SelectStartTimeWidget()
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.VIEW_STATS.value] = (
            ViewStatsWidget()
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
        if current_widget and hasattr(current_widget, "cancel"):
            current_widget.cancel()
        self.exit_requested.emit()

    def set_match(self, match_name: str):
        self.match_name = match_name
        self.working_directory = os.path.join(MATCH_LIST_FOLDER, match_name)
        self.task_graph.set_working_directory(self.working_directory)

    def _switch_to(self, widget: QWidget):
        current = self.stack.currentWidget()
        if current and hasattr(current, "cancel"):
            current.cancel()

        self.stack.setCurrentWidget(widget)

    @Slot(str)
    def _open_task(self, task_id: str):
        if self.task_graph.state(task_id) != TaskState.LOCKED:
            self._switch_to(self.tasks_to_widgets[task_id])
            self.tasks_to_widgets[task_id].set_working_directory(self.working_directory)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = MomentumGraphMainWidget()
    widget.set_match("sabre_1")
    widget.show()
    sys.exit(app.exec())
