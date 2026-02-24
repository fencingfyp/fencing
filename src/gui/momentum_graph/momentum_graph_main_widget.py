import cProfile
import pstats

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QWidget

from src.gui.navbar.app_navigator import AppNavigator, View
from src.gui.util.task_graph import MomentumGraphTasksToIds, Task, TaskGraph, TaskState
from src.gui.util.task_graph_navbar import TaskGraphLocalNav
from src.pyside.MatchContext import MatchContext
from src.util.file_names import (
    CROPPED_SCORE_LIGHTS_VIDEO_NAME,
    CROPPED_SCOREBOARD_VIDEO_NAME,
    CROPPED_TIMER_VIDEO_NAME,
    DETECT_LIGHTS_OUTPUT_CSV_NAME,
    MOMENTUM_DATA_CSV_NAME,
    MOMENTUM_GRAPH_IMAGE_NAME,
    OCR_OUTPUT_CSV_NAME,
    PERIODS_JSON_NAME,
)

from .crop_regions_widget import CropRegionsWidget
from .detect_score_lights_widget import DetectScoreLightsWidget
from .generate_momentum_graph_widget import GenerateMomentumGraphWidget
from .momentum_graph_overview_widget import MomentumGraphOverviewWidget
from .perform_ocr_widget import PerformOcrWidget
from .select_periods_widget import SelectPeriodsWidget
from .view_stats_widget import ViewStatsWidget

TASK_DEPENDENCIES = [
    Task(
        MomentumGraphTasksToIds.CROP_REGIONS.value,
        [
            CROPPED_SCOREBOARD_VIDEO_NAME,
            CROPPED_SCORE_LIGHTS_VIDEO_NAME,
            # CROPPED_TIMER_VIDEO_NAME,
        ],
    ),
    Task(
        MomentumGraphTasksToIds.PERFORM_OCR.value,
        [OCR_OUTPUT_CSV_NAME],
        deps=[MomentumGraphTasksToIds.CROP_REGIONS.value],
    ),
    Task(
        MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS.value,
        [DETECT_LIGHTS_OUTPUT_CSV_NAME],
        deps=[MomentumGraphTasksToIds.CROP_REGIONS.value],
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
        MomentumGraphTasksToIds.SELECT_PERIODS.value,
        [PERIODS_JSON_NAME],
        deps=[],
    ),
    Task(
        MomentumGraphTasksToIds.VIEW_STATS.value,
        [MOMENTUM_GRAPH_IMAGE_NAME],
        deps=[
            MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH.value,
            MomentumGraphTasksToIds.SELECT_PERIODS.value,
        ],
    ),
]


def navigation(nav: AppNavigator, match_ctx: MatchContext):
    nav.register(
        view=View.MOMENTUM,
        title="Momentum Graph",
        widget=MomentumGraphMainWidget(match_ctx),
        parent=View.MANAGE_MATCH,
    )


class MomentumGraphMainWidget(QWidget):
    exit_requested = Signal()

    def __init__(self, match_context: MatchContext, parent=None):
        super().__init__(parent)
        self.match_context = match_context

        self.task_graph: TaskGraph = TaskGraph(TASK_DEPENDENCIES, match_context)
        self.task_graph.graph_changed.connect(self._update_navbar_states)

        self.local_navbar = self.initialise_navbar()

        self.stack = QStackedWidget()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # root.addWidget(self.local_navbar)
        root.addWidget(self.stack, 1)
        self.setLayout(root)

        self.overview_widget = MomentumGraphOverviewWidget(self.task_graph)
        self.stack.addWidget(self.overview_widget)
        self.overview_widget.task_selected.connect(self._open_task)

        self.initialise_task_widgets()
        self._update_navbar_states()

        self.match_context.match_changed.connect(self._on_match_changed)

    def initialise_navbar(self):
        ordered_tasks = self.task_graph.topological_order()
        navbar = TaskGraphLocalNav(ordered_tasks)
        navbar.task_requested.connect(self._open_task)
        navbar.overview_requested.connect(lambda: self._switch_to(self.overview_widget))
        navbar.back_requested.connect(self._on_exit_requested)
        return navbar

    def initialise_task_widgets(self):
        self.tasks_to_widgets = {}
        self.tasks_to_widgets[MomentumGraphTasksToIds.CROP_REGIONS.value] = (
            CropRegionsWidget(self.match_context)
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.PERFORM_OCR.value] = (
            PerformOcrWidget(self.match_context)
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS.value] = (
            DetectScoreLightsWidget(self.match_context)
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH.value] = (
            GenerateMomentumGraphWidget(self.match_context)
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.SELECT_PERIODS.value] = (
            SelectPeriodsWidget(self.match_context)
        )
        self.tasks_to_widgets[MomentumGraphTasksToIds.VIEW_STATS.value] = (
            ViewStatsWidget(self.match_context)
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
            self.local_navbar.update_task_state(tid, state)

    def _on_exit_requested(self):
        self.exit_requested.emit()

    @Slot()
    def _on_match_changed(self):
        self.match_name = self.match_context.file_manager.get_match_name()

    def _switch_to(self, widget: QWidget):
        self.stack.setCurrentWidget(widget)

    @Slot(str)
    def _open_task(self, task_id: str):
        if self.task_graph.state(task_id) != TaskState.LOCKED:
            self._switch_to(self.tasks_to_widgets[task_id])

    def showEvent(self, event):
        self._switch_to(self.overview_widget)
        return super().showEvent(event)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()

        widget = MomentumGraphMainWidget(match_context)
        match_context.set_file("matches_data/sabre_2.mp4")
        widget.show()
        sys.exit(app.exec())

    # Run the profiler and save stats to a file

    cProfile.run("main()", "profile.stats")

    # Load stats
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()  # remove extraneous path info
    stats.sort_stats("tottime")  # sort by total time

    # Print only top 10 functions
    stats.print_stats(10)
