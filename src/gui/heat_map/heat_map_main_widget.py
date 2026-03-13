from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QApplication, QHBoxLayout, QStackedWidget, QWidget

from src.gui.heat_map.generate_heat_map_widget import GenerateHeatMapWidget
from src.gui.MatchContext import MatchContext
from src.gui.momentum_graph.crop_regions_widget import CropRegionsWidget
from src.gui.momentum_graph.detect_score_lights_widget import DetectScoreLightsWidget
from src.gui.momentum_graph.generate_momentum_graph_widget import (
    GenerateMomentumGraphWidget,
)
from src.gui.momentum_graph.perform_ocr_widget import PerformOcrWidget
from src.gui.navbar.app_navigator import AppNavigator, View
from src.gui.task_graph.task_graph import (
    HeatMapTasksToIds,
    TaskGraph,
    TaskState,
    TasksToIds,
)
from src.gui.task_graph.task_graph_navbar import TaskGraphLocalNav
from src.gui.task_graph.task_graph_view import TaskGraphView

from .heat_map_overview_widget import HeatMapOverviewWidget
from .track_fencers_widget import TrackFencersWidget
from .track_poses_widget import TrackPosesWidget


def navigation(nav: AppNavigator, match_ctx: MatchContext):
    nav.register(
        view=View.HEAT_MAP,
        title="Heat Map",
        widget=HeatMapMainWidget(match_ctx),
        parent=View.MANAGE_MATCH,
    )


class HeatMapMainWidget(QWidget):
    exit_requested = Signal()

    def __init__(self, match_context: MatchContext, parent=None):
        super().__init__(parent)
        self.match_context = match_context
        self.working_directory = None

        self.task_graph: TaskGraph = match_context.task_graph
        self.task_view: TaskGraphView = TaskGraphView(
            self.task_graph, {id.value for id in HeatMapTasksToIds}
        )
        self.task_view.graph_changed.connect(self._update_navbar_states)

        self.local_navbar = self.initialise_navbar()

        self.stack = QStackedWidget()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.stack, 1)
        self.setLayout(root)

        self.overview_widget = HeatMapOverviewWidget(self.task_view)
        self.stack.addWidget(self.overview_widget)
        self.overview_widget.task_selected.connect(self._open_task)

        self.initialise_task_widgets()
        self._update_navbar_states()

        self.match_context.match_changed.connect(self._on_match_changed)

    def initialise_navbar(self):
        ordered_tasks = self.task_view.topological_order()
        navbar = TaskGraphLocalNav(ordered_tasks)
        navbar.task_requested.connect(self._open_task)
        navbar.overview_requested.connect(lambda: self._switch_to(self.overview_widget))
        navbar.back_requested.connect(self._on_exit_requested)
        return navbar

    def initialise_task_widgets(self):
        self.tasks_to_widgets = {}
        self.tasks_to_widgets[TasksToIds.CROP_REGIONS.value] = CropRegionsWidget(
            self.match_context
        )
        self.tasks_to_widgets[TasksToIds.PERFORM_OCR.value] = PerformOcrWidget(
            self.match_context
        )
        self.tasks_to_widgets[TasksToIds.DETECT_SCORE_LIGHTS.value] = (
            DetectScoreLightsWidget(self.match_context)
        )
        self.tasks_to_widgets[TasksToIds.GENERATE_MOMENTUM_GRAPH.value] = (
            GenerateMomentumGraphWidget(self.match_context)
        )
        self.tasks_to_widgets[TasksToIds.GENERATE_HEAT_MAP.value] = (
            GenerateHeatMapWidget(self.match_context)
        )
        self.tasks_to_widgets[TasksToIds.TRACK_POSES.value] = TrackPosesWidget(
            self.match_context
        )
        self.tasks_to_widgets[TasksToIds.TRACK_FENCERS.value] = TrackFencersWidget(
            self.match_context
        )

        for task_id, widget in self.tasks_to_widgets.items():
            self.stack.addWidget(widget)
            widget.run_completed.connect(
                lambda tid=task_id: self.task_view.mark_finished(tid)
            )
            widget.run_started.connect(lambda tid=task_id: self.task_view.rerun(tid))

    def _update_navbar_states(self):
        for tid in self.tasks_to_widgets:
            state = self.task_view.state(tid)
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
        if self.task_view.state(task_id) != TaskState.LOCKED:
            self._switch_to(self.tasks_to_widgets[task_id])

    def hideEvent(self, event):
        self._switch_to(
            self.overview_widget
        )  # this is needed here because switching on showEvent runs the showEvent of the
        # widget being switched from, which might cause issues if the widget isn't supposed to be run.
        return super().hideEvent(event)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    match_context = MatchContext()
    widget = HeatMapMainWidget(match_context)
    match_context.set_file("matches_data/sabre_2.mp4")
    widget.show()
    sys.exit(app.exec())
    match_context = MatchContext()
    widget = HeatMapMainWidget(match_context)
    match_context.set_file("matches_data/sabre_2.mp4")
    widget.show()
    sys.exit(app.exec())
