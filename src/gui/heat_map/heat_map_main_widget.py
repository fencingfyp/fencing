from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QApplication, QHBoxLayout, QStackedWidget, QWidget

from src.gui.navbar.app_navigator import AppNavigator, View
from src.gui.util.task_graph import HeatMapTasksToIds, Task, TaskGraph, TaskState
from src.gui.util.task_graph_navbar import TaskGraphLocalNav
from src.pyside.MatchContext import MatchContext
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

        self.overview_widget = HeatMapOverviewWidget(self.task_graph)
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
        self.tasks_to_widgets[HeatMapTasksToIds.TRACK_POSES.value] = TrackPosesWidget(
            self.match_context
        )
        self.tasks_to_widgets[HeatMapTasksToIds.TRACK_FENCERS.value] = (
            TrackFencersWidget(self.match_context)
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


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    match_context = MatchContext()
    widget = HeatMapMainWidget(match_context)
    match_context.set_file("matches_data/sabre_2.mp4")
    widget.show()
    sys.exit(app.exec())
