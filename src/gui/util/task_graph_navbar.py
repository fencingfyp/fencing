from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from src.gui.util.task_graph import TaskState
from src.gui.util.task_graph_view import TASK_STATE_CSS


class TaskGraphLocalNav(QWidget):
    back_requested = Signal()
    overview_requested = Signal()
    task_requested = Signal(str)

    def __init__(self, ordered_task_ids: list[str], parent=None):
        super().__init__(parent)

        self._task_buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # self.back_button = QPushButton("← Back")
        self.overview_button = QPushButton("Overview")

        # layout.addWidget(self.back_button)
        layout.addWidget(self.overview_button)
        layout.addSpacing(12)

        # ---- task buttons ----
        for index, tid in enumerate(ordered_task_ids, start=1):
            btn = QPushButton(self._task_label(tid, index))
            btn.setEnabled(False)
            layout.addWidget(btn)

            btn.clicked.connect(lambda _, t=tid: self.task_requested.emit(t))
            self._task_buttons[tid] = btn

        layout.addStretch()

        # ---- signals ----
        # self.back_button.clicked.connect(self.back_requested)
        self.overview_button.clicked.connect(self.overview_requested)

    # ---------- public API ----------

    def update_task_state(self, task_id: str, state: TaskState):
        btn = self._task_buttons.get(task_id)
        if not btn:
            return

        if state == TaskState.LOCKED:
            btn.setEnabled(False)
            btn.setStyleSheet(TASK_STATE_CSS["LOCKED"])
        elif state == TaskState.READY:
            btn.setEnabled(True)
            btn.setStyleSheet(TASK_STATE_CSS["READY"])
        elif state == TaskState.DONE:
            btn.setEnabled(True)
            btn.setStyleSheet(TASK_STATE_CSS["DONE"])

    # ---------- helpers ----------

    @staticmethod
    def _task_label(task_id: str, index: int) -> str:
        return f"{index}. {task_id.replace('_', ' ').title()}"
