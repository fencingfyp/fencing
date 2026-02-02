from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from src.gui.util.task_graph import TaskState
from src.gui.util.task_graph_view import TASK_STATE_CSS


class TaskGraphNavbar(QWidget):
    back_clicked = Signal()
    overview_clicked = Signal()
    task_clicked = Signal(str)

    def __init__(self, ordered_task_ids: list[str], parent=None):
        super().__init__(parent)

        self.setFixedWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.back_button = QPushButton("← Back")
        self.overview_button = QPushButton("Overview")

        layout.addWidget(self.back_button)
        layout.addWidget(self.overview_button)
        layout.addSpacing(12)

        self.task_buttons: dict[str, QPushButton] = {}

        for index, tid in enumerate(ordered_task_ids, start=1):
            label = self.create_task_label(tid, index)
            btn = QPushButton(label)
            btn.setEnabled(False)  # initial state
            layout.addWidget(btn)

            self.task_buttons[tid] = btn
            btn.clicked.connect(lambda _, t=tid: self.task_clicked.emit(t))

        layout.addStretch()

        self.back_button.clicked.connect(self.back_clicked)
        self.overview_button.clicked.connect(self.overview_clicked)

    def create_task_label(self, tid: str, index: int) -> QPushButton:
        return f"{index}. {tid.replace('_', ' ').title()}"

    def update_task_state(self, tid: str, state: TaskState):
        btn = self.task_buttons[tid]

        if state == TaskState.LOCKED:
            btn.setEnabled(False)
            btn.setStyleSheet(TASK_STATE_CSS["LOCKED"])
        elif state == TaskState.READY:
            btn.setEnabled(True)
            btn.setStyleSheet(TASK_STATE_CSS["READY"])
        else:
            btn.setEnabled(True)
            btn.setStyleSheet(TASK_STATE_CSS["DONE"])
