from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QPushButton, QWidget

from src.gui.util.io import load_ui_dynamic
from src.gui.util.task_graph import MomentumGraphTasksToIds, TaskGraph, TaskState

BUTTON_CSS_DONE = """
QPushButton {
    background-color: #2ecc71;
    color: black;
    border: none;
}
QPushButton:hover {
    background-color: #27ae60;
}
QPushButton:pressed {
    background-color: #1e8449;
}
"""
BUTTON_CSS_READY = """
QPushButton {
    background-color: #f1c40f;
    color: black;
    border: none;
}
QPushButton:hover {
    background-color: #d4ac0d;
}
QPushButton:pressed {
    background-color: #b7950b;
}
"""
BUTTON_CSS_LOCKED = """
QPushButton {
    background-color: #7f8c8d;
    color: black;
    border: none;
}
"""


class MomentumGraphMenuWidget(QWidget):
    navigate_to_manage_match = Signal()
    navigate_to_task = Signal(str)

    def __init__(self, task_graph: TaskGraph, parent=None):
        super().__init__(parent)

        file_name = "src/gui/momentum_graph/momentum_graph_menu_widget.ui"
        ui = load_ui_dynamic(file_name, self)
        self.task_buttons = self.build_task_button_mapping(ui)
        self.back_button = ui.findChild(QPushButton, "backButton")

        self.task_graph = task_graph

        for task_id, button in self.task_buttons.items():
            button.clicked.connect(
                lambda _, tid=task_id: self.navigate_to_task.emit(tid)
            )

        self.back_button.clicked.connect(lambda: self.navigate_to_manage_match.emit())

        # Update button states based on task graph
        self.update_button_states()
        self.task_graph.graph_changed.connect(self.update_button_states)

    @staticmethod
    def build_task_button_mapping(ui) -> dict[str, QPushButton]:
        task_buttons = {}

        task_buttons[MomentumGraphTasksToIds.CROP_SCOREBOARD.value] = ui.findChild(
            QPushButton, "cropScoreboardButton"
        )
        task_buttons[MomentumGraphTasksToIds.CROP_SCORE_LIGHTS.value] = ui.findChild(
            QPushButton, "cropScoreLightsButton"
        )
        task_buttons[MomentumGraphTasksToIds.PERFORM_OCR.value] = ui.findChild(
            QPushButton, "performOcrButton"
        )
        task_buttons[MomentumGraphTasksToIds.DETECT_SCORE_LIGHTS.value] = ui.findChild(
            QPushButton, "detectScoreLightsButton"
        )
        task_buttons[MomentumGraphTasksToIds.GENERATE_MOMENTUM_GRAPH.value] = (
            ui.findChild(QPushButton, "generateMomentumGraphButton")
        )
        return task_buttons

    @Slot()
    def update_button_states(self):
        for task_id, button in self.task_buttons.items():
            state = self.task_graph.state(task_id)
            if state == TaskState.LOCKED:
                button.setEnabled(False)
                button.setStyleSheet(BUTTON_CSS_LOCKED)
                # button.setText(f"{task_id.name.replace('_', ' ').title()} (Locked)")
            elif state == TaskState.READY:
                button.setEnabled(True)
                button.setStyleSheet(BUTTON_CSS_READY)
                # button.setText(f"{task_id.name.replace('_', ' ').title()} (Ready)")
            else:
                button.setEnabled(True)
                button.setStyleSheet(BUTTON_CSS_DONE)
                # button.setText(f"{task_id.name.replace('_', ' ').title()} (Done)")
