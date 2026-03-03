from pathlib import Path

from PySide6.QtCore import QObject, Signal

from src.gui.task_dependencies import TASK_DEPENDENCIES
from src.gui.util.task_graph import TaskGraph
from src.model.FileManager import FileManager


class MatchContext(QObject):
    match_changed = Signal()

    def __init__(self):
        super().__init__()
        self.file_manager: FileManager | None = None
        self.match_name: str | None = None
        self.task_graph = TaskGraph(TASK_DEPENDENCIES, None)

    def set_file(self, video_file_path: str):
        self.file_manager = FileManager(video_file_path)
        self.match_name = self.file_manager.get_match_name()
        self.task_graph.file_manager = self.file_manager
        self.task_graph._on_match_changed()
        self.match_changed.emit()
