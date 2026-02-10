from pathlib import Path

from PySide6.QtCore import QObject, Signal

from src.model.FileManager import FileManager


class MatchContext(QObject):
    match_changed = Signal()

    def __init__(self):
        super().__init__()
        self.file_manager: FileManager | None = None
        self.match_name: str | None = None

    def set_file(self, video_file_path: str):
        self.file_manager = FileManager(video_file_path)
        self.match_name = self.file_manager.get_match_name()
        self.match_changed.emit()
