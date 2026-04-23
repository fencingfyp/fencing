from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton

from src.gui.util.tag import Tag
from src.gui.util.video_player_widget import VideoPlayerWidget

from .add_tag_dialog import AddTagDialog


class TaggableVideoPlayerWidget(VideoPlayerWidget):

    tag_requested = Signal(Tag)

    def _build_controls(self) -> QHBoxLayout:
        hbox = super()._build_controls()

        self._add_tag_button = QPushButton("Add Tag")
        self._add_tag_button.clicked.connect(self._on_add_tag)
        hbox.insertWidget(1, self._add_tag_button)  # after Play, before stretch

        return hbox

    def _on_add_tag(self):
        time_msec = int(self.get_current_time_msec())
        frame_idx = int(self.get_current_frame_number())
        if time_msec is None:
            return

        dialog = AddTagDialog(time_msec, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        tag = dialog.result_tag(frame_idx, time_msec)
        print(f"Requesting tag at frame {frame_idx}: {tag}")
        self.tag_requested.emit(tag)
