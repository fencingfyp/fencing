import json
import os
from dataclasses import dataclass
from typing import List, Optional, override

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.gui.video_player_widget import VideoPlayerWidget
from src.util.file_names import ORIGINAL_VIDEO_NAME, PERIODS_JSON_NAME

from .base_task_widget import BaseTaskWidget


@dataclass
class Period:
    start_ms: int
    end_ms: int


class PeriodSelectorController:
    """Handles period marking logic for a video player."""

    def __init__(self, video_player: VideoPlayerWidget, max_periods: int):
        self.player = video_player
        self.max_periods = max_periods
        self.periods: List[Period] = []
        self.current_start_ms: Optional[int] = None

        # Finish callback
        self.on_finish: Optional[callable] = None

    def mark_start(self):
        if self.current_start_ms is not None:
            return
        self.current_start_ms = self.player.video_frame.get_current_time_msec()

    def mark_end(self):
        if self.current_start_ms is None:
            return
        end_ms = self.player.video_frame.get_current_time_msec()
        if end_ms <= self.current_start_ms:
            return
        self.periods.append(Period(self.current_start_ms, end_ms))
        self.current_start_ms = None

        # Auto-finish if max periods reached
        if len(self.periods) >= self.max_periods:
            self.finish()

    def finish(self):
        if self.on_finish:
            self.on_finish()


class SelectPeriodsWidget(BaseTaskWidget):
    """UI wrapper for periods selection."""

    def __init__(self, parent=None, *, max_periods: int = 3):
        super().__init__(parent)
        self.max_periods = max_periods
        self.save_path: Optional[str] = None
        self.player = VideoPlayerWidget(self)

        # Hide unused base task UI
        self.ui.runButton.hide()
        self.ui.uiTextLabel.hide()
        self.ui.videoLabel.hide()

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.player)

        self.info = QLabel(self)
        layout.addWidget(self.info)

        self.finish_button = QPushButton("Finish selection", self)
        self.finish_button.clicked.connect(self._finish)
        layout.addWidget(self.finish_button)
        self.setLayout(layout)

        self.controller = None

    @override
    def setup(self):
        self.is_running = True
        video_path = os.path.join(self.working_dir, ORIGINAL_VIDEO_NAME)
        self.save_path = os.path.join(self.working_dir, PERIODS_JSON_NAME)

        self.player.set_video_source(video_path)

        self.controller = PeriodSelectorController(self.player, self.max_periods)
        self.controller.on_finish = self._finish

        self.activate()
        self._update_info()

    @override
    def cancel(self):
        self.deactivate()

    # ------------------- lifecycle -------------------
    def activate(self):
        """Activate video player and shortcuts."""
        self.player.activate()
        self.finish_button.setEnabled(False)

        self.player.register_shortcut(Qt.Key.Key_S, self._mark_start)
        self.player.register_shortcut(Qt.Key.Key_E, self._mark_end)

    def deactivate(self):
        """Deactivate video player and shortcuts."""
        self.player.deactivate()

    # ------------------- controller wrappers -------------------
    def _mark_start(self):
        self.controller.mark_start()
        self._update_info()
        self._update_finish_button_state()

    def _mark_end(self):
        self.controller.mark_end()
        self._update_info()
        self._update_finish_button_state()

    def _finish(self):
        if not self.controller.periods:
            return
        # Save periods
        with open(self.save_path, "w") as f:
            json.dump(
                [
                    {"start_ms": p.start_ms, "end_ms": p.end_ms}
                    for p in self.controller.periods
                ],
                f,
                indent=2,
            )

        self.run_completed.emit(MomentumGraphTasksToIds.SELECT_PERIODS)
        self.finish_button.setEnabled(False)
        self.info.setText("Selection finished.")
        self.is_running = False
        self.deactivate()

    # ------------------- UI helpers -------------------
    def _update_info(self):
        if self.controller.current_start_ms is None:
            self.info.setText(
                f"Periods: {len(self.controller.periods)}/{self.max_periods}\nS = mark start"
            )
        else:
            self.info.setText(
                f"Periods: {len(self.controller.periods)}/{self.max_periods}\nE = mark end"
            )

    def _update_finish_button_state(self):
        self.finish_button.setEnabled(
            self.is_running and len(self.controller.periods) > 0
        )


if __name__ == "__main__":
    import cProfile
    import pstats
    import sys

    from PySide6.QtWidgets import QApplication, QWidget

    def main():
        app = QApplication(sys.argv)
        widget = SelectPeriodsWidget()
        widget.set_working_directory("matches_data/sabre_3")
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
