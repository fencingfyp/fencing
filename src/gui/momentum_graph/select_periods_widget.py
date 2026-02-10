import json
from dataclasses import dataclass
from typing import List, Optional, override

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.gui.video_player_widget import VideoPlayerWidget
from src.model.FileManager import FileRole
from src.pyside.MatchContext import MatchContext

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

        if len(self.periods) >= self.max_periods:
            self.finish()

    def finish(self):
        if self.on_finish:
            self.on_finish()

    def cancel(self):
        self.periods = []
        self.current_start_ms = None


class SelectPeriodsWidget(BaseTaskWidget):
    """UI wrapper for periods selection."""

    def __init__(self, match_context, parent=None, *, max_periods: int = 3):
        super().__init__(match_context, parent)
        self.max_periods = max_periods
        self.save_path: Optional[str] = None
        self.controller: Optional[PeriodSelectorController] = None

        self._content = QWidget(self)
        layout = QVBoxLayout(self._content)

        self.player = VideoPlayerWidget(self._content)
        layout.addWidget(self.player)

        self.info = QLabel(self._content)
        layout.addWidget(self.info)

        self.finish_button = QPushButton("Finish selection", self._content)
        self.finish_button.clicked.connect(self._finish)
        layout.addWidget(self.finish_button)
        self.register_widget(self._content)

    @override
    def setup(self):
        if not self.match_context.file_manager.get_working_directory():
            return
        self.is_running = True

        video_path = self.match_context.file_manager.get_original_video()
        self.save_path = self.match_context.file_manager.get_path(FileRole.PERIODS)

        self.player.set_video_source(video_path)

        self.controller = PeriodSelectorController(self.player, self.max_periods)
        self.controller.on_finish = self._finish

        self.show_widget(self._content)
        self.activate()
        self._update_info()

    @override
    def cancel(self):
        self.deactivate()
        super().cancel()

    # ------------------- lifecycle -------------------
    def activate(self):
        """Activate video player and shortcuts."""
        self.player.play()
        self.finish_button.setEnabled(False)

        self.player.register_shortcut(Qt.Key.Key_S, self._mark_start)
        self.player.register_shortcut(Qt.Key.Key_E, self._mark_end)

    def deactivate(self):
        """Deactivate video player and shortcuts."""
        self.finish_button.setEnabled(False)
        self.player.pause()

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
        if not self.controller or not self.controller.periods:
            return

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
        if len(self.controller.periods) == self.max_periods:
            self.info.setText(f"All {self.max_periods} periods selected.")
            return
        if self.controller.current_start_ms is None:
            self.info.setText(
                f"Periods: {len(self.controller.periods)}/{self.max_periods}\nS = mark start"
            )
            return
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

    from PySide6.QtWidgets import QApplication

    def main():
        app = QApplication(sys.argv)
        match_context = MatchContext()
        widget = SelectPeriodsWidget(match_context)
        match_context.set_file("matches_data/epee_3/epee_3.mp4")
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
