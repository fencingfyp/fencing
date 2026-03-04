from abc import ABC

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from src.gui.base_task_widget.actions_panel_widget import ActionsPanelWidget
from src.gui.util.conversion import np_to_pixmap
from src.model.Quadrilateral import Quadrilateral
from src.model.Ui import Ui

from .fencer_selection_controller import FencerSelectionController
from .loading_overlay import LoadingOverlay
from .n_point_picker import NPointPicker
from .video_renderer import VideoRenderer


class ABCQObjectMeta(type(QObject), type(ABC)):
    """This allows using PySide6 as the implementation for the UI abstraction defined in Ui.py."""

    pass


class PysideUi(QObject, Ui, metaclass=ABCQObjectMeta):
    task_completed = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        video_label: QLabel,
        action_panel: ActionsPanelWidget | None,
        text_label: QLabel | None,
        parent: QObject,
    ):
        super().__init__(parent)
        self.video_label = video_label
        self.action_panel = action_panel
        self.text_label = text_label
        self.video_renderer = VideoRenderer(video_label)
        self._loading_overlay = LoadingOverlay(video_label)

        self._additional_windows: dict[int | str, QWidget] = {}

        self.timer = QTimer(self)
        QApplication.instance().aboutToQuit.connect(self.close_additional_windows)
        self.fps = None

    def initialise(self, fps: float):
        self.fps = fps

    # ------------------------------------------------------------------
    # Rendering API (matches OpenCvUi intent)
    # ------------------------------------------------------------------

    def write(self, text: str, silent=False):
        if self.text_label:
            self.text_label.setText(text, silent=silent)

    def set_fresh_frame(self, frame: np.ndarray) -> QPixmap:
        return self.video_renderer.set_frame(frame)

    def plot_points(self, pts, color=None):
        self.video_renderer.render_points(pts, color=color)

    def get_current_frame(self) -> np.ndarray | None:
        return self.video_renderer.get_current_frame()

    def show_text(self, text: str):
        if self.text_label:
            self.text_label.setText(text)

    def show_frame(self):
        pass

    def show_additional(self, key: int | str, frame: np.ndarray):
        """
        Show the given frame in a separate top-level window.
        Each index corresponds to a separate window.
        """

        pixmap = np_to_pixmap(frame)

        # Create window if it doesn't exist
        if key not in self._additional_windows:
            win = QWidget(None)  # top-level window
            win.setWindowTitle(f"Additional View {key}")

            # Set a reasonable size (or match frame)
            win.resize(frame.shape[1], frame.shape[0])

            label = QLabel(win)
            label.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout(win)
            layout.addWidget(label)
            win.setLayout(layout)

            self._additional_windows[key] = win
            win.show()
        else:
            win = self._additional_windows[key]
            label = win.findChild(QLabel)

        # Scale pixmap to label size while preserving aspect ratio
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        label.update()

    def draw_quadrilateral(self, quad: Quadrilateral, color=(0, 255, 0)):
        self.video_renderer.render_quadrilateral(quad, color=color)

    def draw_detections(self, detections: dict, highlight_id=None):
        self.video_renderer.render_detections(detections, highlight_id)

    def draw_objects(self, drawables: list):
        self.video_renderer.render(drawables)

    def show_loading(self, message: str = "Loading...", progress: float | None = None):
        """None progress = spinner, 0.0-1.0 = progress bar."""
        self._loading_overlay.show_loading(message, progress)

    def update_loading(self, progress: float, message: str | None = None):
        if message is not None:
            self._loading_overlay.set_message(message)
        self._loading_overlay.set_progress(progress)

    def hide_loading(self):
        self._loading_overlay.hide_loading()

    # ------------------------------------------------------------------
    # Other interactive API
    # ------------------------------------------------------------------

    def cancel_running_subtasks(self):
        if hasattr(self, "point_picker") and self.point_picker:
            self.point_picker.deactivate()
            self.point_picker = None
        if hasattr(self, "fencer_selector") and self.fencer_selector:
            self.fencer_selector.deactivate()
            self.fencer_selector = None
        if self.action_panel:
            self.action_panel.clear()

    def get_n_points_async(self, frame, prompts: list[str], callback):
        if frame is None:
            return

        self.cancel_running_subtasks()
        self.set_fresh_frame(frame)
        self.point_picker = NPointPicker(
            renderer=self.video_renderer,
            text_label=self.text_label,
            action_panel=self.action_panel,
            prompts=prompts,
            on_done=callback,
        )

    def get_fencer_id(self, candidates: dict[int, dict], left: bool, on_done: callable):
        self.fencer_selector = FencerSelectionController(
            ui=self,
            video_label=self.video_label,
            action_panel=self.action_panel,
            on_done=on_done,
        )
        self.fencer_selector.start(candidates, left)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_additional_windows(self):
        for win in self._additional_windows.values():
            win.close()
        self._additional_windows.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def schedule(self, callback, delay_ms=0):
        QTimer.singleShot(delay_ms, callback)
        QTimer.singleShot(delay_ms, callback)
        QTimer.singleShot(delay_ms, callback)
