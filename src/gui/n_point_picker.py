from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QKeySequence, QShortcut

from src.model.drawable.labelled_points_drawable import LabelledPointsDrawable

LABELS = ["TL", "TR", "BR", "BL"]


class NPointPicker(QObject):
    """
    Picks N points on a video frame using VideoRenderer.
    """

    def __init__(self, renderer, text_label, prompts: list[str], on_done):
        """
        renderer: VideoRenderer instance
        text_label: QLabel to show prompts
        prompts: list of prompt strings per point
        on_done: callback(points_list)
        """
        super().__init__(renderer.video_label)
        self.renderer = renderer
        self.text_label = text_label
        self.prompts = prompts
        self.on_done = on_done

        self.current_idx = 0
        self.picked_points: list[tuple[float, float]] = []
        self.current_click: tuple[float, float] | None = None

        self._active = False
        self._shortcut: QShortcut | None = None

        self.activate()

    # ----------------------- Lifecycle -----------------------
    def activate(self):
        if self._active:
            return
        self._active = True

        # connect to renderer's mouse click signal
        self.renderer.mouse_clicked.connect(self._on_mouse_click)

        # shortcut for confirming points
        self._shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_W), self.renderer.video_label
        )
        self._shortcut.activated.connect(self.confirm_point)

        self.current_idx = 0
        self.picked_points.clear()
        self.current_click = None
        self._update_prompt()
        self._redraw_preview()

    def deactivate(self):
        if not self._active:
            return
        self._active = False

        # disconnect signal
        self.renderer.mouse_clicked.disconnect(self._on_mouse_click)

        if self._shortcut:
            self._shortcut.activated.disconnect(self.confirm_point)
            self._shortcut.setParent(None)
            self._shortcut = None

        self.text_label.setText("")

    # ----------------------- Mouse -----------------------
    def _on_mouse_click(self, fx: float, fy: float):
        """Handle frame coordinates emitted by VideoRenderer."""
        self.current_click = (fx, fy)
        self._redraw_preview()

    # ----------------------- Confirm -----------------------
    def confirm_point(self):
        if self.current_click is None:
            return

        self.picked_points.append(self.current_click)
        self.current_click = None
        self.current_idx += 1

        if self.current_idx >= len(self.prompts):
            self.finish()
        else:
            self._update_prompt()
            self._redraw_preview()

    # ----------------------- Helpers -----------------------
    def _update_prompt(self):
        self.text_label.setText(self.prompts[self.current_idx])

    def _redraw_preview(self):
        # draw picked points + current click
        points = self.picked_points.copy()
        if self.current_click:
            points.append(self.current_click)

        self.renderer.render(
            [
                LabelledPointsDrawable(
                    points, labels=LABELS[: len(points)], color=(255, 0, 0), size=5
                )
            ]
        )

    def finish(self):
        self.deactivate()
        self.on_done(self.picked_points)
