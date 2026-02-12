from PySide6.QtCore import QObject, Qt

from src.gui.util.actions_panel_widget import ActionsPanelWidget, TaskAction
from src.model.drawable.labelled_points_drawable import LabelledPointsDrawable

LABELS = ["TL", "TR", "BR", "BL"]


class NPointPicker(QObject):
    """
    Picks N points on a video frame using VideoRenderer.
    """

    def __init__(
        self,
        renderer,
        text_label,
        action_panel: ActionsPanelWidget,
        prompts: list[str],
        on_done,
    ):
        """
        renderer: VideoRenderer instance
        text_label: QLabel to show prompts
        action_panel: ActionPanelWidget instance
        prompts: list of prompt strings per point
        on_done: callback(points_list)
        """
        super().__init__(renderer.video_label)

        self.renderer = renderer
        self.text_label = text_label
        self.action_panel = action_panel
        self.prompts = prompts
        self.on_done = on_done

        self.current_idx = 0
        self.picked_points: list[tuple[float, float]] = []
        self.current_click: tuple[float, float] | None = None

        self._active = False

        self.activate()

    # ----------------------- Lifecycle -----------------------

    def activate(self):
        if self._active:
            return

        self._active = True

        self.renderer.mouse_clicked.connect(self._on_mouse_click)

        self.current_idx = 0
        self.picked_points.clear()
        self.current_click = None

        self._update_prompt()
        self._redraw_preview()
        self._update_actions()

    def deactivate(self):
        if not self._active:
            return

        self._active = False

        self.renderer.mouse_clicked.disconnect(self._on_mouse_click)

        self.action_panel.clear()
        self.text_label.setText("")

    # ----------------------- Mouse -----------------------

    def _on_mouse_click(self, fx: float, fy: float):
        self.current_click = (fx, fy)
        self._redraw_preview()
        self._update_actions()

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
            self._update_actions()

    # ----------------------- UI Updates -----------------------

    def _update_prompt(self):
        self.text_label.setText(self.prompts[self.current_idx])

    def _update_actions(self):
        if not self._active:
            return

        actions = []

        # Confirm only enabled if user has clicked
        actions.append(
            TaskAction(
                id="confirm",
                label="Confirm (W)",
                shortcut=Qt.Key.Key_W,
                callback=self.confirm_point,
                enabled=self.current_click is not None,
            )
        )

        self.action_panel.set_actions(actions)

    # ----------------------- Rendering -----------------------

    def _redraw_preview(self):
        points = self.picked_points.copy()
        if self.current_click:
            points.append(self.current_click)

        self.renderer.render(
            [
                LabelledPointsDrawable(
                    points,
                    labels=LABELS[: len(points)],
                    color=(255, 0, 0),
                    size=5,
                )
            ]
        )

    # ----------------------- Finish -----------------------

    def finish(self):
        self.deactivate()
        self.on_done(self.picked_points)
