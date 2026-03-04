from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget


class InstructionLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Overlay
        self._highlight = QWidget(self)
        self._highlight.setStyleSheet(
            """
            background-color: rgba(255, 255, 0, 180);
            border-radius: 4px;
        """
        )
        self._highlight.hide()

        # Opacity effect
        self._effect = QGraphicsOpacityEffect(self._highlight)
        self._highlight.setGraphicsEffect(self._effect)

        # Animation
        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setDuration(1200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._highlight.hide)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._highlight.setGeometry(self.rect())

    def setText(self, text, silent=False):
        super().setText(text)
        if not silent:
            self.animate_highlight()

    def animate_highlight(self):
        self._highlight.show()
        self._effect.setOpacity(1.0)

        self._anim.stop()
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.start()
