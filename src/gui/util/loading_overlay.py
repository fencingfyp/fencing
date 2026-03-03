from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget


class LoadingOverlay(QWidget):
    """
    Semi-transparent loading overlay that sits on top of a QLabel.

    Usage:
        overlay = LoadingOverlay(video_label)
        overlay.show_loading("Initialising model...")    # spinner
        overlay.show_loading("Calibrating…", progress=0.0)  # determinate bar
        overlay.set_progress(0.6)
        overlay.hide_loading()
    """

    def __init__(self, parent: QLabel):
        super().__init__(parent)
        self._bg_color = QColor(10, 10, 18, 195)
        self._bar_bg_color = QColor(40, 40, 60)
        self._bar_fg_color = QColor(80, 200, 140)
        self._text_color = QColor(220, 220, 240)
        self._sub_color = QColor(130, 130, 160)

        self._message = "Loading..."
        self._progress = None  # None = spinner, float = bar
        self._spin_angle = 0.0

        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(16)
        self._spin_timer.timeout.connect(self._tick_spinner)

        self._fit_to_parent()
        self.hide()

    # --- public API ---

    def show_loading(self, message: str = "Loading...", progress=None):
        self._message = message
        self._progress = progress
        self._fit_to_parent()
        self.show()
        self.raise_()
        if progress is None:
            self._spin_timer.start()
        else:
            self._spin_timer.stop()

    def hide_loading(self):
        self._spin_timer.stop()
        self.hide()

    def set_progress(self, value):
        """0.0-1.0 for determinate bar, None to switch back to spinner."""
        self._progress = value
        if value is None:
            self._spin_timer.start()
        else:
            self._spin_timer.stop()
        self.update()

    def set_message(self, message: str):
        self._message = message
        self.update()

    # --- internals ---

    def _fit_to_parent(self):
        if self.parent():
            self.setGeometry(self.parent().rect())

    def _tick_spinner(self):
        self._spin_angle = (self._spin_angle + 4.0) % 360.0
        self.update()

    def resizeEvent(self, event):
        self._fit_to_parent()
        super().resizeEvent(event)

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self._bg_color)

        cx, cy = w // 2, h // 2

        if self._progress is None:
            # indeterminate spinner
            r = max(24.0, min(min(w, h) * 0.10, 56.0))
            thickness = max(3, r * 0.12)
            rect = QRectF(cx - r, cy - r - 28, r * 2, r * 2)

            track_pen = QPen(self._bar_bg_color, thickness)
            track_pen.setCapStyle(Qt.RoundCap)
            p.setPen(track_pen)
            p.drawArc(rect, 0, 360 * 16)

            arc_pen = QPen(self._bar_fg_color, thickness)
            arc_pen.setCapStyle(Qt.RoundCap)
            p.setPen(arc_pen)
            p.drawArc(rect, int(-self._spin_angle * 16), int(250 * 16))
        else:
            # determinate bar
            bar_w = min(w * 0.65, 320.0)
            bar_h = max(6.0, min(h * 0.025, 14.0))
            bar_x = cx - bar_w / 2
            bar_y = cy - bar_h / 2 - 24
            radius = bar_h / 2

            p.setPen(Qt.NoPen)
            p.setBrush(self._bar_bg_color)
            p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), radius, radius)

            fill_w = bar_w * max(0.0, min(1.0, self._progress))
            if fill_w > 0:
                p.setBrush(self._bar_fg_color)
                p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), radius, radius)

            pct_font = QFont("Courier New", max(8, int(bar_h * 0.95)))
            pct_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
            p.setFont(pct_font)
            p.setPen(self._sub_color)
            fm = QFontMetrics(pct_font)
            pct_text = f"{int(self._progress * 100)}%"
            p.drawText(
                int(cx - fm.horizontalAdvance(pct_text) / 2),
                int(bar_y + bar_h + fm.height() + 4),
                pct_text,
            )

        # message
        font_size = max(10, min(int(h * 0.028), 18))
        msg_font = QFont("Courier New", font_size)
        msg_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        p.setFont(msg_font)
        p.setPen(self._text_color)
        fm = QFontMetrics(msg_font)
        msg_y = cy + int(min(w, h) * 0.12) + 10
        p.drawText(
            int(cx - fm.horizontalAdvance(self._message) / 2), msg_y, self._message
        )
        p.end()
