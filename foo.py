import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from src.model import Quadrilateral
from src.model.drawable import DetectionsDrawable, PointsDrawable, QuadrilateralDrawable
from src.pyside.video_renderer import VideoRenderer

app = QApplication(sys.argv)

# Main window
win = QWidget()
layout = QVBoxLayout(win)
video_label = QLabel()
video_label.setFixedSize(640, 480)
layout.addWidget(video_label)
win.show()

renderer = VideoRenderer(video_label)

# Dummy frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Example drawables
quad = Quadrilateral([[0, 0], [100, 0], [100, 100], [0, 100]])
quad_drawable = QuadrilateralDrawable(quad, color=(0, 255, 255))

detections = {
    1: {"box": [300, 150, 400, 250], "id": 1},
    2: {"box": [100, 300, 200, 400], "id": 2},
}
detections_drawable = DetectionsDrawable(detections, highlight_id=2)
points_drawable = PointsDrawable(
    points=[(320, 240), (330, 250), (340, 260)], color=(255, 0, 255)
)


# Render a single frame with points and drawables
def render_frame():
    renderer.set_frame(frame)
    # Use the existing adapter pattern
    renderer.render([quad_drawable, detections_drawable, points_drawable])


# Simple timer to refresh the frame
timer = QTimer()
timer.timeout.connect(render_frame)
timer.start(1000 // 30)  # 30 FPS

sys.exit(app.exec())
