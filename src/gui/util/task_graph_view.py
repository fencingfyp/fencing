import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .task_graph import GraphLayout

TASK_STATE_CSS = {
    "DONE": """
        QPushButton {
            background-color: #2ecc71;
            color: black;
            border: none;
            text-align: left;
            padding: 6px;
        }
        QPushButton:hover { background-color: #27ae60; }
        QPushButton:pressed { background-color: #1e8449; }
    """,
    "READY": """
        QPushButton {
            background-color: #f1c40f;
            color: black;
            border: none;
            text-align: left;
            padding: 6px;
        }
        QPushButton:hover { background-color: #d4ac0d; }
        QPushButton:pressed { background-color: #b7950b; }
    """,
    "LOCKED": """
        QPushButton {
            background-color: #7f8c8d;
            color: black;
            border: none;
            text-align: left;
            padding: 6px;
        }
    """,
}


# -------------------------------
# Node Widget
# -------------------------------
class TaskNodeWidget(QWidget):
    STATE_COLORS = {
        "DONE": {
            "base": "#2ecc71",
            "hover": "#27ae60",
            "pressed": "#1e8449",
            "border": "#27ae60",
        },
        "READY": {
            "base": "#f1c40f",
            "hover": "#d4ac0d",
            "pressed": "#b7950b",
            "border": "#d4ac0d",
        },
        "LOCKED": {
            "base": "#7f8c8d",
            "hover": "#7f8c8d",
            "pressed": "#7f8c8d",
            "border": "#636e72",
        },
    }

    def __init__(self, tid: str, state: str):
        super().__init__()
        self.tid = tid
        self.state = state
        self.hovered = False
        self.pressed = False

        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self.title_label = QLabel(tid)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        palette = self.title_label.palette()
        palette.setColor(QPalette.WindowText, QColor("black"))
        self.title_label.setPalette(palette)

        self.state_label = QLabel(state)
        self.state_label.setAlignment(Qt.AlignCenter)
        palette = self.state_label.palette()
        palette.setColor(QPalette.WindowText, QColor("black"))
        self.state_label.setPalette(palette)

        layout.addWidget(self.title_label)
        layout.addWidget(self.state_label)

        self.setMinimumSize(120, 60)

    def enterEvent(self, event):
        self.hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressed = False
            self.update()
        super().mouseReleaseEvent(event)

    def update_state(self, state: str):
        self.state = state
        self.state_label.setText(state)
        self.update()  # trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        colors = self.STATE_COLORS.get(self.state, self.STATE_COLORS["LOCKED"])
        bg_color = colors["base"]
        if self.pressed:
            bg_color = colors["pressed"]
        elif self.hovered:
            bg_color = colors["hover"]

        # draw background
        painter.setBrush(QColor(bg_color))
        painter.setPen(QPen(QColor(colors["border"])))
        painter.drawRoundedRect(self.rect(), 6, 6)

        super().paintEvent(event)


# -------------------------------
# Proxy wrapper for scene
# -------------------------------
class TaskNodeItem(QGraphicsProxyWidget):
    def __init__(self, widget: QWidget):
        super().__init__()
        self.setWidget(widget)


# -------------------------------
# Edge
# -------------------------------
class DependencyEdge(QGraphicsPathItem):
    def __init__(self, src: TaskNodeItem, dst: TaskNodeItem):
        super().__init__()

        src_rect = src.sceneBoundingRect()
        dst_rect = dst.sceneBoundingRect()

        # Right-middle of source → left-middle of target
        p1 = QPointF(src_rect.right(), src_rect.center().y())
        p2 = QPointF(dst_rect.left(), dst_rect.center().y())

        dx = (p2.x() - p1.x()) * 0.5
        path = QPainterPath(p1)
        path.cubicTo(
            QPointF(p1.x() + dx, p1.y()),
            QPointF(p2.x() - dx, p2.y()),
            p2,
        )
        self.setPath(path)
        self.setPen(QPen(Qt.gray, 2))
        self.setZValue(-1)  # behind nodes
        self.setAcceptedMouseButtons(Qt.NoButton)


# -------------------------------
# Scene & Renderer
# -------------------------------
class TaskGraphScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.node_items: dict[str, TaskNodeItem] = {}
        self.edges: list[DependencyEdge] = []

    def render(
        self, layout: GraphLayout, states: dict[str, str], index_map: dict[str, int]
    ):
        """Build nodes, edges, and positions once."""
        self.clear()
        self.node_items.clear()
        self.edges.clear()

        # Build DAG for layout
        G = nx.DiGraph()
        for src, dst in layout.edges:
            G.add_edge(src, dst)

        # Compute positions via Graphviz (left→right)
        pos = graphviz_layout(G, prog="dot", args="-Grankdir=LR")

        # Scale & offset positions
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, min_y = min(xs), min(ys)
        x_scale, y_scale = 1, 2  # magic numbers to ensure the boxes dont overlap
        x_offset, y_offset = -min_x * x_scale + 50, -min_y * y_scale + 50

        # --- create nodes ---
        for tid, (gx, gy) in pos.items():
            number = index_map.get(tid, "?")
            widget = TaskNodeWidget(f"{number}. {tid}", states.get(tid, "UNKNOWN"))
            item = TaskNodeItem(widget)
            item.setPos(gx * x_scale + x_offset, gy * y_scale + y_offset)
            self.addItem(item)
            self.node_items[tid] = item

        # --- create edges ---
        for src, dst in layout.edges:
            edge = DependencyEdge(self.node_items[src], self.node_items[dst])
            self.addItem(edge)
            self.edges.append(edge)

    def update_nodes(self, states: dict[str, str]):
        """Update the contents of each node widget based on new states."""
        for tid, item in self.node_items.items():
            widget = item.widget()
            if widget:
                widget.update_state(states.get(tid, "UNKNOWN"))


# -------------------------------
# View
# -------------------------------
class TaskGraphView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)  # avoid hand cursor
