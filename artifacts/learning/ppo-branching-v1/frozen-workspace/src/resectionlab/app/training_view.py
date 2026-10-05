"""Compact native inspection of actual optimization and checkpoint selection."""

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget, QTextBrowser


class SelectionCurve(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.setMinimumHeight(145)

    def set_history(self, history):
        self.points = [(point["gradient_steps"], point["mean_return"]) for point in history]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b1215"))
        painter.setPen(QColor("#899d9f"))
        painter.drawText(10, 20, "CHECKPOINT SELECTION RETURN · surrogate units")
        if not self.points:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Selection measurements appear after evaluation")
            return
        xs, ys = zip(*self.points)
        xmin, xmax = min(xs), max(max(xs), min(xs) + 1)
        ymin, ymax = min(ys), max(ys)
        if ymax == ymin:
            ymin, ymax = ymin - 1, ymax + 1
        area = QRectF(45, 36, self.width() - 65, self.height() - 65)
        painter.setPen(QPen(QColor("#2b4245"), 1))
        painter.drawLine(int(area.left()), int(area.bottom()), int(area.right()), int(area.bottom()))
        painter.drawText(4, int(area.top()) + 5, f"{ymax:.1f}")
        painter.drawText(4, int(area.bottom()), f"{ymin:.1f}")
        path = QPainterPath()
        points = []
        for index, (x, y) in enumerate(self.points):
            px = area.left() + (x - xmin) / (xmax - xmin) * area.width()
            py = area.bottom() - (y - ymin) / (ymax - ymin) * area.height()
            path.moveTo(px, py) if index == 0 else path.lineTo(px, py)
            points.append((px, py))
        painter.setPen(QPen(QColor("#87c5b5"), 2))
        painter.drawPath(path)
        for x, y in points:
            painter.drawEllipse(QRectF(x - 2, y - 2, 4, 4))
        painter.setPen(QColor("#899d9f"))
        painter.drawText(int(area.left()), self.height() - 8, f"{xmin} updates")
        painter.drawText(int(area.right()) - 70, self.height() - 8, f"{max(xs)} updates")


class TrainingDialog(QDialog):
    start_requested = Signal(float)
    resume_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Refine for this case · coarse research simulation")
        self.resize(560, 580)
        layout = QVBoxLayout(self)
        title = QLabel("Patient-specific policy refinement")
        title.setObjectName("title")
        layout.addWidget(title)
        info = QLabel("Actual gradient updates inside a frozen, coarse structural simulation. Initial search routes remain available. Function and vessels are unassessed; this deterministic scenario is not uncertainty validation.")
        info.setWordWrap(True)
        info.setObjectName("muted")
        layout.addWidget(info)
        controls = QHBoxLayout()
        self.budget = QComboBox()
        for seconds in (30, 60, 120):
            self.budget.addItem(f"{seconds} s training budget", float(seconds))
        self.budget.setCurrentIndex(1)
        controls.addWidget(self.budget)
        self.start_button = QPushButton("Start scratch training")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self.budget.currentData()))
        controls.addWidget(self.start_button)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.resume_button.setEnabled(False)
        controls.addWidget(self.resume_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)
        self.counts = QLabel("0 gradient updates · No training run yet")
        self.counts.setWordWrap(True)
        layout.addWidget(self.counts)
        self.curve = SelectionCurve()
        layout.addWidget(self.curve)
        self.details = QTextBrowser()
        self.details.setPlainText("Optimization and selection use disjoint seed manifests. Final-evaluation worlds remain untouched. Coarse-cell replay is shown only when the independent sequence checker accepts it; native-resolution removal certification remains outstanding.")
        layout.addWidget(self.details, 1)

    def set_running(self, running):
        self.start_button.setEnabled(not running)
        self.resume_button.setEnabled(False)
        self.budget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def set_report(self, report):
        self.counts.setText(f"{report.get('gradient_steps', 0)} gradient updates · {report.get('optimization_environment_steps', 0)} optimization transitions\n"
                           f"{report.get('selection_environment_steps', 0)} selection transitions · {report.get('elapsed_seconds', 0):.1f} s · {report.get('status', 'preparing')}")
        self.curve.set_history(report.get("selection_history", []))
        initial = report.get("initial_selection_return")
        selected = report.get("selected_selection_return")
        values = "Pending selection" if initial is None or selected is None else f"Initial selection return: {initial:.3f}\nSelected checkpoint return: {selected:.3f}"
        self.details.setPlainText(values + "\n\nMode: PATIENT_SCRATCH_RL\nCheckpoint selection uses surrogate returns. An unchanged or worse learned policy is retained as a measured negative result.\n\n"
                                 + f"Checkpoint: {report.get('selected_checkpoint_hash', 'pending')}\nOutput: {report.get('output_dir', 'pending')}\n\nFinal evaluation is not read by this view.")
        replay = report.get("replay") or {}
        native = replay.get("native_removal_audit") or {}
        if native and not native.get("feasible"):
            self.details.append("\nNATIVE REMOVAL AUDIT: REJECTED\n" + ", ".join(native.get("failures", ()))
                                + "\nThe coarse footprint is not supported at source resolution. Removal/residual replay is withheld; this remains a development experiment.")
        self.resume_button.setEnabled(report.get("status") == "cancelled")
