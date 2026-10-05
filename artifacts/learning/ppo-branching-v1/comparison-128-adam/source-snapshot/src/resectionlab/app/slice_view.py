"""Linked physical-coordinate MRI canvas with inspectable source overlays."""

import numpy as np
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .reslice import PLANE_LABELS, sample_slice, slice_geometry, window_to_rgb


class SliceView(QWidget):
    point_selected = Signal(object)
    scrolled = Signal(str, int)

    def __init__(self, plane: str, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.geometry = None
        self.image = None
        self.cursor = None
        self.rgb = None
        self.setMinimumSize(160, 150)
        self.setMouseTracking(True)
        self.setAccessibleName(f"{plane.title()} MRI slice")
        self.setToolTip("Click to link all views at this RAS coordinate. Scroll to move through slices.")

    def set_data(self, case, affine, cursor, window, overlays):
        self.cursor = np.array(cursor)
        self.geometry = slice_geometry(case.mri.shape, affine, cursor, self.plane)
        values = sample_slice(case.mri, affine, self.geometry)
        rgb = window_to_rgb(values, *window).astype(float)
        for overlay in overlays:
            mask, color, opacity = overlay[:3]
            overlay_affine = overlay[3] if len(overlay) > 3 else affine
            visible = sample_slice(mask.astype(np.uint8, copy=False), overlay_affine, self.geometry, order=0) > 0
            rgb[visible] = (1 - opacity) * rgb[visible] + opacity * np.asarray(color)
        self.rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        h, w, _ = self.rgb.shape
        self.image = QImage(self.rgb.data, w, h, self.rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def image_rect(self):
        if not self.geometry:
            return QRectF()
        available = QRectF(25, 30, self.width() - 50, self.height() - 57)
        width_mm, height_mm = self.geometry.physical_size
        scale = min(available.width() / width_mm, available.height() / height_mm)
        width, height = width_mm * scale, height_mm * scale
        return QRectF(available.center().x() - width / 2, available.center().y() - height / 2, width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#06090b"))
        painter.setPen(QColor("#2b373b"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(QColor("#c9d5d6"))
        painter.drawText(12, 19, self.plane.upper())
        if self.image is None:
            painter.setPen(QColor("#607175"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "MRI view\nLoad a case to inspect")
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        a, b, _ = self.geometry.axes
        x = rect.left() + (self.cursor[a] - self.geometry.bounds[0, a]) / np.ptp(self.geometry.bounds[:, a]) * rect.width()
        y = rect.top() + (self.geometry.bounds[1, b] - self.cursor[b]) / np.ptp(self.geometry.bounds[:, b]) * rect.height()
        painter.setPen(QPen(QColor(135, 197, 181, 145), 1))
        painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
        painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
        labels = PLANE_LABELS[self.plane]
        painter.setPen(QColor("#a5b6b9"))
        painter.drawText(9, int(rect.center().y()), labels[0])
        painter.drawText(self.width() - 19, int(rect.center().y()), labels[1])
        painter.drawText(int(rect.center().x()), 38, labels[2])
        painter.drawText(int(rect.center().x()), self.height() - 10, labels[3])
        painter.drawText(12, self.height() - 10, f"{self.geometry.position_mm:+.1f} mm")
        # A physical scale bar uses the same millimeter-to-screen transform.
        maximum_bar = min(20., self.geometry.physical_size[0] / 4)
        bar = max(value for value in (.1, .5, 1., 2., 5., 10., 20.) if value <= maximum_bar)
        pixels = bar / self.geometry.physical_size[0] * rect.width()
        right = self.width() - 16
        painter.drawLine(int(right - pixels), self.height() - 16, right, self.height() - 16)
        painter.drawText(QRectF(right - 65, self.height() - 34, 65, 16), Qt.AlignmentFlag.AlignRight, f"{bar:g} mm")

    def mousePressEvent(self, event):
        rect = self.image_rect()
        if self.geometry and rect.contains(event.position()):
            x = (event.position().x() - rect.left()) / rect.width()
            y = (event.position().y() - rect.top()) / rect.height()
            self.point_selected.emit(self.geometry.world_at(x, y))

    def wheelEvent(self, event):
        if self.geometry:
            self.scrolled.emit(self.plane, 1 if event.angleDelta().y() > 0 else -1)
            event.accept()
