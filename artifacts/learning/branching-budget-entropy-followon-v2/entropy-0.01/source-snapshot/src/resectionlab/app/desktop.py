"""Case-centered native research workspace.

Operations run in cancellable background jobs. All displayed measurements are
computed from the current case or retained planner output; unsupported clinical
quantities remain explicitly unavailable.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import html
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, QPoint, QRect, QStandardPaths
from PySide6.QtGui import QAction, QKeySequence, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QDoubleSpinBox, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSlider, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from resectionlab.core import thaw_json
from resectionlab.imaging import (
    create_synthetic_case, load_case, load_nifti_case, read_case_artifacts, save_case, revise_compartment,
)
from .annotation import edit_sphere
from .reslice import PLANE_AXES, ras_affine, world_bounds
from .scene import AnatomyScene, mask_surface
from .slice_view import SliceView
from .style import COLORS, STYLE
from .workers import BackgroundJob


def label(text, kind=None, wrap=False):
    widget = QLabel(text)
    if kind:
        widget.setObjectName(kind)
    widget.setWordWrap(wrap)
    return widget


def button(text, callback, *, primary=False):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    if primary:
        widget.setObjectName("primary")
    return widget


def plain(value):
    """Normalize public evaluator data for the portable JSON artifact."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    from dataclasses import fields, is_dataclass
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    return value


def prepare_case(loader, cancelled, progress):
    """Load, window and mesh once off the GUI thread."""
    started = time.perf_counter()
    progress(10, "Validating source images and coordinates…")
    case = loader()
    if cancelled.is_set():
        return None
    affine = ras_affine(case.affine, case.frame)
    positive = case.mri.ravel()[::max(1, case.mri.size // 200000)]
    positive = positive[np.isfinite(positive) & (positive != 0)]
    window = tuple(np.percentile(positive, [1, 99])) if len(positive) else (0., 1.)
    meshes = {}
    masks = dict(case.compartments)
    if case.brain_mask is not None:
        masks = {"brain": case.brain_mask, **masks}
    for index, (name, mask) in enumerate(masks.items()):
        if cancelled.is_set():
            return None
        progress(20 + int(65 * index / max(len(masks), 1)), f"Building {name.replace('_', ' ')} surface…")
        if np.any(mask):
            meshes[name] = mask_surface(mask)
    progress(95, "Preparing linked views…")
    return {"case": case, "affine": affine, "meshes": meshes, "window": window,
            "load_seconds": time.perf_counter() - started}


class MainWindow(QMainWindow):
    case_ready = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RessectionLab · Patient-specific planning research")
        self.resize(1460, 920)
        self.setMinimumSize(1080, 720)
        self.case = None
        self.affine = None
        self.cursor = np.zeros(3)
        self.window = (0., 1.)
        self.colors = {}
        self.layers = {}
        self.routes = []
        self.current_job = None
        self.annotation_dialog = None
        self.training_dialog = None
        self.training_run = None
        self.training_report = None
        self.training_active = False
        self.replay = None
        self.replay_overlay = None
        self.bundle_path = None
        self.pending_artifacts = None
        self.load_seconds = None
        self.dirty = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self._build_ui()
        self._build_menu()

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 10)
        outer.setSpacing(12)
        top = QHBoxLayout()
        top.addWidget(label("RessectionLab", "brand"))
        top.addSpacing(16)
        top.addWidget(label("PATIENT-SPECIFIC PLANNING", "muted"))
        top.addStretch()
        top.addWidget(label("RESEARCH ONLY", "badge"))
        self.open_button = button("Open case", self.open_bundle_dialog)
        self.import_button = button("Import MRI", self.import_dialog)
        self.save_button = button("Save case", self.save_dialog)
        self.save_button.setEnabled(False)
        top.addWidget(self.open_button)
        top.addWidget(self.import_button)
        top.addWidget(self.save_button)
        outer.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._left_panel())
        split.addWidget(self._center_panel())
        split.addWidget(self._right_panel())
        split.setSizes([230, 850, 310])
        outer.addWidget(split, 1)

        self.timeline = QFrame()
        self.timeline.setObjectName("panel")
        timeline_layout = QVBoxLayout(self.timeline)
        timeline_layout.setContentsMargins(12, 8, 12, 8)
        self.timeline_label = label("Coarse selection replay · source MRI remains unchanged", "muted", True)
        timeline_layout.addWidget(self.timeline_label)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.valueChanged.connect(self.replay_step)
        timeline_layout.addWidget(self.timeline_slider)
        self.timeline.hide()
        outer.addWidget(self.timeline)

        footer = QHBoxLayout()
        self.operation_label = label("Ready · Load a public case or explore the synthetic geometry fixture", "muted")
        footer.addWidget(self.operation_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(200)
        self.progress.hide()
        footer.addWidget(self.progress)
        self.cancel_button = button("Cancel", self.cancel_job)
        self.cancel_button.hide()
        footer.addWidget(self.cancel_button)
        self.coordinate_label = label("RAS+ · millimeters", "muted")
        footer.addWidget(self.coordinate_label)
        outer.addLayout(footer)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Local processing · Clinical deficit probabilities are not available")

    def _left_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(310)
        content = QVBoxLayout(panel)
        content.setContentsMargins(14, 12, 14, 14)
        content.addWidget(label("CASE & EVIDENCE", "section"))
        self.case_title = label("No case open", "title", True)
        content.addWidget(self.case_title)
        self.case_subtitle = label("Your imaging stays on this Mac.", "muted", True)
        content.addWidget(self.case_subtitle)
        self.demo_button = button("Explore synthetic fixture", self.load_demo)
        content.addWidget(self.demo_button)
        content.addWidget(label("SOURCE LAYERS", "section"))
        self.mri_label = label("Structural MRI · not loaded", "muted", True)
        content.addWidget(self.mri_label)
        self.layer_container = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_container)
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_layout.setSpacing(0)
        content.addWidget(self.layer_container)
        self.plane_checkbox = QCheckBox("MRI plane in 3-D")
        self.plane_checkbox.setChecked(True)
        self.plane_checkbox.toggled.connect(self.refresh_views)
        content.addWidget(self.plane_checkbox)
        content.addWidget(label("Overlay opacity", "muted"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 80)
        self.opacity.setValue(35)
        self.opacity.valueChanged.connect(self.refresh_views)
        content.addWidget(self.opacity)
        content.addWidget(label("FUNCTIONAL EVIDENCE", "section"))
        self.motor_label = label("Motor · unassessed", "muted", True)
        self.language_label = label("Language · unassessed", "muted", True)
        content.addWidget(self.motor_label)
        content.addWidget(self.language_label)
        content.addWidget(label("Missing evidence remains unknown. Atlas priors require registration and review.", "muted", True))
        content.addWidget(label("CASE QUALITY", "section"))
        self.qc_label = label("Import validates geometry, units, orientation and source labels.", "muted", True)
        content.addWidget(self.qc_label)
        self.inspect_button = button("Inspect sources / transforms", self.inspect_sources)
        self.inspect_button.setEnabled(False)
        content.addWidget(self.inspect_button)
        self.correct_button = button("Correct annotation…", self.correct_annotation_dialog)
        self.correct_button.setEnabled(False)
        content.addWidget(self.correct_button)
        content.addStretch()
        self.case_hash_label = label("No case version", "muted", True)
        content.addWidget(self.case_hash_label)
        return panel

    def _center_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(label("Anatomy workspace", "title"))
        top.addStretch()
        top.addWidget(button("Reset camera", lambda: self.scene.reset_camera()))
        layout.addLayout(top)
        self.mode_label = label("Linked physical MRI views · RAS+ · neurological orientation", "muted")
        layout.addWidget(self.mode_label)
        view_split = QSplitter(Qt.Orientation.Horizontal)
        view_split.setChildrenCollapsible(False)
        scene_panel = QFrame()
        scene_panel.setObjectName("panel")
        scene_layout = QVBoxLayout(scene_panel)
        scene_layout.setContentsMargins(1, 1, 1, 1)
        scene_title = QHBoxLayout()
        scene_title.setContentsMargins(10, 5, 10, 2)
        scene_title.addWidget(label("3-D  ·  RAS mm", "muted"))
        scene_title.addStretch()
        scene_title.addWidget(label("Drag to rotate", "muted"))
        scene_layout.addLayout(scene_title)
        self.scene = AnatomyScene()
        scene_layout.addWidget(self.scene, 1)
        scene_layout.addWidget(label("  Anatomy estimates + complete instrument envelopes", "muted"))
        view_split.addWidget(scene_panel)
        slice_panel = QWidget()
        slice_panel.setMinimumWidth(205)
        slice_layout = QVBoxLayout(slice_panel)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        slice_layout.setSpacing(5)
        self.slices = {}
        for plane in ("axial", "coronal", "sagittal"):
            view = SliceView(plane)
            view.point_selected.connect(self.set_cursor)
            view.scrolled.connect(self.scroll_slice)
            self.slices[plane] = view
            slice_layout.addWidget(view, 1)
        view_split.addWidget(slice_panel)
        view_split.setSizes([510, 280])
        layout.addWidget(view_split, 1)
        self.anatomy_note = label("Source imaging under every overlay. Click a slice to inspect the same physical point in all views.", "muted", True)
        layout.addWidget(self.anatomy_note)
        return container

    def _right_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(270)
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(label("ROUTE COMPARISON", "section"))
        layout.addWidget(label("Inspect the alternatives", "title", True))
        layout.addWidget(label("Compare modeled access and geometric exposure under explicit assumptions.", "muted", True))
        layout.addWidget(label("INSTRUMENT GEOMETRY", "section"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("All generic research instruments", None)
        from resectionlab.geometry import GENERIC_TOOLS
        for tool in GENERIC_TOOLS:
            self.tool_combo.addItem(f"{tool.tool_id.replace('generic_', '').title()} · shaft Ø{2 * tool.shaft_radius_mm:g} mm", tool.tool_id)
        self.tool_combo.currentIndexChanged.connect(self.tool_changed)
        layout.addWidget(self.tool_combo)
        self.tool_note = label("Dimensions are research assumptions, not a commercial device specification.", "muted", True)
        layout.addWidget(self.tool_note)
        self.support_checkbox = QCheckBox("Use estimated image support\nfor hypothetical access windows")
        self.support_checkbox.setToolTip("Explicit research assumption. Nonzero MRI support is not a reviewed brain or cortical segmentation. Does not establish clinical access or remove tissue.")
        self.support_checkbox.hide()
        layout.addWidget(self.support_checkbox)
        self.generate_button = button("Generate candidate routes", self.generate_routes, primary=True)
        self.generate_button.setEnabled(False)
        layout.addWidget(self.generate_button)
        self.refine_button = button("Refine for this case", self.refine_case)
        self.refine_button.setEnabled(False)
        self.refine_button.setToolTip("Case-specific policy training becomes available when a validated simulation is prepared.")
        layout.addWidget(self.refine_button)
        self.route_filter = QComboBox()
        self.route_filter.addItem("Retained alternatives", "pareto")
        self.route_filter.addItem("Dominated candidates", "dominated")
        self.route_filter.addItem("Rejected candidates", "rejected")
        self.route_filter.currentIndexChanged.connect(self.populate_routes)
        layout.addWidget(self.route_filter)
        self.route_list = QListWidget()
        self.route_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.route_list.setMinimumHeight(125)
        self.route_list.itemSelectionChanged.connect(self.route_selection_changed)
        layout.addWidget(self.route_list, 1)
        self.route_summary = label("No routes generated. Select two candidates to compare full tool envelopes.", "muted", True)
        layout.addWidget(self.route_summary)
        self.metrics = QTextBrowser()
        self.metrics.setMinimumHeight(180)
        self.metrics.setMaximumHeight(340)
        self.metrics.setHtml("<p style='color:#91a1a6'>Generate routes after reviewing the case. Accessible target volume will be reported separately from simulated removed volume.</p>")
        layout.addWidget(self.metrics, 1)
        self.unknown_label = label("Clinical outcomes are not predicted. Functional and vascular anatomy remain unassessed.", "warning", True)
        layout.addWidget(self.unknown_label)
        return panel

    def _build_menu(self):
        menu = self.menuBar().addMenu("File")
        for title, shortcut, callback in [
            ("Open case…", QKeySequence.StandardKey.Open, self.open_bundle_dialog),
            ("Save case…", QKeySequence.StandardKey.Save, self.save_dialog),
            ("Import MRI…", "Ctrl+I", self.import_dialog),
            ("Synthetic fixture", "Ctrl+D", self.load_demo),
        ]:
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            menu.addAction(action)
        menu.addSeparator()
        export_action = menu.addAction("Export workspace image…")
        export_action.triggered.connect(self.export_image)
        help_menu = self.menuBar().addMenu("Research")
        source_action = help_menu.addAction("Inspect current case evidence")
        source_action.triggered.connect(self.inspect_sources)

    def start_job(self, operation, finished, message, *, structured_update=None):
        if self.current_job is not None:
            return
        job = BackgroundJob(operation, with_updates=structured_update is not None)
        self.current_job = job
        job.signals.progress.connect(self.job_progress)
        job.signals.finished.connect(lambda result: self.job_finished(result, finished))
        job.signals.failed.connect(self.job_failed)
        job.signals.cancelled.connect(lambda: self.job_finished(None, None))
        if structured_update is not None:
            job.signals.updated.connect(structured_update)
        self.operation_label.setText(message)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        self.set_busy(True)
        self.pool.start(job)

    def set_busy(self, busy):
        for control in (self.open_button, self.import_button, self.demo_button):
            control.setEnabled(not busy)
        self.save_button.setEnabled(not busy and self.case is not None)
        self.generate_button.setEnabled(not busy and self.case is not None and bool(self.case.compartments))
        self.correct_button.setEnabled(not busy and self.case is not None and bool(self.case.compartments))
        self.refine_button.setEnabled(not busy and self.case is not None and bool(self.routes))

    def job_progress(self, value, message):
        self.progress.setValue(value)
        self.operation_label.setText(message)

    def job_finished(self, result, callback):
        was_cancelled = self.current_job and self.current_job.cancel_event.is_set()
        self.current_job = None
        self.progress.hide()
        self.cancel_button.hide()
        self.set_busy(False)
        if callback is not None and result is not None:
            callback(result)
        elif was_cancelled:
            self.operation_label.setText("Operation cancelled · Current case preserved")
        if self.training_active and callback is None:
            self.training_active = False
            if self.training_dialog:
                self.training_dialog.set_running(False)
            if self.training_run:
                result_path = Path(self.training_run["output_dir"]) / "result.json"
                if result_path.exists():
                    self.training_progress(json.loads(result_path.read_text()))

    def job_failed(self, message):
        self.job_finished(None, None)
        self.operation_label.setText("Operation failed · Current case preserved")
        QMessageBox.warning(self, "Unable to complete operation", message)

    def cancel_job(self):
        if self.current_job:
            self.current_job.cancel()
            self.operation_label.setText("Cancelling after the current computation…")

    def load_with(self, loader, *, bundle_path=None):
        self.pending_artifacts = None
        self.bundle_path = bundle_path
        def operation(cancel, progress):
            result = prepare_case(loader, cancel, progress)
            if result is not None and bundle_path:
                result["artifacts"] = read_case_artifacts(bundle_path)
            return result
        self.start_job(operation, self.install_case, "Opening case…")

    def load_demo(self):
        self.load_with(create_synthetic_case)

    def load_nifti(self, mri, mask=None, *, ucsf_labels=False):
        names = {1: "necrotic_non_enhancing_core", 2: "flair_abnormality", 4: "enhancing_tumor"} if ucsf_labels else None
        self.load_with(lambda: load_nifti_case(mri, mask, label_map=names))

    def install_case(self, prepared):
        self.case = prepared["case"]
        self.affine = prepared["affine"]
        self.window = prepared["window"]
        self.load_seconds = prepared["load_seconds"]
        self.routes = []
        self.replay = self.replay_overlay = None
        self.timeline.hide()
        if self.training_run and self.training_run.get("case_hash") != self.case.semantic_hash:
            self.training_run = self.training_report = None
        self.dirty = False
        self.colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(self.case.compartments)}
        self.scene.set_surfaces(prepared["meshes"], self.affine, self.colors)
        target = np.zeros(self.case.mri.shape, dtype=bool)
        for mask in self.case.compartments.values():
            target |= mask
        voxels = np.argwhere(target)
        center = np.mean(voxels, axis=0) if len(voxels) else (np.array(self.case.mri.shape) - 1) / 2
        self.cursor = self.affine[:3, :3] @ center + self.affine[:3, 3]
        while self.layer_layout.count():
            self.layer_layout.takeAt(0).widget().deleteLater()
        self.layers.clear()
        if self.case.brain_mask is not None:
            self._add_layer("brain", "Brain mask · derived surface", (130, 151, 160))
        for name, mask in self.case.compartments.items():
            volume = np.count_nonzero(mask) * self.case.voxel_volume_mm3 / 1000
            self._add_layer(name, f"{name.replace('_', ' ')}\n{volume:.2f} mL · annotation", self.colors[name])
        synthetic = any(source.provenance == "simulated" for source in self.case.source_refs)
        self.case_title.setText("Synthetic access fixture" if synthetic else self.case.case_id)
        self.case_subtitle.setText("Synthetic geometry fixture · no patient data" if synthetic else "Annotation-assisted · structural mode")
        if self.case.metadata.get("primary_source_equivalence") == "unverified":
            self.case_subtitle.setText("Public structural mirror\nOfficial source equivalence unverified")
        self.support_checkbox.setChecked(False)
        self.support_checkbox.setVisible(self.case.brain_mask is None)
        self.mri_label.setText(f"Source MRI · {' × '.join(map(str, self.case.mri.shape))}\n" + " × ".join(f"{v:.2f}" for v in np.linalg.norm(self.affine[:3, :3], axis=0)) + " mm voxels")
        self.qc_label.setText("Header geometry checks passed\nVisual alignment review: " + str(self.case.metadata.get("annotation_review", "pending")) + "\nVascular anatomy unassessed")
        self.case_hash_label.setText(f"Case version {self.case.revision}\n{self.case.semantic_hash[:18]}…")
        self.inspect_button.setEnabled(True)
        self.set_busy(False)
        artifacts = prepared.get("artifacts", {})
        ui = artifacts.get("workspace", {})
        if ui.get("case_hash") == self.case.semantic_hash:
            cursor = ui.get("cursor_ras_mm")
            if cursor is not None:
                self.cursor = np.asarray(cursor, dtype=float)
            self.routes = list(ui.get("routes", []))
            self.opacity.setValue(int(ui.get("overlay_opacity", 35)))
            self.plane_checkbox.setChecked(bool(ui.get("show_mri_plane", True)))
            for name, checked in ui.get("visible_layers", {}).items():
                if name in self.layers:
                    self.layers[name].setChecked(bool(checked))
            camera = ui.get("camera")
            if camera:
                current = self.scene.renderer.GetActiveCamera()
                current.SetPosition(*camera["position"])
                current.SetFocalPoint(*camera["focal_point"])
                current.SetViewUp(*camera["view_up"])
        self.populate_routes()
        self.refresh_views()
        self.operation_label.setText(f"Case ready · Loaded and surfaced in {self.load_seconds:.2f} s · Local")
        self.case_ready.emit()

    def _add_layer(self, name, text, color):
        checkbox = QCheckBox(text)
        checkbox.setChecked(True)
        checkbox.setStyleSheet(f"QCheckBox {{ color: rgb({color[0]}, {color[1]}, {color[2]}); }}")
        checkbox.toggled.connect(lambda visible, key=name: self.layer_toggled(key, visible))
        self.layers[name] = checkbox
        self.layer_layout.addWidget(checkbox)

    def layer_toggled(self, name, visible):
        self.scene.set_visibility(name, visible)
        self.refresh_views()

    def refresh_views(self, *_):
        if self.case is None:
            return
        overlays = [(mask, self.colors[name], self.opacity.value() / 100)
                    for name, mask in self.case.compartments.items()
                    if self.layers.get(name) and self.layers[name].isChecked()]
        if self.replay_overlay is not None:
            overlays.append((self.replay_overlay, (94, 192, 240), .5, ras_affine(np.asarray(self.replay["affine"]), self.case.frame)))
        for view in self.slices.values():
            view.set_data(self.case, self.affine, self.cursor, self.window, overlays)
        axial = self.slices["axial"]
        self.scene.set_axial_image(axial.rgb, axial.geometry, self.plane_checkbox.isChecked())
        self.scene.set_cursor(self.cursor)
        self.scene.render()
        self.coordinate_label.setText("RAS mm  " + "  ".join(f"{v:+.1f}" for v in self.cursor))

    def set_cursor(self, point):
        if self.case is None:
            return
        bounds = world_bounds(self.case.mri.shape, self.affine)
        self.cursor = np.clip(np.asarray(point), bounds[0], bounds[1])
        self.refresh_views()

    def scroll_slice(self, plane, direction):
        point = self.cursor.copy()
        normal = PLANE_AXES[plane][2]
        point[normal] += direction * np.min(np.linalg.norm(self.affine[:3, :3], axis=0))
        self.set_cursor(point)

    def import_dialog(self):
        mri, _ = QFileDialog.getOpenFileName(self, "Select structural MRI", "", "NIfTI images (*.nii *.nii.gz)")
        if not mri:
            return
        mask, _ = QFileDialog.getOpenFileName(self, "Optional supplied target annotation · Cancel to view MRI only", str(Path(mri).parent), "NIfTI images (*.nii *.nii.gz)")
        self.load_nifti(mri, mask or None)

    def open_bundle_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open RessectionLab case", "", "RessectionLab case (*.ressectionlab *.rslab *.zip)")
        if path:
            self.load_with(lambda: load_case(path), bundle_path=path)

    def save_dialog(self):
        if self.case is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save portable case", self.bundle_path or f"{self.case.case_id}.rslab", "RessectionLab case (*.rslab)")
        if path:
            self.save_to(path)

    def save_to(self, path):
        case = self.case
        camera = self.scene.renderer.GetActiveCamera()
        artifacts = {"workspace": {"case_hash": case.semantic_hash, "cursor_ras_mm": self.cursor.tolist(), "routes": self.routes,
                                  "overlay_opacity": self.opacity.value(), "show_mri_plane": self.plane_checkbox.isChecked(),
                                  "visible_layers": {name: checkbox.isChecked() for name, checkbox in self.layers.items()},
                                  "camera": {"position": list(camera.GetPosition()), "focal_point": list(camera.GetFocalPoint()), "view_up": list(camera.GetViewUp())}}}
        def save_operation(cancel, progress):
            progress(15, "Saving source references, images and workspace…")
            return save_case(case, path, artifacts=artifacts)
        def finished(result):
            self.bundle_path = str(result)
            self.dirty = False
            self.operation_label.setText(f"Saved portable case · {Path(result).name}")
        self.start_job(save_operation, finished, "Saving case…")

    def inspect_sources(self):
        if self.case is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Case evidence, provenance & transforms")
        dialog.resize(760, 660)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Source evidence & physical coordinates", "title"))
        browser = QTextBrowser()
        details = {"case_id": self.case.case_id, "case_hash": self.case.semantic_hash,
                   "frame": self.case.frame, "voxel_to_world_mm": self.case.affine.tolist(),
                   "source_refs": [plain(source) for source in self.case.source_refs],
                   "unknowns": list(self.case.unknowns), "metadata": thaw_json(self.case.metadata),
                   "patient_context": plain(self.case.context) if self.case.context else "Not supplied; postoperative results cannot be assumed preoperative."}
        browser.setPlainText(json.dumps(details, indent=2, default=str))
        layout.addWidget(browser)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.exec()

    def correct_annotation_dialog(self):
        if self.case is None or not self.case.compartments:
            return
        if self.annotation_dialog is not None:
            self.annotation_dialog.raise_()
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Correct source annotation")
        dialog.resize(370, 310)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Review and correct a target mask", "title", True))
        layout.addWidget(label("Move the linked cursor in the MRI views, then apply a spherical annotation brush. Original source masks are preserved. Each edit creates a new case version and invalidates routes.", "muted", True))
        target = QComboBox()
        target.addItems(list(self.case.compartments))
        layout.addWidget(target)
        mode = QComboBox()
        mode.addItem("Erase annotation", False)
        mode.addItem("Add annotation", True)
        layout.addWidget(mode)
        radius = QDoubleSpinBox()
        radius.setRange(.5, 15.)
        radius.setValue(2.)
        radius.setSuffix(" mm radius")
        layout.addWidget(radius)
        reason = QLineEdit()
        reason.setPlaceholderText("Required correction reason")
        layout.addWidget(reason)
        apply = button("Apply at linked cursor", lambda: None, primary=True)
        def apply_edit():
            if self.current_job is not None or not reason.text().strip():
                reason.setFocus()
                return
            case, cursor, name = self.case, self.cursor.copy(), target.currentText()
            edited = edit_sphere(case.compartments[name], self.affine, cursor, radius.value(), add=mode.currentData())
            if np.array_equal(edited, case.compartments[name]):
                self.operation_label.setText("Annotation unchanged · No voxels affected")
                return
            revised = revise_compartment(case, name, edited, reason=reason.text().strip())
            def completed(prepared):
                self.install_case(prepared)
                self.set_cursor(cursor)
                self.dirty = True
                self.operation_label.setText("Annotation revised · Previous routes invalidated · Save to retain changes")
            self.start_job(lambda cancel, progress: prepare_case(lambda: revised, cancel, progress), completed, "Applying annotation correction…")
        apply.clicked.connect(apply_edit)
        layout.addWidget(apply)
        dialog.finished.connect(lambda: setattr(self, "annotation_dialog", None))
        self.annotation_dialog = dialog
        dialog.show()

    def generate_routes(self):
        if self.case is None:
            return
        case = self.case
        from resectionlab.geometry import GENERIC_TOOLS
        selected_tool = self.tool_combo.currentData()
        tools = tuple(tool for tool in GENERIC_TOOLS if tool.tool_id == selected_tool) if selected_tool else None
        support = None
        support_provenance = None
        if case.brain_mask is None:
            if not self.support_checkbox.isChecked():
                self.operation_label.setText("Review the explicit estimated-support assumption before generating hypothetical windows.")
                self.support_checkbox.setFocus()
                return
            support = np.isfinite(case.mri) & (case.mri != 0)
            support_provenance = {"source": case.semantic_hash, "method": "finite_nonzero_structural_MRI_support_v1", "evidence_type": "estimated"}
        def operation(cancel, progress):
            from resectionlab.planning import generate_candidate_routes
            result = generate_candidate_routes(case, tools=tools, support_mask=support, support_provenance=support_provenance, cancel=cancel.is_set,
                progress=lambda done, total: progress(int(100 * done / max(total, 1)), f"Checking complete instruments · {done}/{total}"))
            return result
        self.start_job(operation, self.install_routes, "Generating instrument-aware candidate routes…")

    def install_routes(self, result):
        self.routes = [plain(candidate) for candidate in result.candidates]
        self.dirty = True
        self.populate_routes()
        self.refine_button.setEnabled(bool(self.routes))
        self.operation_label.setText(f"{len(self.routes)} candidate evaluations · {result.elapsed_seconds:.2f} s · SEARCH")

    def populate_routes(self, *_):
        if not hasattr(self, "route_list"):
            return
        self.route_list.clear()
        category = self.route_filter.currentData()
        matches = [(index, route) for index, route in enumerate(self.routes) if route.get("category") == category]
        for index, route in matches:
            volume = route.get("accessible_target_volume_mm3", 0) / 1000
            name = route.get("route_id", route.get("candidate_id", f"Route {index + 1:02d}"))
            item = QListWidgetItem(f"{name}\n{route.get('tool_id', 'Generic tool')} · {volume:.2f} mL accessible")
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.route_list.addItem(item)
        self.route_summary.setText(f"{len(matches)} {category} candidates · Select up to two to compare" if self.routes else "No routes generated. Select two candidates to compare full tool envelopes.")
        if matches:
            self.route_list.item(0).setSelected(True)
        else:
            self.scene.set_routes([])

    def route_selection_changed(self):
        selected = self.route_list.selectedItems()[:2]
        visible, sections = [], []
        for item in selected:
            route = self.routes[item.data(Qt.ItemDataRole.UserRole)]
            geometry = route.get("geometry", {})
            radius = route.get("shaft_radius_mm", route.get("tool", {}).get("shaft_radius_mm", 1.5))
            length = route.get("tool", {}).get("working_length_mm", 120.)
            failures = geometry.get("failures", [])
            point = failures[0].get("position_mm") if failures else None
            entry, target = np.asarray(route["entry_mm"], dtype=float), np.asarray(route["target_mm"], dtype=float)
            if self.case and self.case.frame == "LPS+":
                entry, target = entry * [-1, -1, 1], target * [-1, -1, 1]
                if point is not None:
                    point = np.asarray(point) * [-1, -1, 1]
            visible.append((entry, target, radius, length, route.get("category") == "rejected", point))
            volume = route.get("accessible_target_volume_mm3", 0) / 1000
            failure = "; ".join(f"{item.get('reason', '')}: {item.get('detail', '')}" for item in failures)
            exposure = route.get("normal_tissue_exposure_mm3")
            exposure_text = "Not available" if exposure is None else f"{exposure / 1000:.3f} mL (conservative)"
            length_text = f"{route.get('route_length_mm', 0):.1f} mm"
            clearance = geometry.get("clearance_mm")
            clearance_text = "Not available" if clearance is None else f"{clearance:.2f} mm (lower bound)"
            compartment_text = "<br>".join(f"{html.escape(name.replace('_', ' '))}: {amount / 1000:.3f} mL" for name, amount in route.get("accessible_target_volume_mm3_by_compartment", {}).items())
            sections.append(f"<h3>{html.escape(route.get('route_id', route.get('candidate_id', 'Candidate route')))}</h3>"
                f"<p><b>Accessible target</b> {volume:.3f} mL<br>"
                f"<b>Simulated removed volume</b> Not computed<br>"
                f"<b>Route length</b> {length_text}<br>"
                f"<b>Normal-tissue exposure</b> {exposure_text}<br>"
                f"<b>Clearance</b> {clearance_text}<br>"
                f"<b>Assessment</b> {html.escape(str(route.get('assessment', 'Incomplete anatomy')))}<br>"
                "</p>"
                f"<p>{compartment_text}</p>"
                f"<p>{html.escape(str(failure))}</p>"
                "<p style='color:#9caeb1'>Conditional static access under the displayed hypothetical window and generic tool model.</p>")
        self.scene.set_routes(visible)
        if sections:
            self.metrics.setHtml("<hr>".join(sections))

    def tool_changed(self, *_):
        from resectionlab.geometry import GENERIC_TOOLS
        selected = self.tool_combo.currentData()
        tool = next((tool for tool in GENERIC_TOOLS if tool.tool_id == selected), None)
        if tool:
            self.tool_note.setText(f"Tip Ø{2 * tool.tip_radius_mm:g} mm · shaft Ø{2 * tool.shaft_radius_mm:g} mm · working length {tool.working_length_mm:g} mm. Generic research geometry.")
        else:
            self.tool_note.setText("Compare two generic instruments. Dimensions are research assumptions.")

    def refine_case(self):
        if self.case is None:
            return
        if self.training_dialog is None:
            from .training_view import TrainingDialog
            self.training_dialog = TrainingDialog(self)
            self.training_dialog.start_requested.connect(self.start_refinement)
            self.training_dialog.resume_requested.connect(self.resume_refinement)
            self.training_dialog.cancel_requested.connect(self.cancel_job)
        if self.training_report:
            self.training_dialog.set_report(self.training_report)
        self.training_dialog.set_running(self.training_active)
        if self.training_report and not self.training_active:
            self.training_dialog.set_report(self.training_report)
        self.training_dialog.show()
        self.training_dialog.raise_()

    def start_refinement(self, budget_seconds):
        if self.current_job is not None or self.case is None:
            return
        if self.case.brain_mask is None and not self.support_checkbox.isChecked():
            self.operation_label.setText("Review estimated image-support assumptions before preparing the coarse research simulation.")
            return
        destination = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / "runs" / uuid.uuid4().hex
        self.training_run = {"case_hash": self.case.semantic_hash, "output_dir": str(destination),
                             "budget_seconds": float(budget_seconds), "seed": 0, "block_size": 6}
        self._run_refinement(resume=False)

    def resume_refinement(self):
        if not self.training_run or self.case.semantic_hash != self.training_run["case_hash"]:
            return
        self._run_refinement(resume=True)

    def _run_refinement(self, *, resume):
        if self.current_job is not None:
            return
        case, settings = self.case, dict(self.training_run)
        self.training_active = True
        self.training_dialog.set_running(True)
        def operation(cancel, progress, update):
            from .refinement import run_refinement
            progress(0, "Preparing frozen coarse patient simulation…")
            def notify(report):
                update(report)
                progress(min(98, int(100 * report.get("elapsed_seconds", 0) / settings["budget_seconds"])),
                    f"PATIENT_SCRATCH_RL · {report.get('gradient_steps', 0)} actual updates · {report.get('optimization_environment_steps', 0)} optimization transitions")
            return run_refinement(case, settings["output_dir"], budget_seconds=settings["budget_seconds"],
                seed=settings["seed"], block_size=settings["block_size"], resume=resume,
                cancelled=cancel.is_set, progress=notify)
        self.start_job(operation, self.install_refinement, "Training policy inside this case…", structured_update=self.training_progress)

    def training_progress(self, report):
        self.training_report = dict(report)
        if self.training_run:
            result = Path(self.training_run["output_dir"]) / "result.json"
            if result.exists():
                self.training_report.update(json.loads(result.read_text()))
        if self.training_dialog:
            self.training_dialog.set_report(self.training_report)

    def install_refinement(self, report):
        self.training_active = False
        self.training_dialog.set_running(False)
        self.training_report = report
        self.training_dialog.set_report(report)
        self.dirty = True
        self.operation_label.setText(f"Patient refinement finished · {report.get('gradient_steps', 0)} actual updates · {report.get('status')} · Initial search retained")
        replay = report.get("replay")
        if replay and replay.get("independent_certificate", {}).get("feasible"):
            self.replay = replay
            self.timeline_slider.setRange(0, len(replay["metrics"]["history"]))
            self.timeline_slider.setValue(0)
            self.timeline.show()
            self.replay_step(0)
        elif replay:
            self.operation_label.setText("Refinement saved · Selection replay failed independent geometry checks; removal display withheld")

    def replay_step(self, step):
        if self.replay is None:
            return
        from .refinement import replay_mask, replay_volumes
        self.replay_overlay = replay_mask(self.replay, step)
        metrics = replay_volumes(self.replay, step)
        self.timeline_label.setText(f"COARSE SELECTION REPLAY · step {step}/{len(self.replay['metrics']['history'])} · "
            f"modeled removed target {metrics['removed_target_mm3'] / 1000:.3f} mL · residual {metrics['residual_target_mm3'] / 1000:.3f} mL\n"
            "Blue = simulated coarse-cell footprint. Source MRI unchanged; native-resolution removal is not certified.")
        self.scene.set_replay_surface(self.replay_overlay, ras_affine(np.asarray(self.replay["affine"]), self.case.frame))
        self.refresh_views()

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export current workspace", "RessectionLab-workspace.png", "PNG image (*.png)")
        if path:
            self.save_workspace_image(path)

    def save_workspace_image(self, path):
        """Capture Qt plus the actual native VTK framebuffer in one image."""
        pixmap = self.grab()
        rgb = self.scene.capture_rgb()
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        painter = QPainter(pixmap)
        position = self.scene.mapTo(self, QPoint(0, 0))
        painter.drawImage(QRect(position, self.scene.size()), image)
        painter.end()
        return pixmap.save(str(path))

    def closeEvent(self, event):
        if self.current_job is not None:
            self.current_job.cancel()
            self.operation_label.setText("Cancelling current operation before closing…")
            event.ignore()
            QTimer.singleShot(200, self.close)
            return
        self.scene.Finalize()
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="RessectionLab native patient-specific planning research workspace")
    parser.add_argument("--demo", action="store_true", help="Open the explicitly synthetic geometry fixture")
    parser.add_argument("--case", type=Path, help="Open a saved portable case bundle")
    parser.add_argument("--mri", type=Path, help="Import a NIfTI structural image")
    parser.add_argument("--mask", type=Path, help="Import a same-frame supplied annotation")
    parser.add_argument("--ucsf-labels", action="store_true", help="Use documented UCSF/BraTS labels 1,2,4")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    application = QApplication.instance() or QApplication([sys.argv[0]])
    application.setApplicationName("RessectionLab")
    application.setOrganizationName("RessectionLab")
    application.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    window.scene.Initialize()
    window.scene.show_orientation()

    if args.smoke_test:
        def finish_smoke():
            checks = {plane: bool(view.image and not view.image.isNull() and view.rgb.shape[0] > 0)
                      for plane, view in window.slices.items()}
            checks["three_dimensional_anatomy"] = bool(window.scene.actors)
            checks["aligned_axial_plane"] = window.scene.plane_actor is not None
            pixels = window.scene.capture_rgb()
            checks["nonempty_rendered_anatomy"] = bool(np.ptp(pixels.astype(float)) > 80)
            report = {"passed": all(checks.values()), "checks": checks, "case_id": window.case.case_id,
                      "case_hash": window.case.semantic_hash, "shape": list(window.case.mri.shape),
                      "frame": "RAS+", "load_seconds": window.load_seconds,
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "scope": "Native rendering smoke test; no clinical validity claim"}
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                report["screenshot_saved"] = window.save_workspace_image(args.screenshot)
            if args.smoke_report:
                args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
                args.smoke_report.write_text(json.dumps(report, indent=2))
            application.exit(0 if report["passed"] else 1)
        window.case_ready.connect(lambda: QTimer.singleShot(1500, finish_smoke))
        QTimer.singleShot(120000, lambda: application.exit(2))
    if args.case:
        QTimer.singleShot(0, lambda: window.load_with(lambda: load_case(args.case), bundle_path=args.case))
    elif args.mri:
        QTimer.singleShot(0, lambda: window.load_nifti(args.mri, args.mask, ucsf_labels=args.ucsf_labels))
    elif args.demo or args.smoke_test:
        QTimer.singleShot(0, window.load_demo)
    if args.screenshot and not args.smoke_test:
        window.case_ready.connect(lambda: QTimer.singleShot(1000, lambda: window.save_workspace_image(args.screenshot)))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
