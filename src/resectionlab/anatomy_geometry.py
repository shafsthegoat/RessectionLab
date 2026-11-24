"""Compiled, cached display meshes and source-derived vascular geometry."""

from __future__ import annotations
import os
import json
import tempfile
from dataclasses import dataclass
from time import perf_counter
from itertools import product
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import digest, affine, world, file_hash
from .core import immutable_array
from .surface_fairing import (
    fair_surface, intersection_vertices, surface_kind, broad_context_surface, NATIVE_POLICY, POLICY,
    NATIVE_VASCULAR_POLICY, OPEN_VASCULAR_POLICY,
)


def _save_cache(path, values):
    """Interrupted writers cannot leave a partially readable geometry cache."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _poly(values, probability):
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkSurfaceNets3D
    from vtkmodules.util.numpy_support import numpy_to_vtk
    image = vtkImageData()
    image.SetDimensions(*values.shape)
    image.GetPointData().SetScalars(numpy_to_vtk(np.asarray(values).ravel(order="F"), deep=True))
    if probability:
        contour = vtkFlyingEdges3D()
        contour.SetValue(0, 0.5)
    else:
        contour = vtkSurfaceNets3D()
        contour.SetValue(0, 1)
        contour.SetOutputMeshTypeToQuads()
        contour.SmoothingOff()
    contour.SetInputData(image)
    contour.Update()
    return contour.GetOutput()


def _arrays(poly):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    if not poly.GetNumberOfPoints():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32)
    points = vtk_to_numpy(poly.GetPoints().GetData()).copy()
    cells = poly.GetPolys()
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    widths = np.diff(offsets)
    if np.all(widths == 3):
        faces = connectivity.reshape(-1, 3).astype(np.uint32)
    elif np.all(widths == 4):
        quads = connectivity.reshape(-1, 4)
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(
            np.uint32
        )
    else:
        raise ValueError("Unexpected VTK surface cell topology")
    return points, faces


def _indexed_contour(values, probability):
    raw, faces = _arrays(_poly(values, probability))
    triangles = raw[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    # Threshold ties may emit zero-area VTK triangles.
    keep = areas > 1e-12
    return raw, faces[keep], int(np.count_nonzero(~keep))


def _suppress_vascular_scores(values, mask):
    """Reflect excluded positives without shifting the retained score samples."""
    excluded = ~mask & (values >= .5)
    if excluded.any():
        below = np.nextafter(values.dtype.type(.5), values.dtype.type(0))
        values[excluded] = np.minimum(1 - values[excluded], below)
    return values


def _vascular_score_contour(scores, mask):
    if np.any(scores[mask] < .5) or not mask.any():
        # A correction may introduce categorically supplied cells. The old
        # score field cannot describe that revision.
        return None
    axes = [np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis))) for axis in range(3)]
    lo = np.maximum(0, np.array([axis[0] for axis in axes]) - 2)
    hi = np.minimum(mask.shape, np.array([axis[-1] for axis in axes]) + 3)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    cropped = mask[crop]
    values = _suppress_vascular_scores(scores[crop].copy(order="C"), cropped)
    positions, indices, _ = _indexed_contour(np.pad(values, 1), probability=True)
    origin = lo - 1
    return (positions + origin).astype(np.float32), indices, cropped, origin


def vascular_probability_mesh(scores, mask):
    """Raw native isosurface for a score-bound case compartment."""
    result = _vascular_score_contour(scores, mask)
    if result is None:
        return None
    positions, indices, _, _ = result
    return positions, indices, normals(positions, indices)


@dataclass(frozen=True)
class ConditionedVascularWall:
    """Source edges and a conditionable wall in the same native IJK frame."""

    source: np.ndarray
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    source_cells: np.ndarray
    diagnostics: dict


def _retain_isolated_score_components(source, positions, faces, owners, mask, native_mask, affine, cancelled):
    """Retain native crossings of completely sampled, single- or two-center components."""
    labels, count = ndi.label(native_mask, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    eligible = sizes <= 2
    eligible[0] = False
    for axis in range(3):
        border = np.swapaxes(labels, 0, axis)[[0, -1]]
        eligible[np.unique(border)] = False
    if mask is not native_mask:
        # An edit can split the original material. Its new fragments do not
        # provide the original isolated-component evidence.
        eligible &= np.bincount(labels[mask], minlength=count + 1) == sizes
    if not eligible.any():
        return positions, {}
    groups = labels.ravel()[owners]
    corner_groups = groups[faces]
    corner_selected = eligible[corner_groups]
    mixed = corner_selected.any(axis=1) & ~corner_selected.all(axis=1)
    eligible[corner_groups[mixed].ravel()] = False
    changed = eligible[groups] & np.any(source != positions, axis=1)
    if not changed.any():
        return positions, {}

    candidate = positions.copy()
    candidate[changed] = source[changed]
    # Each selected closed component uses its original faces and coordinates.
    # Check contact with other components before accepting the replacement.
    linear = np.asarray(affine)[:3, :3]
    physical = candidate.astype(float) @ linear.T
    raw = source.astype(float) @ linear.T
    previous = positions.astype(float) @ linear.T
    if (len(intersection_vertices(raw, physical, faces, cancelled))
            or len(intersection_vertices(previous, physical, faces, cancelled))):
        return positions, {}
    return candidate, {
        'sourceRetainedIsolatedComponents': len(np.unique(groups[changed])),
        'maxDisplacementMm': float(np.linalg.norm(physical - raw, axis=1).max(initial=0.)),
    }


def conditioned_vascular_probability_mesh(scores, mask, affine, *, retain_isolated_components=False,
                                          native_mask=None, cancelled=lambda: False):
    """Build the versioned wall shared by research collision and rendering."""
    if cancelled():
        raise InterruptedError("Vascular wall preparation cancelled")
    result = _vascular_score_contour(scores, mask)
    if result is None:
        return None
    raw, faces, cropped, origin = result
    owners = vascular_mesh_source_cells(raw, mask)
    positions, diagnostics = fair_surface(
        raw, faces, affine, "vessel", mask=np.pad(cropped, 1), mask_origin=origin,
        source_shape=mask.shape, cancelled=cancelled,
    )
    if retain_isolated_components:
        positions, retention = _retain_isolated_score_components(
            raw, positions, faces, owners, mask, mask if native_mask is None else native_mask, affine, cancelled,
        )
        diagnostics.update(retention)
    return ConditionedVascularWall(
        immutable_array(raw), immutable_array(positions), immutable_array(faces),
        immutable_array(normals(positions, faces)), immutable_array(owners), diagnostics,
    )


def vascular_mesh_source_cells(positions, mask):
    """Bind every score crossing to the nearest positive endpoint of its native edge."""
    rounded = np.rint(positions)
    positions = np.where(np.abs(positions - rounded) <= 1e-5, rounded, positions)
    floor = np.floor(positions).astype(np.int64)
    ceil = np.ceil(positions).astype(np.int64)
    if np.any(np.count_nonzero(floor != ceil, axis=1) > 1):
        raise ValueError("Vascular surface must use native score edges")
    distance = np.full(len(positions), np.inf)
    source = np.full(len(positions), np.iinfo(np.uint32).max, dtype=np.uint32)
    for step in product((0, 1), repeat=3):
        candidate = np.where(step, ceil, floor)
        inside = ((candidate >= 0) & (candidate < mask.shape)).all(axis=1)
        rows = np.flatnonzero(inside)
        occupied = rows[mask[tuple(candidate[rows].T)]]
        error = np.sum((candidate[occupied] - positions[occupied])**2, axis=1)
        take = error < distance[occupied]
        accepted = occupied[take]
        distance[accepted] = error[take]
        source[accepted] = np.ravel_multi_index(candidate[accepted].T, mask.shape)
    if not np.isfinite(distance).all():
        raise ValueError("Vascular surface contains an unbound source crossing")
    return source


def normals(points, faces):
    result = np.zeros_like(points, dtype=np.float64)
    f = points[faces]
    n = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    for i in range(3):
        np.add.at(result, faces[:, i], n)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-20)
    return result.astype(np.float32)


def display_surface_kind(row):
    ontology = row.get("label_ontology") or row.get("provenance", {}).get("label_ontology") or {}
    if (row["kind"] == "context" and row.get("provenance", {}).get("model") == "siam-v2"
            and isinstance(ontology, dict) and ontology.get("value") == 16):
        return "outer-head-tissue"
    return row["kind"]


def appearance_kind(row):
    """Cortical display classification does not alter the cached mesh policy."""
    kind = display_surface_kind(row)
    if kind == "brain-region":
        provenance = row.get("provenance") or {}
        model = str(provenance.get("model", "")).lower()
        ontology = row.get("label_ontology") or provenance.get("label_ontology") or {}
        value = ontology.get("value", row.get("label_value")) if isinstance(ontology, dict) else None
        if (model == "siam-v2" and value == 1
                or model in {"supersynth", "synthseg", "fastsurfer"} and value in {3, 42}):
            return "gray-matter"
    return kind


CONDITIONED_SURFACE = "source-bound-conditioned-surface-v1"
CONDITIONED_MAX_ADDITIONAL_MM = .55
CONDITIONED_MAX_SOURCE_MM = .9
CONDITIONED_MAX_SOURCE_CELLS = 1.18


def _display_geometry(collection, row, result, cache_hit, started):
    attachment = row.get("conditioned_surface")
    if attachment is not None and not row.get("edits"):
        from .core import array_digest

        if not isinstance(attachment, dict) or attachment.get("schema") != CONDITIONED_SURFACE:
            raise ValueError("Unsupported conditioned display surface")
        if row["revision"] == attachment["mask_revision"]:
            scan = collection.series(row["source_series"])
            if (row["kind"] not in {"skull", "bone", "brain"} or row["role"] != "context"
                    or row["data"]["sha256"] != attachment["mask_sha256"]
                    or row["label_value"] != attachment["label_value"]
                    or scan["data"]["sha256"] != attachment["source_array_sha256"]
                    or scan["original"]["sha256"] != attachment["source_original_sha256"]
                    or row["affine"] != attachment["affine"] or scan["affine"] != row["affine"]
                    or array_digest(result["source"]) != attachment["source_mesh_hash"]
                    or array_digest(result["indices"]) != attachment["indices_hash"]
                    or array_digest(result["positions"]) != attachment["stock_conditioned_hash"]):
                raise ValueError("Conditioned surface and native anatomy bindings differ")
            points = collection.array(attachment["positions"])
            stock = result["positions"]
            source = result["source"]
            if points.dtype != np.float32 or points.shape != source.shape or not np.isfinite(points).all():
                raise ValueError("Invalid conditioned display positions")
            linear = affine(row["affine"])[:3, :3]
            protected = source[:, 2] <= attachment["protected_native_plane"]
            if "protected_crop_margin_cells" in attachment:
                margin = attachment["protected_crop_margin_cells"]
                face_distance = np.minimum(source, np.asarray(row["data"]["shape"]) - 1 - source)
                protected |= (face_distance <= margin).any(axis=1)
            displacement = np.linalg.norm((points - source) @ linear.T, axis=1)
            if (not protected.any() or not np.array_equal(points[protected], stock[protected])
                    or displacement.max(initial=0.) > CONDITIONED_MAX_SOURCE_MM
                    or np.linalg.norm(points - source, axis=1).max(initial=0.) > CONDITIONED_MAX_SOURCE_CELLS
                    or np.linalg.norm((points - stock) @ linear.T, axis=1).max(initial=0.) > CONDITIONED_MAX_ADDITIONAL_MM):
                raise ValueError("Conditioned display positions exceed the source bounds")
            result = {**result, "positions": points, "normals": normals(points, result["indices"]),
                      "policy_version": CONDITIONED_SURFACE, "max_displacement_mm": displacement.max()}
    return {**result, "cache_hit": cache_hit, "preparation_ms": (perf_counter() - started) * 1000}


def _field_values(collection, row, cropped, slices, display_source):
    source = display_source if display_source is not None else row
    probability = source.get("probability")
    if probability is None or source.get("edits"):
        return cropped.astype(np.uint8), "categorical"
    values = collection.array(probability)[slices]
    vascular = surface_kind(row["kind"]) == "vessel"
    if vascular:
        # Materialize the small native crop in storage order once. Repeated
        # strided scans against an external-drive memmap delay cold extraction.
        values = np.array(values, dtype=np.result_type(values.dtype, np.float32), order="C", copy=True)
    original = collection.array(source["data"])[slices] == source["label_value"] if display_source is not None else cropped
    if values.shape != cropped.shape or not np.isfinite(values).all():
        return cropped.astype(np.uint8), "categorical"
    if not vascular:
        if np.array_equal(values >= .5, cropped):
            return values, "native-probability"
        return cropped.astype(np.uint8), "categorical"
    if (np.any((values > .5) & ~original) or np.any((values < .5) & original)
            or np.any(cropped & ~original)):
        return cropped.astype(np.uint8), "categorical"
    # Match explicit source membership for scores exactly at the isovalue.
    # Only removed positives and excluded ties cross below the contour.
    # Uncut interfaces retain their source field.
    return _suppress_vascular_scores(values, cropped), "native-probability" if display_source is None else "filtered-native-probability"


def local_source_contour(collection, row, slices):
    """Read a native sublattice and leave any cut surface edges open."""
    mask = collection.array(row['data'])[slices] == row['label_value']
    values, field = _field_values(collection, row, mask, slices, vascular_cleanup_source(collection, row))
    if not mask.any() or min(mask.shape) < 2:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32), field)
    positions, indices, _ = _indexed_contour(values, field != 'categorical')
    return positions, indices, field


def condition_open_vascular_contour(source, faces, mask, affine, *, cancelled=lambda: False):
    """Condition a native score surface while holding its open rims fixed."""
    from .vessel_section_stability import preserve_open_sections

    if len(faces) > 80_000:
        return source, {'policy': OPEN_VASCULAR_POLICY, 'kind': 'vessel', 'maxDisplacementMm': 0.,
                        'fallbackReason': 'Field exceeds the interactive vessel geometry budget. Source contour retained.'}
    positions, diagnostics = fair_surface(source, faces, affine, 'vessel', mask=mask,
                                         source_shape=mask.shape, cancelled=cancelled, open_field=True)
    owners = vascular_mesh_source_cells(source, mask)
    positions, retained = _retain_isolated_score_components(source, positions, faces, owners, mask, mask, affine, cancelled)
    guarded, sections = preserve_open_sections(source, positions, faces, mask, affine, cancelled=cancelled)
    if sections['sectionRestoredVertices']:
        linear = np.asarray(affine)[:3, :3]
        raw, guarded_mm = source.astype(float) @ linear.T, guarded.astype(float) @ linear.T
        changed = np.any(guarded != positions, axis=1)[faces].any(axis=1)
        original_tri, new_tri = raw[faces[changed]], guarded_mm[faces[changed]]
        old_normal = np.cross(original_tri[:, 1] - original_tri[:, 0], original_tri[:, 2] - original_tri[:, 0])
        normal = np.cross(new_tri[:, 1] - new_tri[:, 0], new_tri[:, 2] - new_tri[:, 0])
        flipped = np.any(np.einsum('ij,ij->i', normal, old_normal) <= 0.)
        collapsed = np.any(np.linalg.norm(normal, axis=1) < POLICY['minimumFaceAreaRatio'] * np.linalg.norm(old_normal, axis=1))
        if flipped or collapsed or len(intersection_vertices(raw, guarded_mm, faces, cancelled)):
            guarded, guarded_mm = source, raw
            sections['fallbackReason'] = 'Source contour retained after the section repair geometry check.'
        sections['maxDisplacementMm'] = float(np.linalg.norm(guarded_mm - raw, axis=1).max(initial=0.))
    return guarded, {**diagnostics, **retained, **sections}


def vascular_cleanup_source(collection, row):
    """Admit an exact, display-only score binding to an unchanged native layer."""
    if surface_kind(row["kind"]) != "vessel" or row.get("edits"):
        return None
    binding = row.get("provenance", {}).get("vascular_cleanup")
    if not isinstance(binding, dict) or binding.get("result_mask_sha256") != row["data"]["sha256"]:
        return None
    source = next((layer for layer in collection.document["layers"]
                   if layer["id"] == binding.get("source_layer_id")), None)
    if (source is None or source["id"] == row["id"] or source["kind"] != row["kind"]
            or source["source_series"] != row["source_series"] or source["affine"] != row["affine"]
            or source["data"]["shape"] != row["data"]["shape"]
            or source["data"]["sha256"] != binding.get("source_mask_sha256")
            or source.get("edits") or not source.get("probability")
            or source["probability"]["sha256"] != binding.get("source_probability_sha256")):
        return None
    return source


def _attached_erase_source(row):
    from .vascular_cleanup import ATTACHED_SCORE_SURFACE

    edits = row.get("edits")
    if (row["kind"] != "vessels" or not row.get("probability") or not row.get("original_data")
            or not edits or edits[-1].get("surface_policy") != ATTACHED_SCORE_SURFACE):
        return None
    return {**row, "data": row["original_data"], "label_value": row["original_label_value"], "edits": []}


def _brain_envelope_signal(collection, row):
    """Identify the actual same-grid acquisition of an ordinary brain estimate."""
    if (row["kind"] != "brain" or row.get("role") != "context" or row.get("edits")
            or row.get("conditioned_surface") or row.get("probability")
            or row.get("provenance", {}).get("kind") != "image_derived_estimate"):
        return None
    from .anatomy_model_registry import source_volume

    scan = collection.series(row["source_series"])
    if (scan["modality"] not in {"CT", "CTA", "MRI", "T1", "T1CE", "T1C", "T2", "FLAIR"}
            or not source_volume(scan) or scan["affine"] != row["affine"]
            or scan["data"]["shape"] != row["data"]["shape"]):
        return None
    try:
        collection.require_native_geometry(scan)
        collection.require_native_geometry(row)
    except ValueError:
        return None
    return scan["data"]


def mesh(collection, row, *, cancelled=lambda: False, display_source=None, conditioned=True, cache_directory=None):
    """Separate source/display mesh; same connectivity, bounded local movement."""
    started = perf_counter()
    display_source = _attached_erase_source(display_source if display_source is not None else row) or display_source
    if display_source is not None:
        if (display_source["kind"] != row["kind"] or display_source["affine"] != row["affine"]
                or display_source["data"]["shape"] != row["data"]["shape"]):
            raise ValueError("Vascular display filtering requires the source lattice")
        if display_source["data"] == row["data"] and display_source["label_value"] == row["label_value"]:
            return mesh(collection, display_source, cancelled=cancelled, conditioned=conditioned,
                        cache_directory=cache_directory)
    if display_source is None and row.get("publisher_surface") is not None:
        from .anatomy_publisher_surface import conditioned_publisher_cta, publisher_surface

        supplied = publisher_surface(collection, row, cancelled=cancelled)
        if supplied is not None:
            if conditioned:
                supplied = conditioned_publisher_cta(collection, row, supplied, cancelled=cancelled)
            if "normals" not in supplied:
                supplied["normals"] = normals(supplied["positions"], supplied["indices"])
            return {**supplied,
                    "preparation_ms": (perf_counter() - started) * 1000}
    display_kind = display_surface_kind(row)
    profile = surface_kind(display_kind)
    native = profile != "vessel"
    version = NATIVE_POLICY["version"] if native else NATIVE_VASCULAR_POLICY["version"]
    if broad_context_surface(display_kind):
        version = NATIVE_POLICY["broadContext"]["version"]
    brain_signal = _brain_envelope_signal(collection, row) if conditioned and display_source is None else None
    identity = {"algorithm": version, "data": row["data"], "probability": row.get("probability"),
                "affine": row["affine"], "label": row["label_value"],
                "revision": row["revision"], "kind": row["kind"]}
    if profile == "vessel":
        identity["fairing_parameters"] = NATIVE_VASCULAR_POLICY
    if brain_signal is not None:
        from .anatomy_brain_surface import BRAIN_ENVELOPE_POLICY

        identity["brain_envelope"] = {"policy": BRAIN_ENVELOPE_POLICY, "signal": brain_signal["sha256"]}
    if profile == "vessel" and (display_source is not None or row.get("probability")):
        identity["field_policy"] = "native-vascular-filter-v2"
        identity["edited"] = bool(row.get("edits"))
    if display_source is not None:
        identity["display_source"] = {
            "data": display_source["data"], "probability": display_source.get("probability"),
            "label": display_source["label_value"], "edited": bool(display_source.get("edits")),
        }
    directory = collection.root / "meshes" if cache_directory is None else cache_directory
    conditioned_cache = directory / f"{digest(identity)}.npz"
    cache = conditioned_cache if conditioned else directory / f"{digest({**identity, 'source_only': True})}.npz"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
        return _display_geometry(collection, row, result, True, started)
    if not conditioned and conditioned_cache.is_file():
        # The conditioned cache records the separately retained raw contour
        # with the same source, mask, affine and field policy. Reuse it verbatim.
        with np.load(conditioned_cache, allow_pickle=False) as archive:
            raw, faces = archive["source"], archive["indices"]
            field = archive["surface_field"] if "surface_field" in archive else np.array("categorical")
        result = {"source": raw, "positions": raw, "indices": faces, "normals": normals(raw, faces),
                  "max_displacement_mm": np.array(0.), "policy_version": np.array("native-source-isosurface-v1"),
                  "surface_field": field}
        return _display_geometry(collection, row, result, True, started)
    mask = collection.mask(row)
    bounds = [np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis))) for axis in range(3)]
    if not len(bounds[0]):
        return {
            "source": np.empty((0, 3), np.float32),
            "positions": np.empty((0, 3), np.float32),
            "normals": np.empty((0, 3), np.float32),
            "indices": np.empty((0, 3), np.uint32),
            "max_displacement_mm": np.array(0.0),
            "policy_version": version,
            "cache_hit": False,
            "preparation_ms": (perf_counter()-started)*1000,
        }
    lo = np.maximum(0, np.array([axis[0] for axis in bounds]) - 2)
    hi = np.minimum(mask.shape, np.array([axis[-1] for axis in bounds]) + 3)
    slices = tuple(slice(a, b) for a, b in zip(lo, hi))
    cropped = mask[slices]
    values, field = _field_values(collection, row, cropped, slices, display_source)
    values = np.pad(values, 1)
    if cancelled():
        raise InterruptedError("Mesh preparation cancelled")
    raw, faces, degenerate = _indexed_contour(values, field != "categorical")
    a = affine(row["affine"])
    # Quantize in final source coordinates before all geometry safety checks.
    offset = lo - 1
    raw = (raw + offset).astype(np.float32)
    if conditioned:
        smoothed, diagnostics = fair_surface(
            raw, faces, a, display_kind, mask=np.pad(cropped, 1) if profile == "vessel" else None,
            mask_origin=offset, native=native, source_shape=mask.shape,
            categorical=field == "categorical", cancelled=cancelled,
        )
        if brain_signal is not None and not diagnostics["restoredComponents"]:
            from .anatomy_brain_surface import MAX_TRIANGLES, refine_brain_envelope

            if len(faces) <= MAX_TRIANGLES:
                refined = refine_brain_envelope(
                    raw, smoothed, faces, a[:3, :3], mask, collection.array(brain_signal), cancelled=cancelled,
                )
                if refined is not None:
                    smoothed, backed_off = refined
                    version = BRAIN_ENVELOPE_POLICY
                    diagnostics["backedOffVertices"] += backed_off
                    diagnostics["maxDisplacementMm"] = np.linalg.norm(
                        (smoothed - raw) @ a[:3, :3].T, axis=1,
                    ).max(initial=0.)
    else:
        smoothed, diagnostics = raw, {"restoredComponents": 0, "backedOffVertices": 0, "maxDisplacementMm": 0.}
    rejected = diagnostics["restoredComponents"] > 0
    result = {
        "source": raw,
        "positions": smoothed,
        "normals": normals(smoothed, faces),
        "indices": faces,
        "smoothing_rejected": np.array(rejected),
        "restored_components": np.array(diagnostics["restoredComponents"]),
        "backed_off_vertices": np.array(diagnostics["backedOffVertices"]),
        "policy_version": np.array(version if conditioned else "native-source-isosurface-v1"),
        "fallback_reason": np.array(diagnostics.get("fallbackReason", "")),
        "source_degenerate_faces_removed": np.array(degenerate),
        "max_displacement_mm": np.array(diagnostics["maxDisplacementMm"]),
        "surface_field": np.array(field),
    }
    if "maxComponentVolumeChangeFraction" in diagnostics:
        result["max_component_volume_change_fraction"] = np.array(diagnostics["maxComponentVolumeChangeFraction"])
        result["volume_restored_components"] = np.array(diagnostics["volumeRestoredComponents"])
    if cancelled():
        raise InterruptedError("Mesh preparation cancelled before cache admission")
    _save_cache(cache, result)
    return _display_geometry(collection, row, result, False, started)


def vessel_graph(mask, native_affine):
    """Source-owned Lee skeleton with explicit topology/coverage diagnostics.

    This is annotation geometry, not a measured vascular network. Diagonal
    skeleton adjacency is distinguished from a local six-face foreground path.
    Neither category estimates lumen, conductance, flow or missing branches.
    """
    a = affine(native_affine)
    mask = np.asarray(mask)
    if mask.ndim != 3 or not all(mask.shape) or mask.dtype.kind not in 'bui':
        raise ValueError('Vessel graph requires an explicit binary source mask')
    if mask.size > 128_000_000:
        raise ValueError('Vessel graph source exceeds declared grid cap')
    if mask.dtype.kind != 'b' and not np.isin(mask, [0, 1]).all():
        raise ValueError('Vessel graph requires an explicit binary source mask')
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    if not np.allclose(
        (a[:3, :3] / spacing).T @ (a[:3, :3] / spacing), np.eye(3), atol=1e-5
    ):
        raise ValueError("Radius extraction requires an orthogonal source grid")
    if np.count_nonzero(mask) > 1_000_000:
        raise ValueError('Vessel graph foreground exceeds declared one-million-cell cap')
    source_ids = np.flatnonzero(mask)
    source_ijk = np.column_stack(np.unravel_index(source_ids, mask.shape))
    lower = source_ijk.min(axis=0) if len(source_ids) else np.zeros(3, int)
    upper = source_ijk.max(axis=0)+1 if len(source_ids) else np.ones(3, int)
    if int(np.prod(upper-lower+2)) > 32_000_000:
        raise ValueError('Vessel graph cropped workspace exceeds declared cap')
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    # Padding represents unassessed exterior for thinning/EDT only, never a
    # negative anatomical finding. Boundary-limited distances are flagged.
    work = np.pad(mask[crop].astype(bool), 1)
    from skimage.morphology import skeletonize
    skeleton = skeletonize(work, method='lee')
    local_indices = np.argwhere(skeleton)
    if len(local_indices) > 100_000:
        raise ValueError('Vessel graph skeleton exceeds declared node cap')
    indices = local_indices+lower-1
    lookup = {tuple(p): i for i, p in enumerate(indices)}
    edges = []
    for i, p in enumerate(indices):
        for delta in np.ndindex(3, 3, 3):
            q = tuple(p + np.array(delta) - 1)
            j = lookup.get(q)
            if j is not None and j > i:
                edges.append((i, j))
                if len(edges) > 1_000_000:
                    raise ValueError('Vessel graph skeleton exceeds declared edge cap')
    edges = np.array(edges, dtype=np.uint32).reshape(-1, 2)
    source6, count6 = ndi.label(work, structure=ndi.generate_binary_structure(3, 1))
    source26, count26 = ndi.label(work, structure=np.ones((3, 3, 3)))
    sampled6 = source6[tuple(local_indices.T)]
    sampled26 = source26[tuple(local_indices.T)]
    represented = set(map(int, sampled26))
    represented6 = set(map(int, sampled6))
    lost = [i for i in range(1, count26+1) if i not in represented]
    lost6 = [i for i in range(1, count6+1) if i not in represented6]
    sizes = np.bincount(source26.ravel())
    sizes6 = np.bincount(source6.ravel())
    local_supported = []
    for left, right in edges:
        start, end = local_indices[[left, right]]
        # At most eight source voxels; no path outside the edge's local box.
        lo, hi = np.minimum(start, end), np.maximum(start, end)
        visited = {tuple(start)}; pending = [tuple(start)]
        while pending:
            cell = np.array(pending.pop())
            for axis in range(3):
                for sign in (-1, 1):
                    other = cell.copy(); other[axis] += sign; key = tuple(other)
                    if (np.all(other >= lo) and np.all(other <= hi) and work[key] and key not in visited):
                        visited.add(key); pending.append(key)
        local_supported.append(tuple(end) in visited)
    radius = ndi.distance_transform_edt(work, sampling=spacing)[tuple(local_indices.T)]
    edge_face = np.abs(indices[edges[:, 0]]-indices[edges[:, 1]]).sum(axis=1) == 1
    boundary_distance = np.minimum((indices+.5)*spacing, (np.array(mask.shape)-.5-indices)*spacing).min(axis=1)
    _, skeleton_components = ndi.label(skeleton, structure=np.ones((3, 3, 3)))
    from skimage.measure import euler_number
    source_euler = int(euler_number(work, connectivity=3))
    skeleton_euler = int(euler_number(skeleton, connectivity=3))
    # Euler of a voxel complex differs from V-E of its all-26-neighbor graph:
    # corner/edge/face cliques can add spurious graph cycles. Expose both.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adjacency = coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(indices),len(indices))).tocsr()
    graph_components = int(connected_components(adjacency,directed=False,return_labels=False))
    graph_cycles = len(edges)-len(indices)+graph_components
    _, background_components = ndi.label(~skeleton,structure=ndi.generate_binary_structure(3,1))
    cavities = int(background_components)-1
    voxel_cycles = int(skeleton_components)+cavities-skeleton_euler
    _, source_background_components = ndi.label(~work,structure=ndi.generate_binary_structure(3,1))
    source_cavities = int(source_background_components)-1
    source_cycles = int(count26)+source_cavities-source_euler
    from .vascular_homology import representative_subgraph, HomologyLimit
    try:
        representative = representative_subgraph(indices,edges)
        representative_fields = {'h1_'+key:value for key,value in representative.items()}
        representative_fields['h1_status'] = np.array('certified-H0-H1-over-GF2')
        representative_fields['h1_matches_source_betti'] = (representative['components'] == count26
            and representative['betti1'] == source_cycles and source_cavities == 0)
    except HomologyLimit as exc:
        # Raw source diagnostics remain useful, but a partial basis must never
        # masquerade as a representative graph or certify a subset of cycles.
        representative_fields = {'h1_status':np.array('cap-abstained'), 'h1_cap_reason':np.array(str(exc)),
            'h1_matches_source_betti':False}
    return {
        'points_ras_mm': world(a, indices).astype(np.float64),
        'source_cell_ids': np.ravel_multi_index(indices.T, mask.shape).astype(np.int64),
        'source_shape': np.array(mask.shape, np.int64), 'source_affine_ras_mm': a.copy(),
        'edges': edges, 'edge_face_adjacent': edge_face,
        'edge_local_face_supported': np.asarray(local_supported, bool),
        'edge_same_source6_component': sampled6[edges[:,0]] == sampled6[edges[:,1]],
        'source_component6': sampled6, 'source_component26': sampled26,
        'radius_mm': radius.astype(np.float64),
        'radius_limited_by_image_extent': radius >= boundary_distance,
        'components': int(count26), 'source_components6': int(count6),
        'skeleton_components26': int(skeleton_components),
        'source_euler26': source_euler, 'skeleton_euler26': skeleton_euler,
        'unrepresented_source_components26': len(lost),
        'unrepresented_source_components6': count6-len(represented6),
        'unrepresented_source26_voxels': int(sizes[lost].sum()) if lost else 0,
        'unrepresented_source6_voxels': int(sizes6[lost6].sum()) if lost6 else 0,
        'source_foreground_voxels': len(source_ids),
        'source_boundary_foreground_voxels': int(np.any((source_ijk == 0) | (source_ijk == np.array(mask.shape)-1), axis=1).sum()),
        'voxel_skeleton_betti_preserved26': not lost and count26 == skeleton_components and source_euler == skeleton_euler and source_cavities == cavities,
        'edge_graph_components': graph_components, 'edge_graph_cycle_rank': graph_cycles,
        'edge_graph_euler': len(indices)-len(edges), 'voxel_skeleton_cycle_rank26': voxel_cycles,
        'voxel_skeleton_cavities6_background': cavities,
        'source_cavities6_background': source_cavities, 'source_cycle_rank26': source_cycles,
        'edge_graph_matches_voxel_betti': graph_components == skeleton_components and graph_cycles == voxel_cycles and cavities == 0,
        **representative_fields,
        'meaning': 'Derived source-voxel skeleton and 26-neighbor adjacency. Voxel and edge-graph topology are separate checks. Annotation-to-background-center distance is not measured lumen radius. No inferred flow.',
    }


def source_branch_cache_path(collection, source):
    return collection.path(f'graphs/{source["data"]["sha256"]}.native-tof-source-branches-v1.json')


def read_source_branch_graph(collection, source, path):
    """Validate prepared branch evidence against the original acquisition and mask."""
    from .vascular_source_branches import _mask_digest
    if path.stat().st_size > 64*1024**2:
        raise ValueError("Source branch attachment exceeds the inspection budget")
    graph = json.loads(path.read_text())
    scan = collection.series(source["source_series"])
    if (graph.get("algorithm") != "native-tof-source-branches-v1"
            or graph.get("source_scan_sha256") != scan["original"]["sha256"]
            or graph.get("source_shape") != source["data"]["shape"]
            or not np.array_equal(graph.get("source_affine_ras_mm"), source["affine"])
            or graph.get("source_prediction_sha256") != _mask_digest(collection.mask(source))):
        raise ValueError("Source branch attachment belongs to different anatomy")
    return graph


def _source_branch_query(collection, row, point, displayed_affine):
    """Reuse the native branch extraction and query paths admitted by this layer."""
    from scipy.spatial import cKDTree

    source = vascular_cleanup_source(collection, row) or row
    path = source_branch_cache_path(collection, source)
    if not path.is_file():
        from .data_root import atomic_json
        from .vascular_source_branches import build_source_branches

        source_scan = collection.series(source['source_series'])
        graph = build_source_branches(collection.mask(source), source['affine'],
                                      source_scan_sha256=source_scan['original']['sha256'])
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, graph)
    stat = path.stat()
    cache_key = (row["data"]["sha256"], str(path), stat.st_mtime_ns, displayed_affine.tobytes())
    cache = getattr(collection, "_attached_vessel_branch_cache", None)
    if not cache or cache[0] != cache_key:
        attachment_sha = file_hash(path)
        graph = read_source_branch_graph(collection, source, path)
        mask = collection.mask(row).ravel()
        branches = []
        points = []
        cells = []
        present = {}
        for branch in graph["branches"]:
            ids = np.asarray(branch["source_cell_ids"])
            if not ids.size or ids.dtype.kind not in "ui" or ids.min() < 0 or ids.max() >= mask.size:
                raise ValueError("Invalid source branch cell binding")
            if not mask[ids].all():
                continue
            indices = np.column_stack(np.unravel_index(ids, tuple(graph["source_shape"])))
            native_positions = world(affine(row["affine"]), indices)
            if not np.allclose(native_positions, branch["points_ras_mm"], atol=1e-5, rtol=0):
                raise ValueError("Source branch coordinates differ from their cells")
            positions = world(displayed_affine, indices)
            points.extend(positions)
            cells.extend(ids.tolist())
            for offset, cell in enumerate(ids.tolist()):
                present.setdefault(cell, []).append((len(branches), offset))
            branches.append((branch, positions))
        tree = cKDTree(np.asarray(points).reshape(-1, 3))
        cache = (cache_key, tree, cells, present, branches, attachment_sha)
        collection._attached_vessel_branch_cache = cache
    _, tree, cells, present, branch_paths, attachment_sha = cache
    result = {"source_graph_sha256": attachment_sha, "branches": []}
    if not cells:
        return result
    cursor_distance, index = tree.query(point)
    cell = cells[index]
    for branch_index, offset in present[cell]:
        branch, positions = branch_paths[branch_index]
        if "nearest" not in result:
            radius = branch["annotation_radii_mm"][offset]
            result["nearest"] = {"point_ras_mm": positions[offset].tolist(), "distance_mm": float(cursor_distance),
                                 "radius_mm": radius, "source_cell_id": cell,
                                 "radius_limited_by_image_extent": branch["radius_limited_by_image_extent"][offset],
                                 "radius_limited_by_workspace": branch["radius_limited_by_workspace"][offset]}
        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        midpoint = int(np.searchsorted(np.r_[0., np.cumsum(lengths)], lengths.sum()/2))
        preview = np.linspace(0, len(positions)-1, min(len(positions), 2048), dtype=int)
        if any(entry["branch_id"] == branch["id"] for entry in result["branches"]):
            continue
        result["branches"].append({"branch_id": branch["id"], "closed": branch["closed"],
                                   "length_mm": branch["length_mm"], "source_cells": branch["source_cell_ids"],
                                   "start_ras_mm": positions[0].tolist(), "end_ras_mm": positions[-1].tolist(),
                                   "midpoint_ras_mm": positions[midpoint].tolist(),
                                   "preview_ras_mm": positions[preview].tolist(),
                                   "unsupported_diagonal_edges": branch["edge_local_face_supported"].count(False),
                                   "connectivity": branch["topology_status"]})
    return result


def inspect_vessel_geometry(collection, layer_id, point=None, display_series=None):
    """Revision-keyed derived graph with source identity and unknown connectivity."""
    from scipy.spatial import cKDTree
    import skimage

    row = collection.layer(layer_id)
    if row["kind"] not in {"vessels", "artery", "vein", "sinus"}:
        raise ValueError("Select a vascular layer")
    from .anatomy_evidence import is_siam_context
    if is_siam_context(row):
        raise ValueError("SIAM vascular tissue supplies image context only")
    if display_series is None:
        a = collection.patient_affine(row)
        coordinate_frame = "patient"
    else:
        if display_series != row["source_series"]:
            raise ValueError("Inspect vascular branches on the layer's source acquisition")
        from .anatomy_display import display_affine
        try:
            a, estimated = display_affine(collection, row, preview=True)
            coordinate_frame = "alignment-preview" if estimated else "patient"
        except ValueError:
            collection.require_native_geometry(row)
            a = affine(row["affine"])
            coordinate_frame = "independent-source"
    if point is not None and (np.asarray(point).shape != (3,) or not np.isfinite(point).all()):
        raise ValueError("Invalid physical query")
    if point is not None and display_series is not None:
        branch_query = _source_branch_query(collection, row, point, a)
        return {"snapshot_id": collection.identity(), "layer_id": layer_id,
                "coordinate_frame": coordinate_frame, **branch_query}
    key = digest(
        {
            "algorithm": "source-owned-skeleton-26-v4",
            "skimage_version": skimage.__version__,
            "data": row["data"],
            "label": row["label_value"],
            "affine": a.tolist(),
        }
    )
    path = collection.root / "graphs" / f"{key}.npz"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            graph = {name: archive[name] for name in archive.files}
    else:
        graph = vessel_graph(collection.mask(row), a)
        graph = {k: v for k, v in graph.items() if isinstance(v, (np.ndarray, int))}
        _save_cache(path, graph)
    points = graph["points_ras_mm"]
    edges = graph["edges"]
    degree = np.bincount(edges.ravel(), minlength=len(points))
    result = {
        "snapshot_id": collection.identity(),
        "graph_id": key,
        "layer_id": layer_id,
        "coordinate_frame": coordinate_frame,
        "nodes": len(points),
        "edges": len(edges),
        "components": int(graph["components"]),
        "endpoints": int((degree == 1).sum()),
        "junction_voxels": int((degree > 2).sum()),
        "path": str(path),
        "source_components6": int(graph['source_components6']),
        "skeleton_components26": int(graph['skeleton_components26']),
        "voxel_skeleton_betti_preserved26": bool(graph['voxel_skeleton_betti_preserved26']),
        "edge_graph_cycle_rank": int(graph['edge_graph_cycle_rank']),
        "voxel_skeleton_cycle_rank26": int(graph['voxel_skeleton_cycle_rank26']),
        "edge_graph_matches_voxel_betti": bool(graph['edge_graph_matches_voxel_betti']),
        "unrepresented_source_components26": int(graph['unrepresented_source_components26']),
        "unrepresented_source_components6": int(graph['unrepresented_source_components6']),
        "unrepresented_source26_voxels": int(graph['unrepresented_source26_voxels']),
        "unrepresented_source6_voxels": int(graph['unrepresented_source6_voxels']),
        "diagonal_edges_without_local_face_support": int((~graph['edge_local_face_supported']).sum()),
        "edges_across_source6_components": int((~graph['edge_same_source6_component']).sum()),
        "source_boundary_foreground_voxels": int(graph['source_boundary_foreground_voxels']),
        "h1_status": str(graph['h1_status']),
        "h1_representative_edges": len(graph['h1_edges']) if 'h1_edges' in graph else None,
        "h1_cycle_rank": int(graph['h1_betti1']) if 'h1_betti1' in graph else None,
        "h1_matches_source_betti": bool(graph['h1_matches_source_betti']),
        "boundary_limited_distance_nodes": int(graph['radius_limited_by_image_extent'].sum()),
        "radius_is_lumen_measurement": False,
        "flow_model_admitted": False,
        "meaning": "Source-owned derived skeleton; annotation distance is not lumen radius. Diagonal-only contacts, lost components and unassessed branches remain unknown; no flow inference.",
    }
    if point is not None and len(points):
        distance, index = cKDTree(points).query(point)
        result["nearest"] = {
            "point_ras_mm": points[index].tolist(),
            "distance_mm": float(distance),
            "radius_mm": float(graph["radius_mm"][index]),
            "degree": int(degree[index]),
            "source_cell_id": int(graph['source_cell_ids'][index]),
            "radius_limited_by_image_extent": bool(graph['radius_limited_by_image_extent'][index]),
        }
    return result
