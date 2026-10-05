"""Opt-in permitted image preparation, independent of policy and native geometry.

Coarse arrays are conservative cell averages, not new geometry or calibrated risk.
Channel declarations cannot prove the honesty of an upstream estimator; the caller
must enforce its source/role boundary before constructing this explicit input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .core import array_digest, freeze_json, semantic_digest
from .spatial_observations import CHANNEL_NAMES, SOURCE_KINDS, TRACKS

PREPROCESSING_VERSION = "permitted-multiscale-cell-averages-v1"
MAX_SOURCE_VOXELS = 32_000_000
MAX_VIEW_AXIS = 64
_PREPARATION_KEYS = frozenset({"source_shape", "requested_local_shape", "requested_coarse_shape",
    "local_shape", "coarse_shape", "local_start_ijk", "target_bbox_min_ijk", "target_bbox_max_ijk",
    "centering_rule", "coarse_rule", "coverage_rule", "native_geometry_eligible"})
_REPORT_KEYS = _PREPARATION_KEYS | frozenset({"preprocessing_version", "source_hash",
    "source_affine_ras_mm", "source_cell_volume_mm3", "coarse_cell_volume_mm3",
    "source_cell_corners_ras_mm", "coarse_cell_corners_ras_mm", "maximum_extent_corner_error_mm",
    "nominal_target_total_mass_mm3", "target_local_retained_mass_mm3",
    "coarse_nominal_target_mass_error_mm3", "coarse_nominal_target_mass_relative_error",
    "target_local_clipped_fraction", "channel_integrals", "target_local_fingerprint",
    "whole_source_fingerprint", "clinical_deficit_probability", "proposal_effect"})


class MultiscaleInputError(ValueError):
    pass


class _FrozenArray(np.ndarray):
    def __setattr__(self, name, value):
        if name in {"shape", "dtype", "strides", "data"}:
            raise MultiscaleInputError("Image array interpretation is immutable")
        super().__setattr__(name, value)


def _freeze(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape).view(_FrozenArray)


def _layout(value):
    root = value
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if value.flags.writeable or not isinstance(root, bytes):
        raise MultiscaleInputError("Image lost immutable byte backing")
    return (id(value), id(root), value.shape, value.dtype.str, value.strides,
            value.__array_interface__["data"][0])


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise MultiscaleInputError(f"{name} requires finite real values")
    with np.errstate(over="ignore"):
        converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise MultiscaleInputError(f"{name} exceeds finite float64 representation")
    return converted


def _affine(value):
    affine = _real(value, "affine_ras_mm")
    if affine.shape != (4, 4) or not np.array_equal(affine[3], [0, 0, 0, 1]):
        raise MultiscaleInputError("A canonical homogeneous RAS+ millimetre affine is required")
    determinant = float(np.linalg.det(affine[:3, :3]))
    if not math.isfinite(determinant) or determinant == 0 or np.linalg.cond(affine[:3, :3]) > 1e8:
        raise MultiscaleInputError("Source affine must be nonsingular and numerically usable")
    return affine


def _grid_shape(value, maximum=MAX_VIEW_AXIS):
    shape = tuple(value)
    if len(shape) != 3 or any(type(n) is not int or not 1 <= n <= maximum for n in shape):
        raise MultiscaleInputError(f"Grid requires three integer lengths from 1 to {maximum}")
    return shape


def _preparation(record):
    if not isinstance(record, Mapping) or set(record) != _PREPARATION_KEYS:
        raise MultiscaleInputError("View preparation requires the exact permitted preprocessing fields")
    record = freeze_json(record)
    source_shape = _grid_shape(record["source_shape"], MAX_SOURCE_VOXELS)
    if math.prod(source_shape) > MAX_SOURCE_VOXELS:
        raise MultiscaleInputError("Source preparation exceeds the 32M-cell bound")
    for prefix in ("local", "coarse"):
        requested = _grid_shape(record[f"requested_{prefix}_shape"])
        actual = _grid_shape(record[f"{prefix}_shape"])
        if actual != tuple(min(a, b) for a, b in zip(requested, source_shape)):
            raise MultiscaleInputError("Prepared shape must equal requested shape clamped to source")
    for key in ("local_start_ijk", "target_bbox_min_ijk", "target_bbox_max_ijk"):
        p = record[key]
        if len(p) != 3 or any(type(v) is not int or v < 0 or v >= source_shape[i] for i, v in enumerate(p)):
            raise MultiscaleInputError("Prepared source indices must be valid integer triples")
    lo, hi = record["target_bbox_min_ijk"], record["target_bbox_max_ijk"]
    if any(a > b for a, b in zip(lo, hi)):
        raise MultiscaleInputError("Nominal target bounding box is reversed")
    expected_start = tuple(max(0, min((lo[i] + hi[i] - (record["local_shape"][i] - 1)) // 2,
                                    source_shape[i] - record["local_shape"][i])) for i in range(3))
    if record["local_start_ijk"] != expected_start:
        raise MultiscaleInputError("Local start must follow the fixed nominal bbox rule")
    if (record["centering_rule"] != "bbox_midpoint_lower_start_tie_then_source_clamp"
            or record["coarse_rule"] != "separable_exact_cell_overlap_whole_cell_average"
            or record["coverage_rule"] != "covered_volume_fraction_uncovered_values_zero"
            or record["native_geometry_eligible"] is not False):
        raise MultiscaleInputError("Unknown preparation semantics or geometry eligibility")
    return record


def _provenance(kind, derivation, dependencies):
    if type(kind) is not str or kind not in SOURCE_KINDS:
        raise MultiscaleInputError("Reference, outcome and reward channels are not permitted inputs")
    if type(derivation) is not str or len(derivation) > 2048:
        raise MultiscaleInputError("Channel derivation requires bounded text")
    dependencies = tuple(dependencies)
    if any(type(d) is not str or d not in CHANNEL_NAMES for d in dependencies) or len(set(dependencies)) != len(dependencies):
        raise MultiscaleInputError("Dependencies must name distinct permitted image channels")
    return {"source_kind": kind, "derivation": derivation, "derived_from": dependencies}


def _channel_semantics(name, values, provenance, track, *, fractional_cavity):
    kind = provenance["source_kind"]
    if name == "structural_intensity":
        allowed = {"synthetic_scan"} if track == "synthetic_scan" else {"observed_scan"}
        if kind not in allowed:
            raise MultiscaleInputError("Structural intensity must match its declared scan track")
    elif name == "observed_cavity":
        if kind != "observed_procedure_state" or np.any(values < 0) or np.any(values > 1):
            raise MultiscaleInputError("Cavity must be observed procedure state between zero and one")
        if not fractional_cavity and not np.isin(values, (0, 1)).all():
            raise MultiscaleInputError("Native observed cavity must be binary")
    else:
        if kind == "derived_from_scan":
            if provenance["derived_from"] != ("structural_intensity",) or not provenance["derivation"].strip():
                raise MultiscaleInputError("Estimates require a declared structural-scan derivation")
        elif not (track == "annotation_assisted" and kind == "supplied_annotation"):
            raise MultiscaleInputError("Nominal evidence requires permitted scan estimates or an explicit annotation track")
        if np.any(values < 0) or (name in {"nominal_tissue", "nominal_target"} and np.any(values > 1)):
            raise MultiscaleInputError("Nominal occupancy must be in [0,1]; functional surrogates must be nonnegative")


@dataclass(frozen=True, slots=True)
class PermittedSourceChannel:
    data: np.ndarray | None = None
    coverage: np.ndarray | None = None
    source_kind: str = "unavailable"
    derivation: str = ""
    derived_from: tuple[str, ...] = ()
    _seal: tuple = field(init=False, repr=False)

    def __post_init__(self):
        record = _provenance(self.source_kind, self.derivation, self.derived_from)
        object.__setattr__(self, "derived_from", record["derived_from"])
        if self.data is None:
            if self.source_kind != "unavailable" or self.coverage is not None or self.derived_from:
                raise MultiscaleInputError("Unavailable evidence cannot carry data, coverage or dependencies")
        else:
            raw = np.asarray(self.data)
            if raw.ndim != 3 or any(n < 1 for n in raw.shape) or raw.size > MAX_SOURCE_VOXELS:
                raise MultiscaleInputError("Full-source channel exceeds the three-dimensional 32M-cell bound")
            if self.source_kind == "unavailable":
                raise MultiscaleInputError("Unavailable evidence cannot carry data")
            values = _real(raw, "channel")
            coverage = np.ones(raw.shape, bool) if self.coverage is None else np.asarray(self.coverage)
            if coverage.dtype != bool or coverage.shape != values.shape:
                raise MultiscaleInputError("Full-source coverage requires a matching boolean grid")
            object.__setattr__(self, "data", _freeze(np.where(coverage, values, 0)))
            object.__setattr__(self, "coverage", _freeze(coverage))
        object.__setattr__(self, "_seal", self._signature())

    def _signature(self):
        layouts = () if self.data is None else (_layout(self.data), _layout(self.coverage))
        return (self.source_kind, self.derivation, self.derived_from, layouts)

    def assert_intact(self):
        if self._signature() != self._seal:
            raise MultiscaleInputError("Permitted source channel was replaced or reinterpreted")

    @property
    def provenance(self):
        self.assert_intact()
        return freeze_json(_provenance(self.source_kind, self.derivation, self.derived_from))


@dataclass(frozen=True, slots=True)
class PermittedVolumeSource:
    channels: Mapping[str, PermittedSourceChannel]
    affine_ras_mm: np.ndarray
    track: str
    source_id: str
    _seal: tuple = field(init=False, repr=False)
    _permitted_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise MultiscaleInputError("An explicit research track is required")
        if type(self.source_id) is not str or not self.source_id.strip() or len(self.source_id) > 256:
            raise MultiscaleInputError("source_id is a bounded audit label, not an actor feature")
        if not isinstance(self.channels, Mapping) or set(self.channels) - set(CHANNEL_NAMES):
            raise MultiscaleInputError("Only the exact six permitted image channel names are accepted")
        channels = {name: self.channels.get(name, PermittedSourceChannel()) for name in CHANNEL_NAMES}
        if any(type(c) is not PermittedSourceChannel for c in channels.values()):
            raise MultiscaleInputError("Full-source channels require typed permitted evidence")
        for name in ("structural_intensity", "observed_cavity"):
            if channels[name].data is None or not channels[name].coverage.any():
                raise MultiscaleInputError(f"ESSENTIAL_EVIDENCE_MISSING:{name}")
        shape = channels["structural_intensity"].data.shape
        for name, channel in channels.items():
            channel.assert_intact()
            if channel.data is not None:
                if channel.data.shape != shape:
                    raise MultiscaleInputError("Permitted channels must share the exact full-source grid")
                _channel_semantics(name, channel.data, channel.provenance, self.track, fractional_cavity=False)
        affine = _freeze(_affine(self.affine_ras_mm))
        object.__setattr__(self, "affine_ras_mm", affine)
        object.__setattr__(self, "channels", MappingProxyType(channels))
        record = {"schema": "permitted-full-source-v1", "track": self.track,
            "affine": array_digest(affine), "channels": {name: {
                "provenance": c.provenance,
                "data": None if c.data is None else array_digest(c.data),
                "coverage": None if c.coverage is None else array_digest(c.coverage),
            } for name, c in channels.items()}}
        object.__setattr__(self, "_permitted_hash", semantic_digest(record))
        object.__setattr__(self, "_seal", self._signature())

    def _signature(self):
        return (_layout(self.affine_ras_mm), self.track, self.source_id, id(self.channels),
                tuple((name, id(c)) for name, c in self.channels.items()), self._permitted_hash)

    def assert_intact(self):
        if self._signature() != self._seal:
            raise MultiscaleInputError("Permitted full-source identity or frame was replaced")
        for channel in self.channels.values():
            channel.assert_intact()

    @property
    def shape(self):
        self.assert_intact()
        return self.channels["structural_intensity"].data.shape

    @property
    def permitted_hash(self):
        self.assert_intact()
        return self._permitted_hash

    @property
    def source_hash(self):
        return self.permitted_hash


@dataclass(frozen=True, slots=True)
class PermittedView:
    kind: str
    image_channels: np.ndarray
    coverage_fraction: np.ndarray
    channel_available: np.ndarray
    affine_ras_mm: np.ndarray
    channel_provenance: Mapping
    track: str
    source_hash: str
    preparation: Mapping
    preprocessing_version: str = PREPROCESSING_VERSION
    _seal: tuple = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)
    _preprocessing_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        if self.kind not in {"target_local_native", "whole_source_coarse"} or self.preprocessing_version != PREPROCESSING_VERSION:
            raise MultiscaleInputError("Unknown multiscale view or preprocessing version")
        if type(self.track) is not str or self.track not in TRACKS:
            raise MultiscaleInputError("View requires an explicit research track")
        if type(self.source_hash) is not str or len(self.source_hash) != 71 or not self.source_hash.startswith("sha256:") or any(c not in "0123456789abcdef" for c in self.source_hash[7:]):
            raise MultiscaleInputError("View requires the computed permitted-source hash")
        images = np.asarray(self.image_channels)
        if images.ndim != 4 or images.shape[0] != len(CHANNEL_NAMES):
            raise MultiscaleInputError("Views require the exact six-channel XYZ schema")
        _grid_shape(images.shape[1:])
        images = _real(images, "view values")
        coverage = _real(self.coverage_fraction, "view coverage")
        available = np.asarray(self.channel_available)
        if coverage.shape != images.shape or available.shape != (len(CHANNEL_NAMES),) or available.dtype != bool:
            raise MultiscaleInputError("View coverage/availability schema mismatch")
        if np.any(coverage < 0) or np.any(coverage > 1) or np.any(images[coverage == 0] != 0):
            raise MultiscaleInputError("Unknown view cells must be zero with explicit coverage in [0,1]")
        if self.kind == "target_local_native" and not np.isin(coverage, (0, 1)).all():
            raise MultiscaleInputError("Native local coverage must remain binary")
        provenance = freeze_json(self.channel_provenance)
        if set(provenance) != set(CHANNEL_NAMES):
            raise MultiscaleInputError("Every view channel needs exact provenance")
        for index, name in enumerate(CHANNEL_NAMES):
            p = provenance[name]
            if not isinstance(p, Mapping) or set(p) != {"source_kind", "derivation", "derived_from"}:
                raise MultiscaleInputError("Unexpected channel provenance fields")
            _provenance(p["source_kind"], p["derivation"], p["derived_from"])
            if not available[index]:
                if p["source_kind"] != "unavailable" or p["derived_from"] or images[index].any() or coverage[index].any():
                    raise MultiscaleInputError("Unavailable evidence cannot carry values or coverage")
            else:
                _channel_semantics(name, images[index], p, self.track, fractional_cavity=self.kind == "whole_source_coarse")
        for name in ("structural_intensity", "observed_cavity"):
            if not available[CHANNEL_NAMES.index(name)]:
                raise MultiscaleInputError(f"ESSENTIAL_EVIDENCE_MISSING:{name}")
        for name, value in (("image_channels", images), ("coverage_fraction", coverage),
                            ("channel_available", available), ("affine_ras_mm", _affine(self.affine_ras_mm))):
            object.__setattr__(self, name, _freeze(value))
        object.__setattr__(self, "channel_provenance", provenance)
        preparation = _preparation(self.preparation)
        expected_shape = preparation["local_shape" if self.kind == "target_local_native" else "coarse_shape"]
        if images.shape[1:] != expected_shape:
            raise MultiscaleInputError("View image shape disagrees with its preprocessing record")
        object.__setattr__(self, "preparation", preparation)
        spec = {"version": self.preprocessing_version, "kind": self.kind, "source_hash": self.source_hash,
                "affine_hash": array_digest(self.affine_ras_mm),
                "track": self.track, "preparation": self.preparation, "provenance": provenance}
        object.__setattr__(self, "_preprocessing_hash", semantic_digest(spec))
        object.__setattr__(self, "_fingerprint", semantic_digest({"preprocessing": spec,
            "arrays": {name: array_digest(getattr(self, name)) for name in self._array_names()}}))
        object.__setattr__(self, "_seal", self._signature())

    @staticmethod
    def _array_names():
        return ("image_channels", "coverage_fraction", "channel_available", "affine_ras_mm")

    def _signature(self):
        return (tuple(_layout(getattr(self, name)) for name in self._array_names()),
            self.kind, self.track, self.source_hash, self.preprocessing_version,
            semantic_digest(self.channel_provenance), semantic_digest(self.preparation),
            self._preprocessing_hash, self._fingerprint)

    def assert_intact(self):
        if self._signature() != self._seal:
            raise MultiscaleInputError("Multiscale view content or interpretation was replaced")

    @property
    def shape(self):
        self.assert_intact()
        return self.image_channels.shape[1:]

    @property
    def fingerprint(self):
        self.assert_intact()
        return self._fingerprint

    @property
    def preprocessing_hash(self):
        self.assert_intact()
        return self._preprocessing_hash


@dataclass(frozen=True, slots=True)
class MultiscaleViews:
    target_local: PermittedView
    whole_source: PermittedView
    report: Mapping
    _seal: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.target_local) is not PermittedView or type(self.whole_source) is not PermittedView:
            raise MultiscaleInputError("Prepared multiscale views require typed outputs")
        if (self.target_local.kind != "target_local_native" or self.whole_source.kind != "whole_source_coarse"
                or self.target_local.source_hash != self.whole_source.source_hash
                or self.target_local.track != self.whole_source.track
                or self.target_local.preparation != self.whole_source.preparation):
            raise MultiscaleInputError("Multiscale views must bind the same permitted source")
        report = freeze_json(self.report)
        if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
            raise MultiscaleInputError("Multiscale report requires its exact schema")
        if (report["source_hash"] != self.target_local.source_hash
                or report["preprocessing_version"] != PREPROCESSING_VERSION
                or report["target_local_fingerprint"] != self.target_local.fingerprint
                or report["whole_source_fingerprint"] != self.whole_source.fingerprint
                or any(report[k] != self.target_local.preparation[k] for k in _PREPARATION_KEYS)
                or report["clinical_deficit_probability"] is not None
                or report["proposal_effect"] != "none_no_task_or_action_inventory_input"):
            raise MultiscaleInputError("Report identity does not match its bound multiscale views")
        affine = _affine(report["source_affine_ras_mm"])
        local_affine = affine.copy()
        local_affine[:3, 3] = affine[:3, :3] @ report["local_start_ijk"] + affine[:3, 3]
        scale = np.asarray(report["source_shape"]) / report["coarse_shape"]
        coarse_affine = affine.copy()
        coarse_affine[:3, :3] *= scale
        coarse_affine[:3, 3] = affine[:3, :3] @ ((scale - 1) / 2) + affine[:3, 3]
        if not np.array_equal(local_affine, self.target_local.affine_ras_mm) or not np.array_equal(coarse_affine, self.whole_source.affine_ras_mm):
            raise MultiscaleInputError("Report source frame does not generate the actual view affines")
        object.__setattr__(self, "report", report)
        object.__setattr__(self, "_seal", self._signature())

    def _signature(self):
        return (id(self.target_local), self.target_local.fingerprint,
                id(self.whole_source), self.whole_source.fingerprint, semantic_digest(self.report))

    def assert_intact(self):
        if self._signature() != self._seal:
            raise MultiscaleInputError("Prepared multiscale report or views were replaced")


def _axis_average(values, axis, output_length):
    """Exact rational cell overlaps, evaluated in float64 without a dense matrix."""
    source_length = values.shape[axis]
    if source_length == output_length:
        return values
    moved = np.moveaxis(values, axis, 0)
    result = np.empty((output_length, *moved.shape[1:]), dtype=np.float64)
    # Source edges are integer multiples of M and output edges multiples of N.
    # Integer overlap arithmetic prevents accidental missed small boundary cells.
    for j in range(output_length):
        lower, upper = j * source_length, (j + 1) * source_length
        first, last = lower // output_length, (upper - 1) // output_length
        if first == last:
            result[j] = moved[first]
        else:
            first_weight = (first + 1) * output_length - lower
            last_weight = upper - last * output_length
            result[j] = moved[first] * (first_weight / source_length) + moved[last] * (last_weight / source_length)
            if last > first + 1:
                result[j] += np.sum(moved[first + 1:last], axis=0, dtype=np.float64) * (output_length / source_length)
    return np.moveaxis(result, 0, axis)


def _coarse(values, shape):
    # Reduce the largest ratio first to bound intermediate array storage.
    axes = sorted(range(3), key=lambda axis: (-values.shape[axis] / shape[axis], axis))
    result = values
    for axis in axes:
        result = _axis_average(result, axis, shape[axis])
    return np.asarray(result, dtype=np.float64)


def _corners(shape, affine):
    indices = np.asarray(list(product(*[(-0.5, n - 0.5) for n in shape])), dtype=np.float64)
    return indices @ affine[:3, :3].T + affine[:3, 3]


def prepare_multiscale_views(source: PermittedVolumeSource, *, local_shape=(64, 64, 64),
                             coarse_shape=(64, 64, 64)) -> MultiscaleViews:
    """Prepare both views using only the explicit permitted source; no task or proposals.

    Local bbox midpoint ties select the lower native start, then clamp to the
    source extent. Oversized targets remain clipped and disclosed. No padding,
    normalization, estimator, clinical probability or geometry certification occurs.
    """
    if type(source) is not PermittedVolumeSource:
        raise MultiscaleInputError("Typed full-source permitted inputs are required")
    source.assert_intact()
    requested_local, requested_coarse = _grid_shape(local_shape), _grid_shape(coarse_shape)
    shape = source.shape
    local_shape = tuple(min(a, b) for a, b in zip(shape, requested_local))
    coarse_shape = tuple(min(a, b) for a, b in zip(shape, requested_coarse))
    nominal = source.channels["nominal_target"]
    if nominal.data is None or not np.any(nominal.data > 0):
        raise MultiscaleInputError("TARGET_LOCAL_UNAVAILABLE: positive covered permitted nominal target is required")
    # Axis projections avoid materializing a potentially 32M x 3 index list.
    positive = nominal.data > 0
    bounds = [np.flatnonzero(np.any(positive, axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    lo, hi = np.asarray([b[0] for b in bounds]), np.asarray([b[-1] for b in bounds])
    unclamped = (lo + hi - (np.asarray(local_shape) - 1)) // 2
    start = np.clip(unclamped, 0, np.asarray(shape) - local_shape)
    slices = tuple(slice(int(a), int(a + b)) for a, b in zip(start, local_shape))
    local_affine = np.array(source.affine_ras_mm, copy=True)
    local_affine[:3, 3] = source.affine_ras_mm[:3, :3] @ start + source.affine_ras_mm[:3, 3]
    scale = np.asarray(shape, dtype=np.float64) / coarse_shape
    coarse_affine = np.array(source.affine_ras_mm, copy=True)
    coarse_affine[:3, :3] = source.affine_ras_mm[:3, :3] * scale
    coarse_affine[:3, 3] = source.affine_ras_mm[:3, :3] @ ((scale - 1) / 2) + source.affine_ras_mm[:3, 3]
    images = [np.zeros((len(CHANNEL_NAMES), *s), np.float64) for s in (local_shape, coarse_shape)]
    coverage = [np.zeros_like(v) for v in images]
    available = np.zeros(len(CHANNEL_NAMES), bool)
    provenance = {}
    for i, name in enumerate(CHANNEL_NAMES):
        channel = source.channels[name]
        provenance[name] = channel.provenance
        if channel.data is not None:
            available[i] = True
            images[0][i] = channel.data[slices]
            coverage[0][i] = channel.coverage[slices]
            images[1][i] = _coarse(channel.data, coarse_shape)
            coverage[1][i] = _coarse(channel.coverage, coarse_shape)
    # Coverage/occupancy roundoff is bounded by float64 averaging, not extrapolation.
    for i, name in enumerate(CHANNEL_NAMES):
        np.clip(coverage[1][i], 0, 1, out=coverage[1][i])
        if name in {"nominal_tissue", "nominal_target", "observed_cavity"}:
            np.clip(images[1][i], 0, 1, out=images[1][i])
    spec = {"source_shape": shape, "requested_local_shape": requested_local,
        "requested_coarse_shape": requested_coarse, "local_shape": local_shape,
        "coarse_shape": coarse_shape, "local_start_ijk": start.tolist(),
        "target_bbox_min_ijk": lo.tolist(), "target_bbox_max_ijk": hi.tolist(),
        "centering_rule": "bbox_midpoint_lower_start_tie_then_source_clamp",
        "coarse_rule": "separable_exact_cell_overlap_whole_cell_average",
        "coverage_rule": "covered_volume_fraction_uncovered_values_zero",
        "native_geometry_eligible": False}
    local = PermittedView("target_local_native", images[0], coverage[0], available,
        local_affine, provenance, source.track, source.permitted_hash, spec)
    whole = PermittedView("whole_source_coarse", images[1], coverage[1], available,
        coarse_affine, provenance, source.track, source.permitted_hash, spec)
    cell_volume = abs(float(np.linalg.det(source.affine_ras_mm[:3, :3])))
    coarse_volume = abs(float(np.linalg.det(coarse_affine[:3, :3])))
    total = float(np.sum(nominal.data, dtype=np.float64)) * cell_volume
    kept = float(np.sum(nominal.data[slices], dtype=np.float64)) * cell_volume
    mass = {}
    for i, name in enumerate(CHANNEL_NAMES):
        channel = source.channels[name]
        mass[name] = {"available": bool(available[i]),
            "source_value_integral_mm3": None if channel.data is None else float(np.sum(channel.data, dtype=np.float64)) * cell_volume,
            "coarse_value_integral_mm3": None if channel.data is None else float(np.sum(images[1][i], dtype=np.float64)) * coarse_volume,
            "source_covered_volume_mm3": None if channel.coverage is None else float(np.count_nonzero(channel.coverage)) * cell_volume,
            "coarse_covered_volume_mm3": None if channel.coverage is None else float(np.sum(coverage[1][i], dtype=np.float64)) * coarse_volume}
    source_corners = _corners(shape, source.affine_ras_mm)
    coarse_corners = _corners(coarse_shape, coarse_affine)
    coarse_target_mass = mass["nominal_target"]["coarse_value_integral_mm3"]
    mass_error = abs(coarse_target_mass - total)
    report = {"preprocessing_version": PREPROCESSING_VERSION, "source_hash": source.permitted_hash,
        **spec, "source_affine_ras_mm": source.affine_ras_mm.tolist(),
        "source_cell_volume_mm3": cell_volume, "coarse_cell_volume_mm3": coarse_volume,
        "source_cell_corners_ras_mm": source_corners.tolist(),
        "coarse_cell_corners_ras_mm": coarse_corners.tolist(),
        "maximum_extent_corner_error_mm": float(np.max(np.linalg.norm(source_corners - coarse_corners, axis=1))),
        "nominal_target_total_mass_mm3": total, "target_local_retained_mass_mm3": kept,
        "coarse_nominal_target_mass_error_mm3": mass_error,
        "coarse_nominal_target_mass_relative_error": mass_error / total,
        "target_local_clipped_fraction": max(0.0, min(1.0, 1 - kept / total)),
        "channel_integrals": mass, "target_local_fingerprint": local.fingerprint,
        "whole_source_fingerprint": whole.fingerprint,
        "clinical_deficit_probability": None,
        "proposal_effect": "none_no_task_or_action_inventory_input"}
    source.assert_intact()
    return MultiscaleViews(local, whole, report)
