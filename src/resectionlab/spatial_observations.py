"""Small permitted-observation boundary for spatial procedural policy research.

This module accepts images, declared estimates and observed procedure state. It
has no simulator, reference anatomy, reward, native preview or oracle-mask input.
Provenance is an explicit contract, not proof that an upstream estimator obeyed
it; task-level hidden-truth invariance tests remain necessary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .core import array_digest, freeze_json, semantic_digest
from .geometry import AccessWindow, ToolGeometry

SPATIAL_OBSERVATION_VERSION = "permitted-spatial-observation-v1"
CHANNEL_NAMES = ("structural_intensity", "nominal_tissue", "nominal_target",
                 "observed_cavity", "nominal_motor", "nominal_language")
ACTION_GEOMETRY_NAMES = ("stop", "entry_x_mm", "entry_y_mm", "entry_z_mm",
    "tip_x_mm", "tip_y_mm", "tip_z_mm", "axis_x", "axis_y", "axis_z",
    "tip_radius_mm", "shaft_radius_mm", "working_length_mm", "tip_length_mm",
    "max_access_angle_deg", "tool_change")
STATE_FEATURE_NAMES = ("steps_taken", "max_steps", "current_tool_present",
    "access_center_x_mm", "access_center_y_mm", "access_center_z_mm",
    "access_normal_x", "access_normal_y", "access_normal_z", "access_radius_mm")
TRACKS = frozenset({"synthetic_scan", "annotation_assisted", "inference_only"})
SOURCE_KINDS = frozenset({"unavailable", "observed_scan", "synthetic_scan",
    "derived_from_scan", "observed_procedure_state", "supplied_annotation"})
MAX_VOLUME_VOXELS = 64 ** 3
MAX_ACTIONS = 1024


class SpatialInputError(ValueError):
    """Required permitted input is absent, malformed or outside this contract."""


class _FrozenArray(np.ndarray):
    def __setattr__(self, name, value):
        if name in {"shape", "dtype", "strides", "data"}:
            raise ValueError("Spatial observation array interpretation is immutable")
        super().__setattr__(name, value)


def _freeze(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise SpatialInputError("Object arrays are not spatial observations")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape).view(_FrozenArray)


def _layout(array):
    root = array
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if array.flags.writeable or not isinstance(root, bytes):
        raise SpatialInputError("Spatial observation lost immutable backing")
    return (id(array), id(root), array.shape, array.dtype.str, array.strides,
            array.__array_interface__["data"][0])


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SpatialInputError(f"{name} must be a bounded nonempty string")
    return value


def _point(value, name):
    array = _real(value, name, np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise SpatialInputError(f"{name} requires three finite RAS+ millimetre coordinates")
    return tuple(float(v) for v in array)


def _real(value, name, dtype=np.float32):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise SpatialInputError(f"{name} requires finite real numeric values")
    with np.errstate(over="ignore"):
        result = np.asarray(array, dtype=dtype)
    if not np.isfinite(result).all():
        raise SpatialInputError(f"{name} exceeds its finite numeric representation")
    return result


@dataclass(frozen=True)
class ObservedChannel:
    data: np.ndarray | None = None
    coverage: np.ndarray | None = None
    source_kind: str = "unavailable"
    derivation: str = ""
    derived_from: tuple[str, ...] = ()
    _layouts: tuple = field(init=False, repr=False)
    _metadata: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.source_kind, str) or self.source_kind not in SOURCE_KINDS:
            raise SpatialInputError("Unknown observation source kind; reference truth is not an input")
        if not isinstance(self.derivation, str) or len(self.derivation) > 2048:
            raise SpatialInputError("Channel derivation must be bounded text")
        dependencies = tuple(self.derived_from)
        if any(not isinstance(v, str) or v not in CHANNEL_NAMES for v in dependencies) or len(set(dependencies)) != len(dependencies):
            raise SpatialInputError("Channel dependencies must name distinct permitted channels")
        object.__setattr__(self, "derived_from", dependencies)
        object.__setattr__(self, "_metadata", (self.source_kind, self.derivation, dependencies))
        if self.data is None:
            if self.source_kind != "unavailable" or self.coverage is not None or dependencies:
                raise SpatialInputError("Unavailable channels cannot carry values, coverage or dependencies")
            object.__setattr__(self, "_layouts", ())
            return
        if self.source_kind == "unavailable":
            raise SpatialInputError("Unavailable channels cannot carry a nonempty payload")
        data = np.asarray(self.data)
        if data.ndim != 3 or any(n < 1 or n > 128 for n in data.shape) or data.size > MAX_VOLUME_VOXELS:
            raise SpatialInputError("Spatial channels require a bounded three-dimensional crop")
        if data.dtype.kind not in "biuf" or not np.isfinite(data).all():
            raise SpatialInputError("Spatial channels must contain finite real values")
        if self.coverage is None:
            coverage = np.ones(data.shape, bool)
        else:
            coverage = np.asarray(self.coverage)
            if coverage.dtype != np.bool_ or coverage.shape != data.shape:
                raise SpatialInputError("Coverage must be a matching boolean grid")
        with np.errstate(over="ignore"):
            values = np.asarray(data, dtype=np.float32)
        if not np.isfinite(values).all():
            raise SpatialInputError("Channel values exceed finite float32 representation")
        # Unknown samples cannot remain a hidden second image behind the mask.
        values = _freeze(np.where(coverage, values, 0), np.float32)
        coverage = _freeze(coverage, bool)
        object.__setattr__(self, "data", values)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "_layouts", (_layout(values), _layout(coverage)))

    def assert_intact(self):
        actual = () if self.data is None else (_layout(self.data), _layout(self.coverage))
        if actual != self._layouts or (self.source_kind, self.derivation, self.derived_from) != self._metadata:
            raise SpatialInputError("Observed channel was replaced or reinterpreted")


@dataclass(frozen=True)
class SpatialInputs:
    channels: Mapping[str, ObservedChannel]
    affine_ras_mm: np.ndarray
    track: str
    source_id: str
    _layout: tuple = field(init=False, repr=False)
    _metadata: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.track, str) or self.track not in TRACKS:
            raise SpatialInputError("Choose an explicit spatial research track")
        _text(self.source_id, "source_id")
        channels = dict(self.channels)
        if set(channels) - set(CHANNEL_NAMES) or any(not isinstance(c, ObservedChannel) for c in channels.values()):
            raise SpatialInputError("Only declared permitted image channels may enter the actor")
        for name in ("structural_intensity", "observed_cavity"):
            if name not in channels or channels[name].data is None or not channels[name].coverage.any():
                raise SpatialInputError(f"ESSENTIAL_EVIDENCE_MISSING:{name}")
        shape = channels["structural_intensity"].data.shape
        for name, channel in channels.items():
            channel.assert_intact()
            if channel.data is None:
                continue
            if channel.data.shape != shape:
                raise SpatialInputError("All spatial evidence must use the same declared physical grid")
            if name == "structural_intensity":
                allowed = {"synthetic_scan"} if self.track == "synthetic_scan" else {"observed_scan", "synthetic_scan"}
                if self.track == "inference_only":
                    allowed = {"observed_scan"}
                if channel.source_kind not in allowed:
                    raise SpatialInputError("Structural intensity must be a scan, not a reference annotation")
            elif name == "observed_cavity":
                if channel.source_kind != "observed_procedure_state" or not np.isin(channel.data, (0, 1)).all():
                    raise SpatialInputError("Cavity requires binary observed procedure state")
            elif self.track != "annotation_assisted":
                if (channel.source_kind != "derived_from_scan" or channel.derived_from != ("structural_intensity",)
                        or not channel.derivation.strip()):
                    raise SpatialInputError("Scan tracks require declared scan-derived estimates; reference annotations are forbidden")
            elif channel.source_kind not in {"derived_from_scan", "supplied_annotation"}:
                raise SpatialInputError("Annotation-assisted fields still require explicit estimate or annotation provenance")
            if name in {"nominal_tissue", "nominal_target"} and (np.any(channel.data < 0) or np.any(channel.data > 1)):
                raise SpatialInputError("Nominal tissue/target support must lie between zero and one")
            if name in {"nominal_motor", "nominal_language"} and np.any(channel.data < 0):
                raise SpatialInputError("Functional evidence must be a nonnegative surrogate, not a clinical probability")
        affine = _real(self.affine_ras_mm, "affine", np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.array_equal(affine[3], (0, 0, 0, 1)):
            raise SpatialInputError("A finite homogeneous RAS+ affine in millimetres is required")
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if np.any(spacing <= 0) or not np.allclose((affine[:3, :3] / spacing).T @ (affine[:3, :3] / spacing), np.eye(3), rtol=0, atol=1e-8):
            raise SpatialInputError("The first spatial crop supports orthogonal grids, including oblique rotations, not shear")
        affine = _freeze(affine)
        object.__setattr__(self, "affine_ras_mm", affine)
        object.__setattr__(self, "channels", MappingProxyType(channels))
        object.__setattr__(self, "_layout", _layout(affine))
        object.__setattr__(self, "_metadata", self._metadata_record())

    def _metadata_record(self):
        return (self.track, self.source_id, id(self.channels),
                tuple((name, id(value)) for name, value in self.channels.items()))

    def assert_intact(self):
        if _layout(self.affine_ras_mm) != self._layout or self._metadata_record() != self._metadata:
            raise SpatialInputError("Spatial source frame was replaced or reinterpreted")
        for channel in self.channels.values():
            channel.assert_intact()


@dataclass(frozen=True)
class SpatialAction:
    action_id: str
    entry_mm: tuple[float, float, float] | None = None
    tip_mm: tuple[float, float, float] | None = None
    tool: ToolGeometry | None = None

    def __post_init__(self):
        _text(self.action_id, "action_id", 128)
        if self.action_id == "STOP":
            if any(v is not None for v in (self.entry_mm, self.tip_mm, self.tool)):
                raise SpatialInputError("STOP carries no instrument geometry")
        else:
            if not isinstance(self.tool, ToolGeometry):
                raise SpatialInputError("A spatial action needs explicit complete-tool geometry")
            entry, tip = _point(self.entry_mm, "entry"), _point(self.tip_mm, "tip")
            if np.linalg.norm(np.subtract(tip, entry)) <= 1e-9:
                raise SpatialInputError("A spatial action needs a nonzero declared instrument axis")
            object.__setattr__(self, "entry_mm", entry)
            object.__setattr__(self, "tip_mm", tip)


@dataclass(frozen=True)
class ObservedProcedureState:
    access: AccessWindow
    steps_taken: int
    max_steps: int
    current_tool_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.access, AccessWindow):
            raise SpatialInputError("Declared physical access is required")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 1024 or type(self.steps_taken) is not int or not 0 <= self.steps_taken <= self.max_steps:
            raise SpatialInputError("Observed step count must lie within a declared finite action budget")
        if self.current_tool_id is not None:
            _text(self.current_tool_id, "current_tool_id", 128)


@dataclass(frozen=True)
class SpatialObservation:
    image_channels: np.ndarray
    coverage: np.ndarray
    channel_available: np.ndarray
    affine_ras_mm: np.ndarray
    spacing_mm: np.ndarray
    action_geometry: np.ndarray
    action_mask: np.ndarray
    state_features: np.ndarray
    action_ids: tuple[str, ...]
    action_tool_ids: tuple[str | None, ...]
    track: str
    source_id: str
    channel_provenance: Mapping[str, Mapping]
    public_target_context: object | None = None
    _layouts: tuple = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)
    _metadata: tuple = field(init=False, repr=False)

    def __post_init__(self):
        # This DTO is public: constructing/replacing it directly must not bypass
        # the permitted-input gates applied by the convenience builder.
        images = _real(self.image_channels, "image_channels")
        if images.ndim != 4 or images.shape[0] != len(CHANNEL_NAMES):
            raise SpatialInputError("Actor volumes require the exact six-channel XYZ schema")
        coverage, available = np.asarray(self.coverage), np.asarray(self.channel_available)
        if coverage.dtype != bool or coverage.shape != images.shape or available.dtype != bool or available.shape != (len(CHANNEL_NAMES),):
            raise SpatialInputError("Coverage and channel availability require exact boolean schema")
        if np.any(images[~coverage] != 0):
            raise SpatialInputError("Uncovered samples must be zero with explicit false coverage")
        if not isinstance(self.channel_provenance, Mapping) or set(self.channel_provenance) != set(CHANNEL_NAMES):
            raise SpatialInputError("Every channel requires its exact provenance record")
        channels = {}
        for index, name in enumerate(CHANNEL_NAMES):
            record = self.channel_provenance[name]
            if not isinstance(record, Mapping) or set(record) != {"source_kind", "derivation", "derived_from"}:
                raise SpatialInputError("Channel provenance has unknown or missing fields")
            if not available[index] and (coverage[index].any() or images[index].any()):
                raise SpatialInputError("Unavailable channels cannot carry values or coverage")
            channels[name] = ObservedChannel(images[index] if available[index] else None,
                coverage[index] if available[index] else None, **record)
        permitted = SpatialInputs(channels, self.affine_ras_mm, self.track, self.source_id)
        spacing = _real(self.spacing_mm, "spacing_mm", np.float64)
        if spacing.shape != (3,) or not np.allclose(spacing, np.linalg.norm(permitted.affine_ras_mm[:3, :3], axis=0), rtol=1e-8, atol=1e-12):
            raise SpatialInputError("Voxel spacing must match the declared physical affine")
        identifiers, tool_ids = tuple(self.action_ids), tuple(self.action_tool_ids)
        if not 1 <= len(identifiers) <= MAX_ACTIONS or len(tool_ids) != len(identifiers):
            raise SpatialInputError("Bounded aligned action and tool identifiers are required")
        for identifier in identifiers:
            _text(identifier, "action_id", 128)
        if identifiers[0] != "STOP" or len(set(identifiers)) != len(identifiers) or tool_ids[0] is not None:
            raise SpatialInputError("Distinct action IDs must begin with geometry-free STOP")
        for tool_id in tool_ids[1:]:
            _text(tool_id, "action_tool_id", 128)
        # Physical coordinates retain float64 until the policy transforms them
        # into normalized source-grid coordinates. Float32 RAS rounding can
        # move an exactly covered oblique-boundary ray outside the image.
        geometry = _real(self.action_geometry, "action_geometry", np.float64)
        if geometry.shape != (len(identifiers), len(ACTION_GEOMETRY_NAMES)):
            raise SpatialInputError("Action geometry requires the exact sixteen-column schema")
        if geometry[0, 0] != 1 or np.any(geometry[0, 1:] != 0) or np.any(geometry[1:, 0] != 0):
            raise SpatialInputError("Only STOP may carry the stop flag; STOP has no tool geometry")
        tool_dimensions = {}
        for row, tool_id in zip(geometry[1:], tool_ids[1:]):
            vector = row[4:7].astype(np.float64) - row[1:4]
            distance = np.linalg.norm(vector)
            if distance <= 1e-9 or not np.allclose(row[7:10], vector / distance, rtol=2e-5, atol=2e-6):
                raise SpatialInputError("Instrument axis must agree with the declared entry and tip")
            if np.any(row[10:14] <= 0) or row[13] >= row[12] or not 0 <= row[14] < 90 or row[15] not in (0, 1):
                raise SpatialInputError("Complete-tool dimensions or tool-change flag are invalid")
            dimensions = tuple(float(v) for v in row[10:15])
            if tool_id in tool_dimensions and dimensions != tool_dimensions[tool_id]:
                raise SpatialInputError("A tool ID cannot name different instrument dimensions")
            tool_dimensions[tool_id] = dimensions
        state = _real(self.state_features, "state_features", np.float64)
        if state.shape != (len(STATE_FEATURE_NAMES),):
            raise SpatialInputError("Observed procedure state requires the exact ten-column schema")
        if (state[0] != int(state[0]) or state[1] != int(state[1]) or not 0 <= state[0] <= state[1]
                or not 1 <= state[1] <= 1024 or state[2] not in (0, 1)
                or not np.isclose(np.linalg.norm(state[6:9]), 1, rtol=0, atol=2e-6) or state[9] <= 0):
            raise SpatialInputError("Observed action budget, tool presence or access geometry is invalid")
        if state[2] == 0 and np.any(geometry[1:, 15] != 0):
            raise SpatialInputError("An absent current instrument cannot trigger a tool-change flag")
        mask = np.asarray(self.action_mask)
        expected_mask = np.full(len(identifiers), state[0] < state[1], bool)
        expected_mask[0] = True
        if mask.dtype != bool or mask.shape != expected_mask.shape or not np.array_equal(mask, expected_mask):
            raise SpatialInputError("Actor masking is restricted to STOP and observed action budget")
        for name, value in (("image_channels", images), ("affine_ras_mm", permitted.affine_ras_mm),
                            ("spacing_mm", spacing), ("action_geometry", geometry), ("state_features", state)):
            object.__setattr__(self, name, value)
        names = ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm",
                 "action_geometry", "action_mask", "state_features")
        for name in names:
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        object.__setattr__(self, "action_ids", tuple(self.action_ids))
        object.__setattr__(self, "action_tool_ids", tuple(self.action_tool_ids))
        provenance = freeze_json(self.channel_provenance)
        object.__setattr__(self, "channel_provenance", provenance)
        if self.public_target_context is not None:
            from .public_target_context import PublicTargetContext
            if type(self.public_target_context) is not PublicTargetContext:
                raise SpatialInputError("Exact optional public target context required")
            self.public_target_context.require_observation(self)
        object.__setattr__(self, "_layouts", tuple(_layout(getattr(self, name)) for name in names))
        object.__setattr__(self, "_metadata", self._metadata_record())
        record = {"version": SPATIAL_OBSERVATION_VERSION, "track": self.track,
            "channels": CHANNEL_NAMES, "action_geometry": ACTION_GEOMETRY_NAMES, "state": STATE_FEATURE_NAMES,
            "arrays": {name: array_digest(getattr(self, name)) for name in names},
            "action_ids": self.action_ids, "action_tool_ids": self.action_tool_ids,
            "provenance": {k: dict(v) for k, v in provenance.items()}}
        if self.public_target_context is not None:
            record['public_target_context']=self.public_target_context.fingerprint
        object.__setattr__(self, "_fingerprint", semantic_digest(record))

    def _metadata_record(self):
        return (self.track, self.source_id, self.action_ids, self.action_tool_ids,
                tuple((name, record["source_kind"], record["derivation"], record["derived_from"])
                      for name, record in self.channel_provenance.items()),
                *((self.public_target_context.fingerprint,) if self.public_target_context is not None else ()))

    def assert_intact(self):
        names = ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm",
                 "action_geometry", "action_mask", "state_features")
        if (tuple(_layout(getattr(self, name)) for name in names) != self._layouts
                or self._metadata_record() != self._metadata):
            raise SpatialInputError("Spatial actor tensors were replaced or reinterpreted")

    @property
    def fingerprint(self):
        """Tensor/provenance identity; join source receipts separately by source_id.

        A display/source identifier is not a neural input and cannot perturb the
        tensor identity. The task must independently bind its permitted source.
        """
        self.assert_intact()
        return self._fingerprint


def build_spatial_observation(inputs: SpatialInputs, actions: Sequence[SpatialAction],
                              state: ObservedProcedureState) -> SpatialObservation:
    """Build tensors without future outcomes; masking is STOP/action-budget only.

    The upstream action generator must use permitted observations and declared
    geometry alone. Actual tool feasibility is not certified by this adapter.
    """
    if not isinstance(inputs, SpatialInputs) or not isinstance(state, ObservedProcedureState):
        raise SpatialInputError("Typed permitted inputs and observed procedure state are required")
    inputs.assert_intact()
    actions = tuple(actions)
    if not 1 <= len(actions) <= MAX_ACTIONS or any(not isinstance(a, SpatialAction) for a in actions):
        raise SpatialInputError("A bounded geometric action inventory is required")
    if actions[0].action_id != "STOP" or len({a.action_id for a in actions}) != len(actions):
        raise SpatialInputError("Distinct action identifiers with STOP first are required")
    shape = inputs.channels["structural_intensity"].data.shape
    images = np.zeros((len(CHANNEL_NAMES), *shape), np.float32)
    coverage = np.zeros(images.shape, bool)
    available = np.zeros(len(CHANNEL_NAMES), bool)
    provenance = {}
    for index, name in enumerate(CHANNEL_NAMES):
        channel = inputs.channels.get(name, ObservedChannel())
        if channel.data is not None:
            images[index], coverage[index], available[index] = channel.data, channel.coverage, True
        provenance[name] = {"source_kind": channel.source_kind, "derivation": channel.derivation,
                            "derived_from": channel.derived_from}
    geometry = np.zeros((len(actions), len(ACTION_GEOMETRY_NAMES)), np.float64)
    geometry[0, 0] = 1
    tool_ids: list[str | None] = [None]
    for index, action in enumerate(actions[1:], 1):
        vector = np.subtract(action.tip_mm, action.entry_mm)
        axis = vector / np.linalg.norm(vector)
        tool = action.tool
        geometry[index] = (0, *action.entry_mm, *action.tip_mm, *axis,
            tool.tip_radius_mm, tool.shaft_radius_mm, tool.working_length_mm, tool.tip_length_mm,
            tool.max_access_angle_deg, float(state.current_tool_id is not None and state.current_tool_id != tool.tool_id))
        tool_ids.append(tool.tool_id)
    access = state.access
    state_features = np.asarray((state.steps_taken, state.max_steps, float(state.current_tool_id is not None),
        *access.center_mm, *access.normal_inward, access.radius_mm), np.float64)
    if not np.isfinite(geometry).all() or not np.isfinite(state_features).all():
        raise SpatialInputError("Physical actor geometry exceeds finite float64 representation")
    mask = np.full(len(actions), state.steps_taken < state.max_steps, bool)
    mask[0] = True
    return SpatialObservation(images, coverage, available, inputs.affine_ras_mm,
        np.linalg.norm(inputs.affine_ras_mm[:3, :3], axis=0), geometry, mask, state_features,
        tuple(a.action_id for a in actions), tuple(tool_ids), inputs.track, inputs.source_id, provenance)
