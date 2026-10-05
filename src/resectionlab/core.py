"""Versioned patient and plan contracts for the research workspace.

Geometry uses voxel-center coordinates transformed into a declared physical
frame in millimeters. Source annotations, active annotations, accessibility,
and simulated removal have distinct representations. This module deliberately
has no clinical outcome predictor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .functional_evidence import FunctionalEvidence
    from .prior_proposals import RegisteredPriorProposal
    from .structural_evidence import StructuralEvidence


CLINICAL_RISK_UNAVAILABLE = "no_validated_clinical_outcome_model"
PROVENANCE_TYPES = frozenset({"observed", "estimated", "prior", "simulated"})
CONTEXT_EVIDENCE_TYPES = frozenset(
    {"observed", "estimated", "unknown", "not_yet_available", "scenario_assumption"}
)
OPTIMIZER_MODES = frozenset(
    {"SEARCH", "PATIENT_SCRATCH_RL", "POPULATION_FROZEN", "POPULATION_ADAPTED", "MANUAL", "STOP"}
)


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))


def freeze_json(value: Any) -> Any:
    """Copy JSON-compatible metadata into recursively immutable containers.

    Reject arbitrary objects and nonfinite numbers rather than allowing mutable
    objects or nonstandard JSON to bypass versioning.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata cannot contain nonfinite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"metadata must be JSON-compatible; got {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return a detached JSON-compatible copy of frozen metadata."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def semantic_digest(value: Any) -> str:
    """Content digest for JSON metadata; mapping insertion order is irrelevant."""
    payload = json.dumps(thaw_json(freeze_json(value)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def immutable_array(value: ArrayLike, dtype: Any = None) -> NDArray[Any]:
    """Own an immutable byte snapshot, including protection from setflags().

    Merely setting ``writeable=False`` on an owning ndarray is reversible. A
    bytes-backed ndarray prevents that loophole and isolates caller mutations.
    """
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise ValueError("object arrays are not supported")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def array_digest(value: NDArray[Any]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode())
    digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def _mask(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match MRI {shape}")
    if array.dtype.kind not in "buif" or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only boolean/binary labels")
    return immutable_array(array, np.bool_)


def _compartments(value: Mapping[str, ArrayLike], shape: tuple[int, ...], name: str) -> Mapping[str, NDArray[np.bool_]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must map compartment names to binary masks")
    result = {}
    for label, array in value.items():
        _nonempty(label, "compartment name")
        result[label] = _mask(array, shape, f"{name}.{label}")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A source artifact, whose frame may be unknown until import validation."""

    source_id: str
    uri: str
    sha256: str | None = None
    license: str | None = None
    native_frame: str = "unknown"
    provenance: str = "observed"

    def __post_init__(self) -> None:
        _nonempty(self.source_id, "source_id")
        _nonempty(self.uri, "uri")
        _nonempty(self.native_frame, "native_frame")
        if self.license is not None:
            _nonempty(self.license, "license")
        if self.provenance not in PROVENANCE_TYPES:
            raise ValueError(f"unsupported provenance: {self.provenance}")
        if self.sha256 is not None:
            _nonempty(self.sha256, "sha256")
            normalized = self.sha256.removeprefix("sha256:").lower()
            if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
                raise ValueError("sha256 must contain 64 hexadecimal digits")
            object.__setattr__(self, "sha256", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceRef:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ContextField:
    """Recorded context; existence in a dataset does not imply preoperative availability."""

    name: str
    value: Any
    source: SourceRef
    available_at: datetime | None = None
    measurement_at: datetime | None = None
    unit: str | None = None
    evidence_type: str = "observed"

    def __post_init__(self) -> None:
        _nonempty(self.name, "context field name")
        if not isinstance(self.source, SourceRef):
            raise TypeError("context source must be a SourceRef")
        if self.evidence_type not in CONTEXT_EVIDENCE_TYPES:
            raise ValueError(f"unsupported context evidence type: {self.evidence_type}")
        if self.unit is not None:
            _nonempty(self.unit, "unit")
        object.__setattr__(self, "value", freeze_json(self.value))
        for name in ("available_at", "measurement_at"):
            timestamp = getattr(self, name)
            if timestamp is not None:
                object.__setattr__(self, name, _utc(timestamp, name))
        if self.available_at is not None and self.measurement_at is not None and self.available_at < self.measurement_at:
            raise ValueError("available_at cannot precede measurement_at")

    def exclusion_reason(self, planning_as_of: datetime, *, include_scenarios: bool = False) -> str | None:
        cutoff = _utc(planning_as_of, "planning_as_of")
        if self.value is None or self.evidence_type == "unknown":
            return "unknown_value"
        if self.evidence_type == "not_yet_available":
            return "not_yet_available"
        if self.evidence_type == "scenario_assumption" and not include_scenarios:
            return "scenario_excluded_from_primary_analysis"
        if self.available_at is None:
            return "availability_time_unknown"
        if self.available_at > cutoff:
            return "available_after_planning_cutoff"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": thaw_json(self.value), "source": self.source.to_dict(),
            "available_at": None if self.available_at is None else self.available_at.isoformat(),
            "measurement_at": None if self.measurement_at is None else self.measurement_at.isoformat(),
            "unit": self.unit, "evidence_type": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextField:
        data = dict(value)
        data["source"] = SourceRef.from_dict(data["source"])
        for name in ("available_at", "measurement_at"):
            data[name] = _parse_time(data.get(name))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PatientContext:
    planning_as_of: datetime
    fields: tuple[ContextField, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "planning_as_of", _utc(self.planning_as_of, "planning_as_of"))
        object.__setattr__(self, "fields", tuple(self.fields))
        if any(not isinstance(item, ContextField) for item in self.fields):
            raise TypeError("fields must contain ContextField records")
        if len({item.name for item in self.fields}) != len(self.fields):
            raise ValueError("context field names must be unique")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("context version must be a positive integer")

    def available_fields(self, *, include_scenarios: bool = False) -> tuple[ContextField, ...]:
        return tuple(item for item in self.fields if item.exclusion_reason(self.planning_as_of, include_scenarios=include_scenarios) is None)

    def withheld_fields(self) -> tuple[ContextField, ...]:
        return tuple(item for item in self.fields if item.exclusion_reason(self.planning_as_of) is not None)

    def planner_values(self, *, include_scenarios: bool = False) -> dict[str, Any]:
        """Only permitted values. Do not send raw provenance records to an actor."""
        return {item.name: thaw_json(item.value) for item in self.available_fields(include_scenarios=include_scenarios)}

    def planning_view(self) -> dict[str, Any]:
        """Display all field names, withholding unavailable values with a reason."""
        result = {}
        for item in self.fields:
            reason = item.exclusion_reason(self.planning_as_of)
            result[item.name] = {
                "value": thaw_json(item.value) if reason is None else None,
                "unit": item.unit, "evidence_type": item.evidence_type, "exclusion_reason": reason,
            }
        return result

    @property
    def semantic_hash(self) -> str:
        data = self.to_dict()
        data["fields"] = sorted(data["fields"], key=lambda item: item["name"])
        return semantic_digest(data)

    def to_dict(self) -> dict[str, Any]:
        return {"planning_as_of": self.planning_as_of.isoformat(), "fields": [item.to_dict() for item in self.fields], "version": self.version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PatientContext:
        return cls(planning_as_of=_parse_time(value["planning_as_of"]), fields=tuple(ContextField.from_dict(item) for item in value.get("fields", ())), version=value.get("version", 1))


@dataclass(frozen=True, slots=True, eq=False)
class CaseData:
    """Immutable physical-grid case, retaining original labels after corrections.

    MRI is a selected structural volume. Compartments need not be disjoint;
    callers must name their target policy and compute union volumes explicitly.
    Empty or missing functional anatomy must be represented in ``unknowns`` or
    evidence records, never inferred as absence of functional tissue.
    """

    case_id: str
    mri: NDArray[Any]
    compartments: Mapping[str, NDArray[np.bool_]]
    affine: NDArray[np.float64]
    source_refs: tuple[SourceRef, ...]
    context: PatientContext | None = None
    revision: int = 1
    unknowns: tuple[str, ...] = ()
    frame: str = "RAS+"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_compartments: Mapping[str, NDArray[np.bool_]] | None = None
    brain_mask: NDArray[np.bool_] | None = None
    structural_evidence: Mapping[str, StructuralEvidence] = field(default_factory=dict)
    prior_proposals: Mapping[str, RegisteredPriorProposal] = field(default_factory=dict)
    functional_evidence: FunctionalEvidence | None = None
    _semantic_hash: str = field(init=False, repr=False)
    _planning_hash: str = field(init=False, repr=False)
    _prior_registration_input_hash: str = field(init=False, repr=False)
    _array_state: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _nonempty(self.case_id, "case_id")
        image = np.asarray(self.mri)
        if image.ndim != 3 or any(size == 0 for size in image.shape):
            raise ValueError("MRI must be a nonempty three-dimensional volume")
        if image.dtype.kind not in "buif" or not np.all(np.isfinite(image)):
            raise ValueError("MRI must contain finite real numeric values")
        transform = np.asarray(self.affine, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("affine must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError("affine must be a homogeneous voxel-to-world transform")
        if abs(float(np.linalg.det(transform[:3, :3]))) < 1e-12:
            raise ValueError("affine must be invertible")
        if self.frame not in {"RAS+", "LPS+"}:
            raise ValueError("frame must explicitly be RAS+ or LPS+")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("case revision must be a positive integer")
        refs = tuple(self.source_refs)
        if not refs or any(not isinstance(item, SourceRef) for item in refs):
            raise ValueError("at least one SourceRef is required, including for synthetic fixtures")
        if len({item.source_id for item in refs}) != len(refs):
            raise ValueError("source_ids must be unique within a case")
        if self.context is not None and not isinstance(self.context, PatientContext):
            raise TypeError("context must be a PatientContext or None")
        object.__setattr__(self, "mri", immutable_array(image))
        object.__setattr__(self, "affine", immutable_array(transform))
        object.__setattr__(self, "compartments", _compartments(self.compartments, image.shape, "compartments"))
        original = self.compartments if self.source_compartments is None else self.source_compartments
        object.__setattr__(self, "source_compartments", _compartments(original, image.shape, "source_compartments"))
        if self.brain_mask is not None:
            object.__setattr__(self, "brain_mask", _mask(self.brain_mask, image.shape, "brain_mask"))
        object.__setattr__(self, "source_refs", refs)
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("case metadata must be a mapping")
        unknowns = tuple(self.unknowns)
        for item in unknowns:
            _nonempty(item, "unknown factor")
        object.__setattr__(self, "unknowns", tuple(sorted(set(unknowns))))
        from .structural_evidence import StructuralEvidence
        if not isinstance(self.structural_evidence, Mapping):
            raise ValueError("structural_evidence must map IDs to StructuralEvidence")
        evidence = dict(self.structural_evidence)
        for identity, proposal in evidence.items():
            if not isinstance(proposal, StructuralEvidence) or identity != proposal.evidence_id:
                raise ValueError("Structural evidence IDs and typed records must agree")
            proposal.assert_matches(self)
        object.__setattr__(self, "structural_evidence", MappingProxyType(evidence))
        if not isinstance(self.prior_proposals, Mapping):
            raise ValueError("prior_proposals must map IDs to RegisteredPriorProposal")
        proposals = dict(self.prior_proposals)
        if proposals:
            from .prior_proposals import RegisteredPriorProposal
            for identity, proposal in proposals.items():
                if not isinstance(proposal, RegisteredPriorProposal) or identity != proposal.proposal_id:
                    raise ValueError("Prior proposal IDs and typed records must agree")
        object.__setattr__(self, "prior_proposals", MappingProxyType(proposals))
        if self.functional_evidence is not None:
            from .functional_evidence import FunctionalEvidence
            if not isinstance(self.functional_evidence, FunctionalEvidence):
                raise TypeError("functional_evidence must be typed FunctionalEvidence or None")
            self.functional_evidence.assert_matches(self)
        manifest = self.to_manifest(include_hash=False)
        object.__setattr__(self, "_semantic_hash", semantic_digest(manifest))
        object.__setattr__(self, "_planning_hash", semantic_digest(self._planning_manifest(manifest)))
        registration_manifest = self._planning_manifest(manifest)
        registration_manifest.pop("functional_evidence", None)
        object.__setattr__(self, "_prior_registration_input_hash", semantic_digest(registration_manifest))
        object.__setattr__(self, "_array_state", self._array_signature())
        # A registered preview refers to the historical registration case, while
        # current eligibility binds the unchanged image/frame/anatomy inputs.
        # Compute the proposal-excluding planning identity before checking it.
        for proposal in proposals.values():
            proposal.assert_matches(self)

    @property
    def semantic_hash(self) -> str:
        # Array contents are bytes-backed; NumPy still permits changing a view's
        # shape/dtype metadata. Such a change must never retain an old hash.
        if self._array_signature() != self._array_state:
            return semantic_digest(self.to_manifest(include_hash=False))
        return self._semantic_hash

    @property
    def planning_hash(self) -> str:
        """Seed identity containing only information permitted at planning time.

        Full provenance remains in ``semantic_hash``. Future pathology cannot
        influence primary simulator seeds through an otherwise opaque hash.
        """
        if self._array_signature() != self._array_state:
            return semantic_digest(self._planning_manifest(self.to_manifest(include_hash=False)))
        return self._planning_hash

    def _array_signature(self) -> tuple[Any, ...]:
        arrays = [self.mri, self.affine, *self.compartments.values(), *self.source_compartments.values()]
        if self.brain_mask is not None:
            arrays.append(self.brain_mask)
        arrays.extend(item.mask for item in self.structural_evidence.values())
        for item in self.prior_proposals.values():
            arrays.extend((item.data, item.sampling_coverage, item.affine_ras_mm,
                           item.source_prior_affine_ras_mm, item.mni_ras_to_patient_ras_mm))
        if self.functional_evidence is not None:
            arrays.extend(value for value in (self.functional_evidence.motor,
                self.functional_evidence.language, self.functional_evidence.motor_coverage,
                self.functional_evidence.language_coverage, self.functional_evidence.affine_ras_mm)
                if value is not None)
        return tuple((array.shape, array.dtype.str, array.strides, array.__array_interface__["data"][0]) for array in arrays)

    @property
    def prior_registration_input_hash(self) -> str:
        """Original registration inputs, unaffected by selecting copied evidence.

        The ordinary planning identity includes selected functional evidence.
        Historical view-only proposals bind their unchanged parent inputs.
        """
        if self._array_signature() == self._array_state:
            return self._prior_registration_input_hash
        manifest = self._planning_manifest(self.to_manifest(include_hash=False))
        manifest.pop("functional_evidence", None)
        return semantic_digest(manifest)

    def _planning_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(manifest)
        result.pop("revision", None)
        # Unselected structural proposals cannot perturb optimization seeds.
        # An explicitly selected working brain_mask remains in this identity.
        result.pop("structural_evidence", None)
        # Registered functional previews are view-only population evidence.
        result.pop("prior_proposals", None)
        if self.context is not None:
            result["context"] = {
                "planning_as_of": self.context.planning_as_of.isoformat(),
                "fields": [item.to_dict() for item in sorted(self.context.available_fields(), key=lambda item: item.name)],
            }
        return result

    @property
    def voxel_volume_mm3(self) -> float:
        return abs(float(np.linalg.det(self.affine[:3, :3])))

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in np.linalg.norm(self.affine[:3, :3], axis=0))

    def voxel_to_world(self, points: ArrayLike) -> NDArray[np.float64]:
        return self._transform_points(points, self.affine)

    def world_to_voxel(self, points: ArrayLike) -> NDArray[np.float64]:
        return self._transform_points(points, np.linalg.inv(self.affine))

    @staticmethod
    def _transform_points(points: ArrayLike, transform: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError("points must have a final dimension of three finite coordinates")
        return array @ transform[:3, :3].T + transform[:3, 3]

    def revised(self, **changes: Any) -> CaseData:
        """Create an edited case while retaining source annotations and lineage."""
        if "revision" in changes or "source_compartments" in changes:
            raise ValueError("revised() preserves source compartments and increments revision itself")
        return replace(self, revision=self.revision + 1, **changes)

    def to_manifest(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": 1, "case_id": self.case_id, "revision": self.revision,
            "frame": self.frame, "physical_units": "mm", "shape": list(self.mri.shape),
            "affine": self.affine.tolist(), "mri_hash": array_digest(self.mri),
            "compartments": {key: array_digest(value) for key, value in self.compartments.items()},
            "source_compartments": {key: array_digest(value) for key, value in self.source_compartments.items()},
            "brain_mask_hash": None if self.brain_mask is None else array_digest(self.brain_mask),
            "source_refs": [item.to_dict() for item in sorted(self.source_refs, key=lambda item: item.source_id)],
            "context": None if self.context is None else self.context.to_dict(),
            "unknowns": list(self.unknowns), "metadata": thaw_json(self.metadata),
        }
        if self.structural_evidence:
            result["structural_evidence"] = {key: value.to_manifest() for key, value in self.structural_evidence.items()}
        if self.prior_proposals:
            result["prior_proposals"] = {key: value.to_manifest() for key, value in self.prior_proposals.items()}
        if self.functional_evidence is not None:
            result["functional_evidence"] = self.functional_evidence.to_manifest()
        if include_hash:
            result["semantic_hash"] = self.semantic_hash
        return result


CaseState = CaseData


class StalePlanError(ValueError):
    """A plan depends on a different case version and must be recomputed."""


@dataclass(frozen=True, slots=True, eq=False)
class Plan:
    plan_id: str
    case_hash: str
    route_points_mm: NDArray[np.float64]
    tool_id: str
    plan_type: str = "route_only"
    optimizer_mode: str = "SEARCH"
    accessible_target_volume_mm3: float | None = None
    simulated_removed_target_volume_mm3: float | None = None
    clinical_deficit_probability: None = None
    clinical_risk_reason: str = CLINICAL_RISK_UNAVAILABLE
    world_model_version: str | None = None
    world_partition: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unknowns: tuple[str, ...] = ()
    removal_replay_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("plan_id", "case_hash", "tool_id", "clinical_risk_reason"):
            _nonempty(getattr(self, name), name)
        route = np.asarray(self.route_points_mm, dtype=np.float64)
        if route.ndim != 2 or route.shape[1] != 3 or not np.all(np.isfinite(route)):
            raise ValueError("route_points_mm must be an N x 3 finite array")
        if self.plan_type not in {"route_only", "simulated_resection"}:
            raise ValueError("unsupported plan_type")
        if self.optimizer_mode not in OPTIMIZER_MODES:
            raise ValueError("unsupported optimizer_mode")
        if self.clinical_deficit_probability is not None:
            raise ValueError("clinical deficit probability must remain null without a validated clinical outcome model")
        for name in ("accessible_target_volume_mm3", "simulated_removed_target_volume_mm3"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite, nonnegative, or null")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.plan_type == "route_only" and self.simulated_removed_target_volume_mm3 is not None:
            raise ValueError("route-only accessibility cannot be reported as simulated tissue removal")
        if self.simulated_removed_target_volume_mm3 is not None:
            _nonempty(self.removal_replay_hash, "removal_replay_hash")
        for name in ("world_model_version", "world_partition"):
            if getattr(self, name) is not None:
                _nonempty(getattr(self, name), name)
        object.__setattr__(self, "route_points_mm", immutable_array(route))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("plan metadata must be a mapping")
        unknowns = tuple(self.unknowns)
        for item in unknowns:
            _nonempty(item, "unknown factor")
        object.__setattr__(self, "unknowns", tuple(sorted(set(unknowns))))

    @property
    def semantic_hash(self) -> str:
        return semantic_digest(self.to_dict())

    def assert_current(self, case: CaseData) -> None:
        if self.case_hash != case.semantic_hash:
            raise StalePlanError(f"plan {self.plan_id} depends on an obsolete case version; recompute it")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id, "case_hash": self.case_hash,
            "route_points_mm": self.route_points_mm.tolist(), "tool_id": self.tool_id,
            "plan_type": self.plan_type, "optimizer_mode": self.optimizer_mode,
            "accessible_target_volume_mm3": self.accessible_target_volume_mm3,
            "simulated_removed_target_volume_mm3": self.simulated_removed_target_volume_mm3,
            "clinical_deficit_probability": None, "clinical_risk_reason": self.clinical_risk_reason,
            "world_model_version": self.world_model_version, "world_partition": self.world_partition,
            "metadata": thaw_json(self.metadata), "unknowns": list(self.unknowns),
            "removal_replay_hash": self.removal_replay_hash,
            "clinical_use_status": "research_only",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Plan:
        data = dict(value)
        if data.pop("clinical_use_status", "research_only") != "research_only":
            raise ValueError("only research_only plans are supported")
        if isinstance(data.get("route_points_mm"), list) and not data["route_points_mm"]:
            data["route_points_mm"] = np.empty((0, 3), dtype=np.float64)
        return cls(**data)
