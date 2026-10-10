"""Source-bound functional fields for explicit, uncalibrated research planning.

Attaching this object is distinct from viewing a registered prior. Population
fields remain population fields, including after engineering alignment checks.
Their intensities and sampled encounters never predict a clinical deficit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import SourceRef, array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .worlds import WorldGeneratorConfig


def anatomy_identity(case: Any) -> str:
    return semantic_digest({"shape": list(case.mri.shape), "frame": case.frame,
        "affine": array_digest(case.affine),
        "compartments": {k: array_digest(v) for k, v in case.compartments.items()}})


@dataclass(frozen=True, slots=True, eq=False)
class FunctionalEvidence:
    evidence_id: str
    motor: np.ndarray | None
    language: np.ndarray | None
    motor_coverage: np.ndarray | None
    language_coverage: np.ndarray | None
    affine_ras_mm: np.ndarray
    source_image_hash: str
    case_anatomy_hash: str
    source_records: Mapping[str, Any]
    uncertainty: WorldGeneratorConfig
    mode: str = "population_prior_sensitivity"
    provenance: str = "prior"
    review_status: str = "alignment_review_required"
    language_aggregation: str = "maximum_of_declared_components"
    _array_state: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("Functional evidence requires a stable ID")
        if self.mode != "population_prior_sensitivity" or self.provenance != "prior":
            raise ValueError("This version supports explicitly population-derived sensitivity evidence only")
        if self.review_status not in {"alignment_review_required", "engineering_qc_only"}:
            raise ValueError("Engineering import cannot establish expert approval or patient function")
        if self.language_aggregation not in {"maximum_of_declared_components", "single_released_map"}:
            raise ValueError("Declare the language aggregation")
        if not isinstance(self.uncertainty, WorldGeneratorConfig):
            raise TypeError("Functional evidence requires a frozen world generator")
        for identity in (self.source_image_hash, self.case_anatomy_hash):
            if (not isinstance(identity, str) or not identity.startswith("sha256:")
                    or len(identity) != 71 or any(c not in "0123456789abcdef" for c in identity[7:])):
                raise ValueError("Evidence must bind exact image and anatomy SHA-256 identities")
        affine = np.asarray(self.affine_ras_mm, dtype=float)
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
                or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
            raise ValueError("Functional fields need an invertible RAS+ millimeter affine")
        object.__setattr__(self, "affine_ras_mm", immutable_array(affine))
        shape = None
        for name in ("motor", "language"):
            values, coverage = getattr(self, name), getattr(self, name + "_coverage")
            if values is None:
                if coverage is not None:
                    raise ValueError("Absent functional fields cannot claim coverage")
                continue
            values, coverage = np.asarray(values), np.asarray(coverage)
            if (values.ndim != 3 or not values.size or values.dtype.kind not in "buif"
                    or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("Functional fields must be finite 3D values in [0,1]")
            if coverage.shape != values.shape or not np.isin(coverage, (0, 1)).all():
                raise ValueError("Functional sampling coverage must be an aligned binary field")
            if np.any(values[~coverage.astype(bool)] != 0):
                raise ValueError("Uncovered raw samples must remain zero with separate unknown coverage")
            if shape is not None and shape != values.shape:
                raise ValueError("Functional fields must share one physical grid")
            shape = values.shape
            object.__setattr__(self, name, immutable_array(values, np.float32))
            object.__setattr__(self, name + "_coverage", immutable_array(coverage, bool))
        if shape is None:
            raise ValueError("Use structural-only mode when no functional map is supplied")
        if not isinstance(self.source_records, Mapping) or not self.source_records:
            raise ValueError("Functional fields require source and transform records")
        for name, record in self.source_records.items():
            if not isinstance(name, str) or not name or not isinstance(record, Mapping):
                raise ValueError("Source records must map IDs to evidence records")
            source = SourceRef.from_dict(record["source"])
            if source.provenance != "prior" or source.sha256 is None:
                raise ValueError("Every population source must remain hash-pinned and prior-derived")
            transform = np.asarray(record["mni_ras_to_patient_ras_mm"], dtype=float)
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
                    or np.linalg.det(transform[:3, :3]) <= 0):
                raise ValueError("A source record requires a non-reflecting template-to-patient transform")
        object.__setattr__(self, "source_records", freeze_json(self.source_records))
        object.__setattr__(self, "_array_state", self._layout())

    def _layout(self) -> tuple:
        return tuple((a.shape, a.dtype.str, a.strides, a.__array_interface__["data"][0])
            for a in (self.motor, self.language, self.motor_coverage,
                      self.language_coverage, self.affine_ras_mm) if a is not None)

    def assert_intact(self) -> None:
        if self._layout() != self._array_state:
            raise ValueError("Functional evidence array metadata changed after construction")

    def assert_matches(self, case: Any) -> None:
        self.assert_intact()
        affine = case.affine if case.frame == "RAS+" else np.diag([-1., -1., 1., 1.]) @ case.affine
        if (array_digest(case.mri) != self.source_image_hash or anatomy_identity(case) != self.case_anatomy_hash
                or not np.allclose(affine, self.affine_ras_mm, rtol=0, atol=1e-8)
                or any(a is not None and a.shape != case.mri.shape for a in (self.motor, self.language))):
            raise ValueError("Functional evidence belongs to different image, anatomy or physical frame")

    @property
    def fingerprint(self) -> str:
        return semantic_digest(self.to_manifest(include_hash=False))

    def to_manifest(self, *, include_hash: bool = True) -> dict[str, Any]:
        self.assert_intact()
        arrays = {name: None if getattr(self, name) is None else array_digest(getattr(self, name))
                  for name in ("motor", "language", "motor_coverage", "language_coverage")}
        result = {"schema_version": 1, "evidence_id": self.evidence_id,
            "mode": self.mode, "provenance": self.provenance, "review_status": self.review_status,
            "language_aggregation": self.language_aggregation,
            "source_image_hash": self.source_image_hash, "case_anatomy_hash": self.case_anatomy_hash,
            "affine_ras_mm": self.affine_ras_mm.tolist(), "array_hashes": arrays,
            "source_records": thaw_json(self.source_records), "uncertainty": thaw_json(freeze_json(self.uncertainty.to_dict())),
            "world_generator_hash": self.uncertainty.fingerprint,
            "coverage_meaning": "released_atlas_sampling_support_not_patient_functional_coverage",
            "zero_meaning": "no_signal_in_released_map_patient_function_unknown",
            "unknown_cost_policy": "unit_upper_bound_surrogate_outside_supplied_map_coverage",
            "patient_specific_function": False, "expert_approval": False,
            "clinical_deficit_probability": None, "physical_units": "mm", "value_units": "unitless_surrogate"}
        if include_hash:
            result["evidence_hash"] = semantic_digest(result)
        return result

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], **arrays: Any) -> FunctionalEvidence:
        names = ("evidence_id", "affine_ras_mm", "source_image_hash", "case_anatomy_hash",
                 "source_records", "mode", "provenance", "review_status", "language_aggregation")
        result = cls(**{name: manifest[name] for name in names}, **arrays,
                     uncertainty=WorldGeneratorConfig(**manifest["uncertainty"]))
        if result.to_manifest() != dict(manifest):
            raise ValueError("Functional evidence payload or metadata differs from its manifest")
        return result

    def planning_arrays(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Conservative declared surrogate, never an imputation of patient function."""
        self.assert_intact()
        return tuple(None if getattr(self, name) is None else immutable_array(
            np.where(getattr(self, name + "_coverage"), getattr(self, name), 1.), np.float32)
            for name in ("motor", "language"))


def population_prior_sensitivity(case: Any, *, uncertainty: WorldGeneratorConfig,
                                evidence_id: str = "registered-population-prior-sensitivity-v1") -> FunctionalEvidence:
    """Explicitly copy existing verified proposals into an unreviewed research mode.

    This does not mutate, approve or change eligibility of any original proposal.
    No fitting, source download, map normalization or mirroring takes place.
    """
    selected = {item.map_id: item for item in case.prior_proposals.values()
                if item.map_kind == "functional_concordance"}
    if not selected:
        raise ValueError("No registered functional concordance proposals are available")
    for item in selected.values():
        item.assert_matches(case)
    motor = selected.get("motor_functional_concordance")
    languages = [selected.get(name + "_functional_concordance")
                 for name in ("phonology", "semantics", "speech_articulation")]
    # A partly supplied combined language map must not hide absent components.
    if any(item is not None for item in languages) and not all(item is not None for item in languages):
        raise ValueError("Combined language sensitivity requires all three declared components")
    language = coverage = None
    if all(item is not None for item in languages):
        coverage = np.logical_and.reduce([item.sampling_coverage for item in languages])
        language = np.maximum.reduce([item.data for item in languages])
        language = np.where(coverage, language, 0)
    affine = case.affine if case.frame == "RAS+" else np.diag([-1., -1., 1., 1.]) @ case.affine
    return FunctionalEvidence(evidence_id=evidence_id, motor=None if motor is None else motor.data,
        language=language, motor_coverage=None if motor is None else motor.sampling_coverage,
        language_coverage=coverage, affine_ras_mm=affine, source_image_hash=array_digest(case.mri),
        case_anatomy_hash=anatomy_identity(case), uncertainty=uncertainty,
        source_records={name: item.to_manifest() for name, item in selected.items()})
