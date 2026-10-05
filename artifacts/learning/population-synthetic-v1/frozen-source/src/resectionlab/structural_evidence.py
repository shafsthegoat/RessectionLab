"""Traceable structural proposals and explicit, limited anatomical review.

A brain envelope is neither a cortical surface nor a safe entry annotation.
Proposals never become planner anatomy merely by being imported or reviewed.
The caller must separately choose an exact reviewed mask as working support.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def _hash(value: Any, name: str) -> str:
    digest = _text(value, name).removeprefix("sha256:").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return "sha256:" + digest


def _time(value: Any, name: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return value.astimezone(timezone.utc)


def structural_frame_hash(case: Any) -> str:
    return semantic_digest({"shape": list(case.mri.shape), "affine": case.affine.tolist(),
                            "frame": case.frame, "physical_units": "mm"})


@dataclass(frozen=True, slots=True)
class BrainEnvelopeReview:
    """Named attestation bound to the exact proposal, not a clinical clearance."""
    evidence_hash: str
    reviewer_id: str
    reviewed_at: datetime
    decision: str
    rationale: str
    scope: str = "research_brain_envelope_only"

    def __post_init__(self):
        object.__setattr__(self, "evidence_hash", _hash(self.evidence_hash, "evidence_hash"))
        _text(self.reviewer_id, "reviewer_id")
        _text(self.rationale, "rationale")
        object.__setattr__(self, "reviewed_at", _time(self.reviewed_at, "reviewed_at"))
        if self.decision not in {"accepted", "rejected"}:
            raise ValueError("Review decision must be accepted or rejected")
        if self.scope != "research_brain_envelope_only":
            raise ValueError("Brain-envelope review cannot grant cortical or clinical clearance")

    def to_manifest(self) -> dict:
        return {"evidence_hash": self.evidence_hash, "reviewer_id": self.reviewer_id,
                "reviewed_at": self.reviewed_at.isoformat(), "decision": self.decision,
                "rationale": self.rationale, "scope": self.scope}

    @classmethod
    def from_manifest(cls, value: Mapping) -> BrainEnvelopeReview:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class StructuralEvidence:
    evidence_id: str
    mask: np.ndarray
    source_image_hash: str
    source_frame_hash: str
    source_file_sha256: str | None
    model_sha256: str | None
    run_sha256: str
    method: str
    provenance: str = "estimated"
    review: BrainEnvelopeReview | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _text(self.evidence_id, "evidence_id")
        _text(self.method, "method")
        mask = np.asarray(self.mask)
        if mask.ndim != 3 or not mask.size or not np.all((mask == 0) | (mask == 1)) or not mask.any():
            raise ValueError("Structural evidence mask must be a nonempty binary 3D volume")
        object.__setattr__(self, "mask", immutable_array(mask, np.bool_))
        for name in ("source_image_hash", "source_frame_hash", "run_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        for name in ("source_file_sha256", "model_sha256"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _hash(getattr(self, name), name))
        if self.provenance not in {"estimated", "supplied", "simulated"}:
            raise ValueError("Structural provenance must be estimated, supplied, or simulated")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Structural evidence metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        if self.review is not None:
            if not isinstance(self.review, BrainEnvelopeReview) or self.review.evidence_hash != self.evidence_hash:
                raise ValueError("Structural review does not match the exact source, frame, mask and model")

    @property
    def mask_hash(self) -> str:
        return array_digest(self.mask)

    @property
    def review_status(self) -> str:
        return "review_required" if self.review is None else self.review.decision

    @property
    def cortical_access_permitted(self) -> bool:
        return False

    def _content(self) -> dict:
        return {"schema_version": 1, "kind": "whole_brain_envelope", "evidence_id": self.evidence_id,
                "shape": list(self.mask.shape), "mask_hash": self.mask_hash,
                "source_image_hash": self.source_image_hash, "source_frame_hash": self.source_frame_hash,
                "source_file_sha256": self.source_file_sha256, "model_sha256": self.model_sha256,
                "run_sha256": self.run_sha256, "method": self.method, "provenance": self.provenance,
                "metadata": thaw_json(self.metadata)}

    @property
    def evidence_hash(self) -> str:
        return semantic_digest(self._content())

    def to_manifest(self) -> dict:
        return {**self._content(), "evidence_hash": self.evidence_hash,
                "review": None if self.review is None else self.review.to_manifest(),
                "review_status": self.review_status, "review_required": self.review is None,
                "cortical_access_permitted": False, "clinical_deficit_probability": None}

    @classmethod
    def from_manifest(cls, value: Mapping, *, mask: np.ndarray) -> StructuralEvidence:
        value = dict(value)
        names = {"evidence_id", "source_image_hash", "source_frame_hash", "source_file_sha256",
                 "model_sha256", "run_sha256", "method", "provenance", "metadata"}
        evidence = cls(mask=mask, review=None if value.get("review") is None else BrainEnvelopeReview.from_manifest(value["review"]),
                       **{name: value[name] for name in names})
        if value != evidence.to_manifest():
            raise ValueError("Structural evidence manifest has changed or contradicts its mask/review")
        return evidence

    def assert_matches(self, case: Any) -> None:
        if (self.mask.shape != case.mri.shape or self.source_image_hash != array_digest(case.mri)
                or self.source_frame_hash != structural_frame_hash(case)):
            raise ValueError("Structural evidence belongs to another source image or physical frame")
        if self.source_file_sha256 is not None and not any(
                source.sha256 is not None and _hash(source.sha256, "source SHA") == self.source_file_sha256
                for source in case.source_refs):
            raise ValueError("Structural evidence source file is absent from case provenance")


def planning_brain_support(case: Any) -> tuple[np.ndarray | None, dict]:
    """Require an explicit research assumption or exact review for supplied masks.

    This permits hypothetical research support only. Even an accepted envelope
    never certifies a cortical surface or grants cortical access permission.
    """
    if case.brain_mask is None:
        return None, {}
    base = {"source": array_digest(case.brain_mask), "cortical_access_permitted": False}
    refs = getattr(case, "source_refs", ())
    if refs and all(source.provenance == "simulated" for source in refs):
        return case.brain_mask, {**base, "method": "explicit_simulated_tissue_support", "evidence_type": "simulated",
                                 "review_status": "not_applicable_synthetic_fixture"}
    matching = [item for item in getattr(case, "structural_evidence", {}).values() if item.mask_hash == base["source"]]
    for item in matching:
        item.assert_matches(case)
        if item.review_status == "accepted":
            return case.brain_mask, {**base, "method": "explicitly_selected_reviewed_brain_envelope",
                                     "evidence_type": item.provenance, "review_status": "accepted",
                                     "evidence_hash": item.evidence_hash, "review": item.review.to_manifest()}
    if matching:
        raise ValueError("BRAIN_MASK_REVIEW_REQUIRED: an unreviewed or rejected extraction cannot be working anatomy")
    assumption = thaw_json(case.metadata.get("brain_mask_support_assumption"))
    if isinstance(assumption, dict):
        expected = {"scope": "hypothetical_tissue_support", "mask_hash": base["source"],
                    "source_image_hash": array_digest(case.mri), "source_frame_hash": structural_frame_hash(case)}
        if any(assumption.get(key) != value for key, value in expected.items()):
            raise ValueError("BRAIN_MASK_ASSUMPTION_STALE: source, frame, mask or scope changed")
        _text(assumption.get("declared_by"), "declared_by")
        _text(assumption.get("rationale"), "rationale")
        _time(assumption.get("declared_at"), "declared_at")
        if assumption.get("cortical_access_permitted", False) is not False:
            raise ValueError("Brain support assumptions cannot grant cortical access")
        return case.brain_mask, {**base, "method": "supplied_unverified_research_assumption",
                                 "evidence_type": "estimated", "review_status": "supplied_unverified",
                                 "assumption": assumption}
    raise ValueError("BRAIN_MASK_REVIEW_REQUIRED: supplied masks require explicit source-bound research assumptions or review")
