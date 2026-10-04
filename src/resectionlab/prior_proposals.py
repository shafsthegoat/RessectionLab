"""Persistent, view-only registered population priors.

These records deliberately cannot become accepted ``RegisteredPrior`` planning
evidence. They retain registered values and atlas field-of-view coverage without
bundling source template images or claiming individual functional localization.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import itertools
import json
from zipfile import ZipFile

import numpy as np

from .core import SourceRef, array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .structural_evidence import structural_frame_hash


TEMPLATE_FRAME = "FSL_MNI152_RAS_mm"
COMPONENTS = {"motor", "phonology", "semantics", "speech_articulation"}
MAP_KINDS = {"functional_concordance", "structural_mask"}
DISPLAY_TITLES = {
    "motor_functional_concordance": "Motor · functional network",
    "phonology_functional_concordance": "Phonology · functional network",
    "semantics_functional_concordance": "Semantics · functional network",
    "speech_articulation_functional_concordance": "Speech arrest / articulation · functional network",
    "phonology_structural_mask": "Phonology · structural network mask",
    "semantics_structural_mask": "Semantics · structural network mask",
    "speech_articulation_structural_mask": "Speech arrest / articulation · structural network mask",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def _hash(value: Any, name: str) -> str:
    value = _text(value, name).removeprefix("sha256:").lower()
    if len(value) != 64 or any(letter not in "0123456789abcdef" for letter in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return "sha256:" + value


def _affine(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(result[:3, :3])) < 1e-12):
        raise ValueError(f"{name} must be a finite invertible physical affine")
    return immutable_array(result)


def _ras_affine(case: Any) -> np.ndarray:
    return np.diag([-1., -1., 1., 1.]) @ case.affine if case.frame == "LPS+" else case.affine


def _compartments_hash(compartments: Mapping) -> str:
    return semantic_digest({name: array_digest(mask) for name, mask in compartments.items()})


def _case_source_hashes(case: Any) -> set[str]:
    result = {source.sha256 for source in case.source_refs if source.sha256}
    for item in case.metadata.get("source_files", ()):
        if isinstance(item, Mapping) and item.get("sha256"):
            result.add(_hash(item["sha256"], "additional source SHA").removeprefix("sha256:"))
    return result


def _same_physical_grid(first: Any, second: Any, shape: tuple[int, ...], *, tolerance_mm: float = 0.01) -> bool:
    first, second = _affine(first, "first affine"), _affine(second, "second affine")
    corners = np.array(list(itertools.product(*[(0, size - 1) for size in shape])))
    delta = corners @ (first[:3, :3] - second[:3, :3]).T + first[:3, 3] - second[:3, 3]
    return bool(np.max(np.linalg.norm(delta, axis=1)) <= tolerance_mm)


@dataclass(frozen=True, slots=True, eq=False)
class RegisteredPriorProposal:
    proposal_id: str
    map_id: str
    component: str
    map_kind: str
    data: np.ndarray
    sampling_coverage: np.ndarray
    affine_ras_mm: np.ndarray
    source: SourceRef
    source_prior_hash: str
    source_prior_shape: tuple[int, int, int]
    source_prior_affine_ras_mm: np.ndarray
    source_image_hash: str
    source_frame_hash: str
    source_file_sha256: str
    registration_image_sha256: str
    registration_lesion_sha256: str
    registration_compartments_hash: str
    source_case_planning_hash: str
    registration_case_hash: str
    registration_hash: str
    mni_ras_to_patient_ras_mm: np.ndarray
    registration_method: str
    registration_version: str
    registration_report_sha256: str
    registration_inventory_sha256: str
    preview_sha256: str
    prior_manifest_sha256: str
    template_identity: str
    template_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _array_state: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("proposal_id", "map_id", "registration_method", "registration_version", "template_identity"):
            _text(getattr(self, name), name)
        if self.component not in COMPONENTS or self.map_kind not in MAP_KINDS:
            raise ValueError("Unknown population prior component or map kind")
        if self.map_id not in DISPLAY_TITLES or self.map_id != f"{self.component}_{self.map_kind}":
            raise ValueError("Canonical prior map identity must agree with its component and scalar meaning")
        if (not isinstance(self.source, SourceRef) or self.source.provenance != "prior"
                or self.source.native_frame != TEMPLATE_FRAME or self.source.sha256 is None):
            raise ValueError("Source must remain a hash-pinned population prior in the declared template frame")
        values, coverage = np.asarray(self.data), np.asarray(self.sampling_coverage)
        if (values.ndim != 3 or not values.size or values.dtype.kind not in "buif"
                or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
            raise ValueError("Prior values must be a finite nonempty 3D grid in [0,1]")
        if coverage.shape != values.shape or not np.all((coverage == 0) | (coverage == 1)):
            raise ValueError("Sampling coverage must be binary and share the values grid")
        coverage = coverage.astype(bool)
        if np.any(values[~coverage] != 0):
            raise ValueError("Uncovered atlas samples must not carry nonzero evidence")
        if self.map_kind == "structural_mask" and not np.all((values == 0) | (values == 1)):
            raise ValueError("Structural network masks must remain binary")
        shape = tuple(self.source_prior_shape)
        if len(shape) != 3 or any(not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0 for size in shape):
            raise ValueError("Source prior shape must contain three positive integer dimensions")
        object.__setattr__(self, "source_prior_shape", shape)
        object.__setattr__(self, "data", immutable_array(values, np.float32))
        object.__setattr__(self, "sampling_coverage", immutable_array(coverage, bool))
        for name in ("affine_ras_mm", "source_prior_affine_ras_mm", "mni_ras_to_patient_ras_mm"):
            object.__setattr__(self, name, _affine(getattr(self, name), name))
        if np.linalg.det(self.mni_ras_to_patient_ras_mm[:3, :3]) <= 0:
            raise ValueError("A population-prior registration may not reflect physical anatomy")
        for name in ("source_prior_hash", "source_image_hash", "source_frame_hash", "source_file_sha256",
                     "registration_image_sha256", "registration_lesion_sha256", "registration_compartments_hash",
                     "source_case_planning_hash", "registration_case_hash", "registration_hash",
                     "registration_report_sha256", "registration_inventory_sha256", "preview_sha256",
                     "prior_manifest_sha256", "template_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Prior proposal metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        object.__setattr__(self, "_array_state", self._array_layout())

    def _array_layout(self) -> tuple:
        return tuple((value.shape, value.dtype.str, value.strides, value.__array_interface__["data"][0])
                     for value in (self.data, self.sampling_coverage, self.affine_ras_mm,
                                   self.source_prior_affine_ras_mm, self.mni_ras_to_patient_ras_mm))

    def _assert_array_layout(self) -> None:
        if self._array_layout() != self._array_state:
            raise ValueError("Prior proposal array metadata changed after creation")

    @property
    def review_status(self) -> str:
        return "alignment_review_required"

    @property
    def view_only(self) -> bool:
        return True

    @property
    def planning_eligible(self) -> bool:
        return False

    @property
    def patient_specific_function(self) -> bool:
        return False

    @property
    def interpolation(self) -> str:
        return "nearest_neighbor" if self.map_kind == "structural_mask" else "linear"

    def _content(self) -> dict[str, Any]:
        self._assert_array_layout()
        names = ("proposal_id", "map_id", "component", "map_kind", "source_prior_hash", "source_image_hash",
                 "source_frame_hash", "source_file_sha256", "source_case_planning_hash", "registration_case_hash",
                 "registration_image_sha256", "registration_lesion_sha256", "registration_compartments_hash",
                 "registration_hash", "registration_method", "registration_version", "registration_report_sha256",
                 "registration_inventory_sha256", "preview_sha256", "prior_manifest_sha256", "template_identity", "template_sha256")
        return {**{name: getattr(self, name) for name in names}, "schema_version": 1,
                "kind": "registered_population_prior_proposal", "source": self.source.to_dict(),
                "source_prior_shape": list(self.source_prior_shape), "source_prior_affine_ras_mm": self.source_prior_affine_ras_mm.tolist(),
                "shape": list(self.data.shape), "affine_ras_mm": self.affine_ras_mm.tolist(),
                "data_hash": array_digest(self.data), "sampling_coverage_hash": array_digest(self.sampling_coverage),
                "sampling_coverage_fraction": float(self.sampling_coverage.mean()),
                "mni_ras_to_patient_ras_mm": self.mni_ras_to_patient_ras_mm.tolist(),
                "metadata": thaw_json(self.metadata), "interpolation": self.interpolation,
                "provenance": "prior", "physical_units": "mm", "map_value_units": "unitless", "template_frame": TEMPLATE_FRAME,
                "coverage_meaning": "atlas field of view, not patient functional coverage"}

    @property
    def evidence_hash(self) -> str:
        return semantic_digest(self._content())

    def to_manifest(self) -> dict[str, Any]:
        content = self._content()
        return {**content, "evidence_hash": semantic_digest(content), "review_status": self.review_status,
                "view_only": True, "planning_eligible": False, "patient_specific_function": False,
                "clinical_deficit_probability": None, "review": None, "template_arrays_embedded": False}

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any], *, data: np.ndarray,
                      sampling_coverage: np.ndarray) -> RegisteredPriorProposal:
        names = ("proposal_id", "map_id", "component", "map_kind", "source_prior_hash", "source_prior_shape",
                 "source_prior_affine_ras_mm", "source_image_hash", "source_frame_hash", "source_file_sha256",
                 "registration_image_sha256", "registration_lesion_sha256", "registration_compartments_hash",
                 "source_case_planning_hash", "registration_case_hash", "registration_hash", "affine_ras_mm",
                 "mni_ras_to_patient_ras_mm", "registration_method", "registration_version",
                 "registration_report_sha256", "registration_inventory_sha256", "preview_sha256",
                 "prior_manifest_sha256", "template_identity", "template_sha256", "metadata")
        result = cls(data=data, sampling_coverage=sampling_coverage, source=SourceRef(**value["source"]),
                     **{name: value[name] for name in names})
        if dict(value) != result.to_manifest():
            raise ValueError("Prior proposal manifest differs from its arrays, provenance or mandatory review gates")
        return result

    def assert_matches(self, case: Any) -> None:
        self._assert_array_layout()
        if (self.data.shape != case.mri.shape or self.source_image_hash != array_digest(case.mri)
                or self.source_frame_hash != structural_frame_hash(case)
                or self.source_case_planning_hash != case.planning_hash
                or not _same_physical_grid(self.affine_ras_mm, _ras_affine(case), case.mri.shape)):
            raise ValueError("Prior proposal belongs to different case inputs, image or physical frame")
        if not any(source.sha256 and _hash(source.sha256, "source SHA") == self.source_file_sha256
                   for source in case.source_refs):
            raise ValueError("Registered prior patient source file is absent from case provenance")
        if (self.registration_image_sha256.removeprefix("sha256:") not in _case_source_hashes(case)
                or self.registration_lesion_sha256.removeprefix("sha256:") not in _case_source_hashes(case)
                or self.registration_compartments_hash != _compartments_hash(case.compartments)
                or self.registration_compartments_hash != _compartments_hash(case.source_compartments)):
            raise ValueError("Registration image or exact lesion labels changed from the source-bound proposal")


def _load_json(path: Path) -> Any:
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Prior import JSON exceeds the 16 MiB local limit")
    return json.loads(path.read_text())


def _within(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Prior source or preview path escapes its declared directory")
    return result


def import_registered_prior_proposals(case: Any, *, registration_directory: str | Path,
                                      source_cache_directory: str | Path, prior_manifest_path: str | Path,
                                      source_image_path: str | Path, registration_image_path: str | Path,
                                      lesion_path: str | Path, cancelled: Callable[[], bool] | None = None) -> Any:
    """Verify saved previews by reconstructing their exact prior resampling.

    The original registration case hash remains historical provenance. Current
    patient source bytes, values and physical grid must match that registration;
    the resulting proposal is also bound to the current planning-input identity.
    Registration is never re-estimated and no alignment review is accepted.
    """
    import nibabel as nib
    from .anatomy import load_functional_prior, register_prior
    from .imaging import file_sha256, inspect_nifti

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise ValueError("PRIOR_IMPORT_CANCELLED: no partial proposals were attached")

    def bounded_image(path: Path) -> None:
        image = nib.load(path)
        if len(image.shape) != 3 or 0 in image.shape or np.prod(image.shape, dtype=object) * 4 > 512 * 1024**2:
            raise ValueError("Prior source image exceeds the 512 MiB 3D expanded float32 limit")

    check_cancelled()

    directory, cache = Path(registration_directory).resolve(), Path(source_cache_directory).resolve()
    report_path, inventory_path = _within(directory, "registration_comparison.json"), _within(directory, "functional_overlay_inventory.json")
    manifest_path, image_path = Path(prior_manifest_path).resolve(), Path(source_image_path).resolve()
    registration_image_path, lesion_path = Path(registration_image_path).resolve(), Path(lesion_path).resolve()
    report, inventory, manifest = _load_json(report_path), _load_json(inventory_path), _load_json(manifest_path)
    if (report.get("schema") != "resectionlab.prior_registration/1"
            or report.get("qc_state") != "alignment_review_required"
            or report.get("patient_specific_function") is not False
            or report.get("clinical_deficit_probability") is not None):
        raise ValueError("Only unreviewed population registration previews may be imported")
    if manifest.get("schema_version") != 1 or manifest.get("evidence_type") != "prior":
        raise ValueError("A versioned population prior source manifest is required")
    if not isinstance(inventory, list) or not 0 < len(inventory) <= 16:
        raise ValueError("Prior inventory must contain between one and 16 layers")
    anticipated_ids = set(getattr(case, "prior_proposals", {})) | {
        f"{item['map_id']}_{item['registration_hash'][-12:]}" for item in inventory}
    if len(anticipated_ids) > 16:
        raise ValueError("A case may retain at most 16 registered prior proposals")
    if case.mri.size * 5 * len(anticipated_ids) > 1024**3:
        raise ValueError("Projected registered prior arrays exceed the 1 GiB local limit")
    specifications = {item["map_id"]: item for item in manifest["maps"]}
    if len(specifications) != len(manifest["maps"]) or len({item["map_id"] for item in inventory}) != len(inventory):
        raise ValueError("Prior manifest and inventory must have unique map identifiers")
    if file_sha256(_within(directory, "implementation_snapshot.py")) != report["implementation_sha256"]:
        raise ValueError("Registration implementation snapshot differs from its recorded hash")
    source_sha = file_sha256(image_path)
    registration_image_sha = file_sha256(registration_image_path)
    if (registration_image_sha != report["patient_sha256"]
            or registration_image_sha not in _case_source_hashes(case)
            or not any(source.sha256 == source_sha for source in case.source_refs)
            or file_sha256(lesion_path) != report["lesion_sha256"]
            or not any(source.sha256 == report["lesion_sha256"] for source in case.source_refs)):
        raise ValueError("Saved registration patient/lesion source hashes do not match this case")
    for path in (image_path, registration_image_path, lesion_path):
        check_cancelled()
        bounded_image(path)
    qc = inspect_nifti(image_path)
    registration_qc, lesion_qc = inspect_nifti(registration_image_path), inspect_nifti(lesion_path)
    if (tuple(qc["shape"]) != case.mri.shape or tuple(report["patient_shape"]) != case.mri.shape
            or tuple(registration_qc["shape"]) != case.mri.shape or tuple(lesion_qc["shape"]) != case.mri.shape
            or not _same_physical_grid(qc["affine_ras_mm"], _ras_affine(case), case.mri.shape)
            or not _same_physical_grid(registration_qc["affine_ras_mm"], _ras_affine(case), case.mri.shape)
            or not _same_physical_grid(lesion_qc["affine_ras_mm"], _ras_affine(case), case.mri.shape)
            or not _same_physical_grid(report["patient_affine_ras_mm"], _ras_affine(case), case.mri.shape)):
        raise ValueError("Saved registration patient grid differs from this case")
    if not np.array_equal(nib.load(image_path).get_fdata(dtype=np.float32), case.mri):
        raise ValueError("Current display MRI values differ from the explicitly supplied display source")
    labels = nib.load(lesion_path).get_fdata(dtype=np.float32)
    conventions = case.metadata.get("label_conventions", {})
    if (not conventions or not np.isfinite(labels).all() or np.any(labels < 0)
            or not np.array_equal(labels, np.rint(labels))):
        raise ValueError("Registration import needs an explicit categorical source lesion convention")
    expected = {name: labels == int(label) for label, name in conventions.items() if int(label) > 0}
    observed_labels = {int(label) for label in np.unique(labels) if label > 0}
    if (not observed_labels.issubset({int(label) for label in conventions})
            or set(expected) != set(case.source_compartments) or set(expected) != set(case.compartments)
            or any(not np.array_equal(expected[name], case.source_compartments[name])
                   or not np.array_equal(expected[name], case.compartments[name]) for name in expected)):
        raise ValueError("Current or original compartment labels differ from the lesion used in registration")
    source_image_hash, source_frame_hash = array_digest(case.mri), structural_frame_hash(case)
    report_sha, inventory_sha, manifest_sha = file_sha256(report_path), file_sha256(inventory_path), file_sha256(manifest_path)
    proposals = dict(getattr(case, "prior_proposals", {}))
    changed = False
    for item in inventory:
        check_cancelled()
        map_id = item["map_id"]
        if map_id not in specifications:
            raise ValueError("Inventory map is absent from the pinned source manifest")
        if (item.get("qc_state") != "alignment_review_required" or item.get("review") is not None
                or item.get("patient_specific") is not False or item.get("overlay_enabled") is not False
                or item.get("clinical_deficit_probability") is not None or item.get("case_hash") != report["case_hash"]):
            raise ValueError("Inventory must preserve the original case anchor and unreviewed prior gates")
        transform = _affine(item["mni_ras_to_patient_ras_mm"], "registration transform")
        candidates = [candidate for candidate in report["candidates"]
                      if candidate.get("status") == "candidate_requires_alignment_review"
                      and candidate.get("qc_state") == "alignment_review_required"
                      and np.array_equal(np.asarray(candidate.get("mni_ras_to_patient_ras_mm")), transform)]
        if len(candidates) != 1 or item["registration_version"] != report["version"]:
            raise ValueError("Preview transform does not identify one unchanged saved registration candidate")
        specification = specifications[map_id]
        prior_path = _within(cache, specification["cache_file"])
        bounded_image(prior_path)
        prior = load_functional_prior(prior_path, specification)
        registered = register_prior(prior, target_shape=case.mri.shape, target_affine=case.affine,
                                    target_frame=case.frame, case_hash=report["case_hash"],
                                    mni_ras_to_patient_ras_mm=transform,
                                    method=item["registration_method"], version=item["registration_version"])
        # This comparison covers source metadata, frame, interpolation and coverage
        # as well as the exact legacy registration hash.
        if item != registered.inventory_item():
            raise ValueError("Inventory differs from independently reconstructed registered source evidence")
        preview_path = _within(directory, f"{map_id}_review_preview.npz")
        if preview_path.stat().st_size > 512 * 1024**2:
            raise ValueError("Prior preview exceeds the 512 MiB local import limit")
        with ZipFile(preview_path) as archive:
            if sum(member.file_size for member in archive.infolist()) > case.mri.size * 6 + 65536:
                raise ValueError("Expanded prior preview exceeds its declared native array budget")
        with np.load(preview_path, allow_pickle=False) as arrays:
            if len(arrays.files) != 3 or set(arrays.files) != {"values", "sampling_coverage", "affine_ras_mm"}:
                raise ValueError("Prior preview has unexpected array members")
            if (arrays["values"].dtype != registered.data.dtype
                    or arrays["affine_ras_mm"].dtype != registered.affine_ras_mm.dtype
                    or not np.array_equal(arrays["values"], registered.data)
                    or arrays["sampling_coverage"].dtype != np.dtype(bool)
                    or not np.array_equal(arrays["sampling_coverage"], registered.sampling_coverage)
                    or not np.array_equal(arrays["affine_ras_mm"], registered.affine_ras_mm)):
                raise ValueError("Saved prior preview differs from pinned source/transform resampling")
        metadata = {
            "title": DISPLAY_TITLES.get(map_id, map_id.replace("_", " ")),
            "source_category": "SPEECH_ARREST" if prior.component == "speech_articulation" else prior.component.upper(),
            "candidate_id": candidates[0]["candidate_id"], "source_header_qc": thaw_json(prior.header_qc),
            "registration_limitations": report["limitations"],
            "template_redistribution": "unresolved; template image bytes remain external local research inputs",
            "zero_meaning": "zero concordance or mask exclusion is not evidence of absent patient function",
            "intensity_meaning": ("Atlas concordance (unitless, 0–1): fraction of DES-seeded normative connectivity maps supporting this location; no patient deficit probability."
                                  if prior.map_kind == "functional_concordance" else
                                  "Binary mask derived from filtered normative tractography; 1=included, 0=not included in released mask; not this patient's reconstructed tract."),
            "seed_origin": "cortical DES" if prior.component == "motor" else "subcortical/white-matter DES",
            "verification": "exact saved preview equals pinned source map resampled with saved transform; no transform refitting",
        }
        proposal = RegisteredPriorProposal(
            proposal_id=f"{map_id}_{registered.registration_hash[-12:]}", map_id=map_id,
            component=prior.component, map_kind=prior.map_kind, data=registered.data,
            sampling_coverage=registered.sampling_coverage, affine_ras_mm=registered.affine_ras_mm,
            source=prior.source, source_prior_hash=prior.semantic_hash,
            source_prior_shape=prior.data.shape, source_prior_affine_ras_mm=prior.affine_ras_mm,
            source_image_hash=source_image_hash, source_frame_hash=source_frame_hash,
            source_file_sha256=source_sha, registration_image_sha256=registration_image_sha,
            registration_lesion_sha256=report["lesion_sha256"], registration_compartments_hash=_compartments_hash(case.compartments),
            source_case_planning_hash=case.planning_hash,
            registration_case_hash=report["case_hash"], registration_hash=registered.registration_hash,
            mni_ras_to_patient_ras_mm=transform, registration_method=registered.registration_method,
            registration_version=registered.registration_version, registration_report_sha256=report_sha,
            registration_inventory_sha256=inventory_sha, preview_sha256=file_sha256(preview_path),
            prior_manifest_sha256=manifest_sha, template_identity=report["template_identity"],
            template_sha256=report["template_sha256"], metadata=metadata)
        if proposal.proposal_id in proposals:
            if proposals[proposal.proposal_id].to_manifest() != proposal.to_manifest():
                raise ValueError("Prior proposal identity already exists with different provenance or arrays")
        else:
            proposals[proposal.proposal_id] = proposal
            changed = True
    check_cancelled()
    return case.revised(prior_proposals=proposals) if changed else case
