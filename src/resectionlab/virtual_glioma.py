"""Bounded, source-bound virtual glioma geometry using pinned TumorGrowthToolkit.

The upstream FK get_D/FK_update kernels are executed unchanged. This wrapper
supplies physical grids, an explicitly qualified host, a masked Gaussian,
zero-padding/no-flux boundaries, a sufficient positivity bound, and exact step
bookkeeping. It does not call upstream solve(), whose cropping, unmasked seed,
time-series origin, and volume-stop indexing do not meet this adapter contract.

Outputs are generated concentrations and explicitly chosen threshold masks.
They are not acquired tumors, BraTS labels, edema, enhancement, mass effect,
perfusion, neurological risk, or patient-specific growth forecasts.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
from types import MappingProxyType, ModuleType
import urllib.request

import numpy as np
import scipy
from scipy.ndimage import affine_transform

from .core import CaseData, array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .rehearsal_episode import _require_development_source
from .structural_evidence import structural_frame_hash

SCHEMA = "resectionlab.virtual-glioma-fk.v1"
QUALIFICATION_SCHEMA = "resectionlab.virtual-glioma-host-qualification.v1"
UPSTREAM_COMMIT = "162c2cbbab5882fff7c3c05d9fedd87453298a0f"
UPSTREAM_VERSION = "0.2"
UPSTREAM_BASE = f"https://raw.githubusercontent.com/m1balcerak/TumorGrowthToolkit/{UPSTREAM_COMMIT}"
SOURCE_FILES = {
    "TumorGrowthToolkit/FK/FK.py": "cf66c619a42876351ca90f75ef92fe1fb8bf611cdda0f10404b3ab28cced8f88",
    "TumorGrowthToolkit/base_solver.py": "4d8df08041a8170dcb5db0d0a8e4ee1d9e016a6287875adfcecbc1ba2a80e84a",
    "LICENSE": "fe6b4ce9724b2e8ce5e281875448293931a6164c66f49f774d54b55599d8684d",
}
GM_WM_FRACTIONS = "GM_WM_FRACTIONS"
UNIFORM_DIFFUSION_DOMAIN = "UNIFORM_DIFFUSION_DOMAIN"
TISSUE_REVIEW_SCHEMA = "resectionlab.fk-categorical-tissue-pair-review.v1"
TISSUE_REVIEW_SCOPE = "research_categorical_GM_WM_support_for_FK"
MAX_SHAPE = 96
MAX_SOURCE_VOXELS = 32_000_000
MAX_STEPS = 2048
MAX_CELL_UPDATES = 180_000_000
MAX_WALL_SECONDS = 45.
# Order-1 interpolation of disjoint binary compartments may accumulate a few
# float64 rounding units outside [0, 1]. This is an admission tolerance only:
# retain the mapped values exactly; never clip either fractions or PDE results.
FRACTION_ROUNDOFF_TOLERANCE = 8 * np.finfo(np.float64).eps
DEFAULT_CACHE = Path.home() / ".cache" / "resectionlab" / "tumor-growth-toolkit" / UPSTREAM_COMMIT


def _json(value):
    return thaw_json(freeze_json(value))


def _hash(value, name):
    if not isinstance(value, str):
        raise ValueError(name + " must be a SHA-256 digest")
    result = value.removeprefix("sha256:")
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(name + " must be a SHA-256 digest")
    return result


def _number(value, name, *, minimum=0., maximum=math.inf, positive=False):
    if not isinstance(value, (float, int, np.floating, np.integer)) or isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or not minimum <= value <= maximum or positive and value <= 0:
        raise ValueError(name + " must be finite and within its declared bounds")
    return float(value)


def ensure_source_assets(cache=DEFAULT_CACHE, *, allow_download=False):
    """Fetch only three small exact-pin files; no installation or model download."""
    cache = Path(cache)
    records = {}
    for name, expected in SOURCE_FILES.items():
        path = cache / name
        url = UPSTREAM_BASE + "/" + name
        if not path.exists():
            if not allow_download:
                raise ValueError("Pinned upstream source missing; explicitly fetch source assets first: " + str(path))
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read(128 * 1024 + 1)
            if len(payload) > 128 * 1024 or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Pinned upstream source integrity/size failure: " + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if path.stat().st_size > 128 * 1024:
            raise ValueError("Cached upstream source size exceeds the bounded asset budget: " + name)
        payload = path.read_bytes()
        if len(payload) > 128 * 1024 or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Cached upstream source integrity failure: " + name)
        records[name] = {"url": url, "sha256": expected, "bytes": len(payload)}
    return {"name": "TumorGrowthToolkit FK", "version": UPSTREAM_VERSION, "commit": UPSTREAM_COMMIT,
            "license": "MIT", "copyright": "Copyright (c) 2023 Michal Balcerak", "files": records}


def load_pinned_kernel(cache=DEFAULT_CACHE):
    """Compile verified source bytes directly, avoiding unverified .pyc caches."""
    receipt = ensure_source_assets(cache)
    namespace = "_resectionlab_tgt_" + UPSTREAM_COMMIT
    names = (namespace, namespace + ".FK", namespace + ".base_solver", namespace + ".FK.FK")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names[:2]:
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        for suffix, source in ((".base_solver", "TumorGrowthToolkit/base_solver.py"), (".FK.FK", "TumorGrowthToolkit/FK/FK.py")):
            payload = (Path(cache) / source).read_bytes()
            if hashlib.sha256(payload).hexdigest() != SOURCE_FILES[source]:
                raise ValueError("Upstream source changed between verification and execution")
            module = ModuleType(namespace + suffix)
            module.__package__ = module.__name__.rpartition(".")[0]
            module.__file__ = str(Path(cache) / source)
            sys.modules[module.__name__] = module
            exec(compile(payload, module.__file__, "exec"), module.__dict__)
        solver = sys.modules[names[-1]].Solver({})
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return solver, receipt


def _admit_host_metadata(case, role):
    # Keep this first: rejection must precede hashes, masks, resampling or I/O.
    if type(case) is not CaseData:
        raise TypeError("Exact immutable CaseData host required")
    _require_development_source(case)
    if role not in {"TRAIN", "DEVELOPMENT"}:
        raise ValueError("Virtual descendants inherit an explicit TRAIN or DEVELOPMENT host role")
    split = case.metadata.get("split", {})
    declared = case.metadata.get("role", case.metadata.get("split_role"))
    if isinstance(split, Mapping):
        declared = split.get("role", declared)
    elif isinstance(split, str):
        declared = split
    normalized = str(declared).upper()
    if normalized in {"TRAIN", "TRAINING", "POPULATION_TRAINING"}:
        normalized = "TRAIN"
    elif normalized in {"DEVELOPMENT", "DEVELOPMENT_DEMO", "SINGLE_DEVELOPMENT_CASE"}:
        normalized = "DEVELOPMENT"
    if normalized != role and not (declared is None and case.metadata.get("is_synthetic") is True):
        raise ValueError("Qualification role must match the host's declared source role")


def _grid(shape, affine):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 3
            or any(type(v) is not int or not 2 <= v <= MAX_SHAPE for v in shape)):
        raise ValueError("Explicit simulation grid must have 2..96 cells per axis")
    transform = np.asarray(affine, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.array_equal(transform[3], [0., 0., 0., 1.])):
        raise ValueError("Simulation affine must be a finite homogeneous RAS+ transform")
    spacing = np.linalg.norm(transform[:3, :3], axis=0)
    if (np.any(spacing < .5) or np.any(spacing > 10.)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.diag(spacing ** 2), rtol=0, atol=1e-8)):
        raise ValueError("FK requires an orthogonal physical grid with spacing .5..10 mm")
    return tuple(shape), transform, spacing


def _voxel_mapping(source_affine, grid_affine):
    # Inverting an oblique/reflected matrix can move an exact boundary by an
    # ulp. Preserve algebraic identity only for exactly equal affines; genuinely
    # shifted/resampled grids still use their full physical mapping.
    if np.array_equal(source_affine, grid_affine):
        return np.eye(4, dtype=np.float64)
    return np.linalg.inv(source_affine) @ grid_affine


def _resample(array, source_affine, grid_affine, shape, order):
    if tuple(shape) == array.shape and np.array_equal(source_affine, grid_affine):
        return np.array(array, dtype=np.float64, copy=True)
    mapping = _voxel_mapping(source_affine, grid_affine)
    return affine_transform(np.asarray(array, dtype=np.float64), mapping[:3, :3], mapping[:3, 3],
                            output_shape=shape, order=order, mode="constant", cval=0., prefilter=False)


def _reviewed_tissue_pair(case, wm_layer, gm_layer, healthy_source_evidence):
    """Admit an exact reviewed categorical pair, never a case-wide acceptance.

    The named support and UNKNOWN masks partition the source brain envelope.
    The review seals their arrays and all four layers' source/method provenance.
    It grants neither physical partial-volume semantics nor host/clinical review.
    Generated fixtures are exempt only when every lineage condition agrees.
    """
    layers = case.metadata.get("anatomy_layers", {})
    if (not isinstance(wm_layer, str) or not isinstance(gm_layer, str) or wm_layer == gm_layer
            or wm_layer not in case.compartments or gm_layer not in case.compartments):
        raise ValueError("Strict mode requires two named attached source WM/GM layers")
    generated = (case.metadata.get("is_synthetic") is True
                 and all(ref.provenance == "simulated" for ref in case.source_refs)
                 and isinstance(healthy_source_evidence, Mapping)
                 and healthy_source_evidence.get("status") == "generated_numerical_control"
                 and all(layers.get(name, {}).get("source_role") == "generated_numerical_control"
                         for name in (wm_layer, gm_layer)))
    if generated:
        return None
    if case.metadata.get("is_synthetic") is True:
        raise ValueError("Generated tissue exemption requires exclusively simulated source and layer lineage")
    value = case.metadata.get("fk_tissue_pair_review")
    if not isinstance(value, Mapping) or set(value) != {"schema", "evidence", "evidence_hash", "review"}:
        raise ValueError("Real strict tissue admission requires an explicit pair-bound tissue review")
    record = _json(value)
    evidence, review = record["evidence"], record["review"]
    if (record["schema"] != TISSUE_REVIEW_SCHEMA or not isinstance(evidence, dict)
            or not isinstance(review, dict)
            or set(review) != {"evidence_hash", "reviewer_id", "reviewed_at", "decision", "rationale", "scope"}
            or review["scope"] != TISSUE_REVIEW_SCOPE or review["decision"] != "accepted"):
        raise ValueError("Tissue review schema, decision or limited research scope is invalid")
    for key in ("reviewer_id", "rationale", "reviewed_at"):
        if not isinstance(review[key], str) or not review[key].strip() or len(review[key]) > 4096:
            raise ValueError("Tissue review requires a named reviewer, rationale and timestamp")
    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Tissue review timestamp must be timezone-aware") from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("Tissue review timestamp must be timezone-aware")
    support_name, unknown_name = evidence.get("classification_support_layer"), evidence.get("unknown_layer")
    names = (wm_layer, gm_layer, support_name, unknown_name)
    if (any(not isinstance(name, str) or name not in case.compartments for name in names)
            or len(set(names)) != 4):
        raise ValueError("Tissue review requires distinct attached GM, WM, classification support and UNKNOWN masks")
    image_hash, frame_hash = array_digest(case.mri), structural_frame_hash(case)
    source_hashes = {ref.sha256 for ref in case.source_refs if ref.sha256 is not None}
    bound_layers = {}
    for kind, name in zip(("white_matter", "gray_matter", "classification_support", "unknown"), names):
        provenance = layers.get(name, {})
        if (not isinstance(provenance, Mapping)
                or provenance.get("source_image_hash") != image_hash
                or provenance.get("source_frame_hash") != frame_hash
                or provenance.get("provenance") not in {"observed", "estimated", "supplied"}
                or any(not isinstance(provenance.get(key), str) or not provenance[key].strip()
                       or len(provenance[key]) > 4096 for key in ("method", "source_role"))
                or provenance.get("source_role") == "generated_numerical_control"
                or kind in {"white_matter", "gray_matter"} and provenance.get("tissue_class") != kind):
            raise ValueError("Reviewed tissue/support layers require matching source, frame, class and method provenance")
        for key in ("source_sha256", "run_sha256"):
            if _hash(provenance.get(key), "tissue " + key) not in source_hashes:
                raise ValueError("Reviewed tissue source and method receipt must occur in attached source provenance")
        bound_layers[kind] = {"name": name, "array_hash": array_digest(case.compartments[name]),
                              "provenance": _json(provenance)}
    support, unknown = (case.compartments[name] for name in (support_name, unknown_name))
    wm, gm = case.compartments[wm_layer], case.compartments[gm_layer]
    if (np.any(support & unknown) or not np.array_equal(support | unknown, case.brain_mask)
            or np.any(wm & gm) or np.any((wm | gm) & ~support)):
        raise ValueError("Reviewed tissue classes must be disjoint inside known support; support/UNKNOWN must partition the brain")
    split = case.metadata.get("split", {})
    patient_group = case.metadata.get("patient_group", split.get("patient_group") if isinstance(split, Mapping) else None)
    expected = {"case_id": case.case_id, "patient_group": case.case_id if patient_group is None else patient_group,
                "source_image_hash": image_hash, "source_frame_hash": frame_hash,
                "source_refs_hash": semantic_digest([ref.to_dict() for ref in case.source_refs]),
                "brain_envelope_hash": array_digest(case.brain_mask), "classification_support_layer": support_name,
                "unknown_layer": unknown_name, "representation": "disjoint_binary_tissue_classes",
                "physical_partial_volume_claim": False, "layers": bound_layers}
    digest = semantic_digest(expected)
    if evidence != expected or record["evidence_hash"] != digest or review["evidence_hash"] != digest:
        raise ValueError("Tissue pair review is stale or belongs to different arrays, coverage, source, frame or method")
    return record


def _derived_domain(case, shape, affine, mode, threshold, wm_layer, gm_layer, *, healthy_source_evidence=None):
    if case.mri.size > MAX_SOURCE_VOXELS:
        raise ValueError("Host source exceeds the bounded resampling workspace budget")
    if case.frame != "RAS+" or case.brain_mask is None or not case.brain_mask.any():
        raise ValueError("Host needs an immutable source-bound RAS+ brain envelope")
    # Missing/stale real-source review is rejected before even brain resampling.
    tissue_review = (_reviewed_tissue_pair(case, wm_layer, gm_layer, healthy_source_evidence)
                     if mode == GM_WM_FRACTIONS else None)
    layers = case.metadata.get("anatomy_layers", {})
    for name, mask in case.compartments.items():
        if (layers.get(name, {}).get("kind") in {"tumor", "glioma", "lesion"}
                or any(token in name.casefold() for token in ("tumor", "glioma", "lesion", "enhancing_target", "flair_abnormality"))) and mask.any():
            raise ValueError("Virtual tumor-free host cannot contain a visible observed/estimated lesion")
    brain = _resample(case.brain_mask, case.affine, affine, shape, 0) > .5
    if not brain.any():
        raise ValueError("Simulation grid contains no mapped brain support")
    source_layers = {}
    reviewed_coverage = None
    if mode == GM_WM_FRACTIONS:
        if not isinstance(wm_layer, str) or not isinstance(gm_layer, str) or wm_layer == gm_layer:
            raise ValueError("Strict mode requires two named source WM/GM layers")
        if wm_layer not in case.compartments or gm_layer not in case.compartments:
            raise ValueError("WM/GM fractions must derive from attached source compartments")
        for kind, name in (("white_matter", wm_layer), ("gray_matter", gm_layer)):
            evidence = layers.get(name, {})
            if (evidence.get("tissue_class") != kind or not evidence.get("source_sha256")
                    or not evidence.get("source_role")
                    or evidence.get("source_image_hash") != array_digest(case.mri)
                    or evidence.get("source_frame_hash") != structural_frame_hash(case)):
                raise ValueError("Named tissue layers require explicit source-bound tissue classification provenance")
            _hash(evidence["source_sha256"], "tissue layer source")
            source_layers[kind] = {"name": name, "arrayHash": array_digest(case.compartments[name]), "provenance": _json(evidence)}
        if np.any(case.compartments[wm_layer] & case.compartments[gm_layer]):
            raise ValueError("Source WM/GM classes overlap; no silent fraction renormalization")
        wm = _resample(case.compartments[wm_layer], case.affine, affine, shape, 1)
        gm = _resample(case.compartments[gm_layer], case.affine, affine, shape, 1)
        tolerance = FRACTION_ROUNDOFF_TOLERANCE
        if (not np.isfinite(wm).all() or not np.isfinite(gm).all()
                or wm.min() < -tolerance or gm.min() < -tolerance
                or wm.max() > 1 + tolerance or gm.max() > 1 + tolerance
                or np.max(wm + gm) > 1 + tolerance):
            raise ValueError("Mapped WM/GM fractions are not nonnegative bounded disjoint tissue fractions")
        domain = brain & (wm + gm >= threshold)
        if tissue_review is not None:
            evidence = tissue_review["evidence"]
            known = _resample(case.compartments[evidence["classification_support_layer"]], case.affine, affine, shape, 0) > .5
            unknown = _resample(case.compartments[evidence["unknown_layer"]], case.affine, affine, shape, 0) > .5
            # Linear class weights may enter a cell whose nearest source label
            # is UNKNOWN. Coverage remains categorical; interpolation cannot
            # invent reviewed support or fill an explicit abstention.
            domain &= known & ~unknown
            reviewed_coverage = {"interpolation": "nearest_neighbor_order0_constant0_voxel_centers",
                                 "classificationSupportHash": array_digest(known), "unknownHash": array_digest(unknown),
                                 "domainRule": "brain_and_known_support_and_not_UNKNOWN_and_GM_plus_WM_at_least_threshold"}
    elif mode == UNIFORM_DIFFUSION_DOMAIN:
        if wm_layer is not None or gm_layer is not None:
            raise ValueError("Uniform diffusion must not masquerade as classified WM/GM")
        wm = gm = None
        domain = brain.copy()
    else:
        raise ValueError("Explicit GM_WM_FRACTIONS or UNIFORM_DIFFUSION_DOMAIN mode required")
    if not domain.any():
        raise ValueError("No admitted matter domain after source-bound mapping and threshold")
    mapping = {"sourceAffineHash": array_digest(case.affine), "targetAffineHash": array_digest(affine),
               "targetShape": list(shape), "targetVoxelToSourceVoxel": _voxel_mapping(case.affine, affine).tolist(),
               "exactGridIdentityCopy": tuple(shape) == case.mri.shape and np.array_equal(case.affine, affine),
               "brainEnvelopeInterpolation": "nearest_neighbor_order0_constant0_voxel_centers",
               "tissueFractionInterpolation": "linear_order1_constant0_no_renormalization" if mode == GM_WM_FRACTIONS else None,
               "sourceBrainEnvelopeHash": array_digest(case.brain_mask), "resampledBrainEnvelopeHash": array_digest(brain),
               "admittedDomainHash": array_digest(domain), "sourceTissues": source_layers,
               "wmFractionHash": None if wm is None else array_digest(wm),
               "gmFractionHash": None if gm is None else array_digest(gm), "matterThreshold": threshold,
               "fractionRoundoffTolerance": FRACTION_ROUNDOFF_TOLERANCE if wm is not None else None,
               "fractionValuesClippedOrRenormalized": False,
               "mappedFractionRange": None if wm is None else {
                   "wm": [float(wm.min()), float(wm.max())], "gm": [float(gm.min()), float(gm.max())],
                   "sum": [float((wm + gm).min()), float((wm + gm).max())]},
               "domainRule": "brain_envelope_and_GM_plus_WM_at_least_threshold" if mode == GM_WM_FRACTIONS else "brain_envelope_only_explicit_parameter_fiction"}
    if tissue_review is not None:
        mapping["tissuePairReview"] = tissue_review
        mapping["tissueInterpolationSemantics"] = "reviewed_categorical_classes_mapped_to_geometric_weights_not_physical_PVE"
        mapping["reviewedCoverageMapping"] = reviewed_coverage
        mapping["domainRule"] = reviewed_coverage["domainRule"]
    return domain, wm, gm, mapping


def _source_binding(case):
    return {"caseId": case.case_id, "caseHash": case.semantic_hash, "imageHash": array_digest(case.mri),
            "frameHash": structural_frame_hash(case), "affineHash": array_digest(case.affine),
            "brainEnvelopeHash": array_digest(case.brain_mask), "sourceRefsHash": semantic_digest([x.to_dict() for x in case.source_refs]),
            "compartmentHashes": {key: array_digest(value) for key, value in case.compartments.items()},
            "sourceCompartmentHashes": {key: array_digest(value) for key, value in case.source_compartments.items()},
            "criticalEvidenceHashes": {key: value.evidence_hash for key, value in case.critical_evidence.items()}}


def qualify_host(case, *, patient_group, role, healthy_source_evidence, qualified_by,
                 grid_shape, grid_affine, tissue_mode=GM_WM_FRACTIONS, matter_threshold=.1,
                 source_wm_layer=None, source_gm_layer=None):
    """Create an auditable explicit host-QA receipt; not a clinical certificate.

    The caller supplies an actual source health statement and reviewer identity.
    Missing tumor masks are never taken as evidence of a tumor-free person.
    This helper performs only binding/resampling, never tumor growth.
    """
    _admit_host_metadata(case, role)
    if not isinstance(patient_group, str) or not patient_group or not isinstance(qualified_by, str) or not qualified_by.strip():
        raise ValueError("Host patient group and explicit qualification reviewer are required")
    declared_group = case.metadata.get("patient_group")
    if declared_group is None and isinstance(case.metadata.get("split"), Mapping):
        declared_group = case.metadata["split"].get("patient_group")
    if patient_group != (case.case_id if declared_group is None else declared_group):
        raise ValueError("Qualification patient group must match the explicit host identity; aliases need source metadata")
    evidence = _json(healthy_source_evidence)
    if (not isinstance(evidence, dict) or set(evidence) != {"status", "patient_group", "source_url", "source_sha256", "statement"}
            or evidence["patient_group"] != patient_group or not evidence["statement"]
            or not isinstance(evidence["source_url"], str)):
        raise ValueError("Healthy-host source evidence must bind this exact patient group")
    _hash(evidence["source_sha256"], "healthy-host source evidence")
    if case.metadata.get("is_synthetic") is True:
        if evidence["status"] != "generated_numerical_control" or any(x.provenance != "simulated" for x in case.source_refs):
            raise ValueError("Generated controls require simulated source lineage and explicit generated status")
    elif evidence["status"] not in {"source_reported_healthy", "source_reported_tumor_free"} or not evidence["source_url"].startswith("https://"):
        raise ValueError("A real host needs reviewed source-reported health evidence, not absence of a tumor mask")
    shape, affine, _ = _grid(grid_shape, grid_affine)
    threshold = _number(matter_threshold, "matter_threshold", maximum=1., positive=True)
    _, _, _, mapping = _derived_domain(case, shape, affine, tissue_mode, threshold, source_wm_layer, source_gm_layer,
                                      healthy_source_evidence=evidence)
    record = {"schema": QUALIFICATION_SCHEMA, "sourceBinding": _source_binding(case), "patientGroup": patient_group,
              "role": role, "healthySourceEvidence": evidence, "qualifiedBy": qualified_by,
              "qualificationScope": "source_reported_host_research_only_not_clinical_certificate",
              "gridShape": list(shape), "gridAffineRasMm": affine.tolist(), "tissueMode": tissue_mode,
              "sourceWmLayer": source_wm_layer, "sourceGmLayer": source_gm_layer, "mapping": mapping}
    record["qualificationHash"] = semantic_digest(record)
    return record


def _validate_qualification(case, qualification):
    if not isinstance(qualification, Mapping):
        raise ValueError("Explicit completed host qualification required before growth")
    # This call is before even hashing a source array or opening solver files.
    _admit_host_metadata(case, qualification.get("role"))
    record = _json(qualification)
    seal = record.pop("qualificationHash", None)
    if qualification.get("schema") != QUALIFICATION_SCHEMA or semantic_digest(record) != seal:
        raise ValueError("Host qualification integrity failure")
    rebuilt = qualify_host(case, patient_group=qualification["patientGroup"], role=qualification["role"],
        healthy_source_evidence=qualification["healthySourceEvidence"], qualified_by=qualification["qualifiedBy"],
        grid_shape=qualification["gridShape"], grid_affine=qualification["gridAffineRasMm"],
        tissue_mode=qualification["tissueMode"], matter_threshold=qualification["mapping"]["matterThreshold"],
        source_wm_layer=qualification["sourceWmLayer"], source_gm_layer=qualification["sourceGmLayer"])
    if rebuilt != _json(qualification):
        raise ValueError("Host source identity, grid, domain or tissue provenance differs from qualification")
    shape, affine, spacing = _grid(qualification["gridShape"], qualification["gridAffineRasMm"])
    domain, wm, gm, mapping = _derived_domain(case, shape, affine, qualification["tissueMode"],
        qualification["mapping"]["matterThreshold"], qualification["sourceWmLayer"], qualification["sourceGmLayer"],
        healthy_source_evidence=qualification["healthySourceEvidence"])
    return shape, affine, spacing, domain, wm, gm, mapping


@dataclass(frozen=True)
class FKParameters:
    seed_center_ras_mm: tuple[float, float, float]
    seed_sigma_mm: float = 3.
    seed_peak: float = .2
    seed_support_sigma: float = 3.
    diffusion_mm2_per_day: float = .05
    growth_per_day: float = .025
    wm_to_gm_diffusion_ratio: float = 10.
    duration_days: float = 60.
    max_dt_days: float = 1.
    stability_safety: float = .9
    concentration_thresholds: tuple[float, ...] = (.1, .5)
    stopping_integral_mm3: float | None = None

    def __post_init__(self):
        center = tuple(self.seed_center_ras_mm)
        if len(center) != 3 or any(not isinstance(x, (float, int, np.floating, np.integer)) or isinstance(x, (bool, np.bool_)) or not math.isfinite(x) for x in center):
            raise ValueError("An explicit finite physical seed position is required")
        object.__setattr__(self, "seed_center_ras_mm", tuple(float(x) for x in center))
        bounds = {"seed_sigma_mm": (0., 20., True), "seed_peak": (0., 1., True),
                  "seed_support_sigma": (1., 5., True), "diffusion_mm2_per_day": (0., 10., False),
                  "growth_per_day": (0., 1., False), "wm_to_gm_diffusion_ratio": (1., 100., True),
                  "duration_days": (0., 730., False), "max_dt_days": (0., 20., True), "stability_safety": (0., 1., True)}
        for key, (minimum, maximum, positive) in bounds.items():
            object.__setattr__(self, key, _number(getattr(self, key), key, minimum=minimum, maximum=maximum, positive=positive))
        thresholds = tuple(self.concentration_thresholds)
        if not 1 <= len(thresholds) <= 8 or len(set(thresholds)) != len(thresholds):
            raise ValueError("Declare one to eight distinct scalar concentration thresholds")
        object.__setattr__(self, "concentration_thresholds", tuple(sorted(_number(x, "threshold", maximum=1., positive=True) for x in thresholds)))
        if self.stopping_integral_mm3 is not None:
            object.__setattr__(self, "stopping_integral_mm3", _number(self.stopping_integral_mm3, "stopping_integral_mm3", positive=True))


def masked_gaussian_seed(domain, affine, params):
    if type(params) is not FKParameters:
        raise TypeError("Exact validated FKParameters required")
    domain = np.asarray(domain, dtype=bool)
    if domain.ndim != 3 or not domain.any():
        raise ValueError("Seed requires a nonempty admitted 3-D domain")
    _, transform, spacing = _grid(list(domain.shape), affine)
    voxel = (np.linalg.inv(transform) @ np.r_[params.seed_center_ras_mm, 1.])[:3]
    if np.any(voxel < -.5) or np.any(voxel >= np.array(domain.shape) - .5):
        raise ValueError("Physical seed lies outside the simulation grid")
    nearest = np.floor(voxel + .5).astype(int)
    if not domain[tuple(nearest)]:
        raise ValueError("Physical seed lies outside the admitted matter domain")
    axes = np.ogrid[tuple(slice(0, n) for n in domain.shape)]
    radius_sq = sum(((axis - voxel[i]) * spacing[i]) ** 2 for i, axis in enumerate(axes))
    field = params.seed_peak * np.exp(-.5 * radius_sq / params.seed_sigma_mm ** 2)
    field[(~domain) | (radius_sq > (params.seed_sigma_mm * params.seed_support_sigma) ** 2)] = 0.
    if not np.any(field > 0):
        raise ValueError("Physical Gaussian has no sampled support; increase sigma or move the seed")
    return field


def prepare_fk_coefficients(kernel, domain, wm, gm, *, tissue_mode, threshold, diffusion, ratio):
    """One zero-valued guard cell blocks every periodic np.roll wrap edge."""
    if tissue_mode == UNIFORM_DIFFUSION_DOMAIN:
        w = domain.astype(np.float64)
        g = np.zeros(domain.shape, np.float64)
        kernel_ratio = 1.
    else:
        w, g = np.where(domain, wm, 0.), np.where(domain, gm, 0.)
        kernel_ratio = ratio
    padded_w, padded_g = np.pad(w, 1), np.pad(g, 1)
    coefficients = kernel.get_D(padded_w, padded_g, threshold, diffusion, kernel_ratio)
    if set(coefficients) != {"D_minus_" + a for a in "xyz"} | {"D_plus_" + a for a in "xyz"}:
        raise ValueError("Upstream diffusion coefficient contract changed")
    for values in coefficients.values():
        if values.shape != padded_w.shape or not np.isfinite(values).all() or values.min() < 0:
            raise ValueError("Upstream diffusion coefficients invalid")
    return coefficients


def stable_dt_limit(coefficients, spacing, growth):
    outflow = np.zeros_like(next(iter(coefficients.values())))
    for axis, h in zip("xyz", spacing):
        outflow += (coefficients["D_minus_" + axis] + coefficients["D_plus_" + axis]) / h ** 2
    rate = float(outflow.max()) + growth
    return (math.inf if rate == 0 else 1. / rate), float(outflow.max())


@dataclass(frozen=True)
class VirtualGliomaResult:
    concentration: np.ndarray
    initial_concentration: np.ndarray
    threshold_masks: Mapping
    manifest: Mapping

    def __post_init__(self):
        object.__setattr__(self, "concentration", immutable_array(self.concentration, np.float64))
        object.__setattr__(self, "initial_concentration", immutable_array(self.initial_concentration, np.float64))
        object.__setattr__(self, "threshold_masks", MappingProxyType({key: immutable_array(value, bool) for key, value in self.threshold_masks.items()}))
        object.__setattr__(self, "manifest", freeze_json(self.manifest))

    def assert_intact(self):
        if (array_digest(self.concentration) != self.manifest["concentrationHash"]
                or array_digest(self.initial_concentration) != self.manifest["initialConcentrationHash"]
                or {row["name"]: row["maskHash"] for row in self.manifest["thresholdMasks"]}
                   != {key: array_digest(value) for key, value in self.threshold_masks.items()}):
            raise ValueError("Virtual lesion arrays differ from their generated provenance")

    def save(self, directory):
        self.assert_intact()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "virtual-glioma.npz"
        np.savez_compressed(path, concentration=self.concentration, initial_concentration=self.initial_concentration,
                            **dict(self.threshold_masks))
        record = _json(self.manifest)
        record["outputFiles"] = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}}
        (directory / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return directory


def simulate_virtual_glioma(case, qualification, params, *, source_cache=DEFAULT_CACHE,
                           domain_mask=None, wm_fraction=None, gm_fraction=None):
    """Run one bounded CPU episode only after completed source-host qualification.

    Optional caller arrays must equal the independently reconstructed source
    mapping exactly. No acquired images, source labels, private references or
    vascular/function annotations are modified or used to shape the lesion.
    """
    shape, affine, spacing, domain, wm, gm, mapping = _validate_qualification(case, qualification)
    if type(params) is not FKParameters:
        raise TypeError("Exact FKParameters required")
    for supplied, expected, label in ((domain_mask, domain, "domain"), (wm_fraction, wm, "WM fraction"), (gm_fraction, gm, "GM fraction")):
        if supplied is not None:
            if expected is None or not np.array_equal(np.asarray(supplied), expected):
                raise ValueError("Supplied " + label + " differs from qualified source-bound resampling")
    source_before = case.semantic_hash
    initial = masked_gaussian_seed(domain, affine, params)
    kernel, kernel_receipt = load_pinned_kernel(source_cache)
    started = time.perf_counter()
    coefficients = prepare_fk_coefficients(kernel, domain, wm, gm, tissue_mode=qualification["tissueMode"],
        threshold=mapping["matterThreshold"], diffusion=params.diffusion_mm2_per_day, ratio=params.wm_to_gm_diffusion_ratio)
    stable_limit, max_outflow = stable_dt_limit(coefficients, spacing, params.growth_per_day)
    proposed_dt = min(params.max_dt_days, params.stability_safety * stable_limit)
    steps = 0 if params.duration_days == 0 else int(math.ceil(params.duration_days / proposed_dt))
    padded_cells = int(np.prod(np.array(shape) + 2))
    if steps > MAX_STEPS or steps * padded_cells > MAX_CELL_UPDATES:
        raise ValueError("Bounded pilot step/cell-update budget exceeded before growth")
    dt = 0. if steps == 0 else params.duration_days / steps
    field = np.pad(initial, 1)
    padded_domain = np.pad(domain, 1)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    history = []

    def record(step):
        return {"step": step, "timeDays": step * dt, "integralConcentrationMm3": float(field.sum(dtype=np.float64)) * voxel_volume,
                "minimum": float(field.min()), "maximum": float(field.max())}

    history.append(record(0))
    actual_steps = 0
    max_balance_residual = 0.
    balance_tolerance = 64. * np.finfo(np.float64).eps * padded_cells
    stop = "initial_integral_threshold" if params.stopping_integral_mm3 is not None and history[0]["integralConcentrationMm3"] >= params.stopping_integral_mm3 else "time"
    cadence = max(1, math.ceil(max(1, steps) / 20))
    if stop != "initial_integral_threshold":
        for step in range(1, steps + 1):
            if time.perf_counter() - started > MAX_WALL_SECONDS:
                raise ValueError("Bounded pilot wall-time budget exceeded; no result published")
            expected_integral = float(field.sum(dtype=np.float64)) + params.growth_per_day * dt * float(np.sum(field * (1. - field), dtype=np.float64))
            field = kernel.FK_update(field, coefficients, params.growth_per_day, dt, *spacing)
            if not np.isfinite(field).all() or field.min() < 0. or field.max() > 1.:
                raise ValueError("FK stability invariant failed; bad values are not silently clipped")
            if np.any(field[~padded_domain] != 0.):
                raise ValueError("FK escaped admitted matter domain; no silent masking of bad updates")
            balance_residual = abs(float(field.sum(dtype=np.float64)) - expected_integral)
            max_balance_residual = max(max_balance_residual, balance_residual)
            if balance_residual > balance_tolerance:
                raise ValueError("FK no-flux integral/reaction balance failed; no result published")
            actual_steps = step
            current = record(step)
            if step % cadence == 0 or step == steps:
                history.append(current)
            if params.stopping_integral_mm3 is not None and current["integralConcentrationMm3"] >= params.stopping_integral_mm3:
                stop = "first_discrete_step_at_integral_threshold"
                if history[-1]["step"] != step:
                    history.append(current)
                break
    final = field[1:-1, 1:-1, 1:-1].copy()
    if case.semantic_hash != source_before:
        raise ValueError("Original source anatomy/provenance changed during virtual-lesion generation")
    masks, mask_records = {}, []
    for threshold in params.concentration_thresholds:
        name = "virtual_concentration_ge_" + repr(threshold).replace(".", "p")
        masks[name] = (final >= threshold) & domain
        mask_records.append({"name": name, "threshold": threshold, "maskHash": array_digest(masks[name]),
                             "volumeMm3": int(masks[name].sum()) * voxel_volume, "semantics": "declared_scalar_concentration_threshold_only"})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    manifest = {"schema": SCHEMA, "label": "Virtual glioma geometry; generated FK concentration on a source-qualified host",
                "evidenceKind": "simulator_generated_geometry_only", "originalAcquiredStructuresChanged": False,
                "sourceBinding": _json(qualification["sourceBinding"]), "qualification": _json(qualification),
                "hostPatientGroup": qualification["patientGroup"], "inheritedHostRole": qualification["role"],
                "newPatientCount": 0, "virtualLesionCount": 1, "clinicalValidation": False,
                "frame": "RAS+", "gridShape": list(shape), "gridAffineRasMm": affine.tolist(), "spacingMm": spacing.tolist(),
                "tissueMode": qualification["tissueMode"], "mapping": mapping, "parameters": asdict(params),
                "kernel": {**kernel_receipt, "executedMethods": ["Solver.get_D", "Solver.FK_update"], "upstreamSolveExecuted": False},
                "wrapperSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numericalContract": {"dtype": "float64", "zeroPaddingCells": 1,
                    "boundary": "zero_face_conductivity_no_flux; padded_roll_edges_cannot_connect",
                    "initialCondition": "physical_gaussian_truncated_at_declared_sigma_multiple_and_masked_to_admitted_domain",
                    "stabilityBoundDays": None if math.isinf(stable_limit) else stable_limit,
                    "maxDiffusiveOutflowPerDay": max_outflow, "sufficientBound": "dt*(max_sum_neighbor_diffusive_rates+rho)<=1",
                    "dtDays": dt, "plannedSteps": steps, "executedSteps": actual_steps,
                    "timeZeroIsExactInitialCondition": True, "cropping": "none", "resamplingDuringSolve": "none",
                    "badResultClipping": False, "randomDraws": 0,
                    "maximumIntegralBalanceResidualMm3": max_balance_residual * voxel_volume,
                    "integralBalanceToleranceMm3": balance_tolerance * voxel_volume,
                    "integralBalanceRule": "sum(new)=sum(old)+dt*rho*sum(old*(1-old)); diffusion contributes zero",
                    "uniformEncoding": "Homogeneous coefficient encoded in upstream WM argument; not white-matter anatomy" if qualification["tissueMode"] == UNIFORM_DIFFUSION_DOMAIN else None},
                "timeSeriesMetrics": history, "finalTimeDays": actual_steps * dt,
                "stopReason": stop, "integralConcentrationMm3": float(final.sum(dtype=np.float64)) * voxel_volume,
                "integralMeaning": "model concentration integral; not measured tumor volume or tissue mass",
                "initialConcentrationHash": array_digest(initial), "concentrationHash": array_digest(final), "thresholdMasks": mask_records,
                "resources": {"wallSeconds": time.perf_counter() - started, "processPeakRssBytes": rss_bytes,
                    "rssScope": "whole_process_lifetime_not_isolated_increment", "estimatedArrayWorkingBytesUpperBound": padded_cells * 8 * 40 + case.mri.size * 8 * 4,
                    "arrayEstimateScope": "solver_and_resampling_temporaries; excludes_preexisting_immutable_host_and_interpreter",
                    "maxSourceVoxels": MAX_SOURCE_VOXELS,
                    "cellUpdates": actual_steps * padded_cells, "maxCellUpdates": MAX_CELL_UPDATES, "maxWallSeconds": MAX_WALL_SECONDS,
                    "maxShapePerAxis": MAX_SHAPE, "execution": "one_CPU_process_sequential_steps_no_training"},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                "limitations": ["geometry_only_no_mass_effect_or_host_deformation", "no_edema_necrosis_or_enhancement_labels",
                    "no_vascular_avoidance_perfusion_or_functional_risk", "uncalibrated_growth_parameters_not_patient_forecast",
                    "source_hazards_and_private_annotations_preserved_unchanged",
                    *( ["uniform_parameter_fiction_no_GM_WM_classification_or_CSF_barriers"] if qualification["tissueMode"] == UNIFORM_DIFFUSION_DOMAIN
                       else ["reviewed_categorical_tissue_interpolation_weights_not_physical_PVE_or_histology"] if "tissuePairReview" in mapping
                       else ["source_estimated_tissue_fractions_not_histology"])]}
    # A deterministic artifact identity omits measured performance metadata.
    manifest["simulationId"] = semantic_digest({key: value for key, value in manifest.items() if key != "resources"})
    return VirtualGliomaResult(final, initial, masks, manifest)
