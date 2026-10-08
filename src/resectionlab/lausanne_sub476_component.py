"""Pinned, retrospective positive-aneurysm component from actual Lausanne data.

Reads existing originals only. No acquisition, registration, routes, targets,
brain support, model, training or clinical admission is performed. The source
methods XML must already be archived; missing source files are a hard dependency.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import gzip
import json
from pathlib import Path
import pickletools

import nibabel as nib
import numpy as np

from .core import SourceRef, array_digest, semantic_digest, thaw_json
from .critical_evidence import (CriticalStructureEvidence, nifti1_header_record,
                                resolve_critical_evidence, source_reference_grid_metrics)
from .imaging import file_sha256, load_nifti_case


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "manifests/lausanne-sub476-manual-annotation-v1.json"
MANIFEST_SHA256 = "39537b60333a22e8e8e81f792ac5ba9dad306ef72ff3a4a4f768f6b7ab59230f"
MASK_SHA256 = "1687ac04ca817691ce91de4ccb4457b31dbbbfca414db019d4e29bd88b0a03ee"
METHODS_SHA256 = "105ae182b2b7da8c0d29f23700b5a3b82c6feb3255e5f09c8c4349495fd1e954"
METHODS_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9931814/fullTextXML"
DATA = Path("data/anatomy/ds003949-v1.0.1")
METHODS = DATA / "manual-annotation-sub476-v1/source/methods-PMC9931814.xml"
MEMBERSHIP_LICENSE = DATA / "manual-annotation-sub476-v1/source/Aneurysm_Detection-LICENSE"
MEMBERSHIP_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
EVIDENCE_ID = "Lausanne-sub476-manual-aneurysm-positive-v1"
SCOPE = "retrospective_source_coded_frame_component"


def _verify(path: Path, digest: str, size: int | None = None) -> None:
    if (size is not None and path.stat().st_size != size) or file_sha256(path) != digest:
        raise ValueError(f"Pinned source differs: {path.name}")


def _source(identity, uri, digest, license="CC0", frame="not_applicable"):
    return SourceRef(identity, uri, sha256=digest, license=license,
                     native_frame=frame, provenance="observed")


def _raw_header(path, digest):
    with gzip.open(path, "rb") as stream:
        return nifti1_header_record(stream.read(348), digest)


def build_sub476_component(root: Path = ROOT):
    """Create one source-bound case; all imported arrays retain original order."""
    root = Path(root)
    manifest_path = root / MANIFEST
    _verify(manifest_path, MANIFEST_SHA256)
    m = json.loads(manifest_path.read_bytes())
    if (m["subject"], m["session"], m["role"]) != ("sub-476", "ses-20140519", "TRAIN"):
        raise ValueError("This adapter is restricted to the declared TRAIN acquisition")
    for entry in [m["cohort"], m["index"], m["original_acquisition"], *m["source_metadata"], *m["rights"]]:
        _verify(root/entry["path"], entry["sha256"], entry["bytes"])
    cohort = json.loads((root/m["cohort"]["path"]).read_bytes())
    person = next(row for row in cohort["members"] if row["subject"] == m["subject"])
    if person["role"] != "TRAIN" or "20140519" not in person["sessions"]:
        raise ValueError("Frozen person role or session differs")
    acquisition = json.loads((root/m["original_acquisition"]["path"]).read_bytes())
    if (acquisition["scanner_frame_admitted"] is not False
            or acquisition["spatial_planning_admitted"] is not False):
        raise ValueError("Source acquisition limits changed; reassess the adapter")
    if json.loads((root/m["rights"][0]["path"]).read_bytes())["License"] != "CC0":
        raise ValueError("Source annotation rights differ")

    strong, sidecar_entry, _ = m["source_metadata"]
    # Inspect only string-list opcodes. Never execute an upstream pickle.
    operations = list(pickletools.genops((root/strong["path"]).read_bytes()))
    allowed = {"APPENDS", "BINPUT", "BINUNICODE", "EMPTY_LIST", "MARK", "PROTO", "STOP"}
    if not {op.name for op, _, _ in operations} <= allowed:
        raise ValueError("Unrecognized annotation-membership metadata")
    members = [value for op, value, _ in operations if op.name == "BINUNICODE"]
    if len(members) != 38 or "sub-476_ses-20140519" not in members:
        raise ValueError("Exact voxelwise annotation membership is missing")
    sidecar_bytes = (root/sidecar_entry["path"]).read_bytes()
    if json.loads(sidecar_bytes) != {"Type": "Lesion", "RawSources": Path(m["original_tof"]["path"]).name, "Space": "orig"}:
        raise ValueError("Original-acquisition annotation linkage differs")
    _verify(root/METHODS, METHODS_SHA256, 124982)
    _verify(root/MEMBERSHIP_LICENSE, MEMBERSHIP_LICENSE_SHA256, 11357)
    methods = _source("DiNoto2022-annotation-methods", METHODS_URL, METHODS_SHA256, "CC-BY-4.0")
    membership = _source("Lausanne-source-voxelwise-membership", strong["url"], strong["sha256"], "Apache-2.0")
    linkage = _source("Lausanne-sub476-annotation-sidecar", sidecar_entry["url"], sidecar_entry["sha256"])

    image_path, mask_path = root/DATA/m["original_tof"]["path"], root/DATA/m["file"]["path"]
    _verify(image_path, m["original_tof"]["sha256"], m["original_tof"]["bytes"])
    _verify(mask_path, MASK_SHA256, m["file"]["bytes"])
    label_image = nib.load(mask_path)
    labels = np.asarray(label_image.dataobj)
    if (labels.dtype != np.uint8 or labels.shape != (306, 384, 160)
            or not np.array_equal(np.unique(labels), [0, 1]) or np.count_nonzero(labels) != 193):
        raise ValueError("Pinned binary aneurysm annotation differs")
    positive = labels.astype(bool)
    annotation_grid = _raw_header(mask_path, MASK_SHA256)
    reference_grid = _raw_header(image_path, m["original_tof"]["sha256"])
    precision = source_reference_grid_metrics(annotation_grid, reference_grid)
    case = load_nifti_case(image_path, case_id="Lausanne-sub476-ses20140519-TOF-component", license="CC0",
        metadata={"evidence_identity": {"dataset": "ds003949-v1.0.1", "participant": m["subject"], "timepoint": m["session"]},
                  "evidence_scope": SCOPE, "scanner_frame_admitted": False, "spatial_planning_admitted": False,
                  "allow_nonzero_mri_access_support": False, "anatomical_coverage": "unreviewed",
                  "registration": "unverified", "role": "TRAIN", "component_optimizer_updates": 0,
                  "recorded_rl_transitions": 0, "annotation_available_at": None, "source_review_available_at": None})
    annotation = _source("Lausanne-sub476-original-manual-aneurysm", m["file"]["source_url"], MASK_SHA256,
                         frame="source_orig_declared_units_missing_retained")
    derivation = {
        "method": "source_reference_grid_normalization", "source_image_sha256": m["original_tof"]["sha256"],
        "annotation_sha256": MASK_SHA256, "annotation_raw_grid": annotation_grid, "reference_raw_grid": reference_grid,
        "array_operation": "unchanged_binary_positive_support", "positive_mask_hash": array_digest(positive),
        "sidecar_source": linkage.to_dict(), "sidecar_base64": base64.b64encode(sidecar_bytes).decode("ascii"),
        "reference_filename": image_path.name, "precision_equivalence": precision,
        "unit_interpretation": "reference_mm_from_explicit_RawSources_orig", "scope": SCOPE,
        "scanner_frame_admitted": False, "spatial_planning_admitted": False,
        "inference_rationale": "The pinned source declares this exact RawSources and Space=orig. Identical shape and a bounded float32 serialization difference support source-grid unit inheritance and coordinate normalization. This is an explicit interpretation, not proven software history or scanner validation. Both raw headers and original files remain unchanged.",
    }
    derivation["normalization_record_hash"] = semantic_digest(derivation)
    evidence = CriticalStructureEvidence(evidence_id=EVIDENCE_ID, structure="vessels", mask=positive,
        annotation_coverage=positive, affine_ras_mm=case.affine, case_id=case.case_id,
        reference_image_hash=array_digest(case.mri), reference_source_id="structural",
        reference_source_sha256=m["original_tof"]["sha256"], source=annotation,
        source_binding={"dataset": "ds003949-v1.0.1", "participant": m["subject"], "timepoint": m["session"],
                        "linkage_source": linkage.to_dict(), "coverage_source": annotation.to_dict(),
                        "coverage_meaning": "source_documented_annotation_domain", "coverage_policy": "positive_support_only",
                        "positive_class": "source_manual_voxelwise_aneurysm_region", "background_meaning": "unknown_for_vascular_anatomy",
                        "coverage_restriction": "Adapter restricts consumption to source positive voxels; no negative vessel annotation domain is inferred.",
                        "original_release_manifest_sha256": MANIFEST_SHA256, "cohort_sha256": m["cohort"]["sha256"]},
        derivation=derivation,
        lineage={"kind": "manual", "model_sha256": [], "source": methods.to_dict(), "membership_source": membership.to_dict(),
                 "membership_license_sha256": MEMBERSHIP_LICENSE_SHA256,
                 "membership": "sub-476_ses-20140519", "method": "Radiologist drew axial slices in ITK-SNAP 3.6.0; no learned initializer in the documented voxelwise protocol."})
    review = {"status": "source_human_reviewed", "source": methods.to_dict(), "available_at": None,
              "content_hash": evidence.content_hash,
              "binding_kind": "software_provenance_binding_not_clinician_signature",
              "scope": "Paper-reported source annotation double-check by a senior neuroradiologist to exclude false positives/negatives. This does not assert review of our content hash, normalization, positive-only domain, adapter, or planning use.",
              "annotation_source_sha256": MASK_SHA256, "membership_source": membership.to_dict()}
    evidence = replace(evidence, review=review)
    return replace(case, critical_evidence={EVIDENCE_ID: evidence})


def component_receipt(case) -> dict:
    evidence = case.critical_evidence[EVIDENCE_ID]
    constraints = resolve_critical_evidence(case)
    return {"schema": "lausanne-sub476-positive-component-v1", "scope": SCOPE,
            "case_hash": case.semantic_hash, "critical_binding_hash": constraints.fingerprint,
            "evidence": evidence.to_manifest(), "constraints": thaw_json(constraints.receipt),
            "positive_voxels": int(np.count_nonzero(evidence.mask)), "negative_annotation_voxels": 0,
            "background_unknown_voxels": int(evidence.mask.size-np.count_nonzero(evidence.mask)),
            "scanner_frame_admitted": False, "spatial_planning_admitted": False,
            "generated_routes": 0, "simulator_episodes": 0, "component_optimizer_updates": 0,
            "recorded_rl_transitions": 0}


if __name__ == "__main__":
    print(json.dumps(component_receipt(build_sub476_component()), indent=2, allow_nan=False))
