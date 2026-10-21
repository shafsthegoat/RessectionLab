#!/usr/bin/env python3
"""Read-only, role-aware corpus inventory; downloads and model inference are absent.

The default reads declared JSON evidence and allowlisted bundle manifests only.
It never decodes source imaging, opens protected bundles, or scans holdout trees.
Recorded source hashes are explicitly distinguished from freshly hashed metadata.
Optional tumor geometry reads annotation arrays from TRAIN/DEVELOPMENT bundles.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
READABLE_ROLES = {"TRAIN", "DEVELOPMENT"}
UNKNOWN = "UNKNOWN"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_patient(value):
    """Known release aliases only; no fuzzy cross-cohort identity matching."""
    for pattern, prefix in ((r"UCSF[-:]PDGM[-:]0*(\d+)", "UCSF-PDGM:"),
                            (r"BTC.*?(sub-PAT\d+)", "BTC:"),
                            (r"ReMIND[-:]0*(\d+)", "ReMIND:"),
                            (r"Lausanne[-:](sub-\d+)", "Lausanne:"),
                            (r"SynthRAD2023[-:](1B[A-Z]\d+)", "SynthRAD2023:")):
        match = re.search(pattern, value)
        if match:
            identifier = match.group(1)
            if prefix == "UCSF-PDGM:":
                identifier = f"{int(identifier):03d}"
            elif prefix == "ReMIND:":
                identifier = f"{int(identifier):03d}"
            return prefix + identifier
    return None


def read_bundle_manifest(path, patient, roles):
    """Role check deliberately precedes opening the ZIP, including metadata."""
    if roles.get(patient) not in READABLE_ROLES:
        return {"path": str(path), "patient_group": patient, "status": "protected_not_opened"}
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo("manifest.json").file_size > 8 * 1024 * 1024:
            raise ValueError("Oversized bundle manifest")
        payload = archive.read("manifest.json")
    manifest = json.loads(payload)
    if canonical_patient(manifest["case_id"]) != patient:
        raise ValueError(f"Bundle patient identity mismatch: {path.name}")
    metadata = manifest.get("metadata", {})
    layers = metadata.get("anatomy_layers", {})
    kinds = {layer.get("kind") for layer in layers.values()}
    if manifest.get("structural_evidence"):
        kinds.add("brain_estimate")
    source_names = set(manifest.get("array_index", {}).get("source_compartments", {}))
    if source_names.intersection({"fractional_source_target_threshold_scenario", "FLAIR_abnormality",
                                  "enhancing_target", "nonenhancing_or_necrotic_core"}):
        kinds.add("tumor_annotation_or_estimate")
    missing = set(metadata.get("missing_anatomy", []))
    for kind in ("brain", "tumor", "skull", "vessels", "ventricles"):
        if kind not in kinds and not (kind == "brain" and "brain_estimate" in kinds) and not (
                kind == "tumor" and "tumor_annotation_or_estimate" in kinds):
            missing.add(kind)
    return {"path": str(path), "patient_group": patient, "role": roles[patient],
            "status": "manifest_inspected_arrays_not_read", "case_id": manifest["case_id"],
            "manifest_sha256_measured": sha256_bytes(payload),
            "case_semantic_hash_recorded": manifest.get("case_semantic_hash"),
            "is_synthetic": metadata.get("is_synthetic", UNKNOWN),
            "available_layer_kinds": sorted(k for k in kinds if k),
            "layers": {k: {field: v.get(field) for field in
                       ("kind", "provenance", "review_status", "coverage", "source_sha256")}
                       for k, v in layers.items()},
            "missing_anatomy": sorted(missing),
            "registration_evidence": metadata.get("source_frame_declaration", UNKNOWN),
            "tumor_location": {"anatomical_region": UNKNOWN, "centroid_ras_mm": None,
                               "basis": "No accepted localization measurement in metadata inventory"}}


def measure_tumor_geometry(path, patient, roles):
    """Measure source-label union only; coordinates do not imply a lobe or function."""
    if roles.get(patient) not in READABLE_ROLES:
        raise ValueError("Protected patient: annotation array access forbidden")
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if canonical_patient(manifest["case_id"]) != patient:
            raise ValueError("Bundle identity mismatch before annotation access")
        if manifest.get("metadata", {}).get("is_synthetic") is not False:
            raise ValueError("Synthetic or unclassified annotation cannot measure observed tumor diversity")
        layers = manifest.get("metadata", {}).get("anatomy_layers", {})
        indices = manifest["array_index"]["source_compartments"]
        selected = {name: key for name, key in indices.items()
                    if (layers.get(name, {}).get("source_role") == "imported_source_annotation"
                        or (not layers and "fractional" in name))}
        if not selected:
            return {"anatomical_region": UNKNOWN, "centroid_ras_mm": None,
                    "basis": "No eligible source-label arrays"}
        if archive.getinfo("arrays.npz").file_size > 512 * 1024 * 1024:
            raise ValueError("Annotation container exceeds bounded inventory limit")
        payload = archive.read("arrays.npz")
    if sha256_bytes(payload) != manifest["array_sha256"]:
        raise ValueError("Array container hash mismatch")
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        affine = arrays["affine"].copy()
        if manifest["frame"] == "LPS+":
            affine = np.diag([-1., -1., 1., 1.]) @ affine
        elif manifest["frame"] != "RAS+":
            raise ValueError("Unknown physical frame")
        masks = [arrays[key].astype(bool) for key in selected.values()]
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise ValueError("Invalid measurement affine")
    if any(mask.ndim != 3 or mask.shape != masks[0].shape for mask in masks):
        raise ValueError("Annotation grids disagree")
    union = np.logical_or.reduce(masks)
    xyz = np.argwhere(union)
    if not len(xyz):
        raise ValueError("Empty source annotation")
    centroid = (affine @ np.r_[xyz.mean(axis=0), 1.])[:3]
    return {"anatomical_region": UNKNOWN, "hemisphere": UNKNOWN,
            "centroid_ras_mm": centroid.tolist(), "positive_voxels": len(xyz),
            "source_label_union_volume_ml": float(len(xyz) * abs(np.linalg.det(affine[:3, :3])) / 1000),
            "label_names": sorted(selected), "arrays_sha256_verified": manifest["array_sha256"],
            "basis": "Measured source-label union in saved bundle frame; includes declared threshold/FLAIR semantics; no lobe or functional localization"}


class Inventory:
    def __init__(self, root, data_root, output_root):
        self.root, self.data_root, self.output_root = map(Path, (root, data_root, output_root))
        self.metadata = []
        self.roles = {}
        self.sources = []
        self.warnings = []
        self.bundles = []

    def evidence(self, relative, *, data=False):
        path = (self.data_root if data else self.root) / relative
        if not path.is_file():
            self.warnings.append(f"Missing evidence: {path}")
            return {}
        payload = path.read_bytes()
        self.metadata.append({"path": str(path), "sha256_measured": sha256_bytes(payload), "bytes": len(payload)})
        return json.loads(payload)

    def role(self, patient, role):
        if patient in self.roles and self.roles[patient] != role:
            raise ValueError(f"Conflicting patient roles: {patient}")
        self.roles[patient] = role

    def source(self, patient, path, record, *, evidence, reported_verified=False, kind="source"):
        role = self.roles[patient]
        # Protected files are not even stat'ed. Receipt evidence remains a historical claim.
        local = Path(path)
        size = local.stat().st_size if role in READABLE_ROLES and local.is_file() else None
        expected = record.get("bytes", record.get("size_bytes", record.get("expected_bytes")))
        self.sources.append({"patient_group": patient, "role": role, "kind": kind,
                             "path": str(local), "expected_bytes": expected, "present_bytes": size,
                             "presence": "protected_not_inspected" if role not in READABLE_ROLES else
                                ("present_size_matches" if size is not None and size == expected else
                                 "present_size_unbound" if size is not None and expected is None else
                                 "size_mismatch" if size is not None else "missing"),
                             "sha256_recorded_not_rehashed": record.get("sha256"),
                             "md5_recorded_not_rehashed": record.get("expected_md5", record.get("expected_annex_md5", record.get("source_checksum"))),
                             "source_url": record.get("source_url", record.get("url")),
                             "reported_byte_verified": reported_verified, "evidence": evidence})

    def load_roles(self):
        for name, key in (("lausanne-component-cohort-v1.json", "canonical_person"),
                          ("synthrad-component-cohort-v1.json", "patient_group"),
                          ("experiments/remind-component-cohort-v1.json", "patient_group")):
            for row in self.evidence("manifests/" + name).get("members", []):
                self.role(row[key], row["role"])
        btc = self.evidence("manifests/experiments/btc-spatial-development-cohort-v1.json")
        for row in btc.get("existing_development_records", []):
            self.role(row["patient_group"], "TRAIN")
        for row in btc.get("candidates", []):
            declared = row["development_role"]
            self.role(row["patient_group"], "TRAIN" if declared == "population_training" else
                      "SELECT" if "selection" in declared else "UNOPENED_TRANSFER")
        self.role("UCSF-PDGM:004", "DEVELOPMENT")

    def load_sources(self):
        path = "manifests/data_acquisition_ucsf.json"
        for row in self.evidence(path).get("files", []):
            if ".nii" in row["path"]:
                self.source("UCSF-PDGM:004", self.data_root / row["path"], row, evidence=path, reported_verified=True)
        for subject in ("05", "16", "20", "28", "22", "25", "26", "27"):
            if subject in {"22", "25", "26", "27"}:
                path = f"diffusion_source/ds001226-v5.0.1/btc_sub-PAT{subject}_verified_manifest.json"
                record = self.evidence(path, data=True)
            else:
                path = "manifests/" + ("btc_acquisition.json" if subject == "28" else f"btc_pat{subject}_acquisition.json")
                record = self.evidence(path)
            for row in record.get("files", []):
                if ".nii" in row["path"]:
                    self.source(f"BTC:sub-PAT{subject}", self.data_root / "diffusion_source/ds001226-v5.0.1" / row["path"], row, evidence=path, reported_verified=True)
        path = "manifests/lausanne-train-originals-v1.json"
        for session in self.evidence(path).get("sessions", []):
            for row in session["files"]:
                self.source("Lausanne:" + session["subject"], self.data_root / "anatomy/ds003949-v1.0.1" / row["path"], row, evidence=path)
        path = "manifests/lausanne-train-annotations-v1.json"
        for row in self.evidence(path).get("records", []):
            if row.get("status") == "metadata_qualified":
                self.source("Lausanne:" + row["subject"], self.data_root / "anatomy/ds003949-v1.0.1" / row["path"], row["file"], evidence=path, kind="annotation")
        for intake in ("remind-train-originals-v1", "remind-train-seg-v1", "remind-select-public-v1"):
            path = f"acquisition/{intake}/completion.json"
            for row in self.evidence(path, data=True).get("files", []):
                patient = canonical_patient(row["patient_id"])
                self.source(patient, self.data_root / "acquisition" / intake / "verified" / row["path"], row,
                            evidence=path, reported_verified=row.get("status") == "byte_verified",
                            kind="annotation" if "seg-" in intake else "source")
        # Converted 001 sources are already source-bound in this saved receipt;
        # they establish one existing individual, not four new cases.
        path = "remind_source/converted-native/ReMIND-001/native-conversion.json"
        for row in self.evidence(path, data=True).get("native_series", []):
            self.source("ReMIND:001", self.data_root / "remind_source/converted-native/ReMIND-001" / row["file"], row,
                        evidence=path, kind="converted_source")
        for modality in ("ct", "mr"):
            path = f"qc/synthrad-task1-first-pair-v1/{modality}-extraction.json"
            row = self.evidence(path, data=True)
            if row:
                self.source("SynthRAD2023:1BA336", self.data_root / "qc/synthrad-task1-first-pair-v1/restricted" / f"{modality}.nii.gz", row, evidence=path, reported_verified=True)

    def load_bundles(self, measure=False):
        for directory in (self.output_root / "cases", self.output_root / "poc-cases"):
            for path in sorted(directory.glob("*.ressectionlab")):
                patient = canonical_patient(path.name)
                if patient is None or patient not in self.roles:
                    self.warnings.append(f"Unmapped bundle not opened (may be duplicate workspace): {path.name}")
                    continue
                row = read_bundle_manifest(path, patient, self.roles)
                self.bundles.append(row)
        if measure:
            # One best source representation per person, deterministic and not every copy.
            candidates = {}
            for row in self.bundles:
                if row["status"] != "protected_not_opened" and row["is_synthetic"] is False:
                    candidates[row["patient_group"]] = row
            for patient, row in candidates.items():
                row["tumor_location"] = measure_tumor_geometry(Path(row["path"]), patient, self.roles)

    def report(self):
        acquired = {r["patient_group"] for r in self.sources if
                    r["kind"] != "annotation" and (".nii" in r["path"] or r["path"].endswith(".dcm")) and
                    (r["presence"] in {"present_size_matches", "present_size_unbound"} or
                     (r["presence"] == "protected_not_inspected" and r["reported_byte_verified"]))}
        inspected = [r for r in self.bundles if r["status"] != "protected_not_opened"]
        real = {r["patient_group"] for r in inspected if r["is_synthetic"] is False}
        virtual = [r for r in inspected if r["is_synthetic"] is True]
        anatomy = {}
        for row in inspected:
            if row["is_synthetic"] is not False:
                continue
            kinds = set(row["available_layer_kinds"])
            if "brain_estimate" in kinds:
                kinds.add("brain")
            if "tumor_annotation_or_estimate" in kinds:
                kinds.add("tumor")
            anatomy.setdefault(row["patient_group"], set()).update(kinds)
        combinations = {label: sum(required <= kinds for kinds in anatomy.values())
                        for label, required in {
                            "brain_plus_tumor": {"brain", "tumor"},
                            "brain_plus_skull": {"brain", "skull"},
                            "brain_plus_vessels": {"brain", "vessels"},
                            "brain_tumor_skull_vessels_same_person": {"brain", "tumor", "skull", "vessels"}
                        }.items()}
        counts = {}
        for family in ("UCSF-PDGM", "BTC", "ReMIND", "SynthRAD2023", "Lausanne"):
            people = sorted(p for p in acquired if p.startswith(family + ":"))
            counts[family] = {"materialized_source_patient_groups": len(people),
                              "roles": dict(Counter(self.roles[p] for p in people)),
                              "patients": people,
                              "inspected_real_bundle_patient_groups": len(real.intersection(people))}
        return {"schema": "resectionlab.patient-corpus-inventory.v1", "scope": "Five named source families; no whole-disk or protected imaging scan",
                "payload_hashes_recomputed": False, "protected_imaging_opened": False,
                "counts": counts, "materialized_source_patient_groups": len(acquired),
                "identity_limitation": "Unique within source-family namespaces; cross-cohort independence unresolved. UCSF004 aliases BraTS2021_00097; Lausanne/TopCoW derivatives add no people.",
                "inspected_real_bundle_patient_groups": len(real), "virtual_bundle_representations_in_inspected_roots": len(virtual),
                "prepared_layer_presence_not_anatomical_acceptance": combinations,
                "virtual_patient_count": 0, "pde_generator_executed_by_inventory": False,
                "training_or_planning_admission": "Not established by inventory; see case-specific QC and latest Gate ledger",
                "tumor_location_policy": "UNKNOWN unless measured; source-frame centroid is not anatomical/functional localization",
                "protected_collections": {"UPENN-GBM": "external final reserved; no patient imaging enumerated",
                                          "UTSW-Glioma": "future structural validation; no patient imaging enumerated"},
                "role_counts_declared": dict(Counter(self.roles.values())),
                "metadata_evidence": self.metadata, "source_files": self.sources,
                "bundles": self.bundles, "warnings": self.warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, help="Explicit existing output location; no automatic sibling lookup")
    parser.add_argument("--output", type=Path, help="JSON report, exclusive create; omit for stdout")
    parser.add_argument("--measure-tumor-geometry", action="store_true", help="Read only allowed source-label arrays; keep lobe/hemisphere UNKNOWN")
    args = parser.parse_args(argv)
    inventory = Inventory(args.root, args.data_root or args.root / "data", args.output_root or args.root / "outputs")
    inventory.load_roles()
    inventory.load_sources()
    inventory.load_bundles(measure=args.measure_tumor_geometry)
    report = inventory.report()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(encoded)
        print(json.dumps({key: report[key] for key in ("counts", "materialized_source_patient_groups", "inspected_real_bundle_patient_groups", "virtual_bundle_representations_in_inspected_roots", "warnings")}, indent=2))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
