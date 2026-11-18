#!/usr/bin/env python3
"""Inspect a frozen study inventory and migrate admitted rows to native anatomy.

Without --run, print the dispositions without opening any image bytes. New
native studies stay private until separately admitted to the app catalog.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile

from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import atomic_json, checked_root, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection


SCHEMA = "resectionlab.anatomy-migration-ledger.v1"
RECEIPT_SCHEMA = "resectionlab.anatomy-migration-receipt.v1"
PROCESSING_ROLES = {"TRAIN", "DEVELOPMENT"}
RETENTION_OUTCOMES = {"unavailable", "protected", "unsupported", "ambiguous", "retained"}
LOCAL_OUTPUTS = Path(__file__).resolve().parents[1] / "testcases"
SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def file_ref(value):
    path = Path(value["path"])
    if not path.is_absolute() or not SHA256.fullmatch(value["sha256"]):
        raise ValueError("File references require an absolute path and a SHA-256")
    if type(value["bytes"]) is not int or value["bytes"] < 1:
        raise ValueError("File references require a positive byte count")
    return path


def load_ledger(path):
    raw = path.read_bytes()
    ledger = json.loads(raw)
    if ledger.get("schema") != SCHEMA or not SAFE_KEY.fullmatch(ledger["version"]):
        raise ValueError("Unsupported migration ledger or version")
    authorities = {}
    for reference in ledger["authorities"]:
        filename = str(file_ref(reference))
        if filename in authorities:
            raise ValueError(f"Duplicate authority: {filename}")
        authorities[filename] = reference
    cases = {}
    roles = defaultdict(set)
    for row in ledger["cases"]:
        key = row["key"]
        if not SAFE_KEY.fullmatch(key) or key in cases:
            raise ValueError("Migration keys must be distinct and filesystem safe")
        identity_fields = ("person_id", "patient_id", "case_id", "source_version", "timepoint", "role", "reason")
        if any(not isinstance(row[field], str) or not row[field].strip() for field in identity_fields):
            raise ValueError("Migration requires explicit study identity, role and disposition")
        if row["role"] != "UNKNOWN":
            roles[row["person_id"]].add(row["role"])
        if row["outcome"] == "eligible":
            validate_admission(row, authorities)
        elif row["outcome"] not in RETENTION_OUTCOMES:
            raise ValueError(f"Unknown migration outcome: {row['outcome']}")
        cases[key] = row
    for key, row in cases.items():
        if row["outcome"] == "eligible" and roles[row["person_id"]] != {row["role"]}:
            raise ValueError(f"A conflicting or protected person role prevents migration: {key}")
    return ledger, cases, authorities, hashlib.sha256(raw).hexdigest()


def validate_admission(row, authorities):
    key = row["key"]
    if row["role"] not in PROCESSING_ROLES or row.get("exposure") != "private":
        raise ValueError(f"Study role or exposure prevents migration: {key}")
    if not row.get("admission") or any(path not in authorities for path in row["admission"]):
        raise ValueError(f"Missing or unbound processing admission metadata: {key}")
    inputs = [file_ref(item) for item in row["inputs"]]
    if len(inputs) != len(set(inputs)):
        raise ValueError(f"Duplicate frozen inputs: {key}")
    if row["route"] == "verified-native-reuse":
        used = {Path(row["native_manifest"]), *(Path(path) for path in row.get("native_sidecars", []))}
    elif row["route"] == "raw-intake":
        spec = row["intake"]
        if not spec.get("series"):
            raise ValueError(f"Raw route requires scans: {key}")
        used = {Path(item["path"]) for item in spec["series"] + spec.get("layers", [])}
        used.update(Path(item["sidecar"]) for item in spec["series"] if item.get("sidecar"))
        if any(item.get("kind", "nifti") != "nifti" for item in spec["series"]):
            raise ValueError("Batch DICOM requires a frozen per-instance file manifest")
    elif row["route"] == "eden-cached-intake":
        if (row["family"] != "EDEN2020" or row["role"] != "DEVELOPMENT"
                or row["eden_variants"] != list(range(1, 9))):
            raise ValueError(f"Invalid EDEN cached preparation identity: {key}")
        used = set(inputs)
    else:
        raise ValueError(f"Unsupported migration route: {key}")
    if used != set(inputs):
        raise ValueError(f"Intake paths and frozen inputs differ: {key}")


def guard_root(path, expected_id, required_bytes=0):
    root = checked_root(path, required_bytes=required_bytes)
    marker = json.loads((root / ".ressectionlab-data-root.json").read_text())
    if not expected_id or marker.get("root_id") != expected_id:
        raise ValueError(f"Mounted source does not match the frozen migration root: {path}")
    return root


def guard_output(root, version, external_roots):
    if root.name == version and root.is_relative_to(LOCAL_OUTPUTS.resolve()):
        return
    for candidate, identity in external_roots.items():
        if root == candidate / "prepared" / version:
            guard_root(candidate, identity)
            return
    raise ValueError("Choose a versioned output under testcases/ or the frozen external prepared/<version>")


def verify_inputs(row, authorities, external_roots, check):
    for reference in [*(authorities[path] for path in row["admission"]), *row["inputs"]]:
        check()
        path = file_ref(reference)
        if path.resolve(strict=True) != path:
            raise ValueError(f"Frozen inputs must use canonical paths: {path}")
        if path.is_relative_to("/Volumes"):
            root = next((root for root in external_roots if path.is_relative_to(root)), None)
            if root is None:
                raise ValueError(f"No verified external root for frozen input: {path}")
            guard_root(root, external_roots[root])
        if not path.is_file() or path.stat().st_size != reference["bytes"] or (
                sha256_file(path) != reference["sha256"]):
            raise ValueError(f"Frozen input size or checksum changed: {path}")


def validate_members(collection, check):
    members = {}
    for name, checksum in referenced_files(collection.document):
        if members.setdefault(name, checksum) != checksum:
            raise ValueError(f"Conflicting native member checksum: {name}")
    for name, checksum in members.items():
        check()
        if sha256_file(collection.path(name)) != checksum:
            raise ValueError(f"Native source member failed verification: {name}")


def copy_native(row, destination, check):
    source = Path(row["native_manifest"])
    if source.name != "patient-anatomy.json":
        raise ValueError("Native reuse requires a patient-anatomy.json manifest")
    source_checksums = {item["path"]: item["sha256"] for item in row["inputs"]}
    origin = PatientImagingCollection(source.parent)
    members = {}
    for name, checksum in referenced_files(origin.document):
        if members.setdefault(name, checksum) != checksum:
            raise ValueError(f"Conflicting native member checksum: {name}")
    for path in (source, *(Path(path) for path in row.get("native_sidecars", []))):
        if path.parent != source.parent or path.name not in {
                "patient-anatomy.json", "workspace-view.json", "anatomy-approaches.json",
                "surgical-toolbox.json"}:
            raise ValueError("Native migration sidecars must belong to the source study")
        members[path.name] = source_checksums[str(path)]
    with tempfile.TemporaryDirectory(prefix=".native-", dir=destination.parent) as staging:
        stage = Path(staging)
        for name, checksum in members.items():
            check()
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin.path(name), target)
            if sha256_file(target) != checksum:
                raise ValueError(f"Source member changed during native copy: {name}")
        copied = PatientImagingCollection(stage)
        if copied.identity() != origin.identity():
            raise ValueError("Native source revision changed during copy")
        current = copied.document.get("snapshot_id")
        if current:
            revision = stage / "revisions" / f"{current}.json"
            revision.parent.mkdir(exist_ok=True)
            shutil.copyfile(stage / source.name, revision)
        stage.rename(destination)


def verify_raw_intake(row, collection):
    """Check the source bytes actually retained by intake against the ledger."""
    expected = {reference["path"]: reference["sha256"] for reference in row["inputs"]}
    for source in row["intake"]["series"]:
        scan = collection.series(source["id"])
        if scan["original"]["sha256"] != expected[source["path"]]:
            raise ValueError(f"Retained scan differs from frozen input: {source['id']}")
        if source.get("sidecar"):
            checksum = scan.get("companions", {}).get("metadata", {}).get("sha256")
            if checksum != expected[source["sidecar"]]:
                raise ValueError(f"Retained sidecar differs from frozen input: {source['id']}")
    for source in row["intake"].get("layers", []):
        layer = collection.layer(source["id"])
        if layer["provenance"]["source"]["sha256"] != expected[source["path"]]:
            raise ValueError(f"Retained annotation differs from frozen input: {source['id']}")


def open_study(row, destination, initialize, check):
    if row["route"] == "raw-intake":
        timepoint = row["timepoint"] if row["timepoint"] in {"preoperative", "postoperative"} else None
        spec = {**row["intake"], "collection": str(destination),
                "patient_id": row["patient_id"], "timepoint": timepoint}
        if (destination / "patient-anatomy.json").exists():
            collection = initialize(spec)
            verify_raw_intake(row, collection)
        else:
            with tempfile.TemporaryDirectory(prefix=".raw-", dir=destination.parent) as staging:
                stage = Path(staging)
                pending = initialize({**spec, "collection": str(stage)})
                verify_raw_intake(row, pending)
                check()
                stage.rename(destination)
            collection = PatientImagingCollection(destination)
    elif row["route"] == "eden-cached-intake":
        from prepare_eden2020_planning import prepare
        if not (destination / "patient-anatomy.json").exists():
            with tempfile.TemporaryDirectory(prefix=".eden-", dir=destination.parent) as staging:
                stage = Path(staging)
                prepare(Path(row["eden_root"]), row["eden_subject"], output=stage,
                        candidate_variants=row["eden_variants"], check=check)
                check()
                source_map = stage / "source-map.json"
                mapping = json.loads(source_map.read_text())
                mapping["collection"] = str(destination / "patient-anatomy.json")
                atomic_json(source_map, mapping)
                stage.rename(destination)
        collection = PatientImagingCollection(destination)
    else:
        if not (destination / "patient-anatomy.json").exists():
            copy_native(row, destination, check)
        collection = PatientImagingCollection(destination)
    if collection.document["patient_id"] != row["patient_id"]:
        raise ValueError("Native collection belongs to a different patient")
    if collection.document.get("study_timepoint") not in {None, row["timepoint"]}:
        raise ValueError("Native collection belongs to a different timepoint")
    binding = {
        "person_id": row["person_id"], "patient_id": row["patient_id"],
        "case_id": row["case_id"], "source_version": row["source_version"],
        "role": row["role"], "timepoint": row["timepoint"],
        "route": row["route"], "exposure": "private",
        "source_hashes": [reference["sha256"] for reference in row["inputs"]],
    }
    current = collection.document.get("migration_source")
    if current is not None and current != binding:
        raise ValueError("Native output already binds another frozen migration row")
    if current is None:
        from resectionlab.anatomy_workspace import read_view, save_view

        view = read_view(collection)
        expected = collection.document.get("snapshot_id")
        collection.document["migration_source"] = binding
        for scan_id, limitation in row.get("scan_coverage", {}).items():
            scan = collection.series(scan_id)
            coverage = scan.get("coverage", {})
            limitations = list(coverage.get("limitations", []))
            limitations.append({"kind": limitation, "basis": "Frozen source inventory"})
            scan["coverage"] = {**coverage, "status": "limited", "limitations": limitations}
        collection.save("Record private migration source identity", expected=expected)
        if view is not None:
            save_view(collection, {**view, "snapshotId": collection.identity()})

    return collection


def discover_brain_estimates(cases, external_roots):
    """Index retained estimate receipts for the exact, admitted raw inputs."""
    eligible = {item["sha256"] for row in cases.values()
                if row["outcome"] == "eligible" and row["route"] == "raw-intake"
                for item in row["inputs"]}
    estimates = defaultdict(list)
    if not eligible:
        return estimates
    for root, identity in external_roots.items():
        try:
            guard_root(root, identity)
        except (OSError, ValueError):
            continue
        cache = root / "cache/inference"
        for path in sorted(cache.glob("*/synthstrip-main/*/inference.json")):
            try:
                if path.resolve(strict=True) != path or path.stat().st_size > 1024**2:
                    continue
                record = json.loads(path.read_text())
                checksum = record["input"]["sha256"]
                if checksum in eligible and record.get("status") == "succeeded":
                    estimates[checksum].append((path, record))
            except (OSError, KeyError, TypeError, ValueError):
                continue
    return estimates


def reuse_brain_estimates(collection, estimates, check):
    """Admit pinnable, source-grid SynthStrip results without rerunning the worker."""
    import nibabel as nib
    import numpy as np

    from resectionlab.anatomy_model_cache import SYNTHSTRIP_SHA, reusable_model_layer
    from resectionlab.anatomy_models import import_proposal
    from resectionlab.anatomy_workspace import read_view, save_view
    from resectionlab.poc_models import MODEL_CONFIG

    initial = collection.identity()
    for scan in collection.document["series"]:
        checksum = scan["original"]["sha256"]
        if checksum not in estimates or any(layer["kind"] == "brain" and layer["source_series"] == scan["id"]
                                             for layer in collection.document["layers"]):
            continue
        for receipt_path, receipt in estimates[checksum]:
            check()
            try:
                if (receipt["model_id"] != "synthstrip-main"
                        or receipt["model"]["version"] != MODEL_CONFIG["synthstrip-main"]["version"]
                        or [item["sha256"] for item in receipt["weights"]] != [SYNTHSTRIP_SHA]):
                    continue
                output = next(item for item in receipt["outputs"] if item["name"] == "brain_mask.nii.gz")
                mask = Path(output["path"])
                if (mask.resolve(strict=True) != mask or not mask.is_relative_to(receipt_path.parent)
                        or output.get("foreground_voxels", 0) <= 0 or sha256_file(mask) != output["sha256"]):
                    continue
                image = nib.load(mask)
                if tuple(image.shape) != tuple(scan["data"]["shape"]) or not np.allclose(
                        image.affine, scan["affine"], atol=1e-5, rtol=0):
                    continue
                candidate = PatientImagingCollection(collection.root, collection.document)
                layer_id = import_proposal(candidate, scan["id"], {
                    "model": "synthstrip", "kind": "brain", "source_series": scan["id"],
                    "source": checksum, "inference": receipt, "path": str(mask),
                    "output_sha256": output["sha256"], "clinical_validation": False,
                    "method": "image_derived_estimate", "evidence_use": "context_only",
                    "version": receipt["model"]["version"], "parameters": receipt["model"]["parameters"],
                    "label_ontology": {"value": 1, "meaning": "Estimated brain envelope"},
                    "inference_receipt_original": collection.original(receipt_path),
                    "model_mask_original": collection.original(mask),
                })
                reused = reusable_model_layer(candidate, scan, {"brain"}, model="synthstrip", native=True)
                if reused is not None and reused["id"] == layer_id:
                    collection.document = candidate.document
                    break
            except (OSError, KeyError, StopIteration, TypeError, ValueError, nib.filebasedimages.ImageFileError):
                continue
    if collection.identity() != initial:
        prior = PatientImagingCollection(collection.root)
        view = read_view(prior)
        collection.save("Reuse checksum-verified source-native brain estimate", expected=initial)
        if view is not None:
            save_view(collection, {**view, "snapshotId": collection.identity()})

def limit_initial_vascular_display(collection):
    """Bound first-load work for studies with dense fine-label annotations."""
    document = collection.document
    if document.get("display", {}).get("initial_layers") is not None:
        return
    available = [row for row in document["layers"]
                 if row.get("evidence_status") != "empty-output"
                 and row.get("availability") != "empty_output"]
    vascular = {"artery", "vein", "sinus", "vessel-candidate", "vessels"}
    if sum(row["kind"] in vascular for row in available) <= 12:
        return
    from resectionlab.anatomy_workspace import read_view

    if read_view(collection) is not None:
        return
    initial = []
    for kind, limit in (("artery", 2), ("vein", 1), ("sinus", 1), ("skull", 1), ("brain", 1)):
        initial.extend([row["id"] for row in available if row["kind"] == kind][:limit])
    if not initial:
        initial = [row["id"] for row in available if row["kind"] in vascular][:2]
    expected = collection.identity()
    document["display"] = {**document.get("display", {}), "initial_layers": initial}
    collection.save("Show representative layers on first open", expected=expected)


def prepare_case(row, destination, receipt, checkpoint, initialize, prepare, cancelled, estimates):
    def check():
        if cancelled():
            raise InterruptedError("Migration cancelled")

    def stage_update(stage):
        receipt["stages"][stage["id"]] = stage
        checkpoint()

    collection = open_study(row, destination, initialize, check)
    if row["route"] == "raw-intake":
        reuse_brain_estimates(collection, estimates, check)
    report = prepare(collection, on_stage=stage_update,
                     cancelled=cancelled, retry_failed=receipt["retry_failed"],
                     **row.get("prepare_options", {}))
    check()
    collection = PatientImagingCollection(destination)
    if row["route"] == "raw-intake":
        limit_initial_vascular_display(collection)
    validate_members(collection, check)
    manifest = destination / "patient-anatomy.json"
    outcomes = report["outcomes"]
    incomplete = any(stage["status"] not in {"succeeded", "complete", "cached", "reused", "available"}
                     for stage in outcomes)
    receipt.update(
        outcome="migrated_with_limitations" if incomplete else "migrated",
        manifest=str(manifest), manifest_sha256=sha256_file(manifest),
        snapshot_id=collection.identity(), stages=outcomes,
        reason="Native study prepared with recorded stage outcomes",
    )


def migrate(ledger, cases, authorities, checksum, output, *, only, retry_failed, initialize):
    from resectionlab.anatomy_preparation import prepare_anatomy

    external_roots = {Path(item["path"]): item["root_id"] for item in ledger["external_roots"]}
    guard_output(output, ledger["version"], external_roots)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "migration-receipt.json"
    stop = False

    def cancel(_signum, _frame):
        nonlocal stop
        stop = True

    def check():
        if stop:
            raise InterruptedError("Migration cancelled")

    previous_handler = signal.signal(signal.SIGINT, cancel)
    try:
        with (output / ".migration.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("A migration already writes this output directory") from exc
            result = (json.loads(result_path.read_text()) if result_path.exists() else
                      {"schema": RECEIPT_SCHEMA, "ledger_sha256": checksum, "cases": {}})
            if result["schema"] != RECEIPT_SCHEMA or result["ledger_sha256"] != checksum:
                raise ValueError("Output belongs to another frozen inventory; use a new version")

            def checkpoint():
                guard_output(output, ledger["version"], external_roots)
                atomic_json(result_path, result)

            selected = {key: cases[key] for key in only or cases}
            estimates = discover_brain_estimates(selected, external_roots)
            for key, row in selected.items():
                if stop:
                    break
                if row["outcome"] != "eligible":
                    continue
                receipt = result["cases"].get(key)
                destination = output / key
                if destination.is_symlink() or (receipt is None and destination.exists()):
                    raise ValueError(f"Preserve existing non-migration output: {destination}")
                receipt = result["cases"].setdefault(key, {})
                previous_status = receipt.get("outcome")
                manifest = destination / "patient-anatomy.json"
                if previous_status in {"migrated", "migrated_with_limitations"}:
                    if not manifest.is_file() or sha256_file(manifest) != receipt["manifest_sha256"]:
                        receipt.update(outcome="preserved_newer_or_missing", reason="Prior migration snapshot differs")
                        checkpoint()
                        continue
                    if not retry_failed or previous_status == "migrated":
                        continue
                if previous_status == "preserved_newer_or_missing" or previous_status == "failed" and not retry_failed:
                    continue
                receipt.update(outcome="running", reason="", stages={}, retry_failed=retry_failed)
                checkpoint()
                print(f"{key}: preparing native study", flush=True)
                try:
                    verify_inputs(row, authorities, external_roots, check)
                    prepare_case(row, destination, receipt, checkpoint, initialize,
                                 prepare_anatomy, lambda: stop, estimates)
                except InterruptedError as exc:
                    receipt.update(outcome="cancelled", reason=str(exc))
                    stop = True
                except Exception as exc:
                    receipt.update(outcome="failed", reason=str(exc))
                    print(f"{key}: {exc}", file=sys.stderr, flush=True)
                checkpoint()
            print(result_path, flush=True)
            if stop:
                return 130
            return int(any(row["outcome"] == "failed" for key, row in result["cases"].items()
                           if not only or key in only))
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="Frozen local migration inventory")
    parser.add_argument("--only", action="append", help="Include a specific migration key (repeatable)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", type=Path, metavar="OUTPUT", help="Prepare eligible rows sequentially")
    mode.add_argument("--results", type=Path, metavar="OUTPUT", help="Summarize this ledger with saved migration outcomes")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed cases and recoverable incomplete stages")
    args = parser.parse_args()
    try:
        ledger, cases, authorities, checksum = load_ledger(args.ledger)
        if args.only and not set(args.only) <= cases.keys():
            raise ValueError("--only specifies an unknown migration key")
        if args.run is None:
            saved = {}
            if args.results:
                receipt = json.loads((args.results / "migration-receipt.json").read_text())
                if receipt["schema"] != RECEIPT_SCHEMA or receipt["ledger_sha256"] != checksum:
                    raise ValueError("The result directory does not belong to this frozen ledger")
                saved = receipt["cases"]
            counts = Counter(saved.get(key, cases[key])["outcome"] for key in args.only or cases)
            print(json.dumps(dict(sorted(counts.items()))))
            return 0
        from standardize_case_library import initialize

        return migrate(ledger, cases, authorities, checksum, args.run.expanduser().resolve(),
                       only=args.only, retry_failed=args.retry_failed, initialize=initialize)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
