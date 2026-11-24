#!/usr/bin/env python3
"""Run native TOF component review once per restricted EDEN2020 subject."""

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import nibabel as nib
import numpy as np

from resectionlab.data_root import ROOT_ENV, atomic_json, checked_root, safe_path
from resectionlab.patient_anatomy import PatientImagingCollection, digest, file_hash
from resectionlab.vascular_cleanup import (
    HIDDEN, RETAINED, UNCERTAIN, ComponentPolicy, classify_components, review_component_visibility,
)


DATA_ROOT = Path("/Volumes/shafrir/old coding projects/RessectionLabData")
VOLUME_UUID = "9CC3623C-E38B-3EEE-B6BB-B9A06C193700"
ACQUISITION = Path("catalog/eden2020-glioma-5714812/acquisition.json")
COLLECTION = Path("prepared/Eden 2020 ResectionLab Projects (restricted)/collection.json")
RUN = Path("qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1")
SCHEMA = "eden2020-restricted-vascular-review-v1"


def load_json(path):
    return json.loads(path.read_text())


def write_partition(path, value):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def inventory(data_root):
    acquired = load_json(safe_path(data_root, ACQUISITION))
    collected = load_json(safe_path(data_root, COLLECTION))
    if (acquired["doi"] != "10.5281/zenodo.5714812" or acquired["access"] != "restricted"
            or not acquired["complete"] or acquired["missing"] or collected["access"] != "restricted"):
        raise ValueError("Unexpected restricted EDEN2020 source inventory")
    expected = {Path(name).stem for name in acquired["files"] if re.fullmatch(r"PTNT_\d{4}\.zip", name)}
    cases = {case["subject"]: case for case in collected["cases"]}
    if set(cases) != expected or len(cases) != len(collected["cases"]) or len(cases) != 30:
        raise ValueError("Published archive and prepared subject inventories differ")
    return cases


def native_signal(collection, source):
    collection.require_native_geometry(source)
    path = collection.path(source["original"]["path"])
    if file_hash(path) != source["original"]["sha256"]:
        raise ValueError("Native TOF acquisition checksum differs")
    # Sequentially decode the compact original. Random samples against a
    # cold 3-D array on the rotating data volume are substantially slower.
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def process(case, out, data_root):
    path = Path(case["manifest"]).resolve()
    if not path.is_relative_to(safe_path(data_root, COLLECTION).parent) or path.name != "patient-anatomy.json":
        raise ValueError("Prepared study is outside the private source collection")
    source_manifest_hash = file_hash(path)
    if source_manifest_hash != case["sha256"]:
        raise ValueError("Prepared study differs from its frozen collection manifest")
    subject = case["subject"]
    collection = PatientImagingCollection(path.parent)
    if collection.document["patient_id"] != f"EDEN2020:{subject}":
        raise ValueError("Prepared EDEN2020 subject does not match its source identity")
    key = {
        "schema": SCHEMA,
        "subject": subject,
        "source_manifest_sha256": source_manifest_hash,
        "source_snapshot_id": collection.identity(),
        "component_policy": asdict(ComponentPolicy()),
    }
    report = out / "report.json"
    partition_path = out / "partition.npy"
    if report.is_file():
        previous = load_json(report)
        if any(previous.get(k) != v for k, v in key.items()):
            raise ValueError("Existing case belongs to a different source or policy; preserve it in its run")
        if previous["status"] == "missing_input" or (
            partition_path.is_file() and previous["partition_sha256"] == file_hash(partition_path)
        ):
            return previous, True

    sources = [row for row in collection.document["series"] if row["modality"] == "TOF"]
    if not sources:
        result = {**key, "status": "missing_input", "reason": "No acquired TOF in this prepared source study"}
        atomic_json(report, result)
        return result, False
    if len(sources) != 1:
        raise ValueError("An explicit TOF source choice is required")
    source = sources[0]
    candidates = [
        row for row in collection.document["layers"]
        if row["source_series"] == source["id"] and row["provenance"].get("model") == "vesselfm"
    ]
    if len(candidates) != 1:
        raise ValueError("Expected a single pinned native vesselFM baseline")
    baseline = candidates[0]
    if (baseline["data"]["shape"] != source["data"]["shape"]
            or not np.array_equal(baseline["affine"], source["affine"])):
        raise ValueError("vesselFM mask does not match the TOF native grid")
    signal = native_signal(collection, source)
    mask = collection.mask(baseline)
    scores = collection.array(baseline["probability"])
    if scores.shape != mask.shape or scores.dtype.kind != "f":
        raise ValueError("Retained vesselFM scores differ from the native grid")
    if any(not np.isfinite(scores[k:k + 16]).all()
           or np.any((scores[k:k + 16] > 0.5) != mask[k:k + 16])
           for k in range(0, mask.shape[0], 16)):
        raise ValueError("Retained scores do not reproduce the vesselFM mask at its isovalue")
    partition, analysis = classify_components(mask, signal, scores, source["affine"])
    if not np.array_equal(partition > 0, mask):
        raise ValueError("The reversible partition differs from baseline support")
    write_partition(partition_path, partition)
    result = {
        **key,
        "status": "candidate" if analysis["hidden_voxels"] else "retained_baseline",
        "source_tof_sha256": source["original"]["sha256"],
        "source_signal_sha256": source["data"]["sha256"],
        "source_mask_sha256": baseline["data"]["sha256"],
        "source_probability_sha256": baseline["probability"]["sha256"],
        "source_series": source["id"],
        "source_layer_id": baseline["id"],
        "source_model": baseline["provenance"].get("inference", {}).get("model"),
        "source_shape": list(mask.shape),
        "source_affine_ras_mm": source["affine"],
        "source_to_structural_status": collection.document["transforms"][source["id"]]["status"],
        "partition_sha256": file_hash(partition_path),
        "partition_states": {
            "0": "background",
            str(RETAINED): "retained candidate",
            str(UNCERTAIN): "uncertain and visible",
            str(HIDDEN): "proposed display exclusion; baseline retained",
        },
        "source_review_status": "pending",
        "application_review_status": "pending",
        "analysis": analysis,
    }
    atomic_json(report, result)
    return result, False


def save_component_review(case, out, *, restore, exclude, reason):
    parent = out / "report.json"
    report = load_json(parent)
    partition = out / "partition.npy"
    if (report.get("schema") != SCHEMA or case["subject"] != report["subject"]
            or case["sha256"] != report["source_manifest_sha256"]
            or file_hash(partition) != report["partition_sha256"]):
        raise ValueError("The base report does not bind this exact subject and partition")
    target = safe_path(out, "reviewed")
    target.mkdir(exist_ok=True)
    manual_review = {
        "restored_components": sorted(set(restore)),
        **({"excluded_components": sorted(set(exclude))} if exclude else {}),
        "reason": reason.strip(),
        "parent_report_sha256": file_hash(parent),
        "parent_partition_sha256": report["partition_sha256"],
    }
    revised_report, revised_partition = target / "report.json", target / "partition.npy"
    with (target / ".review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if revised_report.exists():
            existing = load_json(revised_report)
            if existing.get("manual_review") != manual_review:
                raise ValueError("An existing component review records a different decision")
            if revised_partition.is_file() and file_hash(revised_partition) == existing["partition_sha256"]:
                return existing, True
        values, analysis = review_component_visibility(
            np.load(partition, mmap_mode="r", allow_pickle=False), report["analysis"],
            restore=restore, exclude=exclude,
        )
        write_partition(revised_partition, values)
        result = {
            **report,
            "manual_review": manual_review,
            "status": "candidate" if analysis["hidden_voxels"] else "retained_baseline",
            "source_review_status": "explicit component review",
            "partition_sha256": file_hash(revised_partition),
            "analysis": analysis,
        }
        atomic_json(revised_report, result)
        return result, False


def process_graph(case, out):
    from resectionlab import vascular_source_branches

    report_path = out / "report.json"
    if not report_path.exists():
        return {"status": "pending_component_review"}, False
    report = load_json(report_path)
    if report["status"] == "missing_input":
        return {"status": "missing_input"}, True
    stamp = {
        "algorithm_sha256": file_hash(vascular_source_branches.__file__),
        "partition_sha256": report["partition_sha256"],
        "source_manifest_sha256": report["source_manifest_sha256"],
    }
    partition = out / "partition.npy"
    if file_hash(partition) != stamp["partition_sha256"]:
        raise ValueError("Reviewed partition checksum differs")
    graph_path, gap_path, receipt_path = (out / name for name in (
        "source-branches.json", "source-gaps.json", "graph-receipt.json"
    ))
    if all(path.is_file() for path in (receipt_path, graph_path, gap_path)):
        previous = load_json(receipt_path)
        if (all(previous.get(k) == v for k, v in stamp.items())
                and previous["graph_sha256"] == file_hash(graph_path)
                and previous["gap_sha256"] == file_hash(gap_path)):
            return previous, True
    manifest = Path(case["manifest"])
    if file_hash(manifest) != stamp["source_manifest_sha256"]:
        raise ValueError("Original native manifest changed since component review")
    collection = PatientImagingCollection(manifest.parent)
    source = collection.series(report["source_series"])
    signal = native_signal(collection, source)
    mask = np.load(partition, mmap_mode="r", allow_pickle=False) > 0
    graph = vascular_source_branches.build_source_branches(
        mask, source["affine"], source_scan_sha256=report["source_tof_sha256"], signal=signal
    )
    gaps = vascular_source_branches.evaluate_source_gaps(graph, mask, signal)
    atomic_json(graph_path, graph)
    atomic_json(gap_path, gaps)
    result = {
        **stamp,
        "status": graph["coverage"]["status"],
        "graph_sha256": file_hash(graph_path),
        "gap_sha256": file_hash(gap_path),
        "graph_coverage": graph["coverage"],
        "gap_coverage": gaps["coverage"],
    }
    atomic_json(receipt_path, result)
    return result, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", action="append", help="Repeat to run individual pilot subjects")
    parser.add_argument("--cohort", action="store_true", help="Run every authoritative restricted subject")
    parser.add_argument("--graph", action="store_true", help="Audit branches and possible gaps using existing case partitions")
    parser.add_argument("--restore", action="append", help="Restore an explicitly reviewed hidden component to uncertain")
    parser.add_argument("--exclude", action="append", help="Hide an explicitly reviewed uncertain component")
    parser.add_argument("--reason", help="Source evidence for an explicit visibility decision")
    args = parser.parse_args()
    if bool(args.subject) == args.cohort:
        parser.error("Select --subject or --cohort")
    component_review = bool(args.restore or args.exclude)
    if component_review and (args.cohort or args.graph or len(args.subject) != 1 or not (args.reason or "").strip()):
        parser.error("Component review requires exactly one --subject and a source-review --reason")
    if args.reason and not component_review:
        parser.error("--reason belongs to a component review")
    data_root = checked_root(os.environ.get(ROOT_ENV, DATA_ROOT), expected_uuid=VOLUME_UUID)
    cases = inventory(data_root)
    subjects = sorted(cases if args.cohort else set(args.subject))
    if not set(subjects) <= cases.keys():
        parser.error("Unknown restricted EDEN2020 subject")
    root = safe_path(data_root, RUN)
    directory = safe_path(root, "cases")
    directory.mkdir(parents=True, exist_ok=True)
    if component_review:
        subject = subjects[0]
        out = safe_path(directory, subject)
        report, cached = save_component_review(
            cases[subject], out, restore=args.restore or (), exclude=args.exclude or (), reason=args.reason
        )
        print(subject, report["manual_review"], "(cached)" if cached else "", flush=True)
        return 0
    stage = "graphs" if args.graph else "processing"
    receipt = root / f"{stage}.json"
    failed = False
    with (root / f".{stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {
            "schema": SCHEMA,
            "acquisition_sha256": file_hash(safe_path(data_root, ACQUISITION)),
            "collection_sha256": file_hash(safe_path(data_root, COLLECTION)),
            "policy_sha256": digest(asdict(ComponentPolicy())),
        }
        run = load_json(receipt) if receipt.exists() else {
            **binding,
            "cases": {subject: {"status": "pending"} for subject in sorted(cases)},
        }
        if any(run.get(key) != value for key, value in binding.items()) or set(run["cases"]) != cases.keys():
            raise ValueError("Existing run belongs to a different cohort or cleanup policy")
        for subject in subjects:
            out = safe_path(directory, subject)
            out.mkdir(exist_ok=True)
            try:
                if args.graph:
                    row, cached = process_graph(cases[subject], out)
                else:
                    report, cached = process(cases[subject], out, data_root)
                    row = {
                        "status": report["status"],
                        "report": str((out / "report.json").relative_to(root)),
                        "hidden_voxels": report.get("analysis", {}).get("hidden_voxels"),
                        "hidden_components": report.get("analysis", {}).get("hidden_components"),
                    }
            except (OSError, ValueError, KeyError) as error:
                row = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
                failed = True
                cached = False
            print(subject, row, "(cached)" if cached else "", flush=True)
            run["cases"][subject] = row
            atomic_json(receipt, run)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
