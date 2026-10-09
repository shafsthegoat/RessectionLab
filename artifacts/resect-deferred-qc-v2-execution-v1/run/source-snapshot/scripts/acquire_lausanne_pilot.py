#!/usr/bin/env python3
"""Freeze grouped Lausanne roles, then acquire one original T1/TOF pair.

No derivatives, model weights, labels, learned preprocessing or training. Reuse
the existing bounded, resumable OpenNeuro transfer implementation. The pilot
is an ingestion check; the declared full TRAIN cohort remains the next stage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time
from urllib.request import Request

from acquire_btc_case import acquire_file, open_without_redirect, verify_file
from acquire_public_case import AcquisitionError, sha256_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/anatomy/ds003949-v1.0.1"
COHORT = ROOT / "manifests/lausanne-component-cohort-v1.json"
MANIFEST = ROOT / "manifests/lausanne-original-pilot-v1.json"
RESULT = ROOT / "artifacts/lausanne-original-pilot-v1/acquisition.json"
COMMIT = "896b8846d899acee68c0246cc987ca96e77267d4"
RAW = f"https://raw.githubusercontent.com/OpenNeuroDatasets/ds003949/{COMMIT}/"
S3 = "https://s3.amazonaws.com/openneuro.org/ds003949/"
PREFIX = "sub-000/ses-20110101/anat/sub-000_ses-20110101_"
NAMESPACE = "RessectionLab:Lausanne:component:v1"
METADATA = {
    "README": "b66e136ee47c7a520d8c22380a9bdf7600b9bc387f5991a89f28c4d205faa23f",
    "dataset_description.json": "39baee3ef549835c0ae13c83eed3235a6597d7ea24b1f94bf361fc1f32fffc35",
    "participants.tsv": "4f3fde676fec18cc6f3cfaddc5e3c22c0d194881a8c65f4af60c8d7042cf2c96",
}
COUNTS = {"control": {"TRAIN": 89, "SELECT": 19, "MEASUREMENT_EVAL": 19},
          "patient": {"TRAIN": 110, "SELECT": 24, "MEASUREMENT_EVAL": 23}}
TOPCOW = {f"sub-{n:03}" for n in [0, 2, 5, 6, 7, 15, 21, 30, 33, 36,
                                  319, 334, 337, 344, 345, 347, 349, 351, 352, 354]}


def expected_files() -> list[dict]:
    records = []
    for suffix, size, md5, version, pointer_sha in [
        ("T1w", 8250277, "97a96ee22f374a7a4d166e50b1460013", "Fd_dldJuHqxHdkXkR8no4V7bSeXyhYuZ",
         "eb1d6dfc53943c93c83e258cbf7750808921a72b1972c5b240a7197f7ac91571"),
        ("angio", 28408253, "66f8a1a067e70c7bfa756133d46f1074", "WcfGrEnFSmBdKz4mehhm24qZYBAf2Q5n",
         "f316b6b2330814a0158ec191e184563f1552eb67c35236c9e97a2f3bd9f39320"),
    ]:
        path = PREFIX + suffix + ".nii.gz"
        records.append({"path": path, "bytes": size, "expected_md5": md5, "sha256": None,
                        "git_url": RAW + path, "pointer_sha256": pointer_sha,
                        "source_url": S3 + path + "?versionId=" + version})
    for suffix, size, digest in [
        ("T1w", 1630, "adf8342710f042d4370f82ceddefe859c04381d1d86ebdae850936be1830f0a6"),
        ("angio", 1569, "63983f59dc5c1495394c4625a49a29089de0bac96e62295e340ef3339af144d5"),
    ]:
        path = PREFIX + suffix + ".json"
        records.append({"path": path, "bytes": size, "sha256": digest,
                        "git_url": RAW + path, "source_url": RAW + path})
    return records


def fetch_metadata(url: str, expected_sha: str) -> bytes:
    if not url.startswith(RAW):
        raise AcquisitionError("Metadata must use the pinned official Git release")
    with open_without_redirect(Request(url, headers={"Accept-Encoding": "identity"}), timeout=45) as response:
        if response.status != 200 or response.geturl() != url:
            raise AcquisitionError("Unexpected metadata source response")
        payload = response.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024 or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise AcquisitionError("Metadata size/hash mismatch")
    return payload


def preserve(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise AcquisitionError(f"Preserve existing different record: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def json_bytes(record: dict) -> bytes:
    return (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()


def members_from_source(payload: bytes) -> list[dict]:
    if hashlib.sha256(payload).hexdigest() != METADATA["participants.tsv"]:
        raise AcquisitionError("Participant source differs from the pinned release")
    people = {}
    for row in csv.DictReader(payload.decode().splitlines(), delimiter="\t"):
        person = people.setdefault(row["participant_id"], {"group": row["group"], "sessions": []})
        if person["group"] != row["group"] or row["exam_date"] in person["sessions"]:
            raise AcquisitionError("Conflicting or duplicate participant/session metadata")
        person["sessions"].append(row["exam_date"])
    if len(people) != 284 or sum(len(p["sessions"]) for p in people.values()) != 296:
        raise AcquisitionError("Cohort counts differ from source audit")
    members = []
    for group, counts in COUNTS.items():
        ids = [key for key, value in people.items() if value["group"] == group]
        ids.sort(key=lambda key: (hashlib.sha256(f"{NAMESPACE}:{key}".encode()).hexdigest(), key))
        if group == "control":
            ids.remove("sub-000")
            ids.insert(0, "sub-000")  # Explicit prospective ingestion anchor.
        if len(ids) != sum(counts.values()):
            raise AcquisitionError("Source groups differ from declared role counts")
        offset = 0
        for role, count in counts.items():
            for subject in ids[offset:offset + count]:
                members.append({"subject": subject, "canonical_person": "Lausanne:" + subject,
                                "group": group, "role": role,
                                "sessions": sorted(people[subject]["sessions"]),
                                "topcow_overlap": subject in TOPCOW})
            offset += count
    return sorted(members, key=lambda member: member["subject"])


def prepare() -> None:
    for name, digest in METADATA.items():
        path = DATA / "source-metadata" / name
        payload = path.read_bytes() if path.exists() else fetch_metadata(RAW + name, digest)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise AcquisitionError("Cached source metadata hash mismatch")
        preserve(path, payload)
    description = json.loads((DATA / "source-metadata/dataset_description.json").read_bytes())
    if description["License"] != "CC0":
        raise AcquisitionError("Source rights changed")
    members = members_from_source((DATA / "source-metadata/participants.tsv").read_bytes())
    if not COHORT.exists():
        preserve(COHORT, json_bytes({"schema": "lausanne-component-cohort-v1", "git_commit": COMMIT,
            "declared_at": datetime.now(timezone.utc).isoformat(), "metadata_sha256": METADATA,
            "split_namespace": NAMESPACE, "role_counts_by_group": COUNTS,
            "rule": "Pin 000 TRAIN; SHA256 UTF-8 namespace:subject rank within control/patient; role blocks in listed order",
            "members": members, "scope": "single-site anatomy component development; not surgical RL or site holdout",
            "derivative_rule": "All sessions and derivatives inherit canonical person's role",
            "overlap_source": "https://zenodo.org/records/15692630/preview/MRA_Lausanne.zip?include_deleted=0"}))
    cohort = json.loads(COHORT.read_bytes())
    if cohort["members"] != members or cohort["git_commit"] != COMMIT:
        raise AcquisitionError("Preserve existing different cohort declaration")
    for entry in expected_files()[:2]:
        pointer = fetch_metadata(entry["git_url"], entry["pointer_sha256"])
        expected_key = f"MD5E-s{entry['bytes']}--{entry['expected_md5']}.nii.gz"
        if not pointer.decode().endswith(expected_key):
            raise AcquisitionError("Unexpected source annex pointer")
        preserve(DATA / "source-metadata" / (Path(entry["path"]).name + ".annex-pointer.txt"), pointer)
    preserve(MANIFEST, json_bytes({"schema": "lausanne-original-pilot-v1", "git_commit": COMMIT,
        "license": "CC0", "subject": "sub-000", "session": "ses-20110101", "role": "TRAIN",
        "cohort_sha256": sha256_file(COHORT), "files": expected_files(),
        "payload_bytes": 36661729, "max_wall_seconds": 240, "max_decoded_image_bytes": 512 * 1024**2,
        "excluded": ["all derivatives", "atlas labels", "model weights", "other patient images"],
        "direct_recorded_rl_transitions": 0}))
    print(json.dumps({"status": "roles_and_pilot_frozen", "people": len(members),
                      "roles": {role: sum(m['role'] == role for m in members) for role in COUNTS['control']},
                      "cohort_sha256": sha256_file(COHORT), "manifest_sha256": sha256_file(MANIFEST)}))


def validate_declaration() -> dict:
    record = json.loads(MANIFEST.read_bytes())
    cohort = json.loads(COHORT.read_bytes())
    members = members_from_source((DATA / "source-metadata/participants.tsv").read_bytes())
    if (record["files"] != expected_files() or record["git_commit"] != COMMIT
            or record["cohort_sha256"] != sha256_file(COHORT) or record["role"] != "TRAIN"
            or record["subject"] != "sub-000" or record["license"] != "CC0"
            or cohort["members"] != members or cohort["git_commit"] != COMMIT
            or record["payload_bytes"] != 36661729):
        raise AcquisitionError("Frozen source/role/pilot contract changed")
    return record


def acquire() -> None:
    from resectionlab.imaging import inspect_nifti
    import nibabel as nib
    import numpy as np

    record = validate_declaration()
    if RESULT.exists():
        raise AcquisitionError("Pilot receipt already exists; preserve it")
    if shutil.disk_usage(DATA).free < 1024**3:
        raise AcquisitionError("At least 1 GiB free required for bounded pilot")
    started = time.monotonic()
    acquired = []
    for entry in record["files"]:
        status = acquire_file(entry, DATA)
        path = DATA / entry["path"]
        verify_file(path, entry)
        acquired.append({"path": entry["path"], "status": status, "bytes": path.stat().st_size,
                         "sha256": sha256_file(path), "source_url": entry["source_url"]})
        print(json.dumps(acquired[-1]), flush=True)
    qc = []
    for entry in record["files"][:2]:
        path = DATA / entry["path"]
        item = {"path": entry["path"], "status": "unassessed"}
        try:
            header = inspect_nifti(path)
            if int(np.prod(header["shape"])) * 4 > 512 * 1024**2:
                raise AcquisitionError("Decoded image exceeds pilot memory bound")
            array = nib.load(path).get_fdata(dtype=np.float32)
            if not np.isfinite(array).all():
                raise AcquisitionError("Image contains nonfinite values")
            item.update(status="passed_header_and_finite_values", header=header,
                        minimum=float(array.min()), maximum=float(array.max()),
                        nonzero_voxels=int(np.count_nonzero(array)), voxels=int(array.size))
            del array
        except TimeoutError:
            raise
        except (ValueError, OSError, AcquisitionError) as error:
            item.update(status="qc_failed", reason=str(error))
        qc.append(item)
    report = {"status": "acquired", "source_manifest_sha256": sha256_file(MANIFEST),
        "cohort_sha256": sha256_file(COHORT), "script_sha256": sha256_file(Path(__file__)),
        "acquired_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "subject": "sub-000", "role": "TRAIN", "files": acquired, "integrity_qc": qc,
        "unique_people_acquired": 1, "component_optimizer_updates": 0, "recorded_rl_transitions": 0,
        "limitations": ["No source vascular annotation acquired", "No clinical anatomy review",
            "T1/TOF registration unverified; shared subject does not prove voxel correspondence",
            "One-site control anatomy ingestion only; no glioma surgical planning or RL validation",
            "TopCoW sub-000 derivative inherits TRAIN; never independent evaluation"]}
    preserve(RESULT, json_bytes(report))
    print(json.dumps({"result": str(RESULT), "qc": [item["status"] for item in qc]}))
    if any(item["status"] == "qc_failed" for item in qc):
        raise AcquisitionError("Acquired bytes retained; inspect failed QC receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "acquire"])
    args = parser.parse_args()
    def timeout(signum, frame):
        raise TimeoutError("Pilot exceeded fixed 240-second wall limit; verified files and partials retained")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(240)
    try:
        prepare() if args.action == "prepare" else acquire()
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
