#!/usr/bin/env python3
"""Acquire and inspect the frozen real RESECT Case3 cavity-label pilot.

Separate source-rights acquisition, original-byte acquisition and structural
QC. No model, registration, resampling, training or Case4 measurement access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesHeaderParser
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from real_intake_io import atomic_preserve, check_deadline, supervise, termination_cleanup, verify_source_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/annotations/resect-seg-v1"
COHORT = ROOT / "manifests/resect-component-cohort-v1.json"
COHORT_SHA = "4fc51f01e42253d395660d850cda2296a3e7f23e5a5ade80bbd5a3ed411b6a77"
MANIFEST = ROOT / "manifests/resect-case3-cavity-pilot-v1.json"
SOURCES = (
    {"path": "rights/README.txt", "bytes": 1656,
     "source_url": "https://osf.io/download/4mkfg/",
     "metadata_url": "https://api.osf.io/v2/files/65cddcd16d0cb801d21a96bc/",
     "expected_md5": "bae7e9b837bfc3b53d7b74d750df0b4b",
     "sha256": "0a75fef344487ee2649a831919388940f2000bf4cb936979e0da2f73a9aa81a9",
     "kind": "annotation_rights_and_release_notes", "file_revision": 2},
    {"path": "originals/Case3-US-during.nii.gz", "bytes": 9156131,
     "source_url": "https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/RESECT/NIFTI/Case3/US/Case3-US-during.nii.gz",
     "expected_md5": "5043491e6b4cea1fe3be1003a42a2cf5", "sha256": None,
     "kind": "acquired_during_resection_ultrasound", "license": "CC-BY-4.0"},
    {"path": "originals/Case3-US-during-resection.nii.gz", "bytes": 28518,
     "source_url": "https://osf.io/download/64cd21fe2fe4962f1561b1e2/?revision=2",
     "metadata_url": "https://api.osf.io/v2/files/64cd21fe2fe4962f1561b1e2/versions/2/",
     "expected_md5": "569676cbc4bd98a800497c03f91959d9",
     "sha256": "88a8534d2a6fd90d7abf4655dccf2b75a9a87d45adb82e29f1ef197f8eb3ba63",
     "kind": "human_reviewed_visible_cavity_annotation", "file_revision": 2,
     "license": "CC-BY-NC-SA-4.0"},
)
SOURCE_CODE = ("scripts/acquire_resect_cavity.py", "scripts/real_intake_io.py",
               "src/resectionlab/imaging.py", "src/resectionlab/core.py", "src/resectionlab/data_policy.py",
               "src/resectionlab/structural_evidence.py")


def encode(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declaration() -> dict:
    if sha(COHORT) != COHORT_SHA:
        raise ValueError("Frozen RESECT family roles changed")
    cohort = json.loads(COHORT.read_bytes())
    selected = [row for row in cohort["members"] if row["patient_group"] == "RESECT:Case3"]
    if len(selected) != 1 or selected[0]["role"] != "TRAIN":
        raise ValueError("Case3 is not the frozen TRAIN pilot")
    return {"schema": "resect-case3-cavity-pilot-v1", "patient_group": "RESECT:Case3", "role": "TRAIN",
            "cohort_sha256": COHORT_SHA, "image_release_doi": "10.11582/2017.00004",
            "annotation_release_doi": "10.17605/OSF.IO/JV8BK", "annotation_dataset_version": "v1 2024-02-15",
            "files": list(SOURCES), "scientific_payload_bytes": 9184649,
            "purpose": "Noncommercial visible-cavity observation-component research",
            "annotation_ancestry": "manual_contours + morphological_interpolation + expert_review",
            "learned_teacher": "none_reported_in_documented_protocol",
            "methods": "https://doi.org/10.1002/mp.17317 sections2.3 and2.7",
            "limits": "Visible dark US cavity may omit blood-filled regions; not complete removed tissue or surgical reward",
            "time_availability": "during-resection acquisition; not available to preoperative planning",
            "max_float32_image_bytes": 512 * 1024**2, "maximum_grid_corner_difference_mm": .01,
            "training_admitted": False, "recorded_rl_transitions": 0, "optimizer_updates": 0}


def require_manifest() -> dict:
    expected = declaration()
    if MANIFEST.read_bytes() != encode(expected):
        raise ValueError("Source manifest differs from the fixed two-file pilot")
    return expected


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(url: str, source: dict) -> None:
    # http.client includes a malformed request target in InvalidURL messages.
    # Refuse it here before a signed query can reach an exception/log.
    if any(not 33 <= ord(character) <= 126 for character in url):
        raise ValueError("Unreviewed source URL encoding; no request sent")
    parsed = urlparse(url)
    allowed = {"osf.io", "files.osf.io", "files.de-1.osf.io"}
    # OSF's authenticated redirect now uses this content-addressed bucket.
    # Admit only the exact reviewed object for this frozen annotation/notice;
    # the complete response still has to match its declared size and hashes.
    digest = source.get("sha256")
    if (source["kind"] in {"annotation_rights_and_release_notes", "human_reviewed_visible_cavity_annotation"}
            and isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)
            and parsed.path == f"/cos-osf-prod-files-de-1/{digest}"):
        allowed.add("storage.googleapis.com")
    if source["kind"] == "acquired_during_resection_ultrasound":
        allowed = {"s3.nird.sigma2.no"}
        if url != source["source_url"]:
            raise ValueError("Original image must retain its exact NIRD source URL")
    if (parsed.scheme != "https" or parsed.hostname not in allowed or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("Unreviewed source redirect destination; no request sent there")


def safe_route(url: str) -> dict:
    # Never put expiring redirect query credentials into public receipts/logs.
    parsed = urlparse(url)
    return {"host": parsed.hostname, "path": parsed.path,
            "full_url_sha256": hashlib.sha256(url.encode()).hexdigest()}


def system_curl_environment() -> dict:
    # Use the existing macOS system trust diagnosed for this exact source.
    # Ambient CA/backend overrides must not silently change that trust choice.
    return {key: value for key, value in os.environ.items()
            if key not in {"CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_SSL_BACKEND"}}


def system_curl_binding(*, timeout: float = 5.) -> dict:
    executable = Path("/usr/bin/curl")
    binding = {"executable": str(executable), "platform": sys.platform,
               "macos_version": platform.mac_ver()[0], "trust": "existing_macos_system_store"}
    if sys.platform != "darwin" or not executable.is_file():
        return {**binding, "available": False}
    try:
        result = subprocess.run([str(executable), "-q", "--version"], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=timeout, check=False,
                                env=system_curl_environment())
    except (OSError, subprocess.TimeoutExpired):
        return {**binding, "available": False}
    version = result.stdout.decode("ascii", errors="replace").splitlines()[0] if result.stdout else ""
    return {**binding, "available": result.returncode == 0 and "(SecureTransport)" in version,
            "sha256": sha(executable), "version": version[:1024]}


def nird_response_headers(path: Path) -> tuple[int, object]:
    # Proxy CONNECT and informational responses may precede the final response.
    # Keep raw headers local; only reviewed status/length/encoding enter receipts.
    with path.open("rb") as handle:
        data = handle.read(65537)
    if len(data) > 65536:
        raise ValueError("Original image response headers exceed byte limit")
    blocks = [block for block in data.replace(b"\r\n", b"\n").split(b"\n\n")
              if block.startswith(b"HTTP/")]
    if not blocks:
        raise ValueError("Original image response headers missing")
    status_line, separator, fields = blocks[-1].partition(b"\n")
    parts = status_line.split()
    if not separator or len(parts) < 2 or len(parts[1]) != 3 or not parts[1].isdigit():
        raise ValueError("Original image response status invalid")
    return int(parts[1]), BytesHeaderParser().parsebytes(fields + b"\n\n")


def transfer_nird_original(source: dict, attempt: Path, *, deadline: float, events: list) -> dict:
    """One exact NIRD GET through verified native macOS trust; never a fallback."""
    if source != SOURCES[1]:
        raise ValueError("System curl is restricted to the exact frozen NIRD original image")
    validate_url(source["source_url"], source)
    check_deadline(deadline)
    target = DATA / source["path"]
    if target.exists():
        digest = verify_source_file(target, source, deadline=deadline)
        return {"path": source["path"], "status": "existing_verified", "sha256": digest, "bytes": source["bytes"]}
    client = system_curl_binding(timeout=min(5., max(.001, deadline - time.monotonic())))
    check_deadline(deadline)
    if not client["available"]:
        raise ValueError("Verified macOS system curl SecureTransport is unavailable")
    event = {"source_path": source["path"], **safe_route(source["source_url"]), "transport": client}
    events.append(event)
    number = len(events)
    atomic_preserve(attempt / f"request-{number:02d}.json", encode(event))
    partial = attempt / (target.name + ".partial")
    headers_path = attempt / f"nird-headers-{number:02d}.txt"
    try:
        # curl receives already-exclusive file descriptors. It cannot reopen or
        # replace a source/destination path; it remains in the supervised worker's
        # process group so the outer watchdog also cleans up this child.
        with partial.open("xb") as output, headers_path.open("xb") as headers:
            remaining = deadline - time.monotonic()
            check_deadline(deadline)
            command = ["/usr/bin/curl", "-q", "--no-location", "--max-redirs", "0",
                       "--silent", "--fail", "--proto", "=https", "--tlsv1.2", "--retry", "0",
                       "--connect-timeout", str(min(30., remaining)), "--max-time", str(remaining),
                       "--max-filesize", str(source["bytes"]), "--header", "Accept-Encoding: identity",
                       "--dump-header", f"/dev/fd/{headers.fileno()}", "--output", "-", source["source_url"]]
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.DEVNULL,
                                       pass_fds=(headers.fileno(),), env=system_curl_environment())
            try:
                event["curl_exit_code"] = process.wait(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                event["deadline_exceeded"] = True
                raise TimeoutError("Original image system curl deadline exceeded; no automatic retry") from None
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            output.flush()
            os.fsync(output.fileno())
        check_deadline(deadline)
        if event["curl_exit_code"]:
            raise RuntimeError(f"Original image system curl failed (exit {event['curl_exit_code']}); no automatic retry")
        status, headers = nird_response_headers(headers_path)
        lengths = headers.get_all("Content-Length", [])
        encodings = headers.get_all("Content-Encoding", [])
        event.update(status=status, content_length=lengths[0] if len(lengths) == 1 else None,
                     content_encoding=encodings[0] if len(encodings) == 1 else "identity" if not encodings else "invalid")
        if (status != 200 or lengths != [str(source["bytes"])]
                or (encodings and encodings != ["identity"])):
            raise ValueError("Original image response status, encoding or declared byte length differs")
        digest = verify_source_file(partial, source, deadline=deadline)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(partial, target)
        partial.unlink()
        return {"path": source["path"], "status": "downloaded_verified", "sha256": digest,
                "bytes": source["bytes"], "transport": client}
    finally:
        atomic_preserve(attempt / f"response-{number:02d}.json", encode(event))


def transfer(source: dict, attempt: Path, *, deadline: float, events: list) -> dict:
    if source["kind"] == "acquired_during_resection_ultrasound":
        return transfer_nird_original(source, attempt, deadline=deadline, events=events)
    target = DATA / source["path"]
    if target.exists():
        digest = verify_source_file(target, source, deadline=deadline)
        return {"path": source["path"], "status": "existing_verified", "sha256": digest, "bytes": source["bytes"]}
    opener = build_opener(NoRedirect())
    url = source["source_url"]
    partial = attempt / (target.name + ".partial")
    for _ in range(5):
        check_deadline(deadline)
        validate_url(url, source)
        event = {"source_path": source["path"], **safe_route(url)}
        events.append(event)
        atomic_preserve(attempt / f"request-{len(events):02d}.json", encode(event))
        try:
            response = opener.open(Request(url, headers={"Accept-Encoding": "identity"}),
                                   timeout=min(30., max(.1, deadline - time.monotonic())))
        except HTTPError as error:
            event.update(status=error.code, retry_after=error.headers.get("Retry-After"))
            atomic_preserve(attempt / f"response-{len(events):02d}.json", encode(event))
            if error.code in (301, 302, 303, 307, 308) and error.headers.get("Location"):
                url = urljoin(url, error.headers["Location"])
                error.close()
                continue
            error.close()
            raise RuntimeError(f"Source HTTP {event['status']}; no automatic retry") from None
        with response:
            event.update(status=response.status)
            atomic_preserve(attempt / f"response-{len(events):02d}.json", encode(event))
            if (response.status != 200 or response.geturl() != url
                    or response.headers.get("Content-Encoding", "identity") != "identity"
                    or int(response.headers.get("Content-Length", -1)) != source["bytes"]):
                raise ValueError("Source response identity, encoding or declared byte length differs")
            size = 0
            with partial.open("xb") as handle:
                while block := response.read(64 * 1024):
                    check_deadline(deadline)
                    size += len(block)
                    if size > source["bytes"]:
                        raise ValueError("Source exceeds pinned byte limit")
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        digest = verify_source_file(partial, source, deadline=deadline)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(partial, target)
        partial.unlink()
        return {"path": source["path"], "status": "downloaded_verified", "sha256": digest, "bytes": size}
    raise ValueError("Source exceeds five-request redirect limit")


def structural_qc() -> dict:
    import nibabel as nib
    import numpy as np
    import resectionlab.imaging as imaging
    if Path(imaging.__file__).resolve() != (ROOT / "src/resectionlab/imaging.py").resolve():
        raise ValueError("QC import does not resolve to the reviewed checkout")
    inspect_nifti = imaging.inspect_nifti
    _maximum_corner_displacement = imaging._maximum_corner_displacement
    headers = [inspect_nifti(DATA / source["path"]) for source in SOURCES[1:]]
    if any(math.prod(header["shape"]) * 4 > 512 * 1024**2 for header in headers):
        raise ValueError("Image exceeds float32 decoded-size allowance")
    if headers[0]["shape"] != headers[1]["shape"]:
        raise ValueError("Original image and source mask shapes differ")
    corner = _maximum_corner_displacement(np.asarray(headers[0]["affine_ras_mm"]),
                                         np.asarray(headers[1]["affine_ras_mm"]), headers[0]["shape"])
    if corner > .01:
        raise ValueError("Source image/mask physical grids differ; no implicit repair")
    image, labels = [nib.load(DATA / source["path"]).get_fdata(dtype=np.float32) for source in SOURCES[1:]]
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite acquired image or source labels")
    vocabulary = np.unique(labels).tolist()
    if vocabulary != [0., 1.]:
        raise ValueError(f"Source does not contain the declared binary cavity vocabulary: {vocabulary}")
    return {"headers": headers, "maximum_grid_corner_difference_mm": corner,
            "image_minimum": float(image.min()), "image_maximum": float(image.max()),
            "label_values": vocabulary, "annotated_cavity_voxels": int(np.count_nonzero(labels)),
            "status": "passed_byte_header_scalar_and_binary_grid_checks",
            "anatomical_review": "pending", "clinical_accuracy": "not_established",
            "training_admitted": False, "spatial_planning_admitted": False}


def code_binding() -> dict:
    import importlib.metadata
    return {"files": {name: sha(ROOT / name) for name in SOURCE_CODE}, "python": sys.version,
            "numpy": importlib.metadata.version("numpy"), "nibabel": importlib.metadata.version("nibabel"),
            "nird_original_transport": system_curl_binding()}


def worker(scope: str, attempt: Path) -> None:
    source = json.loads((attempt / "source.json").read_bytes())
    if code_binding() != source:
        raise ValueError("Execution source changed before worker")
    manifest = require_manifest()
    events, files = [], []
    record = {"scope": scope, "patient_group": "RESECT:Case3", "role": "TRAIN", "status": "started",
              "manifest_sha256": sha(MANIFEST), "cohort_sha256": COHORT_SHA,
              "execution_source_sha256": sha(attempt / "source.json"), "events": events, "files": files,
              "optimizer_updates": 0, "recorded_rl_transitions": 0, "training_admitted": False}
    started = time.monotonic()
    try:
        if scope != "rights":
            verify_source_file(DATA / SOURCES[0]["path"], SOURCES[0])
        if scope == "qc":
            for entry in SOURCES[1:]:
                files.append({"path": entry["path"], "sha256": verify_source_file(DATA / entry["path"], entry)})
            record["structural_qc"] = structural_qc()
        else:
            for entry in SOURCES[:1] if scope == "rights" else SOURCES[1:]:
                files.append(transfer(entry, attempt, deadline=started + 295, events=events))
        if code_binding() != source or require_manifest() != manifest:
            raise ValueError("Execution inputs changed during worker")
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        atomic_preserve(attempt / "worker-result.json", encode(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "rights", "acquire", "qc"])
    parser.add_argument("--worker-attempt", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        atomic_preserve(MANIFEST, encode(declaration()))
        print(json.dumps({"manifest": str(MANIFEST.relative_to(ROOT)), "sha256": sha(MANIFEST)}))
        return
    if args.worker_attempt is not None:
        watchdog = threading.Timer(300, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            worker(args.action, args.worker_attempt)
        finally:
            watchdog.cancel()
        return
    require_manifest()
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "pilot.lock").open("a+") as lock, termination_cleanup():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = DATA / "attempts" / (args.action + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        attempt.mkdir(parents=True)
        result = {"scope": args.action, "status": "interrupted", "max_seconds": 300}
        started = time.monotonic()
        try:
            source = code_binding()
            for name, digest in source["files"].items():
                content = (ROOT / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("Source changed during snapshot")
                atomic_preserve(attempt / "source-snapshot" / name, content)
            atomic_preserve(attempt / "source.json", encode(source))
            atomic_preserve(attempt / "manifest.json", MANIFEST.read_bytes())
            result.update(manifest_sha256=sha(MANIFEST), execution_source_sha256=sha(attempt / "source.json"))
            command = [sys.executable, str(Path(__file__).resolve()), args.action, "--worker-attempt", str(attempt)]
            status, code = supervise(command, attempt / "worker.log", deadline=started + 300,
                on_start=lambda pid: atomic_preserve(attempt / "started.json", encode({"worker_pid": pid})))
            result.update(status=status, exit_code=code)
        except BaseException as error:
            result.update(status="setup_or_supervision_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            atomic_preserve(attempt / "supervision.json", encode(result))
            print(json.dumps({"attempt": str(attempt.relative_to(ROOT)), **result}), flush=True)
        if result["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
