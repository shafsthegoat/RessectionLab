"""Verify saved fixed-plane evidence and compressed bytes without image decoding."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREFIX = HERE.relative_to(ROOT).as_posix()
RELEASE_COMMIT = "981f6c9c0d94291db5fc199b3ec145e5e9bb15d0"
SOURCE_COMMIT = "abd2af40d91c647e29ba21de526a4bf84e92752c"
RELEASE_SHA = "ff8c8d9005e0e60055de5867fae52ee5fcd59cfad59dbb80b2ce2ced0000f942"
ARCHIVE_SHA = "74d03e76fed057e3d6f50000ba11fe1b3a7288f3ed56af0950aeca8aa62f085c"
SOURCE = "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
ORIGINAL_SHA = "b3b9fa69b87221062261b8b16fbddfdac2c678104b354ba371b069fd784e3625"
inputs = {}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def path(relative):
    pieces = PurePosixPath(relative)
    need(not pieces.is_absolute() and ".." not in pieces.parts, "unsafe input path")
    current = ROOT
    for piece in pieces.parts:
        current /= piece
        need(not current.is_symlink(), "symlink input")
    return current


def read(relative, cap=2 * 1024**2):
    with path(relative).open("rb") as stream:
        data = stream.read(cap + 1)
    need(len(data) <= cap, "oversized saved metadata or display artifact")
    inputs[relative] = {"sha256": digest(data), "bytes": len(data)}
    return data


def record(name):
    return json.loads(read(f"{PREFIX}/{name}"))


def bound(binding):
    data = read(binding["path"])
    need(digest(data) == binding["sha256"], "bound bytes differ")
    return data


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=10)


def audit():
    release = record("release.json")
    root_result = record("root-terminal.json")
    execution_release = record("execution-release-record.json")
    archive_record = record("source-archive.json")
    need(inputs[f"{PREFIX}/release.json"]["sha256"] == RELEASE_SHA and
         release["released"] is True and release["action"] == "fixed_three_planes_two_windows_once",
         "unexpected render release")
    need(execution_release["release_sha256"] == RELEASE_SHA and
         execution_release["source_commit"] == archive_record["source_commit"] == SOURCE_COMMIT,
         "source or release identity mismatch")
    need(execution_release["source_archive_sha256"] == archive_record["archive_sha256"] == ARCHIVE_SHA,
         "archive identity mismatch")
    need(digest(read(f"{PREFIX}/source.tar.gz")) == ARCHIVE_SHA, "source archive changed")
    for name in ("release.json", "execution-release-record.json", "source-archive.json", "source.tar.gz"):
        need(git("show", f"{RELEASE_COMMIT}:{PREFIX}/{name}") == read(f"{PREFIX}/{name}"),
             "release differs from committed bytes")
    git("merge-base", "--is-ancestor", SOURCE_COMMIT, RELEASE_COMMIT)
    git_time = datetime.fromisoformat(git("show", "-s", "--format=%cI", RELEASE_COMMIT).decode().strip())
    released = datetime.fromisoformat(execution_release["released_at_utc"])
    started = datetime.fromisoformat(root_result["started_at_utc"])
    finished = datetime.fromisoformat(root_result["finished_at_utc"])
    need(git_time < started and released < started < finished, "release did not precede execution")

    # This is the source-code tar archive, never the patient gzip file.
    with tarfile.open(path(f"{PREFIX}/source.tar.gz"), "r:gz") as archive:
        files = [m for m in archive.getmembers() if not m.isdir()]
        need(len(files) == len({m.name for m in files}) == 13 and
             {m.name for m in files} == set(archive_record["files_sha256"]), "source inventory mismatch")
        for item in files:
            need(item.isfile() and item.size <= 2 * 1024**2, "unsafe source archive member")
            data = archive.extractfile(item).read()
            need(digest(data) == archive_record["files_sha256"][item.name] and
                 data == read(item.name) == git("show", f"{SOURCE_COMMIT}:{item.name}"),
                 "source archive/current/Git mismatch")

    renderer_path = "scripts/render_rhuh_fixed_planes.py"
    renderer = read(renderer_path)
    need(digest(renderer) == release["renderer_sha256"], "renderer release mismatch")
    # Parse literal configuration without importing the renderer, decoder or imaging libraries.
    constants = {}
    for node in ast.parse(renderer).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    need(constants["SOURCE"] == release["source"] == SOURCE and
         constants["ORIGINAL_SHA"] == release["original_compressed_sha256"] == ORIGINAL_SHA,
         "renderer input constants mismatch")
    need(constants["DECODER_SHA"] == release["decoder_sha256"] == digest(read(constants["DECODER_PATH"])),
         "decoder identity mismatch")
    inspection = json.loads(bound(constants["REPORT"]))
    prior_audit = json.loads(bound(constants["AUDIT"]))
    for key in ("PARENT", "TERMINAL"):
        bound(constants[key])
    need(constants["REPORT"]["sha256"] == release["inspection_sha256"] and
         constants["AUDIT"]["sha256"] == release["inspection_audit_sha256"] and
         prior_audit["accepted_for_saved_result_consistency"] is True, "inspection identity mismatch")

    output = release["output_directory"]
    need(output == "outputs/rhuh-fixed-plane-qc-v1/run-first", "unexpected render output")
    result = json.loads(read(output + "/receipt.json"))
    supervision = "outputs/rhuh-fixed-plane-qc-v1/supervision-first"
    supervisor = json.loads(read(supervision + "/supervisor.json"))
    progress = json.loads(read(supervision + "/supervisor-progress.json"))
    worker = json.loads(read(supervision + "/worker.log"))
    need(worker == result and root_result["supervisor"] == supervisor, "worker or parent result differs")
    need(result["schema"] == "resectionlab.rhuh-fixed-plane-display.v1" and
         result["accepted"] is True and result["status"] == "fixed_planes_rendered_review_required" and
         root_result["accepted_for_render_completion"] is True and
         root_result["source_and_release_unchanged"] is True, "render completion missing")
    need(supervisor["status"] == "complete" and supervisor["returncode"] == 0 and
         supervisor["timed_out"] is False and supervisor["termination_reason"] is None and
         supervisor["automatic_retry"] is False, "unaccepted supervised render")
    need(supervisor["declaration_sha256"] == progress["declaration_sha256"] == RELEASE_SHA and
         progress["termination_reason"] is None, "supervisor release mismatch")
    need(0 < progress["elapsed_seconds"] <= supervisor["seconds"] <= root_result["whole_call_seconds"] <= 60,
         "saved wall-time bounds exceeded")
    need(supervisor["sampled_peak_rss_bytes"] == progress["sampled_peak_rss_bytes"] <= 2 * 1024**3 and
         0 < supervisor["rss_samples"] == progress["samples"] == 8 and
         supervisor["sampling_interval_seconds"] == 0.2, "saved sampled RSS mismatch")
    need(execution_release["settings"] == {"max_wall_seconds": 60.0, "max_rss_bytes": 2 * 1024**3} and
         execution_release["supervisor_source_sha256"] == digest(read("scripts/preflight_real_spatial_policy.py")),
         "supervisor setting or source mismatch")
    need(root_result["actual_command"] == [str(ROOT / ".venv/bin/python"), str(ROOT / renderer_path),
         "--execute", "--release", f"{PREFIX}/release.json", "--release-sha256", RELEASE_SHA],
         "recorded command mismatch")
    need(result["release"] == {"path": f"{PREFIX}/release.json", "sha256": RELEASE_SHA} and
         result["inspection"] == constants["REPORT"] and result["inspection_audit"] == constants["AUDIT"] and
         result["renderer_sha256"] == release["renderer_sha256"] and
         result["decoder_sha256"] == release["decoder_sha256"] and
         result["source"] == SOURCE and result["original_compressed_sha256"] == ORIGINAL_SHA,
         "render receipt provenance mismatch")
    need(result["scientific_use_released"] is False and result["anatomical_validation"] is False and
         result["decoded_snapshot_passes"] == 2, "render scope or declared decode count changed")

    shape, indices, windows = result["shape"], result["indices"], result["windows"]
    need(shape == list(constants["SHAPE"]) == inspection["shape"] == [240, 240, 155] and
         indices == list(constants["INDICES"]) == [120, 120, 77], "fixed plane selection changed")
    need(windows == [list(w) for w in constants["WINDOWS"]] and
         windows[0] == [inspection["intensity"]["scaled_finite_min"], inspection["intensity"]["scaled_finite_max"]]
         and windows[1] == [-3, 3], "fixed display windows changed")
    affine = result["affine_ras_mm"]
    need(affine == inspection["geometry"]["qform"]["affine_in_source_units"] and
         inspection["geometry"]["spatial_units"] == "mm", "display affine differs from saved coded geometry")
    spacing, codes = [], []
    for column in range(3):
        v = [affine[row][column] for row in range(3)]
        norm = math.sqrt(math.fsum(x*x for x in v))
        axis = max(range(3), key=lambda row: abs(v[row]))
        need(norm > 0 and all(v[row] == 0 for row in range(3) if row != axis), "noncardinal saved affine")
        spacing.append(norm)
        codes.append(("R", "A", "S")[axis] if v[axis] > 0 else ("L", "P", "I")[axis])
    opposite = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
    expected_planes = []
    for fixed, index in enumerate(indices):
        horizontal, vertical = [axis for axis in range(3) if axis != fixed]
        expected_planes.append({"fixed_axis": fixed, "index": index,
            "horizontal_axis": horizontal, "vertical_axis": vertical,
            "left": opposite[codes[horizontal]], "right": codes[horizontal],
            "bottom": opposite[codes[vertical]], "top": codes[vertical],
            "horizontal_step_mm": spacing[horizontal], "vertical_step_mm": spacing[vertical],
            "extent": [-0.5, shape[horizontal] - 0.5, -0.5, shape[vertical] - 0.5]})
    need(result["planes"] == expected_planes, "plane descriptor arithmetic mismatch")
    need(set(result["artifacts"]) == {"fixed-planes.png", "fixed-planes.svg"}, "unexpected display artifacts")
    for name, identity in result["artifacts"].items():
        data = read(output + "/" + name)
        need(len(data) == identity["bytes"] and digest(data) == identity["sha256"], "render artifact changed")

    request = json.loads(bound(inspection["bindings"]["request"]))
    payload = request["payload"]
    need(payload["sha256"] == ORIGINAL_SHA and payload["measured_compressed_bytes"] == 6337221,
         "original payload metadata changed")
    original = path(payload["path"])
    before = original.stat()
    need(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o444, "original no longer regular/read-only")
    sha256 = hashlib.sha256()
    length = 0
    with original.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        while chunk := stream.read(1024**2):
            length += len(chunk)
            need(length <= 64 * 1024**2, "original compressed cap exceeded")
            sha256.update(chunk)
        ended = os.fstat(stream.fileno())
    after = original.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)
    need(stamp(before) == stamp(opened) == stamp(ended) == stamp(after) and
         length == 6337221 and sha256.hexdigest() == ORIGINAL_SHA, "original changed")
    for name, identity in list(inputs.items()):
        need(digest(path(name).read_bytes()) == identity["sha256"], "saved input drift during audit")
    return {"schema": "resectionlab.rhuh-fixed-plane-saved-result-audit.v1",
            "accepted_for_saved_render_consistency": True,
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "reviewer_source": {"path": Path(__file__).relative_to(ROOT).as_posix(),
                                "sha256": digest(Path(__file__).read_bytes())},
            "release_commit": RELEASE_COMMIT, "source_commit": SOURCE_COMMIT,
            "committed_release_preceded_execution": True, "archive_current_and_git_files_verified": 13,
            "source_archive_sha256": ARCHIVE_SHA,
            "observed_receipt_bounds": {"whole_call_seconds": root_result["whole_call_seconds"],
                "supervised_seconds": supervisor["seconds"], "sampled_worker_rss_bytes": supervisor["sampled_peak_rss_bytes"],
                "rss_samples": supervisor["rss_samples"], "worker_returncode": 0, "recorded_automatic_retry": False},
            "declared_fixed_display": {"shape": shape, "indices": indices, "windows": windows,
                                       "independently_recomputed_plane_descriptors": expected_planes,
                                       "recorded_snapshot_decompressions": result["decoded_snapshot_passes"]},
            "render_artifacts": result["artifacts"],
            "original_compressed_bytes": {"path": payload["path"], "bytes": length,
                                          "sha256": sha256.hexdigest(), "mode": "0o444", "stable_during_read": True},
            "consulted_inputs": inputs, "checker_failures": [], "patient_image_decodes": 0,
            "renderer_invocations": 0, "network_requests": 0,
            "scope": "Saved metadata/source/history audit and byte hashing only. Neither MRI gzip/header/voxels nor PNG/SVG pixels were decoded. No renderer/decoder imports or rerender occurred.",
            "limits": ["Saved PNG/SVG hashes and plane descriptors are verified; pixel fidelity and qualitative anatomy are not independently assessed here.",
                       "The fixed planes sample only three locations and cannot establish whole-volume anatomy, normalization suitability, a support mask, registration or clinical validity.",
                       "Resource measurements are recorded sampled worker RSS and recorded elapsed times, not continuous memory bounds or independently reconstructed process execution.",
                       "Two snapshot decompressions are the renderer's reported count, not new decompressions by this audit.",
                       "The original remains hash-consistent and read-only; this is not an OS immutable-file guarantee. Case import, support derivation and learning remain unreleased."]}


if __name__ == "__main__":
    result = audit()
    destination = HERE / "independent-result-audit.json"
    need(not destination.exists(), "preserve any existing audit")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": True, "receipt_sha256": digest(destination.read_bytes()),
                      "reviewer_sha256": result["reviewer_source"]["sha256"]}))
