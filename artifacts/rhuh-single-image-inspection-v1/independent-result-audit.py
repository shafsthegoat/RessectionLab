"""Audit saved inspection records; never decompress or interpret image bytes."""

from datetime import datetime, timezone
import hashlib
import itertools
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
SOURCE = "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
RELEASE_COMMIT = "dd633a21edfb48b27cacb72da1c2e042f6cd7c21"
SOURCE_COMMIT = "285a00fde70c11c097f052b75d23645e80b6c521"
PAYLOAD_SHA = "b3b9fa69b87221062261b8b16fbddfdac2c678104b354ba371b069fd784e3625"
TOKEN = "a5c579950d0431d04928e5b59e91ce9d"
MANIFEST_SHA = "a8d5770eabaad9cad264df46e2e1fea82bc12617e0d17800914ed011407ec9fa"
consulted = {}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local(relative):
    item = PurePosixPath(relative)
    need(not item.is_absolute() and ".." not in item.parts, "unsafe local path")
    path = ROOT
    for part in item.parts:
        path = path / part
        need(not path.is_symlink(), "symlink input")
    return path


def raw(relative, limit=2 * 1024**2):
    path = local(relative)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    need(len(data) <= limit, "oversized metadata/source")
    consulted[relative] = {"sha256": sha(data), "bytes": len(data)}
    return data


def record(name):
    return json.loads(raw(f"{PREFIX}/{name}"))


def bound(binding):
    data = raw(binding["path"])
    need(sha(data) == binding["sha256"], "binding digest mismatch")
    return data


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=10)


def determinant(a):
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def audit():
    manifest = record("launch-manifest.json")
    need(consulted[f"{PREFIX}/launch-manifest.json"]["sha256"] == MANIFEST_SHA,
         "unexpected launch manifest")
    request = json.loads(bound(manifest["request"]))
    release = json.loads(bound(manifest["release"]))
    for key in ("inspector", "launcher", "public_transfer"):
        bound(manifest[key])
    for key, value in request.items():
        if isinstance(value, dict) and set(value) == {"path", "sha256"}:
            bound(value)
    for binding in request["acquisition_sources"].values():
        bound(binding)

    started = record("execution-started.json")
    parent = record("parent-summary.json")
    terminal = record("root-terminal.json")
    release_record = record("execution-release-record.json")
    archive_record = record("source-archive.json")
    for value in release_record.values():
        if isinstance(value, dict) and set(value) == {"path", "sha256"}:
            bound(value)
    need(all(x["source"] == SOURCE for x in (manifest, request, release, parent)),
         "public source mismatch")
    need(manifest["released"] is True and release["released"] is True,
         "inspection was not released")
    need(release["action"] == "bounded_header_and_voxel_inspection_once",
         "unexpected release action")
    need(release["request"] == manifest["request"], "release request mismatch")
    need(started["release_commit"] == terminal["release_commit"] == RELEASE_COMMIT,
         "release commit mismatch")
    need(started["maximum_attempts"] == release_record["maximum_attempts"] == 1,
         "attempt declaration mismatch")
    need(started["manifest_sha256"] == MANIFEST_SHA and
         started["release_sha256"] == manifest["release"]["sha256"], "launch pins mismatch")
    expected_command = [str(ROOT / ".venv/bin/python"), str(ROOT / manifest["launcher"]["path"]),
                        "--manifest", f"{PREFIX}/launch-manifest.json", "--manifest-sha256", MANIFEST_SHA,
                        "--request-sha256", manifest["request"]["sha256"],
                        "--release-sha256", manifest["release"]["sha256"]]
    need(started["command"] == expected_command, "recorded command differs from release")
    for name in ("launch-manifest.json", "request.json", "release.json", "execution-release-record.json",
                 "source-archive.json", "source.tar.gz"):
        need(git("show", f"{RELEASE_COMMIT}:{PREFIX}/{name}") == raw(f"{PREFIX}/{name}"),
             "release commit contents changed")
    source_commit = archive_record["source_commit"]
    need(source_commit == SOURCE_COMMIT, "unexpected archive source commit")
    for older, newer in ((source_commit, release_record["preparation_commit"]),
                         (release_record["preparation_commit"], RELEASE_COMMIT)):
        git("merge-base", "--is-ancestor", older, newer)
    release_git_time = datetime.fromisoformat(git("show", "-s", "--format=%cI", RELEASE_COMMIT).decode().strip())
    release_time = datetime.fromisoformat(release_record["released_utc"])
    start_time = datetime.fromisoformat(started["started_utc"])
    end_time = datetime.fromisoformat(terminal["finished_utc"])
    need(release_git_time < start_time and release_time < start_time < end_time,
         "release did not precede execution")

    archive_path = f"{PREFIX}/source.tar.gz"
    archive_bytes = raw(archive_path)
    need(sha(archive_bytes) == archive_record["archive_sha256"] == terminal["source_archive_sha256"],
         "source archive digest mismatch")
    need(len(archive_bytes) == archive_record["archive_bytes"], "source archive length mismatch")
    # Only the SOURCE archive is decompressed. The patient .nii.gz is hash-read below.
    with tarfile.open(local(archive_path), "r:gz") as archive:
        members = [item for item in archive.getmembers() if not item.isdir()]
        need(len(members) == 23 == archive_record["archive_file_count"], "archive file count mismatch")
        need(len({item.name for item in members}) == len(members), "duplicate archive member")
        need({item.name for item in members} == set(archive_record["files_sha256"]), "archive inventory mismatch")
        for item in members:
            need(item.isfile() and 0 <= item.size <= 2 * 1024**2, "unsafe source archive member")
            data = archive.extractfile(item).read()
            need(sha(data) == archive_record["files_sha256"][item.name], "archived source hash mismatch")
            need(data == raw(item.name) == git("show", f"{source_commit}:{item.name}"),
                 "archived source differs from current or original commit")

    output = manifest["output_directory"]
    stdout = raw(f"{output}/stdout.json", 65536)
    stderr = raw(f"{output}/stderr.txt", 65536)
    launch_bytes = raw(f"{output}/launch.json", 65536)
    need(launch_bytes == raw(f"{PREFIX}/parent-summary.json"), "saved and parent launch records differ")
    result = json.loads(stdout)
    need(not stderr and not terminal["stderr"], "unexpected stderr")
    need(terminal["exit_code"] == 0 and terminal["accepted"] is True and
         parent["accepted"] is True and parent["inspector_exit_code"] == 0,
         "terminal launch was not accepted")
    need(terminal["parent_summary_sha256"] == sha(launch_bytes), "parent hash mismatch")
    need(parent["stdout_sha256"] == sha(stdout), "child output hash mismatch")
    need(parent["launcher_exit_zero_required"] is True and parent["source_closure_unchanged"] is True,
         "missing final launcher conditions")
    need(0 < parent["elapsed_seconds"] <= terminal["root_elapsed_seconds"] < 60,
         "recorded elapsed bounds exceeded")
    need(0 < parent["peak_sampled_combined_rss_bytes"] <= 2 * 1024**3 and
         parent["rss_samples"] == 3 and parent["rss_sampling_interval_seconds"] == 0.1,
         "recorded RSS bounds mismatch")
    need(result["bindings"] == {"request": manifest["request"], "release": manifest["release"],
                                "reconciliation": request["reconciliation"], "source": SOURCE},
         "result binding mismatch")
    need(parent["request"] == manifest["request"] and parent["release"] == manifest["release"] and
         parent["manifest"] == {"path": f"{PREFIX}/launch-manifest.json", "sha256": MANIFEST_SHA},
         "parent binding mismatch")
    need(result["schema"] == "resectionlab.rhuh-single-image-inspection.v1" and
         result["status"] == parent["status"] == "format_inspected_scientific_use_unreleased" and
         result["binary_structure_valid"] is True and result["gzip_footer_and_single_member_verified"] is True,
         "inspection success scope mismatch")
    need(result["clinical_validation"] is False and result["scientific_use_released"] is False and
         parent["scientific_use_released"] is False and
         release_record["clinical_use_authorized"] is False and
         release_record["case_import_or_learning_authorized"] is False,
         "scientific authorization overstatement")

    shape = result["shape"]
    need(len(shape) == 3 and all(type(x) is int and x > 0 for x in shape), "invalid saved shape")
    count = math.prod(shape)
    need(result["nifti_version"] == 1 and result["dtype"] == "<f4", "unexpected reported storage")
    need(count == result["voxel_count"] == 8928000 and
         count * 4 == result["declared_voxel_bytes"] == 35712000, "storage product mismatch")
    need(result["voxel_offset"] == 352 and result["extension_count"] == 0 and
         result["uncompressed_bytes"] == result["voxel_offset"] + count * 4,
         "offset or uncompressed length arithmetic mismatch")
    intensity = result["intensity"]
    need(intensity["scaled_finite_voxels"] + intensity["scaled_nonfinite_voxels"] == count and
         intensity["scaled_nonfinite_voxels"] == 0, "finite count mismatch")
    lo, mean, hi = (intensity[k] for k in ("scaled_finite_min", "scaled_finite_mean", "scaled_finite_max"))
    need(all(math.isfinite(x) for x in (lo, mean, hi)) and lo <= mean <= hi, "intensity arithmetic mismatch")
    need(intensity["effective_slope"] == 1 and intensity["effective_intercept"] == 0 and
         intensity["scaling_enabled"] is True, "unexpected recorded scaling")

    geometry = result["geometry"]
    q, s = geometry["qform"], geometry["sform"]
    need(q["present"] is True and q["code"] == 1 and q["finite"] is True and q["invertible"] is True,
         "qform status mismatch")
    need(s == {"affine_in_source_units": None, "axis_codes": None, "code": 0,
               "finite": None, "invertible": None, "present": False}, "unexpected coded sform")
    a = q["affine_in_source_units"]
    need(len(a) == 4 and all(len(row) == 4 for row in a) and a[3] == [0, 0, 0, 1] and
         all(math.isfinite(x) for row in a for x in row), "invalid reported affine")
    columns = [[a[row][col] for row in range(3)] for col in range(3)]
    zooms = [math.sqrt(sum(x*x for x in col)) for col in columns]
    det = determinant(a)
    need(det != 0 and zooms == geometry["zooms_in_source_units"] == [1, 1, 1], "affine scale mismatch")
    need(all(sum(columns[i][k]*columns[j][k] for k in range(3)) == 0
             for i in range(3) for j in range(i)), "reported affine is not orthogonal")
    axes = [max(range(3), key=lambda row: abs(col[row])) for col in columns]
    need(len(set(axes)) == 3, "ambiguous dominant reported axes")
    codes = [("R", "A", "S")[row] if col[row] > 0 else ("L", "P", "I")[row]
             for col, row in zip(columns, axes)]
    need(codes == q["axis_codes"] == ["L", "P", "S"] and geometry["spatial_units"] == "mm",
         "reported orientation/unit mismatch")
    corners = [[sum(a[row][col] * point[col] for col in range(3)) + a[row][3] for row in range(3)]
               for point in itertools.product(*[(-0.5, n - 0.5) for n in shape])]
    bounds = [[min(p[k] for p in corners), max(p[k] for p in corners)] for k in range(3)]
    need(geometry["max_full_cell_corner_difference_mm"] is None and geometry["issues"] == [] and
         geometry["anatomical_registration_accepted"] is False and
         geometry["scanner_or_atlas_provenance_verified"] is False,
         "geometry interpretation scope mismatch")

    payload = request["payload"]
    need(payload["sha256"] == result["compressed_sha256"] == PAYLOAD_SHA and
         payload["measured_compressed_bytes"] == result["compressed_bytes"] == 6337221,
         "compressed identity mismatch")
    need(payload["publisher_algorithm_confirmed"] is False and
         payload["expected_compressed_bytes"] is None and payload["prior_length_match_claim"] is False,
         "provider algorithm or prior length overstatement")
    path = local(payload["path"])
    original_stat = path.stat()
    need(stat.S_ISREG(original_stat.st_mode) and stat.S_IMODE(original_stat.st_mode) == 0o444,
         "original is not regular read-only preserved file")
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    length = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        while chunk := stream.read(1024**2):
            length += len(chunk)
            need(length <= 64 * 1024**2, "compressed byte cap exceeded")
            sha256.update(chunk)
            md5.update(chunk)
        final_opened = os.fstat(stream.fileno())
    final_stat = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)
    need(stamp(original_stat) == stamp(opened) == stamp(final_opened) == stamp(final_stat),
         "original changed during compressed hash read")
    need(length == 6337221 and sha256.hexdigest() == PAYLOAD_SHA and
         md5.hexdigest() == result["candidate_compressed_md5"] == TOKEN,
         "original compressed bytes differ")
    for relative, identity in list(consulted.items()):
        need(sha(local(relative).read_bytes()) == identity["sha256"], "consulted record changed during audit")

    return {
        "schema": "resectionlab.rhuh-saved-inspection-independent-audit.v1",
        "accepted_for_saved_result_consistency": True,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_source": {"path": Path(__file__).relative_to(ROOT).as_posix(),
                            "sha256": sha(Path(__file__).read_bytes())},
        "release_and_history": {"release_commit": RELEASE_COMMIT, "source_commit": source_commit,
                                "release_preceded_execution": True, "archive_files_verified": 23,
                                "archive_current_and_committed_bytes_match": True,
                                "declared_maximum_attempts": 1},
        "observed_receipt_bounds": {"root_elapsed_seconds": terminal["root_elapsed_seconds"],
                                    "launcher_elapsed_seconds": parent["elapsed_seconds"],
                                    "peak_sampled_combined_rss_bytes": parent["peak_sampled_combined_rss_bytes"],
                                    "rss_samples": parent["rss_samples"],
                                    "stdout_bytes": len(stdout), "stderr_bytes": len(stderr)},
        "independent_saved_arithmetic": {"shape": shape, "voxels": count, "voxel_storage_bytes": count * 4,
                                        "uncompressed_bytes": 352 + count * 4,
                                        "reported_finite_counts_consistent": True,
                                        "reported_min_mean_max_ordered_and_finite": [lo, mean, hi],
                                        "coded_qform_determinant": det, "column_norms_mm": zooms,
                                        "numeric_axis_codes": codes, "full_cell_bounds_mm": bounds,
                                        "second_coded_form_available": False},
        "original_compressed_verification": {"path": payload["path"], "bytes": length,
                                             "sha256": sha256.hexdigest(), "candidate_md5": md5.hexdigest(),
                                             "mode": "0o444", "unchanged_during_read": True},
        "consulted_inputs": consulted,
        "checker_failures": [],
        "scope": "Saved JSON/source/history audit plus one compressed-byte hash pass. Source tar archive only was unpacked; no patient gzip, header or voxel re-decode, no inspector invocation and no network.",
        "limits": ["Intensity summaries and gzip/footer validity are reported by the frozen child; only their saved arithmetic and bindings were independently checked here.",
                   "Affine math and numeric axis codes do not verify anatomy, modality, scanner provenance, registration or clinical usefulness.",
                   "Recorded timings and sampled memory do not reconstruct OS execution or prove a continuous memory maximum or unrecorded attempt count.",
                   "Read-only permissions and stable hashes do not create an OS immutable-file guarantee.",
                   "Publisher checksum algorithm and prior compressed length remain undeclared. Scientific use, case import and training remain unreleased."]}


if __name__ == "__main__":
    report = audit()
    destination = HERE / "independent-result-audit.json"
    need(not destination.exists(), "preserve existing audit; do not overwrite")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": True, "receipt_sha256": sha(destination.read_bytes()),
                      "reviewer_sha256": report["reviewer_source"]["sha256"]}))
