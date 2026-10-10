"""Backend-only admission for a completed generated contact-family pilot.

The publication SHA is deliberately unset until the actual root-owned pilot,
both final checkpoints, and the measurement pass have been independently
reviewed. No renderer argument or environment variable can set it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RELEASE_VERSION = "public-contact-desktop-release-v1"
RELEASE_RELATIVE_PATH = "artifacts/public-contact-learning-v1/desktop-release.json"
PILOT_PREFIX = "outputs/learning/public-contact-v1/attempt-01/"
RELEASE_MANIFEST_SHA256 = "4bc4a93f453fb4064c3291b55c60481de554adfe706ab7db24a3a023f5e20dae"  # Pin the actual reviewed release in a later exact-source change.
MAX_MANIFEST_BYTES = 32 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_FREEZE_BYTES = 128 * 1024
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024
METHODS = ("IL", "RL")
ONLINE_METHODS = frozenset(("IL", "RL", "SEARCH", "STOP"))


class ContactReleaseUnavailable(ValueError):
    """The server has no reviewed complete pilot publication."""


def _sha256(path: Path, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ContactReleaseUnavailable("Published contact file exceeds its fixed bound")
            digest.update(chunk)
    return digest.hexdigest(), size


def _file(root: Path, relative: str, expected_relative: str, max_bytes: int,
          expected_sha256: str, expected_bytes: int) -> Path:
    if relative != expected_relative or type(expected_sha256) is not str or len(expected_sha256) != 64:
        raise ContactReleaseUnavailable("Published contact file path/hash differs from its fixed slot")
    if type(expected_bytes) is not int or not 0 < expected_bytes <= max_bytes:
        raise ContactReleaseUnavailable("Published contact file size differs from its bound")
    path = root
    for part in Path(relative).parts:
        if part in ("", ".", ".."):
            raise ContactReleaseUnavailable("Published contact path is not a fixed relative path")
        path = path / part
        if path.is_symlink():
            raise ContactReleaseUnavailable("Published contact path traverses a symlink")
    if not path.is_file():
        raise ContactReleaseUnavailable("Published contact file is absent")
    observed_hash, observed_bytes = _sha256(path, max_bytes)
    if observed_hash != expected_sha256 or observed_bytes != expected_bytes:
        raise ContactReleaseUnavailable("Published contact file bytes changed")
    return path


def _json_file(path: Path, limit: int, expected_sha256: str) -> dict:
    if path.stat().st_size > limit:
        raise ContactReleaseUnavailable("Published contact JSON exceeds its bound")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContactReleaseUnavailable("Published contact JSON has a duplicate key")
            result[key] = value
        return result
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise ContactReleaseUnavailable("Published contact JSON grew past its bound")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ContactReleaseUnavailable("Published contact JSON changed after validation")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (ValueError, UnicodeError) as error:
        raise ContactReleaseUnavailable("Published contact JSON is invalid") from error
    if type(value) is not dict:
        raise ContactReleaseUnavailable("Published contact JSON root is not an object")
    return value


def _file_record(root: Path, record: dict, name: str, limit: int) -> Path:
    if type(record) is not dict or set(record) != {"relativePath", "sha256", "bytes"}:
        raise ContactReleaseUnavailable("Published contact file record has the wrong schema")
    return _file(root, record["relativePath"], PILOT_PREFIX + name, limit,
                 record["sha256"], record["bytes"])


def _online_rows(result: dict, key: str, count: int, role_by_layout: dict) -> None:
    rows = result.get(key)
    if type(rows) is not list or len(rows) != count:
        raise ContactReleaseUnavailable("Declared contact online pass is missing rows")
    seen = set()
    for row in rows:
        if type(row) is not dict or row.get("role") != key or row.get("status") not in (
                "complete", "limit_or_unresolved"):
            raise ContactReleaseUnavailable("Declared contact online row lacks a terminal outcome")
        layout, goal, method = row.get("layout_id"), row.get("goal_id"), row.get("method")
        if (type(layout) is not str or role_by_layout.get(layout) != key or
                goal not in ("surface", "deep") or method not in ONLINE_METHODS):
            raise ContactReleaseUnavailable("Declared contact online row changed fixed role/goal/method")
        identity = (layout, goal, method)
        if identity in seen:
            raise ContactReleaseUnavailable("Declared contact online row is duplicated")
        seen.add(identity)
    expected = {(layout, goal, method) for layout, role in role_by_layout.items() if role == key
                for goal in ("surface", "deep") for method in ONLINE_METHODS}
    if seen != expected:
        raise ContactReleaseUnavailable("Declared contact online pass omitted a fixed cell")


def read_published_contact_release(root: Path, *, family_manifest: dict,
                                   experiment_hash: str) -> dict:
    """Verify the fixed publication without decoding weights or making a task.

    Only the owned child may subsequently decode BOTH final checkpoints with
    the experiment and kind='final'. A SHA/JSON receipt alone is not training
    authenticity or a license to use a historical checkpoint.
    """
    if RELEASE_MANIFEST_SHA256 is None:
        raise ContactReleaseUnavailable("No reviewed completed contact pilot has been published")
    root = Path(root)
    manifest_path = root
    for part in Path(RELEASE_RELATIVE_PATH).parts:
        manifest_path = manifest_path / part
        if manifest_path.is_symlink():
            raise ContactReleaseUnavailable("Backend-owned contact release traverses a symlink")
    if not manifest_path.is_file():
        raise ContactReleaseUnavailable("Backend-owned contact release is absent")
    manifest_hash, manifest_bytes = _sha256(manifest_path, MAX_MANIFEST_BYTES)
    if manifest_hash != RELEASE_MANIFEST_SHA256 or manifest_bytes == 0:
        raise ContactReleaseUnavailable("Backend-owned contact release bytes differ from review")
    manifest = _json_file(manifest_path, MAX_MANIFEST_BYTES, manifest_hash)
    if (set(manifest) != {"version", "experimentHash", "familyHash", "pilotResult",
                          "finalFreeze", "checkpoints"} or manifest["version"] != RELEASE_VERSION or
            manifest["experimentHash"] != experiment_hash or
            manifest["familyHash"] != family_manifest.get("family_hash") or
            type(manifest["checkpoints"]) is not dict or set(manifest["checkpoints"]) != set(METHODS)):
        raise ContactReleaseUnavailable("Backend-owned contact release has wrong experiment or schema")
    pilot_path = _file_record(root, manifest["pilotResult"], "result.json", MAX_RESULT_BYTES)
    freeze_path = _file_record(root, manifest["finalFreeze"], "final-checkpoint-freeze.json",
                               MAX_FREEZE_BYTES)
    checkpoint_paths = {}
    for method in METHODS:
        record = manifest["checkpoints"][method]
        if type(record) is not dict or set(record) != {"file", "parameterHash"}:
            raise ContactReleaseUnavailable("Final checkpoint record has the wrong schema")
        checkpoint_paths[method] = _file_record(root, record["file"], method + "-final.gmckpt",
                                                MAX_CHECKPOINT_BYTES)
    pilot = _json_file(pilot_path, MAX_RESULT_BYTES, manifest["pilotResult"]["sha256"])
    freeze = _json_file(freeze_path, MAX_FREEZE_BYTES, manifest["finalFreeze"]["sha256"])
    if (pilot.get("status") != "complete_with_all_failures_preserved" or
            pilot.get("measurement_status") != "single_frozen_pass_finished_no_checkpoint_selection" or
            type(pilot.get("patient_reads")) is not int or pilot["patient_reads"] != 0 or
            pilot.get("experiment_hash") != experiment_hash or
            type(pilot.get("training")) is not dict or set(pilot["training"]) != set(METHODS) or
            freeze.get("version") != "public-contact-final-checkpoint-freeze-v1" or
            freeze.get("checkpoint_selection") != "none_fixed_32_update_endpoint" or
            freeze.get("experiment_hash") != experiment_hash or
            type(freeze.get("checkpoints")) is not dict or set(freeze["checkpoints"]) != set(METHODS)):
        raise ContactReleaseUnavailable("Contact pilot/freeze is partial or does not match the experiment")
    role_by_layout = {row["layout_id"]: row["role"] for row in family_manifest["source_bindings"]}
    _online_rows(pilot, "SELECT", 4 * 2 * 4, role_by_layout)
    _online_rows(pilot, "MEASUREMENT_EVAL", 8 * 2 * 4, role_by_layout)
    for method in METHODS:
        record = manifest["checkpoints"][method]
        training = pilot["training"][method]
        frozen = freeze["checkpoints"][method]
        if (type(training) is not dict or type(training.get("checkpoint")) is not dict or
                type(frozen) is not dict):
            raise ContactReleaseUnavailable("Contact training/freeze row has the wrong schema")
        if (training.get("status") != "completed_fixed_endpoint" or
                type(training.get("updates")) is not int or training["updates"] != 32 or
                training["checkpoint"].get("sha256") != record["file"]["sha256"] or
                training["checkpoint"].get("parameter_hash") != record["parameterHash"] or
                frozen.get("file_sha256") != record["file"]["sha256"] or
                frozen.get("parameter_hash") != record["parameterHash"]):
            raise ContactReleaseUnavailable("Contact checkpoint is not the completed frozen endpoint")
    return {"manifestSha256": manifest_hash, "manifest": manifest,
            "pilotResultSha256": manifest["pilotResult"]["sha256"],
            "finalFreezeSha256": manifest["finalFreeze"]["sha256"],
            "finalFreezeRecord": freeze,
            "checkpointPaths": checkpoint_paths}
