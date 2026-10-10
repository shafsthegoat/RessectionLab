"""Generated receipt admission controls; never open a real checkpoint."""
from __future__ import annotations

import hashlib
import json

import pytest


from resectionlab import contact_family_desktop_release as release


def _save(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
               if isinstance(content, dict) else content)
    path.write_bytes(payload)
    return {"relativePath": relative, "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}


def _fixture(root):
    roles = {**{f"train-{n}": "TRAIN" for n in range(12)},
             **{f"select-{n}": "SELECT" for n in range(4)},
             **{f"eval-{n}": "MEASUREMENT_EVAL" for n in range(8)}}
    family = {"family_hash": "sha256:" + "a" * 64,
              "source_bindings": [{"layout_id": layout, "role": role}
                                  for layout, role in roles.items()]}
    experiment_hash = "sha256:" + "b" * 64
    rows = {role: [{"role": role, "status": "complete", "layout_id": layout,
                    "goal_id": goal, "method": method}
                   for layout, actual_role in roles.items() if actual_role == role
                   for goal in ("surface", "deep") for method in ("STOP", "SEARCH", "IL", "RL")]
            for role in ("SELECT", "MEASUREMENT_EVAL")}
    checkpoints = {}
    for method in ("IL", "RL"):
        file = _save(root, release.PILOT_PREFIX + method + "-final.gmckpt",
                     ("generated-" + method).encode())
        checkpoints[method] = {"file": file, "parameterHash": "sha256:" + method.lower() * 32}
    pilot = {"status": "complete_with_all_failures_preserved",
             "measurement_status": "single_frozen_pass_finished_no_checkpoint_selection",
             "patient_reads": 0,
             "experiment_hash": experiment_hash,
             "training": {method: {"status": "completed_fixed_endpoint", "updates": 32,
                         "checkpoint": {"sha256": checkpoints[method]["file"]["sha256"],
                                        "parameter_hash": checkpoints[method]["parameterHash"]}}
                          for method in ("IL", "RL")}, **rows}
    frozen = {"version": "public-contact-final-checkpoint-freeze-v1",
              "checkpoint_selection": "none_fixed_32_update_endpoint",
              "experiment_hash": experiment_hash,
              "checkpoints": {method: {"file_sha256": checkpoints[method]["file"]["sha256"],
                                          "parameter_hash": checkpoints[method]["parameterHash"]}
                              for method in ("IL", "RL")}}
    manifest = {"version": release.RELEASE_VERSION, "experimentHash": experiment_hash,
                "familyHash": family["family_hash"],
                "pilotResult": _save(root, release.PILOT_PREFIX + "result.json", pilot),
                "finalFreeze": _save(root, release.PILOT_PREFIX + "final-checkpoint-freeze.json", frozen),
                "checkpoints": checkpoints}
    manifest_record = _save(root, release.RELEASE_RELATIVE_PATH, manifest)
    return family, experiment_hash, manifest, pilot, manifest_record


def test_absent_publication_refuses_before_any_artifact_read(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "RELEASE_MANIFEST_SHA256", None)
    with pytest.raises(release.ContactReleaseUnavailable, match="No reviewed completed"):
        release.read_published_contact_release(tmp_path, family_manifest={}, experiment_hash="sha256:none")


def test_exact_complete_generated_publication_and_partial_refusal(tmp_path, monkeypatch):
    family, experiment, manifest, pilot, manifest_record = _fixture(tmp_path)
    monkeypatch.setattr(release, "RELEASE_MANIFEST_SHA256", manifest_record["sha256"])
    checked = release.read_published_contact_release(tmp_path,
        family_manifest=family, experiment_hash=experiment)
    assert checked["manifestSha256"] == manifest_record["sha256"]
    assert set(checked["checkpointPaths"]) == {"IL", "RL"}
    pilot["training"]["IL"]["updates"] = 31
    manifest["pilotResult"] = _save(tmp_path, release.PILOT_PREFIX + "result.json", pilot)
    changed_manifest = _save(tmp_path, release.RELEASE_RELATIVE_PATH, manifest)
    monkeypatch.setattr(release, "RELEASE_MANIFEST_SHA256", changed_manifest["sha256"])
    with pytest.raises(release.ContactReleaseUnavailable, match="completed frozen endpoint"):
        release.read_published_contact_release(tmp_path,
            family_manifest=family, experiment_hash=experiment)


def test_tampered_file_and_symlink_refuse(tmp_path, monkeypatch):
    family, experiment, manifest, _, manifest_record = _fixture(tmp_path)
    monkeypatch.setattr(release, "RELEASE_MANIFEST_SHA256", manifest_record["sha256"])
    result = tmp_path / manifest["pilotResult"]["relativePath"]
    result.write_bytes(result.read_bytes() + b"x")
    with pytest.raises(release.ContactReleaseUnavailable, match="bytes changed"):
        release.read_published_contact_release(tmp_path,
            family_manifest=family, experiment_hash=experiment)
    result.unlink()
    result.symlink_to(tmp_path / release.PILOT_PREFIX / "RL-final.gmckpt")
    with pytest.raises(release.ContactReleaseUnavailable, match="symlink"):
        release.read_published_contact_release(tmp_path,
            family_manifest=family, experiment_hash=experiment)


def test_missing_measurement_row_cannot_publish_learned_method(tmp_path, monkeypatch):
    family, experiment, manifest, pilot, _ = _fixture(tmp_path)
    pilot["MEASUREMENT_EVAL"].pop()
    manifest["pilotResult"] = _save(tmp_path, release.PILOT_PREFIX + "result.json", pilot)
    manifest_record = _save(tmp_path, release.RELEASE_RELATIVE_PATH, manifest)
    monkeypatch.setattr(release, "RELEASE_MANIFEST_SHA256", manifest_record["sha256"])
    with pytest.raises(release.ContactReleaseUnavailable, match="missing rows"):
        release.read_published_contact_release(tmp_path,
            family_manifest=family, experiment_hash=experiment)


def test_complete_negative_learned_outcomes_remain_reportable(tmp_path, monkeypatch):
    family, experiment, manifest, pilot, _ = _fixture(tmp_path)
    for role in ("SELECT", "MEASUREMENT_EVAL"):
        for row in pilot[role]:
            if row["method"] in ("IL", "RL"):
                row["metrics"] = {"goal_retained": True,
                                  "goal_contacted_and_retained": False,
                                  "total_reward": 0.0}
    manifest["pilotResult"] = _save(tmp_path, release.PILOT_PREFIX + "result.json", pilot)
    manifest_record = _save(tmp_path, release.RELEASE_RELATIVE_PATH, manifest)
    monkeypatch.setattr(release, "RELEASE_MANIFEST_SHA256", manifest_record["sha256"])
    admitted = release.read_published_contact_release(tmp_path,
        family_manifest=family, experiment_hash=experiment)
    assert admitted["pilotResultSha256"] == manifest["pilotResult"]["sha256"]
