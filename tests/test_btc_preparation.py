"""Creator-source case preparation preserves coordinates, data, and missingness."""
from datetime import datetime, timezone
from hashlib import md5, sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from resectionlab.imaging import load_case, read_case_artifacts


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_btc_case.py"
SPEC = importlib.util.spec_from_file_location("prepare_btc_case", SCRIPT)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


def image(path, values, affine=None):
    affine = np.eye(4) if affine is None else affine
    volume = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
    volume.header.set_xyzt_units("mm")
    volume.set_qform(affine, 1)
    volume.set_sform(affine, 1)
    nib.save(volume, path)


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "sources"
    shape = (24, 25, 26)
    for name in preparation.EXPECTED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json"):
            path.write_text("{}")
        elif name in {"README", "CHANGES"}:
            path.write_text("Synthetic source fixture. Not a real participant.")
    image(root / preparation.T1, np.indices(shape)[0] + 1)
    fractional = np.zeros(shape)
    fractional[2:4, 3:5, 4:6] = 0.8
    fractional[4:6, 3:5, 4:6] = 0.3
    flip = np.eye(4)
    flip[0, 0], flip[0, 3] = -1, shape[0] - 1
    image(root / preparation.ANNOTATION, fractional, flip)
    vectors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
    vectors[1:] /= np.linalg.norm(vectors[1:], axis=1)[:, None]
    for base in (preparation.AP, preparation.PA):
        image(root / (base + ".nii.gz"), np.ones((4, 5, 6, 7)))
        np.savetxt(root / (base + ".bval"), [[0] + [1000] * 6])
        np.savetxt(root / (base + ".bvec"), vectors.T)
    (root / "participants.tsv").write_text(
        "participant_id\t age\ttumor type & grade\ttumor size (cub cm)\ttumor location\thandedness\n"
        "sub-PAT28\t44\tOligodendroglioma II\t11.49\tFrontal\t1\n"
    )
    manifest = {"accession": preparation.ACCESSION, "release": preparation.RELEASE,
                "git_commit": preparation.REVISION, "subject": preparation.SUBJECT,
                "dataset": "BTC_preop", "license": "CC0", "files": []}
    for name in sorted(preparation.EXPECTED_FILES):
        payload = (root / name).read_bytes()
        manifest["files"].append({"path": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest(),
                                  "expected_md5": md5(payload, usedforsecurity=False).hexdigest(),
                                  "expected_bytes": len(payload), "source_url": "synthetic://source/" + name})
    manifest_path = tmp_path / "acquisition.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path, manifest


def test_preparation_roundtrip_gates_full_head_and_preserves_fractional_source(source_tree, tmp_path):
    root, manifest, _ = source_tree
    source_bytes = (root / preparation.ANNOTATION).read_bytes()
    output = tmp_path / "prepared.rslcase"
    report = preparation.prepare(manifest, root, output)
    case = load_case(output)
    assert report["source_hashes_checked"] == 15
    assert report["reopened_identical"] and report["source_files_unchanged"]
    assert case.brain_mask is None
    assert case.metadata["structural_coverage"] == "full_head"
    assert case.metadata["allow_nonzero_mri_access_support"] is False
    assert report["automatic_cortical_access_status"] == "blocked_without_reviewed_cerebral_mask"
    assert (root / preparation.ANNOTATION).read_bytes() == source_bytes
    mask = case.compartments["fractional_source_target_threshold_scenario"]
    assert mask.sum() == 8 and mask[20, 3, 4]
    assert not mask[2, 3, 4]
    assert report["annotation_derivation"]["threshold_evidence"] == "declared_research_assumption"
    assert report["threshold_sensitivity"][0]["source_voxel_count"] == 16
    assert case.context is None
    assert report["planning_cutoff_status"] == "historical_preoperative_cutoff_unknown"
    assert all(item["value"] is None for item in report["withheld_context"].values())
    assert not report["diffusion_input_audit"]["usable_for_reconstruction"]
    assert "PREPROCESSING_UNREVIEWED" in report["diffusion_input_audit"]["issues"]
    assert read_case_artifacts(output)["plans"] == []


def test_replay_cutoff_never_releases_unknown_timed_pathology(source_tree, tmp_path):
    root, manifest, _ = source_tree
    output = tmp_path / "with_context.rslcase"
    cutoff = datetime(2026, 10, 4, tzinfo=timezone.utc)
    report = preparation.prepare(manifest, root, output, planning_as_of=cutoff)
    case = load_case(output)
    assert case.context.planning_as_of == cutoff
    assert case.context.planner_values() == {}
    assert case.context.planning_view()["source_diagnosis"]["value"] is None
    assert case.context.planning_view()["source_diagnosis"]["exclusion_reason"] == "availability_time_unknown"
    assert report["planning_cutoff_status"] == "declared_research_replay_cutoff"


@pytest.mark.parametrize("kind", ["wrong_release", "duplicate", "missing", "sha", "md5", "path"])
def test_invalid_source_set_cannot_prepare(source_tree, tmp_path, kind):
    root, manifest_path, manifest = source_tree
    if kind == "wrong_release":
        manifest["release"] = "6.0.0"
    elif kind == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif kind == "missing":
        manifest["files"].pop()
    elif kind == "sha":
        manifest["files"][0]["sha256"] = "0" * 64
    elif kind == "md5":
        manifest["files"][0]["expected_md5"] = "0" * 32
    else:
        # Exact relative source names can still be symlinks outside the root.
        name = manifest["files"][0]["path"]
        external = tmp_path / "outside"
        external.write_bytes((root / name).read_bytes())
        (root / name).unlink()
        (root / name).symlink_to(external)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        preparation.prepare(manifest_path, root, tmp_path / "should_not_exist.rslcase")
    assert not (tmp_path / "should_not_exist.rslcase").exists()


def test_output_cannot_overwrite_source_or_manifest(source_tree):
    root, manifest, _ = source_tree
    with pytest.raises(ValueError, match="outside the immutable source"):
        preparation.prepare(manifest, root, root / preparation.T1)
    with pytest.raises(ValueError, match="outside the immutable source"):
        preparation.prepare(manifest, root, manifest)


def test_cli_produces_reopenable_case_and_json_report(source_tree, tmp_path):
    root, manifest, _ = source_tree
    output, report = tmp_path / "cli.rslcase", tmp_path / "qc.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest), "--data-root", str(root),
                             "--output", str(output), "--report", str(report), "--annotation-threshold", "0.25"],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["reopened_identical"]
    assert load_case(output).compartments["fractional_source_target_threshold_scenario"].sum() == 16
    assert json.loads(report.read_text())["source_hashes_checked"] == 15


def test_cli_rejects_naive_cutoff(source_tree, tmp_path):
    root, manifest, _ = source_tree
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest), "--data-root", str(root),
                             "--output", str(tmp_path / "bad.rslcase"), "--report", str(tmp_path / "bad.json"),
                             "--planning-as-of", "2026-10-04"], text=True, capture_output=True)
    assert result.returncode == 2
    assert "timezone offset" in result.stderr


@pytest.mark.parametrize("alias", ["manifest", "bundle"])
def test_cli_report_cannot_overwrite_manifest_or_bundle(source_tree, tmp_path, alias):
    root, manifest, _ = source_tree
    output = tmp_path / "case.rslcase"
    output.write_bytes(b"Existing bundle must survive argument validation")
    report = manifest if alias == "manifest" else output
    before_manifest = manifest.read_bytes()
    before_output = output.read_bytes()
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest), "--data-root", str(root),
                             "--output", str(output), "--report", str(report)],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert manifest.read_bytes() == before_manifest
    assert output.read_bytes() == before_output


def test_pat05_structural_preparation_preserves_source_and_missingness(source_tree, tmp_path, monkeypatch):
    root, manifest_path, manifest = source_tree
    t1, annotation, _, _, names = preparation.source_layout("sub-PAT05")
    for name in names:
        if "sub-PAT05" in name:
            source = root / name.replace("sub-PAT05", "sub-PAT28")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    participants = root / "participants.tsv"
    participants.write_text(participants.read_text() + "sub-PAT05\t50\tOligo-astrocytoma II\t8\tFrontal\t1\n")
    manifest["subject"] = "sub-PAT05"
    manifest["files"] = []
    for name in sorted(names):
        payload = (root / name).read_bytes()
        manifest["files"].append({"path": name, "bytes": len(payload), "sha256": sha256(payload).hexdigest(),
                                  "expected_md5": md5(payload, usedforsecurity=False).hexdigest(),
                                  "expected_bytes": len(payload), "source_url": "synthetic://" + name})
    manifest_path.write_text(json.dumps(manifest))
    original_annotation = (root / annotation).read_bytes()
    monkeypatch.setattr(preparation, "audit_diffusion", lambda *a, **k: pytest.fail("Structural mode must not invent DWI"))
    output = tmp_path / "pat05.rslcase"
    report = preparation.prepare(manifest_path, root, output, planning_as_of=datetime(2026, 10, 4, tzinfo=timezone.utc))
    case = load_case(output)
    assert report["source_hashes_checked"] == 7
    assert report["case_id"] == "BTC-ds001226-sub-PAT05-preop"
    assert report["reopened_identical"]
    assert report["diffusion_input_audit"]["issues"] == ["MISSING_DIRECTIONAL_DIFFUSION"]
    assert case.brain_mask is None and case.metadata["allow_nonzero_mri_access_support"] is False
    assert case.metadata["split"]["external_holdout_eligible"] is False
    assert case.context.planner_values() == {}
    assert (root / annotation).read_bytes() == original_annotation
    assert read_case_artifacts(output)["plans"] == []


@pytest.mark.parametrize("subject", ["sub-PAT16", "sub-PAT20"])
def test_queued_preparation_rejects_unacquired_images_before_opening(subject, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import acquire_btc_case as acquisition

    manifest = tmp_path / "pending.json"
    manifest.write_text(json.dumps(acquisition.queued_manifest(subject)))
    monkeypatch.setattr(preparation, "load_fractional_annotation_case",
                        lambda *a, **k: pytest.fail("Pending sources must not be opened"))
    with pytest.raises(ValueError, match="pending annex"):
        preparation.prepare(manifest, tmp_path / "absent", tmp_path / "case.rslcase")


@pytest.mark.parametrize("subject", ["sub-PAT16", "sub-PAT20"])
def test_queued_preparation_rejects_role_or_source_relabeling(subject, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import acquire_btc_case as acquisition

    manifest = acquisition.queued_manifest(subject)
    manifest["role"] = "final_evaluation"
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen development selection"):
        preparation.prepare(path, tmp_path / "absent", tmp_path / "case.rslcase")


@pytest.mark.parametrize("subject", ["sub-PAT16", "sub-PAT20"])
def test_queued_structural_preparation_retains_all_gates(source_tree, tmp_path, monkeypatch, subject):
    """Tiny explicit synthetic queue fixture; no real candidate image is read."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import acquire_btc_case as acquisition

    root, manifest_path, _ = source_tree
    t1, annotation, _, _, names = preparation.source_layout(subject)
    for name in names:
        if subject in name:
            original = root / name.replace(subject, "sub-PAT28")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
    participants = root / "participants.tsv"
    participants.write_text(participants.read_text() +
                           "sub-PAT05\t40\tOligo-astrocytoma II\t8\tFrontal\t1\n"
                           "sub-PAT16\t39\tAnaplastic astrocytoma II-III\t8\tFrontal\t1\n"
                           "sub-PAT20\t70\tAnaplastic astrocytoma III\t8\tParietal\t1\n")
    queue = json.loads(acquisition.QUEUE_MANIFEST.read_text())
    for entry in queue["shared_release_files"]:
        payload = (root / entry["path"]).read_bytes()
        entry.update(bytes=len(payload), sha256=sha256(payload).hexdigest())
    candidate = next(item for item in queue["candidates"] if item["subject"] == subject)
    for entry in candidate["files"]:
        if entry["scope"] != "structural_minimum":
            continue
        payload = (root / entry["path"]).read_bytes()
        entry["expected_bytes"] = len(payload)
        if "expected_annex_md5" in entry:
            entry["expected_annex_md5"] = md5(payload, usedforsecurity=False).hexdigest()
        else:
            entry["sha256"] = sha256(payload).hexdigest()
    synthetic_queue = tmp_path / "synthetic-only-queue.json"
    synthetic_queue.write_text(json.dumps(queue))
    monkeypatch.setattr(acquisition, "QUEUE_MANIFEST", synthetic_queue)
    monkeypatch.setattr(acquisition, "QUEUE_SHA256", sha256(synthetic_queue.read_bytes()).hexdigest())
    manifest = acquisition.queued_manifest(subject)
    for entry in manifest["files"]:
        entry["sha256"] = sha256((root / entry["path"]).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    original_annotation = (root / annotation).read_bytes()
    monkeypatch.setattr(preparation, "audit_diffusion", lambda *a, **k: pytest.fail("Unrequested DWI"))
    output = tmp_path / "queued.rslcase"
    report = preparation.prepare(manifest_path, root, output, planning_as_of=datetime(2026, 10, 4, tzinfo=timezone.utc))
    case = load_case(output)
    assert report["source_hashes_checked"] == 7 and report["reopened_identical"]
    assert case.brain_mask is None and case.metadata["allow_nonzero_mri_access_support"] is False
    assert case.metadata["split"]["patient_group"] == f"BTC:{subject}"
    assert case.metadata["split"]["external_holdout_eligible"] is False
    assert case.metadata["selection_queue_sha256"] == acquisition.QUEUE_SHA256
    assert case.context.planner_values() == {}
    assert report["diffusion_input_audit"]["issues"] == ["MISSING_DIRECTIONAL_DIFFUSION"]
    assert (root / annotation).read_bytes() == original_annotation
    assert read_case_artifacts(output)["plans"] == []
