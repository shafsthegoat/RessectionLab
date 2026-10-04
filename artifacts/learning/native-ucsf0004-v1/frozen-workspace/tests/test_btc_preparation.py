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
