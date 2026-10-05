"""Engineering profile contracts; numerical fixtures never train or score a model."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_spatial_policy_scale.py"
spec = importlib.util.spec_from_file_location("spatial_scale_profile", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
VERIFY_SOURCE = profile.verified_source


@pytest.fixture
def source(tmp_path):
    """A geometric file fixture only; no policy inference or learning."""
    path = tmp_path / profile.IMAGE_RELATIVE
    path.parent.mkdir(parents=True)
    image = nib.Nifti1Image(np.arange(64 ** 3, dtype=np.float32).reshape(64, 64, 64), np.diag([1.5, 2., 2.5, 1.]))
    image.header.set_xyzt_units("mm")
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 1)
    nib.save(image, path)
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"candidates": [{"subject": profile.SUBJECT,
        "development_role": "population_training", "role_locked_before_image_access": True}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"subject": profile.SUBJECT, "development_role": "population_training",
        "selection_cohort_sha256": profile.digest(cohort), "files": [{"path": profile.IMAGE_RELATIVE,
        "bytes": path.stat().st_size, "sha256": profile.digest(path)}]}))
    return VERIFY_SOURCE(tmp_path, manifest, cohort)


@pytest.fixture
def declared_source(source, monkeypatch):
    def verify(source_root=None, manifest_path=None, cohort_path=None):
        if source_root is None:
            return source
        return VERIFY_SOURCE(source_root, manifest_path, cohort_path)
    monkeypatch.setattr(profile, "verified_source", verify)
    return source


def test_prepare_snapshots_sources_without_running_a_model(tmp_path, monkeypatch, declared_source):
    monkeypatch.setattr(profile, "profile_cell", lambda *a: pytest.fail("Preparation ran a model"))
    output = tmp_path / "prepared"
    declaration = profile.prepare(output)
    assert declaration["plan"] == [{"side": s, "non_stop_rays": n} for s, n in profile.PLAN]
    assert declaration["optimizer_steps_per_cell"] == declaration["training_forward_calls_per_cell"] == 0
    assert declaration["synthetic_scan"] is False
    assert not (output / "attempt.json").exists()
    for path, expected in declaration["source_sha256"].items():
        assert profile.digest(output / "source_snapshot" / path) == expected
    before = (output / "declaration.json").read_bytes()
    with pytest.raises(FileExistsError):
        profile.prepare(output)
    assert (output / "declaration.json").read_bytes() == before


def test_source_drift_refuses_before_attempt_or_child(tmp_path, monkeypatch, declared_source):
    output = tmp_path / "prepared"
    profile.prepare(output)
    monkeypatch.setattr(profile, "source_hashes", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="Source changed"):
        profile.run_prepared(output)
    assert not (output / "attempt.json").exists()


def test_image_drift_refuses_before_attempt(tmp_path, declared_source):
    output = tmp_path / "prepared"
    profile.prepare(output)
    path = Path(declared_source["image_path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum or byte count"):
        profile.run_prepared(output)
    assert not (output / "attempt.json").exists()


def test_modified_resource_cap_is_not_a_valid_declaration(tmp_path, declared_source):
    output = tmp_path / "prepared"
    profile.prepare(output)
    path = output / "declaration.json"
    declaration = json.loads(path.read_text())
    declaration["rss_limit_bytes"] *= 2
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="bounded plan"):
        profile.run_prepared(output)


def test_nontraining_role_rejected_even_when_manifest_hash_refreshed(source):
    cohort_path, manifest_path = Path(source["cohort_path"]), Path(source["manifest_path"])
    cohort = json.loads(cohort_path.read_text())
    cohort["candidates"][0]["development_role"] = "checkpoint_selection_development"
    cohort_path.write_text(json.dumps(cohort))
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_cohort_sha256"] = profile.digest(cohort_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prospective locked training"):
        VERIFY_SOURCE(source["source_root"], manifest_path, cohort_path)


def test_center_crop_uses_case_statistics_and_preserves_oblique_reflected_coordinates():
    scan = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    affine = np.array([[0., -2., 0., 18.], [-1.5, 0., 0., -7.], [0., 0., 3.5, 44.], [0., 0., 0., 1.]])
    crop, cropped_affine, receipt = profile.centered_case_crop(scan, affine, 4)
    assert receipt["start_voxel"] == [2, 3, 4]
    np.testing.assert_allclose(crop, ((scan[2:6, 3:7, 4:8] - scan.mean(dtype=np.float64)) /
                                    scan.std(dtype=np.float64)).astype(np.float32), atol=1e-7)
    for point in ([0, 0, 0, 1], [3, 3, 3, 1], [1, 2, 3, 1]):
        original = np.asarray(point, dtype=float)
        original[:3] += receipt["start_voxel"]
        np.testing.assert_array_equal(cropped_affine @ point, affine @ original)
    assert receipt["normalization_voxels"] == scan.size and not receipt["annotation_used"]
    assert not receipt["resampled"]


@pytest.mark.parametrize("value", [0., np.nan, np.inf])
def test_invalid_or_constant_scan_rejected(value):
    with pytest.raises(ValueError):
        profile.centered_case_crop(np.full((4, 4, 4), value), np.eye(4), 4)


def test_observation_fixture_preserves_missingness_and_native_translation(source):
    observation, crop = profile.real_observation(4, 2, source)
    assert observation.track == "inference_only"
    assert observation.channel_available.tolist() == [True, False, False, True, False, False]
    assert not observation.image_channels[[1, 2, 4, 5]].any()
    assert not observation.coverage[[1, 2, 4, 5]].any()
    np.testing.assert_array_equal(observation.affine_ras_mm, crop["crop_affine_ras_mm"])
    world = observation.affine_ras_mm @ [1, 1, .5, 1]
    np.testing.assert_array_equal(observation.action_geometry[1, 1:4], world[:3])


def test_deadline_terminates_child_and_retains_log(tmp_path):
    log = tmp_path / "deadline.log"
    result = profile.supervise([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"],
        log, time.monotonic() + .1, rss_reader=lambda pid: 0)
    assert result["refusal"] == "total_deadline_reached" and result["exit_code"] != 0
    assert log.is_file()


@pytest.mark.parametrize("mode", ["memory", "monitor_failure"])
def test_resource_monitor_failure_stops_and_retains_child_attempt(tmp_path, mode):
    def reader(pid):
        if mode == "monitor_failure":
            raise RuntimeError("injected monitor failure")
        return profile.RSS_STOP_BYTES + 1
    result = profile.supervise([sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / f"{mode}.log", time.monotonic() + 2, rss_reader=reader)
    assert result["exit_code"] != 0
    assert ("rss_headroom" if mode == "memory" else "resource_monitor_failed") in result["refusal"]


def test_first_failure_stops_without_retry_and_attempt_cannot_be_reused(tmp_path, monkeypatch, declared_source):
    output = tmp_path / "prepared"
    profile.prepare(output)
    calls = []
    def failed(*args, **kwargs):
        calls.append(args)
        return {"exit_code": 1, "refusal": "injected failure", "sampled_peak_rss_bytes": 0}
    monkeypatch.setattr(profile, "supervise", failed)
    report = profile.run_prepared(output)
    assert report["status"] == "failed_stopped_without_retry"
    assert len(calls) == 1 and report["unexecuted_cells"] == 3 and report["retries"] == 0
    saved = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        profile.run_prepared(output)
    assert (output / "results.json").read_bytes() == saved


def test_import_root_rejects_an_already_loaded_external_project_module(tmp_path, monkeypatch):
    from types import SimpleNamespace
    escaped = tmp_path / "outside.py"
    escaped.write_text("# external fixture\n")
    monkeypatch.setitem(sys.modules, "resectionlab.unexpected_external_module", SimpleNamespace(__file__=str(escaped)))
    with pytest.raises(ValueError, match="escaped the profile source archive"):
        profile.enforce_import_root()


def test_import_root_is_first_and_loaded_module_paths_are_bound():
    evidence = profile.enforce_import_root()
    assert sys.path[0] == str((profile.ROOT / "src").resolve())
    for item in evidence.values():
        assert Path(item["path"]).is_relative_to((profile.ROOT / "src/resectionlab").resolve())
        assert profile.digest(item["path"]) == item["sha256"]
    assert "src/resectionlab/core.py" in profile.source_paths()
    assert set(profile.source_hashes()) == set(profile.source_paths())


def test_declaration_cannot_move_to_another_source_root(tmp_path, declared_source):
    output = tmp_path / "prepared"
    profile.prepare(output)
    path = output / "declaration.json"
    declaration = json.loads(path.read_text())
    declaration["source_root"] = str(tmp_path)
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="declared complete source archive"):
        profile.run_prepared(output)
    assert not (output / "attempt.json").exists()


def test_reprepare_reuses_only_exact_verified_absolute_source_references(tmp_path, declared_source):
    original = tmp_path / "original"
    profile.prepare(original)
    derived = profile.prepare(tmp_path / "derived", original / "declaration.json")
    assert derived["scan_source"] == declared_source
    assert derived["import_root"] == str((profile.ROOT / "src").resolve())
