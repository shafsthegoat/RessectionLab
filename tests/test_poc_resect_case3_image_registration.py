"""Generated controls only: no acquired images, tags, or evaluation points."""
import copy
import gzip
import importlib.util
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("case3_image_runner", Path(__file__).resolve().parents[1] /
                                             "scripts/poc_resect_case3_image_registration.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def header(shape=(31, 32, 33), affine=None):
    value = nib.Nifti1Header()
    value.set_data_shape(shape)
    value.set_data_dtype(np.float32)
    value.set_xyzt_units("mm")
    value.set_sform(np.eye(4) if affine is None else affine, code=1)
    value["qform_code"] = 0
    value["pixdim"][0] = 0  # Inactive source qform must not be reconstructed.
    value["vox_offset"] = 352
    value["scl_slope"], value["scl_inter"] = .9, 0.
    return value.binaryblock


def cohort():
    return {"members": [{"patient_group": runner.CASE, "case_id": "Case3", "role": "TRAIN",
                         "all_timepoints_and_derivatives_grouped": True}]}


def candidate():
    return {"schema": runner.SCHEMA + "-candidate", "case_id": runner.CASE,
            "preflight_sha256": "p" * 64, "fit_release_sha256": "r" * 64,
            "sources_rechecked_after_each_decode": True,
            "source_fixity_after_fit": {name: entry["sha256"] for name, entry in runner.INPUTS.items()},
            "runtime": runner.runtime_record(),
            "runner_sha256": runner.digest(runner.__file__), "report": {
                "before_RAS_mm_to_during_RAS_mm": np.eye(4).tolist(),
                "during_RAS_mm_to_before_RAS_mm": np.eye(4).tolist(),
                "fixed": "before_resection_US", "moving": "during_resection_US",
                "sampling_direction": "before-RAS output physical point -> during-RAS input physical point",
                "identity_baseline_before_RAS_to_during_RAS": np.eye(4).tolist(),
                "landmarks_used": False, "configuration": runner.CONFIG}}


def test_active_sform_inactive_invalid_qfac_header_without_array_proxy(monkeypatch):
    monkeypatch.setattr(nib, "load", lambda *_: pytest.fail("header preflight constructed an image proxy"))
    result = runner.inspect_header(header())
    assert result["qform_code"] == 0 and result["sform_code"] == 1
    assert result["qform_qfac_raw"] == 0 and result["inactive_qform_reconstructed"] is False
    assert result["float32_array_bytes"] == 31 * 32 * 33 * 4
    assert result["image_array_accessed"] is False
    assert result["full_payload_length_and_finiteness_verified"] is False


def test_header_rejects_shear_and_oversized_shape():
    affine = np.eye(4)
    affine[0, 1] = .1
    with pytest.raises(ValueError, match="NONORTHOGONAL"):
        runner.inspect_header(header(affine=affine))
    with pytest.raises(ValueError, match="RESOURCE_LIMIT"):
        runner.inspect_header(header(shape=(400, 400, 400)))


@pytest.mark.parametrize("change", ["id", "source_hash", "role", "cohort_hash", "duplicate"])
def test_wrong_identity_hash_or_role_refused(change):
    identity, inputs, roles, digest = runner.CASE, copy.deepcopy(runner.INPUTS), cohort(), runner.COHORT_SHA
    if change == "id": identity = "RESECT:Case4"
    if change == "source_hash": inputs["before"]["sha256"] = "0" * 64
    if change == "role": roles["members"][0]["role"] = "MEASUREMENT_EVAL"
    if change == "cohort_hash": digest = "0" * 64
    if change == "duplicate": roles["members"] *= 2
    with pytest.raises(ValueError):
        runner.validate_identity(identity, inputs, roles, digest)


def test_generated_source_mismatch_rejected_before_decode(tmp_path, monkeypatch):
    path = tmp_path / "generated.nii.gz"
    path.write_bytes(b"generated corrupt source")
    monkeypatch.setattr(nib, "load", lambda *_: pytest.fail("decoded hash-mismatched source"))
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.rigid.checked_source(path, "0" * 64)


def test_signal_support_is_positive_signal_and_native_coverage_only():
    array = np.ones((7, 8, 9), np.float32)
    array[3, 3, 3], array[4, 4, 4] = 0, -2
    support = runner.signal_support(array)
    assert support.dtype == bool
    assert not support[0].any() and not support[-1].any()
    assert support[2, 2, 2] and not support[3, 3, 3] and not support[4, 4, 4]
    assert "never a brain" in runner.SUPPORT
    with pytest.raises(ValueError, match="NO_POSITIVE"):
        runner.signal_support(np.zeros((7, 8, 9)))
    array[2, 2, 2] = np.nan
    with pytest.raises(ValueError, match="INVALID_SIGNAL"):
        runner.signal_support(array)


def test_partial_moving_coverage_is_geometric_not_known_tissue():
    sitk = pytest.importorskip("SimpleITK")
    slab = runner.rigid.interior_coverage((10, 12, 6))
    affine = np.eye(4)
    affine[2, 3] = 7
    result = runner.rigid.resample(slab, affine, np.zeros((10, 12, 24)), np.eye(4),
                                   sitk.Transform(3, sitk.sitkIdentity), label=True)
    assert not result[:, :, :8].any() and not result[:, :, 12:].any()
    assert result[4, 5, 8:12].tolist() == [1, 1, 1, 1]


def test_oblique_ras_lps_sampling_direction_and_generated_points():
    sitk = pytest.importorskip("SimpleITK")
    from resectionlab.preprocessing import RAS_TO_LPS, sitk_transform_to_ras
    during = np.zeros((12, 13, 14), np.float32)
    during[7, 6, 8] = 1
    affine = np.array([[0., -2., 0., 20.], [1.5, 0., 0., -10.], [0., 0., 3., 7.], [0., 0., 0., 1.]])
    translation_ras = affine[:3, 0] * 2
    transform = sitk.TranslationTransform(3, tuple(RAS_TO_LPS[:3, :3] @ translation_ras))
    on_before = runner.rigid.resample(during, affine, during, affine, transform, label=True)
    assert np.argwhere(on_before == 1).tolist() == [[5, 6, 8]]
    matrix = sitk_transform_to_ras(transform)
    before_point = nib.affines.apply_affine(affine, [5, 6, 8])
    during_point = nib.affines.apply_affine(affine, [7, 6, 8])
    np.testing.assert_allclose(nib.affines.apply_affine(matrix, before_point), during_point)
    np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(matrix), during_point), before_point)


def test_fixed_algorithm_generated_translation_without_landmark_access(monkeypatch):
    sitk = pytest.importorskip("SimpleITK")
    from scipy.ndimage import map_coordinates
    observed_initial = []
    original_initial = sitk.ImageRegistrationMethod.SetInitialTransform
    def observe_initial(method, transform, *args, **kwargs):
        for point in [(0., 0., 0.), (11., -9., 3.)]:
            np.testing.assert_allclose(transform.TransformPoint(point), point, atol=1e-12)
        observed_initial.append(True)
        return original_initial(method, transform, *args, **kwargs)
    monkeypatch.setattr(sitk.ImageRegistrationMethod, "SetInitialTransform", observe_initial)
    # All values generated in memory; any attempted payload open is a failure.
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".tag" or "data/anatomy" in str(path) or "data/annotations" in str(path):
            pytest.fail("array-only fit attempted patient or landmark access")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    grid = np.indices((42, 42, 42), dtype=float)
    during = np.zeros((42, 42, 42), np.float32)
    for center, sigma, strength in [((20, 21, 22), (8, 9, 10), .4), ((14, 16, 25), (3, 4, 5), .9),
                                    ((29, 25, 14), (4, 3, 5), .7), ((24, 12, 17), (3, 4, 3), .8)]:
        during += strength * np.exp(-sum(((grid[d] - center[d]) / sigma[d])**2 for d in range(3)) / 2)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = -42
    translation = np.array([3., -2., 1.])
    before = map_coordinates(during, grid + translation[:, None, None, None] / 2, order=1)
    _, report = runner.fit_arrays(before, affine, during, affine)
    matrix = np.asarray(report["before_RAS_mm_to_during_RAS_mm"])
    np.testing.assert_allclose(matrix[:3, 3], translation, atol=.65)
    np.testing.assert_allclose(matrix @ report["during_RAS_mm_to_before_RAS_mm"], np.eye(4), atol=1e-10)
    assert report["configuration"] == runner.CONFIG
    assert observed_initial == [True]
    assert report["landmarks_used"] is False and report["anatomical_registration_accepted"] is False
    assert "T1" not in str(report) and "MRA" not in str(report)


def test_insufficient_coverage_is_a_fixed_failure_without_fallback():
    pytest.importorskip("SimpleITK")
    values = np.arange(40**3, dtype=np.float32).reshape((40, 40, 40))
    outside = np.eye(4)
    outside[0, 3] = 1000
    with pytest.raises(ValueError, match="Insufficient partial-field overlap"):
        runner.fit_arrays(values, np.eye(4), values, outside)


def test_worker_audit_guard_refuses_tags_labels_other_cases_and_source_writes(tmp_path):
    data = tmp_path / "data"
    allowed = data / "Case3-before.nii.gz"
    guard = runner.patient_read_guard(data, [allowed])
    guard("open", (str(allowed), "r", 0))
    for path, mode in [(data / "Case3-full.tag", "r"), (data / "mask.nii.gz", "r"),
                       (data / "Case4.nii.gz", "r"), (tmp_path / "outside.tag", "r"), (allowed, "w")]:
        with pytest.raises(PermissionError):
            guard("open", (str(path), mode, 0))


@pytest.mark.parametrize("status,code", [("timeout", None), ("worker_failed", 1), ("memory_stop", -9),
                                         ("supervisor_or_seal_failed", None)])
def test_failed_or_timed_out_candidate_never_freezes_or_enables_evaluation(tmp_path, status, code):
    runner.publish(tmp_path / "candidate.json", candidate())
    runner.seal_attempt(tmp_path, status, code, "p" * 64, "r" * 64)
    assert not (tmp_path / "frozen-transform.json").exists()
    with pytest.raises(ValueError, match="SUPERVISED_FREEZE"):
        runner.evaluation_gate(tmp_path, {})


def test_success_freeze_is_exclusive_and_needs_separate_bound_evaluation_release(tmp_path):
    runner.publish(tmp_path / "candidate.json", candidate())
    runner.publish(tmp_path / "supervision.json", {"status": "completed", "exit_code": 0, "worker_reaped": True,
        "samples": 2, "sampled_family_rss_stop_bytes": runner.LIMITS["sampled_family_rss_stop_bytes"],
        "peak_family_rss_bytes_sampled": 1024})
    runner.seal_attempt(tmp_path, "completed", 0, "p" * 64, "r" * 64)
    with pytest.raises(FileExistsError):
        runner.publish(tmp_path / "frozen-transform.json", {"overwrite": True})
    with pytest.raises(ValueError, match="EVALUATION_RELEASE"):
        runner.evaluation_gate(tmp_path, {})
    release = {"schema": runner.SCHEMA + "-evaluation-release", "case_id": runner.CASE,
               "decision": "release_separate_evaluation", "fit_result_sha256": runner.digest(tmp_path / "result.json"),
               "frozen_transform_sha256": runner.digest(tmp_path / "frozen-transform.json"),
               "full_tag_role": "evaluation_only_all_points", "refitting_or_parameter_selection_permitted": False}
    assert runner.evaluation_gate(tmp_path, release)["anatomical_alignment_accepted"] is False
    release["full_tag_role"] = "fit"
    with pytest.raises(ValueError):
        runner.evaluation_gate(tmp_path, release)


def test_real_supervisor_timeout_kills_generated_worker(tmp_path):
    result = runner.supervise_sampled([sys.executable, "-c", "import time; time.sleep(2)"],
                                     tmp_path / "log", deadline=time.monotonic() + .15, on_start=lambda _: None)
    assert result["status"] == "timeout" and result["worker_reaped"]
    assert result["minimum_available_memory_bytes_sampled"] > 0


def test_sampled_memory_stop_cleans_worker_and_generated_descendant(tmp_path):
    import psutil
    child_path = tmp_path / "child.pid"
    script = ("import subprocess,sys,time; from pathlib import Path; "
              "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)']); "
              "Path(sys.argv[1]).write_text(str(child.pid)); "
              "data=bytearray(32*1024**2); time.sleep(3)")
    # Small generated-only stop threshold; production bound remains fixed at 4 GiB.
    result = runner.supervise_sampled([sys.executable, "-c", script, str(child_path)], tmp_path / "log",
        deadline=time.monotonic() + 3, on_start=lambda _: None, rss_stop=24 * 1024**2)
    assert result["status"] == "memory_stop" and result["worker_reaped"]
    assert result["process_group_cleanup_attempted"]
    assert result["peak_family_rss_bytes_sampled"] > 24 * 1024**2
    assert result["memory_control"] == "sampled_stop_enforcement" and result["os_hard_memory_limit"] is False
    child_pid = int(child_path.read_text())
    try:
        child = psutil.Process(child_pid)
        for _ in range(20):
            if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(.01)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_memory_observer_failure_cleans_up_and_cannot_look_successful(tmp_path):
    def fail_sample(*_):
        raise RuntimeError("generated monitor failure")
    result = runner.supervise_sampled([sys.executable, "-c", "import time; time.sleep(2)"],
        tmp_path / "log", deadline=time.monotonic() + 2, on_start=lambda _: None, observer=fail_sample)
    assert result["status"] == "supervisor_failed" and result["worker_reaped"]
    assert result["process_group_cleanup_attempted"]


def test_memory_stop_receipt_cannot_be_promoted_to_successful_freeze(tmp_path):
    runner.publish(tmp_path / "candidate.json", candidate())
    runner.publish(tmp_path / "supervision.json", {"status": "memory_stop", "exit_code": 0, "worker_reaped": True})
    with pytest.raises(ValueError, match="MEMORY_SUPERVISION"):
        runner.seal_attempt(tmp_path, "completed", 0, "p" * 64, "r" * 64)
    assert not (tmp_path / "frozen-transform.json").exists()


@pytest.mark.parametrize("change", ["inverse", "direction", "identity_baseline"])
def test_bad_transform_semantics_cannot_freeze(tmp_path, change):
    value = candidate()
    if change == "inverse": value["report"]["during_RAS_mm_to_before_RAS_mm"][0][3] = 2.
    if change == "direction": value["report"]["sampling_direction"] = "during to before"
    if change == "identity_baseline": value["report"]["identity_baseline_before_RAS_to_during_RAS"][0][3] = 2.
    runner.publish(tmp_path / "candidate.json", value)
    runner.publish(tmp_path / "supervision.json", {"status": "completed", "exit_code": 0, "worker_reaped": True,
        "samples": 2, "sampled_family_rss_stop_bytes": runner.LIMITS["sampled_family_rss_stop_bytes"],
        "peak_family_rss_bytes_sampled": 1024})
    with pytest.raises(ValueError):
        runner.seal_attempt(tmp_path, "completed", 0, "p" * 64, "r" * 64)
    assert not (tmp_path / "frozen-transform.json").exists()


def test_group_permission_error_still_reaps_worker_and_records_failure(tmp_path, monkeypatch):
    def denied(*_):
        raise PermissionError("generated group cleanup denial")
    monkeypatch.setattr(runner.os, "killpg", denied)
    result = runner.supervise_sampled([sys.executable, "-c", "import time; time.sleep(2)"], tmp_path / "log",
        deadline=time.monotonic() + .15, on_start=lambda _: None)
    assert result["status"] == "cleanup_failed" and result["worker_reaped"]
    assert "PermissionError" in result["cleanup_error"]


@pytest.mark.parametrize("mutation_phase", ["after_decode", "after_fit"])
def test_source_mutation_during_generated_worker_refuses_candidate(tmp_path, monkeypatch, mutation_phase):
    inputs, headers = {}, {}
    for name in ("before", "during"):
        path = tmp_path / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6), np.float32), np.eye(4)), path)
        inputs[name] = {"path": path.name, "sha256": runner.digest(path), "bytes": path.stat().st_size}
        headers[name] = {"selected_affine": np.eye(4).tolist(), "expected_expanded_bytes": 352 + 4 * 5 * 6 * 4}
    baseline = {"headers": headers}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "INPUTS", inputs)
    monkeypatch.setattr(runner, "require_fit_release", lambda *_: baseline)
    monkeypatch.setattr(runner, "preflight", lambda: baseline)
    monkeypatch.setattr(runner.sys, "addaudithook", lambda _: None)
    source = tmp_path / inputs["before"]["path"]
    def mutate():
        with source.open("ab") as handle:
            handle.write(b"generated source changed")
    original_checked = runner.rigid.checked_source
    def decode(path, expected):
        result = original_checked(path, expected)
        if mutation_phase == "after_decode": mutate()
        return result
    def fit(*_):
        assert mutation_phase == "after_fit"
        mutate()
        return None, {}
    monkeypatch.setattr(runner.rigid, "checked_source", decode)
    monkeypatch.setattr(runner, "fit_arrays", fit)
    with pytest.raises(ValueError, match="Source size/type mismatch"):
        runner.worker(tmp_path / "preflight", tmp_path / "release", tmp_path)
    assert not (tmp_path / "candidate.json").exists()


def test_preflight_reads_only_generated_headers_and_streaming_hashes(tmp_path, monkeypatch):
    generated = {}
    for name in ("before", "during"):
        path = tmp_path / (name + ".nii.gz")
        # Deliberately no voxel payload: header preflight cannot establish payload validity.
        path.write_bytes(gzip.compress(header()))
        generated[name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": runner.digest(path)}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "INPUTS", generated)
    monkeypatch.setattr(runner, "verify_metadata", lambda: None)
    monkeypatch.setattr(nib, "load", lambda *_: pytest.fail("preflight decoded image"))
    result = runner.preflight()
    assert result["decoded_array_bytes"] == 0 and result["fit_run"] is False
    assert all(h["decompressed_bytes_returned"] == 348 for h in result["headers"].values())
    assert result["pair_float32_bytes"] == 2 * 31 * 32 * 33 * 4
