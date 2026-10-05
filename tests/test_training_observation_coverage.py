"""Small analytic controls only; never decode a patient, forward a policy or train."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_training_observation_coverage as runner
from resectionlab.geometry import ToolGeometry
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.spatial_multiscale_views import PermittedSourceChannel, PermittedVolumeSource, prepare_multiscale_views


def analytic_source(shape=(9, 7, 5), affine=None):
    image = np.indices(shape).sum(axis=0).astype(np.float32)
    target = np.zeros(shape, np.float32)
    target[1, 2, 2] = 1.
    target[-2, -2, -2] = .25
    channels = {
        "structural_intensity": PermittedSourceChannel(image, source_kind="observed_scan"),
        "nominal_tissue": PermittedSourceChannel(np.ones(shape), source_kind="supplied_annotation"),
        "nominal_target": PermittedSourceChannel(target, source_kind="supplied_annotation"),
        "observed_cavity": PermittedSourceChannel(np.zeros(shape), source_kind="observed_procedure_state"),
    }
    return PermittedVolumeSource(channels, np.eye(4) if affine is None else affine, "annotation_assisted", "analytic-only")


def test_current_policy_samples_distinguish_fractional_coverage_from_extent():
    coverage = np.ones((6, 5, 5, 5))
    coverage[2] = .25
    coverage[4:] = 0
    result = runner.segment_visibility(np.array([-1., 2., 2.]), np.array([5., 2., 2.]), np.eye(4), (5, 5, 5), coverage)
    assert result["sample_inside_center_domain"] == [False, True, True, True, False]
    assert result["continuous_center_domain_fraction"] == pytest.approx(4 / 6)
    assert result["continuous_fullcell_extent_fraction"] == pytest.approx(5 / 6)
    samples = np.asarray(result["sample_channel_coverage_fraction"])
    np.testing.assert_array_equal(samples[:, 2], [0, .25, .25, .25, 0])
    assert not samples[:, 4:].any()


def test_oblique_reflected_grid_roundtrip_and_material_outside():
    angle = .3
    affine = np.array([[-np.cos(angle), -2*np.sin(angle), 0, 14.],
                       [-np.sin(angle), 2*np.cos(angle), 0, -8.], [0, 0, 1.5, 4.], [0, 0, 0, 1.]])
    first = affine[:3, 3]
    last = affine[:3, :3] @ [4, 4, 4] + first
    coverage = np.ones((6, 5, 5, 5))
    result = runner.segment_visibility(first, last, affine, (5, 5, 5), coverage)
    assert all(result["sample_inside_center_domain"])
    outside = first - affine[:3, 0] * 1e-5
    result = runner.segment_visibility(outside, outside, affine, (5, 5, 5), coverage)
    assert not any(result["sample_inside_center_domain"])
    assert all(result["sample_inside_fullcell_extent"])


def test_union_uses_overlapping_segment_intervals_once():
    assert runner._union_length([[.1, .7], [.4, .9], None]) == pytest.approx(.8)
    assert runner._union_length([[0, .2], [.7, 1.]]) == pytest.approx(.5)
    assert runner._union_length([None, None]) == 0.


def test_full_source_conservation_and_shaft_convention_reported_without_clearance():
    source = analytic_source()
    views = prepare_multiscale_views(source, local_shape=(5, 5, 5), coarse_shape=(4, 3, 2))
    tool = ToolGeometry("analytic", .1, .1, 12., tip_length_mm=2.)
    inventory = {"emitted": [{"action_id": "row", "tool_id": tool.tool_id,
        "entry_mm": [0., 3., 2.], "tip_mm": [6., 3., 2.], "feasible": True, "reason": "OK"}]}
    report = runner.compare_views(views.target_local, views, inventory, (tool,), source.shape,
                                  source.affine_ras_mm, source.channels["nominal_target"].data)
    assert report["global_extent_max_error_mm"] < 1e-8
    assert report["coarse_nominal_mass_relative_error"] < 1e-6
    assert report["views"]["target_local"]["nominal_mass_omitted_mm3"] > 0
    row = report["actions"][0]["visibility"]
    # Shaft excludes the active distal2mm; working length is tip→proximal.
    approach = row["approach_shaft_centerline"]["whole_source"]["sample_points_ras_mm"]
    deepest = row["deepest_shaft_centerline"]["whole_source"]["sample_points_ras_mm"]
    assert approach[0] == [-12., 3., 2.] and approach[-1] == [-2., 3., 2.]
    assert deepest[0] == [-6., 3., 2.] and deepest[-1] == [4., 3., 2.]
    assert row["approach_shaft_centerline"]["whole_source"]["continuous_fullcell_extent_fraction"] == 0
    assert "no clearance" in report["interpretation"]


def test_native_catalog_and_legacy_observation_remain_exactly_unchanged():
    task = make_native_opening_task()
    before = runner.task_invariants(task)
    report = runner.inspect_task(task)
    assert report["task_before"] == report["task_after"] == before
    assert report["physical_catalog_unchanged"]
    assert report["executed_transitions"] == report["policy_forwards"] == report["optimizer_updates"] == 0
    assert not report["coverage"]["views"]["whole_source"]["channel_available"][4]


def test_private_reference_changes_neither_permitted_source_nor_views():
    first = make_native_opening_task()
    second = NativeSpatialTask(replace(first.case, reference_target=np.ones(first.case.reference_target.shape)), max_steps=first.max_steps)
    a = runner.full_source_from_task(first, first.observation())
    b = runner.full_source_from_task(second, second.observation())
    assert a.permitted_hash == b.permitted_hash
    x, y = prepare_multiscale_views(a), prepare_multiscale_views(b)
    assert x.target_local.fingerprint == y.target_local.fingerprint
    assert x.whole_source.fingerprint == y.whole_source.fingerprint
    assert runner.task_invariants(first) == runner.task_invariants(second)


@pytest.mark.parametrize("subject", ["sub-PAT16", "sub-PAT20", "sub-PAT26", "sub-PAT27", "sub-PAT29", "sub-PAT31", "../PAT05"])
def test_noneligible_subject_rejects_before_any_input_loader(monkeypatch, subject):
    monkeypatch.setattr(runner.prep, "_decode_case", lambda *a: pytest.fail("Image decoding forbidden"))
    monkeypatch.setattr(runner.prep, "load_prepared_training_case", lambda *a, **k: pytest.fail("Task construction forbidden"))
    with pytest.raises(ValueError, match="successful TRAIN"):
        runner.load_initial(subject, {}, {}, lambda: pytest.fail("Role guard must run first"))


def test_role_and_configuration_tampering_reject_before_declaration_reads(monkeypatch):
    monkeypatch.setattr(runner, "declaration", lambda: pytest.fail("Invalid role/settings reached metadata"))
    for subjects in (list(reversed(runner.SUBJECTS)), ["sub-PAT26"], list(runner.SUBJECTS[:-1])):
        with pytest.raises(ValueError, match="ordered six-TRAIN"):
            runner.validate({"version": runner.VERSION, "subjects": subjects, "settings": runner.SETTINGS})


def test_preview_budget_and_deadline_precede_constructor_and_restore_observer():
    old = sys.getprofile()
    with runner.InitialInventoryGuard(lambda: None) as guard:
        guard.begin_case("sub-PAT05")
        make_native_opening_task()
        assert guard.case_calls == guard.total > 0
        guard.subject = None
        with pytest.raises(RuntimeError, match="preview cap"):
            make_native_opening_task()
    assert sys.getprofile() is old
    guard = runner.InitialInventoryGuard(lambda: (_ for _ in ()).throw(TimeoutError("expired")))
    with pytest.raises(TimeoutError, match="expired"):
        guard.begin_case("sub-PAT05")
    assert guard.total == 0


def test_step_is_rejected_before_committing_analytic_native_state():
    task = make_native_opening_task()
    before = runner.task_invariants(task)
    with runner.InitialInventoryGuard(lambda: None):
        with pytest.raises(RuntimeError, match="committed transitions"):
            task.step("STOP")
    assert runner.task_invariants(task) == before


def test_blank_report_retains_all_six_with_historical_block_and_null_outcome():
    result = runner.blank_report()
    assert list(result["subjects"]) == list(runner.SUBJECTS)
    assert result["cohort_denominator"] == 6 and result["new_task_attempts"] == 0
    for subject in runner.BLOCKED:
        row = result["subjects"][subject]
        assert row["status"] == "historical_support_block"
        assert row["representation"] is None and not row["new_task_attempted"]


def test_worker_fake_success_keeps_historical_denominator_and_no_duplicate_case(monkeypatch, tmp_path):
    originals = {subject: {"coverage": {"outside": 19 if subject.endswith("16") else 125}} for subject in runner.BLOCKED}
    record = {"members": {subject: {"member": {"case_bundle": subject, "case_bundle_sha256": "constant"}} for subject in runner.SUBJECTS},
              "source_sha256": {"tiny": "constant"}}
    monkeypatch.setattr(runner, "validate_release", lambda *a: None)
    monkeypatch.setattr(runner, "validate", lambda *a: originals)
    monkeypatch.setattr(runner, "original_records", lambda: originals)
    monkeypatch.setattr(runner, "source_inventory", lambda: record["source_sha256"])
    monkeypatch.setattr(runner, "sha256", lambda p: "constant")
    calls = []
    monkeypatch.setattr(runner, "load_initial", lambda subject, *a: calls.append(subject))
    monkeypatch.setattr(runner, "inspect_task", lambda *a: {"zero_updates": True})
    result = runner.worker(record, tmp_path, manifest_sha256="test", release={})
    assert calls == [x for x in runner.SUBJECTS if x not in runner.BLOCKED]
    assert result["status"] == "complete" and result["new_task_attempts"] == result["new_representation_completions"] == 4
    assert result["cohort_denominator"] == len(result["subjects"]) == 6
    for subject in runner.BLOCKED:
        assert result["subjects"][subject]["representation"] is None


def test_release_requires_archive_not_just_authorized_flag(tmp_path):
    release = {"version": runner.RELEASE_VERSION, "authorized": True, "manifest_sha256": "test",
               "runtime_root": str(runner.ROOT.resolve()), "source_commit": "a" * 40,
               "source_archive": str(tmp_path / "absent.tar"), "source_archive_sha256": "0" * 64}
    with pytest.raises(ValueError, match="archive changed or is absent"):
        runner.validate_release(release, "test")
