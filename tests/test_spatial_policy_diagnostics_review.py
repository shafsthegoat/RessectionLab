"""Independent diagnostics contracts; artificial arrays only, no learning."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_spatial_task import DEFAULT_NATIVE_SPATIAL_REWARD
from resectionlab.spatial_observations import (
    ObservedChannel, ObservedProcedureState, SpatialAction, SpatialInputs, build_spatial_observation,
)
from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler, nominal_depth_coverage, spatial_coverage

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("preflight_diagnostics_independent", ROOT / "scripts/preflight_real_spatial_policy.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def observation_fixture(*, include_target=True):
    angle = .63
    source = np.eye(4)
    source[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.],
                              [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]) @ np.diag([-1.25, 2., 3.])
    source[:3, 3] = [30., -80., 110.]
    origin, shape = np.array([2, 2, 2]), (5, 7, 9)
    crop = source.copy()
    crop[:3, 3] += source[:3, :3] @ origin
    world = lambda p: crop[:3, :3] @ p + crop[:3, 3]
    full = np.zeros((9, 11, 13), np.float32)
    full[2, 2, 2], full[8, 10, 12] = .25, .75
    channels = {"structural_intensity": ObservedChannel(np.ones(shape), source_kind="observed_scan"),
        "observed_cavity": ObservedChannel(np.zeros(shape), source_kind="observed_procedure_state")}
    if include_target:
        channels["nominal_target"] = ObservedChannel(full[2:7, 2:9, 2:11], source_kind="supplied_annotation")
    tool = ToolGeometry("unit-probe", .2, .2, 30., tip_length_mm=1.)
    access = AccessWindow(world([-1., 3., 4.]), crop[:3, 0] / np.linalg.norm(crop[:3, 0]), 2.)
    actions = (SpatialAction("STOP"),
        SpatialAction("cross", world([-1., 3., 4.]), world([3., 3., 4.]), tool),
        SpatialAction("corners", world([0., 0., 0.]), world([4., 6., 8.]), tool),
        SpatialAction("outside", world([-1e-8, 0., 0.]), world([-1e-8, 6., 8.]), tool))
    observation = build_spatial_observation(SpatialInputs(channels, crop, "annotation_assisted", "analytic-only"),
        actions, ObservedProcedureState(access, 0, 3))
    return observation, source, full, tool


def test_oblique_reflected_crop_bounds_use_voxel_centers_and_distinguish_continuous_fraction():
    observation, source, target, _ = observation_fixture()
    before = observation.fingerprint
    result = spatial_coverage(observation, source_shape=target.shape, source_affine=source, nominal_target=target)
    np.testing.assert_allclose(result["crop_origin_source_voxels"], [2., 2., 2.], atol=1e-12)
    np.testing.assert_allclose(result["crop_last_center_source_voxels"], [6., 8., 10.], atol=1e-12)
    rows = {row["action_id"]: row for row in result["actions"]}
    assert rows["cross"]["sample_fraction_inside"] == .8
    assert rows["cross"]["straight_segment_fraction_inside"] == pytest.approx(.75)
    assert rows["corners"]["ray_samples_inside"] == 5
    assert rows["corners"]["straight_segment_fraction_inside"] == 1.
    assert rows["outside"]["ray_samples_inside"] == 0
    assert rows["outside"]["straight_segment_fraction_inside"] == 0.
    assert result["sample_coverage_counts"] == {"none": 1, "partial": 1, "full": 1}
    assert result["nominal_target_mass_fraction_visible"] == .25
    assert result["nominal_target_positive_voxels_total"] == 2
    assert result["nominal_target_positive_voxels_in_crop"] == 1
    assert observation.fingerprint == before
    observation.assert_intact()


def test_missing_nominal_target_is_reported_unknown_without_private_fallback():
    observation, source, target, _ = observation_fixture(include_target=False)
    report = spatial_coverage(observation, source_shape=target.shape, source_affine=source, nominal_target=None)
    assert report["nominal_target_available"] is False
    for field in ("mass_in_crop", "mass_total", "mass_fraction_visible", "positive_voxels_total"):
        assert report["nominal_target_" + field] is None
    assert report["legal_nonstop_actions"] == 3


def test_negative_source_axis_depths_use_mm_and_declared_capsule_radius_only():
    _, affine, target, tool = observation_fixture()
    nominal = np.zeros(target.shape)
    nominal[2, 4, 4], nominal[7, 4, 4] = .2, .8
    center_voxel = np.array([8.5, 4., 4.])
    center = affine[:3, :3] @ center_voxel + affine[:3, 3]
    normal = -affine[:3, 0] / np.linalg.norm(affine[:3, 0])
    case = SimpleNamespace(affine_ras_mm=affine, access=AccessWindow(center, normal, 2.),
        nominal_target=nominal, tools=(tool,), _candidate_voxels=((8, 4, 4), (6, 4, 4)),
        _candidate_scope="fixed_access_grid_3columns_8depths_within_actor_crop")
    report = nominal_depth_coverage(case)
    assert report["nominal_source_axis"] == 0 and report["inward_sign"] == -1
    assert report["retained_depth_offsets_voxels"] == [0, 2]
    np.testing.assert_allclose(report["endpoint_depth_range_mm"], [.625, 3.125])
    assert report["distal_tip_capsule_axial_upper_bound_mm"] == pytest.approx(3.325)
    np.testing.assert_allclose(report["nominal_target_center_depth_range_mm"], [1.875, 8.125])
    assert report["nominal_target_centers_beyond_axial_upper_bound"] == 1


def test_profiler_preserves_exact_calls_results_and_exception_identity_across_nested_phases():
    calls, error = [], KeyboardInterrupt("unit cancellation")
    result = SimpleNamespace(feasible=True, microsteps=(1, 2), reason="")

    class Engine:
        def preview_stroke(self, *args, **kwargs):
            calls.append((self, args, kwargs))
            if kwargs.get("cancel"):
                raise error
            return result

    engine, token, original = Engine(), object(), Engine.preview_stroke
    with pytest.raises(KeyboardInterrupt) as caught:
        with NativePreviewProfiler(Engine) as profiler:
            with profiler.phase("outer"):
                with profiler.phase("inner"):
                    assert engine.preview_stroke(token, exact=token) is result
                assert profiler.phase_name == "outer"
                snapshot = profiler.snapshot()
                snapshot["phases"]["inner"]["started"] = 99
                assert profiler.snapshot()["phases"]["inner"]["started"] == 1
                engine.preview_stroke(token, cancel=True)
    assert caught.value is error and Engine.preview_stroke is original
    assert calls == [(engine, (token,), {"exact": token}), (engine, (token,), {"cancel": True})]
    assert profiler.phase_name == "unclassified"
    assert profiler.snapshot()["phases"]["outer"]["raised"] == 1
    assert profiler.snapshot()["phases"]["inner"]["returned_microsteps"] == 2


@pytest.mark.parametrize("grid_case", ["default", "derived", "mismatch"])
def test_profile_worker_never_constructs_optimizer_or_takes_update_branch(monkeypatch, tmp_path, grid_case):
    import torch
    import resectionlab.native_spatial_task as task_module
    import resectionlab.spatial_policy as policy_module
    import resectionlab.spatial_policy_diagnostics as diagnostic_module
    calls, diagnostic_calls, observation = [], [], SimpleNamespace(action_ids=("STOP", "CUT"))
    derived = np.eye(4).tolist() if grid_case != "default" else None
    grid_record = {} if derived is None else {"method": "unit-derived", "derived_affine_ras_mm": derived}

    def forbidden(*args, **kwargs):
        raise AssertionError("Profile must not enter an optimizer, gradient or checkpoint path")

    class Policy:
        architecture_hash = "unit-architecture"
        config = SimpleNamespace(ray_samples=5)
        def __init__(self, *args):
            pass
        def architecture_record(self):
            return {"architecture": "fake-unit-only"}

    fake_case = SimpleNamespace(structural_intensity=np.zeros((3, 3, 3)), affine_ras_mm=np.eye(4), nominal_target=None)
    fake_task = SimpleNamespace(case=fake_case, reward_spec=DEFAULT_NATIVE_SPATIAL_REWARD,
        metrics=lambda: {"native_grid_reconciliation": grid_record},
        observation=lambda: observation, candidate_inventory=lambda: {})
    declaration = {"version": runner.VERSION, "subject": "unit-only", "mode": "profile", "track": "annotation_assisted",
        "settings": {"seed": 11, "max_steps": 3, "max_rss_bytes": 10**9, "max_wall_seconds": 30},
        "policy_config": {}, "expected_policy_architecture_hash": "unit-architecture",
        "access": {"center_mm": [0., 0., 0.], "normal_inward": [0., 0., 1.], "radius_mm": 2.},
        "tools": [asdict(ToolGeometry("unit", .2, .2, 30., tip_length_mm=1.))],
        "objective": asdict(DEFAULT_NATIVE_SPATIAL_REWARD)}
    if derived is not None:
        declaration["expected_native_grid_binding"] = {"method": "unit-derived", "derived_affine_ras_mm": derived}
    if grid_case == "mismatch":
        declaration["expected_native_grid_binding"]["method"] = "stale-declared-grid"
    for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms", "manual_seed"):
        monkeypatch.setattr(torch, name, lambda *args: None)
    monkeypatch.setattr(torch, "Generator", lambda: SimpleNamespace(manual_seed=lambda value: None))
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    monkeypatch.setattr(policy_module, "SpatialPolicy", Policy)
    monkeypatch.setattr(policy_module, "parameter_hash", lambda value: "unit-parameters")
    monkeypatch.setattr(policy_module, "reinforce_loss", forbidden)
    monkeypatch.setattr(policy_module, "gradient_step", forbidden)
    monkeypatch.setattr(task_module, "native_spatial_task_from_case", lambda *args, **kwargs: fake_task)
    monkeypatch.setattr(runner, "load_inputs", lambda value: (fake_case, "unit-only", {"source": "frozen"}))
    monkeypatch.setattr(runner, "numerical_source_inventory", lambda: {"source": "frozen"})
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 1)
    monkeypatch.setattr(diagnostic_module, "nominal_depth_coverage",
        lambda case, **kwargs: diagnostic_calls.append(("depth", kwargs["native_affine"])) or {})
    monkeypatch.setattr(diagnostic_module, "spatial_coverage",
        lambda *args, **kwargs: diagnostic_calls.append(("coverage", kwargs["native_affine"])) or {})
    monkeypatch.setattr(runner, "episode", lambda *args, **kwargs: (calls.append(kwargs) or [], {"fake_episode": True}))
    profiler = SimpleNamespace(phase=lambda name: nullcontext(), snapshot=lambda: {})
    if grid_case == "mismatch":
        with pytest.raises(ValueError, match="Executed native grid differs"):
            runner._worker_with_profiler(declaration, tmp_path, profiler)
        assert not calls and not diagnostic_calls
        return
    runner._worker_with_profiler(declaration, tmp_path, profiler)
    assert len(calls) == 1 and calls[0]["profile_actions"] and not calls[0]["stochastic"]
    assert calls[0]["native_affine"] == derived
    assert diagnostic_calls == [("depth", derived), ("coverage", derived)]
    report = json.loads((tmp_path / "receipt.json").read_text())
    assert report["optimizer_updates"] == 0
    assert [row["phase"] for row in report["episodes"]] == ["untrained_fixed_inventory_profile"]
    assert not (tmp_path / "latest-policy.pt").exists()


def test_supervisor_rss_measurement_failure_stops_group_and_preserves_failure_identity(monkeypatch, tmp_path):
    signals, launches = [], []

    class Process:
        pid, returncode = 999999, None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            return self.returncode

    process = Process()

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return process

    def kill(pid, signal):
        signals.append((pid, signal))
        process.returncode = -signal

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="unavailable"))
    monkeypatch.setattr(runner.os, "killpg", kill)
    command = ["never-launched", "--expected-declaration-sha256", "f" * 64]
    result = runner.supervise_worker(command, tmp_path,
        {"max_wall_seconds": 30., "max_rss_bytes": 10**9}, "f" * 64)
    assert result["status"] == "failed"
    assert result["termination_reason"].startswith("parent_supervision_error:RuntimeError:Worker RSS unavailable")
    assert result["declaration_sha256"] == "f" * 64 and not result["automatic_retry"]
    assert signals == [(process.pid, runner.signal.SIGTERM)]
    assert launches[0][0] is command and launches[0][1]["start_new_session"]
    assert launches[0][1]["env"]["OMP_NUM_THREADS"] == "1"
    assert json.loads((tmp_path / "supervisor-failure.json").read_text()) == result


def test_roundoff_reporting_uses_derived_physical_frame_and_preserves_original_index_authority():
    from resectionlab.native_spatial_task import make_native_opening_task
    original = np.eye(4)
    original[2, 0] = 1e-9
    source = replace(make_native_opening_task().case, affine_ras_mm=original,
        native_grid_reconciliation="orthogonal_roundoff_1e-6mm", crop_shape=(5, 5, 5))
    inputs = source.spatial_inputs(np.zeros(source.structural_intensity.shape, bool))
    observation = build_spatial_observation(inputs, (SpatialAction("STOP"),),
        ObservedProcedureState(source.access, 0, 3))
    arguments = {"source_shape": source.structural_intensity.shape, "source_affine": original,
                 "nominal_target": source.nominal_target}
    default = spatial_coverage(observation, **arguments)
    assert default == spatial_coverage(observation, **arguments, native_affine=original)
    physical = source._native_affine_ras_mm
    corrected = spatial_coverage(observation, **arguments, native_affine=physical)
    np.testing.assert_allclose(corrected["crop_origin_source_voxels"], source._crop_origin, rtol=0, atol=1e-14)
    assert np.max(np.abs(np.asarray(default["crop_origin_source_voxels"]) - source._crop_origin)) > 1e-11
    assert corrected["original_source_affine_ras_mm"] == original.tolist()
    assert corrected["native_physical_affine_ras_mm"] == physical.tolist()
    depths = nominal_depth_coverage(source, native_affine=physical)
    default_depths = nominal_depth_coverage(source)
    assert default_depths == nominal_depth_coverage(source, native_affine=original)
    assert depths["retained_depth_offsets_voxels"] == default_depths["retained_depth_offsets_voxels"]
    points = np.column_stack((source._candidate_voxels, np.ones(len(source._candidate_voxels)))) @ physical.T
    manual = (points[:, :3] - source.access.center_mm) @ source.access.normal_inward
    np.testing.assert_allclose(depths["endpoint_depth_range_mm"], [manual.min(), manual.max()], rtol=0, atol=1e-14)
    assert depths["endpoint_depth_range_mm"] != default_depths["endpoint_depth_range_mm"]
