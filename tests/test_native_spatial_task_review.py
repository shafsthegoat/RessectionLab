"""Independent analytic geometry/authority checks; no training or patient runs."""
from dataclasses import replace
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.native_spatial_task import (DEFAULT_NATIVE_SPATIAL_REWARD, NativeSpatialCase,
    NativeSpatialTask, OPENING_TOOLS)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_real_spatial_preflight", ROOT / "scripts/preflight_real_spatial_policy.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def analytic_case(*, private_target=False):
    shape, shift = (24, 25, 18), np.array([7, 8, 2])
    support = np.zeros(shape, bool)
    points = [np.array([4, 4, depth]) + shift for depth in range(1, 6)] + [np.array([5, 5, 1]) + shift]
    for point in points:
        support[tuple(point)] = True
    nominal = np.zeros(shape, np.float32)
    for point in points[3:5]:
        nominal[tuple(point)] = 1
    reference = np.ones(shape, np.float32) if private_target else nominal
    angle = .23
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, -1.]])
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = rotation, [14, -8, 12]
    access_voxel = np.array([3.9, 4, .5]) + shift
    access = AccessWindow(rotation @ access_voxel + affine[:3, 3], rotation @ [0, 0, 1], 2.4,
                          "independent-analytic-hypothetical-aperture")
    source = NativeSpatialCase(np.full(shape, 100., np.float32), support, reference, affine, access,
        OPENING_TOOLS, track="inference_only", support_source_kind="derived_from_scan",
        support_derivation="declared analytic source generator; no human anatomy",
        nominal_target=nominal, target_source_kind="derived_from_scan",
        target_derivation="declared analytic estimator held fixed across private references", crop_shape=(5, 5, 5))
    return source, shift


def cut(task, tool, point):
    return next(row["action_id"] for row in task.candidate_inventory()["ledger"]
                if row["voxel"] == list(point) and row["tool_id"] == tool and row["feasible"])


def test_native_inventory_is_complete_over_declared_primitives_and_exact_entries():
    case, _ = analytic_case()
    task = NativeSpatialTask(case, max_steps=2)
    inventory, observation = task.candidate_inventory(), task.observation()
    expected = {(tuple(point), tool.tool_id) for point, tool in itertools.product(np.argwhere(case.observed_support), case.tools)}
    actual = {(tuple(row["voxel"]), row["tool_id"]) for row in inventory["ledger"]}
    assert actual == expected and len(actual) == inventory["declared_slots"] == inventory["evaluated_slots"] == 12
    assert inventory["complete"] and inventory["omitted_count"] == 0
    assert inventory["accepted_count"] + inventory["rejected_count"] == 12
    assert observation.action_ids[0] == "STOP" and observation.action_mask.all()
    assert set(observation.action_ids[1:]) == {row["action_id"] for row in inventory["ledger"] if row["feasible"]}
    for row in inventory["ledger"]:
        tip = case.affine_ras_mm[:3, :3] @ row["voxel"] + case.affine_ras_mm[:3, 3]
        entry = tip - np.dot(tip - case.access.center_mm, case.access.normal_inward) * case.access.normal_inward
        np.testing.assert_allclose(row["tip_mm"], tip, rtol=0, atol=1e-12)
        np.testing.assert_allclose(row["entry_mm"], entry, rtol=0, atol=1e-12)
        if row["feasible"]:
            i = observation.action_ids.index(row["action_id"])
            np.testing.assert_allclose(observation.action_geometry[i, 1:7], [*entry, *tip], rtol=0, atol=2e-6)
        else:
            assert row["reason"]


def test_actor_crop_affine_does_not_crop_native_removal_or_retained_contact():
    case, shift = analytic_case()
    task = NativeSpatialTask(case, max_steps=2)
    obs = task.observation()
    origin = np.array(case._crop_origin)
    assert obs.image_channels.shape[-3:] == (5, 5, 5)
    assert task._config.tissue_mask.shape == case.structural_intensity.shape
    for voxel in ([0, 0, 0], [4, 4, 4]):
        actual = obs.affine_ras_mm[:3, :3] @ voxel + obs.affine_ras_mm[:3, 3]
        expected = case.affine_ras_mm[:3, :3] @ (origin + voxel) + case.affine_ras_mm[:3, 3]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    first = task.step(cut(task, OPENING_TOOLS[0].tool_id, np.array([4, 4, 1]) + shift))
    retained = task._engine.contact_mask & ~task._engine.removed_mask
    assert retained.any() and task._engine.remaining_mask[retained].all()
    metrics = task.metrics()
    assert metrics["partial_contact_weight"] == 0 and metrics["currently_retained_contacted_tissue_upper_bound_mm3"] > 0
    assert metrics["simulated_removed_volume_mm3"] == pytest.approx(3.)
    assert first.info["motor_surrogate"] is first.info["language_surrogate"] is None
    assert first.info["clinical_deficit_probability"] is None
    assert not any(metrics["functional_evidence_available"].values())
    second = task.step(cut(task, OPENING_TOOLS[1].tool_id, np.array([4, 4, 5]) + shift))
    removed = np.asarray(second.info["removed_indices_native"])
    assert np.any(np.any((removed < origin) | (removed >= origin + 5), axis=1))
    assert task.metrics()["simulated_removed_volume_mm3"] == pytest.approx(6.)
    assert task.independent_geometry_check().feasible


def test_private_reference_change_never_changes_proposals_policy_inputs_or_nominal_teacher():
    source, shift = analytic_case()
    other, _ = analytic_case(private_target=True)
    a, b = NativeSpatialTask(source, max_steps=2), NativeSpatialTask(other, max_steps=2)
    assert source.reference_hash != other.reference_hash and source.source_hash == other.source_hash
    assert a.decision_model_hash == b.decision_model_hash
    for stage in range(2):
        assert a.candidate_inventory() == b.candidate_inventory()
        first, second = a.observation(), b.observation()
        assert first.fingerprint == second.fingerprint and first.action_ids == second.action_ids
        for name in ("image_channels", "coverage", "action_geometry", "state_features", "action_mask"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        assert a.planning_clone().metrics() == b.planning_clone().metrics()
        if stage == 0:
            identifier = cut(a, OPENING_TOOLS[0].tool_id, np.array([4, 4, 1]) + shift)
            assert a.step(identifier).reward != b.step(identifier).reward


@pytest.mark.parametrize("mutation", ["tool", "access", "budget", "reward"])
def test_world_tool_access_and_objective_cannot_change_mid_episode(mutation):
    source, _ = analytic_case()
    task = NativeSpatialTask(source, max_steps=2)
    if mutation == "tool": object.__setattr__(source, "tools", (replace(OPENING_TOOLS[0], tip_radius_mm=2.3),))
    elif mutation == "access": object.__setattr__(source, "access", replace(source.access, radius_mm=3.))
    elif mutation == "budget": task.max_steps = 3
    else: task.reward_spec = replace(DEFAULT_NATIVE_SPATIAL_REWARD, normal_per_mm3=.1)
    with pytest.raises((ValueError, RuntimeError)):
        task.observation()


def test_unsupported_worlds_and_functional_objectives_are_explicitly_refused():
    source, _ = analytic_case()
    task = NativeSpatialTask(source)
    with pytest.raises(ValueError, match="deterministic"):
        task.reset(42)
    for field in ("motor_per_mm3", "language_per_mm3", "graph_edge_cost"):
        with pytest.raises(ValueError, match="function remains unassessed"):
            NativeSpatialTask(source, reward=replace(DEFAULT_NATIVE_SPATIAL_REWARD, **{field: 1.}))


def declaration(subject="sub-PAT22"):
    return {"version": runner.VERSION, "mode": "profile", "track": "annotation_assisted",
        "source_sha256": runner.numerical_source_inventory(), "cohort_path": str(runner.COHORT_PATH),
        "cohort_sha256": "962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985",
        "subject": subject, "case_bundle": "THIS_MUST_NOT_BE_READ.ressectionlab"}


def test_actual_frozen_train_metadata_and_complete_source_inventory_validate_without_images():
    record = declaration()
    cohort, group, sources = runner.validate_declaration(record)
    assert group == "BTC:sub-PAT22" and cohort["source"]["git_commit"] == "359d372c5e972a161966312128adb365870df949"
    assert sources == record["source_sha256"]
    assert {"scripts/preflight_real_spatial_policy.py", "src/resectionlab/native_spatial_task.py",
            "src/resectionlab/native_resection.py", "src/resectionlab/geometry.py",
            "src/resectionlab/spatial_policy.py", "src/resectionlab/evaluation.py"} <= set(sources)


@pytest.mark.parametrize("subject", ["sub-PAT26", "sub-PAT27", "sub-PAT29", "sub-PAT31"])
def test_selection_or_sealed_subject_refused_before_bundle_checksum_or_load(monkeypatch, subject):
    import resectionlab.imaging as imaging
    record = declaration(subject)
    original = runner.sha256
    def checksum(path):
        assert "THIS_MUST_NOT_BE_READ" not in str(path), "Forbidden role reached case checksum"
        return original(path)
    monkeypatch.setattr(runner, "sha256", checksum)
    monkeypatch.setattr(imaging, "load_case", lambda *a, **kw: pytest.fail("Forbidden role reached images"))
    with pytest.raises(ValueError, match="only.*TRAIN"):
        runner.load_inputs(record)


@pytest.mark.parametrize("subject", ["sub-PAT26", "sub-PAT29"])
def test_resealed_role_promotion_cannot_replace_frozen_cohort(tmp_path, monkeypatch, subject):
    import resectionlab.imaging as imaging
    record = declaration(subject)
    cohort = json.loads((ROOT / "manifests/experiments/btc-spatial-development-cohort-v1.json").read_text())
    next(row for row in cohort["candidates"] if row["subject"] == subject)["development_role"] = "population_training"
    path = tmp_path / "resealed-cohort.json"
    path.write_text(json.dumps(cohort))
    record.update(cohort_path=str(path), cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(imaging, "load_case", lambda *a, **kw: pytest.fail("Resealed role reached images"))
    with pytest.raises(ValueError, match="Cohort path and identity"):
        runner.load_inputs(record)


@pytest.mark.parametrize("mutation", ["omit_engine", "readme_only", "changed_policy"])
def test_numerical_source_contract_is_closed_before_images(monkeypatch, mutation):
    import resectionlab.imaging as imaging
    record = declaration()
    if mutation == "omit_engine":
        del record["source_sha256"]["src/resectionlab/native_resection.py"]
    elif mutation == "readme_only":
        record["source_sha256"] = {"README.md": runner.sha256(ROOT / "README.md")}
    else:
        record["source_sha256"]["src/resectionlab/spatial_policy.py"] = "0" * 64
    monkeypatch.setattr(imaging, "load_case", lambda *a, **kw: pytest.fail("Incomplete source freeze reached images"))
    with pytest.raises(ValueError, match="Complete numerical source"):
        runner.load_inputs(record)


@pytest.mark.parametrize("changed", ["case_bundle", "access", "tools", "settings"])
def test_parent_worker_byte_binding_includes_case_tool_access_and_settings(tmp_path, changed):
    record = {"case_bundle": "declared-source", "access": {"radius_mm": 2.4},
              "tools": [{"tool_id": "declared-tool"}], "settings": {"max_steps": 2}}
    path = tmp_path / "declaration.json"
    original = json.dumps(record, indent=2).encode() + b"\n"
    path.write_bytes(original)
    value, digest, data = runner.read_declaration(path)
    assert value == record and data == original and digest == hashlib.sha256(original).hexdigest()
    assert runner.read_declaration(path, expected_sha256=digest)[2] == original
    record[changed] = "changed-after-parent-read"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Declaration bytes changed"):
        runner.read_declaration(path, expected_sha256=digest)
