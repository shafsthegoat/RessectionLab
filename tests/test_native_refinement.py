"""Real native updates reach the desktop only through an independent replay gate."""
import copy

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.native_simulation import NativeSequentialSimulator
from resectionlab.native_refinement import (native_replay_mask, replay_artifact_hash,
    inspect_native_refinement, native_partial_contact_accounting, recheck_native_replay,
    run_native_refinement, validate_native_replay)


@pytest.fixture
def native_case(monkeypatch):
    tissue = np.ones((5, 5, 4), bool)
    target = np.zeros(tissue.shape, bool)
    target[1:4, 1:4, 2:] = True
    case = CaseData(case_id="desktop-native-fixture", mri=tissue.astype(np.float32),
        compartments={"target": target}, affine=np.eye(4), brain_mask=tissue,
        source_refs=(SourceRef("fixture", "synthetic://native-refinement", provenance="simulated"),))
    def factory(case, **kwargs):
        config = NativeResectionConfig(tissue, target.astype(np.int16), np.eye(4),
            kwargs.get("access") or AccessWindow((2, 2, -.5), (0, 0, 1), 3.),
            kwargs.get("tools") or NATIVE_GENERIC_TOOLS,
            case.semantic_hash, "synthetic solid cube")
        target_point = kwargs.get("selected_target_mm") or (2, 2, 3)
        entry = kwargs.get("selected_entry_mm")
        return NativeSequentialSimulator(config, [target_point],
            candidate_entries_mm=None if entry is None else [entry], max_steps=kwargs["max_steps"],
            max_actions=kwargs["max_actions"], cancelled=kwargs["cancelled"])
    monkeypatch.setattr("resectionlab.native_simulation.make_native_patient_simulator", factory)
    return case


def test_actual_updates_and_source_grid_replay_are_certified(native_case, tmp_path):
    report = run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11, max_steps=1)
    assert report["gradient_steps"] > 0
    assert report["actor_parameters_changed"] is True
    assert report["role"] == "selection" and report["final_evaluation"] is False
    assert report["replay_status"] == "accepted_independent_geometry"
    replay = report["replay"]
    assert validate_native_replay(native_case, replay)
    removed = native_replay_mask(replay, len(replay["metrics"]["history"]))
    assert removed.sum() == pytest.approx(replay["metrics"]["simulated_removed_target_volume_mm3"] +
                                          replay["metrics"]["simulated_removed_normal_volume_mm3"])
    changed = copy.deepcopy(replay)
    changed["role"] = "final_evaluation"
    changed["artifact_hash"] = replay_artifact_hash(changed)
    with pytest.raises(ValueError, match="selection artifact"):
        validate_native_replay(native_case, changed)
    changed = copy.deepcopy(replay)
    changed["metrics"]["simulated_removed_target_volume_mm3"] += 1
    with pytest.raises(ValueError, match="changed after"):
        validate_native_replay(native_case, changed)
    forged = copy.deepcopy(replay)
    forged["native_certificate"]["feasible"] = False
    forged["artifact_hash"] = replay_artifact_hash(forged)
    repaired = recheck_native_replay(native_case, forged, max_steps=1)
    assert repaired["native_certificate"]["feasible"] is True
    forged = copy.deepcopy(replay)
    forged["actions"] = ["UNKNOWN_NATIVE_ACTION"]
    forged["artifact_hash"] = replay_artifact_hash(forged)
    with pytest.raises(ValueError, match="Unknown or stale"):
        recheck_native_replay(native_case, forged, max_steps=1)


def test_cancelled_training_exposes_no_uncertified_replay_and_resumes(native_case, tmp_path):
    state = {"cancelled": False}
    first = run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11, max_steps=1,
        cancelled=lambda: state["cancelled"], progress=lambda _: state.update(cancelled=True))
    assert first["status"] == "cancelled" and first["gradient_steps"] == 2
    assert first["replay"] is None
    state["cancelled"] = False
    resumed = run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11,
                                    max_steps=1, resume=True)
    assert resumed["gradient_steps"] >= first["gradient_steps"]
    assert resumed["replay_status"] == "accepted_independent_geometry"


def test_failed_independent_check_never_releases_native_replay(native_case, tmp_path, monkeypatch):
    from resectionlab.evaluation import NativeRemovalAudit
    def rejected(*args, **kwargs):
        return NativeRemovalAudit(False, ("injected_independent_failure",), None, None, None,
                                  0., 0., 0., native_case.semantic_hash, 1., 0)
    monkeypatch.setattr("resectionlab.evaluation.independent_check_native_history", rejected)
    report = run_native_refinement(native_case, tmp_path, budget_seconds=1., seed=11, max_steps=1)
    assert report["replay"] is None
    assert report["replay_status"] == "rejected_independent_geometry"
    assert (tmp_path / "native-candidate-freeze.json").exists()
    assert not (tmp_path / "native-selection-replay.json").exists()


def test_partial_contact_is_not_permanently_retained_when_a_later_stroke_removes_it(native_case):
    history = [
        {"removed_indices_native": [[0, 0, 0]], "contact_indices_native": [[0, 0, 0], [0, 0, 1]]},
        {"removed_indices_native": [[0, 0, 1]], "contact_indices_native": [[0, 0, 1]]},
    ]
    assert native_partial_contact_accounting(native_case, history) == {
        "cumulative_partial_normal_contact_mm3": 1.,
        "currently_retained_partial_normal_contact_mm3": 0.,
        "previously_partial_normal_later_removed_mm3": 1.,
    }


def test_exact_selected_route_is_preserved_through_updates_reopen_and_resume(native_case, tmp_path):
    options = {"access": AccessWindow((2, 2, -.5), (0, 0, 1), 3., "selected-window"),
               "tools": (NATIVE_GENERIC_TOOLS[0],), "selected_entry_mm": [1., 2., -.5],
               "selected_target_mm": [1., 2., 3.], "candidate_count": 9, "max_steps": 1}
    ready = inspect_native_refinement(native_case, **options)
    assert ready["legalNonStopActions"] == 1
    binding = ready["route_binding"]
    assert binding["candidate_entries_mm"] == [[1., 2., -.5]]
    assert binding["candidate_targets_mm"] == [[1., 2., 3.]]
    assert binding["mode"] == "exact_selected_route"
    report = run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11, **options)
    assert report["gradient_steps"] > 0
    assert report["route_binding"] == binding
    assert report["replay"]["route_binding"] == binding
    assert recheck_native_replay(native_case, report["replay"], **options)["route_binding"] == binding
    modified = {**options, "selected_target_mm": [2., 2., 3.]}
    before = (tmp_path / "native-refinement.json").read_bytes()
    with pytest.raises(ValueError, match="Resume route or settings changed"):
        run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11, resume=True, **modified)
    assert (tmp_path / "native-refinement.json").read_bytes() == before
    with pytest.raises(ValueError, match="route binding differs"):
        recheck_native_replay(native_case, report["replay"], **modified)


def test_stop_only_preflight_creates_no_optimizer_or_checkpoint(native_case, tmp_path, monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError("STOP-only preflight must not create an optimizer or take a transition")
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(NativeSequentialSimulator, "step", forbidden)
    options = {"access": AccessWindow((2, 2, -.5), (0, 0, 1), 3.),
        "tools": (ToolGeometry("subvoxel-tip", .05, .05, 10., tip_length_mm=.1),),
        "selected_entry_mm": [2., 2., -.5], "selected_target_mm": [2., 2., 3.]}
    readiness = inspect_native_refinement(native_case, **options)
    assert readiness["status"] == "no_actionable_moves" and readiness["legalNonStopActions"] == 0
    assert readiness["reasons"]
    report = run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11, **options)
    assert report["status"] == report["replay_status"] == "no_actionable_moves"
    assert report["gradient_steps"] == report["optimization_environment_steps"] == report["selection_environment_steps"] == 0
    assert report["optimizer_mode"] is None and report["replay"] is None
    assert report["role"] == "preflight" and report["selection_history"] == []
    assert (tmp_path / "native-refinement.json").exists()
    for name in ("checkpoint.pt", "initial.pt", "contract.json", "native-selection-replay.json"):
        assert not (tmp_path / name).exists()
    previous = (tmp_path / "native-refinement.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_native_refinement(native_case, tmp_path, budget_seconds=3., seed=11, **options)
    assert (tmp_path / "native-refinement.json").read_bytes() == previous


def test_selected_geometry_requires_paired_points_window_and_explicit_tool(native_case):
    with pytest.raises(ValueError, match="together"):
        inspect_native_refinement(native_case, selected_entry_mm=[2, 2, -.5])
    with pytest.raises(ValueError, match="explicit access window and tool"):
        inspect_native_refinement(native_case, selected_entry_mm=[2, 2, -.5], selected_target_mm=[2, 2, 3])
