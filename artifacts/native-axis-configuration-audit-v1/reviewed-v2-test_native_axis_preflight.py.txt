"""Preflight gates on tiny generated anatomy; no patient execution or gradients."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_native_axis as runner
from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.worlds import WorldGeneratorConfig, WorldRole, content_hash, generate_partitions


def fixture(tmp_path):
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros_like(tissue)
    target[1:6, 1:6, 2:7] = True
    case = CaseData("axis-preflight-test", tissue.astype(float), {"target": target}, np.eye(4),
        (SourceRef("analytic", "simulated:preflight", provenance="simulated"),), brain_mask=tissue)
    cfg = NativeResectionConfig(tissue, target.astype(np.int16), np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        case.semantic_hash, "explicit synthetic tissue", case_id=case.case_id)
    factory = lambda guard: AxisColumnNativeSimulator(cfg,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 2), max_steps=1, cancelled=guard)
    panels = generate_partitions(case.semantic_hash, WorldGeneratorConfig(), 71,
        optimization=1, selection=2, final_evaluation=1, stress=1, planning_hash=case.planning_hash)
    source = tmp_path / "stable-source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/preflight_native_axis.py").write_text("# private immutable source for orchestration test\n")
    return case, factory, panels, source


def guard():
    return runner.ResourceGuard(runner.PreflightBudget(90, 4 * 1024**3, 1))


def test_complete_greedy_and_untrained_panels_are_separate_and_certified(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("preflight must not construct an optimizer"))
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("preflight must not backpropagate"))
    case, factory, panels, source = fixture(tmp_path)
    output = tmp_path / "run"
    result = runner.run_preflight(case, factory, panels.optimization, panels.selection, output,
                                  guard(), source_root=source)
    assert result["status"] == "completed"
    assert result["completed_selection_worlds"] == result["completed_untrained_policy_worlds"] == 2
    assert result["eligible_candidate_count"] == 2 and result["certified_complete_episodes"] == 5
    assert not result["final_worlds_used"] and not result["stress_worlds_used"]
    assert result["gradient_steps"] == 0
    initial = json.loads((output / "initial/inventory.json").read_text())
    assert initial["complete_inventory"]["status"] == "complete"
    assert initial["complete_inventory"]["batch"]["slot_count"] == 2
    records = json.loads((output / "candidate-records.json").read_text())["candidates"]
    greedy, policy = records["frozen_greedy_sequence"], records["untrained_raw_policy"]
    assert policy["seed"] == 11 and policy["input_profile"]["profile_id"] == "RAW"
    assert policy["optimizer_constructed"] is False
    assert greedy["selection_partition_hash"] == policy["selection_partition_hash"] == panels.selection.partition_hash
    assert json.loads((output / "cancellation-probe.json").read_text())["committed_state_unchanged"]
    assert all(row["history_hash"] == greedy["independent_history_hash"] for row in greedy["selection_worlds"])


@pytest.mark.parametrize("incomplete_stage", ["greedy", "selection-2", "untrained-selection-2"])
def test_unfinished_panels_preserve_denominator_and_never_export_candidate(tmp_path, monkeypatch, incomplete_stage):
    case, factory, panels, source = fixture(tmp_path)
    original = runner.run_episode
    def interrupted(sim, budget, folder, **kwargs):
        if folder.name == incomplete_stage:
            raise InterruptedError("controlled incomplete episode")
        return original(sim, budget, folder, **kwargs)
    monkeypatch.setattr(runner, "run_episode", interrupted)
    output = tmp_path / "failed"
    with pytest.raises(InterruptedError, match="incomplete episode"):
        runner.run_preflight(case, factory, panels.optimization, panels.selection,
            output, guard(), source_root=source)
    status = json.loads((output / "status.json").read_text())
    assert status["eligible_candidate_count"] == 0
    assert status["expected_selection_worlds"] == status["expected_untrained_policy_worlds"] == 2
    assert not list(output.glob("eligible-*.json"))
    assert (output / "failure-state.json").exists()
    if incomplete_stage == "selection-2":
        assert status["completed_selection_worlds"] == 1
    if incomplete_stage == "untrained-selection-2":
        assert status["completed_selection_worlds"] == 2 and status["completed_untrained_policy_worlds"] == 1


def test_after_commit_interruption_preserves_actual_cut_and_reward(tmp_path, monkeypatch):
    _, factory, _, _ = fixture(tmp_path)
    budget = guard()
    sim = factory(budget)
    original = sim.engine.commit_preview
    def commit_then_cancel(preview):
        original(preview)
        budget.reason = "controlled_after_commit"
    monkeypatch.setattr(sim.engine, "commit_preview", commit_then_cancel)
    with pytest.raises(CommittedTransitionInterrupted):
        runner.run_episode(sim, budget, tmp_path / "episode")
    failure = json.loads((tmp_path / "episode/failure.json").read_text())
    assert failure["interrupted_transition_committed"] is True
    assert failure["raw_state"]["committed_cut_count"] == 1
    assert failure["interrupted_reward"] == failure["raw_state"]["total_reward"]
    assert failure["raw_state"]["published_inventory_available"] is False
    assert not (tmp_path / "episode/episode.json").exists()


@pytest.mark.parametrize("attack", ["final", "stress", "seed_count", "overlap", "case", "planning", "generator"])
def test_world_role_and_binding_fail_before_rollouts(tmp_path, attack):
    case, _, panels, _ = fixture(tmp_path)
    selection = panels.selection
    if attack in {"final", "stress"}:
        selection = replace(selection, role=WorldRole.FINAL_EVALUATION if attack == "final" else WorldRole.STRESS)
    elif attack == "seed_count":
        selection = replace(selection, seeds=selection.seeds[:1])
    elif attack == "overlap":
        selection = replace(selection, seeds=(panels.optimization.seeds[0], selection.seeds[1]))
    elif attack == "case":
        selection = replace(selection, case_hash="other")
    elif attack == "planning":
        selection = replace(selection, planning_hash="other")
    else:
        selection = replace(selection, generator=WorldGeneratorConfig(translation_scale_mm=(1, 0, 0)))
    with pytest.raises(ValueError):
        runner.validate_panel(case, WorldGeneratorConfig(), panels.optimization, selection)


def test_memory_and_wall_budgets_latch_without_hiding_cancellation_overhead():
    clock = [10.]
    memory = [10]
    budget = runner.ResourceGuard(runner.PreflightBudget(5., 20, 1), clock=lambda: clock[0], memory=lambda: memory[0])
    assert not budget()
    memory[0] = 21
    assert budget() and budget.reason == "process_peak_rss_budget"
    memory[0] = 1
    assert budget()  # A later measurement cannot undo exceeded peak memory.
    clock[0] = 16
    with pytest.raises(InterruptedError):
        budget.require()
    receipt = budget.receipt()
    assert receipt["observed_peak_rss_bytes"] == 21 and receipt["cancellation_callback_calls"] == 4
    assert receipt["budget_request_to_receipt_seconds"] == 6
    second = runner.ResourceGuard(runner.PreflightBudget(5., 20, 1), clock=lambda: clock[0], memory=lambda: 1)
    clock[0] = 21
    assert second() and second.reason == "worker_wall_budget"


def test_source_mutation_rejects_completed_panel_and_no_eligible_export(tmp_path, monkeypatch):
    case, factory, panels, source = fixture(tmp_path)
    original = runner.independent_check_native_history
    def audit_and_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (source / "scripts/preflight_native_axis.py").write_text("# changed during run\n")
        return result
    monkeypatch.setattr(runner, "independent_check_native_history", audit_and_mutate)
    output = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="SOURCE_CHANGED"):
        runner.run_preflight(case, factory, panels.optimization, panels.selection, output, guard(), source_root=source)
    assert not list(output.glob("eligible-*.json"))


def test_coherently_resealed_budget_declaration_is_refused(tmp_path):
    value = json.loads((ROOT / runner.DECLARATION_PATH).read_text())
    value["resource_budget"]["worker_wall_seconds"] = 30
    value.pop("declaration_content_hash")
    value["declaration_content_hash"] = content_hash(value)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="committed declaration"):
        runner._checked_manifest(path, runner.DECLARATION_HASH)


def test_default_cli_only_writes_declaration_without_case_loading(tmp_path, monkeypatch):
    output = tmp_path / "declared"
    monkeypatch.setattr(sys, "argv", ["preflight", "--output", str(output)])
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No patient worker without --execute"))
    runner.main()
    status = json.loads((output / "launcher-status.json").read_text())
    assert status["status"] == "declared_not_executed" and status["gradient_steps"] == 0
    assert not (output / "profile").exists()
    with pytest.raises(FileExistsError):
        runner.main()


@pytest.mark.parametrize("changed", ["proposal_rule", "max_cuts", "max_actions_including_stop", "input_profile"])
def test_actual_public_defaults_must_match_pinned_action_model(tmp_path, changed):
    _, factory, _, _ = fixture(tmp_path)
    small = factory(guard())
    sim = AxisColumnNativeSimulator(small.native_config, max_steps=3)
    declaration = runner._checked_manifest(ROOT / runner.DECLARATION_PATH, runner.DECLARATION_HASH)
    reference = runner._checked_manifest(ROOT / runner.REFERENCE_PATH, runner.REFERENCE_HASH)
    runner.assert_public_action_model(sim, declaration, reference)
    if changed == "proposal_rule":
        declaration["action_model"][changed]["max_primary_rays"] = 2
    elif changed == "input_profile":
        declaration["action_model"][changed] = "FEATURE_UNITS"
    else:
        declaration["action_model"][changed] += 1
    with pytest.raises(ValueError, match="contract changed|differs from declaration"):
        runner.assert_public_action_model(sim, declaration, reference)
