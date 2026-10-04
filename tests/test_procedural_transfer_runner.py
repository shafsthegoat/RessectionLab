"""Counterfactual declaration/source/action-set checks; no registered training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.core import array_digest
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.native_simulation import NativeSequentialSimulator
from resectionlab.worlds import content_hash, generate_partitions

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("run_procedural_transfer", ROOT / "scripts/run_procedural_transfer.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def native_fixture():
    tissue = np.ones((7, 7, 6), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 3:] = 1
    cfg = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 3.), NATIVE_GENERIC_TOOLS,
        "source-fixture", "procedural test cube", case_id="synthetic_native_contract_test")
    sim = NativeSequentialSimulator(cfg, [(3, 3, 4), (2, 3, 4)], max_steps=3, max_actions=7)
    case = SimpleNamespace(case_id=cfg.case_id, semantic_hash=cfg.source_hash,
        planning_hash="planning-fixture", mri=tissue, affine=np.eye(4), frame="RAS+",
        compartments={"target": labels > 0})
    panels = generate_partitions(case.semantic_hash, sim.config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=case.planning_hash)
    target = {"case_id": case.case_id, "semantic_hash": case.semantic_hash,
        "planning_hash": case.planning_hash, "source_shape": list(tissue.shape),
        "source_affine": case.affine.tolist(), "source_frame": case.frame,
        "compartment_mask_hashes": {"target": array_digest(labels > 0)},
        "world_partitions": panels.to_dict(), **runner.model_inventory(sim)}
    declaration = {"target": target, "selection_and_evaluation": {"common_world_master_seed": 20261004}}
    return case, sim, declaration


def test_committed_declaration_refuses_budget_rewrite_even_after_reseal(tmp_path):
    declaration = runner.load_declaration(ROOT / runner.DECLARATION_PATH)
    declaration["budgets"]["online_scratch_and_adapted_each_seed"]["max_wall_seconds"] = 60
    payload = dict(declaration)
    payload.pop("declaration_content_hash")
    declaration["declaration_content_hash"] = content_hash(payload)
    altered = tmp_path / "resealed.json"
    altered.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="committed preregistration"):
        runner.load_declaration(altered)
    # The public comparison API also refuses a resealed mapping before even
    # constructing a target, loading a checkpoint or importing its trainer.
    destination = tmp_path / "refused"
    with pytest.raises(ValueError, match="committed preregistration"):
        runner.run_target_comparison(None, None, None, tmp_path / "absent.pt", destination, declaration)
    status = json.loads((destination / "status.json").read_text())
    assert status["status"] == "failed"
    assert len(status["unfinished_learning_runs"]) == 6
    assert not (destination / "training.json").exists()


def test_incomplete_first_selection_cannot_be_ranked_but_evaluated_stop_can():
    for missing in (None, np.nan, np.inf):
        with pytest.raises(RuntimeError, match="no completed selection panel"):
            runner.require_complete_selection(SimpleNamespace(selected_selection_return=missing), [], 2)
    stop = SimpleNamespace(selected_selection_return=0., selected_checkpoint_hash="initial")
    complete = {"mean_return": 0., "checkpoint_hash": "initial", "world_count": 2}
    runner.require_complete_selection(stop, [complete], 2)
    for changed in ({}, {**complete, "checkpoint_hash": "different"},
                    {**complete, "world_count": 1}, {**complete, "mean_return": 1.}):
        with pytest.raises(RuntimeError, match="matching complete selection-history"):
            runner.require_complete_selection(stop, [changed], 2)


def test_available_public_bundle_matches_declared_inventory_without_stepping():
    from dataclasses import replace
    from resectionlab.imaging import load_case
    from resectionlab.learning import TrainingConfig
    from resectionlab.procedural_learning import TransferTarget, validate_procedural_target_worlds
    declaration = runner.load_declaration(ROOT / runner.DECLARATION_PATH)
    bundle = ROOT / declaration["target"]["bundle_path"]
    if not bundle.is_file():
        pytest.skip("Pinned public development bundle is not installed")
    case = load_case(bundle)
    sim = runner.make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7)
    panels = runner.assert_declared_target(case, sim, declaration)
    identity = declaration["target"]
    target = TransferTarget(case.semantic_hash, case.planning_hash, identity["group_id"],
        tuple(identity["aliases"]), identity["source_kind"])
    settings = TrainingConfig(seed=11, **declaration["budgets"]["online_scratch_and_adapted_each_seed"])
    receipt = validate_procedural_target_worlds(target, sim, panels.optimization, panels.selection, settings)
    assert receipt["final_worlds_used"] is False
    with pytest.raises(ValueError, match="(?i)(world|partition|planning)"):
        validate_procedural_target_worlds(target, sim,
            replace(panels.optimization, planning_hash="postoperative-replacement"), panels.selection, settings)
    assert not sim.removed_mask.any()
    assert sim.metrics()["history"] == []


@pytest.mark.parametrize("field", ["candidate_entries_mm", "tools", "evidence_available", "max_steps"])
def test_target_binding_refuses_action_geometry_or_evidence_drift(field):
    case, sim, declaration = native_fixture()
    runner.assert_declared_target(case, sim, declaration)
    corrupted = copy.deepcopy(declaration)
    if field == "candidate_entries_mm":
        corrupted["target"][field][0][0] += .1
    elif field == "tools":
        corrupted["target"][field][0]["shaft_radius_mm"] += .1
    elif field == "evidence_available":
        corrupted["target"][field][0] = True
    else:
        corrupted["target"][field] += 1
    with pytest.raises(ValueError, match=field):
        runner.assert_declared_target(case, sim, corrupted)


def test_seed_identity_and_all_partition_roles_are_bound():
    case, sim, declaration = native_fixture()
    changed = copy.deepcopy(declaration)
    changed["target"]["world_partitions"]["selection"]["seeds"][0] += 1
    with pytest.raises(ValueError, match="world partitions"):
        runner.assert_declared_target(case, sim, changed)
    case.planning_hash = "postoperative-replacement"
    with pytest.raises(ValueError, match="planning_hash"):
        runner.assert_declared_target(case, sim, declaration)


def test_procedural_preflight_audits_paid_cut_and_refuses_partition_substitution(tmp_path):
    case, sim, declaration = native_fixture()
    panels = runner.assert_declared_target(case, sim, declaration)
    member = SimpleNamespace(family_id="procedural:test-only", aliases=("procedural:test-alias",),
        factory=sim.clone, optimization=panels.optimization, selection=panels.selection)
    declared_member = {"group_id": member.family_id, "aliases": member.aliases,
        "source_hash": sim.case_hash, "model_hash": sim.decision_model_hash,
        "native_config_hash": sim.native_config.fingerprint,
        "initial_action_ids": list(sim.observation().action_ids), "world_partitions": panels.to_dict()}
    fixture_declaration = {"procedural_training": {"members": [declared_member]}}
    records = runner.procedural_preflight((member,), fixture_declaration, tmp_path, lambda: False)
    assert records[0]["audit"]["feasible"]
    assert records[0]["normal_removed_mm3"] > 0
    assert records[0]["gradient_steps"] == 0
    assert not sim.removed_mask.any()
    member.selection = panels.final_evaluation
    with pytest.raises(ValueError, match="procedural selection worlds"):
        runner.procedural_preflight((member,), fixture_declaration, tmp_path, lambda: False)


def test_exact_runner_bytes_are_part_of_runtime_freeze(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    file = scripts / "run_procedural_transfer.py"
    file.write_text("# before\n")
    snapshot = runner.transfer_source_snapshot(tmp_path)
    assert "scripts/run_procedural_transfer.py" in snapshot["numerical_runtime_sha256"]
    file.write_text("# modified during search\n")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        runner.assert_transfer_source_unchanged(snapshot, tmp_path)


def test_reused_failed_audit_keeps_every_candidate_in_denominator(tmp_path, monkeypatch):
    case, sim, _ = native_fixture()
    calls = []
    def rejected(*args):
        calls.append(args)
        return {"feasible": False, "failures": ["test_unsupported_native_removal"]}, {"removed": 0}
    monkeypatch.setattr(runner, "native_audit", rejected)
    candidates = [runner.FrozenSequence(name, sim.case_hash, ("STOP",), name, name)
                  for name in ("SEARCH", "SCRATCH:11", "ADAPTED:11")]
    result = runner.validate_frozen_candidates(sim, case, candidates, 1, tmp_path, lambda: False)
    assert len(calls) == result["unique_audits"] == 1
    assert result["candidate_count"] == 3
    assert result["rejected_candidate_ids"] == ["SEARCH", "SCRATCH:11", "ADAPTED:11"]
    audits = json.loads((tmp_path / "native-history-audit.json").read_text())
    assert not audits["SEARCH"]["shared_audit_reused"]
    assert audits["ADAPTED:11"]["shared_audit_reused"]
    assert result["full_validation_seconds"] > 0
    assert not result["final_worlds_used"]


def test_all_arm_orchestration_uses_explicit_nonpatient_test_scope(tmp_path, monkeypatch):
    """Exercise real trainers/audits on tiny geometry, never the registered case.

    Pytest-only validators substitute this fixture for the fixed study. No
    production flag or checkpoint accepts these substitutions on a patient.
    """
    from dataclasses import asdict
    from resectionlab.learning import TrainingConfig
    from resectionlab.procedural_learning import (TEST_SCOPE, make_procedural_test_target,
        make_native_procedural_fixture, train_procedural_native_policy)
    target, factory = make_procedural_test_target()
    members = make_native_procedural_fixture(target)
    config = TrainingConfig(hidden_features=16, max_gradient_steps=4, max_environment_steps=64,
        max_wall_seconds=15, max_episode_steps=4, episodes_per_update=2, checkpoint_interval=2)
    shared = train_procedural_native_policy(members, excluded_targets=(target,), config=config,
                                            output_dir=tmp_path / "pretraining")
    assert shared["scope"] == TEST_SCOPE
    base = factory()
    case = SimpleNamespace(mri=base.native_config.tissue_mask, affine=base.native_config.affine,
                          frame="RAS+", semantic_hash=base.case_hash)
    panels = generate_partitions(base.case_hash, base.config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    declaration = runner.load_declaration(ROOT / runner.DECLARATION_PATH)
    settings = asdict(config)
    settings.pop("seed")
    settings.update(max_gradient_steps=1, max_environment_steps=16, checkpoint_interval=1)
    declaration["budgets"]["online_scratch_and_adapted_each_seed"] = settings
    declaration["budgets"]["search"] = {"beam_width": 2, "max_expansions": 16, "max_wall_seconds": 15}
    monkeypatch.setattr(runner, "SCOPE", TEST_SCOPE)
    monkeypatch.setattr(runner, "assert_declaration", lambda value: None)
    monkeypatch.setattr(runner, "assert_declared_sources", lambda value: None)
    monkeypatch.setattr(runner, "assert_declared_target", lambda *args: panels)
    output = tmp_path / "comparison"
    result = runner.run_target_comparison(base, case, target, Path(shared["checkpoint_path"]), output, declaration)
    assert result["status"] == "completed"
    assert result["scope"] == TEST_SCOPE
    assert len(result["completed_learning_runs"]) == 6
    records = json.loads((output / "training.json").read_text())
    assert len(records) == 6
    assert all(row["gradient_steps"] == 1 for row in records)
    assert all(row["optimization_environment_steps"] <= 16 for row in records)
    assert all(row["algorithm"] == "masked_reinforce_state_value_v2" for row in records)
    adapted = [row for row in records if row["optimizer_mode"] == runner.ADAPTED]
    assert len(adapted) == 3
    assert all(row["initial_checkpoint_hash"] == shared["policy_hash"] for row in adapted)
    frozen = json.loads((output / "procedural-frozen.json").read_text())
    assert frozen["gradient_steps"] == frozen["optimization_environment_steps"] == 0
    assert frozen["selection_panel_complete"]
    assert result["validation"]["candidate_count"] == 13
    assert result["validation"]["rejected_candidate_ids"] == []
    assert not result["final_worlds_used"]
    assert not list(output.rglob("final_evaluation.json"))
    assert not list(output.rglob("stress.json"))
    assert not list(output.rglob("ledger.json"))
