"""Experiment isolation and independent replay checks, not performance tests."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.learning import TrainingConfig
from resectionlab.simulation import make_patient_simulator, make_synthetic_simulator

spec = importlib.util.spec_from_file_location(
    "run_patient_learning", Path(__file__).parents[1] / "scripts" / "run_patient_learning.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def stable_sources(tmp_path, monkeypatch):
    """Exercise real snapshot protection on private files during orchestration tests."""
    root = tmp_path / "private-source-fixture"
    source = root / "src/resectionlab"
    source.mkdir(parents=True)
    (source / "learning.py").write_text("# private provenance fixture, not executed\n")
    snapshot = runner.source_snapshot
    preserve = runner.preserve_source
    monkeypatch.setattr(runner, "source_snapshot", lambda *_: snapshot(root))
    monkeypatch.setattr(runner, "preserve_source",
                        lambda record, destination, *_: preserve(record, destination, root))
    return root


def test_source_snapshot_detects_uncommitted_file_edits(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    file = source / "uncommitted.py"
    file.write_text("answer = 1\n")
    snapshot = runner.source_snapshot(tmp_path)
    assert "src/uncommitted.py" in snapshot["file_sha256"]
    runner.preserve_source(snapshot, tmp_path / "archive", tmp_path)
    assert (tmp_path / "archive/src/uncommitted.py").read_text() == "answer = 1\n"
    file.write_text("answer = 2\n")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        runner.assert_source_unchanged(snapshot, tmp_path)


def test_independent_checker_certifies_connected_sequence_and_rejects_teleport():
    case = make_synthetic_simulator()
    actions = tuple(f"REMOVE:synthetic-cannula:1,1,{z}" for z in range(3)) + ("STOP",)
    candidate = runner.FrozenSequence("legal", case.case_hash, actions, "SEARCH")
    assert runner.independent_sequence_check(make_synthetic_simulator, candidate).feasible
    illegal = runner.FrozenSequence("teleport", case.case_hash, actions[2:], "SEARCH")
    assert not runner.independent_sequence_check(make_synthetic_simulator, illegal).feasible
    after_stop = runner.FrozenSequence("after-stop", case.case_hash, ("STOP",) + actions, "SEARCH")
    assert not runner.independent_sequence_check(make_synthetic_simulator, after_stop).feasible


def test_runner_persists_freeze_and_disjoint_evaluation(tmp_path, stable_sources):
    output = tmp_path / "run"
    result = runner.run_experiment(make_synthetic_simulator, output,
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1,
                              max_wall_seconds=10, checkpoint_interval=1),
        seeds=(11,), counts=(1, 1, 1, 1))
    assert result["status"] == "completed"
    manifest = json.loads((output / "manifest.json").read_text())
    freeze = json.loads((output / "candidate-freeze.json").read_text())
    evaluation_dir = output / result["evaluation_run_id"]
    evaluation = json.loads((evaluation_dir / "final_evaluation.json").read_text())
    assert result["run_id"] != result["evaluation_run_id"]
    assert evaluation["candidate_freeze_hash"] == freeze["fingerprint"]
    assert evaluation["independent_patient_count"] == 0
    assert evaluation["unit_of_independent_patient_inference"] == "synthetic_fixture"
    assert manifest["source"]["file_sha256"]["src/resectionlab/learning.py"]
    panels = manifest["world_partitions"]
    assert not set(panels["optimization"]["seeds"]) & set(panels["final_evaluation"]["seeds"])
    assert len(json.loads((evaluation_dir / "ledger.json").read_text())) == 2
    assert all(c["clinical_deficit_probability"] is None for c in evaluation["candidates"])
    assert all(c["geometry"]["feasible"] for c in evaluation["candidates"])
    timing = json.loads((evaluation_dir / "validation-timings.json").read_text())
    assert timing["full_validation_seconds"] > 0
    assert set(timing["partition_seconds"]) == {"final_evaluation", "stress"}
    with pytest.raises(FileExistsError):
        runner.run_experiment(make_synthetic_simulator, output, config=TrainingConfig())


def test_development_only_never_opens_final_evaluator(tmp_path, monkeypatch, stable_sources):
    def forbidden(*args, **kwargs):
        raise AssertionError("development-only screening must not evaluate final worlds")
    monkeypatch.setattr(runner, "evaluate_frozen_candidates", forbidden)
    result = runner.run_experiment(make_synthetic_simulator, tmp_path / "dev",
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1, max_wall_seconds=10),
        seeds=(11,), counts=(1, 1, 1, 1), evaluate=False)
    assert result["status"] == "completed"
    assert result["evaluation_status"] == "not_requested_development_selection_only"
    assert not list((tmp_path / "dev").glob("evaluation-*"))


def test_ppo_runner_records_distinct_algorithm_and_actual_adam_steps(tmp_path, monkeypatch, stable_sources):
    from resectionlab.learning_ppo import PPOConfig, train_patient_ppo
    def forbidden(*args, **kwargs):
        raise AssertionError("PPO development run must not inspect final worlds")
    monkeypatch.setattr(runner, "evaluate_frozen_candidates", forbidden)
    output = tmp_path / "ppo"
    runner.run_experiment(make_synthetic_simulator, output,
        config=PPOConfig(max_environment_steps=32, max_gradient_steps=4, max_wall_seconds=10),
        trainer=train_patient_ppo, seeds=(11,), counts=(1, 1, 1, 1), evaluate=False)
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "scratch-11/result.json").read_text())
    assert manifest["trainer"] == "resectionlab.learning_ppo.train_patient_ppo"
    assert report["algorithm"] == "masked_clipped_ppo_gae_v1"
    assert report["gradient_steps"] == 4
    assert report["rollout_batches"] == 1
    assert report["optimization_sample_presentations"] == report["optimization_environment_steps"] * 4


def test_native_footprint_failure_invalidates_coarse_pass(tmp_path, stable_sources):
    mask = np.ones((6, 6, 6), dtype=bool)
    case = CaseData(case_id="native-audit-fixture", mri=mask.astype(np.float32),
                    compartments={"enhancing": mask}, affine=np.eye(4), brain_mask=mask,
                    source_refs=(SourceRef("fixture", "synthetic://native-audit", provenance="simulated"),))
    template = make_patient_simulator(case, block_size=3, max_steps=2, max_actions=4)
    output = tmp_path / "native-audit"
    result = runner.run_experiment(template.clone, output,
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1, max_wall_seconds=10),
        seeds=(11,), counts=(1, 1, 1, 1), source_case=case)
    assert result["status"] == "invalidated"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preoperative_planning_hash"] == case.planning_hash
    assert manifest["world_partitions"]["optimization"]["planning_hash"] == case.planning_hash
    evaluation_dir = output / result["evaluation_run_id"]
    native = json.loads((evaluation_dir / "native-removal-audit.json").read_text())
    coarse = json.loads((evaluation_dir / "coarse-sequence-audit.json").read_text())
    assert native["required"] is True
    assert native["candidates"]["STOP"]["feasible"] is True
    assert native["candidates"]["SEARCH"]["feasible"] is False
    assert native["candidates"]["SEARCH"]["unsupported_source_tissue_volume_mm3"] > 0
    assert coarse["SEARCH"]["feasible"] is True
    evaluation = json.loads((evaluation_dir / "final_evaluation.json").read_text())
    search = next(candidate for candidate in evaluation["candidates"] if candidate["plan_id"] == "SEARCH")
    assert search["status"] == "rejected_geometry"
    assert search["world_outcomes"] == []


def test_search_and_rl_share_nonstop_horizon(tmp_path, stable_sources):
    output = tmp_path / "short-horizon"
    runner.run_experiment(make_synthetic_simulator, output,
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1,
                              max_wall_seconds=10, max_episode_steps=2),
        seeds=(11,), counts=(1, 1, 1, 1), evaluate=False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["requested_simulator_nonstop_horizon"] == 4
    assert manifest["effective_shared_nonstop_horizon"] == 1
    frozen = json.loads((output / "candidate-freeze.json").read_text())
    assert all(sum(action != "STOP" for action in candidate["actions"]) <= 1
               for candidate in frozen["candidates"])


def test_unknown_adapter_cannot_silently_become_coarse(tmp_path, stable_sources):
    from resectionlab.simulation import SequentialSimulator
    class UnregisteredAdapter(SequentialSimulator):
        pass
    output = tmp_path / "unsupported-adapter"
    with pytest.raises(TypeError, match="must not reconstruct"):
        runner.run_experiment(lambda: UnregisteredAdapter(make_synthetic_simulator().config),
            output, config=TrainingConfig(), seeds=(11,), evaluate=False)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "training.json").exists()


def test_frozen_population_inference_rejects_final_worlds_and_incomplete_panels():
    from resectionlab.learning import MaskedPatientPolicy, policy_hash
    from resectionlab.worlds import generate_partitions
    simulator = make_synthetic_simulator()
    observation = simulator.observation()
    policy = MaskedPatientPolicy(observation.action_features.shape[1], observation.state_features.size, 16)
    partitions = generate_partitions(simulator.case_hash, simulator.config.world_generator,
        42, optimization=1, selection=2, final_evaluation=1, stress=1)
    config = TrainingConfig(hidden_features=16, max_wall_seconds=10)
    with pytest.raises(ValueError, match="selection worlds only"):
        runner.frozen_population_rollouts(make_synthetic_simulator, policy,
            partitions.final_evaluation, config, cancelled=lambda: False)
    original = policy_hash(policy)
    interrupted, actions = runner.frozen_population_rollouts(make_synthetic_simulator,
        policy, partitions.selection, config, cancelled=lambda: True)
    assert interrupted["status"] == "cancelled"
    assert interrupted["selection_return"] is None
    assert interrupted["gradient_steps"] == interrupted["optimization_environment_steps"] == 0
    assert actions == ()
    complete, actions = runner.frozen_population_rollouts(make_synthetic_simulator,
        policy, partitions.selection, config, cancelled=lambda: False)
    assert complete["selection_panel_complete"]
    assert len(complete["selection_returns"]) == 2
    assert complete["selection_environment_steps"] > 0
    assert actions
    assert policy_hash(policy) == original


@pytest.fixture
def analytic_population(tmp_path):
    from resectionlab.population_learning import make_analytic_population_fixture, train_population_policy
    members, target_factory, target = make_analytic_population_fixture()
    result = train_population_policy(members, exclusions=(target,),
        config=TrainingConfig(hidden_features=16, max_gradient_steps=4,
            max_environment_steps=64, max_wall_seconds=15, episodes_per_update=2,
            max_episode_steps=8, checkpoint_interval=2), output_dir=tmp_path / "population")
    return Path(result["checkpoint_path"]), members, target_factory, target


def test_population_runner_keeps_all_arms_isolated_and_final_worlds_closed(tmp_path, stable_sources, analytic_population, monkeypatch):
    import hashlib
    checkpoint, members, target_factory, target = analytic_population
    original_bytes = checkpoint.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic population development must not inspect final worlds")
    monkeypatch.setattr(runner, "evaluate_frozen_candidates", forbidden)
    output = tmp_path / "all-arms"
    result = runner.run_experiment(target_factory, output,
        config=TrainingConfig(hidden_features=16, max_gradient_steps=4,
            max_environment_steps=64, max_wall_seconds=15, episodes_per_update=2,
            max_episode_steps=8, checkpoint_interval=2),
        seeds=(11, 23, 47), counts=(2, 2, 2, 2), evaluate=False,
        population_checkpoint=checkpoint, population_case_group=target.group_id,
        population_case_aliases=target.aliases)
    assert result["status"] == "completed"
    assert checkpoint.read_bytes() == original_bytes
    assert (output / "population-source.pt").read_bytes() == original_bytes
    frozen = json.loads((output / "population-frozen.json").read_text())
    assert frozen["gradient_steps"] == frozen["optimization_environment_steps"] == 0
    assert frozen["selection_panel_complete"]
    provenance = json.loads((output / "population-provenance.json").read_text())
    assert provenance["checkpoint_file_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    for seed in (11, 23, 47):
        scratch = json.loads((output / f"scratch-{seed}/contract.json").read_text())
        adapted = json.loads((output / f"adapted-{seed}/contract.json").read_text())
        assert scratch["config"] == adapted["config"]
        assert scratch["partitions"] == adapted["partitions"]
        assert scratch["shared_checkpoint_hash"] is None
        assert adapted["initial_checkpoint_hash"] == frozen["shared_checkpoint_hash"]
        report = json.loads((output / f"adapted-{seed}/result.json").read_text())
        assert report["optimizer_mode"] == "POPULATION_ADAPTED"
        assert 0 < report["gradient_steps"] <= 4
    freeze = json.loads((output / "candidate-freeze.json").read_text())
    modes = {candidate["optimizer_mode"] for candidate in freeze["candidates"]}
    assert {"SEARCH", "PATIENT_SCRATCH_RL", "POPULATION_FROZEN", "POPULATION_ADAPTED"} <= modes
    audit = output / result["geometry_validation_run_id"]
    assert not (audit / "ledger.json").exists()
    assert not (audit / "final_evaluation.json").exists()


def test_population_overlap_fails_before_any_online_optimization(tmp_path, stable_sources, analytic_population):
    checkpoint, members, target_factory, target = analytic_population
    member = members[0]
    output = tmp_path / "overlap"
    with pytest.raises(ValueError, match="overlap|development|excluded"):
        runner.run_experiment(member.simulator_factory, output,
            config=TrainingConfig(hidden_features=16), seeds=(11,), evaluate=False,
            population_checkpoint=checkpoint,
            population_case_group=member.identity.group_id,
            population_case_aliases=member.identity.aliases)
    assert not (output / "search.json").exists()
    assert not (output / "training.json").exists()


@pytest.mark.parametrize("evaluate", [False, True])
def test_native_backend_preserved_through_horizon_training_audit_and_worlds(tmp_path, stable_sources, monkeypatch, evaluate):
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_resection import NativeResectionConfig, NATIVE_GENERIC_TOOLS
    from resectionlab.native_simulation import NativeSequentialSimulator
    tissue = np.ones((5, 5, 3), bool)
    case = CaseData(case_id="native-runner", mri=tissue.astype(np.float32),
        compartments={"enhancing": tissue}, affine=np.eye(4), brain_mask=tissue,
        source_refs=(SourceRef("fixture", "synthetic://native-runner", provenance="simulated"),))
    native = NativeResectionConfig(tissue, tissue.astype(np.int16), np.eye(4),
        AccessWindow((2, 2, -.5), (0, 0, 1), 3.), (NATIVE_GENERIC_TOOLS[0],),
        case.semantic_hash, "synthetic solid cube", case_id=case.case_id)
    def factory():
        return NativeSequentialSimulator(native, [(2, 2, 1)], max_steps=3, max_actions=2)
    if not evaluate:
        def forbidden(*args, **kwargs):
            raise AssertionError("Native development geometry must not inspect final worlds")
        monkeypatch.setattr(runner, "evaluate_frozen_candidates", forbidden)
    output = tmp_path / "native-run"
    result = runner.run_experiment(factory, output,
        config=TrainingConfig(max_environment_steps=8, max_gradient_steps=1,
            max_wall_seconds=10, max_episode_steps=2, episodes_per_update=1),
        seeds=(11,), counts=(1, 1, 1, 1), source_case=case, evaluate=evaluate)
    assert result["status"] == "completed"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["simulator_backend"] == "NativeSequentialSimulator"
    assert manifest["effective_shared_nonstop_horizon"] == 1
    frozen = json.loads((output / "candidate-freeze.json").read_text())
    assert any(c["plan_id"] == "INITIAL:11" for c in frozen["candidates"])
    assert all(action == "STOP" or action.startswith("NATIVE:")
               for c in frozen["candidates"] for action in c["actions"])
    audit_dir = output / result["evaluation_run_id" if evaluate else "geometry_validation_run_id"]
    audits = json.loads((audit_dir / "native-history-audit.json").read_text())
    assert all(audit["feasible"] and audit["complete_tool_checked"] for audit in audits.values())
    assert audits["SEARCH"]["shared_audit_reused"]
    assert audits["SEARCH"]["audit_key"] == audits["GREEDY"]["audit_key"]
    assert not json.loads((audit_dir / "coarse-sequence-audit.json").read_text())
    if evaluate:
        for role in ("final_evaluation", "stress"):
            replay = json.loads((audit_dir / f"{role}-replay.json").read_text())
            assert all(item["simulation_version"].startswith("native-") for item in replay)
            report = json.loads((audit_dir / f"{role}.json").read_text())
            assert report["geometry_scope"] == "independent_native_history_complete_tool_and_source_cell_removal"
    else:
        assert not (audit_dir / "ledger.json").exists()
        assert not (audit_dir / "final_evaluation.json").exists()
