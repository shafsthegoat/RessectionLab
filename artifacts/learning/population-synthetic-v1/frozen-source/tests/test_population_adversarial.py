"""Independent leakage, immutability, and cost-boundary checks for shared actors."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from resectionlab import learning
from resectionlab.learning import TrainingConfig, load_policy, policy_hash, train_patient_policy
from resectionlab.population_learning import (PopulationIdentity, _cohort_record, feature_schema,
    load_frozen_population_policy, make_analytic_population_fixture, train_population_policy,
    validate_population_checkpoint)
from resectionlab.simulation import SequentialSimulator
from resectionlab.worlds import WorldRole, generate_partitions

spec = importlib.util.spec_from_file_location("population_audit_runner", Path(__file__).parents[1] / "scripts/run_patient_learning.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def small_config(**changes):
    return replace(TrainingConfig(hidden_features=16, max_gradient_steps=4,
        max_environment_steps=128, max_wall_seconds=30., max_episode_steps=8,
        episodes_per_update=2, checkpoint_interval=2, seed=11), **changes)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    members, target_factory, target = make_analytic_population_fixture()
    folder = tmp_path_factory.mktemp("population-audit") / "pretraining"
    trained = train_population_policy(members, exclusions=(target,), config=small_config(), output_dir=folder)
    assert trained["status"] == "completed"
    assert trained["provenance"]["training_gradient_steps"] == 4
    return members, target_factory, target, Path(trained["checkpoint_path"]), trained


def validation(factory, identity):
    simulator = factory()
    observation = simulator.observation()
    return {"target_case_hash": identity.case_hash, "target_group": identity.group_id,
        "target_aliases": identity.aliases,
        "expected_dimensions": (observation.action_features.shape[1], observation.state_features.size, 16),
        "expected_feature_schema": feature_schema(simulator)}


@pytest.mark.parametrize("target_alias", ["synthetic:patient-a", "dataset:visit-a"])
def test_group_and_alias_names_connect_transitively_before_split_check(target_alias):
    development = (
        PopulationIdentity("synthetic-case-a", "synthetic:patient-a", ("dataset:visit-a",)),
        PopulationIdentity("synthetic-case-b", "synthetic:patient-b"),
    )
    bridge = PopulationIdentity("synthetic-unscored-visit", "synthetic:bridge", (target_alias, "dataset:later-visit"), "excluded")
    target = PopulationIdentity("synthetic-heldout", "synthetic:heldout", ("dataset:later-visit",), "final_evaluation")
    with pytest.raises(ValueError, match="overlaps"):
        _cohort_record(development, (bridge, target))


def test_identity_snapshots_alias_input_before_caller_mutation():
    aliases = ["source:preoperative"]
    identity = PopulationIdentity("synthetic-example", "synthetic:example", aliases)
    before = identity.tokens
    aliases.append("source:postoperative")
    assert identity.aliases == ("source:preoperative",)
    assert identity.tokens == before


@pytest.mark.parametrize("role", [WorldRole.FINAL_EVALUATION, WorldRole.STRESS])
def test_shared_gradient_pool_refuses_final_or_stress_selection_worlds(tmp_path, role):
    members, _, target = make_analytic_population_fixture()
    altered = (replace(members[0], selection_manifest=replace(members[0].selection_manifest, role=role)), members[1])
    with pytest.raises(ValueError, match="expected selection"):
        train_population_policy(altered, exclusions=(target,), config=small_config(), output_dir=tmp_path / "forbidden")
    assert not (tmp_path / "forbidden/population.pt").exists()


def test_population_feature_order_and_undeclared_aliases_cannot_be_silently_accepted(shared):
    _, factory, target, checkpoint, _ = shared
    expected = validation(factory, target)
    expected["expected_feature_schema"]["action_feature_names"].reverse()
    with pytest.raises(ValueError, match="feature semantics"):
        validate_population_checkpoint(checkpoint, **expected)
    expected = validation(factory, target)
    expected["target_aliases"] = ()
    with pytest.raises(ValueError, match="predeclared excluded"):
        validate_population_checkpoint(checkpoint, **expected)


def test_frozen_arm_cannot_construct_optimizer_or_create_gradients(shared, monkeypatch):
    _, factory, target, checkpoint, trained = shared
    policy = load_frozen_population_policy(checkpoint, **validation(factory, target))
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in policy.parameters())
    original_hash = policy_hash(policy)
    original_bytes = checkpoint.read_bytes()
    simulator = factory()
    partitions = generate_partitions(simulator.case_hash, simulator.config.world_generator, 101,
        optimization=2, selection=2, final_evaluation=2, stress=2)

    def forbidden_optimizer(*_args, **_kwargs):
        raise AssertionError("Frozen inference attempted to construct an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden_optimizer)
    report, actions = runner.frozen_population_rollouts(factory, policy, partitions.selection,
        small_config(), cancelled=lambda: False)
    assert report["gradient_steps"] == report["optimization_environment_steps"] == 0
    assert report["selection_panel_complete"] is True
    assert report["selection_environment_steps"] >= len(partitions.selection.seeds)
    assert actions and actions[-1] == "STOP"
    assert report["pretraining_cost_accounted_separately"] is True
    assert trained["provenance"]["offline_pretraining"]["gradient_steps"] == 4
    assert policy_hash(policy) == original_hash
    assert checkpoint.read_bytes() == original_bytes
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_frozen_selection_refuses_wrong_world_binding_and_changing_models(shared):
    _, factory, target, checkpoint, _ = shared
    policy = load_frozen_population_policy(checkpoint, **validation(factory, target))
    simulator = factory()
    partitions = generate_partitions(simulator.case_hash, simulator.config.world_generator, 101,
        optimization=2, selection=2, final_evaluation=2, stress=2)
    wrong_case = replace(partitions.selection, case_hash="synthetic-another-case")
    with pytest.raises(ValueError, match="case"):
        runner.frozen_population_rollouts(factory, policy, wrong_case, small_config(), cancelled=lambda: False)
    calls = {"count": 0}

    def changing_factory():
        calls["count"] += 1
        base = factory()
        if calls["count"] > 1:
            return SequentialSimulator(replace(base.config, reward=replace(base.config.reward, action_cost=.5)))
        return base

    with pytest.raises((ValueError, RuntimeError), match="model"):
        runner.frozen_population_rollouts(changing_factory, policy, partitions.selection, small_config(), cancelled=lambda: False)


def test_adaptation_copies_exact_shared_weights_with_fresh_optimizer_and_full_online_limits(shared, tmp_path):
    _, factory, target, checkpoint, trained = shared
    original_bytes = checkpoint.read_bytes()
    simulator = factory()
    partitions = generate_partitions(simulator.case_hash, simulator.config.world_generator, 202,
        optimization=3, selection=2, final_evaluation=2, stress=2)
    config = small_config(max_gradient_steps=1, checkpoint_interval=1, seed=19)
    shared_args = {"shared_checkpoint": checkpoint, "population_case_group": target.group_id,
                   "population_case_aliases": target.aliases}
    untouched = train_patient_policy(factory, partitions.optimization, partitions.selection,
        config=config, output_dir=tmp_path / "cancelled", cancelled=lambda: True, **shared_args)
    untouched_checkpoint = torch.load(tmp_path / "cancelled/checkpoint.pt", weights_only=True)
    assert untouched.gradient_steps == 0
    assert untouched_checkpoint["optimizer"]["state"] == {}, "Offline Adam moments leaked into patient adaptation"
    assert untouched.initial_checkpoint_hash == trained["policy_hash"]
    adapted = train_patient_policy(factory, partitions.optimization, partitions.selection,
        config=config, output_dir=tmp_path / "adapted", **shared_args)
    scratch = train_patient_policy(factory, partitions.optimization, partitions.selection,
        config=config, output_dir=tmp_path / "scratch")
    assert adapted.initial_checkpoint_hash == adapted.shared_checkpoint_hash == trained["policy_hash"]
    assert policy_hash(load_policy(tmp_path / "adapted/initial.pt")) == trained["policy_hash"]
    assert adapted.gradient_steps == scratch.gradient_steps == config.max_gradient_steps
    for name in ("adapted", "scratch"):
        record = json.loads((tmp_path / name / "contract.json").read_text())
        assert record["config"] == json.loads((tmp_path / "scratch/contract.json").read_text())["config"]
        assert record["partitions"] == json.loads((tmp_path / "scratch/contract.json").read_text())["partitions"]
    latest = torch.load(tmp_path / "adapted/checkpoint.pt", weights_only=True)
    assert latest["optimizer"]["state"]
    assert all(float(state["step"]) == 1 for state in latest["optimizer"]["state"].values())
    assert checkpoint.read_bytes() == original_bytes
    assert hashlib.sha256((tmp_path / "adapted/population-source.pt").read_bytes()).digest() == hashlib.sha256(original_bytes).digest()


def test_numerical_source_change_during_shared_updates_withholds_checkpoint(tmp_path, monkeypatch):
    members, _, target = make_analytic_population_fixture()
    original = learning.numerical_source_hashes
    state = {"changed": False}

    def hashes():
        record = original()
        if state["changed"]:
            record["geometry.py"] = "0" * 64
        return record

    monkeypatch.setattr(learning, "numerical_source_hashes", hashes)
    with pytest.raises(ValueError, match="source changed during training"):
        train_population_policy(members, exclusions=(target,), config=small_config(),
            output_dir=tmp_path / "source-mutated", progress=lambda _: state.update(changed=True))
    assert not (tmp_path / "source-mutated/population.pt").exists()


def test_interrupted_frozen_selection_never_retains_a_partial_panel_candidate(shared):
    _, factory, target, checkpoint, _ = shared
    policy = load_frozen_population_policy(checkpoint, **validation(factory, target))
    simulator = factory()
    partitions = generate_partitions(simulator.case_hash, simulator.config.world_generator, 101,
        optimization=2, selection=2, final_evaluation=2, stress=2)
    calls = {"count": 0}

    def stop_after_first_world():
        calls["count"] += 1
        return factory()

    report, actions = runner.frozen_population_rollouts(stop_after_first_world, policy,
        partitions.selection, small_config(), cancelled=lambda: calls["count"] >= 2)
    assert len(report["selection_returns"]) == 1
    assert report["selection_environment_steps"] > 0
    assert report["selection_panel_complete"] is False
    assert report["selection_return"] is None
    assert actions == ()


def test_clinical_patient_identity_cannot_be_relabeled_as_synthetic_by_zero_patient_count(shared):
    _, factory, target, checkpoint, _ = shared
    expected = validation(factory, target)
    expected["target_case_hash"] = "sha256:" + "a" * 64
    expected["target_group"] = "synthetic:misdeclared-clinical-case"
    with pytest.raises(ValueError, match="explicit synthetic"):
        validate_population_checkpoint(checkpoint, **expected)


@pytest.mark.parametrize("contamination", ["final_used", "final_selection_rule", "member_final_panel", "member_world_overlap"])
def test_supplied_checkpoint_declaring_contaminated_selection_cannot_be_resealed(shared, tmp_path, contamination):
    from resectionlab.worlds import content_hash
    _, factory, target, checkpoint, _ = shared
    copied = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = copied["population_provenance"]
    if contamination == "final_used":
        provenance["final_worlds_used"] = True
    elif contamination == "final_selection_rule":
        provenance["checkpoint_rule"] = "highest_final_evaluation_return"
    elif contamination == "member_final_panel":
        provenance["members"][0]["partitions"]["selection"]["role"] = "final_evaluation"
    else:
        panels = provenance["members"][0]["partitions"]
        panels["selection"]["seeds"][0] = panels["optimization"]["seeds"][0]
    if contamination in {"member_final_panel", "member_world_overlap"}:
        from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest
        selected = provenance["members"][0]["partitions"]["selection"]
        selected["partition_hash"] = WorldPartitionManifest(WorldRole(selected["role"]),
            selected["case_hash"], WorldGeneratorConfig(**selected["generator"]),
            tuple(selected["seeds"]), selected.get("planning_hash")).partition_hash
    copied["provenance_hash"] = content_hash(provenance)
    altered = tmp_path / "declared-contamination.pt"
    torch.save(copied, altered)
    with pytest.raises(ValueError):
        validate_population_checkpoint(altered, **validation(factory, target))


@pytest.mark.parametrize("counter", ["elapsed_seconds", "gradient_steps"])
def test_resealed_population_checkpoint_cannot_understate_declared_offline_cost(shared, tmp_path, counter):
    from resectionlab.worlds import content_hash
    _, factory, target, checkpoint, _ = shared
    copied = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = copied["population_provenance"]
    provenance["offline_pretraining"][counter] = -1 if counter == "elapsed_seconds" else 0
    copied["provenance_hash"] = content_hash(provenance)
    altered = tmp_path / "inconsistent-offline-cost.pt"
    torch.save(copied, altered)
    with pytest.raises(ValueError):
        validate_population_checkpoint(altered, **validation(factory, target))
