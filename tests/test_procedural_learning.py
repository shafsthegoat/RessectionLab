"""Tiny explicitly nonpatient contract tests; the registered study is not run."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from resectionlab import learning
from resectionlab.procedural_learning import (TEST_SCOPE, TransferTarget,
    make_procedural_test_target, make_native_procedural_fixture, native_observation_schema,
    train_procedural_native_policy, validate_procedural_checkpoint,
    load_frozen_procedural_policy, train_procedural_adapted_policy)
from resectionlab.worlds import WorldRole, content_hash, generate_partitions


def settings(**changes):
    return replace(learning.TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=4,
        max_environment_steps=128, max_episode_steps=4, max_wall_seconds=20.,
        episodes_per_update=2, checkpoint_interval=2), **changes)


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("procedural-contract")
    target, factory = make_procedural_test_target()
    members = make_native_procedural_fixture(target)
    result = train_procedural_native_policy(members, excluded_targets=(target,), config=settings(),
                                            output_dir=directory / "shared")
    assert result["status"] == "completed"
    return target, factory, members, result


def args(target, factory):
    return {"target": target, "simulator": factory(), "hidden_features": 16}


def panels(target, factory):
    return generate_partitions(target.case_hash, factory().config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)


def test_recipe_features_and_immutable_identity_snapshot():
    target, factory = make_procedural_test_target()
    aliases = list(target.aliases)
    copied = replace(target, aliases=aliases)
    aliases.clear()
    assert copied.aliases == target.aliases
    members = make_native_procedural_fixture(target)
    manifest = members[0].generator_manifest
    manifest["recipe"]["tools"][0]["shaft_radius_mm"] += 5
    assert members[0].generator_manifest != manifest
    schema = native_observation_schema(factory())
    assert len(schema["action_feature_names"]) == 15
    assert len(schema["state_feature_names"]) == 6
    assert schema["action_feature_units"][9] == "fraction_of_action_budget"
    assert "not physical depth" in schema["depth_semantics"]
    assert all(native_observation_schema(member.factory()) == schema for member in members)


def test_actual_shared_updates_preserve_source_evidence_and_both_families(pretrained):
    target, factory, members, result = pretrained
    checked = validate_procedural_checkpoint(result["checkpoint_path"], **args(target, factory))
    provenance = checked["provenance"]
    assert checked["scope"] == TEST_SCOPE
    assert provenance["training_gradient_steps"] == 4
    assert provenance["actor_parameters_changed"] is True
    assert provenance["policy_hash"] != provenance["initial_policy_hash"]
    assert provenance["human_patients_in_pretraining"] == 0
    assert provenance["clinical_population_training"] is False
    assert provenance["final_worlds_used"] is False
    assert set(provenance["gradient_episode_sources"]) == {member.optimization.case_hash for member in members}
    source_dir = Path(result["checkpoint_path"]).parent / "source-snapshot/resectionlab"
    for name, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == digest


def test_frozen_actor_never_constructs_optimizer_or_receives_gradients(pretrained, monkeypatch):
    target, factory, _members, result = pretrained
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen arm constructed Adam")
    monkeypatch.setattr(learning.torch.optim, "Adam", forbidden)
    policy = load_frozen_procedural_policy(result["checkpoint_path"], **args(target, factory))
    before = learning.policy_hash(policy)
    output = learning.rollout_policy(policy, factory(), seed=panels(target, factory).selection.seeds[0])
    assert output.actions[-1] == "STOP"
    assert learning.policy_hash(policy) == before
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in policy.parameters())


def test_adaptation_exact_initial_weights_fresh_adam_and_same_caps(pretrained, tmp_path, monkeypatch):
    target, factory, _members, shared = pretrained
    worlds = panels(target, factory)
    original = learning.torch.optim.Adam
    initial_states = []
    def counted(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        initial_states.append(len(optimizer.state))
        return optimizer
    monkeypatch.setattr(learning.torch.optim, "Adam", counted)
    config = settings(max_gradient_steps=2)
    result = train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
        checkpoint=shared["checkpoint_path"], target=target, config=config, output_dir=tmp_path / "adapt")
    scratch = learning.train_patient_policy(factory, worlds.optimization, worlds.selection,
        config=config, output_dir=tmp_path / "scratch")
    assert initial_states == [0, 0]
    assert result.initial_checkpoint_hash == shared["policy_hash"] == result.shared_checkpoint_hash
    assert result.optimizer_mode == "PROCEDURAL_PRETRAINED_ADAPTED"
    assert result.gradient_steps == scratch.gradient_steps == 2
    assert result.optimization_environment_steps <= config.max_environment_steps
    assert learning.policy_hash(learning.load_policy(shared["checkpoint_path"])) == shared["policy_hash"]
    saved = json.loads((tmp_path / "adapt/contract.json").read_text())
    assert saved["procedural_initialization"]["scope"] == TEST_SCOPE
    assert saved["population_initialization"] is None


def test_cancel_resume_retains_procedural_initialization(pretrained, tmp_path):
    target, factory, _members, shared = pretrained
    worlds = panels(target, factory)
    config = settings(max_gradient_steps=2)
    options = dict(checkpoint=shared["checkpoint_path"], target=target, config=config, output_dir=tmp_path / "adapt")
    cancelled = train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
                                               cancelled=lambda: True, **options)
    assert cancelled.status == "cancelled" and cancelled.gradient_steps == 0
    resumed = train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection, resume=True, **options)
    assert resumed.gradient_steps == 2
    assert resumed.initial_checkpoint_hash == shared["policy_hash"]


def test_slow_initialization_excluded_but_selection_and_resumed_budget_are_charged(pretrained, tmp_path, monkeypatch):
    from resectionlab import procedural_learning
    target, factory, _members, shared = pretrained
    worlds = panels(target, factory)
    clock = [0.]
    executed = [0]
    monkeypatch.setattr(learning, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original_validation = procedural_learning.validate_procedural_adaptation
    def slow_validation(*args, **kwargs):
        result = original_validation(*args, **kwargs)
        clock[0] += 100.  # Much longer than the entire declared learner budget.
        return result
    monkeypatch.setattr(procedural_learning, "validate_procedural_adaptation", slow_validation)
    def measured_factory():
        simulator = factory()
        original_step = simulator.step
        def measured_step(action):
            value = original_step(action)
            clock[0] += .05
            executed[0] += 1
            return value
        simulator.step = measured_step
        return simulator
    stop = [False]
    config = settings(max_gradient_steps=2, checkpoint_interval=1, max_wall_seconds=10.)
    options = dict(checkpoint=shared["checkpoint_path"], target=target,
                   config=config, output_dir=tmp_path / "timed")
    first = train_procedural_adapted_policy(measured_factory, worlds.optimization, worlds.selection,
        cancelled=lambda: stop[0], progress=lambda _: stop.__setitem__(0, True), **options)
    assert first.status == "cancelled" and first.gradient_steps == 1
    assert first.initialization_seconds == pytest.approx(100.)
    assert first.selection_environment_steps > 0
    assert first.elapsed_seconds == pytest.approx(executed[0] * .05)
    assert first.elapsed_seconds >= first.selection_environment_steps * .05 - 1e-9
    consumed = first.elapsed_seconds
    resumed = train_procedural_adapted_policy(measured_factory, worlds.optimization, worlds.selection,
                                              resume=True, **options)
    assert resumed.gradient_steps == 2
    assert resumed.initialization_seconds == pytest.approx(200.)
    assert resumed.elapsed_seconds > consumed
    assert resumed.elapsed_seconds == pytest.approx(executed[0] * .05)
    saved = json.loads((tmp_path / "timed/result.json").read_text())
    assert saved["initialization_seconds_this_invocation"] == pytest.approx(100.)
    assert "initialization excluded" in saved["elapsed_seconds_scope"]
    contract = json.loads((tmp_path / "timed/contract.json").read_text())
    assert contract["timing_contract"] == "optimization_selection_budget_v2_initialization_separate"


def test_final_worlds_rejected_before_adaptation(pretrained, tmp_path):
    target, factory, _members, shared = pretrained
    worlds = panels(target, factory)
    with pytest.raises(ValueError, match="expected selection"):
        train_procedural_adapted_policy(factory, worlds.optimization, worlds.final_evaluation,
            checkpoint=shared["checkpoint_path"], target=target, config=settings(), output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad/initial.pt").exists()


@pytest.mark.parametrize("mutation", ["world_role", "source_recipe", "negative_time", "final_worlds", "kind"])
def test_resealed_inconsistent_provenance_is_refused(pretrained, tmp_path, mutation):
    target, factory, _members, shared = pretrained
    saved = torch.load(shared["checkpoint_path"], map_location="cpu", weights_only=True)
    provenance = saved["procedural_provenance"]
    if mutation == "world_role":
        provenance["members"][0]["partitions"]["selection"]["role"] = "final_evaluation"
    elif mutation == "source_recipe":
        provenance["members"][0]["generator_manifest"]["recipe"]["target_boxes_half_open"] = []
    elif mutation == "negative_time":
        provenance["offline_pretraining"]["elapsed_seconds"] = -1
    elif mutation == "final_worlds":
        provenance["final_worlds_used"] = True
    else:
        saved["kind"] = "population_checkpoint"
    saved["provenance_hash"] = content_hash(provenance)
    path = tmp_path / "resealed.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError):
        validate_procedural_checkpoint(path, **args(target, factory))


def test_test_target_cannot_be_relabeled_public_patient(pretrained):
    target, factory, _members, shared = pretrained
    with pytest.raises(ValueError, match="declared exclusion"):
        validate_procedural_checkpoint(shared["checkpoint_path"],
            **args(replace(target, source_kind="public_patient_structural_mirror"), factory))
    with pytest.raises(ValueError, match="declared exclusion"):
        validate_procedural_checkpoint(shared["checkpoint_path"], **args(replace(target, aliases=()), factory))
