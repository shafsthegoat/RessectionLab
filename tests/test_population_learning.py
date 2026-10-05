"""Synthetic shared pretraining, cohort exclusions and isolated online arms."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from resectionlab import learning
from resectionlab.population_learning import (PopulationIdentity, feature_schema,
    load_frozen_population_policy, make_analytic_population_fixture,
    train_population_policy, validate_population_checkpoint)
from resectionlab.worlds import WorldRole, generate_partitions
from resectionlab.worlds import content_hash


def settings(**changes):
    return replace(learning.TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=8,
        max_environment_steps=256, max_episode_steps=6, max_wall_seconds=30,
        episodes_per_update=4, checkpoint_interval=2), **changes)


@pytest.fixture
def pretrained(tmp_path):
    members, factory, excluded = make_analytic_population_fixture()
    result = train_population_policy(members, exclusions=(excluded,), config=settings(), output_dir=tmp_path / "population")
    assert result["checkpoint_path"]
    return members, factory, excluded, result


def validation_args(factory, excluded):
    return {"target_case_hash": excluded.case_hash, "target_group": excluded.group_id,
            "target_aliases": excluded.aliases, "expected_dimensions": (12, 6, 16),
            "expected_feature_schema": feature_schema(factory())}


def test_actual_shared_gradients_bind_source_runtime_and_both_groups(pretrained):
    members, factory, excluded, result = pretrained
    validated = validate_population_checkpoint(result["checkpoint_path"], **validation_args(factory, excluded))
    provenance = validated["provenance"]
    assert provenance["training_gradient_steps"] == 8
    assert provenance["initial_policy_hash"] != provenance["policy_hash"]
    assert provenance["actor_parameters_changed"] is True
    assert set(provenance["gradient_episode_sources"]) == {member.identity.case_hash for member in members}
    assert all(count > 0 for count in provenance["gradient_episode_sources"].values())
    assert provenance["cohort"]["synthetic_group_count"] == 2
    assert provenance["cohort"]["independent_patient_count"] == 0
    assert provenance["final_worlds_used"] is False
    assert provenance["runtime"]["device"] == "cpu"
    from pathlib import Path
    root = Path(result["checkpoint_path"]).parent
    for name, digest in provenance["source_sha256"].items():
        assert hashlib.sha256((root / "source-snapshot/resectionlab" / name).read_bytes()).hexdigest() == digest


def test_frozen_policy_has_no_gradients_and_clones_cannot_change_shared_weights(pretrained):
    _, factory, excluded, result = pretrained
    from pathlib import Path
    path = Path(result["checkpoint_path"])
    before = path.read_bytes()
    first = load_frozen_population_policy(path, **validation_args(factory, excluded))
    second = load_frozen_population_policy(path, **validation_args(factory, excluded))
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in first.parameters())
    initial = learning.policy_hash(first)
    learning.rollout_policy(first, factory(), seed=123, max_steps=6)
    assert learning.policy_hash(first) == initial
    with torch.no_grad():
        next(second.parameters()).add_(1)
    assert learning.policy_hash(second) != initial
    assert path.read_bytes() == before and learning.policy_hash(first) == initial


def test_adaptation_uses_same_initial_weights_fresh_case_state_and_budget(pretrained, tmp_path):
    _, factory, excluded, result = pretrained
    simulator = factory()
    worlds = generate_partitions(simulator.case_hash, simulator.config.world_generator, 12,
        optimization=3, selection=2, final_evaluation=3, stress=2)
    for seed in (11, 23):
        config = settings(seed=seed, max_gradient_steps=4)
        adapted = learning.train_patient_policy(factory, worlds.optimization, worlds.selection,
            config=config, output_dir=tmp_path / f"adapt-{seed}", shared_checkpoint=result["checkpoint_path"],
            population_case_group=excluded.group_id, population_case_aliases=excluded.aliases)
        assert adapted.initial_checkpoint_hash == adapted.shared_checkpoint_hash == result["policy_hash"]
        assert adapted.optimizer_mode == "POPULATION_ADAPTED"
        assert adapted.gradient_steps == 4
        contract = json.loads((tmp_path / f"adapt-{seed}/contract.json").read_text())
        assert contract["config"] == learning.asdict(config)
        assert contract["population_initialization"]["provenance_hash"] == result["provenance_hash"]
    assert validate_population_checkpoint(result["checkpoint_path"], **validation_args(factory, excluded))["policy_hash"] == result["policy_hash"]


def test_aliases_repeat_groups_and_undeclared_targets_cannot_bypass_exclusions(pretrained):
    members, factory, excluded, result = pretrained
    args = validation_args(factory, excluded)
    for changes in ({"target_case_hash": members[0].identity.case_hash},
                    {"target_group": members[0].identity.group_id},
                    {"target_aliases": (members[0].identity.group_id,)},
                    {"target_aliases": members[0].identity.aliases}):
        with pytest.raises(ValueError, match="overlap"):
            validate_population_checkpoint(result["checkpoint_path"], **{**args, **changes})
    with pytest.raises(ValueError, match="predeclared"):
        validate_population_checkpoint(result["checkpoint_path"], **{**args, "target_aliases": ()})
    with pytest.raises(ValueError, match="Synthetic population"):
        validate_population_checkpoint(result["checkpoint_path"], **{**args, "target_case_hash": "sha256:real-patient"})


def test_final_groups_and_final_worlds_never_enter_shared_updates(tmp_path):
    members, _, excluded = make_analytic_population_fixture()
    final_member = replace(members[0], identity=replace(members[0].identity, outer_split="final_evaluation"))
    with pytest.raises(ValueError, match="outer-development"):
        train_population_policy((final_member, members[1]), exclusions=(excluded,), config=settings(), output_dir=tmp_path / "outer")
    final_world = replace(members[0], selection_manifest=replace(members[0].selection_manifest, role=WorldRole.FINAL_EVALUATION))
    with pytest.raises(ValueError, match="expected selection"):
        train_population_policy((final_world, members[1]), exclusions=(excluded,), config=settings(), output_dir=tmp_path / "world")


def test_population_export_requires_actual_updates_and_frozen_source(tmp_path, monkeypatch):
    members, _, excluded = make_analytic_population_fixture()
    result = train_population_policy(members, exclusions=(excluded,), config=settings(),
        output_dir=tmp_path / "cancel", cancelled=lambda: True)
    assert result["checkpoint_path"] is None
    assert not (tmp_path / "cancel/population.pt").exists()
    original = learning.numerical_source_hashes()
    changed = {"value": False}
    monkeypatch.setattr(learning, "numerical_source_hashes", lambda: {**original, "simulation.py": "changed"} if changed["value"] else original)
    with pytest.raises(ValueError, match="source changed during training"):
        train_population_policy(members, exclusions=(excluded,), config=settings(max_gradient_steps=2),
            output_dir=tmp_path / "changed", progress=lambda _: changed.update(value=True))
    assert not (tmp_path / "changed/population.pt").exists()


def test_identity_copies_alias_lists_and_dimensions_require_actual_semantics(pretrained):
    aliases = ["synthetic:original"]
    identity = PopulationIdentity("synthetic-identity", "synthetic:test", aliases)
    aliases.append("synthetic:changed")
    assert identity.aliases == ("synthetic:original",)
    _, factory, excluded, result = pretrained
    args = validation_args(factory, excluded)
    wrong = {**args["expected_feature_schema"], "state_feature_names": list(reversed(args["expected_feature_schema"]["state_feature_names"]))}
    with pytest.raises(ValueError, match="feature semantics"):
        validate_population_checkpoint(result["checkpoint_path"], **{**args, "expected_feature_schema": wrong})


def test_adaptation_resume_retains_shared_source_contract(pretrained, tmp_path):
    _, factory, excluded, result = pretrained
    simulator = factory()
    worlds = generate_partitions(simulator.case_hash, simulator.config.world_generator, 32,
        optimization=3, selection=2, final_evaluation=3, stress=2)
    config = settings(max_gradient_steps=6)
    cancel = {"value": False}
    folder = tmp_path / "resume"
    first = learning.train_patient_policy(factory, worlds.optimization, worlds.selection,
        config=config, output_dir=folder, shared_checkpoint=result["checkpoint_path"],
        population_case_group=excluded.group_id, population_case_aliases=excluded.aliases,
        cancelled=lambda: cancel["value"], progress=lambda _: cancel.update(value=True))
    assert first.status == "cancelled" and first.gradient_steps == 2
    resumed = learning.train_patient_policy(factory, worlds.optimization, worlds.selection,
        config=config, output_dir=folder, resume=True)
    whole = learning.train_patient_policy(factory, worlds.optimization, worlds.selection,
        config=config, output_dir=tmp_path / "whole", shared_checkpoint=result["checkpoint_path"],
        population_case_group=excluded.group_id, population_case_aliases=excluded.aliases)
    assert resumed.latest_checkpoint_hash == whole.latest_checkpoint_hash
    assert resumed.selected_checkpoint_hash == whole.selected_checkpoint_hash


def test_checkpoint_cannot_replace_missing_training_settings_with_defaults(pretrained, tmp_path):
    _, factory, excluded, result = pretrained
    source = torch.load(result["checkpoint_path"], weights_only=True)
    source["population_provenance"]["training_config"].pop("max_environment_steps")
    source["provenance_hash"] = content_hash(source["population_provenance"])
    changed = tmp_path / "missing-budget.pt"
    torch.save(source, changed)
    with pytest.raises(ValueError, match="explicit and complete"):
        validate_population_checkpoint(changed, **validation_args(factory, excluded))
