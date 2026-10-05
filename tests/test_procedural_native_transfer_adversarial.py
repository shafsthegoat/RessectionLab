"""Independent registered-recipe, exclusion and transfer-contract adversaries.

Only tiny procedural fixtures are executed here; this file never launches the
registered patient experiment or opens final/stress worlds.
"""
from __future__ import annotations

import json
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.evaluation import independent_check_native_history
from resectionlab.native_resection import NativeResectionConfig
from resectionlab.native_simulation import NATIVE_ACTION_FEATURE_NAMES, NativeSequentialSimulator
from resectionlab.simulation import RewardSpec
from resectionlab.worlds import content_hash


DECLARATION = Path(__file__).resolve().parents[1] / "manifests/experiments/procedural-native-to-ucsf-v1.json"


@pytest.fixture(scope="module")
def transfer_runner():
    scripts = DECLARATION.parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location("procedural_transfer_audit_runner", scripts / "run_procedural_transfer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def declared_fixture(index):
    declaration = json.loads(DECLARATION.read_text())
    member = declaration["procedural_training"]["members"][index]
    recipe = member["recipe"]
    assert recipe["tissue"] == "all_true"
    tissue = np.ones(tuple(recipe["shape"]), dtype=bool)
    labels = np.zeros(tissue.shape, dtype=np.int16)
    for box in recipe["target_boxes_half_open"]:
        labels[tuple(slice(start, stop) for start, stop in box)] = recipe["target_label"]
    affine = np.asarray(recipe["affine"], dtype=float)
    source_hash = content_hash({"recipe": recipe, "tissue_mask": tissue,
                               "target_labels": labels, "affine": affine})
    native = NativeResectionConfig(tissue, labels, affine, AccessWindow(**recipe["access"]),
        tuple(ToolGeometry(**tool) for tool in recipe["tools"]), source_hash,
        member["tissue_support_provenance"], case_id=recipe["case_id"])
    simulator = NativeSequentialSimulator(native, recipe["candidate_targets_mm"],
        candidate_entries_mm=recipe["candidate_entries_mm"],
        reward=RewardSpec(**declaration["target"]["reward"]), max_steps=recipe["max_steps"],
        max_actions=recipe["max_actions"], partial_contact_weight=recipe["partial_contact_weight"])
    return declaration, member, simulator


@pytest.mark.parametrize("index", [0, 1])
def test_registered_procedural_geometry_reproduces_without_any_patient_arrays(index):
    declaration, member, simulator = declared_fixture(index)
    assert simulator.case_hash == member["source_hash"]
    assert simulator.native_config.fingerprint == member["native_config_hash"]
    assert simulator.decision_model_hash == member["model_hash"]
    assert simulator.config.tissue_mask.shape == (9, 9, 8)
    observation = simulator.observation()
    assert observation.action_features.shape == (5, 15)
    assert observation.state_features.shape == (6,)
    assert list(NATIVE_ACTION_FEATURE_NAMES) == declaration["target"]["action_feature_names"]
    assert observation.action_ids == tuple(member["initial_action_ids"])
    np.testing.assert_array_equal(observation.state_features, [1., 0., 0., 0., 0., 0.])
    assert simulator.config.evidence_available == (False, False)
    assert declaration["claim_boundary"]["human_patients_in_pretraining"] == 0
    assert declaration["claim_boundary"]["clinical_population_training"] is False


@pytest.mark.parametrize("index", [0, 1])
def test_every_procedural_initial_cut_pays_exact_normal_and_partial_contact_cost(index):
    declaration, _member, simulator = declared_fixture(index)
    weights = declaration["target"]["reward"]
    observation = simulator.observation()
    for row, action in enumerate(simulator.proposed_actions()[1:], 1):
        episode = simulator.clone()
        transition = episode.step(action.action_id)
        record = episode.metrics()["history"][0]
        removed = {tuple(point) for point in record["removed_indices_native"]}
        contact = {tuple(point) for point in record["contact_indices_native"]}
        labels = simulator.native_config.target_labels
        target = sum(labels[point] > 0 for point in removed)
        normal = sum(labels[point] == 0 for point in removed)
        partial_normal = sum(labels[point] == 0 for point in contact - removed)
        assert normal > 0, "Procedural training must not inherit a free approach corridor"
        features = observation.action_features[row]
        np.testing.assert_array_equal(features[[0, 1, 2, 3, 4, 8, 9, 10, 12, 13, 14]],
                                      [0., target, normal, 0., 0., 0., 0., 1., partial_normal, 0., 0.])
        tool = next(tool for tool in simulator.native_config.tools if tool.tool_id == action.tool_id)
        assert features[5] == pytest.approx(action.insertion_distance_mm)
        assert features[6] == pytest.approx(tool.tip_radius_mm)
        assert features[7] == pytest.approx(tool.shaft_radius_mm)
        expected = (weights["target_per_mm3"] * target - weights["normal_per_mm3"] * normal
                    - simulator.partial_contact_weight * weights["normal_per_mm3"] * partial_normal
                    - weights["action_cost"])
        assert transition.reward == pytest.approx(expected)
        metrics = episode.metrics()
        assert metrics["simulated_removed_target_volume_mm3"] == target
        assert metrics["simulated_removed_normal_volume_mm3"] == normal
        assert metrics["clinical_deficit_probability"] is None
        assert transition.observation.state_features[1] == pytest.approx(1. / simulator.config.max_steps)
        for next_features in transition.observation.action_features[1:]:
            assert next_features[9] == pytest.approx(1. / simulator.config.max_steps)
        native = simulator.native_config
        reference = SimpleNamespace(mri=np.zeros(native.tissue_mask.shape), affine=native.affine,
                                    frame="RAS+", semantic_hash=simulator.case_hash)
        audit = independent_check_native_history(reference, native.tools, metrics["history"],
            tissue_mask=native.tissue_mask, access=native.access, hard_exclusion=native.hard_exclusion)
        assert audit.feasible and audit.complete_tool_checked and audit.frontier_checked


@pytest.mark.parametrize("alteration", ["budget", "target_alias", "patient_pretraining_claim"])
def test_coherently_resealed_declaration_cannot_change_registered_protocol(transfer_runner, tmp_path, alteration):
    declaration = json.loads(DECLARATION.read_text())
    declaration.pop("declaration_content_hash")
    if alteration == "budget":
        declaration["budgets"]["online_scratch_and_adapted_each_seed"]["max_gradient_steps"] += 1
    elif alteration == "target_alias":
        declaration["target"]["aliases"] = []
    else:
        declaration["claim_boundary"]["human_patients_in_pretraining"] = 2
    declaration["declaration_content_hash"] = content_hash(declaration)
    path = tmp_path / "resealed-declaration.json"
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="(?i)(declaration|preregistration)"):
        transfer_runner.load_declaration(path)
    with pytest.raises(ValueError, match="(?i)(declaration|preregistration)"):
        transfer_runner.assert_declaration(declaration)


def tiny_settings():
    from resectionlab.learning import TrainingConfig
    return TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=4,
        max_environment_steps=64, max_episode_steps=4, max_wall_seconds=30.,
        episodes_per_update=2, checkpoint_interval=2)


@pytest.fixture(scope="module")
def procedural_checkpoint(tmp_path_factory):
    from resectionlab import procedural_learning as transfer
    target, factory = transfer.make_procedural_test_target()
    members = transfer.make_native_procedural_fixture(target)
    observed_worlds = []
    original = NativeSequentialSimulator.reset
    def tracked_reset(simulator, seed=0):
        observed_worlds.append((simulator.case_hash, seed))
        return original(simulator, seed)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(NativeSequentialSimulator, "reset", tracked_reset)
        result = transfer.train_procedural_native_policy(members, excluded_targets=(target,),
            config=tiny_settings(), output_dir=tmp_path_factory.mktemp("procedural-audit") / "shared")
    assert result["status"] == "completed"
    return transfer, target, factory, members, result, observed_worlds


def test_actual_shared_training_never_resets_final_stress_or_patient_worlds(procedural_checkpoint):
    _transfer, target, _factory, members, result, worlds = procedural_checkpoint
    allowed = {member.optimization.case_hash:
        set(member.optimization.seeds) | set(member.selection.seeds) for member in members}
    assert worlds and {case_hash for case_hash, _ in worlds} == set(allowed)
    # Construction initializes a native simulator with seed 0. Every explicitly
    # scheduled world after construction belongs to its exact registered panel.
    assert all(seed == 0 or seed in allowed[case_hash] for case_hash, seed in worlds)
    assert target.case_hash not in allowed
    provenance = result["provenance"]
    assert provenance["human_patients_in_pretraining"] == 0
    assert provenance["training_gradient_steps"] == tiny_settings().max_gradient_steps
    assert sum(provenance["gradient_episode_sources"].values()) == 8


@pytest.mark.parametrize("attack", ["unknown_status", "unchanged_actor", "world_map",
    "understated_postprocessing", "feature_units", "runtime", "source_snapshot", "gradient_counter"])
def test_resealed_transfer_checkpoint_cannot_contradict_numerical_contract(
        procedural_checkpoint, tmp_path, attack):
    transfer, target, factory, _members, result, _worlds = procedural_checkpoint
    artifact = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    provenance = artifact["procedural_provenance"]
    training = provenance["training_record"]
    if attack == "unknown_status":
        training["status"] = "failed"
    elif attack == "unchanged_actor":
        training["initial_checkpoint_hash"] = provenance["policy_hash"]
        provenance["initial_policy_hash"] = provenance["policy_hash"]
    elif attack == "world_map":
        first = sorted(provenance["world_map"])[0]
        provenance["world_map"][first][1] += 1
    elif attack == "understated_postprocessing":
        offline = provenance["offline_pretraining"]
        offline["postprocessing_seconds_before_export"] = offline["total_offline_seconds_before_checkpoint_write"] + 10.
    elif attack == "feature_units":
        provenance["observation_schema"]["action_feature_units"][1] = "fraction"
    elif attack == "runtime":
        provenance["runtime"]["device"] = "cuda"
    elif attack == "source_snapshot":
        name = sorted(provenance["source_snapshot"])[0]
        provenance["source_snapshot"][name] += "\n# altered audit source\n"
    else:
        training["optimization_history"][-1]["gradient_steps"] += 1
    provenance["training_run_hash"] = content_hash(training)
    artifact["provenance_hash"] = content_hash(provenance)
    changed = tmp_path / f"{attack}.pt"
    torch.save(artifact, changed)
    with pytest.raises(ValueError):
        transfer.validate_procedural_checkpoint(changed, target=target, simulator=factory(), hidden_features=16)


@pytest.mark.parametrize("attack", ["group_alias_overlap", "target_alias", "factory_anatomy", "selection_role"])
def test_pretraining_source_substitution_is_rejected_before_learner(tmp_path, monkeypatch, attack):
    from resectionlab import learning, procedural_learning as transfer
    from resectionlab.worlds import WorldRole
    target, factory = transfer.make_procedural_test_target()
    members = list(transfer.make_native_procedural_fixture(target))
    if attack == "group_alias_overlap":
        members[0] = replace(members[0], aliases=(members[1].family_id,))
    elif attack == "target_alias":
        members[0] = replace(members[0], aliases=target.aliases)
    elif attack == "factory_anatomy":
        members[0] = replace(members[0], factory=factory)
    else:
        members[0] = replace(members[0], selection=replace(members[0].selection, role=WorldRole.FINAL_EVALUATION))
    monkeypatch.setattr(learning, "train_patient_policy", lambda *a, **k: pytest.fail("Invalid source reached learner"))
    with pytest.raises(ValueError):
        transfer.train_procedural_native_policy(members, excluded_targets=(target,),
            config=tiny_settings(), output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid/procedural.pt").exists()


def test_common_learner_cannot_route_procedural_checkpoint_through_old_population_gate(
        procedural_checkpoint, tmp_path, monkeypatch):
    from resectionlab import learning
    from resectionlab.worlds import generate_partitions
    _transfer, target, factory, _members, shared, _worlds = procedural_checkpoint
    worlds = generate_partitions(target.case_hash, factory().config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    monkeypatch.setattr(learning.torch.optim, "Adam", lambda *a, **k: pytest.fail("Invalid domain reached Adam"))
    with pytest.raises(ValueError):
        learning.train_patient_policy(factory, worlds.optimization, worlds.selection,
            config=tiny_settings(), output_dir=tmp_path / "wrong-domain",
            shared_checkpoint=shared["checkpoint_path"], population_case_group=target.group_id,
            population_case_aliases=target.aliases)


def test_coherently_relabeled_final_panel_is_rejected_by_direct_adaptation_seam(
        procedural_checkpoint, tmp_path, monkeypatch):
    from resectionlab import learning
    from resectionlab.worlds import WorldRole, generate_partitions
    _transfer, target, factory, _members, shared, _worlds = procedural_checkpoint
    worlds = generate_partitions(target.case_hash, factory().config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    relabeled = replace(worlds.final_evaluation, role=WorldRole.SELECTION)
    monkeypatch.setattr(learning.torch.optim, "Adam", lambda *a, **k: pytest.fail("Relabeled final panel reached Adam"))
    with pytest.raises(ValueError, match="worlds differ"):
        learning.train_patient_policy(factory, worlds.optimization, relabeled,
            config=tiny_settings(), output_dir=tmp_path / "wrong-worlds",
            procedural_checkpoint=shared["checkpoint_path"], procedural_target=target)


def test_public_online_budget_and_seeds_are_exactly_preregistered():
    from resectionlab import learning, procedural_learning as transfer
    declaration = json.loads(DECLARATION.read_text())
    item = declaration["target"]
    target = transfer.TransferTarget(item["semantic_hash"], item["planning_hash"], item["group_id"],
        tuple(item["aliases"]), item["source_kind"])
    config = learning.TrainingConfig(**declaration["budgets"]["online_scratch_and_adapted_each_seed"])
    for seed in declaration["policy"]["optimization_seeds_online"]:
        transfer._validate_budget(replace(config, seed=seed), target, online=True)
    with pytest.raises(ValueError, match="seed"):
        transfer._validate_budget(replace(config, seed=999), target, online=True)
    with pytest.raises(ValueError, match="budget"):
        transfer._validate_budget(replace(config, seed=11, max_gradient_steps=config.max_gradient_steps + 1), target, online=True)


def test_cancelled_pretraining_cannot_export_a_transfer_initialization(tmp_path):
    from resectionlab import procedural_learning as transfer
    target, _factory = transfer.make_procedural_test_target()
    result = transfer.train_procedural_native_policy(transfer.make_native_procedural_fixture(target),
        excluded_targets=(target,), config=tiny_settings(), output_dir=tmp_path / "cancelled",
        cancelled=lambda: True)
    assert result["status"] == "cancelled"
    assert result["checkpoint_path"] is None
    assert result["training"]["gradient_steps"] == 0
    assert not (tmp_path / "cancelled/procedural.pt").exists()


def test_resumed_adaptation_rejects_changed_source_bytes_and_preserves_checkpoint(
        procedural_checkpoint, tmp_path):
    from resectionlab.worlds import generate_partitions
    transfer, target, factory, _members, shared, _worlds = procedural_checkpoint
    worlds = generate_partitions(target.case_hash, factory().config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    destination = tmp_path / "adaptation"
    options = dict(checkpoint=shared["checkpoint_path"], target=target,
        config=tiny_settings(), output_dir=destination)
    cancelled = transfer.train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
        cancelled=lambda: True, **options)
    assert cancelled.gradient_steps == 0
    paths = [destination / name for name in ("checkpoint.pt", "initial.pt", "contract.json", "result.json")]
    originals = {path: path.read_bytes() for path in paths}
    local_source = destination / "procedural-source.pt"
    assert local_source.read_bytes() == Path(shared["checkpoint_path"]).read_bytes()
    # Torch accepts trailing bytes, so this attack leaves weights/provenance
    # readable but changes the source artifact's exact retained identity.
    with local_source.open("ab") as changed:
        changed.write(b"changed-source-container")
    with pytest.raises(ValueError, match="resume contract changed"):
        transfer.train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
            resume=True, **options)
    assert all(path.read_bytes() == payload for path, payload in originals.items())


class BudgetClock:
    def __init__(self):
        self.seconds = 0.

    def perf_counter(self):
        return self.seconds


class TimedStopFixture:
    """One known STOP transition with virtual cost; no anatomy or risk claims."""
    decision_model_hash = "sha256:timed-stop-contract-v1"
    case_hash = "timed-stop-contract"

    def __init__(self, clock, transition_seconds):
        from resectionlab.worlds import WorldGeneratorConfig
        self.clock = clock
        self.transition_seconds = transition_seconds
        self.world_generator_fingerprint = WorldGeneratorConfig().fingerprint

    def reset(self, seed=0):
        return SimpleNamespace(action_ids=("STOP",), action_features=np.array([[1., 0.]], np.float32),
            state_features=np.array([1.], np.float32), action_mask=np.array([True]))

    def step(self, action):
        assert action == 0
        self.clock.seconds += self.transition_seconds
        return SimpleNamespace(observation=self.reset(), reward=0., terminated=True, info={})

    def metrics(self):
        return {"source": "timing_contract_fixture", "clinical_deficit_probability": None}


def timed_factory(clock, transition_seconds):
    constructions = 0
    def factory():
        nonlocal constructions
        if constructions == 0:
            clock.seconds += 50.  # Measured initialization, outside learning allowance.
        constructions += 1
        return TimedStopFixture(clock, transition_seconds)
    return factory


def timing_panels():
    from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole
    generator = WorldGeneratorConfig()
    return (WorldPartitionManifest(WorldRole.OPTIMIZATION, TimedStopFixture.case_hash, generator, (1, 2)),
            WorldPartitionManifest(WorldRole.SELECTION, TimedStopFixture.case_hash, generator, (3, 4)))


def test_initial_selection_time_is_charged_and_incomplete_panel_has_no_selected_actor(tmp_path, monkeypatch):
    import time
    from resectionlab import learning
    clock = BudgetClock()
    monkeypatch.setattr(learning, "time", SimpleNamespace(perf_counter=clock.perf_counter, time=time.time))
    config = learning.TrainingConfig(max_wall_seconds=1., max_gradient_steps=4, max_environment_steps=20,
        max_episode_steps=1, episodes_per_update=1, checkpoint_interval=1, hidden_features=4)
    result = learning.train_patient_policy(timed_factory(clock, 1.5), *timing_panels(),
        config=config, output_dir=tmp_path)
    assert result.status == "wall_time_budget"
    assert result.initialization_seconds == pytest.approx(50.)
    assert result.elapsed_seconds == pytest.approx(1.5)
    assert result.gradient_steps == result.optimization_environment_steps == 0
    assert result.selection_environment_steps == 1
    assert result.initial_selection_return is result.selected_selection_return is None
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["selection_history"] == []
    assert saved["selection_seconds"] == pytest.approx(1.5)


def test_resume_keeps_cumulative_learning_budget_without_charging_setup_or_paused_time(tmp_path, monkeypatch):
    import time
    from resectionlab import learning
    clock = BudgetClock()
    monkeypatch.setattr(learning, "time", SimpleNamespace(perf_counter=clock.perf_counter, time=time.time))
    config = learning.TrainingConfig(max_wall_seconds=1.3, max_gradient_steps=8, max_environment_steps=20,
        max_episode_steps=1, episodes_per_update=1, checkpoint_interval=1, hidden_features=4)
    cancel_state = {"cancelled": False}
    first = learning.train_patient_policy(timed_factory(clock, .2), *timing_panels(),
        config=config, output_dir=tmp_path, cancelled=lambda: cancel_state["cancelled"],
        progress=lambda _: cancel_state.update(cancelled=True))
    assert first.status == "cancelled" and first.gradient_steps == 1
    assert first.elapsed_seconds == pytest.approx(1.)
    assert first.initialization_seconds == pytest.approx(50.)
    initial_selected = first.selected_checkpoint_hash
    clock.seconds += 1000.  # User pause between invocations, never learning time.
    resumed = learning.train_patient_policy(timed_factory(clock, .2), *timing_panels(),
        config=config, output_dir=tmp_path, resume=True)
    assert resumed.status == "wall_time_budget"
    assert resumed.gradient_steps == 2
    assert resumed.elapsed_seconds == pytest.approx(1.4)
    assert resumed.initialization_seconds == pytest.approx(100.)
    assert resumed.optimization_environment_steps == 2
    assert resumed.selection_environment_steps == 5
    assert resumed.selected_checkpoint_hash == initial_selected
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["initialization_seconds_this_invocation"] == pytest.approx(50.)
    assert [row["gradient_steps"] for row in saved["selection_history"]] == [0, 1]
    exhausted = learning.train_patient_policy(timed_factory(clock, .2), *timing_panels(),
        config=config, output_dir=tmp_path, resume=True)
    assert exhausted.status == "wall_time_budget"
    assert exhausted.gradient_steps == resumed.gradient_steps
    assert exhausted.optimization_environment_steps == resumed.optimization_environment_steps
    assert exhausted.selection_environment_steps == resumed.selection_environment_steps
    assert exhausted.elapsed_seconds == pytest.approx(1.4)
    assert exhausted.initialization_seconds == pytest.approx(150.)
