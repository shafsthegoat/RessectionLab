"""Auditable shared-policy training on explicitly synthetic development groups.

This bounded implementation does not train a clinical population policy. It
tests the four-arm protocol, group exclusions, genuine shared updates and isolated
case adaptation. Procedural anatomy groups are not independent human patients.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import inspect
import math
from pathlib import Path
import platform
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import learning
from .cohort import OUTER_SPLITS
from .worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole, content_hash, generate_partitions


@dataclass(frozen=True)
class PopulationIdentity:
    case_hash: str
    group_id: str
    aliases: tuple[str, ...] = ()
    outer_split: str = "development"
    source_kind: str = "analytic_synthetic"

    def __post_init__(self) -> None:
        if (not isinstance(self.case_hash, str) or not self.case_hash.startswith("synthetic-")
                or not isinstance(self.group_id, str) or not self.group_id.startswith("synthetic:")):
            raise ValueError("Synthetic population identities require an explicit synthetic group")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        if self.source_kind != "analytic_synthetic":
            raise ValueError("Clinical population pretraining is not implemented by this synthetic protocol")
        if self.outer_split not in OUTER_SPLITS:
            raise ValueError("Unsupported outer cohort split")
        if (any(not isinstance(alias, str) or not alias or ":" not in alias for alias in self.aliases)
                or len(set(self.aliases)) != len(self.aliases)):
            raise ValueError("Aliases must be distinct nonempty namespaced identities")

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(("case:" + self.case_hash, "identity:" + self.group_id,
                          *("identity:" + value for value in self.aliases)))


@dataclass(frozen=True)
class PopulationMember:
    identity: PopulationIdentity
    simulator_factory: Callable[[], Any]
    optimization_manifest: WorldPartitionManifest
    selection_manifest: WorldPartitionManifest


def feature_schema(simulator: Any) -> dict[str, Any]:
    from .native_simulation import NATIVE_ACTION_FEATURE_NAMES, NativeSequentialSimulator
    from .simulation import ACTION_FEATURE_NAMES, SequentialSimulator
    if type(simulator) is NativeSequentialSimulator:
        action_names = NATIVE_ACTION_FEATURE_NAMES
        version = "native_candidate_features_v1"
    elif type(simulator) is SequentialSimulator:
        action_names = ACTION_FEATURE_NAMES
        version = "coarse_candidate_features_v1"
    else:
        raise ValueError("Population transfer requires a known named observation schema")
    scope = "analytic_synthetic" if simulator.config.case_id.startswith("synthetic") and simulator.case_hash.startswith("synthetic-") else "patient_or_unverified"
    return {"version": version, "anatomy_scope": scope, "action_feature_names": list(action_names),
        "state_feature_names": ["remaining_target_fraction", "action_budget_fraction_used",
            "removed_tissue_fraction", "nominal_severed_edge_fraction", "motor_evidence_available",
            "language_evidence_available"]}


def _cohort_record(members: Sequence[PopulationIdentity], exclusions: Sequence[PopulationIdentity]) -> dict[str, Any]:
    if len(members) < 2 or not exclusions:
        raise ValueError("Shared pretraining requires two development groups and explicit target exclusions")
    if any(member.outer_split != "development" for member in members):
        raise ValueError("Population updates may only use outer-development groups")
    if len({member.case_hash for member in members}) != len(members):
        raise ValueError("Duplicate development case or derivative must not count twice")
    # Connected alias components bind repeat visits and derivatives, including
    # transitive aliases. Even a reported overlap is grouped conservatively.
    components: list[set[str]] = []
    for record in (*members, *exclusions):
        merged = set(record.tokens)
        for component in list(components):
            if merged & component:
                merged |= component
                components.remove(component)
        components.append(merged)
    for component in components:
        if any(member.tokens & component for member in members) and any(excluded.tokens & component for excluded in exclusions):
            raise ValueError("Development/excluded patient group, alias or case overlaps")
    development_groups = {index for index, component in enumerate(components)
                          if any(member.tokens & component for member in members)}
    if len(development_groups) < 2:
        raise ValueError("Repeat visits/derivatives do not establish two development groups")
    return {"development": [asdict(member) for member in members],
            "exclusions": [asdict(record) for record in exclusions],
            "identity_components": [sorted(value) for value in components],
            "synthetic_group_count": len(development_groups), "independent_patient_count": 0}


def _identities(records: Sequence[dict[str, Any]]) -> tuple[PopulationIdentity, ...]:
    return tuple(PopulationIdentity(**{**record, "aliases": tuple(record.get("aliases", ()))}) for record in records)


def validate_population_checkpoint(checkpoint: str | Path, *, target_case_hash: str,
                                    target_group: str, target_aliases: tuple[str, ...] = (),
                                    expected_dimensions: tuple[int, int, int] | None = None,
                                    expected_feature_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(checkpoint)
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("kind") != "population_checkpoint" or source.get("schema_version") != 1:
        raise ValueError("Population adaptation requires an actual shared training checkpoint with provenance")
    provenance = source.get("population_provenance", {})
    if (source.get("provenance_hash") != content_hash(provenance)
            or provenance.get("scope") != "analytic_synthetic_only"
            or provenance.get("policy_hash") != source.get("policy_hash")
            or type(provenance.get("training_gradient_steps")) is not int
            or provenance.get("training_gradient_steps", 0) <= 0
            or provenance.get("actor_parameters_changed") is not True):
        raise ValueError("Population checkpoint provenance or actual update evidence is invalid")
    saved_settings = provenance.get("training_config")
    if not isinstance(saved_settings, dict) or set(saved_settings) != set(asdict(learning.TrainingConfig())):
        raise ValueError("Population pretraining settings must be explicit and complete")
    settings = learning.TrainingConfig(**saved_settings)
    offline = provenance.get("offline_pretraining", {})
    for key in ("gradient_steps", "optimization_environment_steps", "selection_environment_steps"):
        if type(offline.get(key)) is not int or offline[key] < 0:
            raise ValueError("Population offline resource counters must be nonnegative integers")
    if (offline["gradient_steps"] != provenance["training_gradient_steps"]
            or offline["gradient_steps"] > settings.max_gradient_steps
            or not 0 < offline["optimization_environment_steps"] <= settings.max_environment_steps):
        raise ValueError("Population offline counters contradict recorded training budget or updates")
    durations = ("elapsed_seconds", "preparation_seconds", "initialization_seconds", "postprocessing_seconds_before_export",
                 "total_offline_seconds_before_checkpoint_write")
    for key in durations:
        value = offline.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Population offline costs must be finite and nonnegative")
    if offline["total_offline_seconds_before_checkpoint_write"] + 1e-6 < offline["elapsed_seconds"] + offline["preparation_seconds"] + offline["initialization_seconds"]:
        raise ValueError("Population total offline cost omits preparation or learning")
    if provenance.get("final_worlds_used") is not False or provenance.get("checkpoint_rule") != "fixed_budget_latest":
        raise ValueError("Population checkpoint declares final-world exposure or an unsupported checkpoint rule")
    members = _identities(provenance["cohort"]["development"])
    exclusions = _identities(provenance["cohort"]["exclusions"])
    if _cohort_record(members, exclusions) != provenance["cohort"]:
        raise ValueError("Population cohort/group binding changed")
    member_records = provenance.get("members", [])
    if (len(member_records) != len(members)
            or _identities([record["identity"] for record in member_records]) != members
            or provenance.get("development_case_hashes") != [member.case_hash for member in members]):
        raise ValueError("Population simulator membership differs from the declared development cohort")
    for member, record in zip(members, member_records):
        saved = record["partitions"]
        reconstructed = []
        for key in ("optimization", "selection"):
            value = saved[key]
            partition = WorldPartitionManifest(WorldRole(value["role"]), value["case_hash"],
                WorldGeneratorConfig(**value["generator"]), tuple(value["seeds"]), value.get("planning_hash"))
            if partition.case_hash != member.case_hash or partition.partition_hash != value["partition_hash"]:
                raise ValueError("Population world provenance does not match its source case/partition hash")
            reconstructed.append(partition)
        if learning._validate_partitions(*reconstructed) != saved:
            raise ValueError("Population optimization/selection provenance changed")
    if not all(provenance.get("runtime", {}).get(key) for key in ("python", "numpy", "torch", "device")):
        raise ValueError("Population runtime provenance is incomplete")
    if not all(provenance.get("source_sha256", {}).get(key) for key in ("learning.py", "population_learning.py", "simulation.py")):
        raise ValueError("Population numerical source provenance is incomplete")
    target = PopulationIdentity(target_case_hash, target_group, tuple(target_aliases))
    # Check overlaps before exact exclusions so aliases cannot be dropped or a
    # case renamed into a purportedly unseen patient group.
    _cohort_record(members, (*exclusions, target))
    if not any(target.case_hash == record.case_hash and target.group_id == record.group_id
               and set(target.aliases) == set(record.aliases) for record in exclusions):
        raise ValueError("Target case/group/aliases were not predeclared excluded from population training")
    dimensions = tuple(source["dimensions"])
    if dimensions[2] != settings.hidden_features:
        raise ValueError("Population actor width disagrees with recorded pretraining settings")
    if expected_dimensions is not None and dimensions != tuple(expected_dimensions):
        raise ValueError("Population checkpoint observation dimensions differ")
    if expected_feature_schema is not None and provenance["feature_schema"] != expected_feature_schema:
        raise ValueError("Population checkpoint feature semantics differ")
    policy = learning.load_policy(path)
    if learning.policy_hash(policy) != provenance["policy_hash"]:
        raise ValueError("Population checkpoint weight hash mismatch")
    exposures = provenance.get("gradient_episode_sources", {})
    if any(exposures.get(member.case_hash, 0) < 1 for member in members):
        raise ValueError("A development member never contributed a completed gradient batch")
    return {"checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy_hash": source["policy_hash"], "provenance_hash": source["provenance_hash"],
        "provenance": provenance, "offline_pretraining": provenance["offline_pretraining"],
        "scope": provenance["scope"]}


def load_frozen_population_policy(checkpoint: str | Path, **validation: Any) -> learning.MaskedPatientPolicy:
    validate_population_checkpoint(checkpoint, **validation)
    policy = learning.clone_checkpoint_policy(checkpoint)
    policy.eval()
    policy.requires_grad_(False)
    return policy


class _PopulationSimulator:
    """Explicit pool wrapper; observations are exactly those of the active member."""
    def __init__(self, members: tuple[PopulationMember, ...], records: tuple[dict[str, Any], ...],
                 lookup: dict[int, tuple[int, int]], case_hash: str, model_hash: str):
        self.members, self.records, self.lookup = members, records, lookup
        self.case_hash, self.decision_model_hash = case_hash, model_hash
        self.world_generator_fingerprint = members[0].optimization_manifest.generator.fingerprint
        self.active = None
        self.active_index = None

    @property
    def optimization_source_hash(self) -> str:
        return self.active.case_hash

    def reset(self, seed: int = 0):
        if seed not in self.lookup:
            raise ValueError("Population reset seed is not a registered development world")
        self.active_index, inner_seed = self.lookup[seed]
        self.active = self.members[self.active_index].simulator_factory()
        self.assert_model_frozen()
        return self.active.reset(inner_seed)

    def assert_model_frozen(self) -> None:
        if self.active is not None:
            learning._assert_model(self.active, self.records[self.active_index]["decision_model_hash"])

    def step(self, action):
        return self.active.step(action)

    def metrics(self):
        return {**self.active.metrics(), "population_member_case_hash": self.active.case_hash,
                "population_scope": "analytic_synthetic_only"}


@learning._record_failures
def train_population_policy(members: Sequence[PopulationMember], *, exclusions: Sequence[PopulationIdentity],
                            config: learning.TrainingConfig, output_dir: str | Path,
                            cancelled=None, progress=None) -> dict[str, Any]:
    """Train one shared actor across development members and export latest weights.

    The fixed-budget latest-checkpoint rule is set before training. Development
    selection scores are logged by the shared learner but never change this rule.
    No final or stress world manifests can enter the member specification.
    """
    started = time.perf_counter()
    members = tuple(members)
    cohort = _cohort_record(tuple(member.identity for member in members), tuple(exclusions))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    sources = learning.numerical_source_hashes()
    runtime = {"python": platform.python_version(), "numpy": str(np.__version__),
               "torch": str(torch.__version__), "device": "cpu"}
    source_directory = directory / "source-snapshot" / "resectionlab"
    source_directory.mkdir(parents=True)
    for name, digest in sources.items():
        payload = (Path(__file__).parent / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Population numerical source changed while preserving bytes")
        (source_directory / name).write_bytes(payload)
    records, lookup, opt_seeds, sel_seeds = [], {}, [], []
    generator = members[0].optimization_manifest.generator
    schemas, dimensions = [], []
    for member in members:
        partitions = learning._validate_partitions(member.optimization_manifest, member.selection_manifest)
        if member.identity.case_hash != member.optimization_manifest.case_hash:
            raise ValueError("Population identity differs from its actual world case")
        simulator = member.simulator_factory()
        if not simulator.config.case_id.startswith("synthetic"):
            raise ValueError("This population training entry point accepts explicit analytic synthetic fixtures only")
        learning._assert_partition_binding(simulator, member.optimization_manifest)
        if simulator.world_generator_fingerprint != generator.fingerprint:
            raise ValueError("Development members must share the declared world sampler")
        observation = simulator.reset(member.optimization_manifest.seeds[0])
        schemas.append(feature_schema(simulator))
        dimensions.append((observation.action_features.shape[1], observation.state_features.size, config.hidden_features))
        factory_path = Path(inspect.getfile(member.simulator_factory))
        factory_bytes = factory_path.read_bytes()
        factory_hash = hashlib.sha256(factory_bytes).hexdigest()
        (directory / "source-snapshot" / (factory_hash + ".py")).write_bytes(factory_bytes)
        records.append({"identity": asdict(member.identity), "partitions": partitions,
                        "decision_model_hash": simulator.decision_model_hash,
                        "factory_source_sha256": factory_hash})
    if any(schema != schemas[0] for schema in schemas) or len(set(dimensions)) != 1:
        raise ValueError("Population members need one shared named observation schema")
    # Interleave members rather than assigning all early episodes to one case.
    for role, outer, offset in (("optimization_manifest", opt_seeds, 0), ("selection_manifest", sel_seeds, 1_000_000)):
        largest = max(len(getattr(member, role).seeds) for member in members)
        for world_index in range(largest):
            for member_index, member in enumerate(members):
                seeds = getattr(member, role).seeds
                if world_index < len(seeds):
                    seed = offset + len(outer)
                    lookup[seed] = (member_index, seeds[world_index])
                    outer.append(seed)
    pool_hash = content_hash({"cohort": cohort, "members": records})
    model_hash = content_hash({"pool": pool_hash, "world_map": lookup, "feature_schema": schemas[0]})
    optimization = WorldPartitionManifest(WorldRole.OPTIMIZATION, pool_hash, generator, tuple(opt_seeds))
    selection = WorldPartitionManifest(WorldRole.SELECTION, pool_hash, generator, tuple(sel_seeds))
    factory = lambda: _PopulationSimulator(members, tuple(records), lookup, pool_hash, model_hash)
    learning._atomic_json(directory / "population-design.json", {"cohort": cohort, "members": records,
        "config": asdict(config), "checkpoint_rule": "fixed_budget_latest", "final_worlds_used": False,
        "source_sha256": sources, "runtime": runtime, "world_lookup": lookup})
    preparation_seconds = time.perf_counter() - started
    result = learning.train_patient_policy(factory, optimization, selection, config=config,
        output_dir=directory / "training", cancelled=cancelled, progress=progress)
    after_training = time.perf_counter()
    training = json.loads((directory / "training/result.json").read_text())
    if learning.numerical_source_hashes() != sources:
        raise ValueError("Population numerical source changed during training; no checkpoint can be exported")
    for member, record in zip(members, records):
        if hashlib.sha256(Path(inspect.getfile(member.simulator_factory)).read_bytes()).hexdigest() != record["factory_source_sha256"]:
            raise ValueError("Population factory source changed during training")
    if result.status == "cancelled" or not result.gradient_steps or not training["actor_parameters_changed"]:
        record = {"status": result.status, "checkpoint_path": None, "reason": "no_completed_shared_actor_export",
                  "training": asdict(result), "scope": "analytic_synthetic_only"}
        learning._atomic_json(directory / "population-result.json", record)
        return record
    exposures: dict[str, int] = {}
    for update in training["optimization_history"]:
        for case_hash in update["episode_source_case_hashes"]:
            exposures[case_hash] = exposures.get(case_hash, 0) + 1
    if any(exposures.get(member.identity.case_hash, 0) < 1 for member in members):
        raise ValueError("Budget ended before every development member contributed an actual gradient batch")
    policy = learning.load_policy(directory / "training/checkpoint.pt", selected=False)
    policy_hash = learning.policy_hash(policy)
    provenance = {"scope": "analytic_synthetic_only", "cohort": cohort, "members": records,
        "development_case_hashes": [member.identity.case_hash for member in members],
        "training_run_hash": content_hash(training), "training_gradient_steps": result.gradient_steps,
        "training_config": asdict(config),
        "actor_parameters_changed": training["actor_parameters_changed"], "gradient_episode_sources": exposures,
        "initial_policy_hash": result.initial_checkpoint_hash, "policy_hash": policy_hash,
        "checkpoint_rule": "fixed_budget_latest", "feature_schema": schemas[0],
        "source_sha256": sources, "runtime": runtime, "final_worlds_used": False,
        "offline_pretraining": {"gradient_steps": result.gradient_steps,
            "optimization_environment_steps": result.optimization_environment_steps,
            "selection_environment_steps": result.selection_environment_steps,
            "elapsed_seconds": result.elapsed_seconds,
            "initialization_seconds": result.initialization_seconds,
            "elapsed_seconds_scope": "optimization and selection including initial selection; initialization and preparation separately measured",
            "preparation_seconds": preparation_seconds,
            "postprocessing_seconds_before_export": time.perf_counter() - after_training,
            "total_offline_seconds_before_checkpoint_write": time.perf_counter() - started},
        "limits": ["No clinical population training", "Procedural groups are not independent patients",
                   "Known declared identity overlap only; no assurance about undiscovered relationships"]}
    checkpoint = directory / "population.pt"
    export_started = time.perf_counter()
    learning._atomic_checkpoint(checkpoint, {"kind": "population_checkpoint", "schema_version": 1,
        "dimensions": list(policy.dimensions), "policy": copy.deepcopy(policy.state_dict()),
        "policy_hash": policy_hash, "population_provenance": provenance, "provenance_hash": content_hash(provenance)})
    record = {"status": "completed", "checkpoint_path": str(checkpoint.resolve()),
        "policy_hash": policy_hash, "provenance_hash": content_hash(provenance), "provenance": provenance,
        "checkpoint_export_seconds": time.perf_counter() - export_started,
        "total_offline_seconds": time.perf_counter() - started}
    learning._atomic_json(directory / "population-result.json", record)
    return record


def make_analytic_population_fixture() -> tuple[tuple[PopulationMember, ...], Callable[[], Any], PopulationIdentity]:
    """Two development anatomies; branching task excluded from shared training.

    A straight delayed-benefit corridor and a shallow target layer are different
    procedural groups. Their shared policy is tested on the previously studied
    branching fixture. This is a small software experiment, not patient sampling.
    """
    from .geometry import AccessWindow, ToolGeometry
    from .simulation import (SequentialSimulator, SimulationConfig,
                             make_branching_simulator, make_synthetic_simulator)

    corridor = make_synthetic_simulator()
    corridor = SequentialSimulator(replace(corridor.config, case_id="synthetic_population_corridor",
        source_hash="synthetic-population-corridor-v1"))
    tissue = np.ones((3, 3, 2), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[:, 1, 1] = 1
    narrow = ToolGeometry("shallow-narrow", .12, .12, 12., max_access_angle_deg=70, tip_length_mm=.15)
    wide = ToolGeometry("shallow-wide", .12, .45, 12., max_access_angle_deg=70, tip_length_mm=.15)
    layer = SequentialSimulator(SimulationConfig(tissue, labels, np.eye(4),
        AccessWindow((1., 1., -.5), (0., 0., 1.), 2.), (narrow, wide), max_steps=5,
        case_id="synthetic_population_shallow_layer", source_hash="synthetic-population-layer-v1",
        derivation={"track": "analytic_synthetic_population_curriculum"}))
    members = []
    for simulator, group, alias in ((corridor, "synthetic:corridor-family", "analytic:corridor-v1"),
                                     (layer, "synthetic:shallow-layer-family", "analytic:layer-v1")):
        identity = PopulationIdentity(simulator.case_hash, group, (alias,))
        panels = generate_partitions(simulator.case_hash, simulator.config.world_generator, 20261004,
            optimization=3, selection=2, final_evaluation=3, stress=2)
        members.append(PopulationMember(identity, simulator.fresh, panels.optimization, panels.selection))
    target = make_branching_simulator()
    excluded = PopulationIdentity(target.case_hash, "synthetic:branching-family", ("analytic:branching-v1",), "excluded")
    return tuple(members), target.fresh, excluded
