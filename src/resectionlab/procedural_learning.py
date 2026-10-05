"""Declared procedural-native initialization, distinct from population training.

Only the two registered generating recipes can contribute shared gradients.
The real target remains a previously studied structural development case. A
separate, explicitly nonpatient target supports contract tests without running
the registered experiment or weakening the public-target provenance gate.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import learning
from .policy_inputs import policy_input_profile
from .geometry import AccessWindow, ToolGeometry
from .native_resection import NATIVE_RESECTION_VERSION, NativeResectionConfig
from .native_simulation import NATIVE_ACTION_FEATURE_NAMES, NATIVE_ADAPTER_VERSION, NativeSequentialSimulator
from .simulation import RewardSpec
from .worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole, content_hash, generate_partitions

KIND = "procedural_pretraining_checkpoint"
SCOPE = "procedural_native_to_patient_development"
TEST_SCOPE = "procedural_native_contract_test_only"
DECLARATION_HASH = "sha256:3abf10162ed9d096a7e21ec84a87c1a88ebf12bb074345a78040dfdebb828ac9"
DECLARATION_PATH = Path(__file__).resolve().parents[2] / "manifests/experiments/procedural-native-to-ucsf-v1.json"
FEATURE_STUDY_ID = "procedural-native-feature-units-v1"
FEATURE_DECLARATION_HASH = "sha256:b19aac85f9241e37b98ae53d539ad923f3258bf42a106d67e520f37abcf6a237"


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: asdict(item) if hasattr(item, "__dataclass_fields__")
                               else item.tolist(), allow_nan=False))


def _declaration() -> dict[str, Any]:
    record = json.loads(DECLARATION_PATH.read_text())
    if (record.get("declaration_content_hash") != DECLARATION_HASH
            or content_hash({key: value for key, value in record.items() if key != "declaration_content_hash"}) != DECLARATION_HASH):
        raise ValueError("Procedural preregistration changed; declare a separate study")
    return record


def _study_binding(input_profile: str = "RAW", study_id: str | None = None) -> dict[str, Any]:
    """Bind a named input intervention separately from physical anatomy/worlds."""
    profile = policy_input_profile(input_profile)
    if study_id is None or study_id == "procedural-native-to-ucsf-v1":
        if input_profile != "RAW":
            raise ValueError("Original procedural study requires RAW inputs; declare the feature-unit study")
        identifier, digest, scope = "procedural-native-to-ucsf-v1", DECLARATION_HASH, SCOPE
    elif study_id == FEATURE_STUDY_ID:
        path = DECLARATION_PATH.with_name(FEATURE_STUDY_ID + ".json")
        record = json.loads(path.read_text())
        digest = content_hash({key: value for key, value in record.items() if key != "declaration_content_hash"})
        if record.get("declaration_content_hash") != FEATURE_DECLARATION_HASH or digest != FEATURE_DECLARATION_HASH:
            raise ValueError("Feature-unit study declaration changed")
        if record["reference_design"]["declaration_content_hash"] != DECLARATION_HASH:
            raise ValueError("Feature-unit physical source declaration differs")
        expected = {**profile.to_dict(), "profile_content_hash": profile.fingerprint}
        if record["profiles"].get(input_profile) != expected:
            raise ValueError("Feature-unit registry differs from the registered profile")
        identifier, scope = study_id, record["scope"]
    else:
        raise ValueError("Unknown procedural transfer study")
    return {"study_id": identifier, "study_declaration_hash": digest, "study_scope": scope,
            "input_profile": profile.to_dict(), "input_profile_hash": profile.fingerprint}


@dataclass(frozen=True)
class TransferTarget:
    case_hash: str
    planning_hash: str
    group_id: str
    aliases: tuple[str, ...] = ()
    source_kind: str = "public_patient_structural_mirror"
    outer_split: str = "development"
    excluded_from_pretraining: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        if (not all(isinstance(value, str) and value for value in (self.case_hash, self.planning_hash, self.group_id))
                or ":" not in self.group_id
                or any(not isinstance(value, str) or ":" not in value for value in self.aliases)
                or len(set(self.aliases)) != len(self.aliases)):
            raise ValueError("Transfer target requires immutable, distinct namespaced identities")
        if self.source_kind not in ("public_patient_structural_mirror", "procedural_test_only"):
            raise ValueError("Unsupported procedural transfer target domain")
        if self.outer_split != "development" or self.excluded_from_pretraining is not True:
            raise ValueError("Transfer targets must be development cases excluded from pretraining")


@dataclass(frozen=True)
class ProceduralMember:
    family_id: str
    aliases: tuple[str, ...]
    factory: Callable[[], NativeSequentialSimulator]
    optimization: WorldPartitionManifest
    selection: WorldPartitionManifest
    generator_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        # Store serialized bytes, not a caller-owned mutable nested dictionary.
        object.__setattr__(self, "generator_json", json.dumps(json.loads(self.generator_json), sort_keys=True))

    @property
    def generator_manifest(self) -> dict[str, Any]:
        return json.loads(self.generator_json)


def _build_recipe(record: dict[str, Any]) -> NativeSequentialSimulator:
    recipe = record["recipe"]
    if recipe["version"] != "native-procedural-families-v1" or recipe["tissue"] != "all_true":
        raise ValueError("Unsupported procedural generator recipe")
    tissue = np.ones(tuple(recipe["shape"]), bool)
    labels = np.zeros(tissue.shape, np.int16)
    for box in recipe["target_boxes_half_open"]:
        labels[tuple(slice(start, stop) for start, stop in box)] = recipe["target_label"]
    affine = np.asarray(recipe["affine"], float)
    source_hash = content_hash({"recipe": recipe, "tissue_mask": tissue, "target_labels": labels, "affine": affine})
    if source_hash != record["source_hash"]:
        raise ValueError("Procedural source bytes do not reproduce the declared hash")
    native = NativeResectionConfig(tissue, labels, affine, AccessWindow(**recipe["access"]),
        tuple(ToolGeometry(**tool) for tool in recipe["tools"]), source_hash,
        record["tissue_support_provenance"], case_id=recipe["case_id"])
    simulator = NativeSequentialSimulator(native, recipe["candidate_targets_mm"],
        candidate_entries_mm=recipe["candidate_entries_mm"], reward=RewardSpec(),
        max_steps=recipe["max_steps"], max_actions=recipe["max_actions"],
        partial_contact_weight=recipe["partial_contact_weight"])
    if (record.get("model_hash", simulator.decision_model_hash) != simulator.decision_model_hash
            or record.get("native_config_hash", native.fingerprint) != native.fingerprint):
        raise ValueError("Procedural native model differs from preregistered generating geometry")
    return simulator


def _build_serialized_recipe(payload: str) -> NativeSequentialSimulator:
    return _build_recipe(json.loads(payload))


def _test_target_record() -> dict[str, Any]:
    record = copy.deepcopy(_declaration()["procedural_training"]["members"][0])
    recipe = record["recipe"]
    recipe["case_id"] = "synthetic_native_contract_target_v1"
    recipe["family_id"] = "procedural_test:native-target-v1"
    recipe["target_boxes_half_open"] = [[[2, 7], [2, 7], [3, 7]]]
    tissue = np.ones(tuple(recipe["shape"]), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[2:7, 2:7, 3:7] = 1
    record["source_hash"] = content_hash({"recipe": recipe, "tissue_mask": tissue,
        "target_labels": labels, "affine": np.asarray(recipe["affine"], float)})
    record.pop("model_hash")
    record.pop("native_config_hash")
    record["group_id"] = recipe["family_id"]
    record["aliases"] = ["procedural_test_generator:native-target-v1"]
    return record


def make_procedural_test_target() -> tuple[TransferTarget, Callable[[], NativeSequentialSimulator]]:
    """A distinct domain for tiny contract tests; cannot stand in for a patient."""
    record = _test_target_record()
    target = TransferTarget(record["source_hash"], record["source_hash"], record["group_id"],
                            tuple(record["aliases"]), "procedural_test_only")
    return target, partial(_build_serialized_recipe, json.dumps(record, sort_keys=True))


def _target_scope(target: TransferTarget) -> str:
    if not isinstance(target, TransferTarget):
        raise ValueError("Transfer target must carry typed source and exclusion provenance")
    if target.source_kind == "procedural_test_only":
        expected, _ = make_procedural_test_target()
        scope = TEST_SCOPE
    else:
        declared = _declaration()["target"]
        expected = TransferTarget(declared["semantic_hash"], declared["planning_hash"], declared["group_id"],
                                  tuple(declared["aliases"]), declared["source_kind"])
        scope = SCOPE
    if target != expected:
        raise ValueError("Transfer target case/group/aliases/domain differs from its declared exclusion")
    return scope


def _partition(record: dict[str, Any]) -> WorldPartitionManifest:
    return WorldPartitionManifest(WorldRole(record["role"]), record["case_hash"],
        WorldGeneratorConfig(**record["generator"]), tuple(record["seeds"]), record.get("planning_hash"))


def make_native_procedural_fixture(target: TransferTarget, *, input_profile: str = "RAW",
                                  study_id: str | None = None) -> tuple[ProceduralMember, ...]:
    _study_binding(input_profile, study_id)
    _target_scope(target)
    result = []
    for record in _declaration()["procedural_training"]["members"]:
        payload = json.dumps(record, sort_keys=True)
        result.append(ProceduralMember(record["group_id"], tuple(record["aliases"]),
            partial(_build_serialized_recipe, payload), _partition(record["world_partitions"]["optimization"]),
            _partition(record["world_partitions"]["selection"]), payload))
    return tuple(result)


def native_observation_schema(simulator: Any) -> dict[str, Any]:
    """Name actual units and transforms; anatomy provenance lives separately."""
    if type(simulator) is not NativeSequentialSimulator:
        raise ValueError("Procedural transfer requires the exact native observation backend")
    simulator.assert_model_frozen()
    observation = simulator.observation()
    if observation.action_features.shape[1] != 15 or observation.state_features.shape != (6,):
        raise ValueError("Native observation dimensions differ from the 15+6 contract")
    return _plain({"version": "native-15x6-actual-semantics-v1", "action_feature_names": NATIVE_ACTION_FEATURE_NAMES,
        "action_feature_units": ["binary", "mm3", "mm3", "spatial_surrogate_mm3", "spatial_surrogate_mm3",
            "mm", "mm", "mm", "binary", "fraction_of_action_budget", "fraction", "fraction", "mm3",
            "spatial_surrogate_mm3", "spatial_surrogate_mm3"],
        "state_feature_names": ["remaining_target_fraction", "action_budget_fraction_used", "removed_tissue_fraction",
            "nominal_severed_edge_fraction", "motor_evidence_available", "language_evidence_available"],
        "state_feature_units": ["fraction", "fraction", "fraction", "fraction", "binary", "binary"],
        "scaling": "identity; raw physical volumes/distances and existing fractions; no learned normalization",
        "depth_semantics": "historical name; completed nonSTOP actions / max_steps; not physical depth",
        "actor_evidence": "nominal fields and actual cavity only; hidden world excluded",
        "adapter_version": NATIVE_ADAPTER_VERSION, "removal_version": NATIVE_RESECTION_VERSION,
        "reward": asdict(simulator.config.reward), "partial_contact_weight": simulator.partial_contact_weight,
        "tools": [asdict(tool) for tool in simulator.native_config.tools],
        "max_steps": simulator.config.max_steps, "max_actions": simulator.config.max_actions,
        "evidence_available": simulator.config.evidence_available,
        "world_generator": asdict(simulator.config.world_generator),
        "removal_semantics": "fully contained exterior-connected native cells only; partial cells retained and costed"})


def _tokens(case_hash: str, group: str, aliases: Sequence[str]) -> set[str]:
    return {"case:" + case_hash, "identity:" + group, *("identity:" + alias for alias in aliases)}


def _validate_members(members: Sequence[ProceduralMember], targets: Sequence[TransferTarget]) -> tuple[list[dict], dict]:
    expected = _declaration()["procedural_training"]["members"]
    if len(members) != len(expected) or len(targets) != 1:
        raise ValueError("The procedural study requires exactly its two families and one excluded target")
    _target_scope(targets[0])
    target_tokens = _tokens(targets[0].case_hash, targets[0].group_id, targets[0].aliases)
    seen: set[str] = set()
    records, schemas = [], []
    for member, declared in zip(members, expected):
        if (member.generator_manifest != declared or member.family_id != declared["group_id"]
                or member.aliases != tuple(declared["aliases"])):
            raise ValueError("Procedural family/alias/recipe differs from the declared source generator")
        tokens = _tokens(declared["source_hash"], member.family_id, member.aliases)
        if tokens & (target_tokens | seen):
            raise ValueError("Procedural source/target group or alias overlaps")
        seen |= tokens
        partitions = learning._validate_partitions(member.optimization, member.selection)
        if (member.optimization.to_dict() != _partition(declared["world_partitions"]["optimization"]).to_dict()
                or member.selection.to_dict() != _partition(declared["world_partitions"]["selection"]).to_dict()):
            raise ValueError("Procedural worlds differ from declared optimization/selection partitions")
        simulator = member.factory()
        learning._assert_partition_binding(simulator, member.optimization)
        learning._assert_model(simulator, declared["model_hash"])
        # Rebuild from source recipe: a source_kind string cannot bless patient arrays.
        reference = _build_recipe(declared)
        if simulator.native_config.fingerprint != reference.native_config.fingerprint:
            raise ValueError("Procedural factory substituted anatomy, tool or access geometry")
        schemas.append(native_observation_schema(simulator))
        records.append({"family_id": member.family_id, "aliases": list(member.aliases),
            "generator_manifest": declared, "partitions": partitions,
            "decision_model_hash": simulator.decision_model_hash})
    if any(schema != schemas[0] for schema in schemas):
        raise ValueError("Procedural families disagree on native feature/reward/action semantics")
    return records, schemas[0]


def _runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": str(np.__version__),
            "torch": str(torch.__version__), "device": "cpu"}


def _validate_budget(config: learning.TrainingConfig, target: TransferTarget, *, online: bool,
                     input_profile: str = "RAW", study_id: str | None = None) -> None:
    _study_binding(input_profile, study_id)
    if _target_scope(target) == SCOPE:
        declaration = _declaration()
        key = "online_scratch_and_adapted_each_seed" if online else "offline_training"
        expected = dict(declaration["budgets"][key])
        if online:
            if config.seed not in declaration["policy"]["optimization_seeds_online"]:
                raise ValueError("Online seed differs from procedural preregistration")
            expected["seed"] = config.seed
        if asdict(config) != expected:
            raise ValueError("Procedural training budget/settings differ from the registered experiment")


def _pool_layout(members):
    lookup, opt, sel = {}, [], []
    for attribute, output, offset in (("optimization", opt, 0), ("selection", sel, 1000000)):
        for world_index in range(max(len(getattr(member, attribute).seeds) for member in members)):
            for index, member in enumerate(members):
                seeds = getattr(member, attribute).seeds
                if world_index < len(seeds):
                    outer = offset + len(output)
                    lookup[outer] = (index, seeds[world_index])
                    output.append(outer)
    return lookup, opt, sel


class _ProceduralPool:
    def __init__(self, members, records, lookup, case_hash, model_hash):
        self.members, self.records, self.lookup = members, records, lookup
        self.case_hash, self.decision_model_hash = case_hash, model_hash
        self.world_generator_fingerprint = members[0].optimization.generator.fingerprint
        self.active = None
        self.active_index = None

    @property
    def optimization_source_hash(self):
        return self.active.case_hash

    def reset(self, seed=0):
        if seed not in self.lookup:
            raise ValueError("Procedural pool refuses undeclared or final/stress worlds")
        self.active_index, inner = self.lookup[seed]
        self.active = self.members[self.active_index].factory()
        self.assert_model_frozen()
        return self.active.reset(inner)

    def assert_model_frozen(self):
        if self.active is not None:
            learning._assert_model(self.active, self.records[self.active_index]["decision_model_hash"])

    def step(self, action):
        self.assert_model_frozen()
        return self.active.step(action)

    def metrics(self):
        return {**self.active.metrics(), "pretraining_source_domain": "procedural_native",
                "procedural_source_hash": self.active.case_hash}


@learning._record_failures
def train_procedural_native_policy(members: Sequence[ProceduralMember], *, excluded_targets: Sequence[TransferTarget],
        config: learning.TrainingConfig, output_dir: str | Path, cancelled=None, progress=None,
        input_profile: str = "RAW", study_id: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    study = _study_binding(input_profile, study_id)
    members, targets = tuple(members), tuple(excluded_targets)
    records, schema = _validate_members(members, targets)
    target = targets[0]
    _validate_budget(config, target, online=False, input_profile=input_profile, study_id=study_id)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    sources = learning.numerical_source_hashes()
    source_bytes = {name: (Path(__file__).parent / name).read_text() for name in sources}
    if any(hashlib.sha256(value.encode()).hexdigest() != sources[name] for name, value in source_bytes.items()):
        raise ValueError("Procedural numerical source changed before preserving exact source bytes")
    source_dir = directory / "source-snapshot" / "resectionlab"
    source_dir.mkdir(parents=True)
    for name, payload in source_bytes.items():
        (source_dir / name).write_text(payload)
    learning._atomic_json(directory / "declaration.json", _declaration())
    if study_id == FEATURE_STUDY_ID:
        (directory / "input-study-declaration.json").write_bytes(DECLARATION_PATH.with_name(FEATURE_STUDY_ID + ".json").read_bytes())
    lookup, opt, sel = _pool_layout(members)
    scope = _target_scope(target)
    pool_hash = content_hash({"members": records, "excluded_targets": [asdict(target)], "scope": scope})
    model_hash = content_hash({"pool": pool_hash, "world_map": lookup, "schema": schema})
    generator = members[0].optimization.generator
    optimization = WorldPartitionManifest(WorldRole.OPTIMIZATION, pool_hash, generator, tuple(opt))
    selection = WorldPartitionManifest(WorldRole.SELECTION, pool_hash, generator, tuple(sel))
    factory = lambda: _ProceduralPool(members, records, lookup, pool_hash, model_hash)
    design = _plain({"scope": scope, **study, "members": records, "excluded_targets": [asdict(target)],
        "training_config": asdict(config), "world_map": lookup, "observation_schema": schema,
        "declaration_hash": DECLARATION_HASH, "checkpoint_rule": "fixed_budget_latest",
        "source_sha256": sources, "runtime": _runtime(), "final_worlds_used": False})
    learning._atomic_json(directory / "procedural-design.json", design)
    preparation_seconds = time.perf_counter() - started
    result = learning.train_patient_policy(factory, optimization, selection, config=config,
        output_dir=directory / "training", cancelled=cancelled, progress=progress, input_profile=input_profile)
    after_training = time.perf_counter()
    training = json.loads((directory / "training/result.json").read_text())
    if learning.numerical_source_hashes() != sources:
        raise ValueError("Procedural numerical source changed during training; no export")
    if (result.status == "cancelled" or result.gradient_steps < 1
            or training["actor_parameters_changed"] is not True):
        record = {"status": result.status, "checkpoint_path": None, "scope": scope,
                  "reason": "no_completed_shared_actor_export", "training": asdict(result)}
        learning._atomic_json(directory / "procedural-result.json", record)
        return record
    exposures: dict[str, int] = {}
    for update in training["optimization_history"]:
        for source in update["episode_source_case_hashes"]:
            exposures[source] = exposures.get(source, 0) + 1
    if any(exposures.get(record["generator_manifest"]["source_hash"], 0) < 1 for record in records):
        raise ValueError("Every procedural family must contribute actual completed gradient batches")
    policy = learning.load_policy(directory / "training/checkpoint.pt", selected=False,
                                  expected_input_profile=input_profile)
    policy_hash = learning.policy_hash(policy)
    provenance = {**design, "human_patients_in_pretraining": 0, "clinical_population_training": False,
        "training_gradient_steps": result.gradient_steps, "actor_parameters_changed": True,
        "gradient_episode_sources": exposures, "training_run_hash": content_hash(training),
        "training_record": training, "policy_hash": policy_hash,
        "initial_policy_hash": result.initial_checkpoint_hash, "source_snapshot": source_bytes,
        "initial_trainable_parameter_hash": training["initial_trainable_parameter_hash"],
        "trainable_parameter_hash": learning.trainable_parameter_hash(policy),
        "offline_pretraining": {"gradient_steps": result.gradient_steps,
            "optimization_environment_steps": result.optimization_environment_steps,
            "selection_environment_steps": result.selection_environment_steps,
            "elapsed_seconds": result.elapsed_seconds, "preparation_seconds": preparation_seconds,
            "initialization_seconds": result.initialization_seconds,
            "elapsed_seconds_scope": "optimization and selection including initial selection; initialization and preparation separately measured",
            "postprocessing_seconds_before_export": time.perf_counter() - after_training,
            "total_offline_seconds_before_checkpoint_write": time.perf_counter() - started}}
    checkpoint = directory / "procedural.pt"
    export_started = time.perf_counter()
    learning._atomic_checkpoint(checkpoint, {"kind": KIND, "schema_version": 1,
        **policy.checkpoint_profile(), "trainable_parameter_hash": learning.trainable_parameter_hash(policy),
        "dimensions": list(policy.dimensions), "policy": copy.deepcopy(policy.state_dict()),
        "policy_hash": policy_hash, "procedural_provenance": provenance, "provenance_hash": content_hash(provenance)})
    record = {"status": "completed", "checkpoint_path": str(checkpoint.resolve()), "scope": scope,
        "policy_hash": policy_hash, "provenance_hash": content_hash(provenance),
        "provenance": {key: value for key, value in provenance.items() if key != "source_snapshot"},
        "checkpoint_export_seconds": time.perf_counter() - export_started,
        "total_offline_seconds": time.perf_counter() - started}
    learning._atomic_json(directory / "procedural-result.json", record)
    return record


def _validate_target_simulator(target: TransferTarget, simulator: NativeSequentialSimulator) -> None:
    scope = _target_scope(target)
    if type(simulator) is not NativeSequentialSimulator or simulator.case_hash != target.case_hash:
        raise ValueError("Transfer target does not match actual native simulator source")
    if scope == SCOPE:
        expected = _declaration()["target"]
        model, native = expected["decision_model_hash"], expected["native_config_hash"]
    else:
        _, factory = make_procedural_test_target()
        reference = factory()
        model, native = reference.decision_model_hash, reference.native_config.fingerprint
    learning._assert_model(simulator, model)
    if simulator.native_config.fingerprint != native:
        raise ValueError("Target native anatomy/tool/window differs from declared model")


def validate_procedural_checkpoint(checkpoint: str | Path, *, target: TransferTarget,
        simulator: NativeSequentialSimulator, hidden_features: int, input_profile: str = "RAW",
        study_id: str | None = None) -> dict[str, Any]:
    study = _study_binding(input_profile, study_id)
    _validate_target_simulator(target, simulator)
    path = Path(checkpoint)
    payload = path.read_bytes()
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("kind") != KIND or source.get("schema_version") != 1:
        raise ValueError("Procedural transfer requires its distinct provenance-bearing checkpoint kind")
    learning.checkpoint_input_profile(source, expected_input_profile=input_profile)
    provenance = source.get("procedural_provenance", {})
    if source.get("provenance_hash") != content_hash(provenance):
        raise ValueError("Procedural provenance hash changed")
    if any(provenance.get(key) != value for key, value in study.items()):
        # Pre-intervention RAW exports lack these fields. They remain subject
        # to the existing exact source/runtime provenance check below.
        legacy = input_profile == "RAW" and study_id is None and all(key not in provenance for key in study)
        if not legacy:
            raise ValueError("Procedural input profile or study provenance differs")
    if (provenance.get("scope") != _target_scope(target) or provenance.get("declaration_hash") != DECLARATION_HASH
            or provenance.get("excluded_targets") != [_plain(asdict(target))]
            or type(provenance.get("human_patients_in_pretraining")) is not int
            or provenance.get("human_patients_in_pretraining") != 0
            or provenance.get("clinical_population_training") is not False
            or provenance.get("final_worlds_used") is not False
            or provenance.get("checkpoint_rule") != "fixed_budget_latest"):
        raise ValueError("Procedural source domain, excluded target, world roles or export rule contradict declaration")
    members = make_native_procedural_fixture(target)
    records, schema = _validate_members(members, (target,))
    if provenance.get("members") != _plain(records):
        raise ValueError("Procedural source recipes, family aliases or world provenance differ")
    lookup, _opt, sel = _pool_layout(members)
    if provenance.get("world_map") != _plain(lookup):
        raise ValueError("Procedural pooled world mapping differs from declared optimization/selection roles")
    if provenance.get("observation_schema") != schema or schema != native_observation_schema(simulator):
        raise ValueError("Native feature/reward/action semantics changed across procedural transfer")
    settings = provenance.get("training_config")
    if not isinstance(settings, dict) or set(settings) != set(asdict(learning.TrainingConfig())):
        raise ValueError("Procedural training settings must be explicit and complete")
    config = learning.TrainingConfig(**settings)
    _validate_budget(config, target, online=False, input_profile=input_profile, study_id=study_id)
    if source.get("dimensions") != [15, 6, hidden_features] or config.hidden_features != hidden_features:
        raise ValueError("Procedural checkpoint dimensions or actor width differ")
    if provenance.get("runtime") != _runtime():
        raise ValueError("Procedural checkpoint runtime differs from frozen implementation")
    sources = provenance.get("source_sha256", {})
    snapshots = provenance.get("source_snapshot", {})
    if sources != learning.numerical_source_hashes() or set(snapshots) != set(sources):
        raise ValueError("Procedural source implementation differs from saved numerical contract")
    if any(not isinstance(snapshots[name], str) or hashlib.sha256(snapshots[name].encode()).hexdigest() != digest
           for name, digest in sources.items()):
        raise ValueError("Procedural exact source snapshot bytes are inconsistent")
    training = provenance.get("training_record", {})
    offline = provenance.get("offline_pretraining", {})
    if (provenance.get("training_run_hash") != content_hash(training)
            or training.get("actor_parameters_changed") is not True
            or provenance.get("actor_parameters_changed") is not True
            or training.get("status") not in ("gradient_budget", "environment_budget", "wall_time_budget")
            or training.get("latest_actor_hash") == training.get("initial_actor_hash")):
        raise ValueError("Procedural training record lacks completed actual actor updates")
    for name in ("gradient_steps", "optimization_environment_steps", "selection_environment_steps"):
        if type(offline.get(name)) is not int or offline[name] < 0 or offline[name] != training.get(name):
            raise ValueError("Procedural offline counters contradict saved optimization record")
    if (not 0 < offline["gradient_steps"] <= config.max_gradient_steps
            or type(provenance.get("training_gradient_steps")) is not int
            or offline["gradient_steps"] != provenance.get("training_gradient_steps")
            or not 0 < offline["optimization_environment_steps"] <= config.max_environment_steps):
        raise ValueError("Procedural offline updates or samples violate their declared budget")
    for name in ("elapsed_seconds", "preparation_seconds", "initialization_seconds", "postprocessing_seconds_before_export", "total_offline_seconds_before_checkpoint_write"):
        if type(offline.get(name)) not in (int, float) or not math.isfinite(offline[name]) or offline[name] < 0:
            raise ValueError("Procedural offline elapsed cost must be finite and nonnegative")
    if (offline["elapsed_seconds"] != training.get("elapsed_seconds")
            or offline["initialization_seconds"] != training.get("initialization_seconds")
            or offline["total_offline_seconds_before_checkpoint_write"] + 1e-6 <
                offline["elapsed_seconds"] + offline["preparation_seconds"] + offline["initialization_seconds"] + offline["postprocessing_seconds_before_export"]):
        raise ValueError("Procedural offline timing omits declared work")
    exposures: dict[str, int] = {}
    updates = training.get("optimization_history", [])
    if len(updates) != offline["gradient_steps"]:
        raise ValueError("Procedural gradient history does not match optimizer-step count")
    previous_steps = 0
    for number, update in enumerate(updates, 1):
        if (type(update.get("gradient_steps")) is not int or update["gradient_steps"] != number
                or type(update.get("optimization_environment_steps")) is not int
                or not previous_steps < update["optimization_environment_steps"] <= offline["optimization_environment_steps"]
                or not 0 < len(update.get("episode_source_case_hashes", [])) <= config.episodes_per_update):
            raise ValueError("Procedural completed gradient history counters are inconsistent")
        previous_steps = update["optimization_environment_steps"]
        for name in ("mean_return", "loss", "gradient_norm_before_clip", "actor_gradient_norm_after_clip"):
            value = update.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or ("norm" in name and value < 0):
                raise ValueError("Procedural gradient/return history contains invalid numerical statistics")
        for source_hash in update["episode_source_case_hashes"]:
            exposures[source_hash] = exposures.get(source_hash, 0) + 1
    expected_sources = {record["generator_manifest"]["source_hash"] for record in records}
    if (set(exposures) != expected_sources or any(value < 1 for value in exposures.values())
            or provenance.get("gradient_episode_sources") != exposures):
        raise ValueError("Procedural updates used an undeclared source or missed a required family")
    policy = learning.load_policy(path, expected_input_profile=input_profile)
    policy_hash = learning.policy_hash(policy)
    if (policy_hash != source.get("policy_hash") or policy_hash != provenance.get("policy_hash")
            or policy_hash != training.get("latest_checkpoint_hash")
            or policy_hash == provenance.get("initial_policy_hash")
            or provenance.get("initial_policy_hash") != training.get("initial_checkpoint_hash")):
        raise ValueError("Procedural latest-checkpoint selection or weight binding changed")
    if "input_profile" in provenance:
        if (training.get("input_profile") != study["input_profile"]
                or training.get("input_profile_hash") != study["input_profile_hash"]
                or provenance.get("trainable_parameter_hash") != learning.trainable_parameter_hash(policy)
                or source.get("trainable_parameter_hash") != learning.trainable_parameter_hash(policy)
                or training.get("latest_trainable_parameter_hash") != learning.trainable_parameter_hash(policy)
                or provenance.get("initial_trainable_parameter_hash") != training.get("initial_trainable_parameter_hash")):
            raise ValueError("Procedural input profile or paired tensor accounting changed")
    selection_history = training.get("selection_history", [])
    if (not selection_history or selection_history[0].get("gradient_steps") != 0
            or any(record.get("world_count") != len(sel) or record.get("gradient_steps", -1) > offline["gradient_steps"]
                   for record in selection_history)):
        raise ValueError("Procedural checkpoint ranking contains an incomplete or inconsistent selection panel")
    if path.read_bytes() != payload:
        raise ValueError("Procedural checkpoint bytes changed during validation")
    return {"checkpoint_file_sha256": hashlib.sha256(payload).hexdigest(), "policy_hash": policy_hash,
        **study, "trainable_parameter_hash": learning.trainable_parameter_hash(policy),
        "provenance_hash": source["provenance_hash"], "scope": provenance["scope"],
        "provenance": {key: value for key, value in provenance.items() if key != "source_snapshot"},
        "offline_pretraining": offline}


def load_frozen_procedural_policy(checkpoint: str | Path, **validation: Any) -> learning.MaskedPatientPolicy:
    validated = validate_procedural_checkpoint(checkpoint, **validation)
    policy = learning.clone_checkpoint_policy(checkpoint)
    if learning.policy_hash(policy) != validated["policy_hash"]:
        raise ValueError("Procedural policy changed between validation and loading")
    policy.eval()
    policy.requires_grad_(False)
    return policy


def validate_procedural_world_panels(target: TransferTarget, optimization: WorldPartitionManifest,
                                     selection: WorldPartitionManifest) -> dict[str, Any]:
    """Check whole declared panels before training, independent of a checkpoint.

    JSON stores immutable generator vectors as arrays, while typed manifests
    retain tuples. Canonical serialization preserves every key and exact value
    while making those two representations comparable. Nothing is rounded or
    omitted: roles, ordered seeds, identities and generator fields all bind.
    """
    learning._validate_partitions(optimization, selection)
    if _target_scope(target) == SCOPE:
        expected = _declaration()["target"]["world_partitions"]
    else:
        panels = generate_partitions(target.case_hash, WorldGeneratorConfig(),
            20261004, optimization=3, selection=2, final_evaluation=3, stress=2,
            planning_hash=target.planning_hash)
        expected = panels.to_dict()
    for role, panel in (("optimization", optimization), ("selection", selection)):
        if content_hash(panel.to_dict()) != content_hash(expected[role]):
            raise ValueError(f"Procedural {role} worlds differ from the complete registered panel")
    return {"declaration_hash": DECLARATION_HASH, "target_case_hash": target.case_hash,
            "optimization_partition_hash": optimization.partition_hash,
            "selection_partition_hash": selection.partition_hash, "final_worlds_used": False}


def validate_procedural_target_worlds(target, simulator, optimization, selection, config, *,
        input_profile: str = "RAW", study_id: str | None = None) -> dict[str, Any]:
    """Public-model preflight before any offline/online optimization or Adam."""
    _validate_target_simulator(target, simulator)
    study = _study_binding(input_profile, study_id)
    learning._validate_simulator_profile(simulator, input_profile)
    _validate_budget(config, target, online=True, input_profile=input_profile, study_id=study_id)
    learning._assert_partition_binding(simulator, optimization)
    learning._assert_partition_binding(simulator, selection)
    receipt = validate_procedural_world_panels(target, optimization, selection)
    return {**receipt, **study, "decision_model_hash": simulator.decision_model_hash,
            "training_config_hash": content_hash(asdict(config))}


def validate_procedural_adaptation(checkpoint, target, simulator, optimization, selection, config, *,
        input_profile: str = "RAW", study_id: str | None = None):
    """The shared learner calls this gate itself, before constructing Adam."""
    validate_procedural_target_worlds(target, simulator, optimization, selection, config,
        input_profile=input_profile, study_id=study_id)
    validation = validate_procedural_checkpoint(checkpoint, target=target,
        simulator=simulator, hidden_features=config.hidden_features, input_profile=input_profile, study_id=study_id)
    return validation


def train_procedural_adapted_policy(factory, optimization, selection, *, checkpoint, target,
        config, output_dir, cancelled=None, progress=None, resume=False,
        input_profile: str = "RAW", study_id: str | None = None):
    return learning.train_patient_policy(factory, optimization, selection, config=config,
        output_dir=output_dir, cancelled=cancelled, progress=progress, resume=resume,
        procedural_checkpoint=None if resume else checkpoint, procedural_target=target,
        input_profile=input_profile, procedural_study_id=study_id)
