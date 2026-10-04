#!/usr/bin/env python3
"""Measure one experimental native action model; never train or compare policies.

The public CLI requires --execute and always starts a fresh subprocess/output.
A complete greedy sequence is frozen before two deterministic selection replays.
Only complete, independently certified results become an eligible candidate.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from resectionlab.core import array_digest
from resectionlab.evaluation import independent_check_native_history
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, AXIS_ADAPTER_VERSION
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, native_config_from_case
from resectionlab.simulation import RewardSpec
from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole, content_hash

REFERENCE_PATH = Path("manifests/experiments/procedural-native-to-ucsf-v1.json")
REFERENCE_HASH = "sha256:3abf10162ed9d096a7e21ec84a87c1a88ebf12bb074345a78040dfdebb828ac9"
DECLARATION_PATH = Path("manifests/experiments/native-axis-preflight-v2.json")
DECLARATION_HASH = "sha256:089dca680b992a5808c104a559ab562a034de9bea91de6df939da40d87650fe3"


def write_json(path: Path, value) -> None:
    def encode(item):
        if isinstance(item, Mapping):
            return dict(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=encode, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_snapshot(root: Path = ROOT) -> dict:
    files = {*root.glob("src/resectionlab/**/*.py"), root / "scripts/preflight_native_axis.py",
             root / REFERENCE_PATH, root / DECLARATION_PATH, root / "requirements-lock.txt",
             root / "pyproject.toml"}
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(files) if path.is_file()}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "nibabel")}
    return {"file_sha256": hashes, "runtime_content_hash": content_hash({"files": hashes, "versions": versions, "python": sys.version}),
            "runtime_versions": versions,
            "python": sys.version, "platform": platform.platform(), "source_root": str(root)}


def assert_source(snapshot: dict, root: Path = ROOT) -> None:
    if source_snapshot(root)["runtime_content_hash"] != snapshot["runtime_content_hash"]:
        raise RuntimeError("SOURCE_CHANGED_DURING_PREFLIGHT")


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


@dataclass(frozen=True)
class PreflightBudget:
    worker_wall_seconds: float = 600.
    process_peak_rss_bytes: int = 6 * 1024**3
    hard_termination_grace_seconds: float = 15.

    def __post_init__(self):
        if (not np.isfinite(self.worker_wall_seconds) or self.worker_wall_seconds <= 0
                or type(self.process_peak_rss_bytes) is not int or self.process_peak_rss_bytes <= 0
                or not np.isfinite(self.hard_termination_grace_seconds)
                or self.hard_termination_grace_seconds < 0):
            raise ValueError("Preflight resource limits must be finite and positive")


class ResourceGuard:
    """Cooperative bounds; parent process supplies a separate hard wall timeout."""
    def __init__(self, budget: PreflightBudget, *, clock=time.perf_counter, memory=peak_rss_bytes):
        self.budget, self.clock, self.memory = budget, clock, memory
        self.started = clock()
        self.calls, self.callback_seconds, self.peak_rss = 0, 0., 0
        self.reason = None
        self.requested_at = None
        self.probe = False

    def __call__(self) -> bool:
        start = self.clock()
        self.calls += 1
        self.peak_rss = max(self.peak_rss, self.memory())
        if self.reason is None:
            if start - self.started >= self.budget.worker_wall_seconds:
                self.reason = "worker_wall_budget"
            elif self.peak_rss > self.budget.process_peak_rss_bytes:
                self.reason = "process_peak_rss_budget"
            if self.reason:
                self.requested_at = start
        cancelled = self.reason is not None or self.probe
        self.callback_seconds += self.clock() - start
        return cancelled

    def require(self):
        if self():
            raise InterruptedError(self.reason or "deliberate_boundary_probe")

    def cancel(self, reason):
        if self.reason is None:
            self.reason, self.requested_at = reason, self.clock()

    def receipt(self) -> dict:
        now = self.clock()
        return {"budget": asdict(self.budget), "elapsed_seconds": now - self.started,
                "cancellation_callback_calls": self.calls, "cancellation_callback_seconds": self.callback_seconds,
                "observed_peak_rss_bytes": max(self.peak_rss, self.memory()), "cancellation_reason": self.reason,
                "budget_request_to_receipt_seconds": None if self.requested_at is None else now - self.requested_at,
                "scope": "worker process RSS; cooperative callback CPU cost; no OS memory reservation"}


def _checked_manifest(path: Path, expected_hash: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    supplied = body.pop("declaration_content_hash", None)
    if supplied != expected_hash or content_hash(body) != expected_hash:
        raise ValueError("Preflight declaration differs from the committed declaration")
    return value


def _partition(value: dict) -> WorldPartitionManifest:
    result = WorldPartitionManifest(value["role"], value["case_hash"],
        WorldGeneratorConfig(**value["generator"]), tuple(value["seeds"]), value.get("planning_hash"))
    if content_hash(result.to_dict()) != content_hash(value):
        raise ValueError("World manifest has extra or changed fields")
    return result


def validate_panel(case, generator, optimization, selection) -> None:
    if optimization.role is not WorldRole.OPTIMIZATION or selection.role is not WorldRole.SELECTION:
        raise ValueError("Only optimization and selection worlds may be opened")
    if len(selection.seeds) != 2 or set(optimization.seeds) & set(selection.seeds):
        raise ValueError("Preflight needs two distinct, disjoint selection worlds")
    for panel in (optimization, selection):
        if (panel.case_hash != case.semantic_hash or panel.planning_hash != case.planning_hash
                or panel.generator.fingerprint != generator.fingerprint or not panel.generator.deterministic):
            raise ValueError("Preflight worlds must bind this case and deterministic generator")


def assert_public_action_model(sim, declaration: dict, reference: dict) -> None:
    """Defaults are convenient construction syntax, never an implicit contract."""
    if type(sim) is not AxisColumnNativeSimulator:
        raise TypeError("Public preflight requires the exact axis backend")
    sim.assert_model_frozen()
    declared = declaration["action_model"]
    actual = {"adapter_version": AXIS_ADAPTER_VERSION, "proposal_rule": asdict(sim.proposal_config),
        "max_cuts": sim.config.max_steps, "max_actions_including_stop": sim.config.max_actions}
    for key, value in actual.items():
        if content_hash(value) != content_hash(declared[key]):
            raise ValueError(f"Actual public action model differs from declaration: {key}")
    if (declared["input_profile"] != "RAW" or tuple(sim.config.evidence_available) != (False, False)
            or content_hash(asdict(sim.config.reward)) != content_hash(reference["target"]["reward"])
            or sim.partial_contact_weight != reference["target"]["partial_contact_weight"]):
        raise ValueError("Public RAW profile, evidence or reward contract changed")


def native_configuration_record(cfg: NativeResectionConfig) -> dict:
    """Bind every constructor component as well as the full native fingerprint."""
    components = {}
    for field in fields(NativeResectionConfig):
        if field.name.startswith("_"):
            continue
        value = getattr(cfg, field.name)
        if isinstance(value, np.ndarray):
            value = {"shape": list(value.shape), "dtype": str(value.dtype), "array_digest": array_digest(value)}
        elif is_dataclass(value):
            value = asdict(value)
        elif field.name == "tools":
            value = [asdict(tool) for tool in value]
        components[field.name] = value
    return {"fingerprint": cfg.fingerprint, "components": components}


def assert_declared_native_configuration(cfg: NativeResectionConfig, declaration: dict, reference: dict) -> dict:
    actual = native_configuration_record(cfg)
    if content_hash(actual) != content_hash(declaration["native_configuration"]):
        raise ValueError("Native configuration differs from the complete V2 declaration")
    historical = declaration["historical_native_configuration"]
    if (historical["fingerprint"] != reference["target"]["native_config_hash"]
            or historical["components"]["tissue_support_provenance"] != reference["target"]["tissue_support_provenance"]
            or set(actual["components"]) != set(historical["components"])):
        raise ValueError("Historical native identity is not bound to its original declaration")
    differing = [name for name, value in actual["components"].items()
                 if content_hash(value) != content_hash(historical["components"][name])]
    if differing != ["tissue_support_provenance"]:
        raise ValueError("V2 repair may change only the explicitly declared provenance wording")
    return {"actual": actual, "historical_native_config_hash": historical["fingerprint"],
        "different_components": differing, "physical_component_equality": True,
        "interpretation": "Distinct truthful provenance identities; no anatomy/tool/access/microstep change"}


def raw_failure_state(sim) -> dict:
    """Diagnostic state only: cancelled inventory may prevent certified metrics()."""
    return {"status": "incomplete_not_a_candidate", "committed_cut_count": len(sim._history),
            "history": copy.deepcopy(sim._history), "total_reward": sim.total_reward,
            "inventory_receipts": copy.deepcopy(sim._inventory_receipts),
            "published_inventory_available": sim._proposals is not None,
            "proposal_accounting": {"integrity_calls": sim._integrity_calls,
                "integrity_seconds": sim._integrity_seconds, "preview_calls": sim._preview_calls,
                "preview_seconds": sim._preview_seconds}}


def _inventory(sim, guard, folder: Path) -> dict:
    guard.require()
    started = time.perf_counter()
    actions = sim.proposed_actions()
    observation = sim.observation()
    metrics = sim.metrics()
    receipts = metrics["inventory_receipts"]
    if (not receipts or receipts[-1]["status"] != "complete"
            or receipts[-1]["batch"]["slot_count"] != len(sim.proposal_config.offsets_source_voxels) * len(sim.native_config.tools)
            or tuple(action.action_id for action in actions) != observation.action_ids
            or receipts[-1]["certified_action_ids"] != list(observation.action_ids[1:])):
        raise RuntimeError("Full ordered native inventory is incomplete")
    record = {"action_ids": observation.action_ids, "action_features": observation.action_features,
        "state_features": observation.state_features, "action_mask": observation.action_mask,
        "nominal_action_values": [sim.nominal_action_value(action) for action in actions],
        "complete_inventory": receipts[-1], "proposal_accounting": metrics["proposal_accounting"],
        "inspection_seconds": time.perf_counter() - started,
        "scope": "complete constructor/reset inventory inspected again; all verification cost charged"}
    guard.require()
    write_json(folder / "inventory.json", record)
    return record


def run_episode(sim, guard, folder: Path, *, fixed_actions: tuple[str, ...] | None = None, policy=None) -> dict:
    """Greedy decisions use nominal complete inventories; replay never chooses anew."""
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    transitions = []
    actions = []
    try:
        while not sim.terminated:
            guard.require()
            decision_started = time.perf_counter()
            options = sim.proposed_actions()
            if policy is not None:
                if fixed_actions is not None:
                    raise ValueError("Policy rollout cannot also request frozen sequence replay")
                import torch
                with torch.no_grad():
                    observation = sim.observation()
                    logits, _ = policy(observation)
                    mask = torch.as_tensor(np.array(observation.action_mask, copy=True), dtype=torch.bool)
                    if tuple(action.action_id for action in options) != observation.action_ids or not torch.isfinite(logits[mask]).all():
                        raise RuntimeError("Untrained policy has changed inventory or nonfinite legal logits")
                    selected = options[int(torch.argmax(logits).item())].action_id
            elif fixed_actions is None:
                values = [sim.nominal_action_value(action) for action in options]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite nominal action value")
                selected = options[int(np.argmax(values))].action_id  # STOP wins zero ties.
            else:
                if len(actions) >= len(fixed_actions):
                    raise RuntimeError("Frozen sequence ends before episode termination")
                selected = fixed_actions[len(actions)]
            decision_seconds = time.perf_counter() - decision_started
            transition_started = time.perf_counter()
            result = sim.step(selected)
            row = {"action_id": selected, "reward": result.reward,
                "decision_seconds": decision_seconds,
                "transition_and_next_inventory_seconds": time.perf_counter() - transition_started,
                "terminated": result.terminated, "committed": selected != "STOP"}
            transitions.append(row)
            actions.append(selected)
            write_json(folder / "progress.json", {"status": "running", "transitions": transitions})
            if len(actions) > sim.config.max_steps + 1:
                raise RuntimeError("Preflight episode exceeded the declared action horizon")
        if fixed_actions is not None and tuple(actions) != tuple(fixed_actions):
            raise RuntimeError("Frozen sequence has trailing actions after termination")
        metrics = sim.metrics()
        if any(row["status"] != "complete" for row in metrics["inventory_receipts"]):
            raise RuntimeError("An incomplete inventory cannot finish an episode")
        guard.require()
        record = {"status": "completed", "actions": actions, "transitions": transitions,
            "total_reward": metrics["total_reward"], "metrics": metrics,
            "episode_seconds_before_export": time.perf_counter() - started}
        write_json(folder / "episode.json", record)
        record["episode_seconds_including_export"] = time.perf_counter() - started
        write_json(folder / "timing.json", {key: value for key, value in record.items() if key.startswith("episode_seconds")})
        return record
    except Exception as error:
        write_json(folder / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "completed_transitions": transitions, "interrupted_transition_committed": bool(getattr(error, "committed", False)),
            "interrupted_reward": getattr(error, "reward", None), "raw_state": raw_failure_state(sim),
            "elapsed_seconds": time.perf_counter() - started, "resource": guard.receipt()})
        raise


def run_preflight(case, factory: Callable, optimization, selection, output: Path, guard: ResourceGuard,
                  *, source_root: Path = ROOT) -> dict:
    """Shared testable orchestration. The public CLI separately binds real anatomy."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"status": "running", "stage": "cold_setup", "completed_selection_worlds": 0,
        "completed_untrained_policy_worlds": 0, "expected_selection_worlds": 2,
        "expected_untrained_policy_worlds": 2, "eligible_candidate_count": 0, "gradient_steps": 0,
        "final_worlds_used": False, "stress_worlds_used": False}
    write_json(output / "status.json", status)
    sim = None
    try:
        source = source_snapshot(source_root)
        write_json(output / "source.json", source)
        guard.require()
        cold = time.perf_counter()
        sim = factory(guard)
        cold_seconds = time.perf_counter() - cold
        if type(sim) is not AxisColumnNativeSimulator:
            raise TypeError("Preflight requires the exact experimental axis backend")
        validate_panel(case, sim.config.world_generator, optimization, selection)
        model_hash = sim.decision_model_hash
        write_json(output / "model.json", {"decision_model_hash": model_hash,
            "native_config_hash": sim.native_config.fingerprint, "case_hash": sim.case_hash,
            "planning_hash": case.planning_hash, "adapter_version": AXIS_ADAPTER_VERSION,
            "proposal_model_hash": sim._proposer.model_hash, "proposal_rule": asdict(sim.proposal_config),
            "max_steps": sim.config.max_steps, "max_actions": sim.config.max_actions, "input_profile": "RAW",
            "reward": asdict(sim.config.reward), "partial_contact_weight": sim.partial_contact_weight,
            "evidence_available": sim.config.evidence_available, "access": asdict(sim.native_config.access),
            "tools": [asdict(tool) for tool in sim.native_config.tools]})
        initial = _inventory(sim, guard, output / "initial")
        write_json(output / "cold-setup.json", {"factory_seconds": cold_seconds,
            "adapter_initialization": sim.metrics()["initialization_timing"],
            "initial_proposal_accounting": initial["proposal_accounting"],
            "scope": "source configuration is prepared separately; cold factory includes complete initial inventory"})
        clone_started = time.perf_counter()
        clone = sim.clone()
        clone_seconds = time.perf_counter() - clone_started
        if clone.decision_model_hash != model_hash or clone.engine.state_hash != sim.engine.state_hash:
            raise RuntimeError("Initial clone changed the model or initial cavity")
        write_json(output / "clone-probe.json", {"seconds": clone_seconds,
            "model_hash": clone.decision_model_hash, "cavity_state_hash": clone.engine.state_hash,
            "scope": "one integrity-checked initial clone; panels below reset one instance, so panel timings exclude per-world clone/factory cost"})
        del clone
        guard.require()
        status["stage"] = "greedy_optimization_episode"
        write_json(output / "status.json", status)
        reset_started = time.perf_counter()
        sim.reset(optimization.seeds[0])
        write_json(output / "greedy-reset.json", {"seconds": time.perf_counter() - reset_started,
            "optimization_seed": optimization.seeds[0], "partition_hash": optimization.partition_hash})
        greedy = run_episode(sim, guard, output / "greedy")
        actions = tuple(greedy["actions"])
        freeze = {"actions": actions, "decision_model_hash": model_hash,
            "selection_partition_hash": selection.partition_hash,
            "source_hash": source["runtime_content_hash"], "status": "frozen_pending_panel_and_geometry",
            "greedy_history_hash": content_hash(greedy["metrics"]["history"])}
        write_json(output / "sequence-freeze.json", {**freeze, "freeze_hash": content_hash(freeze)})
        # Retain one simulator, reset between worlds; cold factory cost remains explicit.
        panel = []
        for index, seed in enumerate(selection.seeds):
            status["stage"] = f"selection_world_{index + 1}"
            write_json(output / "status.json", status)
            guard.require()
            phase = time.perf_counter()
            sim.reset(seed)
            reset_seconds = time.perf_counter() - phase
            if sim.decision_model_hash != model_hash or sim.case_hash != case.semantic_hash:
                raise RuntimeError("Selection replay changed model or case")
            row = run_episode(sim, guard, output / f"selection-{index + 1}", fixed_actions=actions)
            panel.append({"seed": seed, "return": row["total_reward"], "reset_seconds": reset_seconds,
                "episode_seconds": row["episode_seconds_including_export"],
                "history_hash": content_hash(row["metrics"]["history"]),
                "episode_world_hash": row["metrics"]["episode_world_hash"]})
            status["completed_selection_worlds"] = len(panel)
            write_json(output / "selection-progress.json", {"status": "incomplete" if len(panel) < 2 else "complete",
                "worlds": panel, "partition": selection.to_dict()})
        if len(panel) != 2 or not np.isfinite([row["return"] for row in panel]).all():
            raise RuntimeError("Selection panel is incomplete or nonfinite")
        status["stage"] = "untrained_raw_initialization"
        write_json(output / "status.json", status)
        policy_started = time.perf_counter()
        import torch
        from resectionlab.learning import MaskedPatientPolicy, policy_hash
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            policy = MaskedPatientPolicy(15, 6, 16, input_profile="RAW")
        policy.eval().requires_grad_(False)
        initial_hash = policy_hash(policy)
        checkpoint = {"policy": policy.state_dict(), "policy_hash": initial_hash,
            "dimensions": policy.dimensions, **policy.checkpoint_profile()}
        torch.save(checkpoint, output / "untrained-raw-initial.pt")
        policy_freeze = {"seed": 11, "hidden_features": 16, "policy_hash": initial_hash,
            "checkpoint_sha256": hashlib.sha256((output / "untrained-raw-initial.pt").read_bytes()).hexdigest(),
            "decision_model_hash": model_hash, "selection_partition_hash": selection.partition_hash,
            **policy.checkpoint_profile(), "gradient_steps": 0, "optimizer_constructed": False,
            "initialization_seconds": time.perf_counter() - policy_started,
            "scope": "includes imports, seeded tensor initialization, freeze and checkpoint write; no policy selection"}
        write_json(output / "untrained-policy-freeze.json", policy_freeze)
        policy_panel = []
        policy_record = None
        for index, seed in enumerate(selection.seeds):
            status["stage"] = f"untrained_raw_selection_world_{index + 1}"
            write_json(output / "status.json", status)
            guard.require()
            phase = time.perf_counter()
            sim.reset(seed)
            reset_seconds = time.perf_counter() - phase
            if sim.decision_model_hash != model_hash or sim.case_hash != case.semantic_hash:
                raise RuntimeError("Untrained policy replay changed model or case")
            row = run_episode(sim, guard, output / f"untrained-selection-{index + 1}", policy=policy)
            if policy_hash(policy) != initial_hash or any(parameter.grad is not None for parameter in policy.parameters()):
                raise RuntimeError("Untrained policy weights or gradients changed")
            policy_record = row
            policy_panel.append({"seed": seed, "return": row["total_reward"], "actions": row["actions"],
                "reset_seconds": reset_seconds, "episode_seconds": row["episode_seconds_including_export"],
                "history_hash": content_hash(row["metrics"]["history"]),
                "episode_world_hash": row["metrics"]["episode_world_hash"]})
            status["completed_untrained_policy_worlds"] = len(policy_panel)
            write_json(output / "untrained-selection-progress.json", {
                "status": "incomplete" if len(policy_panel) < 2 else "complete", "worlds": policy_panel,
                "partition": selection.to_dict(), "policy_hash": initial_hash})
        if len(policy_panel) != 2 or not np.isfinite([row["return"] for row in policy_panel]).all():
            raise RuntimeError("Untrained policy panel is incomplete or nonfinite")
        status["stage"] = "independent_native_audit"
        write_json(output / "status.json", status)
        audit_started = time.perf_counter()
        history_hashes = {freeze["greedy_history_hash"], *(row["history_hash"] for row in panel)}
        if len(history_hashes) != 1:
            raise RuntimeError("Deterministic frozen replays differ; audit cannot be shared")
        audit = independent_check_native_history(case, sim.native_config.tools, greedy["metrics"]["history"],
            tissue_mask=sim.native_config.tissue_mask, access=sim.native_config.access,
            hard_exclusion=sim.native_config.hard_exclusion, cancelled=guard)
        audit_seconds = time.perf_counter() - audit_started
        write_json(output / "independent-audit.json", {"audit": audit.to_dict(), "seconds": audit_seconds,
            "shared_exact_history_hash": freeze["greedy_history_hash"], "certified_complete_episodes": 3,
            "deduplication": "same source/model and byte-equivalent normalized history; all replays actually executed"})
        if not audit.feasible or not audit.complete_tool_checked or not audit.frontier_checked:
            raise RuntimeError("Independent native certification failed")
        if len({row["history_hash"] for row in policy_panel}) != 1:
            raise RuntimeError("Deterministic untrained policy replays differ")
        policy_audit_started = time.perf_counter()
        if policy_panel[0]["history_hash"] == freeze["greedy_history_hash"]:
            policy_audit, unique_audits = audit, 1
        else:
            policy_audit = independent_check_native_history(case, sim.native_config.tools,
                policy_record["metrics"]["history"], tissue_mask=sim.native_config.tissue_mask,
                access=sim.native_config.access, hard_exclusion=sim.native_config.hard_exclusion, cancelled=guard)
            unique_audits = 2
        policy_audit_seconds = time.perf_counter() - policy_audit_started
        write_json(output / "untrained-independent-audit.json", {"audit": policy_audit.to_dict(),
            "seconds": policy_audit_seconds, "shared_exact_history_hash": policy_panel[0]["history_hash"],
            "certified_complete_episodes": 2, "reused_greedy_certificate": unique_audits == 1})
        if not policy_audit.feasible or not policy_audit.complete_tool_checked or not policy_audit.frontier_checked:
            raise RuntimeError("Untrained policy independent native certification failed")
        guard.require()
        before = (sim.engine.state_hash, len(sim._history))
        probe_started = time.perf_counter()
        guard.probe = True
        try:
            sim.proposed_actions()
        except InterruptedError:
            probe_seconds = time.perf_counter() - probe_started
        else:
            raise RuntimeError("Adapter ignored cancellation boundary")
        finally:
            guard.probe = False
        if before != (sim.engine.state_hash, len(sim._history)):
            raise RuntimeError("Boundary cancellation mutated the completed episode")
        write_json(output / "cancellation-probe.json", {"boundary_response_seconds": probe_seconds,
            "committed_state_unchanged": True, "scope": "request before cached inventory; not mid-preview latency"})
        guard.require()
        assert_source(source, source_root)
        if (policy_hash(policy) != initial_hash
                or hashlib.sha256((output / "untrained-raw-initial.pt").read_bytes()).hexdigest() != policy_freeze["checkpoint_sha256"]):
            raise RuntimeError("Untrained policy checkpoint changed before publication")
        final_fields = dict(status="completed", stage="complete", eligible_candidate_count=2,
            mean_selection_return=float(np.mean([row["return"] for row in panel])),
            untrained_mean_selection_return=float(np.mean([row["return"] for row in policy_panel])),
            independent_audit_seconds=audit_seconds + policy_audit_seconds,
            unique_native_audits=unique_audits, certified_complete_episodes=5,
            full_preflight_seconds=time.perf_counter() - started, resource=guard.receipt())
        candidates = {"frozen_greedy_sequence": {"actions": actions, "decision_model_hash": model_hash,
            "freeze_hash": content_hash(freeze), "selection_partition_hash": selection.partition_hash,
            "selection_worlds": panel, "mean_selection_return": final_fields["mean_selection_return"],
            "independent_history_hash": freeze["greedy_history_hash"], "input_profile": "RAW",
            "status": "development_preflight_only", "clinical_probability": None},
            "untrained_raw_policy": {**policy_freeze,
            "selection_worlds": policy_panel, "mean_selection_return": final_fields["untrained_mean_selection_return"],
            "independent_history_hash": policy_panel[0]["history_hash"], "status": "development_preflight_only"}}
        # Payload presence alone grants no eligibility. The final atomic status
        # is the sole consumer authority, and binds both complete payloads.
        write_json(output / "candidate-records.json", {"publication_authority": "status.json must be completed and hash-match this file",
            "candidates": candidates})
        guard.require()
        assert_source(source, source_root)
        status.update(final_fields, candidate_records_sha256=hashlib.sha256((output / "candidate-records.json").read_bytes()).hexdigest(),
            full_preflight_seconds=time.perf_counter() - started, resource=guard.receipt())
        write_json(output / "status.json", status)
        return status
    except Exception as error:
        status.update(status="cancelled" if isinstance(error, InterruptedError) else "failed",
            error_type=type(error).__name__, error=str(error), resource=guard.receipt(),
            full_preflight_seconds=time.perf_counter() - started, eligible_candidate_count=0)
        if sim is not None:
            write_json(output / "failure-state.json", raw_failure_state(sim))
        write_json(output / "status.json", status)
        raise


def public_worker(output: Path, case_bundle: Path) -> None:
    declaration = _checked_manifest(ROOT / DECLARATION_PATH, DECLARATION_HASH)
    budget = PreflightBudget(**declaration["resource_budget"])
    guard = ResourceGuard(budget)
    signal.signal(signal.SIGTERM, lambda *_: guard.cancel("parent_cancellation"))
    source = source_snapshot()
    try:
        launch = json.loads((output / "launch-source.json").read_text())
        if launch["runtime_content_hash"] != source["runtime_content_hash"]:
            raise RuntimeError("Frozen worker differs from launch source")
        _public_worker(output, case_bundle, guard)
        guard.require()
        assert_source(source)
    except Exception as error:
        write_json(output / "worker-failure.json", {"status": "failed", "error_type": type(error).__name__,
            "error": str(error), "eligible_candidate_count": 0, "resource": guard.receipt(),
            "stage": "profile_or_export" if (output / "profile").exists() else "case_and_native_preparation"})
        raise
    finally:
        write_json(output / "worker-resource.json", guard.receipt())


def _public_worker(output: Path, case_bundle: Path, guard: ResourceGuard) -> None:
    reference = _checked_manifest(ROOT / REFERENCE_PATH, REFERENCE_HASH)
    declaration = _checked_manifest(ROOT / DECLARATION_PATH, DECLARATION_HASH)
    target = reference["target"]
    from resectionlab.imaging import load_case
    preparation = time.perf_counter()
    if hashlib.sha256(case_bundle.read_bytes()).hexdigest() != target["bundle_sha256"]:
        raise ValueError("Case bundle differs from the declared patient derivative")
    case = load_case(case_bundle)
    guard.require()
    if (case.semantic_hash != target["semantic_hash"] or case.planning_hash != target["planning_hash"]
            or case.frame != "RAS+" or [asdict(tool) for tool in NATIVE_GENERIC_TOOLS] != target["tools"]):
        raise ValueError("Public anatomy, frame or native tool contract changed")
    cfg = native_config_from_case(case, access=AccessWindow(**target["access"]))
    guard.require()
    config_receipt = assert_declared_native_configuration(cfg, declaration, reference)
    write_json(output / "native-configuration-preflight.json", config_receipt)
    for actual, expected in ((array_digest(cfg.tissue_mask), target["tissue_support_hash"]),
                            (array_digest(cfg.hard_exclusion), target["hard_exclusion_hash"])):
        if actual != expected:
            raise ValueError("Native anatomy differs from prior declared source model")
    optimization = _partition(target["world_partitions"]["optimization"])
    selection = _partition(target["world_partitions"]["selection"])
    validate_panel(case, optimization.generator, optimization, selection)
    write_json(output / "preparation.json", {"case_and_native_config_seconds": time.perf_counter() - preparation,
        "bundle_sha256": target["bundle_sha256"], "native_config_hash": cfg.fingerprint,
        "old_fixed_action_model_hash_for_distinction_only": target["decision_model_hash"],
        "source_frame": case.frame, "tissue_support_provenance": cfg.tissue_support_provenance,
        "optimization": optimization.to_dict(), "selection": selection.to_dict(), "resource": guard.receipt()})
    def factory(cancelled):
        result = AxisColumnNativeSimulator(cfg, proposal_config=AxisColumnProposalConfig(), max_steps=3,
            reward=RewardSpec(**target["reward"]), partial_contact_weight=target["partial_contact_weight"],
            world_generator=optimization.generator,
            compartment_names={i: name for i, name in enumerate(sorted(case.compartments), 1)}, cancelled=cancelled)
        if result.decision_model_hash == target["decision_model_hash"]:
            raise RuntimeError("New axis action model must have a distinct identity")
        assert_public_action_model(result, declaration, reference)
        return result
    run_preflight(case, factory, optimization, selection, output / "profile", guard)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-bundle", type=Path)
    parser.add_argument("--execute", action="store_true", help="run public development preflight after external execution release")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        public_worker(args.output, args.case_bundle)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = _checked_manifest(ROOT / DECLARATION_PATH, DECLARATION_HASH)
    write_json(args.output / "declaration.json", declaration)
    status = {"status": "declared_not_executed", "gradient_steps": 0, "final_worlds_used": False,
        "stress_worlds_used": False, "eligible_candidate_count": 0}
    write_json(args.output / "launcher-status.json", status)
    if not args.execute:
        return
    if args.case_bundle is None:
        raise ValueError("--execute requires an explicit --case-bundle")
    launched = time.perf_counter()
    snapshot = source_snapshot()
    write_json(args.output / "launch-source.json", snapshot)
    frozen = args.output / "frozen-source"
    for name, digest in snapshot["file_sha256"].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("SOURCE_CHANGED_WHILE_COPYING")
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert_source(snapshot)
    env = {**os.environ, "PYTHONPATH": str(frozen / "src"),
           "GIT_CEILING_DIRECTORIES": os.pathsep.join((str(frozen.parent), str(args.output.parent)))}
    command = [sys.executable, str(frozen / "scripts/preflight_native_axis.py"), "--worker",
               "--output", str(args.output), "--case-bundle", str(args.case_bundle.resolve())]
    budget = PreflightBudget(**declaration["resource_budget"])
    status.update(status="running", launch_pid=os.getpid(), worker_command=command,
        launch_preparation_seconds=time.perf_counter() - launched, resource_budget=asdict(budget))
    write_json(args.output / "launcher-status.json", status)
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen(command, cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT)
        hard_timeout, parent_timeout = False, False
        try:
            process.wait(timeout=budget.worker_wall_seconds)
        except subprocess.TimeoutExpired:
            parent_timeout = True
            process.terminate()
            try:
                process.wait(timeout=budget.hard_termination_grace_seconds)
            except subprocess.TimeoutExpired:
                hard_timeout = True
                process.kill()
                process.wait()
    outcome_path = args.output / "profile/status.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    complete = not parent_timeout and process.returncode == 0 and outcome.get("status") == "completed"
    status.update(status="completed" if complete else "failed",
        worker_returncode=process.returncode, hard_killed=hard_timeout,
        parent_timeout_requested=parent_timeout,
        full_launcher_seconds=time.perf_counter() - launched,
        eligible_candidate_count=outcome.get("eligible_candidate_count", 0) if complete else 0,
        consumer_authority="Both launcher-status.json and profile/status.json must be completed; payload presence alone is insufficient")
    write_json(args.output / "launcher-status.json", status)
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
