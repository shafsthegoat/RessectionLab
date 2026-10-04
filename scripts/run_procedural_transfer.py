#!/usr/bin/env python3
"""Declared procedural-to-patient development transfer; final worlds stay closed.

This distinct checkpoint path does not relax the analytic population runner.
The declaration fixes one previously studied patient-derived target, two
procedural families, the common native action model, and all online budgets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from run_patient_learning import (FrozenSequence, frozen_population_rollouts,
    preserve_source, source_snapshot, write_json)
from resectionlab.core import array_digest
from resectionlab.evaluation import freeze_candidates, independent_check_native_history
from resectionlab.learning import TrainingConfig, load_policy, policy_hash, rollout_policy, train_patient_policy
from resectionlab.native_resection import NATIVE_RESECTION_VERSION
from resectionlab.native_simulation import (NATIVE_ACTION_FEATURE_NAMES, NATIVE_ADAPTER_VERSION,
    NativeSequentialSimulator, make_native_patient_simulator, native_beam_search, native_greedy_search)
from resectionlab.worlds import FrozenDecisionModel, content_hash, generate_partitions

DECLARATION_PATH = Path("manifests/experiments/procedural-native-to-ucsf-v1.json")
DECLARATION_HASH = "sha256:3abf10162ed9d096a7e21ec84a87c1a88ebf12bb074345a78040dfdebb828ac9"
SCOPE = "procedural_native_to_patient_development"
FROZEN = "PROCEDURAL_PRETRAINED_FROZEN"
ADAPTED = "PROCEDURAL_PRETRAINED_ADAPTED"


def execution_context() -> dict[str, Any]:
    """Observed machine load is a timing covariate, not a workload guarantee."""
    return {"recorded_at": datetime.now(timezone.utc).isoformat(), "cpu_count": os.cpu_count(),
        "load_average_1_5_15_minutes": list(os.getloadavg()),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                                  * (1 if sys.platform == "darwin" else 1024)}


def require_complete_selection(result: Any, history: list[dict[str, Any]], world_count: int) -> None:
    # STOP's fully evaluated return of zero is valid. An interrupted first
    # panel leaves None, which must never become a selected candidate.
    value = result.selected_selection_return
    if type(value) not in (int, float) or not np.isfinite(value):
        raise RuntimeError("Learned policy has no completed selection panel; preserve the unfinished run")
    complete = any(isinstance(row, dict) and type(row.get("world_count")) is int
        and row["world_count"] == world_count and row.get("checkpoint_hash") == result.selected_checkpoint_hash
        and type(row.get("mean_return")) in (int, float) and np.isfinite(row["mean_return"])
        and row["mean_return"] == value for row in history)
    if not complete:
        raise RuntimeError("Selected checkpoint lacks a matching complete selection-history record")


def load_declaration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert_declaration(value)
    return value


def assert_declaration(value: dict[str, Any]) -> None:
    payload = dict(value)
    supplied = payload.pop("declaration_content_hash", None)
    if supplied != DECLARATION_HASH or content_hash(payload) != supplied:
        raise ValueError("Declaration differs from the committed preregistration")


def _equal(actual: Any, declared: Any, field: str) -> None:
    if content_hash(actual) != content_hash(declared):
        raise ValueError(f"Declared native contract changed: {field}")


def model_inventory(sim: NativeSequentialSimulator) -> dict[str, Any]:
    if type(sim) is not NativeSequentialSimulator:
        raise TypeError("Procedural transfer requires the exact native simulator backend")
    sim.assert_model_frozen()
    cfg = sim.native_config
    return {"decision_model_hash": sim.decision_model_hash,
        "native_config_hash": cfg.fingerprint, "candidate_tips_mm": sim.candidate_tips_mm.tolist(),
        "candidate_entries_mm": sim.candidate_entries_mm.tolist(), "access": asdict(cfg.access),
        "tools": [asdict(tool) for tool in cfg.tools], "reward": asdict(sim.config.reward),
        "partial_contact_weight": sim.partial_contact_weight, "max_steps": sim.config.max_steps,
        "max_actions": sim.config.max_actions, "initial_action_ids": list(sim.observation().action_ids),
        "evidence_available": list(sim.config.evidence_available),
        "tissue_support_hash": array_digest(cfg.tissue_mask),
        "tissue_support_provenance": cfg.tissue_support_provenance,
        "hard_exclusion_hash": array_digest(cfg.hard_exclusion),
        "hard_exclusion_true_cells": int(np.count_nonzero(cfg.hard_exclusion)),
        "native_adapter_version": NATIVE_ADAPTER_VERSION,
        "native_resection_version": NATIVE_RESECTION_VERSION,
        "action_feature_names": list(NATIVE_ACTION_FEATURE_NAMES)}


def assert_declared_target(case: Any, sim: NativeSequentialSimulator,
                           declaration: dict[str, Any]) -> Any:
    """Bind actual anatomy, hypotheses, action ordering and seed identity."""
    target = declaration["target"]
    actual = {"case_id": case.case_id, "semantic_hash": case.semantic_hash,
        "planning_hash": case.planning_hash, "source_shape": list(case.mri.shape),
        "source_affine": np.asarray(case.affine).tolist(), "source_frame": case.frame,
        "compartment_mask_hashes": {name: array_digest(mask) for name, mask in case.compartments.items()},
        **model_inventory(sim)}
    for name, value in actual.items():
        _equal(value, target[name], name)
    if sim.case_hash != case.semantic_hash:
        raise ValueError("Patient simulator source does not match the actual patient bundle")
    panels = generate_partitions(case.semantic_hash, sim.config.world_generator,
        declaration["selection_and_evaluation"]["common_world_master_seed"],
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=case.planning_hash)
    _equal(panels.to_dict(), target["world_partitions"], "all world partitions")
    return panels


def assert_declared_sources(declaration: dict[str, Any], root: Path = ROOT) -> None:
    for name, digest in declaration["source_hashes_at_declaration"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Declared source changed before execution: {name}; declare an amendment")


def transfer_source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    snapshot = source_snapshot(root)
    runtime = snapshot["numerical_runtime_sha256"]
    name = "scripts/run_procedural_transfer.py"
    runtime[name] = snapshot["file_sha256"][name]
    snapshot["numerical_runtime_content_hash"] = content_hash(runtime)
    return snapshot


def assert_transfer_source_unchanged(snapshot: dict[str, Any], root: Path = ROOT) -> None:
    if transfer_source_snapshot(root)["numerical_runtime_content_hash"] != snapshot["numerical_runtime_content_hash"]:
        raise ValueError("SOURCE_CHANGED_DURING_RUN: preserve this attempt")


def preserve_declaration_inputs(declaration: dict[str, Any], directory: Path,
                                 root: Path = ROOT) -> None:
    """Keep non-code provenance needed by the narrow checkpoint validator."""
    names = (str(DECLARATION_PATH), *(name for name in declaration["source_hashes_at_declaration"]
                                    if not name.startswith("src/")))
    for name in names:
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / name).read_bytes())
    load_declaration(directory / DECLARATION_PATH)
    for name in names[1:]:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != declaration["source_hashes_at_declaration"][name]:
            raise ValueError(f"Declared input changed while preserving it: {name}")


def native_audit(sim: NativeSequentialSimulator, case: Any, actions: tuple[str, ...],
                 seed: int, cancelled: Callable[[], bool]) -> tuple[dict[str, Any], dict[str, Any]]:
    instance = sim.clone()
    instance.replay(actions, seed)
    metrics = instance.metrics()
    cfg = instance.native_config
    audit = independent_check_native_history(case, cfg.tools, metrics["history"],
        tissue_mask=cfg.tissue_mask, access=cfg.access, hard_exclusion=cfg.hard_exclusion,
        geometry_frame="RAS+", cancelled=cancelled).to_dict()
    return audit, metrics


def validate_frozen_candidates(base: NativeSequentialSimulator, case: Any,
        candidates: list[FrozenSequence], seed: int, output: Path,
        cancelled: Callable[[], bool]) -> dict[str, Any]:
    """Independent native development audit after selection, with no final worlds."""
    started = time.perf_counter()
    cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    audits, replays, timings = {}, {}, {}
    for candidate in candidates:
        if cancelled():
            raise InterruptedError("Independent candidate audit cancelled")
        step_started = time.perf_counter()
        key = content_hash({"model": base.decision_model_hash, "actions": candidate.actions})
        reused = key in cache
        if not reused:
            cache[key] = native_audit(base, case, candidate.actions, seed, cancelled)
        audit, metrics = cache[key]
        audits[candidate.plan_id] = {**audit, "audit_key": key, "shared_audit_reused": reused}
        replays[candidate.plan_id] = {"checkpoint_hash": candidate.selected_checkpoint_hash,
            "shared_checkpoint_hash": candidate.shared_checkpoint_hash, "audit_key": key, "metrics": metrics}
        timings[candidate.plan_id] = time.perf_counter() - step_started
        write_json(output / "native-history-audit.json", audits)
        write_json(output / "native-history-replay.json", replays)
    result = {"unique_audits": len(cache), "candidate_count": len(candidates),
        "full_validation_seconds": time.perf_counter() - started, "candidate_seconds": timings,
        "rejected_candidate_ids": [name for name, audit in audits.items() if not audit["feasible"]],
        "final_worlds_used": False}
    write_json(output / "validation-timings.json", result)
    return result


def run_target_comparison(base: NativeSequentialSimulator, case: Any, target: Any,
        checkpoint: Path, output: Path, declaration: dict[str, Any], *,
        cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    status = {"run_id": f"procedural-transfer-development-{uuid.uuid4()}", "status": "running",
        "scope": SCOPE, "final_worlds_used": False, "started_at": datetime.now(timezone.utc).isoformat(),
        "planned_learning_runs": [f"{mode}:{seed}" for mode in ("PATIENT_SCRATCH_RL", ADAPTED)
                                  for seed in (11, 23, 47)], "completed_learning_runs": []}
    write_json(output / "status.json", status)
    try:
        assert_declaration(declaration)
        assert_declared_sources(declaration)
        from resectionlab.procedural_learning import (validate_procedural_checkpoint,
            load_frozen_procedural_policy, train_procedural_adapted_policy)
        snapshot = transfer_source_snapshot()
        write_json(output / "source.json", snapshot)
        preserve_source(snapshot, output / "source-snapshot")
        preserve_declaration_inputs(declaration, output / "source-snapshot")
        panels = assert_declared_target(case, base, declaration)
        config = TrainingConfig(**declaration["budgets"]["online_scratch_and_adapted_each_seed"])
        if config.max_episode_steps != base.config.max_steps + 1:
            raise ValueError("Policy and search must share the declared non-STOP horizon")
        initialization_started = time.perf_counter()
        validation = dict(target=target, simulator=base, hidden_features=config.hidden_features)
        shared = validate_procedural_checkpoint(checkpoint, **validation)
        saved = output / "procedural-source.pt"
        saved.write_bytes(Path(checkpoint).read_bytes())
        _equal(validate_procedural_checkpoint(saved, **validation), shared, "copied shared checkpoint")
        frozen_policy = load_frozen_procedural_policy(saved, **validation)
        initialization_seconds = time.perf_counter() - initialization_started
        write_json(output / "shared-provenance.json", {**shared,
            "initialization_seconds": initialization_seconds,
            "initialization_scope": "validation_copy_and_frozen_load_once_per_comparison"})
        model = FrozenDecisionModel.create(case_hash=base.case_hash,
            geometry={"simulation_hash": base.decision_model_hash},
            objectives={**asdict(base.config.reward), "partial_contact_weight": base.partial_contact_weight},
            tools=[asdict(tool) for tool in base.native_config.tools],
            world_generator=base.config.world_generator.to_dict(),
            action_primitives={"native_adapter": NATIVE_ADAPTER_VERSION,
                "inventory": model_inventory(base)})
        write_json(output / "manifest.json", {**status, "declaration_hash": DECLARATION_HASH,
            "decision_model": model.to_dict(), "target": asdict(target),
            "world_partitions": panels.to_dict(), "training_config": asdict(config),
            "optimization_seeds": declaration["policy"]["optimization_seeds_online"],
            "budget_matching": declaration["budgets"]["primary_matching"],
            "transition_budget_claim": declaration["budgets"]["transition_budget_claim"],
            "clinical_deficit_probability": None})
        candidates = [FrozenSequence("STOP", base.case_hash, ("STOP",), "STOP")]
        searches = []
        for mode in ("GREEDY", "SEARCH"):
            assert_transfer_source_unchanged(snapshot)
            if cancelled():
                raise InterruptedError("Transfer experiment cancelled")
            preparation_started = time.perf_counter()
            arm = base.fresh()
            _equal(model_inventory(arm), model_inventory(base), "fresh search arm")
            preparation_seconds = time.perf_counter() - preparation_started
            options = declaration["budgets"]["search"]
            context_before = execution_context()
            result = (native_greedy_search(arm, max_wall_seconds=options["max_wall_seconds"])
                      if mode == "GREEDY" else native_beam_search(arm, **options))
            searches.append({"method": mode, **asdict(result), "preparation_seconds": preparation_seconds,
                "execution_context_before": context_before, "execution_context_after": execution_context()})
            candidates.append(FrozenSequence(mode, base.case_hash, result.actions, mode))
        write_json(output / "search.json", searches)
        preparation_started = time.perf_counter()
        frozen_arm = base.fresh()
        _equal(model_inventory(frozen_arm), model_inventory(base), "fresh frozen arm")
        preparation_seconds = time.perf_counter() - preparation_started
        context_before = execution_context()
        frozen_record, actions = frozen_population_rollouts(frozen_arm.clone, frozen_policy,
            panels.selection, config, cancelled=cancelled)
        frozen_record.update(optimizer_mode=FROZEN, preparation_seconds=preparation_seconds,
            shared_checkpoint_initialization_seconds=initialization_seconds,
            execution_context_before=context_before, execution_context_after=execution_context(),
            elapsed_seconds_scope="selection_rollouts_only; cold preparation and shared checkpoint initialization separate")
        write_json(output / "procedural-frozen.json", frozen_record)
        if not frozen_record["selection_panel_complete"]:
            raise RuntimeError("Frozen procedural policy did not complete the selection panel")
        candidates.append(FrozenSequence(FROZEN, base.case_hash, actions, FROZEN,
            shared["policy_hash"], shared_checkpoint_hash=shared["policy_hash"]))
        records = []
        for mode in ("PATIENT_SCRATCH_RL", ADAPTED):
            for seed in declaration["policy"]["optimization_seeds_online"]:
                assert_transfer_source_unchanged(snapshot)
                if cancelled():
                    raise InterruptedError("Transfer experiment cancelled")
                preparation_started = time.perf_counter()
                arm = base.fresh()
                _equal(model_inventory(arm), model_inventory(base), "fresh learned arm")
                preparation_seconds = time.perf_counter() - preparation_started
                folder = output / f"{'scratch' if mode == 'PATIENT_SCRATCH_RL' else 'adapted'}-{seed}"
                status["current_learning_run"] = f"{mode}:{seed}"
                write_json(output / "status.json", status)
                kwargs = dict(config=replace(config, seed=seed), output_dir=folder, cancelled=cancelled)
                context_before = execution_context()
                call_started = time.perf_counter()
                result = (train_patient_policy(arm.clone, panels.optimization, panels.selection, **kwargs)
                    if mode == "PATIENT_SCRATCH_RL" else train_procedural_adapted_policy(
                        arm.clone, panels.optimization, panels.selection,
                        checkpoint=saved, target=target, **kwargs))
                call_seconds = time.perf_counter() - call_started
                detailed = json.loads((folder / "result.json").read_text())
                contract = json.loads((folder / "contract.json").read_text())
                if contract["algorithm"] != declaration["policy"]["algorithm"]:
                    raise ValueError("Online learner algorithm differs from preregistration")
                record = {**asdict(result), "optimizer_mode": mode, "seed": seed,
                    "preparation_seconds": preparation_seconds, "execution_context_before": context_before,
                    "execution_context_after": execution_context(), "trainer_call_seconds": call_seconds,
                    "trainer_call_scope": "full wrapper, initialization, learner and export; elapsed_seconds is the learner's optimization-plus-selection timer",
                    "learner_wall_budget_overshoot_seconds": max(0., result.elapsed_seconds - config.max_wall_seconds),
                    "actor_parameters_changed": detailed["actor_parameters_changed"],
                    "initial_actor_hash": detailed["initial_actor_hash"],
                    "latest_actor_hash": detailed["latest_actor_hash"],
                    "selected_is_initial": result.selected_checkpoint_hash == result.initial_checkpoint_hash,
                    "algorithm": contract["algorithm"]}
                records.append(record)
                write_json(output / "training.json", records)
                if result.status in {"failed", "cancelled"}:
                    raise RuntimeError(f"{mode} seed {seed} did not finish: {result.status}")
                require_complete_selection(result, detailed.get("selection_history", []), len(panels.selection.seeds))
                if mode == ADAPTED and result.initial_checkpoint_hash != shared["policy_hash"]:
                    raise ValueError("Adapted actor did not start from the unchanged shared policy")
                extraction_started = time.perf_counter()
                checkpoints = [(f"{mode}:{seed}", "checkpoint.pt", result.selected_checkpoint_hash)]
                if mode == "PATIENT_SCRATCH_RL":
                    checkpoints.append((f"INITIAL:{seed}", "initial.pt", result.initial_checkpoint_hash))
                for label, filename, expected in checkpoints:
                    actor = load_policy(folder / filename)
                    if policy_hash(actor) != expected:
                        raise ValueError("Saved actor differs from the reported checkpoint")
                    rollout = rollout_policy(actor, base.clone(), seed=panels.selection.seeds[0],
                        max_steps=config.max_episode_steps, expected_model_hash=base.decision_model_hash,
                        interrupt=cancelled)
                    candidates.append(FrozenSequence(label, base.case_hash, rollout.actions,
                        "INITIAL_POLICY" if label.startswith("INITIAL:") else mode, expected,
                        shared_checkpoint_hash=shared["policy_hash"] if mode == ADAPTED else None))
                record["candidate_extraction_seconds"] = time.perf_counter() - extraction_started
                write_json(output / "training.json", records)
                assert_transfer_source_unchanged(snapshot)
                status["completed_learning_runs"].append(f"{mode}:{seed}")
                write_json(output / "status.json", status)
        if (policy_hash(frozen_policy) != shared["policy_hash"]
                or hashlib.sha256(saved.read_bytes()).hexdigest() != shared["checkpoint_file_sha256"]):
            raise ValueError("Shared procedural checkpoint changed during adaptation")
        freeze = freeze_candidates(candidates, model, panels.selection,
            "declared baselines, initial scratch actors, selected scratch/adapted actors; maximum selection mean, earliest tie",
            optimization_manifest=panels.optimization)
        write_json(output / "candidate-freeze.json", {**freeze.to_dict(), "candidates": [asdict(c) for c in candidates]})
        audit_dir = output / f"development-geometry-{uuid.uuid4()}"
        audit_dir.mkdir()
        write_json(audit_dir / "manifest.json", {"candidate_freeze_hash": freeze.fingerprint,
            "final_worlds_used": False, "development_run_id": status["run_id"]})
        audit = validate_frozen_candidates(base, case, candidates, panels.selection.seeds[0], audit_dir, cancelled)
        assert_transfer_source_unchanged(snapshot)
        status.update(status="invalidated" if audit["rejected_candidate_ids"] else "completed",
            validation=audit, geometry_validation_run_id=audit_dir.name,
            total_comparison_seconds=time.perf_counter() - started,
            finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "status.json", status)
        return status
    except Exception as exc:
        status.update(status="cancelled" if isinstance(exc, InterruptedError) else "failed",
            exception=type(exc).__name__, reason=str(exc), total_comparison_seconds=time.perf_counter() - started,
            unfinished_learning_runs=[name for name in status["planned_learning_runs"]
                                      if name not in status["completed_learning_runs"]])
        write_json(output / "status.json", status)
        raise


def procedural_preflight(members: tuple[Any, ...], declaration: dict[str, Any],
                         output: Path, cancelled: Callable[[], bool]) -> list[dict[str, Any]]:
    """Validate the prespecified first fine-tool stroke, without policy updates."""
    from types import SimpleNamespace
    expected = declaration["procedural_training"]["members"]
    if len(members) != len(expected):
        raise ValueError("Procedural family count differs from declaration")
    records = []
    for member, declared in zip(members, expected):
        started = time.perf_counter()
        sim = member.factory()
        _equal(member.family_id, declared["group_id"], "procedural family order")
        _equal(member.aliases, declared["aliases"], "procedural aliases")
        _equal(sim.case_hash, declared["source_hash"], "procedural source bytes")
        _equal(sim.decision_model_hash, declared["model_hash"], "procedural model")
        _equal(sim.native_config.fingerprint, declared["native_config_hash"], "procedural native config")
        _equal(list(sim.observation().action_ids), declared["initial_action_ids"], "procedural initial action set")
        _equal(member.optimization.to_dict(), declared["world_partitions"]["optimization"], "procedural optimization worlds")
        _equal(member.selection.to_dict(), declared["world_partitions"]["selection"], "procedural selection worlds")
        # This geometry-only adapter records the exact generated source hash;
        # it is not a human CaseData record or an independent patient.
        cfg = sim.native_config
        audit_source = SimpleNamespace(mri=cfg.tissue_mask, affine=cfg.affine,
                                       frame="RAS+", semantic_hash=cfg.source_hash)
        actions = (declared["initial_action_ids"][1], "STOP")
        audit, metrics = native_audit(sim, audit_source, actions, member.optimization.seeds[0], cancelled)
        row = {"family_id": member.family_id, "actions": actions, "audit": audit,
            "normal_removed_mm3": metrics["simulated_removed_normal_volume_mm3"],
            "target_removed_mm3": metrics["simulated_removed_target_volume_mm3"],
            "elapsed_seconds": time.perf_counter() - started, "gradient_steps": 0,
            "final_worlds_used": False}
        records.append(row)
        write_json(output / "procedural-preflight.json", records)
        if not audit["feasible"] or row["normal_removed_mm3"] <= 0:
            raise ValueError("Declared procedural fixture failed independent paid-removal preflight; retain and amend")
    return records


def worker(output: Path, bundle: Path, *, execute: bool,
           cancelled: Callable[[], bool]) -> dict[str, Any]:
    from resectionlab.imaging import load_case
    from resectionlab.procedural_learning import (TransferTarget, make_native_procedural_fixture,
                                                 train_procedural_native_policy)
    started = time.perf_counter()
    declaration = load_declaration(ROOT / DECLARATION_PATH)
    assert_declared_sources(declaration)
    snapshot = transfer_source_snapshot()
    write_json(output / "worker-source.json", snapshot)
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != declaration["target"]["bundle_sha256"]:
        raise ValueError("Source bundle differs from declared bytes")
    preparation_started = time.perf_counter()
    case = load_case(bundle)
    base = make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7,
                                         cancelled=cancelled)
    assert_declared_target(case, base, declaration)
    target_data = declaration["target"]
    target = TransferTarget(case_hash=case.semantic_hash, planning_hash=case.planning_hash,
        group_id=target_data["group_id"], aliases=tuple(target_data["aliases"]),
        source_kind=target_data["source_kind"], outer_split="development", excluded_from_pretraining=True)
    target_preparation_seconds = time.perf_counter() - preparation_started
    fixture_started = time.perf_counter()
    members = tuple(make_native_procedural_fixture(target))
    fixture_seconds = time.perf_counter() - fixture_started
    write_json(output / "cohort.json", {"target": asdict(target),
        "procedural_families": [member.family_id for member in members],
        "human_patients_in_pretraining": 0, "patient_derived_development_cases": 1,
        "verified_primary_source_target_cases": 0, "locked_final_patients": 0})
    preflight = procedural_preflight(members, declaration, output, cancelled)
    assert_transfer_source_unchanged(snapshot)
    preparation = {"target_preparation_seconds": target_preparation_seconds,
        "procedural_fixture_construction_seconds": fixture_seconds,
        "preflight_seconds": sum(row["elapsed_seconds"] for row in preflight),
        "source_hash": snapshot["numerical_runtime_content_hash"]}
    if not execute:
        result = {"status": "preflight_passed_no_training", "training_executed": False,
            "final_worlds_used": False, **preparation, "total_worker_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", result)
        return result
    offline = TrainingConfig(**declaration["budgets"]["offline_training"])
    pretraining_started = time.perf_counter()
    pretraining_context = execution_context()
    pretrained = train_procedural_native_policy(members, excluded_targets=(target,), config=offline,
                                                output_dir=output / "pretraining", cancelled=cancelled)
    pretraining_call_seconds = time.perf_counter() - pretraining_started
    if pretrained.get("status") != "completed" or not pretrained.get("checkpoint_path"):
        raise RuntimeError("Procedural pretraining did not export a qualified checkpoint")
    assert_transfer_source_unchanged(snapshot)
    result = run_target_comparison(base, case, target, Path(pretrained["checkpoint_path"]),
        output / "comparison", declaration, cancelled=cancelled)
    assert_transfer_source_unchanged(snapshot)
    comparison = output / "comparison"
    read = lambda name: json.loads((comparison / name).read_text())
    audit_dir = comparison / result["geometry_validation_run_id"]
    summary = {"status": result["status"], "study_id": declaration["study_id"], "scope": SCOPE,
        "claim_boundary": declaration["claim_boundary"], "target": asdict(target),
        "declaration_hash": DECLARATION_HASH, "source_hash": snapshot["numerical_runtime_content_hash"],
        "preparation": preparation, "offline_pretraining": pretrained,
        "pretraining_call_seconds": pretraining_call_seconds,
        "pretraining_context_before": pretraining_context,
        "shared": read("shared-provenance.json"), "search": read("search.json"),
        "frozen": read("procedural-frozen.json"), "learning": read("training.json"),
        "validation": result["validation"], "independent_geometry": {
            name: audit["feasible"] for name, audit in json.loads((audit_dir / "native-history-audit.json").read_text()).items()},
        "total_worker_seconds": time.perf_counter() - started, "final_worlds_used": False,
        "clinical_deficit_probability": None, "training_executed": True,
        "limitations": declaration["deferred_gates"], "budget_claim": declaration["budgets"]["primary_matching"],
        "transition_budget_claim": declaration["budgets"]["transition_budget_claim"]}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-bundle", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="Run declared training after preflight; default is preflight only")
    parser.add_argument("--frozen-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    if args.frozen_worker:
        worker(output, args.case_bundle.resolve(), execute=args.execute, cancelled=stopped.is_set)
        return
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "scope": SCOPE, "declared_training_requested": args.execute,
              "final_worlds_used": False, "declaration_hash": DECLARATION_HASH}
    write_json(output / "experiment-status.json", status)
    try:
        declaration = load_declaration(ROOT / DECLARATION_PATH)
        assert_declared_sources(declaration)
        snapshot = transfer_source_snapshot()
        write_json(output / "launch-source.json", snapshot)
        frozen = output / "frozen-source"
        preserve_source(snapshot, frozen)
        preserve_declaration_inputs(declaration, frozen)
        # Preserve source bytes for this small public bundle as well as code.
        original_bundle = args.case_bundle or ROOT / declaration["target"]["bundle_path"]
        preserved_bundle = output / "source-case.ressectionlab"
        preserved_bundle.write_bytes(original_bundle.read_bytes())
        if hashlib.sha256(preserved_bundle.read_bytes()).hexdigest() != declaration["target"]["bundle_sha256"]:
            raise ValueError("Patient bundle bytes differ from declaration")
        assert_transfer_source_unchanged(snapshot)
        command = [sys.executable, str(frozen / "scripts/run_procedural_transfer.py"),
            "--output", str(output), "--case-bundle", str(preserved_bundle), "--frozen-worker"]
        if args.execute:
            command.append("--execute")
        process = subprocess.Popen(command, cwd=frozen)
        interruption_sent = False
        while process.poll() is None:
            if stopped.is_set() and not interruption_sent:
                process.terminate()
                interruption_sent = True
            time.sleep(.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        summary = json.loads((output / "summary.json").read_text())
        status.update(status=summary["status"], total_seconds_including_source_freeze=time.perf_counter() - started)
    except Exception as exc:
        status.update(status="cancelled" if stopped.is_set() else "failed", reason=str(exc),
            exception=type(exc).__name__, total_seconds_including_source_freeze=time.perf_counter() - started)
        write_json(output / "experiment-status.json", status)
        raise
    write_json(output / "experiment-status.json", status)
    print(f"{status['status']}: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
