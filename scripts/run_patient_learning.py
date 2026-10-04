#!/usr/bin/env python3
"""Reproducible local development comparison; never a locked cohort claim.

Train on optimization worlds, select checkpoints on selection worlds, freeze
fixed action sequences, then separately evaluate untouched world manifests.
The independent checker certifies each insertion against the evolving cavity;
the discrete suction primitive and surrogate consequences remain model inputs.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
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

from resectionlab.evaluation import (EvaluationLedger, IndependentGeometryResult,
    WorldOutcome, evaluate_frozen_candidates, freeze_candidates,
    independent_check_sequence, independent_native_removal_check, independent_check_native_history)
from resectionlab.learning import (TrainingConfig, load_policy, policy_hash,
    rollout_policy, train_patient_policy)
from resectionlab.simulation import (SequentialSimulator, beam_search,
    greedy_search, make_synthetic_simulator)
from resectionlab.native_simulation import (NativeSequentialSimulator,
    native_beam_search, native_greedy_search)
from resectionlab.worlds import (FrozenDecisionModel, WorldGenerator,
    content_hash, generate_partitions)


def write_json(path: Path, value: Any) -> None:
    def encode(item: Any) -> Any:
        if isinstance(item, Mapping):
            return dict(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=encode) + "\n")
    temporary.replace(path)


def source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Hash actual working files, including code not committed yet."""
    files = sorted({*root.glob("src/**/*.py"), *root.glob("scripts/*.py"),
                    *root.glob("tests/*.py"), *root.glob("*.md"),
                    *root.glob("docs/*.md"), *root.glob("*requirements*.txt"),
                    *root.glob("pyproject.toml")})
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files if path.is_file()}
    runtime = {name: digest for name, digest in hashes.items()
               if (name.startswith("src/resectionlab/") and "/app/" not in name)
               or name in {"scripts/run_patient_learning.py", "scripts/run_native_learning.py"}
               or name in {"pyproject.toml", "requirements-lock.txt"}}
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"file_sha256": hashes, "content_hash": content_hash(hashes),
            "numerical_runtime_sha256": runtime, "numerical_runtime_content_hash": content_hash(runtime),
            "git_revision": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "python": sys.version, "platform": platform.platform()}


def assert_source_unchanged(snapshot: dict[str, Any], root: Path = ROOT, *, runtime_only: bool = False) -> None:
    field = "numerical_runtime_content_hash" if runtime_only else "content_hash"
    if source_snapshot(root)[field] != snapshot[field]:
        raise ValueError("SOURCE_CHANGED_DURING_RUN: preserve this attempt and start a new run")


def preserve_source(snapshot: dict[str, Any], directory: Path, root: Path = ROOT) -> None:
    """Keep exact bytes for reproducing a run from an uncommitted working tree."""
    for name, digest in snapshot["file_sha256"].items():
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("SOURCE_CHANGED_DURING_RUN: snapshot could not be preserved")
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


@dataclass(frozen=True)
class FrozenSequence:
    plan_id: str
    case_hash: str
    actions: tuple[str, ...]
    optimizer_mode: str
    selected_checkpoint_hash: str | None = None
    plan_type: str = "simulated_resection_sequence"

    @property
    def semantic_hash(self) -> str:
        return content_hash(asdict(self))


def independent_sequence_check(factory: Callable[[], SequentialSimulator],
                               candidate: FrozenSequence) -> IndependentGeometryResult:
    """Use the shared independent checker; native footprint is a separate gate."""
    return independent_check_sequence(factory, candidate.actions)


def run_experiment(factory: Callable[[], SequentialSimulator], output: Path, *,
                   config: TrainingConfig, seeds: tuple[int, ...] = (11, 23, 47),
                   counts: tuple[int, int, int, int] = (16, 8, 16, 8),
                   master_seed: int = 20261004, beam_width: int = 8,
                   independent_patient_count: int = 0,
                   source_case: Any | None = None,
                   planning_hash: str | None = None,
                   evaluate: bool = True,
                   trainer: Callable[..., Any] = train_patient_policy,
                   cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide distinct nonnegative optimization seeds")
    if len(counts) != 4 or any(count < 1 for count in counts):
        raise ValueError("Provide four positive world-panel sizes")
    if config.max_episode_steps < 2:
        raise ValueError("Comparison horizon needs at least one removal slot and one STOP slot")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = source_snapshot()
    run_id = f"development-{uuid.uuid4()}"
    status = {"run_id": run_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        preserve_source(snapshot, output / "source-snapshot")
        preprocessing_started = time.perf_counter()
        base = factory()
        if type(base) not in {SequentialSimulator, NativeSequentialSimulator}:
            raise TypeError("Unsupported simulator adapter: this runner must not reconstruct a native model as coarse")
        native_backend = type(base) is NativeSequentialSimulator
        if native_backend and source_case is None:
            raise ValueError("Native simulation requires its exact source_case for independent history validation")
        if source_case is not None and source_case.semantic_hash != base.case_hash:
            raise ValueError("Simulator and source_case identities differ")
        requested_simulator_horizon = base.config.max_steps
        # Policy rollout reserves its last slot for STOP; give search exactly
        # the same number of non-STOP actions, with the effective limit saved.
        effective_horizon = min(requested_simulator_horizon, config.max_episode_steps - 1)
        if base.config.max_steps != effective_horizon:
            base = base.fresh(max_steps=effective_horizon)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        factory = base.clone
        generator = base.config.world_generator
        seed_identity = source_case.planning_hash if source_case is not None else planning_hash
        if independent_patient_count > 0 and not seed_identity:
            raise ValueError("Patient runs require source_case or an explicit preoperative planning_hash")
        partitions = generate_partitions(base.case_hash, generator, master_seed,
            optimization=counts[0], selection=counts[1], final_evaluation=counts[2], stress=counts[3],
            planning_hash=seed_identity)
        model = FrozenDecisionModel.create(case_hash=base.case_hash,
            geometry={"simulation_hash": base.decision_model_hash},
            objectives={**asdict(base.config.reward), **({"partial_contact_weight": base.partial_contact_weight}
                                                        if native_backend else {})},
            tools=[asdict(t) for t in base.config.tools],
            world_generator=generator.to_dict(),
            action_primitives={"simulation_version": base.metrics()["simulation_version"],
                               "max_actions": base.config.max_actions,
                               "proposal_scan_limit": base.config.proposal_scan_limit})
        write_json(output / "manifest.json", {
            **status, "protocol_id": "patient-pilot-v0.1",
            "experiment_label": "synthetic_smoke" if independent_patient_count == 0 else "development_patient",
            "independent_patient_count": independent_patient_count,
            "training_config": asdict(config), "optimization_seeds": seeds,
            "trainer": f"{trainer.__module__}.{trainer.__name__}",
            "simulator_backend": type(base).__name__, "preprocessing_seconds": preprocessing_seconds,
            "source": snapshot, "decision_model": model.to_dict(),
            "simulation_hash": base.decision_model_hash, "world_partitions": partitions.to_dict(),
            "preoperative_planning_hash": seed_identity,
            "requested_simulator_nonstop_horizon": requested_simulator_horizon,
            "effective_shared_nonstop_horizon": effective_horizon,
            "simulation_derivation": dict(base.config.derivation),
            "evidence_available": list(base.config.evidence_available),
            "interpretation": "deterministic_optimization" if generator.deterministic else "registration_sensitivity_scenario",
            "clinical_deficit_probability": None,
            "benchmark_limitations": ["development_only_no_locked_cohort_claim",
                "single_default_preference_not_a_pareto_frontier", "no_population_checkpoint",
                "search_uses_nominal_scoring_not_world_ensemble_optimization"]})
        candidates = [FrozenSequence("STOP", base.case_hash, ("STOP",), "STOP")]
        search_records = []
        for label in ("GREEDY", "SEARCH"):
            preparation_started = time.perf_counter()
            # A new cache per arm prevents preceding search from warming RL's
            # candidate geometry for free. Immutable anatomy remains shared.
            arm = base.fresh()
            preparation_seconds = time.perf_counter() - preparation_started
            if label == "GREEDY":
                result = (native_greedy_search if native_backend else greedy_search)(arm, max_wall_seconds=config.max_wall_seconds)
            else:
                result = (native_beam_search if native_backend else beam_search)(arm, beam_width=beam_width,
                    max_expansions=config.max_environment_steps, max_wall_seconds=config.max_wall_seconds)
            search_records.append({"method": label, "preparation_seconds": preparation_seconds, **asdict(result)})
            candidates.append(FrozenSequence(label, base.case_hash, result.actions, label))
        write_json(output / "search.json", search_records)
        training = []
        for seed in seeds:
            if cancelled():
                status["status"] = "cancelled"
                write_json(output / "status.json", status)
                return status
            preparation_started = time.perf_counter()
            arm_template = base.fresh()
            preparation_seconds = time.perf_counter() - preparation_started
            result = trainer(arm_template.clone, partitions.optimization, partitions.selection,
                config=replace(config, seed=seed), output_dir=output / f"scratch-{seed}", cancelled=cancelled)
            training.append({**asdict(result), "preparation_seconds": preparation_seconds})
            write_json(output / "training.json", training)
            if result.status in {"cancelled", "failed"}:
                status["status"] = result.status
                write_json(output / "status.json", status)
                return status
            policy = load_policy(output / f"scratch-{seed}" / "checkpoint.pt")
            if policy_hash(policy) != result.selected_checkpoint_hash:
                raise ValueError("Selected checkpoint does not match reported checkpoint hash")
            # Extract the fixed sequence before evaluation; never select using final worlds.
            extraction_started = time.perf_counter()
            selected = rollout_policy(policy, factory(), seed=partitions.selection.seeds[0],
                                      max_steps=config.max_episode_steps, interrupt=cancelled)
            candidates.append(FrozenSequence(f"PATIENT_SCRATCH_RL:{seed}", base.case_hash,
                selected.actions, "PATIENT_SCRATCH_RL", result.selected_checkpoint_hash))
            if native_backend:
                initial_policy = load_policy(output / f"scratch-{seed}" / "initial.pt")
                if policy_hash(initial_policy) != result.initial_checkpoint_hash:
                    raise ValueError("Initial checkpoint does not match reported checkpoint hash")
                initial = rollout_policy(initial_policy, factory(), seed=partitions.selection.seeds[0],
                                         max_steps=config.max_episode_steps, interrupt=cancelled)
                candidates.append(FrozenSequence(f"INITIAL:{seed}", base.case_hash,
                    initial.actions, "INITIAL_POLICY", result.initial_checkpoint_hash))
            training[-1]["candidate_extraction_seconds"] = time.perf_counter() - extraction_started
            write_json(output / "training.json", training)
        assert_source_unchanged(snapshot, runtime_only=True)
        freeze = freeze_candidates(candidates, model, partitions.selection,
            "all baseline sequences and one selected checkpoint per prespecified seed; native runs include initial policies; max selection mean, earliest tie",
            optimization_manifest=partitions.optimization)
        write_json(output / "candidate-freeze.json", {**freeze.to_dict(),
                   "candidates": [asdict(c) for c in candidates]})
        if not evaluate and not native_backend:
            status.update(status="completed", evaluation_status="not_requested_development_selection_only",
                          finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "status.json", status)
            return status
        evaluation_id = f"{'evaluation' if evaluate else 'development-geometry'}-{uuid.uuid4()}"
        evaluation_dir = output / evaluation_id
        evaluation_dir.mkdir()
        ledger = EvaluationLedger()
        write_json(evaluation_dir / "manifest.json", {"run_id": evaluation_id,
            "development_run_id": run_id, "candidate_freeze_hash": freeze.fingerprint,
            "source_content_hash": snapshot["content_hash"], "independent_patient_count": independent_patient_count,
            "final_world_evaluation_requested": evaluate, "simulator_backend": type(base).__name__,
            "clinical_use_status": "research_only", "started_at": datetime.now(timezone.utc).isoformat()})
        validation_started = time.perf_counter()
        certificates: dict[str, IndependentGeometryResult] = {}
        native_audits: dict[str, dict[str, Any]] = {}
        coarse_audits: dict[str, dict[str, Any]] = {}
        validation_timings: dict[str, Any] = {"coarse_sequence_seconds": {}, "native_removal_seconds": {},
                                              "partition_seconds": {}}
        requires_native = independent_patient_count > 0 or source_case is not None
        native_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        native_replays: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if cancelled():
                raise InterruptedError("Candidate validation cancelled")
            started = time.perf_counter()
            if native_backend:
                # Identical frozen sequences share geometry under this fixed
                # case/model; each policy identity still retains its own record.
                audit_key = content_hash({"model": base.decision_model_hash, "actions": candidate.actions})
                cache_hit = audit_key in native_cache
                if not cache_hit:
                    instance = factory()
                    instance.replay(candidate.actions, partitions.selection.seeds[0])
                    metrics = instance.metrics()
                    native_config = instance.native_config
                    native = independent_check_native_history(source_case, native_config.tools,
                        metrics["history"], tissue_mask=native_config.tissue_mask,
                        access=native_config.access, hard_exclusion=native_config.hard_exclusion,
                        geometry_frame="RAS+", cancelled=cancelled).to_dict()
                    native["status"] = "passed" if native["feasible"] else "rejected"
                    native_cache[audit_key] = (native, metrics)
                native, metrics = native_cache[audit_key]
                native_audits[candidate.plan_id] = {**native, "audit_key": audit_key,
                                                   "shared_audit_reused": cache_hit}
                native_replays[candidate.plan_id] = {"checkpoint_hash": candidate.selected_checkpoint_hash,
                                                    "audit_key": audit_key, "metrics": metrics}
                validation_timings["native_removal_seconds"][candidate.plan_id] = time.perf_counter() - started
                certificates[candidate.plan_id] = IndependentGeometryResult(native["feasible"],
                    tuple(native["failures"]), native.get("first_unsupported_position_mm"),
                    checker_version=native["checker_version"],
                    unknowns=("tissue_mechanics_unvalidated", "hypothetical_intracranial_access"))
                write_json(evaluation_dir / "native-history-replay.json", native_replays)
                write_json(evaluation_dir / "native-history-audit.json", native_audits)
                continue
            checked = independent_sequence_check(factory, candidate)
            validation_timings["coarse_sequence_seconds"][candidate.plan_id] = time.perf_counter() - started
            coarse_audits[candidate.plan_id] = asdict(checked)
            if requires_native:
                started = time.perf_counter()
                if source_case is None:
                    native = {"status": "unassessed", "feasible": False,
                              "failures": ["native_source_case_required"],
                              "interpretation": "A coarse-grid pass cannot establish native removal validity"}
                else:
                    instance = factory()
                    try:
                        instance.replay(candidate.actions, 0)
                        native = independent_native_removal_check(source_case, instance.config,
                                                                 instance.metrics()["history"]).to_dict()
                        native["status"] = "passed" if native["feasible"] else "rejected"
                    except ValueError as exc:
                        native = {"status": "unassessed", "feasible": False,
                                  "failures": ["native_removal_audit_unavailable"], "reason": str(exc)}
                native_audits[candidate.plan_id] = native
                validation_timings["native_removal_seconds"][candidate.plan_id] = time.perf_counter() - started
                if not native["feasible"]:
                    checked = IndependentGeometryResult(False,
                        tuple(sorted(set(checked.failures) | set(native["failures"]))),
                        native.get("first_unsupported_position_mm"),
                        checker_version="independent-sequence-plus-native-removal-v1",
                        unknowns=tuple(sorted(set(checked.unknowns) | {"native_complete_tool_replay_unverified"})))
            certificates[candidate.plan_id] = checked
        write_json(evaluation_dir / "coarse-sequence-audit.json", coarse_audits)
        write_json(evaluation_dir / "native-removal-audit.json", {
            "required": requires_native, "source_case_hash": None if source_case is None else source_case.semantic_hash,
            "candidates": native_audits,
            "remaining_gate": "tissue_mechanics_not_validated" if native_backend else "native_complete_tool_replay_and_tissue_mechanics_not_validated"})
        if not evaluate:
            assert_source_unchanged(snapshot, runtime_only=True)
            validation_timings["full_validation_seconds"] = time.perf_counter() - validation_started
            write_json(evaluation_dir / "validation-timings.json", validation_timings)
            rejected = [name for name, certificate in certificates.items() if not certificate.feasible]
            status.update(status="invalidated" if rejected else "completed",
                evaluation_status="not_requested_development_selection_only",
                geometry_validation_run_id=evaluation_id,
                validation_status="rejected_candidates" if rejected else "passed_declared_native_checks",
                rejected_candidate_ids=rejected,
                full_validation_seconds=validation_timings["full_validation_seconds"],
                finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "status.json", status)
            return status
        for partition in (partitions.final_evaluation, partitions.stress):
            partition_started = time.perf_counter()
            world_template = base.fresh(world_generator=partition.generator)
            world_metrics = []
            latent_seeds = {WorldGenerator(partition.generator).sample(partition, i).world_id: seed
                            for i, seed in enumerate(partition.seeds)}
            def world_evaluator(candidate: FrozenSequence, latent: Any) -> WorldOutcome:
                instance = world_template.clone()
                instance.replay(candidate.actions, latent_seeds[latent.world_id])
                metrics = instance.metrics()
                world_metrics.append({"plan_id": candidate.plan_id, "world_id": latent.world_id, **metrics})
                return WorldOutcome(events={}, costs={
                    "motor_removed_evidence": metrics["motor_surrogate"],
                    "language_removed_evidence": metrics["language_surrogate"],
                    "normal_removed_volume": metrics["simulated_removed_normal_volume_mm3"]},
                    simulated_removed_target_volume_mm3=metrics["simulated_removed_target_volume_mm3"],
                    unknowns=tuple(metrics["unassessed"]) + (("functional_coverage_unknown",)
                        if metrics["removed_unknown_coverage_volume_mm3"] else ()))
            # Save exposure claim first, including on interruption/failure.
            ledger.claim(partition, freeze)
            write_json(evaluation_dir / "ledger.json", ledger.to_dict())
            evaluated = evaluate_frozen_candidates(candidates, freeze, model, partition,
                event_definitions={}, cost_units={"motor_removed_evidence": "field-weighted mm3 surrogate",
                    "language_removed_evidence": "field-weighted mm3 surrogate", "normal_removed_volume": "mm3"},
                world_evaluator=world_evaluator,
                geometry_checker=lambda candidate: certificates[candidate.plan_id], ledger=ledger)
            evaluated["independent_patient_count"] = independent_patient_count
            evaluated["unit_of_independent_patient_inference"] = "synthetic_fixture" if independent_patient_count == 0 else "one_patient"
            evaluated["geometry_scope"] = ("independent_native_history_complete_tool_and_source_cell_removal" if native_backend
                else "coarse_complete_tool_and_native_removal_footprint" if requires_native
                else "independent_complete_tool_checker_on_synthetic_grid")
            evaluated["native_removal_audits"] = native_audits
            evaluated["remaining_geometry_gate"] = ("tissue_mechanics_unvalidated" if native_backend
                else "native_complete_tool_replay_unverified" if requires_native else None)
            evaluated["event_evaluation_status"] = "structural_contact_events_not_implemented_in_this_runner"
            write_json(evaluation_dir / f"{partition.role.value}.json", evaluated)
            write_json(evaluation_dir / f"{partition.role.value}-replay.json", world_metrics)
            validation_timings["partition_seconds"][partition.role.value] = time.perf_counter() - partition_started
        assert_source_unchanged(snapshot, runtime_only=True)
        validation_timings["full_validation_seconds"] = time.perf_counter() - validation_started
        write_json(evaluation_dir / "validation-timings.json", validation_timings)
        rejected = [name for name, certificate in certificates.items() if not certificate.feasible]
        status.update(status="invalidated" if rejected else "completed", evaluation_run_id=evaluation_id,
                      validation_status="rejected_candidates" if rejected else "passed_declared_checks",
                      rejected_candidate_ids=rejected,
                      full_validation_seconds=validation_timings["full_validation_seconds"],
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "status.json", status)
        return status
    except Exception as exc:
        status.update(status="invalidated" if "SOURCE_CHANGED" in str(exc) else "failed",
                      error_type=type(exc).__name__, reason=str(exc),
                      finished_at=datetime.now(timezone.utc).isoformat())
        if "validation_started" in locals():
            status["full_validation_seconds"] = time.perf_counter() - validation_started
        write_json(output / "status.json", status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrites a run")
    parser.add_argument("--case-bundle", type=Path, help="Reviewed public-case bundle; omitted uses analytic synthetic fixture")
    parser.add_argument("--backend", choices=("coarse", "native"), default="coarse",
                        help="Native preserves source cells and validates complete stroke histories")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--world-counts", type=int, nargs=4, default=[16, 8, 16, 8], metavar=("OPT", "SELECT", "EVAL", "STRESS"))
    parser.add_argument("--environment-steps", type=int, default=4096)
    parser.add_argument("--gradient-steps", type=int, default=128)
    parser.add_argument("--wall-seconds", type=float, default=60)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--development-only", action="store_true", help="Do not inspect final-evaluation or stress worlds")
    parser.add_argument("--algorithm", choices=("reinforce", "ppo"), default="reinforce",
                        help="Scratch learner; every Adam step counts against the gradient budget")
    parser.add_argument("--master-seed", type=int, default=20261004)
    arguments = parser.parse_args()
    if arguments.backend == "native" and arguments.case_bundle is None:
        parser.error("--backend native requires --case-bundle")
    factory = make_synthetic_simulator
    count = 0
    case = None
    if arguments.case_bundle:
        from resectionlab.imaging import load_case
        from resectionlab.simulation import make_patient_simulator
        case = load_case(arguments.case_bundle)
        if arguments.backend == "native":
            from resectionlab.native_simulation import make_native_patient_simulator
            factory = lambda: make_native_patient_simulator(case)
        else:
            factory = lambda: make_patient_simulator(case)
        count = 0 if case.metadata.get("is_synthetic") or all(ref.provenance == "simulated" for ref in case.source_refs) else 1
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    trainer = train_patient_policy
    config_type = TrainingConfig
    if arguments.algorithm == "ppo":
        from resectionlab.learning_ppo import PPOConfig, train_patient_ppo
        trainer, config_type = train_patient_ppo, PPOConfig
    result = run_experiment(factory, arguments.output,
        config=config_type(max_environment_steps=arguments.environment_steps,
                              max_gradient_steps=arguments.gradient_steps,
                              max_wall_seconds=arguments.wall_seconds,
                              hidden_features=arguments.hidden_features,
                              learning_rate=arguments.learning_rate,
                              entropy_weight=arguments.entropy_weight,
                              max_episode_steps=arguments.episode_steps),
        seeds=tuple(arguments.seeds), counts=tuple(arguments.world_counts),
        master_seed=arguments.master_seed, independent_patient_count=count, source_case=case,
        evaluate=not arguments.development_only, trainer=trainer, cancelled=stop.is_set)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
