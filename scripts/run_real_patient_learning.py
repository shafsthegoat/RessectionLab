#!/usr/bin/env python3
"""Small real PAT05 geometric REINFORCE experiment; no transfer/clinical claim.

Uses the existing annotation-assisted spatial actor and native actions unchanged.
Fixed initial/latest readouts are not a checkpoint-selection or patient test set.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_real_spatial_policy as support
import compare_real_spatial_search as inputs

VERSION = "pat05-real-geometric-learning-v1"
SETTINGS = {"seed": 11, "optimizer_updates": 2, "episodes_per_update": 2,
    "random_episodes": 3, "max_steps": 3, "learning_rate": .001,
    "gamma": 1., "entropy_weight": .01, "value_weight": .5,
    "gradient_clip": 5., "search_seconds": 90., "max_wall_seconds": 600., "max_rss_bytes": 6 * 1024 ** 3,
    "device": "cpu", "torch_threads": 1, "blas_thread_caps": 1}
PHYSICAL_FIELDS = ("member", "tools", "objective", "objective_context", "adapter_options",
                   "policy_config", "expected_policy_architecture_hash", "track",
                   "cohort_path", "cohort_sha256")
BASE_DECLARATION = ROOT / "manifests/experiments/pat05-real-spatial-search-comparison-v1.json"


def source_inventory():
    paths = [Path(__file__), Path(inputs.__file__), Path(support.__file__),
             *(ROOT / "src/resectionlab").rglob("*.py")]
    return {str(path.relative_to(ROOT)): support.sha256(path) for path in sorted(paths)}


def declaration():
    original = json.loads(BASE_DECLARATION.read_text())
    return {"version": VERSION, **{key: original[key] for key in PHYSICAL_FIELDS},
        "source_sha256": source_inventory(), "settings": SETTINGS,
        "base_definition_sha256": support.sha256(BASE_DECLARATION),
        "methods": ["initial_policy", "random_legal", "greedy_search", "scratch_reinforce", "latest_policy"],
        "success": "at least one independently checked fully contained source target cell",
        "scope": "one TRAIN anatomy, annotation-assisted bounded partial removal, no full-resection or generalization claim",
        "search": "repeat all current certified previews scored by permitted nominal objective; stop when no positive marginal action; maximum three actions",
        "checkpoint_rule": "fixed latest after exactly two updates, no reward-selected checkpoint",
        "representation_limit": "same permitted annotation/scan/tool information; planner accesses full nominal field while CNN uses declared64-cubed crop",
        "initial_cache": "shared immutable prepared initial inventory; fresh mutable cavity/history clone per episode; preparation cost reported"}


def validate_declaration(record):
    from resectionlab.real_patient_learning import read_development_cohort, require_development_role
    if record.get("version") != VERSION or record.get("settings") != SETTINGS:
        raise ValueError("This runner requires the fixed two-update development configuration")
    original = json.loads(BASE_DECLARATION.read_text())
    if any(record.get(key) != original[key] for key in PHYSICAL_FIELDS):
        raise ValueError("Patient roles, permitted inputs, geometry, model and rewards must remain fixed")
    if record.get("base_definition_sha256") != support.sha256(BASE_DECLARATION):
        raise ValueError("Original task definition changed")
    if record.get("source_sha256") != source_inventory():
        raise ValueError("Numerical source changed before execution")
    cohort = read_development_cohort(ROOT / record["cohort_path"])
    require_development_role(cohort, record["member"]["subject"], role="TRAIN")
    if record["member"]["subject"] != "sub-PAT05":
        raise ValueError("This experiment opens only TRAIN PAT05")
    return cohort


def checked_batch(episodes, policy):
    from resectionlab.real_patient_learning import validate_on_policy_batch
    from resectionlab.spatial_policy import parameter_hash
    return validate_on_policy_batch(episodes, parameter_hash=parameter_hash(policy))


def run_episode(base, policy, generator, *, mode, name, output, guard, audit, sequence=None):
    """Persist attempted/committed decisions, then audit the entire terminal history.

    No partial episode is returned for gradient computation. Policy forwards only
    receive SpatialObservation, while reference metrics are written afterwards.
    """
    import numpy as np
    import torch
    from resectionlab.spatial_policy import SpatialTransition, parameter_hash
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    if mode not in {"argmax", "sample", "random", "sequence"}:
        raise ValueError("Unknown episode decision mode")
    if mode == "sequence" and not sequence:
        raise ValueError("Search replay requires a nonempty complete sequence")
    started = time.perf_counter()
    before = parameter_hash(policy)
    report = {"status": "preparing", "mode": mode, "name": name,
        "behavior_parameter_hash": before, "attempted_actions": 0, "committed_transitions": 0,
        "invalid_actions": 0, "decisions": [], "policy_forward_calls": 0}
    path = output / (name + ".json")
    task = None

    def preserve():
        report["elapsed_seconds"] = time.perf_counter() - started
        support.write_json(path, report)
        guard()

    try:
        preserve()
        setup = time.perf_counter()
        task = base.clone()
        report["initial_state_clone_seconds"] = time.perf_counter() - setup
        if task.terminated or task.metrics()["steps"] != 0:
            raise ValueError("Each method must start from the same empty initial cavity")
        transitions = []
        while not task.terminated:
            guard()
            observation = task.observation()
            mask = np.asarray(observation.action_mask)
            legal = np.flatnonzero(mask)
            if not len(legal):
                raise ValueError("STOP must remain legal")
            decision = {"observation_hash": observation.fingerprint,
                "action_ids": list(observation.action_ids), "action_mask": mask.tolist(),
                "step": len(transitions), "behavior_parameter_hash": before}
            decision_started = time.perf_counter()
            if mode in {"argmax", "sample"}:
                with torch.no_grad():
                    logits, value = policy(observation)
                    probabilities = logits.softmax(-1)
                    index = int(logits.argmax()) if mode == "argmax" else int(torch.multinomial(probabilities, 1, generator=generator))
                decision.update(logits=[float(x) if np.isfinite(x) else None for x in logits.detach().cpu().numpy()],
                    probabilities=probabilities.detach().cpu().tolist(), value=float(value.detach()),
                    decision_rule="argmax_first; STOP wins exact ties" if mode == "argmax" else "categorical_sample")
                report["policy_forward_calls"] += 1
            elif mode == "random":
                index = int(legal[int(torch.randint(len(legal), (1,), generator=generator))])
                decision["decision_rule"] = "uniform_over_all_current_legal_actions_including_STOP"
            else:
                if len(transitions) >= len(sequence):
                    raise ValueError("Search sequence ended before STOP/horizon")
                requested = sequence[len(transitions)]
                if requested not in observation.action_ids:
                    raise ValueError("Search replay action is absent from current inventory")
                index = observation.action_ids.index(requested)
                decision["decision_rule"] = "frozen_search_sequence"
            if not mask[index]:
                raise ValueError("Selected action is masked")
            action = observation.action_ids[index]
            decision.update(action_id=action, selected_index=index, decision_seconds=time.perf_counter() - decision_started,
                            status="attempted")
            report["decisions"].append(decision)
            report["attempted_actions"] += 1
            preserve()
            transition_started = time.perf_counter()
            try:
                result = task.step(action)
            except CommittedTransitionInterrupted as error:
                decision.update(status="committed_unreturned", committed_info=copy.deepcopy(error.info))
                report["committed_transitions"] += 1
                raise
            except (ValueError, IndexError, KeyError):
                report["invalid_actions"] += 1
                raise
            decision.update(status="returned", reward=float(result.reward), terminated=bool(result.terminated),
                            transition_seconds=time.perf_counter() - transition_started,
                            committed_info=copy.deepcopy(result.info))
            report["committed_transitions"] += 1
            transitions.append(SpatialTransition(observation, action, result.reward, result.terminated))
            preserve()
        if mode == "sequence" and len(transitions) != len(sequence):
            raise ValueError("Search supplied actions after its terminal transition")
        if parameter_hash(policy) != before:
            raise RuntimeError("Rollout mutated frozen policy weights")
        report.update(status="awaiting_independent_check", metrics=task.metrics(),
                      simulated_return=sum(t.reward for t in transitions),
                      online_seconds=time.perf_counter() - started)
        preserve()
        audit_started = time.perf_counter()
        report["independent_evaluation"] = audit(task, metrics=report["metrics"])
        report["independent_evaluation_seconds"] = time.perf_counter() - audit_started
        if report["independent_evaluation"].get("accepted") is not True:
            raise RuntimeError("Independent evaluator rejected the executed history")
        # The evaluator raises for contradictory accounting/certification; a
        # genuine zero-target complete episode is valid negative task evidence.
        report["status"] = "complete"
        preserve()
        return tuple(transitions), report
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                      elapsed_seconds=time.perf_counter() - started)
        if task is not None:
            try:
                report["failure_metrics"] = task.metrics()
            except BaseException as metric_error:
                report["failure_metrics_error"] = str(metric_error)
        support.write_json(path, report)
        raise


def worker(record, output):
    cohort = validate_declaration(record)
    import torch
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash, reinforce_loss, gradient_step
    from resectionlab.real_patient_learning import PatientEpisode
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SETTINGS["seed"])
    policy = SpatialPolicy(SpatialPolicyConfig(**record["policy_config"]))
    if policy.architecture_hash != record["expected_policy_architecture_hash"]:
        raise ValueError("Changed spatial policy architecture")
    generator = torch.Generator().manual_seed(SETTINGS["seed"] + 100000)
    random_generator = torch.Generator().manual_seed(SETTINGS["seed"] + 200000)
    report = {"version": VERSION, "status": "running", "patient": "sub-PAT05", "role": "TRAIN",
        "track": "annotation_assisted", "algorithm": "masked_REINFORCE_with_spatial_value_baseline",
        "settings": SETTINGS, "parameters": sum(p.numel() for p in policy.parameters()),
        "architecture_hash": policy.architecture_hash, "initial_parameter_hash": parameter_hash(policy),
        "optimizer_updates": 0, "gradient_curve": [], "episodes": {},
        "other_patients_opened": 0, "checkpoint_selection": "fixed_latest_not_best_return",
        "generalization_measured": False, "clinical_accuracy_measured": False}

    def guard():
        if time.perf_counter() - started > SETTINGS["max_wall_seconds"] or support.peak_rss_bytes() > SETTINGS["max_rss_bytes"]:
            raise RuntimeError("Whole experiment resource limit exceeded")

    def preserve():
        report.update(elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=support.peak_rss_bytes())
        support.write_json(output / "receipt.json", report)
        guard()

    def save_weights(label, optimizer=None):
        torch.save({"policy": policy.state_dict(), "architecture": policy.architecture_record(),
            "parameter_hash": parameter_hash(policy), "update": report["optimizer_updates"],
            "optimizer": None if optimizer is None else optimizer.state_dict()}, output / (label + ".pt"))

    preserve()
    save_weights("initial")
    prep_started = time.perf_counter()
    base = inputs.load_member(record, cohort)
    report.update(preparation_seconds=time.perf_counter() - prep_started,
        initial_task_metrics=base.metrics(), initial_inventory=base.candidate_inventory(),
        decision_model_hash=base.decision_model_hash)
    preserve()

    def audit(task, *, metrics):
        return evaluate_native_spatial_episode(task, metrics=metrics, minimum_target_cells=1,
                                               distance_backend="batch", distance_batch_size=256)

    def episode(name, mode, sequence=None):
        transitions, result = run_episode(base, policy, random_generator if mode == "random" else generator,
            mode=mode, name=name, output=output, guard=guard, audit=audit, sequence=sequence)
        report["episodes"][name] = {"receipt": name + ".json", "return": result["simulated_return"],
            "actions": [x["action_id"] for x in result["decisions"]],
            "target_removed_mm3": result["metrics"]["target_removed_mm3"],
            "normal_removed_mm3": result["metrics"]["normal_removed_mm3"],
            "invalid_actions": result["invalid_actions"], "seconds": result["elapsed_seconds"],
            "independent_evaluation": result["independent_evaluation"],
            "parameter_hash": result["behavior_parameter_hash"]}
        preserve()
        return transitions

    try:
        episode("initial_policy", "argmax")
        for index in range(SETTINGS["random_episodes"]):
            episode("random_legal_" + str(index), "random")
        search_started = time.perf_counter()
        sequence, accounting = base.observed_greedy_search(seconds=SETTINGS["search_seconds"])
        report["search"] = {"sequence": list(sequence), "accounting": accounting,
                            "planning_seconds": time.perf_counter() - search_started}
        preserve()
        episode("greedy_search", "sequence", sequence)
        optimizer = torch.optim.Adam(policy.parameters(), lr=SETTINGS["learning_rate"])
        for update in range(SETTINGS["optimizer_updates"]):
            behavior = parameter_hash(policy)
            batch = [PatientEpisode("BTC:sub-PAT05", behavior,
                episode(f"optimization_{update}_{index}", "sample"))
                for index in range(SETTINGS["episodes_per_update"])]
            batch_record = checked_batch(batch, policy)
            guard()
            loss_started = time.perf_counter()
            loss, loss_record = reinforce_loss(policy, [row.transitions for row in batch],
                gamma=SETTINGS["gamma"], entropy_weight=SETTINGS["entropy_weight"], value_weight=SETTINGS["value_weight"])
            gradient_record = gradient_step(policy, optimizer, loss, max_norm=SETTINGS["gradient_clip"])
            report["optimizer_updates"] += gradient_record["optimizer_steps"]
            report["gradient_curve"].append({"update": report["optimizer_updates"], **batch_record,
                **loss_record, **gradient_record, "loss_backward_optimizer_seconds": time.perf_counter() - loss_started})
            save_weights("update_" + str(report["optimizer_updates"]), optimizer)
            save_weights("latest", optimizer)
            del loss, batch
            preserve()
        episode("latest_policy", "argmax")
        report.update(status="complete", latest_parameter_hash=parameter_hash(policy),
            behavior_changed=report["episodes"]["initial_policy"]["actions"] != report["episodes"]["latest_policy"]["actions"],
            return_change=report["episodes"]["latest_policy"]["return"] - report["episodes"]["initial_policy"]["return"],
            latest_minus_greedy_return=report["episodes"]["latest_policy"]["return"] - report["episodes"]["greedy_search"]["return"],
            success_interpretation="Any target cell is a reachability criterion only; useful behavior is assessed by continuous return and target/normal volumes against initial/random/greedy")
        if source_inventory() != record["source_sha256"]:
            raise RuntimeError("Source bytes changed during the experiment")
        preserve()
        return report
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                      latest_parameter_hash=parameter_hash(policy), elapsed_seconds=time.perf_counter() - started)
        support.write_json(output / "receipt.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--write-declaration", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-declaration-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.write_declaration:
        if args.declaration.exists():
            raise SystemExit("Preserve existing declarations")
        support.write_json(args.declaration, declaration())
        return
    record, digest, payload = support.read_declaration(args.declaration, args.expected_declaration_sha256)
    validate_declaration(record)
    if args.worker:
        if not args.expected_declaration_sha256:
            raise ValueError("Worker requires frozen declaration identity")
        try:
            worker(record, args.output)
        except BaseException as error:
            support.write_json(args.output / "failure.json", {"type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc()})
            raise
        return
    if not args.execute:
        print(json.dumps({"status": "validated_not_executed", "declaration_sha256": digest}))
        return
    if args.output is None or args.output.exists():
        raise SystemExit("Choose a new output directory to preserve earlier attempts")
    args.output.mkdir(parents=True)
    (args.output / "declaration-input.json").write_bytes(payload)
    for relative in record["source_sha256"]:
        destination = args.output / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    outcome = support.supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker",
        "--declaration", str((args.output / "declaration-input.json").resolve()),
        "--expected-declaration-sha256", digest, "--output", str(args.output.resolve())], args.output, SETTINGS, digest)
    support.write_json(args.output / "output-sha256.json", {path.name: support.sha256(path)
        for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "output-sha256.json"})
    if outcome["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
