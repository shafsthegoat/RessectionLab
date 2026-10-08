#!/usr/bin/env python3
"""One generated native opening microtask: BC, scratch RL and exact nominal search.

No patient loader, transfer claim, physical-mechanics validation or clinical use.
The source-bound declaration is prospective; partial attempts never become zero.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_real_spatial_policy as io
from run_real_patient_learning import run_episode

VERSION = "native-opening-learning-v1"
SETTINGS = {"seed": 11, "bc_updates": 16, "rl_updates": 16, "episodes_per_update": 4,
    "learning_rate": .001, "gamma": 1., "entropy_weight": .01, "value_weight": .5,
    "gradient_clip": 5., "random_episodes": 3, "max_steps": 2,
    "online_seconds": 10., "online_native_previews": 2048,
    "max_wall_seconds": 300., "max_rss_bytes": 2 * 1024 ** 3, "cpu_threads": 1}
SCRIPTS = ("scripts/run_native_opening_learning.py", "scripts/run_real_patient_learning.py",
           "scripts/preflight_real_spatial_policy.py", "scripts/compare_real_spatial_search.py")
METHODS = ("STOP", "initial", "random-0", "random-1", "random-2", "greedy", "depth2-search", "BC", "scratch-RL")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def dependency_paths():
    """Conservative local import closure, including imports inside functions.

    Unrelated desktop modules are excluded. Runtime module-origin checks also
    refuse a dynamic import absent from this frozen static closure.
    """
    pending = set(SCRIPTS) | {"src/resectionlab/__init__.py"}
    found = set()
    def add_module(name):
        base = ROOT / ("src/" + name.replace(".", "/") if name.startswith("resectionlab")
                       else "scripts/" + name.replace(".", "/"))
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file(): pending.add(str(path.relative_to(ROOT)))
    while pending:
        name = pending.pop()
        if name in found: continue
        found.add(name)
        module = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names: add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = ".".join(package.split(".")[:len(package.split(".")) - node.level + 1])
                    imported = prefix + ("." + node.module if node.module else "")
                else: imported = node.module or ""
                add_module(imported)
                for alias in node.names: add_module(imported + "." + alias.name)
    return tuple(sorted(found))


def sources(paths=None):
    expected = dependency_paths()
    paths = expected if paths is None else tuple(sorted(paths))
    if paths != expected or "src/resectionlab/data_policy.py" not in paths:
        raise ValueError("Source closure omitted the runner or admission boundary")
    result = {}
    for name in paths:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".py":
            raise ValueError("Source closure must contain local Python files")
        result[name] = io.sha256(path)
    return result


def imported_sources(record):
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename: continue
        path = Path(filename).resolve()
        owned = name == "resectionlab" or name.startswith("resectionlab.") or name in {
            Path(p).stem for p in SCRIPTS}
        if owned and (not path.is_relative_to(ROOT) or str(path.relative_to(ROOT)) not in record["source_sha256"]):
            raise ValueError("Executing undeclared local source: " + name)


def specification():
    return {"version": VERSION, "settings": SETTINGS,
        "task": {"factory": "make_native_opening_task", "arguments": {}, "horizon": 2,
            "scope": "one six-cell analytic native fixture; no anatomical population", "expected_optimum": 1.1},
        "policy": {"encoder_channels": [8, 16], "hidden_features": 32, "ray_samples": 5,
                   "physical_reference_mm": 10., "critic_candidate_context": True},
        "lineage": {"images": "generated", "transitions": "simulator_generated",
            "cavity_channel": "exact simulated committed removal; not observed operative cavity",
            "teacher": "complete depth-two search over permitted nominal planning clones",
            "real_patient_count": 0, "human_roles_opened": [], "clinical_use": False},
        "methods": list(METHODS),
        "checkpoint_rule": "initial and fixed latest only; no performance selection",
        "comparison": "full tiny source fits both actor and planner; same native actions/reward/horizon",
        "costs": "shared preparation, offline teacher/BC/RL, online planning/replay and independent audit reported separately"}


def declaration():
    return {**specification(), "source_sha256": sources()}


def validate(record):
    if canonical({key: value for key, value in record.items() if key != "source_sha256"}) != canonical(specification()):
        raise ValueError("Only the fixed generated opening development declaration is supported")
    if record.get("source_sha256") != sources(record.get("source_sha256", {})):
        raise ValueError("Declared numerical source changed")


def exact_teacher(base, *, check):
    """Enumerate the complete finite nominal tree; never use evaluator rewards."""
    from resectionlab.core import semantic_digest
    samples, proofs, rows, calls, terminals = {}, {}, [], 0, 0
    root = base.planning_clone()
    if not root.metrics()["planning_estimator_only"]:
        raise ValueError("Teacher must use the nominal planning model")

    def solve(task, prefix=()):
        nonlocal calls, terminals
        check()
        if task.terminated:
            terminals += 1
            return 0., ()
        observation = task.observation()
        scores, best, sequence = [], -float("inf"), None
        for identifier in observation.action_ids:
            check(); child = task.clone(); outcome = child.step(identifier); calls += 1
            remaining, suffix = solve(child, (*prefix, identifier))
            score = float(outcome.reward) + remaining
            scores.append({"action_id": identifier, "nominal_return_to_go": score})
            if score > best:  # STOP is first; deterministic earliest tie.
                best, sequence = score, (identifier, *suffix)
        key = observation.fingerprint
        prior = samples.get(key)
        if prior is not None and prior[1] != sequence[0]:
            raise ValueError("Identical actor observation has conflicting optimal teacher labels")
        samples[key] = (observation, sequence[0])
        proofs.setdefault(key, {"observation_hash": key, "selected_action_id": sequence[0],
            "prefix": list(prefix), "complete_demonstration": list((*prefix, *sequence))})
        rows.append({"observation_hash": key, "state_hash": semantic_digest(task.metrics()),
            "action_ids": list(observation.action_ids), "selected_action_id": sequence[0], "scores": scores})
        return best, sequence

    score, sequence = solve(root)
    return tuple(samples.values()), sequence, {"complete": True, "nominal_optimum": score,
        "model_transition_calls": calls, "terminal_sequences": terminals,
        "unique_supervised_states": len(samples), "state_rows": rows, "sequence": list(sequence),
        "supervised_state_demonstrations": list(proofs.values()),
        "observation_conflicts": 0, "reference_fields_used": False}


def guarded_episode(base, model, output, name, whole_guard, *, mode="argmax", generator=None,
                    planner=None, seconds=10., previews=2048):
    """Reuse the real episode writer, closing online cost before native audit."""
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.planning_budget import PlanningBudget
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    budget = PlanningBudget(NativeResectionEngine, max_native_previews=previews, seconds=seconds)
    active, extra = True, {}
    def check():
        whole_guard()
        if active: budget.check()
    try:
        with ExitStack() as stack:
            stack.enter_context(budget)
            sequence = None
            if planner is not None:
                with budget.phase("planning"):
                    sequence, extra = planner(check)
            stack.enter_context(budget.phase("execution"))
            def audit(task, *, metrics):
                nonlocal active
                terminal = json.loads((output / (name + ".json")).read_text())
                if (terminal.get("status") != "awaiting_independent_check" or not task.terminated
                        or canonical(terminal["metrics"]) != canonical(metrics)
                        or canonical(task.metrics()) != canonical(metrics)
                        or metrics["decision_model_hash"] != base.decision_model_hash):
                    raise ValueError("Complete durable history must match the actual native task")
                io.write_json(output / (name + "-terminal.json"), terminal)
                budget.complete(history_complete=True); stack.close(); active = False
                whole_guard()
                result = evaluate_native_spatial_episode(task, metrics=metrics, cancelled=lambda: bool(whole_guard()))
                whole_guard(); return result
            episodes, report = run_episode(base, model, generator, mode=mode, name=name, output=output,
                guard=check, audit=audit, sequence=sequence)
        accounting = budget.snapshot()
        if accounting["failure"] is not None or not accounting["counting_reliable"]:
            raise ValueError("Unreliable online preview accounting")
        io.write_json(output / (name + "-cost.json"), {"online": accounting, "planning": extra,
                     "independent_audit_seconds": report["independent_evaluation_seconds"]})
        report["guarded_online_seconds"] = accounting["elapsed_seconds"]
        return episodes, report, extra
    except BaseException:
        io.write_json(output / (name + "-cost.json"), {"online": budget.snapshot(), "planning": extra,
                                                     "outcomes": None})
        raise


def worker(record, output, *, declaration_sha256, profile=False):
    import torch
    from resectionlab.data_policy import GeneratedDevelopmentContext
    from resectionlab.native_spatial_task import make_native_opening_task
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.spatial_policy import (SpatialPolicy, SpatialPolicyConfig, SpatialTransition,
        parameter_hash, imitation_loss, reinforce_loss, gradient_step)
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    validate(record); torch.set_num_threads(1)
    started = time.perf_counter()
    results = {"version": VERSION, "status": "running", "real_patient_count": 0,
        "initial_parameter_hash": None, "preparation": None,
        "methods": {name: {"status": "not_started", "outcomes": None} for name in METHODS},
        "training": {name: {"status": "not_started", "updates": 0} for name in ("BC", "scratch-RL")}}
    if not profile: io.write_json(output / "result.json", results)
    def guard():
        limit = 60. if profile else SETTINGS["max_wall_seconds"]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak if sys.platform == "darwin" else peak * 1024
        if time.perf_counter() - started > limit or peak > SETTINGS["max_rss_bytes"]:
            raise InterruptedError("Declared worker time/memory envelope exhausted")
    prep_started = time.perf_counter()
    with NativePreviewProfiler(NativeResectionEngine) as profiler:
        base = make_native_opening_task(cancelled=lambda: bool(guard()))
        initial_observation = base.observation()
    prep = {"seconds": time.perf_counter() - prep_started, "preview_profile": profiler.snapshot(),
        "source_hash": base.case.source_hash, "reference_hash": base.case.reference_hash,
        "decision_model_hash": base.decision_model_hash, "initial_observation_hash": initial_observation.fingerprint,
        "inventory": base.candidate_inventory(), "lineage": record["lineage"]}
    io.write_json(output / "preparation.json", prep)
    context = GeneratedDevelopmentContext(declaration_sha256, (base.case.source_hash,), base.decision_model_hash)
    context.require_task(base)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SETTINGS["seed"])
        initial = SpatialPolicy(SpatialPolicyConfig(**record["policy"]))
    initial_weights = copy.deepcopy(initial.state_dict())
    initial_hash = parameter_hash(initial)
    torch.save({"policy": initial_weights, "parameter_hash": initial_hash,
                "architecture": initial.architecture_record(), "context": asdict(context)}, output / "initial.pt")
    if profile:
        # Exactly one known opening→cut episode and one update; no configuration search.
        task, transitions, forward_seconds = base.clone(), [], 0.
        with NativePreviewProfiler(NativeResectionEngine) as profile_preview:
            for tool, voxel in (("short-wide-opener", [4, 4, 1]), ("long-narrow-cutter", [4, 4, 5])):
                guard(); observation = task.observation()
                action = next(row["action_id"] for row in task.candidate_inventory()["ledger"]
                              if row["feasible"] and row["tool_id"] == tool and row["voxel"] == voxel)
                began = time.perf_counter()
                with torch.no_grad(): initial(observation)
                forward_seconds += time.perf_counter() - began
                result = task.step(action)
                transitions.append(SpatialTransition(observation, action, result.reward, result.terminated))
        audit = evaluate_native_spatial_episode(task, cancelled=lambda: bool(guard()))
        if audit.get("accepted") is not True:
            io.write_json(output / "profile-rejected-audit.json", audit)
            raise ValueError("Independent audit refused the profile demonstration before learning")
        optimizer = torch.optim.Adam(initial.parameters(), lr=SETTINGS["learning_rate"])
        began = time.perf_counter()
        loss, stats = imitation_loss(initial, [(t.observation, t.action_id) for t in transitions], learning_context=context)
        update = gradient_step(initial, optimizer, loss, learning_context=context)
        io.write_json(output / "profile.json", {"mode": "single_generated_development_episode_update",
            "preparation": prep, "actor_seconds": forward_seconds, "loss_update_seconds": time.perf_counter()-began,
            "native_preview_profile": profile_preview.snapshot(), "independent_evaluation": audit,
            "loss": stats, "update": update, "elapsed_seconds": time.perf_counter()-started,
            "scope": "one certified trajectory/BC gradient execution-cost check; discarded profile weights, no on-policy gain claim"})
        guard(); validate(record); imported_sources(record); return
    results.update(initial_parameter_hash=initial_hash, preparation=prep)
    def save():
        results["elapsed_seconds"] = time.perf_counter()-started
        io.write_json(output / "result.json", results); guard()
    save()
    def evaluate(name, model, *, mode="argmax", planner=None, generator=None):
        results["methods"][name]["status"] = "running"; save()
        try:
            _, report, extra = guarded_episode(base, model, output, name, guard, mode=mode,
                planner=planner, generator=generator, seconds=SETTINGS["online_seconds"], previews=SETTINGS["online_native_previews"])
            results["methods"][name] = {"status": "complete", "actions": [d["action_id"] for d in report["decisions"]],
                "parameter_hash": parameter_hash(model), "outcomes": report["independent_evaluation"]["outcomes"],
                "accepted": True, "online_seconds": report["guarded_online_seconds"],
                "replay_seconds": report["online_seconds"], "planning": extra}
            save()
        except BaseException as error:
            results["methods"][name] = {"status": "failed", "outcomes": None,
                "failure": {"type": type(error).__name__, "message": str(error)}}
            io.write_json(output / "result.json", results)
            raise
    evaluate("STOP", initial, mode="sequence", planner=lambda check: (("STOP",), {}))
    evaluate("initial", initial)
    for i in range(SETTINGS["random_episodes"]):
        evaluate("random-" + str(i), initial, mode="random",
                 generator=torch.Generator().manual_seed(SETTINGS["seed"] + 1000 + i))
    evaluate("greedy", initial, mode="sequence", planner=lambda check: base.observed_greedy_search(seconds=10.))
    teacher_samples = None
    def teacher(check):
        nonlocal teacher_samples
        teacher_samples, sequence, receipt = exact_teacher(base, check=check)
        io.write_json(output / "teacher.json", receipt)
        return sequence, receipt
    evaluate("depth2-search", initial, mode="sequence", planner=teacher)
    if teacher_samples is None or not teacher_samples:
        raise RuntimeError("Complete replayed teacher is required for imitation")
    teacher_started, teacher_checks = time.perf_counter(), []
    for index, state in enumerate(results["methods"]["depth2-search"]["planning"]["supervised_state_demonstrations"]):
        sequence = tuple(state["complete_demonstration"])
        episode, report, _ = guarded_episode(base, initial, output, f"teacher-state-{index:02}", guard,
            mode="sequence", planner=lambda check, sequence=sequence: (sequence, {}),
            seconds=SETTINGS["online_seconds"], previews=SETTINGS["online_native_previews"])
        label_step = episode[len(state["prefix"])]
        if (label_step.observation.fingerprint != state["observation_hash"]
                or label_step.action_id != state["selected_action_id"]):
            raise ValueError("Independently replayed teacher state/action differs from searched label")
        teacher_checks.append({**state, "accepted": True, "transitions": len(episode),
            "guarded_replay_seconds": report["guarded_online_seconds"],
            "audit_seconds": report["independent_evaluation_seconds"]})
        io.write_json(output / "teacher-demonstration-checks.json", teacher_checks)
    results["offline_teacher_validation"] = {"seconds": time.perf_counter()-teacher_started,
        "complete_episodes": len(teacher_checks), "transitions": sum(row["transitions"] for row in teacher_checks),
        "reference_reward_not_used_for_labels": True}
    save()
    for arm in ("BC", "scratch-RL"):
        model = copy.deepcopy(initial)
        if parameter_hash(model) != initial_hash:
            raise ValueError("Paired initialization changed")
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
        rng = torch.Generator().manual_seed(SETTINGS["seed"])
        rows = []; fit_started = time.perf_counter()
        results["training"][arm]["status"] = "running"; save()
        for step in range(SETTINGS["bc_updates"] if arm == "BC" else SETTINGS["rl_updates"]):
            guard(); context.require_task(base); episodes, episode_rows = [], []
            if arm == "scratch-RL":
                before_collection = parameter_hash(model)
                for i in range(SETTINGS["episodes_per_update"]):
                    ep, report, _ = guarded_episode(base, model, output, f"rl-{step:02}-{i}", guard,
                        mode="sample", generator=rng, seconds=SETTINGS["online_seconds"], previews=SETTINGS["online_native_previews"])
                    episodes.append(ep); episode_rows.append({"return": report["simulated_return"],
                        "decisions": len(ep), "outcomes": report["independent_evaluation"]["outcomes"]})
                if parameter_hash(model) != before_collection:
                    raise ValueError("On-policy batch weights changed during collection")
            loss_started = time.perf_counter()
            if arm == "BC":
                loss, stats = imitation_loss(model, teacher_samples, learning_context=context)
            else:
                loss, stats = reinforce_loss(model, episodes, gamma=SETTINGS["gamma"],
                    entropy_weight=SETTINGS["entropy_weight"], value_weight=SETTINGS["value_weight"], learning_context=context)
            update = gradient_step(model, optimizer, loss, max_norm=SETTINGS["gradient_clip"], learning_context=context)
            rows.append({"update": step+1, "loss": stats, "gradient": update, "episodes": episode_rows,
                "loss_update_seconds": time.perf_counter()-loss_started, "fit_elapsed_seconds": time.perf_counter()-fit_started})
            io.write_json(output / (arm + "-updates.json"), rows)
            torch.save({"policy": model.state_dict(), "optimizer": optimizer.state_dict(), "updates": step+1,
                "parameter_hash": parameter_hash(model), "initial_parameter_hash": initial_hash,
                "architecture": model.architecture_record(), "context": asdict(context)}, output / (arm + "-latest.pt"))
            results["training"][arm].update(updates=step+1, latest_parameter_hash=parameter_hash(model)); save()
        results["training"][arm] = {"status": "complete", "updates": len(rows), "fit_seconds": time.perf_counter()-fit_started,
            "initial_parameter_hash": initial_hash, "latest_parameter_hash": parameter_hash(model),
            "optimization_transitions": sum(e["decisions"] for row in rows for e in row["episodes"]),
            "loss_forward_calls": sum(row["loss"]["loss_forward_calls"] for row in rows)}
        evaluate(arm, model)
    validate(record); imported_sources(record)
    results["status"] = "complete"; save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--write-declaration", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.write_declaration:
        if args.declaration.exists(): raise ValueError("Preserve earlier declaration bytes")
        args.declaration.parent.mkdir(parents=True, exist_ok=True)
        io.write_json(args.declaration, declaration()); return
    record, declared_hash, payload = io.read_declaration(args.declaration, args.expected_sha256)
    validate(record)
    if args.worker:
        if not args.expected_sha256: raise ValueError("Worker requires frozen declaration bytes")
        try: worker(record, args.output, declaration_sha256=declared_hash, profile=args.profile)
        except BaseException as error:
            io.write_json(args.output / "failure.json", {"type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "outcomes": None})
            result_path = args.output / "result.json"
            if result_path.exists():
                partial = json.loads(result_path.read_text()); partial["status"] = "failed"
                partial["failure"] = {"type": type(error).__name__, "message": str(error)}
                io.write_json(result_path, partial)
            raise
        return
    if not (args.execute or args.profile):
        print("Prospective declaration validated; no task or policy created."); return
    if args.output is None or args.output.exists(): raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    frozen = args.output / "declaration-input.json"; frozen.write_bytes(payload)
    for name in record["source_sha256"]:
        dest = args.output / "source-snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--declaration", str(frozen.resolve()),
               "--expected-sha256", declared_hash, "--output", str(args.output.resolve())]
    if args.profile: command.append("--profile")
    limits = {"max_wall_seconds": 60. if args.profile else SETTINGS["max_wall_seconds"],
              "max_rss_bytes": SETTINGS["max_rss_bytes"]}
    supervised = io.supervise_worker(command, args.output, limits, declared_hash)
    io.write_json(args.output / "output-sha256.json", {str(p.relative_to(args.output)): io.sha256(p)
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "output-sha256.json"})
    if supervised["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__":
    main()
