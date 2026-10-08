#!/usr/bin/env python3
"""One longer unchanged scratch REINFORCE fit; cached tree is evaluation only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_native_opening_bc_capacity as evaluation

prior, io, canonical = evaluation.prior, evaluation.io, evaluation.canonical
VERSION = "native-opening-rl-capacity-v1"
SCRIPT = "scripts/run_native_opening_rl_capacity.py"
SAVED, INITIAL_HASH = evaluation.SAVED, evaluation.INITIAL_HASH
RL16_HASH = "sha256:5134b5ff817c87dbc31f9b51fb082626a06883efb75471cfe38e8d41f6ccbfa7"
READOUTS = (0, 16, 32, 64, 128, 256)
SETTINGS = {"seed": 11, "updates": 256, "episodes_per_update": 4, "learning_rate": .001,
    "gamma": 1., "entropy_weight": .01, "value_weight": .5, "gradient_clip": 5.,
    "readouts": list(READOUTS), "max_steps": 2, "cpu_threads": 1,
    "max_wall_seconds": 180., "max_rss_bytes": 2 * 1024 ** 3,
    "online_seconds": 10., "online_native_previews": 2048}
ORIGINAL_EPISODES = tuple(f"rl-{update:02}-{episode}.json" for update in range(16) for episode in range(4))
INPUTS = (*evaluation.INPUTS, "scratch-RL-updates.json", "scratch-RL-latest.pt", *ORIGINAL_EPISODES)


def specification():
    return {"version": VERSION, "settings": SETTINGS, "policy": prior.specification()["policy"],
        "old_declaration_sha256": evaluation.OLD_DECLARATION_SHA256,
        "initial_parameter_hash": INITIAL_HASH, "update16_parameter_hash": RL16_HASH,
        "algorithm": "unchanged masked REINFORCE with spatial value baseline; no PPO, imitation or replay updates",
        "sampling": "four fresh complete on-policy episodes per update; original seed11 private action RNG",
        "optimizer": "fresh original Adam defaults and learning rate; clip5; no warm start",
        "evaluation": "five reconstructed states and sixteen authenticated terminal routes; privileged model assessment only",
        "primary": "fixed update256; no performance selection, early success stop, retry or budget extension",
        "lineage": prior.specification()["lineage"],
        "scope": "one generated training-fit diagnostic; no patient or held-out generalization claim",
        "costs": "all sampled episodes, repeated loss forwards, audits, preparation, evaluation reconstruction and readouts counted"}


def sources():
    return {**evaluation.sources(), SCRIPT: io.sha256(ROOT / SCRIPT)}


def declaration():
    return {**specification(), "source_sha256": sources(),
        "input_sha256": {name: io.sha256(SAVED / name) for name in INPUTS}}


def evaluation_binding(record):
    return {**evaluation.specification(),
        "source_sha256": {name: digest for name, digest in record["source_sha256"].items() if name != SCRIPT},
        "input_sha256": {name: record["input_sha256"][name] for name in evaluation.INPUTS}}


def validate(record):
    if canonical({k: v for k, v in record.items() if k not in ("source_sha256", "input_sha256")}) != canonical(specification()):
        raise ValueError("Only the fixed unchanged RL256 declaration is supported")
    if record.get("source_sha256") != sources() or set(record.get("input_sha256", {})) != set(INPUTS):
        raise ValueError("Complete source/input closure changed")
    evaluation.validate(evaluation_binding(record))
    index = json.loads((SAVED / "output-sha256.json").read_text())
    for name, expected in record["input_sha256"].items():
        if io.sha256(SAVED / name) != expected or (name != "output-sha256.json" and index.get(name) != expected):
            raise ValueError("Original RL receipt/checkpoint changed: " + name)
    if Path(sys.modules[__name__].__file__).resolve() != (ROOT / SCRIPT).resolve():
        raise ValueError("Executing runner outside its declared source root")


def trajectory_signature(report):
    """Only numerical behavior, never clocks, filenames or new declaration IDs."""
    fields = ("observation_hash", "action_ids", "action_mask", "action_id", "selected_index",
              "logits", "probabilities", "value", "reward", "terminated", "behavior_parameter_hash")
    return [{name: decision[name] for name in fields} for decision in report["decisions"]]


def load_evaluation_tree(record):
    """Read complete old outcomes for assessment; nothing returned is a training batch."""
    validate(record)
    bundle = evaluation.load_frozen_teacher(evaluation_binding(record))
    original_updates = json.loads((SAVED / "scratch-RL-updates.json").read_text())
    if (len(original_updates) != 16 or bundle["result"]["training"]["scratch-RL"]["latest_parameter_hash"] != RL16_HASH
            or bundle["result"]["training"]["scratch-RL"]["updates"] != 16):
        raise ValueError("Original scratch RL ancestry differs")
    root_hash = bundle["preparation"]["initial_observation_hash"]
    root = bundle["rows"][root_hash]
    children = {tuple(p["prefix"]): bundle["rows"][p["observation_hash"]]
                for p in bundle["teacher-demonstration-checks"] if p["prefix"]}
    expected_paths = {("STOP",)} | {(first, second) for first in root["action_ids"][1:]
        for second in children[(first,)]["action_ids"]}
    if len(expected_paths) != 16:
        raise ValueError("Original complete finite route denominator differs")
    root_scores = {s["action_id"]: s["nominal_return_to_go"] for s in root["scores"]}
    routes, originals = {}, {}
    for name in ORIGINAL_EPISODES:
        report = json.loads((SAVED / name).read_text())
        outcome = report["independent_evaluation"]["outcomes"]
        if (report["status"] != "complete" or report["independent_evaluation"]["accepted"] is not True
                or report["metrics"]["terminated"] is not True or report["invalid_actions"] != 0):
            raise ValueError("Only completed independently accepted old routes support assessment")
        path = tuple(d["action_id"] for d in report["decisions"])
        if path not in expected_paths:
            raise ValueError("Saved route outside the complete nominal tree")
        nodes = []
        for i, decision in enumerate(report["decisions"]):
            state = root if i == 0 else children[(path[0],)]
            if decision["observation_hash"] != state["observation_hash"] or decision["action_ids"] != state["action_ids"]:
                raise ValueError("Route observation/inventory differs from evaluation tree")
            nodes.append({"observation_hash": decision["observation_hash"], "action_id": decision["action_id"]})
        if len(path) == 1:
            nominal = root_scores["STOP"]
        else:
            continuation = {s["action_id"]: s["nominal_return_to_go"] for s in children[(path[0],)]["scores"]}
            # At depth one, scores are immediate terminal rewards. Root Q* minus
            # max child reward recovers the fixed opening reward, not a teacher label.
            nominal = root_scores[path[0]] - max(continuation.values()) + continuation[path[1]]
        if (not math.isfinite(nominal) or not math.isclose(nominal, report["simulated_return"], abs_tol=1e-12)
                or not math.isclose(nominal, outcome["total_reward"], abs_tol=1e-12)):
            raise ValueError("Nominal tree return disagrees with the audited fixture route")
        row = {"path": list(path), "nodes": nodes, "nominal_return": nominal,
            "any_target": outcome["positive_target_source_cells_removed"] > 0,
            "full_target": math.isclose(outcome["target_removed_mm3"], outcome["total_reference_target_mm3"], abs_tol=1e-12),
            "target_removed_mm3": outcome["target_removed_mm3"], "normal_removed_mm3": outcome["normal_removed_mm3"]}
        if path in routes and canonical(routes[path]) != canonical(row):
            raise ValueError("Repeated saved route has inconsistent outcomes")
        routes[path] = row; originals[name] = report
    if set(routes) != expected_paths:
        raise ValueError("A complete terminal route is missing from independent saved outcomes")
    return bundle, {"root_observation_hash": root_hash, "routes": [routes[p] for p in sorted(routes)],
        "state_actions": {key: row["action_ids"] for key, row in bundle["rows"].items()},
        "scope": "privileged nominal-model expectation; target indicators from matching audited generated fixture routes",
        "original_episodes": originals, "original_updates": original_updates}


def expected_metrics(readout, tree):
    by_state = {}
    for row in readout["states"]:
        key, ids, probabilities = row["observation_hash"], row["action_ids"], row["probabilities"]
        if (key in by_state or tree["state_actions"].get(key) != ids or len(ids) != len(probabilities)
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or not math.isclose(math.fsum(probabilities), 1., abs_tol=2e-6)):
            raise ValueError("Readout state/action/probability binding differs")
        # Float32 softmax may sum to 1 +/- roundoff. Explicitly normalize the
        # saved probabilities for finite-tree arithmetic, never for training.
        total = math.fsum(probabilities)
        by_state[key] = {action: p / total for action, p in zip(ids, probabilities)}
    if set(by_state) != set(tree["state_actions"]):
        raise ValueError("All five evaluation states are required")
    rows = []
    for route in tree["routes"]:
        probability = math.prod(by_state[n["observation_hash"]][n["action_id"]] for n in route["nodes"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid evaluation route probability")
        rows.append({**route, "probability": probability})
    total = math.fsum(row["probability"] for row in rows)
    if len(rows) != 16 or not math.isclose(total, 1., abs_tol=2e-6):
        raise ValueError("Evaluation probabilities do not cover the complete terminal tree")
    return {"nominal_expected_return": math.fsum(r["probability"] * r["nominal_return"] for r in rows),
        "generated_any_target_probability": math.fsum(r["probability"] for r in rows if r["any_target"]),
        "generated_full_target_probability": math.fsum(r["probability"] for r in rows if r["full_target"]),
        "expected_target_mm3": math.fsum(r["probability"] * r["target_removed_mm3"] for r in rows),
        "expected_normal_mm3": math.fsum(r["probability"] * r["normal_removed_mm3"] for r in rows),
        "root_stop_probability": by_state[tree["root_observation_hash"]]["STOP"],
        "terminal_probability_sum": total, "routes": rows, "scope": tree["scope"]}


def verify_original_episode(update, episode, report, tree):
    original = tree["original_episodes"][f"rl-{update-1:02}-{episode}.json"]
    if canonical(trajectory_signature(report)) != canonical(trajectory_signature(original)):
        raise ValueError("First16 sampled observation/action/reward trajectory differs")


def verify_original_update(update, row, tree):
    original = tree["original_updates"][update - 1]
    if canonical(row["gradient"]) != canonical(original["gradient"]) or canonical(row["loss"]) != canonical(original["loss"]):
        raise ValueError("First16 loss/gradient/hash trajectory differs")
    if update == 16 and row["gradient"]["updated_parameter_hash"] != RL16_HASH:
        raise ValueError("Update16 original checkpoint not reproduced")


def worker(record, output, *, declaration_sha256):
    import torch
    from resectionlab.data_policy import GeneratedDevelopmentContext
    from resectionlab.native_spatial_task import make_native_opening_task
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash, reinforce_loss, gradient_step
    started = time.perf_counter()
    result = {"version": VERSION, "status": "running", "updates": 0, "completed_training_episodes": 0,
        "training_transitions": 0, "collection_forward_calls": 0, "loss_forward_calls": 0, "readout_forward_calls": 0,
        "active_update": None, "readouts": {}, "real_patient_count": 0,
        "methods": {name: {"status": "not_started", "outcomes": None} for name in ("initial", "RL256")}}
    def guard():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if time.perf_counter()-started >= SETTINGS["max_wall_seconds"] or peak > SETTINGS["max_rss_bytes"]:
            raise InterruptedError("Fixed worker time/memory envelope exhausted")
    def save():
        result["elapsed_seconds"] = time.perf_counter()-started
        io.write_json(output / "result.json", result)
    save()
    try:
        validate(record); torch.set_num_threads(1)
        bundle, tree = load_evaluation_tree(record)
        began = time.perf_counter()
        with NativePreviewProfiler(NativeResectionEngine) as profiler:
            base = make_native_opening_task(cancelled=lambda: bool(guard()))
        result["preparation"] = {"seconds": time.perf_counter()-began, "preview_profile": profiler.snapshot()}
        context = GeneratedDevelopmentContext(declaration_sha256, (base.case.source_hash,), base.decision_model_hash)
        # Cached nominal states are readout inputs only. Collection below always
        # executes the actual current policy in fresh native simulator clones.
        evaluation_samples, result["evaluation_reconstruction"] = evaluation.reconstruct_teacher(base, bundle, context, guard)
        io.write_json(output / "evaluation-tree.json", {k: v for k, v in tree.items() if not k.startswith("original_")})
        result["inherited_evaluation_costs"] = {"source_experiment_seconds": bundle["result"]["elapsed_seconds"],
            "teacher_model_calls": bundle["teacher"]["model_transition_calls"],
            "teacher_validation": bundle["result"]["offline_teacher_validation"],
            "prior_RL_training": bundle["result"]["training"]["scratch-RL"],
            "scope": "previously paid complete-tree and audited-route evidence; assessment only, not training replay"}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SETTINGS["seed"])
            model = SpatialPolicy(SpatialPolicyConfig(**record["policy"]))
        if parameter_hash(model) != INITIAL_HASH:
            raise ValueError("Fresh scratch initialization differs")
        result.update(initial_parameter_hash=parameter_hash(model), architecture_hash=model.architecture_hash,
            context=asdict(context), parameter_count=sum(p.numel() for p in model.parameters()))
        def checkpoint(update, optimizer=None):
            torch.save({"policy": model.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(),
                "updates": update, "parameter_hash": parameter_hash(model), "initial_parameter_hash": INITIAL_HASH,
                "architecture": model.architecture_record(), "context": asdict(context),
                "action_rng": None if update == 0 else rng.get_state()}, output / f"RL-{update:03}.pt")
        def inspect(update):
            before_rng = rng.get_state().clone()
            row = evaluation.readout(model, evaluation_samples, guard)
            row["privileged_model_assessment"] = expected_metrics(row, tree)
            if not torch.equal(rng.get_state(), before_rng):
                raise ValueError("Readout changed training action RNG")
            result["readouts"][str(update)] = row
            result["readout_forward_calls"] += row["forward_calls"]; save()
        def evaluate(name):
            before = parameter_hash(model)
            result["methods"][name]["status"] = "running"; save()
            _, report, _ = prior.guarded_episode(base, model, output, name, guard,
                seconds=SETTINGS["online_seconds"], previews=SETTINGS["online_native_previews"])
            if report["independent_evaluation"]["accepted"] is not True or parameter_hash(model) != before:
                raise ValueError("Actual inference audit refused or weights changed")
            result["methods"][name] = {"status": "complete", "accepted": True, "parameter_hash": before,
                "actions": [d["action_id"] for d in report["decisions"]],
                "outcomes": report["independent_evaluation"]["outcomes"],
                "online_seconds": report["guarded_online_seconds"], "audit_seconds": report["independent_evaluation_seconds"]}; save()
        rng = torch.Generator().manual_seed(SETTINGS["seed"])
        checkpoint(0); inspect(0); evaluate("initial")
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
        updates, fit_started = [], time.perf_counter()
        for update in range(1, SETTINGS["updates"] + 1):
            guard(); context.require_task(base)
            batch, episode_rows, before = [], [], parameter_hash(model)
            result["active_update"] = {"update": update, "status": "collecting",
                "episodes": [{"status": "not_started", "outcomes": None} for _ in range(4)]}; save()
            for episode_index in range(4):
                name = f"rl-{update-1:03}-{episode_index}"
                active = result["active_update"]["episodes"][episode_index]
                active.update(status="running", receipt=name + ".json"); save()
                episode, report, _ = prior.guarded_episode(base, model, output, name, guard,
                    mode="sample", generator=rng, seconds=SETTINGS["online_seconds"], previews=SETTINGS["online_native_previews"])
                if (report["independent_evaluation"]["accepted"] is not True or not episode[-1].terminated
                        or parameter_hash(model) != before):
                    raise ValueError("Complete independently audited unchanged-policy collection required")
                if update <= 16: verify_original_episode(update, episode_index, report, tree)
                costs = json.loads((output / (name + "-cost.json")).read_text())
                row = {"receipt": name + ".json", "return": report["simulated_return"], "decisions": len(episode),
                    "outcomes": report["independent_evaluation"]["outcomes"],
                    "online_seconds": report["guarded_online_seconds"], "audit_seconds": report["independent_evaluation_seconds"],
                    "native_preview_entries": costs["online"]["native_preview_entries"]}
                batch.append(episode); episode_rows.append(row); active.update(status="complete", **row)
                result["completed_training_episodes"] += 1; result["training_transitions"] += len(episode)
                result["collection_forward_calls"] += report["policy_forward_calls"]; save()
            result["active_update"]["status"] = "updating"; save(); guard()
            loss_started = time.perf_counter()
            # Only the newly executed on-policy transitions enter this loss.
            loss, stats = reinforce_loss(model, batch, gamma=SETTINGS["gamma"], entropy_weight=SETTINGS["entropy_weight"],
                value_weight=SETTINGS["value_weight"], learning_context=context)
            gradient = gradient_step(model, optimizer, loss, max_norm=SETTINGS["gradient_clip"], learning_context=context)
            row = {"update": update, "loss": stats, "gradient": gradient, "episodes": episode_rows,
                "loss_update_seconds": time.perf_counter()-loss_started}
            updates.append(row); result["updates"] = update; result["loss_forward_calls"] += stats["loss_forward_calls"]
            io.write_json(output / "updates.json", updates)
            if update <= 16: verify_original_update(update, row, tree)
            if update == 16: result["first16_exact_reproduction"] = True
            result["active_update"]["status"] = "complete"
            if update in READOUTS: checkpoint(update, optimizer); inspect(update)
            save()
        result["fit_with_readout_export_seconds"] = time.perf_counter()-fit_started
        result["training_online_seconds"] = sum(e["online_seconds"] for r in updates for e in r["episodes"])
        result["training_audit_seconds"] = sum(e["audit_seconds"] for r in updates for e in r["episodes"])
        result["training_native_previews"] = sum(e["native_preview_entries"] for r in updates for e in r["episodes"])
        result["loss_update_seconds"] = sum(r["loss_update_seconds"] for r in updates)
        evaluate("RL256"); validate(record)
        if (result["completed_training_episodes"] != 1024 or result["readout_forward_calls"] != 30
                or result["training_transitions"] > 2048 or result["loss_forward_calls"] != result["training_transitions"]):
            raise ValueError("Fixed actual workload accounting differs")
        result["status"] = "complete"; save()
    except BaseException as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if result["active_update"] is not None:
            result["active_update"]["status"] = "failed"
            for row in result["active_update"]["episodes"]:
                if row["status"] == "running": row.update(status="failed", outcomes=None)
        for row in result["methods"].values():
            if row["status"] == "running": row.update(status="failed", outcomes=None)
        save(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--write-declaration", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.write_declaration:
        if args.declaration.exists(): raise ValueError("Preserve previous declaration bytes")
        io.write_json(args.declaration, declaration()); return
    record, declared_hash, payload = io.read_declaration(args.declaration, args.expected_sha256); validate(record)
    if args.worker:
        if not args.expected_sha256: raise ValueError("Worker requires exact declaration SHA")
        worker(record, args.output, declaration_sha256=declared_hash); return
    if not args.execute:
        print("RL256 declaration validated; no model or task created."); return
    if args.output is None or args.output.exists(): raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    frozen = args.output / "declaration-input.json"; frozen.write_bytes(payload)
    for name in record["source_sha256"]:
        destination = args.output / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, destination)
    io.write_json(args.output / "inherited-input-sha256.json", record["input_sha256"])
    supervised = io.supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker", "--declaration",
        str(frozen.resolve()), "--expected-sha256", declared_hash, "--output", str(args.output.resolve())], args.output, SETTINGS, declared_hash)
    io.write_json(args.output / "output-sha256.json", {str(p.relative_to(args.output)): io.sha256(p)
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "output-sha256.json"})
    if supervised["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__":
    main()
