#!/usr/bin/env python3
"""Fixed longer fit of the original five audited generated teacher states."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
import run_native_opening_learning as prior

io, canonical = prior.io, prior.canonical
VERSION = "native-opening-bc-capacity-v1"
SAVED = ROOT / "artifacts/native-opening-learning-v1"
SCRIPT = "scripts/run_native_opening_bc_capacity.py"
OLD_DECLARATION_SHA256 = "963238d34dce36f4a9128c782f0d3ca667640123a58a2eb2498abb01d68c5212"
INITIAL_HASH = "sha256:2fa97c0e730db09371786d5ba903ffdf467b3cc7149a111c182d6a7499cdd9b8"
BC16_HASH = "sha256:1fbf11d5bd9e78ce26a55a21ceb4e4a1a6126cac5842b17bfa26353e1b453a7f"
READOUTS = (0, 16, 32, 64, 128, 256)
SETTINGS = {"seed": 11, "updates": 256, "samples_per_update": 5, "learning_rate": .001,
    "gradient_clip": 5., "readouts": list(READOUTS), "max_steps": 2, "cpu_threads": 1,
    "max_wall_seconds": 60., "max_rss_bytes": 2 * 1024 ** 3,
    "online_seconds": 10., "online_native_previews": 2048}
INPUTS = ("output-sha256.json", "declaration-input.json", "result.json", "preparation.json",
    "teacher.json", "teacher-demonstration-checks.json", "initial.pt", "BC-latest.pt",
    "BC-updates.json", *(f"teacher-state-{i:02}.json" for i in range(5)), "depth2-search-cost.json")


def specification():
    return {"version": VERSION, "settings": SETTINGS, "policy": prior.specification()["policy"],
        "old_declaration_sha256": OLD_DECLARATION_SHA256,
        "initial_parameter_hash": INITIAL_HASH, "update16_parameter_hash": BC16_HASH,
        "teacher": "same five audited states in original order; four nominal prefix transitions; no new search",
        "sampling": "all five equally weighted on every update; no reweighting, shuffle or filtering",
        "optimizer": "fresh Adam with original defaults, learning rate and clip; no state warm start",
        "primary": "fixed update256 even if worse; intermediate readouts never choose a checkpoint",
        "lineage": prior.specification()["lineage"],
        "scope": "one generated training-fit diagnostic; no RL, patient, holdout, generalization or clinical claim",
        "costs": "inherited teacher/search/audit costs plus new preparation, reconstruction, BC, readouts and audited inference"}


def sources():
    return {**prior.sources(), SCRIPT: io.sha256(ROOT / SCRIPT)}


def declaration():
    return {**specification(), "source_sha256": sources(),
        "input_sha256": {name: io.sha256(SAVED / name) for name in INPUTS}}


def validate(record):
    if canonical({k: v for k, v in record.items() if k not in ("source_sha256", "input_sha256")}) != canonical(specification()):
        raise ValueError("Only the fixed 256-update capacity diagnostic is supported")
    if record.get("source_sha256") != sources():
        raise ValueError("Complete numerical source closure changed")
    if set(record.get("input_sha256", {})) != set(INPUTS):
        raise ValueError("Missing original teacher/checkpoint inputs")
    for name, expected in record["input_sha256"].items():
        if io.sha256(SAVED / name) != expected:
            raise ValueError("Original immutable input changed: " + name)
    if record["input_sha256"]["declaration-input.json"] != OLD_DECLARATION_SHA256:
        raise ValueError("Wrong original experiment declaration")
    index = json.loads((SAVED / "output-sha256.json").read_text())
    if any(index.get(name) != value for name, value in record["input_sha256"].items()
           if name != "output-sha256.json"):
        raise ValueError("Original execution index differs from inherited inputs")
    old = json.loads((SAVED / "declaration-input.json").read_text())
    if old["source_sha256"] != {k: v for k, v in record["source_sha256"].items() if k != SCRIPT}:
        raise ValueError("Production numerical sources differ from the original fit")
    prior.imported_sources(record)
    own = Path(sys.modules[__name__].__file__).resolve()
    if own != (ROOT / SCRIPT).resolve():
        raise ValueError("Capacity runner imported outside its frozen source root")


def load_frozen_teacher(record):
    """Authenticate original declarations, nominal labels and completed audits."""
    validate(record)
    bundle = {name: json.loads((SAVED / (name + ".json")).read_text())
              for name in ("teacher", "teacher-demonstration-checks", "result", "preparation", "BC-updates")}
    teacher, proofs, result = bundle["teacher"], bundle["teacher-demonstration-checks"], bundle["result"]
    if (result["status"] != "complete" or result["initial_parameter_hash"] != INITIAL_HASH
            or result["training"]["BC"]["latest_parameter_hash"] != BC16_HASH
            or result["training"]["BC"]["updates"] != 16 or not teacher["complete"]
            or teacher["reference_fields_used"] is not False
            or teacher["terminal_sequences"] != 16 or teacher["model_transition_calls"] != 20
            or teacher["unique_supervised_states"] != 5 or len(proofs) != 5):
        raise ValueError("Original complete teacher and 16-update ancestry required")
    rows = {row["observation_hash"]: row for row in teacher["state_rows"]}
    if len(rows) != 5 or len(teacher["supervised_state_demonstrations"]) != 5:
        raise ValueError("Original five-state teacher denominator changed")
    for i, proof in enumerate(proofs):
        declared = teacher["supervised_state_demonstrations"][i]
        if (any(proof.get(key) != value for key, value in declared.items()) or proof.get("accepted") is not True
                or len(proof["prefix"]) not in (0, 1)
                or proof["complete_demonstration"][:len(proof["prefix"])] != proof["prefix"]):
            raise ValueError("Teacher prefix or validation lineage differs")
        episode = json.loads((SAVED / f"teacher-state-{i:02}.json").read_text())
        decisions = episode["decisions"]
        at_label = decisions[len(proof["prefix"])]
        expected = rows[proof["observation_hash"]]
        if (episode["status"] != "complete" or episode["independent_evaluation"]["accepted"] is not True
                or episode["metrics"]["terminated"] is not True
                or [d["action_id"] for d in decisions] != proof["complete_demonstration"]
                or at_label["observation_hash"] != proof["observation_hash"]
                or at_label["action_id"] != proof["selected_action_id"]
                or expected["selected_action_id"] != proof["selected_action_id"]
                or at_label["action_ids"] != expected["action_ids"]):
            raise ValueError("Inherited independent demonstration does not bind the teacher label")
    if sum(len(p["prefix"]) for p in proofs) != 4 or sum(p["selected_action_id"] == "STOP" for p in proofs) != 2:
        raise ValueError("Original prefix count or label composition changed")
    bundle["rows"] = rows
    return bundle


def reconstruct_teacher(base, bundle, context, guard):
    """Replay only the four original one-step nominal prefixes; never relabel."""
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
    started, samples, states, transitions = time.perf_counter(), [], [], 0
    context.require_task(base)
    prep = bundle["preparation"]
    if (base.case.source_hash != prep["source_hash"] or base.decision_model_hash != prep["decision_model_hash"]
            or base.observation().fingerprint != prep["initial_observation_hash"]):
        raise ValueError("Original source/model/initial observation changed")
    with NativePreviewProfiler(NativeResectionEngine) as profiler:
        for proof in bundle["teacher-demonstration-checks"]:
            guard(); context.require_task(base)
            task = base.planning_clone()
            if not task.metrics()["planning_estimator_only"]:
                raise ValueError("Reconstruction must use only nominal planning clones")
            for action in proof["prefix"]:
                guard(); transitions += 1
                if transitions > 4 or task.terminated:
                    raise ValueError("Original nonterminal prefix budget changed")
                task.step(action)
            observation = task.observation()
            context.require_observations((observation,))
            expected = bundle["rows"][proof["observation_hash"]]
            if (task.terminated or observation.fingerprint != proof["observation_hash"]
                    or list(observation.action_ids) != expected["action_ids"]
                    or proof["selected_action_id"] != expected["selected_action_id"]
                    or not all(observation.action_mask)):
                raise ValueError("Reconstructed prefix observation/inventory/label differs")
            samples.append((observation, proof["selected_action_id"]))
            states.append({"observation_hash": observation.fingerprint, "prefix": proof["prefix"],
                "action_ids": list(observation.action_ids), "label": proof["selected_action_id"]})
    if transitions != 4 or len(samples) != 5:
        raise ValueError("Exactly five inherited states and four prefix transitions required")
    return tuple(samples), {"seconds": time.perf_counter()-started, "prefix_transitions": transitions,
        "preview_profile": profiler.snapshot(), "states": states, "new_teacher_search_calls": 0}


def readout(model, samples, guard):
    import torch
    from resectionlab.spatial_policy import parameter_hash
    began, before, rows = time.perf_counter(), parameter_hash(model), []
    with torch.no_grad():
        for observation, label in samples:
            guard(); logits, value = model(observation)
            index = observation.action_ids.index(label)
            probabilities = logits.softmax(-1)
            other = logits.clone(); other[index] = -torch.inf
            top = int(logits.argmax())
            rows.append({"observation_hash": observation.fingerprint, "teacher_action": label,
                "teacher_cross_entropy": float(-logits.log_softmax(-1)[index]),
                "teacher_probability": float(probabilities[index]),
                "teacher_rank": 1 + int((logits > logits[index]).sum()),
                "teacher_logit_margin": float(logits[index] - other.max()),
                "stop_minus_best_nonstop_logit": float(logits[0] - logits[1:].max()),
                "stop_probability": float(probabilities[0]), "selected_action": observation.action_ids[top],
                "correct": top == index, "logits": logits.tolist(), "probabilities": probabilities.tolist(),
                "action_ids": list(observation.action_ids), "critic_value": float(value)})
    if parameter_hash(model) != before:
        raise ValueError("No-gradient readout changed policy parameters")
    return {"parameter_hash": before, "forward_calls": len(rows), "seconds": time.perf_counter()-began,
        "mean_cross_entropy": sum(r["teacher_cross_entropy"] for r in rows)/len(rows),
        "correct_states": sum(r["correct"] for r in rows), "states": rows}


def verify_update16(model, rows):
    from resectionlab.spatial_policy import parameter_hash
    if len(rows) != 16 or parameter_hash(model) != BC16_HASH:
        raise ValueError("Update16 did not reproduce the original fixed BC checkpoint")


def worker(record, output, *, declaration_sha256):
    import torch
    from resectionlab.data_policy import GeneratedDevelopmentContext
    from resectionlab.native_spatial_task import make_native_opening_task
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash, imitation_loss, gradient_step
    started = time.perf_counter()
    results = {"version": VERSION, "status": "running", "updates": 0, "loss_forward_calls": 0,
        "diagnostic_forward_calls": 0, "readouts": {}, "real_patient_count": 0,
        "methods": {name: {"status": "not_started", "outcomes": None} for name in ("initial", "BC256")}}
    def guard():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if time.perf_counter()-started >= SETTINGS["max_wall_seconds"] or peak > SETTINGS["max_rss_bytes"]:
            raise InterruptedError("Fixed worker time/memory envelope exhausted")
    def save():
        results["elapsed_seconds"] = time.perf_counter()-started
        io.write_json(output / "result.json", results)
    save()
    try:
        validate(record); torch.set_num_threads(1)
        bundle = load_frozen_teacher(record)
        began = time.perf_counter()
        with NativePreviewProfiler(NativeResectionEngine) as profiler:
            base = make_native_opening_task(cancelled=lambda: bool(guard()))
        results["preparation"] = {"seconds": time.perf_counter()-began, "preview_profile": profiler.snapshot()}
        context = GeneratedDevelopmentContext(declaration_sha256, (base.case.source_hash,), base.decision_model_hash)
        samples, reconstruction = reconstruct_teacher(base, bundle, context, guard)
        results["reconstruction"] = reconstruction
        old_search_cost = json.loads((SAVED / "depth2-search-cost.json").read_text())
        results["inherited_costs"] = {"depth2_search_online_seconds": bundle["result"]["methods"]["depth2-search"]["online_seconds"],
            "original_shared_preparation": bundle["result"]["preparation"],
            "depth2_search_audit_seconds": old_search_cost["independent_audit_seconds"],
            "teacher_search_model_transitions": bundle["teacher"]["model_transition_calls"],
            "teacher_validation": bundle["result"]["offline_teacher_validation"],
            "teacher_search_preview_profile": old_search_cost["online"],
            "prior_BC16_training_seconds": bundle["result"]["training"]["BC"]["fit_seconds"],
            "scope": "previously paid teacher/search and audit costs; not repeated or counted as new runtime"}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SETTINGS["seed"])
            model = SpatialPolicy(SpatialPolicyConfig(**record["policy"]))
        if parameter_hash(model) != INITIAL_HASH:
            raise ValueError("Fresh initialization differs from the original initial weights")
        results.update(initial_parameter_hash=parameter_hash(model), architecture_hash=model.architecture_hash,
                       context=asdict(context), parameter_count=sum(p.numel() for p in model.parameters()))
        def checkpoint(update, optimizer=None):
            torch.save({"policy": model.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(),
                "updates": update, "parameter_hash": parameter_hash(model), "initial_parameter_hash": INITIAL_HASH,
                "architecture": model.architecture_record(), "context": asdict(context)}, output / f"BC-{update:03}.pt")
        def inspect(update):
            row = readout(model, samples, guard)
            results["readouts"][str(update)] = row
            results["diagnostic_forward_calls"] += row["forward_calls"]
            save()
        def evaluate(name):
            before = parameter_hash(model)
            results["methods"][name]["status"] = "running"; save()
            _, report, _ = prior.guarded_episode(base, model, output, name, guard,
                seconds=SETTINGS["online_seconds"], previews=SETTINGS["online_native_previews"])
            if report["independent_evaluation"]["accepted"] is not True or parameter_hash(model) != before:
                raise ValueError("Independent execution rejected or inference weights changed")
            results["methods"][name] = {"status": "complete", "accepted": True, "parameter_hash": before,
                "actions": [d["action_id"] for d in report["decisions"]],
                "outcomes": report["independent_evaluation"]["outcomes"],
                "online_seconds": report["guarded_online_seconds"], "replay_seconds": report["online_seconds"],
                "audit_seconds": report["independent_evaluation_seconds"]}; save()
        checkpoint(0); inspect(0); evaluate("initial")
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
        rows, fit_started = [], time.perf_counter()
        for update in range(1, SETTINGS["updates"] + 1):
            guard(); context.require_task(base)
            began = time.perf_counter()
            loss, stats = imitation_loss(model, samples, learning_context=context)
            gradient = gradient_step(model, optimizer, loss, max_norm=SETTINGS["gradient_clip"], learning_context=context)
            rows.append({"update": update, "loss": stats, "gradient": gradient, "seconds": time.perf_counter()-began})
            results["updates"] = update; results["loss_forward_calls"] += stats["loss_forward_calls"]
            io.write_json(output / "updates.json", rows)
            if update == 16:
                verify_update16(model, rows)
                # Hash-chain identity verifies the complete original prefix, not just its last state.
                if [r["gradient"] for r in rows] != [r["gradient"] for r in bundle["BC-updates"]]:
                    raise ValueError("Original16 gradient/hash trajectory differs")
                results["update16_exact_reproduction"] = True
            if update in READOUTS:
                checkpoint(update, optimizer); inspect(update)
            save()
        results["fit_with_readout_export_seconds"] = time.perf_counter()-fit_started
        results["loss_update_seconds"] = sum(r["seconds"] for r in rows)
        evaluate("BC256")
        validate(record)
        if results["updates"] != 256 or results["loss_forward_calls"] != 1280 or results["diagnostic_forward_calls"] != 30:
            raise ValueError("Fixed workload counters differ")
        results["status"] = "complete"; save()
    except BaseException as error:
        results.update(status="failed", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        for row in results["methods"].values():
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
        if args.declaration.exists(): raise ValueError("Preserve prior declaration bytes")
        io.write_json(args.declaration, declaration()); return
    record, declared_hash, payload = io.read_declaration(args.declaration, args.expected_sha256)
    validate(record)
    if args.worker:
        if not args.expected_sha256: raise ValueError("Worker requires exact declaration SHA")
        worker(record, args.output, declaration_sha256=declared_hash); return
    if not args.execute:
        print("Fixed longer-fit declaration validated; no task, policy or optimizer created."); return
    if args.output is None or args.output.exists(): raise ValueError("Choose a new output directory")
    args.output.mkdir(parents=True)
    frozen = args.output / "declaration-input.json"; frozen.write_bytes(payload)
    for name in record["source_sha256"]:
        destination = args.output / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, destination)
    io.write_json(args.output / "inherited-input-sha256.json", record["input_sha256"])
    supervised = io.supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker", "--declaration",
        str(frozen.resolve()), "--expected-sha256", declared_hash, "--output", str(args.output.resolve())],
        args.output, SETTINGS, declared_hash)
    io.write_json(args.output / "output-sha256.json", {str(p.relative_to(args.output)): io.sha256(p)
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "output-sha256.json"})
    if supervised["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__":
    main()
