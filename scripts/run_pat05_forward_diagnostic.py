#!/usr/bin/env python3
"""Exactly two frozen checkpoints on the unchanged historical PAT05 initial DTO."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from resectionlab import pat05_forward_diagnostic as bridge
from preflight_real_spatial_policy import supervise_worker, write_json, sha256

SCRIPT = "scripts/run_pat05_forward_diagnostic.py"
SETTINGS = {"cpu_threads": 1, "max_wall_seconds": 60., "max_rss_bytes": 2 * 1024**3,
    "forward_order": ["initial", "RL256"], "forwards_per_checkpoint": 1,
    "native_previews": 0, "executed_actions": 0, "gradient_updates": 0, "automatic_retry": False}


def dependency_paths(root=ROOT):
    """Conservative AST local-import closure, including function-local imports."""
    root = Path(root)
    seeds = [SCRIPT, "scripts/preflight_real_spatial_policy.py", "src/resectionlab/pat05_forward_diagnostic.py"]
    result, pending = set(), list(seeds)
    def resolve(module):
        if module == "resectionlab" or module.startswith("resectionlab."):
            path = root / "src" / Path(*module.split("."))
        else:
            path = root / "scripts" / module.replace(".", "/")
        candidates = (path.with_suffix(".py"), path / "__init__.py")
        return next((str(p.relative_to(root)) for p in candidates if p.is_file()), None)
    while pending:
        relative = pending.pop()
        if relative in result:
            continue
        result.add(relative)
        module = relative[4:-3].replace("/", ".") if relative.startswith("src/") else relative[8:-3]
        package = module if module.endswith(".__init__") else module.rsplit(".",1)[0]
        package = package.removesuffix(".__init__")
        tree = ast.parse((root / relative).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = package.split(".")[:len(package.split("."))-node.level+1]
                    base = ".".join(parent + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                imports += [base, *(base + "." + a.name for a in node.names)]
        for name in imports:
            resolved = resolve(name)
            if resolved and resolved not in result:
                pending.append(resolved)
    result.add("src/resectionlab/__init__.py")
    return tuple(sorted(result))


def sources():
    return {path: sha256(ROOT / path) for path in dependency_paths()}


def specification():
    return {"version": bridge.VERSION, "subject": "sub-PAT05", "role": "TRAIN", "settings": SETTINGS,
        "metadata_sha256": bridge.METADATA_SHA256, "checkpoints": bridge.CHECKPOINTS,
        "architecture_hash": bridge.ARCHITECTURE, "expected_observation_hash": bridge.OBSERVATION,
        "expected_native_case_source_hash": bridge.SOURCE,
        "scope": "paired frozen-checkpoint outputs on one historical annotation-assisted initial DTO",
        "admission": "exact historical hypothetical estimated support for observation reconstruction only; default anatomy/model/learning guards unchanged",
        "historical_inventory": "70 simulator-feasibility-filtered proposals plus STOP; simulator-derived information, not scan-only deployment input",
        "prohibited": ["native_engine", "preview", "task_step", "search", "loss", "backward", "optimizer", "adaptation", "new_patient_or_holdout"],
        "cavity": "hypothetical empty initial state; retained observed_procedure_state label only for exact DTO continuity",
        "clinical_availability": "unknown; historical acknowledgment time is not clinical evidence availability",
        "clinical_performance_measured": False, "deployment_compatibility_established": False,
        "resources_scope": "60-second supervised worker including preparation, both loads/forwards and export; sampled2GiB worker RSS",
        "cost_limit": "prior MRI acquisition, support estimation and historical proposal certification were previously paid and are not new inference time"}


def declaration():
    return {**specification(), "source_sha256": sources(),
        "runtime": {"python": sys.version, "platform": sys.platform,
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "nibabel")}}}


def assert_imports(record):
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        local_script = name in {"preflight_real_spatial_policy", "run_pat05_forward_diagnostic"}
        if name == "resectionlab" or name.startswith("resectionlab.") or local_script:
            if not path.is_relative_to(ROOT.resolve()) or str(path.relative_to(ROOT.resolve())) not in record["source_sha256"]:
                raise ValueError("Executing local module outside declared source closure: " + name)


def validate(record):
    if bridge.canonical(record) != bridge.canonical(declaration()):
        raise ValueError("Only the exact prospective source/input/runtime/settings declaration is permitted")
    assert_imports(record)
    return bridge.read_authorities(ROOT)


def worker(record, output, declaration_sha256):
    import torch
    started = time.perf_counter()
    context = bridge.Pat05ForwardContext(declaration_sha256)
    result = {"version": bridge.VERSION, "status": "running", "subject": "sub-PAT05", "role": "TRAIN",
        "methods": {name: {"status": "not_started", "outputs": None} for name in SETTINGS["forward_order"]},
        "clinical_performance_measured": False, "deployment_compatibility_established": False}
    def guard():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if time.perf_counter()-started >= SETTINGS["max_wall_seconds"] or peak > SETTINGS["max_rss_bytes"]:
            raise InterruptedError("Fixed diagnostic worker resource envelope exhausted")
        return peak
    def save():
        result.update(elapsed_seconds=time.perf_counter()-started, accounting=context.snapshot())
        write_json(output / "result.json", result)
    save()
    try:
        torch.set_num_threads(1)
        authorities = validate(record)
        guard()
        began = time.perf_counter()
        observation, reconstruction = bridge.reconstruct_pat05(ROOT, authorities)
        context.require_observation(observation)
        result["reconstruction_seconds"] = time.perf_counter()-began
        write_json(output / "reconstruction.json", reconstruction)
        write_json(output / "input-description.json", bridge.describe_observation(observation))
        result["historical_preparation"] = {"seconds": authorities[bridge.HISTORICAL+"receipt.json"]["preparation_seconds"],
            "scope": "previously paid historical native certification; excluded from this observation-only clock",
            "proposals_emitted": 78, "proposals_accepted": 70, "new_previews": 0}
        for name in SETTINGS["forward_order"]:
            guard(); validate(record); context.require_observation(observation)
            result["methods"][name]["status"] = "running"; save()
            began = time.perf_counter()
            row = context.forward_checkpoint(ROOT, name, observation)
            guard(); validate(record)
            row["load_and_forward_seconds"] = time.perf_counter()-began
            write_json(output / (name + ".json"), row)
            result["methods"][name] = {"status": "complete", "outputs": name + ".json",
                "parameter_hash": row["parameter_hash"], "forward_seconds": row["forward_seconds"],
                "load_and_forward_seconds": row["load_and_forward_seconds"],
                "highest_ranked_id_not_executed": row["highest_ranked_id_not_executed"]}
            save()
        if context.snapshot()["completed_forwards"] != 2:
            raise ValueError("Both fixed checkpoint forwards must complete")
        result["peak_worker_rss_bytes"] = guard()
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        for method in result["methods"].values():
            if method["status"] == "running":
                method.update(status="failed", outputs=None)
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        save()


def write_output_index(output):
    """Include inherited index files; exclude only this root index itself."""
    index = output / "output-sha256.json"
    write_json(index, {str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path != index})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-declaration-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw = args.declaration.read_bytes(); identity = bridge.digest_bytes(raw); record = json.loads(raw)
    if args.expected_declaration_sha256 and identity != args.expected_declaration_sha256:
        raise ValueError("Supervisor-bound declaration bytes changed")
    validate(record)  # metadata only; no case decode, checkpoint load or forward
    if args.worker:
        if not args.expected_declaration_sha256 or args.output is None:
            raise ValueError("Worker requires declaration identity and output")
        worker(record, args.output, identity)
        return
    if not args.execute:
        print(json.dumps({"status": "metadata_preflight_passed", "declaration_sha256": identity,
            "patient_arrays_decoded": 0, "model_forwards": 0, "execution_requires": "--execute after source freeze"}))
        return
    if args.output is None or args.output.exists():
        raise ValueError("Execution requires a fresh output directory; no retries/overwrite")
    args.output.mkdir(parents=True)
    (args.output / "declaration-input.json").write_bytes(raw)
    for relative, expected in record["source_sha256"].items():
        target = args.output / "source-snapshot" / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bridge.checked_bytes(ROOT, relative, expected))
    for relative, expected in bridge.METADATA_SHA256.items():
        target = args.output / "input-metadata" / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bridge.checked_bytes(ROOT, relative, expected))
    command = [sys.executable, str(ROOT / SCRIPT), "--declaration", str(args.output.resolve()/"declaration-input.json"),
        "--output", str(args.output.resolve()), "--worker", "--expected-declaration-sha256", identity]
    supervised = supervise_worker(command, args.output, SETTINGS, identity)
    validate(record)
    write_output_index(args.output)
    if supervised["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
