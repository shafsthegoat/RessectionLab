#!/usr/bin/env python3
"""Bounded PAT05 SEARCH/untrained-policy comparison; no optimizer execution.

Validation is the default. Execution requires a frozen declaration hash and an
explicit --execute flag. This first real-only runner reuses native execution,
preflight auditing and process supervision. It opens no other patient, trains
nothing, and does not claim validated tissue mechanics or clinical accuracy.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_real_spatial_policy as preflight
from resectionlab.real_patient_learning import COHORT_SHA256, read_development_cohort, require_development_role

VERSION = "real-spatial-search-comparison-v1"
COHORT_PATH = "manifests/experiments/btc-spatial-development-cohort-v1.json"
SUBJECT = "sub-PAT05"
BUNDLE_PATH = "outputs/cases/BTC-sub-PAT05-structural-evidence.ressectionlab"


def numerical_source_inventory():
    paths = [Path(__file__).resolve(), ROOT / "scripts/preflight_real_spatial_policy.py",
             *(ROOT / "src/resectionlab").rglob("*.py")]
    return {str(path.relative_to(ROOT)): preflight.sha256(path) for path in sorted(paths)}


def _digest(value, *, prefix=False):
    if not isinstance(value, str) or (prefix and not value.startswith("sha256:")):
        return False
    text = value[7:] if prefix else value
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_declaration(declaration):
    """Check metadata and source closure before any patient file is read."""
    if (declaration.get("version") != VERSION or declaration.get("status") != "prospective_frozen"
            or declaration.get("track") != "annotation_assisted"
            or declaration.get("methods") != ["SEARCH", "untrained_policy"]):
        raise ValueError("A frozen annotation-assisted SEARCH/untrained-policy declaration is required")
    if declaration.get("source_sha256") != numerical_source_inventory():
        raise ValueError("Complete numerical source hashes are absent or changed")
    if declaration.get("cohort_path") != COHORT_PATH or declaration.get("cohort_sha256") != COHORT_SHA256:
        raise ValueError("Original cohort identity must remain unchanged")
    cohort = read_development_cohort(ROOT / COHORT_PATH)
    member = declaration.get("member", {})
    if member.get("subject") != SUBJECT or member.get("role") != "TRAIN":
        raise ValueError("This bounded first comparison may open only TRAIN sub-PAT05")
    require_development_role(cohort, member["subject"], role=member["role"])
    if (member.get("case_bundle") != BUNDLE_PATH
            or not _digest(member.get("case_bundle_sha256"))
            or not _digest(member.get("case_semantic_hash"), prefix=True)
            or not isinstance(member.get("access"), dict)
            or not isinstance(member.get("research_support_acknowledgment"), dict)
            or not isinstance(member.get("expected_native_grid_binding"), dict)
            or not member["expected_native_grid_binding"]):
        raise ValueError("Explicit patient bundle, support, access and grid bindings are required")
    if (not isinstance(declaration.get("tools"), list) or not declaration["tools"]
            or not isinstance(declaration.get("objective"), dict)
            or not isinstance(declaration.get("policy_config"), dict)
            or not _digest(declaration.get("expected_policy_architecture_hash"), prefix=True)
            or not isinstance(declaration.get("adapter_options"), dict)
            or "research_support_acknowledgment" in declaration["adapter_options"]):
        raise ValueError("Common tools, objective, architecture and adapter settings must be explicit")
    for field, directory in (("coverage_evidence", "artifacts"), ("source_profile_declaration", "manifests")):
        reference = declaration.get(field, {})
        if not isinstance(reference.get("path"), str) or not _digest(reference.get("sha256")):
            raise ValueError("Frozen metadata path/hash required: " + field)
        evidence_path = (ROOT / reference["path"]).resolve()
        if evidence_path.suffix != ".json" or not evidence_path.is_relative_to((ROOT / directory).resolve()):
            raise ValueError(field + " must be a local JSON artifact, not a patient image")
        if preflight.sha256(evidence_path) != reference["sha256"]:
            raise ValueError("Declared metadata changed: " + field)
    settings = declaration.get("settings", {})
    if type(settings.get("optimizer_updates")) is not int or settings["optimizer_updates"] != 0:
        raise ValueError("Optimizer execution is disabled; declare optimizer_updates=0")
    if settings.get("device") != "cpu" or settings.get("torch_threads") != 1 or settings.get("blas_thread_caps") != 1:
        raise ValueError("This comparison requires explicitly declared single-thread CPU execution")
    for name in ("seed", "max_steps", "max_rss_bytes"):
        value = settings.get(name)
        minimum = 0 if name == "seed" else 1
        if type(value) is not int or not minimum <= value < 2**63 - 100000:
            raise ValueError("Missing or invalid declared integer setting: " + name)
    if settings["max_steps"] > 6:
        raise ValueError("Native adapter horizon is limited to six actions")
    for name in ("max_wall_seconds", "online_episode_seconds"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("No implicit runtime budget is allowed: " + name)
    search = declaration.get("search", {})
    if (any(type(search.get(key)) is not int or search[key] < 1 for key in ("max_calls", "beam_width"))
            or not isinstance(search.get("seconds"), (float, int)) or isinstance(search["seconds"], bool)
            or not math.isfinite(search["seconds"]) or search["seconds"] <= 0
            or search.get("transition_mode") not in {"eager", "lazy_planning"}):
        raise ValueError("Search limits and transition mode require a prospective measurement-based choice")
    return cohort


def load_member(declaration, cohort):
    """Role and byte identity precede decoding this one bound real source."""
    member = declaration["member"]
    if (member.get("subject") != SUBJECT or member.get("role") != "TRAIN"
            or member.get("case_bundle") != BUNDLE_PATH):
        raise ValueError("Only the bound TRAIN sub-PAT05 input is allowed")
    require_development_role(cohort, SUBJECT, role="TRAIN")
    bundle = ROOT / member["case_bundle"]
    if preflight.sha256(bundle) != member["case_bundle_sha256"]:
        raise RuntimeError("Declared patient bundle bytes changed")
    from resectionlab.imaging import load_case
    from resectionlab.geometry import AccessWindow, ToolGeometry
    from resectionlab.native_spatial_task import native_spatial_task_from_case
    case = load_case(bundle)
    source = case.metadata.get("source_collection", {})
    if (case.case_id != "BTC-ds001226-" + SUBJECT + "-preop"
            or case.semantic_hash != member["case_semantic_hash"] or case.metadata.get("is_synthetic")
            or source.get("accession") != "ds001226" or source.get("release") != "5.0.1"
            or source.get("git_commit") != cohort["source"]["git_commit"]
            or not any(ref.source_id == "structural" and ref.provenance == "observed" for ref in case.source_refs)):
        raise RuntimeError("Bundle does not establish its declared real preoperative patient identity")
    task = native_spatial_task_from_case(case, access=AccessWindow(**member["access"]),
        tools=tuple(ToolGeometry(**row) for row in declaration["tools"]),
        max_steps=declaration["settings"]["max_steps"], track=declaration["track"],
        research_support_acknowledgment=member["research_support_acknowledgment"],
        **declaration["adapter_options"])
    if asdict(task.reward_spec) != declaration["objective"]:
        raise RuntimeError("Executed physical objective differs from its declaration")
    grid = task.metrics()["native_grid_reconciliation"]
    if any(grid.get(key) != value for key, value in member["expected_native_grid_binding"].items()):
        raise RuntimeError("Executed native/source frame binding differs from its declaration")
    return task


def replay_search(base, settings, *, preserve, profiler):
    from resectionlab.observed_search import observed_beam_search
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    started = time.perf_counter()
    with profiler.phase("SEARCH:planning"):
        sequence, search = observed_beam_search(base, **settings,
            objective_source="permitted_annotation_assisted_nominal_model")
    preserve({"status": "planned", "sequence": list(sequence), "search": search})
    committed = []
    def record(info):
        committed.append(copy.deepcopy(info))
        preserve({"latest_committed_transition": info, "committed_transition_count": len(committed),
                  "committed_transition_records": copy.deepcopy(committed)})
    with profiler.phase("SEARCH:execution"):
        task = base.fresh()
        for action in sequence:
            try:
                result = task.step(action)
            except CommittedTransitionInterrupted as error:
                record(error.info)
                raise
            record(result.info)
    if not task.terminated:
        raise RuntimeError("Search replay did not complete its declared episode")
    metrics = task.metrics()
    online_seconds = time.perf_counter() - started
    preserve({"status": "awaiting_independent_audit", "metrics": metrics, "online_seconds": online_seconds})
    audit_started = time.perf_counter()
    with profiler.phase("SEARCH:independent_audit"):
        checked = asdict(task.independent_geometry_check())
    preserve({"independent_geometry_check": checked})
    if not checked["feasible"]:
        raise RuntimeError("Independent native checker rejected search replay")
    return {"status": "complete", "metrics": metrics, "sequence": list(sequence), "search": search,
            "online_seconds": online_seconds, "independent_audit_seconds": time.perf_counter() - audit_started,
            "independent_geometry_check": checked}


def worker(declaration, output):
    """One bounded comparison. There is no optimizer or training branch."""
    cohort = validate_declaration(declaration)
    import torch
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler, spatial_coverage, runtime_proposal_coverage
    settings, started = declaration["settings"], time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings["seed"])
    policy = SpatialPolicy(SpatialPolicyConfig(**declaration["policy_config"]))
    if policy.architecture_hash != declaration["expected_policy_architecture_hash"]:
        raise RuntimeError("Policy architecture differs from its declaration")
    before = parameter_hash(policy)
    generator = torch.Generator().manual_seed(settings["seed"] + 100000)
    receipt = {"version": VERSION, "subject": SUBJECT, "source_sha256": declaration["source_sha256"],
        "architecture": policy.architecture_record(), "initial_parameter_hash": before,
        "optimizer_updates": 0, "population_generalization_measured": False,
        "external_final_patients_opened": 0, "other_patients_opened": 0,
        "methods": {name: {"status": "not_executed"} for name in declaration["methods"]},
        "representation_limit": "same permitted source and certified actions; SEARCH reads full nominal fields while actor receives its declared crop",
        "scope": "real anatomy; simulated rigid-cell actions; mechanics and clinical accuracy unvalidated"}
    with NativePreviewProfiler(NativeResectionEngine) as profiler:
        def preserve(*, enforce=True):
            receipt.update(elapsed_seconds=time.perf_counter() - started,
                           peak_rss_bytes=preflight.peak_rss_bytes(), native_preview_cost=profiler.snapshot())
            preflight.write_json(output / "receipt.json", receipt)
            if enforce and (receipt["elapsed_seconds"] > settings["max_wall_seconds"] or receipt["peak_rss_bytes"] > settings["max_rss_bytes"]):
                raise RuntimeError("Declared whole-run resource budget exceeded")

        preserve()
        with profiler.phase("shared_input_preparation"):
            preparation_started = time.perf_counter()
            base = load_member(declaration, cohort)
            receipt["shared_input_preparation_seconds"] = time.perf_counter() - preparation_started
        receipt["initial_task_metrics"] = base.metrics()
        native_affine = receipt["initial_task_metrics"]["native_grid_reconciliation"].get("derived_affine_ras_mm")
        observation, inventory = base.observation(), base.candidate_inventory()
        receipt["initial_candidate_inventory"] = inventory
        receipt["initial_observation_coverage"] = spatial_coverage(observation,
            source_shape=base.case.structural_intensity.shape, source_affine=base.case.affine_ras_mm,
            nominal_target=base.case.nominal_target, ray_samples=policy.config.ray_samples, native_affine=native_affine)
        receipt["initial_proposal_coverage"] = runtime_proposal_coverage(base.case, inventory, observation,
            native_affine=native_affine, ray_samples=policy.config.ray_samples)
        preserve()
        for method in declaration["methods"]:
            row = receipt["methods"][method]
            row["status"] = "starting"
            def save(value):
                row.update(value)
                preserve()
            try:
                preserve()  # Make an in-flight method durable before expensive work.
                if method == "SEARCH":
                    row.update(replay_search(base, declaration["search"], preserve=save, profiler=profiler))
                else:
                    with profiler.phase("untrained_policy"):
                        _, result = preflight.episode(base, policy, generator, stochastic=False,
                            checkpoint=save, diagnostics=False, profiler=profiler)
                    row.update(result, status="complete")
                if row["online_seconds"] > settings["online_episode_seconds"]:
                    raise RuntimeError("Method exceeded its declared online episode budget")
            except BaseException as error:
                row.update(status="failed", error_type=type(error).__name__, reason=str(error),
                           partial_search_accounting=getattr(error, "accounting", None),
                           partial_best_sequence=getattr(error, "best_sequence", None))
                receipt["status"] = "failed"
                preserve(enforce=False)
                raise
            preserve()
        if parameter_hash(policy) != before:
            raise RuntimeError("Zero-update comparison changed model weights or buffers")
        if numerical_source_inventory() != declaration["source_sha256"]:
            raise RuntimeError("Numerical source changed during the comparison")
        receipt.update(status="complete", final_parameter_hash=parameter_hash(policy))
        preserve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-declaration-sha256")
    parser.add_argument("--execute", action="store_true", help="Explicit release of a frozen declaration")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    declaration, digest, payload = preflight.read_declaration(args.declaration, args.expected_declaration_sha256)
    validate_declaration(declaration)
    if not args.execute and not args.worker:
        print("Declaration/source validation passed; no patient opened and no optimizer executed.")
        return
    if not args.expected_declaration_sha256 or args.output is None:
        raise SystemExit("Execution requires the frozen declaration hash and a new output directory")
    if args.worker:
        try:
            worker(declaration, args.output)
        except BaseException as error:
            preflight.write_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error),
                "accounting": getattr(error, "accounting", None), "traceback": traceback.format_exc()})
            raise
        return
    if args.output.exists():
        raise SystemExit("Preserve earlier attempts: choose a new output directory")
    args.output.mkdir(parents=True)
    snapshot = args.output / "declaration-input.json"
    snapshot.write_bytes(payload)
    result = preflight.supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker",
        "--declaration", str(snapshot.resolve()), "--expected-declaration-sha256", digest,
        "--output", str(args.output.resolve())], args.output, declaration["settings"], digest)
    preflight.write_json(args.output / "output-sha256.json", {path.name: preflight.sha256(path)
        for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "output-sha256.json"})
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
