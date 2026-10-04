#!/usr/bin/env python3
"""Evaluate fixed, existing UCSF routes/histories under declared prior scenarios.

Run from a committed source snapshot. This runner does not train, generate new
paths, approve anatomy, or overwrite source cases. Its declaration precedes the
first event evaluation and all attempted panels, including failures, are kept.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
import platform
from pathlib import Path
import resource
import signal
import subprocess
import sys
from time import perf_counter, sleep
from types import SimpleNamespace


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def select_static_routes(rows, declaration):
    selected = [min((row for row in rows if row["window_id"] == name), key=lambda row: row["route_id"])
                for name in sorted({row["window_id"] for row in rows})]
    if [row["route_id"] for row in selected] != declaration["selected_static_route_ids"]:
        raise ValueError("Historical route selection differs from prospective declaration")
    return selected


def panel_specs(declaration):
    result = []
    for uncertainty in declaration["uncertainty_panels"]:
        for threshold in declaration["thresholds"]:
            result.append({"name": f"{uncertainty['name']}_support_{threshold:g}",
                "uncertainty": uncertainty, "threshold": threshold, "components": "motor_and_language"})
    for components in declaration["evidence"]["missing_component_variants"]:
        result.append({"name": f"{declaration['uncertainty_panels'][0]['name']}_{components}",
            "uncertainty": declaration["uncertainty_panels"][0],
            "threshold": declaration["thresholds"][0], "components": components})
    return result


def verify_inputs(data_root, declaration):
    result = {}
    for key, item in declaration["inputs"].items():
        path = (Path(data_root) / item["path"]).resolve()
        if not path.is_relative_to(Path(data_root).resolve()):
            raise ValueError("Declared input escapes data root")
        actual = file_sha(path)
        if actual != item["sha256"]:
            raise ValueError(f"Source bytes changed: {key}")
        result[key] = {**item, "actual_sha256": actual, "bytes": path.stat().st_size}
    return result


def source_inventory():
    paths = [Path(__file__).resolve(), *sorted((SOURCE_ROOT / "src/resectionlab").rglob("*.py"))]
    return {str(path.relative_to(SOURCE_ROOT)): file_sha(path) for path in paths}


def run(data_root, output, bundle_output, declaration_path):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    import numpy as np
    import scipy
    from resectionlab.core import array_digest
    from resectionlab.evaluation import (EvaluationLedger, IndependentGeometryResult,
        freeze_candidates, independent_check_native_history, independent_check_route)
    from resectionlab.functional_evidence import anatomy_identity, population_prior_sensitivity
    from resectionlab.functional_events import (FunctionalEventConfig, evaluate_functional_candidates,
        prepare_functional_exposure, sweeps_from_native_history, sweeps_from_route)
    from resectionlab.geometry import AccessWindow, GeometryScene, ToolGeometry
    from resectionlab.imaging import load_case, save_case
    from resectionlab.native_resection import native_config_from_case
    from resectionlab.worlds import FrozenDecisionModel, WorldGeneratorConfig, content_hash, generate_partitions

    started = perf_counter()
    output, bundle_output = Path(output), Path(bundle_output)
    output.mkdir(parents=True, exist_ok=False)
    if bundle_output.exists():
        raise FileExistsError("New evidence bundle path already exists")
    declaration_bytes = Path(declaration_path).read_bytes()
    declaration_sha = sha256(declaration_bytes).hexdigest()
    declaration = json.loads(declaration_bytes)
    write_json(output / "declaration.json", declaration)
    source_before = source_inventory()
    write_json(output / "source-before.json", source_before)
    write_json(output / "runtime.json", {"python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(), "machine": platform.machine(),
        "thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}})
    clocks, reports = {}, []

    def rss_gib():
        scale = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

    def guard():
        if perf_counter() - started > declaration["budgets"]["wall_seconds"]:
            raise TimeoutError("Declared full worker wall budget exceeded")
        if rss_gib() > declaration["budgets"]["peak_rss_gib"]:
            raise MemoryError("Declared peak worker RSS budget exceeded")
        return False

    def phase(name):
        guard()
        print(json.dumps({"phase": name, "elapsed_seconds": perf_counter() - started,
                          "peak_rss_gib": rss_gib()}), flush=True)
        return perf_counter()

    try:
        clock = phase("verify_pinned_inputs")
        receipts = verify_inputs(data_root, declaration)
        write_json(output / "inputs-before.json", receipts)
        paths = {name: Path(data_root) / item["path"] for name, item in declaration["inputs"].items()}
        original, prior_case = load_case(paths["original_case"]), load_case(paths["prior_case"])
        if original.semantic_hash != declaration["original_case_hash"] or prior_case.semantic_hash != declaration["prior_case_hash"]:
            raise ValueError("Case semantics differ from the pinned historical inputs")
        if (array_digest(original.mri) != array_digest(prior_case.mri)
                or anatomy_identity(original) != anatomy_identity(prior_case)):
            raise ValueError("Prior bundle changes original physical anatomy")
        if original.frame != "RAS+":
            raise ValueError("This frozen retained-history experiment declares RAS+ inputs")
        clocks["input_verification_and_loading_seconds"] = perf_counter() - clock
        original_routes = json.loads(paths["original_routes"].read_text())
        selected = select_static_routes(original_routes["routes"], declaration)
        native_records = [json.loads(paths[name].read_text()) for name in ("fine_history", "wide_history")]
        geometries = []
        clock = phase("independent_geometry_and_exposure")
        first = native_records[0]["route"]
        first_access = AccessWindow(**first["window"])
        first_config = native_config_from_case(original, access=first_access, tools=(ToolGeometry(**first["tool"]),))
        tissue, affine = first_config.tissue_mask, first_config.affine
        # No vascular/hard-exclusion input exists in this source case. The empty
        # constraint array is explicitly unknown anatomy, never verified absence.
        scene = GeometryScene(np.zeros(original.mri.shape, bool), affine, enforce_tip_in_bounds=False)
        geometry_clocks, exposures = {}, {}
        for record in native_records:
            guard()
            source_payload = {key: value for key, value in record.items() if key != "candidate_hash"}
            if content_hash(source_payload) != record["candidate_hash"] or record["source_hash"] != original.semantic_hash:
                raise ValueError("Historical native history identity mismatch")
            route_data = record["route"]
            tool, access = ToolGeometry(**route_data["tool"]), AccessWindow(**route_data["window"])
            config = native_config_from_case(original, access=access, tools=(tool,))
            tick = perf_counter()
            audit = independent_check_native_history(original, (tool,), record["history"],
                tissue_mask=tissue, access=access, hard_exclusion=config.hard_exclusion, cancelled=guard)
            audit_seconds = perf_counter() - tick
            identity = "native_" + tool.tool_id
            sweeps = sweeps_from_native_history(record["history"], (tool,))
            tick = perf_counter()
            exposure = prepare_functional_exposure(sweeps, tissue_mask=tissue, affine_ras_mm=affine,
                candidate_hash=record["candidate_hash"], case_hash=original.semantic_hash, cancelled=guard)
            geometry_clocks[identity] = {"independent_audit_seconds": audit_seconds,
                                        "footprint_seconds": perf_counter() - tick}
            certificate = IndependentGeometryResult(audit.feasible, audit.failures,
                checker_version=audit.checker_version, unknowns=tuple(original.unknowns))
            removal_arrays = [np.asarray(step.get("removed_indices_native", ()), int).reshape(-1, 3)
                              for step in record["history"] if step.get("action_id") != "STOP"]
            removed = np.unique(np.concatenate(removal_arrays, axis=0), axis=0) if removal_arrays else np.empty((0, 3), int)
            removed_by_compartment = {name: int(mask[tuple(removed.T)].sum()) * exposure.voxel_volume_mm3
                                      for name, mask in original.compartments.items()}
            geometries.append({"id": identity, "source_candidate_hash": record["candidate_hash"],
                "plan_type": "simulated_resection", "tool": asdict(tool), "entry_mm": route_data["entry_mm"],
                "target_mm": route_data["target_mm"], "certificate": certificate, "independent_native_audit": audit.to_dict(),
                "removed_by_compartment_mm3": removed_by_compartment if audit.feasible else None,
                "residual_by_compartment_mm3": {name: int(mask.sum()) * exposure.voxel_volume_mm3 - removed_by_compartment[name]
                    for name, mask in original.compartments.items()} if audit.feasible else None,
                "removed_normal_tissue_mm3": len(removed) * exposure.voxel_volume_mm3 - sum(removed_by_compartment.values()) if audit.feasible else None,
                "removed_tissue_mm3": len(removed) * exposure.voxel_volume_mm3 if audit.feasible else None})
            exposures[identity] = exposure
        for row in selected:
            guard()
            identity = "static_" + row["route_id"]
            tool = ToolGeometry(**row["tool"])
            access = AccessWindow(row["entry_mm"], declaration["static_window_normals_ras"][row["window_id"]],
                                  declaration["static_window_radius_mm"], row["window_id"])
            route = SimpleNamespace(entry_mm=row["entry_mm"], target_mm=row["target_mm"], tool=tool,
                                    window=access, unknowns=tuple(original.unknowns))
            tick = perf_counter()
            checked = independent_check_route(route, scene)
            audit_seconds = perf_counter() - tick
            tick = perf_counter()
            exposures[identity] = prepare_functional_exposure(sweeps_from_route(route), tissue_mask=tissue,
                affine_ras_mm=affine, candidate_hash=content_hash(row), case_hash=original.semantic_hash, cancelled=guard)
            geometry_clocks[identity] = {"independent_audit_seconds": audit_seconds,
                                        "footprint_seconds": perf_counter() - tick}
            executable = checked if row["exact_native_stroke_feasible"] else IndependentGeometryResult(False,
                ("historical_native_execution_rejection:" + row["exact_native_stroke_reason"],),
                checker_version="historical-execution-gate-plus-independent-static-check-v1", unknowns=checked.unknowns)
            geometries.append({"id": identity, "source_candidate_hash": content_hash(row), "plan_type": "route_only",
                "tool": asdict(tool), "entry_mm": row["entry_mm"], "target_mm": row["target_mm"],
                "certificate": executable, "static_geometry_certificate": asdict(checked),
                "historical_native_execution_rejection": row["exact_native_stroke_reason"],
                "historical_native_rejection_retested": False, "removed_tissue_mm3": None})
        exposures["STOP"] = prepare_functional_exposure((), tissue_mask=tissue, affine_ras_mm=affine,
            candidate_hash=content_hash({"action": "STOP"}), case_hash=original.semantic_hash)
        geometries.append({"id": "STOP", "source_candidate_hash": content_hash({"action": "STOP"}),
            "plan_type": "route_only", "tool": None, "certificate": IndependentGeometryResult(True),
            "removed_tissue_mm3": None})
        clocks["geometry_and_exposure_seconds"] = perf_counter() - clock
        geometry_report = [{**row, "certificate": asdict(row["certificate"])} for row in geometries]
        write_json(output / "geometry.json", {"alternatives": geometry_report, "clocks": geometry_clocks,
            "tissue_support_provenance": first_config.tissue_support_provenance,
            "hard_exclusions": "unavailable vascular anatomy; no supplied hard mask; no claim of vessel clearance",
            "retained_historical_rejections": [{"route_id": row["route_id"], "reason": row["exact_native_stroke_reason"]}
                for row in original_routes["routes"] if not row["exact_native_stroke_feasible"]]})
        clock = phase("freeze_all_panels_before_events")
        centre = affine[:3, :3] @ ((np.asarray(original.mri.shape) - 1) / 2) + affine[:3, 3]
        case_versions, panels = {}, []
        for spec in panel_specs(declaration):
            guard()
            key = (spec["uncertainty"]["name"], spec["components"])
            if key not in case_versions:
                item = spec["uncertainty"]
                generator = WorldGeneratorConfig(family=item["family"],
                    translation_scale_mm=tuple(item["translation_scale_mm"]),
                    rotation_scale_deg=tuple(item["rotation_scale_deg"]), rotation_center_mm=tuple(centre),
                    version=declaration["generator_version"], parameter_basis=declaration["parameter_basis"])
                evidence = population_prior_sensitivity(prior_case, uncertainty=generator)
                if spec["components"] == "motor_only":
                    evidence = replace(evidence, language=None, language_coverage=None)
                elif spec["components"] == "language_only":
                    evidence = replace(evidence, motor=None, motor_coverage=None)
                case_versions[key] = replace(prior_case, functional_evidence=evidence)
            case = case_versions[key]
            evidence = case.functional_evidence
            candidates = [SimpleNamespace(plan_id=row["id"], semantic_hash=content_hash({
                "source_candidate_hash": row["source_candidate_hash"], "case_hash": case.semantic_hash}),
                case_hash=case.semantic_hash, plan_type=row["plan_type"]) for row in geometries]
            bound_exposures = {candidate.plan_id: replace(exposures[candidate.plan_id],
                candidate_hash=candidate.semantic_hash, case_hash=candidate.case_hash) for candidate in candidates}
            event_config = FunctionalEventConfig(spec["threshold"], spec["threshold"], declaration["cvar_alpha"])
            model = FrozenDecisionModel.create(case_hash=case.semantic_hash, geometry={
                "functional_evidence_hash": evidence.fingerprint, "functional_event_config": event_config.to_dict(),
                "tissue_support_hash": exposures["STOP"].tissue_support_hash,
                "functional_footprint_hashes": {name: item.fingerprint for name, item in bound_exposures.items()},
                "original_anatomy_hash": anatomy_identity(original)}, objectives={"mode": "fixed_plan_sensitivity_no_optimization"},
                tools=[row["tool"] for row in geometries if row["tool"] is not None],
                world_generator=evidence.uncertainty, action_primitives="fixed_historical_routes_and_native_histories")
            partitions = generate_partitions(case.semantic_hash, evidence.uncertainty, declaration["master_seed"],
                optimization=declaration["optimization_worlds"], selection=declaration["selection_worlds"],
                final_evaluation=spec["uncertainty"]["worlds"], stress=2,
                planning_hash="prospectively_paired_UCSF0004_prior_sensitivity_v1")
            freeze = freeze_candidates(candidates, model, partitions.selection, declaration["selection_rule"],
                                       optimization_manifest=partitions.optimization)
            panels.append((spec, case, candidates, bound_exposures, event_config, model, partitions, freeze))
        write_json(output / "all-panels-frozen.json", [{"panel": spec, "case_hash": case.semantic_hash,
            "decision_model": model.to_dict(), "candidate_freeze": freeze.to_dict(), "partitions": partitions.to_dict()}
            for spec, case, _, _, _, model, partitions, freeze in panels])
        clocks["evidence_and_freeze_seconds"] = perf_counter() - clock
        clock = phase("save_primary_evidence_case")
        primary_case = panels[0][1]
        save_case(primary_case, bundle_output)
        reopened = load_case(bundle_output)
        if reopened.semantic_hash != primary_case.semantic_hash or reopened.functional_evidence.fingerprint != primary_case.functional_evidence.fingerprint:
            raise ValueError("New primary evidence case did not round trip")
        del reopened
        clocks["new_evidence_bundle_roundtrip_seconds"] = perf_counter() - clock
        certificates = {row["id"]: row["certificate"] for row in geometries}
        for spec, case, candidates, bound_exposures, event_config, model, partitions, freeze in panels:
            clock = phase("evaluate_" + spec["name"])
            report = evaluate_functional_candidates(candidates, freeze, model, partitions.final_evaluation,
                evidence=case.functional_evidence, footprints=bound_exposures,
                geometry_checker=lambda candidate: certificates[candidate.plan_id],
                ledger=EvaluationLedger(), config=event_config)
            report["elapsed_seconds"] = perf_counter() - clock
            report["panel_specification"] = spec
            report["scope"] = "prospectively paired development sensitivity; no parameter updates or outcome-driven selection"
            for candidate in report["candidates"]:
                original_geometry = next(row for row in geometries if row["id"] == candidate["plan_id"])
                if "historical_native_execution_rejection" in original_geometry:
                    candidate["status"] = "rejected_native_execution"
                    candidate["historical_native_execution_rejection"] = original_geometry["historical_native_execution_rejection"]
                    candidate["static_geometry_certificate"] = original_geometry["static_geometry_certificate"]
                    candidate["historical_native_rejection_retested"] = False
            write_json(output / (spec["name"] + ".json"), report)
            reports.append({"panel": spec["name"], "file": spec["name"] + ".json", "seconds": report["elapsed_seconds"],
                "distinct_transforms": report["distinct_anatomy_transforms"], "candidates": [{
                    "plan_id": row["plan_id"], "status": row["status"], "events": row["model_events"],
                    "costs": row["surrogate_costs"]} for row in report["candidates"]]})
            guard()
        if (source_inventory() != source_before or verify_inputs(data_root, declaration) != receipts
                or file_sha(declaration_path) != declaration_sha):
            raise ValueError("Source or data changed during the declared analysis")
        report = {"schema": declaration["schema"], "status": "completed", "human_patients": 1,
            "training_updates": 0, "clinical_deficit_probability": None,
            "limitations": declaration["limitations"], "declared_configuration_sha256": declaration_sha,
            "input_receipts": receipts, "new_bundle": {"path": str(bundle_output), "sha256": file_sha(bundle_output),
                "case_hash": primary_case.semantic_hash, "functional_evidence_hash": primary_case.functional_evidence.fingerprint},
            "clocks": clocks, "geometry_clocks": geometry_clocks, "panels": reports,
            "worker_seconds": perf_counter() - started, "peak_rss_gib": rss_gib(),
            "source_and_inputs_unchanged": True, "final_evaluation_not_external_patient_validation": True}
        write_json(output / "report.json", report)
        print(json.dumps({"status": "completed", "seconds": report["worker_seconds"],
                          "peak_rss_gib": report["peak_rss_gib"]}), flush=True)
        return report
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
            "elapsed_seconds": perf_counter() - started, "peak_rss_gib": rss_gib(),
            "completed_panels": [row["panel"] for row in reports], "clocks": clocks,
            "declared_configuration_sha256": declaration_sha,
            "automatic_retry": False, "clinical_deficit_probability": None})
        raise


def supervise(data_root, output, bundle_output, declaration_path):
    """Bound the whole worker independently of its cooperative checkpoints."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    declaration_path = Path(declaration_path).resolve()
    declaration_bytes = declaration_path.read_bytes()
    declaration = json.loads(declaration_bytes)
    frozen_declaration = output / "frozen-declaration.json"
    frozen_declaration.write_bytes(declaration_bytes)
    started = perf_counter()
    log = output / "worker.log"
    worker_output = output / "analysis"
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker",
        "--data-root", str(Path(data_root).resolve()), "--output", str(worker_output.resolve()),
        "--bundle-output", str(Path(bundle_output).resolve()), "--declaration", str(frozen_declaration.resolve())]
    reason, maximum_rss = None, 0.
    def stop_worker(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = perf_counter() - started
                sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                        capture_output=True, text=True, timeout=2, check=False)
                try:
                    current_rss = int(sample.stdout.strip()) / (1024 ** 2)
                except ValueError:
                    if process.poll() is None:
                        raise RuntimeError("Worker RSS unavailable while process is running")
                    current_rss = 0.
                maximum_rss = max(maximum_rss, current_rss)
                if elapsed > declaration["budgets"]["wall_seconds"]:
                    reason = "parent_wall_budget_exceeded"
                elif current_rss > declaration["budgets"]["peak_rss_gib"]:
                    reason = "parent_sampled_rss_budget_exceeded"
                if reason is not None:
                    stop_worker(process)
                    break
                sleep(.2)
        except BaseException as exc:
            reason = "parent_supervision_error:" + type(exc).__name__
            stop_worker(process)
        returncode = process.wait()
    if returncode == 0 and reason is None:
        try:
            completed_report = json.loads((worker_output / "report.json").read_text())
            if (completed_report.get("status") != "completed"
                    or completed_report.get("declared_configuration_sha256") != sha256(declaration_bytes).hexdigest()
                    or completed_report.get("source_and_inputs_unchanged") is not True):
                raise ValueError("Invalid worker completion receipt")
        except (OSError, ValueError):
            reason = "worker_exit_without_valid_completion_receipt"
    result = {"status": "completed" if returncode == 0 and reason is None else "failed",
        "worker_exit_code": returncode, "termination_reason": reason,
        "parent_wall_seconds": perf_counter() - started, "parent_sampled_peak_rss_gib": maximum_rss,
        "rss_sampling_interval_seconds": .2, "termination_grace_seconds": 2,
        "worker_log": str(log), "worker_output": str(worker_output), "automatic_retry": False,
        "command": command, "declaration_sha256": sha256(declaration_bytes).hexdigest()}
    write_json(output / "supervisor.json", result)
    if result["status"] != "completed":
        write_json(output / "failure.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run(args.data_root, args.output, args.bundle_output, args.declaration)
    else:
        result = supervise(args.data_root, args.output, args.bundle_output, args.declaration)
        if result["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
