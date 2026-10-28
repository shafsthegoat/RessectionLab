#!/usr/bin/env python3
"""Run a sealed, no-optimizer sector assessment on one fixed UCSF anatomy.

The scene and checkpoint must have been sealed before assessment image access.
All 14 predeclared controls/actors are retained, including unpassed episodes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from prepare_frozen_ucsf_assessment import (
    ACCESS_METHOD, CASE, CHOSEN, ROSTER, NativeCallCounter, digest, freeze_path,
    output_root, protocol_path, validate_acquired_role, validate_freeze)
from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.patient_sector_learning import (
    OptionActor, actor_choice, fixed_choice, frozen_actor, parameter_hash,
    prepare_case, rollout)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.workspace_bundle import load_workspace, save_workspace


def _write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_identity(case, artifacts, protocol, source: Path,
                     package_root: Path) -> None:
    if (not source.resolve().is_relative_to(Path(package_root).resolve()) or
            source.name != "frozen-access-comparison.ressectionlab"):
        raise ValueError("Assessment access source is outside the frozen output namespace")
    adapter = case.metadata.get("surgical_adapter", {})
    manifest = adapter.get("sourceManifest", {})
    series = manifest.get("series", [])
    revisions = {row.get("source_mirror_revision") for row in series}
    if (adapter.get("sourcePatient") != CASE or manifest.get("patient_id") != CASE or
            not series or revisions != {protocol["mirrorRevision"]} or
            artifacts.get("surgical_toolbox", {}).get("planning", {}).get("task") != "anatomy-access"):
        raise ValueError("Assessment source patient, revision or access planning differs")


def _specs(freeze: dict):
    yield "script-near-left", lambda: fixed_choice((
        "corridor:left", "near:left", "far:left", "STOP")), None
    for hand in ("left", "right"):
        for order in (("near", "far"), ("far", "near")):
            label = f"fixed-{hand}-{order[0]}-first"
            sequence = (f"corridor:{hand}", f"{order[0]}:{hand}",
                        f"{order[1]}:{hand}", "STOP")
            yield label, lambda sequence=sequence: fixed_choice(sequence), None
    for seed in (17, 29, 43):
        def untrained(seed=seed):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                actor = OptionActor()
            actor.eval()
            return actor_choice(actor), {"stage": "untrained", "seed": seed,
                                          "actorSha256": parameter_hash(actor)}
        yield f"untrained-seed{seed}", untrained, None
    for name in ROSTER:
        def selected(name=name):
            row = freeze["checkpoints"][name]
            actor = frozen_actor(row["path"], expected_producer=freeze["producerIdentity"])
            if parameter_hash(actor) != row["actorSha256"]:
                raise ValueError("Sealed actor changed before execution")
            return actor_choice(actor), {"stage": row["stage"],
                "checkpoint": name, "checkpointFileSha256": row["fileSha256"],
                "actorSha256": row["actorSha256"],
                "trainingReceiptSha256": row["trainingReceiptSha256"]}
        yield name.removesuffix(".pt"), selected, name


def _export(source: Path, prepared, trial: dict, label: str, model: dict | None,
            destination: Path, protocol: dict, freeze: dict,
            package_root: Path) -> dict:
    """Save a genuine frozen assessment branch with its own role metadata."""
    case, artifacts, workspace = load_workspace(source)
    arts = copy.deepcopy(artifacts)
    record = arts["surgical_toolbox"]
    branch_id = "frozen-assessment-" + label
    if branch_id in record["branches"] or destination.exists():
        raise ValueError("Assessment export branch or destination already exists")
    record["branches"][branch_id] = trial["engine"].snapshot()
    record.setdefault("parents", {})[branch_id] = {
        "branchId": record["branchId"], "position": 0}
    report = trial["report"]
    metadata = {
        "schema": "resectionlab.frozen-anatomy-assessment.v1",
        "branchId": branch_id, "sourceRole": "FROZEN_ASSESSMENT",
        "evaluationRole": "FROZEN_ASSESSMENT", "optimizationRole": "NONE_ON_ASSESSMENT",
        "optimizerUpdates": 0, "method": label,
        "trainingRole": "paired DEVELOPMENT 0007/0005" if model and model.get("checkpoint") else None,
        "model": model, "patientId": CASE,
        "protocolSha256": digest(protocol_path(package_root)),
        "freezeSha256": digest(freeze_path(package_root)),
        "sourceArchiveSha256": prepared.source_sha256,
        "sourceSceneId": prepared.scene.identity,
        "sourceAccessComparisonSeal": prepared.access_comparison_seal,
        "sourceAccessPlanSeal": prepared.access["planSeal"],
        "sourceAccessMethod": prepared.access_method,
        "primaryCheckpointPredeclared": protocol["primaryCheckpoint"],
        "status": report["status"], "taskGateMet": report["taskGateMet"],
        "cleanupComplete": report["cleanupComplete"],
        "deltaTargetMm3": report["deltaTargetMm3"],
        "deltaNonTargetMm3": report["deltaNonTargetMm3"],
        "totalNonTargetMm3": report["totalNonTargetMm3"],
        "perSectorRemovedMm3": report["perSectorRemovedMm3"],
        "policyDecisions": copy.deepcopy(report["decisions"]),
        "executionCost": copy.deepcopy(report["cost"]),
        "nativeActions": report["nativeActions"],
        "finalStateId": report["finalStateId"],
        "exactNativeReplay": report["exactNativeReplay"],
        "promotionStatus": "assessment-observation-only-no-clinical-promotion",
        "limitations": [protocol["claimLimit"],
                        "Modeled contact/removal is not calibrated patient injury probability"]}
    record["frozenAssessmentExecutions"] = [metadata]
    arts["frozen_anatomy_assessment"] = metadata
    if workspace is None:
        save_case(case, destination, artifacts=arts)
    else:
        save_workspace(case, destination, sources=workspace["sources"],
            imaging=workspace["imagingState"], episode=workspace["episode"],
            selection=workspace["selection"], originals=workspace["originals"],
            artifacts=arts)
    reopened_case, reopened_artifacts, _ = load_workspace(destination)
    saved = reopened_artifacts["surgical_toolbox"]["branches"][branch_id]
    replay = SurgicalEngine.restore(prepared.scene, saved)
    if (reopened_case.semantic_hash != case.semantic_hash or
            replay.state_id != trial["engine"].state_id or
            replay.actions != trial["engine"].actions or
            reopened_artifacts["frozen_anatomy_assessment"] != metadata):
        raise ValueError("Frozen assessment native export did not reopen/replay exactly")
    return {"path": str(destination), "fileSha256": digest(destination),
            "branchId": branch_id, "finalStateId": replay.state_id,
            "nativeReplayCalls": len(replay.actions), "reopenedExact": True,
            "metadataSha256": semantic_digest(metadata)}


def run(package_root: Path, access_preparation: Path, output: Path, *, pixel_go: bool) -> dict:
    if pixel_go is not True:
        raise ValueError("Explicit PIXEL GO is required for frozen assessment execution")
    package_root = Path(package_root).resolve()
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to((ROOT / "testcases").resolve()):
        raise ValueError("Choose a new ignored assessment output directory")
    started = time.monotonic()
    protocol, freeze = validate_freeze(package_root)
    validate_acquired_role(package_root)
    preparation = json.loads(Path(access_preparation).read_text())
    source = Path(preparation.get("output", ""))
    selection_file = Path(access_preparation).parent / "source-only-selection.json"
    source_preparation_calls = preparation.get("nativeCalls", {}).get("physicalCalls")
    source_preparation_seconds = preparation.get("nativeCalls", {}).get("wallSeconds")
    if (preparation.get("status") != "saved" or
            preparation.get("sourceRole") != "FROZEN_ASSESSMENT" or
            preparation.get("patientId") != CASE or
            preparation.get("protocolSha256") != digest(protocol_path(package_root)) or
            preparation.get("freezeSha256") != digest(freeze_path(package_root)) or
            not selection_file.is_file() or
            digest(selection_file) != preparation.get("sourceOnlySelectionFileSha256") or
            not source.is_file() or digest(source) != preparation.get("outputSha256") or
            type(source_preparation_calls) is not int or
            not 0 <= source_preparation_calls <= 1500 or
            type(source_preparation_seconds) not in (int, float) or
            not math.isfinite(source_preparation_seconds) or
            not 0 <= source_preparation_seconds <= 120):
        raise ValueError("Frozen access preparation is missing, changed or axis0 unsupported")
    if preparation.get("assessmentRouteSupported") is not True:
        output.mkdir(parents=True)
        unsupported = {"schema": "resectionlab.frozen-sector-assessment.v1",
            "status": "unsupported", "reason": "Predeclared axis0-access route did not execute",
            "patientId": CASE, "sourceRole": "FROZEN_ASSESSMENT",
            "source": str(source.resolve()), "sourceSha256": digest(source),
            "protocolSha256": digest(protocol_path(package_root)),
            "freezeSha256": digest(freeze_path(package_root)),
            "accessPreparationReceiptSha256": digest(access_preparation),
            "optimizerUpdates": 0, "executions": [],
            "sourcePreparationPhysicalCalls": source_preparation_calls,
            "sourcePreparationWallSeconds": source_preparation_seconds}
        _write_json(output / "receipt.json", unsupported)
        return unsupported
    remaining_calls = protocol["nativeBudget"]["assessmentAggregateMaxPhysicalCalls"] - source_preparation_calls
    remaining_seconds = (protocol["nativeBudget"]["assessmentAggregateMaxWallSeconds"] -
                         source_preparation_seconds - (time.monotonic()-started))
    if remaining_calls <= 0 or remaining_seconds <= 0:
        raise ValueError("Source preparation exhausted the complete assessment budget")
    output.mkdir(parents=True)
    receipt = {"schema": "resectionlab.frozen-sector-assessment.v1",
        "status": "running", "patientId": CASE,
        "sourceRole": "FROZEN_ASSESSMENT", "optimizerUpdates": 0,
        "source": str(source.resolve()), "sourceSha256": digest(source),
        "protocolSha256": digest(protocol_path(package_root)),
        "freezeSha256": digest(freeze_path(package_root)),
        "accessPreparationReceiptSha256": digest(access_preparation),
        "selectedAccessMethod": ACCESS_METHOD,
        "predeclaredPrimaryCheckpoint": CHOSEN,
        "sourcePreparationPhysicalCalls": source_preparation_calls,
        "sourcePreparationWallSeconds": source_preparation_seconds,
        "executions": [], "failure": None}
    counter = NativeCallCounter(remaining_calls, remaining_seconds)
    try:
        with counter:
            counter.phase = "assessment-source-validation"
            case, artifacts, _ = load_workspace(source)
            _source_identity(case, artifacts, protocol, source, package_root)
            prepared = prepare_case(source, access_method=ACCESS_METHOD)
            if prepared.case_id != case.case_id:
                raise ValueError("Frozen prepared source changed identity")
            specifications = list(_specs(freeze))
            if len(specifications) != 14:
                raise ValueError("Predeclared assessment control/checkpoint roster changed")
            for index, (label, make_choice, checkpoint_name) in enumerate(specifications):
                row = {"index": index, "method": label, "checkpoint": checkpoint_name,
                       "status": "running"}
                receipt["executions"].append(row)
                counter.phase = "execution-" + label
                counter.begin_execution(max_calls=512, max_seconds=600)
                try:
                    made = make_choice()
                    if isinstance(made, tuple):
                        choose, model = made
                    else:
                        choose, model = made, None
                    trial = rollout(prepared, choose, method=label,
                                    max_wall_seconds=min(600., remaining_seconds))
                    destination = output / (label + ".ressectionlab")
                    counter.phase = "export-" + label
                    exported = _export(source, prepared, trial, label, model,
                                       destination, protocol, freeze, package_root)
                    counter.check_time()
                    row.update(status="evaluated", report=trial["report"],
                               model=model, export=exported)
                except Exception:
                    row.update(status="failed", failure=traceback.format_exc())
                finally:
                    row["meter"] = counter.end_execution()
                    _write_json(output / f"{index:02d}-{label}.json", row)
                if counter.calls >= counter.max_calls or time.monotonic()-counter.started >= counter.max_seconds:
                    break
        receipt["status"] = ("evaluated" if len(receipt["executions"]) == 14 and
                             all(row["status"] == "evaluated" for row in receipt["executions"])
                             else "completed_with_failures")
    except Exception:
        receipt.update(status="failed", failure=traceback.format_exc())
    finally:
        receipt["nativeCalls"] = counter.report()
        receipt["totalPhysicalCallsIncludingPreparation"] = source_preparation_calls + counter.calls
        receipt["totalWallSecondsIncludingPreparation"] = source_preparation_seconds + (
            time.monotonic()-started)
        if (receipt["totalPhysicalCallsIncludingPreparation"] > 10000 or
                receipt["totalWallSecondsIncludingPreparation"] > 900):
            receipt.update(status="failed", failure="Frozen aggregate native-call or wall budget exceeded")
        receipt["sourceUnchanged"] = digest(source) == receipt["sourceSha256"]
        receipt["checkpointFreezeUnchanged"] = digest(freeze_path(package_root)) == receipt["freezeSha256"]
        _write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-preparation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=ROOT / "testcases")
    parser.add_argument("--pixel-go", action="store_true")
    args = parser.parse_args()
    if not args.package_root.resolve().is_relative_to((ROOT / "testcases").resolve()):
        parser.error("Package root must remain under ignored testcases/")
    result = run(args.package_root, args.access_preparation_receipt, args.output,
                 pixel_go=args.pixel_go)
    print(json.dumps({"status": result["status"],
                      "receipt": str(Path(args.output).resolve() / "receipt.json")}))
    if result["status"] != "evaluated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
