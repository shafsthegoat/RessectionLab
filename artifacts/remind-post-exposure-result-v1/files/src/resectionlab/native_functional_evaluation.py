"""Seal one selected native history before revealing independent functional worlds.

This local interface evaluates the entire complete-tool sequence. Its scenarios
are uncalibrated sensitivity assumptions, not postoperative outcome predictions.
The shared ledger is claimed durably before any event is computed. Cancellation
can resume that same evaluation; it cannot reopen optimizer training.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .core import Plan, freeze_json, thaw_json
from .evaluation import (CandidateFreezeManifest, EvaluationLedger,
                         IndependentGeometryResult, freeze_candidates)
from .functional_events import (FunctionalEventConfig, evaluate_functional_candidates,
                                prepare_functional_exposure, sweeps_from_native_history)
from .worlds import (FrozenDecisionModel, WorldGeneratorConfig, WorldPartitionManifest,
                     WorldPartitions, content_hash)

SEAL_FILE = "native-functional-freeze.json"
REPORT_FILE = "native-functional-events.json"
LEDGER_FILE = "functional-evaluation-ledger.json"
VERSION = "native-sequence-functional-evaluation-v1"


def _json(value):
    return thaw_json(freeze_json(value))


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Functional evaluation artifacts cannot be redirected")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _check(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("Functional evaluation cancelled; any existing candidate seal remains in force")


def partitions_from_record(record: dict) -> WorldPartitions:
    def parse(role):
        item = record[role]
        result = WorldPartitionManifest(item["role"], item["case_hash"],
            WorldGeneratorConfig(**item["generator"]), tuple(item["seeds"]), item.get("planning_hash"))
        if _json(result.to_dict()) != item:
            raise ValueError("Saved functional world partition changed")
        return result
    return WorldPartitions(*(parse(role) for role in
        ("optimization", "selection", "final_evaluation", "stress")))


def replay_source_hash(replay: dict) -> str:
    # Certificates and functional reports are independently derived from this
    # fixed selection artifact. Excluding them avoids circular evidence hashes.
    return content_hash({key: value for key, value in replay.items()
        if key not in {"artifact_hash", "native_certificate", "functional_assessment"}})


def validate_assessment_binding(case, replay: dict) -> None:
    assessment = replay.get("functional_assessment")
    if assessment is None:
        return
    if not isinstance(assessment, dict):
        raise ValueError("Functional assessment must be a sealed report")
    evidence = case.functional_evidence
    if (evidence is None or assessment.get("version") != VERSION
            or assessment.get("assessment_hash") != content_hash({k: v for k, v in assessment.items()
                                                                  if k != "assessment_hash"})
            or assessment.get("source_replay_hash") != replay_source_hash(replay)
            or assessment.get("functional_evidence_hash") != evidence.fingerprint
            or assessment.get("world_generator_hash") != evidence.uncertainty.fingerprint
            or assessment.get("selected_checkpoint_hash") != replay.get("checkpoint_hash")
            or assessment.get("native_decision_model_hash") != replay.get("decision_model_hash")):
        raise ValueError("Functional assessment differs from its frozen replay, checkpoint or evidence")


def _prepare(case, replay, native, partitions, *, seal=None, cancelled=None):
    _check(cancelled)
    evidence = case.functional_evidence
    if evidence is None:
        raise ValueError("Functional evaluation requires explicitly selected evidence; structural-only mode is unassessed")
    evidence.assert_matches(case)
    if (partitions.selection.partition_hash != replay["selection_partition_hash"]
            or partitions.final_evaluation.case_hash != case.semantic_hash
            or partitions.final_evaluation.planning_hash != case.planning_hash
            or partitions.final_evaluation.generator.fingerprint != evidence.uncertainty.fingerprint):
        raise ValueError("Functional worlds differ from the original selected case and uncertainty")
    history = replay["metrics"]["history"]
    history_hash = content_hash(history)
    points = np.array([point for step in history for point in (step["entry_mm"], step["tip_mm"])],
                      dtype=float).reshape(-1, 3)
    candidate = Plan("native-sequence-" + history_hash[7:23], case.semantic_hash, points,
        "+".join(sorted({step["tool_id"] for step in history})) or "STOP",
        plan_type="simulated_resection", optimizer_mode="PATIENT_SCRATCH_RL",
        simulated_removed_target_volume_mm3=replay["metrics"]["simulated_removed_target_volume_mm3"],
        removal_replay_hash=history_hash, world_model_version=evidence.uncertainty.version,
        world_partition=partitions.final_evaluation.partition_hash,
        metadata={"native_history_hash": history_hash, "tool_catalog": [asdict(tool) for tool in native.tools],
                  "selected_checkpoint_hash": replay["checkpoint_hash"],
                  "native_decision_model_hash": replay["decision_model_hash"]},
        unknowns=("vascular_anatomy_unassessed", "patient_specific_function_unassessed"))
    config = FunctionalEventConfig()
    footprint = prepare_functional_exposure(sweeps_from_native_history(history, native.tools),
        tissue_mask=native.tissue_mask, affine_ras_mm=native.affine,
        candidate_hash=candidate.semantic_hash, case_hash=case.semantic_hash, cancelled=cancelled)
    model = FrozenDecisionModel.create(case_hash=case.semantic_hash,
        geometry={"native_decision_model_hash": replay["decision_model_hash"],
            "route_binding_hash": replay["route_binding"]["binding_hash"],
            "functional_evidence_hash": evidence.fingerprint,
            "functional_event_config": config.to_dict(), "tissue_support_hash": footprint.tissue_support_hash,
            "functional_footprint_hashes": {candidate.plan_id: footprint.fingerprint}},
        objectives={"scope": "independent_complete_sequence_sensitivity; no_checkpoint_selection",
                    "clinical_deficit_probability": None},
        tools=[asdict(tool) for tool in native.tools], world_generator=evidence.uncertainty.to_dict(),
        action_primitives={"native_history_hash": history_hash},
        planning_as_of=None if case.context is None else case.context.planning_as_of.isoformat())
    binding = _json({"version": VERSION, "case_hash": case.semantic_hash,
        "planning_hash": case.planning_hash, "source_replay_hash": replay_source_hash(replay),
        "candidate": candidate.to_dict(), "candidate_hash": candidate.semantic_hash,
        "decision_model": model.to_dict(), "partitions": partitions.to_dict()})
    if seal is None:
        frozen = freeze_candidates([candidate], model, partitions.selection,
            "already_selected_checkpoint_and_exact_native_history; no_event_based_reselection",
            optimization_manifest=partitions.optimization)
        seal = {**binding, "candidate_freeze": _json(frozen.to_dict())}
        seal["seal_hash"] = content_hash(seal)
    else:
        if (any(seal.get(key) != value for key, value in binding.items())
                or seal.get("seal_hash") != content_hash({k: v for k, v in seal.items() if k != "seal_hash"})):
            raise ValueError("Sealed functional evaluation belongs to a different candidate, evidence or world configuration")
        frozen = CandidateFreezeManifest(**{k: v for k, v in seal["candidate_freeze"].items() if k != "fingerprint"})
        if frozen.fingerprint != seal["candidate_freeze"].get("fingerprint"):
            raise ValueError("Functional candidate freeze changed")
    return candidate, footprint, model, frozen, seal


def _evaluate(case, replay, prepared, partitions, *, cancelled=None):
    candidate, footprint, model, frozen, seal = prepared
    certificate = replay["native_certificate"]
    def checked_geometry(_candidate):
        _check(cancelled)
        return IndependentGeometryResult(certificate["feasible"], tuple(certificate["failures"]),
            checker_version=certificate["checker_version"],
            unknowns=("vascular_anatomy_unassessed", "patient_specific_function_unassessed"))
    events = evaluate_functional_candidates([candidate], frozen, model, partitions.final_evaluation,
        evidence=case.functional_evidence, footprints={candidate.plan_id: footprint},
        geometry_checker=checked_geometry, ledger=EvaluationLedger())
    _check(cancelled)
    result = {"version": VERSION, "status": "complete", "scope": "entire_selected_native_sequence",
        "source_replay_hash": replay_source_hash(replay), "selected_checkpoint_hash": replay["checkpoint_hash"],
        "native_decision_model_hash": replay["decision_model_hash"],
        "functional_evidence_hash": case.functional_evidence.fingerprint,
        "world_generator_hash": case.functional_evidence.uncertainty.fingerprint,
        "seal": seal, "event_report": events, "vascular_evidence_status": "unassessed",
        "clinical_deficit_probability": None,
        "interpretation": "Held-out model-conditioned population-map encounters; no clinical outcome prediction or checkpoint reselection"}
    result = _json(result)
    result["assessment_hash"] = content_hash(result)
    return result


@contextmanager
def _ledger_lock(path, cancelled):
    lock = path.with_suffix(path.suffix + ".lock")
    if path.is_symlink() or lock.is_symlink():
        raise ValueError("Functional evaluation ledger cannot be redirected")
    with lock.open("a") as handle:
        while True:
            _check(cancelled)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.02)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluate_selected_native_sequence(case, replay, native, partitions: WorldPartitions,
                                      directory: str | Path, *, cancelled=None, on_seal=None) -> dict:
    """Persist a seal before outcomes; repeat calls use exactly that same seal.

    Caller independently rechecks native geometry and checkpoint identity first.
    The directory's parent owns a shared ledger so separate local runs cannot
    reuse revealed worlds for another frozen candidate under the same model.
    """
    directory = Path(directory)
    seal_path, report_path = directory / SEAL_FILE, directory / REPORT_FILE
    if seal_path.is_symlink() or report_path.is_symlink():
        raise ValueError("Functional evaluation artifacts cannot be redirected")
    prior = json.loads(seal_path.read_text()) if seal_path.exists() else None
    prepared = _prepare(case, replay, native, partitions, seal=prior, cancelled=cancelled)
    frozen, seal = prepared[3], prepared[4]
    if prior is None:
        _write(seal_path, seal)
    if on_seal is not None:
        on_seal(seal)  # Desktop signs this durable state before any outcome is read.
    ledger_path = directory.parent / LEDGER_FILE
    with _ledger_lock(ledger_path, cancelled):
        ledger = EvaluationLedger(json.loads(ledger_path.read_text()) if ledger_path.exists() else {})
        ledger.claim(partitions.final_evaluation, frozen)
        _write(ledger_path, ledger.to_dict())
    result = _evaluate(case, replay, prepared, partitions, cancelled=cancelled)
    if report_path.exists():
        if json.loads(report_path.read_text()) != result:
            raise ValueError("Saved functional outcomes differ from independent deterministic re-evaluation")
    else:
        _write(report_path, result)
    return result


def verify_selected_native_assessment(case, replay, native, *, cancelled=None) -> None:
    """Recompute supplied events from the same frozen history and worlds."""
    validate_assessment_binding(case, replay)
    report = replay.get("functional_assessment")
    if report is None:
        return
    seal = report["seal"]
    partitions = partitions_from_record(seal["partitions"])
    prepared = _prepare(case, replay, native, partitions, seal=seal, cancelled=cancelled)
    if _evaluate(case, replay, prepared, partitions, cancelled=cancelled) != report:
        raise ValueError("Functional event outcomes do not reproduce from the frozen full-tool history")
