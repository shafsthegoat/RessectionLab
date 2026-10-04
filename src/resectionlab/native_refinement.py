"""Headless desktop facade for real native-cell updates and certified selection replay."""
from __future__ import annotations

import json
import copy
from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .worlds import content_hash
from .selection_contract import has_completed_selection


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def replay_artifact_hash(replay: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in replay.items() if key != "artifact_hash"})


def require_completed_selection_replay(report: dict[str, Any], contract: dict[str, Any]) -> None:
    """Gate fresh and historical desktop replay against its frozen learner contract."""
    replay = report.get("replay")
    if replay is None:
        return
    partitions = contract.get("partitions", {})
    selection = partitions.get("selection", {}) if isinstance(partitions, dict) else {}
    if (not isinstance(replay, dict) or not isinstance(selection, dict)
            or report.get("status") in {"cancelled", "failed", "running", "incomplete_selection"}
            or not has_completed_selection(report, selection)
            or not isinstance(selection.get("partition_hash"), str) or not selection["partition_hash"]
            or replay.get("checkpoint_hash") != report.get("selected_checkpoint_hash")
            or replay.get("selection_partition_hash") != selection.get("partition_hash")):
        raise ValueError("Replay requires a completed selection panel for its frozen checkpoint and worlds")


def _point(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite canonical RAS+ three-vector")
    return array.tolist()


def _access_record(access: Any) -> dict[str, Any] | None:
    if access is None:
        return None
    return {"center_mm": np.asarray(access.center_mm).tolist(),
            "normal_inward": np.asarray(access.normal_inward).tolist(),
            "radius_mm": float(access.radius_mm), "window_id": access.window_id}


def _geometry_request(*, access, tools, selected_entry_mm, selected_target_mm,
                      candidate_count: int, max_steps: int, max_actions: int) -> dict[str, Any]:
    entry = _point(selected_entry_mm, "Selected entry")
    target = _point(selected_target_mm, "Selected target")
    if (entry is None) != (target is None):
        raise ValueError("Selected entry and target must be supplied together")
    if entry is not None and (access is None or not tools):
        raise ValueError("A selected route requires its explicit access window and tool geometry")
    return {"geometry_frame": "RAS+", "selected_entry_mm": entry, "selected_target_mm": target,
        "access": _access_record(access), "tools": None if tools is None else [asdict(tool) for tool in tools],
        "candidate_count": candidate_count, "max_steps": max_steps, "max_actions": max_actions}


def _prepare_native(case: Any, *, access, tools, selected_entry_mm, selected_target_mm,
                    candidate_count: int, max_steps: int, max_actions: int, cancelled):
    from .native_simulation import make_native_patient_simulator
    from .route_native_diagnostics import initial_native_action_diagnostic

    requested = _geometry_request(access=access, tools=tools, selected_entry_mm=selected_entry_mm,
        selected_target_mm=selected_target_mm, candidate_count=candidate_count, max_steps=max_steps,
        max_actions=max_actions)
    template = make_native_patient_simulator(case, access=access, tools=tools,
        selected_entry_mm=requested["selected_entry_mm"], selected_target_mm=requested["selected_target_mm"],
        candidate_count=candidate_count, max_steps=max_steps, max_actions=max_actions, cancelled=cancelled)
    actual_access = _access_record(template.native_config.access)
    actual_tools = [asdict(tool) for tool in template.native_config.tools]
    if requested["selected_entry_mm"] is not None:
        if (len(template.candidate_tips_mm) != 1 or len(template.candidate_entries_mm) != 1
                or not np.array_equal(template.candidate_entries_mm[0], requested["selected_entry_mm"])
                or not np.array_equal(template.candidate_tips_mm[0], requested["selected_target_mm"])):
            raise ValueError("Native factory did not preserve the exact selected entry/target pair")
        if actual_access != requested["access"]:
            raise ValueError("Native factory did not preserve the exact selected access window")
        if actual_tools != requested["tools"]:
            raise ValueError("Native factory did not preserve the complete selected tool geometry")
    binding = {"version": "native-route-binding-v1", "geometry_frame": "RAS+",
        "mode": "exact_selected_route" if requested["selected_entry_mm"] is not None else "default_candidate_set",
        "requested_geometry": requested, "access": actual_access, "tools": actual_tools,
        "candidate_entries_mm": template.candidate_entries_mm.tolist(),
        "candidate_targets_mm": template.candidate_tips_mm.tolist(),
        "decision_model_hash": template.decision_model_hash}
    binding["binding_hash"] = content_hash(binding)
    diagnostic = initial_native_action_diagnostic(template)
    legal = diagnostic["non_stop_action_count"]
    readiness = {"status": "ready" if legal else "no_actionable_moves", "role": "preflight", "legalNonStopActions": legal,
        "action_ids": diagnostic["non_stop_action_ids"], "reasons": sorted(diagnostic["rejection_counts"]),
        "diagnostic": diagnostic, "decision_model_hash": template.decision_model_hash,
        "case_hash": case.semantic_hash, "route_binding": binding, "clinical_deficit_probability": None}
    if not legal and not readiness["reasons"]:
        readiness["reasons"] = ["no_legal_native_nonstop_action"]
    return template, readiness


def inspect_native_refinement(case: Any, *, access=None, tools=None, selected_entry_mm=None,
                              selected_target_mm=None, candidate_count: int = 4, max_steps: int = 3,
                              max_actions: int = 7, cancelled=None) -> dict[str, Any]:
    """Geometry-only preflight; no optimizer, checkpoint, policy rollout or removal."""
    started = time.perf_counter()
    _, readiness = _prepare_native(case, access=access, tools=tools, selected_entry_mm=selected_entry_mm,
        selected_target_mm=selected_target_mm, candidate_count=candidate_count, max_steps=max_steps,
        max_actions=max_actions, cancelled=cancelled)
    readiness["preparation_seconds"] = time.perf_counter() - started
    return readiness


def native_replay_mask(replay: dict[str, Any], step: int) -> np.ndarray:
    """Only fully contained, actually removed source cells enter the visible cavity."""
    history = replay["metrics"]["history"]
    if type(step) is not int or not 0 <= step <= len(history):
        raise ValueError("Replay step is outside the retained removal history")
    mask = np.zeros(tuple(replay["shape"]), bool)
    for record in history[:step]:
        indices = np.asarray(record.get("removed_indices_native", []))
        if not indices.size:
            continue
        if (indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= np.asarray(mask.shape))):
            raise ValueError("Invalid native removal indices")
        if mask[tuple(indices.T)].any() or len(np.unique(indices, axis=0)) != len(indices):
            raise ValueError("Replay repeats already removed source cells")
        mask[tuple(indices.T)] = True
    return mask


def native_partial_contact_accounting(case: Any, history: list[dict[str, Any]]) -> dict[str, float]:
    """Separate ever partially contacted normal cells from those still retained."""
    removed: set[tuple[int, int, int]] = set()
    partial: set[tuple[int, int, int]] = set()
    for record in history:
        current = {tuple(point) for point in record.get("removed_indices_native", [])}
        contact = {tuple(point) for point in record.get("contact_indices_native", [])}
        partial |= contact - removed - current
        removed |= current
    normal = {point for point in partial if not any(mask[point] for mask in case.compartments.values())}
    volume = case.voxel_volume_mm3
    return {"cumulative_partial_normal_contact_mm3": float(len(normal) * volume),
            "currently_retained_partial_normal_contact_mm3": float(len(normal - removed) * volume),
            "previously_partial_normal_later_removed_mm3": float(len(normal & removed) * volume)}


def validate_native_replay(case: Any, replay: dict[str, Any]) -> bool:
    if replay.get("case_hash") != case.semantic_hash or replay.get("role") != "selection" or replay.get("final_evaluation") is not False:
        raise ValueError("Native replay is stale or is not a selection artifact")
    if replay.get("artifact_hash") != replay_artifact_hash(replay):
        raise ValueError("Native replay artifact changed after independent checking")
    certificate = replay.get("native_certificate", {})
    if not all(certificate.get(key) is True for key in ("feasible", "complete_tool_checked", "frontier_checked")):
        raise ValueError("Native replay lacks complete independent geometry checks")
    if certificate.get("source_case_hash") != case.semantic_hash:
        raise ValueError("Native certificate belongs to another source case")
    if tuple(replay.get("shape", ())) != case.mri.shape:
        raise ValueError("Native replay grid differs from source")
    affine = np.asarray(replay.get("affine"))
    expected = np.asarray(case.affine) if case.frame == "RAS+" else np.diag([-1., -1., 1., 1.]) @ case.affine
    if affine.shape != (4, 4) or not np.allclose(affine, expected, rtol=0, atol=1e-7):
        raise ValueError("Native replay physical frame differs from source")
    metrics = replay["metrics"]
    binding = replay.get("route_binding")
    if binding is not None:
        if binding.get("binding_hash") != content_hash({key: value for key, value in binding.items() if key != "binding_hash"}):
            raise ValueError("Native route binding changed after preparation")
        if binding.get("decision_model_hash") != replay.get("decision_model_hash"):
            raise ValueError("Native route binding belongs to a different decision model")
        entries = np.asarray(binding["candidate_entries_mm"], float)
        targets = np.asarray(binding["candidate_targets_mm"], float)
        allowed_tools = {tool["tool_id"] for tool in binding["tools"]}
        for record in metrics["history"]:
            matches = np.all(np.isclose(entries, record.get("entry_mm"), rtol=0, atol=1e-7), axis=1)
            matches &= np.all(np.isclose(targets, record.get("tip_mm"), rtol=0, atol=1e-7), axis=1)
            if not matches.any() or record.get("tool_id") not in allowed_tools:
                raise ValueError("Native replay departed from its retained entry, target or tool choices")
    if any(record.get("source_hash") != case.semantic_hash for record in metrics["history"]):
        raise ValueError("Native removal history belongs to another source case")
    removed = native_replay_mask(replay, len(metrics["history"]))
    target = np.zeros(case.mri.shape, bool)
    for mask in case.compartments.values():
        target |= mask
    volume = case.voxel_volume_mm3
    expected_totals = {"simulated_removed_target_volume_mm3": float((removed & target).sum() * volume),
        "simulated_removed_normal_volume_mm3": float((removed & ~target).sum() * volume),
        "modeled_residual_target_volume_mm3": float((target & ~removed).sum() * volume)}
    if any(not np.isclose(metrics.get(key, -1), value, rtol=1e-7, atol=1e-6) for key, value in expected_totals.items()):
        raise ValueError("Native replay volume totals disagree with source-cell removal")
    if not np.isclose(certificate.get("contained_source_tissue_volume_mm3", -1), removed.sum() * volume,
                      rtol=1e-7, atol=1e-6):
        raise ValueError("Native certificate volume differs from retained removal")
    contacts = native_partial_contact_accounting(case, metrics["history"])
    if any(not np.isclose(metrics.get(key, -1), value, rtol=1e-7, atol=1e-6) for key, value in contacts.items()):
        raise ValueError("Native partial-contact totals disagree with retained source history")
    if metrics.get("clinical_deficit_probability") is not None:
        raise ValueError("Clinical deficit probabilities are unsupported")
    return True


def recheck_native_replay(case: Any, replay: dict[str, Any], *, access=None, tools=None,
                         cancelled=None, candidate_count: int = 4, max_steps: int = 3,
                         max_actions: int = 7, selected_entry_mm=None,
                         selected_target_mm=None) -> dict[str, Any]:
    """Reopen disk artifacts without trusting their claimed certificate or digest.

    Fresh model replay must reproduce the saved native removal history. The
    independent checker then certifies that history anew. Policy provenance is a
    separate checkpoint-weight check for the caller; geometry cannot prove it.
    """
    from .evaluation import independent_check_native_history

    if replay.get("case_hash") != case.semantic_hash or replay.get("role") != "selection":
        raise ValueError("Saved native replay has a stale case or wrong world role")
    selected_seed = replay.get("selection_seed")
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("Saved native replay requires its selection seed")
    template, readiness = _prepare_native(case, access=access, tools=tools,
        selected_entry_mm=selected_entry_mm, selected_target_mm=selected_target_mm,
        candidate_count=candidate_count, max_steps=max_steps, max_actions=max_actions, cancelled=cancelled)
    if replay.get("route_binding") is not None:
        if replay["route_binding"] != readiness["route_binding"]:
            raise ValueError("Saved native replay route binding differs from the selected entry, target, window or tool")
    elif selected_entry_mm is not None:
        raise ValueError("Historical replay has no exact selected-route binding")
    if template.decision_model_hash != replay.get("decision_model_hash"):
        raise ValueError("Saved replay decision model differs from current native model")
    template.replay(replay["actions"], selected_seed)
    current = template.metrics()
    if content_hash(current["history"]) != content_hash(replay["metrics"]["history"]):
        raise ValueError("Saved history differs from current native action replay")
    native = template.native_config
    certificate = independent_check_native_history(case, native.tools, current["history"],
        tissue_mask=native.tissue_mask, access=native.access, hard_exclusion=native.hard_exclusion,
        geometry_frame="RAS+", cancelled=cancelled)
    refreshed = copy.deepcopy(replay)
    refreshed["native_certificate"] = certificate.to_dict()
    refreshed["artifact_hash"] = replay_artifact_hash(refreshed)
    validate_native_replay(case, refreshed)
    return refreshed


def run_native_refinement(case: Any, output_dir: str | Path, *, budget_seconds: float = 30.,
                          seed: int = 0, resume: bool = False, cancelled=None, progress=None,
                          access=None, tools=None, candidate_count: int = 4,
                          max_steps: int = 3, max_actions: int = 7, selected_entry_mm=None,
                          selected_target_mm=None) -> dict[str, Any]:
    """Train within fixed assumptions; only independently accepted replay is exposed.

    The budget covers optimization and selection; learner initialization, native
    preprocessing and independent certification are separate measured costs.
    Resume retains the original total learner budget and frozen source contract.
    """
    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda _: None)
    if type(seed) is not int or seed < 0 or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("Native refinement needs a nonnegative seed and positive finite budget")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "native-request.json"
    request = {"schema_version": 1, "case_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "seed": seed, "budget_seconds": float(budget_seconds),
        "geometry": _geometry_request(access=access, tools=tools, selected_entry_mm=selected_entry_mm,
            selected_target_mm=selected_target_mm, candidate_count=candidate_count, max_steps=max_steps,
            max_actions=max_actions)}
    previous = None
    if resume:
        if not request_path.is_file() or not (directory / "checkpoint.pt").is_file():
            raise ValueError("No matching native route request and optimizer checkpoint exist to resume")
        previous = json.loads(request_path.read_text())
        if previous.get("request") != request:
            raise ValueError("Resume route or settings changed; use a new run for edited geometry or budget")
        from .learning import numerical_source_hashes
        contract_path = directory / "contract.json"
        if not contract_path.is_file() or json.loads(contract_path.read_text()).get("numerical_source_sha256") != numerical_source_hashes():
            raise ValueError("Resume numerical source changed; historical native results remain preserved")
    elif any((directory / name).exists() for name in ("native-request.json", "native-refinement.json", "checkpoint.pt")):
        raise FileExistsError("Native refinement run exists; resume its exact request or use a new directory")
    started = time.perf_counter()
    template, readiness = _prepare_native(case, access=access, tools=tools,
        selected_entry_mm=selected_entry_mm, selected_target_mm=selected_target_mm,
        candidate_count=candidate_count, max_steps=max_steps, max_actions=max_actions, cancelled=cancelled)
    preparation_seconds = time.perf_counter() - started
    if previous is not None and previous.get("route_binding") != readiness["route_binding"]:
        raise ValueError("Resume native route model changed; retained source geometry cannot be substituted")
    if not resume:
        _write(request_path, {"request": request, "route_binding": readiness["route_binding"]})
    if readiness["status"] == "no_actionable_moves":
        report = {"status": "no_actionable_moves", "replay_status": "no_actionable_moves", "replay": None,
            "optimizer_mode": None, "requested_optimizer_mode": "PATIENT_SCRATCH_RL",
            "gradient_steps": 0, "optimization_environment_steps": 0, "selection_environment_steps": 0,
            "actor_parameters_changed": False, "initial_selection_return": None, "selected_selection_return": None,
            "case_hash": case.semantic_hash, "planning_hash": case.planning_hash, "role": "preflight",
            "final_evaluation": False, "elapsed_seconds": 0., "preparation_seconds": preparation_seconds,
            "decision_model_hash": template.decision_model_hash, "readiness": readiness,
            "route_binding": readiness["route_binding"], "selection_history": [], "optimization_history": [],
            "clinical_deficit_probability": None}
        _write(directory / "native-refinement.json", report)
        return report
    # Import the learner only after native geometry establishes a real action
    # choice. A STOP-only setup never creates a policy or optimizer checkpoint.
    from .evaluation import independent_check_native_history
    from .learning import TrainingConfig, load_policy, rollout_policy, train_patient_policy
    from .worlds import generate_partitions

    partitions = generate_partitions(case.semantic_hash, template.config.world_generator, seed,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=case.planning_hash)
    config = TrainingConfig(seed=seed, max_environment_steps=256, max_gradient_steps=32,
        max_wall_seconds=float(budget_seconds), max_episode_steps=max_steps + 1,
        episodes_per_update=2, checkpoint_interval=2, hidden_features=16)
    if not resume:
        _write(directory / "native-world-partitions.json", partitions.to_dict())
    result = train_patient_policy(template.clone, partitions.optimization, partitions.selection,
        config=config, output_dir=directory, cancelled=cancelled, progress=progress, resume=resume)
    report = json.loads((directory / "result.json").read_text())
    report.update(case_hash=case.semantic_hash, planning_hash=case.planning_hash,
        role="selection", final_evaluation=False, preparation_seconds=preparation_seconds,
        readiness=readiness, route_binding=readiness["route_binding"],
        replay=None, replay_status="cancelled" if cancelled() else "pending_independent_check")
    report["selection_panel_complete"] = has_completed_selection(
        {**report, **asdict(result)}, partitions.selection.to_dict())
    if cancelled() or result.status == "cancelled":
        report.update(status="cancelled", replay_status="cancelled")
        _write(directory / "native-refinement.json", report)
        return report
    if result.status == "failed" or not report["selection_panel_complete"]:
        report.update(training_status=result.status,
            status="failed" if result.status == "failed" else "incomplete_selection",
            replay_status="failed" if result.status == "failed" else "no_completed_selection")
        _write(directory / "native-refinement.json", report)
        return report
    check_started = time.perf_counter()
    policy = load_policy(directory / "checkpoint.pt")
    selected = rollout_policy(policy, template.clone(), seed=partitions.selection.seeds[0],
        max_steps=config.max_episode_steps, expected_model_hash=template.decision_model_hash, interrupt=cancelled)
    selected.metrics.update(native_partial_contact_accounting(case, selected.metrics["history"]))
    replay = {"role": "selection", "case_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "decision_model_hash": template.decision_model_hash, "checkpoint_hash": result.selected_checkpoint_hash,
        "selection_partition_hash": partitions.selection.partition_hash, "actions": list(selected.actions),
        "selection_seed": partitions.selection.seeds[0],
        "route_binding": readiness["route_binding"],
        "metrics": selected.metrics, "shape": list(case.mri.shape), "affine": template.config.affine.tolist(),
        "final_evaluation": False, "scope": "native_contained_cell_structural_research",
        "interpretation": "Independent geometric checks within rigid modeled anatomy; tissue mechanics and clinical consequences unvalidated"}
    # Freeze the exact candidate before any independent outcome is available.
    _write(directory / "native-candidate-freeze.json", {**replay, "candidate_hash": content_hash(replay)})
    native = template.native_config
    certificate = independent_check_native_history(case, native.tools, selected.metrics["history"],
        tissue_mask=native.tissue_mask, access=native.access, hard_exclusion=native.hard_exclusion,
        geometry_frame="RAS+", cancelled=cancelled)
    report["native_certificate"] = certificate.to_dict()
    report["selection_replay_and_check_seconds"] = time.perf_counter() - check_started
    if certificate.feasible:
        replay["native_certificate"] = certificate.to_dict()
        replay["artifact_hash"] = replay_artifact_hash(replay)
        validate_native_replay(case, replay)
        _write(directory / "native-selection-replay.json", replay)
        report.update(replay=replay, replay_status="accepted_independent_geometry")
    else:
        report["replay_status"] = "rejected_independent_geometry"
    _write(directory / "native-refinement.json", report)
    return report
