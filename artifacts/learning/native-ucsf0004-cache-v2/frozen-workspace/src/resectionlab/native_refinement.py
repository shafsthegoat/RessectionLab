"""Headless desktop facade for real native-cell updates and certified selection replay."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .worlds import content_hash


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def replay_artifact_hash(replay: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in replay.items() if key != "artifact_hash"})


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
    if metrics.get("clinical_deficit_probability") is not None:
        raise ValueError("Clinical deficit probabilities are unsupported")
    return True


def run_native_refinement(case: Any, output_dir: str | Path, *, budget_seconds: float = 30.,
                          seed: int = 0, resume: bool = False, cancelled=None, progress=None,
                          access=None, tools=None, candidate_count: int = 4,
                          max_steps: int = 3, max_actions: int = 7) -> dict[str, Any]:
    """Train within fixed assumptions; only independently accepted replay is exposed.

    The budget covers learner initialization, optimization and selection; native
    preprocessing and independent certification are separate measured costs.
    Resume retains the original total learner budget and frozen source contract.
    """
    from .evaluation import independent_check_native_history
    from .learning import TrainingConfig, load_policy, rollout_policy, train_patient_policy
    from .native_simulation import make_native_patient_simulator
    from .worlds import generate_partitions

    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda _: None)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    template = make_native_patient_simulator(case, access=access, tools=tools,
        candidate_count=candidate_count, max_steps=max_steps, max_actions=max_actions, cancelled=cancelled)
    preparation_seconds = time.perf_counter() - started
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
        replay=None, replay_status="cancelled" if cancelled() else "pending_independent_check")
    if cancelled() or result.status == "cancelled":
        _write(directory / "native-refinement.json", report)
        return report
    check_started = time.perf_counter()
    policy = load_policy(directory / "checkpoint.pt")
    selected = rollout_policy(policy, template.clone(), seed=partitions.selection.seeds[0],
        max_steps=config.max_episode_steps, expected_model_hash=template.decision_model_hash, interrupt=cancelled)
    replay = {"role": "selection", "case_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "decision_model_hash": template.decision_model_hash, "checkpoint_hash": result.selected_checkpoint_hash,
        "selection_partition_hash": partitions.selection.partition_hash, "actions": list(selected.actions),
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
