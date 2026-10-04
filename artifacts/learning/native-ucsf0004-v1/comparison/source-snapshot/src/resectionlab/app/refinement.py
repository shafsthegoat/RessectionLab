"""Headless bridge from the desktop to bounded, real policy-gradient updates.

The viewer consumes selection-set replay only. It never reads the final
evaluation partition while training, selecting checkpoints or preparing replay.
"""

from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time

import numpy as np


def run_refinement(case, output_dir, *, budget_seconds=60., seed=0, block_size=6,
                   resume=False, cancelled=None, progress=None):
    """Freeze a coarse structural scenario, train, and certify selection replay."""
    # Keep PyTorch out of application launch and ordinary imaging workflows.
    from resectionlab.learning import TrainingConfig, train_patient_policy, load_policy, rollout_policy
    from resectionlab.simulation import make_patient_simulator
    from resectionlab.worlds import WorldGeneratorConfig, generate_partitions

    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda _: None)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    world = WorldGeneratorConfig()
    simulator = make_patient_simulator(case, block_size=block_size, max_steps=24,
        max_actions=8, proposal_scan_limit=64, world_generator=world)
    partitions = generate_partitions(case.semantic_hash, world, seed,
        optimization=8, selection=2, final_evaluation=8, stress=4,
        planning_hash=getattr(case, "planning_hash", case.semantic_hash))
    config = TrainingConfig(seed=seed, max_environment_steps=2048,
        max_gradient_steps=32, max_wall_seconds=float(budget_seconds),
        max_episode_steps=24, episodes_per_update=2, checkpoint_interval=4)
    if not resume:
        (directory / "world-partitions.json").write_text(json.dumps(partitions.to_dict(), indent=2))
        (directory / "desktop-run.json").write_text(json.dumps({
            "case_hash": case.semantic_hash, "case_id": case.case_id, "budget_seconds": budget_seconds,
            "seed": seed, "block_size": block_size, "mode": "PATIENT_SCRATCH_RL",
            "scope": "coarse_annotation_assisted_structural_research",
            "interpretation": "deterministic optimization; different seeds are not anatomical uncertainty",
            "clinical_deficit_probability": None,
            "final_evaluation_used_for_selection": False}, indent=2))
    result = train_patient_policy(simulator.clone, partitions.optimization, partitions.selection,
        config=config, output_dir=directory, cancelled=cancelled, progress=progress, resume=resume)
    report = json.loads((directory / "result.json").read_text())
    report["preparation_and_training_seconds"] = time.perf_counter() - started
    report["case_hash"] = case.semantic_hash
    report["block_size"] = block_size
    report["replay"] = None
    if cancelled() or result.status == "cancelled":
        return report
    replay_started = time.perf_counter()
    policy = load_policy(directory / "checkpoint.pt")
    selected_seed = partitions.selection.seeds[0]
    replay = rollout_policy(policy, simulator.clone(), seed=selected_seed,
        max_steps=config.max_episode_steps, expected_model_hash=simulator.decision_model_hash,
        interrupt=cancelled)
    certificate = {"valid": False, "reason": "independent_sequence_checker_unavailable"}
    try:
        from resectionlab.evaluation import independent_check_sequence
        certificate = independent_check_sequence(simulator.clone, replay.actions, seed=selected_seed, cancelled=cancelled)
        if hasattr(certificate, "to_dict"):
            certificate = certificate.to_dict()
        elif not isinstance(certificate, dict):
            certificate = asdict(certificate)
    except ImportError:
        pass
    from resectionlab.evaluation import independent_native_removal_check
    native_audit = independent_native_removal_check(case, simulator.config, replay.metrics["history"])
    replay_report = {
        "role": "selection", "case_hash": case.semantic_hash,
        "decision_model_hash": simulator.decision_model_hash,
        "checkpoint_hash": result.selected_checkpoint_hash,
        "actions": list(replay.actions), "metrics": replay.metrics,
        "independent_certificate": certificate,
        "native_removal_audit": native_audit.to_dict() if hasattr(native_audit, "to_dict") else asdict(native_audit),
        "affine": simulator.config.affine.tolist(),
        "shape": list(simulator.config.tissue_mask.shape),
        "source_block_size": block_size,
        "interpretation": "coarse cell removal footprint with exact source-fraction volume accounting; not native-resolution removal certification",
        "final_evaluation": False,
    }
    replay_report["artifact_hash"] = replay_artifact_hash(replay_report)
    (directory / "selection-replay.json").write_text(json.dumps(replay_report, indent=2))
    report["replay"] = replay_report
    report["selection_replay_and_check_seconds"] = time.perf_counter() - replay_started
    return report


def replay_artifact_hash(replay):
    """Bind displayed actions, history, source grid and evaluator records."""
    record = {key: value for key, value in replay.items() if key != "artifact_hash"}
    return "sha256:" + hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def replay_mask(replay, step):
    """Accumulate only retained removal history, never route accessibility."""
    result = np.zeros(tuple(replay["shape"]), dtype=bool)
    for item in replay["metrics"]["history"][:step]:
        indices = np.asarray(item.get("removed_indices", []), dtype=int)
        if len(indices):
            result[tuple(indices.T)] = True
    return result


def validate_replay(case, replay):
    """Reject stale, malformed or unaccepted selection artifacts before display."""
    if replay.get("case_hash") != case.semantic_hash or replay.get("role") != "selection":
        raise ValueError("Replay is stale or does not belong to the selection partition")
    if not replay.get("independent_certificate", {}).get("feasible"):
        raise ValueError("Replay has not passed independent simulation-grid checks")
    if not replay.get("native_removal_audit", {}).get("feasible"):
        raise ValueError("Replay has not passed independent native removal checks")
    if replay.get("artifact_hash") != replay_artifact_hash(replay):
        raise ValueError("Replay artifact or evaluator records changed after creation")
    block = replay.get("source_block_size")
    if type(block) is not int or block < 1:
        raise ValueError("Replay needs an explicit source block size")
    expected_shape = tuple(int(np.ceil(n / block)) for n in case.mri.shape)
    if tuple(replay.get("shape", ())) != expected_shape:
        raise ValueError("Replay grid does not match source block derivation")
    expected = np.array(case.affine, copy=True)
    expected[:3, 3] += expected[:3, :3] @ np.full(3, (block - 1) / 2)
    expected[:3, :3] *= block
    affine = np.asarray(replay.get("affine"))
    if affine.shape != (4, 4) or not np.allclose(affine, expected, atol=1e-7, rtol=0):
        raise ValueError("Replay physical frame does not match source derivation")
    source_target = np.zeros(case.mri.shape, dtype=bool)
    for mask in case.compartments.values():
        source_target |= mask
    if case.brain_mask is not None:
        source_tissue = np.asarray(case.brain_mask) | source_target
    else:
        from scipy.ndimage import binary_fill_holes
        source_tissue = binary_fill_holes(case.mri != 0) | source_target
    seen = set()
    total_target, total_normal = 0., 0.
    for item in replay["metrics"]["history"]:
        indices = np.asarray(item.get("removed_indices", []))
        if not len(indices):
            continue
        if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= np.asarray(expected_shape)):
            raise ValueError("Replay contains invalid removal indices")
        expected_target, expected_normal = 0., 0.
        for point in indices:
            key = tuple(point)
            if key in seen:
                raise ValueError("Replay repeats an already removed coarse cell")
            seen.add(key)
            low = point * block
            high = np.minimum(low + block, case.mri.shape)
            region = tuple(slice(lo, hi) for lo, hi in zip(low, high))
            target_count = np.count_nonzero(source_target[region])
            tissue_count = np.count_nonzero(source_tissue[region])
            expected_target += target_count * case.voxel_volume_mm3
            expected_normal += (tissue_count - target_count) * case.voxel_volume_mm3
        for metric in ("target_removed_mm3", "normal_removed_mm3"):
            value = item.get(metric, 0.)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Replay volume increments must be finite and nonnegative")
        if not np.isclose(item.get("target_removed_mm3", 0.), expected_target, rtol=1e-7, atol=1e-6) or not np.isclose(item.get("normal_removed_mm3", 0.), expected_normal, rtol=1e-7, atol=1e-6):
            raise ValueError("Replay increments disagree with source tissue accounting")
        total_target += expected_target
        total_normal += expected_normal
    metrics = replay["metrics"]
    residual = np.count_nonzero(source_target) * case.voxel_volume_mm3 - total_target
    if not np.isclose(metrics.get("simulated_removed_target_volume_mm3", -1), total_target, rtol=1e-7, atol=1e-6) or not np.isclose(metrics.get("modeled_residual_target_volume_mm3", -1), residual, rtol=1e-7, atol=1e-6) or not np.isclose(metrics.get("simulated_removed_normal_volume_mm3", -1), total_normal, rtol=1e-7, atol=1e-6):
        raise ValueError("Replay totals disagree with source tissue accounting")
    if metrics.get("clinical_deficit_probability") is not None:
        raise ValueError("Clinical deficit probabilities are unsupported")
    return True


def replay_volumes(replay, step):
    """Read volume increments from real simulated transitions."""
    selected = replay["metrics"]["history"][:step]
    removed = sum(item.get("target_removed_mm3", 0.) for item in selected)
    normal = sum(item.get("normal_removed_mm3", 0.) for item in selected)
    original = replay["metrics"]["modeled_residual_target_volume_mm3"] + replay["metrics"]["simulated_removed_target_volume_mm3"]
    return {"removed_target_mm3": removed, "residual_target_mm3": original - removed, "removed_normal_mm3": normal}
