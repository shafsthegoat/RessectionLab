"""Headless bridge from the desktop to bounded, real policy-gradient updates.

The viewer consumes selection-set replay only. It never reads the final
evaluation partition while training, selecting checkpoints or preparing replay.
"""

from dataclasses import asdict
import json
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
        certificate = independent_check_sequence(simulator.clone, replay.actions, seed=selected_seed)
        if hasattr(certificate, "to_dict"):
            certificate = certificate.to_dict()
        elif not isinstance(certificate, dict):
            certificate = asdict(certificate)
    except ImportError:
        pass
    replay_report = {
        "role": "selection", "case_hash": case.semantic_hash,
        "decision_model_hash": simulator.decision_model_hash,
        "checkpoint_hash": result.selected_checkpoint_hash,
        "actions": list(replay.actions), "metrics": replay.metrics,
        "independent_certificate": certificate,
        "affine": simulator.config.affine.tolist(),
        "shape": list(simulator.config.tissue_mask.shape),
        "source_block_size": block_size,
        "interpretation": "coarse cell removal footprint with exact source-fraction volume accounting; not native-resolution removal certification",
        "final_evaluation": False,
    }
    (directory / "selection-replay.json").write_text(json.dumps(replay_report, indent=2))
    report["replay"] = replay_report
    report["selection_replay_and_check_seconds"] = time.perf_counter() - replay_started
    return report


def replay_mask(replay, step):
    """Accumulate only retained removal history, never route accessibility."""
    result = np.zeros(tuple(replay["shape"]), dtype=bool)
    for item in replay["metrics"]["history"][:step]:
        indices = np.asarray(item.get("removed_indices", []), dtype=int)
        if len(indices):
            result[tuple(indices.T)] = True
    return result


def replay_volumes(replay, step):
    """Read volume increments from real simulated transitions."""
    selected = replay["metrics"]["history"][:step]
    removed = sum(item.get("target_removed_mm3", 0.) for item in selected)
    normal = sum(item.get("normal_removed_mm3", 0.) for item in selected)
    original = replay["metrics"]["modeled_residual_target_volume_mm3"] + replay["metrics"]["simulated_removed_target_volume_mm3"]
    return {"removed_target_mm3": removed, "residual_target_mm3": original - removed, "removed_normal_mm3": normal}
