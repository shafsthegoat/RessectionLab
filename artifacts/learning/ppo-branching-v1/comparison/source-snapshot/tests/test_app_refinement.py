"""The desktop bridge must retain true updates, role separation and replay units."""

import json

import numpy as np

from resectionlab.app.refinement import replay_mask, replay_volumes, run_refinement
from resectionlab.imaging import create_synthetic_case


def test_replay_uses_history_increments_not_accessibility():
    replay = {"shape": [4, 4, 4], "metrics": {
        "history": [{"removed_indices": [[1, 1, 1]], "target_removed_mm3": 3., "normal_removed_mm3": 1.},
                    {"removed_indices": [[1, 1, 2]], "target_removed_mm3": 2., "normal_removed_mm3": 2.}],
        "simulated_removed_target_volume_mm3": 5., "modeled_residual_target_volume_mm3": 7.}}
    assert not replay_mask(replay, 0).any()
    assert replay_mask(replay, 1).sum() == 1
    assert replay_mask(replay, 2).sum() == 2
    assert replay_volumes(replay, 1) == {"removed_target_mm3": 3., "residual_target_mm3": 9., "removed_normal_mm3": 1.}


def test_refinement_performs_updates_without_final_world_feedback(tmp_path):
    case = create_synthetic_case((24, 24, 24))
    report = run_refinement(case, tmp_path / "run", budget_seconds=20, seed=11, block_size=3)
    assert report["gradient_steps"] > 0
    assert report["optimization_environment_steps"] > 0
    assert report["initial_checkpoint_hash"] != report["latest_checkpoint_hash"]
    assert report["replay"]["role"] == "selection"
    assert report["replay"]["final_evaluation"] is False
    partitions = json.loads((tmp_path / "run" / "world-partitions.json").read_text())
    all_seeds = [seed for item in partitions.values() for seed in item["seeds"]]
    assert len(all_seeds) == len(set(all_seeds))
    assert report["replay"]["metrics"]["seed"] in partitions["selection"]["seeds"]
    assert (tmp_path / "run" / "checkpoint.pt").is_file()
    assert not list((tmp_path / "run").glob("*final*result*"))
