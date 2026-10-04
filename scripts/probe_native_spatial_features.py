#!/usr/bin/env python3
"""One tiny synthetic descriptor probe. No policy, optimizer or patient input."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np

from research.native_spatial_features import (access_source_frame, candidate_coordinates,
    cavity_residual_summary, descriptor_contract)
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NativeResectionConfig, NATIVE_GENERIC_TOOLS


def fixture():
    tissue = np.ones((7, 7, 8), dtype=bool)
    target = np.zeros(tissue.shape, np.int16)
    target[1:6, 1:6, 2:7] = 1
    cfg = NativeResectionConfig(tissue, target, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        "synthetic-axis-learner-fixture-v1", "explicit synthetic support", case_id="synthetic_axis_learner")
    return AxisColumnNativeSimulator(cfg,
        proposal_config=AxisColumnProposalConfig(((0, 0), (1, 0)), 4), max_steps=2)


def coordinates(sim):
    ids = sim.observation().action_ids
    entries = [sim.native_config.access.center_mm if action == "STOP" else sim._action_provenance[action]["entry_mm"] for action in ids]
    tips = [sim.native_config.access.center_mm if action == "STOP" else sim._action_provenance[action]["tip_mm"] for action in ids]
    return ids, np.asarray(entries), np.asarray(tips)


def source_receipt():
    paths = [*ROOT.glob("src/resectionlab/*.py"), ROOT / "research/native_spatial_features.py", Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    return {"files": hashes, "python": sys.version, "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}}


def run_probe():
    before = source_receipt()
    started = time.perf_counter()
    sim = fixture()
    cfg = sim.native_config
    frame = access_source_frame(cfg.affine, cfg.access.center_mm, cfg.access.normal_inward)
    ids, entries, tips = coordinates(sim)
    spatial = candidate_coordinates(ids, entries, tips, frame)
    raw = sim.observation().action_features
    assert np.array_equal(raw[1], raw[3])
    assert not np.array_equal(spatial[1], spatial[3])
    branches, transitions = [], 0
    for index in (1, 3):
        branch = sim.clone()
        first = branch.step(ids[index]); transitions += 1
        continuations = []
        for action in branch.observation().action_ids:
            last = branch.clone().step(action); transitions += 1
            assert last.terminated
            continuations.append({"action_id": action, "second_reward": last.reward,
                                  "two_step_return": first.reward + last.reward})
        summary = cavity_residual_summary(cfg.tissue_mask, cfg.target_labels > 0,
                                         branch.engine.removed_mask, cfg.affine, frame)
        branches.append({"action_id": ids[index], "entry_mm": entries[index].tolist(),
            "tip_mm": tips[index].tolist(), "raw_action_features": raw[index].tolist(),
            "spatial_action_features": spatial[index].tolist(), "first_reward": first.reward,
            "existing_six_state_features_after_cut": branch.observation().state_features.tolist(),
            "proposed_eight_state_features_after_cut": summary.tolist(),
            "best_two_step_return": max(row["two_step_return"] for row in continuations),
            "all_one_step_continuations": continuations})
    # Algebraic checks use the same saved geometry; they do not replay another world.
    angle = .731
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    transforms = {"translation": (np.eye(3), np.asarray((71., -18., 3.))),
                  "proper_rotation_and_translation": (rotation, np.asarray((9., 17., -24.))),
                  "RAS_to_LPS": (np.diag((-1., -1., 1.)), np.zeros(3))}
    removed = sim.clone(); removed.step(ids[1]); transitions += 1
    original_summary = cavity_residual_summary(cfg.tissue_mask, cfg.target_labels > 0,
                                               removed.engine.removed_mask, cfg.affine, frame)
    errors = {}
    for name, (matrix, shift) in transforms.items():
        transform = np.eye(4); transform[:3, :3] = matrix; transform[:3, 3] = shift
        affine = transform @ cfg.affine
        transformed_frame = access_source_frame(affine,
            matrix @ np.asarray(cfg.access.center_mm) + shift, matrix @ np.asarray(cfg.access.normal_inward))
        candidate = candidate_coordinates(ids, entries @ matrix.T + shift, tips @ matrix.T + shift, transformed_frame)
        state = cavity_residual_summary(cfg.tissue_mask, cfg.target_labels > 0,
                                        removed.engine.removed_mask, affine, transformed_frame)
        errors[name] = {"candidate_max_abs_error": float(np.max(np.abs(candidate-spatial))),
                       "state_max_abs_error": float(np.max(np.abs(state-original_summary)))}
        assert errors[name]["candidate_max_abs_error"] < 1e-10
        assert errors[name]["state_max_abs_error"] < 1e-10
    order = np.asarray((3, 0, 4, 1, 2))
    permuted = candidate_coordinates(tuple(ids[i] for i in order), entries[order], tips[order], frame)
    assert np.array_equal(permuted, spatial[order])
    # Explicit summary collision: different abstract occupancy patterns, same
    # volume and first moment. Their native reachability is not asserted.
    a, b = np.zeros(cfg.tissue_mask.shape, bool), np.zeros(cfg.tissue_mask.shape, bool)
    a[2:4, 2:4, 3] = True
    b[1, 2:4, 3] = True; b[4, 2:4, 3] = True
    sa = cavity_residual_summary(cfg.tissue_mask, cfg.target_labels > 0, a, cfg.affine, frame)
    sb = cavity_residual_summary(cfg.tissue_mask, cfg.target_labels > 0, b, cfg.affine, frame)
    assert np.array_equal(sa, sb) and not np.array_equal(a, b)
    # The source-axis order is part of the declaration, not an invariant gauge.
    reindexed_affine = cfg.affine.copy(); reindexed_affine[:3, [0, 1]] = reindexed_affine[:3, [1, 0]]
    reindexed_frame = access_source_frame(reindexed_affine, cfg.access.center_mm, cfg.access.normal_inward)
    reordered = candidate_coordinates(ids, entries, tips, reindexed_frame)
    assert not np.array_equal(reordered, spatial)
    after = source_receipt()
    if before != after:
        raise RuntimeError("Source/runtime changed during the synthetic probe")
    return {"status": "completed_synthetic_descriptor_probe", "contract": descriptor_contract(),
        "source_receipt": before, "case_hash": sim.case_hash,
        "decision_model_hash": sim.decision_model_hash, "native_config_hash": cfg.fingerprint,
        "frame": frame.to_dict(), "initial_action_ids": list(ids), "branches": branches,
        "raw_actor_alias_equal": True, "spatial_actor_alias_equal": False,
        "old_post_cut_state_equal": branches[0]["existing_six_state_features_after_cut"] == branches[1]["existing_six_state_features_after_cut"],
        "proposed_post_cut_state_equal": branches[0]["proposed_eight_state_features_after_cut"] == branches[1]["proposed_eight_state_features_after_cut"],
        "coordinate_transform_errors": errors, "row_permutation_max_abs_error": float(np.max(np.abs(permuted-spatial[order]))),
        "summary_counterexample": {"cavity_a_indices": np.argwhere(a).tolist(), "cavity_b_indices": np.argwhere(b).tolist(),
            "equal_summary": sa.tolist(), "reachable_native_cavities_verified": False},
        "source_axis_reindexing_counterexample": {"max_abs_change": float(np.max(np.abs(reordered-spatial))),
            "interpretation": "Holding physical points fixed while changing declared source-axis order changes the tangent basis. No source-reindex invariance claimed."},
        "native_transitions_executed": transitions, "gradients": 0, "optimizers_constructed": 0,
        "patients_loaded": 0, "policies_evaluated": 0, "elapsed_seconds": time.perf_counter()-started,
        "limits": ["No trained policy or efficacy test", "No production15+6 input or adapter change",
            "Coordinate checks test descriptor algebra, not transformed whole-engine geometry",
            "Raw physical units may still require a separately declared scaling study",
            "Counts and centroids omit geometry, topology, connectivity and candidate interactions",
            "Finite fixture distinction does not establish generalization or Markov completeness"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = run_probe()
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "native_transitions_executed", "gradients", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
