"""Prospective runner checks with tiny synthetic tissue; no public case load."""
from contextlib import nullcontext
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_public_native_capsule_cache as probe
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache, inject_native_capsule_cache
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NativeResectionConfig, NATIVE_GENERIC_TOOLS


def tiny_simulator():
    tissue = np.zeros((7, 7, 7), bool)
    tissue[1:6, 1:6, 1:6] = True
    labels = np.zeros_like(tissue, dtype=np.int16)
    labels[3, 3, 3:5] = 1
    cfg = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow([3, 3, .5], [0, 0, 1], 4), (NATIVE_GENERIC_TOOLS[0],),
        "synthetic-runner-check", "synthetic explicit cube")
    return AxisColumnNativeSimulator(cfg,
        proposal_config=AxisColumnProposalConfig(offsets_source_voxels=((0, 0),), max_primary_rays=1),
        max_steps=1)


def test_prospective_declaration_and_complete_source_closure_are_bound():
    value = probe.declaration()
    probe._check_bound_inputs(value)
    snapshot = probe.source_snapshot()
    for path in ("src/resectionlab/native_proposals.py", "src/resectionlab/experimental_capsule_cache.py",
                 "src/resectionlab/native_axis_simulation.py", "scripts/preflight_native_axis.py",
                 "scripts/probe_public_native_capsule_cache.py", str(probe.DECLARATION_PATH)):
        assert path in snapshot["file_sha256"]
    assert set(value["bound_input_file_sha256"]) <= set(snapshot["file_sha256"])
    budget = value["resource_budget"]
    assert budget["worker_wall_seconds"] + budget["hard_termination_grace_seconds"] <= 600
    assert budget["process_peak_rss_bytes"] == 6 * 1024**3
    assert value["fixed_work"]["gradient_updates"] == 0


def test_cli_without_execute_only_declares_and_refuses_existing_output(tmp_path):
    output = tmp_path / "declaration-only"
    command = [sys.executable, str(ROOT / "scripts/probe_public_native_capsule_cache.py"),
               "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    assert json.loads((output / "launcher-status.json").read_text())["status"] == "declared_not_executed"
    assert {path.name for path in output.iterdir()} == {"declaration.json", "launcher-status.json"}
    second = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr


def test_scientific_filters_remove_only_explicit_timing_paths():
    inventory = {"attempts": [{"elapsed_seconds": 1., "reason": "PASS", "tip_mm": [1, 2, 3]}],
                 "scientific_elapsed_seconds": 7., "batch": {"source_hash": "source"}}
    metrics = {"initialization_timing": {"seconds": 2},
        "proposal_accounting": {"integrity_seconds": 1., "preview_seconds": 2., "preview_calls": 4},
        "inventory_receipts": [inventory], "proposal_failures": [{"elapsed_seconds": 3., "reason": "BLOCKED"}],
        "history": [{"elapsed_seconds": "this unknown field must remain"}], "source_hash": "source"}
    result = probe.scientific_metrics(metrics)
    assert "initialization_timing" not in result
    assert result["proposal_accounting"] == {"preview_calls": 4}
    assert result["inventory_receipts"][0]["attempts"][0] == {"reason": "PASS", "tip_mm": [1, 2, 3]}
    assert result["inventory_receipts"][0]["scientific_elapsed_seconds"] == 7.
    assert result["history"] == metrics["history"] and result["source_hash"] == "source"
    assert metrics["proposal_accounting"]["preview_seconds"] == 2.


def test_measurement_wrapper_counts_exceptions_rejects_nesting_and_restores():
    class Proposer:
        def propose(self, value):
            if value == "fail":
                raise LookupError("intentional")
            return value
    proposer = Proposer()
    with probe.measure_proposer_calls(proposer) as counters:
        assert proposer.propose(7) == 7
        with pytest.raises(RuntimeError):
            with probe.measure_proposer_calls(proposer):
                pass
        with pytest.raises(LookupError):
            proposer.propose("fail")
    assert counters["calls"] == 2 and counters["seconds"] >= 0
    assert "propose" not in proposer.__dict__ and proposer.propose(8) == 8


def test_fixed_trace_four_modes_preserve_certificates_observations_masks_and_rewards(tmp_path):
    sim = tiny_simulator()
    actions = (sim.proposed_actions()[1].action_id,)
    cache = ExactCapsuleCoverCache(sim.native_config)
    guard = probe.pre.ResourceGuard(probe.pre.PreflightBudget(worker_wall_seconds=60.))
    expected_text = expected_masks = None
    for mode in probe.MODES:
        with inject_native_capsule_cache(cache) if mode.startswith("cached") else nullcontext():
            trace, masks, timing = probe.run_fixed_trace(sim, actions, 13, guard, tmp_path / mode)
        actual = probe._canonical(trace)
        if expected_text is None:
            expected_text, expected_masks = actual, masks
        else:
            assert actual == expected_text and masks == expected_masks
        assert len(trace["snapshots"]) == 2 and len(trace["snapshots"][0]["certificates"]) == 1
        assert timing["full_proposal_verification"]["calls"] > timing["adapter_proposal_accounting"]["integrity_calls"]
        assert "propose" not in sim._proposer.__dict__
    assert cache.stats()["hits"] > 0


def test_cancelled_trace_retains_failure_and_restores_measurement_hook(tmp_path):
    sim = tiny_simulator()
    actions = (sim.proposed_actions()[1].action_id,)
    guard = probe.pre.ResourceGuard(probe.pre.PreflightBudget(worker_wall_seconds=60.))
    guard.cancel("synthetic_cancellation")
    before = sim.engine.state_hash
    with pytest.raises(InterruptedError):
        probe.run_fixed_trace(sim, actions, 13, guard, tmp_path / "cancelled")
    assert sim.engine.state_hash == before
    assert "propose" not in sim._proposer.__dict__
    assert (tmp_path / "cancelled/failure.json").is_file()


def test_gzip_scientific_export_roundtrip_is_deterministic(tmp_path):
    value = {"certificates": {"action": "exact scientific record"}, "reward": 3.2}
    first = probe._write_gzip(tmp_path / "one.gz", value)
    second = probe._write_gzip(tmp_path / "two.gz", value)
    assert first == second
    assert probe.gzip.decompress((tmp_path / "one.gz").read_bytes()).decode() == probe._canonical(value)
