"""Synthetic instrumentation for saved-argument reconstruction; no public data."""
from dataclasses import asdict
from pathlib import Path
import ast
import copy
import gzip
import importlib.util
import json
import math
import random
import struct
import sys

import numpy as np
import pytest

from resectionlab import geometry, native_resection as native
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reconstruct_queries_review", ROOT / "scripts/reconstruct_native_cache_queries.py")
reconstruct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruct)


@pytest.fixture
def saved_fixture(monkeypatch, tmp_path):
    tissue = np.zeros((9, 9, 9), bool)
    tissue[1:8, 1:8, 1:8] = True
    labels = np.zeros_like(tissue, dtype=np.int16)
    labels[3, 3, 3:6] = labels[5, 5, 3:6] = 1
    cfg = native.NativeResectionConfig(tissue, labels, np.eye(4),
        geometry.AccessWindow([3., 3., .5], [0., 0., 1.], 5.), (native.NATIVE_GENERIC_TOOLS[0],),
        "synthetic-key-reconstruction", "synthetic explicit cube")
    sim = AxisColumnNativeSimulator(cfg,
        proposal_config=AxisColumnProposalConfig(offsets_source_voxels=((0, 0), (2, 2)), max_primary_rays=2),
        max_steps=2)
    cache = ExactCapsuleCoverCache(cfg)
    observed, frames = [], []
    original = native.capsule_voxel_indices
    def instrumented(scene, start, end, radius):
        # Independent observation of original native arguments, before calling
        # the unchanged coverage function. No reconstruction helper is used.
        observed.append((np.asarray(start, dtype=np.float64).tobytes()
                         + np.asarray(end, dtype=np.float64).tobytes()
                         + np.asarray(radius, dtype=np.float64).tobytes()).hex())
        frames.append((scene.shape, scene.affine.tobytes(), scene._inverse.tobytes(),
                       scene._cell_radius_mm, scene._orthogonal_spacing.tobytes()))
        return original(scene, start, end, radius)
    monkeypatch.setattr(native, "capsule_voxel_indices", instrumented)
    snapshots = []
    def capture(observation):
        snapshots.append({"action_ids": list(observation.action_ids),
            "inventory": copy.deepcopy(sim._inventory_receipts[-1]),
            "cavity_state_hash": sim.engine.state_hash,
            "certificates": {action: json.dumps(sim._previews[action].to_history_record())
                             for action in observation.action_ids[1:]}})
    observation = sim.reset(11)
    capture(observation)
    while not sim.terminated:
        assert len(observation.action_ids) > 1
        observation = sim.step(observation.action_ids[1]).observation
        capture(observation)
    assert len(sim._history) == 2 and frames and len(set(frames)) == 1
    fixture = {"trace": {"snapshots": snapshots}, "tools": [asdict(tool) for tool in cfg.tools],
               "cache_metadata": cache.stats()}
    fixture_path = tmp_path / "saved-native-fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    observed_path = tmp_path / "independently-observed-arguments.json"
    observed_path.write_text(json.dumps(observed))
    return fixture_path, observed_path


def test_saved_native_certificates_reproduce_every_instrumented_argument_in_order(saved_fixture):
    fixture_path, observed_path = saved_fixture
    fixture = json.loads(fixture_path.read_text())
    observed = json.loads(observed_path.read_text())
    result = reconstruct.reconstruct(**fixture, expected_calls=len(observed))
    assert [query["argument_float64_le_hex"] for query in result["queries"]] == observed
    assert [query["kind"] for query in result["queries"]] == ["active", "shaft"] * (len(observed) // 2)
    assert sum(result["inventory_query_counts"]) == len(observed)
    assert result["binding"]["source_hash"] == "synthetic-key-reconstruction"
    assert result["binding"]["omitted_invariant_cache_frame_fields"]
    assert len({row["key_projection_hash"] for row in result["queries"]}) == len(set(observed))


@pytest.mark.parametrize("change", ["rejected", "missing_certificate", "attempt_order", "source", "frame", "microstep", "radius"])
def test_incomplete_or_mismatched_saved_evidence_rejects(saved_fixture, change):
    fixture = json.loads(saved_fixture[0].read_text())
    first = fixture["trace"]["snapshots"][0]
    action = first["action_ids"][1]
    cert = json.loads(first["certificates"][action])
    if change == "rejected":
        first["inventory"]["attempts"][0]["feasible"] = False
    elif change == "missing_certificate":
        first["certificates"].pop(action)
    elif change == "attempt_order":
        first["inventory"]["attempts"].reverse()
    elif change == "source":
        cert["source_hash"] = "foreign-case"
    elif change == "frame":
        second = first["action_ids"][2]
        cert = json.loads(first["certificates"][second])
        cert["native_affine"][0][3] = 10.
        action = second
    elif change == "microstep":
        cert["microsteps"][1]["tip_start_mm"][0] += .125
    else:
        cert["microsteps"][0]["active_radius_mm"] += .25
    if change not in ("rejected", "missing_certificate", "attempt_order"):
        first["certificates"][action] = json.dumps(cert)
    with pytest.raises(ValueError):
        reconstruct.reconstruct(**fixture)


def test_denominator_and_query_cap_fail_closed(saved_fixture):
    fixture = json.loads(saved_fixture[0].read_text())
    observed = json.loads(saved_fixture[1].read_text())
    with pytest.raises(ValueError, match="count"):
        reconstruct.reconstruct(**fixture, expected_calls=len(observed) + 2)
    with pytest.raises(ValueError, match="cap"):
        reconstruct.reconstruct(**fixture, maximum_queries=1)


def test_float_arguments_preserve_signed_zero_ulp_and_subtraction_order():
    for value in (0., -0., 1., math.nextafter(1., 2.), -181.5):
        assert reconstruct.float_bytes([value]) == np.asarray([value], dtype=np.float64).tobytes()
    assert reconstruct.float_bytes([0.]) != reconstruct.float_bytes([-0.])
    assert reconstruct.float_bytes([1.]) != reconstruct.float_bytes([math.nextafter(1., 2.)])
    randomizer = random.Random(91)
    for _ in range(100):
        point = [randomizer.uniform(-200., 200.) for _ in range(3)]
        axis = [randomizer.uniform(-1., 1.) for _ in range(3)]
        scale = randomizer.choice([2., 3., 120.])
        actual = np.asarray(point, dtype=np.float64) - scale * np.asarray(axis, dtype=np.float64)
        assert reconstruct.float_bytes(reconstruct.subtract_scaled(point, scale, axis)) == actual.tobytes()


def test_reuse_distances_match_independent_brute_force_definition():
    randomizer = random.Random(12)
    for keys in (["a", "b", "a", "c", "a", "b"], ["a"] * 10,
                 [str(randomizer.randrange(17)) for _ in range(200)]):
        expected, seen = [], {}
        for index, key in enumerate(keys):
            expected.append(None if key not in seen else len(set(keys[seen[key] + 1:index])))
            seen[key] = index
        assert reconstruct.reuse_distances(keys) == expected
        for cap in (0, 1, 2, 8, 17, 256):
            result = reconstruct.entry_only_lru(keys, cap)
            assert result["hits"] == sum(value is not None and value < cap for value in expected)


def test_summary_keeps_payload_unknown_and_does_not_count_hash_as_identity(saved_fixture):
    fixture = json.loads(saved_fixture[0].read_text())
    result = reconstruct.reconstruct(**fixture)
    for query in result["queries"]:
        query["key_projection_hash"] = "deliberate-identical-hash"
    summary = reconstruct.summarize(result, (0, 16384))
    exact_keys = [row["argument_float64_le_hex"] for row in result["queries"]]
    assert summary["distinct_argument_keys"] == len(set(exact_keys)) > 1
    assert summary["phases"][1]["compulsory_misses"] == 0
    assert summary["exact_total_cover_payload_bytes"] is None
    assert summary["payload_weighted_reuse_distances"] is None
    assert "same no-bypass admission policy" in summary["interpretation"]
    assert summary["entry_only_lru_unlimited_payload"]["16384"]["repeated_increment"]["hits"] == len(exact_keys)


def test_diagnostic_has_only_standard_library_imports():
    tree = ast.parse((ROOT / "scripts/reconstruct_native_cache_queries.py").read_text())
    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dependencies.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            dependencies.add(node.module.split(".")[0])
    assert dependencies <= sys.stdlib_module_names


def test_default_cli_does_not_open_inputs_and_refuses_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(reconstruct, "ROOT", tmp_path)
    declaration = {"input_sha256": {"nonexistent-public-input": "never-read"}}
    declaration["declaration_content_hash"] = reconstruct.digest(declaration)
    path = tmp_path / reconstruct.DECLARATION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(declaration))
    output = tmp_path / "declaration-only"
    monkeypatch.setattr(sys, "argv", ["reconstruct", "--output", str(output)])
    def forbidden(*args):
        raise AssertionError("default declaration must not hash or open any input")
    monkeypatch.setattr(reconstruct, "file_hash", forbidden)
    reconstruct.main()
    assert json.loads((output / "status.json").read_text())["status"] == "declared_not_executed"
    with pytest.raises(FileExistsError):
        reconstruct.main()


def test_executed_synthetic_saved_fixture_never_calls_geometry(monkeypatch, tmp_path, saved_fixture):
    fixture = json.loads(saved_fixture[0].read_text())
    observed = json.loads(saved_fixture[1].read_text())
    root = tmp_path / "synthetic-only-cli"
    run = root / "synthetic-run"
    run.mkdir(parents=True)
    def write(name, value):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    runtime = "synthetic-frozen-runtime"
    result = {"status": "completed", "runtime_content_hash": runtime}
    write("result.json", result)
    write("worker-status.json", {"status": "completed", "result_hash": reconstruct.digest(result),
                                 "runtime_content_hash": runtime})
    write("launcher-status.json", {"status": "completed"})
    source_binding = {"synthetic-frozen-callsite": "synthetic-source-hash"}
    write("launch-source.json", {"file_sha256": source_binding, "runtime_content_hash": runtime})
    write("preparation.json", {"native_configuration": {"actual": {"components": {"tools": fixture["tools"]}}}})
    count = len(observed)
    cache = fixture["cache_metadata"]
    write("cached_cold/receipt.json", {"cache_before": {**cache, "calls": 0}, "cache_after": {**cache, "calls": count}})
    write("cached_warm/receipt.json", {"cache_before": {**cache, "calls": count}, "cache_after": {**cache, "calls": 2 * count}})
    trace = run / "reference_before/scientific-trace.json.gz"
    trace.parent.mkdir()
    trace.write_bytes(gzip.compress(json.dumps(fixture["trace"]).encode(), mtime=0))
    declaration = {"input_run": "synthetic-run", "input_sha256": {
        str(path.relative_to(root)): reconstruct.file_hash(path) for path in run.rglob("*") if path.is_file()},
        "frozen_native_source_sha256": source_binding, "expected_queries_per_phase": count,
        "source_runtime_content_hash": runtime,
        "maximum_queries": count, "maximum_uncompressed_trace_bytes": 1024**2,
        "maximum_wall_seconds": 60, "maximum_process_peak_rss_bytes": 512 * 1024**2,
        "entry_capacities": [0, 4096]}
    declaration["declaration_content_hash"] = reconstruct.digest(declaration)
    path = root / reconstruct.DECLARATION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(declaration))
    monkeypatch.setattr(reconstruct, "ROOT", root)
    output = root / "output"
    monkeypatch.setattr(sys, "argv", ["reconstruct", "--output", str(output), "--execute"])
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-artifact reconstruction cannot call cover geometry")
    monkeypatch.setattr(geometry, "capsule_voxel_indices", forbidden)
    monkeypatch.setattr(native, "capsule_voxel_indices", forbidden)
    reconstruct.main()
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["queries_per_phase"] == count and summary["cover_geometry_calls"] == 0
    with gzip.open(output / "ordered-query-arguments.json.gz", "rt") as stream:
        queries = json.load(stream)["queries"]
    assert [row["argument_float64_le_hex"] for row in queries] == observed
