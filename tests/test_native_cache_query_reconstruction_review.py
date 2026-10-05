"""Independent saved-query audit: synthetic JSON only, no native execution."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reconstruct_native_cache_queries.py"
SPEC = importlib.util.spec_from_file_location("independent_query_reconstruction", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
SYNTHETIC = ROOT / "artifacts/native-cache-query-review-v1/synthetic-instrumented-fixture.json.gz"


@pytest.fixture
def saved():
    # The independent observer already saved these bytes from a tiny phantom.
    # Loading this fixture does not construct any simulator or geometry object.
    return json.loads(gzip.decompress(SYNTHETIC.read_bytes()))


def edit_certificate(fixture, change, *, inventory=0, row=1):
    snapshot = fixture["trace"]["snapshots"][inventory]
    action = snapshot["action_ids"][row]
    cert = json.loads(snapshot["certificates"][action])
    change(cert)
    snapshot["certificates"][action] = json.dumps(cert)


def private_cli(tmp_path, monkeypatch, saved, *, runtime_mismatch=None):
    root = tmp_path / "private-synthetic-inputs"
    run = root / "saved"
    run.mkdir(parents=True)
    fixture = saved["fixture"]
    runtime = "sha256:synthetic-frozen-runtime"

    def write(name, value):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    result = {"status": "completed", "runtime_content_hash": runtime}
    source = {"file_sha256": {"synthetic-callsite": "synthetic-source"}, "runtime_content_hash": runtime}
    if runtime_mismatch == "source":
        source["runtime_content_hash"] = "sha256:foreign-runtime"
    if runtime_mismatch == "result":
        result["runtime_content_hash"] = "sha256:foreign-runtime"
    write("result.json", result)
    write("launch-source.json", source)
    write("worker-status.json", {"status": "completed", "result_hash": MOD.digest(result),
        "runtime_content_hash": "sha256:foreign-runtime" if runtime_mismatch == "worker" else runtime})
    write("launcher-status.json", {"status": "completed", "worker_returncode": 0,
        "parent_timeout_requested": False, "hard_killed": False})
    write("preparation.json", {"native_configuration": {"actual": {"components": {"tools": fixture["tools"]}}}})
    count = len(saved["independently_observed_argument_hex"])
    cache = fixture["cache_metadata"]
    write("cached_cold/receipt.json", {"cache_before": {**cache, "calls": 0}, "cache_after": {**cache, "calls": count}})
    write("cached_warm/receipt.json", {"cache_before": {**cache, "calls": count}, "cache_after": {**cache, "calls": 2 * count}})
    trace = run / "reference_before/scientific-trace.json.gz"
    trace.parent.mkdir()
    trace.write_bytes(gzip.compress(json.dumps(fixture["trace"]).encode(), mtime=0))
    declaration = {"input_run": "saved", "input_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*") if path.is_file()},
        "source_runtime_content_hash": runtime,
        "frozen_native_source_sha256": source["file_sha256"], "expected_queries_per_phase": count,
        "maximum_queries": count, "maximum_uncompressed_trace_bytes": 1024**2,
        "maximum_wall_seconds": 60, "maximum_process_peak_rss_bytes": 512 * 1024**2,
        "entry_capacities": [0, 32, 4096]}
    declaration["declaration_content_hash"] = MOD.digest(declaration)
    path = root / MOD.DECLARATION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(declaration))
    output = root / "output"
    monkeypatch.setattr(MOD, "ROOT", root)
    # These in-process tests target one guard at a time. Lifetime pytest RSS
    # includes unrelated earlier imaging/model tests, unlike the isolated CLI.
    # Give only this module a private resource facade; the explicit RSS-negative
    # test overrides it and the owner suite exercises the actual child process.
    units = 1 if sys.platform == "darwin" else 1024
    monkeypatch.setattr(MOD, "resource", SimpleNamespace(RUSAGE_SELF=resource.RUSAGE_SELF,
        getrusage=lambda *args: SimpleNamespace(ru_maxrss=(64 * 1024**2)//units)))
    monkeypatch.setattr(sys, "argv", ["reconstruct", "--output", str(output), "--execute"])
    return root, output, declaration


def test_saved_instrumentation_matches_every_byte_and_query_label(saved):
    result = MOD.reconstruct(**saved["fixture"], expected_calls=114)
    assert [q["argument_float64_le_hex"] for q in result["queries"]] == saved["independently_observed_argument_hex"]
    assert [q["index"] for q in result["queries"]] == list(range(114))
    assert result["inventory_query_counts"] == [76, 38, 0]
    assert [q["kind"] for q in result["queries"]] == ["active", "shaft"] * 57
    for i in range(0, 114, 2):
        left, right = result["queries"][i:i+2]
        assert all(left[k] == right[k] for k in ("inventory_index", "action_id", "tool_id", "microstep_index"))
        assert len(bytes.fromhex(left["argument_float64_le_hex"])) == 7 * 8


def test_certificate_mapping_insertion_order_cannot_reorder_queries(saved):
    fixture = saved["fixture"]
    expected = MOD.reconstruct(**fixture)
    for snapshot in fixture["trace"]["snapshots"]:
        snapshot["certificates"] = dict(reversed(tuple(snapshot["certificates"].items())))
    assert MOD.reconstruct(**fixture) == expected


def test_duplicate_action_id_is_not_a_second_measured_attempt(saved):
    fixture = saved["fixture"]
    snapshot = fixture["trace"]["snapshots"][0]
    action = snapshot["action_ids"][1]
    snapshot["action_ids"].append(action)
    snapshot["inventory"]["certified_action_ids"].append(action)
    snapshot["inventory"]["attempts"].append(copy.deepcopy(snapshot["inventory"]["attempts"][0]))
    with pytest.raises(ValueError, match="(?i)(unique|duplicate|complete)"):
        MOD.reconstruct(**fixture)


@pytest.mark.parametrize("field", ["tip_mm", "entry_mm"])
def test_one_ulp_endpoint_change_is_rejected_against_attempt(saved, field):
    def change(cert):
        cert[field][0] = math.nextafter(cert[field][0], math.inf)
    edit_certificate(saved["fixture"], change)
    with pytest.raises(ValueError, match="bound"):
        MOD.reconstruct(**saved["fixture"])


@pytest.mark.parametrize("change", ["shape", "affine_ulp", "affine_zero_sign", "axis", "tool", "target", "step_order", "active_end", "shaft_length"])
def test_frame_tool_and_microstep_mismatch_fails_closed(saved, change):
    fixture = saved["fixture"]
    def mutate(cert):
        if change == "shape":
            cert["source_shape"][0] += 1
        elif change == "affine_ulp":
            cert["native_affine"][0][0] = math.nextafter(1., math.inf)
        elif change == "affine_zero_sign":
            cert["native_affine"][0][1] = -0.
        elif change == "axis":
            cert["axis_unit"][0] = .1
        elif change == "tool":
            cert["tool_id"] = "foreign-tool"
        elif change == "target":
            cert["microsteps"][-1]["tip_end_mm"][2] += .125
            cert["microsteps"][-1]["active_stroke_end_mm"][2] += .125
        elif change == "step_order":
            cert["microsteps"][1:3] = reversed(cert["microsteps"][1:3])
        elif change == "active_end":
            cert["microsteps"][1]["active_stroke_end_mm"][1] += .125
    if change == "shaft_length":
        # This dimension is not directly repeated in saved certificates; its
        # provenance comes from the hash-bound preparation model. Changing it
        # demonstrably changes only the reconstructed shaft arguments.
        original = MOD.reconstruct(**fixture)
        fixture["tools"][0]["working_length_mm"] += .25
        changed = MOD.reconstruct(**fixture)
        for before, after in zip(original["queries"], changed["queries"]):
            assert (before["argument_float64_le_hex"] == after["argument_float64_le_hex"]) == (before["kind"] == "active")
        return
    edit_certificate(fixture, mutate, row=2 if change in ("shape", "affine_ulp", "affine_zero_sign") else 1)
    with pytest.raises((ValueError, KeyError)):
        MOD.reconstruct(**fixture)


def test_projection_binds_common_frame_without_fabricating_derived_bytes(saved):
    fixture = saved["fixture"]
    before = MOD.reconstruct(**fixture)
    for snapshot in fixture["trace"]["snapshots"]:
        for action, payload in snapshot["certificates"].items():
            cert = json.loads(payload)
            cert["native_affine"][0][1] = -0.
            snapshot["certificates"][action] = json.dumps(cert)
    after = MOD.reconstruct(**fixture)
    assert before["binding_hash"] != after["binding_hash"]
    assert [q["argument_float64_le_hex"] for q in before["queries"]] == [q["argument_float64_le_hex"] for q in after["queries"]]
    assert all(a["key_projection_hash"] != b["key_projection_hash"] for a, b in zip(before["queries"], after["queries"]))
    omitted = after["binding"]["omitted_invariant_cache_frame_fields"]
    assert set(omitted) == {"inverse_affine_bytes", "cell_radius_mm", "orthogonal_spacing_bytes"}
    assert all(name not in after["binding"] for name in omitted)
    assert MOD.summarize(before) == MOD.summarize(after)


def test_float64_bit_identity_is_not_python_numeric_equality():
    assert 0. == -0.
    assert MOD.float_bytes([0.]).hex() == "0000000000000000"
    assert MOD.float_bytes([-0.]).hex() == "0000000000000080"
    assert MOD.float_bytes([1.]).hex() == "000000000000f03f"
    assert MOD.float_bytes([math.nextafter(1., math.inf)]).hex() == "010000000000f03f"
    assert MOD.float_bytes([math.nextafter(0., 1.)]).hex() == "0100000000000000"
    assert not MOD.same_vector([0., 1., 2.], [-0., 1., 2.])


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1.0"])
def test_nonfinite_or_non_numeric_arguments_reject(value):
    with pytest.raises(ValueError):
        MOD.float_bytes([value])


def brute_reuse(keys):
    result = []
    for index, key in enumerate(keys):
        earlier = [i for i in range(index) if keys[i] == key]
        result.append(None if not earlier else len(set(keys[earlier[-1]+1:index])))
    return result


def test_exhaustive_short_sequences_reuse_and_lru_agree_with_definition():
    for length in range(1, 7):
        for keys in itertools.product("abc", repeat=length):
            distance = brute_reuse(keys)
            assert MOD.reuse_distances(keys) == distance
            for capacity in range(5):
                actual = MOD.entry_only_lru(keys, capacity)
                expected_hits = sum(d is not None and d < capacity for d in distance)
                assert actual["hits"] == expected_hits
                assert actual["misses"] == length - expected_hits
                assert actual["entries"] == min(len(set(keys)), capacity)
                assert actual["evictions"] == (max(0, actual["misses"] - capacity) if capacity else 0)


@pytest.mark.parametrize("capacity", [-1, 1.5, True])
def test_non_integer_entry_capacities_reject(capacity):
    with pytest.raises(ValueError):
        MOD.entry_only_lru(["a"], capacity)


def test_repeated_pass_boundary_and_hash_collision_keep_payload_unknown():
    keys = ["a", "b", "a", "c", "a", "b"]
    result = MOD.summarize({"queries": [{"argument_float64_le_hex": k, "key_projection_hash": "collision"} for k in keys]}, (0, 1, 2, 3))
    assert result["distinct_argument_keys"] == 3
    assert result["within_first_phase_repeat_count"] == 3
    assert result["phases"][0]["compulsory_misses"] == 3
    assert result["phases"][1]["compulsory_misses"] == 0
    assert result["entry_capacity_for_all_repeat_accesses_to_hit_without_payload_limit"] == 3
    assert result["entry_only_lru_unlimited_payload"]["3"]["repeated_increment"] == {"hits": 6, "misses": 0, "evictions": 0}
    assert result["exact_total_cover_payload_bytes"] is None
    assert result["payload_weighted_reuse_distances"] is None
    assert "upper bounds" in result["interpretation"]
    assert "shaft" in result["payload_limitation"]


def test_entry_only_bound_is_qualified_by_unchanged_no_bypass_admission():
    # A byte-oversized item can bypass admission and avoid evicting a small item.
    # Consequently an arbitrary smaller-byte-cap policy need not have fewer
    # hits than this unlimited-payload entry-LRU model. No geometry is needed
    # to exhibit the counterexample; the recorded run had zero bypasses.
    keys = ["small", "oversized", "small"]
    entry_only = MOD.entry_only_lru(keys, 1)
    admitted = [key for key in keys if key != "oversized"]
    byte_policy_hits = MOD.entry_only_lru(admitted, 1)["hits"]
    assert entry_only["hits"] == 0 < byte_policy_hits == 1
    summary = MOD.summarize({"queries": [{"argument_float64_le_hex": key} for key in keys]}, (1,))
    assert "no-bypass" in summary["interpretation"]


@pytest.mark.parametrize("authority", ["source", "worker", "result"])
def test_resealed_private_inputs_cannot_change_declared_runtime(tmp_path, monkeypatch, saved, authority):
    _, output, _ = private_cli(tmp_path, monkeypatch, saved, runtime_mismatch=authority)
    with pytest.raises(ValueError, match="(?i)runtime"):
        MOD.main()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "summary.json").exists()


def test_default_only_reads_declaration_even_when_public_paths_are_missing(tmp_path, monkeypatch):
    declaration = {"input_run": "forbidden-public-run", "input_sha256": {"forbidden-public-run/input": "missing"}}
    declaration["declaration_content_hash"] = MOD.digest(declaration)
    path = tmp_path / MOD.DECLARATION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(declaration))
    output = tmp_path / "default-output"
    monkeypatch.setattr(MOD, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["reconstruct", "--output", str(output)])
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "forbidden-public-run" not in str(path), "default must not open any public input"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    MOD.main()
    assert json.loads((output / "status.json").read_text()) == {"status": "declared_not_executed"}


def test_cooperative_budget_failure_retains_failed_attempt(tmp_path, monkeypatch, saved):
    _, output, _ = private_cli(tmp_path, monkeypatch, saved)
    monkeypatch.setattr(MOD.time, "perf_counter", iter([0., 100.]).__next__)
    with pytest.raises(InterruptedError, match="wall budget"):
        MOD.main()
    receipt = json.loads((output / "status.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "InterruptedError"
    assert not (output / "ordered-query-arguments.json.gz").exists()


def test_decompression_cap_has_failed_authority(tmp_path, monkeypatch, saved):
    root, output, declaration = private_cli(tmp_path, monkeypatch, saved)
    declaration["maximum_uncompressed_trace_bytes"] = 1
    declaration.pop("declaration_content_hash")
    declaration["declaration_content_hash"] = MOD.digest(declaration)
    (root / MOD.DECLARATION).write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="decompression"):
        MOD.main()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_input_hash_change_rejects_before_reconstruction(tmp_path, monkeypatch, saved):
    root, output, _ = private_cli(tmp_path, monkeypatch, saved)
    (root / "saved/preparation.json").write_text("{}")
    monkeypatch.setattr(MOD, "reconstruct", lambda *a, **k: pytest.fail("changed source must reject before reconstruction"))
    with pytest.raises(ValueError, match="Bound input changed"):
        MOD.main()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def reseal_one_private_input(root, declaration, relative, value):
    path = root / relative
    path.write_text(json.dumps(value))
    declaration["input_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    declaration.pop("declaration_content_hash")
    declaration["declaration_content_hash"] = MOD.digest(declaration)
    (root / MOD.DECLARATION).write_text(json.dumps(declaration))


def test_coherently_rehashed_callsite_drift_still_rejects(tmp_path, monkeypatch, saved):
    root, output, declaration = private_cli(tmp_path, monkeypatch, saved)
    relative = "saved/launch-source.json"
    value = json.loads((root / relative).read_text())
    value["file_sha256"]["synthetic-callsite"] = "different-capsule-formula"
    reseal_one_private_input(root, declaration, relative, value)
    with pytest.raises(ValueError, match="callsite"):
        MOD.main()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "ordered-query-arguments.json.gz").exists()


def test_coherently_rehashed_warm_denominator_mismatch_rejects(tmp_path, monkeypatch, saved):
    root, output, declaration = private_cli(tmp_path, monkeypatch, saved)
    relative = "saved/cached_warm/receipt.json"
    value = json.loads((root / relative).read_text())
    value["cache_after"]["calls"] += 2
    reseal_one_private_input(root, declaration, relative, value)
    with pytest.raises(ValueError, match="denominator"):
        MOD.main()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_process_rss_limit_preserves_failed_receipt(tmp_path, monkeypatch, saved):
    _, output, _ = private_cli(tmp_path, monkeypatch, saved)
    units = 1 if sys.platform == "darwin" else 1024
    monkeypatch.setattr(MOD.resource, "getrusage", lambda *a: SimpleNamespace(ru_maxrss=(1024**3)//units))
    with pytest.raises(InterruptedError, match="RSS"):
        MOD.main()
    receipt = json.loads((output / "status.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "InterruptedError"
    assert not (output / "ordered-query-arguments.json.gz").exists()


def test_in_process_runtime_guard_is_independent_of_earlier_pytest_peak(tmp_path, monkeypatch, saved):
    units = 1 if sys.platform == "darwin" else 1024
    monkeypatch.setattr(resource, "getrusage", lambda *args: SimpleNamespace(ru_maxrss=(1024**3)//units))
    _, output, declaration = private_cli(tmp_path, monkeypatch, saved, runtime_mismatch="source")
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units > declaration["maximum_process_peak_rss_bytes"]
    assert MOD.resource is not resource
    with pytest.raises(ValueError, match="runtime"):
        MOD.main()
    receipt = json.loads((output / "status.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"


def test_late_input_mutation_keeps_diagnostic_payload_ineligible(tmp_path, monkeypatch, saved):
    root, output, _ = private_cli(tmp_path, monkeypatch, saved)
    original = MOD.summarize
    def mutate_after_analysis(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "saved/preparation.json").write_text("{}")
        return result
    monkeypatch.setattr(MOD, "summarize", mutate_after_analysis)
    with pytest.raises(ValueError, match="during reconstruction"):
        MOD.main()
    assert (output / "ordered-query-arguments.json.gz").exists()
    assert not (output / "summary.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_no_native_or_numerical_modules_needed_for_saved_synthetic_cli(tmp_path, monkeypatch, saved):
    _, output, _ = private_cli(tmp_path, monkeypatch, saved)
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"resectionlab", "numpy", "scipy", "torch", "nibabel"}, name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    MOD.main()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["summary_sha256"] == hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[key] == 0 for key in ("cover_geometry_calls", "patient_loads", "simulator_steps", "gradient_updates"))
    assert summary["final_worlds_used"] is False and summary["stress_worlds_used"] is False
