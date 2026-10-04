"""Small independent arithmetic and tamper tests; no public diagnostic rerun."""
import ast
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("query_result_audit", ROOT / "scripts/audit_native_cache_query_results.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def tiny():
    metadata = {key: key + "-synthetic" for key in ("namespace", "cache_version", "native_config_hash", "source_hash",
        "geometry_source_sha256", "source_guard_sha256", "geometry_version", "native_version", "mode")}
    metadata["geometry_epsilon"] = 1e-9
    tool = {"tool_id": "T", "tip_length_mm": 1., "working_length_mm": 3., "tip_radius_mm": 2., "shaft_radius_mm": .5}
    cert = {"tool_id": "T", "entry_mm": [2., 3., 0.], "tip_mm": [2., 3., 1.], "axis_unit": [0., 0., 1.],
        "source_state_hash": "cavity", "source_hash": metadata["source_hash"], "decision_model_hash": metadata["native_config_hash"],
        "source_shape": [5, 5, 5], "native_affine": [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
        "microsteps": [{"tip_start_mm": [2.,3.,0.], "tip_end_mm": [2.,3.,float(z)],
            "active_stroke_start_mm": [2.,3.,-1.], "active_stroke_end_mm": [2.,3.,float(z)], "active_radius_mm": 2.}
                       for z in (0,1)]}
    attempt = {"status": "complete", "feasible": True, "tool_id": "T", "tip_mm": cert["tip_mm"], "entry_mm": cert["entry_mm"]}
    trace = {"snapshots": [{"action_ids": ["STOP", "A"], "cavity_state_hash": "cavity", "certificates": {"A": json.dumps(cert)},
        "inventory": {"status": "complete", "certified_action_ids": ["A"], "attempts": [attempt]}}]}
    binding = {**metadata, "version": "saved-native-exact-query-arguments-v1", "float64_byte_order": "little",
        "omitted_invariant_cache_frame_fields": ["inverse_affine_bytes", "cell_radius_mm", "orthogonal_spacing_bytes"],
        "source_shape": cert["source_shape"], "native_affine_float64_le_hex": AUDIT.doubles(sum(cert["native_affine"], [])).hex()}
    bh = AUDIT.content_hash(binding)
    # Independent literal four-call calculation, rather than producer import.
    arguments = [[2.,3.,-1.,2.,3.,0.,2.], [2.,3.,-3.,2.,3.,-1.,.5],
                 [2.,3.,-1.,2.,3.,1.,2.], [2.,3.,-3.,2.,3.,0.,.5]]
    rows = []
    for i, values in enumerate(arguments):
        hx = AUDIT.doubles(values).hex()
        rows.append({"index": i, "inventory_index": 0, "action_id": "A", "tool_id": "T", "microstep_index": i//2,
            "kind": ["active", "shaft"][i%2], "argument_float64_le_hex": hx,
            "key_projection_hash": AUDIT.content_hash({"binding": bh, "arguments": hx})})
    return trace, [tool], metadata, {"binding": binding, "binding_hash": bh, "inventory_query_counts": [4], "queries": rows}


def test_literal_active_shaft_four_call_sequence(tiny):
    result = AUDIT.verify_arguments(*tiny)
    assert result["queries"] == 4 and result["microsteps_by_inventory"] == [2]
    assert result["exact_argument_byte_mismatches"] == result["ordering_mismatches"] == 0


@pytest.mark.parametrize("attack", ["byte", "order", "tool", "microstep", "query_missing", "query_added", "projection_hash", "binding", "frame", "attempt", "cavity", "active_radius"])
def test_saved_argument_tampering_rejects(tiny, attack):
    trace, tools, metadata, projected = tiny
    if attack == "byte":
        projected["queries"][0]["argument_float64_le_hex"] = "01" + projected["queries"][0]["argument_float64_le_hex"][2:]
    elif attack == "order":
        projected["queries"][:2] = reversed(projected["queries"][:2])
    elif attack == "tool":
        tools[0]["working_length_mm"] += .1
    elif attack == "microstep":
        projected["queries"][2]["microstep_index"] = 0
    elif attack == "query_missing":
        projected["queries"].pop()
    elif attack == "query_added":
        projected["queries"].append(copy.deepcopy(projected["queries"][-1]))
    elif attack == "projection_hash":
        projected["queries"][0]["key_projection_hash"] = "sha256:foreign"
    elif attack == "binding":
        projected["binding"]["cell_radius_mm"] = 1.
    elif attack == "frame":
        projected["binding"]["native_affine_float64_le_hex"] = "00"
    elif attack == "attempt":
        trace["snapshots"][0]["inventory"]["attempts"][0]["feasible"] = False
    elif attack == "cavity":
        trace["snapshots"][0]["cavity_state_hash"] = "foreign"
    else:
        cert = json.loads(trace["snapshots"][0]["certificates"]["A"])
        cert["microsteps"][0]["active_radius_mm"] += .5
        trace["snapshots"][0]["certificates"]["A"] = json.dumps(cert)
    with pytest.raises(ValueError):
        AUDIT.verify_arguments(trace, tools, metadata, projected)


def test_signed_zero_and_ulp_remain_distinct():
    assert AUDIT.doubles([-0.]).hex() == "0000000000000080"
    assert AUDIT.doubles([0.]).hex() == "0000000000000000"
    assert AUDIT.doubles([math.nextafter(1., math.inf)]).hex() == "010000000000f03f"
    assert len(set(AUDIT.doubles([value]) for value in (-0., 0., 1., math.nextafter(1.,math.inf)))) == 4


def brute_distances(keys):
    result = []
    for i, key in enumerate(keys):
        before = [j for j in range(i) if keys[j] == key]
        result.append(None if not before else len(set(keys[before[-1]+1:i])))
    return result


def test_sorted_positions_and_heap_lru_match_exhaustive_brute_force():
    for size in range(1, 7):
        for keys in itertools.product("abc", repeat=size):
            expected = brute_distances(keys)
            assert AUDIT.independent_distances(keys) == expected
            for cap in range(5):
                actual = AUDIT.independent_lru(keys, cap)
                hits = sum(d is not None and d < cap for d in expected)
                assert actual == {"hits": hits, "misses": size-hits, "entries": min(cap,len(set(keys))),
                    "evictions": max(0,size-hits-cap) if cap else 0}


@pytest.fixture
def summary_fixture():
    # A,B,A has one first-pass reuse at distance 1; the phase boundary repeats
    # A immediately at distance 0, followed by distances 1 and 1.
    projected = {"queries": [{"argument_float64_le_hex": key, "key_projection_hash": "same-hash"} for key in ("00","01","00")]}
    summary = {"queries_per_phase": 3, "distinct_argument_keys": 2, "within_first_phase_repeat_count": 1,
        "occurrence_frequency_histogram": {"1": 1, "2": 1},
        "phases": [{"phase": "first", "calls": 3, "compulsory_misses": 2,
            "distinct_key_reuse_distance_histogram": {"1":1}, "maximum_distinct_key_reuse_distance": 1},
            {"phase": "repeated", "calls": 3, "compulsory_misses": 0,
            "distinct_key_reuse_distance_histogram": {"0":1,"1":2}, "maximum_distinct_key_reuse_distance": 1}],
        "entry_capacity_for_all_repeat_accesses_to_hit_without_payload_limit": 2,
        "entry_only_lru_unlimited_payload": {"1": {"first": {"hits":0,"misses":3,"entries":1,"evictions":2},
            "repeated_increment": {"hits":1,"misses":2,"evictions":2}},
            "2": {"first": {"hits":1,"misses":2,"entries":2,"evictions":0},
            "repeated_increment": {"hits":3,"misses":0,"evictions":0}}},
        "exact_total_cover_payload_bytes": None, "payload_weighted_reuse_distances": None,
        "interpretation": "same no-bypass admission; not measured speedup"}
    return projected, summary


def test_repeat_boundary_and_collision_agnostic_statistics(summary_fixture):
    assert AUDIT.verify_statistics(*summary_fixture, [1,2])["distinct_argument_keys"] == 2


@pytest.mark.parametrize("attack", ["distinct", "histogram", "lru", "payload", "wording"])
def test_summary_tampering_rejects(summary_fixture, attack):
    projected, summary = summary_fixture
    if attack == "distinct":
        summary["distinct_argument_keys"] += 1
    elif attack == "histogram":
        summary["phases"][1]["distinct_key_reuse_distance_histogram"]["1"] += 1
    elif attack == "lru":
        summary["entry_only_lru_unlimited_payload"]["1"]["repeated_increment"]["hits"] += 1
    elif attack == "payload":
        summary["exact_total_cover_payload_bytes"] = 0
    else:
        summary["interpretation"] = "predicts cache speedup"
    with pytest.raises(ValueError):
        AUDIT.verify_statistics(projected, summary, [1,2])


@pytest.fixture
def authority_fixture():
    declaration = json.loads((ROOT / AUDIT.DECLARATION_PATH).read_text())
    status = {"status":"completed", "summary_sha256":"summary-bytes"}
    summary = {"query_artifact_sha256":"query-bytes", "declaration_content_hash":AUDIT.DECLARATION_HASH,
        "script_sha256":AUDIT.PRODUCER_SHA256, "queries_per_phase":22364, "elapsed_seconds":1.,
        "cover_geometry_calls":0, "patient_loads":0, "simulator_steps":0, "gradient_updates":0,
        "final_worlds_used":False, "stress_worlds_used":False}
    return declaration, status, summary


def test_pinned_authority_chain(authority_fixture):
    AUDIT.verify_authority(*authority_fixture,"summary-bytes","query-bytes")


@pytest.mark.parametrize("attack", ["incomplete", "summary_hash", "query_hash", "source", "declaration", "count", "wall", "bool_counter", "final", "stress"])
def test_failed_or_coherently_resealed_authority_is_ineligible(authority_fixture, attack):
    declaration, status, summary = authority_fixture
    if attack == "incomplete": status["status"] = "failed"
    elif attack == "summary_hash": status["summary_sha256"] = "foreign"
    elif attack == "query_hash": summary["query_artifact_sha256"] = "foreign"
    elif attack == "source": summary["script_sha256"] = "foreign"
    elif attack == "declaration":
        declaration.pop("declaration_content_hash")
        declaration["maximum_queries"] += 1
        declaration["declaration_content_hash"] = AUDIT.content_hash(declaration)
    elif attack == "count": summary["queries_per_phase"] -= 2
    elif attack == "wall": summary["elapsed_seconds"] = 61.
    elif attack == "bool_counter": summary["gradient_updates"] = False
    elif attack == "final": summary["final_worlds_used"] = True
    else: summary["stress_worlds_used"] = True
    with pytest.raises(ValueError):
        AUDIT.verify_authority(declaration,status,summary,"summary-bytes","query-bytes")


def test_auditor_is_standard_library_only_and_does_not_import_producer():
    tree = ast.parse((ROOT / "scripts/audit_native_cache_query_results.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom): names = [node.module.split(".")[0]]
        else: continue
        assert set(names) <= sys.stdlib_module_names
