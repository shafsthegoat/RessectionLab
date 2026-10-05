"""Independent pure artifact adversaries; no native simulation/cache execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_public_native_capsule_cache as audit


def seal(stats):
    record = {key: stats[key] for key in audit.NAMESPACE_FIELDS}
    stats["namespace"] = "sha256:" + hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def cache_fixture():
    declared = {"cache": {"version": "fixture-cache", "max_entries": 10, "max_payload_bytes": 100},
                "v2_model": {"native_config_hash": "geometry"}, "source_case": {"semantic_hash": "case"}}
    source = {"file_sha256": {"src/resectionlab/geometry.py": "geometry-source", "src/resectionlab/native_proposals.py": "source-guard"}}
    constants = {"geometry_version": "geometry-v", "native_version": "native-v", "geometry_epsilon": 1e-9}
    empty = {"cache_version": "fixture-cache", **constants, "geometry_source_sha256": "geometry-source",
        "source_guard_sha256": "source-guard", "mode": "exact_orthogonal_cells", "native_config_hash": "geometry", "source_hash": "case",
        "guarded_direct_geometry_callables": ["world_to_voxel"], "max_entries": 10, "max_payload_bytes": 100,
        "calls": 0, "hits": 0, "misses": 0, "evictions": 0, "bypasses": 0, "entries": 0, "retained_payload_bytes": 0,
        "query_seconds": 0., "compute_seconds": 0., "feasibility_cached": False, "cavity_cached": False, "integrity_cached": False}
    seal(empty)
    cold = {**empty, "calls": 10, "hits": 5, "misses": 5, "entries": 5, "retained_payload_bytes": 60,
            "query_seconds": .02, "compute_seconds": .01}
    warm = {**cold, "calls": 20, "hits": 15, "query_seconds": .03}
    phases = [{"mode": "reference_before", "cache_before": None, "cache_after": None},
              {"mode": "cached_cold", "cache_before": empty, "cache_after": cold},
              {"mode": "cached_warm", "cache_before": cold, "cache_after": warm},
              {"mode": "reference_after", "cache_before": warm, "cache_after": warm}]
    return declared, source, constants, phases, {"stats": empty}


def test_cold_warm_cache_accounting_and_reference_reversal():
    declaration, source, constants, phases, construction = cache_fixture()
    deltas = audit.check_cache_sequence(phases, construction, declaration, source, constants)
    assert deltas["cached_cold"]["hits"] == deltas["cached_cold"]["misses"] == 5
    assert deltas["cached_warm"]["hits"] == 10 and deltas["cached_warm"]["misses"] == 0


def test_zero_hit_eviction_cycle_is_retained_as_valid_negative_result():
    declaration, source, constants, phases, construction = cache_fixture()
    cold = {**phases[1]["cache_after"], "calls": 10, "hits": 0, "misses": 10, "evictions": 8,
            "entries": 2, "retained_payload_bytes": 72}
    warm = {**cold, "calls": 20, "misses": 20, "evictions": 18, "query_seconds": .04, "compute_seconds": .02}
    phases[1]["cache_after"] = phases[2]["cache_before"] = cold
    phases[2]["cache_after"] = phases[3]["cache_before"] = phases[3]["cache_after"] = warm
    result = audit.check_cache_sequence(phases, construction, declaration, source, constants)
    assert result["cached_cold"]["hits"] == result["cached_warm"]["hits"] == 0
    assert result["cached_warm"]["misses"] == result["cached_warm"]["evictions"] == 10


@pytest.mark.parametrize("attack", ["hit_sum", "entries", "payload", "evictions", "negative", "bool_count", "timer", "cavity_scope", "source", "geometry"])
def test_invalid_cache_stats_rejected_even_with_resealed_namespace(attack):
    declaration, source, constants, phases, _ = cache_fixture()
    stats = copy.deepcopy(phases[1]["cache_after"])
    key, value = {"hit_sum": ("hits", 99), "entries": ("entries", 20), "payload": ("retained_payload_bytes", 101),
        "evictions": ("evictions", 9), "negative": ("misses", -1), "bool_count": ("calls", True),
        "timer": ("compute_seconds", 8.), "cavity_scope": ("cavity_cached", True),
        "source": ("source_hash", "different-case"), "geometry": ("geometry_epsilon", .1)}[attack]
    stats[key] = value
    seal(stats)
    with pytest.raises(audit.AuditError):
        audit.check_cache_stats(stats, declaration, source, constants)


@pytest.mark.parametrize("attack", ["order", "nonempty_cold", "warm_reset", "reference_after_query"])
def test_phase_cache_continuity_is_not_inferred_from_final_hit_count(attack):
    declaration, source, constants, phases, construction = cache_fixture()
    phases = copy.deepcopy(phases)
    if attack == "order":
        phases[1], phases[2] = phases[2], phases[1]
    elif attack == "nonempty_cold":
        construction["stats"]["calls"] = 1
    elif attack == "warm_reset":
        phases[2]["cache_before"] = copy.deepcopy(construction["stats"])
    else:
        phases[3]["cache_after"] = {**phases[3]["cache_after"], "calls": 21, "hits": 16}
    with pytest.raises(audit.AuditError):
        audit.check_cache_sequence(phases, construction, declaration, source, constants)


def test_exterior_flood_preserves_sealed_pocket_and_forbids_diagonal_contact():
    remaining = np.ones((5, 5, 5), bool)
    remaining[0, 1, 1] = False
    remaining[1, 2, 1] = False  # only diagonal from exterior
    remaining[3, 3, 3] = False  # sealed pocket
    connected = audit.exterior_free_mask(remaining)
    assert connected[0, 1, 1] and not connected[1, 2, 1] and not connected[3, 3, 3]
    remaining[1, 1, 1] = False
    assert audit.exterior_free_mask(remaining)[1, 2, 1]
    assert not audit.exterior_free_mask(remaining)[3, 3, 3]


def test_empty_volume_and_each_boundary_face_seed_the_flood():
    assert audit.exterior_free_mask(np.zeros((3, 3, 3), bool)).all()
    for axis in range(3):
        for coordinate in (0, 2):
            remaining = np.ones((3, 3, 3), bool)
            point = [1, 1, 1]; point[axis] = coordinate
            remaining[tuple(point)] = False
            assert audit.exterior_free_mask(remaining).sum() == 1


def initial_inventory():
    folder = ROOT / "artifacts/preflight/native-axis-v2/profile"
    model = json.loads((folder / "model.json").read_text())
    inventory = json.loads((folder / "initial/inventory.json").read_text())["complete_inventory"]
    return model, audit.strip_inventory_timing(inventory)


def test_frozen_original_inventory_is_complete_with_actual_order():
    model, inventory = initial_inventory()
    result = audit.check_inventory(inventory, model)
    assert result["slots"] == result["previews"] == result["certified"] == 26
    assert result["fallback_previews"] == result["rejected_previews"] == 0


@pytest.mark.parametrize("attack", ["reordered_slots", "missing_slot", "unconditional_fallback", "lost_certificate", "counter", "source"])
def test_inventory_cannot_hide_missing_work_or_change_science(attack):
    model, inventory = initial_inventory()
    if attack == "reordered_slots":
        inventory["batch"]["ledger"][0], inventory["batch"]["ledger"][1] = inventory["batch"]["ledger"][1], inventory["batch"]["ledger"][0]
    elif attack == "missing_slot":
        inventory["batch"]["ledger"].pop()
    elif attack == "unconditional_fallback":
        inventory["attempts"][0]["phase"] = "fallback"
    elif attack == "lost_certificate":
        inventory["certified_action_ids"].pop()
    elif attack == "counter":
        inventory["batch"]["counts"]["PROPOSED_UNCERTIFIED"] += 1
    else:
        inventory["batch"]["engine_model_hash"] = "changed"
    with pytest.raises(audit.AuditError):
        audit.check_inventory(inventory, model)


def test_timing_filter_does_not_drop_unrelated_elapsed_or_geometry_fields():
    value = {"attempts": [{"elapsed_seconds": 1., "radius_mm": 2., "nested": {"elapsed_seconds": 3.}}],
             "science_elapsed_seconds": 4.}
    stripped = audit.strip_inventory_timing(value)
    assert stripped == {"attempts": [{"radius_mm": 2., "nested": {"elapsed_seconds": 3.}}], "science_elapsed_seconds": 4.}
    assert value["attempts"][0]["elapsed_seconds"] == 1.


def test_ancestry_checksum_includes_partial_contact_not_only_removed_cells():
    cert = {"removed_indices_native": [[1, 1, 1]], "contact_indices_native": [[1, 1, 1], [1, 1, 2]],
            "native_affine": np.eye(4).tolist(), "microsteps": [
                {"removed_indices_native": [[1, 1, 1]], "contact_indices_native": [[1, 1, 1], [1, 1, 2]]}]}
    original = audit.certificate_digest(cert)
    changed = copy.deepcopy(cert)
    changed["contact_indices_native"][1] = [1, 2, 1]
    changed["microsteps"][0]["contact_indices_native"][1] = [1, 2, 1]
    assert audit.certificate_digest(changed) != original
    assert cert["contact_indices_native"][1] == [1, 1, 2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def authorities(tmp_path):
    result = {"status": "completed", "runtime_content_hash": "runtime", "scientific_model_hash": "model",
              "gradient_updates": 0, "final_worlds_used": False, "stress_worlds_used": False, "clinical_deficit_probability": None}
    worker = {"status": "completed", "runtime_content_hash": "runtime", "scientific_model_hash": "model", "result_hash": audit.canonical_hash(result)}
    launcher = {"status": "completed", "worker_returncode": 0, "parent_timeout_requested": False,
                "hard_killed": False, "receipt_error": None}
    for name, value in (("result.json", result), ("worker-status.json", worker), ("launcher-status.json", launcher)):
        write(tmp_path / name, value)
    return launcher, worker, result


def test_atomic_authority_chain_accepts_only_complete_matching_result(tmp_path):
    _, _, result = authorities(tmp_path)
    actual, _ = audit.check_authorities(audit.Reader(), tmp_path)
    assert actual == result
    result["extra"] = True
    write(tmp_path / "result.json", result)
    with pytest.raises(audit.AuditError, match="worker authority"):
        audit.check_authorities(audit.Reader(), tmp_path)


@pytest.mark.parametrize("attack", ["timeout", "failed_worker", "missing_result", "gradients", "final_worlds"])
def test_failed_or_out_of_scope_result_cannot_pass(tmp_path, attack):
    launcher, worker, result = authorities(tmp_path)
    if attack == "timeout":
        launcher["parent_timeout_requested"] = True
        write(tmp_path / "launcher-status.json", launcher)
    elif attack == "failed_worker":
        worker["status"] = "failed"; write(tmp_path / "worker-status.json", worker)
    elif attack == "missing_result":
        (tmp_path / "result.json").unlink()
    else:
        result["gradient_updates" if attack == "gradients" else "final_worlds_used"] = 1 if attack == "gradients" else True
        worker["result_hash"] = audit.canonical_hash(result)
        write(tmp_path / "result.json", result); write(tmp_path / "worker-status.json", worker)
    with pytest.raises(audit.AuditError):
        audit.check_authorities(audit.Reader(), tmp_path)


def test_incomplete_attempt_retains_diagnostics_without_equivalence_claim(tmp_path):
    authorities(tmp_path)
    write(tmp_path / "launcher-status.json", {"status": "failed"})
    result, receipt = audit.check_authorities(audit.Reader(), tmp_path)
    assert result is None and receipt["audit_status"] == "incomplete_attempt_retained"
    assert not receipt["scientific_equivalence_verified"] and (tmp_path / "result.json").exists()
