"""Synthetic tamper tests; no patient replay or production simulation imports."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("axis_result_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


@pytest.mark.parametrize("attack", ["worker", "launcher", "timeout", "payload", "gradients", "stress", "denominator"])
def test_publication_cannot_ignore_either_authority_or_payload(attack):
    raw = b'{"candidates":{}}'
    common = {"status":"completed", "eligible_candidate_count":2, "gradient_steps":0,
        "final_worlds_used":False, "stress_worlds_used":False}
    launcher = {**common, "worker_returncode":0, "parent_timeout_requested":False,"hard_killed":False}
    status = {**common, "candidate_records_sha256":hashlib.sha256(raw).hexdigest()}
    audit.check_publication(launcher,status,raw)
    if attack == "worker": status["status"] = "failed"
    elif attack == "launcher": launcher["status"] = "running"
    elif attack == "timeout": launcher["parent_timeout_requested"] = True
    elif attack == "payload": raw += b' '
    elif attack == "gradients": status["gradient_steps"] = 1
    elif attack == "stress": status["stress_worlds_used"] = True
    else: status["eligible_candidate_count"] = 1
    with pytest.raises(audit.AuditError):
        audit.check_publication(launcher,status,raw)


def fixture_inventory():
    model = {"proposal_rule":{"offsets_source_voxels":[[0,0]]},"tools":[{"tool_id":"tool"}],
        "decision_model_hash":"model", "native_config_hash":"native", "proposal_model_hash":"proposer",
        "case_hash":"case", "max_actions":2}
    proposal = {"column_index":0,"offset_source_voxels":[0,0],"tool_id":"tool","proposal_id":"p",
        "entry_mm":[0,0,0],"primary_target_mm":[0,0,2],"fallback_target_mm":[0,0,1],
        "fallback_condition":"primary_preview_rejected"}
    batch = {"slot_count":1,"ledger":[{k:proposal[k] for k in ("column_index","offset_source_voxels","tool_id","proposal_id")}],
        "counts":{"PROPOSED_UNCERTIFIED":1},"source_hash":"case","engine_model_hash":"native",
        "proposal_model_hash":"proposer","geometry_certified":False,"removal_authorized":False,
        "unsupported_reason":None,"proposals":[proposal]}
    batch["ledger"][0]["reason"] = "PROPOSED_UNCERTIFIED"
    attempts = [{"status":"complete","proposal_id":"p","phase":phase,"feasible":phase=="fallback",
        "tool_id":"tool","entry_mm":[0,0,0],"tip_mm":tip,"elapsed_seconds":.1}
        for phase,tip in (("primary",[0,0,2]),("fallback",[0,0,1]))]
    return {"batch":batch,"status":"complete","terminated":False,"attempts":attempts,
        "certified_action_ids":[audit.action_id("model","p","fallback")]},model


@pytest.mark.parametrize("attack", ["fallback_without_rejection","missing_primary","extra_slot","wrong_action","wrong_geometry"])
def test_complete_inventory_requires_all_slots_and_legal_fallback(attack):
    record, model = fixture_inventory()
    result = audit.check_inventory(record,model)
    assert result["previews"] == 2 and result["fallback"] == 1 and result["certified"] == 1
    if attack == "fallback_without_rejection": record["attempts"][0]["feasible"] = True
    elif attack == "missing_primary": record["attempts"].pop(0)
    elif attack == "extra_slot": record["batch"]["ledger"].append(copy.deepcopy(record["batch"]["ledger"][0]))
    elif attack == "wrong_action": record["certified_action_ids"][0] = "other"
    else: record["attempts"][1]["tip_mm"] = [0,0,9]
    with pytest.raises(audit.AuditError): audit.check_inventory(record,model)


def test_terminal_unpreviewed_provider_rays_are_not_certified_actions():
    record, model = fixture_inventory()
    record.update(terminated=True, attempts=[], certified_action_ids=[])
    result = audit.check_inventory(record,model)
    assert result["proposals"] == 1 and result["previews"] == result["certified"] == 0
    record["certified_action_ids"] = ["invented"]
    with pytest.raises(audit.AuditError): audit.check_inventory(record,model)


@pytest.mark.parametrize("attack", ["duplicate_removal","missing_contact","outside_tissue","wrong_union"])
def test_microstep_union_and_source_support_are_enforced(attack):
    tissue = np.ones((2,1,1),bool)
    step = {"removed_indices_native":[[0,0,0]],"contact_indices_native":[[0,0,0],[1,0,0]],
        "microsteps":[{"removed_indices_native":[[0,0,0]],"contact_indices_native":[[0,0,0],[1,0,0]]}]}
    assert len(audit.check_microstep_accounting(step,tissue)[0]) == 1
    if attack == "duplicate_removal":step["microsteps"].append(copy.deepcopy(step["microsteps"][0]))
    elif attack == "missing_contact":
        step["contact_indices_native"]=[[1,0,0]];step["microsteps"][0]["contact_indices_native"]=[[1,0,0]]
    elif attack == "outside_tissue":tissue[0,0,0]=False
    else:step["microsteps"][0]["contact_indices_native"]=[[0,0,0]]
    with pytest.raises(audit.AuditError):audit.check_microstep_accounting(step,tissue)


def test_raw_and_gzip_receipts_produce_identical_byte_and_semantic_identity(tmp_path):
    raw=b'{"value": 1, "ordered": [1, 2]}\n'
    path=tmp_path/'receipt.json'
    path.write_bytes(raw)
    direct=audit.Reader(); expected=direct.json(path)
    path.unlink(); Path(str(path)+'.gz').write_bytes(gzip.compress(raw))
    compressed=audit.Reader()
    assert compressed.json(path)==expected and compressed.hashes==direct.hashes


def test_raw_and_gzip_disagreement_is_not_silently_ignored(tmp_path):
    path=tmp_path/'receipt.json'
    path.write_text('{"value":1}')
    Path(str(path)+'.gz').write_bytes(gzip.compress(b'{"value":2}'))
    with pytest.raises(audit.AuditError, match="Raw/gzip disagreement"):
        audit.Reader().json(path)


def test_tensor_mutation_changes_independent_hash():
    weights={"actor.0.weight":np.ones((2,3),np.float32)}
    expected=audit.tensor_hash(weights)
    weights["actor.0.weight"][0,0]+=1
    assert audit.tensor_hash(weights)!=expected


def cell_metrics():
    target = np.zeros((3, 1, 1), bool)
    target[1, 0, 0] = True
    history = []
    for index, tool in enumerate(("A", "A", "B")):
        history.append({"removed_indices_native": [[index, 0, 0]],
            "contact_indices_native": [[index, 0, 0], [2, 0, 0]] if index != 2 else [[2, 0, 0]],
            "tip_mm": [0., 0., 1.], "entry_mm": [0., 0., 0.], "tool_id": tool,
            "target_removed_mm3": 2. if index == 1 else 0.,
            "normal_removed_mm3": 0. if index == 1 else 2.,
            "partial_normal_contact_mm3": 2. if index == 0 else 0.,
            "motor_surrogate_delta": 0., "language_surrogate_delta": 0.,
            "partial_motor_contact_surrogate": 0., "partial_language_contact_surrogate": 0.})
    metrics = {"history": history, "total_reward": .73,
        "simulated_removed_target_volume_mm3": 2., "simulated_removed_normal_volume_mm3": 4.,
        "cumulative_partial_normal_contact_mm3": 2., "modeled_residual_target_volume_mm3": 0.,
        "clinical_deficit_probability": None, "functional_evidence_available": {"motor": False, "language": False},
        "motor_surrogate": None, "language_surrogate": None,
        "removed_by_compartment_mm3": {"target": 2.}, "residual_by_compartment_mm3": {"target": 0.}}
    reward = {"target_per_mm3": 1., "normal_per_mm3": .2, "action_cost": .01,
        "motion_per_mm": .02, "tool_change_cost": .3}
    return metrics, {"target": target}, reward

def test_independent_score_counts_partial_contact_once_even_if_later_removed():
    metrics, compartments, reward = cell_metrics()
    result = audit.recompute_score(metrics, compartments, 2., reward, .05)
    assert result["score"] == pytest.approx(.73)
    assert result["cumulative_partial_normal_contact_mm3"] == 2.

@pytest.mark.parametrize("change", ["partial_twice", "removed_twice", "inflated_target", "clinical_zero"])

def test_recomputed_score_rejects_bookkeeping_or_unknown_evidence_relabeling(change):
    metrics, compartments, reward = cell_metrics()
    if change == "partial_twice":
        metrics["history"][1]["partial_normal_contact_mm3"] = 2.
    elif change == "removed_twice":
        metrics["history"][1]["removed_indices_native"] = [[0, 0, 0]]
    elif change == "inflated_target":
        metrics["simulated_removed_target_volume_mm3"] = 3.
    else:
        metrics["clinical_deficit_probability"] = 0.
    with pytest.raises(audit.AuditError):
        audit.recompute_score(metrics, compartments, 2., reward, .05)
