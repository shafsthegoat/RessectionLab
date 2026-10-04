#!/usr/bin/env python3
"""Paired saved-forward diagnostics for the completed one-update axis pilot.

Stdlib only. No policy forward, simulator, optimizer, random draw or new world.
All six prescribed selection pairs must match before any diagnostic is emitted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

SOURCE_RECEIPT_SHA256 = "d2fb38139898d842961220d19aec7f481a9a8567bdb288737ffc627cd2450ea0"
FORWARD_AUDIT_SHA256 = "37b9bec653545486446cd78f845615867654838938a7f9a69ba07c0451d0eb21"
SOURCE_COMMIT = "f3a05910bc142e44c253cae7b747da3fb315d3ea"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return "sha256:" + digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def f32_bytes(values):
    packed = bytearray()
    for value in values:
        require(type(value) in (int, float) and math.isfinite(value), "Nonfinite/non-numeric float32 evidence")
        encoded = struct.pack("<f", value)
        require(struct.unpack("<f", encoded)[0] == value, "Saved evidence is not exactly represented by float32")
        packed.extend(encoded)
    return bytes(packed)


def array(record, shape, *, boolean=False):
    require(isinstance(record, dict) and record["shape"] == list(shape), "Saved array dimensions differ")
    values = record["values"]
    if len(shape) == 2:
        require(len(values) == shape[0] and all(len(row) == shape[1] for row in values), "Ragged array")
        flattened = [value for row in values for value in row]
    else:
        require(len(values) == shape[0], "Array length differs")
        flattened = values
    if boolean:
        require(record["dtype"] == "|b1" and all(type(value) is bool for value in flattened), "Invalid boolean array")
        encoded = bytes(flattened)
    else:
        require(record["dtype"] == "<f4", "Actual capture must use the recorded little-endian float32 format")
        encoded = f32_bytes(flattened)
    return values, encoded


def validate_forward(event):
    payload = event["payload"]
    require(event["status"] == "step_returned" and payload["version"] == "learner-decision-observer-v1"
            and payload["role"] == "selection" and payload["forward_evaluated"] is True
            and payload["decision_rule"] == "deterministic_argmax" and payload["forced_reason"] is None,
            "Expected a complete recorded deterministic selection forward")
    inputs = payload["inputs"]
    ids = inputs["action_ids"]
    require(ids and ids[0] == "STOP" and len(ids) == len(set(ids)), "Invalid ordered action inventory")
    mask, mask_bytes = array(inputs["action_mask"], (len(ids),), boolean=True)
    require(all(mask), "Axis inventory contains an uncertified action")
    actions, action_bytes = array(inputs["action_features"], (len(ids), 15))
    state, state_bytes = array(inputs["state_features"], (6,))
    for key in ("source_action_features", "actor_action_features"):
        require(array(inputs[key], (len(ids), 15))[1] == action_bytes, "RAW action input transformation differs")
    require(array(inputs["source_state_features"], (6,))[1] == state_bytes, "RAW state input differs")
    logits, _ = array(payload["logits"], (len(ids),))
    f32_bytes([payload["value"]])
    best = max(range(len(logits)), key=logits.__getitem__)
    require(payload["selected_index"] == best and payload["selected_action_id"] == ids[best], "Saved selection is not first argmax")
    identity = canonical_hash({"action_ids": ids, "action_float32_bytes": action_bytes.hex(),
        "state_float32_bytes": state_bytes.hex(), "mask_bytes": mask_bytes.hex()})
    return {"action_ids": ids, "action_features": actions, "state_features": state,
        "identity": identity, "logits": logits, "value": payload["value"], "chosen_index": best,
        "decision_id": event["decision_id"], "payload_hash": canonical_hash(payload)}


def rank_values(logits):
    ordered = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    ranks = [0] * len(logits)
    for rank, index in enumerate(ordered, 1):
        ranks[index] = rank
    return ordered, ranks


def diagnose_pairs(events, selection_seeds, initial_policy_hash, latest_policy_hash, *, steps=3):
    """Pure arithmetic on all prescribed saved selection states; no state selection."""
    require(len(selection_seeds) == 2 and len(set(selection_seeds)) == 2, "Two distinct declared selection seeds required")
    selected = [event for event in events if event["kind"] == "decision" and event["role"] == "selection"]
    require(len(selected) == 4 * steps, "All initial and updated selection forwards are required")
    by_key = {}
    for event in selected:
        payload = event["payload"]
        require(event["seed"] == payload["seed"] and payload["seed"] in selection_seeds,
                "Selection seed/context mismatch")
        require(payload["update"] in (0, 1) and payload["panel"] == payload["update"]
                and payload["episode"] == selection_seeds.index(payload["seed"])
                and type(payload["step"]) is int and 0 <= payload["step"] < steps,
                "Selection update/panel/episode/step mismatch")
        key = (payload["seed"], payload["step"], payload["update"])
        require(key not in by_key, "Duplicate saved selection state")
        by_key[key] = validate_forward(event)
    pairs = []
    for seed in selection_seeds:
        for step in range(steps):
            before = by_key[(seed, step, 0)]
            after = by_key[(seed, step, 1)]
            require(before["identity"] == after["identity"], "Initial/latest states, ordering or RAW inputs differ; pairing refused")
            first, first_rank = rank_values(before["logits"])
            last, last_rank = rank_values(after["logits"])
            first_mean = math.fsum(before["logits"]) / len(first)
            last_mean = math.fsum(after["logits"]) / len(last)
            classes = defaultdict(list)
            # STOP is excluded: the question concerns competing physical actions.
            for index, features in enumerate(before["action_features"][1:], 1):
                classes[f32_bytes(features)].append(index)
            aliases = [{"indices": indices, "action_ids": [before["action_ids"][i] for i in indices],
                "feature_float32_sha256": digest(features),
                "initial_logits": [before["logits"][i] for i in indices],
                "latest_logits": [after["logits"][i] for i in indices]}
                for features, indices in classes.items() if len(indices) > 1]
            action_rows = []
            for index, action_id in enumerate(before["action_ids"]):
                b, a = before["logits"][index], after["logits"][index]
                action_rows.append({"index": index, "action_id": action_id, "initial_logit": b,
                    "latest_logit": a, "logit_delta": a - b,
                    "centered_logit_delta": (a - last_mean) - (b - first_mean),
                    "initial_rank": first_rank[index], "latest_rank": last_rank[index],
                    "rank_delta": last_rank[index] - first_rank[index]})
            pairs.append({"world_seed": seed, "step": step, "state_identity": before["identity"],
                "initial_decision_id": before["decision_id"], "latest_decision_id": after["decision_id"],
                "initial_payload_hash": before["payload_hash"], "latest_payload_hash": after["payload_hash"],
                "initial_policy_hash": initial_policy_hash, "latest_policy_hash": latest_policy_hash,
                "initial_chosen_id": before["action_ids"][first[0]], "latest_chosen_id": after["action_ids"][last[0]],
                "chosen_action_unchanged": first[0] == last[0],
                "initial_top_two_margin": before["logits"][first[0]] - before["logits"][first[1]],
                "latest_top_two_margin": after["logits"][last[0]] - after["logits"][last[1]],
                "initial_value": before["value"], "latest_value": after["value"],
                "value_delta": after["value"] - before["value"],
                "logits_exactly_unchanged": before["logits"] == after["logits"],
                "rank_changed_actions": sum(row["rank_delta"] != 0 for row in action_rows),
                "max_absolute_centered_logit_delta": max(abs(row["centered_logit_delta"]) for row in action_rows),
                "legal_nonstop_actions": len(first) - 1, "unique_nonstop_feature_rows": len(classes),
                "alias_groups": aliases, "aliased_nonstop_actions": sum(len(group["indices"]) for group in aliases),
                "action_rows": action_rows})
    grouped = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[pair["state_identity"]].append(index)
    unique = []
    for identity, indices in grouped.items():
        first = pairs[indices[0]]
        for index in indices[1:]:
            other = pairs[index]
            require(first["action_rows"] == other["action_rows"]
                    and first["initial_value"] == other["initial_value"]
                    and first["latest_value"] == other["latest_value"],
                    "Same exact state has inconsistent saved deterministic outputs")
        unique.append({"state_identity": identity, "pair_indices": indices,
            "representative_pair": indices[0], "occurrences": len(indices)})
    return {"selection_record_pairs": len(pairs), "unique_states": len(unique), "pairs": pairs,
        "unique_state_groups": unique, "all_chosen_actions_unchanged": all(row["chosen_action_unchanged"] for row in pairs),
        "rank_definition": "Descending saved float32 logits; exact ties retain the original action index. Ranks are ordinal, not clinical preferences.",
        "centering": "Subtract each state's mean logit including STOP before taking paired difference; no softmax computed.",
        "alias_definition": "Identical 15-element actual RAW float32 row bytes, excluding STOP, with the same shared six-feature state. No approximate grouping or future-outcome equality is asserted."}


class Evidence:
    def __init__(self, root):
        self.root, self.hashes = root, {}

    def raw(self, name, expected=None):
        path = self.root / name
        compressed = path.with_suffix(path.suffix + ".gz")
        raw = path.read_bytes() if path.exists() else gzip.decompress(compressed.read_bytes())
        if path.exists() and compressed.exists():
            require(gzip.decompress(compressed.read_bytes()) == raw, "Raw/gzip disagreement: " + name)
        sha = digest(raw)
        if expected is not None:
            require(sha == expected, "Pinned evidence bytes changed: " + name)
        self.hashes[name] = sha
        return raw

    def read(self, name, expected=None):
        return json.loads(self.raw(name, expected), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def extract(root, audit_path):
    source = Evidence(root)
    receipt = source.read("report-v1/report-source.json", SOURCE_RECEIPT_SHA256)
    require(receipt["execution_baseline"]["record"]["source_commit"] == SOURCE_COMMIT,
            "Different executed source baseline")
    expected = receipt["input_files"]
    def read(name):
        return source.read(name, expected[name]["uncompressed_sha256"])
    launcher, worker, pilot = (read(name) for name in ("launcher-status.json", "worker-status.json", "pilot/status.json"))
    require(launcher["status"] == worker["status"] == pilot["status"] == "completed"
            and launcher["worker_returncode"] == 0 and not launcher["parent_timeout_requested"]
            and not launcher["hard_killed"], "Completed final publication authorities required")
    require(worker["pilot_status_sha256"] == source.hashes["pilot/status.json"], "Final authority mismatch")
    source.raw("pilot/candidate-record.json", expected["pilot/candidate-record.json"]["uncompressed_sha256"])
    require(worker["candidate_record_sha256"] == pilot["candidate_record_sha256"]
            == source.hashes["pilot/candidate-record.json"], "Candidate identity mismatch")
    for name in ("pilot/learner/initial.pt", "pilot/learner/checkpoint.pt"):
        source.raw(name, expected[name]["uncompressed_sha256"])
    declaration, contract, result, journal = (read(name) for name in
        ("declaration.json", "pilot/learner/contract.json", "pilot/learner/result.json", "pilot/accounting.json"))
    audit_data = audit_path.read_bytes()
    require(digest(audit_data) == FORWARD_AUDIT_SHA256, "Completed independent forward audit differs")
    require(result["gradient_steps"] == 1 and result["actor_parameters_changed"] is True
            and result["initial_checkpoint_hash"] != result["latest_checkpoint_hash"]
            and contract["input_profile"]["profile_id"] == "RAW", "Different learning/profile contract")
    require(journal["status"] == "learner_returned" and journal["counts_complete"]
            and journal["decision_model_hash"] == result["decision_model_hash"], "Different model or incomplete journal")
    require(journal["receipt_hash"] == "sha256:" + digest(json.dumps({key: value for key, value in journal.items()
        if key != "receipt_hash"}, sort_keys=True, allow_nan=False).encode()), "Journal content hash differs")
    analysis = diagnose_pairs(journal["events"], declaration["world_partitions"]["selection"]["seeds"],
        result["initial_checkpoint_hash"], result["latest_checkpoint_hash"], steps=3)
    return {"scope": "Post hoc descriptive diagnostic of all six already-recorded selection pairs; no efficacy inference or new tuning choice",
        "source_commit": SOURCE_COMMIT, "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "independent_forward_audit_sha256": FORWARD_AUDIT_SHA256,
        "decision_model_hash": result["decision_model_hash"], "profile_hash": result["input_profile_hash"],
        "initial_policy_hash": result["initial_checkpoint_hash"], "latest_policy_hash": result["latest_checkpoint_hash"],
        "selected_policy_hash": result["selected_checkpoint_hash"],
        "initial_selection_return": result["initial_selection_return"],
        "updated_selection_return": result["selection_history"][1]["mean_return"],
        "source_file_sha256": source.hashes, "analysis": analysis,
        "diagnostic_script_sha256": digest(Path(__file__).read_bytes()),
        "new_policy_forwards": 0, "new_simulator_calls": 0, "new_random_draws": 0,
        "new_gradient_steps": 0, "final_worlds_used": False, "stress_worlds_used": False}


def render(report):
    analysis = report["analysis"]
    representatives = [analysis["pairs"][group["representative_pair"]] for group in analysis["unique_state_groups"]]
    return "\n".join(["# Saved native-axis decisions after one RAW update", "",
        f"All {analysis['selection_record_pairs']} prescribed initial/latest pairs matched exactly on ordered action IDs, RAW feature rows, state and mask. They represent {analysis['unique_states']} distinct states repeated across deterministic selection seeds. Chosen actions were {'unchanged in every pair' if analysis['all_chosen_actions_unchanged'] else 'changed in at least one pair'}.", "",
        "| State / step | Non-STOP rows / unique rows | Aliased rows | Initial → latest top-two margin | Actions changing rank | Initial → latest value |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {i+1} / {row['step']} | {row['legal_nonstop_actions']} / {row['unique_nonstop_feature_rows']} | {row['aliased_nonstop_actions']} | {row['initial_top_two_margin']:.6g} → {row['latest_top_two_margin']:.6g} | {row['rank_changed_actions']} | {row['initial_value']:.6g} → {row['latest_value']:.6g} |" for i,row in enumerate(representatives)], "",
        "The JSON retains every action's actual saved logit, rank and paired change, including logit differences after subtracting each state's mean. These describe the recorded update; no policy forward or softmax was computed. Selection uses the first maximum row, and the latest weights remain distinct from the selected checkpoint.", "",
        "Exact repeated 15-feature rows expose actions that this representation presents identically to the actor in a shared state. This is a representational observation; it does not establish equal future outcomes, prove that every optimal route is unrepresentable, or attribute the unchanged selection score to aliasing. Distinct feature rows can also produce equal logits. No near-equality threshold or favorable subset was chosen after seeing results.", "",
        "These are three steps in one previously studied development patient, with deterministic repeats and the same unreviewed support/hypothetical access as the completed pilot. This post hoc diagnostic establishes neither clinical efficacy nor robustness and chooses no new settings. It reads immutable saved outputs and checkpoint bytes only: zero new model forwards, simulation episodes, random draws, gradients, final worlds or stress worlds.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.run, args.audit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "DIAGNOSTIC.md").write_text(render(report))


if __name__ == "__main__":
    main()
