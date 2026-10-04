"""Independent constructed saved-record adversaries; no model or environment."""
import copy
import gzip
import hashlib
import importlib.util
from pathlib import Path
import struct

import pytest


SPEC = importlib.util.spec_from_file_location(
    "independent_decision_report_target",
    Path(__file__).resolve().parents[1] / "scripts/report_native_axis_decision_diagnostics.py")
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


def next_f32(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


def array(values, shape, dtype="<f4"):
    return {"values": copy.deepcopy(values), "shape": shape, "dtype": dtype}


def events():
    rows = []
    for update in (0, 1):
        for episode, seed in enumerate((101, 103)):
            for step in range(3):
                actions = [[1.] + [0.] * 14] + [
                    [0., float(index), float(step)] + [0.] * 12 for index in range(1, 6)]
                state = [float(step), 0., 0., 0., 0., 0.]
                ids = ["STOP", "one", "two", "three", "four", "five"]
                logits = [0., 5., 4., 3., 2., 1.] if update == 0 else [.5, 4., 5., 3., 1., 2.]
                chosen = 1 if update == 0 else 2
                inputs = {key: array(actions, [6, 15]) for key in
                          ("action_features", "source_action_features", "actor_action_features")}
                inputs.update({key: array(state, [6]) for key in ("state_features", "source_state_features")})
                inputs.update(action_ids=ids, action_mask=array([True] * 6, [6], "|b1"))
                payload = {"version": "learner-decision-observer-v1", "role": "selection",
                    "seed": seed, "panel": update, "update": update, "episode": episode, "step": step,
                    "forward_evaluated": True, "decision_rule": "deterministic_argmax", "forced_reason": None,
                    "inputs": inputs, "logits": array(logits, [6]), "value": float(step + update),
                    "selected_index": chosen, "selected_action_id": ids[chosen]}
                rows.append({"kind": "decision", "role": "selection", "seed": seed,
                    "status": "step_returned", "decision_id": f"constructed-{len(rows)}", "payload": payload})
    return rows


def diagnose(rows):
    return owner.diagnose_pairs(rows, [101, 103], "initial", "updated")


def set_logits(payload, values):
    payload["logits"]["values"] = values
    chosen = max(range(len(values)), key=values.__getitem__)
    payload["selected_index"] = chosen
    payload["selected_action_id"] = payload["inputs"]["action_ids"][chosen]


def test_independent_grouping_does_not_count_repeat_worlds_as_distinct_states():
    report = diagnose(events())
    assert report["selection_record_pairs"] == 6
    assert report["unique_states"] == 3
    assert [group["pair_indices"] for group in report["unique_state_groups"]] == [[0, 3], [1, 4], [2, 5]]
    assert not report["all_chosen_actions_unchanged"]


def test_event_order_does_not_change_declared_pair_order_or_denominators():
    rows = events()
    assert diagnose(rows) == diagnose(list(reversed(rows)))


def test_alias_count_counts_actions_not_combinatorial_pairs_and_excludes_stop():
    rows = events()
    for event in rows:
        for key in ("action_features", "source_action_features", "actor_action_features"):
            matrix = event["payload"]["inputs"][key]["values"]
            matrix[2] = copy.deepcopy(matrix[1])
            matrix[3] = copy.deepcopy(matrix[1])
            matrix[5] = copy.deepcopy(matrix[4])
        set_logits(event["payload"], [0., 3., 3., 3., 2., 2.])
    for pair in diagnose(rows)["pairs"]:
        assert pair["unique_nonstop_feature_rows"] == 2
        assert pair["aliased_nonstop_actions"] == pair["legal_nonstop_actions"] == 5
        assert sorted(len(group["indices"]) for group in pair["alias_groups"]) == [2, 3]
        assert all("STOP" not in group["action_ids"] for group in pair["alias_groups"])


@pytest.mark.parametrize("replacement", [next_f32(1.), -0.])
def test_exact_feature_bytes_do_not_merge_one_ulp_or_signed_zero(replacement):
    rows = events()
    for event in rows:
        for key in ("action_features", "source_action_features", "actor_action_features"):
            matrix = event["payload"]["inputs"][key]["values"]
            matrix[2] = copy.deepcopy(matrix[1])
            column = 1 if replacement > 0 else 4
            matrix[2][column] = replacement
    assert all(pair["alias_groups"] == [] for pair in diagnose(rows)["pairs"])


def test_one_ulp_logit_difference_is_not_a_tie():
    rows = events()
    for event in rows:
        set_logits(event["payload"], [0., 1., next_f32(1.), -.5, -1., -2.])
    for pair in diagnose(rows)["pairs"]:
        assert pair["initial_chosen_id"] == "two"
        assert pair["initial_top_two_margin"] == next_f32(1.) - 1.


def test_exact_ties_retain_inventory_order_and_refuse_later_equal_choice():
    rows = events()
    for event in rows:
        set_logits(event["payload"], [0., 5., 5., 3., 2., 1.])
    pair = diagnose(rows)["pairs"][0]
    assert pair["initial_chosen_id"] == "one"
    assert pair["initial_top_two_margin"] == 0
    assert [row["initial_rank"] for row in pair["action_rows"]] == [6, 1, 2, 3, 4, 5]
    rows[0]["payload"].update(selected_index=2, selected_action_id="two")
    with pytest.raises(ValueError, match="first argmax"):
        diagnose(rows)


def test_common_exact_logit_shift_has_zero_centered_delta():
    rows = events()
    for event in rows:
        set_logits(event["payload"], [value + 16 * event["payload"]["update"] for value in [0., 5., 4., 3., 2., 1.]])
    for pair in diagnose(rows)["pairs"]:
        assert pair["rank_changed_actions"] == 0
        assert pair["max_absolute_centered_logit_delta"] == 0
        assert all(row["logit_delta"] == 16 for row in pair["action_rows"])


def test_one_ulp_different_paired_state_is_not_nearly_equal():
    rows = events()
    for key in ("state_features", "source_state_features"):
        rows[7]["payload"]["inputs"][key]["values"][0] = next_f32(1.)
    with pytest.raises(ValueError, match="pairing refused"):
        diagnose(rows)


def test_repeated_identical_state_cannot_hide_one_ulp_output_disagreement():
    rows = events()
    rows[4]["payload"]["value"] = next_f32(1.)
    with pytest.raises(ValueError, match="inconsistent saved deterministic outputs"):
        diagnose(rows)


@pytest.mark.parametrize("bad_mask", [False, 1])
def test_masked_or_nonboolean_row_is_not_counted_as_certified(bad_mask):
    rows = events()
    rows[0]["payload"]["inputs"]["action_mask"]["values"][5] = bad_mask
    with pytest.raises(ValueError):
        diagnose(rows)


def test_duplicate_state_slot_is_refused_even_with_a_fresh_decision_id():
    rows = events()
    rows[-1] = copy.deepcopy(rows[-2])
    rows[-1]["decision_id"] = "fresh-identifier-same-state-slot"
    with pytest.raises(ValueError, match="Duplicate saved selection state"):
        diagnose(rows)


def test_duplicate_decision_id_is_refused_even_in_distinct_state_slots():
    rows = events()
    rows[-1]["decision_id"] = rows[0]["decision_id"]
    with pytest.raises(ValueError, match="(?i)decision.*(identifier|identity|id|unique)|duplicate"):
        diagnose(rows)


def test_gzip_evidence_requires_exact_raw_bytes_and_pinned_digest(tmp_path):
    payload = b'{"evidence":"constructed-only"}\n'
    compressed = tmp_path / "record.json.gz"
    compressed.write_bytes(gzip.compress(payload, mtime=0))
    reader = owner.Evidence(tmp_path)
    assert reader.raw("record.json", hashlib.sha256(payload).hexdigest()) == payload
    with pytest.raises(ValueError, match="Pinned evidence bytes changed"):
        reader.raw("record.json", "0" * 64)
    (tmp_path / "record.json").write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="Raw/gzip disagreement"):
        reader.raw("record.json")
