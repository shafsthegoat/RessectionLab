"""Generated-only leakage canary: no patient loader, optimizer, or FEBio call."""
import json

import pytest

from scripts import limited_input_leakage_canary_v1 as canary


@pytest.fixture(scope="module")
def result():
    return canary.run()


def test_fixed_estimate_private_truth_swap_preserves_all_deployment_facing_inputs(result):
    fixed = result["fixed_private_swap"]
    assert fixed["invariants_passed"]
    assert len(set(fixed["reference_hashes"])) == 2
    assert len(set(fixed["observation_fingerprints"])) == 1
    assert fixed["observation_array_hashes"][0] == fixed["observation_array_hashes"][1]
    assert fixed["candidate_ids"][0] == fixed["candidate_ids"][1]
    assert fixed["legal_masks"][0] == fixed["legal_masks"][1]
    assert fixed["search_decisions"][0] == fixed["search_decisions"][1]
    assert fixed["search_cost_stable"][0] == fixed["search_cost_stable"][1]
    assert fixed["private_evaluator_returns"][0] != fixed["private_evaluator_returns"][1]


def test_masking_only_target_image_exposes_annotation_dependent_proposals(result):
    negative = result["annotation_mask_only_negative"]
    assert negative["side_channel_exposed"]
    assert negative["masked_image_hashes"][0] == negative["masked_image_hashes"][1]
    assert negative["candidate_ids"][0] != negative["candidate_ids"][1]
    assert negative["endpoints"][0] != negative["endpoints"][1]
    # Root legal geometry can still coincide; the same physical opening makes
    # different target-derived strokes legal and changes the actor's tensor.
    after = negative["same_physical_opening_then_masked_actor"]
    assert after["masked_image_hashes"][0] == after["masked_image_hashes"][1]
    assert after["candidate_ids"][0] != after["candidate_ids"][1]
    assert after["action_geometry_hashes"][0] != after["action_geometry_hashes"][1]
    assert after["legal_masks"][0] != after["legal_masks"][1]
    assert after["search_decisions"][0] != after["search_decisions"][1]


def test_missing_nominal_fails_search_but_generic_actor_remains_available(result):
    missing = result["missing_estimate"]
    assert "ESSENTIAL_EVIDENCE_MISSING" in missing["nominal_cavity_constructor"]
    assert "ESSENTIAL_EVIDENCE_MISSING" in missing["generic_fixed_lattice_search"]
    assert "Exact finite generated-fixture arrays" in missing["sealed_limited_planning_spec"]
    assert missing["generic_fixed_lattice_actor_nonstop_actions"] > 0
    assert result["status"] == "diagnostic_complete_boundary_hold"


@pytest.mark.parametrize("field", ["source_sha256", "config", "expected"])
def test_mutated_frozen_declaration_fails_before_fixture_construction(tmp_path, monkeypatch, field):
    record = json.loads(canary.MANIFEST.read_text())
    if field == "source_sha256":
        record[field][canary.SOURCE_FILES[0]] = "0" * 64
    elif field == "config":
        record[field]["search"]["max_calls"] += 1
    else:
        record[field]["negative_control"] = "different outcome after seeing labels"
    path = tmp_path / "changed-manifest.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(canary, "make_native_opening_task",
        lambda: pytest.fail("Changed declaration reached generated fixture"))
    with pytest.raises(ValueError, match="frozen source hash changed"):
        canary.run(path)


def test_canary_rejects_an_accidental_private_truth_handoff(monkeypatch):
    original = canary._case

    def leaked(base, reference, *, nominal, annotation):
        return original(base, reference, nominal=reference, annotation=annotation)

    monkeypatch.setattr(canary, "_case", leaked)
    with pytest.raises(RuntimeError, match="CANARY_INVARIANT_FAILED"):
        canary.run()
