"""Analytical and metadata-only controls; no patient decode or training."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_pat25_access as diagnostic
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask


def fixture_case():
    support = np.zeros((11, 11, 11), bool)
    support[1:10, 1:10, 1:10] = True
    target = np.zeros_like(support)
    target[2:9, 2:9, 2:9] = True
    image = np.indices(support.shape).sum(axis=0).astype(np.float32)
    access, derivation = diagnostic.prep.derive_access(target, support, np.eye(4), "sub-PAT25")
    source = NativeSpatialCase(image, support, target, np.eye(4), AccessWindow(**access),
        (ToolGeometry("analytic-fine", 1.25, .45, 20., 35., 2.),
         ToolGeometry("analytic-wide", 2.25, 1.1, 20., 35., 3.)),
        track="annotation_assisted", support_source_kind="supplied_annotation",
        support_derivation="analytical unchanged support", nominal_target=target,
        target_source_kind="supplied_annotation", target_derivation="analytical supplied target",
        crop_shape=(11, 11, 11), proposal_mode="nominal_cavity_v1")
    return source, diagnostic.six_accesses(derivation, np.eye(4), access), derivation, access


def test_six_existing_exits_preserve_order_and_selected_exact_identity():
    source, exits, derivation, access = fixture_case()
    assert [(row["axis"], row["outward_sign"]) for row in exits] == diagnostic.EXIT_ORDER
    assert exits[0]["selected_original"] and exits[0]["access"] == access
    for row in exits:
        axis, sign = row["axis"], row["outward_sign"]
        assert row["boundary_voxel"] == derivation["six_axis_exit_distances_mm"][row["exit_index"]]["boundary_voxel"]
        expected = np.zeros(3)
        expected[axis] = -sign
        np.testing.assert_array_equal(row["access"]["normal_inward"], expected)
        assert row["access"]["radius_mm"] == source.access.radius_mm == 6.


@pytest.mark.parametrize("damage", ["order", "distance", "boundary", "selected_id"])
def test_malformed_existing_exits_fail_closed(damage):
    _, _, derivation, access = fixture_case()
    if damage == "order":
        derivation["six_axis_exit_distances_mm"].reverse()
    elif damage == "distance":
        derivation["six_axis_exit_distances_mm"][0]["distance_mm"] += .01
    elif damage == "boundary":
        derivation["six_axis_exit_distances_mm"][0]["boundary_voxel"][1] += 1
    else:
        access["center_mm"][0] += .01
    with pytest.raises(ValueError):
        diagnostic.six_accesses(derivation, np.eye(4), access)


def test_trace_preserves_complete_inventory_acceptance_and_initial_state():
    source, exits, _, _ = fixture_case()
    plain = NativeSpatialTask(source, max_steps=3)
    expected = plain.candidate_inventory()
    made = []
    def factory():
        task = NativeSpatialTask(source, max_steps=3)
        made.append(task)
        return task
    record = diagnostic.inspect_task(factory, exits[0])
    actual = made[0]
    assert record["status"] == "complete" and record["initial_inventory"] == expected
    assert record["executed_transitions"] == record["commits"] == 0
    assert len(record["trace"]) == expected["emitted_count"]
    assert record["exterior_neighbor"]["connected_to_external_free"]
    for name in ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"):
        np.testing.assert_array_equal(getattr(actual._engine, name), getattr(plain._engine, name))
    assert actual._engine.state_hash == plain._engine.state_hash
    assert not actual._engine.history and actual._steps == 0
    assert sys.getprofile() is None


def blocked_engine():
    support = np.zeros((7, 7, 8), bool)
    support[3, 3, 1:7] = True
    # The intentionally internal access has an uncut proximal shaft obstacle.
    config = NativeResectionConfig(support, np.zeros_like(support, dtype=np.int16), np.eye(4),
        AccessWindow((3., 3., 3.5), (0., 0., 1.), 6., "analytic-internal-gap"),
        (ToolGeometry("analytic-shaft", 1.25, .45, 10., 35., 2.),),
        "sha256:" + "a" * 64, "analytical test, no patient")
    return NativeResectionEngine(config)


def test_trace_captures_actual_first_blocked_cells_without_reexecuting_or_removing():
    engine = blocked_engine()
    initial = engine.remaining_mask.copy()
    expected = engine.preview_stroke("analytic-shaft", (3., 3., 5.))
    with diagnostic.PreviewTrace() as trace:
        actual = engine.preview_stroke("analytic-shaft", (3., 3., 5.))
    assert actual.reason == expected.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
    assert actual.failure_tip_mm == expected.failure_tip_mm
    row, = trace.rows
    assert row["failure_tip_mm"] == [3., 3., 3.5]
    assert row["failure_step_index"] == row["completed_preview_microsteps"] == 0
    # The shaft's end cap extends from z=1.5 to 1.95, intersecting cell 2.
    assert row["blocked_cell_index_bounds"] == [[3, 3, 1], [3, 3, 2]]
    assert row["blocked_cell_ras_aabb_mm"] == [[2.5, 2.5, .5], [3.5, 3.5, 2.5]]
    assert row["blocked_cell_count"] == 2
    np.testing.assert_array_equal(engine.remaining_mask, initial)
    assert not engine.history and not engine.removed_mask.any()


def test_commit_is_refused_before_any_state_change():
    source, _, _, _ = fixture_case()
    task = NativeSpatialTask(source)
    result = next(iter(task._inventory.values()))
    with pytest.raises(RuntimeError, match="Commit is prohibited"):
        with diagnostic.PreviewTrace():
            task._engine.commit_preview(result)
    assert not task._engine.history and not task._engine.removed_mask.any()
    assert sys.getprofile() is None


def test_interrupted_trace_remains_partial_not_stop_only_success():
    source, exits, _, _ = fixture_case()
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("analytical cancellation after previews")
    with pytest.raises(TimeoutError) as caught:
        diagnostic.inspect_task(lambda: NativeSpatialTask(source), exits[0], check)
    receipt = caught.value.diagnostic
    assert receipt["status"] == "unassessed" and not receipt["inventory_complete"]
    assert receipt["partial_preview_count"] == 2
    assert "initial_inventory" not in receipt and sys.getprofile() is None


def test_cap_omissions_are_not_a_complete_anatomical_result():
    source, exits, _, _ = fixture_case()
    capped = replace(source, proposal_config=NominalCavityProposalConfig(max_candidates=1))
    with pytest.raises(RuntimeError, match="complete declared inventory") as caught:
        diagnostic.inspect_task(lambda: NativeSpatialTask(capped), exits[0])
    assert caught.value.diagnostic["status"] == "unassessed"


def test_enclosed_non_support_cell_is_not_labeled_external_free():
    source, _, _, _ = fixture_case()
    support = np.array(source.observed_support)
    support[5, 5, 5] = False
    target = np.zeros_like(support)
    target[4, 5, 5] = True
    access, derivation = diagnostic.prep.derive_access(target, support, np.eye(4), "sub-PAT25")
    exits = diagnostic.six_accesses(derivation, np.eye(4), access)
    selected = next(row for row in exits if row["selected_original"])
    source = replace(source, observed_support=support, reference_target=target,
                     nominal_target=target, access=AccessWindow(**access))
    report = diagnostic.inspect_task(lambda: NativeSpatialTask(source), selected)
    exterior = report["exterior_neighbor"]
    assert exterior["outside_neighbor_voxel"] == [5, 5, 5]
    assert not exterior["in_estimated_support"] and not exterior["connected_to_external_free"]
    assert report["executed_transitions"] == 0


@pytest.mark.parametrize("subject", ["sub-PAT05", "sub-PAT26", "sub-PAT27", "sub-PAT29"])
def test_other_subjects_reject_before_metadata_or_decode(monkeypatch, subject):
    def forbidden(*args, **kwargs):
        pytest.fail("No source read before PAT25 role gate")
    monkeypatch.setattr(diagnostic.prep, "read_development_cohort", forbidden)
    monkeypatch.setattr(diagnostic.prep, "_decode_case", forbidden)
    with pytest.raises(ValueError, match="fixed PAT25"):
        diagnostic.load_bound_case({"version": diagnostic.VERSION, "subject": subject,
            "role": "TRAIN", "settings": diagnostic.SETTINGS}, lambda: None)


def test_byte_mismatch_fails_before_decode(monkeypatch, tmp_path):
    path = tmp_path / "analytical.fixture"
    path.write_bytes(b"not a patient")
    monkeypatch.setattr(diagnostic, "validate", lambda record: ({}, {}))
    monkeypatch.setattr(diagnostic.prep, "_decode_case", lambda path: pytest.fail("No decode after byte mismatch"))
    with pytest.raises(ValueError, match="bundle bytes"):
        diagnostic.load_bound_case({"member": {"case_bundle": str(path), "case_bundle_sha256": "0" * 64}}, lambda: None)
