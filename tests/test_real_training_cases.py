"""Tiny source/geometry controls; never load a real patient or train a policy."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_real_training_cases as prep
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.structural_evidence import StructuralEvidence, structural_frame_hash

STAMP = "2026-10-04T23:30:00+00:00"


@pytest.fixture
def prepared_source(monkeypatch, tmp_path):
    """Fabricated IO fixture with real task geometry; no patient data."""
    image = np.indices((9, 9, 9)).sum(axis=0).astype(np.float32)
    support = np.zeros(image.shape, bool)
    support[1:8, 1:8, 1:8] = True
    target = np.zeros(image.shape, bool)
    target[4, 4, 4] = True
    cohort = prep.read_development_cohort(prep.ROOT / prep.COHORT_PATH)
    case = CaseData("BTC-ds001226-sub-PAT22-preop", image, {"supplied_target": target}, np.eye(4),
                    (SourceRef("structural", "generated://analytic-preparation-fixture", provenance="observed"),),
                    metadata={"structural_coverage": "full_head", "source_collection": cohort["source"]})
    evidence = StructuralEvidence("synthstrip_main_unit", support, array_digest(case.mri),
                                  structural_frame_hash(case), None, "sha256:" + "1" * 64,
                                  "sha256:" + "2" * 64, "analytic support model fixture")
    case = case.revised(structural_evidence={evidence.evidence_id: evidence})
    path = tmp_path / "case.fixture"
    path.write_bytes(b"stand-in for encoded analytic source")
    member = {"subject": "sub-PAT22", "role": "TRAIN", "case_bundle": str(path),
              "case_bundle_sha256": prep.sha256(path), "case_semantic_hash": case.semantic_hash,
              "planning_hash": case.planning_hash, "evidence_id": evidence.evidence_id,
              "evidence_hash": evidence.evidence_hash}
    monkeypatch.setattr(prep, "known_members", lambda: {"sub-PAT22": copy.deepcopy(member)})
    monkeypatch.setattr(prep, "_decode_case", lambda path: case)
    return case, member, evidence


def test_same_axis_rule_uses_lexicographic_cell_then_physical_distance():
    support = np.zeros((9, 9, 9), bool)
    support[1:8, 1:8, 1:8] = True
    target = np.zeros_like(support)
    target[3, 4, 4] = target[5, 4, 4] = True
    access, record = prep.derive_access(target, support, np.diag([2., 1., 1., 1.]), "sub-PAT22")
    assert record["representative_voxel"] == [3, 4, 4]
    # The nearer x-index face is farther in millimetres; y wins the y/z tie.
    assert access["center_mm"] == [6., .5, 4.]
    assert access["normal_inward"] == [0., 1., 0.]
    assert access["radius_mm"] == 6.
    assert record["depth_to_annotation_representative_mm"] == 3.5
    assert len(record["six_axis_exit_distances_mm"]) == 6


def test_axis_rule_preserves_oblique_physical_landmark_and_input_arrays():
    support = np.ones((5, 5, 5), bool)
    target = np.zeros_like(support)
    target[2, 2, 2] = True
    affine = np.array([[0., -1., 0., 10.], [1., 0., 0., 20.], [0., 0., 1., 30.], [0., 0., 0., 1.]])
    originals = target.copy(), support.copy(), affine.copy()
    access, record = prep.derive_access(target, support, affine, "sub-PAT22")
    assert record["selected_boundary_voxel"] == [-.5, 2., 2.]
    assert access["center_mm"] == [8., 19.5, 32.]
    assert access["normal_inward"] == [0., 1., 0.]
    for first, second in zip(originals, (target, support, affine)):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("subject", ["sub-PAT05", "sub-PAT26", "sub-PAT27", "sub-PAT29", "sub-PAT31", "../PAT22"])
def test_forbidden_subject_refuses_before_registry_or_bundle(monkeypatch, subject):
    def forbidden(*args, **kwargs):
        pytest.fail("No input read is permitted before the subject gate")
    monkeypatch.setattr(prep, "read_development_cohort", forbidden)
    monkeypatch.setattr(prep, "known_members", forbidden)
    monkeypatch.setattr(prep, "_decode_case", forbidden)
    with pytest.raises(ValueError, match="five remaining TRAIN"):
        prep.prepare_training_case(subject, declared_at=STAMP)


def test_real_native_initial_preparation_and_frozen_reload(prepared_source):
    case, _, _ = prepared_source
    original = case.semantic_hash
    task, record = prep.prepare_training_case("sub-PAT22", declared_at=STAMP)
    assert record["status"] == "prepared" and record["executed_transitions"] == 0
    assert record["coverage"]["full_target_source_cells"] == 1
    assert record["coverage"]["target_outside_actor_crop_source_cells"] == 0
    assert record["initial_inventory"]["declared_slots"] == 78
    assert record["initial_inventory"]["complete"]
    assert record["binding"]["common_task"]["max_steps"] == 3
    assert record["binding"]["common_task"]["adapter_options"]["crop_shape"] == [64, 64, 64]
    assert task.metrics()["steps"] == 0 and not task._engine.history
    assert case.semantic_hash == original and case.brain_mask is None
    rebuilt = prep.load_prepared_training_case(record)
    assert rebuilt.observation().fingerprint == task.observation().fingerprint
    assert rebuilt.decision_model_hash == task.decision_model_hash


def test_support_conflict_retains_full_target_and_never_constructs_native(prepared_source, monkeypatch):
    case, member, _ = prepared_source
    target = np.array(case.compartments["supplied_target"])
    target[0, 0, 0] = True
    changed = case.revised(compartments={"supplied_target": target})
    # This fixture simulates a source-bound proposal with incomplete inclusion,
    # as in the retained PAT16/PAT20 receipts. Its image/frame did not change.
    member.update(case_semantic_hash=changed.semantic_hash, planning_hash=changed.planning_hash)
    monkeypatch.setattr(prep, "_decode_case", lambda path: changed)
    monkeypatch.setattr(prep, "_construct_task", lambda *a: pytest.fail("Support conflict must precede native previews"))
    task, record = prep.prepare_training_case("sub-PAT22", declared_at=STAMP)
    assert task is None and record["status"] == "blocked_support_conflict"
    assert record["failure_code"] == "TARGET_OUTSIDE_SUPPORT"
    assert record["coverage"]["full_target_source_cells"] == 2
    assert record["coverage"]["target_outside_support_source_cells"] == 1
    assert record["coverage"]["reference_or_nominal_clipped"] is False
    assert "access" not in record["binding"]["member"]
    with pytest.raises(ValueError, match="successful"):
        prep.load_prepared_training_case(record)


@pytest.mark.parametrize("field", ["access", "common_task", "coverage"])
def test_resealed_binding_cannot_change_access_task_or_coverage(prepared_source, field):
    _, record = prep.prepare_training_case("sub-PAT22", declared_at=STAMP)
    if field == "access":
        record["binding"]["member"]["access"]["radius_mm"] += .1
    elif field == "common_task":
        record["binding"]["common_task"]["max_steps"] = 4
    else:
        record["coverage"]["full_target_source_cells"] = 900
    record["binding_hash"] = prep.binding_hash(record["binding"])
    with pytest.raises(ValueError, match="binding"):
        prep.load_prepared_training_case(record)


def test_cancellation_prevents_decode_or_partial_success(prepared_source, monkeypatch):
    monkeypatch.setattr(prep, "_decode_case", lambda path: pytest.fail("Cancelled request cannot decode a source"))
    with pytest.raises(InterruptedError):
        prep.prepare_training_case("sub-PAT22", declared_at=STAMP, cancelled=lambda: True)


def test_missing_source_bytes_refuses_before_decode(prepared_source, monkeypatch):
    _, member, _ = prepared_source
    member["case_bundle_sha256"] = "0" * 64
    monkeypatch.setattr(prep, "_decode_case", lambda path: pytest.fail("Mismatch must precede decode"))
    with pytest.raises(ValueError, match="byte identity"):
        prep.prepare_training_case("sub-PAT22", declared_at=STAMP)
