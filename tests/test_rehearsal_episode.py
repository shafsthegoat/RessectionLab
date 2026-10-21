"""Independent invariants for the bounded source-bound research runtime."""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest, semantic_digest
from resectionlab.geometry import ToolGeometry, ToolPose
from resectionlab.rehearsal_episode import (AUTHORITY, _synchronized_collision, create_episode,
                                           execute_alternatives, restore_episode)


def source_case(**changes):
    shape = (32, 32, 48)
    brain = np.zeros(shape, bool)
    brain[3:29, 3:29, 4:44] = True
    target = np.zeros(shape, bool)
    target[12:18, 12:18, 20:36] = True
    params = dict(case_id="generated-software-control", mri=brain.astype(np.float32),
                  compartments={"visible_target": target}, source_compartments={"private_reference": target},
                  brain_mask=brain, affine=np.diag([1., 1., 1., 1.]),
                  source_refs=(SourceRef("fixture", "generated://fixture", provenance="simulated", native_frame="RAS+"),),
                  metadata={"is_synthetic": True, "anatomy_layers": {"visible_target": {"kind": "tumor", "provenance": "simulated"}}})
    params.update(changes)
    return CaseData(**params)


def move(episode, tool, vector, duration=2., kind="move"):
    return {"kind": kind, "toolId": tool, "durationSeconds": duration,
            "tipRasMm": (episode.poses[tool].tip_mm + np.asarray(vector)).tolist()}


def assert_accounting(episode):
    m = episode.metrics()
    assert m["sourceTissueMm3"] == m["remainingTissueMm3"] + m["assumedExposureTissueMm3"] + m["removedMm3"]
    assert m["sourceTargetMm3"] == m["residualTargetMm3"] + m["assumedExposureTargetMm3"] + m["targetRemovedMm3"]
    assert m["removedMm3"] == m["targetRemovedMm3"] + m["nonTargetRemovedMm3"]
    assert not episode.removed.intersection(episode.assumed_exposure)


def test_source_is_unchanged_and_frame_binding_rejects_another_case():
    case = source_case()
    hashes = [array_digest(x) for x in (case.mri, case.affine, case.brain_mask, *case.compartments.values(), *case.source_compartments.values())]
    episode = create_episode(case)
    assert episode.step(move(episode, "aspirator", 6 * episode.direction, 4., "aspirate"))["accepted"]
    assert hashes == [array_digest(x) for x in (case.mri, case.affine, case.brain_mask, *case.compartments.values(), *case.source_compartments.values())]
    record = episode.serialize()
    with pytest.raises(ValueError, match="source identity"):
        restore_episode(replace(case, case_id="other-patient"), record)
    changed = np.array(case.affine)
    changed[0, 3] += 1.
    with pytest.raises(ValueError, match="source identity"):
        restore_episode(replace(case, affine=changed), record)
    with pytest.raises(ValueError):
        episode.target.flat[0] = True
    assert_accounting(episode)


def test_persistent_tools_clock_removal_field_management_and_pending_events():
    episode = create_episode(source_case())
    parked = episode.poses["field_manager"]
    assert episode.step(move(episode, "aspirator", 6 * episode.direction, 4., "aspirate"))["accepted"]
    assert np.array_equal(episode.poses["field_manager"].tip_mm, parked.tip_mm)
    removed = set(episode.removed)
    burden = episode.field_burden
    assert removed and burden > 0 and episode.pending_events
    assert episode.step(move(episode, "field_manager", .5 * episode.direction, 2., "clear_field"))["accepted"]
    assert episode.removed == removed
    assert episode.field_managed and episode.field_burden < burden
    history = dict(episode.field_managed)
    assert episode.step({"kind": "inspect", "durationSeconds": 1.})["accepted"]
    assert episode.field_managed == history and episode.elapsed == 7.
    assert episode.step({"kind": "reassess", "durationSeconds": 1.})["accepted"]
    assert episode.last_inspection == 8.
    assert_accounting(episode)
    assert episode.step({"kind": "STOP", "durationSeconds": 0.})["accepted"]
    stopped = episode.state_id
    assert not episode.step({"kind": "inspect", "durationSeconds": 1.})["accepted"]
    assert episode.state_id == stopped


def test_branch_identical_snapshot_replay_and_sparse_delta_prefixes():
    episode = create_episode(source_case())
    assert episode.step({"kind": "inspect", "durationSeconds": .5})["accepted"]
    saved = episode.serialize()
    comparison = execute_alternatives(episode)
    assert episode.serialize() == saved
    assert comparison["identicalSnapshot"]
    results = []
    for branch in comparison["branches"]:
        assert branch["branchPointStateId"] == episode.state_id
        record = branch["episode"]
        reopened = restore_episode(episode.case, json.loads(json.dumps(record["savedEpisode"])))
        assert reopened.export() == record
        assert record["replayFrames"][:len(episode.frames)] == episode.frames
        seen = set()
        prior = record["initialStateId"]
        seconds = -1
        for f in record["replayFrames"]:
            assert f["stateBefore"] == prior
            assert f["elapsedSeconds"] >= seconds
            ids = {tuple(v) for v in f["removedIndicesNative"]}
            assert not ids & seen
            seen |= ids
            prior, seconds = f["stateAfter"], f["elapsedSeconds"]
        assert seen == {tuple(v) for v in record["finalRemovedIndicesNative"]}
        assert prior == record["finalStateId"]
        assert_accounting(reopened)
        results.append(record["metrics"]["removedMm3"])
    assert results[0] != results[1]
    damaged = copy.deepcopy(saved)
    damaged["finalStateId"] = "sha256:" + "0" * 64
    damaged["recordHash"] = semantic_digest({k: v for k, v in damaged.items() if k != "recordHash"})
    with pytest.raises(ValueError, match="exact final state"):
        restore_episode(episode.case, damaged)


def test_tool_tool_and_retained_tissue_collision_rejections_are_atomic():
    episode = create_episode(source_case())
    original = episode.serialize()
    result = episode.step(move(episode, "field_manager", -6 * episode.lateral))
    assert not result["accepted"] and "TOOL_COLLISION" in result["reason"]
    assert episode.serialize() == original
    result = episode.step(move(episode, "aspirator", 6 * episode.direction))
    assert not result["accepted"] and "RETAINED_TISSUE_COLLISION" in result["reason"]
    assert episode.serialize() == original
    # Distal tips are separated by >10 mm, but the parallel full shafts overlap.
    vector = -4.9 * episode.lateral - 10. * episode.direction
    result = episode.step(move(episode, "field_manager", vector, 4.))
    assert not result["accepted"] and "TOOL_COLLISION" in result["reason"]
    assert episode.serialize() == original


def test_synchronized_crossing_collides_between_clear_endpoints():
    a = ToolGeometry("a", .2, .2, 10., tip_length_mm=1.)
    b = ToolGeometry("b", .2, .2, 10., tip_length_mm=1.)
    starts = {"a": ToolPose((-2, 0, 4), (0, 0, 1)), "b": ToolPose((2, 0, 4), (0, 0, 1))}
    ends = {"a": ToolPose((2, 0, 4), (0, 0, 1)), "b": ToolPose((-2, 0, 4), (0, 0, 1))}
    # Analytic independent witness: both centerlines coincide at t=.5 while
    # endpoints have 4 mm separation, exceeding the total .4 mm radius.
    assert np.linalg.norm(starts["a"].tip_mm - starts["b"].tip_mm) == 4.
    assert np.linalg.norm(ends["a"].tip_mm - ends["b"].tip_mm) == 4.
    assert _synchronized_collision({"a": a, "b": b}, starts, ends, 8)
    ends = {"a": ToolPose((-2, 0, 5), (0, 0, 1)), "b": ToolPose((2, 0, 5), (0, 0, 1))}
    assert _synchronized_collision({"a": a, "b": b}, starts, ends, 8) is None


def test_synchronized_runtime_move_is_continuous_and_tools_persist():
    episode = create_episode(source_case())
    before = dict(episode.poses)
    motions = [{"toolId": key, "tipRasMm": (pose.tip_mm - 2 * pose.axis_unit).tolist()} for key, pose in before.items()]
    assert episode.step({"kind": "move", "motions": motions, "durationSeconds": 1.})["accepted"]
    for key in before:
        assert np.allclose(episode.poses[key].tip_mm, before[key].tip_mm - 2 * before[key].axis_unit)
    assert episode.elapsed == 1.
    assert not episode.removed
    frames = episode.frames[1:]
    assert len(frames) == 4
    for a, b in zip(episode.frames, episode.frames[1:]):
        for first, second in zip(a["tools"], b["tools"]):
            assert np.linalg.norm(np.array(second["tipRasMm"]) - first["tipRasMm"]) <= .5 + 1e-10


def test_forbidden_input_fields_and_privileged_reference_do_not_reach_planner():
    case = source_case()
    changed = replace(case, source_compartments={"private_reference": np.zeros(case.mri.shape, bool)},
                      metadata={**case.metadata, "future_pathology": "not-permitted", "outcome": "secret"})
    first, second = create_episode(case), create_episode(changed)
    assert first.planning_observation() == second.planning_observation()
    assert first.source_binding["caseHash"] != second.source_binding["caseHash"]
    before = first.state_id
    for field in ("private_reference", "reward", "future_pathology", "critical_truth", "referenceMask"):
        assert not first.step({"kind": "inspect", "durationSeconds": 1., field: 1})["accepted"]
        assert first.state_id == before
    public = json.dumps(first.planning_observation())
    assert "private_reference" not in public and "future_pathology" not in public and "sourceRefs" not in public
    assert first.metrics()["vascularEvents"] is None
    assert first.metrics()["motorRisk"] is None and first.metrics()["languageRisk"] is None


def test_motion_activation_exchange_and_authority_guards():
    episode = create_episode(source_case())
    assert not episode.step(move(episode, "aspirator", 20 * episode.direction, .1))["accepted"]
    assert not episode.step({"kind": "aspirate", "toolId": "field_manager", "durationSeconds": 1})["accepted"]
    assert not episode.step({"kind": "move", "toolId": "aspirator", "durationSeconds": 1, "axis": (1, 0, 0)})["accepted"]
    assert episode.step({"kind": "exchange", "toolId": "aspirator", "durationSeconds": 1})["accepted"]
    assert episode.step(move(episode, "aspirator", -episode.direction, .5))["accepted"]
    assert not episode.step({"kind": "exchange", "toolId": "aspirator", "durationSeconds": 1})["accepted"]
    with pytest.raises(ValueError, match="authority"):
        create_episode(source_case(), research_authority="disable_policy")
    with pytest.raises(ValueError, match="Protected"):
        create_episode(replace(source_case(), case_id="UPENN-GBM-0001"))
    for role in ("external_holdout", "final_evaluation", "stress_test", "restricted"):
        case = source_case()
        with pytest.raises(ValueError, match="Protected"):
            create_episode(replace(case, metadata={**case.metadata, "split": {"role": role}}))


def test_full_shaft_sweeps_acquired_constraint_between_clear_endpoints():
    case = source_case()
    obstacle = np.zeros(case.mri.shape, bool)
    obstacle[21, 12, 10] = True
    modified = replace(case, compartments={**case.compartments, "observed_vessel": obstacle},
                       metadata={**case.metadata, "anatomy_layers": {**case.metadata["anatomy_layers"],
                           "observed_vessel": {"kind": "vessels", "source_sha256": "1" * 64,
                                               "source_role": "generated_geometry_control"}}})
    episode = create_episode(modified)
    tip = episode.poses["field_manager"].tip_mm
    assert np.linalg.norm(tip - np.array([21, 12, 10])) > 6.
    before = episode.state_id
    result = episode.step(move(episode, "field_manager", 6 * episode.lateral, 2.))
    assert not result["accepted"] and "ANATOMICAL_COLLISION" in result["reason"]
    assert episode.state_id == before
    # Moving the observed obstacle onto the initial shaft rejects creation even
    # though it is far from the active tip and within assumed exposure volume.
    obstacle = np.zeros(case.mri.shape, bool)
    obstacle[12, 12, 10] = True
    with pytest.raises(ValueError, match="Initial post-exposure"):
        create_episode(replace(modified, compartments={**modified.compartments, "observed_vessel": obstacle}))


def test_stop_after_budget_roundtrip_and_public_observation_detachment():
    episode = create_episode(source_case())
    public = episode.planning_observation()
    original = episode.state_id
    public["targetNames"].append("not-in-source")
    public["tools"][0]["tipRasMm"][0] += 100.
    assert episode.state_id == original
    for _ in range(128):
        assert episode.step({"kind": "inspect", "durationSeconds": .001})["accepted"]
    assert not episode.step({"kind": "inspect", "durationSeconds": .001})["accepted"]
    assert episode.step({"kind": "STOP", "durationSeconds": 0.})["accepted"]
    assert restore_episode(episode.case, episode.serialize()).state_id == episode.state_id


def test_malformed_motion_payload_is_an_atomic_rejection():
    episode = create_episode(source_case())
    before = episode.state_id
    for motions in ([1], [{}], [{"toolId": "aspirator", "tipRasMm": [float("nan"), 0, 0]}]):
        assert not episode.step({"kind": "move", "durationSeconds": 1., "motions": motions})["accepted"]
        assert episode.state_id == before


@pytest.mark.parametrize("declaration", [
    {"role": "SELECT"}, {"role": "EVAL"}, {"role": "test"}, {"role": "MEASUREMENT_EVAL"},
    {"split": "SELECT"}, {"split": {"partition": "test"}},
    {"split": {"role": "development", "development_role": "checkpoint_selection_development"}},
    {"cohort": {"binding": {"role": {"name": "SELECT"}}}},
    {"split": {"role": {"current": {"value": "EVAL"}}}},
    {"source_files": [{"source_role": "fixed_test"}]},
    {"source_assets": [{"provenance": {"role": "MEASUREMENT_EVAL"}}]},
    {"role": "held-out"}, {"role": "validation"}, {"role": "UNOPENED_TRANSFER"},
    {"source_assets": [{"policy": {"restricted": True}}]},
    {"roles": {"TRAIN": True, "SELECT": True}}, {"split": {"name": "SELECT"}},
    {"split": {"SELECT": True}},
    {"roles": {"SELECT": "true"}}, {"roles": {"SELECT": 1}},
    {"roles": {"SELECT": "false"}}, {"roles": {"SELECT": 0}},
    {"parent": {"split": {"MEASUREMENT_EVAL": "enabled"}}},
    {"parent": {"restricted": "true"}},
])
def test_nested_protected_role_rejected_before_geometry_without_source_access(monkeypatch, declaration):
    import builtins
    import resectionlab.rehearsal_episode as runtime
    case = source_case()
    case = replace(case, metadata={**case.metadata, **declaration})

    def forbidden_access(*args, **kwargs):
        pytest.fail("Protected metadata must be refused before source access or geometry")

    monkeypatch.setattr(builtins, "open", forbidden_access)
    monkeypatch.setattr(runtime, "resolve_critical_evidence", forbidden_access)
    with pytest.raises(ValueError, match="Protected"):
        create_episode(case)


@pytest.mark.parametrize("identity,metadata", [
    ("BTC-ds001226-sub-PAT26-preop", {}),
    ("BTC-ds001226-sub-PAT27-preop", {"role": "TRAIN"}),
    ("ReMIND-013", {}), ("ReMIND-037", {"split": {"role": "TRAIN"}}),
    ("renamed-fixture", {"binding": {"patient_group": "BTC:sub-PAT26"}}),
    ("renamed-fixture", {"source": {"patientGroup": "ReMIND:037"}}),
    ("UPENN-GBM-0001", {}),
    ("renamed-fixture", {"source_collection": {"name": "UPENN-GBM"}}),
    ("renamed-fixture", {"parent": {"source_collection": {"name": "UPENN-GBM"}}}),
    ("renamed-fixture", {"source_collection": {"name": "UTSW"}}),
    ("renamed-fixture", {"parent": {"sourceDataset": {"identifier": "UTSW"}}}),
    ("renamed-fixture", {"parent": {"collectionName": "UPENN-GBM"}}),
])
def test_locked_select_and_upenn_identities_cannot_be_relabelled(monkeypatch, identity, metadata):
    import builtins
    import resectionlab.rehearsal_episode as runtime
    case = source_case()
    case = replace(case, case_id=identity, metadata={**case.metadata, **metadata})

    def forbidden_access(*args, **kwargs):
        pytest.fail("Identity exclusion must precede source access or geometry")

    monkeypatch.setattr(builtins, "open", forbidden_access)
    monkeypatch.setattr(runtime, "resolve_critical_evidence", forbidden_access)
    with pytest.raises(ValueError, match="Protected"):
        create_episode(case)


@pytest.mark.parametrize("declaration", [
    {}, {"role": "TRAIN"}, {"split": "DEVELOPMENT"},
    {"split": {"role": "development_demo", "reason": "Not final evaluation or held-out evidence",
               "external_holdout_eligible": False}},
    {"split": {"role": "development", "development_role": "population_training"}},
    {"cohort": {"binding": {"role": {"name": "TRAIN"}}}},
    {"source_files": [{"source_role": "imported_source_annotation"}]},
    {"roles": {"TRAIN": True, "SELECT": False}}, {"split": {"name": "TRAIN"}},
    {"roles": {"TRAIN": 1, "SELECT": None}},
    {"parent": {"source_collection": {"name": "Generated", "notes": "No UPENN or UTSW data used"}}},
])
def test_training_development_and_legacy_absent_role_remain_admitted(declaration):
    case = source_case()
    case = replace(case, metadata={**case.metadata, **declaration})
    episode = create_episode(case)
    assert episode.metrics()["removedMm3"] == 0.
