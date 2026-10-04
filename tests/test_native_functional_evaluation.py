"""Analytic integration controls for sealing and independent full-tool outcomes."""
import copy
import json

import pytest
import torch

from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS
from resectionlab import native_functional_evaluation as events
from resectionlab.native_refinement import (evaluate_native_refinement_candidate,
    recheck_native_replay, replay_artifact_hash, run_native_refinement)
from resectionlab.native_simulation import make_native_patient_simulator
from resectionlab.worlds import content_hash
from test_functional_native_bridge import case_with_evidence


@pytest.fixture
def selected(tmp_path):
    torch.set_num_threads(1)
    case = case_with_evidence()
    directory = tmp_path / "run"
    options = dict(access=AccessWindow((3., 3., -.5), (0., 0., 1.), 4.),
        tools=(NATIVE_GENERIC_TOOLS[0],), selected_entry_mm=(3., 3., -.5),
        selected_target_mm=(3., 3., 4.), candidate_count=1, max_steps=1, max_actions=2)
    report = run_native_refinement(case, directory, budget_seconds=1., seed=11, **options)
    assert report["gradient_steps"] > 0
    assert report["replay_status"] == "accepted_independent_geometry"
    return case, directory, options, report["replay"]


def test_interruption_durably_seals_worlds_and_only_same_evaluation_can_resume(selected, monkeypatch):
    case, directory, options, replay = selected
    seen = []
    original = events._evaluate
    def interrupt(*args, **kwargs):
        assert (directory / events.SEAL_FILE).is_file()
        assert (directory.parent / events.LEDGER_FILE).is_file()
        assert seen, "External caller must receive the durable seal before outcomes"
        raise InterruptedError("analytical interruption before first outcome")
    monkeypatch.setattr(events, "_evaluate", interrupt)
    with pytest.raises(InterruptedError, match="first outcome"):
        evaluate_native_refinement_candidate(case, directory, replay, on_seal=seen.append, **options)
    assert not (directory / events.REPORT_FILE).exists()
    with pytest.raises(ValueError, match="FUNCTIONAL_EVALUATION_SEALED"):
        run_native_refinement(case, directory, budget_seconds=1., seed=11, resume=True, **options)
    seal_before = (directory / events.SEAL_FILE).read_bytes()
    monkeypatch.setattr(events, "_evaluate", original)
    result = evaluate_native_refinement_candidate(case, directory, replay, **options)
    assert (directory / events.SEAL_FILE).read_bytes() == seal_before
    again = evaluate_native_refinement_candidate(case, directory, result["replay"], **options)
    assert again["replay"]["functional_assessment"] == result["replay"]["functional_assessment"]
    report = result["replay"]["functional_assessment"]["event_report"]
    assert report["nonidentical_worlds_verified"]
    assert report["world_partition"]["role"] == "final_evaluation"
    assert all(event["denominator"] == 3 for event in report["candidates"][0]["model_events"])
    assert "language_evidence_unavailable" in report["candidates"][0]["unknowns"]
    assert "vascular_anatomy_unassessed" in report["candidates"][0]["unknowns"]


def test_reopening_recomputes_events_even_if_a_forged_report_has_new_hashes(selected):
    case, directory, options, replay = selected
    evaluated = evaluate_native_refinement_candidate(case, directory, replay, **options)["replay"]
    altered = copy.deepcopy(evaluated)
    assessment = altered["functional_assessment"]
    event = assessment["event_report"]["candidates"][0]["model_events"][0]
    event["numerator"] = (event["numerator"] + 1) % (event["denominator"] + 1)
    assessment["assessment_hash"] = content_hash({k: v for k, v in assessment.items() if k != "assessment_hash"})
    altered["artifact_hash"] = replay_artifact_hash(altered)
    with pytest.raises(ValueError, match="outcomes do not reproduce"):
        recheck_native_replay(case, altered, **options)


def test_shared_ledger_rejects_a_new_candidate_on_already_revealed_worlds(selected):
    case, directory, options, replay = selected
    completed = evaluate_native_refinement_candidate(case, directory, replay, **options)
    sealed = completed["replay"]["functional_assessment"]["seal"]
    partitions = events.partitions_from_record(sealed["partitions"])
    changed = copy.deepcopy(replay)
    changed["checkpoint_hash"] = "different-candidate-checkpoint-control"
    native = make_native_patient_simulator(case, **options).native_config
    next_directory = directory.parent / "new-run"
    next_directory.mkdir()
    with pytest.raises(ValueError, match="EVALUATION_WORLDS_REVEALED"):
        events.evaluate_selected_native_sequence(case, changed, native, partitions, next_directory)
    assert not (next_directory / events.REPORT_FILE).exists()


def test_edited_final_partition_cannot_reset_worlds_before_sealing(selected):
    case, directory, options, replay = selected
    path = directory / "native-world-partitions.json"
    record = json.loads(path.read_text())
    record["final_evaluation"]["seeds"][0] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="predeclared native run"):
        evaluate_native_refinement_candidate(case, directory, replay, **options)
    assert not (directory / events.SEAL_FILE).exists()
