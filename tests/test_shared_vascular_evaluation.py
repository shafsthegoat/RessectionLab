"""Generated-only post-seal vascular encounter and non-leakage controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
import json
import tempfile

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest
from resectionlab.development_episode import execute_development_episode, make_development_task
from resectionlab import shared_vascular_evaluation as v


@pytest.fixture
def script():
    return execute_development_episode()[1]


@pytest.fixture
def output():
    root = Path(__file__).resolve().parents[1] / "build"
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def _binding(episode):
    return v._binding(episode, semantic_digest(episode["history"]))


def _changed_reference(episode, *, positive=False, unknown=False, short_grid=False):
    shape = (13, 13, 12) if short_grid else v.SHAPE
    affine = np.eye(4) if short_grid else np.asarray(v.AFFINE)
    mask = np.zeros(shape, bool)
    coverage = np.ones(shape, bool)
    if positive:
        mask[(6, 6, 2) if short_grid else (12, 12, 34)] = True
    if unknown:
        coverage[(6, 6, 3) if short_grid else (12, 12, 36)] = False
    binding = replace(_binding(episode),
        mask_hash=array_digest(mask), coverage_hash=array_digest(coverage),
        reference_frame_hash=semantic_digest({"shape": list(shape),
            "affine_ras_mm": affine.tolist()}))
    return binding, v.GeneratedReference(binding, mask, coverage, affine)


def test_post_seal_contact_sidecar_binds_same_replay_and_separates_probe_removal(script, output):
    result = v.evaluate_development_episode_vascular(episode=script,
        output_directory=output / "success")
    assert result["status"] == "evaluated_generated_vascular_reference"
    assert result["evaluationId"] == semantic_digest({k: x for k, x in result.items()
                                                       if k != "evaluationId"})
    assert result["episodeId"] == script["episodeId"]
    assert result["strategySeal"] == script["planning"]["strategySeal"]
    assert result["physicalHistoryHash"] == semantic_digest(script["history"])
    assert json.loads(result["physicalHistoryCanonicalJson"]) == script["history"]
    assert result["actionIds"] == [row["action_id"] for row in script["history"]]
    assert len(result["perAction"]) == len(script["history"]) == 6
    assert [(row["actionIndex"], row["actionId"], row["interactionMode"]) for row in result["perAction"]] == [
        (i, native["action_id"], native["interaction_mode"]) for i, native in enumerate(script["history"])]
    assert all(row["sweepCount"] == 1 for row in result["perAction"][:-1])
    assert result["perAction"][-1]["sweepCount"] == 0
    assert all(result["perAction"][-1][part] is None for part in ("shaft", "tip", "wholeTool"))
    for i in (1, 3):
        assert script["history"][i]["interaction_mode"] == "probe"
        assert script["history"][i]["removed_indices_native"] == []
        assert result["perAction"][i]["wholeTool"]["touched_reference_cells"] > 0
    assert result["wholeTool"]["positive_reference_cells"] == 1
    assert result["shaft"]["positive_reference_cells"] == 1
    assert result["tip"]["positive_reference_cells"] == 1
    # Shaft and tip unions overlap; whole-tool positives are not their sum.
    assert result["wholeTool"]["positive_reference_cells"] < (
        result["shaft"]["positive_reference_cells"] + result["tip"]["positive_reference_cells"])
    assert result["wholeTool"]["unknown_reference_cells"] == 1
    assert result["wholeTool"]["biological_vessel_free"] is None
    assert result["clinicalInjuryProbability"] is None
    assert result["removedOverlap"]["outcomes"] is None
    assert (output / "success" / "attempt.json").exists()
    assert json.loads((output / "success" / "report.json").read_text()) == result


@pytest.mark.parametrize("mutation", ["action", "mode", "microstep", "source", "affine",
                                      "strategy", "tool", "access", "source_binding"])
def test_tampered_episode_never_calls_private_loader(script, output, mutation):
    changed = copy.deepcopy(script)
    if mutation == "action":
        changed["history"][0]["action_id"] = "STOP"
    elif mutation == "mode":
        changed["history"][1]["interaction_mode"] = "aspirate"
    elif mutation == "microstep":
        changed["history"][0]["microsteps"][0]["tip_end_mm"][2] += .1
    elif mutation == "source":
        changed["sourceHash"] = "sha256:" + "1" * 64
    elif mutation == "affine":
        changed["affine"][0][3] += 1.
    elif mutation == "tool":
        changed["tools"][0]["shaft_radius_mm"] += .1
    elif mutation == "access":
        changed["access"]["radius_mm"] += .1
    elif mutation == "source_binding":
        changed["sourceBinding"]["private_reference_published"] = True
    else:
        changed["planning"]["strategy"]["actions"][0] = "STOP"
    changed["episodeId"] = semantic_digest({k: x for k, x in changed.items() if k != "episodeId"})
    called = []
    with pytest.raises(ValueError):
        v._evaluate_development_episode_vascular(episode=changed,
            output_directory=output / mutation, reference_binding=_binding(changed),
            load_reference=lambda: called.append(True))
    assert called == []
    assert not (output / mutation).exists()


def test_private_reference_perturbation_changes_only_sidecar_not_episode_or_actions(script, output):
    original = json.dumps(script, sort_keys=True, separators=(",", ":"))
    observed = make_development_task().observation().fingerprint
    reference_binding, reference = _changed_reference(script, positive=False, unknown=False)
    calls = []
    alternative = v._evaluate_development_episode_vascular(episode=script,
        output_directory=output / "alternative", reference_binding=reference_binding,
        load_reference=lambda: (calls.append(True), reference)[1])
    default = v.evaluate_development_episode_vascular(episode=script,
        output_directory=output / "default")
    assert calls == [True]
    assert alternative["wholeTool"]["positive_reference_cells"] == 0
    assert alternative["wholeTool"]["annotated_positive_encounter"] is False
    assert default["wholeTool"]["positive_reference_cells"] == 1
    assert json.dumps(script, sort_keys=True, separators=(",", ":")) == original
    assert make_development_task().observation().fingerprint == observed
    assert alternative["episodeId"] == default["episodeId"] == script["episodeId"]
    assert alternative["physicalHistoryHash"] == default["physicalHistoryHash"]
    assert alternative["actionIds"] == default["actionIds"]


def test_outside_fov_and_unlabelled_are_unknown_not_vessel_free(script, output):
    binding, reference = _changed_reference(script, positive=False, unknown=True, short_grid=True)
    result = v._evaluate_development_episode_vascular(episode=script,
        output_directory=output / "short", reference_binding=binding,
        load_reference=lambda: reference)
    whole = result["wholeTool"]
    assert whole["positive_reference_cells"] == 0
    assert whole["outside_reference_fov"] is True
    assert whole["annotated_positive_encounter"] is None
    assert whole["biological_vessel_free"] is None
    assert result["patientAdmission"] is False


def test_wrong_loaded_binding_and_loader_failure_have_no_partial_outcomes(script, output):
    binding = _binding(script)
    wrong_binding, wrong_reference = _changed_reference(script, positive=False)
    result = v._evaluate_development_episode_vascular(episode=script,
        output_directory=output / "wrong", reference_binding=binding,
        load_reference=lambda: wrong_reference)
    assert result["status"] == "evaluation_failed"
    assert result["perAction"] is result["wholeTool"] is None
    secret = "private_generated_label_values_do_not_emit"
    def failure():
        raise OSError(secret)
    second = v._evaluate_development_episode_vascular(episode=script,
        output_directory=output / "failed", reference_binding=binding,
        load_reference=failure)
    assert second["status"] == "evaluation_failed"
    assert secret not in (output / "failed" / "report.json").read_text()
    assert wrong_binding != binding


def test_private_loader_cannot_repose_validated_history_or_replace_binding(script, output):
    original_history = copy.deepcopy(script["history"])
    original_hash = semantic_digest(original_history)
    binding = _binding(script)
    def mutate_episode():
        for row in script["history"]:
            if row["action_id"] == "STOP":
                continue
            row["entry_mm"][0] += 3.
            row["tip_mm"][0] += 3.
            for micro in row["microsteps"]:
                for name in ("tip_start_mm", "tip_end_mm",
                             "active_stroke_start_mm", "active_stroke_end_mm"):
                    micro[name][0] += 3.
        return v._load_generated_reference(binding)
    failed = v._evaluate_development_episode_vascular(episode=script,
        output_directory=output / "mutated-pose", reference_binding=binding,
        load_reference=mutate_episode)
    assert failed["status"] == "evaluation_failed"
    assert failed["wholeTool"] is failed["perAction"] is None
    assert failed["physicalHistoryHash"] == original_hash
    assert json.loads(failed["physicalHistoryCanonicalJson"]) == original_history

    clean = execute_development_episode()[1]
    other_binding = _binding(clean)
    def mutate_binding():
        object.__setattr__(other_binding, "source_hash", "sha256:" + "3" * 64)
        return v._load_generated_reference(other_binding)
    wrong = v._evaluate_development_episode_vascular(episode=clean,
        output_directory=output / "mutated-binding", reference_binding=other_binding,
        load_reference=mutate_binding)
    assert wrong["status"] == "evaluation_failed"
    assert wrong["perAction"] is None
    assert wrong["referenceBindingHash"] == _binding(clean).fingerprint


def test_reference_binding_swap_during_scoring_cannot_publish_counts(script, output, monkeypatch):
    binding = _binding(script)
    reference = v._load_generated_reference(binding)
    original = v.evaluate_contacts
    switched = []
    def swap(*args, **kwargs):
        if not switched:
            other = replace(binding, episode_id="sha256:" + "4" * 64)
            object.__setattr__(reference, "binding", other)
            switched.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(v, "evaluate_contacts", swap)
    result = v._evaluate_development_episode_vascular(episode=script,
        output_directory=output / "swapped-during-score", reference_binding=binding,
        load_reference=lambda: reference)
    assert switched == [True]
    assert result["status"] == "evaluation_failed"
    assert result["referenceBindingHash"] == binding.fingerprint
    assert result["wholeTool"] is result["perAction"] is None


def test_wrong_binding_rejected_before_private_loader(script, output):
    wrong = replace(_binding(script), source_hash="sha256:" + "2" * 64)
    opened = []
    with pytest.raises(ValueError, match="evaluator_reference_binding_mismatch"):
        v._evaluate_development_episode_vascular(episode=script,
            output_directory=output / "bad-binding", reference_binding=wrong,
            load_reference=lambda: opened.append(True))
    assert opened == []
    assert not (output / "bad-binding").exists()


def test_search_stop_only_has_no_vascular_capsule_or_loader_sampling(output):
    episode = execute_development_episode(selector="SEARCH")[1]
    result = v.evaluate_development_episode_vascular(episode=episode,
        output_directory=output / "search")
    assert result["status"] == "evaluated_generated_vascular_reference"
    assert [row["actionId"] for row in result["perAction"]] == [row["action_id"] for row in episode["history"]]
    if result["actionIds"] == ["STOP"]:
        assert result["wholeTool"]["touched_reference_cells"] == 0
        assert result["perAction"] == [{"actionIndex": 0, "actionId": "STOP",
            "interactionMode": "stop", "sweepCount": 0,
            "shaft": None, "tip": None, "wholeTool": None}]


def test_private_reference_variation_cannot_change_search_actions_or_nominal_seal(output):
    first = execute_development_episode(selector="SEARCH")[1]
    before_actions = [row["action_id"] for row in first["history"]]
    before_seal = first["planning"]["strategySeal"]
    binding, reference = _changed_reference(first, positive=False, unknown=True)
    changed = v._evaluate_development_episode_vascular(episode=first,
        output_directory=output / "private-changed", reference_binding=binding,
        load_reference=lambda: reference)
    second = execute_development_episode(selector="SEARCH")[1]
    assert changed["actionIds"] == before_actions
    assert [row["action_id"] for row in second["history"]] == before_actions
    assert second["planning"]["strategySeal"] == before_seal
    assert second["sourceHash"] == first["sourceHash"]
