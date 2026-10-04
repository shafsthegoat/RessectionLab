"""Metadata and batch arithmetic only; no image access or learning execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.real_patient_learning import (
    PatientEpisode, SELECT_SUBJECTS, TRAIN_GROUPS, TRAIN_SUBJECTS,
    patient_uniform_bc_indices, read_development_cohort, require_development_role,
    training_patient_schedule, validate_on_policy_batch,
)

MANIFEST = Path(__file__).parents[1] / "manifests/experiments/btc-spatial-development-cohort-v1.json"
POLICY_HASH = "sha256:" + "a" * 64


def test_exact_six_train_and_two_select_roles_require_only_metadata():
    cohort = read_development_cohort(MANIFEST)
    for role, subjects in (("TRAIN", TRAIN_SUBJECTS), ("SELECT", SELECT_SUBJECTS)):
        for subject in subjects:
            member = require_development_role(cohort, subject, role=role)
            assert member.patient_group == "BTC:" + subject
            assert member.role == role


@pytest.mark.parametrize("subject,role", [("sub-PAT29", "TRAIN"), ("sub-PAT31", "SELECT"),
    ("sub-PAT26", "TRAIN"), ("sub-PAT05", "SELECT"), ("UPENN-GBM-00001", "TRAIN"),
    ("sub-PAT22", "ADAPT"), ("BTC:sub-PAT22", "TRAIN")])
def test_forbidden_or_relabelled_patients_fail_before_loading(subject, role):
    with pytest.raises(ValueError, match="permitted development"):
        require_development_role({}, subject, role=role)


def test_role_group_visit_and_full_cohort_changes_are_rejected():
    cohort = read_development_cohort(MANIFEST)
    for field, value in (("patient_group", "BTC:sub-PAT29"), ("visit", "ses-postop"),
                         ("development_role", "population_training"), ("eligible_for_external_final", True)):
        altered = deepcopy(cohort)
        altered["existing_development_records"][0][field] = value
        with pytest.raises(ValueError, match="locked|Duplicate"):
            require_development_role(altered, "sub-PAT22", role="TRAIN")
    altered = deepcopy(cohort)
    altered["existing_development_records"].pop()
    with pytest.raises(ValueError, match="locked"):
        require_development_role(altered, "sub-PAT22", role="TRAIN")
    altered = deepcopy(cohort)
    altered["candidates"].append(deepcopy(altered["candidates"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        require_development_role(altered, "sub-PAT22", role="TRAIN")


def test_manifest_identity_is_checked_before_json_parse(tmp_path):
    changed = tmp_path / "cohort.json"
    changed.write_text("not JSON or a permitted cohort")
    with pytest.raises(ValueError, match="bytes changed"):
        read_development_cohort(changed)


def test_bc_patient_draw_is_independent_of_teacher_path_length():
    class FixedDraws:
        def __init__(self):
            self.bounds = []
            self.values = iter([0, 0, 5, 999, 0, 0])
        def integers(self, bound):
            self.bounds.append(bound)
            return next(self.values)
    counts = {group: 1 for group in TRAIN_GROUPS}
    counts[TRAIN_GROUPS[-1]] = 1000
    generator = FixedDraws()
    indices = patient_uniform_bc_indices(counts, batch_size=3, generator=generator)
    assert generator.bounds == [6, 1, 6, 1000, 6, 1]
    assert indices == ((TRAIN_GROUPS[0], 0), (TRAIN_GROUPS[-1], 999), (TRAIN_GROUPS[0], 0))


def test_bc_is_reproducible_and_does_not_drop_missing_or_stop_only_patients():
    counts = {group: 1 for group in TRAIN_GROUPS}
    a = patient_uniform_bc_indices(counts, batch_size=20, generator=np.random.default_rng(23))
    b = patient_uniform_bc_indices(counts, batch_size=20, generator=np.random.default_rng(23))
    assert a == b and all(index == 0 for _, index in a)
    for changed in ({**counts, TRAIN_GROUPS[0]: 0}, {**counts, "BTC:sub-PAT26": 1},
                    {group: 1 for group in TRAIN_GROUPS[:-1]}):
        with pytest.raises(ValueError):
            patient_uniform_bc_indices(changed, batch_size=2, generator=np.random.default_rng(23))


def test_full_patient_schedule_has_no_outcome_or_length_weighting():
    schedule = training_patient_schedule(rounds=3, generator=np.random.default_rng(23))
    assert len(schedule) == 18
    for offset in (0, 6, 12):
        assert set(schedule[offset:offset + 6]) == set(TRAIN_GROUPS)
    assert schedule == training_patient_schedule(rounds=3, generator=np.random.default_rng(23))


def episode(group=TRAIN_GROUPS[0], rewards=(-3., 1.), policy_hash=POLICY_HASH):
    return PatientEpisode(group, policy_hash,
        tuple(SimpleNamespace(reward=reward, terminated=index == len(rewards) - 1)
              for index, reward in enumerate(rewards)))


def test_complete_negative_and_stop_episodes_count_by_patient_not_steps():
    episodes = [episode(), episode(TRAIN_GROUPS[1], rewards=(0.,))]
    report = validate_on_policy_batch(episodes, parameter_hash=POLICY_HASH)
    assert report["completed_episodes"] == 2 and report["transitions"] == 3
    assert report["episodes_by_patient"] == {group: int(index < 2) for index, group in enumerate(TRAIN_GROUPS)}
    assert report["actor_reduction"] == "mean_episodes_sum_discounted_score_terms"


@pytest.mark.parametrize("group", ["BTC:sub-PAT26", "BTC:sub-PAT29", "UPENN-GBM-00001"])
def test_population_gradient_batch_rejects_nontrain_patient(group):
    with pytest.raises(ValueError, match="TRAIN"):
        validate_on_policy_batch([episode(group)], parameter_hash=POLICY_HASH)


def test_gradient_batch_rejects_stale_incomplete_and_nonfinite_episodes():
    with pytest.raises(ValueError, match="stale"):
        validate_on_policy_batch([episode(policy_hash="sha256:" + "b" * 64)], parameter_hash=POLICY_HASH)
    bad = [PatientEpisode(TRAIN_GROUPS[0], POLICY_HASH, ()),
           PatientEpisode(TRAIN_GROUPS[0], POLICY_HASH, (SimpleNamespace(reward=1., terminated=False),)),
           PatientEpisode(TRAIN_GROUPS[0], POLICY_HASH, (SimpleNamespace(reward=1., terminated=True),) * 2),
           episode(rewards=(float("nan"),))]
    for value in bad:
        with pytest.raises(ValueError, match="complete"):
            validate_on_policy_batch([value], parameter_hash=POLICY_HASH)
    with pytest.raises(ValueError, match="measured"):
        validate_on_policy_batch([episode()], parameter_hash="not-a-measured-hash")
