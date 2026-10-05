"""Independent tiny boundary controls; no real images or native calls."""
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_real_training_cases as prep


STAMP = '2026-10-04T12:00:00+00:00'


def forbid(*args, **kwargs):
    raise AssertionError('This boundary must reject before downstream access')


@pytest.mark.parametrize('subject', ['sub-PAT05', 'sub-PAT26', 'sub-PAT27', 'sub-PAT99'])
def test_forbidden_subject_is_rejected_before_metadata_or_decode(monkeypatch, subject):
    monkeypatch.setattr(prep, 'read_development_cohort', forbid)
    monkeypatch.setattr(prep, 'known_members', forbid)
    monkeypatch.setattr(prep, '_decode_case', forbid)
    with pytest.raises(ValueError, match='five remaining TRAIN'):
        prep.prepare_training_case(subject, declared_at=STAMP)


def metadata_fixture(monkeypatch, tmp_path):
    bundle = tmp_path / 'fixture.bundle'
    bundle.write_bytes(b'constructed metadata-only fixture')
    member = {'subject': 'sub-PAT16', 'role': 'TRAIN', 'case_bundle': 'fixture.bundle',
              'case_bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
              'case_semantic_hash': 'fixture-case', 'planning_hash': 'fixture-plan',
              'evidence_id': 'fixture-evidence', 'evidence_hash': 'fixture-evidence-hash'}
    common = {'track': 'annotation_assisted', 'max_steps': 3, 'tools': [],
              'objective': {}, 'adapter_options': {'crop_shape': [64, 64, 64]}}
    calls = []
    monkeypatch.setattr(prep, 'ROOT', tmp_path)
    monkeypatch.setattr(prep, 'read_development_cohort', lambda path: {'source': {'git_commit': 'source-commit'}})

    def role(cohort, subject, *, role):
        calls.append((subject, role))
        return SimpleNamespace(patient_group='BTC:' + subject)

    monkeypatch.setattr(prep, 'require_development_role', role)
    monkeypatch.setattr(prep, 'known_members', lambda: {'sub-PAT16': copy.deepcopy(member)})
    monkeypatch.setattr(prep, 'common_task_definition', lambda: copy.deepcopy(common))
    return bundle, member, common, calls


def test_changed_source_bytes_reject_before_decode_after_role_check(monkeypatch, tmp_path):
    bundle, _, _, calls = metadata_fixture(monkeypatch, tmp_path)
    bundle.write_bytes(b'changed')
    monkeypatch.setattr(prep, '_decode_case', forbid)
    with pytest.raises(ValueError, match='retained byte identity'):
        prep.prepare_training_case('sub-PAT16', declared_at=STAMP)
    assert calls == [('sub-PAT16', 'TRAIN')]


def test_union_target_conflict_stays_in_denominator_and_never_constructs_task(monkeypatch, tmp_path):
    _, member, _, _ = metadata_fixture(monkeypatch, tmp_path)
    first = np.zeros((3, 3, 3), dtype=bool)
    second = first.copy()
    support = first.copy()
    first[1, 1, 1] = support[1, 1, 1] = True
    second[2, 1, 1] = True
    evidence = SimpleNamespace(
        evidence_id=member['evidence_id'], evidence_hash=member['evidence_hash'],
        provenance='estimated', review_status='review_required', model_sha256='model',
        review=None, mask=support, source_image_hash='image', source_frame_hash='frame',
        mask_hash='mask', run_sha256='run', assert_matches=lambda case: None)
    case = SimpleNamespace(
        case_id='BTC-ds001226-sub-PAT16-preop', semantic_hash='fixture-case', planning_hash='fixture-plan',
        metadata={'source_collection': {'accession': 'ds001226', 'release': '5.0.1', 'git_commit': 'source-commit'}},
        source_refs=[SimpleNamespace(source_id='structural', provenance='observed')],
        structural_evidence={member['evidence_id']: evidence}, compartments={'first': first, 'second': second})
    monkeypatch.setattr(prep, '_decode_case', lambda path: case)
    monkeypatch.setattr(prep, '_construct_task', forbid)
    monkeypatch.setattr(prep, 'derive_access', forbid)
    task, record = prep.prepare_training_case('sub-PAT16', declared_at=STAMP)
    assert task is None and record['status'] == 'blocked_support_conflict'
    assert record['coverage']['full_target_source_cells'] == 2
    assert record['coverage']['target_outside_support_source_cells'] == 1
    assert record['coverage']['target_inside_support_source_cells'] == 1
    assert not record['coverage']['reference_or_nominal_clipped']
    assert not record['coverage']['support_expanded']
    assert record['executed_transitions'] == record['optimizer_updates'] == 0
    assert first.sum() == second.sum() == support.sum() == 1


@pytest.mark.parametrize('change', ['source', 'task'])
def test_resealed_foreign_binding_rejects_before_reprepare(monkeypatch, tmp_path, change):
    _, member, common, _ = metadata_fixture(monkeypatch, tmp_path)
    binding = {'subject': 'sub-PAT16', 'role': 'TRAIN', 'member': member, 'common_task': common}
    record = {'version': prep.VERSION, 'status': 'prepared', 'subject': 'sub-PAT16',
              'role': 'TRAIN', 'binding': copy.deepcopy(binding)}
    if change == 'source':
        record['binding']['member']['case_bundle_sha256'] = '0' * 64
    else:
        record['binding']['common_task']['adapter_options']['crop_shape'] = [32, 32, 32]
    record['binding_hash'] = prep.binding_hash(record['binding'])
    monkeypatch.setattr(prep, 'prepare_training_case', forbid)
    monkeypatch.setattr(prep, '_decode_case', forbid)
    with pytest.raises(ValueError, match='authoritative input records'):
        prep.load_prepared_training_case(record)


def test_access_rule_uses_physical_distance_and_fixed_lexicographic_tie():
    support = np.ones((3, 3, 3), dtype=bool)
    target = np.zeros_like(support)
    target[1, 1, 1] = True
    affine = np.diag([2., 1., 3., 1.])
    access, derivation = prep.derive_access(target, support, affine, 'sub-PAT16')
    assert access['center_mm'] == [2., -.5, 3.]
    assert access['normal_inward'] == [0., 1., 0.]
    assert access['radius_mm'] == 6.
    assert derivation['depth_to_annotation_representative_mm'] == 1.5
    assert not derivation['selection_uses_reward_or_native_preview']
