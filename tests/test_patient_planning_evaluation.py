"""Generated native controls; no patient images, labels, weights or training."""
from dataclasses import asdict
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.native_spatial_task import make_native_opening_task, OPENING_TOOLS
from resectionlab.patient_planning_evaluation import evaluate_sealed_annotation_plans


def fixture(tmp_path, stop_after=None):
    task = make_native_opening_task()
    initial = task.observation().fingerprint
    for index, (tool, voxel) in enumerate(zip(OPENING_TOOLS, ((4, 4, 1), (4, 4, 5)))):
        if index == stop_after:
            task.step('STOP')
            break
        action = next(row['action_id'] for row in task.candidate_inventory()['ledger']
                      if row['feasible'] and row['tool_id'] == tool.tool_id and row['voxel'] == list(voxel))
        task.step(action)
    metrics = task.metrics()
    context = {'patient_group': 'generated-evaluator-control'}
    nominal_history = deepcopy(metrics['history'])
    for step in nominal_history:
        if step['action_id'] != 'STOP':
            step['outcome_scope'] = 'permitted_nominal_model'
    plan = dict(context_hash=semantic_digest(context), source_hash=task.case.source_hash,
                decision_model_hash=task.decision_model_hash, initial_observation_hash=initial,
                max_steps=2, actions=[row['action_id'] for row in metrics['history']],
                history=nominal_history)
    audit = evaluate_native_spatial_episode(task)
    record = dict(version='patient-native-preflight-sealed-plans-v1', status='complete',
        patient_context=context, source_hash=task.case.source_hash, decision_model_hash=task.decision_model_hash,
        initial_observation_hash=initial, tools=[asdict(t) for t in task.case.tools],
        expected_methods=['SEARCH', 'IL', 'RL'], plans=[dict(method=m, plan=plan,
            plan_seal=semantic_digest(plan), replayed_history=metrics['history'],
            independent_geometry=audit) for m in ['SEARCH', 'IL', 'RL']])
    path = tmp_path / 'plans.json'
    def write():
        path.write_text(json.dumps(record))
        return sha256(path.read_bytes()).hexdigest()
    mask = np.zeros(task.case.structural_intensity.shape, bool)
    mask[4, 4, 1] = True
    reference = dict(patient_group=context['patient_group'], public_source_hash=task.case.source_hash,
        frame_correspondence='qualified_common_RAS_mm_frame', mask=mask, coverage=np.ones(mask.shape, bool),
        affine_ras_mm=task.case.affine_ras_mm, source_kind='generated_control',
        label_name='generated critical annotation', source_sha256='a'*64)
    return path, record, reference, write


def test_full_tool_positive_contact_and_private_changes_do_not_change_plans(tmp_path):
    path, record, reference, write = fixture(tmp_path)
    digest = write()
    positive = evaluate_sealed_annotation_plans(path, expected_sha256=digest, load_reference=lambda: reference)
    assert len(positive['methods']) == 3
    assert positive['clinical_injury_probability'] is None
    assert all(row['contacts']['whole_tool']['annotated_positive_encounter'] is True for row in positive['methods'])
    reference['mask'][:] = False
    reference['coverage'][:] = False
    unknown = evaluate_sealed_annotation_plans(path, expected_sha256=digest, load_reference=lambda: reference)
    assert all(row['contacts']['whole_tool']['annotated_positive_encounter'] is None for row in unknown['methods'])
    assert [r['plan_seal'] for r in positive['methods']] == [r['plan_seal'] for r in unknown['methods']]
    assert sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('failure', ['missing_method', 'seal', 'geometry', 'input'])
def test_every_method_checked_before_private_read(tmp_path, failure):
    path, record, reference, write = fixture(tmp_path)
    if failure == 'missing_method': record['plans'].pop()
    if failure == 'seal': record['plans'][-1]['plan_seal'] = 'wrong'
    if failure == 'geometry': record['plans'][-1]['independent_geometry']['accepted'] = False
    if failure == 'input': record['source_hash'] = 'wrong'
    calls = []
    with pytest.raises(ValueError):
        evaluate_sealed_annotation_plans(path, expected_sha256=write(), load_reference=lambda: calls.append(1))
    assert calls == []


def test_mutation_during_private_read_is_refused(tmp_path):
    path, record, reference, write = fixture(tmp_path)
    digest = write()
    def load():
        path.write_text('{}')
        return reference
    with pytest.raises(ValueError, match='changed_during_reference'):
        evaluate_sealed_annotation_plans(path, expected_sha256=digest, load_reference=load)


def test_wrong_patient_and_unknown_positive_coverage_are_refused(tmp_path):
    path, record, reference, write = fixture(tmp_path)
    digest = write()
    reference['patient_group'] = 'another-person'
    with pytest.raises(ValueError, match='patient_or_frame'):
        evaluate_sealed_annotation_plans(path, expected_sha256=digest, load_reference=lambda: reference)
    reference['patient_group'] = record['patient_context']['patient_group']
    reference['coverage'][:] = False
    with pytest.raises(ValueError, match='explicit_coverage'):
        evaluate_sealed_annotation_plans(path, expected_sha256=digest, load_reference=lambda: reference)


def test_replay_pose_difference_is_not_hidden_by_outcome_scope_handling(tmp_path):
    path, record, reference, write = fixture(tmp_path)
    step = record['plans'][0]['replayed_history'][0]
    step['tip_mm'] = [step['tip_mm'][0] + 1, *step['tip_mm'][1:]]
    calls = []
    with pytest.raises(ValueError, match='nominal_and_replayed_history'):
        evaluate_sealed_annotation_plans(path, expected_sha256=write(), load_reference=lambda: calls.append(1))
    assert not calls


@pytest.mark.parametrize('stop_after', [0, 1])
def test_real_STOP_scope_and_complete_history_are_preserved(tmp_path, stop_after):
    path, record, reference, write = fixture(tmp_path, stop_after=stop_after)
    result = evaluate_sealed_annotation_plans(path, expected_sha256=write(), load_reference=lambda: reference)
    assert result['status'] == 'complete'
    if stop_after == 0:
        assert all(row['contacts']['whole_tool']['touched_reference_cells'] == 0 for row in result['methods'])
