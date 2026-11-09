"""Source-sized frozen gates, exact prefix cost and explicit missing evidence."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, semantic_digest, thaw_json
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_patient_task import freeze_patient_task, assess_patient_task
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet


def source(*, protected=True, role='TRAIN'):
    shape = (64, 64, 96)
    target = np.zeros(shape, bool); target[30:34, 30:34, 58:62] = True
    stem = np.zeros(shape, bool); stem[50, 50, 40] = True
    metadata = {'is_synthetic': True, 'generated': True,
        'data_case': {'family_id': 'GENERATED:frozen-task', 'role': role, 'split': role},
        'integrated_rehearsal': {'stage': 'generated_existing_exposure', 'assumptions': ['GENERATED geometry, not patient anatomy']},
        'anatomy_layers': {'target': {'kind': 'tumor'},
            'exclusion': {'kind': 'brain', 'structure_id': 'brainstem', 'protected': protected,
                          'review_status': 'engineering-reviewed'}}}
    case = CaseData(case_id='GENERATED-frozen-task', mri=np.zeros(shape, np.float32), affine=np.eye(4),
        compartments={'target': target, 'exclusion': stem}, metadata=metadata,
        source_refs=(SourceRef('generated', 'synthetic://frozen-task', provenance='simulated'),))
    scene = from_case(case)
    # This fixture's full-grid box is explicit engineering geometry. No patient
    # coverage is inferred from its contents or the name of the excluded voxel.
    adapter = {'reviewedTaskDomainV1': {'mode': 'restrict-unknown', 'shape': list(shape),
        'minimumVoxelBoundary': [-.5, -.5, -.5], 'maximumVoxelBoundary': [63.5, 63.5, 95.5],
        'reviewEvidenceSha256': '1'*64, 'maskSha256': '2'*64}}
    case = replace(case, metadata={**metadata, 'surgical_adapter': adapter})
    engine = SurgicalEngine(from_case(case))
    assert engine.step({'slot': 'right', 'operation': 'configure',
                        'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1})['status'] == 'accepted'
    return case, engine


def execute(engine, operation, z, duration):
    for attempt in range(8 if operation == 'apply' else 1):
        result = engine.step({'slot': 'right', 'operation': operation,
            'pose': {'position': [31.5, 31.5, float(z)], 'quaternion': [0., 0., 0., 1.]},
            'duration': duration if attempt == 0 else 30.})
        if result['status'] == 'accepted': return
        assert (result['status'], result['reason']) == ('partial', 'TISSUE_ASPIRATION_IN_PROGRESS'), result
    pytest.fail('Finite native aspiration did not finish within the fixed test budget')


def test_small_target_uses_frozen_original_fraction_instead_of_impossible_500mm3_floor():
    case, baseline = source()
    task = freeze_patient_task(case, baseline)
    assert task['originalTargetMm3'] == 64 and task['minimumWholeTargetMm3'] == 32
    assert task['minimumTargetPerZoneMm3'] == 3.2
    assert task['policy']['maximumWholeNonTargetMm3'] == 200
    assert task['policy']['maximumTreatmentNonTargetMm3'] == 20
    assert task['sourceRole'] == 'TRAIN' and task['generatedControl']
    assert task['eligibilityBlockers'] == []
    work = baseline.clone(); execute(work, 'place', 45., .1)
    treatment = work.clone(); execute(work, 'apply', 63., 4.)
    execute(work, 'withdraw', 45., 2.)
    restored = SurgicalEngine.restore(work.scene, work.snapshot())
    assert restored.snapshot() == work.snapshot()
    assessment = assess_patient_task(case, work, task, baseline=baseline,
        treatment_baseline=treatment, exact_replay=True,
        physical_calls=len(work.actions)*2, wall_seconds=1.)
    assert assessment['taskSatisfied'], assessment['failedGates']
    assert assessment['outcome']['originalTarget']['fractionRemoved'] >= .5
    assert len([v for v in assessment['zoneTreatmentMm3'].values() if v >= 3.2]) >= 3


def test_missing_protection_keeps_task_ineligible_even_when_all_source_target_is_removed():
    case, baseline = source(protected=False); task = freeze_patient_task(case, baseline)
    assert 'MISSING_CRITICAL_ANNOTATIONS' in task['eligibilityBlockers']
    work = baseline.clone(); execute(work, 'place', 45., .1); treatment = work.clone()
    execute(work, 'apply', 63., 4.); execute(work, 'withdraw', 45., 2.)
    result = assess_patient_task(case, work, task, baseline=baseline, treatment_baseline=treatment,
        exact_replay=True, physical_calls=len(work.actions)*2, wall_seconds=1.)
    assert not result['taskSatisfied'] and not result['gates']['declaredEligibility']


@pytest.mark.parametrize('required', [(), ('brainstem',)])
def test_empty_required_list_cannot_skip_review_of_mandatory_brainstem(required):
    case, _ = source()
    # Match the declared source review to the supplied annotation so automatic
    # source-mask union cannot hide the case behind a reviewed metadata claim.
    metadata = thaw_json(case.metadata)
    metadata['anatomy_layers']['exclusion']['review_status'] = 'unreviewed'
    case = replace(case, metadata=metadata); scene = from_case(case)
    annotation = ProtectionAnnotation.from_mask(scene, 'brainstem',
        case.compartments['exclusion'], provenance='GENERATED review regression',
        review_status='unreviewed')
    engine = SurgicalEngine(scene, protection=ProtectionSet(scene, [annotation], required=required))
    assert engine.step({'slot': 'right', 'operation': 'configure',
        'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1})['status'] == 'accepted'
    assert not engine.guard_report()['missingRequired']
    assert 'CRITICAL_ANNOTATION_REVIEW_REQUIRED' in freeze_patient_task(case, engine)['eligibilityBlockers']


@pytest.mark.parametrize('role', ['EVAL', 'TEST', 'FROZEN_ASSESSMENT'])
def test_synthetic_descendant_cannot_relabel_reserved_family_as_development(role):
    case, engine = source(role=role)
    with pytest.raises(ValueError, match='UNRESERVED_SOURCE_ROLE'):
        freeze_patient_task(case, engine)


def test_resealing_weaker_caps_or_source_target_goals_cannot_change_frozen_task():
    case, baseline = source(); task = freeze_patient_task(case, baseline)
    for mutation in ('cap', 'target', 'prefix'):
        changed = copy.deepcopy(task)
        if mutation == 'cap': changed['policy']['maximumWholeNonTargetMm3'] = 300.
        if mutation == 'target': changed['minimumWholeTargetMm3'] = 1.
        if mutation == 'prefix': changed['prefixActions'] = 0
        changed['taskSeal'] = semantic_digest({k:v for k,v in changed.items() if k != 'taskSeal'})
        with pytest.raises(ValueError, match='FROZEN_INPUTS_CHANGED'):
            assess_patient_task(case, baseline, changed, baseline=baseline)


def test_missing_cost_replay_or_treatment_baseline_never_becomes_zero_burden_pass():
    case, baseline = source(); task = freeze_patient_task(case, baseline)
    work = baseline.clone(); execute(work, 'place', 45., .1); treatment = work.clone()
    execute(work, 'apply', 63., 4.); execute(work, 'withdraw', 45., 2.)
    result = assess_patient_task(case, work, task, baseline=baseline)
    assert {'exactReplay', 'physicalCalls', 'wallBudget', 'treatmentNonTarget'} <= set(result['failedGates'])
    with pytest.raises(ValueError, match='MUST_PRECEDE_TARGET_REMOVAL'):
        assess_patient_task(case, work, task, baseline=baseline, treatment_baseline=work)
