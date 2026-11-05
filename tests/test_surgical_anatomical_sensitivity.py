"""Generated source-label stress controls; no clinical calibration evidence."""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_anatomical_encounter import AnatomicalObservation
from resectionlab import surgical_anatomical_sensitivity as sensitivity
from resectionlab.surgical_strategy_sensitivity import _observation, validate_sensitivity
from test_surgical_anatomical_encounter import context_case, context_report
from test_surgical_two_tool import qualified, prepared


@pytest.fixture(scope='module')
def label_studies(context_case, context_report):
    return {step: sensitivity.generate_sensitivity(context_case[1], context_report,
        {'contextVoxelShift': step}) for step in (0, 1)}


def test_source_grid_translation_uses_affine_direction_and_does_not_wrap_or_fill_unknown():
    mask = np.zeros((4, 5, 6), bool); mask[0, 1, 2] = True; mask[3, 2, 4] = True
    affine = np.array([[0., -1., 0., 20.], [.5, 0., 0., 40.], [0., 0., -2., 60.], [0., 0., 0., 1.]])
    context = AnatomicalObservation(mask, affine, 'unreviewed')
    scenarios = sensitivity.label_scenarios(context, 1)
    assert len(scenarios) == 7
    for scenario in scenarios:
        shift = scenario['voxelShift']
        assert np.array_equal(scenario['translationRasMm'], affine[:3, :3]@shift)
        for dest in np.ndindex(mask.shape):
            src = np.array(dest)-shift
            known = bool(np.all(src >= 0) and np.all(src < mask.shape))
            assert scenario['known'][dest] == known
            assert scenario['mask'][dest] == (bool(mask[tuple(src)]) if known else False)
        assert scenario['outOfFieldContextCells'] == int(mask.sum()-scenario['mask'].sum())
    assert context.mask.tobytes() == mask.tobytes() and np.array_equal(context.affine, affine)
    tiny = AnatomicalObservation(np.ones((1, 1, 1), bool), np.eye(4), 'unreviewed')
    assert all(not s['known'].any() and not s['mask'].any() for s in sensitivity.label_scenarios(tiny, 2)[1:])


def test_zero_shift_preserves_nominal_programs_physics_and_original_context(context_case, context_report, label_studies):
    before = context_case[1].snapshot(); assessment = label_studies[0]['assessment']
    assert assessment['scenarioCount'] == 1 and not assessment['preferenceChanges']
    assert set(assessment['policies'].values()) == {context_report['publicPlan']['decision']['selectedId']}
    for row, original in zip(assessment['candidates'], context_report['publicPlan']['alternatives']):
        assert row['physicalStateHashes'] == original.get('physicalStateHashes', [])
        assert row['summary'] == original['summary'] and row['withdrawalComplete']
        for key, value in original['anatomicalEncounter'].items():
            if key in row['outcomes'][0]['encounter']: assert row['outcomes'][0]['encounter'][key] == value
    assert context_case[1].snapshot() == before


def test_annotation_changes_do_not_change_held_physics_but_can_exclude_context_work(label_studies):
    nominal = label_studies[0]['assessment']; shifted = label_studies[1]['assessment']
    assert shifted['scenarioCount'] == 7 and not shifted['calibratedUncertainty']['supported']
    assert shifted['scenarioConfiguration']['distribution'] is None
    assert shifted['fixedMarginDefinition']['isGuarantee'] is False
    for first, row in zip(nominal['candidates'], shifted['candidates']):
        assert first['physicalStateHashes'] == row['physicalStateHashes']
        assert first['summary'] == row['summary']
        assert len(row['outcomes']) == 7
        for outcome in row['outcomes']:
            encounter = outcome['encounter']
            assert outcome['meetsTask'] == (not outcome['violations'])
            assert bool(encounter['newContextWorkCells']) == ('NEW_CONTEXT_MANIPULATION' in outcome['violations'])
            assert bool(encounter['unknownEncounteredCells']) == ('UNKNOWN_CONTEXT_COVERAGE' in outcome['violations'])
            assert encounter['newContextWorkMm3'] >= encounter['newRemovedContextMm3']


def test_work_and_unknown_membership_are_checked_even_when_new_removal_excludes_near_cells():
    from types import SimpleNamespace
    mask = np.zeros((2, 2, 2), bool); mask[1, 1, 1] = True
    scenario = {'mask': mask, 'known': np.ones_like(mask)}
    engine = SimpleNamespace(state={'removed': [7]}, scene=SimpleNamespace(voxel_mm3=1., points=lambda ids: np.zeros((len(ids), 3))))
    footprint = sensitivity._Footprint(0.)
    endpoint = sensitivity._endpoint(engine, {'removed': []}, footprint, scenario, {7}, {'marginMm': 0., 'maxNearMm3': 0.})
    assert endpoint['nearMm3'] == 0. and endpoint['newContextWorkCells'] == [7] and endpoint['newRemovedContextMm3'] == 1.
    scenario['known'][1, 1, 1] = False
    endpoint = sensitivity._endpoint(engine, {'removed': []}, footprint, scenario, {7}, {'marginMm': 0., 'maxNearMm3': 0.})
    assert endpoint['unknownEncounteredCells'] == [7]


def test_complete_native_program_rejects_injected_context_in_partial_work_and_removed_cells(context_case, context_report):
    # A labeled software hazard injection, not an accepted source-shift study.
    # The ordinary wide-cavity source fixture has no target/context boundary
    # within the permitted shifts. Injecting membership tests the full replay
    # work ledger without inventing a biological annotation or calibration.
    engine = context_case[1]; observation = _observation(engine, context_report)
    rows = context_report['publicPlan']['alternatives']; cells = rows[1]['summary']['removedSourceCells']
    assert cells
    scenarios = sensitivity.label_scenarios(observation.context, 0)
    injected = copy.deepcopy(scenarios[0]); injected['id'] = 'software-injected-context-overlap'
    injected['mask'].ravel()[cells] = True; scenarios.append(injected)
    for row in rows[:2]:
        result = sensitivity._replay(engine.clone(), row, context_report['publicPlan']['constraints'], scenarios, 0., check=lambda: None)
        outcome = result['outcomes'][1]
        assert result['physicalStateHashes'] == row['physicalStateHashes']
        assert outcome['encounter']['newContextWorkMm3'] > 0.
        assert 'NEW_CONTEXT_MANIPULATION' in outcome['violations'] and not outcome['meetsTask']
        if row['id'] == 'short': assert outcome['encounter']['newRemovedContextMm3'] == 0.
        else: assert outcome['encounter']['newRemovedContextMm3'] > 0.


@pytest.mark.parametrize('raw', [{'contextVoxelShift': True}, {'contextVoxelShift': 1.}, {'contextVoxelShift': -1},
    {'contextVoxelShift': 3}, {'contextVoxelShift': 1, 'patientProbability': .99}, {'offsetMm': 1}])
def test_controls_cannot_claim_calibration_or_unbounded_sampling(context_case, context_report, raw):
    with pytest.raises(ValueError): sensitivity.assess_observed(_observation(context_case[1], context_report), context_report['publicPlan'], raw)


def reseal(study):
    a = study['assessment']; a['assessmentSeal'] = semantic_digest({k: v for k, v in a.items() if k != 'assessmentSeal'})
    study['studySeal'] = semantic_digest({k: v for k, v in study.items() if k != 'studySeal'})
    return study


@pytest.mark.parametrize('field', ['policy', 'scenario', 'work', 'unknown', 'physical', 'calibration'])
def test_self_resealed_forgery_requires_canonical_native_and_public_evidence(context_case, context_report, label_studies, field):
    study = copy.deepcopy(label_studies[0]); a = study['assessment']; row = a['candidates'][0]
    if field == 'policy': a['policies']['allDeclaredScenarios'] = 'invented'
    elif field == 'scenario': a['scenarioConfiguration']['scenarios'][0]['maskHash'] = 'invented'
    elif field == 'work': row['outcomes'][0]['encounter']['newContextWorkMm3'] = 999.
    elif field == 'unknown': row['outcomes'][0]['encounter']['unknownEncounteredCells'] = [-1]
    elif field == 'physical': row['physicalStateHashes'][0] = 'invented'
    else: a['calibratedUncertainty']['supported'] = True
    with pytest.raises(ValueError, match='canonical anatomical'): validate_sensitivity(context_case[1], context_report, reseal(study))


def test_private_truth_and_callback_mutations_do_not_change_public_label_decisions(context_case, context_report, label_studies):
    engine = context_case[1].clone(); engine._private_reference_truth = np.ones(engine.scene.shape, bool)
    engine._private_evaluator = lambda: pytest.fail('Private reference read')
    assert sensitivity.generate_sensitivity(engine, context_report, {'contextVoxelShift': 0}) == label_studies[0]
    plan = copy.deepcopy(context_report['publicPlan'])
    def progress(*_): plan['constraints']['minimumTargetMm3'] = 0.
    observed = sensitivity.assess_observed(_observation(engine, context_report), plan, {'contextVoxelShift': 0}, progress=progress)
    assert observed == label_studies[0]['assessment']


def test_shared_history_preserves_both_sensitivity_kinds_and_cancellation_is_atomic(context_case, context_report, label_studies):
    from resectionlab import surgical_strategy_workspace as workspace
    from resectionlab import surgical_strategy_sensitivity as execution
    from resectionlab.desktop_bridge import _Request, BridgeError
    case, engine, source, *_ = context_case; record = copy.deepcopy(source)
    item = workspace.install_report(engine, record, case.semantic_hash, context_report)
    old = execution.generate_sensitivity(engine, context_report, {'offsetMm': 0.})
    workspace.install_sensitivity(engine, record, case.semantic_hash, item, old)
    workspace.install_sensitivity(engine, record, case.semantic_hash, item, label_studies[0])
    shown = workspace.planning_view(engine, record, case.semantic_hash)
    assert shown['executionSensitivity']['study'] == old
    assert shown['anatomicalSensitivity']['study'] == label_studies[0]
    request = _Request('label-cancel'); before = copy.deepcopy(record)
    def progress(*_): request.cancel()
    with pytest.raises(BridgeError, match='cancelled'):
        sensitivity.generate_sensitivity(engine, context_report, {'contextVoxelShift': 1}, check=request.check, progress=progress)
    assert record == before


def test_bridge_requires_one_stress_control_and_persists_a_cold_native_study(context_case, context_report, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.imaging import save_case, read_case_artifacts
    from resectionlab import surgical_strategy_workspace as workspace
    from resectionlab import surgical_strategy_sensitivity as execution
    case, engine, source, *_ = context_case; record = copy.deepcopy(source)
    workspace.install_report(engine, record, case.semantic_hash, context_report)
    path = tmp_path/'source.ressectionlab'; saved = tmp_path/'saved.ressectionlab'
    save_case(case, path, artifacts={'surgical_toolbox': record}); original = path.read_bytes()
    session = BridgeSession(tmp_path/'transfers')
    def call(**args):
        return session.execute('surgicalRehearsal', {'caseHash': case.semantic_hash,
            'command': 'analyzeStrategySensitivity', 'expectedStateId': engine.state_id,
            'expectedPlanSeal': context_report['reportSeal'], 'acknowledgeResearchUnknowns': True,
            **args}, _Request('label-study'), lambda *_: None)
    try:
        session.execute('loadCase', {'path': str(path)}, _Request('load'), lambda *_: None)
        with pytest.raises(ValueError): call(contextVoxelShift=0, offsetMm=0.)
        with pytest.raises(ValueError): call(contextVoxelShift=True)
        result = call(contextVoxelShift=0); attachment = result['strategyPlanning']['anatomicalSensitivity']
        assert attachment['study']['nativeReplayVerified'] and result['view']['stateId'] == engine.state_id
        session.execute('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved)}, _Request('save'), lambda *_: None)
        assert path.read_bytes() == original
        assert len(read_case_artifacts(saved)['surgical_toolbox']['strategySensitivityStudies']) == 1
    finally: session.close()
    workspace._VALIDATED.clear(); execution._VALIDATED.clear(); sensitivity.planning._VALIDATED_REPORTS.clear()
    fresh = BridgeSession(tmp_path/'fresh-transfers')
    try:
        fresh.execute('loadCase', {'path': str(saved)}, _Request('load'), lambda *_: None)
        result = fresh.execute('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': 'inspect'}, _Request('inspect'), lambda *_: None)
        assert result['strategyPlanning']['anatomicalSensitivity'] == attachment and result['view']['stateId'] == engine.state_id
    finally: fresh.close()


def test_final_progress_cancellation_cannot_publish_validation_authority(context_case, context_report, monkeypatch):
    from resectionlab.desktop_bridge import _Request, BridgeError
    from resectionlab import surgical_strategy_sensitivity as execution
    request = _Request('final-label-cancel'); remembered = []
    monkeypatch.setattr(execution, '_remember', remembered.append)
    def progress(fraction, _):
        if fraction == 1.: request.cancel()
    with pytest.raises(BridgeError, match='cancelled'):
        sensitivity.generate_sensitivity(context_case[1], context_report, {'contextVoxelShift': 0}, check=request.check, progress=progress)
    assert not remembered
