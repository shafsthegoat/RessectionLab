"""Generated execution-offset stress controls, not measured patient uncertainty."""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.worlds import AxisTranslationStress
from resectionlab import surgical_strategy_sensitivity as sensitivity
from test_surgical_anatomical_encounter import context_case, context_report
from test_surgical_two_tool import qualified, prepared


@pytest.fixture(scope='module')
def zero_study(context_case, context_report):
    return sensitivity.generate_sensitivity(context_case[1], context_report, {'offsetMm': 0.})


def test_finite_worlds_have_exact_coherent_physical_translations_and_no_sampling_claim():
    config = AxisTranslationStress(1.)
    worlds = config.worlds()
    assert len(worlds) == 7 and worlds[0].world_id == 'nominal'
    assert config.to_dict()['distribution'] is None and config.to_dict()['patientCalibration'] is None
    assert len(AxisTranslationStress(0.).worlds()) == 1
    point = np.array([[0., 2., 4.], [1., 3., 5.]])
    for world in worlds:
        transformed = world.transform_points(point)
        assert np.array_equal(transformed[1]-transformed[0], point[1]-point[0])
        assert not world.anatomy_transform_mm.flags.writeable
        assert np.linalg.norm(world.anatomy_transform_mm[:3, 3]) == (0. if world.deterministic else 1.)
        assert world.actor_observation()['uncertainty_mode'] == ('deterministic' if world.deterministic else 'execution_offset_stress')
    for bad in (True, -1., np.nan):
        with pytest.raises(ValueError): AxisTranslationStress(bad)


def test_fixed_shaft_margin_includes_retained_target_cells():
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene
    labels = np.zeros((25, 25, 25), np.uint16); labels[12, 10, 5] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}], {}, {}))
    for hand, position in [('left', [10., 10., 10.]), ('right', [-50., -50., -50.])]:
        engine.state['slots'][hand].update(instrument='cusa', position=position)
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .01}
    previous = copy.deepcopy(engine.state['slots']); controls = {'geometry': {'nonTargetClearanceMm': 0.}}
    sensitivity._fixed_clearance(engine, action, previous, controls, 0.)
    with pytest.raises(ValueError, match='FIXED_MARGIN_SHAFT_CLEARANCE'):
        sensitivity._fixed_clearance(engine, action, previous, controls, 1.)


def test_subvoxel_offsets_change_physical_pose_prefixes_without_resampling(context_case, context_report):
    observation = sensitivity._observation(context_case[1], context_report)
    row = context_report['publicPlan']['alternatives'][1]
    world = AxisTranslationStress(.25).worlds()[2]
    outcome = sensitivity._replay(observation.planning_engine(), row,
        context_report['publicPlan']['constraints'], observation.context, world)
    index = next(i for i, action in enumerate(row['actionProgram']) if 'pose' in action)
    assert np.array_equal(np.asarray(outcome['actionProgram'][index]['pose']['position'])-
        row['actionProgram'][index]['pose']['position'], [.25, 0., 0.])
    assert len(outcome['physicalStateHashes']) > index
    assert outcome['physicalStateHashes'][index] != row['physicalStateHashes'][index]


def test_zero_offset_keeps_exact_canonical_nominal_programs_and_selection(context_case, context_report, zero_study):
    engine = context_case[1]; before = engine.snapshot()
    study = zero_study
    assessment = study['assessment']
    assert assessment['scenarioCount'] == 1 and not assessment['preferenceChanges']
    assert set(assessment['policies'].values()) == {context_report['publicPlan']['decision']['selectedId']}
    assert not assessment['calibratedUncertainty']['supported']
    for candidate, original in zip(assessment['candidates'], context_report['publicPlan']['alternatives']):
        nominal = candidate['outcomes'][0]
        assert semantic_digest(nominal['actionProgram']) == semantic_digest(original['actionProgram'])
        assert nominal['physicalStateHashes'] == original.get('physicalStateHashes', [])
        assert nominal['actionReceipts'] == original.get('actionReceipts', [])
        assert nominal['completeExecution'] and nominal['withdrawalComplete']
    assert engine.snapshot() == before


def reseal(study):
    a = study['assessment']; a['assessmentSeal'] = semantic_digest({k: v for k, v in a.items() if k != 'assessmentSeal'})
    study['studySeal'] = semantic_digest({k: v for k, v in study.items() if k != 'studySeal'})
    return study


@pytest.mark.parametrize('field', ['policy', 'prefix', 'probability', 'omission'])
def test_resealed_studies_cannot_change_decisions_prefixes_or_evidence(context_case, context_report, zero_study, field):
    study = copy.deepcopy(zero_study); a = study['assessment']
    if field == 'policy': a['policies']['allDeclaredScenarios'] = 'private-canary'
    elif field == 'prefix': a['candidates'][0]['outcomes'][0]['physicalStateHashes'][0] = 'private-canary'
    elif field == 'probability': a['calibratedUncertainty'] = {'supported': True, 'clinicalSafetyProbability': .999}
    else: a['candidates'].pop(0)
    with pytest.raises(ValueError, match='canonical sensitivity'):
        sensitivity.validate_sensitivity(context_case[1], context_report, reseal(study))


def test_completed_cache_rebuilds_live_inputs_and_never_keeps_payload_owners(context_case, context_report, zero_study, monkeypatch):
    engine = context_case[1]
    def forbidden(*a, **k): pytest.fail('Recomputed a previously complete exact study')
    monkeypatch.setattr(sensitivity, 'assess_observed', forbidden)
    assert sensitivity.validate_sensitivity(engine, context_report, copy.deepcopy(zero_study))
    changed = engine.clone(); changed.state['elapsedSeconds'] += .1
    with pytest.raises(ValueError): sensitivity.validate_sensitivity(changed, context_report, zero_study)
    assert len(sensitivity._VALIDATED) <= 32 and all(type(k) is str and v is True for k, v in sensitivity._VALIDATED.items())


def test_private_evaluator_owners_do_not_affect_stress_inputs_or_selections(context_case, context_report, zero_study):
    changed = context_case[1].clone()
    class PrivateOwner:
        def __deepcopy__(self, memo): pytest.fail('Copied a private evaluator')
        def __call__(self): pytest.fail('Called a private evaluator')
    changed._private_truth = np.ones(changed.scene.shape, bool); changed._private_evaluator = PrivateOwner()
    assert sensitivity.generate_sensitivity(changed, context_report, {'offsetMm': 0.}) == zero_study


def test_callback_edits_cannot_change_the_frozen_candidate_snapshot(context_case, context_report, zero_study):
    engine = context_case[1]; plan = copy.deepcopy(context_report['publicPlan'])
    changed = []
    def progress(stage, _):
        if not changed:
            changed.append(1); plan['alternatives'][0]['id'] = 'private-canary'
            plan['alternatives'][0]['actionProgram'][0]['duration'] = 99.
    result = sensitivity.assess_observed(sensitivity._observation(engine, context_report), plan, {'offsetMm': 0.}, progress=progress)
    assert changed and result == zero_study['assessment']


def test_workspace_cancel_keeps_existing_native_record(context_case, context_report):
    from resectionlab import surgical_strategy_workspace as workspace
    from resectionlab.desktop_bridge import _Request, BridgeError
    case, engine, original, _, _ = context_case; record = copy.deepcopy(original)
    item = workspace.install_report(engine, record, case.semantic_hash, context_report)
    before = copy.deepcopy(record); request = _Request('stress-cancel')
    def progress(fraction, message):
        if message.startswith('Checking public execution offsets'): request.cancel()
    with pytest.raises(BridgeError, match='cancelled'):
        study = sensitivity.generate_sensitivity(engine, context_report, {'offsetMm': 0.}, check=request.check, progress=progress)
        workspace.install_sensitivity(engine, record, case.semantic_hash, item, study, check=request.check)
    assert record == before


def test_native_bridge_persists_source_linked_study_and_cold_reopens(context_case, tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.imaging import save_case, read_case_artifacts
    from resectionlab import surgical_strategy_workspace as workspace
    case, engine, record, controls, _ = context_case
    source = tmp_path/'source.ressectionlab'; saved = tmp_path/'saved.ressectionlab'
    save_case(case, source, artifacts={'surgical_toolbox': record}); initial_bytes = source.read_bytes()
    session = BridgeSession(tmp_path/'transfers')
    progress_events = []
    def progress(fraction, message):
        assert type(fraction) in (int, float) and 0 <= float(fraction) <= 1 and type(message) is str
        progress_events.append((fraction, message))
    def call(command, **kwargs):
        return session.execute('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': command, **kwargs}, _Request(command), progress)
    try:
        session.execute('loadCase', {'path': str(source)}, _Request('load'), lambda *_: None)
        baseline = call('inspect')['view']['stateId']
        comparison = call('generateStrategies', expectedStateId=baseline, constraints=controls, acknowledgeResearchUnknowns=True)['strategyPlanning']
        result = call('analyzeStrategySensitivity', expectedStateId=baseline, expectedPlanSeal=comparison['reportSeal'], offsetMm=0., acknowledgeResearchUnknowns=True)
        assert result['view']['stateId'] == baseline == engine.state_id
        attachment = result['strategyPlanning']['executionSensitivity']
        assert attachment['study']['nativeReplayVerified'] and attachment['comparisonRevision'] == comparison['revision']
        assert any('native execution offsets' in message for _, message in progress_events)
        assert progress_events[-1][0] == 1.
        session.execute('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved)}, _Request('save'), lambda *_: None)
        assert source.read_bytes() == initial_bytes
        assert len(read_case_artifacts(saved)['surgical_toolbox']['strategySensitivityStudies']) == 1
    finally: session.close()
    workspace._VALIDATED.clear(); sensitivity._VALIDATED.clear()
    from resectionlab import surgical_two_tool
    surgical_two_tool._VALIDATED_REPORTS.clear()
    fresh = BridgeSession(tmp_path/'fresh-transfers')
    try:
        fresh.execute('loadCase', {'path': str(saved)}, _Request('load'), lambda *_: None)
        result = fresh.execute('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': 'inspect'}, _Request('inspect'), lambda *_: None)
        assert result['view']['stateId'] == baseline and result['strategyPlanning']['executionSensitivity'] == attachment
    finally: fresh.close()


def test_offset_worlds_replay_every_held_candidate_with_explicit_failures(context_case, context_report):
    engine = context_case[1]; before = engine.snapshot()
    study = sensitivity.generate_sensitivity(engine, context_report, {'offsetMm': 1.})
    assessment = study['assessment']
    assert assessment['scenarioCount'] == 7
    assert len(assessment['candidates']) == len(context_report['publicPlan']['alternatives'])
    assert assessment['frozenDecisionModel']['action_primitives'] == {
        r['id']: r['actionProgram'] for r in context_report['publicPlan']['alternatives']}
    for candidate in assessment['candidates']:
        assert len(candidate['outcomes']) == 7
        for outcome in candidate['outcomes']:
            assert len(outcome['actionReceipts']) == len(outcome['physicalStateHashes'])
            assert outcome['meetsTask'] == (not outcome['violations'])
            assert outcome['completeExecution'] == (outcome['failure'] is None)
            if outcome['failure']:
                assert 'INCOMPLETE_EXECUTION' in outcome['violations']
    assert engine.snapshot() == before


@pytest.mark.parametrize('raw', [{'offsetMm': True}, {'offsetMm': -1.}, {'offsetMm': 3.},
    {'offsetMm': 1., 'patientProbability': .99}])
def test_uncalibrated_input_cannot_smuggle_probability_or_unbounded_worlds(context_case, context_report, raw):
    with pytest.raises(ValueError): sensitivity.generate_sensitivity(context_case[1], context_report, raw)
