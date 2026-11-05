"""Independent native stress audit and falsification; generated anatomy only."""
import ast
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab import surgical_strategy_sensitivity as producer
from resectionlab import surgical_two_tool as planning
from resectionlab.surgical_anatomical_encounter import EncounterTracker
from test_surgical_anatomical_encounter import context_case, context_report
from test_surgical_two_tool import qualified, prepared

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_strategy_sensitivity.py'
spec = importlib.util.spec_from_file_location('independent_execution_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope='module')
def studies(context_case, context_report):
    engine = context_case[1]
    return {offset: producer.generate_sensitivity(engine, context_report, {'offsetMm': offset}) for offset in (0., 1.)}


def reseal(study):
    assessment = study['assessment']
    assessment['assessmentSeal'] = semantic_digest({k: v for k, v in assessment.items() if k != 'assessmentSeal'})
    study['studySeal'] = semantic_digest({k: v for k, v in study.items() if k != 'studySeal'})
    return study


def test_world_family_and_both_hand_commands_are_coherent_without_duration_or_rotation_changes():
    worlds = audit.worlds(1.)
    assert [name for name, _ in worlds] == ['nominal', 'R-', 'R+', 'A-', 'A+', 'S-', 'S+']
    assert audit.worlds(0.) == [('nominal', [0., 0., 0.])]
    for hand, op in (('left', 'apply'), ('right', 'withdraw')):
        original = {'slot': hand, 'operation': op, 'duration': .75,
            'pose': {'position': [2., 4., 8.], 'quaternion': [0., 0., 0., 1.]}, 'parameters': {'radiusMm': 1.5}}
        for _, delta in worlds:
            shifted = audit.translate(original, delta)
            assert np.array_equal(np.asarray(shifted['pose']['position']) - original['pose']['position'], delta)
            assert shifted['duration'] == .75 and shifted['pose']['quaternion'] == original['pose']['quaternion']
            assert shifted['parameters'] == original['parameters'] and shifted is not original
        assert audit.translate(original, [0., 0., 0.]) == original
    inspect = {'slot': 'right', 'operation': 'inspect', 'duration': .1}
    assert audit.translate(inspect, [1., 0., 0.]) == inspect


@pytest.mark.parametrize('bad', [True, -1., 2.1, float('nan')])
def test_unbounded_or_fake_probability_worlds_are_refused(bad):
    with pytest.raises(ValueError): audit.worlds(bad)


def test_independent_checker_uses_no_planner_or_sensitivity_prediction_imports():
    modules = [n.module for n in ast.walk(ast.parse(PATH.read_text())) if isinstance(n, ast.ImportFrom)]
    assert not any(m and ('surgical_strategy_sensitivity' in m or 'surgical_two_tool' in m) for m in modules)


def test_native_zero_and_seven_worlds_verify_without_predictor_or_private_owner_access(context_case, context_report, studies, monkeypatch):
    engine = context_case[1]; before = engine.snapshot()
    class PrivateOwner:
        def __deepcopy__(self, memo): pytest.fail('Private evaluator was cloned')
    engine._private_evaluation_reference = PrivateOwner()
    engine._private_truth = np.full(engine.scene.shape, 99., dtype=np.float32)
    def forbidden(*args, **kwargs): pytest.fail('Production prediction used as audit oracle')
    monkeypatch.setattr(producer, '_replay', forbidden)
    monkeypatch.setattr(planning, '_compile', forbidden)
    monkeypatch.setattr(planning, '_summary', forbidden)
    monkeypatch.setattr(planning, '_check_burdens', forbidden)
    monkeypatch.setattr(EncounterTracker, 'observe', forbidden)
    try:
        results = {offset: audit.verify(study, context_report, engine) for offset, study in studies.items()}
    finally:
        del engine._private_evaluation_reference, engine._private_truth
    assert results[0.]['passed'] and results[0.]['scenarioCount'] == 1
    assert results[1.]['passed'] and results[1.]['scenarioCount'] == 7
    assert set(results[0.]['policies'].values()) == {context_report['publicPlan']['decision']['selectedId']}
    outcomes = [o for c in results[1.]['candidates'] for o in c['outcomes']]
    assert any(not o['completeExecution'] for o in outcomes)
    assert any(o['failure'] for o in outcomes)
    assert all(o['anatomicalEncounter']['baselineRemovedContextMm3'] > 0. for o in outcomes)
    assert engine.snapshot() == before


@pytest.mark.parametrize('damage', ['candidate-omission', 'world-omission', 'world-duplicate', 'matrix', 'both-hands', 'duration', 'zero-offset'])
def test_resealed_frozen_family_or_transform_forgery_is_refused_before_replay(context_case, context_report, studies, monkeypatch, damage):
    changed = copy.deepcopy(studies[1.]); assessment = changed['assessment']
    if damage == 'candidate-omission': assessment['candidates'].pop()
    elif damage == 'world-omission': assessment['candidates'][0]['outcomes'].pop()
    elif damage == 'world-duplicate': assessment['candidates'][0]['outcomes'][1] = copy.deepcopy(assessment['candidates'][0]['outcomes'][0])
    elif damage == 'matrix': assessment['scenarioConfiguration']['offsetMm'] = .5
    elif damage == 'zero-offset': assessment['controls']['offsetMm'] = 0.
    else:
        # First nominal outcome is checked before any replay. Altering either
        # hand's held program or its duration must fail independently of seals.
        program = assessment['candidates'][0]['outcomes'][0]['actionProgram']
        action = next(a for a in program if a['slot'] == 'right' and 'pose' in a) if damage == 'both-hands' else program[0]
        if damage == 'both-hands': action['pose']['position'][0] += .5
        else: action['duration'] += .01
    reseal(changed)
    # Duplicate nonnominal worlds reach the first nominal replay before their
    # altered action program is inspected, so let that case use the real audit.
    if damage != 'world-duplicate':
        def forbidden(*args, **kwargs): pytest.fail('Malformed structural input reached native replay')
        monkeypatch.setattr(audit, 'replay', forbidden)
    with pytest.raises(ValueError): audit.verify(changed, context_report, context_case[1])


@pytest.mark.parametrize('damage', ['physical-prefix', 'context-prefix', 'union', 'progress', 'withdrawal', 'reassessment', 'failure-prefix'])
def test_self_resealed_native_outcome_forgery_is_refused(context_case, context_report, studies, damage):
    changed = copy.deepcopy(studies[1.]); outcomes = changed['assessment']['candidates'][0]['outcomes']
    nominal = outcomes[0]
    if damage == 'physical-prefix': nominal['physicalStateHashes'][0] = 'invented'
    elif damage == 'context-prefix': nominal['contextPrefixHashes'].pop()
    elif damage == 'union': nominal['anatomicalEncounter']['nearSourceCells'].append(-1)
    elif damage == 'progress': nominal['summary']['additionalTargetMm3'] += .5
    elif damage == 'withdrawal': nominal['withdrawalComplete'] = False
    elif damage == 'reassessment': nominal['reassessment'] = None
    else:
        failed = next(o for o in outcomes if o['failure'])
        failed['failure']['actionIndex'] += 1
    reseal(changed)
    with pytest.raises(ValueError, match='outcome differs'):
        audit.verify(changed, context_report, context_case[1])


def test_resealed_policy_and_fixed_margin_witness_cannot_be_invented(context_case, context_report, studies):
    changed = copy.deepcopy(studies[0.])
    changed['assessment']['policies']['fixedMargin'] = 'not-a-candidate'
    reseal(changed)
    with pytest.raises(ValueError, match='policy'):
        audit.verify(changed, context_report, context_case[1])
    changed = copy.deepcopy(studies[1.])
    changed['assessment']['candidates'][0]['outcomes'][0]['fixedMarginFailure'] = {'actionIndex': 0, 'reason': 'invented'}
    reseal(changed)
    with pytest.raises(ValueError, match='outcome differs'):
        audit.verify(changed, context_report, context_case[1])


def test_original_annotation_bytes_and_public_context_are_checked(context_case, context_report, studies):
    collection = context_case[4]
    path = collection.path(context_report['contextBinding']['original']['path'])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b'corrupt-after-study')
        with pytest.raises(ValueError, match='original|annotation|bytes'):
            audit.verify(studies[0.], context_report, context_case[1])
    finally:
        path.write_bytes(original)
    changed = copy.deepcopy(context_report)
    changed['publicPlan']['publicObservation']['anatomicalContext']['maskHash'] = 'invented-source'
    changed['publicPlan']['publicPlanSeal'] = semantic_digest({k: v for k, v in changed['publicPlan'].items() if k != 'publicPlanSeal'})
    changed['reportSeal'] = semantic_digest({k: v for k, v in changed.items() if k != 'reportSeal'})
    with pytest.raises(ValueError, match='public source context'):
        audit.verify(studies[0.], changed, context_case[1])


def test_burden_violation_is_checked_even_when_native_action_was_accepted(context_case, context_report):
    engine = context_case[1].clone(); before = copy.deepcopy(engine.state)
    controls = context_report['publicPlan']['constraints']
    targets = {r['id'] for r in engine.scene.regions if r.get('target')}
    cell = next(int(c) for c in np.flatnonzero((engine.scene.labels != 0) & ~np.isin(engine.scene.labels, list(targets))) if int(c) not in engine.state['removed'])
    engine.state['removed'].append(cell)
    assert audit.burden_reason(engine, before, controls) == 'TWO_TOOL_TASK: new non-target removal exceeds task'


def test_policy_uses_complete_scenario_matrix_and_deterministic_ties():
    def outcome(progress, seconds, *, context=True, fixed_failure=None):
        return {'meetsTask': progress, 'fixedMarginEncounter': {'withinLimit': context}, 'fixedMarginFailure': fixed_failure,
                'summary': {'elapsedSeconds': seconds, 'additionalTargetMm3': 1.}}
    rows = [{'id': 'short', 'outcomes': [outcome(True, .5), outcome(False, .5)]},
            {'id': 'long', 'outcomes': [outcome(True, 1.), outcome(True, 1.)]},
            {'id': 'stop', 'outcomes': [outcome(False, 0.), outcome(False, 0.)]}]
    assert audit.choose(rows) == 'short'
    assert audit.choose(rows, every=True) == 'long'
    rows[0]['outcomes'][0]['fixedMarginFailure'] = {'actionIndex': 1}
    assert audit.choose(rows, fixed=True) == 'long'
    rows[1]['outcomes'][1]['meetsTask'] = False
    assert audit.choose(rows, every=True) == 'stop'


def test_fixed_shaft_margin_counts_retained_targets_but_preserves_working_contact():
    from types import SimpleNamespace
    from resectionlab.surgical_scene import SurgicalScene
    labels = np.zeros((32, 32, 64), np.uint16)
    labels[0, 0, 0] = 1
    labels[12, 10, 20] = 2
    regions = [{'id': 1, 'name': 'Generated tissue', 'material': 'soft_tissue'},
               {'id': 2, 'name': 'Generated target', 'material': 'tumor', 'target': True}]
    scene = SurgicalScene(labels, np.eye(4), regions, {'kind': 'generated-audit-control'}, {})
    slots = {'left': {'instrument': 'cusa', 'position': [10., 10., 45.], 'quaternion': [0., 0., 0., 1.], 'parameters': {}},
             'right': {'instrument': 'frazier_suction', 'position': [30., 30., -10.], 'quaternion': [0., 0., 0., 1.], 'parameters': {}}}
    engine = SimpleNamespace(scene=scene, state={'slots': slots, 'removed': []}, protection=None)
    controls = {'geometry': {'nonTargetClearanceMm': 0.}}
    action = {'slot': 'left', 'operation': 'inspect'}
    assert audit.clearance_reason(engine, action, slots, controls, 0., fixed=True) is None
    assert audit.clearance_reason(engine, action, slots, controls, 1., fixed=True) == 'STRATEGY_SENSITIVITY: FIXED_MARGIN_SHAFT_CLEARANCE'
    engine.state['removed'] = [int(np.ravel_multi_index((12, 10, 20), labels.shape))]
    assert audit.clearance_reason(engine, action, slots, controls, 1., fixed=True) is None
    labels[12, 10, 20] = 0; labels[12, 10, 45] = 2
    engine.scene = SurgicalScene(labels, np.eye(4), regions, {'kind': 'generated-working-contact'}, {})
    engine.state['removed'] = []
    assert audit.clearance_reason(engine, action, slots, controls, .25, fixed=True) is None
