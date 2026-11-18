"""Behavioral checks for source accounting, research intent and canonical STOP."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_procedure_policy import (
    LEGACY_SELECTION_RULE, PROPERTY, procedure_admission, procedure_preservation_preference,
    procedure_source_memberships,
)
from resectionlab.surgical_treatment_intent import (
    PRESERVATION_PREFERENCE_SCHEMA, PRESERVATION_SELECTION_RULE, assess_treatment_execution,
    decide_treatment_intent, make_treatment_intent, procedure_treatment_intent,
)
from test_surgical_procedure_policy import action, prepared, revised, source


CST = 'structure:corticospinal:removedMm3'
MOTOR = 'structure:motor-cortex:removedMm3'


def option(name, benefit, *, cst=0., motor=0., admissible=True, blockers=()):
    return dict(id=name, treatmentMm3=benefit, harms={CST: cst, MOTOR: motor},
        admissible=admissible, blockingBoundaries=list(blockers))


def decide(alternatives, **options):
    return decide_treatment_intent(alternatives, incumbent=option('exit', 0.),
        required_harms=(CST, MOTOR), **options)


def test_declared_named_harm_preference_keeps_full_frontier_and_uses_no_exchange_rate():
    alternatives = [option('less-cst', 81., motor=54.), option('more-cst', 81., cst=54.), option('less-treatment', 27.)]
    unresolved = decide(alternatives)
    assert unresolved['status'] == 'reassess'
    assert unresolved['selectedId'] is None
    assert unresolved['reason'] == 'unresolved_treatment_harm_tradeoff'
    assert set(unresolved['controllingHarmMetrics']) == {CST, MOTOR}
    selected = decide(alternatives, preference_harms=(CST,))
    assert selected['status'] == 'continue' and selected['selectedId'] == 'less-cst'
    assert set(selected['frontierIds']) == {row['id'] for row in alternatives}
    assert selected['preferenceFrontierIds'] == ['less-cst']
    assert selected['controllingHarmMetrics'] == [CST]
    higher_harm = [option('more', 101., cst=1.), option('less', 100.)]
    assert decide(higher_harm, preference_harms=(CST,))['status'] == 'reassess'


def test_missing_evidence_reassesses_and_paired_worlds_are_never_averaged():
    missing = decide([option('observed-positive', 27.), option('missing-cst', 54., cst=None)],
        preference_harms=(CST,))
    assert missing['status'] == 'reassess' and missing['reason'] == 'required_evidence_unavailable'
    worlds = {'a': 0., 'b': 0.}
    paired = decide_treatment_intent([
        option('biased-a', {'a': 54., 'b': 27.}),
        option('biased-b', {'a': 27., 'b': 54.}),
    ], incumbent=option('exit', worlds), required_harms=(CST,), preference_harms=(CST,))
    assert paired['status'] == 'reassess'
    assert paired['frontierIds'] == ['biased-a', 'biased-b']


def intent_scene():
    base = source()
    cell = lambda point: int(np.ravel_multi_index(point, base.shape))
    first, second = cell((8, 8, 20)), cell((11, 8, 20))
    cerebellum = next(iter(procedure_source_memberships(base)['structure:cerebellum']))
    evidence = base.properties[PROPERTY]['evidence']
    boundary = dict(evidence, label='Declared cerebellar preservation boundary',
        structure='cerebellum', cells=[first, cerebellum])
    treatment = make_treatment_intent(lesion_id='engineered-lesion', revision='v2',
        original_cells=[first, second, cerebellum], eligible_cells=[second], evidence=evidence,
        boundaries={'preserve-cerebellum': boundary},
        compartments={'generated-enhancement': dict(evidence, kind='enhancing', cells=[second])})
    def declare(raw):
        raw['structures']['cerebellum']['cells'] = sorted([first, cerebellum])
        raw.update(selectionRule=PRESERVATION_SELECTION_RULE, treatmentIntent=treatment,
            preservationPreference={'schema': PRESERVATION_PREFERENCE_SCHEMA,
                'metrics': ['structure:cerebellum:removedMm3']})
    return revised(base, declare)


def test_source_original_burden_retains_protected_overlap_after_checked_removal():
    engine = prepared(intent_scene())
    before = procedure_treatment_intent(engine)
    assert before['lesionId'] == 'engineered-lesion' and before['lesionRevision'] == 'v2'
    assert before['coverageComplete'] is True
    assert before['originalBurdenMm3'] == pytest.approx(375.)
    assert before['eligibleTreatmentMm3'] == pytest.approx(125.)
    assert before['residual']['protectedResidualMm3'] == pytest.approx(250.)
    assert before['diagnosis'] == {'status': 'unknown', 'label': None}
    assert before['baselineFunction'] == {'status': 'unknown', 'summary': None}
    assert engine.step(action('apply', [55., 40., 100.], duration=1.))['status'] == 'accepted'
    after = procedure_treatment_intent(engine)
    assert after['sourceBinding'] == before['sourceBinding']
    assert after['originalRemovedMm3'] == pytest.approx(125.)
    assert after['tumorCompartments'][0]['residualMm3'] == 0.
    assert after['unclassifiedResidualMm3'] == pytest.approx(250.)
    decision = decide([], residual=after['residual'], coverage_complete=after['coverageComplete'])
    assert decision['status'] == 'stop_with_residual'
    assert [row['id'] for row in decision['supportingBoundaries']] == ['preserve-cerebellum']
    restored = SurgicalEngine.restore(engine.scene, engine.snapshot())
    assert procedure_treatment_intent(restored) == after


def test_source_policy_legacy_behavior_and_independent_source_declaration_are_enforced():
    old = source()
    assert old.properties[PROPERTY]['selectionRule'] == LEGACY_SELECTION_RULE
    assert procedure_preservation_preference(old) is None
    assert procedure_treatment_intent(SurgicalEngine(old))['originalBurdenMm3'] is None
    wrong_source_mask = revised(intent_scene(), lambda raw: raw['treatmentIntent'].update(originalCells=[]))
    assert procedure_admission(wrong_source_mask)['blockers'][0]['code'] == 'PROCEDURE_INVALID_INTENT_ORIGINAL_TARGET'
    no_named_preference = revised(intent_scene(), lambda raw: raw.pop('preservationPreference'))
    assert procedure_admission(no_named_preference)['blockers'][0]['code'] == 'PROCEDURE_VERSIONED_PRESERVATION_PREFERENCE_REQUIRED'


def test_eligible_residual_requires_supported_refusals_and_keeps_unknown_separate():
    boundary = dict(id='cst', label='CST preservation boundary', residualMm3=60., eligibleResidualMm3=60.)
    residual = dict(residualMm3=60., eligibleResidualMm3=60., unexplainedResidualMm3=0., boundaries=[boundary])
    rejected = [option('behind-cst', 60., admissible=False, blockers=['cst'])]
    assert decide(rejected, residual=residual, coverage_complete=False)['status'] == 'reassess'
    result = decide(rejected, residual=residual, coverage_complete=True)
    assert result['status'] == 'stop_with_residual' and result['supportingBoundaries'] == [boundary]
    failed = copy.deepcopy(rejected)
    failed[0]['blockingBoundaries'] = ['tool_refusal_only']
    assert decide(failed, residual=residual, coverage_complete=True)['status'] == 'reassess'
    unknown = [option('not-replayed', 60., admissible=None)]
    assert decide(unknown, residual=residual, coverage_complete=True)['status'] == 'reassess'
    assert decide([], residual={'residualMm3': 0.}, coverage_complete=True)['status'] == 'stop_after_treatment'


def test_actual_stop_refused_while_inserted_then_accepted_without_changing_intent():
    pytest.importorskip('gymnasium')
    from test_surgical_rl_procedure import command, env, prepare

    test_env = env(intent_scene())
    prepare(test_env, x=55.)
    result = test_env.step_action(command('apply', [55., 40., 100.], duration=1.))
    assert result[4]['outcome']['status'] == 'accepted'
    report = procedure_treatment_intent(test_env.engine)
    intent = decide([], residual=report['residual'], coverage_complete=report['coverageComplete'])
    assert intent['status'] == 'stop_with_residual'
    failed = test_env.stop('Named preservation boundary')
    assessment = assess_treatment_execution(intent, stop_outcome=failed[4]['outcome'],
        terminated=failed[2], truncated=failed[3], termination_reason=failed[4]['termination_reason'])
    assert assessment['intentStatus'] == 'stop_with_residual'
    assert assessment['executionStatus'] == 'stop_refused'
    assert assessment['stopReasonCode'] == 'PROCEDURE_STOP_REQUIRES_CHECKED_EXIT'
    withdrawal = test_env.step_action(command('withdraw', [55., 40., 65.], duration=1.))
    assert withdrawal[4]['outcome']['status'] == 'accepted'
    before = test_env.engine.state_id
    accepted = test_env.stop('Named preservation boundary')
    assessment = assess_treatment_execution(intent, stop_outcome=accepted[4]['outcome'],
        terminated=accepted[2], truncated=accepted[3], termination_reason=accepted[4]['termination_reason'])
    assert assessment['executionStatus'] == 'justified_residual_stop'
    assert test_env.engine.state_id == before
    continuing = decide([option('more-with-preservation', 27.)])
    assert assess_treatment_execution(continuing, stop_outcome=accepted[4]['outcome'])['executionStatus'] == 'premature_stop'
    assert assess_treatment_execution(continuing, truncated=True, termination_reason='attempt_limit')['executionStatus'] == 'truncated'
    assert assess_treatment_execution(continuing, terminated=True,
        termination_reason='damage_budget:neural')['executionStatus'] == 'failure'
