"""Generated numerical controls for sealed user policies; no patient execution."""
import copy

import numpy as np
import pytest

from resectionlab import surgical_consequences
from resectionlab.core import semantic_digest
from resectionlab.surgical_consequences import consequence_vector, seal_report
from resectionlab.surgical_consequence_analysis import (
    MAX_REVISIONS, analysis_view, normalize_policy_input, reevaluate_analysis,
    validate_saved_analyses,
)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene

CASE_HASH = 'a' * 64


def policy(**changes):
    return {'minimumAdditionalTargetMm3': 0., 'maximumCumulativeNonTargetMm3': 0.,
            'requiredMetrics': [], 'maxBrainStrainSquaredMm3S': None,
            'maxBrainStressSquaredPa2Mm3S': None, **changes}


@pytest.fixture
def admitted(monkeypatch):
    def build(*, tie=False, dirty_stop=False):
        labels = np.zeros((5, 5, 5), np.uint16)
        labels[2, 2, 2] = 1
        scene = SurgicalScene(labels, np.eye(4),
            [{'id': 1, 'name': 'Generated target', 'material': 'tumor', 'target': True}],
            {'kind': 'generated-control'}, {'tissuePhysicsV2': default_profile()})
        base = SurgicalEngine(scene)
        assert base.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
        engines = {'main': base}
        for name, duration in [('short', .1), ('long', .1 if tie else .2), ('STOP', .05)]:
            e = base.clone()
            assert e.step({'slot': 'left', 'operation': 'inspect', 'duration': duration})['status'] == 'accepted'
            engines[name] = e
        if dirty_stop:
            # A real saved activation leaves cleanup incomplete while removal
            # increments remain zero; comparison must refuse this STOP.
            assert engines['STOP'].step({'slot': 'left', 'operation': 'activate', 'duration': .05})['status'] == 'accepted'
        vectors = [{'branchId': name, 'scenarioId': 'nominal', 'vector': consequence_vector(engines[name], base)}
                   for name in ['short', 'long', 'STOP']]
        frozen_policy = {'id': 'generated-frozen', 'basis': 'declared_research_task_not_clinical_threshold',
                         'objectives': ['elapsedSeconds'], 'constraints': [], 'requiredMetrics': []}
        declaration = {'sourceArchiveSha256': '0' * 64, 'sourceSceneId': scene.identity,
            'sourceGeometryHash': vectors[0]['vector']['sourceGeometryHash'],
            'prefixBranch': 'main', 'prefixActions': 1, 'alternativeBranches': ['short', 'long', 'STOP'],
            'stopBranch': 'STOP', 'scenarios': [{'id': 'nominal', 'relaxationTimeScale': 1.}],
            'policies': [frozen_policy]}
        # A separately constructed generated pin, never an API override.
        monkeypatch.setattr(surgical_consequences, 'APP_DECLARATION_HASH', semantic_digest(declaration))
        scenarios = [{'id': 'nominal', 'relaxationTimeScale': 1., 'sourceSceneId': scene.identity,
                      'prefixStateId': base.state_id, 'profileHash': semantic_digest(None)}]
        report = seal_report(scene, vectors, scenarios, [frozen_policy], baseline_branch='main',
                             stop_branch='STOP', source_archive_sha256='0' * 64, declaration=declaration)
        record = {'branchId': 'main', 'position': 1,
                  'branches': {name: e.snapshot() for name, e in engines.items()},
                  'consequenceComparison': report}
        return base, record, report, engines
    return build


def evaluate(base, record, report, p=None):
    return reevaluate_analysis(base, record, report, CASE_HASH, policy() if p is None else p,
                               report['comparisonSeal'])


def test_threshold_revision_required_unknown_and_no_original_mutation(admitted):
    base, record, report, _ = admitted()
    original = copy.deepcopy((record, report, base.snapshot()))
    first = evaluate(base, record, report)
    saved = first['consequenceAnalyses'][0]
    assert saved['decision']['selectedBranchId'] == 'short'
    assert saved['decision']['taskSatisfied']
    assert saved['policy']['objectives'] == ['targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds']
    assert saved['policy']['constraints'][1]['scope'] == 'cumulative'
    second = evaluate(base, first, report, policy(minimumAdditionalTargetMm3=1.))
    assert second['consequenceAnalyses'][:1] == first['consequenceAnalyses']
    decision = second['consequenceAnalyses'][-1]['decision']
    assert decision['selectedBranchId'] == 'STOP' and not decision['taskSatisfied']
    third = evaluate(base, second, report, policy(requiredMetrics=['motorEncounterMm3', 'languageEncounterMm3']))
    last = third['consequenceAnalyses'][-1]
    assert {r['metric'] for r in last['decision']['violations']['short']} == {'motorEncounterMm3', 'languageEncounterMm3'}
    assert last['revision'] == 3 and last['newPhysicalActions'] == last['optimizerUpdates'] == 0
    assert last['clinicalValidation'] is False
    assert (record, report, base.snapshot()) == original
    view = analysis_view(base, third, report, CASE_HASH)
    assert view['available'] and view['current'] and view['revisionCount'] == 3
    view['latest']['policyInput']['requiredMetrics'].clear()
    assert third['consequenceAnalyses'][-1]['policyInput']['requiredMetrics']


def test_ambiguous_frontier_and_stop_cleanup_residual_preserved(admitted):
    base, record, report, _ = admitted(tie=True)
    saved = evaluate(base, record, report)['consequenceAnalyses'][0]
    decision = saved['decision']
    assert set(decision['pairedScenarioParetoFront']) == {'short', 'long'}
    assert decision['selectedBranchId'] == 'STOP' and decision['stopFallback']
    assert not decision['taskSatisfied'] and decision['stopExecutable']
    stop = next(row['vector'] for row in report['samples'] if row['branchId'] == 'STOP')
    assert stop['cleanupComplete'] and stop['residualTargetMm3'] == 1.
    assert stop['dimensions']['targetRemovedMm3']['incremental'] == 0.
    assert stop['dimensions']['nonTargetRemovedMm3']['incremental'] == 0.


def test_dirty_stop_does_not_become_executable(admitted):
    base, record, report, _ = admitted(dirty_stop=True)
    decision = evaluate(base, record, report, policy(minimumAdditionalTargetMm3=1.))['consequenceAnalyses'][0]['decision']
    assert not decision['stopExecutable'] and decision['selectedBranchId'] is None
    assert not decision['taskSatisfied']


@pytest.mark.parametrize('required', ['supplyDeficitSeconds', 'motorEncounterMm3', 'languageEncounterMm3', 'patientShiftErrorMm'])
def test_all_requested_unknowns_stay_unknown(admitted, required):
    base, record, report, _ = admitted()
    decision = evaluate(base, record, report, policy(requiredMetrics=[required]))['consequenceAnalyses'][0]['decision']
    assert decision['selectedBranchId'] == 'STOP' and not decision['taskSatisfied']
    assert any(v.get('metric') == required and v['reason'] == 'unknown_required_consequence'
               for v in decision['violations']['short'])


def test_optional_mechanics_are_cumulative_and_unavailable_is_not_zero(admitted):
    base, record, report, _ = admitted()
    saved = evaluate(base, record, report, policy(maxBrainStrainSquaredMm3S=0., maxBrainStressSquaredPa2Mm3S=0.))['consequenceAnalyses'][0]
    assert [c['scope'] for c in saved['policy']['constraints']] == ['incremental', 'cumulative', 'cumulative', 'cumulative']
    assert saved['decision']['selectedBranchId'] == 'STOP'
    assert all(row['vector']['dimensions']['brainStrainSquaredMm3S']['cumulative'] is None for row in report['samples'])


@pytest.mark.parametrize('changes', [
    {'minimumAdditionalTargetMm3': True}, {'maximumCumulativeNonTargetMm3': -1.},
    {'minimumAdditionalTargetMm3': float('inf')}, {'minimumAdditionalTargetMm3': 10 ** 400}, {'maxBrainStrainSquaredMm3S': float('nan')},
    {'maxBrainStressSquaredPa2Mm3S': False}, {'requiredMetrics': ['thermalPeakC']},
    {'requiredMetrics': ['motorEncounterMm3', 'motorEncounterMm3']},
    {'objectives': ['elapsedSeconds']}, {'clinicalValidation': True}, {'basis': 'clinical'},
])
def test_policy_rejects_unbounded_or_promoted_fields(changes):
    with pytest.raises(ValueError): normalize_policy_input(policy(**changes))


def test_policy_requires_exact_fields():
    p = policy(); del p['maxBrainStrainSquaredMm3S']
    with pytest.raises(ValueError): normalize_policy_input(p)


def test_stale_seal_and_exact_baseline_required_historical_recovery(admitted):
    base, record, report, engines = admitted()
    saved = evaluate(base, record, report)
    with pytest.raises(ValueError, match='Stale'):
        reevaluate_analysis(base, saved, report, CASE_HASH, policy(), 'sha256:' + '0' * 64)
    moved = copy.deepcopy(saved); moved['branchId'] = 'short'; moved['position'] = len(engines['short'].actions)
    view = analysis_view(engines['short'], moved, report, CASE_HASH)
    assert not view['available'] and not view['current'] and view['latest'] == saved['consequenceAnalyses'][-1]
    assert view['baseline'] == {'branchId': 'main', 'position': 1, 'stateId': base.state_id}
    with pytest.raises(ValueError, match='exact frozen'):
        evaluate(engines['short'], moved, report)
    moved['branchId'] = 'main'; moved['position'] = 0
    assert not analysis_view(base, moved, report, CASE_HASH)['available']
    with pytest.raises(ValueError): evaluate(base, moved, report)
    moved['position'] = 1
    assert analysis_view(base, moved, report, CASE_HASH)['current']
    assert len(evaluate(base, moved, report)['consequenceAnalyses']) == 2
    stale_engine = base.clone(); stale_engine.state['elapsedSeconds'] += .1
    with pytest.raises(ValueError): evaluate(stale_engine, saved, report)


@pytest.mark.parametrize('mutate', [
    lambda a: a.__setitem__('analysisSeal', 'sha256:' + '0' * 64),
    lambda a: a.__setitem__('clinicalValidation', True),
    lambda a: a.__setitem__('optimizerUpdates', False),
    lambda a: a['decision'].__setitem__('selectedBranchId', 'long'),
    lambda a: a['policyInput'].__setitem__('minimumAdditionalTargetMm3', 1.),
    lambda a: a['policy']['objectives'].append('brainStrainSquaredMm3S'),
    lambda a: a['binding']['baseline'].__setitem__('stateId', 'invented'),
    lambda a: a['binding']['programs'][0].__setitem__('snapshotHash', 'invented'),
    lambda a: a['binding'].__setitem__('scenariosHash', 'invented'),
    lambda a: a.__setitem__('caseHash', 'b' * 64),
    lambda a: a.__setitem__('revision', 2),
])
def test_analysis_tampering_rejected_even_if_resealed(admitted, mutate):
    base, record, report, _ = admitted()
    saved = evaluate(base, record, report)
    item = saved['consequenceAnalyses'][0]; mutate(item)
    if item['analysisSeal'] != 'sha256:' + '0' * 64:
        item['analysisSeal'] = semantic_digest({k: v for k, v in item.items() if k != 'analysisSeal'})
    with pytest.raises(ValueError): validate_saved_analyses(base, saved, report, CASE_HASH)
    with pytest.raises(ValueError): analysis_view(base, saved, report, CASE_HASH)
    with pytest.raises(ValueError): evaluate(base, saved, report)


@pytest.mark.parametrize('mutate', [
    lambda r, p: r['branches']['main']['checkpoint'].__setitem__('position', 0),
    lambda r, p: r['branches']['short']['checkpoint']['state'].__setitem__('elapsedSeconds', 10.),
    lambda r, p: r['branches']['short']['actions'][1].__setitem__('duration', .3),
    lambda r, p: r['branches']['short']['checkpoint'].__setitem__('stateId', 'invented'),
    lambda r, p: p['scenarios'][0].__setitem__('prefixStateId', 'invented'),
    lambda r, p: p['scenarios'][0].__setitem__('relaxationTimeScale', 2.),
    lambda r, p: p.__setitem__('sourceSceneId', 'invented'),
    lambda r, p: p.__setitem__('sourceRole', 'CLINICAL'),
    lambda r, p: p['declaration'].__setitem__('prefixActions', 0),
])
def test_frozen_source_program_checkpoint_scenario_tampering_rejected(admitted, mutate):
    base, record, report, _ = admitted()
    saved = evaluate(base, record, report); changed_report = saved['consequenceComparison']
    mutate(saved, changed_report)
    changed_report['comparisonSeal'] = semantic_digest({k: v for k, v in changed_report.items() if k != 'comparisonSeal'})
    with pytest.raises(ValueError): analysis_view(base, saved, changed_report, CASE_HASH)


def test_missing_or_detached_original_report_is_rejected(admitted):
    base, record, report, _ = admitted()
    del record['consequenceComparison']
    with pytest.raises(ValueError, match='original comparison'): evaluate(base, record, report)


def test_revision_limit_preserves_every_revision(admitted):
    base, record, report, _ = admitted()
    for i in range(MAX_REVISIONS):
        record = evaluate(base, record, report, policy(minimumAdditionalTargetMm3=float(i)))
    original = copy.deepcopy(record)
    assert [a['revision'] for a in record['consequenceAnalyses']] == list(range(1, MAX_REVISIONS + 1))
    with pytest.raises(ValueError, match='limit reached'): evaluate(base, record, report)
    assert record == original
    validate_saved_analyses(base, record, report, CASE_HASH)
