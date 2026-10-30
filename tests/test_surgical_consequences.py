"""Paired-world partial orders, missing evidence, exact prefixes and STOP."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_consequences import (
    METRICS, VECTOR_SCHEMA, compare_samples, consequence_vector,
    seal_report, validate_saved_comparison, validate_policy,
    comparison_declaration, APP_DECLARATION_HASH,
)


def policy(required=()):
    return {'id': 'generated-control', 'basis': 'declared_research_task_not_clinical_threshold',
        'objectives': ['targetRemovedMm3', 'nonTargetRemovedMm3', 'brainStrainSquaredMm3S'],
        'constraints': [{'metric': 'targetRemovedMm3', 'scope': 'incremental', 'operator': 'at_least', 'value': 8.}],
        'requiredMetrics': list(required)}


def samples(costs=((1., 2.), (2., 3.))):
    rows = []
    for world in range(2):
        for branch, gain, cost in [('short', 8., costs[0][world]), ('long', 8., costs[1][world]), ('STOP', 0., .5)]:
            known = {'targetRemovedMm3': (560. + gain, gain), 'nonTargetRemovedMm3': (4940., 0.),
                     'brainStrainSquaredMm3S': (cost, cost)}
            dimensions = {k: {'unit': unit, 'direction': direction,
                'status': ('source_count' if k in ('targetRemovedMm3', 'nonTargetRemovedMm3', 'elapsedSeconds') else 'modeled_unvalidated') if k in known else 'unavailable',
                'cumulative': known[k][0] if k in known else None,
                'incremental': known[k][1] if k in known else None,
                'reason': None if k in known else 'Generated unknown evidence control'}
                for k, (unit, direction) in METRICS.items()}
            rows.append({'branchId': branch, 'scenarioId': str(world), 'vector': {
                'schema': VECTOR_SCHEMA, 'sourceGeometryHash': 'same-geometry', 'sourceSceneId': 'scene-'+str(world),
                'baselineStateId': 'prefix-'+str(world), 'prefixActions': 15, 'cleanupComplete': True,
                'clinicalValidation': False,
                'dimensions': dimensions}})
    return rows


def compare(rows=None, p=None):
    return compare_samples(samples() if rows is None else rows, policy() if p is None else p,
                           stop_branch='STOP', scenario_ids=['0', '1'], branch_ids=['short', 'long', 'STOP'])


def test_dominance_keeps_full_access_cost_and_does_not_use_scalar_weights():
    result = compare()
    assert result['selectedBranchId'] == 'short' and result['taskSatisfied']
    assert result['dominatedBy'] == {'long': ['short'], 'short': []}
    assert result['objectiveScopes']['nonTargetRemovedMm3'] == 'cumulative'
    assert samples()[0]['vector']['dimensions']['nonTargetRemovedMm3']['cumulative'] == 4940.
    assert samples()[0]['vector']['dimensions']['nonTargetRemovedMm3']['incremental'] == 0.
    p = policy(); p['weights'] = {'brainStrainSquaredMm3S': .001}
    with pytest.raises(ValueError, match='policy'): validate_policy(p)


def test_packaged_app_declaration_pin_matches_checked_in_experiment():
    p = Path(__file__).resolve().parents[1] / 'manifests/experiments/consequence-comparison-v1.json'
    assert semantic_digest(comparison_declaration(json.loads(p.read_text()))) == APP_DECLARATION_HASH


@pytest.mark.parametrize('edited_branch', ['main', 'short', 'long', 'STOP'])
def test_native_edits_preserve_compared_endpoints_and_prefix_on_reopen(tmp_path, monkeypatch, edited_branch):
    from resectionlab import surgical_consequences
    from resectionlab.core import freeze_json, thaw_json
    from resectionlab.desktop_bridge import BridgeSession, _Request
    session = BridgeSession(tmp_path / 'transfers')
    def call(operation, **args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)
    try:
        case = call('createSurgicalFixture')
        key = case['caseHash']
        call('surgicalRehearsal', caseHash=key, command='create')
        base = session.surgical_sessions[key][1]
        engines = {'main': base, 'STOP': base, 'short': base.clone(), 'long': base.clone()}
        for name, duration in [('short', .1), ('long', .2)]:
            assert engines[name].step({'slot': 'left', 'operation': 'inspect', 'duration': duration})['status'] == 'accepted'
        vectors = [{'scenarioId': 'nominal', 'branchId': name, 'vector': consequence_vector(engines[name], base)}
                   for name in ['short', 'long', 'STOP']]
        p = {'id': 'generated-time', 'basis': 'declared_research_task_not_clinical_threshold',
             'objectives': ['elapsedSeconds'], 'constraints': [], 'requiredMetrics': []}
        declaration = {'alternativeBranches': ['short', 'long', 'STOP'], 'prefixBranch': 'main', 'prefixActions': 0,
                       'stopBranch': 'STOP', 'sourceArchiveSha256': '0'*64, 'sourceSceneId': base.scene.identity,
                       'sourceGeometryHash': vectors[0]['vector']['sourceGeometryHash'],
                       'scenarios': [{'id': 'nominal', 'relaxationTimeScale': 1.}], 'policies': [p]}
        # Independently trusted generated fixture pin; production has no artifact-controlled override.
        monkeypatch.setattr(surgical_consequences, 'APP_DECLARATION_HASH', semantic_digest(declaration))
        scenarios = [{'id': 'nominal', 'sourceSceneId': base.scene.identity, 'relaxationTimeScale': 1.,
                      'prefixStateId': base.state_id, 'profileHash': semantic_digest(None)}]
        report = seal_report(base.scene, vectors, scenarios, [p], baseline_branch='main', stop_branch='STOP',
                             source_archive_sha256='0'*64, declaration=declaration)
        entry = session._get_case(key); artifacts = thaw_json(entry.artifacts)
        record = artifacts['surgical_toolbox']
        frozen_branches = {name: e.snapshot() for name, e in engines.items()}
        record.update(branches=frozen_branches, consequenceComparison=report,
                      timeline={name: e.history for name, e in engines.items()})
        entry.artifacts = freeze_json(artifacts)
        state = call('surgicalRehearsal', caseHash=key, command='inspect')
        policy_input = {'minimumAdditionalTargetMm3': 0., 'maximumCumulativeNonTargetMm3': 0.,
                        'requiredMetrics': [], 'maxBrainStrainSquaredMm3S': None,
                        'maxBrainStressSquaredPa2Mm3S': None}
        state = call('surgicalRehearsal', caseHash=key, command='reevaluateConsequences',
                     expectedStateId=state['view']['stateId'],
                     expectedComparisonSeal=report['comparisonSeal'], policyInput=policy_input)
        analysis = copy.deepcopy(state['consequenceAnalysis']['latest'])
        assert state['consequenceAnalysis']['current'] and analysis['revision'] == 1
        assert analysis['newPhysicalActions'] == analysis['optimizerUpdates'] == 0
        state = call('surgicalRehearsal', caseHash=key, command='selectBranch', branchId=edited_branch,
                     expectedStateId=state['view']['stateId'])
        state = call('surgicalRehearsal', caseHash=key, command='step', expectedStateId=state['view']['stateId'],
                     action={'slot': 'left', 'operation': 'inspect', 'duration': .1})
        assert state['branchId'] not in frozen_branches
        assert state['consequenceComparison'] == report
        assert not state['consequenceAnalysis']['current']
        assert state['consequenceAnalysis']['latest'] == analysis
        saved = thaw_json(entry.artifacts)['surgical_toolbox']
        assert all(saved['branches'][name] == snapshot for name, snapshot in frozen_branches.items())
        assert saved['parents'][state['branchId']] == {'branchId': edited_branch, 'position': len(engines[edited_branch].actions)}
        path = tmp_path / 'compared.ressectionlab'
        call('saveCase', caseHash=key, path=str(path), overwrite=False)
        session.close(); session = BridgeSession(tmp_path / 'fresh')
        reopened = call('loadCase', path=str(path))
        restored = call('surgicalRehearsal', caseHash=reopened['caseHash'], command='inspect')
        assert restored['view']['stateId'] == state['view']['stateId']
        assert restored['consequenceComparison'] == report
        assert restored['consequenceAnalysis']['latest'] == analysis
        assert not restored['consequenceAnalysis']['current']
        assert all(thaw_json(session._get_case(key).artifacts)['surgical_toolbox']['branches'][name] == snapshot
                   for name, snapshot in frozen_branches.items())
    finally:
        session.close()


def test_crossing_worlds_remain_tradeoff_even_when_mean_would_choose_one():
    result = compare(samples(((0., 10.), (4., 7.))))
    assert set(result['pairedScenarioParetoFront']) == {'short', 'long'}
    assert result['selectedBranchId'] == 'STOP' and result['stopFallback']
    assert not result['taskSatisfied']
    assert result['reason'] == 'unresolved_tradeoff_stop_with_residual'


def test_missing_required_motor_and_language_are_separate_unknowns_and_stop():
    result = compare(p=policy(['motorEncounterMm3', 'languageEncounterMm3']))
    assert result['selectedBranchId'] == 'STOP' and not result['taskSatisfied']
    blocked = result['violations']['short']
    assert {v['metric'] for v in blocked} == {'motorEncounterMm3', 'languageEncounterMm3'}
    assert len(blocked) == 4


def test_stop_is_not_claimed_to_meet_target_task_or_to_be_executable_with_new_removal():
    rows = samples(); p = policy(['patientShiftErrorMm'])
    result = compare(rows, p)
    assert result['stopExecutable'] and not result['taskSatisfied']
    for row in rows:
        if row['branchId'] == 'STOP': row['vector']['dimensions']['targetRemovedMm3']['incremental'] = 1.
    result = compare(rows, p)
    assert not result['stopExecutable'] and result['selectedBranchId'] is None
    assert result['reason'] == 'no_executable_stop'


@pytest.mark.parametrize('mutate', [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows.__setitem__(slice(None), [r for r in rows if r['scenarioId'] != '1']),
    lambda rows: rows.__setitem__(slice(None), [r for r in rows if r['branchId'] != 'short']),
    lambda rows: rows[0]['vector'].__setitem__('sourceGeometryHash', 'different-geometry'),
    lambda rows: rows[0]['vector'].__setitem__('baselineStateId', 'other-prefix'),
    lambda rows: rows[0]['vector'].__setitem__('sourceSceneId', 'other-scene'),
    lambda rows: rows[0]['vector']['dimensions']['brainStrainSquaredMm3S'].__setitem__('unit', 'mm'),
    lambda rows: rows[0]['vector']['dimensions']['brainStrainSquaredMm3S'].__setitem__('direction', 'max'),
    lambda rows: rows[0]['vector']['dimensions']['motorEncounterMm3'].__setitem__('cumulative', 0.),
    lambda rows: rows[0]['vector']['dimensions']['brainStrainSquaredMm3S'].__setitem__('cumulative', float('nan')),
])
def test_missing_world_source_unit_direction_and_unknown_coercion_are_refused(mutate):
    rows = samples(); mutate(rows)
    with pytest.raises(ValueError): compare(rows)


def test_sample_order_does_not_change_paired_frontier():
    original = samples(); expected = compare(original)
    for order in itertools.islice(itertools.permutations(range(len(original))), 20):
        assert compare([original[i] for i in order]) == expected


def source():
    labels = np.zeros((5, 5, 5), np.uint16); labels[2, 2, 2] = 1
    return SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Source target', 'material': 'tumor', 'target': True}],
                         {'kind': 'generated-control'}, {'tissuePhysicsV2': default_profile()})


def test_vector_requires_exact_prefix_and_absent_physics_never_becomes_zero():
    e = SurgicalEngine(source()); base = e.clone()
    assert e.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
    vector = consequence_vector(e, base)
    assert vector['dimensions']['brainStrainSquaredMm3S']['cumulative'] is None
    assert vector['dimensions']['thermalPeakC']['cumulative'] is None
    assert vector['dimensions']['supplyDeficitSeconds']['cumulative'] is None
    assert vector['dimensions']['targetRemovedMm3']['cumulative'] == 0.
    other = base.clone(); other.step({'slot': 'right', 'operation': 'inspect', 'duration': .1})
    with pytest.raises(ValueError, match='prefix'): consequence_vector(e, other)


def test_real_heat_adapter_retains_units_and_does_not_turn_missing_vessels_into_zero():
    from resectionlab import surgical_bipolar_thermal as heat
    labels = np.zeros((8, 5, 5), np.uint16); labels[2:4, 1, 2] = 1
    cells = list(map(int, np.flatnonzero(labels)))
    p = heat.research_profile(cells, {'1': {'volumetricHeatCapacityJPerMm3K': .00383,
        'conductivityWPerMmK': .000553, 'source': 'Generated material scenario'}},
        ambient_c=33., watts_per_activation=.02, maximum_c=60.)
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Target', 'material': 'tumor', 'target': True}],
                          {'kind': 'generated-control'}, {'tissuePhysicsV2': default_profile(), heat.PROPERTY: p})
    e = SurgicalEngine(scene); base = e.clone()
    for action in [
        {'operation': 'configure', 'mechanicsRevision': 2},
        {'operation': 'exchange', 'instrument': 'bipolar_forceps'},
        {'operation': 'place', 'pose': {'position': [2.5, 1., -4.], 'quaternion': [0., 0., 0., 1.]},
         'parameters': {'jawOpeningMm': .5, 'forceN': 0., 'power': 0., 'depthMm': 1.}},
        {'operation': 'apply', 'pose': {'position': [2.5, 1., 1.8], 'quaternion': [0., 0., 0., 1.]}, 'duration': .5},
        {'operation': 'apply', 'parameters': {'power': 1.}, 'duration': .2}]:
        assert e.step({'slot': 'left', 'duration': .1, **action})['status'] == 'accepted'
    dimensions = consequence_vector(e, base)['dimensions']
    assert dimensions['thermalPeakC']['cumulative'] == pytest.approx(33. + .004/(2*.00383))
    assert dimensions['thermalPeakC']['unit'] == 'degC'
    assert dimensions['thermalExposureMm3KS']['cumulative'] > 0.
    assert dimensions['bloodGeneratedMm3']['cumulative'] is None


def test_graph_adapter_reports_max_declared_deficit_without_summing_overlapping_regions():
    from resectionlab import surgical_vascular_graph as graph
    labels = np.zeros((5, 5, 5), np.uint16); labels[2, 2, 2] = 1; labels[0:2, 0, 0] = 2
    cells = [int(np.ravel_multi_index((x, 0, 0), labels.shape)) for x in (0, 1)]
    p = {'schema': graph.SCHEMA, 'provenance': {'kind': 'assumed-research', 'description': 'Generated coefficients'},
         'coverage': {'sourceCaseId': 'generated', 'status': 'generated-control', 'missing': ['No patient data']},
         'fieldPressurePa': 0., 'compressionForceN': 1., 'coagulationDose': 2., 'clotRatePerSecond': 0.,
         'nodes': [{'id': 'a', 'complianceMm3PerPa': .001, 'initialStorageMm3': 1., 'source': 'Generated'},
                   {'id': 'b', 'complianceMm3PerPa': .001, 'initialStorageMm3': .5, 'source': 'Generated'}],
         'edges': [{'id': 'edge', 'from': 'a', 'to': 'b', 'kind': 'artery', 'conductanceMm3PerPaS': .001,
                    'sourceCells': cells, 'sourceFace': cells, 'woundConductanceMm3PerPaS': [.001, .001], 'source': 'Generated'}],
         'boundaries': [{'id': 'in', 'node': 'a', 'pressurePa': 1000., 'conductanceMm3PerPaS': .001, 'source': 'Generated'},
                        {'id': 'out', 'node': 'b', 'pressurePa': 0., 'conductanceMm3PerPaS': .001, 'source': 'Generated'}],
         'regions': [{'id': name, 'supply': [{'edge': 'edge', 'direction': 1}], 'drainage': [{'edge': 'edge', 'direction': 1}],
                      'storageNodes': ['b'], 'referenceSupplyMm3PerSecond': ref, 'referenceDrainageMm3PerSecond': ref,
                      'source': 'Deliberately overlapping generated interfaces'} for name, ref in [('one', .5), ('two', 1.)]]}
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Target', 'material': 'tumor', 'target': True},
        {'id': 2, 'name': 'Declared vessel', 'material': 'vessel'}], {'kind': 'generated'},
        {'tissuePhysicsV2': default_profile(), graph.PROPERTY: p})
    e = SurgicalEngine(scene); base = e.clone()
    assert e.step({'slot': 'left', 'operation': 'inspect', 'duration': 1.})['status'] == 'accepted'
    v = consequence_vector(e, base)['dimensions']; regions = graph.metrics(e)['regions']
    values = [r['supplyDeficitSeconds'] for r in regions.values()]
    assert min(values) > 0 and v['supplyDeficitSeconds']['cumulative'] == max(values)
    assert v['supplyDeficitSeconds']['cumulative'] < sum(values)
    assert v['supplyDeficitSeconds']['status'] == 'modeled_unvalidated'


def test_resealed_saved_nominal_ledger_cannot_disagree_with_checkpoint():
    scene = source(); base = SurgicalEngine(scene); a = base.clone(); b = base.clone()
    a.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    b.step({'slot': 'left', 'operation': 'inspect', 'duration': .2})
    vectors = [{'scenarioId': 'nominal', 'branchId': name, 'vector': consequence_vector(e, base)}
               for name, e in [('a', a), ('b', b), ('STOP', base)]]
    p = {'id': 'source-cost', 'basis': 'declared_research_task_not_clinical_threshold',
         'objectives': ['elapsedSeconds'], 'constraints': [], 'requiredMetrics': []}
    scenarios = [{'id': 'nominal', 'sourceSceneId': scene.identity, 'relaxationTimeScale': 1.,
                  'prefixStateId': base.state_id, 'profileHash': semantic_digest(None)}]
    declaration = {'alternativeBranches': ['a', 'b', 'STOP'], 'prefixBranch': 'main', 'prefixActions': 0,
                   'stopBranch': 'STOP', 'sourceArchiveSha256': '0'*64,
                   'sourceSceneId': scene.identity, 'sourceGeometryHash': vectors[0]['vector']['sourceGeometryHash'],
                   'scenarios': [{'id': 'nominal', 'relaxationTimeScale': 1.}], 'policies': [p]}
    report = seal_report(scene, vectors, scenarios, [p], baseline_branch='main', stop_branch='STOP', source_archive_sha256='0'*64, declaration=declaration)
    record = {'branches': {'main': base.snapshot(), 'a': a.snapshot(), 'b': b.snapshot(), 'STOP': base.snapshot()}}
    validate_saved_comparison(scene, record, report, declaration=declaration)
    bad = copy.deepcopy(report)
    bad['samples'][0]['vector']['dimensions']['elapsedSeconds']['cumulative'] = .05
    bad['comparisonSeal'] = semantic_digest({k: v for k, v in bad.items() if k != 'comparisonSeal'})
    with pytest.raises(ValueError, match='ledger'): validate_saved_comparison(scene, record, bad, declaration=declaration)
    for mutate in [lambda r: r.__setitem__('decisions', []),
                   lambda r: r['scenarios'][0].__setitem__('id', 'renamed-world'),
                   lambda r: r['decisions'].append(copy.deepcopy(r['decisions'][0]))]:
        bad = copy.deepcopy(report); mutate(bad)
        bad['comparisonSeal'] = semantic_digest({k: v for k, v in bad.items() if k != 'comparisonSeal'})
        with pytest.raises(ValueError): validate_saved_comparison(scene, record, bad, declaration=declaration)
    for mutate in [lambda r: r.__setitem__('sourceRole', 'MEASUREMENT_EVAL'),
                   lambda r: r['samples'][0]['vector'].__setitem__('sourceGeometryHash', 'invented'),
                   lambda r: r['samples'][0]['vector'].__setitem__('sourceSceneId', 'invented'),
                   lambda r: r['samples'][0]['vector'].__setitem__('clinicalValidation', True),
                   lambda r: r['samples'][0]['vector'].__setitem__('residualTargetMm3', -999.),
                   lambda r: r['samples'][0].__setitem__('scenarioId', 'undeclared-world')]:
        bad = copy.deepcopy(report); mutate(bad)
        bad['comparisonSeal'] = semantic_digest({k: v for k, v in bad.items() if k != 'comparisonSeal'})
        with pytest.raises(ValueError): validate_saved_comparison(scene, record, bad, declaration=declaration)
