"""Accepted engine controls carried through typed paired consequence comparison."""

import copy
import json
import os
from pathlib import Path

import pytest

from resectionlab.surgical_consequences import compare_samples, consequence_vector
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_procedure_policy import PROPERTY
from resectionlab.surgical_retractor_runtime import PROPERTY as RETRACTOR
from resectionlab.surgical_scene import SurgicalScene
from test_surgical_procedure_policy import action as procedure_action, revised, source as procedure_source
from test_surgical_retractor_runtime import H, accepted, action as retractor_action, prepare, source as slab_source
from test_surgical_vascular_graph import engine as graph_fixture


STOP = 'STOP'
SUPPLY = 'vascular:declared-region:supplyDeficitSeconds'
DRAINAGE = 'vascular:declared-region:drainageDeficitSeconds'
BLOOD = 'bloodGeneratedMm3'
CONTACT = 'functional:cortex:recordedContactCellSeconds'
PEAK = 'mechanical:1:peakStressNormPa'
FORCE_PEAK = 'mechanical:1:retractorForcePeakN'
FORCE_TIME = 'mechanical:1:retractorForceTimeNs'
STRESS_TIME = 'mechanical:1:stressSquaredVolumeTimePa2Mm3Seconds'


@pytest.fixture(scope='module')
def receipt():
    evidence = {}
    yield evidence
    output = os.environ.get('RESSECTIONLAB_MECHANISM_RECEIPT')
    if output and len(evidence) == 4:
        with Path(output).open('x') as stream:
            json.dump({'schema': 'resectionlab.generated-mechanism-comparison-receipt.v1',
                       'role': 'DEVELOPMENT', 'clinicalValidation': False, 'optimizerUpdates': 0,
                       'coldReplayVerified': True, 'controls': evidence}, stream, indent=2, allow_nan=False)
            stream.write('\n')


def receipt_rows(vectors, *metrics):
    return {name: {'sourceSceneId': vector['sourceSceneId'], 'sourceGeometryHash': vector['sourceGeometryHash'],
                   'baselineStateId': vector['baselineStateId'], 'stateId': vector['stateId'],
                   'actionHash': vector['actionHash'], 'prefixActions': vector['prefixActions'],
                   'nativeActions': vector['nativeActions'], 'cleanupComplete': vector['cleanupComplete'],
                   'dimensions': {key: vector['dimensions'][key] for key in metrics}}
            for name, vector in vectors.items()}


def displayed(engine, baseline):
    cold = SurgicalEngine.restore(engine.scene, engine.snapshot(), use_cache=False, force_verify=True)
    result = consequence_vector(engine, baseline, extended=True)
    assert consequence_vector(cold, baseline, extended=True) == result
    return result


def ranked(vectors, objectives, *, required=(), constraints=()):
    policy = dict(id='generated-mechanism-control', basis='declared_research_task_not_clinical_threshold',
                  objectives=list(objectives), requiredMetrics=list(required), constraints=list(constraints))
    samples = [dict(branchId=branch, scenarioId='nominal', vector=vector) for branch, vector in vectors.items()]
    return compare_samples(samples, policy, stop_branch=STOP, scenario_ids=['nominal'], branch_ids=list(vectors))


def functional_control():
    def permit_cut(raw):
        raw['maximumNeuralCutSourceMm3'] = 250.
        raw['maximumStructureCutSourceMm3']['cortex'] = 250.

    scene = revised(procedure_source(neural_limit=250., overlap=True, cutting=True), permit_cut)
    baseline = SurgicalEngine(scene)
    accepted(baseline, procedure_action('inspect'))
    removed = baseline.clone()
    accepted(removed, procedure_action('place', [55., 40., 65.]))
    accepted(removed, procedure_action('apply', [55., 40., 100.], duration=1.))
    accepted(removed, procedure_action('withdraw', [55., 40., 65.], duration=1.))

    cut = baseline.clone()
    accepted(cut, procedure_action('exchange', instrument='scalpel'))
    accepted(cut, procedure_action('place', [40., 42.5, 65.]))
    partial = cut.step(procedure_action('apply', [40., 42.5, 90.], duration=1.))
    assert partial['status'] == 'partial' and partial['reasonCode'] == 'SHAFT_COLLISION'
    accepted(cut, procedure_action('deactivate'))
    accepted(cut, procedure_action('withdraw', [40., 42.5, 65.], duration=1.))

    held = baseline.clone()
    accepted(held, procedure_action('exchange', instrument='brain_retractor'))
    accepted(held, procedure_action('place', [40., 40., 65.], parameters={'radiusMm': 1., 'depthMm': 2.}))
    accepted(held, procedure_action('hold', [40., 40., 90.], duration=.5))
    accepted(held, procedure_action('move', [40., 40., 87.], duration=.5))
    accepted(held, procedure_action('release'))
    accepted(held, procedure_action('withdraw', [40., 40., 65.], duration=1.))
    return baseline, {'removed': removed, 'cut': cut, 'held': held, STOP: baseline}


def test_functional_source_removal_cut_and_contact_remain_separate_in_comparison(receipt):
    baseline, branches = functional_control()
    vectors = {name: displayed(engine, baseline) for name, engine in branches.items()}
    for name, expected in {'removed': (1, 0), 'cut': (0, 2), 'held': (0, 0)}.items():
        vector = vectors[name]
        assert vector['cleanupComplete']
        row = vector['mechanismEvidence']['functional']['structures']['cortex']
        assert row['source']['sourceBinding'] == baseline.scene.properties[PROPERTY]['sourceBinding']
        assert row['source']['coverageStatus'] == 'declared_intracranial_annotation_domain'
        for field, cells in zip(('removedMm3', 'cutSourceMm3'), expected):
            assert row[field]['incremental'] == cells * baseline.scene.voxel_mm3
            assert vector['dimensions'][f'structure:cortex:{field}']['incremental'] == row[field]['incremental']
        contact = vector['dimensions'][CONTACT]
        assert contact['unit'] == 'cell*s' and contact['baseline'] == 0.
        assert contact['incremental'] == row['recordedContactCellSeconds']['incremental']
        assert (contact['incremental'] > 0.) == (name == 'held')
        assert vector['dimensions']['motorEncounterMm3']['status'] == 'unavailable'

    result = ranked(vectors, ['targetRemovedMm3'], constraints=[
        dict(metric=CONTACT, scope='incremental', operator='at_most', value=0.)])
    assert result['selectedBranchId'] == 'removed'
    assert result['violations']['held'][0]['metric'] == CONTACT
    unknown = 'functional:language-cortex:recordedContactCellSeconds'
    missing = ranked(vectors, [CONTACT], required=[unknown])
    assert missing['stopFallback'] and not missing['feasibleContinuations']
    assert missing['violations']['held'] == [dict(scenario='nominal', reason='unknown_required_consequence', metric=unknown)]
    receipt['functional'] = dict(branches=receipt_rows(vectors, 'targetRemovedMm3', 'structure:cortex:removedMm3',
        'structure:cortex:cutSourceMm3', CONTACT, unknown), comparison=result, requiredUnknownComparison=missing)


def vascular_control():
    baseline = SurgicalEngine(graph_fixture().scene)
    accepted(baseline, retractor_action('configure', mechanicsRevision=2))

    branches = {STOP: baseline}
    for name, y in (('arterial-occlusion', 1.), ('venous-occlusion', 3.), ('arterial-cut', 1.)):
        end = baseline.clone()
        occlusion = name.endswith('occlusion')
        if occlusion:
            parameters = {'jawOpeningMm': .5, 'forceN': 0., 'power': 0., 'depthMm': 1.}
            instrument = 'bipolar_forceps'
        else:
            parameters = {'jawOpeningMm': 0., 'depthMm': 1., 'forceN': 0.}
            instrument = 'micro_scissors'
        accepted(end, retractor_action('exchange', duration=.1, instrument=instrument))
        for command in (retractor_action('place', [2.5, y, -4.], duration=.1, parameters=parameters),
                        retractor_action('apply', [2.5, y, 1.8], duration=.5)):
            if not occlusion:
                command['pose']['quaternion'] = [0., 0., 2**-.5, 2**-.5]
            accepted(end, command)
        if occlusion:
            accepted(end, retractor_action('apply', parameters={'power': 1.}, duration=2.1))
        for operation, duration in (('deactivate', .1), ('withdraw', .1), ('inspect', 1.)):
            accepted(end, retractor_action(operation, duration=duration))
        branches[name] = end
    return baseline, branches


def test_live_vascular_outcomes_reach_distinct_comparator_dimensions(receipt):
    baseline, branches = vascular_control()
    vectors = {name: displayed(engine, baseline) for name, engine in branches.items()}
    artery, vein, cut = [vectors[key]['dimensions'] for key in ('arterial-occlusion', 'venous-occlusion', 'arterial-cut')]
    assert artery[BLOOD]['unit'] == 'mm3' and artery[BLOOD]['aggregation'] == 'cumulative_volume'
    assert artery[BLOOD]['incremental'] == vein[BLOOD]['incremental'] == 0. < cut[BLOOD]['incremental']
    assert artery[SUPPLY]['incremental'] > vein[SUPPLY]['incremental'] > 0.
    assert vein[DRAINAGE]['incremental'] > artery[DRAINAGE]['incremental'] > 0.
    for metric, kind, edge in ((SUPPLY, 'artery', 'artery'), (DRAINAGE, 'vein', 'vein')):
        dimension = artery[metric]
        assert dimension['unit'] == 's' and dimension['aggregation'] == 'time_integral'
        assert dimension['source']['sceneId'] == baseline.scene.identity
        assert dimension['source']['coverage']['status'] == 'generated-control'
        assert dimension['source']['edgeTerms'] == [dict(edge=edge, direction=1, kind=kind)]
    segments = vectors['arterial-cut']['mechanismEvidence']['vascular']['segments']
    assert segments['artery']['continuity'] == 'severed' and len(segments['artery']['wounds']) == 2

    result = ranked(vectors, [SUPPLY, DRAINAGE], constraints=[
        dict(metric=BLOOD, scope='incremental', operator='at_most', value=0.)])
    assert result['pairedScenarioParetoFront'] == ['arterial-occlusion', 'venous-occlusion']
    assert result['stopFallback'] and result['violations']['arterial-cut'][0]['metric'] == BLOOD
    receipt['vascular'] = dict(branches=receipt_rows(vectors, BLOOD, SUPPLY, DRAINAGE), comparison=result)


def mechanical_control():
    baseline = prepare(track_exposure=True)
    pushed = baseline.clone()
    accepted(pushed, retractor_action('move', [7., 3.16, 1.9]))
    sustained = pushed.clone()
    accepted(sustained, retractor_action('inspect', duration=6 * H))
    branches = {}
    for name, engine in ((STOP, baseline), ('short', pushed), ('dwell', sustained)):
        end = engine.clone()
        accepted(end, retractor_action('move', [7., 2.9, 1.9]))
        accepted(end, retractor_action('release'))
        accepted(end, retractor_action('withdraw', duration=.1))
        branches[name] = end
    return baseline, pushed, sustained, branches


def test_mechanical_peak_and_integrated_loading_reach_the_comparator_at_exact_prefix(receipt):
    baseline, pushed, sustained, branches = mechanical_control()
    assert sustained.state[RETRACTOR]['last']['summedNormalForceN'] < pushed.state[RETRACTOR]['last']['summedNormalForceN']
    interval_vector = displayed(sustained, pushed)
    interval = interval_vector['dimensions']
    for metric, unit in ((PEAK, 'Pa'), (FORCE_PEAK, 'N')):
        row = interval[metric]
        assert row['unit'] == unit and row['aggregation'] == 'cumulative_peak'
        assert row['cumulative'] == row['baseline'] > 0. and row['incremental'] == 0.
        assert 'cumulative peak' in row['incrementalMeaning']
    for metric, unit in ((FORCE_TIME, 'N*s'), (STRESS_TIME, 'Pa2*mm3*s')):
        row = interval[metric]
        assert row['unit'] == unit and row['aggregation'] == 'time_integral'
        assert row['cumulative'] > row['baseline'] > 0. and row['incremental'] > 0.

    vectors = {name: displayed(engine, baseline) for name, engine in branches.items()}
    result = ranked(vectors, [PEAK, FORCE_TIME, STRESS_TIME])
    assert result['selectedBranchId'] == 'short'
    lattice_only = 'mechanical:1:peakNeighborDisplacementRatio'
    unknown = vectors['dwell']['dimensions'][lattice_only]
    assert unknown['status'] == 'unavailable' and unknown['cumulative'] is None
    assert ranked(vectors, [PEAK], required=[lattice_only])['stopFallback']

    forged = copy.deepcopy(vectors)
    forged['dwell']['dimensions'][PEAK]['aggregation'] = 'time_integral'
    with pytest.raises(ValueError, match='scope or identity'):
        ranked(forged, [PEAK])
    receipt['mechanical'] = dict(branches=receipt_rows(vectors, PEAK, FORCE_PEAK, FORCE_TIME, STRESS_TIME),
        loadedDwell=receipt_rows({'dwell': interval_vector}, PEAK, FORCE_PEAK, FORCE_TIME, STRESS_TIME)['dwell'],
        comparison=result)


def test_boundary_spanning_force_unknown_keeps_material_and_recorded_loading(receipt):
    scene = slab_source(track_exposure=True, two_regions=True)
    regions = copy.deepcopy(scene.regions)
    regions[1]['material'] = 'tumor'
    source = SurgicalScene(scene.labels, scene.affine, regions, scene.source, scene.properties)
    engine = SurgicalEngine(source)
    for operation, kwargs in (
        ('configure', {'mechanicsRevision': 2}), ('exchange', {'instrument': 'brain_retractor'}),
        ('place', {'position': [7., 2., 1.9], 'parameters': {'radiusMm': 1., 'depthMm': 2.}}),
        ('hold', {'position': [7., 3.1, 1.9], 'duration': 4 * H}),
    ):
        accepted(engine, retractor_action(operation, **kwargs))
    baseline = engine.clone()
    accepted(engine, retractor_action('move', [7., 3.16, 1.9]))
    vector = displayed(engine, baseline)
    values = vector['dimensions']
    for label, material in ((1, 'soft_tissue'), (2, 'tumor')):
        exposed = values[f'mechanical:{label}:retractorLoadedCellSeconds']
        force = values[f'mechanical:{label}:retractorForceTimeNs']
        assert exposed['source']['material'] == force['source']['material'] == material
        assert exposed['source']['labelId'] == label and exposed['incremental'] > 0.
        assert force['status'] == 'unavailable' and force['incremental'] is None
        assert 'spanning source labels' in force['reason']
    assert values['brainStrainSquaredMm3S']['source']['labelIds'] == [1]
    assert values['tumorStrainSquaredMm3S']['source']['labelIds'] == [2]
    receipt['mechanicalBoundary'] = receipt_rows({'boundary': vector}, 'brainStrainSquaredMm3S', 'tumorStrainSquaredMm3S',
        *(f'mechanical:{label}:{metric}' for label in (1, 2)
          for metric in ('retractorLoadedCellSeconds', 'retractorForceTimeNs')))['boundary']
