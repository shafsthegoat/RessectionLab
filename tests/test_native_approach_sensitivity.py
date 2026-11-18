"""Adversarial controls for declared registration and geometric sensitivity."""

from copy import deepcopy
from itertools import product
from math import hypot

import pytest

from resectionlab.native_approach_comparison import assess_sensitivity, freeze_scenarios


@pytest.mark.parametrize('root_count,registration_count', [(1, 14), (2, 16), (3, 16), (6, 16)])
def test_declared_translations_cover_signed_axes_with_coherent_sources(root_count, registration_count):
    groups = {f'root-{i}': [f'scan-{i}', f'derived-{i}'] for i in range(root_count)}
    config = {'translationMm': 1., 'boundaryMm': .5}
    design = freeze_scenarios(config, groups)
    assert design == freeze_scenarios(config, dict(reversed(list(groups.items()))))
    scenarios = design['scenarios']
    pure = [row for row in scenarios if row['id'].startswith('registration-')]
    combined = [row for row in scenarios if row['id'].startswith('combined-')]
    assert len(scenarios) == 2 + 2 * registration_count
    assert [row['sourceTranslationsMm'] for row in pure] == [row['sourceTranslationsMm'] for row in combined]
    signed_axes = {axis for axis in product((-1, 0, 1), repeat=3) if sum(map(abs, axis)) == 1}
    for scan, derived in groups.values():
        offsets = set()
        for row in pure:
            translation = row['sourceTranslationsMm']
            offset = translation.get(scan, [0, 0, 0])
            assert offset == translation.get(derived, [0, 0, 0])
            offsets.add(tuple(offset))
        assert signed_axes <= offsets
        if root_count == 1:
            signs = {tuple(1 if coordinate > 0 else -1 for coordinate in offset) for offset in offsets - signed_axes}
            assert signs == set(product((-1, 1), repeat=3))


def _field(whole=True):
    return {'wholeEnvelopeWithinField': whole}


def _geometry(length, reach, traversal, clearance):
    return {
        'lengthMm': length, 'coverage': {'withinReferenceEnvelope': True},
        'targets': [{'available': True, 'layerId': 'tumor', 'coverage': _field(),
                     'coveredVoxelCenterVolumeMm3': reach}],
        'normalTraversal': {'available': True, 'layerId': 'brain', 'coverage': _field(),
                            'conservativeVolumeUpperBoundMm3': traversal},
        'structures': [{'available': True, 'layerId': layer_id, 'kind': kind, 'coverage': _field(),
                        'confirmedIntersection': False, 'clearanceLowerBoundMm': clearance}
                       for layer_id, kind in [('vessel', 'vessels'), ('cst', 'tractography'), ('pons', 'brainstem')]],
        'constraintFailures': [], 'detectedConflictLayerIds': [], 'unknowns': [],
    }


def _sensitivity(nominal, perturbations):
    design = {'id': 'control', 'scenarios': [{'id': 'nominal'},
              *({'id': f'shift-{i}', 'index': i} for i in range(len(perturbations)))]}
    return assess_sensitivity(nominal, design, lambda scenario: perturbations[scenario['index']], check=lambda: None)


def test_third_registration_root_can_move_a_parallel_tract_into_the_capsule():
    design = freeze_scenarios({'translationMm': 1., 'boundaryMm': 0.},
                              {'a': ['first'], 'b': ['second'], 'c': ['third', 'tract-scan']})
    nominal = _geometry(20, 10, 15, .8)

    def measure(scenario):
        # The tract and capsule axis run parallel to X, with overlapping
        # longitudinal intervals and a nominal 1 mm separation in Z.
        _, dy, dz = scenario['sourceTranslationsMm'].get('tract-scan', (0, 0, 0))
        geometry = deepcopy(nominal)
        clearance = hypot(dy, 1 + dz) - .2
        geometry['structures'][1]['clearanceLowerBoundMm'] = clearance
        if clearance <= 0:
            geometry['structures'][1]['confirmedIntersection'] = True
            geometry['constraintFailures'] = ['cst: detected capsule contact']
            geometry['detectedConflictLayerIds'] = ['cst']
        return geometry

    result = assess_sensitivity(nominal, design, measure, check=lambda: None)
    metric = result['metrics']['tractClearanceMm']
    assert metric['nominal'] == .8 and metric['worst'] == -.2
    worst = next(row for row in result['scenarios'] if row['id'] == metric['worstScenarioId'])
    assert worst['status'] == 'constraint-violation' and worst['conflictLayerIds'] == ['cst']
    offset = next(row for row in design['scenarios'] if row['id'] == worst['id'])['sourceTranslationsMm']
    assert offset['tract-scan'] == offset['third'] == [0, 0, -1]


def test_adverse_tail_uses_fractional_mass_in_physical_units():
    nominal = _geometry(20, 1, 30, -5)
    perturbations = [_geometry(20, reach, traversal, clearance)
                     for reach, traversal, clearance in [(3, 14, -2), (4, 13, -1), (10, 12, 0),
                                                         (11, 11, 1), (12, 10, 2), (13, 9, 3), (20, 8, 4)]]
    metrics = _sensitivity(nominal, perturbations)['metrics']
    assert metrics['nonTargetTraversalMm3']['tailMean'] == pytest.approx((14 + .4 * 13) / 1.4)
    assert metrics['nonTargetTraversalMm3']['adverseDirection'] == 'higher'
    assert metrics['targetReachMm3']['tailMean'] == pytest.approx((3 + .4 * 4) / 1.4)
    assert metrics['targetReachMm3']['adverseDirection'] == 'lower'
    assert metrics['targetReachMm3']['units'] == 'mm³'
    assert metrics['vascularClearanceMm']['tailMean'] == pytest.approx((-2 + .4 * -1) / 1.4)
    assert metrics['vascularClearanceMm']['worst'] == -5
    assert metrics['vascularClearanceMm']['worstScenarioId'] == 'nominal'
    assert metrics['vascularClearanceMm']['units'] == 'mm'
    assert metrics['lengthMm']['tailMean'] is None


def test_missing_acquisition_field_suppresses_tail_without_hiding_detected_contact():
    nominal = _geometry(20, 10, 15, 4)
    partial = deepcopy(nominal)
    vessel = partial['structures'][0]
    vessel.update(acquisitionCoverage=_field(False), confirmedIntersection=True, clearanceLowerBoundMm=-.1)
    partial['detectedConflictLayerIds'] = ['vessel']
    partial['constraintFailures'] = ['vessel: intersects an excluded structure']
    result = _sensitivity(nominal, [partial] + [nominal] * 4)
    metric = result['metrics']['vascularClearanceMm']
    assert metric['worst'] == 4 and metric['assessedCount'] == 5 and metric['unknownCount'] == 1
    assert metric['tailMean'] is None and not metric['complete']
    scenario = result['scenarios'][1]
    assert scenario['status'] == 'constraint-violation' and scenario['coverageIncomplete']
    assert scenario['unknownSourceLayerIds'] == scenario['conflictLayerIds'] == ['vessel']
    assert scenario['sourceReadings']['vessel'][:3] == ['partial-field', -.1, 'detected']
    assert result['metrics']['targetReachMm3']['tailMean'] == 10


def test_missing_vessels_never_supply_a_clearance_or_complete_status():
    nominal = _geometry(20, 10, 15, 4)
    nominal['structures'] = nominal['structures'][1:]
    result = _sensitivity(nominal, [])
    assert result['scenarios'][0]['status'] == 'partial-coverage'
    assert result['scenarios'][0]['coverageIncomplete']
    assert result['metrics']['vascularClearanceMm']['worst'] is None
    assert result['metrics']['tractClearanceMm']['worst'] == 4


def test_declared_missing_anatomy_is_visible_even_when_every_objective_has_a_value():
    nominal = _geometry(20, 10, 15, 4)
    nominal['declaredMissingAnatomy'] = nominal['unknowns'] = ['Small perforating vessels unsegmented']
    result = _sensitivity(nominal, [])
    scenario, = result['scenarios']
    assert scenario['status'] == 'partial-coverage' and scenario['coverageIncomplete']
    assert scenario['missingMetrics'] == [] and scenario['unknownSourceLayerIds'] == []
    assert scenario['unknowns'] == ['Small perforating vessels unsegmented']


def test_other_critical_metric_does_not_sum_or_repeat_overlapping_geometry():
    nominal = _geometry(20, 10, 15, 4)
    nominal['structures'][0]['clearanceLowerBoundMm'] = 1
    nominal['structures'][1]['clearanceLowerBoundMm'] = 2
    nominal['structures'].append({**nominal['structures'][2], 'layerId': 'pons-overlap'})
    metrics = _sensitivity(nominal, [])['metrics']
    assert [metrics[key]['nominal'] for key in ('vascularClearanceMm', 'tractClearanceMm', 'otherCriticalClearanceMm')] == [1, 2, 4]
