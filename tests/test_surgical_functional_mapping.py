"""functionalMappingV1: labelled stimulation estimates and structure-aware harm on synthetic grids."""
from dataclasses import asdict

import numpy as np
import pytest

from resectionlab import surgical_functional_mapping as fm
from resectionlab.surgical_effects import apply_effect, initialize_effect_state
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import ProtectionSet
from resectionlab.surgical_rl_rewards import RewardConfig, RewardTracker, StructureRewardConfig, compute_reward
from resectionlab.surgical_scene import SurgicalScene

SHAPE = (30, 20, 20)
REGIONS = [dict(id=1, name='Cerebral white matter', material='soft_tissue'),
           dict(id=2, name='CSTs (tract)', material='soft_tissue'),
           dict(id=3, name='Arcuates (tract)', material='soft_tissue'),
           dict(id=4, name='Tumour rim', material='tumor', target=True),
           dict(id=5, name='Artery', material='vessel'),
           dict(id=6, name='Vein', material='vessel')]


def labels(*, tracts=True, air_below=0):
    volume = np.ones(SHAPE, np.uint16)
    if tracts:
        volume[20:22] = 2          # corticospinal estimate, x = 20..21
        volume[0:2] = 3            # arcuate estimate, x = 0..1
    volume[8:14, 8:12, 8:12] = 4
    volume[15, 2:18, 3] = 5
    volume[15, 2:18, 16] = 6
    volume[:, :, :air_below] = 0
    return volume


def scene(mapping=True, *, tracts=True, awake=False, max_search_mm=20., extra=None, air_below=0):
    properties = {'functionalMappingV1': fm.declaration(awake=awake, max_search_mm=max_search_mm)} if mapping else {}
    properties.update(extra or {})
    regions = REGIONS if tracts else [r for r in REGIONS if r['id'] not in (2, 3)]
    return SurgicalScene(labels(tracts=tracts, air_below=air_below), np.eye(4), regions, {}, properties)


def cell(x, y, z):
    return int(np.ravel_multi_index((x, y, z), SHAPE))


def probe(engine, tip=(15, 10, 10), tool='subcortical_probe', **parameters):
    engine.state['slots']['left']['instrument'] = tool
    engine.state['slots']['left']['parameters'] = dict(parameters)
    return apply_effect(engine, 'left', 'apply', [cell(*tip)], np.asarray(tip, float), [0, 0, 0, 1], 1)[0]


def fresh(s):
    e = SurgicalEngine(s)
    initialize_effect_state(e)
    return e


@pytest.fixture
def reference_only(monkeypatch):
    """The compiled actor has no mapping model; run the reference transition (see the reported engine hook)."""
    import resectionlab.surgical_native_engine as native
    monkeypatch.setattr(native, 'prepare', lambda engine, action, check: NotImplemented)


def test_no_property_keeps_the_legacy_row_and_never_calls_the_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('mapping model called without its property')
    for name in ('stimulation_observation', 'structure_harm', 'declare', '_model'):
        monkeypatch.setattr(fm, name, forbidden)
    e = fresh(scene(False))
    row = probe(e, amplitudeMa=10.)
    assert row == {'kind': 'mapping', 'probe': 'subcortical', 'position': [15., 10., 10.], 'cells': [cell(15, 10, 10)],
                   'parameters': {'amplitudeMa': 10.}, 'coverage': 'unknown', 'evidence': [], 'result': 'unknown',
                   'uncertainty': 'no matching supplied functional evidence'}
    assert e.state['observations'] == [row]
    other = fresh(scene(False)); probe(other, amplitudeMa=10.)
    assert other.state_id == e.state_id
    assert 'functionalMappingWarning' not in ProtectionSet(e.scene).report()


def test_declaration_lists_tracts_families_law_and_unknown_ischaemia():
    declared = fm.declare(scene())
    assert [(t['id'], t['family'], t['provenance']) for t in declared['tracts']] == [
        ('CSTs', 'motor', 'estimated'), ('Arcuates', 'language', 'estimated')]
    assert all('ESTIMATE' in t['evidence'] for t in declared['tracts'])
    assert declared['familiesWithoutTracts'] == ['vision', 'other']
    assert declared['responseLaw']['provenance'] == 'owner-supplied'
    assert declared['responseLaw']['secondaryReference']['provenance'] == 'assumed-research'
    assert declared['ischaemicTerritory']['status'] == 'unknown' and declared['ischaemicTerritory']['value'] is None
    assert declared['patientOutcome']['deficitProbability'] is None
    assert set(declared['vessels']) == {'artery', 'vein'}
    assert [fm.family_for(n) for n in ('CST_left', 'IFOF', 'Frontal aslant', 'Optic radiations', 'Cingulum', 'Fornix')] == [
        'motor', 'language', 'language', 'vision', 'other', 'unclassified']
    with pytest.raises(ValueError, match='ISCHAEMIC'):
        fm.declare(scene(extra={'functionalMappingV1': {'ischaemicTerritory': 0}}))


def test_response_appears_at_the_predicted_current_and_disappears_below_it():
    threshold = (5. - 1.42) / .81          # tip 5 mm from the nearest CST cell centre
    e = fresh(scene())
    below, above = probe(e, amplitudeMa=threshold - .01), probe(e, amplitudeMa=threshold + .01)
    assert below['kind'] == above['kind'] == 'model-estimate'
    assert below['families']['motor']['estimatedDistanceMm'] == 5.
    assert below['result'] == 'model_estimate_no_response' and above['result'] == 'model_estimate_response'
    assert above['distanceBand'].startswith('tract within ~5.0 mm')
    assert 'does not establish absence' in below['distanceBand']
    assert above['provenance']['responseLaw'] == 'owner-supplied' and above['parametersAssumedAtReference'] == [
        'pulseWidthUs', 'frequencyHz']
    assert above['uncertaintyMm'] == pytest.approx(above['gridUncertaintyMm'] + abs(above['estimatedReachMm'] - above['secondaryReferenceReachMm']))
    assert above['gridUncertaintyMm'] == pytest.approx(np.sqrt(3) / 2)
    assert above['ischaemicTerritory'] == 'unknown' and above['deficitProbability'] is None
    assert e.state['observations'][-2:] == [below, above]


def test_higher_current_reaches_further():
    e = fresh(scene())
    near = probe(e, tip=(12, 10, 10), amplitudeMa=4.43)       # 8 mm from the CST
    far = probe(e, tip=(12, 10, 10), amplitudeMa=8.2)
    assert near['result'] == 'model_estimate_no_response' and far['result'] == 'model_estimate_response'
    assert far['estimatedReachMm'] > near['estimatedReachMm']


def test_pulse_width_and_frequency_scale_effective_current_monotonically_up_to_the_reference():
    low = 'low-frequency-bipolar'
    currents = [fm.effective_current(4., width, 60., low)[0] for width in (50., 100., 200., 500., 800., 1000.)]
    assert currents == sorted(currents) and currents[3] == 4.
    rates = [fm.effective_current(4., 500., f, low)[0] for f in (5., 20., 60., 120., 250., 500.)]
    assert rates == sorted(rates) and rates[2] == 4.
    # No extrapolation: a setting stronger than the declared reference never reaches further.
    assert currents[3:] == [4.] * 3 and rates[2:] == [4.] * 4
    assert fm.effective_current(4., 1e-3, 1e-3)[0] >= 4. * .25 * .8 and fm.effective_current(4., 1000., 5000.)[0] == 4.
    # The default reference is the high-frequency monopolar train, not 60 Hz.
    assert fm.effective_current(4., 500., 400.)[0] == 4. and fm.effective_current(4., 500., 60.)[0] < 4.
    e = fresh(scene())
    results = [probe(e, amplitudeMa=4.5, pulseWidthUs=w)['result'] for w in (200., 500., 1000.)]
    assert results == ['model_estimate_no_response', 'model_estimate_response', 'model_estimate_response']
    results = [probe(e, amplitudeMa=4.5, frequencyHz=f)['result'] for f in (20., 60., 400., 1000.)]
    assert results == ['model_estimate_no_response', 'model_estimate_no_response',
                       'model_estimate_response', 'model_estimate_response']
    assert probe(e, amplitudeMa=0.)['result'] == 'unknown'
    assert probe(e)['families']['motor']['reason'].startswith('No stimulation current')


def test_stimulation_reference_is_a_declared_preset_stated_in_each_row():
    assert set(fm.STIMULATION_PRESETS) == {'high-frequency-monopolar-train', 'low-frequency-bipolar'}
    assert all(p['provenance'] == 'assumed-research' for p in fm.STIMULATION_PRESETS.values())
    assert 50. <= fm.STIMULATION_PRESETS['low-frequency-bipolar']['referenceFrequencyHz'] <= 60.
    assert fm.STIMULATION_PRESETS['high-frequency-monopolar-train']['referenceFrequencyHz'] >= 250.
    row = probe(fresh(scene()), amplitudeMa=4.)
    reference = row['stimulationReference']
    assert reference['preset'] == 'high-frequency-monopolar-train' and reference['declared']
    assert reference['referenceFrequencyHz'] == 400. and reference['provenance'] == 'assumed-research'
    bipolar = fm.declaration(stimulation_reference='low-frequency-bipolar')
    row = probe(fresh(scene(extra={'functionalMappingV1': bipolar})), amplitudeMa=4.)
    assert row['stimulationReference']['preset'] == 'low-frequency-bipolar'
    assert 'extrapolates the technique' in row['stimulationReference']['statement']
    legacy = {k: v for k, v in fm.declaration().items() if k != 'stimulationReference'}
    row = probe(fresh(scene(extra={'functionalMappingV1': legacy})), amplitudeMa=4.)
    assert row['stimulationReference']['preset'] == fm.DEFAULT_STIMULATION_REFERENCE
    assert not row['stimulationReference']['declared'] and row['stimulationReference']['statement'].startswith('Assumed')
    assert fm.declare(scene())['stimulationReference']['preset'] == fm.DEFAULT_STIMULATION_REFERENCE
    with pytest.raises(ValueError, match='STIMULATION_REFERENCE'):
        fm.declare(scene(extra={'functionalMappingV1': {**legacy, 'stimulationReference': 'sixty-hertz'}}))


def test_removed_tract_cells_are_ignored_for_distance():
    e = fresh(scene())
    e.state['removed'] = [cell(20, y, z) for y in range(SHAPE[1]) for z in range(SHAPE[2])]
    row = probe(e, amplitudeMa=4.43)
    assert row['families']['motor']['estimatedDistanceMm'] == 6.
    assert row['result'] == 'model_estimate_no_response'


def test_no_tract_evidence_or_outside_coverage_stays_unknown():
    row = probe(fresh(scene(tracts=False)), amplitudeMa=20.)
    assert row['result'] == 'unknown' and row['coverage'] == 'unknown'
    assert row['families']['motor']['reason'] == 'The scene declares no tract evidence for this function family.'
    row = probe(fresh(scene(max_search_mm=3.)), amplitudeMa=20.)
    assert row['result'] == 'unknown' and 'outside declared tract coverage' in row['families']['motor']['reason']
    assert row['tracts'][0]['estimatedDistanceMm'] is None


def test_language_needs_an_awake_scenario_and_cortical_no_response_is_unknown():
    tip = (5, 10, 10)                      # 4 mm from the arcuate, 15 mm from the CST
    asleep = probe(fresh(scene()), tip=tip, amplitudeMa=4.)
    assert asleep['families']['language']['result'] == 'unknown' and 'awake' in asleep['families']['language']['reason']
    assert asleep['families']['motor']['result'] == 'model_estimate_no_response'
    assert 'language' in asleep['unknownFamilies']
    awake = probe(fresh(scene(awake=True)), tip=tip, amplitudeMa=4.)
    assert awake['families']['language']['result'] == 'model_estimate_response'
    assert 'lawExtrapolated' in awake['families']['language']
    cortical = probe(fresh(scene()), tool='cortical_probe', amplitudeMa=4.)
    assert cortical['probe'] == 'cortical' and cortical['result'] == 'unknown'
    assert probe(fresh(scene()), tool='cortical_probe', amplitudeMa=20.)['result'] == 'model_estimate_response'


def test_supplied_evidence_still_wins():
    evidence = [{'id': 'site', 'cell': cell(15, 10, 10), 'probe': 'subcortical', 'result': 'test-only'}]
    mapped = probe(fresh(scene(extra={'functionalEvidence': evidence})), amplitudeMa=10.)
    plain = probe(fresh(scene(False, extra={'functionalEvidence': evidence})), amplitudeMa=10.)
    assert mapped == plain and mapped['result'] == 'supplied_evidence'


def cavity_engine():
    e = fresh(scene())
    e.state['removed'] = [cell(x, y, z) for x in range(16, 19) for y in range(9, 12) for z in range(9, 12)]
    return e


def brute(e, region):
    cavity = e.scene.points(e.state['removed'])
    cells = [c for c in np.flatnonzero(e.scene.labels.ravel() == region) if c not in set(e.state['removed'])]
    d = np.linalg.norm(e.scene.points(cells)[:, None] - cavity[None], axis=2).min(axis=1)
    return d.min(), int((d <= 2).sum()), int(((d > 2) & (d <= 5)).sum())


def test_structure_harm_matches_a_hand_built_cavity():
    e = cavity_engine()
    harm = fm.structure_harm(e)
    cst = harm['tracts']['CSTs']
    assert cst['removedMm3'] == 0. and cst['minDistanceMm'] == 2.
    assert (cst['minDistanceMm'], cst['band0To2Mm']['cells'], cst['band2To5Mm']['cells']) == brute(e, 2)
    assert cst['band0To2Mm']['cells'] == 9                 # the 3 x 3 face at x = 20
    artery = harm['vessels']['artery']
    assert artery['minDistanceMm'] == pytest.approx(np.sqrt(1 + 36))
    assert (artery['minDistanceMm'], artery['band0To2Mm']['cells'], artery['band2To5Mm']['cells']) == brute(e, 5)
    assert harm['cavity']['cells'] == 27 and harm['ischaemicTerritory']['status'] == 'unknown'
    assert harm['families']['vision']['status'] == 'unknown'
    e.state['removed'] += [cell(20, 10, 10), cell(20, 11, 10)]
    cst = fm.structure_harm(e)['tracts']['CSTs']
    assert cst['removedMm3'] == 2. and cst['minDistanceMm'] == 0.


def test_consequence_dimensions_only_with_the_property():
    from resectionlab.surgical_consequences import (
        _metric_spec, _vector_metrics, compare_samples, consequence_vector)
    e = cavity_engine()
    dims, estimate = fm.consequence_dimensions(e, e.fresh())
    assert dims['functional:CSTs:removedMm3']['cumulative'] == 0.
    distance = dims['functional:CSTs:minDistanceMm']
    assert (distance['cumulative'], distance['baseline'], distance['incremental']) == (2., 20., -18.)
    assert _metric_spec('functional:CSTs:minDistanceMm') == ('mm', 'max')
    assert estimate['motorTractMinDistanceMmModelEstimate'] == 2. and estimate['patientOutcome']['deficitProbability'] is None
    assert estimate['ischaemicTerritory']['value'] is None

    def run(s):
        engine = SurgicalEngine(s)
        for action in ({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa'},
                       {'slot': 'left', 'operation': 'place', 'pose': {'position': [15., 10., 1.], 'quaternion': [0., 0., 0., 1.]}},
                       {'slot': 'left', 'operation': 'apply', 'duration': 1.,
                        'pose': {'position': [15., 10., 5.], 'quaternion': [0., 0., 0., 1.]}}):
            engine.step(action)
        return engine
    plain = run(scene(False, air_below=3))
    vector = consequence_vector(plain, plain.fresh())
    assert 'functionalMappingMetricIds' not in vector and not any(k.endswith('minDistanceMm') for k in vector['dimensions'])
    mapped = run(scene(air_below=3))
    vector = consequence_vector(mapped, mapped.fresh())
    assert vector['functionalMappingMetricIds'] == sorted(dims)
    assert set(_vector_metrics(vector)) == set(vector['dimensions'])
    for name in ('motorEncounterMm3', 'languageEncounterMm3'):
        assert vector['dimensions'][name]['status'] == 'unavailable' and vector['dimensions'][name]['cumulative'] is None
    assert vector['functionalModelEstimate']['evidenceClass'] == 'model_estimate'
    assert mapped.state['removed'] and vector['dimensions']['functional:CSTs:minDistanceMm']['incremental'] < 0

    # A shrinking distance is a valid incremental value and larger is better.
    def sample(branch, distance):
        dimensions = {k: dict(v) for k, v in vector['dimensions'].items()}
        d = dimensions['functional:CSTs:minDistanceMm']
        d.update(cumulative=distance, incremental=distance - d['baseline'])
        return {'branchId': branch, 'scenarioId': 'nominal',
                'vector': {**vector, 'dimensions': dimensions, 'cleanupComplete': True}}
    policy = {'id': 'mapping', 'basis': 'declared_research_task_not_clinical_threshold',
              'objectives': ['functional:CSTs:minDistanceMm'], 'constraints': [], 'requiredMetrics': []}
    result = compare_samples([sample('near', 3.), sample('far', 9.), sample('STOP', 20.)], policy, stop_branch='STOP',
                             scenario_ids=['nominal'], branch_ids=['near', 'far', 'STOP'])
    assert result['dominatedBy']['near'] == ['far']


def tracker_engine():
    e = SurgicalEngine(scene())
    return e, RewardTracker(e)


def test_structure_reward_replaces_the_flat_healthy_weight():
    e, tracker = tracker_engine()
    before = tracker.measure(e)
    e.state['removed'] = [cell(18, 10, 10), cell(19, 10, 10), cell(20, 10, 10), cell(15, 5, 3), cell(10, 10, 10)]
    after = tracker.measure(e)
    assert after['healthy_removed_mm3'] == 4.               # flat ledger unchanged; one cell is tumour
    assert after['structure_tract_motor_removed_mm3'] == 1. and after['structure_vessel_artery_removed_mm3'] == 1.
    assert after['structure_other_healthy_removed_mm3'] == 2.
    assert after['structure_tract_motor_band_0_2mm_mm3'] > 0
    config = StructureRewardConfig()
    _, terms = compute_reward(before, after, {'status': 'accepted'}, config)
    assert terms['healthy_removal'] == pytest.approx(-2. * 2. / 1000.)
    assert terms['tract_removal:motor'] == pytest.approx(-10. / 1000.)
    assert terms['vessel_removal:artery'] == pytest.approx(-8. / 1000.)
    assert terms['tract_band_0_2mm:motor'] == pytest.approx(
        -10. * .25 * after['structure_tract_motor_band_0_2mm_mm3'] / 1000.)
    assert terms['tract_removal:language'] == 0.
    # The plain config keeps the flat weight and its legacy term names.
    _, flat = compute_reward(before, after, {'status': 'accepted'}, RewardConfig())
    assert flat['healthy_removal'] == pytest.approx(-2. * 4. / 1000.)
    assert not any(':' in name for name in flat)
    assert list(asdict(RewardConfig())) == [
        'tumor_removal_weight', 'healthy_removal_weight', 'bone_removal_weight', 'healthy_thermal_weight',
        'healthy_cut_weight', 'deformation_weight', 'vessel_damage_weight', 'bleeding_weight',
        'protected_damage_weight', 'near_contact_weight', 'action_cost', 'time_cost', 'invalid_action_cost',
        'protected_blocked_weight', 'volume_scale_mm3', 'area_scale_mm2', 'thermal_exposure_scale',
        'near_contact_scale_mm3_seconds', 'time_scale_seconds']
    plain = SurgicalEngine(scene(False)); plain_tracker = RewardTracker(plain)
    with pytest.raises(ValueError, match='STRUCTURE_REWARD_REQUIRES_FUNCTIONAL_MAPPING'):
        compute_reward(plain_tracker.measure(plain), plain_tracker.measure(plain), {'status': 'accepted'}, config)
    with pytest.raises(ValueError):
        StructureRewardConfig(tract_motor_weight=-1.)


def test_protection_report_warns_for_procedure_scale_tracts_without_mapping():
    procedure = {'hostRehearsalV1': {'schema': 'generated-test'}}
    warning = ProtectionSet(scene(False, extra=procedure)).report()['functionalMappingWarning']
    assert warning['code'] == 'TRACT_LAYERS_WITHOUT_FUNCTIONAL_MAPPING'
    assert warning['tractRegions'] == ['Arcuates (tract)', 'CSTs (tract)']
    assert 'functionalMappingWarning' not in ProtectionSet(scene(extra=procedure)).report()
    assert 'functionalMappingWarning' not in ProtectionSet(scene(False)).report()


def test_probe_replay_is_deterministic(reference_only):
    def run():
        engine = SurgicalEngine(scene(air_below=3))
        pose = lambda z: {'position': [15., 10., z], 'quaternion': [0., 0., 0., 1.]}
        for action in ({'slot': 'left', 'operation': 'exchange', 'instrument': 'subcortical_probe'},
                       {'slot': 'left', 'operation': 'place', 'pose': pose(1.), 'parameters': {'amplitudeMa': 8.}},
                       {'slot': 'left', 'operation': 'apply', 'duration': 1., 'pose': pose(4.)}):
            assert engine.step(action)['status'] in ('accepted', 'partial')
        return engine
    first, second = run(), run()
    rows = first.state['observations']
    assert rows and rows[-1]['kind'] == 'model-estimate' and rows == second.state['observations']
    assert first.state_id == second.state_id
    assert first.replay(first.actions).state_id == first.state_id
    restored = SurgicalEngine.restore(first.scene, first.snapshot())
    assert restored.state_id == first.state_id
