"""Declared vascular exposures affect decisions; no motor-risk calibration."""
import copy
from types import SimpleNamespace

import pytest

from resectionlab import surgical_consequences as consequences
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_joint_cranial_search import _compare_checked_nodes, _dominates, _retain_beam
from resectionlab.surgical_sequence_outcomes import sequence_outcome
from test_surgical_procedure_policy import source
from test_surgical_vascular_graph import engine as graph_fixture


@pytest.mark.parametrize('metric', ['supplyDeficitSeconds', 'drainageDeficitSeconds'])
def test_equal_treatment_and_zero_bleeding_cannot_hide_declared_deficit(metric, monkeypatch):
    # Controlled typed decision vectors, separate from physical model evidence.
    baseline = SurgicalEngine(source())
    template = consequences.consequence_vector(baseline, baseline)
    nodes, vectors = [], {}
    for tag, target, deficit in [('stop', 0., 0.), ('deficit', 100., 20.), ('preserved', 100., 0.)]:
        node = SimpleNamespace(exited=SimpleNamespace(scene=baseline.scene, tag=tag),
            goal_assessment={'originalTargetTreatmentMm3': target, 'endpointAssessment': {
                'terminated': True, 'truncated': False, 'validWithdrawal': True,
                'classification': 'valid-stop-with-residual'}})
        vector = copy.deepcopy(template)
        for name in ('bloodGeneratedMm3', 'supplyDeficitSeconds', 'drainageDeficitSeconds'):
            value = deficit if name == metric else 0.
            vector['dimensions'][name].update(status='modeled_unvalidated', cumulative=value, incremental=value, reason=None)
        vectors[tag] = vector
        nodes.append(node)
    monkeypatch.setattr(consequences, 'consequence_vector', lambda e, _: copy.deepcopy(vectors[e.tag]))
    selected, frontier, decision = _compare_checked_nodes(nodes[1:], nodes[0], SimpleNamespace(_source_engine=baseline))
    assert selected is nodes[2] and list(frontier) == [nodes[2]]
    assert metric in decision['policy']['objectives']
    assert metric in decision['policy']['requiredMetrics']


def test_beam_retains_lower_vascular_exposure_and_does_not_treat_unknown_as_zero():
    def node(tag, deficit):
        return SimpleNamespace(goal_assessment={'originalTargetTreatmentMm3': 1.},
            ledger=dict(boneRemovedMm3=0., preparationNonboneRemovedMm3=0.,
                workNonTargetRemovedMm3=0., modeledElapsedSeconds=1.),
            sequence_outcome={'dimensions': {'supplyDeficitSeconds': {
                'status': 'unavailable' if deficit is None else 'modeled_unvalidated', 'cumulative': deficit}}},
            path=({'proposalId': tag, 'preparationOnly': False},))
    damaging, preserved, unknown = node('first', 20.), node('second', 0.), node('unknown', None)
    assert _dominates(preserved, damaging)
    assert not _dominates(unknown, damaging) and not _dominates(preserved, unknown)
    assert _retain_beam([damaging, preserved], 2) == [preserved]


def test_actual_zero_bleed_occlusion_ledgers_match_graph_and_cold_replay():
    engine = SurgicalEngine(graph_fixture().scene)
    baseline = engine.clone()
    def act(operation, point=None, **extra):
        action = {'slot': 'left', 'operation': operation, 'duration': .1, **extra}
        if point is not None:
            action['pose'] = {'position': point, 'quaternion': [0., 0., 0., 1.]}
        result = engine.step(action)
        assert result['status'] == 'accepted', result
    act('configure', mechanicsRevision=2)
    act('exchange', instrument='bipolar_forceps')
    act('place', [2.5, 1., -4.], parameters={'jawOpeningMm': .5, 'forceN': 0., 'power': 0., 'depthMm': 1.})
    act('apply', [2.5, 1., 1.8], duration=.5)
    act('apply', parameters={'power': 1.}, duration=2.1)
    act('withdraw')
    result = sequence_outcome(engine, baseline=baseline)
    dims = result['dimensions']
    assert dims['bloodGeneratedMm3']['cumulative'] == 0.
    for metric in ('supplyDeficitSeconds', 'drainageDeficitSeconds'):
        expected = max(r[metric] for r in engine.metrics()['vesselPhysicsV2']['regions'].values())
        assert dims[metric]['cumulative'] == expected
        assert dims[metric]['incremental'] == expected - dims[metric]['baseline']
        assert dims[metric]['status'] == 'modeled_unvalidated'
    assert dims['supplyDeficitSeconds']['cumulative'] > 0
    assert dims['motorEncounterMm3']['cumulative'] is None
    restored = SurgicalEngine.restore(engine.scene, engine.snapshot())
    assert sequence_outcome(restored, baseline=baseline) == result


def test_graph_without_declared_interfaces_keeps_exposure_unknown():
    from test_surgical_vascular_graph import profile
    declaration = profile()
    declaration['regions'] = []
    result = sequence_outcome(SurgicalEngine(graph_fixture(declaration).scene))
    for metric in ('supplyDeficitSeconds', 'drainageDeficitSeconds'):
        assert result['dimensions'][metric]['status'] == 'unavailable'
        assert result['dimensions'][metric]['cumulative'] is None
        assert metric in result['missingEvidence']


def test_generated_graph_reward_charges_both_shared_exposures_after_checked_withdrawal():
    from resectionlab.surgical_rl_rewards import GeneratedGraphRewardConfig, RewardTracker, compute_reward
    from test_surgical_mechanism_comparison import vascular_control
    baseline, branches = vascular_control()
    config = GeneratedGraphRewardConfig(supply_deficit_weight=2., drainage_deficit_weight=3.)
    tracker = RewardTracker(baseline, graph_evaluation=True)
    before = tracker.measure(baseline)
    for branch in ('arterial-occlusion', 'venous-occlusion', 'arterial-cut'):
        engine = branches[branch]
        after = tracker.clone().measure(engine)
        sequence = sequence_outcome(engine, baseline=baseline)
        assert sequence['withdrawalComplete']
        assert after['bleeding_generated_mm3'] == sequence['dimensions']['bloodGeneratedMm3']['cumulative']
        _, terms = compute_reward(before, after, {'status': 'accepted'}, config)
        for term, name, weight in [('supply_deficit', 'supplyDeficitSeconds', 2.),
                                   ('drainage_deficit', 'drainageDeficitSeconds', 3.)]:
            dim = sequence['dimensions'][name]
            assert dim['cumulative'] == after[name]
            assert terms[term] == pytest.approx(-weight * dim['incremental'])
            assert terms[term] < 0
        with pytest.raises(ValueError, match='graph reward weights required'):
            compute_reward(before, after, {'status': 'accepted'})


@pytest.mark.parametrize('missing', ['regions', 'generated-coverage', 'generated-source'])
def test_graph_reward_holds_unqualified_domains(missing):
    from resectionlab.surgical_rl_rewards import RewardTracker
    from resectionlab.surgical_scene import SurgicalScene
    from test_surgical_vascular_graph import profile
    declaration = profile()
    if missing == 'regions':
        declaration['regions'] = []
    if missing == 'generated-coverage':
        declaration['coverage']['status'] = 'partial-acquired'
    source = graph_fixture(declaration).scene
    if missing == 'generated-source':
        source = SurgicalScene(source.labels, source.affine, source.regions,
            {'kind': 'generated-control', 'patientId': 'test'}, source.properties)
    with pytest.raises(ValueError, match='VASCULAR_GRAPH_RL_HOLD'):
        RewardTracker(SurgicalEngine(source), graph_evaluation=True)


def test_explicit_generated_graph_rl_env_measures_and_replays_refusals():
    from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
    from resectionlab.surgical_rl_rewards import GeneratedGraphRewardConfig
    from resectionlab.surgical_scene import SurgicalScene
    fixture = graph_fixture().scene
    labels = fixture.labels.copy()
    labels[7, 4, 4] = 3
    scene = SurgicalScene(labels, fixture.affine,
        [*fixture.regions, {'id': 3, 'name': 'generated target', 'material': 'tumor', 'target': True}],
        fixture.source, fixture.properties)
    env = SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False),
                        reward_config=GeneratedGraphRewardConfig())
    _, initial = env.reset(seed=1)
    assert env.engine.state['toolMechanicsRevision'] == 2
    assert len(env.engine.actions) == 1
    _, _, _, _, rejected = env.step_action({'slot': 'left', 'operation': 'configure', 'mechanicsRevision': 1})
    assert rejected['outcome']['reasonCode'] == 'GENERATED_GRAPH_MECHANICS_FIXED_AT_RESET'
    assert rejected['stateId'] == initial['stateId']
    assert rejected['reward_terms']['drainage_deficit'] == rejected['reward_terms']['supply_deficit'] == 0.
    def act(operation, point=None, **extra):
        action = {'slot': 'left', 'operation': operation, 'duration': .1, **extra}
        if point is not None:
            action['pose'] = {'position': point, 'quaternion': [0., 0., 0., 1.]}
        _, _, _, _, info = env.step_action(action)
        assert info['outcome']['status'] == 'accepted', info['outcome']
        return info
    act('exchange', instrument='bipolar_forceps')
    act('place', [2.5, 1., -4.], parameters={'jawOpeningMm': .5, 'forceN': 0., 'power': 0., 'depthMm': 1.})
    act('apply', [2.5, 1., 1.8], duration=.5)
    act('apply', parameters={'power': 1.}, duration=2.1)
    withdrawn = act('withdraw')
    assert withdrawn['metrics']['bleeding_generated_mm3'] == 0.
    assert withdrawn['reward_terms']['supply_deficit'] < 0.
    assert withdrawn['reward_terms']['drainage_deficit'] < 0.
    record = env.snapshot()
    env.reset(seed=2)
    _, restored = env.restore(record)
    assert restored['stateId'] == record['stateId']
    assert restored['metrics'] == record['measurements']
