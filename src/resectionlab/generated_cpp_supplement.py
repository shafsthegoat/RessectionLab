"""Fixed generated simulation/RL episodes, never patient or policy training data.

Promoted from the existing public drilling, CUSA, vascular and RL controls. No
pytest imports, success-dependent retries, inferred tissue or new physics.
"""
import numpy as np

from .generated_cpp_fixture import continuum_fixture, request, pose
from .surgical_engine import SurgicalEngine
from .surgical_physics_v2 import default_profile
from .surgical_scene import SurgicalScene


SIMULATION_FAMILIES = ('simulation-target-episode', 'simulation-drill-thermal',
    'simulation-cusa-v3', 'simulation-vascular', 'simulation-vascular-wound', 'simulation-continuum')
RL_FAMILIES = ('rl-success-episode', 'rl-protected-attempt-episode', 'rl-independent-instances')


def target_scene():
    labels = np.zeros((24, 24, 20), np.uint16)
    labels[8:11, 8:11, 8:11] = 1
    return SurgicalScene(labels, np.eye(4), [{'id': 1,
        'name': 'Generated independent RL target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated independent software acceptance control'},
        {'tissuePhysicsV2': default_profile()})


def target_actions():
    def point(x, z):
        return pose(x, 9., z)
    return [request('place', pose={'position': [9., 9., 9.], 'quaternion': [0., 0., 0., 0.]}),
        request('exchange', instrument='cusa'),
        request('place', pose=point(8., -5.), parameters={'radiusMm': 1.5}),
        request('apply', pose=point(8., 11.), duration=1.), request('withdraw', duration=1.),
        request('place', pose=point(10., -5.)), request('apply', pose=point(10., 11.), duration=1.),
        request('withdraw', duration=1.)]


def drill_fixture():
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[4:7, 4:7, 8] = 1
    labels[5, 5, 6] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'bone', 'material': 'bone'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'explicit-thermal-parity-fixture'}, {'tissuePhysicsV2': default_profile()})
    def point(z):
        return {'position': [5., 5., z], 'quaternion': [1., 0., 0., 0.]}
    actions = [request('place', pose=point(15.), duration=.2,
            parameters={'radiusMm': 1., 'forceN': 0., 'rpm': 0.}),
        request('apply', pose=point(7.)),
        request('apply', pose=point(7.), duration=5., parameters={'forceN': 20., 'rpm': 80000.}),
        request('exchange', instrument='cusa', duration=.2),
        request('withdraw', pose=point(15.), duration=.7), request('inspect', slot='right', duration=2.5)]
    return SurgicalEngine(scene), actions


def cusa_fixture():
    from .surgical_cusa_v3 import default_profile as cusa_profile
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = 1
    profile = cusa_profile()
    profile['capacityMm3PerSecond'] = 2.
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Generated contact coupon',
        'material': 'tumor', 'target': True}],
        {'kind': 'generated CUSA contact coupon; no acquired anatomy'}, {'cusaV3': profile})
    actions = [request('exchange', instrument='cusa'), request('place', pose=pose(9., 9., 3.)),
        request('apply', pose=pose(9., 9., 11.), duration=.05)]
    actions += [request('apply', pose=pose(9., 9., 11.)) for _ in range(8)]
    actions += [request('deactivate'), request('withdraw', pose=pose(9., 9., 3.), duration=.3)]
    return SurgicalEngine(scene), actions


def vascular_fixture(*, wound=False):
    from .surgical_vascular_graph import PROPERTY, SCHEMA
    source = 'Generated numerical coefficient; not physiological calibration'
    shape = (8, 5, 5)
    def edge(name, start, end, kind, points):
        cells = [int(np.ravel_multi_index(point, shape)) for point in points]
        return {'id': name, 'from': start, 'to': end, 'kind': kind,
            'conductanceMm3PerPaS': .0001, 'sourceCells': cells, 'sourceFace': cells[:],
            'woundConductanceMm3PerPaS': [.0001, .0001], 'source': source}
    profile = {'schema': SCHEMA,
        'provenance': {'kind': 'assumed-research', 'description': source},
        'coverage': {'sourceCaseId': 'generated-rc-control', 'status': 'generated-control',
            'missing': ['No acquired patient or neurological outcomes']}, 'fieldPressurePa': 0.,
        'compressionForceN': 1., 'coagulationDose': 2., 'clotRatePerSecond': 0.,
        'nodes': [{'id': name, 'complianceMm3PerPa': .001,
            'initialStorageMm3': volume, 'source': source}
            for name, volume in [('a', 7.5), ('b', 5.), ('c', 2.5)]],
        'edges': [edge('artery', 'a', 'b', 'artery', [(2, 1, 2), (3, 1, 2)]),
                  edge('vein', 'b', 'c', 'vein', [(2, 3, 2), (3, 3, 2)])],
        'boundaries': [{'id': name, 'node': node, 'pressurePa': pressure,
            'conductanceMm3PerPaS': .0001, 'source': source}
            for name, node, pressure in [('inlet', 'a', 10000.), ('outlet', 'c', 0.)]],
        'regions': [{'id': 'declared-region', 'supply': [{'edge': 'artery', 'direction': 1}],
            'drainage': [{'edge': 'vein', 'direction': 1}], 'storageNodes': ['b'],
            'referenceSupplyMm3PerSecond': .25, 'referenceDrainageMm3PerSecond': .25, 'source': source}]}
    labels = np.zeros(shape, np.uint16)
    for index, row in enumerate(profile['edges']):
        labels.ravel()[row['sourceCells']] = index + 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': index + 1, 'name': row['kind'], 'material': 'vessel'}
        for index, row in enumerate(profile['edges'])], {'kind': 'generated-control'},
        {'tissuePhysicsV2': default_profile(), PROPERTY: profile})
    actions = [request('configure', mechanicsRevision=2), request('exchange', instrument='bipolar_forceps'),
        request('place', pose=pose(2.5, 1., -4.), parameters={
            'jawOpeningMm': .5, 'forceN': 1., 'power': 0., 'depthMm': 1.}),
        request('apply', pose=pose(2.5, 1., 1.8), duration=.5), request('deactivate'),
        request('inspect', duration=.3), request('apply', parameters={'jawOpeningMm': 2.}),
        request('apply', parameters={'jawOpeningMm': .5, 'forceN': 0., 'power': 0.}),
        request('apply', parameters={'power': 1.}, duration=2.1), request('withdraw')]
    if wound:
        q = [0., 0., 2**-.5, 2**-.5]
        first = {'position': [2.5, 1., -4.], 'quaternion': q}
        contact = {'position': [2.5, 1., 1.8], 'quaternion': q}
        actions = [request('configure', mechanicsRevision=2), request('exchange', instrument='micro_scissors'),
            request('place', pose=first, parameters={'jawOpeningMm': 0., 'depthMm': 1., 'forceN': 0.}),
            request('apply', pose=contact, duration=.5), request('deactivate'), request('withdraw'),
            request('exchange', instrument='bipolar_forceps'),
            request('place', pose=pose(2.5, 1., -4.), parameters={
                'jawOpeningMm': .5, 'forceN': 1., 'power': 0., 'depthMm': 1.}),
            request('apply', pose=pose(2.5, 1., 1.8), duration=.5), request('deactivate'),
            request('inspect', duration=1.), request('apply', parameters={'jawOpeningMm': 2.}, duration=.2),
            request('withdraw')]
    return SurgicalEngine(scene), actions


def simulation_fixture(family):
    if family == 'simulation-target-episode':
        return SurgicalEngine(target_scene()), target_actions()
    if family == 'simulation-drill-thermal':
        return drill_fixture()
    if family == 'simulation-cusa-v3':
        return cusa_fixture()
    if family == 'simulation-vascular':
        return vascular_fixture()
    if family == 'simulation-vascular-wound':
        return vascular_fixture(wound=True)
    if family == 'simulation-continuum':
        engine, _ = continuum_fixture(13)
        return engine, [request('inspect', slot='right', duration=t) for t in (.1, .2, .3)]
    raise ValueError('Unknown generated simulation family: ' + family)


def rl_fixture(family, seed=318):
    from .surgical_rl_env import RLConfig, SurgicalRLEnv
    from .surgical_rl_protection import ProtectionAnnotation, ProtectionSet
    scene = target_scene()
    if family == 'rl-success-episode':
        return SurgicalRLEnv(scene, config=RLConfig(require_critical_coverage=False)), target_actions(), seed
    if family == 'rl-protected-attempt-episode':
        mask = np.zeros(scene.shape, bool)
        mask[9, 9, 8] = True
        annotation = ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
            provenance='Generated overlap-exclusion control; no acquired anatomy')
        protection = ProtectionSet(scene, [annotation], clearance_mm=1., strict=True)
        actions = [request('exchange', instrument='cusa'),
            request('place', pose=pose(9., 9., -5.), parameters={'radiusMm': .5}),
            request('apply', pose=pose(9., 9., 9.)), request('withdraw')]
        return SurgicalRLEnv(scene, protection=protection, config=RLConfig(max_steps=4)), actions, seed
    raise ValueError('Unknown generated RL family: ' + family)
