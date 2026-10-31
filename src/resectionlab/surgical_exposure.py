"""Bounded executable intracerebral preparation from declared public anatomy.

No skull/dura opening or clinical access is inferred from source background.
The CUSA uses fixed catalog geometry, finite capacity and the shared clock.
Every removed cell remains a preparation cost, never subsequent strategy gain.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_planning_observation import PlanningObservation, project_observation, physical_state_hash
from .surgical_task_admission import source_evidence, verify_source, task_spec

SCHEMA = 'resectionlab.source-exposure-preparation.v1'
PUBLIC_SCHEMA = 'resectionlab.public-source-exposure-preparation.v1'
SCOPE = ('Finite annotation-assisted intracerebral preparation through mapped non-target tissue, '
    'with zero additional target removal. Exterior/skull/dura and complete vascular/functional anatomy '
    'remain unassessed. Native masks and source annotations are unchanged; only executed cell removal '
    'and persistent physical burdens evolve. Engineering bounds and assumed aspiration rate are not clinical limits.')
DEFAULTS = {'maxNonTargetMm3': 100., 'maxActions': 64, 'maxSeconds': 90., 'maxCandidates': 3,
            'maxProposals': 256, 'maxPathMm': 25., 'maxWallSeconds': 30., 'lateralSweepMm': 0.}


def _require(value, reason):
    if not value: raise ValueError('SOURCE_EXPOSURE: '+reason)


def normalize(raw):
    _require(type(raw) is dict and set(raw) == set(DEFAULTS), 'exact bounded controls required')
    result = {}
    for key, high in [('maxNonTargetMm3', 200.), ('maxActions', 128), ('maxSeconds', 180.),
                      ('maxCandidates', 12), ('maxProposals', 1024), ('maxPathMm', 40.), ('maxWallSeconds', 90.), ('lateralSweepMm', 1.)]:
        value = raw[key]; integer = key in ('maxActions', 'maxCandidates', 'maxProposals')
        _require(type(value) is int if integer else type(value) in (int, float), 'numeric control differs: '+key)
        _require(np.isfinite(value) and (0 <= value if key == 'lateralSweepMm' else 0 < value) and value <= high,
                 'control outside software bounds: '+key)
        result[key] = value if integer else float(value)
    _require(result['maxActions'] >= 6 and result['maxSeconds'] >= 2. and result['maxPathMm'] > 3.5,
             'insufficient setup/withdrawal or preparation bounds')
    return result


class _ExposureEngine(SurgicalEngine):
    def _validate(self, action):
        _require(action.get('slot') == 'right' and action.get('operation') in ('place', 'apply', 'withdraw'),
                 'operation outside right-hand preparation task')
        slot = self.state['slots']['right']
        _require(slot['instrument'] == 'cusa' and 'instrument' not in action, 'fixed CUSA required')
        params = action.get('parameters', {})
        _require(not set(params)-{'radiusMm', 'power', 'duty'} and params.get('radiusMm', 1.5) == 1.5,
                 'fixed catalog preparation geometry required')
        _require(len(self.actions) < min(self._remaining_actions, self._limits['maxActions']), 'action budget exhausted')
        duration, points = super()._validate(action)
        _require(duration <= 1. and sum(a['duration'] for a in self.actions)+duration <= self._limits['maxSeconds']+1e-10,
                 'finite preparation clock budget exhausted')
        return duration, points

    def fresh(self): raise ValueError('SOURCE_EXPOSURE: cannot reset observed tissue or burdens')
    def reset(self, scene=None): return self.fresh()
    def replay(self, actions): return self.fresh()


@dataclass(frozen=True)
class ExposureObservation:
    baseline: PlanningObservation

    def __post_init__(self):
        _require(type(self.baseline) is PlanningObservation, 'typed public baseline required')

    def manifest(self):
        body = {'schema': 'resectionlab.source-exposure-observation.v1', 'baseline': self.baseline.manifest(),
            'capabilities': {'operations': ['place', 'apply', 'withdraw'], 'instrument': 'cusa',
                'radiusMm': 1.5, 'maxAdditionalTargetMm3': 0., 'sourceDomain': 'observed mapped labels and physical state'},
            'scope': SCOPE}
        return {**body, 'inputHash': semantic_digest(body), 'evaluationReferencesExposed': False}

    def planning_engine(self, controls):
        shell = self.baseline.planning_engine()
        model = _ExposureEngine(shell.scene, protection=shell.protection, guard_config=shell.guard_config)
        model.state = copy.deepcopy(shell.state); model._remaining_actions = shell._remaining_actions
        model._limits = copy.deepcopy(controls)
        return model


def project_exposure(engine):
    """Native qualification envelope only; no provenance enters the model.

    Reuse the existing immutable numerical baseline representation. Its proposal
    spheres do not assert a clear entry or a post-exposure admission. This new
    wrapper alone declares the finite active preparation capability.
    """
    ids = [r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor']
    points = engine.scene.points(np.flatnonzero(np.isin(engine.scene.labels, ids)))
    _require(len(points) > 0, 'nonempty mapped target required')
    center = points.mean(axis=0).tolist()
    proposal = task_spec(entry={'centerMm': center, 'radiusMm': 15.}, target={'centerMm': center, 'radiusMm': 15.},
                         axis=[0., 0., 1.], allow_unreviewed=True)
    return ExposureObservation(project_observation(engine, proposal, source_evidence(engine.scene)['availableEvidence']))


def _shortlist(observation, controls, check):
    model = observation.planning_engine(controls); scene = model.scene
    labels = scene.labels.copy(); labels.flat[model.state['removed']] = 0
    brain = [r['id'] for r in scene.regions if r['material'] == 'soft_tissue']
    targets = [r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor']
    free = labels == 0; seed = np.zeros(labels.shape, bool)
    for axis in range(3):
        for side in (0, -1):
            index = [slice(None)]*3; index[axis] = side; seed[tuple(index)] = free[tuple(index)]
    check(); exterior = ndi.binary_propagation(seed, mask=free)
    surface = np.argwhere(np.isin(labels, brain) & ndi.binary_dilation(exterior))
    target = np.isin(labels, targets); boundary = np.argwhere(target & ~ndi.binary_erosion(target))
    if not len(surface) or not len(boundary): return []
    world = lambda ijk: ijk@scene.affine[:3, :3].T+scene.affine[:3, 3]
    points, goals = world(surface), world(boundary)
    check(); distance, neighbor = cKDTree(points).query(goals, k=min(4, len(points)))
    distance, neighbor = np.asarray(distance).reshape(len(goals), -1), np.asarray(neighbor).reshape(len(goals), -1)
    # A declared preparation family requires some mapped-tissue depth before a
    # 0.25mm target stand-off. Closer/direct paths are outside this family.
    valid = (distance > 3.5) & (distance < controls['maxPathMm'])
    eligible = np.flatnonzero(valid.ravel())
    rank = eligible[np.lexsort((eligible, distance.ravel()[eligible]))][:controls['maxProposals']]
    rows = []
    for flat in rank:
        check(); i, j = np.unravel_index(flat, distance.shape); s, t = points[neighbor[i, j]], goals[i]
        length = float(distance[i, j]); direction = (t-s)/length; outside = s-direction*8.
        samples = np.linspace(s, t, max(2, int(np.ceil(length/.3))))
        ijk = np.rint((samples-scene.affine[:3, 3])@scene.inverse[:3, :3].T).astype(int)
        if not np.isin(labels[tuple(ijk.T)], brain+targets).all(): continue
        quaternion = Rotation.align_vectors([direction], [[0., 0., 1.]])[0].as_quat().tolist()
        basis = Rotation.from_quat(quaternion).as_matrix(); width = controls['lateralSweepMm']
        offsets = [np.zeros(3)] + ([basis[:, 0]*width, -basis[:, 0]*width, basis[:, 1]*width, -basis[:, 1]*width] if width else [])
        paths = []; union = set()
        for offset in offsets:
            shifted = outside+offset
            def touches_target(point):
                return any(int(labels.flat[c]) in targets for c in model.cells(shifted, point, 1.5))
            if touches_target(shifted) or not touches_target(t+offset): break
            low, high = 0., length+8.
            for _ in range(26):
                check(); middle = (low+high)/2.
                if touches_target(shifted+direction*middle): high = middle
                else: low = middle
            endpoint = shifted+direction*(low-.25)
            if np.dot(endpoint-(s+offset), direction) < .5: break
            cells = [int(c) for c in model.cells(shifted, endpoint, 1.5) if labels.flat[c]]
            if not cells or not np.isin(labels.flat[cells], brain).all(): break
            union.update(cells)
            if len(union)*scene.voxel_mm3 > controls['maxNonTargetMm3']: break
            paths.append({'outsideMm': shifted.tolist(), 'endpointMm': endpoint.tolist()})
        if len(paths) != len(offsets): continue
        row = {'surfaceMm': s.tolist(), 'sourceTargetMm': t.tolist(), 'outsideMm': outside.tolist(),
            'endpointMm': paths[0]['endpointMm'], 'axisRasUnit': direction.tolist(), 'quaternion': quaternion,
            'paths': paths, 'sourceTubeMm3': len(union)*scene.voxel_mm3, 'surfaceTargetDistanceMm': length}
        rows.append(row)
        if len(rows) == controls['maxCandidates']: break
    return rows


def _receipt(result):
    return {key: result.get(key) for key in ('status', 'reason', 'reasonCode', 'completedFraction', 'elapsedSeconds')}


def _summary(model, before):
    new = sorted(set(model.state['removed'])-set(before['removed']))
    targets = [r['id'] for r in model.scene.regions if r.get('target') and r['material'] == 'tumor']
    target = sum(int(model.scene.labels.flat[c]) in targets for c in new)*model.scene.voxel_mm3
    return {'additionalTargetMm3': target, 'additionalNonTargetMm3': len(new)*model.scene.voxel_mm3-target,
        'elapsedSeconds': model.state['elapsedSeconds']-before['elapsedSeconds'],
        'withdrawn': all(not s['active'] and not s['inserted'] and not s['grip'] for s in model.state['slots'].values())}


def _compile(model, row, controls, check):
    before = copy.deepcopy(model.state); park = {k: copy.deepcopy(before['slots']['right'][k]) for k in ('position', 'quaternion')}
    pose = lambda point: {'position': point, 'quaternion': row['quaternion']}
    params = {'radiusMm': 1.5, 'power': 1., 'duty': 1.}
    program = []; receipts = []; hashes = []; failure = None
    def step(operation, position, duration):
        nonlocal failure
        check(); action = {'slot': 'right', 'operation': operation, 'pose': position, 'parameters': params.copy(), 'duration': duration}
        result = model.step(copy.deepcopy(action), check=check)
        program.append(action); receipts.append(_receipt(result)); hashes.append(physical_state_hash(model))
        finite = result['status'] == 'partial' and result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS' and abs(result['elapsedSeconds']-duration) < 1e-8
        summary = _summary(model, before)
        if result['status'] != 'accepted' and not finite: failure = result.get('reason') or result['status']
        if summary['additionalTargetMm3'] > 0 or summary['additionalNonTargetMm3'] > controls['maxNonTargetMm3']+1e-9: failure = 'PREPARATION_BURDEN_LIMIT'
        return not failure
    reached = False
    for index, path in enumerate(row['paths']):
        reached = False
        if not step('place', pose(path['outsideMm']), .1): break
        # Every path reserves its withdrawal, subsequent path setup and final
        # remote parking. The source CUSA itself never changes radius.
        remaining = len(row['paths'])-index-1
        reserve_actions = 1+2*remaining+2; reserve_seconds = .5+.6*remaining+.2
        while len(program)+reserve_actions < controls['maxActions']:
            if sum(a['duration'] for a in program)+1.+reserve_seconds > controls['maxSeconds']: break
            if not step('apply', pose(path['endpointMm']), 1.): break
            if np.linalg.norm(np.asarray(model.state['slots']['right']['position'])-path['endpointMm']) < 1e-5:
                reached = True; break
        if not reached or failure: break
        if not step('withdraw', pose(path['outsideMm']), .5): break
    if reached and not failure:
        for operation in ('place', 'withdraw'):
            if not step(operation, park, .1): break
    if not reached and not failure: failure = 'PREPARATION_ACTION_OR_TIME_BUDGET'
    return {'route': row, 'actionProgram': program, 'actionReceipts': receipts, 'physicalStateHashes': hashes,
        'summary': _summary(model, before), 'failure': failure, 'complete': reached and failure is None and _summary(model, before)['withdrawn']}


def plan_observed(observation, raw, *, check=lambda: None):
    _require(type(observation) is ExposureObservation, 'typed public exposure observation required')
    observation = ExposureObservation(PlanningObservation(observation.baseline.labels, observation.baseline.affine, observation.baseline.fields))
    controls = normalize(raw); rows = _shortlist(observation, controls, check); attempts = []
    for row in rows:
        check(); attempt = _compile(observation.planning_engine(controls), row, controls, check); attempts.append(attempt)
        if attempt['complete']: break
    value = {'schema': PUBLIC_SCHEMA, 'publicObservation': observation.manifest(), 'controls': controls,
        'attempts': attempts, 'decision': {'operation': 'PREPARE' if attempts and attempts[-1]['complete'] else 'STOP',
            'selectedIndex': len(attempts)-1 if attempts and attempts[-1]['complete'] else None,
            'rule': 'First complete execution in deterministic source-distance order, under declared budgets; no optimality or clinical-access claim'}, 'scope': SCOPE}
    value['publicSeal'] = semantic_digest(value)
    return value


def prepare(engine, raw, *, native_collection, check=lambda: None):
    """Public choice seals first; every held attempt then executes natively."""
    controls = normalize(raw); before = engine.snapshot(); engine.scene.require_cache_source()
    verification = verify_source(engine.scene, native_collection=native_collection, check=check)
    restored = SurgicalEngine.restore(engine.scene, before, check=check, use_cache=False)
    _require(restored.snapshot() == before, 'source baseline does not cold-replay exactly')
    observation = project_exposure(engine); started = time.monotonic()
    def guard():
        check()
        _require(time.monotonic()-started <= controls['maxWallSeconds'], 'preparation computation budget exhausted; no partial study may publish')
    public = plan_observed(observation, controls, check=guard); selected = None
    _require(public['schema'] == PUBLIC_SCHEMA and public['scope'] == SCOPE
             and semantic_digest(public['publicObservation']) == semantic_digest(observation.manifest())
             and semantic_digest(public['controls']) == semantic_digest(controls),
             'public preparation inputs differ')
    _require(public['publicSeal'] == semantic_digest({k: v for k, v in public.items() if k != 'publicSeal'}),
             'public preparation seal differs')
    _require(public['decision']['selectedIndex'] is None or type(public['decision']['selectedIndex']) is int,
             'public preparation selection type differs')
    for index, attempt in enumerate(public['attempts']):
        model = engine.clone(); phases = []
        _require(type(attempt['complete']) is bool, 'public preparation completion type differs')
        _require(len(attempt['actionProgram']) <= controls['maxActions'] and
            sum(a['duration'] for a in attempt['actionProgram']) <= controls['maxSeconds']+1e-10,
            'public preparation exceeds native task budget')
        for action, receipt, expected in zip(attempt['actionProgram'], attempt['actionReceipts'], attempt['physicalStateHashes'], strict=True):
            _require(action.get('slot') == 'right' and action.get('operation') in ('place', 'apply', 'withdraw')
                and semantic_digest(action.get('parameters')) == semantic_digest({'radiusMm': 1.5, 'power': 1., 'duty': 1.})
                and type(action.get('duration')) in (int, float) and 0 < action['duration'] <= 1., 'public preparation action exceeds native capability')
            guard(); result = model.step(copy.deepcopy(action), check=guard)
            _require(semantic_digest(_receipt(result)) == semantic_digest(receipt) and physical_state_hash(model) == expected, 'native preparation prefix differs from public prediction')
            phases.append({'action': action, 'receipt': _receipt(result), 'slot': copy.deepcopy(model.state['slots']['right'])})
        _require(semantic_digest(_summary(model, engine.state)) == semantic_digest(attempt['summary']), 'native preparation burdens or withdrawal differ')
        summary = _summary(model, engine.state)
        _require(summary['additionalTargetMm3'] == 0 and 0 < summary['additionalNonTargetMm3'] <= controls['maxNonTargetMm3']+1e-9
                 or not attempt['complete'], 'complete preparation exceeds native burden bounds')
        _require(attempt['complete'] == _complete_program(phases, attempt['route'], engine.state['slots']['right'])
                 and (attempt['failure'] is None) == attempt['complete'], 'public preparation completeness differs from native program')
        if index == public['decision']['selectedIndex']: selected = model
    complete = next((i for i, a in enumerate(public['attempts']) if a['complete']), None)
    _require(public['decision']['selectedIndex'] == complete and public['decision']['operation'] == ('PREPARE' if complete is not None else 'STOP'),
             'public preparation choice differs from verified completion')
    guard(); _require(before == engine.snapshot(), 'native source checkpoint changed during preparation')
    _require(project_exposure(engine).manifest() == observation.manifest(), 'public source inputs changed during preparation')
    verify_source(engine.scene, native_collection=native_collection, check=guard)
    value = {'schema': SCHEMA, 'sourceSceneId': engine.scene.identity, 'sourceStateId': before['stateId'],
        'sourceSnapshotHash': semantic_digest(before), 'sourceVerification': verification,
        'publicPlan': public, 'nativeReplayVerified': True, 'scope': SCOPE}
    value['preparationSeal'] = semantic_digest(value)
    return selected, value


def _complete_program(phases, route, original_slot):
    """Native structural/endpoint check, independent of compiler success flags."""
    if not route['paths']: return False
    index = 0
    def accepted(row):
        receipt = row['receipt']
        return receipt['status'] == 'accepted' or (row['action']['operation'] == 'apply'
            and receipt['status'] == 'partial' and receipt['reasonCode'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
            and abs(receipt['elapsedSeconds']-row['action']['duration']) <= 1e-8)
    def matches(row, operation, point, quaternion, duration):
        return (accepted(row) and row['action']['operation'] == operation
            and row['action']['pose'] == {'position': point, 'quaternion': quaternion}
            and row['action']['duration'] == duration)
    for path in route['paths']:
        if index >= len(phases) or not matches(phases[index], 'place', path['outsideMm'], route['quaternion'], .1): return False
        index += 1; reached = False
        while index < len(phases) and phases[index]['action']['operation'] == 'apply':
            row = phases[index]
            if not matches(row, 'apply', path['endpointMm'], route['quaternion'], 1.): return False
            reached = np.linalg.norm(np.asarray(row['slot']['position'])-path['endpointMm']) < 1e-5
            index += 1
            if reached: break
        if not reached or index >= len(phases) or not matches(phases[index], 'withdraw', path['outsideMm'], route['quaternion'], .5): return False
        index += 1
    for operation in ('place', 'withdraw'):
        if index >= len(phases) or not matches(phases[index], operation, original_slot['position'], original_slot['quaternion'], .1): return False
        index += 1
    return index == len(phases) and not phases[-1]['slot']['inserted'] and not phases[-1]['slot']['active'] and not phases[-1]['slot']['grip']
