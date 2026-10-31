"""Executable, annotation-assisted two-tool research strategies.

The existing admission qualifies source bytes and the inactive baseline. This
NEW task separately authorizes a finite CUSA/Frazier suffix. Decisions use the
same immutable public baseline as approach planning, never the rich engine.
No device/patient calibration, optical visibility or neurological prediction.
"""
from __future__ import annotations

import copy
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from threading import RLock

import numpy as np
from scipy.spatial.transform import Rotation

from .core import array_digest, freeze_json, semantic_digest, thaw_json
from .surgical_engine import SurgicalEngine
from .surgical_planning_observation import PlanningObservation, project_observation, physical_state_hash
from .surgical_interactive_planning import (interactive_planning_availability, normalize_constraints,
    plan_observed_approaches, validate_public_approach_plan, _binding, _evidence, _retained_targets)
from .surgical_task_admission import admission, bind_admission, validate_admission
from .surgical_motion import envelopes, pair_conflict
from .surgical_tools import TOOLS, catalog
from .surgical_program_cache import ProgramPrefixCache
from .surgical_anatomical_encounter import (AnatomicalObservation, EncounterTracker, context_options,
    project_context, normalize_limit, axial_envelope)

TASK = 'post-exposure-two-tool-limited-debulking-v1'
SCHEMA = 'resectionlab.two-tool-comparison.v1'
PUBLIC_SCHEMA = 'resectionlab.public-two-tool-plan.v1'
CONTEXT_TASK = 'post-exposure-two-tool-source-context-v2'
CONTEXT_SCHEMA = 'resectionlab.two-tool-comparison.v2'
CONTEXT_PUBLIC_SCHEMA = 'resectionlab.public-two-tool-plan.v2'
SCOPE = ('Source-annotation-assisted CUSA/Frazier sequence with an assumed contact-capacity law. '
         'No new non-target removal, source exclusions stay active, complete withdrawal required. '
         'Missing anatomy, optical visibility, empirical device rates and clinical outcomes remain unassessed.')
MAX_WALL_SECONDS = 60.

# Only successful, complete canonical validations are retained. Keys contain
# content digests, never reports, observations, engines or caller-owned objects.
# Every lookup follows fresh native/source and live public-input validation.
_VALIDATION_CACHE_ENTRIES = 32
_VALIDATED_REPORTS = OrderedDict()
_VALIDATION_LOCK = RLock()
_VALIDATION_COUNTS = dict(hits=0, misses=0, evictions=0)
_LAST_PREFIX_STATS = {}


def prefix_cache_stats():
    """Last invocation diagnostics; never model/checkpoint handles."""
    with _VALIDATION_LOCK: return copy.deepcopy(_LAST_PREFIX_STATS)


class _CompilationScope:
    """One public scene owner and detached checkpoints, released on every exit."""
    def __init__(self, observation, controls, *, kind):
        _require(type(observation) is TwoToolObservation, 'typed public compilation scope required')
        self.binding = semantic_digest({'observation': observation.manifest(), 'controls': controls})
        self.cache = ProgramPrefixCache(); self.kind = kind; self.root = None; self.owner = None

    def _owner_signature(self):
        root = self.root; root.scene.require_cache_source()
        _require(root._occupancy_cache is None, 'public compilation owner changed: occupancy must stay cold')
        guard = root.protection
        grid = root.grid
        return semantic_digest({'scene': root.scene.identity, 'sceneOwner': id(root.scene),
            'state': root.state, 'actions': root.actions, 'history': root.history,
            'protectedIds': sorted(root._protected_ids), 'task': root._task, 'remaining': root._remaining_actions,
            'guardContract': root._protection_contract, 'catalog': catalog(),
            'byLabel': [[key, value] for key, value in root.by_label.items()], 'guardOwner': id(guard),
            'guardConfig': None if root.guard_config is None else asdict(root.guard_config),
            'grid': {'owner': id(grid), 'coverOwner': id(root._cell_cover_cache), 'shape': grid.shape,
                'affine': array_digest(grid.affine), 'inverse': array_digest(grid._inverse),
                'radius': grid._cell_radius_mm,
                'spacing': None if grid._orthogonal_spacing is None else grid._orthogonal_spacing.tolist()},
            'guard': None if guard is None else {'annotations': [asdict(a) for a in guard.annotations],
                'cells': array_digest(guard.cells), 'points': array_digest(guard._points.reshape(-1)),
                'pointsShape': guard._points.shape, 'radius': guard._radius,
                'treeOwner': id(guard._tree), 'treeData': array_digest(guard._tree.data.reshape(-1)),
                'treeShape': guard._tree.data.shape, 'treeIndices': array_digest(guard._tree.indices),
                'treeMin': array_digest(guard._tree.mins), 'treeMax': array_digest(guard._tree.maxes),
                'clearance': guard.clearance_mm, 'required': guard.required, 'missing': guard.missing,
                'config': asdict(root.guard_config)}})

    def model(self, observation, controls):
        _require(self.binding == semantic_digest({'observation': observation.manifest(), 'controls': controls}),
                 'public compilation observation or controls changed')
        _require(thaw_json(observation.baseline.fields)['instrumentCatalog'] == catalog(),
                 'instrument catalog changed during public compilation')
        if self.root is None:
            self.root = observation.planning_engine(); self.owner = self._owner_signature()
        self.require_owner()
        model = self.root.clone(); model._task = copy.deepcopy(self.root._task)
        model._protected_ids = self.root._protected_ids.copy()
        model._protection_contract = copy.deepcopy(self.root._protection_contract)
        return model

    def restore(self, payload):
        self.require_owner()
        # get() already detached the whole checkpoint graph. Start from a cold
        # public shell: only immutable scene/guard and pure cell covers share.
        model = copy.copy(self.root)
        model.state, model.actions, model.history = payload['state'], payload['actions'], payload['history']
        model._task = copy.deepcopy(self.root._task); model._protected_ids = self.root._protected_ids.copy()
        model._protection_contract = copy.deepcopy(self.root._protection_contract)
        return model

    def require_owner(self):
        _require(self.owner == self._owner_signature(), 'public compilation owner changed')

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, traceback):
        values = {**self.cache.stats(), 'publicRoots': int(self.root is not None),
            'sceneLabelBytes': self.root.scene.labels.nbytes if self.root is not None else 0,
            'completed': exc_type is None}
        self.cache.close(); self.root = None; self.owner = None
        values.update(retainedAfterClose=self.cache.stats()['entries'], closed=True)
        with _VALIDATION_LOCK: _LAST_PREFIX_STATS[self.kind] = values


def validation_cache_stats():
    """Process-local diagnostics; no validation authority is exported."""
    with _VALIDATION_LOCK:
        return {**_VALIDATION_COUNTS, 'entries': len(_VALIDATED_REPORTS),
            'maxEntries': _VALIDATION_CACHE_ENTRIES,
            'estimatedResidentBytes': sys.getsizeof(_VALIDATED_REPORTS) + sum(
                sys.getsizeof(key)+sys.getsizeof(value) for key, value in _VALIDATED_REPORTS.items()),
            'reportsRetained': False, 'enginesRetained': False}


def _require(value, message):
    if not value: raise ValueError('TWO_TOOL_TASK: '+message)


def task_spec(*, with_context=False):
    result = {'id': TASK, 'inputContract': 'annotation-assisted-two-tool-native-state-v1',
        'baselineQualification': 'existing source-byte-qualified inactive-access admission',
        'allowedOperations': ['exchange', 'inspect', 'place', 'move', 'apply', 'activate', 'deactivate', 'withdraw'],
        'allowedTools': ['cusa', 'frazier_suction'], 'maxActions': 24, 'maxDurationSeconds': 10.,
        'maxCusaActivationSeconds': 3., 'maxAdditionalTargetMm3': 8., 'maxAdditionalNonTargetMm3': 0.,
        'maxPairCandidatesPerRoute': 12, 'cusaRadiusMm': 1.5, 'suctionRadiusMm': 1.5,
        'clinicalOutcomes': 'unsupported'}
    if with_context:
        result.update(id=CONTEXT_TASK, inputContract='annotation-assisted-two-tool-source-context-v2',
            anatomicalEndpoint='unique retained supplied context cells in complete nominal axial-tool clearance envelope',
            maxContextMarginMm=5., maxContextNearMm3=1000.)
    return result


@dataclass(frozen=True)
class TwoToolObservation:
    baseline: PlanningObservation
    task: object
    context: AnatomicalObservation | None = None

    def __post_init__(self):
        _require(type(self.baseline) is PlanningObservation, 'typed public baseline required')
        frozen = freeze_json(self.task)
        _require(self.context is None or type(self.context) is AnatomicalObservation, 'typed public context required')
        _require(semantic_digest(frozen) == semantic_digest(task_spec(with_context=self.context is not None)), 'task capability/limit declaration differs')
        if self.context is not None:
            _require(self.context.mask.shape == self.baseline.labels.shape and np.array_equal(self.context.affine, self.baseline.affine),
                     'public context frame differs from baseline')
            targets = [r['id'] for r in thaw_json(self.baseline.fields)['regions'] if r.get('target') and r['material'] == 'tumor']
            _require(not np.any(self.context.mask & np.isin(self.baseline.labels, targets)), 'source context overlaps owned target')
        object.__setattr__(self, 'task', frozen)

    def manifest(self):
        baseline = self.baseline.manifest()
        body = {'schema': 'resectionlab.two-tool-observation.v1', 'baseline': baseline,
                'taskHash': semantic_digest(self.task)}
        if self.context is not None:
            body.update(schema='resectionlab.two-tool-observation.v2', anatomicalContext=self.context.manifest())
        return {**body, 'inputHash': semantic_digest(body), 'evaluationReferencesExposed': False}

    def planning_engine(self):
        # Copy only an already projected public model, never a rich native clone.
        public = self.baseline.planning_engine()
        engine = _TaskEngine(public.scene, protection=public.protection, guard_config=public.guard_config)
        engine.state = copy.deepcopy(public.state)
        engine._remaining_actions = public._remaining_actions
        engine._task = thaw_json(self.task)
        return engine


class _TaskEngine(SurgicalEngine):
    def _validate(self, action):
        task = self._task
        _require(len(self.actions) < min(self._remaining_actions, task['maxActions']), 'remaining action budget exhausted')
        _require(action.get('operation') in task['allowedOperations'], 'operation outside the finite two-tool task')
        slot = self.state['slots'].get(action.get('slot'), {})
        tool = action.get('instrument') if action.get('operation') == 'exchange' else slot.get('instrument')
        _require(tool in task['allowedTools'], 'instrument outside the finite two-tool task')
        _require(not set(action.get('parameters', {}))-{'radiusMm', 'depthMm', 'power', 'duty', 'flowMm3PerSecond'},
                 'unsupported task parameter')
        radius = action.get('parameters', {}).get('radiusMm', slot.get('parameters', {}).get('radiusMm', 1.5))
        if action.get('operation') == 'exchange': radius = action.get('parameters', {}).get('radiusMm', 1.5)
        _require(radius == 1.5, 'full catalog tool radius is fixed')
        duration, points = super()._validate(action)
        _require(sum(a.get('duration', 1.) for a in self.actions)+duration <= task['maxDurationSeconds']+1e-10,
                 'finite duration budget exhausted')
        return duration, points

    def fresh(self): raise ValueError('TWO_TOOL_TASK: observed anatomy and burdens cannot be reset')
    def reset(self, scene=None): return self.fresh()
    def replay(self, actions): return self.fresh()


def defaults(engine):
    available = interactive_planning_availability(engine)
    if not available['available'] or admission(engine) is None:
        return {'available': False, 'reason': available['reason'] or 'A qualified task baseline is required', 'defaults': None, 'scope': SCOPE}
    return {'available': True, 'reason': None, 'scope': SCOPE, 'task': task_spec(), 'contextOptions': context_options(engine),
            'defaults': {'geometry': available['defaults'], 'shortSeconds': .25, 'longSeconds': 1., 'minimumTargetMm3': 1.}}


def _normalize(observation, raw):
    keys = {'geometry', 'shortSeconds', 'longSeconds', 'minimumTargetMm3'} | ({'encounter'} if observation.context is not None else set())
    _require(type(raw) is dict and set(raw) == keys,
             'exact geometry, duration and progress controls required')
    result = {}
    for key, low, high in (('shortSeconds', .1, 2.), ('longSeconds', .1, 2.), ('minimumTargetMm3', 0., 8.)):
        value = raw[key]
        _require(type(value) in (int, float) and np.isfinite(value) and low <= value <= high, key+' outside task bounds')
        result[key] = float(value)
    _require(result['shortSeconds'] < result['longSeconds'], 'short duration must be less than long duration')
    fields = thaw_json(observation.baseline.fields)
    result['geometry'] = normalize_constraints(observation.baseline.planning_engine(), raw['geometry'], task=fields['task'])
    if observation.context is not None: result['encounter'] = normalize_limit(raw['encounter'])
    return result


def _tracker(observation, engine, controls):
    return EncounterTracker(observation.context, engine, controls['encounter']) if observation.context is not None else None


def _stop(observation, controls):
    baseline = observation.planning_engine(); before = copy.deepcopy(baseline.state)
    row = {'id': 'stop', 'label': 'STOP now', 'feasible': True, 'taskProgressMet': controls['minimumTargetMm3'] == 0.,
           'actionProgram': [], 'physicalStateHashes': [], 'phases': [], 'summary': _summary(baseline, before)}
    tracker = _tracker(observation, baseline, controls)
    if tracker: row.update(anatomicalEncounter=tracker.receipt(baseline), contextPrefixHashes=[])
    return row


def _summary(engine, before):
    removed = np.setdiff1d(engine.state['removed'], before['removed'])
    labels = engine.scene.labels.ravel(); target = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    n = int(np.count_nonzero(np.isin(labels[removed], list(target))))
    return {'additionalTargetMm3': n*engine.scene.voxel_mm3,
        'additionalNonTargetMm3': (len(removed)-n)*engine.scene.voxel_mm3,
        'removedSourceCells': removed.tolist(), 'residualTargetMm3': len(_retained_targets(engine))*engine.scene.voxel_mm3,
        'elapsedSeconds': engine.state['elapsedSeconds']-before['elapsedSeconds'],
        'cusaActivationSeconds': engine.state['cusaV3']['effectiveActivationSeconds']-before['cusaV3']['effectiveActivationSeconds'],
        'cusaInputCapacityMm3': engine.state['cusaV3']['inputCapacityMm3']-before['cusaV3']['inputCapacityMm3'],
        'cusaPendingMm3': sum(engine.state['cusaV3']['progressMm3'].values()),
        'fluidRemovedMm3': engine.state['fluidRemovedMm3']-before['fluidRemovedMm3']}


def _check_burdens(engine, before, controls):
    summary = _summary(engine, before); task = task_spec()
    _require(summary['additionalNonTargetMm3'] <= task['maxAdditionalNonTargetMm3'], 'new non-target removal exceeds task')
    _require(summary['additionalTargetMm3'] <= task['maxAdditionalTargetMm3'], 'target manipulation exceeds task')
    _require(summary['cusaActivationSeconds'] <= task['maxCusaActivationSeconds']+1e-10, 'CUSA activation exceeds task')
    new_work = [int(cell) for cell, value in engine.state['cusaV3']['progressMm3'].items()
                if value > before['cusaV3']['progressMm3'].get(cell, 0.)]
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    _require(all(int(engine.scene.labels.ravel()[cell]) in targets for cell in new_work), 'CUSA work reached non-target source tissue')
    cells = sorted(set(summary['removedSourceCells']) | set(new_work))
    target = controls['geometry']['target']
    _require(not cells or np.all(np.linalg.norm(engine.scene.points(cells)-target['centerMm'], axis=1) <= target['radiusMm']+1e-9),
             'new removal escapes requested target region')
    return summary


def _pair_poses(observation, route, entry_region=None):
    fields = thaw_json(observation.baseline.fields); task = fields['task']
    region = task['entry'] if entry_region is None else entry_region
    axis = np.asarray(task['axisRasUnit']); entry = np.asarray(region['centerMm'])
    q = Rotation.align_vectors([axis], [[0., 0., 1.]])[0]
    basis = q.as_matrix(); radius = region['radiusMm']
    length = float((np.asarray(route['targetPose']['position'])-entry)@axis)
    offsets = [basis[:, 1], -basis[:, 1], basis[:, 0], -basis[:, 0]]
    for fraction in (.66, .5, .8):
        for offset in offsets:
            radial = offset*radius*.57
            tip = entry+axis*max(0., length*fraction)+radial
            # Stagger outside parking depths so withdrawn working heads do not
            # collide when a narrow entry admits parallel shafts but not two
            # side-by-side 1.5-mm spherical heads in the same plane.
            outside = entry-axis*40.+radial
            yield {'tip': {'position': tip.tolist(), 'quaternion': q.as_quat().tolist()},
                   'outside': {'position': outside.tolist(), 'quaternion': q.as_quat().tolist()}}


def _pair_context_receipt(engine, secondary, context_tracker):
    slot = copy.deepcopy(engine.state['slots']['right'])
    slot.update(instrument='frazier_suction', parameters={'radiusMm': 1.5, 'depthMm': 2.})
    trial = copy.copy(context_tracker); trial.seen = {k: v.copy() for k, v in context_tracker.seen.items()}
    trial.add_cover(engine, axial_envelope(secondary['outside'], secondary['tip'], 'frazier_suction', engine.parameters(slot)))
    return trial.receipt(engine)


def _screen_pair(engine, secondary, clearance, context_tracker=None):
    from .surgical_joint_geometry import _axial_cover
    slot = copy.deepcopy(engine.state['slots']['right']); slot.update(instrument='frazier_suction', parameters={'radiusMm': 1.5, 'depthMm': 2.})
    params = engine.parameters(slot)
    cover = _axial_cover(engine, secondary['outside'], secondary['tip'], params, instrument='frazier_suction')
    for a, b, radius, _ in cover:
        if engine.touched(a, b, radius): return 'SECONDARY_RETAINED_TISSUE'
        if engine.protection is not None and len(engine.protection.capsule_hits(a, b, radius, positions=engine._positions())):
            return 'SECONDARY_PROTECTED_STRUCTURE'
        if clearance:
            targets = {r['id'] for r in engine.scene.regions if r.get('target')}
            if any(int(engine.scene.labels.ravel()[c]) not in targets for c in engine.touched(a, b, radius+clearance)):
                return 'SECONDARY_NON_TARGET_CLEARANCE'
    left = engine.state['slots']['left']
    if pair_conflict(cover, envelopes(TOOLS[left['instrument']], engine.parameters(left), left['position'], left['quaternion'])):
        return 'SECONDARY_TOOL_COLLISION'
    if context_tracker is not None:
        if not _pair_context_receipt(engine, secondary, context_tracker)['withinLimit']: return 'ANATOMICAL_CONTEXT_LIMIT:secondary-access'
    return None


def _check_action_clearance(engine, action, previous_slots, margin):
    if not margin: return
    from .surgical_joint_geometry import _axial_cover
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    for hand, slot in engine.state['slots'].items():
        params = engine.parameters(slot); before = previous_slots[hand]
        if hand == action['slot'] and action['operation'] in ('move', 'apply', 'withdraw'):
            cover = _axial_cover(engine, before, slot, params, instrument=slot['instrument'])
        else:
            cover = envelopes(TOOLS[slot['instrument']], params, slot['position'], slot['quaternion'])
        _require(not any(int(engine.scene.labels.ravel()[cell]) not in targets
                         for a, b, radius, _ in cover for cell in engine.touched(a, b, radius+margin)),
                 'mapped non-target clearance violated during the complete sequence')


def _compile(observation, controls, route, secondary, seconds, identity, *, probe_pose=None, check=lambda: None, prefix_scope=None):
    e = prefix_scope.model(observation, controls) if prefix_scope is not None else observation.planning_engine()
    before = copy.deepcopy(e.state); program = []; states = []; phases = []; receipts = []
    tracker = _tracker(observation, e, controls); context_hashes = []
    def step(action):
        nonlocal e
        check()
        if tracker: tracker.observation.assert_intact()
        prefix = [*program, action]
        found = prefix_scope.cache.get(prefix, check=check) if prefix_scope is not None else None
        if found is not None:
            e = prefix_scope.restore(found)
            if tracker:
                tracker.observation.assert_intact()
                tracker.seen = {key: set(value) for key, value in found['contextSeen'].items()}
            check()
            receipts.append(found['receipt']); program.append(copy.deepcopy(action)); states.append(found['physicalHash'])
            if tracker: context_hashes.append(found['contextHash'])
            return
        previous_slots = copy.deepcopy(e.state['slots'])
        result = e.step(copy.deepcopy(action), check=check)
        finite_contact = (action['slot'] == 'left' and action['operation'] == 'apply' and result['status'] == 'partial'
                          and result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS'
                          and abs(result['elapsedSeconds']-action['duration']) <= 1e-8)
        _require(result['status'] == 'accepted' or finite_contact, 'native public '+action['slot']+' '+action['operation']+' refused: '+str(result.get('reason')))
        receipt = {key: result.get(key) for key in ('status', 'reason', 'completedFraction', 'elapsedSeconds')}
        _check_burdens(e, before, controls)
        _check_action_clearance(e, action, previous_slots, controls['geometry']['nonTargetClearanceMm'])
        context_hash = None
        if tracker:
            tracker.observe(e, action, previous_slots); context_hash = semantic_digest(tracker.receipt(e))
            context_hashes.append(context_hash)
        physical_hash = physical_state_hash(e)
        if prefix_scope is not None:
            prefix_scope.require_owner()
            prefix_scope.cache.put(prefix, {'state': e.state, 'actions': e.actions, 'history': e.history,
                'receipt': receipt, 'physicalHash': physical_hash, 'contextHash': context_hash,
                'contextSeen': {key: sorted(value) for key, value in tracker.seen.items()} if tracker else None}, check=check)
        receipts.append(receipt); program.append(copy.deepcopy(action)); states.append(physical_hash)
    def phase(name, actions):
        start = len(program)
        for action in actions: step(action)
        phases.append({'id': name, 'startAction': start, 'endAction': len(program), 'physicalStateHash': physical_state_hash(e)})
        if tracker: phases[-1]['contextHash'] = semantic_digest(tracker.receipt(e))
    approach = copy.deepcopy(route['actionProgram'])
    setup = [approach.pop(0)] if approach and approach[0]['operation'] == 'exchange' else []
    phase('inspect', setup+[{'slot': 'left', 'operation': 'inspect', 'duration': .01}])
    phase('access', approach)
    _require(_screen_pair(e, secondary, controls['geometry']['nonTargetClearanceMm']) is None, 'paired working room refused')
    phase('position-secondary', [
        {'slot': 'right', 'operation': 'exchange', 'instrument': 'frazier_suction', 'duration': .1},
        {'slot': 'right', 'operation': 'place', 'pose': secondary['outside'], 'parameters': {'radiusMm': 1.5, 'depthMm': 2., 'flowMm3PerSecond': 20.}, 'duration': .1},
        {'slot': 'right', 'operation': 'move', 'pose': secondary['tip'], 'duration': .2},
        {'slot': 'right', 'operation': 'activate', 'duration': .01}])
    phase('operate', [{'slot': 'left', 'operation': 'apply',
        'pose': {'position': route['sourceTargetPointMm'], 'quaternion': route['targetPose']['quaternion']},
        'parameters': {'radiusMm': 1.5, 'power': 1., 'duty': 1.}, 'duration': seconds}])
    if probe_pose is None: probe_pose = {key: copy.deepcopy(e.state['slots']['left'][key]) for key in ('position', 'quaternion')}
    phase('reassess', [{'slot': 'left', 'operation': 'deactivate', 'duration': .01},
        {'slot': 'left', 'operation': 'withdraw', 'pose': route['targetPose'], 'duration': .1}])
    # A common inactive follow-up pose tests whether retained contact actually
    # cleared. Preview failure is a recorded STOP reason, not a committed action.
    probe = {'slot': 'left', 'operation': 'move', 'duration': .1, 'pose': copy.deepcopy(probe_pose)}
    trial = e.clone(); outcome = trial.step(copy.deepcopy(probe), check=check)
    reassessment = {'probeAction': probe, 'probeStatus': outcome['status'], 'probeReason': outcome.get('reason'),
        'summary': _summary(e, before), 'meaning': 'Inactive continuation into the same proposed source space after finite aspiration'}
    if outcome['status'] == 'accepted':
        step(probe); phases[-1].update(endAction=len(program), physicalStateHash=physical_state_hash(e))
        if tracker: phases[-1]['contextHash'] = semantic_digest(tracker.receipt(e))
    phase('withdraw', [
        {'slot': 'right', 'operation': 'deactivate', 'duration': .01},
        {'slot': 'right', 'operation': 'withdraw', 'pose': secondary['outside'], 'duration': .2},
        *route['withdrawalProgram']])
    _require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in e.state['slots'].values()), 'complete withdrawal required')
    summary = _check_burdens(e, before, controls)
    row = {'id': identity, 'label': 'Short contact' if identity == 'short' else 'Longer contact', 'contactSeconds': seconds,
        'routeId': route['id'], 'secondary': secondary, 'actionProgram': program, 'physicalStateHashes': states, 'actionReceipts': receipts,
        'phases': phases, 'summary': summary, 'reassessment': reassessment, 'feasible': True,
        'taskProgressMet': summary['additionalTargetMm3'] >= controls['minimumTargetMm3'],
        'access': 'native-complete-tool', 'workingRoom': 'native-sequential-persistent-pair', 'withdrawal': 'complete',
        'visibility': None, 'visibilityReason': 'NO_QUALIFIED_OPTICAL_VISIBILITY_MODEL', 'clinicalRisk': None}
    if tracker:
        receipt = tracker.receipt(e)
        row.update(anatomicalEncounter=receipt, contextPrefixHashes=context_hashes, feasible=receipt['withinLimit'])
    return row


def _approach_stop_reason(approaches):
    if approaches.get('missingRequiredEvidence'): return 'REQUIRED_EVIDENCE_UNAVAILABLE'
    if any(row.get('reason') == 'GEOMETRY_WALL_BUDGET' for row in approaches['rejections']): return 'COMPUTATION_BUDGET'
    return 'NO_COMPLETE_TOOL_ACCESS'


def _decision(alternatives, controls, reason=None):
    feasible = [row for row in alternatives if row['feasible'] and row['taskProgressMet']]
    selected = min(feasible, key=lambda r: (r['summary']['elapsedSeconds'], -r['summary']['additionalTargetMm3'], r['id'])) if feasible else None
    operate = selected is not None and selected['id'] != 'stop'
    if selected is None and any(row.get('anatomicalEncounter', {}).get('withinLimit') is False and row['taskProgressMet'] for row in alternatives):
        reason = reason or 'ANATOMICAL_CONTEXT_LIMIT'
    return {'operation': 'COMPARE' if operate else 'STOP', 'selectedId': selected['id'] if selected else 'stop',
        'reason': reason or ('MODELED_PROGRESS_THRESHOLD_MET' if operate else 'PROGRESS_ALREADY_SATISFIED'
                            if selected else 'NO_COMPLETE_STRATEGY_MEETS_PROGRESS_THRESHOLD'),
        'rule': 'Meet the declared research progress threshold with zero new non-target removal, '+
            ('respect the supplied context near-volume limit, ' if 'encounter' in controls else '')+
            'then minimize modeled duration; not a clinical recommendation'}


def plan_observed_strategies(observation, raw, *, check=lambda: None, progress=lambda *_: None):
    _require(type(observation) is TwoToolObservation, 'typed public two-tool observation required')
    controls = _normalize(observation, raw); started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started > MAX_WALL_SECONDS: raise TimeoutError('Two-tool comparison wall budget')
    approaches = plan_observed_approaches(observation.baseline, controls['geometry'], check=bounded, progress=progress)
    stop = _stop(observation, controls)
    alternatives = []; rejected = []; chosen = None; reason = None; context_failed = False; rejected_pair = None
    prefix_scope = _CompilationScope(observation, controls, kind='planning')
    try:
        for route in approaches['routes']:
            access = observation.planning_engine()
            access_context = _tracker(observation, access, controls)
            for action in route['actionProgram']:
                previous_slots = copy.deepcopy(access.state['slots'])
                bounded(); result = access.step(copy.deepcopy(action), check=bounded)
                _require(result['status'] == 'accepted', 'source approach no longer executes')
                if access_context: access_context.observe(access, action, previous_slots)
            if access_context and not access_context.receipt(access)['withinLimit']:
                context_failed = True
                rejected.append({'routeId': route['id'], 'reason': 'ANATOMICAL_CONTEXT_LIMIT:primary-access',
                    'anatomicalEncounter': access_context.receipt(access)})
                continue
            for index, secondary in enumerate(_pair_poses(observation, route, controls['geometry']['entry'])):
                bounded(); failure = _screen_pair(access, secondary, controls['geometry']['nonTargetClearanceMm'], access_context)
                if failure:
                    context_failed |= failure.startswith('ANATOMICAL_CONTEXT_LIMIT')
                    rejected.append({'routeId': route['id'], 'pairIndex': index, 'reason': failure,
                        **({'anatomicalEncounter': _pair_context_receipt(access, secondary, access_context)}
                           if failure.startswith('ANATOMICAL_CONTEXT_LIMIT') else {})}); continue
                variants = []
                for identity, key in (('short', 'shortSeconds'), ('long', 'longSeconds')):
                    try:
                        common_probe = variants[0]['reassessment']['probeAction']['pose'] if variants else None
                        variants.append(_compile(observation, controls, route, secondary, controls[key], identity,
                            probe_pose=common_probe, check=bounded, prefix_scope=prefix_scope))
                    except ValueError as error:
                        rejected.append({'routeId': route['id'], 'pairIndex': index, 'strategy': identity, 'reason': str(error)})
                if len(variants) == 2:
                    pair = {'route': route, 'pairIndex': index, 'secondary': secondary}
                    if observation.context is None or any(row['feasible'] and row['taskProgressMet'] for row in variants):
                        alternatives = variants; chosen = pair; break
                    rejected_pair = (variants, pair)
                    if any(not row['feasible'] for row in variants):
                        context_failed = True
                        rejected.append({'routeId': route['id'], 'pairIndex': index, 'reason': 'ANATOMICAL_CONTEXT_LIMIT:complete-sequences',
                            'anatomicalEncounters': [row['anatomicalEncounter'] for row in variants]})
                    else:
                        rejected.append({'routeId': route['id'], 'pairIndex': index, 'reason': 'NO_COMPLETE_STRATEGY_MEETS_PROGRESS_THRESHOLD'})
            if chosen is not None: break
    except TimeoutError:
        reason = 'COMPUTATION_BUDGET'
    finally:
        prefix_scope.__exit__(*sys.exc_info())
    if not approaches['routes']:
        reason = _approach_stop_reason(approaches)
    if chosen is None and reason is None:
        reason = ('COMPUTATION_BUDGET' if observation.context is not None and any(
            r.get('reason') == 'GEOMETRY_WALL_BUDGET' for r in approaches['rejections']) else
            'ANATOMICAL_CONTEXT_LIMIT' if context_failed else None if rejected_pair is not None else 'NO_COMPLETE_TWO_TOOL_SEQUENCE')
    if chosen is None and rejected_pair is not None:
        alternatives, chosen = rejected_pair
    alternatives.append(stop)
    report = {'schema': CONTEXT_PUBLIC_SCHEMA if observation.context is not None else PUBLIC_SCHEMA,
        'task': thaw_json(observation.task), 'publicObservation': observation.manifest(),
        'constraints': controls, 'approaches': approaches, 'chosenPair': chosen, 'alternatives': alternatives,
        'rejections': rejected, 'decision': _decision(alternatives, controls, reason), 'scope': SCOPE}
    report['publicPlanSeal'] = semantic_digest(report)
    return report


def generate_comparison(engine, raw, *, check=lambda: None, progress=lambda *_: None):
    value = admission(engine); _require(value is not None, 'source/baseline admission is required')
    validate_admission(engine, value)
    available = defaults(engine); _require(available['available'], available['reason'])
    context = binding = None; public_raw = copy.deepcopy(raw)
    if type(raw) is dict and 'encounter' in raw:
        selection = raw['encounter']
        _require(type(selection) is dict and set(selection) == {'layerId', 'marginMm', 'maxNearMm3'}
                 and type(selection['layerId']) is str, 'explicit source context selection and limits required')
        context, binding = project_context(engine, selection['layerId'])
        public_raw['encounter'].pop('layerId')
    observation = TwoToolObservation(project_observation(engine, value['task'], _evidence(engine)['availableEvidence']),
        task_spec(with_context=context is not None), context)
    public = plan_observed_strategies(observation, public_raw, check=check, progress=progress)
    report = {'schema': CONTEXT_SCHEMA if context is not None else SCHEMA, 'publicPlan': public, 'taskAdmission': value,
              **_binding(engine), 'sourceEvidence': _evidence(engine)}
    if binding is not None: report['contextBinding'] = binding
    report['reportSeal'] = semantic_digest(report)
    return report


def _validate_context_rejections(observation, controls, public, expected, *, prefix_scope=None, check=lambda: None):
    """Refusal counts are recomputed from public source cells, never self-seals."""
    routes = {row['id']: row for row in public['approaches']['routes']}; access_cache = {}
    _require(type(public['rejections']) is list and len(public['rejections']) <=
        37*controls['geometry']['candidateCount']+1, 'context rejection budget differs')
    if public['decision'].get('reason') == 'ANATOMICAL_CONTEXT_LIMIT':
        # This stronger refusal certifies the complete *declared finite family*,
        # not merely a genuine failure selected from an omitted search. Budget
        # expiry has a separate unresolved reason and cannot claim this proof.
        _require(not any(r.get('reason') == 'GEOMETRY_WALL_BUDGET' for r in public['approaches']['rejections']),
                 'anatomical STOP cannot hide an unfinished approach search')
        # Bind the parent shortlist too: truncating both its routes and their
        # context refusals must not turn a subset into a complete-family proof.
        canonical_approaches = plan_observed_approaches(observation.baseline, controls['geometry'], check=check)
        _require(semantic_digest(canonical_approaches) == semantic_digest(public['approaches']),
                 'anatomical STOP approach-family coverage differs')
        canonical = []; fallback = None
        for route in public['approaches']['routes']:
            check()
            access = observation.planning_engine(); tracker = _tracker(observation, access, controls)
            for action in route['actionProgram']:
                previous = copy.deepcopy(access.state['slots']); result = access.step(copy.deepcopy(action), check=check)
                _require(result['status'] == 'accepted', 'context coverage access differs')
                tracker.observe(access, action, previous)
            receipt = tracker.receipt(access)
            if not receipt['withinLimit']:
                canonical.append({'routeId': route['id'], 'reason': 'ANATOMICAL_CONTEXT_LIMIT:primary-access',
                    'anatomicalEncounter': receipt}); continue
            for index, secondary in enumerate(_pair_poses(observation, route, controls['geometry']['entry'])):
                check()
                failure = _screen_pair(access, secondary, controls['geometry']['nonTargetClearanceMm'], tracker)
                if failure:
                    canonical.append({'routeId': route['id'], 'pairIndex': index, 'reason': failure,
                        **({'anatomicalEncounter': _pair_context_receipt(access, secondary, tracker)}
                            if failure.startswith('ANATOMICAL_CONTEXT_LIMIT') else {})}); continue
                chosen = public['chosenPair']
                variants = copy.deepcopy(expected) if chosen and chosen['route']['id'] == route['id'] and chosen['pairIndex'] == index else []
                if not variants:
                    for identity, key in (('short', 'shortSeconds'), ('long', 'longSeconds')):
                        try:
                            probe = variants[0]['reassessment']['probeAction']['pose'] if variants else None
                            variants.append(_compile(observation, controls, route, secondary, controls[key], identity,
                                probe_pose=probe, prefix_scope=prefix_scope, check=check))
                        except ValueError as error:
                            canonical.append({'routeId': route['id'], 'pairIndex': index, 'strategy': identity, 'reason': str(error)})
                if len(variants) == 2:
                    _require(not any(row['feasible'] and row['taskProgressMet'] for row in variants),
                             'anatomical STOP omitted a complete admissible progress-qualified sequence')
                    fallback = {'route': route, 'pairIndex': index, 'secondary': secondary}
                    canonical.append({'routeId': route['id'], 'pairIndex': index,
                        **({'reason': 'ANATOMICAL_CONTEXT_LIMIT:complete-sequences',
                            'anatomicalEncounters': [row['anatomicalEncounter'] for row in variants]}
                           if any(not row['feasible'] for row in variants) else
                           {'reason': 'NO_COMPLETE_STRATEGY_MEETS_PROGRESS_THRESHOLD'})})
        _require(any(r['reason'].startswith('ANATOMICAL_CONTEXT_LIMIT') for r in canonical),
                 'anatomical STOP has no verified source-cell refusal')
        _require(semantic_digest(canonical) == semantic_digest(public['rejections']) and public['chosenPair'] == fallback,
                 'anatomical STOP candidate coverage, order or fallback differs')
        return
    verified = 0
    for rejection in public['rejections']:
        check()
        _require(type(rejection) is dict and type(rejection.get('reason')) is str, 'rejection schema differs')
        reason = rejection['reason']
        if not reason.startswith('ANATOMICAL_CONTEXT_LIMIT'): continue
        _require(reason in ('ANATOMICAL_CONTEXT_LIMIT:primary-access', 'ANATOMICAL_CONTEXT_LIMIT:secondary-access',
            'ANATOMICAL_CONTEXT_LIMIT:complete-sequences'), 'unknown context refusal')
        route = routes.get(rejection.get('routeId')); _require(route is not None, 'context refusal route differs')
        if route['id'] not in access_cache:
            access = observation.planning_engine(); tracker = _tracker(observation, access, controls)
            for action in route['actionProgram']:
                previous = copy.deepcopy(access.state['slots']); result = access.step(copy.deepcopy(action), check=check)
                _require(result['status'] == 'accepted', 'context refusal access replay differs')
                tracker.observe(access, action, previous)
            access_cache[route['id']] = access, tracker
        access, tracker = access_cache[route['id']]
        if reason.endswith('primary-access'):
            receipt = tracker.receipt(access)
            wanted = {'routeId': route['id'], 'reason': reason, 'anatomicalEncounter': receipt}
            _require(not receipt['withinLimit'], 'invented primary context limit')
        else:
            pairs = list(_pair_poses(observation, route, controls['geometry']['entry'])); index = rejection.get('pairIndex')
            _require(type(index) is int and 0 <= index < len(pairs), 'context refusal pair differs')
            if reason.endswith('secondary-access'):
                receipt = _pair_context_receipt(access, pairs[index], tracker)
                wanted = {'routeId': route['id'], 'pairIndex': index, 'reason': reason, 'anatomicalEncounter': receipt}
                _require(not receipt['withinLimit'], 'invented secondary context limit')
            else:
                chosen = public['chosenPair']
                rows = expected if chosen and chosen['route']['id'] == route['id'] and chosen['pairIndex'] == index else []
                if not rows:
                    for identity, key in (('short', 'shortSeconds'), ('long', 'longSeconds')):
                        probe = rows[0]['reassessment']['probeAction']['pose'] if rows else None
                        rows.append(_compile(observation, controls, route, pairs[index], controls[key], identity,
                            probe_pose=probe, prefix_scope=prefix_scope, check=check))
                receipts = [row['anatomicalEncounter'] for row in rows]
                wanted = {'routeId': route['id'], 'pairIndex': index, 'reason': reason, 'anatomicalEncounters': receipts}
                _require(all(not row['withinLimit'] for row in receipts), 'invented complete-sequence context limit')
        _require(semantic_digest(rejection) == semantic_digest(wanted), 'context refusal measurement differs')
        verified += 1
    if public['chosenPair'] is None and public['decision'].get('reason') == 'ANATOMICAL_CONTEXT_LIMIT':
        _require(verified > 0, 'context STOP has no verified source-cell refusal')


def validate_report(engine, report, *, check=lambda: None):
    """Rebuild live inputs; reuse only previously complete canonical validation.

    The source byte-verification fingerprint, full native binding and current
    public projection are checked even on hits. In particular, serialized guard
    contracts alone cannot hide a changed live protection mask/configuration.
    A cold report still executes the canonical approach and two-tool recipes.
    Native materialization is never cached here.
    """
    check()
    _require(type(report) is dict, 'comparison must be an object')
    _require(report.get('schema') in (SCHEMA, CONTEXT_SCHEMA) and report.get('reportSeal') == semantic_digest({k: v for k, v in report.items() if k != 'reportSeal'}), 'comparison seal differs')
    value = admission(engine)
    _require(value is not None and value == report.get('taskAdmission'), 'source baseline qualification differs')
    validate_admission(engine, value)
    binding = _binding(engine)
    with_context = report['schema'] == CONTEXT_SCHEMA
    _require(set(report) == {'schema', 'publicPlan', 'taskAdmission', 'sourceEvidence', 'reportSeal', *binding} |
             ({'contextBinding'} if with_context else set()), 'comparison schema differs')
    _require(all(report.get(k) == v for k, v in binding.items()) and report['sourceEvidence'] == _evidence(engine), 'source identity or planning checkpoint differs')
    context = None
    if with_context:
        _require(type(report['contextBinding']) is dict, 'source context binding must be an object')
        context, source_binding = project_context(engine, report['contextBinding'].get('layerId'))
        _require(semantic_digest(source_binding) == semantic_digest(report['contextBinding']), 'source context binding differs')
    observation = TwoToolObservation(project_observation(engine, value['task'], report['sourceEvidence']['availableEvidence']),
        task_spec(with_context=with_context), context)
    public = report['publicPlan']
    _require(type(public) is dict and set(public) == {'schema', 'task', 'publicObservation', 'constraints', 'approaches', 'chosenPair', 'alternatives', 'rejections', 'decision', 'scope', 'publicPlanSeal'}, 'public plan schema differs')
    _require(public['schema'] == (CONTEXT_PUBLIC_SCHEMA if with_context else PUBLIC_SCHEMA)
             and semantic_digest(public['task']) == semantic_digest(task_spec(with_context=with_context)) and public['scope'] == SCOPE,
             'two-tool task declaration differs')
    _require(public['publicPlanSeal'] == semantic_digest({k: v for k, v in public.items() if k != 'publicPlanSeal'}), 'public plan seal differs')
    manifest = observation.manifest()
    _require(public['publicObservation'] == manifest, 'public observation differs')
    cache_key = semantic_digest({'report': report, 'nativeBinding': binding, 'publicObservation': manifest})
    with _VALIDATION_LOCK:
        if cache_key in _VALIDATED_REPORTS:
            _VALIDATED_REPORTS.move_to_end(cache_key)
            _VALIDATION_COUNTS['hits'] += 1
            check(); return True
        _VALIDATION_COUNTS['misses'] += 1
    controls = _normalize(observation, public['constraints'])
    _require(semantic_digest(controls) == semantic_digest(public['constraints']), 'public controls differ')
    approaches = public['approaches']
    _require(approaches['publicPlanSeal'] == semantic_digest({k: v for k, v in approaches.items() if k != 'publicPlanSeal'})
             and approaches['publicObservation'] == observation.baseline.manifest()
             and approaches['constraints'] == controls['geometry'], 'public approach inputs differ')
    available = thaw_json(observation.baseline.fields)['availableEvidence']
    _require(approaches['evidence'] == {'availableEvidence': available}, 'public evidence differs')
    missing = sorted(set(controls['geometry']['requiredEvidence'])-set(available))
    if missing:
        _require(not approaches['routes'] and approaches.get('missingRequiredEvidence') == missing and approaches['status'] == 'required_evidence_unavailable', 'missing-evidence refusal differs')
    else:
        _require('missingRequiredEvidence' not in approaches, 'invented missing evidence')
    validate_public_approach_plan(observation.baseline, approaches, check=check)
    with _CompilationScope(observation, controls, kind='validation') as prefix_scope:
        expected = []
        chosen = public['chosenPair']
        if chosen is not None:
            _require(type(chosen) is dict and set(chosen) == {'route', 'pairIndex', 'secondary'}, 'paired geometry schema differs')
            matches = [r for r in approaches['routes'] if r['id'] == chosen['route']['id']]
            _require(len(matches) == 1 and matches[0] == chosen['route'], 'paired approach differs')
            pairs = list(_pair_poses(observation, chosen['route'], controls['geometry']['entry'])); index = chosen['pairIndex']
            _require(type(index) is int and 0 <= index < len(pairs) and pairs[index] == chosen['secondary'], 'paired placement is not a declared candidate')
            for identity, key in (('short', 'shortSeconds'), ('long', 'longSeconds')):
                probe = expected[0]['reassessment']['probeAction']['pose'] if expected else None
                expected.append(_compile(observation, controls, chosen['route'], chosen['secondary'], controls[key], identity,
                    probe_pose=probe, prefix_scope=prefix_scope, check=check))
            claimed = public['decision'].get('reason')
            reason = claimed if with_context and public['decision'].get('operation') == 'STOP' and claimed in ('ANATOMICAL_CONTEXT_LIMIT', 'COMPUTATION_BUDGET') else None
            if reason is not None: _require(reason in ('ANATOMICAL_CONTEXT_LIMIT', 'COMPUTATION_BUDGET'), 'context refusal reason differs')
        else:
            reason = _approach_stop_reason(approaches) if not approaches['routes'] else public['decision'].get('reason')
            _require(reason in {'REQUIRED_EVIDENCE_UNAVAILABLE', 'NO_COMPLETE_TOOL_ACCESS', 'NO_COMPLETE_TWO_TOOL_SEQUENCE', 'COMPUTATION_BUDGET'} |
                     ({'ANATOMICAL_CONTEXT_LIMIT'} if with_context else set()), 'STOP reason differs')
        if with_context: _validate_context_rejections(observation, controls, public, expected, prefix_scope=prefix_scope, check=check)
        expected.append(_stop(observation, controls))
        _require(semantic_digest(public['alternatives']) == semantic_digest(expected), 'canonical programs, physical predictions or reassessment differ')
        _require(semantic_digest(public['decision']) == semantic_digest(_decision(expected, controls, reason)), 'public strategy/STOP decision differs')
    check()
    with _VALIDATION_LOCK:
        _VALIDATED_REPORTS[cache_key] = True
        _VALIDATED_REPORTS.move_to_end(cache_key)
        while len(_VALIDATED_REPORTS) > _VALIDATION_CACHE_ENTRIES:
            _VALIDATED_REPORTS.popitem(last=False)
            _VALIDATION_COUNTS['evictions'] += 1
    return True


def materialize(engine, report, candidate_id, *, check=lambda: None):
    """Independent native execution after the complete public plan is sealed."""
    validate_report(engine, report, check=check)
    matches = [row for row in report['publicPlan']['alternatives'] if row['id'] == candidate_id]
    _require(len(matches) == 1, 'unknown strategy')
    strategy = matches[0]; native = engine.clone(); before = copy.deepcopy(engine.state); checkpoints = {}
    _require(strategy['feasible'], 'declared anatomical context limit prevents installation of this sequence')
    tracker = None
    if report['schema'] == CONTEXT_SCHEMA:
        context, _ = project_context(engine, report['contextBinding']['layerId'])
        tracker = EncounterTracker(context, native, report['publicPlan']['constraints']['encounter'])
    phases = {row['endAction']: row['id'] for row in strategy['phases']}
    reassess = next((row for row in strategy['phases'] if row['id'] == 'reassess'), None)
    for index, action in enumerate(strategy['actionProgram']):
        check(); previous_slots = copy.deepcopy(native.state['slots'])
        result = native.step(copy.deepcopy(action), check=check)
        receipt = {key: result.get(key) for key in ('status', 'reason', 'completedFraction', 'elapsedSeconds')}
        _require(receipt == strategy['actionReceipts'][index] and physical_state_hash(native) == strategy['physicalStateHashes'][index],
                 'native action differs from sealed public prediction; no branch installed')
        _check_burdens(native, before, report['publicPlan']['constraints'])
        _check_action_clearance(native, action, previous_slots, report['publicPlan']['constraints']['geometry']['nonTargetClearanceMm'])
        if tracker:
            tracker.observe(native, action, previous_slots)
            _require(semantic_digest(tracker.receipt(native)) == strategy['contextPrefixHashes'][index],
                     'native anatomical encounter differs from public prefix prediction')
        if reassess and index+1 == reassess['startAction']+2:
            trial = native.clone(); probe = trial.step(copy.deepcopy(strategy['reassessment']['probeAction']), check=check)
            _require(probe['status'] == strategy['reassessment']['probeStatus'] and probe.get('reason') == strategy['reassessment']['probeReason'],
                     'native reassessment differs from public prediction')
        if index+1 in phases: checkpoints[phases[index+1]] = native.clone()
    _require(_summary(native, before) == strategy['summary'], 'native consequence vector differs')
    if tracker:
        _require(tracker.receipt(native) == strategy['anatomicalEncounter'] and tracker.receipt(native)['withinLimit'],
                 'native complete sequence violates or differs from the declared anatomical context limit')
    _require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in native.state['slots'].values()), 'native withdrawal incomplete')
    receipt = {'candidateId': candidate_id, 'reportSeal': report['reportSeal'], 'publicPlanSeal': report['publicPlan']['publicPlanSeal'],
        'publicInputHash': report['publicPlan']['publicObservation']['inputHash'], 'taskId': report['publicPlan']['task']['id'],
        'startStateId': engine.state_id, 'finalStateId': native.state_id, 'actionCount': len(strategy['actionProgram']),
        'nativeMatchesPublicPrediction': True, 'withdrawalComplete': True, 'summary': strategy['summary'],
        'phases': [{'id': key, 'stateId': e.state_id, 'position': len(e.actions)} for key, e in checkpoints.items()]}
    if tracker: receipt.update(anatomicalEncounter=tracker.receipt(native), contextBindingSeal=report['contextBinding']['bindingSeal'])
    return native, checkpoints, receipt
