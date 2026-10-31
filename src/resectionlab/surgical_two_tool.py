"""Executable, annotation-assisted two-tool research strategies.

The existing admission qualifies source bytes and the inactive baseline. This
NEW task separately authorizes a finite CUSA/Frazier suffix. Decisions use the
same immutable public baseline as approach planning, never the rich engine.
No device/patient calibration, optical visibility or neurological prediction.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .core import freeze_json, semantic_digest, thaw_json
from .surgical_engine import SurgicalEngine
from .surgical_planning_observation import PlanningObservation, project_observation, physical_state_hash
from .surgical_interactive_planning import (interactive_planning_availability, normalize_constraints,
    plan_observed_approaches, validate_public_approach_plan, _binding, _evidence, _retained_targets)
from .surgical_task_admission import admission, bind_admission, validate_admission
from .surgical_motion import envelopes, pair_conflict
from .surgical_tools import TOOLS

TASK = 'post-exposure-two-tool-limited-debulking-v1'
SCHEMA = 'resectionlab.two-tool-comparison.v1'
PUBLIC_SCHEMA = 'resectionlab.public-two-tool-plan.v1'
SCOPE = ('Source-annotation-assisted CUSA/Frazier sequence with an assumed contact-capacity law. '
         'No new non-target removal, source exclusions stay active, complete withdrawal required. '
         'Missing anatomy, optical visibility, empirical device rates and clinical outcomes remain unassessed.')
MAX_WALL_SECONDS = 60.


def _require(value, message):
    if not value: raise ValueError('TWO_TOOL_TASK: '+message)


def task_spec():
    return {'id': TASK, 'inputContract': 'annotation-assisted-two-tool-native-state-v1',
        'baselineQualification': 'existing source-byte-qualified inactive-access admission',
        'allowedOperations': ['exchange', 'inspect', 'place', 'move', 'apply', 'activate', 'deactivate', 'withdraw'],
        'allowedTools': ['cusa', 'frazier_suction'], 'maxActions': 24, 'maxDurationSeconds': 10.,
        'maxCusaActivationSeconds': 3., 'maxAdditionalTargetMm3': 8., 'maxAdditionalNonTargetMm3': 0.,
        'maxPairCandidatesPerRoute': 12, 'cusaRadiusMm': 1.5, 'suctionRadiusMm': 1.5,
        'clinicalOutcomes': 'unsupported'}


@dataclass(frozen=True)
class TwoToolObservation:
    baseline: PlanningObservation
    task: object

    def __post_init__(self):
        _require(type(self.baseline) is PlanningObservation, 'typed public baseline required')
        frozen = freeze_json(self.task)
        _require(semantic_digest(frozen) == semantic_digest(task_spec()), 'task capability/limit declaration differs')
        object.__setattr__(self, 'task', frozen)

    def manifest(self):
        baseline = self.baseline.manifest()
        body = {'schema': 'resectionlab.two-tool-observation.v1', 'baseline': baseline,
                'taskHash': semantic_digest(self.task)}
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
    return {'available': True, 'reason': None, 'scope': SCOPE, 'task': task_spec(),
            'defaults': {'geometry': available['defaults'], 'shortSeconds': .25, 'longSeconds': 1., 'minimumTargetMm3': 1.}}


def _normalize(observation, raw):
    _require(type(raw) is dict and set(raw) == {'geometry', 'shortSeconds', 'longSeconds', 'minimumTargetMm3'},
             'exact geometry, duration and progress controls required')
    result = {}
    for key, low, high in (('shortSeconds', .1, 2.), ('longSeconds', .1, 2.), ('minimumTargetMm3', 0., 8.)):
        value = raw[key]
        _require(type(value) in (int, float) and np.isfinite(value) and low <= value <= high, key+' outside task bounds')
        result[key] = float(value)
    _require(result['shortSeconds'] < result['longSeconds'], 'short duration must be less than long duration')
    fields = thaw_json(observation.baseline.fields)
    result['geometry'] = normalize_constraints(observation.baseline.planning_engine(), raw['geometry'], task=fields['task'])
    return result


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


def _screen_pair(engine, secondary, clearance):
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


def _compile(observation, controls, route, secondary, seconds, identity, *, probe_pose=None, check=lambda: None):
    e = observation.planning_engine(); before = copy.deepcopy(e.state); program = []; states = []; phases = []; receipts = []
    def step(action):
        check(); previous_slots = copy.deepcopy(e.state['slots'])
        result = e.step(copy.deepcopy(action), check=check)
        finite_contact = (action['slot'] == 'left' and action['operation'] == 'apply' and result['status'] == 'partial'
                          and result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS'
                          and abs(result['elapsedSeconds']-action['duration']) <= 1e-8)
        _require(result['status'] == 'accepted' or finite_contact, 'native public '+action['slot']+' '+action['operation']+' refused: '+str(result.get('reason')))
        receipts.append({key: result.get(key) for key in ('status', 'reason', 'completedFraction', 'elapsedSeconds')})
        _check_burdens(e, before, controls)
        _check_action_clearance(e, action, previous_slots, controls['geometry']['nonTargetClearanceMm'])
        program.append(copy.deepcopy(action)); states.append(physical_state_hash(e))
    def phase(name, actions):
        start = len(program)
        for action in actions: step(action)
        phases.append({'id': name, 'startAction': start, 'endAction': len(program), 'physicalStateHash': physical_state_hash(e)})
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
    phase('withdraw', [
        {'slot': 'right', 'operation': 'deactivate', 'duration': .01},
        {'slot': 'right', 'operation': 'withdraw', 'pose': secondary['outside'], 'duration': .2},
        *route['withdrawalProgram']])
    _require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in e.state['slots'].values()), 'complete withdrawal required')
    summary = _check_burdens(e, before, controls)
    return {'id': identity, 'label': 'Short contact' if identity == 'short' else 'Longer contact', 'contactSeconds': seconds,
        'routeId': route['id'], 'secondary': secondary, 'actionProgram': program, 'physicalStateHashes': states, 'actionReceipts': receipts,
        'phases': phases, 'summary': summary, 'reassessment': reassessment, 'feasible': True,
        'taskProgressMet': summary['additionalTargetMm3'] >= controls['minimumTargetMm3'],
        'access': 'native-complete-tool', 'workingRoom': 'native-sequential-persistent-pair', 'withdrawal': 'complete',
        'visibility': None, 'visibilityReason': 'NO_QUALIFIED_OPTICAL_VISIBILITY_MODEL', 'clinicalRisk': None}


def _approach_stop_reason(approaches):
    if approaches.get('missingRequiredEvidence'): return 'REQUIRED_EVIDENCE_UNAVAILABLE'
    if any(row.get('reason') == 'GEOMETRY_WALL_BUDGET' for row in approaches['rejections']): return 'COMPUTATION_BUDGET'
    return 'NO_COMPLETE_TOOL_ACCESS'


def _decision(alternatives, controls, reason=None):
    feasible = [row for row in alternatives if row['feasible'] and row['taskProgressMet']]
    selected = min(feasible, key=lambda r: (r['summary']['elapsedSeconds'], -r['summary']['additionalTargetMm3'], r['id'])) if feasible else None
    operate = selected is not None and selected['id'] != 'stop'
    return {'operation': 'COMPARE' if operate else 'STOP', 'selectedId': selected['id'] if selected else 'stop',
        'reason': reason or ('MODELED_PROGRESS_THRESHOLD_MET' if operate else 'PROGRESS_ALREADY_SATISFIED'
                            if selected else 'NO_COMPLETE_STRATEGY_MEETS_PROGRESS_THRESHOLD'),
        'rule': 'Meet the declared research progress threshold with zero new non-target removal, then minimize modeled duration; not a clinical recommendation'}


def plan_observed_strategies(observation, raw, *, check=lambda: None, progress=lambda *_: None):
    _require(type(observation) is TwoToolObservation, 'typed public two-tool observation required')
    controls = _normalize(observation, raw); started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started > MAX_WALL_SECONDS: raise TimeoutError('Two-tool comparison wall budget')
    approaches = plan_observed_approaches(observation.baseline, controls['geometry'], check=bounded, progress=progress)
    baseline = observation.planning_engine(); before = copy.deepcopy(baseline.state)
    stop = {'id': 'stop', 'label': 'STOP now', 'feasible': True, 'taskProgressMet': controls['minimumTargetMm3'] == 0.,
            'actionProgram': [], 'physicalStateHashes': [], 'phases': [], 'summary': _summary(baseline, before)}
    alternatives = []; rejected = []; chosen = None; reason = None
    try:
        for route in approaches['routes']:
            access = observation.planning_engine()
            for action in route['actionProgram']:
                bounded(); result = access.step(copy.deepcopy(action), check=bounded)
                _require(result['status'] == 'accepted', 'source approach no longer executes')
            for index, secondary in enumerate(_pair_poses(observation, route, controls['geometry']['entry'])):
                bounded(); failure = _screen_pair(access, secondary, controls['geometry']['nonTargetClearanceMm'])
                if failure:
                    rejected.append({'routeId': route['id'], 'pairIndex': index, 'reason': failure}); continue
                variants = []
                for identity, key in (('short', 'shortSeconds'), ('long', 'longSeconds')):
                    try:
                        common_probe = variants[0]['reassessment']['probeAction']['pose'] if variants else None
                        variants.append(_compile(observation, controls, route, secondary, controls[key], identity, probe_pose=common_probe, check=bounded))
                    except ValueError as error:
                        rejected.append({'routeId': route['id'], 'pairIndex': index, 'strategy': identity, 'reason': str(error)})
                if len(variants) == 2:
                    alternatives = variants; chosen = {'route': route, 'pairIndex': index, 'secondary': secondary}; break
            if chosen is not None: break
    except TimeoutError:
        reason = 'COMPUTATION_BUDGET'
    if not approaches['routes']:
        reason = _approach_stop_reason(approaches)
    if chosen is None and reason is None: reason = 'NO_COMPLETE_TWO_TOOL_SEQUENCE'
    alternatives.append(stop)
    report = {'schema': PUBLIC_SCHEMA, 'task': task_spec(), 'publicObservation': observation.manifest(),
        'constraints': controls, 'approaches': approaches, 'chosenPair': chosen, 'alternatives': alternatives,
        'rejections': rejected, 'decision': _decision(alternatives, controls, reason), 'scope': SCOPE}
    report['publicPlanSeal'] = semantic_digest(report)
    return report


def generate_comparison(engine, raw, *, check=lambda: None, progress=lambda *_: None):
    value = admission(engine); _require(value is not None, 'source/baseline admission is required')
    validate_admission(engine, value)
    available = defaults(engine); _require(available['available'], available['reason'])
    observation = TwoToolObservation(project_observation(engine, value['task'], _evidence(engine)['availableEvidence']), task_spec())
    public = plan_observed_strategies(observation, raw, check=check, progress=progress)
    report = {'schema': SCHEMA, 'publicPlan': public, 'taskAdmission': value, **_binding(engine), 'sourceEvidence': _evidence(engine)}
    report['reportSeal'] = semantic_digest(report)
    return report


def validate_report(engine, report):
    """Rebuild public inputs and execute canonical recipes, not caller receipts."""
    _require(type(report) is dict, 'comparison must be an object')
    _require(report.get('schema') == SCHEMA and report.get('reportSeal') == semantic_digest({k: v for k, v in report.items() if k != 'reportSeal'}), 'comparison seal differs')
    value = admission(engine)
    _require(value is not None and value == report.get('taskAdmission'), 'source baseline qualification differs')
    validate_admission(engine, value)
    binding = _binding(engine)
    _require(set(report) == {'schema', 'publicPlan', 'taskAdmission', 'sourceEvidence', 'reportSeal', *binding}, 'comparison schema differs')
    _require(all(report.get(k) == v for k, v in binding.items()) and report['sourceEvidence'] == _evidence(engine), 'source identity or planning checkpoint differs')
    observation = TwoToolObservation(project_observation(engine, value['task'], report['sourceEvidence']['availableEvidence']), task_spec())
    public = report['publicPlan']
    _require(type(public) is dict and set(public) == {'schema', 'task', 'publicObservation', 'constraints', 'approaches', 'chosenPair', 'alternatives', 'rejections', 'decision', 'scope', 'publicPlanSeal'}, 'public plan schema differs')
    _require(public['schema'] == PUBLIC_SCHEMA and semantic_digest(public['task']) == semantic_digest(task_spec()) and public['scope'] == SCOPE,
             'two-tool task declaration differs')
    _require(public['publicPlanSeal'] == semantic_digest({k: v for k, v in public.items() if k != 'publicPlanSeal'}), 'public plan seal differs')
    _require(public['publicObservation'] == observation.manifest(), 'public observation differs')
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
    validate_public_approach_plan(observation.baseline, approaches)
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
            expected.append(_compile(observation, controls, chosen['route'], chosen['secondary'], controls[key], identity, probe_pose=probe))
        reason = None
    else:
        reason = _approach_stop_reason(approaches) if not approaches['routes'] else public['decision'].get('reason')
        _require(reason in ('REQUIRED_EVIDENCE_UNAVAILABLE', 'NO_COMPLETE_TOOL_ACCESS', 'NO_COMPLETE_TWO_TOOL_SEQUENCE', 'COMPUTATION_BUDGET'), 'STOP reason differs')
    baseline = observation.planning_engine(); before = copy.deepcopy(baseline.state)
    expected.append({'id': 'stop', 'label': 'STOP now', 'feasible': True, 'taskProgressMet': controls['minimumTargetMm3'] == 0.,
        'actionProgram': [], 'physicalStateHashes': [], 'phases': [], 'summary': _summary(baseline, before)})
    _require(semantic_digest(public['alternatives']) == semantic_digest(expected), 'canonical programs, physical predictions or reassessment differ')
    _require(semantic_digest(public['decision']) == semantic_digest(_decision(expected, controls, reason)), 'public strategy/STOP decision differs')
    return True


def materialize(engine, report, candidate_id, *, check=lambda: None):
    """Independent native execution after the complete public plan is sealed."""
    validate_report(engine, report)
    matches = [row for row in report['publicPlan']['alternatives'] if row['id'] == candidate_id]
    _require(len(matches) == 1, 'unknown strategy')
    strategy = matches[0]; native = engine.clone(); before = copy.deepcopy(engine.state); checkpoints = {}
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
        if reassess and index+1 == reassess['startAction']+2:
            trial = native.clone(); probe = trial.step(copy.deepcopy(strategy['reassessment']['probeAction']), check=check)
            _require(probe['status'] == strategy['reassessment']['probeStatus'] and probe.get('reason') == strategy['reassessment']['probeReason'],
                     'native reassessment differs from public prediction')
        if index+1 in phases: checkpoints[phases[index+1]] = native.clone()
    _require(_summary(native, before) == strategy['summary'], 'native consequence vector differs')
    _require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in native.state['slots'].values()), 'native withdrawal incomplete')
    receipt = {'candidateId': candidate_id, 'reportSeal': report['reportSeal'], 'publicPlanSeal': report['publicPlan']['publicPlanSeal'],
        'publicInputHash': report['publicPlan']['publicObservation']['inputHash'], 'taskId': TASK,
        'startStateId': engine.state_id, 'finalStateId': native.state_id, 'actionCount': len(strategy['actionProgram']),
        'nativeMatchesPublicPrediction': True, 'withdrawalComplete': True, 'summary': strategy['summary'],
        'phases': [{'id': key, 'stateId': e.state_id, 'position': len(e.actions)} for key, e in checkpoints.items()]}
    return native, checkpoints, receipt
