"""Finite complete-program execution stress on source-bound public anatomy.

This does not supply patient-specific registration/segmentation uncertainty.
Candidate programs are fixed before coherent tool-command offsets are applied.
The fixed-margin comparator expands nominal full-tool non-target, protected and
context clearance; it is not a certificate for offset-dependent target progress
or unknown anatomy.
"""
from __future__ import annotations

import copy
import time
from collections import OrderedDict
from threading import RLock

import numpy as np

from .core import semantic_digest, thaw_json, freeze_json
from .worlds import AxisTranslationStress, FrozenDecisionModel
from .surgical_anatomical_encounter import AnatomicalObservation, EncounterTracker, project_context, axial_envelope
from .surgical_motion import envelopes
from .surgical_tools import TOOLS
from .surgical_planning_observation import PlanningObservation, physical_state_hash, project_observation
from .surgical_task_admission import admission
from . import surgical_two_tool as planning

SCHEMA = 'resectionlab.two-tool-execution-sensitivity.v1'
PUBLIC_SCHEMA = 'resectionlab.public-two-tool-execution-sensitivity.v1'
SCOPE = ('Finite coherent command-offset stress against fixed observed anatomy. '
    'Both tools and withdrawal are replayed. Not measured patient uncertainty, '
    'anatomical registration, segmentation, deformation, clinical risk or a guarantee '
    'between tested offsets. Outside the supplied source grid remains unassessed. '
    'Selection covers only the frozen comparison programs; no new route search.')
MAX_WALL_SECONDS = 90.
_VALIDATED = OrderedDict()
_LOCK = RLock()


def _cache_key(report, observation, study):
    return semantic_digest({'comparison': report['reportSeal'], 'observation': observation.manifest(), 'study': study})


def _remember(key):
    # Complete-content digests only. Native/source/public inputs are rebuilt
    # before every hit; no study, scene, engine, callback or mask is retained.
    with _LOCK:
        _VALIDATED[key] = True; _VALIDATED.move_to_end(key)
        while len(_VALIDATED) > 32: _VALIDATED.popitem(last=False)


def _require(value, message):
    if not value: raise ValueError('STRATEGY_SENSITIVITY: '+message)


def _controls(raw, plan):
    _require(type(raw) is dict and set(raw) == {'offsetMm'}, 'exact offsetMm control required')
    value = raw['offsetMm']
    _require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 2., 'offset outside the 0–2mm software stress bound')
    _require('encounter' in plan['constraints'], 'generate a supplied-context comparison first')
    _require(plan['constraints']['encounter']['marginMm']+value <= 5., 'combined fixed context margin exceeds the 5mm software bound')
    return {'offsetMm': float(value)}


def _shift(action, world):
    result = copy.deepcopy(action)
    if 'pose' in result and not world.deterministic:
        result['pose']['position'] = world.transform_points(np.asarray(result['pose']['position'])).tolist()
    return result


def _fixed_clearance(engine, action, previous, controls, extra):
    if not extra: return
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    for hand, slot in engine.state['slots'].items():
        params = engine.parameters(slot)
        cover = (axial_envelope(previous[hand], slot, slot['instrument'], params)
            if hand == action['slot'] and action['operation'] in ('move', 'apply', 'withdraw')
            else envelopes(TOOLS[slot['instrument']], params, slot['position'], slot['quaternion']))
        for a, b, radius, part in cover:
            if part == 'shaft':
                _require(not engine.touched(a, b, radius+extra), 'FIXED_MARGIN_SHAFT_CLEARANCE')
            cells = engine.touched(a, b, radius+controls['geometry']['nonTargetClearanceMm']+extra)
            _require(not any(int(engine.scene.labels.flat[cell]) not in targets for cell in cells), 'FIXED_MARGIN_NON_TARGET_CLEARANCE')
            if engine.protection is not None:
                _require(not len(engine.protection.capsule_hits(a, b, radius+extra, positions=engine._positions())), 'FIXED_MARGIN_PROTECTED_CLEARANCE')


def _replay(engine, row, controls, context, world, *, fixed_extra=None, check=lambda: None):
    """Execute a held program; a failed/partial prefix is retained, never installed."""
    before = copy.deepcopy(engine.state)
    tracker = EncounterTracker(context, engine, controls['encounter'])
    expanded = (EncounterTracker(context, engine, {**controls['encounter'],
        'marginMm': controls['encounter']['marginMm']+fixed_extra}) if fixed_extra is not None else None)
    actions = [_shift(action, world) for action in row['actionProgram']]
    hashes = []; receipts = []; context_hashes = []; phases = []; failure = None; fixed_failure = None; probe = None
    reassess = next((p for p in row['phases'] if p['id'] == 'reassess'), None)
    for index, action in enumerate(actions):
        check(); previous = copy.deepcopy(engine.state['slots'])
        result = engine.step(copy.deepcopy(action), check=check)
        receipts.append({key: result.get(key) for key in ('status', 'reason', 'completedFraction', 'elapsedSeconds')})
        hashes.append(physical_state_hash(engine))
        # An accepted finite-time CUSA action may stop spatially while activation
        # continues. Other partial transitions do not complete the held program.
        finite = (action['slot'] == 'left' and action['operation'] == 'apply' and result['status'] == 'partial'
            and result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS'
            and abs(result['elapsedSeconds']-action['duration']) <= 1e-8)
        tracker.observe(engine, action, previous)
        context_hashes.append(semantic_digest(tracker.receipt(engine)))
        if expanded: expanded.observe(engine, action, previous)
        try:
            _require(result['status'] == 'accepted' or finite, 'execution refused: '+str(result.get('reason')))
            planning._check_burdens(engine, before, controls)
            planning._check_action_clearance(engine, action, previous, controls['geometry']['nonTargetClearanceMm'])
        except ValueError as error:
            failure = {'actionIndex': index, 'operation': action['operation'],
                'phase': next((p['id'] for p in row['phases'] if p['startAction'] <= index < p['endAction']), None),
                'reason': str(error)}
            break
        if expanded and fixed_failure is None:
            try: _fixed_clearance(engine, action, previous, controls, fixed_extra)
            except ValueError as error: fixed_failure = {'actionIndex': index, 'reason': str(error)}
        if reassess and index+1 == reassess['startAction']+2:
            probe_action = _shift(row['reassessment']['probeAction'], world)
            trial = engine.clone(); outcome = trial.step(copy.deepcopy(probe_action), check=check)
            probe = {'action': probe_action, 'status': outcome['status'], 'reason': outcome.get('reason')}
        for phase in row['phases']:
            if phase['endAction'] == index+1:
                phases.append({'id': phase['id'], 'endAction': index+1,
                    'physicalStateHash': physical_state_hash(engine), 'contextHash': context_hashes[-1]})
    complete = failure is None and len(receipts) == len(actions)
    withdrawn = all(not s['active'] and not s['inserted'] and not s['grip'] for s in engine.state['slots'].values())
    summary = planning._summary(engine, before); encounter = tracker.receipt(engine)
    progress = summary['additionalTargetMm3'] >= controls['minimumTargetMm3']
    violations = ([] if complete else ['INCOMPLETE_EXECUTION'])+([] if withdrawn else ['INCOMPLETE_WITHDRAWAL'])
    if not encounter['withinLimit']: violations.append('ANATOMICAL_CONTEXT_LIMIT')
    if not progress: violations.append('PROGRESS_THRESHOLD')
    value = {'scenarioId': world.world_id, 'translationRasMm': world.anatomy_transform_mm[:3, 3].tolist(),
        'actionProgram': actions, 'actionReceipts': receipts, 'physicalStateHashes': hashes,
        'contextPrefixHashes': context_hashes,
        'phases': phases, 'failure': failure, 'completeExecution': complete, 'withdrawalComplete': withdrawn,
        'summary': summary, 'anatomicalEncounter': encounter, 'reassessment': probe,
        'taskProgressMet': progress, 'violations': violations, 'meetsTask': not violations}
    if expanded:
        value.update(fixedMarginEncounter=expanded.receipt(engine), fixedMarginFailure=fixed_failure)
    return value


def _choose(rows):
    eligible = [r for r in rows if r['eligible']]
    return min(eligible, key=lambda r: (r['elapsedSeconds'], -r['additionalTargetMm3'], r['id']))['id'] if eligible else 'stop'


def assess_observed(observation, plan, raw, *, check=lambda: None, progress=lambda *_: None):
    """Only a typed public observation and a sealed, frozen candidate family enter."""
    _require(type(observation) is planning.TwoToolObservation and observation.context is not None, 'typed source-context observation required')
    # Detach before invoking user callbacks. FrozenDecisionModel and all replay
    # inputs must refer to the same snapshot even if a caller edits its objects.
    plan = thaw_json(freeze_json(plan))
    _require(plan['publicPlanSeal'] == semantic_digest({k: v for k, v in plan.items() if k != 'publicPlanSeal'})
        and plan['publicObservation'] == observation.manifest(), 'sealed public comparison input differs')
    observation = planning.TwoToolObservation(PlanningObservation(observation.baseline.labels,
        observation.baseline.affine, observation.baseline.fields), observation.task,
        AnatomicalObservation(observation.context.mask, observation.context.affine, observation.context.review_status))
    normalized = _controls(raw, plan); config = AxisTranslationStress(normalized['offsetMm'])
    controls = planning._normalize(observation, plan['constraints']); worlds = config.worlds()
    model = FrozenDecisionModel.create(case_hash=observation.manifest()['inputHash'],
        geometry=observation.manifest(), objectives=controls, tools=thaw_json(observation.task),
        world_generator=config.to_dict(), action_primitives={r['id']: r['actionProgram'] for r in plan['alternatives']})
    candidates = []
    count = len(plan['alternatives'])*len(worlds)
    for row_index, row in enumerate(plan['alternatives']):
        outcomes = []
        for world_index, world in enumerate(worlds):
            done = row_index*len(worlds)+world_index
            check(); progress(done/(2*count), f'Checking public execution offsets {done+1}/{count}: {row["id"]} {world.world_id}')
            result = _replay(observation.planning_engine(), row, controls, observation.context, world,
                fixed_extra=config.offset_mm if world.world_id == 'nominal' else None, check=check)
            if world.world_id == 'nominal':
                _require(result['completeExecution'] and result['actionReceipts'] == row.get('actionReceipts', [])
                    and result['physicalStateHashes'] == row.get('physicalStateHashes', [])
                    and semantic_digest(result['summary']) == semantic_digest(row['summary'])
                    and semantic_digest(result['anatomicalEncounter']) == semantic_digest(row['anatomicalEncounter']),
                    'zero-offset replay differs from the canonical nominal program')
            outcomes.append(result)
        candidates.append({'id': row['id'], 'outcomes': outcomes})
    def ranks(index=0, fixed=False, all_worlds=False):
        result = []
        for candidate in candidates:
            outcome = candidate['outcomes'][index]
            eligible = all(o['meetsTask'] for o in candidate['outcomes']) if all_worlds else outcome['meetsTask']
            if fixed: eligible &= outcome['fixedMarginEncounter']['withinLimit'] and outcome['fixedMarginFailure'] is None
            result.append({'id': candidate['id'], 'eligible': eligible,
                'elapsedSeconds': outcome['summary']['elapsedSeconds'], 'additionalTargetMm3': outcome['summary']['additionalTargetMm3']})
        return result
    nominal = _choose(ranks()); fixed = _choose(ranks(fixed=True)); stable = _choose(ranks(all_worlds=True))
    _require(nominal == plan['decision']['selectedId'], 'nominal selection differs from the source comparison')
    per_world = [{'scenarioId': world.world_id, 'selectedId': _choose(ranks(i))} for i, world in enumerate(worlds)]
    value = {'schema': PUBLIC_SCHEMA, 'publicObservation': observation.manifest(),
        'comparisonPublicPlanSeal': plan['publicPlanSeal'], 'controls': normalized,
        'frozenDecisionModel': model.to_dict(), 'scenarioConfiguration': config.to_dict(),
        'scenarioCount': len(worlds), 'candidates': candidates,
        'policies': {'nominal': nominal, 'fixedMargin': fixed, 'allDeclaredScenarios': stable},
        'fixedMarginDefinition': {'nonTargetClearanceMm': controls['geometry']['nonTargetClearanceMm']+config.offset_mm,
            'contextMarginMm': controls['encounter']['marginMm']+config.offset_mm, 'protectedExtraMm': config.offset_mm,
            'geometry': 'complete nominal axial tool union; all retained shaft cells, mapped non-target working cells and supplied protected/context cells'},
        'perScenarioPreference': per_world,
        'preferenceChanges': [p for p in per_world if p['selectedId'] != nominal],
        'calibratedUncertainty': {'supported': False, 'reason': 'No patient-specific execution-error calibration is supplied'},
        'scope': SCOPE}
    value['assessmentSeal'] = semantic_digest(value)
    check(); return value


def _observation(engine, report):
    context, _ = project_context(engine, report['contextBinding']['layerId'])
    return planning.TwoToolObservation(project_observation(engine, admission(engine)['task'],
        report['sourceEvidence']['availableEvidence']), planning.task_spec(with_context=True), context)


def generate_sensitivity(engine, report, raw, *, check=lambda: None, progress=lambda *_: None):
    started = time.monotonic()
    report = thaw_json(freeze_json(report))
    def bounded():
        check()
        if time.monotonic()-started > MAX_WALL_SECONDS:
            raise TimeoutError('Sensitivity computation incomplete; no study installed')
    planning.validate_report(engine, report, check=bounded)
    _require(report['schema'] == planning.CONTEXT_SCHEMA, 'a supplied-context comparison is required')
    observation = _observation(engine, report)
    assessment = assess_observed(observation, report['publicPlan'], raw, check=bounded, progress=progress)
    # Seal public decisions before checking them against independent native
    # clones. Native results cannot reorder candidates or alter the policy.
    # Every scenario executes again; only pure geometry caches may be shared.
    controls = planning._normalize(observation, report['publicPlan']['constraints'])
    worlds = AxisTranslationStress(assessment['controls']['offsetMm']).worlds()
    count = len(report['publicPlan']['alternatives'])*len(worlds)
    for row_index, (row, candidate) in enumerate(zip(report['publicPlan']['alternatives'], assessment['candidates'])):
        for world_index, (world, predicted) in enumerate(zip(worlds, candidate['outcomes'])):
            done = row_index*len(worlds)+world_index
            bounded(); progress(.5+done/(2*count), f'Checking native execution offsets {done+1}/{count}: {row["id"]} {world.world_id}')
            actual = _replay(engine.clone(), row, controls, observation.context, world,
                fixed_extra=assessment['controls']['offsetMm'] if world.world_id == 'nominal' else None, check=bounded)
            _require(semantic_digest(actual) == semantic_digest(predicted), 'native execution differs from sealed public scenario')
    value = {'schema': SCHEMA, 'comparisonReportSeal': report['reportSeal'],
        'contextBindingSeal': report['contextBinding']['bindingSeal'], 'assessment': assessment, 'nativeReplayVerified': True}
    value['studySeal'] = semantic_digest(value)
    planning.validate_report(engine, report, check=bounded)
    progress(1., 'Complete public and native execution-offset study checked')
    bounded(); _remember(_cache_key(report, observation, value)); return value


def validate_sensitivity(engine, report, study, *, check=lambda: None):
    check()
    report = thaw_json(freeze_json(report)); study = thaw_json(freeze_json(study))
    _require(type(study) is dict and set(study) == {'schema', 'comparisonReportSeal', 'contextBindingSeal', 'assessment', 'nativeReplayVerified', 'studySeal'}, 'study schema differs')
    planning.validate_report(engine, report, check=check)
    _require(report['schema'] == planning.CONTEXT_SCHEMA, 'a supplied-context comparison is required')
    _require(study['schema'] == SCHEMA and study['studySeal'] == semantic_digest({k: v for k, v in study.items() if k != 'studySeal'})
        and study['comparisonReportSeal'] == report['reportSeal']
        and study['contextBindingSeal'] == report['contextBinding']['bindingSeal'], 'study seal or source comparison differs')
    _require(type(study['assessment']) is dict and 'controls' in study['assessment'], 'public assessment controls missing')
    observation = _observation(engine, report)
    key = _cache_key(report, observation, study)
    with _LOCK:
        if key in _VALIDATED:
            _VALIDATED.move_to_end(key); check(); return True
    expected = generate_sensitivity(engine, report, study['assessment']['controls'], check=check)
    _require(semantic_digest(expected) == semantic_digest(study), 'canonical sensitivity study differs')
    return True
