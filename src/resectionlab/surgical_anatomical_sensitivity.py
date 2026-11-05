"""Declared source-grid label-location stress of held complete programs.

Only context membership changes. Tissue, target, tool poses and mechanics remain
fixed. Integer source-grid translations avoid interpolation; uncovered cells are
unknown. This is a conditional segmentation/localization stress test, not a
patient-calibrated error model, rigid patient registration or brain shift.
"""
from __future__ import annotations

import copy
import time
from fractions import Fraction

import numpy as np

from .surgical_proof_cache import proof_scope
from .core import array_digest, freeze_json, thaw_json, semantic_digest
from .surgical_anatomical_encounter import axial_envelope, AnatomicalObservation
from .surgical_motion import envelopes
from .surgical_tools import TOOLS
from .surgical_planning_observation import physical_state_hash, PlanningObservation
from . import surgical_two_tool as planning

SCHEMA = 'resectionlab.two-tool-context-label-sensitivity.v1'
PUBLIC_SCHEMA = 'resectionlab.public-two-tool-context-label-sensitivity.v1'
SCOPE = ('Declared integer source-grid context-label translations on fixed tissue and target; '
    'held complete nominal tool programs, including withdrawal. No interpolation, '
    'patient error calibration, registration correction, brain shift, clinical risk, '
    'guarantee between tested shifts or new route search. Outside the original source '
    'grid remains unassessed; newly uncovered in-grid cells are unknown, never safe empty space.')


def _require(ok, message):
    if not ok: raise ValueError('ANATOMICAL_SENSITIVITY: '+message)


def _controls(raw, observation, plan):
    _require(type(raw) is dict and set(raw) == {'contextVoxelShift'}, 'exact contextVoxelShift control required')
    step = raw['contextVoxelShift']
    _require(type(step) is int and 0 <= step <= 2, 'use an integer 0–2 source-voxel shift')
    distance = step*float(np.max(np.linalg.norm(observation.context.affine[:3, :3], axis=0)))
    _require(distance <= 2.+1e-10, 'physical label shift exceeds the 2mm software stress bound')
    _require(plan['constraints']['encounter']['marginMm']+distance <= 5.+1e-10,
             'fixed context margin exceeds the 5mm software bound')
    return {'contextVoxelShift': step}, distance


def label_scenarios(context, step):
    """Destination cells sample source cell i-k exactly; no wrap or padding evidence."""
    context.assert_intact()
    shifts = [('nominal', [0, 0, 0])]
    if step:
        for axis in range(3):
            for sign in (-1, 1):
                shift = [0, 0, 0]; shift[axis] = sign*step
                shifts.append((f'grid-{axis}{"-" if sign < 0 else "+"}', shift))
    result = []
    for name, shift in shifts:
        mask = np.zeros_like(context.mask); known = np.zeros_like(mask)
        dest = []; source = []
        for size, delta in zip(mask.shape, shift):
            lo, hi = max(0, delta), min(size, size+delta)
            dest.append(slice(lo, hi)); source.append(slice(lo-delta, hi-delta))
        if all(s.stop > s.start for s in dest):
            mask[tuple(dest)] = context.mask[tuple(source)]; known[tuple(dest)] = True
        result.append({'id': name, 'voxelShift': shift,
            'translationRasMm': (context.affine[:3, :3]@np.asarray(shift)).tolist(),
            'outOfFieldContextCells': int(context.mask.sum())-int(mask.sum()), 'mask': mask, 'known': known})
    return result


def _scenario_manifest(scenario):
    return {k: copy.deepcopy(scenario[k]) for k in ('id', 'voxelShift', 'translationRasMm', 'outOfFieldContextCells')} | {
        'maskHash': array_digest(scenario['mask']), 'knownCoverageHash': array_digest(scenario['known'])}


class _Footprint:
    """Label-independent retained destination cells at every complete-tool sweep."""
    def __init__(self, margin):
        self.margin = margin; self.seen = {'shaft': set(), 'working': set()}

    def observe(self, engine, action, previous):
        removed = set(engine.state['removed'])
        for hand, slot in engine.state['slots'].items():
            params = engine.parameters(slot)
            cover = (axial_envelope(previous[hand], slot, slot['instrument'], params)
                if hand == action['slot'] and action['operation'] in ('move', 'apply', 'withdraw')
                else envelopes(TOOLS[slot['instrument']], params, slot['position'], slot['quaternion']))
            for a, b, radius, part in cover:
                self.seen[part].update(int(c) for c in engine.cells(a, b, radius+self.margin) if int(c) not in removed)


def _endpoint(engine, before, footprint, scenario, worked, limit):
    # Work volume is the union volume of cells receiving ANY new capacity or
    # removal. It is not allocated aspiration capacity or a biological injury.
    mask = scenario['mask'].ravel(); known = scenario['known'].ravel()
    parts = {p: sorted(c for c in cells if mask[c]) for p, cells in footprint.seen.items()}
    union = sorted(set(parts['shaft']) | set(parts['working']))
    unknown = sorted(c for c in footprint.seen['shaft'] | footprint.seen['working'] | worked if not known[c])
    work = sorted(c for c in worked if mask[c]); new = set(engine.state['removed'])-set(before['removed'])
    historical = sum(bool(mask[c]) for c in before['removed']); volume = engine.scene.voxel_mm3
    return {**limit, 'sourceContextMm3': int(mask.sum())*volume,
        'baselineRemovedContextMm3': historical*volume,
        'baselineRetainedContextMm3': (int(mask.sum())-historical)*volume,
        'newRemovedContextMm3': sum(bool(mask[c]) for c in new)*volume,
        'newContextWorkCells': work, 'newContextWorkMm3': len(work)*volume,
        'nearSourceCells': union, 'nearPositionsMm': engine.scene.points(union[:12]).tolist(),
        'nearMm3': len(union)*volume, 'shaftNearMm3': len(parts['shaft'])*volume,
        'workingNearMm3': len(parts['working'])*volume,
        'unknownEncounteredCells': unknown, 'unknownEncounteredMm3': len(unknown)*volume,
        'withinLimit': len(union)*volume <= limit['maxNearMm3']}


def _replay(engine, row, controls, scenarios, extra, *, check):
    before = copy.deepcopy(engine.state); limit = controls['encounter']
    footprint = _Footprint(limit['marginMm']); expanded = _Footprint(limit['marginMm']+extra)
    worked = set(); receipts = []; hashes = []; prefixes = [[] for _ in scenarios]
    for action in row['actionProgram']:
        check(); previous = copy.deepcopy(engine.state['slots'])
        outcome = engine.step(copy.deepcopy(action), check=check)
        finite = (action['slot'] == 'left' and action['operation'] == 'apply' and outcome['status'] == 'partial'
            and outcome.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS'
            and abs(outcome['elapsedSeconds']-action['duration']) <= 1e-8)
        _require(outcome['status'] == 'accepted' or finite, 'held nominal program no longer executes')
        planning._check_burdens(engine, before, controls)
        planning._check_action_clearance(engine, action, previous, controls['geometry']['nonTargetClearanceMm'])
        worked.update(set(engine.state['removed'])-set(before['removed']))
        # In this admitted CUSA/Frazier family exact partial work is monotone
        # until whole-cell removal; no other mechanism discards/reverts it.
        # Removed cells plus increased exact work therefore cover the action.
        worked.update(int(c) for c, amount in engine.state['cusaV3']['progressExact'].items()
            if Fraction(*amount) > Fraction(*before['cusaV3']['progressExact'].get(c, [0, 1])))
        footprint.observe(engine, action, previous); expanded.observe(engine, action, previous)
        hashes.append(physical_state_hash(engine))
        receipts.append({key: outcome.get(key) for key in ('status', 'reason', 'completedFraction', 'elapsedSeconds')})
        for scenario, values in zip(scenarios, prefixes):
            values.append(semantic_digest(_endpoint(engine, before, footprint, scenario, worked, limit)))
    _require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in engine.state['slots'].values()),
             'held nominal program did not completely withdraw')
    summary = planning._summary(engine, before)
    _require(hashes == row.get('physicalStateHashes', []) and receipts == row.get('actionReceipts', [])
        and semantic_digest(summary) == semantic_digest(row['summary']), 'nominal physical replay differs')
    outcomes = []
    for scenario, values in zip(scenarios, prefixes):
        encounter = _endpoint(engine, before, footprint, scenario, worked, limit)
        violations = []
        if not encounter['withinLimit']: violations.append('ANATOMICAL_CONTEXT_LIMIT')
        if encounter['newContextWorkCells']: violations.append('NEW_CONTEXT_MANIPULATION')
        if encounter['unknownEncounteredCells']: violations.append('UNKNOWN_CONTEXT_COVERAGE')
        if summary['additionalTargetMm3'] < controls['minimumTargetMm3']: violations.append('PROGRESS_THRESHOLD')
        outcomes.append({'scenarioId': scenario['id'], 'encounter': encounter,
            'contextPrefixHashes': values, 'violations': violations, 'meetsTask': not violations})
    nominal = outcomes[0]['encounter']; original = row['anatomicalEncounter']
    _require(all(nominal[k] == v for k, v in original.items() if k in nominal), 'nominal context replay differs')
    return {'id': row['id'], 'actionReceipts': receipts, 'physicalStateHashes': hashes,
        'completeExecution': True, 'withdrawalComplete': True, 'summary': summary,
        'outcomes': outcomes, 'fixedMarginEncounter': _endpoint(engine, before, expanded, scenarios[0], worked,
            {**limit, 'marginMm': limit['marginMm']+extra})}


def _choose(candidates, index=0, *, every=False, fixed=False):
    eligible = []
    for row in candidates:
        allowed = all(o['meetsTask'] for o in row['outcomes']) if every else row['outcomes'][index]['meetsTask']
        if fixed: allowed &= row['fixedMarginEncounter']['withinLimit']
        if allowed: eligible.append((row['summary']['elapsedSeconds'], -row['summary']['additionalTargetMm3'], row['id']))
    return min(eligible)[2] if eligible else 'stop'


def assess_observed(observation, plan, raw, *, check=lambda: None, progress=lambda *_: None):
    _require(type(observation) is planning.TwoToolObservation and observation.context is not None, 'typed source-context observation required')
    plan = thaw_json(freeze_json(plan))
    _require(plan['publicPlanSeal'] == semantic_digest({k: v for k, v in plan.items() if k != 'publicPlanSeal'})
        and plan['publicObservation'] == observation.manifest(), 'sealed public comparison differs')
    observation = planning.TwoToolObservation(PlanningObservation(observation.baseline.labels,
        observation.baseline.affine, observation.baseline.fields), observation.task,
        AnatomicalObservation(observation.context.mask, observation.context.affine, observation.context.review_status))
    normalized, extra = _controls(raw, observation, plan)
    controls = planning._normalize(observation, plan['constraints'])
    scenarios = label_scenarios(observation.context, normalized['contextVoxelShift'])
    candidates = []
    for index, row in enumerate(plan['alternatives']):
        check(); progress(index/(2*len(plan['alternatives'])), 'Checking public held program: '+row['id'])
        candidates.append(_replay(observation.planning_engine(), row, controls, scenarios, extra, check=check))
    nominal = _choose(candidates)
    _require(nominal == plan['decision']['selectedId'], 'nominal selection differs from source comparison')
    preferences = [{'scenarioId': scenario['id'], 'selectedId': _choose(candidates, i)} for i, scenario in enumerate(scenarios)]
    value = {'schema': PUBLIC_SCHEMA, 'publicObservation': observation.manifest(),
        'comparisonPublicPlanSeal': plan['publicPlanSeal'], 'controls': normalized,
        'scenarioConfiguration': {'family': 'integer-source-grid-context-label-translations-v1',
            'appliedTo': 'context membership on fixed destination tissue cells only',
            'parameterBasis': 'declared-uncalibrated-label-location-stress', 'distribution': None, 'patientCalibration': None,
            'maximumTranslationMm': extra, 'scenarios': [_scenario_manifest(s) for s in scenarios]},
        'scenarioCount': len(scenarios), 'candidates': candidates,
        'policies': {'nominal': nominal, 'fixedContextMargin': _choose(candidates, fixed=True),
            'allDeclaredScenarios': _choose(candidates, every=True)},
        'fixedMarginDefinition': {'contextMarginMm': controls['encounter']['marginMm']+extra,
            'isGuarantee': False, 'scope': 'Nominal retained context with an expanded tool envelope; does not bound changed label membership in removed or manipulated cells.'},
        'perScenarioPreference': preferences, 'preferenceChanges': [p for p in preferences if p['selectedId'] != nominal],
        'calibratedUncertainty': {'supported': False, 'reason': 'No patient-specific label-location error calibration is supplied'},
        'scope': SCOPE}
    value['assessmentSeal'] = semantic_digest(value)
    check(); return value


def generate_sensitivity(engine, report, raw, *, check=lambda: None, progress=lambda *_: None):
    with proof_scope(engine.scene, report, raw, engine=engine):
        return _canonical_generate_sensitivity(engine, report, raw, check=check, progress=progress)


def _canonical_generate_sensitivity(engine, report, raw, *, check=lambda: None, progress=lambda *_: None):
    from .surgical_strategy_sensitivity import _observation, _remember, _cache_key, MAX_WALL_SECONDS
    started = time.monotonic(); report = thaw_json(freeze_json(report))
    def bounded():
        check()
        if time.monotonic()-started > MAX_WALL_SECONDS:
            raise TimeoutError('Anatomical sensitivity incomplete; no study installed')
    planning.validate_report(engine, report, check=bounded)
    _require(report['schema'] == planning.CONTEXT_SCHEMA, 'a supplied-context comparison is required')
    observation = _observation(engine, report)
    assessment = assess_observed(observation, report['publicPlan'], raw, check=bounded, progress=progress)
    scenarios = label_scenarios(observation.context, assessment['controls']['contextVoxelShift'])
    controls = planning._normalize(observation, report['publicPlan']['constraints'])
    for index, (row, predicted) in enumerate(zip(report['publicPlan']['alternatives'], assessment['candidates'])):
        bounded(); progress(.5+index/(2*len(assessment['candidates'])), 'Checking native held program: '+row['id'])
        actual = _replay(engine.clone(), row, controls, scenarios,
            assessment['scenarioConfiguration']['maximumTranslationMm'], check=bounded)
        _require(semantic_digest(actual) == semantic_digest(predicted), 'native program differs from sealed public assessment')
    value = {'schema': SCHEMA, 'comparisonReportSeal': report['reportSeal'],
        'contextBindingSeal': report['contextBinding']['bindingSeal'], 'assessment': assessment, 'nativeReplayVerified': True}
    value['studySeal'] = semantic_digest(value)
    progress(1., 'Complete public and native source-label study checked')
    planning.validate_report(engine, report, check=bounded)
    bounded(); _remember(_cache_key(report, observation, value))
    return value


def validate_sensitivity(engine, report, study, *, check=lambda: None):
    with proof_scope(engine.scene, report, study, engine=engine):
        return _canonical_validate_sensitivity(engine, report, study, check=check)


def _canonical_validate_sensitivity(engine, report, study, *, check=lambda: None):
    from .surgical_strategy_sensitivity import _observation, _cache_key, _LOCK, _VALIDATED
    check(); report = thaw_json(freeze_json(report)); study = thaw_json(freeze_json(study))
    _require(type(study) is dict and set(study) == {'schema', 'comparisonReportSeal', 'contextBindingSeal', 'assessment', 'nativeReplayVerified', 'studySeal'}, 'study schema differs')
    planning.validate_report(engine, report, check=check)
    _require(study['schema'] == SCHEMA and study['studySeal'] == semantic_digest({k: v for k, v in study.items() if k != 'studySeal'})
        and study['comparisonReportSeal'] == report['reportSeal'] and study['contextBindingSeal'] == report['contextBinding']['bindingSeal'], 'study or source binding differs')
    key = _cache_key(report, _observation(engine, report), study)
    with _LOCK:
        if key in _VALIDATED: _VALIDATED.move_to_end(key); check(); return True
    expected = generate_sensitivity(engine, report, study['assessment']['controls'], check=check)
    _require(semantic_digest(expected) == semantic_digest(study), 'canonical anatomical sensitivity differs')
    return True
