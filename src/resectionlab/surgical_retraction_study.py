"""Bind recorded paired-tool studies to native branches for desktop inspection.

This projection runs no search, simulation or learning. Playback still uses the
ordinary causal native replay path. Receipts retain planning versus replay cost.
"""
import copy
import re
from types import SimpleNamespace
import numpy as np

from .core import semantic_digest
from .surgical_staged_planning import larger_task, source_zones, assess_task
from .surgical_sector_planning import _sector_cells, _prefix_actions

SCHEMA = 'resectionlab.retractor-native-study.v1'


def _source_prefix(record, route):
    comparisons = [record.get('planning')]+[row.get('sourceAccessComparison') for row in
        record.get('jointBaselineHistory', record.get('sectorPlanningHistory', []))]
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        for plan in comparison.get('alternatives', []):
            if plan.get('route') == route:
                return _prefix_actions(plan)
    return None


def _phase_state(history):
    inserted, held, retractor_used = {'left':False,'right':False}, False, False
    for event in history:
        action = event['action']; hand = action['slot']; op = action['operation']
        if op == 'place': inserted[hand] = True
        if op == 'withdraw': inserted[hand] = False
        if hand == 'right':
            retractor_used |= op == 'exchange' and action.get('instrument') == 'brain_retractor'
            if any(item.get('kind') == 'coupled_hold_created' for item in event['events']): held = True
            if op in ('release','withdraw'): held = False
    return inserted, held, retractor_used


def install_retraction_study(scene, record, receipt, *, receipt_sha256):
    """Project an existing measured study; never generate missing native work."""
    report = {'schema': SCHEMA, 'sourceSceneId': scene.identity,
        'sourceStateId': record['branches']['main']['stateId'],
        'sourceRole': 'DEVELOPMENT', 'clinicalOutcomes': 'unavailable',
        'optimizerUpdates': 0, 'receiptSha256': receipt_sha256,
        'route': copy.deepcopy(receipt['route']), 'task': copy.deepcopy(receipt['task']),
        'physicalCalls': receipt['cost']['physicalCalls'],
        'wallSeconds': receipt['wallSeconds'], 'variants': []}
    for value in receipt['variants']:
        snapshot = record['branches'][value['id']]
        if snapshot != value['nativeSnapshot'] or value['exactReplay'] is not True:
            raise ValueError('Recorded study endpoint differs from measured native program')
        history = record['timeline'][value['id']]
        phases = [{'label': 'Intact source', 'position': 0}]
        for position, event in enumerate(history, 1):
            action = event['action']
            if any(item.get('kind') == 'coupled_hold_created' for item in event['events']):
                phases.append({'label': 'Native coupled hold acquired', 'position': position})
            if action['slot'] == 'right' and action['operation'] in ('release', 'withdraw'):
                phases.append({'label': 'Retractor '+action['operation'], 'position': position})
            if action['slot'] == 'left' and action['operation'] == 'withdraw':
                phases.append({'label': 'CUSA withdrawn', 'position': position})
        baseline = value['accessEndpointMetrics']
        _, held_at_access, _ = _phase_state(history[:baseline['nativeActions']])
        phases.append({'label': ('Access complete with tissue held' if held_at_access else
                       'Access complete') if value['accessComplete'] else 'Access stopped',
                       'position': baseline['nativeActions']})
        for stage in value['workStages']:
            phases.append({'label': 'CUSA work stage '+str(stage['stage']+1),
                           'position': stage['end']})
        phases.append({'label': 'Final native endpoint', 'position': len(snapshot['actions'])})
        report['variants'].append({
            **{key: copy.deepcopy(value[key]) for key in ('assisted', 'status',
                'failureReason', 'accessComplete', 'workStages', 'outcome',
                'nativeActions', 'finalMetrics', 'sampledMechanicalPeaks')},
            'branchId': value['id'], 'finalStateId': snapshot['stateId'],
            'baseline': copy.deepcopy(baseline), 'phases': sorted(phases, key=lambda p:p['position']),
            'exactNativeReplay': True,
            'costMode': ('recorded_program_replay' if value.get('recordedProgramReplay') else
                         'planning_and_replay'),
            'publicCalls': value['publicCalls'],
            'originalPlanningAndReplayCalls': value.get('baselineSourcePublicCalls'),
            'outsideRecoveryOffsetMm': value.get('acquisition', {}).get(
                'proposalUsed', {}).get('outsideRecoveryOffsetMm', 0.)})
    report['comparisonSeal'] = semantic_digest(report)
    validate_retraction_study(scene, record, report)
    record['retractionStudy'] = report
    return report


def validate_retraction_study(scene, record, report):
    """Remeasure gates from native source cells and intervals, without replay."""
    if (report.get('schema') != SCHEMA or report.get('sourceSceneId') != scene.identity or
            report.get('sourceRole') != 'DEVELOPMENT' or report.get('optimizerUpdates') != 0 or
            report.get('clinicalOutcomes') != 'unavailable' or
            not re.fullmatch(r'[0-9a-f]{64}', report.get('receiptSha256', '')) or
            report.get('comparisonSeal') != semantic_digest({
                k:v for k,v in report.items() if k != 'comparisonSeal'}) or
            report.get('task') != larger_task(scene)):
        raise ValueError('Paired-tool study source, content or task seal differs')
    source = record['branches'].get('main', {})
    if source.get('actions') != [] or source.get('stateId') != report['sourceStateId']:
        raise ValueError('Paired-tool study lacks its common intact source')
    rows = report['variants']
    if not 1 <= len(rows) <= 3 or len({r['branchId'] for r in rows}) != len(rows):
        raise ValueError('Paired-tool study branch inventory differs')
    if (not 0 <= report['physicalCalls'] <= 1200 or not 0 <= report['wallSeconds'] <= 180 or
            sum(row['publicCalls'] for row in rows) != report['physicalCalls']):
        raise ValueError('Paired-tool study resource accounting differs')
    zones, _ = source_zones(scene)
    sectors, _ = _sector_cells(scene, report['route'])
    prefix = _source_prefix(record, report['route'])
    target_labels = {int(r['id']) for r in scene.regions if r.get('target')}
    target_volume = lambda cells: sum(int(scene.labels.flat[cell]) in target_labels
                                     for cell in set(cells))*scene.voxel_mm3
    for row in rows:
        snapshot = record['branches'].get(row['branchId'], {})
        history = record.get('timeline', {}).get(row['branchId'], [])
        actions = snapshot.get('actions', [])
        if (snapshot.get('sceneId') != scene.identity or
                snapshot.get('stateId') != row['finalStateId'] or
                len(actions) != row['nativeActions'] or len(history) != len(actions) or
                row['exactNativeReplay'] is not True or
                record.get('parents', {}).get(row['branchId']) != {'branchId':'main','position':0}):
            raise ValueError('Paired-tool native endpoint or ancestry differs')
        state = snapshot['checkpoint']['state']
        all_cells = set()
        previous_state = report['sourceStateId']
        for action, entry in zip(actions, history):
            if entry['action'] != action or entry['stateBefore'] != previous_state:
                raise ValueError('Paired-tool native history chain differs')
            previous_state = entry['stateAfter']
            all_cells.update(entry.get('deltas', {}).get('removedCells', []))
        if previous_state != row['finalStateId'] or all_cells != set(state['removed']):
            raise ValueError('Paired-tool native removal ledger differs')
        configs = {}
        previous_end = row['baseline']['nativeActions']
        for stage in row['workStages']:
            start, end = stage['start'], stage['end']
            if not previous_end <= start < end <= len(history) or history[end-1]['stateAfter'] != stage['stateId']:
                raise ValueError('Paired-tool work interval differs')
            cells = set()
            radii = set()
            for index in range(start, end):
                action = actions[index]
                if action['slot'] != 'left' or action['operation'] != 'apply':
                    raise ValueError('Paired-tool work is not the recorded CUSA action')
                radii.add(action['parameters']['radiusMm'])
                cells.update(history[index]['deltas']['removedCells'])
            if len(radii) != 1 or target_volume(cells) != stage['targetIncrementMm3']:
                raise ValueError('Paired-tool work configuration or target ledger differs')
            key = f'radius-{radii.pop():g}'
            configs[key] = configs.get(key, 0.)+target_volume(cells)
            previous_end = end
        baseline_cells = set()
        baseline = row['baseline']
        if not 0 <= baseline['nativeActions'] <= len(history):
            raise ValueError('Paired-tool access interval differs')
        for entry in history[:baseline['nativeActions']]:
            baseline_cells.update(entry['deltas']['removedCells'])
        if (set(baseline['removedCells']) != baseline_cells or
                baseline['targetRemovedMm3'] != target_volume(baseline_cells) or
                baseline['nonTargetRemovedMm3'] != len(baseline_cells)*scene.voxel_mm3-target_volume(baseline_cells)):
            raise ValueError('Paired-tool access removal differs')
        inserted, held, _ = _phase_state(history[:baseline['nativeActions']])
        _, _, used_retractor = _phase_state(history)
        if row['assisted'] is not used_retractor:
            raise ValueError('Paired-tool treatment label differs from native instrument history')
        if row['accessComplete'] and (prefix is None or not inserted['left'] or
                [a for a in actions[:baseline['nativeActions']] if a['slot']=='left'] != prefix or
                row['assisted'] and (not inserted['right'] or not held)):
            raise ValueError('Paired-tool access completion differs from source prefix or held state')
        target = target_volume(all_cells)
        other = len(all_cells)*scene.voxel_mm3-target
        view = SimpleNamespace(scene=scene, state=state, actions=actions, history=history,
            metrics=lambda:{'targetRemovedMm3':target,'nonTargetRemovedMm3':other})
        expected = assess_task(view, report['task'], zones, configs, exact_replay=True,
            baseline=baseline, admitted_strokes=len(row['workStages']), legacy_sectors=sectors)
        if expected != row['outcome']:
            raise ValueError('Paired-tool task gates differ from native effects')
        if row['status'] not in ('completed_native_program','failed_native_program') or (
                row['status']=='completed_native_program' and
                (row['failureReason'] is not None or not row['accessComplete'] or
                 not expected['cleanupComplete'])) or (
                row['status']=='failed_native_program' and not isinstance(row['failureReason'],str)):
            raise ValueError('Paired-tool program status differs from access and cleanup evidence')
        residual = max((float(np.linalg.norm(v)) for v in state.get('deformation',{}).values()),default=0.)
        if (row['finalMetrics']['targetRemovedMm3'] != target or
                row['finalMetrics']['nonTargetRemovedMm3'] != other or
                row['finalMetrics']['brainMaxDisplacementMm'] != residual or
                row['finalMetrics']['contactCellSeconds'] != sum(state['contact'].values()) or
                row['finalMetrics']['elapsedSeconds'] != state['elapsedSeconds'] or
                any(type(p['position']) is not int or not 0 <= p['position'] <= len(actions)
                    for p in row['phases'])):
            raise ValueError('Paired-tool display metrics or phase position differs')
        access_label = ('Access complete with tissue held' if held else 'Access complete') if row['accessComplete'] else 'Access stopped'
        if not {'label':access_label,'position':baseline['nativeActions']} in row['phases']:
            raise ValueError('Paired-tool access phase label differs from native history')
    return True
