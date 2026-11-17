"""Read-only geometry from checked cranial source/history; no new planning.

Contact-limited motion uses recorded dwell positions. Unrecognized partials
leave a gap instead of drawing their requested, unexecuted suffix.
"""
from __future__ import annotations

import copy
from .core import semantic_digest

SCHEMA = 'resectionlab.recorded-cranial-geometry.v1'
MAX_DISPLAY_CELLS = 4096


def recorded_cranial_geometry(base_scene, scene, source_snapshot, snapshot, row, source_state_id):
    from .surgical_procedure_policy import _validated
    from .surgical_scene import original_target_memberships
    policy = _validated(scene)
    if policy is None:
        return None
    history = row['nativeHistory']
    slots = copy.deepcopy(source_snapshot['checkpoint']['state']['slots'])
    paths, omitted, accepted_actions = [], [], []
    for index, event in enumerate(history):
        action = event['action']; hand = action['slot']; slot = slots[hand]
        operation = action['operation']
        if operation == 'exchange':
            slot['instrument'] = action['instrument']
        accepted_actions.append({'actionIndex': index, 'operation': operation,
            'slot': hand, 'instrument': slot['instrument'], 'status': event['status'],
            'stateBefore': event['stateBefore'], 'stateAfter': event['stateAfter'],
            'elapsedSeconds': event.get('elapsedSeconds', 0.)})
        poses = action.get('path') or ([action['pose']] if action.get('pose') else [])
        if not poses:
            continue
        points = [list(slot['position'])] if slot.get('position') is not None else []
        if event['status'] == 'accepted' and event.get('completedFraction') == 1.:
            if operation == 'place':
                points = []  # Exterior placement is not a swept physical path.
            points.extend(copy.deepcopy(p['position']) for p in poses)
            slot['position'] = copy.deepcopy(poses[-1]['position'])
        elif event['status'] == 'partial':
            stops = [e['position'] for e in event.get('events', ())
                if e.get('kind') in ('cusa_contact_dwell', 'drill_contact_dwell') and 'position' in e]
            if len(poses) == 1 and stops:
                points.append(copy.deepcopy(stops[-1]))
                slot['position'] = copy.deepcopy(stops[-1])
            else:
                omitted.append(index); slot['position'] = None
                continue
        else:
            omitted.append(index); slot['position'] = None
            continue
        if operation != 'place' and len(points) >= 2:
            paths.append({'actionIndex': index, 'operation': operation, 'slot': hand,
                'instrument': slot['instrument'], 'status': event['status'],
                'stateBefore': event['stateBefore'], 'stateAfter': event['stateAfter'],
                'pointsRasMm': points})
    state = snapshot['checkpoint']['state']
    removed = set(state['removed'])
    dimensions = row['sequenceOutcome']['dimensions']
    structures = []
    for name, cells in sorted(policy['structureCells'].items()):
        ordered = sorted(cells); declaration = policy['record']['structures'][name]
        shown = ordered[:MAX_DISPLAY_CELLS]
        structures.append({'id': name, 'label': declaration.get('displayLabel', name),
            'kind': declaration['kind'], 'evidenceKind': 'generated-engineering',
            'reviewStatus': declaration.get('reviewStatus', 'unreviewed'),
            'provenance': declaration['provenance'], 'sourceCells': ordered,
            'pointsRasMm': scene.points(shown).tolist(), 'displayTruncated': len(shown) < len(ordered),
            'removedMm3': dimensions['structureRemovedMm3'][name]['cumulative'],
            'cutSourceMm3': dimensions['structureCutSourceMm3'][name]['cumulative']})
    target = sorted({int(c) for values in original_target_memberships(scene).values() for c in values})
    residual = [c for c in target if c not in removed]
    result = {'schema': SCHEMA, 'caseHash': base_scene.source.get('caseHash'),
        'sourceSceneId': base_scene.identity, 'sourceStateId': source_state_id,
        'rootSourceStateId': source_snapshot['stateId'], 'sceneId': scene.identity,
        'policyHash': policy['record']['policyHash'], 'branchId': row['branchId'],
        'finalStateId': snapshot['stateId'], 'nativeActions': len(snapshot['actions']),
        'actionHistoryHash': semantic_digest(history), 'paths': paths,
        'acceptedActions': accepted_actions,
        'acceptedStateIds': [source_snapshot['stateId']] + [e['stateAfter'] for e in history],
        'omittedPartialActions': omitted, 'structures': structures,
        'harmBounds': {key: copy.deepcopy(policy['record'][key]) for key in
            ('maximumNeuralRemovalMm3','maximumNeuralCutSourceMm3','maximumBoneRemovalMm3',
             'maximumNonbonePreparationRemovalMm3','maximumModeledElapsedSeconds',
             'maximumStructureRemovalMm3','maximumStructureCutSourceMm3')},
        'cleanup': {'debrisPresentMm3': float(state['debrisMm3']),
            'debrisRemovedMm3': float(state.get('debrisRemovedMm3', 0.)),
            'fluidPresentMm3': float(sum(state['fluid'].values()) + sum(s.get('reservoirMm3', 0.) for s in state['sources'])),
            'activeBleedingSources': sum(bool(s.get('bleedingActive', True)) for s in state['sources'])},
        'targetSourceCells': target, 'targetPointsRasMm': scene.points(target[:MAX_DISPLAY_CELLS]).tolist(),
        'residualSourceCells': residual, 'residualPointsRasMm': scene.points(residual[:MAX_DISPLAY_CELLS]).tolist(),
        'targetDisplayTruncated': len(target) > MAX_DISPLAY_CELLS,
        'scope': 'Recorded accepted instrument-tip paths and independently declared source memberships; not anatomical encounter or functional deficit. Placement jumps are omitted. Native replay supplies physical animation.'}
    result['geometrySeal'] = semantic_digest(result)
    return result
