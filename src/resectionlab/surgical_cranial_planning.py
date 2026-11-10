"""Source-qualified complete cranial sequences for the desktop comparison.

This adapter proposes only source-owned programs. Native search and literal
export replay decide executable progress; no report or qualification seal grants
patient treatment permission.
"""
from __future__ import annotations

import copy
import math
import re
import time
from types import SimpleNamespace

import numpy as np

from .core import semantic_digest, thaw_json
from .surgical_cranial_planning_roots import (
    FIELD, bind_branch_root, source_binding, validate_registry)
from .surgical_joint_cranial_forest import OpeningComparisonBudget, search_opening_forest

SCHEMA = 'resectionlab.cranial-strategy-comparison.v1'
_LIMIT_FIELDS = frozenset(('max_bone_mm3', 'max_preparation_nonbone_mm3',
    'max_work_nonbone_mm3', 'max_native_actions', 'max_public_calls',
    'max_wall_seconds', 'max_bone_contact_replans', 'max_tissue_contact_replans',
    'export_replay_reserve_calls'))


def _require(value, reason):
    if not value:
        raise ValueError('CRANIAL_PLANNING_' + reason)


def _block(code, message=None):
    return {'code': code, 'message': message or code.replace('_', ' ').capitalize() + '.'}


def _reviewed_evidence(rows):
    if type(rows) is not dict or not rows:
        return False
    proofs = []
    for name, proof in rows.items():
        if name == 'scope' and type(proof) is str and proof.strip():
            continue  # Native opening contracts include this explicit metadata.
        if (type(proof) is not dict or proof.get('status') != 'reviewed' or
                type(proof.get('provenance')) is not str or not proof['provenance'].strip() or
                type(proof.get('evidenceSha256')) is not str or
                re.fullmatch('[0-9a-f]{64}', proof['evidenceSha256']) is None):
            return False
        proofs.append(proof)
    return bool(proofs)


def _base_snapshot(base_scene, record):
    mapping = record.get(FIELD, {}).get('branchRoots', {})
    snapshots = [s for name, s in record['branches'].items() if mapping.get(name) is None and
        s.get('sceneId') == base_scene.identity and not s.get('actions')]
    _require(bool(snapshots) and len({s['stateId'] for s in snapshots}) == 1,
        'UNACTED_COMMON_SOURCE_REQUIRED')
    return snapshots[0]


def _protection_contains(base, candidate):
    """Existing source annotations/configuration cannot disappear in a root."""
    if base is None:
        return True
    if candidate is None:
        return False
    # Configuration/provenance cannot weaken. Additional protected membership
    # or structures are allowed; the source itself stays exact in the registry.
    if (base.get('schema') != candidate.get('schema') or
            base.get('sourceBinding') != candidate.get('sourceBinding') or
            base.get('config') != candidate.get('config') or
            candidate.get('clearanceMm', -1) < base.get('clearanceMm', 0) or
            not set(base.get('required', ())) <= set(candidate.get('required', ()))):
        return False
    extra = {a['structureId']: a for a in candidate.get('annotations', ())}
    return all(a['structureId'] in extra and
        {k: v for k, v in a.items() if k != 'cells'} ==
            {k: v for k, v in extra[a['structureId']].items() if k != 'cells'} and
        set(a['cells']) <= set(extra[a['structureId']]['cells']) for a in base.get('annotations', ()))


def _planning_errors(base_scene, root, base_snapshot):
    planning = thaw_json(root.get('planning'))
    if not isinstance(planning, dict):
        return [_block('SOURCE_PLANNING_INVENTORY_REQUIRED')]
    errors = []
    evidence_sets = (root.get('openingContract', {}).get('evidence'),
        root['sourceSnapshot'].get('cranialStagesContract', {}).get('evidence'))
    if not all(_reviewed_evidence(rows) for rows in evidence_sets):
        errors.append(_block('SOURCE_STAGE_OR_OPENING_EVIDENCE_REQUIRED'))
    q = planning.get('qualification', {})
    if (q.get('status') != 'reviewed' or q.get('sourceBinding') != source_binding(base_scene) or
            type(q.get('evidenceSha256')) is not str or
            re.fullmatch('[0-9a-f]{64}', q['evidenceSha256']) is None):
        errors.append(_block('REVIEWED_SOURCE_QUALIFICATION_REQUIRED'))
    explicit = q.get('blockers', [])
    if not isinstance(explicit, list) or any(type(x) is not dict or
            not isinstance(x.get('code'), str) or not isinstance(x.get('message'), str) for x in explicit):
        errors.append(_block('QUALIFICATION_BLOCKERS_INVALID'))
    else:
        errors.extend(copy.deepcopy(explicit))
    limits = planning.get('limits', {})
    if (type(limits) is not dict or set(limits) != _LIMIT_FIELDS or
            any(type(limits.get(k)) is not int or limits[k] < 1
                for k in ('max_native_actions', 'max_public_calls')) or
            any(type(limits.get(k)) not in (float, int) or not math.isfinite(limits[k]) or limits[k] < 0
                for k in ('max_bone_mm3', 'max_preparation_nonbone_mm3', 'max_work_nonbone_mm3')) or
            type(limits.get('max_wall_seconds')) not in (float, int) or
            not math.isfinite(limits.get('max_wall_seconds', float('nan'))) or limits.get('max_wall_seconds', 0) <= 0 or
            any(type(limits.get(k)) is not int or not 0 <= limits[k] <= 8
                for k in ('max_bone_contact_replans', 'max_tissue_contact_replans')) or
            type(limits.get('export_replay_reserve_calls')) is not int or
            not 0 < limits.get('export_replay_reserve_calls', 0) < limits.get('max_public_calls', 0)):
        errors.append(_block('EXPLICIT_SHARED_NATIVE_AND_EXPORT_BUDGET_REQUIRED'))
    if not _protection_contains(base_snapshot.get('protectionContract'), root['sourceSnapshot'].get('protectionContract')):
        errors.append(_block('COMMON_SOURCE_PROTECTION_MUST_BE_PRESERVED'))
    opening = planning.get('openingSkill', {})
    if opening.get('kind') not in ('opening', 'milling_opening'):
        errors.append(_block('SOURCE_OPENING_SKILL_REQUIRED'))
    else:
        from .surgical_joint_cranial_search import _proposal
        try:
            _proposal({**opening, 'id': opening.get('id', 'source-opening'), 'sourceStateId': 'preflight'}, 'preflight')
        except (ValueError, TypeError, KeyError):
            errors.append(_block('SOURCE_OPENING_PROGRAM_INVALID'))
    if not planning.get('preparationActions'):
        errors.append(_block('SOURCE_PREPARATION_PROGRAM_REQUIRED'))
    routes = planning.get('workRoutes', [])
    if (not routes or any(type(r.get('id')) is not str or not r['id'] or
            type(r.get('motion')) is not dict or r['motion'].get('operation') != 'apply' or
            type(r['motion'].get('pose')) is not dict for r in routes) or
            len({r['id'] for r in routes}) != len(routes)):
        errors.append(_block('SOURCE_CUSA_WORK_ROUTES_REQUIRED'))
    else:
        for route in routes:
            motion = route['motion'];pose = motion['pose']
            if (motion.get('slot') not in ('left', 'right') or
                    type(motion.get('duration')) not in (int, float) or
                    not math.isfinite(motion['duration']) or not 0 < motion['duration'] <= 30 or
                    any(type(pose.get(k)) not in (list, tuple) or len(pose[k]) != n or
                        any(type(x) not in (int, float) or not math.isfinite(x) for x in pose[k])
                        for k, n in (('position', 3), ('quaternion', 4))) or
                    np.linalg.norm(pose.get('quaternion', [0., 0., 0., 0.])) < 1e-12):
                errors.append(_block('SOURCE_CUSA_WORK_MOTION_INVALID'))
                break
    return errors


def cranial_planning_availability(base_scene, record):
    """Presence never falls back to an exposed-brain planner or invents a root."""
    if FIELD not in record and not any(r.get('material') == 'bone' for r in base_scene.regions) and not any(
            name in base_scene.properties for name in ('cranialAccessV1', 'cranialMillingV1')):
        return None
    current = record.get('branches', {}).get(record.get('branchId'), {})
    result = {'present': True, 'available': False, 'roots': [],
        'sourceSceneId': base_scene.identity, 'sourceStateId': current.get('stateId', base_scene.identity), 'blockers': []}
    if FIELD not in record:
        result['blockers'] = [_block('QUALIFIED_SOURCE_OPENING_ROOT_REQUIRED')]
        return result
    try:
        validate_registry(base_scene, thaw_json(record[FIELD]))
        base = _base_snapshot(base_scene, record)
        result['sourceStateId'] = base['stateId']
        for root in record[FIELD]['roots']:
            errors = _planning_errors(base_scene, root, base)
            result['roots'].append({'rootId': root['id'], 'label': root['label'],
                'openingMethod': 'milling' if root['openingProperty'] == 'cranialMillingV1' else 'craniotomy',
                'available': not errors, 'blockers': errors, 'sceneId': root['sceneId'],
                'sourceStateId': root['sourceSnapshot']['stateId']})
        result['available'] = any(r['available'] for r in result['roots'])
        if not result['available']:
            result['blockers'] = [_block('NO_SOURCE_QUALIFIED_COMPLETE_SEQUENCE_ROOT')]
        if (current.get('sceneId') != base_scene.identity or current.get('actions') or
                record.get('position') != 0 or record[FIELD]['branchRoots'].get(record.get('branchId')) is not None):
            result['available'] = False
            result['blockers'].append(_block('RETURN_TO_UNACTED_COMMON_SOURCE'))
    except (ValueError, KeyError, TypeError) as error:
        result['blockers'] = [_block('SOURCE_OPENING_REGISTRY_INVALID', str(error))]
    return result


def _proposer(planning):
    inventory = copy.deepcopy(planning)
    def propose(engine):
        from .surgical_joint_cranial_skills import _readiness
        from .surgical_scene import original_target_memberships
        ready = _readiness(engine) or {}
        if not ready.get('bonePrefixVerified'):
            value = copy.deepcopy(inventory['openingSkill'])
            value.update(id=value.get('id', 'source-opening'), sourceStateId=engine.state_id)
            return [value]
        if not ready.get('duraCleared'):
            return [{'id': 'source-membrane-preparation', 'sourceStateId': engine.state_id,
                'kind': 'preparation', 'actions': copy.deepcopy(inventory['preparationActions'])}]
        membership = original_target_memberships(engine.scene)
        if not membership:
            return []
        owned = {r['id'] for r in engine.scene.regions if r.get('target')}
        remaining = sorted({int(c) for cells in membership.values() for c in cells}
            - set(engine.state['removed']))
        remaining = [c for c in remaining if int(engine.scene.labels.flat[c]) in owned]
        if not remaining:
            return []
        points = engine.scene.points(remaining)
        def rank(route):
            point = np.asarray(route['motion']['pose']['position'], float)
            return float(np.min(np.sum((points-point)**2, axis=1))), route['id']
        return [{'id': 'source-work-' + row['id'], 'sourceStateId': engine.state_id,
            'kind': 'cusa', 'motion': copy.deepcopy(row['motion'])}
            for row in sorted(inventory['workRoutes'], key=rank)[:6]]
    return propose


def compare_cranial_strategies(base_scene, record, *, check=lambda: None,
        progress=lambda *args: None, planning_horizon=5, beam_width=3, started_at=None):
    from .surgical_engine import SurgicalEngine
    from .surgical_joint_cranial_exit import joint_exit_complete
    if started_at is None:
        started_at = time.monotonic()
    available = cranial_planning_availability(base_scene, record)
    _require(available is not None and available['available'], 'QUALIFIED_COMPLETE_SEQUENCE_ROOT_REQUIRED')
    _require(type(planning_horizon) is int and 1 <= planning_horizon <= 5 and
        type(beam_width) is int and 2 <= beam_width <= 4, 'BOUNDED_HORIZON_AND_BEAM_REQUIRED')
    check()
    scenes = validate_registry(base_scene, thaw_json(record[FIELD]))
    rows = [r for r in record[FIELD]['roots'] if next(x for x in available['roots'] if x['rootId'] == r['id'])['available']]
    common = rows[0]['planning']['limits']
    _require(all(r['planning']['limits'] == common for r in rows), 'EQUAL_COMPARISON_LIMITS_REQUIRED')
    budget = OpeningComparisonBudget(common['max_public_calls'], common['max_wall_seconds'],
        export_reserve=common['export_replay_reserve_calls'], check=check, started_at=started_at)
    roots = []
    for row in rows:
        budget.check()
        roots.append({'id': row['id'], 'label': row['label'],
            'engine': SurgicalEngine.restore(scenes[row['id']], thaw_json(row['sourceSnapshot']), use_cache=False, check=budget.check),
            'limits': {k: v for k, v in common.items() if k != 'export_replay_reserve_calls'},
            'propose': _proposer(row['planning']), 'outside_poses': copy.deepcopy(row['planning']['outsidePoses'])})
    result = search_opening_forest(roots, budget=budget, horizon=planning_horizon,
        beam_width=beam_width, progress=progress)
    branches, alternatives = [], []
    for item in result['branches']:
        engine, node = item['engine'], item['node']
        branch_id = 'cranial-' + item['rootId'] + '-' + engine.state_id.split(':')[-1][:16]
        _require(branch_id not in record['branches'], 'BRANCH_ALREADY_PRESENT')
        goal = copy.deepcopy(node.goal_assessment)
        goal.update(originalTargetVolumeMm3=goal['originalTargetCells']*engine.scene.voxel_mm3,
            residualOriginalTargetMm3=goal['residualOriginalTargetCells']*engine.scene.voxel_mm3)
        row = {'rootId': item['rootId'], 'sceneId': engine.scene.identity, 'branchId': branch_id,
            'finalStateId': engine.state_id, 'nativeActions': len(engine.actions),
            'label': next(r['label'] for r in rows if r['id'] == item['rootId']),
            'completedFrozenTask': goal['completedFrozenTask'], 'goalAssessment': goal,
            'ledger': copy.deepcopy(node.ledger), 'sequenceOutcome': copy.deepcopy(node.sequence_outcome),
            'wholeExitComplete': joint_exit_complete(engine),
            'stopReason': 'COMPLETED_FROZEN_ENGINEERING_TASK' if goal['completedFrozenTask'] else
                '; '.join(goal['reasons']) or 'CHECKED_PROGRESS_ONLY',
            'snapshotHash': semantic_digest(engine.snapshot()), 'nativeHistory': copy.deepcopy(engine.history),
            'admittedPatientTask': False}
        branches.append((engine, row)); alternatives.append(row)
    report = {'schema': SCHEMA, 'sourceSceneId': base_scene.identity,
        'sourceBranchId': record['branchId'],
        'sourceStateId': available['sourceStateId'], 'sourceBinding': source_binding(base_scene),
        'rootSeals': {r['id']: r['rootSeal'] for r in rows}, 'decision': result['decision'],
        'selectedBranchId': next((r['branchId'] for r in alternatives if r['completedFrozenTask']), None),
        'alternatives': alternatives, 'failures': copy.deepcopy(result['failures'])[:128],
        'limitations': ['Opening and preparation are charged over the whole sequence; skull clearance is not target treatment.',
            'Source qualification, registration and clinical consequences require independent evidence. No patient-task admission.'],
        'cost': result['cost'], 'admittedPatientTask': False}
    report['comparisonSeal'] = semantic_digest(report)
    budget.check()
    return {'report': report, 'branches': branches, 'budget': budget}


def install_cranial_comparison(record, comparison, base_scene):
    """Stage publication atomically; retain the current common source cursor."""
    comparison['budget'].check()
    _require(len(comparison['branches']) <= 3, 'MAXIMUM_THREE_EXPORTED_BRANCHES')
    _require([row for _, row in comparison['branches']] == comparison['report']['alternatives'],
        'EXPORTED_REPORT_BRANCHES_DIFFER')
    staged = copy.deepcopy(record)
    for engine, row in comparison['branches']:
        comparison['budget'].check()
        identity = row['branchId']
        _require(identity not in staged['branches'], 'BRANCH_ID_COLLISION')
        staged['branches'][identity] = engine.snapshot()
        staged.setdefault('parents', {})[identity] = None
        staged.setdefault('timeline', {})[identity] = copy.deepcopy(engine.history)
        bind_branch_root(base_scene, staged, identity, row['rootId'])
    staged['cranialPlanning'] = copy.deepcopy(comparison['report'])
    validate_cranial_report(base_scene, staged, staged['cranialPlanning'])
    comparison['budget'].check()
    record.clear(); record.update(staged)


def _snapshot_quantities(scene, snapshot, source, history):
    """Read saved native geometry, never execute or restore an acted branch."""
    from .surgical_cranial_stages import source_partition_sets
    from .surgical_scene import original_target_memberships
    state = snapshot['checkpoint']['state']
    removed = state['removed']
    _require(type(removed) is list and len(set(removed)) == len(removed) and
        all(type(c) is int and 0 <= c < scene.labels.size for c in removed), 'NATIVE_REMOVED_SOURCE_IDS_INVALID')
    target = set(int(c) for values in original_target_memberships(scene).values() for c in values)
    owned = {r['id'] for r in scene.regions if r.get('target')}
    physical = target & set(removed)
    credited = {c for c in physical if int(scene.labels.flat[c]) in owned}
    points = scene.points(sorted(target));codes = (points >= points.mean(axis=0)).astype(np.int8) @ np.array([1, 2, 4])
    zones = [set(c for c, code in zip(sorted(target), codes) if code == i) for i in range(8)]
    parts = source_partition_sets(source._cranial_stages);cells = set(removed)
    classified_target = {c for c in cells if int(scene.labels.flat[c]) in owned}
    v = scene.voxel_mm3
    ledger = {'boneRemovedMm3': len(cells & parts['bone'])*v,
        'preparationNonboneRemovedMm3': len(cells & parts['preparation'])*v,
        'workNonTargetRemovedMm3': len((cells & parts['work'])-classified_target)*v,
        'wholeNonTargetRemovedMm3': len(cells-classified_target)*v,
        'targetRemovedMm3': len(classified_target)*v, 'modeledElapsedSeconds': state['elapsedSeconds'],
        'uniqueRemovedSourceCells': len(cells),
        'unauthorizedRemovedSourceCells': len(cells-parts['bone']-parts['preparation']-parts['work'])}
    goal = {'originalTargetCells': len(target), 'removedOriginalTargetCells': len(physical),
        'targetBasis': 'original_annotation' if scene._original_target_proof is not None else 'owned_target_labels_fallback',
        'ownedOriginalTargetCells': sum(int(scene.labels.flat[c]) in owned for c in target),
        'creditedOriginalTargetCells': len(credited), 'residualOriginalTargetCells': len(target-physical),
        'originalTargetVolumeMm3': len(target)*v, 'residualOriginalTargetMm3': len(target-physical)*v,
        'originalTargetRemovedMm3': len(physical)*v, 'originalTargetTreatmentMm3': len(credited)*v,
        'originalTargetFractionRemoved': len(physical)/len(target), 'originalTargetTreatmentFraction': len(credited)/len(target),
        'zoneRemovedCells': [len(physical & z) for z in zones],
        'zoneTreatmentCreditCells': [len(credited & z) for z in zones],
        'zoneResidualCells': [len(z-physical) for z in zones],
        'zonesReachingRequiredOriginalFraction': sum(len(credited & z)/len(target)+1e-12 >= .05 for z in zones)}
    # A read-only endpoint view invokes only stock-body exterior geometry. It
    # cannot be used as a native engine or grant a stage permission.
    from .surgical_engine import SurgicalEngine
    from .surgical_joint_cranial_exit import joint_exit_complete
    view = SimpleNamespace(scene=scene, state=state, actions=snapshot['actions'],
        _cranial_access=source._cranial_access)
    view.parameters = lambda slot: SurgicalEngine.parameters(view, slot)
    return goal, ledger, joint_exit_complete(view)


def validate_cranial_report(base_scene, record, report):
    """Cold/display checks recompute target/zone and B/P/W facts from geometry."""
    from .surgical_engine import SurgicalEngine
    from .surgical_membrane_transport import clearance
    report = thaw_json(report)
    _require(report.get('schema') == SCHEMA and report.get('comparisonSeal') == semantic_digest(
        {k: v for k, v in report.items() if k != 'comparisonSeal'}) and
        report.get('sourceSceneId') == base_scene.identity and report.get('sourceBinding') == source_binding(base_scene) and
        report.get('admittedPatientTask') is False, 'REPORT_SOURCE_OR_SEAL_CHANGED')
    _require(report.get('sourceStateId') == _base_snapshot(base_scene, record)['stateId'],
        'REPORT_COMMON_SOURCE_CHANGED')
    original = record['branches'].get(report.get('sourceBranchId'), {})
    _require(original.get('sceneId') == base_scene.identity and
        original.get('stateId') == report['sourceStateId'] and not original.get('actions') and
        record[FIELD]['branchRoots'].get(report['sourceBranchId']) is None,
        'REPORT_COMMON_SOURCE_BRANCH_CHANGED')
    registry = thaw_json(record[FIELD]);scenes = validate_registry(base_scene, registry)
    roots = {r['id']: r for r in registry['roots']}
    _require(all(roots[k]['rootSeal'] == v for k, v in report['rootSeals'].items()), 'REPORT_ROOT_CHANGED')
    alternatives = report['alternatives'];seen = set()
    _require(type(alternatives) is list and len(alternatives) <= 3, 'REPORT_BRANCH_BOUND')
    for row in alternatives:
        identity, root_id = row['branchId'], row['rootId']
        _require(identity not in seen and registry['branchRoots'].get(identity) == root_id, 'REPORT_ROOT_BRANCH_CHANGED')
        seen.add(identity);snapshot = thaw_json(record['branches'][identity]);scene = scenes[root_id]
        _require(row['sceneId'] == snapshot['sceneId'] == scene.identity and
            row['finalStateId'] == snapshot['stateId'] == snapshot['checkpoint']['stateId'] and
            snapshot['checkpoint']['position'] == row['nativeActions'] == len(snapshot['actions']) and
            row['snapshotHash'] == semantic_digest(snapshot), 'REPORT_NATIVE_ENDPOINT_CHANGED')
        source = SurgicalEngine.restore(scene, roots[root_id]['sourceSnapshot'], use_cache=False)
        state_hash = semantic_digest({'scene': scene.identity, 'state': snapshot['checkpoint']['state'],
            **({'protectionContract': snapshot['protectionContract']['contentHash']} if snapshot.get('protectionContract') else {}),
            'cranialStagesContract': source._cranial_stages.content_hash})
        _require(state_hash == snapshot['stateId'], 'REPORT_PHYSICAL_STATE_CHANGED')
        history = row['nativeHistory'];previous = source.state_id
        _require(len(history) == len(snapshot['actions']), 'REPORT_HISTORY_CHANGED')
        for action, event in zip(snapshot['actions'], history):
            _require(event['action'] == action and event['stateBefore'] == previous and
                event['status'] in ('accepted', 'partial'), 'REPORT_HISTORY_CHAIN_CHANGED')
            previous = event['stateAfter']
        _require(previous == snapshot['stateId'], 'REPORT_HISTORY_ENDPOINT_CHANGED')
        goal, ledger, outside = _snapshot_quantities(scene, snapshot, source, history)
        _require(all(row['goalAssessment'].get(k) == v for k, v in goal.items()) and
            all(row['ledger'].get(k) == v for k, v in ledger.items()), 'REPORT_FULL_TARGET_OR_SOURCE_LEDGER_CHANGED')
        caps = roots[root_id]['planning']['limits']
        within = (ledger['boneRemovedMm3'] <= caps['max_bone_mm3']+1e-9 and
            ledger['preparationNonboneRemovedMm3'] <= caps['max_preparation_nonbone_mm3']+1e-9 and
            ledger['workNonTargetRemovedMm3'] <= caps['max_work_nonbone_mm3']+1e-9 and
            abs(ledger['wholeNonTargetRemovedMm3']-sum(ledger[k] for k in
                ('boneRemovedMm3','preparationNonboneRemovedMm3','workNonTargetRemovedMm3'))) <= 1e-9 and
            row['nativeActions'] <= caps['max_native_actions'] and
            row['ledger']['spentPublicCalls'] <= caps['max_public_calls'] and
            row['ledger']['wallSecondsSinceMeterCreation'] < caps['max_wall_seconds'] and
            not row['ledger']['unauthorizedRemovedSourceCells'])
        _require(row['ledger']['withinLimits'] == within, 'REPORT_CAP_OR_REFUSAL_CHANGED')
        outcome = row['sequenceOutcome']
        _require(outcome['sceneId'] == scene.identity and outcome['stateId'] == snapshot['stateId'] and
            outcome['baselineStateId'] == source.state_id and row['wholeExitComplete'] == outside and
            outcome['withdrawalComplete'] == (not snapshot['checkpoint']['state']['holds'] and
                not any(s['active'] or s['inserted'] or s['grip'] for s in snapshot['checkpoint']['state']['slots'].values())),
            'REPORT_SEQUENCE_OR_EXIT_CHANGED')
        dimensions = outcome['dimensions']
        _require(dimensions['targetRemovedMm3']['cumulative'] == ledger['targetRemovedMm3'] and
            dimensions['nonTargetRemovedMm3']['cumulative'] == ledger['wholeNonTargetRemovedMm3'], 'REPORT_SEQUENCE_COUNTS_CHANGED')
        view = SimpleNamespace(scene=scene, state=snapshot['checkpoint']['state'],
            actions=snapshot['actions'], history=history)
        dura = clearance(view, source._cranial_stages)['cleared']
        completed = goal['originalTargetTreatmentFraction']+1e-12 >= .5 and goal['zonesReachingRequiredOriginalFraction'] >= 3 and outside and dura and row['ledger']['withinLimits']
        reasons = []
        if goal['originalTargetTreatmentFraction']+1e-12 < .5:
            reasons.append('ORIGINAL_TARGET_FRACTION_BELOW_GOAL')
        if goal['zonesReachingRequiredOriginalFraction'] < 3:
            reasons.append('ORIGINAL_SPATIAL_PROGRESS_BELOW_GOAL')
        if not (outside and dura):
            reasons.append('NATIVE_WORK_AND_WHOLE_EXIT_REQUIRED')
        if not within:
            reasons.extend(row['ledger']['refusalReasons'])
        _require(row['completedFrozenTask'] == row['goalAssessment']['completedFrozenTask'] == completed and
            row['goalAssessment']['reasons'] == reasons and row.get('admittedPatientTask') is False,
            'REPORT_COMPLETION_CHANGED')
    selected = report['selectedBranchId']
    _require(selected is None or any(r['branchId'] == selected and r['completedFrozenTask'] for r in alternatives), 'REPORT_SELECTION_CHANGED')
    complete = next((r['branchId'] for r in alternatives if r['completedFrozenTask']), None)
    _require(selected == complete and report['decision'] == ('COMPLETED_FROZEN_TASK' if complete else
        'CHECKED_PROGRESS_ONLY' if alternatives else 'STOP'), 'REPORT_DECISION_CHANGED')
    cost = report['cost']
    _require(cost['physicalCalls'] == cost['previewCalls']+cost['directStepCalls'] and
        0 <= cost['physicalCalls'] <= cost['maxPhysicalCalls'] and
        sum(r['physicalCalls'] for r in cost['roots'].values()) == cost['physicalCalls'], 'REPORT_SHARED_COST_CHANGED')
    _require(all(len(r['trace']) == r['physicalCalls'] and
        sum(x['kind']=='preview' for x in r['trace']) == r['previewCalls'] and
        sum(x['kind']=='step' for x in r['trace']) == r['directStepCalls'] for r in cost['roots'].values()),
        'REPORT_CALL_TRACE_CHANGED')
    return {'verified': True, 'alternatives': len(alternatives), 'admittedPatientTask': False}


def cranial_summary(report):
    """Public DTO preserves every comparison row; private replay logs stay saved."""
    public = {k: copy.deepcopy(report[k]) for k in ('schema', 'comparisonSeal', 'sourceSceneId',
        'sourceStateId', 'sourceBranchId', 'decision', 'selectedBranchId', 'failures', 'limitations')}
    public['alternatives'] = [{k: copy.deepcopy(v) for k, v in row.items()
        if k not in ('snapshotHash', 'nativeHistory')} for row in report['alternatives']]
    public['cost'] = copy.deepcopy(report['cost'])
    public.update(admittedPatientTask=False, clinicalValidation='none')
    return public
