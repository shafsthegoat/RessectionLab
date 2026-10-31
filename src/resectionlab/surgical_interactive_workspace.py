"""Native workspace ownership for source-bound interactive approach reports.

Generated geometry is analysis evidence. Only a separately checked public action
program can create an executable branch. Source checkpoints and installed plans
are retained so later edits fork rather than rewriting their evidence.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

from .core import semantic_digest

SCHEMA = 'resectionlab.interactive-approach-workspace.v1'
MAX_REVISIONS = 16
_APPLICATION_ADMISSIONS = OrderedDict()


def _application_key(scene, baseline, report, snapshot, application):
    return (scene.identity, semantic_digest(scene.properties), semantic_digest(baseline),
            report['reportSeal'], semantic_digest(snapshot), semantic_digest(application))


def _remember_application(key):
    _APPLICATION_ADMISSIONS[key] = True
    _APPLICATION_ADMISSIONS.move_to_end(key)
    while len(_APPLICATION_ADMISSIONS) > 64:
        _APPLICATION_ADMISSIONS.popitem(last=False)


def _seal(value, field):
    result = copy.deepcopy(value)
    result[field] = semantic_digest(result)
    return result


def _checked_seal(value, field):
    if not isinstance(value, dict):
        raise ValueError('A saved interactive planning record must be an object')
    body = {key: item for key, item in value.items() if key != field}
    if semantic_digest(body) != value.get(field):
        raise ValueError('Interactive planning workspace content changed')


def validate_workspace(scene, record, case_hash):
    from .surgical_interactive_planning import validate_saved_interactive_report
    history = record.get('interactiveApproaches', [])
    if not isinstance(history, list) or len(history) > MAX_REVISIONS:
        raise ValueError('Interactive planning history exceeds its revision budget')
    reports = {}
    for revision, item in enumerate(history, 1):
        _checked_seal(item, 'workspaceSeal')
        if (item.get('schema') != SCHEMA or type(item.get('revision')) is not int
                or item['revision'] != revision or item.get('caseHash') != case_hash):
            raise ValueError('Interactive planning source or revision differs')
        baseline = record['branches'].get(item.get('sourceBranchId'))
        if baseline is None or semantic_digest(baseline) != item.get('sourceSnapshotHash'):
            raise ValueError('Interactive planning source checkpoint changed')
        validate_saved_interactive_report(scene, baseline, item['report'])
        timeline = record.get('timeline', {}).get(item['sourceBranchId'])
        if timeline is not None and semantic_digest(timeline) != item['report'].get('baselineHistoryHash'):
            raise ValueError('Interactive planning source receipts changed')
        reports[revision] = item
    applications = record.get('interactiveApproachApplications', [])
    if not isinstance(applications, list) or len(applications) > 64:
        raise ValueError('Interactive planning applications exceed their budget')
    for application in applications:
        _checked_seal(application, 'applicationSeal')
        revision = application.get('revision')
        item = reports.get(revision) if type(revision) is int else None
        snapshot = record['branches'].get(application.get('branchId'))
        route = next((route for route in item['report']['routes']
                      if route['id'] == application.get('candidateId')), None) if item else None
        baseline = record['branches'][item['sourceBranchId']] if item else None
        receipt = application.get('receipt', {})
        if (item is None or application.get('reportSeal') != item['report']['reportSeal']
                or snapshot is None or semantic_digest(snapshot) != application.get('snapshotHash')
                or route is None or receipt.get('approachStateId') != snapshot.get('stateId')
                or snapshot.get('actions') != baseline['actions'] + route['actionProgram']
                or snapshot.get('sceneId') != baseline.get('sceneId')
                or snapshot.get('protectionContract') != baseline.get('protectionContract')
                or receipt.get('candidateId') != route['id']
                or receipt.get('reportSeal') != item['report']['reportSeal']
                or receipt.get('startStateId') != baseline['stateId']
                or receipt.get('actionCount') != len(route['actionProgram'])
                or receipt.get('withdrawalActionCount') != len(route['withdrawalProgram'])
                or receipt.get('exactPhysicsPreserved') is not True
                or receipt.get('publicAccessAccepted') is not True
                or receipt.get('publicWithdrawalAccepted') is not True
                or type(receipt.get('additionalTargetRemovedMm3')) not in (int, float)
                or type(receipt.get('additionalNonTargetRemovedMm3')) not in (int, float)
                or receipt['additionalTargetRemovedMm3'] != 0.
                or receipt['additionalNonTargetRemovedMm3'] != 0.
                or snapshot['checkpoint']['state']['removed'] != baseline['checkpoint']['state']['removed']):
            raise ValueError('Installed approach differs from its checked program or report')
        key = _application_key(scene, baseline, item['report'], snapshot, application)
        if key not in _APPLICATION_ADMISSIONS:
            # A report seal cannot authenticate a recorded withdrawal. Admit
            # unfamiliar persisted applications by unchanged native actions,
            # once per complete source/program/receipt content, never by path.
            from .surgical_engine import SurgicalEngine
            from .surgical_interactive_planning import materialize_interactive_approach
            source = SurgicalEngine.restore(scene, baseline)
            expected, expected_receipt = materialize_interactive_approach(source, item['report'], route['id'])
            if (semantic_digest(expected.snapshot()) != semantic_digest(snapshot)
                    or semantic_digest(expected_receipt) != semantic_digest(receipt)):
                raise ValueError('Installed approach native replay or withdrawal receipt differs')
            _remember_application(key)
    return history


def install_report(engine, record, case_hash, report):
    """Append one report and freeze its exact current native checkpoint."""
    from .surgical_interactive_planning import validate_interactive_report
    history = validate_workspace(engine.scene, record, case_hash)
    if len(history) >= MAX_REVISIONS:
        raise ValueError('Save this planning study before starting another revision (limit 16)')
    validate_interactive_report(engine, report)
    revision = len(history) + 1
    branch = 'approach-source-' + str(len(record['branches']))
    while branch in record['branches']:
        branch += '-next'
    snapshot = engine.snapshot()
    item = _seal({'schema': SCHEMA, 'revision': revision, 'caseHash': case_hash,
        'sourceBranchId': branch, 'sourceSnapshotHash': semantic_digest(snapshot),
        'report': copy.deepcopy(report)}, 'workspaceSeal')
    record['branches'][branch] = snapshot
    record.setdefault('parents', {})[branch] = {'branchId': record['branchId'], 'position': record['position']}
    record.setdefault('timeline', {})[branch] = copy.deepcopy(engine.history)
    record.setdefault('interactiveApproaches', []).append(item)
    return item


def current_report(engine, record, case_hash, expected_seal):
    history = validate_workspace(engine.scene, record, case_hash)
    if not history or history[-1]['report']['reportSeal'] != expected_seal:
        raise ValueError('Approach report changed; generate or inspect the current report')
    item = history[-1]
    from .surgical_interactive_planning import validate_interactive_report
    validate_interactive_report(engine, item['report'])
    return item


def install_application(record, item, branch, engine, candidate_id, receipt):
    applications = record.setdefault('interactiveApproachApplications', [])
    if len(applications) >= 64:
        raise ValueError('Save this planning study before installing more approaches (limit 64)')
    application = _seal({'revision': item['revision'], 'reportSeal': item['report']['reportSeal'],
        'candidateId': candidate_id, 'branchId': branch,
        'snapshotHash': semantic_digest(engine.snapshot()), 'receipt': copy.deepcopy(receipt)}, 'applicationSeal')
    applications.append(application)
    # The caller just materialized this exact private candidate through the
    # public engine. Reuse that admission until any bound content changes.
    baseline = record['branches'][item['sourceBranchId']]
    _remember_application(_application_key(engine.scene, baseline, item['report'],
        record['branches'][branch], application))


def planning_view(engine, record, case_hash):
    history = validate_workspace(engine.scene, record, case_hash)
    if not history:
        return None
    item = history[-1]
    baseline = record['branches'][item['sourceBranchId']]
    current = (engine.state_id == baseline['stateId']
               and semantic_digest(engine.actions) == semantic_digest(baseline['actions']))
    if current:
        from .surgical_interactive_planning import validate_interactive_report
        validate_interactive_report(engine, item['report'])
    return {**copy.deepcopy(item['report']), 'revision': item['revision'],
        'current': current, 'historyCount': len(history), 'sourceBranchId': item['sourceBranchId'],
        'sourcePosition': len(baseline['actions']),
        'applications': [copy.deepcopy(a) for a in record.get('interactiveApproachApplications', [])
                         if a['revision'] == item['revision']]}


def protected_branches(record):
    return ({item['sourceBranchId'] for item in record.get('interactiveApproaches', [])}
            | {item['branchId'] for item in record.get('interactiveApproachApplications', [])})
