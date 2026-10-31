"""Versioned native ownership of finite two-tool comparison studies."""
from __future__ import annotations

import copy
from collections import OrderedDict
from types import SimpleNamespace

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_task_admission import bind_admission, validate_admission
from .surgical_two_tool import validate_report, materialize
from .surgical_interactive_workspace import _seal, _checked_seal

SCHEMA = 'resectionlab.two-tool-workspace.v1'
_VALIDATED = OrderedDict()


def validate_workspace(scene, record, case_hash, *, check=lambda: None):
    check()
    history = record.get('strategyComparisons', []); applications = record.get('strategyApplications', [])
    if type(history) is not list or len(history) > 16 or type(applications) is not list or len(applications) > 64:
        raise ValueError('Two-tool study history exceeds its bounded schema')
    if not history and not applications: return history
    reports = {}
    for revision, item in enumerate(history, 1):
        check()
        _checked_seal(item, 'workspaceSeal')
        if set(item) != {'schema', 'revision', 'caseHash', 'sourceBranchId', 'sourceSnapshotHash', 'report', 'workspaceSeal'} or item['schema'] != SCHEMA or item['revision'] != revision or type(item['revision']) is not int or item['caseHash'] != case_hash:
            raise ValueError('Two-tool study identity/revision differs')
        baseline = record['branches'].get(item['sourceBranchId'])
        if baseline is None or semantic_digest(baseline) != item['sourceSnapshotHash'] or item['report']['taskAdmission'] != record.get('taskAdmission'):
            raise ValueError('Two-tool study source checkpoint differs')
        validate_admission(SimpleNamespace(scene=scene), item['report']['taskAdmission'], require_baseline=False)
        key = (scene.identity, semantic_digest(baseline), semantic_digest(item))
        if key not in _VALIDATED:
            engine = SurgicalEngine.restore(scene, baseline, check=check)
            bind_admission(engine, item['report']['taskAdmission']); validate_report(engine, item['report'], check=check)
            check()
            _VALIDATED[key] = True
        reports[revision] = item
    for app in applications:
        check()
        _checked_seal(app, 'applicationSeal')
        if set(app) != {'revision', 'reportSeal', 'candidateId', 'branchId', 'snapshotHash', 'receipt', 'applicationSeal'}:
            raise ValueError('Two-tool application schema differs')
        if type(app['revision']) is not int: raise ValueError('Two-tool application revision must be an integer')
        item = reports.get(app['revision']); snapshot = record['branches'].get(app['branchId'])
        if item is None or snapshot is None or app['reportSeal'] != item['report']['reportSeal'] or semantic_digest(snapshot) != app['snapshotHash']:
            raise ValueError('Two-tool application identity differs')
        baseline = record['branches'][item['sourceBranchId']]
        row = next((row for row in item['report']['publicPlan']['alternatives'] if row['id'] == app['candidateId']), None)
        if row is None or snapshot['actions'] != baseline['actions']+row['actionProgram']:
            raise ValueError('Two-tool installed program differs')
        key = (scene.identity, semantic_digest(item), semantic_digest(snapshot), semantic_digest(app))
        if key not in _VALIDATED:
            engine = SurgicalEngine.restore(scene, baseline, check=check); bind_admission(engine, item['report']['taskAdmission'])
            candidate, _, receipt = materialize(engine, item['report'], app['candidateId'], check=check)
            if candidate.snapshot() != snapshot or receipt != app['receipt']:
                raise ValueError('Two-tool installed state or receipt differs')
            check(); _VALIDATED[key] = True
    while len(_VALIDATED) > 64: _VALIDATED.popitem(last=False)
    return history


def install_report(engine, record, case_hash, report, *, check=lambda: None):
    history = validate_workspace(engine.scene, record, case_hash, check=check)
    if len(history) >= 16: raise ValueError('Two-tool comparison revision limit reached; save this study')
    validate_report(engine, report, check=check); check()
    branch = 'strategy-source-'+str(len(record['branches']))
    while branch in record['branches']: branch += '-next'
    snapshot = engine.snapshot()
    item = _seal({'schema': SCHEMA, 'revision': len(history)+1, 'caseHash': case_hash,
        'sourceBranchId': branch, 'sourceSnapshotHash': semantic_digest(snapshot), 'report': copy.deepcopy(report)}, 'workspaceSeal')
    record['branches'][branch] = snapshot
    record.setdefault('parents', {})[branch] = {'branchId': record['branchId'], 'position': record['position']}
    record.setdefault('timeline', {})[branch] = copy.deepcopy(engine.history)
    record.setdefault('strategyComparisons', []).append(item)
    _VALIDATED[(engine.scene.identity, semantic_digest(snapshot), semantic_digest(item))] = True
    return item


def current_report(engine, record, case_hash, seal, *, check=lambda: None):
    history = validate_workspace(engine.scene, record, case_hash, check=check)
    if not history or history[-1]['report']['reportSeal'] != seal:
        raise ValueError('Two-tool comparison changed; regenerate or inspect the current study')
    item = history[-1]
    validate_report(engine, item['report'], check=check)
    return item


def install_application(record, item, branch, engine, candidate_id, receipt):
    applications = record.setdefault('strategyApplications', [])
    if len(applications) >= 64: raise ValueError('Two-tool installed-branch limit reached; save this study')
    app = _seal({'revision': item['revision'], 'reportSeal': item['report']['reportSeal'],
        'candidateId': candidate_id, 'branchId': branch, 'snapshotHash': semantic_digest(engine.snapshot()),
        'receipt': copy.deepcopy(receipt)}, 'applicationSeal')
    applications.append(app)
    _VALIDATED[(engine.scene.identity, semantic_digest(item), app['snapshotHash'], semantic_digest(app))] = True


def planning_view(engine, record, case_hash, *, check=lambda: None):
    history = validate_workspace(engine.scene, record, case_hash, check=check)
    if not history: return None
    item = history[-1]; report = item['report']
    return {**copy.deepcopy(report), 'revision': item['revision'], 'sourceBranchId': item['sourceBranchId'],
        'sourcePosition': report['prefixActions'], 'current': engine.state_id == report['startStateId'] and semantic_digest(engine.actions) == report['baselineActionHash'],
        'applications': [copy.deepcopy(a) for a in record.get('strategyApplications', [])]}


def protected_branches(record):
    return {i['sourceBranchId'] for i in record.get('strategyComparisons', [])} | {a['branchId'] for a in record.get('strategyApplications', [])}
