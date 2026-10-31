"""Versioned native ownership of finite two-tool comparison studies."""
from __future__ import annotations

import copy
from collections import OrderedDict
from types import SimpleNamespace

from .core import semantic_digest, _copy_builtin_json, _JSON_COPY_FALLBACK
from .surgical_engine import SurgicalEngine
from .surgical_task_admission import bind_admission, validate_admission
from .surgical_two_tool import validate_report, materialize, _engine_owner_signature
from .surgical_program_cache import _json_size
from .surgical_interactive_workspace import _seal, _checked_seal

SCHEMA = 'resectionlab.two-tool-workspace.v1'
_VALIDATED = OrderedDict()
_LAST_BASELINE_STATS = {}


def baseline_cache_stats():
    """Last invocation diagnostics, never native owners or checkpoints."""
    return dict(_LAST_BASELINE_STATS)


class _BaselineScope:
    """Reuse verified native restores only inside one workspace proof.

    Root state/ancestry are detached for every consumer; admission is rebound
    and source verification still runs at the existing validation boundaries.
    The bounds account JSON state/history, separately from grid/guard owners.
    """
    def __init__(self, scene, *, check):
        self.scene = scene; self.check = check; self.roots = OrderedDict(); self.bytes = 0
        self.hits = self.misses = self.bypasses = self.evictions = self.peak_bytes = 0

    def restore(self, snapshot):
        if self.scene is None: raise ValueError('Native baseline validation scope is closed')
        self.check(); self.scene.require_cache_source()
        copied = _copy_builtin_json(snapshot)
        if copied is _JSON_COPY_FALLBACK:
            # Preserve native validation/dispatch for non-JSON representations;
            # a tuple/subclass must not borrow an equal list's cached authority.
            self.bypasses += 1; self.misses += 1
            return SurgicalEngine.restore(self.scene, snapshot, check=self.check)
        snapshot = copied; key = semantic_digest(snapshot)
        found = self.roots.get(key)
        if found is None:
            self.misses += 1
            root = SurgicalEngine.restore(self.scene, snapshot, check=self.check)
            # Occupancy depends on changing physical state; each consumer starts
            # cold. Only the existing pure geometry cache can share its owner.
            root._occupancy_cache = None
            size = _json_size({'state': root.state, 'actions': root.actions, 'history': root.history})
            if size is None or size > 16*1024**2:
                self.bypasses += 1
                return root
            owner = _engine_owner_signature(root)
            while self.roots and (len(self.roots) >= 2 or self.bytes+size > 16*1024**2):
                _, old = self.roots.popitem(last=False); self.bytes -= old[2]; self.evictions += 1
            self.check(); self.roots[key] = found = (root, owner, size); self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        else:
            self.hits += 1; self.roots.move_to_end(key)
        root, owner, _ = found
        if owner != _engine_owner_signature(root): raise ValueError('Native baseline validation owner changed')
        engine = root.clone(); engine._protection_contract = copy.deepcopy(root._protection_contract)
        engine._protected_ids = root._protected_ids.copy(); engine.by_label = copy.deepcopy(root.by_label)
        self.check(); return engine

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, traceback):
        global _LAST_BASELINE_STATS
        completed = False
        try:
            if exc_type is None:
                for root, owner, _ in self.roots.values():
                    if owner != _engine_owner_signature(root): raise ValueError('Native baseline validation owner changed')
                completed = True
        finally:
            values = {'hits': self.hits, 'restores': self.misses, 'bypasses': self.bypasses,
                'evictions': self.evictions, 'peakAccountedPayloadBytes': self.peak_bytes,
                'maxEntries': 2, 'maxPayloadBytes': 16*1024**2, 'completed': completed}
            self.roots.clear(); self.bytes = 0; self.scene = self.check = None
            _LAST_BASELINE_STATS = {**values, 'retainedAfterClose': 0}


def validate_workspace(scene, record, case_hash, *, check=lambda: None):
    with _BaselineScope(scene, check=check) as scope:
        return _validate_workspace(scene, record, case_hash, check=check, restore=scope.restore)


def _validate_workspace(scene, record, case_hash, *, check, restore):
    check()
    history = record.get('strategyComparisons', []); applications = record.get('strategyApplications', [])
    sensitivities = record.get('strategySensitivityStudies', [])
    if type(history) is not list or len(history) > 16 or type(applications) is not list or len(applications) > 64:
        raise ValueError('Two-tool study history exceeds its bounded schema')
    if type(sensitivities) is not list or len(sensitivities) > 8:
        raise ValueError('Execution sensitivity study history exceeds its bounded schema')
    if not history and not applications and not sensitivities: return history
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
            engine = restore(baseline)
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
            engine = restore(baseline); bind_admission(engine, item['report']['taskAdmission'])
            candidate, _, receipt = materialize(engine, item['report'], app['candidateId'], check=check)
            if candidate.snapshot() != snapshot or receipt != app['receipt']:
                raise ValueError('Two-tool installed state or receipt differs')
            check(); _VALIDATED[key] = True
    for revision, item in enumerate(sensitivities, 1):
        check(); _checked_seal(item, 'workspaceSeal')
        if (set(item) != {'schema', 'revision', 'comparisonRevision', 'study', 'workspaceSeal'}
                or item['schema'] != 'resectionlab.strategy-sensitivity-workspace.v1'
                or type(item['revision']) is not int or item['revision'] != revision
                or type(item['comparisonRevision']) is not int):
            raise ValueError('Execution sensitivity workspace schema differs')
        comparison = reports.get(item['comparisonRevision'])
        if comparison is None: raise ValueError('Execution sensitivity comparison is missing')
        key = (scene.identity, semantic_digest(comparison), semantic_digest(item))
        if key not in _VALIDATED:
            from .surgical_strategy_sensitivity import validate_sensitivity
            engine = restore(record['branches'][comparison['sourceBranchId']])
            bind_admission(engine, comparison['report']['taskAdmission'])
            validate_sensitivity(engine, comparison['report'], item['study'], check=check)
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


def install_sensitivity(engine, record, case_hash, comparison, study, *, check=lambda: None):
    current = current_report(engine, record, case_hash, comparison['report']['reportSeal'], check=check)
    if current != comparison: raise ValueError('Execution sensitivity source comparison changed')
    history = record.get('strategySensitivityStudies', [])
    if len(history) >= 8: raise ValueError('Execution sensitivity study limit reached; save this study')
    from .surgical_strategy_sensitivity import validate_sensitivity
    validate_sensitivity(engine, comparison['report'], study, check=check); check()
    item = _seal({'schema': 'resectionlab.strategy-sensitivity-workspace.v1',
        'revision': len(history)+1, 'comparisonRevision': comparison['revision'], 'study': copy.deepcopy(study)}, 'workspaceSeal')
    record.setdefault('strategySensitivityStudies', []).append(item)
    _VALIDATED[(engine.scene.identity, semantic_digest(comparison), semantic_digest(item))] = True
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
    sensitivity = next((s for s in reversed(record.get('strategySensitivityStudies', []))
        if s['comparisonRevision'] == item['revision']), None)
    return {**copy.deepcopy(report), 'revision': item['revision'], 'sourceBranchId': item['sourceBranchId'],
        'sourcePosition': report['prefixActions'], 'current': engine.state_id == report['startStateId'] and semantic_digest(engine.actions) == report['baselineActionHash'],
        'applications': [copy.deepcopy(a) for a in record.get('strategyApplications', [])],
        **({'executionSensitivity': copy.deepcopy(sensitivity)} if sensitivity else {})}


def protected_branches(record):
    return {i['sourceBranchId'] for i in record.get('strategyComparisons', [])} | {a['branchId'] for a in record.get('strategyApplications', [])}
