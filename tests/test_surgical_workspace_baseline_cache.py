"""Native workspace baseline reuse, isolated generated sources and consumers."""
import copy
import gc
import weakref

import numpy as np
import pytest
from scipy.spatial import cKDTree

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_protection import ProtectionSet, ProtectionAnnotation, GuardConfig
from resectionlab import surgical_strategy_workspace as workspace
from resectionlab import surgical_two_tool as planning
from test_surgical_two_tool import qualified, comparison, prepared


def check():
    pass


class Cancelled(RuntimeError):
    pass


def scene_copy(scene):
    return SurgicalScene(scene.labels, scene.affine, scene.regions, scene.source, scene.properties)


def count_restores(monkeypatch):
    original = SurgicalEngine.restore.__func__
    calls = []
    def counted(cls, scene, snapshot, **kwargs):
        calls.append((id(snapshot), type(snapshot), type(snapshot['actions'])))
        return original(cls, scene, snapshot, **kwargs)
    monkeypatch.setattr(SurgicalEngine, 'restore', classmethod(counted))
    return calls


@pytest.fixture(scope='module')
def guarded(qualified):
    original = qualified[1]; scene = scene_copy(original.scene)
    mask = np.zeros(scene.shape, bool); mask[0, 0, 0] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
        provenance='Generated owner invalidation control')], clearance_mm=0.)
    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig(require_critical_coverage=False))
    for action in original.actions:
        assert engine.step(copy.deepcopy(action))['status'] != 'rejected'
    return engine


def variants(engine):
    result = [engine.snapshot()]
    for duration in (.01, .02):
        candidate = engine.clone()
        assert candidate.step({'slot': 'left', 'operation': 'inspect', 'duration': duration})['status'] == 'accepted'
        result.append(candidate.snapshot())
    return result


def test_exact_snapshot_content_reuses_one_verified_restore(qualified, monkeypatch):
    engine = qualified[1]; snapshot = engine.snapshot(); scene = scene_copy(engine.scene)
    expected = SurgicalEngine.restore(scene, snapshot)
    calls = count_restores(monkeypatch)
    with workspace._BaselineScope(scene, check=check) as scope:
        first = scope.restore(snapshot)
        second = scope.restore(copy.deepcopy(snapshot))
        assert first.snapshot() == second.snapshot() == expected.snapshot()
        assert first.actions == second.actions == expected.actions
        assert first.history == second.history == expected.history
        assert first is not second and first is not next(iter(scope.roots.values()))[0]
        assert len(calls) == 1 and scope.hits == 1 and scope.misses == 1
        # Extra historical metadata is ignored by the native reader, but does
        # not confer another snapshot's cached restore authority.
        changed = copy.deepcopy(snapshot); changed['ignoredHistoricalMetadata'] = True
        assert scope.restore(changed).snapshot() == expected.snapshot()
        assert len(calls) == 2
    assert scope.roots == {} and scope.scene is None and scope.check is None
    assert workspace.baseline_cache_stats()['retainedAfterClose'] == 0


@pytest.mark.parametrize('field', ['state-id', 'checkpoint-state', 'action-log', 'scene-id'])
def test_different_or_forged_snapshot_cannot_borrow_cached_authority(qualified, monkeypatch, field):
    engine = qualified[1]; snapshot = engine.snapshot(); calls = count_restores(monkeypatch)
    with workspace._BaselineScope(scene_copy(engine.scene), check=check) as scope:
        original = scope.restore(snapshot)
        bad = copy.deepcopy(snapshot)
        if field == 'state-id': bad['stateId'] = 'sha256:invented'
        elif field == 'checkpoint-state': bad['checkpoint']['state']['elapsedSeconds'] += .25
        elif field == 'action-log': bad['actions'][0]['duration'] += .01
        else: bad['sceneId'] = 'sha256:other-source'
        with pytest.raises(ValueError): scope.restore(bad)
        assert len(calls) == 2 and len(scope.roots) == 1
        assert scope.restore(snapshot).snapshot() == original.snapshot()
        assert scope.hits == 1


def test_every_consumer_detaches_body_ancestry_contract_and_region_records(guarded):
    snapshot = guarded.snapshot(); scene = scene_copy(guarded.scene)
    with workspace._BaselineScope(scene, check=check) as scope:
        first = scope.restore(snapshot)
        root = next(iter(scope.roots.values()))[0]
        expected_history = copy.deepcopy(root.history)
        label = next(iter(first.by_label))
        first.state['elapsedSeconds'] += 1.
        first.actions[0]['duration'] += 1.
        first.history[0]['consumer-canary'] = {'mutable': []}
        first._protection_contract['config']['critical_max_displacement_mm'] = 99.
        first._protected_ids = frozenset()
        first.by_label[label]['name'] = 'consumer-only-region'
        first.by_label[65535] = {'id': 65535}
        first.occupancy()
        assert root._occupancy_cache is None
        second = scope.restore(snapshot)
        assert second.snapshot() == snapshot and second.history == expected_history
        assert second.by_label[label]['name'] != 'consumer-only-region' and 65535 not in second.by_label
        assert second._protected_ids == root._protected_ids and second._protected_ids
        assert second._occupancy_cache is None
        assert second.scene is first.scene is root.scene
        assert second.grid is first.grid is root.grid
        assert second._cell_cover_cache is first._cell_cover_cache is root._cell_cover_cache
        first.cells([0., 0., 0.], [0., 0., 1.], .1)
        assert scope.restore(snapshot).snapshot() == snapshot


@pytest.mark.parametrize('representation', ['tuple-actions', 'mapping-subclass', 'private-owner'])
def test_non_json_dispatch_is_unmodified_and_never_deepcopies_private_owners(qualified, monkeypatch, representation):
    class PrivateOwner:
        def __deepcopy__(self, memo): pytest.fail('Private owner was deep-copied by baseline reuse')
    class SnapshotSubclass(dict):
        def __deepcopy__(self, memo): pytest.fail('Snapshot subclass was coerced/copied')
    engine = qualified[1]; snapshot = engine.snapshot(); non_json = copy.deepcopy(snapshot)
    owner_ref = None
    if representation == 'tuple-actions': non_json['actions'] = tuple(non_json['actions'])
    elif representation == 'mapping-subclass': non_json = SnapshotSubclass(non_json)
    else:
        owner = PrivateOwner(); owner_ref = weakref.ref(owner)
        non_json['unusedPrivateEvaluator'] = owner
    calls = count_restores(monkeypatch)
    with workspace._BaselineScope(scene_copy(engine.scene), check=check) as scope:
        expected = scope.restore(snapshot).snapshot()
        actual = scope.restore(non_json)
        assert actual.snapshot() == expected
        assert calls[-1] == (id(non_json), type(non_json), type(non_json['actions']))
        assert len(calls) == 2 and scope.bypasses == 1 and scope.hits == 0
        assert len(scope.roots) == 1
        if owner_ref is not None:
            del non_json['unusedPrivateEvaluator'], owner
            gc.collect()
            assert owner_ref() is None


@pytest.mark.parametrize('fence', ['hit', 'exit'])
@pytest.mark.parametrize('field', ['body', 'actions', 'history', 'contract', 'by-label', 'protected-ids',
                                  'guard', 'guard-tree', 'guard-scene', 'grid', 'source', 'scene-owner', 'catalog'])
def test_mutated_native_owner_is_rejected_before_hit_or_successful_exit(guarded, monkeypatch, field, fence):
    scene = scene_copy(guarded.scene); snapshot = guarded.snapshot()
    scope = workspace._BaselineScope(scene, check=check)
    with pytest.raises(ValueError):
        with scope:
            consumer = scope.restore(snapshot)
            root = next(iter(scope.roots.values()))[0]
            if field == 'body': root.state['elapsedSeconds'] += .1
            elif field == 'actions': root.actions.append({'canary': True})
            elif field == 'history': root.history[0]['canary'] = True
            elif field == 'contract': root._protection_contract['config']['critical_max_displacement_mm'] = .1
            elif field == 'by-label': root.by_label[65535] = {'id': 65535}
            elif field == 'protected-ids': root._protected_ids = frozenset()
            elif field == 'guard': consumer.protection.clearance_mm += .1
            elif field == 'guard-tree': consumer.protection._tree = cKDTree(consumer.protection._points + 10.)
            elif field == 'guard-scene': consumer.protection.scene = scene_copy(consumer.scene)
            elif field == 'grid':
                inverse = consumer.grid._inverse.copy(); inverse[0, 3] += 1.
                object.__setattr__(consumer.grid, '_inverse', inverse)
            elif field == 'source': consumer.scene.source['mutated-source'] = True
            elif field == 'scene-owner': root.scene = scene_copy(root.scene)
            else:
                original_catalog = planning.catalog
                altered = original_catalog(); altered[0]['shaftRadiusMm'] += .1
                monkeypatch.setattr(planning, 'catalog', lambda: copy.deepcopy(altered))
            if fence == 'hit': scope.restore(snapshot)
    assert scope.roots == {} and scope.bytes == 0 and scope.scene is None
    assert workspace.baseline_cache_stats()['completed'] is False


def test_closed_scope_cannot_restore_or_retain_another_native_owner(qualified, monkeypatch):
    engine = qualified[1]; snapshot = engine.snapshot(); calls = count_restores(monkeypatch)
    with workspace._BaselineScope(scene_copy(engine.scene), check=check) as scope:
        scope.restore(snapshot)
        root_ref = weakref.ref(next(iter(scope.roots.values()))[0])
    gc.collect()
    assert root_ref() is None and scope.scene is None and scope.check is None
    assert len(calls) == 1 and scope.roots == {} and scope.bytes == 0
    with pytest.raises(ValueError, match='scope is closed'):
        scope.restore(copy.deepcopy(snapshot))
    assert len(calls) == 1 and scope.roots == {} and scope.bytes == 0


def test_two_root_lru_and_total_json_byte_caps(qualified, monkeypatch):
    engine = qualified[1]; snapshots = variants(engine)
    with workspace._BaselineScope(scene_copy(engine.scene), check=check) as scope:
        scope.restore(snapshots[0]); scope.restore(snapshots[1]); scope.restore(snapshots[0])
        scope.restore(snapshots[2])
        assert len(scope.roots) == 2 and scope.evictions == 1
        assert semantic_digest(snapshots[0]) in scope.roots
        assert semantic_digest(snapshots[1]) not in scope.roots
        assert scope.peak_bytes <= 16 * 1024**2
    # Deterministic accounting pressure exercises total bytes independently of
    # the two-root bound without manufacturing a huge native simulation.
    monkeypatch.setattr(workspace, '_json_size', lambda _: 9 * 1024**2)
    with workspace._BaselineScope(scene_copy(engine.scene), check=check) as scope:
        scope.restore(snapshots[0]); scope.restore(snapshots[1])
        assert len(scope.roots) == 1 and scope.evictions == 1 and scope.bytes == 9 * 1024**2
    monkeypatch.setattr(workspace, '_json_size', lambda _: 17 * 1024**2)
    with workspace._BaselineScope(scene_copy(engine.scene), check=check) as scope:
        assert scope.restore(snapshots[0]).snapshot() == snapshots[0]
        assert not scope.roots and scope.bypasses == 1


@pytest.mark.parametrize('failure', ['cancel-hit', 'cancel-cold', 'failed-cold'])
def test_cancellation_and_failed_restore_release_all_native_roots(qualified, failure):
    engine = qualified[1]; snapshots = variants(engine)
    armed = False; checks = 0
    def cancel():
        nonlocal checks
        if armed:
            checks += 1
            if checks == (1 if failure == 'cancel-hit' else 3): raise Cancelled('cancel native baseline proof')
    scope = workspace._BaselineScope(scene_copy(engine.scene), check=cancel)
    with pytest.raises(Cancelled if failure.startswith('cancel') else ValueError):
        with scope:
            scope.restore(snapshots[0])
            root_ref = weakref.ref(next(iter(scope.roots.values()))[0])
            if failure == 'failed-cold':
                bad = copy.deepcopy(snapshots[1]); bad['stateId'] = 'invented-final-state'
                scope.restore(bad)
            else:
                armed = True
                scope.restore(snapshots[0] if failure == 'cancel-hit' else snapshots[1])
    gc.collect()
    assert root_ref() is None and scope.roots == {} and scope.bytes == 0
    assert scope.scene is None and scope.check is None
    assert workspace.baseline_cache_stats()['completed'] is False


def test_cold_workspace_reuses_baseline_without_skipping_native_application(qualified, comparison, monkeypatch):
    case, engine, original, _ = qualified
    record = copy.deepcopy(original); snapshot = engine.snapshot(); source_branch = record['branchId']
    record['strategyComparisons'] = [workspace._seal({'schema': workspace.SCHEMA, 'revision': revision,
        'caseHash': case.semantic_hash, 'sourceBranchId': source_branch, 'sourceSnapshotHash': semantic_digest(snapshot),
        'report': copy.deepcopy(comparison)}, 'workspaceSeal') for revision in range(1, 4)]
    candidate, _, receipt = planning.materialize(engine, comparison, 'long')
    record['branches']['installed-long'] = candidate.snapshot()
    record['strategyApplications'] = [workspace._seal({'revision': 1, 'reportSeal': comparison['reportSeal'],
        'candidateId': 'long', 'branchId': 'installed-long', 'snapshotHash': semantic_digest(candidate.snapshot()),
        'receipt': receipt}, 'applicationSeal')]
    before = copy.deepcopy(record); workspace._VALIDATED.clear()
    calls = count_restores(monkeypatch); native_steps = []
    original_step = SurgicalEngine.step
    def counted_step(self, action, **kwargs):
        if type(self) is SurgicalEngine: native_steps.append(copy.deepcopy(action))
        return original_step(self, action, **kwargs)
    monkeypatch.setattr(SurgicalEngine, 'step', counted_step)
    admissions = []; original_admission = workspace.validate_admission
    def checked_admission(*args, **kwargs):
        admissions.append(1)
        return original_admission(*args, **kwargs)
    monkeypatch.setattr(workspace, 'validate_admission', checked_admission)
    assert workspace.validate_workspace(engine.scene, record, case.semantic_hash) == record['strategyComparisons']
    assert len(calls) == 1
    assert workspace.baseline_cache_stats()['hits'] == 3
    row = next(r for r in comparison['publicPlan']['alternatives'] if r['id'] == 'long')
    reassess = next(p for p in row['phases'] if p['id'] == 'reassess')
    expected_native = copy.deepcopy(snapshot['actions'])
    for index, action in enumerate(row['actionProgram'], 1):
        expected_native.append(action)
        if index == reassess['startAction'] + 2:
            expected_native.append(row['reassessment']['probeAction'])
    assert native_steps == expected_native
    assert len(admissions) == 3 and record == before and engine.snapshot() == snapshot
    assert workspace.baseline_cache_stats()['retainedAfterClose'] == 0
    # Existing cross-call workspace authority still rechecks source admission;
    # this invocation-local optimization contributes no retained roots.
    workspace.validate_workspace(engine.scene, record, case.semantic_hash)
    assert len(calls) == 1 and len(admissions) == 6
