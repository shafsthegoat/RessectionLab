"""Imported endpoint display never promotes a checkpoint into replay evidence."""
import copy
from contextlib import contextmanager
import json
import sys

import numpy as np
import pytest

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.surgical_archive_view import inspect_archive_checkpoint
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene


def fixture(*, v2=False, protected=False):
    labels = np.zeros((8, 8, 8), np.uint16)
    labels[3:5, 3:5, 3:5] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'Generated target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated archive display fixture'},
        {'tissuePhysicsV2': default_profile()} if v2 else {})
    options = {'protection': ProtectionSet(scene), 'guard_config': GuardConfig(require_critical_coverage=False)} if protected else {}
    engine = SurgicalEngine(scene, **options)
    for _ in range(2):
        assert engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
    record = {'branchId': 'full-primary', 'position': 0,
        'branches': {'full-primary': engine.snapshot()},
        'fullCase': {'full-primary': {'status': 'stopped', 'stopReason': 'Generated fixture'}}}
    return scene, engine, record


@contextmanager
def forbid_physics():
    """Observe actual entry even if a future implementation bypasses mocks."""
    if not hasattr(sys, 'monitoring'):
        pytest.skip('Python 3.12 monitoring required')
    monitor = sys.monitoring
    tool_id = next(i for i in range(5, -1, -1) if monitor.get_tool(i) is None)
    codes = {SurgicalEngine.step.__code__, SurgicalEngine.restore.__func__.__code__,
             SurgicalEngine._execute.__code__, SurgicalEngine._substep.__code__}
    def entered(code, offset):
        if code in codes:
            raise AssertionError('Imported display attempted physics or canonical restore')
    monitor.use_tool_id(tool_id, 'archive-display-no-physics')
    monitor.register_callback(tool_id, monitor.events.PY_START, entered)
    monitor.set_events(tool_id, monitor.events.PY_START)
    try:
        yield
    finally:
        monitor.set_events(tool_id, 0)
        monitor.register_callback(tool_id, monitor.events.PY_START, None)
        monitor.free_tool_id(tool_id)


@pytest.mark.parametrize('v2,protected', [(False, False), (True, False), (True, True)])
def test_endpoint_is_detached_explicitly_unverified_and_executes_zero_physics(v2, protected):
    scene, engine, record = fixture(v2=v2, protected=protected)
    before = copy.deepcopy(record)
    with forbid_physics():
        result = inspect_archive_checkpoint(scene, freeze_json(record))
    assert result['view']['stateId'] == engine.state_id
    assert result['position'] == 2  # Endpoint, not the saved position-zero cursor.
    assert result['view']['history'] == []
    assert result['view']['metrics']['collisions'] is None
    assert 'timeline' not in result and 'playback' not in result
    assert result['verification']['status'] == 'unverified-import'
    assert result['verification']['readOnly'] is True
    assert not result['verification']['canonical']
    assert not result['verification']['causalReplayVerified']
    assert result['verification']['physicsActionsExecuted'] == 0
    assert bool(result['verification']['protectionContractId']) == protected
    json.dumps(result, allow_nan=False)
    result['view']['slots']['left']['position'][0] = 999
    result['fullCase']['status'] = 'caller annotation'
    assert record == before


def test_altered_checkpoint_fails_its_state_hash():
    scene, _, record = fixture()
    record['branches']['full-primary']['checkpoint']['state']['fluidAddedMm3'] = 999.
    with forbid_physics(), pytest.raises(ValueError, match='state hash mismatch'):
        inspect_archive_checkpoint(scene, record)


def test_fabricated_self_consistent_endpoint_remains_unverified():
    scene, _, record = fixture()
    snapshot = record['branches']['full-primary']
    # No actual removal action produced this endpoint. A digest cannot prove it.
    cell = int(np.flatnonzero(scene.labels)[0])
    snapshot['checkpoint']['state']['removed'] = [cell]
    digest = semantic_digest({'scene': scene.identity, 'state': snapshot['checkpoint']['state']})
    snapshot['stateId'] = snapshot['checkpoint']['stateId'] = digest
    with forbid_physics():
        result = inspect_archive_checkpoint(scene, record)
    assert result['view']['removed'] == [cell]
    assert result['view']['metrics']['targetRemovedMm3'] == scene.voxel_mm3
    assert result['verification']['checkpointHashMatched']
    assert result['verification']['status'] == 'unverified-import'
    assert not result['verification']['causalReplayVerified']
    assert not result['verification']['canonical']


@pytest.mark.parametrize('mutation,pattern', [
    (lambda r: r['branches']['full-primary'].update(schema='wrong'), 'engine version'),
    (lambda r: r['branches']['full-primary'].update(sceneId='another source'), 'scene'),
    (lambda r: r['branches']['full-primary']['checkpoint'].update(position=True), 'position'),
    (lambda r: r['branches']['full-primary']['checkpoint'].update(position=1), 'position'),
    (lambda r: r['branches']['full-primary']['checkpoint'].update(stateId='another state'), 'identity'),
    (lambda r: r['branches']['full-primary'].pop('checkpoint'), 'position'),
    (lambda r: r.update(branches=[]), 'branch records'),
    (lambda r: r['branches']['full-primary'].update(actions='inspect'), 'array'),
    (lambda r: r.update(fullCase=[]), 'metadata'),
])
def test_malformed_import_fails_closed(mutation, pattern):
    scene, _, record = fixture()
    mutation(record)
    with forbid_physics(), pytest.raises(ValueError, match=pattern):
        inspect_archive_checkpoint(scene, record)


def test_protection_contract_integrity_is_required_and_branch_selection_is_explicit():
    scene, _, record = fixture(protected=True)
    record['branches']['alternative'] = copy.deepcopy(record['branches']['full-primary'])
    with forbid_physics():
        selected = inspect_archive_checkpoint(scene, record, branch_id='alternative')
    assert selected['branchId'] == 'alternative' and selected['fullCase'] is None
    record['branches']['full-primary']['protectionContract']['clearanceMm'] += 1.
    with forbid_physics(), pytest.raises(ValueError, match='integrity mismatch'):
        inspect_archive_checkpoint(scene, record)
    with pytest.raises(ValueError, match='unavailable'):
        inspect_archive_checkpoint(scene, record, branch_id='missing')


def test_cancellation_is_forwarded_without_mutating_import():
    scene, _, record = fixture()
    before = copy.deepcopy(record)
    def cancel():
        raise TimeoutError('cancel archive view')
    with pytest.raises(TimeoutError, match='cancel archive view'):
        inspect_archive_checkpoint(scene, record, check=cancel)
    assert record == before


def test_unversioned_source_edits_cannot_keep_old_checkpoint_identity():
    scene, _, record = fixture()
    scene.regions[0]['target'] = False
    with forbid_physics(), pytest.raises(ValueError, match='Source scene changed'):
        inspect_archive_checkpoint(scene, record)
    with pytest.raises(ValueError, match='Exact source'):
        inspect_archive_checkpoint(None, record)
