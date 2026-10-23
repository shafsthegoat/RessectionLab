"""Fixed prefix roles over generated ordinary saved bridge artifacts."""
from contextlib import contextmanager
import hashlib
import shutil
import sys

import pytest

from resectionlab.core import freeze_json, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_prefix_replay import _owned_bytes
from resectionlab.surgical_prefix_worker import PrefixWorkerError, PrefixWorkerSession


def call(session, operation, args):
    return session.execute(operation, args, _Request('generated-prefix-role'), lambda *_: None)


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    directory = tmp_path_factory.mktemp('prefix-worker-source')
    session = BridgeSession(directory / 'source-transfers')
    try:
        loaded = call(session, 'createSurgicalFixture', {})
        case_hash = loaded['caseHash']
        result = call(session, 'surgicalRehearsal', {'caseHash': case_hash, 'command': 'create'})
        for duration in (.1, .3):
            result = call(session, 'surgicalRehearsal', {'caseHash': case_hash, 'command': 'step',
                'expectedStateId': result['view']['stateId'],
                'action': {'slot': 'left', 'operation': 'inspect', 'duration': duration}})
            assert result['outcome']['status'] == 'accepted'
        path = directory / 'generated.ressectionlab'
        call(session, 'saveCase', {'caseHash': case_hash, 'path': str(path)})
        return {'path': path, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'branch': result['branchId'], 'endpoint': result['view']['stateId']}
    finally:
        session.close()


@pytest.fixture
def workers(tmp_path, monkeypatch, archive):
    monkeypatch.setenv('RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY', str(tmp_path / 'prepared'))
    created = []
    def make(role, *, load=True, source=None, sha256=None):
        session = BridgeSession(tmp_path / ('transfers-' + str(len(created))))
        worker = PrefixWorkerSession(session, role=role, source_path=source or archive['path'],
            source_sha256=sha256 or archive['sha256'], branch_id=archive['branch'])
        created.append(worker)
        opened = call(worker, 'loadCase', {'path': str(source or archive['path'])}) if load else None
        return worker, session, opened
    yield make
    for worker in created:
        worker.close()


def start(worker, opened, archive, role):
    args = {'caseHash': opened['caseHash'], 'command': 'seekPrefix',
            'branchId': archive['branch'], 'position': 0}
    if role == 'consumer':
        args['cacheOnly'] = True
    return call(worker, 'surgicalRehearsal', args)


def frame(worker, opened, archive, position, index, **acks):
    return call(worker, 'surgicalRehearsal', {'caseHash': opened['caseHash'], 'command': 'seekPrefixFrame',
        'branchId': archive['branch'], 'position': position, 'frameIndex': index, **acks})


@contextmanager
def steps(*, forbid=False):
    if not hasattr(sys, 'monitoring'):
        pytest.skip('Python 3.12 monitoring required')
    monitor = sys.monitoring
    tool = next(i for i in range(5, -1, -1) if monitor.get_tool(i) is None)
    observed = []
    code = SurgicalEngine.step.__code__
    def entered(current, _offset):
        if current is not code:
            return monitor.DISABLE
        observed.append(len(sys._getframe(1).f_locals['self'].actions))
        if forbid:
            raise AssertionError('Consumer attempted a public physics step')
    monitor.use_tool_id(tool, 'generated-prefix-worker-steps')
    monitor.register_callback(tool, monitor.events.PY_START, entered)
    monitor.set_events(tool, monitor.events.PY_START)
    try:
        yield observed
    finally:
        monitor.set_events(tool, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)


def test_real_roles_publish_once_consume_signed_packages_and_preserve_source(workers, archive):
    producer, p_session, p_open = workers('producer')
    consumer, c_session, c_open = workers('consumer')
    assert p_open['caseHash'] == c_open['caseHash']
    assert p_open['prefixWorker'] == {'schema': 'resectionlab.prefix-worker.v1',
        'role': 'producer', 'sourceArchiveSha256': archive['sha256']}
    p_origin, c_origin = p_session.cases[p_open['caseHash']].artifacts, c_session.cases[c_open['caseHash']].artifacts
    original = thaw_json(p_origin)
    p_start = start(producer, p_open, archive, 'producer')
    c_start = start(consumer, c_open, archive, 'consumer')
    binding = p_start['prefixWorker']['binding']
    assert binding == c_start['prefixWorker']['binding']
    assert binding == dict(p_session.surgical_prefix_session[3]._prepared_binding)
    assert binding['protectionContractHash'] == p_start['verification']['protectionContractHash']
    assert p_start['prefixWorker']['actionFrameBytes'] == c_start['prefixWorker']['actionFrameBytes'] == 0
    physical = []
    last = None
    for position in range(2):
        with steps() as observed:
            produced = frame(producer, p_open, archive, position, 0)
        physical.extend(observed)
        p_owner = p_session.surgical_prefix_session[3]
        count = produced['playback']['frameCount']
        assert produced['prefixWorker']['store']['publications'] == position + 1
        assert produced['prefixWorker']['actionFrameBytes'] == _owned_bytes(p_owner._frames[position])
        # Metadata is detached from actual owner identity and counters.
        produced['prefixWorker']['binding']['sceneId'] = 'caller edit'
        produced['prefixWorker']['cache']['publishedActions'] = 999
        token, field_tokens = None, {}
        with steps(forbid=True) as consumer_steps:
            for index in range(count):
                last = frame(consumer, c_open, archive, position, index,
                    **({'prefixToken': token, 'prefixFieldTokens': field_tokens} if token else {}))
                assert last['view']['stateId'] == p_owner._frames[position][index]['stateId']
                assert last['prefixWorker']['binding'] == binding
                assert last['prefixWorker']['cache'] == {'mode': 'only', 'loadedActions': position + 1,
                    'publishedActions': 0, 'misses': 0, 'lastSkip': None}
                assert last['readOnly'] and not last['changed'] and last['fullCase'] is None
                token, field_tokens = last['timelineToken'], last['viewFieldTokens']
        assert consumer_steps == []
    assert physical == [0, 1]
    assert last['view']['stateId'] == archive['endpoint']
    assert p_session.cases[p_open['caseHash']].artifacts is p_origin
    assert c_session.cases[c_open['caseHash']].artifacts is c_origin
    assert thaw_json(p_origin) == thaw_json(c_origin) == original
    assert hashlib.sha256(archive['path'].read_bytes()).hexdigest() == archive['sha256']
    assert not hasattr(producer, 'cases') and not hasattr(producer, 'surgical_prefix_session')
    with pytest.raises(AttributeError):
        producer.role = 'consumer'
    producer.prune_transfers()
    assert producer.run_dir is p_session.run_dir


@pytest.mark.parametrize('operation,args', [
    ('loadCase', {}), ('ping', {}), ('listRuns', {}), ('trainPatient', {}),
    ('surgicalRehearsal', {'command': 'inspect'}),
    ('surgicalRehearsal', {'command': 'step'}),
    ('surgicalRehearsal', {'command': 'seekFrame'}),
])
def test_forbidden_operations_permanently_fail_without_physics(workers, archive, operation, args):
    worker, session, opened = workers('producer')
    start(worker, opened, archive, 'producer')
    with steps(forbid=True) as observed:
        with pytest.raises(PrefixWorkerError):
            call(worker, operation, args)
        with pytest.raises(PrefixWorkerError, match='permanently unavailable'):
            frame(worker, opened, archive, 0, 0)
    assert observed == []
    worker.close()
    assert not session.cases and not hasattr(session, 'surgical_prefix_session')


@pytest.mark.parametrize('role,cache_only', [('consumer', False), ('consumer', None), ('producer', True)])
def test_role_mode_cannot_be_changed_by_start_request(workers, archive, role, cache_only):
    worker, session, opened = workers(role)
    args = {'caseHash': opened['caseHash'], 'branchId': archive['branch'], 'command': 'seekPrefix', 'position': 0}
    if cache_only is not None:
        args['cacheOnly'] = cache_only
    with steps(forbid=True), pytest.raises(PrefixWorkerError, match='mode differs'):
        call(worker, 'surgicalRehearsal', args)
    assert not hasattr(session, 'surgical_prefix_session')


@pytest.mark.parametrize('change', ['skip', 'nonzero-frame', 'producer-ack', 'restart', 'branch', 'source'])
def test_producer_rejects_reordering_restart_ack_and_foreign_owner(workers, archive, change):
    worker, _, opened = workers('producer')
    start(worker, opened, archive, 'producer')
    args = {'caseHash': opened['caseHash'], 'branchId': archive['branch'],
            'command': 'seekPrefixFrame', 'position': 0, 'frameIndex': 0}
    if change == 'skip': args['position'] = 1
    elif change == 'nonzero-frame': args['frameIndex'] = 1
    elif change == 'producer-ack': args['prefixToken'] = 'prefix-replay.v1/client'
    elif change == 'restart': args.update(command='seekPrefix', position=0)
    elif change == 'branch': args['branchId'] = 'other'
    elif change == 'source': args['caseHash'] = 'sha256:' + '0' * 64
    with steps(forbid=True) as observed, pytest.raises(PrefixWorkerError):
        call(worker, 'surgicalRehearsal', args)
    assert observed == []


def test_consumer_missing_package_fails_permanently_without_physical_fallback(workers, archive):
    worker, session, opened = workers('consumer')
    start(worker, opened, archive, 'consumer')
    with steps(forbid=True) as observed:
        with pytest.raises(ValueError):
            frame(worker, opened, archive, 0, 0)
        with pytest.raises(PrefixWorkerError, match='permanently unavailable'):
            frame(worker, opened, archive, 0, 0)
    assert observed == []
    assert session.surgical_prefix_session[3].statistics['preparedCache']['misses'] == 1


def test_consumer_only_repeats_latest_resident_tuple_without_load_or_steps(workers, archive):
    producer, _, p_open = workers('producer')
    consumer, session, c_open = workers('consumer')
    start(producer, p_open, archive, 'producer')
    start(consumer, c_open, archive, 'consumer')
    frame(producer, p_open, archive, 0, 0)
    with steps(forbid=True) as observed:
        first = frame(consumer, c_open, archive, 0, 0)
        repeated = frame(consumer, c_open, archive, 0, 0)
        assert repeated['view'] == first['view'] and repeated['view'] is not first['view']
        assert repeated['prefixWorker']['cache'] == first['prefixWorker']['cache']
        assert repeated['prefixWorker']['store']['hits'] == first['prefixWorker']['store']['hits'] == 1
        frame(consumer, c_open, archive, 0, 1)
        with pytest.raises(PrefixWorkerError, match='exact next observation'):
            frame(consumer, c_open, archive, 0, 0)
    assert observed == []
    assert session.surgical_prefix_session[3]._prepared_store.hits == 1


def test_consumer_repeat_cannot_reload_evicted_package(workers, archive):
    producer, _, p_open = workers('producer')
    consumer, session, c_open = workers('consumer')
    start(producer, p_open, archive, 'producer')
    start(consumer, c_open, archive, 'consumer')
    frame(producer, p_open, archive, 0, 0)
    frame(consumer, c_open, archive, 0, 0)
    owner = session.surgical_prefix_session[3]
    owner._frames.clear()
    with steps(forbid=True) as observed, pytest.raises(PrefixWorkerError, match='must remain resident'):
        frame(consumer, c_open, archive, 0, 0)
    assert observed == [] and owner._prepared_store.hits == 1


@pytest.mark.parametrize('args', [
    {'command': 'seekPrefixFrame', 'position': 1, 'frameIndex': 0},
    {'command': 'seekPrefixFrame', 'position': 0, 'frameIndex': 1},
    {'command': 'seekPrefixFrame', 'position': 0, 'frameIndex': 0, 'timelineToken': 'canonical'},
    {'command': 'seekPrefix', 'position': 0, 'cacheOnly': True},
])
def test_consumer_rejects_skips_canonical_acknowledgements_and_restart(workers, archive, args):
    worker, session, opened = workers('consumer')
    start(worker, opened, archive, 'consumer')
    with steps(forbid=True) as observed, pytest.raises(PrefixWorkerError):
        call(worker, 'surgicalRehearsal', {'caseHash': opened['caseHash'], 'branchId': archive['branch'], **args})
    assert observed == [] and session.surgical_prefix_session[3]._prepared_store.misses == 0


@pytest.mark.parametrize('change', ['binding', 'mode', 'owner', 'artifacts', 'store-miss', 'store-cap', 'canonical'])
def test_existing_control_drift_is_rejected_before_next_delegate(workers, archive, change):
    worker, session, opened = workers('producer')
    start(worker, opened, archive, 'producer')
    owned = session.surgical_prefix_session
    owner = owned[3]
    if change == 'binding': owner._prepared_binding['protectionContractHash'] = '0' * 64
    elif change == 'mode': owner._cache_only = True
    elif change == 'owner': session.surgical_prefix_session = (*owned[:3], object())
    elif change == 'artifacts': session.cases[opened['caseHash']].artifacts = freeze_json(thaw_json(owned[1]))
    elif change == 'store-miss': owner._prepared_store.misses = 1
    elif change == 'store-cap': owner._prepared_store.max_action_bytes -= 1
    elif change == 'canonical': session.surgical_sessions = {}
    with steps(forbid=True) as observed, pytest.raises(PrefixWorkerError):
        frame(worker, opened, archive, 0, 0)
    assert observed == []


def test_downstream_error_poisons_role_but_cleanup_survives(workers, archive, monkeypatch):
    worker, session, opened = workers('producer')
    failure = RuntimeError('downstream control failure')
    def failed(*args, **kwargs):
        raise failure
    monkeypatch.setattr(session, 'execute', failed)
    with pytest.raises(RuntimeError) as actual:
        start(worker, opened, archive, 'producer')
    assert actual.value is failure
    with pytest.raises(PrefixWorkerError, match='permanently unavailable'):
        start(worker, opened, archive, 'producer')
    worker.prune_transfers()
    worker.close()
    assert not session.cases


def test_bad_response_authority_permanently_fails_role(workers, archive, monkeypatch):
    worker, session, opened = workers('producer')
    normal_execute = session.execute
    def changed(*args):
        result = normal_execute(*args)
        result['verification']['readOnly'] = False
        return result
    monkeypatch.setattr(session, 'execute', changed)
    with steps(forbid=True) as observed, pytest.raises(PrefixWorkerError, match='authority or progress'):
        start(worker, opened, archive, 'producer')
    assert observed == []
    with pytest.raises(PrefixWorkerError, match='permanently unavailable'):
        start(worker, opened, archive, 'producer')


def test_source_hash_is_required_before_and_after_normal_load(workers, archive, tmp_path, monkeypatch):
    worker, session, _ = workers('producer', load=False, sha256='0' * 64)
    with pytest.raises(PrefixWorkerError, match='archive bytes changed'):
        call(worker, 'loadCase', {'path': str(archive['path'])})
    assert not session.cases
    copied = tmp_path / 'copied.ressectionlab'
    shutil.copyfile(archive['path'], copied)
    worker, session, _ = workers('producer', load=False, source=copied)
    normal_execute = session.execute
    def changed_after_load(*args):
        result = normal_execute(*args)
        with copied.open('ab') as stream:
            stream.write(b'changed after admission')
        return result
    monkeypatch.setattr(session, 'execute', changed_after_load)
    with pytest.raises(PrefixWorkerError, match='archive bytes changed'):
        call(worker, 'loadCase', {'path': str(copied)})
    with pytest.raises(PrefixWorkerError, match='permanently unavailable'):
        call(worker, 'loadCase', {'path': str(copied)})


@pytest.mark.parametrize('changed', [{'role': 'editable'}, {'role': True}, {'source_sha256': 'bad'},
                                    {'branch_id': ''}, {'branch_id': '   '}])
def test_startup_configuration_is_fixed_and_validated(tmp_path, archive, changed):
    session = BridgeSession(tmp_path / 'invalid-transfers')
    configuration = {'role': 'producer', 'source_path': archive['path'],
                     'source_sha256': archive['sha256'], 'branch_id': archive['branch'], **changed}
    try:
        with pytest.raises(PrefixWorkerError):
            PrefixWorkerSession(session, **configuration)
        assert not session.cases
    finally:
        session.close()


def test_explicit_fresh_restart_authenticates_existing_packages_with_separate_actual_write_count(workers, archive):
    first_binding = first_packages = first_frames = None
    observed_steps = []
    for attempt in range(2):
        producer, p_session, p_open = workers('producer')
        consumer, c_session, c_open = workers('consumer')
        start(producer, p_open, archive, 'producer')
        start(consumer, c_open, archive, 'consumer')
        frames = []
        for position in range(2):
            with steps() as observed:
                produced = frame(producer, p_open, archive, position, 0)
            observed_steps.extend(observed)
            owner = p_session.surgical_prefix_session[3]
            frames.append([dict(row) for row in owner._frames[position]])
            with steps(forbid=True) as consumer_steps:
                for index in range(produced['playback']['frameCount']):
                    consumed = frame(consumer, c_open, archive, position, index)
            assert consumer_steps == []
            assert produced['prefixWorker']['cache']['publishedActions'] == position + 1
            assert produced['prefixWorker']['store']['publications'] == (position + 1 if attempt == 0 else 0)
            assert consumed['prefixWorker']['store']['hits'] == position + 1
        assert consumed['view']['stateId'] == archive['endpoint']
        binding = produced['prefixWorker']['binding']
        namespace = owner._prepared_store._namespace
        packages = {str(path.relative_to(namespace)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in namespace.rglob('*') if path.is_file()}
        if attempt == 0:
            first_binding, first_packages, first_frames = binding, packages, frames
        else:
            assert binding == first_binding and packages == first_packages and frames == first_frames
        producer.close()
        consumer.close()
    assert observed_steps == [0, 1, 0, 1]


def test_actual_store_publication_counter_cannot_regress_before_next_action(workers, archive):
    worker, session, opened = workers('producer')
    start(worker, opened, archive, 'producer')
    frame(worker, opened, archive, 0, 0)
    session.surgical_prefix_session[3]._prepared_store.publications = 0
    with steps(forbid=True) as observed, pytest.raises(PrefixWorkerError, match='publication counter'):
        frame(worker, opened, archive, 1, 0)
    assert observed == []
