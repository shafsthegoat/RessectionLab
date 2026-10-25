"""Fixed-role bridge composition for fresh production and prepared consumption.

The normal bridge owns every source admission, transition, signature and view.
This proxy restricts commands and reports actual committed control metadata; it
does not wait for packages, retry failures, or grant cache/replay authority.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from types import MappingProxyType

SCHEMA = 'resectionlab.prefix-worker.v1'
MAX_STAGED_ACTION_BYTES = 64 * 1024**2
_BINDING_KEYS = frozenset({'schema', 'sceneId', 'originSnapshotHash', 'actionLogHash',
    'protectionContractHash', 'implementation', 'prefixImplementation'})
_STORE_KEYS = ('directory', 'namespaceDirectory', 'diskBytes', 'maxDiskBytes', 'maxActionBytes',
               'hits', 'misses', 'publications', 'lastSkip')
_CANONICAL_OWNERS = ('surgical_sessions', 'surgical_replay_cache', 'surgical_frame_cache')


class PrefixWorkerError(ValueError):
    code = 'PREFIX_WORKER_FAILED'


def _need(condition, message):
    if not condition:
        raise PrefixWorkerError(message)


def _digest(value):
    return (type(value) is str and len(value.removeprefix('sha256:')) == 64
            and all(char in '0123456789abcdef' for char in value.removeprefix('sha256:')))


class PrefixWorkerSession:
    """One source, one branch and one immutable producer/consumer startup role."""
    __slots__ = ('__session', '__config', '__failed', '__closed', '__entry', '__artifacts',
                 '__case_hash', '__owner', '__store', '__binding', '__store_limits',
                 '__verified', '__total', '__position', '__frame', '__last', '__store_publications')

    def __init__(self, session, *, role, source_path, source_sha256, branch_id):
        # BridgeRuntime privately supplies its normal session. Do not import
        # desktop_bridge here: its -m entrypoint may be the __main__ module.
        _need(type(role) is str and role in ('producer', 'consumer'), 'Invalid prefix worker role')
        _need(type(source_sha256) is str and len(source_sha256) == 64 and _digest(source_sha256),
              'Invalid prefix source archive SHA256')
        _need(type(branch_id) is str and 0 < len(branch_id) <= 128 and branch_id.strip() and '\x00' not in branch_id,
              'Invalid prefix branch')
        _need(type(source_path) is str or isinstance(source_path, Path), 'Invalid prefix source path')
        path = Path(source_path).resolve(strict=True)
        _need(path.is_file() and len(str(path)) <= 4096, 'Invalid prefix source archive')
        _need(type(session.cases) is OrderedDict and not session.cases
              and not any(name in vars(session) for name in (*_CANONICAL_OWNERS, 'surgical_prefix_session')),
              'Prefix worker requires an unused bridge session')
        self.__session = session
        self.__config = MappingProxyType({'role': role, 'path': str(path),
            'sha256': source_sha256, 'branch': branch_id})
        self.__failed = self.__closed = False
        self.__entry = self.__artifacts = self.__case_hash = None
        self.__owner = self.__store = self.__binding = self.__store_limits = None
        self.__verified = self.__position = self.__frame = 0
        self.__total = self.__last = None
        self.__store_publications = 0

    @property
    def run_dir(self):
        return self.__session.run_dir

    def _source_hash(self, request):
        digest = hashlib.sha256()
        with Path(self.__config['path']).open('rb') as stream:
            while chunk := stream.read(65536):
                request.check()
                digest.update(chunk)
        request.check()
        _need(digest.hexdigest() == self.__config['sha256'], 'Prefix source archive bytes changed')

    def _case(self):
        session = self.__session
        _need(type(session.cases) is OrderedDict and len(session.cases) == 1
              and session.cases.get(self.__case_hash) is self.__entry
              and self.__entry.artifacts is self.__artifacts
              and self.__entry.case.semantic_hash == self.__case_hash,
              'Prefix source owner changed')
        _need(not any(name in vars(session) for name in _CANONICAL_OWNERS),
              'Prefix worker acquired canonical or editable ownership')

    def _control(self, verified):
        from .surgical_prefix_cache import PreparedPrefixStore
        from .surgical_prefix_replay import SurgicalPrefixReplay
        self._case()
        owned = vars(self.__session).get('surgical_prefix_session')
        _need(type(owned) is tuple and len(owned) == 4
              and owned[0] is self.__entry and owned[1] is self.__artifacts
              and owned[2] == self.__config['branch'] and type(owned[3]) is SurgicalPrefixReplay,
              'Prefix worker owner differs')
        owner = owned[3]
        if self.__owner is not None:
            _need(owner is self.__owner and owner._prepared_store is self.__store,
                  'Prefix worker owner or store was replaced')
        _need(owner._snapshot is self.__artifacts['surgical_toolbox']['branches'][self.__config['branch']],
              'Prefix worker snapshot owner changed')
        _need(owner._cache_only is (self.__config['role'] == 'consumer')
              and not owner._blocked and owner._failure is None and owner._invalidated is None,
              'Prefix worker failed or changed mode')
        binding = owner._prepared_binding
        _need(type(binding) is dict and set(binding) == _BINDING_KEYS
              and binding['schema'] == 'resectionlab.prepared-prefix-binding.v1'
              and all(_digest(binding[key]) for key in _BINDING_KEYS - {'schema'})
              and binding == {'schema': 'resectionlab.prepared-prefix-binding.v1',
                  'sceneId': owner._scene.identity, 'originSnapshotHash': owner._origin_hash,
                  'actionLogHash': owner._action_hash, 'protectionContractHash': owner._protection_hash,
                  'implementation': owner._implementation, 'prefixImplementation': owner._prefix_implementation},
              'Prefix prepared binding differs')
        if self.__binding is not None:
            _need(binding == self.__binding, 'Prefix prepared binding changed')
        stats = owner.statistics
        total = len(owner._snapshot['actions'])
        _need(type(verified) is int and type(stats['verifiedThrough']) is int
              and type(stats['totalActions']) is int and stats['verifiedThrough'] == verified == len(owner._history)
              and stats['totalActions'] == total and 0 <= verified <= total
              and (self.__total is None or total == self.__total), 'Prefix action frontier differs')
        for used, limit, maximum in (('checkpointBytes', 'maxCheckpointBytes', 16 * 1024**2),
                ('frameBytes', 'maxFrameBytes', MAX_STAGED_ACTION_BYTES),
                ('historyBytes', 'maxHistoryBytes', 4 * 1024**2)):
            _need(type(stats[used]) is int and type(stats[limit]) is int
                  and 0 <= stats[used] <= stats[limit] <= maximum, 'Prefix retention budget differs')
        _need(type(stats['immutableInputBytes']) is int and 0 <= stats['immutableInputBytes'] <= 128 * 1024**2
              and type(owner._max_json_bytes) is int and 0 < owner._max_json_bytes <= 64 * 1024**2,
              'Prefix input or decoded budget differs')
        cache = stats['preparedCache']
        producer = self.__config['role'] == 'producer'
        _need(type(cache) is dict and cache['mode'] == ('prepare' if producer else 'only')
              and all(type(cache[key]) is int for key in ('loadedActions', 'publishedActions', 'misses'))
              and cache['loadedActions'] == (0 if producer else verified)
              and cache['publishedActions'] == (verified if producer else 0)
              and cache['misses'] == 0 and cache['lastSkip'] is None, 'Prefix prepared counters differ')
        store = owner._prepared_store
        _need(type(store) is PreparedPrefixStore, 'Prefix prepared store differs')
        stored = store.statistics
        _need(all(type(stored[key]) is int for key in ('diskBytes', 'maxDiskBytes', 'maxActionBytes',
                  'hits', 'misses', 'publications'))
              and 0 <= stored['diskBytes'] <= stored['maxDiskBytes'] and 0 < stored['maxDiskBytes'] <= 512 * 1024**2
              and 0 < stored['maxActionBytes'] <= MAX_STAGED_ACTION_BYTES
              and stored['hits'] == (0 if producer else verified)
              and 0 <= stored['publications'] <= (verified if producer else 0)
              and stored['misses'] == 0 and stored['lastSkip'] is None
              and stored['directory'] == str(store.directory)
              and stored['namespaceDirectory'] == str(store._namespace)
              and all(type(stored[key]) is str and len(stored[key]) <= 4096
                      for key in ('directory', 'namespaceDirectory')), 'Prefix store counters or budgets differ')
        limits = tuple(stored[key] for key in ('directory', 'namespaceDirectory', 'maxDiskBytes', 'maxActionBytes'))
        advance = verified - self.__verified
        publication_delta = stored['publications'] - self.__store_publications
        _need(advance in (0, 1) and publication_delta in ((0, 1) if producer and advance else (0,)),
              'Prefix store publication counter changed outside one committed action')
        _need(self.__store_limits is None or limits == self.__store_limits, 'Prefix store configuration changed')
        return owner, total, dict(binding), dict(cache), {key: stored[key] for key in _STORE_KEYS}, limits

    def execute(self, operation, args, request, progress):
        _need(not self.__failed and not self.__closed, 'Prefix worker is permanently unavailable')
        try:
            _need(type(operation) is str and type(args) is dict and all(type(key) is str for key in args),
                  'Invalid prefix worker request')
            request.check()
            if self.__entry is None:
                _need(operation == 'loadCase' and args == {'path': self.__config['path']},
                      'Prefix worker requires its one exact source load')
                _need(not self.__session.cases and 'surgical_prefix_session' not in vars(self.__session),
                      'Prefix source was installed outside the worker')
                self._source_hash(request)
                result = self.__session.execute(operation, args, request, progress)
                self._source_hash(request)
                _need(type(result) is dict and _digest(result.get('caseHash')) and 'prefixWorker' not in result,
                      'Invalid prefix source result')
                self.__case_hash = result['caseHash']
                self.__entry = self.__session.cases.get(self.__case_hash)
                _need(self.__entry is not None, 'Prefix source result has no owner')
                self.__artifacts = self.__entry.artifacts
                branches = self.__artifacts.get('surgical_toolbox', {}).get('branches', {})
                _need(self.__config['branch'] in branches, 'Prefix branch is absent from source')
                self._case()
                result['prefixWorker'] = {'schema': SCHEMA, 'role': self.__config['role'],
                                         'sourceArchiveSha256': self.__config['sha256']}
                return result
            self._case()
            _need(operation == 'surgicalRehearsal' and args.get('caseHash') == self.__case_hash
                  and args.get('branchId') == self.__config['branch'], 'Prefix command source or branch differs')
            starting = self.__owner is None
            repeated = False
            keys = {'caseHash', 'command', 'branchId', 'position'}
            if starting:
                _need('surgical_prefix_session' not in vars(self.__session), 'Prefix owner was installed outside worker')
                _need(args.get('command') == 'seekPrefix' and type(args.get('position')) is int
                      and args['position'] == 0, 'Prefix worker requires one prefix-zero start')
                keys.add('cacheOnly')
                expected_mode = self.__config['role'] == 'consumer'
                _need(args.get('cacheOnly', False) is expected_mode, 'Prefix startup mode differs from role')
                verified = 0
            else:
                self._control(self.__verified)
                keys.add('frameIndex')
                if self.__config['role'] == 'consumer':
                    keys.update(('prefixToken', 'prefixFieldTokens'))
                _need(args.get('command') == 'seekPrefixFrame'
                      and type(args.get('position')) is int and type(args.get('frameIndex')) is int,
                      'Prefix worker requires its exact next observation')
                requested = (args['position'], args['frameIndex'])
                repeated = self.__config['role'] == 'consumer' and requested == self.__last
                _need(repeated or requested == (self.__position, self.__frame)
                      and self.__position < self.__total, 'Prefix worker requires its exact next observation')
                if repeated:
                    rows = self.__owner._frames.get(requested[0])
                    _need(type(rows) is list and requested[1] < len(rows),
                          'Repeated observation must remain resident; no package reload permitted')
                verified = self.__verified if repeated else self.__position + 1
            _need(set(args) <= keys, 'Unsupported prefix worker argument')
            result = self.__session.execute(operation, args, request, progress)
            owner, total, binding, cache, store, limits = self._control(verified)
            verification = result.get('verification') if type(result) is dict else None
            _need(type(verification) is dict and verification.get('status') == 'reconstructed-prefix'
                  and verification.get('readOnly') is True and verification.get('canonical', False) is False
                  and type(verification.get('verifiedThrough')) is int and type(verification.get('totalActions')) is int
                  and verification.get('verifiedThrough') == verified and verification.get('totalActions') == total
                  and verification.get('preparedCache') == cache and verification.get('failure') is None
                  and all(verification.get(key) == binding[field] for key, field in (
                      ('originSnapshotHash', 'originSnapshotHash'), ('actionLogHash', 'actionLogHash'),
                      ('protectionContractHash', 'protectionContractHash'), ('implementation', 'implementation'),
                      ('prefixImplementation', 'prefixImplementation')))
                  and result.get('readOnly') is True and result.get('changed') is False
                  and result.get('fullCase') is None and result.get('outcome') is None
                  and result.get('budgetStopped') is False and result.get('branchId') == self.__config['branch']
                  and result.get('caseHash') == self.__case_hash and 'prefixWorker' not in result,
                  'Prefix response changed authority or progress')
            action_bytes = 0
            if not starting:
                from .surgical_prefix_replay import _owned_bytes
                position, frame = args['position'], args['frameIndex']
                rows = owner._frames.get(position)
                playback = result.get('playback')
                _need(type(rows) is list and type(playback) is dict
                      and playback.get('branchId') == self.__config['branch']
                      and all(type(playback.get(key)) is int for key in
                              ('position', 'actionIndex', 'frameIndex', 'frameCount'))
                      and playback.get('position') == position and playback.get('actionIndex') == position
                      and playback.get('frameIndex') == frame and playback.get('frameCount') == len(rows)
                      and frame < len(rows) and playback.get('readOnly') is True
                      and playback.get('frameStateId') == rows[frame]['stateId'] == result['view']['stateId'],
                      'Prefix playback observation differs')
                action_bytes = _owned_bytes(rows)
                _need(0 < action_bytes <= MAX_STAGED_ACTION_BYTES, 'Prefix action frame graph exceeds bound')
                if not repeated:
                    if self.__config['role'] == 'producer' or frame + 1 == len(rows):
                        self.__position += 1; self.__frame = 0
                    else:
                        self.__frame += 1
                    self.__last = (position, frame)
            self.__owner, self.__store = owner, owner._prepared_store
            self.__binding = MappingProxyType(binding)
            self.__store_limits, self.__verified, self.__total = limits, verified, total
            self.__store_publications = store['publications']
            result['prefixWorker'] = {'schema': SCHEMA, 'role': self.__config['role'], 'binding': dict(binding),
                'verifiedThrough': verified, 'totalActions': total, 'cache': cache, 'store': store,
                'actionFrameBytes': action_bytes, 'maxStagedActionBytes': MAX_STAGED_ACTION_BYTES,
                'sceneId': binding['sceneId'], 'sourceArchiveSha256': self.__config['sha256']}
            request.check()
            return result
        except BaseException:
            self.__failed = True
            raise

    def prune_transfers(self):
        try:
            return self.__session.prune_transfers()
        except BaseException:
            self.__failed = True
            raise

    def close(self):
        if not self.__closed:
            self.__closed = True
            try:
                self.__session.close()
            except BaseException:
                self.__failed = True
                raise
