"""Explicit full-vascular pipeline qualification; never a playback authority.

Only the consumer writes binary observations. The producer records hashes of
already committed compressed bytes. One later consumer audit verifies numerical
truth, joined to producer receipts by exact bytes and complete metadata.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import time

import vascular_playback_monitor as base

MIB = 1024**2
ACTION_MANIFEST_BYTES = 6*MIB
CONTROL_BYTES = {'normal': 512*1024, 'producer': MIB, 'consumer': 2*MIB}
PARENT_BYTES = 512*1024
TOTAL_MANIFEST_BYTES = 16*MIB
BINDING_KEYS = frozenset({'schema', 'sceneId', 'originSnapshotHash', 'actionLogHash',
                          'protectionContractHash', 'implementation', 'prefixImplementation'})
BASELINE = base.ROOT/'testcases/outputs/rl-readiness-20261022/desktop-full-vascular-field-token-v15'
BASELINE_MANIFEST_SHA = 'df47327233c69b3090e7c3fa6aa755cad96c10aff1ea764c289c51e22edf77e7'
require = base.require


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _digest(value):
    return (type(value) is str and len(value.removeprefix('sha256:')) == 64
            and all(char in '0123456789abcdef' for char in value.removeprefix('sha256:')))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _same(left, right, message):
    require(_json(left) == _json(right), message)


def _detach(value):
    # Only retained metadata is copied. State bytes are hashed separately.
    if isinstance(value, Mapping):
        require(all(type(key) is str for key in value), 'Metadata keys must be strings')
        return {key: _detach(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_detach(item) for item in value]
    require(value is None or type(value) in (str, bool, int, float), 'Unsupported metadata type')
    require(type(value) is not float or math.isfinite(value), 'Nonfinite metadata')
    return value


def _binding(value):
    require(type(value) is dict and set(value) == BINDING_KEYS
            and value['schema'] == 'resectionlab.prepared-prefix-binding.v1'
            and all(_digest(value[key]) for key in BINDING_KEYS-{'schema'}), 'Invalid full prepared binding')
    return dict(value)


class DigestFrameSpool:
    """Bounded producer metadata only; never opens a binary output file."""
    def __init__(self, directory, *, max_manifest=ACTION_MANIFEST_BYTES):
        require(_integer(max_manifest, 1, ACTION_MANIFEST_BYTES), 'Invalid digest manifest cap')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = (self.directory/'observed-actions.jsonl').open('xb')
        self.max_manifest = max_manifest
        self.manifest_size = self.actions = self.frames = 0
        self.seconds = 0.
        self.owner_metadata = None

    def append(self, index, frames, receipt):
        require(type(index) is int and index == self.actions and index < 73, 'Digest action sequence differs')
        require(type(frames) is list and len(frames) == base.FRAME_COUNTS[index]
                and self.frames + len(frames) <= 1149, 'Digest observation count differs')
        started = time.perf_counter()
        rows = []
        for frame in frames:
            require(type(frame) is dict and 'state' in frame, 'Invalid committed frame')
            blob = frame['state']
            require(type(blob) is bytes and 0 < len(blob) <= base.MAX_RAW_STATE, 'Invalid committed compressed bytes')
            rows.append({**{key: _detach(value) for key, value in frame.items() if key != 'state'},
                         'blob': {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}})
        record = {'actionIndex': index, 'owner': self.owner_metadata,
                  'receipt': _detach(receipt), 'frames': rows}
        encoded = _json(record) + b'\n'
        require(self.manifest_size + len(encoded) <= self.max_manifest, 'Producer manifest cap exceeded')
        # Check the complete metadata budget before any append. No state bytes
        # or complete action graph survives the owning write.
        self.manifest.write(encoded); self.manifest.flush()
        self.manifest_size += len(encoded); self.actions += 1; self.frames += len(frames)
        self.seconds += time.perf_counter() - started

    def close(self):
        self.manifest.close()

    @property
    def statistics(self):
        return {'binaryBytes': 0, 'maxBinaryBytes': 0, 'manifestBytes': self.manifest_size,
                'maxManifestBytes': self.max_manifest, 'actions': self.actions, 'frames': self.frames,
                'seconds': self.seconds, 'meaning': 'Producer committed compressed SHA/length and complete metadata only; no binary spool or decode.'}


class _NoActionSpool:
    def __init__(self, directory):
        self.actions = self.frames = 0
        self.owner_metadata = None

    def append(self, *_):
        raise ValueError('Ordinary bootstrap cannot capture an action')

    def close(self):
        pass

    @property
    def statistics(self):
        return {'binaryBytes': 0, 'manifestBytes': 0, 'actions': 0, 'frames': 0,
                'maxBinaryBytes': 0, 'maxManifestBytes': 0, 'seconds': 0.}


class PipelineVascularMonitor(base.VascularPlaybackMonitor):
    def __init__(self, engine_base, output, *, role, reference=None, clock=time.monotonic,
                 rss=base.peak_rss, _json_adapter=None):
        require(role in CONTROL_BYTES, 'Unknown vascular pipeline role')
        self.role = role
        self.full_binding = None
        self.store_statistics = None
        self.native_identities = None
        self.next_cursor = (0, 0)
        self.last_cursor = None
        factory = (_NoActionSpool if role == 'normal' else DigestFrameSpool if role == 'producer'
                   else lambda directory: base.FrameSpool(directory, max_bytes=base.MAX_SPOOL,
                                                         max_manifest=ACTION_MANIFEST_BYTES))
        super().__init__(engine_base, Path(output)/'roles'/role, reference=reference, clock=clock,
                         rss=rss, prepared_only=role == 'consumer', _json_adapter=_json_adapter,
                         _spool_factory=factory)

    def resources(self):
        super().resources()
        # Context/counter rows have a fixed shape and retain no request payload.
        # Bound repetition before it could grow the final control graph without
        # repeatedly serializing the entire command history on every hot read.
        maximum = {'normal': 16, 'producer': 76, 'consumer': 1200}[self.role]
        require(len(self.commands) < maximum, 'Role command evidence limit exceeded')

    def admit_command(self, operation, args):
        if operation == 'surgicalRehearsal':
            command = args.get('command')
            if self.role == 'normal':
                require(command in ('inspectArchive', 'seekPrefix'), 'Ordinary bootstrap cannot request frames')
            else:
                require(command in ('seekPrefix', 'seekPrefixFrame'), 'Role cannot inspect or edit archive')
                if command == 'seekPrefixFrame':
                    cursor = (args.get('position'), args.get('frameIndex'))
                    if self.role == 'producer':
                        require(cursor == (len(self.calls), 0)
                                and 'prefixToken' not in args and 'prefixFieldTokens' not in args,
                                'Producer must prepare exactly the next original action at frame zero')
                    else:
                        require(cursor == self.next_cursor or cursor == self.last_cursor,
                                'Consumer must read sequentially or repeat only its latest resident frame')
        elif self.role != 'normal':
            require(operation in ('loadCase', 'ping', 'shutdown'), 'Unexpected fixed-role operation')
        super().admit_command(operation, args)

    def admit_step(self, obj, action, resumed=False):
        if self.role == 'normal':
            row = dict(self.context or {}); row.update(priorActions=len(obj.actions), event='resume' if resumed else 'entry')
            self.forbidden.append(row); self.attempts['ordinaryPipelinePublicStep'] += 1
            raise base.ForbiddenPhysicsStep('Ordinary pipeline bootstrap refused public physics before execution')
        super().admit_step(obj, action, resumed)

    def capture(self, session):
        owned = getattr(session, 'surgical_prefix_session', None)
        if owned:
            owner = owned[3]
            actual = _binding(_detach(owner._prepared_binding))
            expected = {'schema': 'resectionlab.prepared-prefix-binding.v1', 'sceneId': owner._scene.identity,
                        'originSnapshotHash': owner._origin_hash, 'actionLogHash': owner._action_hash,
                        'protectionContractHash': owner._protection_hash,
                        'implementation': owner._implementation, 'prefixImplementation': owner._prefix_implementation}
            _same(actual, expected, 'Prepared binding differs from actual owner')
            if self.full_binding is not None:
                _same(actual, self.full_binding, 'Role prepared binding changed')
            self.full_binding = actual
            store = owner._prepared_store
            require(store is not None, 'Role prepared store unavailable')
            stored = _detach(store.statistics)
            verified = len(owner._history)
            for key in ('diskBytes', 'maxDiskBytes', 'maxActionBytes', 'hits', 'misses', 'publications'):
                require(_integer(stored.get(key), 0, 2**63-1), 'Invalid actual store counter')
            require(0 < stored['maxDiskBytes'] <= 512*MIB and stored['diskBytes'] <= stored['maxDiskBytes']
                    and stored['maxActionBytes'] == 64*MIB and stored['misses'] == 0
                    and stored['lastSkip'] is None, 'Actual role store budget/skip differs')
            producer = self.role == 'producer'
            require(stored['hits'] == (verified if self.role == 'consumer' else 0)
                    and stored['publications'] == (verified if producer else 0),
                    'Role must perform exact fresh writes/loads, not accepted existing publications')
            if self.role == 'normal':
                require(verified == 0, 'Ordinary bootstrap advanced physics/history')
            if self.store_statistics is not None:
                for key in ('directory', 'namespaceDirectory', 'maxDiskBytes', 'maxActionBytes'):
                    _same(stored[key], self.store_statistics[key], 'Role store configuration changed')
            self.store_statistics = stored
        super().capture(session)
        if self.owner_metadata is not None:
            self.owner_metadata['preparedBinding'] = self.full_binding
        if self.role == 'consumer' and self.context and self.context.get('command') == 'seekPrefixFrame':
            cursor = self.context['position'], self.context['frameIndex']
            if cursor != self.last_cursor:
                require(cursor == self.next_cursor, 'Consumer capture cursor differs')
                self.last_cursor = cursor
                action, frame = cursor
                self.next_cursor = ((action + 1, 0) if frame + 1 == base.FRAME_COUNTS[action]
                                    else (action, frame + 1))

    def receipt(self):
        result = super().receipt()
        result.update(pipelineRole=self.role, pipelineSchema='resectionlab.vascular-pipeline-role.v1',
                      workerPid=os.getpid(), parentPid=os.getppid(), caseHash=self.case_hash,
                      preparedBinding=self.full_binding, actualStore=self.store_statistics,
                      nativeIdentities=self.native_identities, maxControlBytes=CONTROL_BYTES[self.role])
        require(len(json.dumps(result, indent=2, allow_nan=False).encode()) + 1 <= CONTROL_BYTES[self.role],
                'Role control manifest cap exceeded')
        return result

    def finish(self):
        failure = None
        try:
            from resectionlab import surgical_prefix_json_native, surgical_idle_native
            self.native_identities = {'prefixJson': surgical_prefix_json_native.strict_live_identity(full=True),
                                      'idle': surgical_idle_native.strict_live_identity()}
        except BaseException as error:
            failure = error
            self.failure = {'type': type(error).__name__, 'message': str(error), 'finalNativeIdentity': True}
        super().finish()
        if failure is not None:
            raise failure


def _read_json(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'Invalid bounded metadata file: '+str(path))
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'Metadata grew beyond cap')
    return json.loads(raw)


def _rows(path, limit, check):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'Invalid action manifest file')
    with path.open('rb') as stream:
        used = 0
        while True:
            check()
            line = stream.readline(limit + 1)
            if not line:
                break
            used += len(line)
            require(used <= limit and line.endswith(b'\n'), 'Action manifest cap or incomplete line')
            yield json.loads(line)


def _owner_pair(current, other, baseline=False):
    require(type(current) is dict and type(other) is dict, 'Missing manifest owner')
    if not baseline:
        require(current.get('observationOrigin') == 'fresh-public-actions'
                and other.get('observationOrigin') == 'prepared-cache', 'Role observation origin differs')
        _same({key: value for key, value in current.items() if key != 'observationOrigin'},
              {key: value for key, value in other.items() if key != 'observationOrigin'}, 'Producer/consumer owner binding differs')
        return {}
    allowed = {'observationOrigin', 'preparedCacheDirectory', 'preparedCacheNamespace', 'preparedBinding', 'verification'}
    _same({key: value for key, value in current.items() if key not in allowed},
          {key: value for key, value in other.items() if key not in allowed}, 'Baseline numerical/source owner differs')
    require(set(current) == set(other) | {'preparedBinding'}, 'Unexpected owner-domain difference')
    require(current['observationOrigin'] == other['observationOrigin'] == 'fresh-public-actions', 'Baseline origin differs')
    require(type(current['verification']) is dict and type(other['verification']) is dict
            and set(current['verification']) == set(other['verification']), 'Baseline verification domains differ')
    for key in current['verification']:
        if key in ('implementation', 'prefixImplementation'):
            require(_digest(current['verification'][key]) and _digest(other['verification'][key]), 'Invalid bound implementation domain')
        else:
            _same(current['verification'][key], other['verification'][key], 'Baseline ancestry differs')
    return {'implementationDomains': {key: {'current': current['verification'][key], 'baseline': other['verification'][key]}
                                      for key in ('implementation', 'prefixImplementation')},
            'preparedCacheDirectory': {'current': current['preparedCacheDirectory'], 'baseline': other['preparedCacheDirectory']},
            'meaning': 'Implementation and disposable cache-owner domains differ after bound source changes; no identity exemption or cache compatibility.'}


def _empty_envelope_fallback(producer, consumer, index, scene_id):
    """Prove the publisher's whole-action optional-envelope budget fallback.

    Only an empty, untruncated envelope may disappear. The complete physical
    descriptor and bytes still compare exactly below; retained flow blocks also
    compare exactly. Reproduce the publisher's metadata admission without state
    bytes, a decoder, an engine, or access to its mutable cache directory.
    """
    from resectionlab.surgical_prefix_cache import _json as cache_json, MAX_MANIFEST_BYTES
    from resectionlab.surgical_flow_display import validate_flow_display
    omitted = []
    for ordinal, (left, right) in enumerate(zip(producer, consumer)):
        if 'flowDisplay' not in left or 'flowDisplay' in right:
            continue
        flow = left['flowDisplay']; previous = producer[max(0, ordinal-1)]
        validate_flow_display(flow, scene_id=scene_id, action_index=index, frame_index=ordinal,
            before_state_id=previous['stateId'], state_id=left['stateId'],
            start_seconds=previous['observedTimeSeconds'], end_seconds=left['observedTimeSeconds'])
        require(flow['operations'] == [] and flow['truncated'] is False,
                'Prepared omission lost flow operations or an incomplete interval')
        omitted.append({key: value for key, value in flow.items() if key not in ('schema', 'sceneId', 'operations', 'truncated')})
    if not omitted:
        return []
    require(all('flowDisplay' not in frame for frame in consumer),
            'Prepared envelope omission must match whole-action fallback')
    def admit(frames):
        budget = [MAX_MANIFEST_BYTES, 0]
        for frame in frames:
            cache_json({key: value for key, value in frame.items() if key != 'blob'}, _budget=budget)
    try:
        admit(producer)
    except ValueError as error:
        require(str(error) == 'Prepared metadata exceeds manifest budget',
                'Prepared envelope omission has no matching metadata byte-budget cause')
    else:
        raise ValueError('Prepared envelope omission without publisher metadata budget exhaustion')
    admit(consumer)
    return omitted


def compare_manifests(producer, consumer, baseline, *, check=lambda: None, _frame_counts=base.FRAME_COUNTS):
    """Metadata only. Never opens, decodes or imports a physical state stream."""
    streams = [_rows(producer, ACTION_MANIFEST_BYTES, check), _rows(consumer, ACTION_MANIFEST_BYTES, check),
               _rows(baseline, base.MAX_MANIFEST, check)]
    observations = 0; owner_difference = None; owners = None; omissions = []
    for index, count in enumerate(_frame_counts):
        records = [next(stream, None) for stream in streams]
        require(all(type(row) is dict and type(row.get('actionIndex')) is int and row['actionIndex'] == index
                    and set(row) == {'actionIndex', 'owner', 'receipt', 'frames'} for row in records), 'Manifest action sequence differs')
        p, c, old = records
        if owners is None:
            owners = [row['owner'] for row in records]
            _owner_pair(p['owner'], c['owner'])
            _binding(p['owner']['preparedBinding'])
            owner_difference = _owner_pair(p['owner'], old['owner'], True)
        else:
            for row, owner in zip(records, owners):
                _same(row['owner'], owner, 'Manifest owner changed within role')
        _same(p['receipt'], c['receipt'], 'Complete producer/consumer public receipt differs')
        _same(p['receipt'], old['receipt'], 'Complete baseline public receipt differs')
        require(all(type(row['frames']) is list and len(row['frames']) == count for row in records), 'Frame count differs')
        omitted = _empty_envelope_fallback(p['frames'], c['frames'], index, p['owner']['sceneId'])
        omissions.extend(omitted)
        for frames in zip(*(row['frames'] for row in records)):
            check(); normalized = []
            for frame, producer_row in zip(frames, (True, False, False)):
                require(type(frame) is dict and type(frame.get('blob')) is dict, 'Invalid frame metadata')
                blob = frame['blob']
                require(set(blob) == ({'sha256', 'bytes'} if producer_row else {'sha256', 'bytes', 'offset'})
                        and type(blob['sha256']) is str and len(blob['sha256']) == 64 and _digest(blob['sha256'])
                        and _integer(blob['bytes'], 1, base.MAX_RAW_STATE), 'Invalid compressed digest/length')
                if not producer_row:
                    require(_integer(blob['offset'], 0, base.MAX_SPOOL-1), 'Invalid spool offset')
                normalized.append({**{key: value for key, value in frame.items() if key != 'blob'},
                                   'blob': {'sha256': blob['sha256'], 'bytes': blob['bytes']}})
            consumer_expected = normalized[0]
            if omitted and 'flowDisplay' in consumer_expected and 'flowDisplay' not in normalized[1]:
                consumer_expected = {key: value for key, value in consumer_expected.items() if key != 'flowDisplay'}
            _same(consumer_expected, normalized[1], 'Producer/consumer full descriptor or compressed bytes differ')
            _same(normalized[0], normalized[2], 'Baseline full descriptor or compressed bytes differ')
            observations += 1
    require(all(next(stream, None) is None for stream in streams), 'Extra action manifest rows')
    return {'actions': len(_frame_counts), 'observations': observations,
            'allCompleteReceiptsDescriptorsFlowAndCompressedShaLengthsExact': not omissions,
            'allCompleteReceiptsPhysicalDescriptorsFlowOperationsAndCompressedShaLengthsExact': True,
            'rawProducerConsumerFullDescriptorsExact': not omissions,
            'preparedEmptyEnvelopeOmissions': {'count': len(omissions), 'intervals': omissions,
                'meaning': 'Exact empty untruncated interval envelopes omitted by independently reproduced whole-action prepared metadata budget fallback; no physical descriptor, operation, receipt or compressed-byte exemption.'},
            'ownerDomainDifferences': owner_difference, 'producerOwner': owners[0],
            'consumerOwner': owners[1], 'stateDecodes': 0}


def _clean_native(guard):
    for key in ('nativeJson', 'nativeCanonicalJson'):
        counter = guard.get(key)
        require(type(counter) is dict and counter.get('started') is True and counter.get('active') is False
                and counter.get('failure') is None and counter.get('cleanupFailure') is None
                and _integer(counter.get('total', {}).get('admittedStrings'), 1, 2**63-1), 'Native admission counter incomplete')
    identities = guard.get('nativeIdentities')
    require(type(identities) is dict and set(identities) == {'idle', 'prefixJson'}
            and all(type(value) is dict and value.get('mode') == 'native' for value in identities.values()), 'Actual selected native identities absent')


def check_parent_proof(proof, guards, report, *, _frame_counts=base.FRAME_COUNTS):
    """Validate bounded parent scheduling/exit and renderer evidence; no decode."""
    total = len(_frame_counts); expected_frames = sum(_frame_counts)
    require(type(proof) is dict and proof.get('schema') == 'resectionlab.vascular-pipeline-parent-proof.v1'
            and proof.get('passed') is True and proof.get('qualificationMode') == 'fresh-pipeline'
            and proof.get('branchId') == base.BRANCH and proof.get('sourceArchiveSha256') == base.NATIVE_SHA
            and _digest(proof.get('caseHash')) and _integer(proof.get('mainPid'), 1, 2**63-1), 'Invalid pipeline parent proof')
    expected_binding = _binding(proof.get('binding'))
    require(type(guards) is dict and set(guards) == set(CONTROL_BYTES), 'Incomplete role guards')
    workers = proof.get('workers')
    require(type(workers) is list and len(workers) == 3
            and {worker.get('role') for worker in workers} == set(CONTROL_BYTES)
            and all(_integer(worker.get('pid'), 1, 2**63-1) and worker['pid'] != proof['mainPid'] and worker.get('exited') is True
                    and type(worker.get('exitCode')) is int and worker['exitCode'] == 0
                    and worker.get('signal') is None for worker in workers)
            and len({worker['pid'] for worker in workers}) == 3, 'Every actual worker exit must be proven before audit')
    resource = proof.get('resources', {})
    require(resource.get('closed') is True and resource.get('failure') is None
            and type(resource.get('maxAlive')) is int and resource['maxAlive'] == 2
            and _integer(resource.get('peakAggregatePythonTreeRssBytes'), 1, base.MAX_RSS)
            and type(resource.get('maxWallMs')) is int and resource['maxWallMs'] == 600000
            and type(resource.get('wallMs')) in (int, float) and math.isfinite(resource['wallMs'])
            and 0 <= resource['wallMs'] < 600000, 'Common pipeline resource budget differs')
    for role, guard in guards.items():
        require(guard.get('pipelineRole') == role and guard.get('pipelineSchema') == 'resectionlab.vascular-pipeline-role.v1'
                and guard.get('failure') is None and guard.get('forbiddenSteps') == []
                and guard.get('publicStepAttempts') == {}, 'Role guard failed or wrong role')
        require(guard.get('caseHash') == proof['caseHash']
                and _integer(guard.get('workerPid'), 1, 2**63-1)
                and guard['workerPid'] == next(worker['pid'] for worker in workers if worker['role'] == role)
                and _integer(guard.get('parentPid'), 1, 2**63-1) and guard['parentPid'] == proof['mainPid']
                and guard.get('qualificationMode') == ('prepared-only' if role == 'consumer' else 'fresh')
                and type(guard.get('processPeakResidentBytes')) is int
                and 0 < guard['processPeakResidentBytes'] <= base.MAX_RSS
                and type(guard.get('wallSeconds')) in (int, float) and math.isfinite(guard['wallSeconds'])
                and 0 <= guard['wallSeconds'] < 600, 'Role PID/mode/resources differ')
        _same(guard.get('preparedBinding'), expected_binding, 'Normal/role full prepared binding differs')
        _clean_native(guard)
        ownership = guard.get('ownership', {})
        require(all(ownership.get(key) is False for key in ('canonicalSessionCreated', 'replayCacheCreated', 'frameCacheCreated', 'frameAdmissionsCreated')), 'Role gained canonical authority')
        stats = ownership.get('prefixStatistics', {}); store = guard.get('actualStore', {})
        verified = total if role != 'normal' else 0
        cache = stats.get('preparedCache', {})
        require(type(stats.get('verifiedThrough')) is int and stats['verifiedThrough'] == verified
                and type(stats.get('totalActions')) is int and stats['totalActions'] == total, 'Role history incomplete')
        expected_cache = {'mode': 'only' if role == 'consumer' else 'prepare',
                          'loadedActions': total if role == 'consumer' else 0,
                          'publishedActions': total if role == 'producer' else 0,
                          'misses': 0, 'lastSkip': None}
        _same(cache, expected_cache, 'Role cache counters incomplete')
        require(store.get('hits') == expected_cache['loadedActions'] and type(store.get('hits')) is int
                and store.get('publications') == expected_cache['publishedActions'] and type(store.get('publications')) is int
                and type(store.get('misses')) is int and store['misses'] == 0 and store.get('lastSkip') is None,
                'Actual store writes/loads incomplete')
        calls = guard.get('permittedSteps')
        require(type(calls) is list and len(calls) == (total if role == 'producer' else 0), 'Wrong role public action count')
        spool = guard.get('spool', {})
        require(type(spool.get('actions')) is int and spool['actions'] == verified
                and type(spool.get('frames')) is int and spool['frames'] == (expected_frames if verified else 0)
                and _integer(spool.get('manifestBytes'), 0, ACTION_MANIFEST_BYTES)
                and _integer(spool.get('binaryBytes'), 0, base.MAX_SPOOL)
                and (role == 'consumer' or spool['binaryBytes'] == 0), 'Wrong role spool inventory')
        for i, call in enumerate(calls):
            require(call.get('operation') == 'surgicalRehearsal' and call.get('command') == 'seekPrefixFrame'
                    and call.get('branchId') == base.BRANCH and call.get('caseHash') == proof['caseHash']
                    and type(call.get('position')) is int and call['position'] == i
                    and type(call.get('frameIndex')) is int and call['frameIndex'] == 0
                    and type(call.get('priorActions')) is int and call['priorActions'] == i
                    and call.get('event') == 'entry', 'Producer original public entry ledger differs')
    for key in ('directory', 'namespaceDirectory', 'maxDiskBytes', 'maxActionBytes'):
        _same(guards['normal']['actualStore'][key], guards['producer']['actualStore'][key], 'Normal/producer store differs')
        _same(guards['producer']['actualStore'][key], guards['consumer']['actualStore'][key], 'Producer/consumer store differs')
    _same(guards['normal']['nativeIdentities'], guards['producer']['nativeIdentities'], 'Normal/producer selected native identities differ')
    _same(guards['producer']['nativeIdentities'], guards['consumer']['nativeIdentities'], 'Producer/consumer selected native identities differ')
    leases = proof.get('leases')
    require(type(leases) is list and len(leases) == 2*total, 'Parent first-load/preparation ledger incomplete')
    indexed = {}
    for row in leases:
        require(type(row) is dict and row.get('role') in ('producer', 'consumer')
                and _integer(row.get('index'), 0, total-1) and type(row.get('requestId')) is str
                and 1 <= len(row['requestId']) <= 256, 'Invalid parent lease identity')
        key = row['role'], row['index']; require(key not in indexed, 'Duplicate parent action lease')
        require(all(type(row.get(field)) in (int, float) and math.isfinite(row[field]) for field in ('start', 'end'))
                and 0 <= row['start'] <= row['end'], 'Invalid parent timing')
        i = row['index']; producer = row['role'] == 'producer'
        require(type(row.get('verifiedThrough')) is int and row['verifiedThrough'] == i+1
                and _integer(row.get('actionFrameBytes'), 1, 64*MIB), 'Parent role frontier/staging cap differs')
        _same(row.get('cache'), {'mode': 'prepare' if producer else 'only', 'loadedActions': 0 if producer else i+1,
                                'publishedActions': i+1 if producer else 0, 'misses': 0, 'lastSkip': None}, 'Parent cache ledger differs')
        _same(row.get('store'), {'hits': 0 if producer else i+1, 'publications': i+1 if producer else 0,
                                'misses': 0, 'lastSkip': None}, 'Parent actual write ledger differs')
        indexed[key] = row
    require(len({row['requestId'] for row in leases}) == len(leases), 'Parent lease IDs not unique')
    for i in range(total):
        require(indexed['producer', i]['end'] <= indexed['consumer', i]['start'], 'Consumer STORE load overlaps producer preparation')
        if i+1 < total:
            require(indexed['consumer', i]['end'] <= indexed['producer', i+1]['start'], 'Producer exceeded one-action lookahead')
    require(type(report) is dict and report.get('qualificationMode') == 'fresh-pipeline', 'GUI report mode differs')
    require(all(type(report.get(key)) in (int, float) and math.isfinite(report[key])
                for key in ('playStarted', 'playEnded', 'playStartedRenderer', 'playEndedRenderer'))
            and 0 <= report['playStarted'] <= report['playEnded']
            and report['playEnded']-report['playStarted'] < 600000
            and report['playStartedRenderer'] <= report['playEndedRenderer']
            and all(report['playStarted'] <= row['start'] <= row['end'] <= report['playEnded'] for row in leases),
            'Startup/action-zero must be included in click-to-completion timing')
    completed = report.get('allCompletedPlaybackIpc'); receipts = report.get('rendererReceipts'); submitted = report.get('submittedPlayback')
    require(all(type(rows) is list and len(rows) == expected_frames for rows in (completed, receipts, submitted)), 'GUI tuple inventory incomplete')
    tuple_lists = [[[row.get(key) for key in ('branchId', 'position', 'frameIndex', 'stateId')] for row in rows]
                   for rows in (completed, receipts, submitted)]
    require(hashlib.sha256(json.dumps(tuple_lists, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            == proof.get('rendererTupleSha256'), 'Parent renderer tuple hash differs')
    expected = [(base.BRANCH, action, frame) for action, count in enumerate(_frame_counts) for frame in range(count)]
    prior_receipt = -math.inf
    for serial, (cursor, rows) in enumerate(zip(expected, zip(completed, receipts, submitted)), 1):
        ipc, receipt, shown = rows
        require(all(type(row.get('position')) is int and type(row.get('frameIndex')) is int
                    and tuple(row.get(key) for key in ('branchId', 'position', 'frameIndex')) == cursor
                    and _digest(row.get('stateId')) and not row.get('unmatched') for row in rows), 'Wrong ordered GUI tuple')
        _same([row['stateId'] for row in rows], [ipc['stateId']]*3, 'GUI state tuple differs')
        require(ipc.get('command') == receipt.get('command') == 'seekPrefixFrame' and not ipc.get('error')
                and type(receipt.get('notificationSerial')) is int and receipt['notificationSerial'] == serial
                and shown.get('receiptNotificationSerial') == serial
                and type(receipt.get('receivedAt')) in (int, float) and math.isfinite(receipt['receivedAt'])
                and type(shown.get('firstSubmittedAt')) in (int, float) and math.isfinite(shown['firstSubmittedAt'])
                and shown.get('rendererReceivedAt') == receipt['receivedAt'] <= shown['firstSubmittedAt'], 'Receipt ordering differs')
        require(report['playStartedRenderer'] <= receipt['receivedAt'] <= shown['firstSubmittedAt'] <= report['playEndedRenderer']
                and receipt['receivedAt'] >= prior_receipt, 'Receipt renderer clock moved backwards/outside playback')
        prior_receipt = receipt['receivedAt']
    return {'roles': 3, 'producerFreshActions': total, 'consumerPublicSteps': 0, 'ordinaryPublicSteps': 0,
            'exactOrderedRendererTuples': expected_frames, 'allActualWorkerExitsProven': True,
            'maxAlive': 2, 'aggregatePeakPythonBytes': resource['peakAggregatePythonTreeRssBytes']}


def audit_pipeline(output, *, baseline=BASELINE, audit_output=None):
    """One consumer numerical audit, then no-decode producer/v15 agreement."""
    output = Path(output); baseline = Path(baseline)
    audit_output = Path(audit_output) if audit_output is not None else output
    audit_output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    def check():
        require(time.monotonic()-began < 180 and base.peak_rss() <= base.MAX_RSS, 'Pipeline audit wall/RSS budget exceeded')
    check()
    require(baseline.resolve() == BASELINE.resolve()
            and base.file_sha(baseline/'observed-actions.jsonl') == BASELINE_MANIFEST_SHA, 'Preserved v15 metadata changed')
    proof_path = output/'pipeline-parent-proof.json'
    proof = _read_json(proof_path, PARENT_BYTES)
    guards = {role: _read_json(output/'roles'/role/'physics-guard.json', limit) for role, limit in CONTROL_BYTES.items()}
    control_bytes = proof_path.stat().st_size + sum((output/'roles'/role/'physics-guard.json').stat().st_size for role in CONTROL_BYTES)
    action_bytes = 0
    for role in ('producer', 'consumer'):
        manifest = output/'roles'/role/'observed-actions.jsonl'
        require(manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size <= ACTION_MANIFEST_BYTES,
                'Role action manifest cap exceeded')
        action_bytes += manifest.stat().st_size
    require(control_bytes <= 4*MIB and action_bytes + control_bytes <= TOTAL_MANIFEST_BYTES, 'Combined pipeline manifest cap exceeded')
    for role in ('producer', 'normal'):
        require(not (output/'roles'/role/'observed-states.bin').exists(), 'Only consumer may own a binary spool')
    require(not (output/'roles'/'normal'/'observed-actions.jsonl').exists(), 'Ordinary cannot own an action manifest')
    binary = output/'roles'/'consumer'/'observed-states.bin'
    require(binary.is_file() and not binary.is_symlink() and 0 < binary.stat().st_size <= base.MAX_SPOOL, 'Consumer binary spool cap exceeded')
    report = _read_json(output/'report.json', 32*MIB)
    parent = check_parent_proof(proof, guards, report)
    check()
    # Existing reader's truthful prepared-only mode/origin remain unchanged.
    # Actual fresh producer authority comes from the independent role proof and
    # byte/metadata equality, never by relabelling consumer cache observations.
    require(guards['consumer'].get('qualificationMode') == 'prepared-only', 'Consumer audit mode must stay prepared-only')
    audit = base.audit_spool(output/'roles'/'consumer', audit_output=audit_output)
    require(audit.get('passed') is True and audit.get('publicStepAttempts') == {}
            and audit.get('allFullCanonicalRecordsExact') is True
            and audit.get('actions') == 73 and audit.get('observations') == 1149, 'Consumer numerical audit incomplete')
    check()
    audited = _read_json(audit_output/'audited-observations.json', 2*MIB)
    require(type(audited) is list and len(audited) == 1149, 'Audited tuple inventory incomplete')
    tuple_keys = ('branchId', 'position', 'frameIndex', 'stateId')
    _same([[row.get(key) for key in tuple_keys] for row in audited],
          [[row.get(key) for key in tuple_keys] for row in report['submittedPlayback']],
          'Actual renderer state tuples differ from independent consumer audit')
    comparison = compare_manifests(output/'roles'/'producer'/'observed-actions.jsonl',
                                   output/'roles'/'consumer'/'observed-actions.jsonl',
                                   baseline/'observed-actions.jsonl', check=check)
    _same(comparison['producerOwner'], guards['producer']['ownerMetadata'], 'Producer manifest/control owner differs')
    _same(comparison['consumerOwner'], guards['consumer']['ownerMetadata'], 'Consumer manifest/control owner differs')
    _same(comparison['producerOwner']['preparedBinding'], proof['binding'], 'Manifest prepared binding differs from normal source')
    result = {'schema': 'resectionlab.vascular-pipeline-audit.v1', 'passed': True,
              'qualificationMode': 'fresh-pipeline', 'parent': parent, 'comparison': comparison,
              'consumerNumericalAuditInvocations': 1, 'consumerAuditMode': 'prepared-only',
              'manifestBytes': {'controls': control_bytes, 'actions': action_bytes, 'combined': control_bytes+action_bytes,
                                'maximum': TOTAL_MANIFEST_BYTES},
              'wallSeconds': time.monotonic()-began, 'maxWallSeconds': 180, 'peakResidentBytes': base.peak_rss(),
              'parentProofSha256': base.file_sha(proof_path),
              'producerManifestSha256': base.file_sha(output/'roles'/'producer'/'observed-actions.jsonl'),
              'consumerManifestSha256': base.file_sha(output/'roles'/'consumer'/'observed-actions.jsonl'),
              'baselineManifestSha256': BASELINE_MANIFEST_SHA,
              'meaning': 'Guarded fresh producer -> exact compressed bytes, physical descriptors, receipts and flow operations -> one no-step consumer numerical audit. Any proven empty display-envelope omission is reported explicitly; raw complete-descriptor equality is separate. No canonical/cache admission or physical replay.'}
    check(); base.atomic_json(audit_output/'pipeline-audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--audit-output', type=Path, help='Separate results directory; preserve original run/audit evidence')
    args = parser.parse_args()
    try:
        audit_pipeline(args.output, baseline=args.baseline, audit_output=args.audit_output)
    except BaseException as error:
        base.atomic_json((args.audit_output or args.output)/'pipeline-audit.json', {'passed': False, 'error': repr(error), 'peakResidentBytes': base.peak_rss()})
        raise
