"""Qualification-only full vascular read-only prefix guard and post-run audit.

No engine/cache function is replaced. execute is wrapped only to reject commands,
record ownership, and copy already COMMITTED compressed frames to an audit spool.
The monitor never supplies state to the application or promotes a cache entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time
import zlib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'testcases/outputs/full-case-rehearsal-20261020/review'
sys.path.insert(0, str(REVIEW))
from readonly_step_monitor import ForbiddenPhysicsStep, ReadOnlyStepMonitor

BRANCH = 'full-vascular'
NATIVE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab'
NATIVE_SHA = '095659e54f381355e8f5655b6c5907990b2ac4767dc9ce58297e3bc4d6f3268d'
REFERENCE = NATIVE.parent / 'vascular-canonical.actions.jsonl'
FRAME_COUNTS = (3,152,152,117,2,2,2,15,2,2,2,48,2,2,2,11,33,36,35,3,3,3,3,48,3,3,3,3,3,52,50,3,3,3,3,3,52,31,3,3,3,13,3,3,13,3,3,7,3,12,27,3,3,3,7,17,3,27,3,3,3,7,12,3,3,3,3,7,12,3,3,27,3)
END_STATE = 'sha256:af496ffbde3ca2ab71477c9226fe9f846f08171efd308362431414584cd3c87e'
MAX_SECONDS = 600
MAX_RSS = 3 * 1024**3
MAX_SPOOL = 1024**3
MAX_MANIFEST = 16 * 1024**2
MAX_RAW_STATE = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_json(value):
    # Preserve JSON type distinctions (not Python's bool/int equality).
    from resectionlab.core import freeze_json, thaw_json
    return json.dumps(thaw_json(freeze_json(value)),sort_keys=True,separators=(',',':'),allow_nan=False)


def peak_rss():
    # macOS reports bytes; Linux reports KiB. This is the whole Python process.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def atomic_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def initialize_guard_inventory():
    # Define finite adapters used by this workflow BEFORE freezing inventory.
    # This matches the prior qualified runner's archive-view preimport. Merely
    # importing these modules executes no engine action or director/policy.
    from resectionlab import surgical_prefix_replay, surgical_frame_replay, surgical_telemetry, surgical_archive_view
    return (surgical_prefix_replay, surgical_frame_replay, surgical_telemetry, surgical_archive_view)


class FrameSpool:
    """One bounded binary file; SHA deduplication is confined to this run."""
    def __init__(self, directory, *, max_bytes=MAX_SPOOL, max_manifest=MAX_MANIFEST):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.data = (self.directory / 'observed-states.bin').open('xb')
        self.manifest = (self.directory / 'observed-actions.jsonl').open('xb')
        self.max_bytes, self.max_manifest = max_bytes, max_manifest
        self.size = self.manifest_size = self.frames = self.actions = 0
        self.blobs = {}
        self.seconds = 0.
        self.owner_metadata = None

    def append(self, index, frames, receipt):
        from resectionlab.core import thaw_json
        require(index == self.actions, 'Spool actions are not contiguous')
        require(self.frames + len(frames) <= sum(FRAME_COUNTS), 'Unexpected extra observations')
        started = time.perf_counter()
        rows, pending = [], {}
        proposed_size = self.size
        for frame in frames:
            blob = frame['state']
            require(type(blob) is bytes and 0 < len(blob) <= MAX_RAW_STATE, 'Invalid compressed committed frame')
            digest = hashlib.sha256(blob).hexdigest()
            location = self.blobs.get(digest) or (pending.get(digest) or (None,None))[0]
            if location is None:
                require(proposed_size + len(blob) <= self.max_bytes, 'One GiB audit spool exceeded')
                location = {'offset': proposed_size, 'bytes': len(blob), 'sha256': digest}
                proposed_size += len(blob)
                pending[digest] = (location, blob)
            rows.append({**{key: thaw_json(value) for key,value in frame.items() if key != 'state'},
                         'blob': location})
        record = {'actionIndex': index, 'owner':self.owner_metadata, 'receipt': thaw_json(receipt), 'frames': rows}
        encoded = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
        require(self.manifest_size + len(encoded) <= self.max_manifest, 'Audit manifest retention exceeded')
        # Both budgets are admitted before the first owning file write.
        for digest,(location,blob) in pending.items():
            self.data.write(blob)
            self.blobs[digest] = location
        self.size = proposed_size
        self.manifest.write(encoded)
        self.data.flush(); self.manifest.flush()
        self.manifest_size += len(encoded)
        self.actions += 1; self.frames += len(frames)
        self.seconds += time.perf_counter() - started

    def close(self):
        self.data.close(); self.manifest.close()

    @property
    def statistics(self):
        return {'binaryBytes': self.size, 'maxBinaryBytes': self.max_bytes,
                'manifestBytes': self.manifest_size, 'maxManifestBytes': self.max_manifest,
                'actions': self.actions, 'frames': self.frames, 'uniqueCompressedBlobs': len(self.blobs),
                'seconds': self.seconds, 'meaning': 'Within-run SHA deduplication of committed compressed observations and full public receipts. Copy/hash/write overhead is INCLUDED in GUI IPC/run timing; no cache authority.'}


class NativeJsonAdmissionCounter:
    """Local normal-return observations; no payloads or extra encoder calls."""
    def __init__(self, adapter):
        self.adapter, self.function = adapter, adapter.try_encode
        self.code = self.function.__code__
        self.monitor = sys.monitoring
        self.events, self.return_event = self.monitor.events, self.monitor.events.PY_RETURN
        self.api = {key:getattr(self.monitor,key) for key in ('get_tool','use_tool_id','free_tool_id',
            'get_events','set_events','get_local_events','set_local_events','register_callback')}
        self.callback = self._returned
        self.callback_code = self.callback.__func__.__code__
        self.name = 'vascular-native-json-returns-' + str(id(self))
        self.tool = None
        self.started = False
        self.current = None
        self.total, self.outside = self.empty(), self.empty()
        self.failure = None

    @staticmethod
    def empty():
        return {'normalReturns':0,'admittedStrings':0,'fallbackNone':0,
                'returnedCharacters':0,'maxReturnedCharacters':0}

    def _failed(self, error):
        self.failure = {'type':type(error).__name__,'message':str(error)}

    def _returned(self, code, offset, value):
        try:
            require(code is self.code, 'Native JSON counter code changed')
            admitted = type(value) is str
            require(admitted or value is None, 'Unexpected native JSON adapter return type')
            length = len(value) if admitted else 0
            for row in (self.total, self.outside if self.current is None else self.current):
                row['normalReturns'] += 1
                row['admittedStrings' if admitted else 'fallbackNone'] += 1
                row['returnedCharacters'] += length
                row['maxReturnedCharacters'] = max(row['maxReturnedCharacters'], length)
                require(all(number <= 2**63-1 for number in row.values()), 'Native JSON counter overflow')
        except BaseException as error:
            self._failed(error)
            raise

    def __enter__(self):
        require(not self.started, 'Native JSON counter cannot be reused')
        self.started = True
        try:
            for tool in (3,2,1,0,5,4):
                if self.api['get_tool'](tool) is None:
                    try:self.api['use_tool_id'](tool,self.name)
                    except ValueError:continue
                    self.tool = tool
                    break
            require(self.tool is not None, 'No native JSON monitoring tool available')
            require(self.api['get_events'](self.tool)==0 and self.api['get_local_events'](self.tool,self.code)==0,
                    'Native JSON monitoring slot contains events')
            require(self.api['register_callback'](self.tool,self.return_event,self.callback) is None,
                    'Native JSON monitoring slot contains a callback')
            self.api['set_local_events'](self.tool,self.code,self.return_event)
            self.check()
            return self
        except BaseException as error:
            self._failed(error)
            self.close()
            raise

    def check(self):
        try:
            require(self.failure is None, 'Native JSON counter previously failed')
            require(self.monitor is getattr(sys,'monitoring',None) and self.tool is not None,
                    'Native JSON monitoring unavailable')
            require(all(getattr(self.monitor,key) is value for key,value in self.api.items()),
                    'Native JSON monitoring API changed')
            require(self.monitor.events is self.events and self.events.PY_RETURN==self.return_event,
                    'Native JSON monitoring event constants changed')
            require(self.adapter.try_encode is self.function and self.function.__code__ is self.code,
                    'Native JSON adapter function/code changed')
            require(self.callback.__func__.__code__ is self.callback_code, 'Native JSON counter callback code changed')
            require(self.api['get_tool'](self.tool)==self.name and self.api['get_events'](self.tool)==0
                    and self.api['get_local_events'](self.tool,self.code)==self.return_event,
                    'Native JSON monitoring ownership/events changed')
            require(self.api['register_callback'](self.tool,self.return_event,self.callback) is self.callback,
                    'Native JSON monitoring callback changed')
        except BaseException as error:
            if self.failure is None:self._failed(error)
            raise

    def begin_command(self):
        self.check()
        previous = self.current
        self.current = self.empty()
        return previous

    def end_command(self, previous):
        result = dict(self.current)
        self.current = previous
        return result

    def close(self):
        tool, self.tool = self.tool, None
        if tool is not None and self.api['get_tool'](tool)==self.name:
            try:
                self.api['set_local_events'](tool,self.code,0)
                self.api['set_events'](tool,0)
                self.api['register_callback'](tool,self.return_event,None)
            finally:self.api['free_tool_id'](tool)

    def __exit__(self, kind, value, traceback):
        try:
            try:self.check()
            except BaseException:
                # check records sticky failure; preserve the original body diagnostic.
                if kind is None:raise
        finally:self.close()

    def receipt(self):
        return {'started':self.started,'active':self.tool is not None,'total':dict(self.total),
            'outsideCommand':dict(self.outside),'failure':self.failure,
            'adapterModule':getattr(self.adapter,'__name__','generated-fixture'),
            'functionModule':self.function.__module__,'functionQualname':self.function.__qualname__,
            'codeFilename':self.code.co_filename,'codeFirstLine':self.code.co_firstlineno,
            'meaning':'Normal returns from the actual loaded adapter: exact str means native admission, None means fallback. Counts are encode calls, not observed frames; no payload retained or extra encoder call. Exceptional exits are not normal-return counts. Observer cost is INCLUDED.'}


class VascularPlaybackMonitor(ReadOnlyStepMonitor):
    def __init__(self, engine_base, output, *, reference=None, clock=time.monotonic, rss=peak_rss,
                 _json_adapter=None):
        super().__init__(engine_base)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.reference = reference if reference is not None else [json.loads(line) for line in REFERENCE.read_text().splitlines()]
        require(len(self.reference) == 73, 'Exactly 73 original vascular actions required')
        self.clock, self.rss = clock, rss
        self.began = clock()
        self.context = None
        self.calls, self.forbidden, self.commands = [], [], []
        self.sessions = []
        self.case_hash = None
        self.prefix_started = False
        self.owner = None
        self.spool = FrameSpool(output)
        self.owner_metadata = None
        self.ownership = None
        self.retention_peaks = {}
        self.failure = None
        self._stop = threading.Event()
        self._watcher = None
        self.json_counter = None
        self._json_adapter = _json_adapter

    def _check_json_counter(self):
        if self.json_counter is not None:
            try:self.json_counter.check()
            except BaseException as error:
                self.failure = {'type':type(error).__name__,'message':str(error),
                                'nativeJsonCounter':True,'context':self.context}
                raise

    def resources(self):
        self._check_json_counter()
        require(self.clock() - self.began < MAX_SECONDS, '600 second qualification wall budget exceeded')
        require(self.rss() <= MAX_RSS, '3 GiB whole Python peak RSS budget exceeded')

    def admit_command(self, operation, args):
        self.resources()
        if operation in ('ping', 'shutdown'):
            return
        if operation == 'loadCase':
            require(self.case_hash is None and Path(args.get('path', '')).resolve() == NATIVE.resolve(), 'Only original native load is permitted')
            require(file_sha(NATIVE) == NATIVE_SHA, 'Original native bytes differ')
            return
        if operation == 'listRuns':
            require(not self.prefix_started and self.case_hash is not None and args.get('caseHash')==self.case_hash and set(args)=={'caseHash'},'Only initial source-bound run inventory is permitted')
            return
        require(operation == 'surgicalRehearsal' and self.case_hash is not None and args.get('caseHash') == self.case_hash,
                'Unexpected operation or source owner')
        command = args.get('command')
        if command == 'inspectArchive':
            require(not self.prefix_started and args.get('branchId') in (None, BRANCH), 'Unexpected imported branch/late inspect')
        elif command == 'seekPrefix':
            require(not self.prefix_started and args.get('branchId') == BRANCH and args.get('position') == 0,
                    'Only one explicit vascular prefix-zero start is permitted')
            self.prefix_started = True
        else:
            require(command == 'seekPrefixFrame' and self.prefix_started and args.get('branchId') == BRANCH,
                    'Only read-only vascular observed playback is permitted')
            position, index = args.get('position'), args.get('frameIndex')
            require(type(position) is int and 0 <= position < 73 and type(index) is int and 0 <= index < FRAME_COUNTS[position],
                    'Unexpected vascular observation cursor')
            require(position <= len(self.calls), 'Observed playback skipped an uncomputed action')

    def admit_step(self, obj, action, resumed=False):
        self.resources()
        row = dict(self.context or {})
        row.update(priorActions=len(obj.actions), event='resume' if resumed else 'entry')
        i = len(self.calls)
        expected = (i, 0, i)
        actual = (row.get('position'), row.get('frameIndex'), row['priorActions'])
        if (resumed or i >= 73 or row.get('operation') != 'surgicalRehearsal'
                or row.get('command') != 'seekPrefixFrame' or row.get('branchId') != BRANCH
                or row.get('caseHash') != self.case_hash or actual != expected
                or self.owner is None or obj.scene is not self.owner._scene
                or canonical_json(obj._protection_contract) != canonical_json(self.owner._template._protection_contract)
                or canonical_json(action) != canonical_json(self.reference[i]['action'])
                or canonical_json(obj.actions) != canonical_json([item['action'] for item in self.reference[:i]])):
            self.forbidden.append(row)
            self.attempts['unexpectedVascularStep'] += 1
            raise ForbiddenPhysicsStep('Unexpected, altered, resumed, or 74th vascular action refused before execution')
        self.calls.append(row)

    def _callback(self, code, offset, *extra):
        if code.co_name != 'step':
            self.other_callbacks += 1
            return None if extra else self.monitor.DISABLE
        frame = sys._getframe(1)
        require(frame.f_code is code, 'Monitoring callback callee differs')
        obj = frame.f_locals.get('self')
        if isinstance(obj, self.engine_base):
            self.admit_step(obj, frame.f_locals.get('action'), bool(extra))

    def capture(self, session):
        owned = getattr(session, 'surgical_prefix_session', None)
        captured = {'beforeClose': True,
            'canonicalSessionCreated': hasattr(session, 'surgical_sessions'),
            'replayCacheCreated': hasattr(session, 'surgical_replay_cache'),
            'frameCacheCreated': hasattr(session, 'surgical_frame_cache'),
            'frameAdmissionsCreated': hasattr(session, 'surgical_frame_admissions'),
            'prefixStatistics': owned[3].statistics if owned else None}
        require(not any(captured[key] for key in ('canonicalSessionCreated','replayCacheCreated','frameCacheCreated','frameAdmissionsCreated')),
                'Canonical session/cache/admission appeared in read-only run')
        if not owned:
            if self.ownership is None: self.ownership=captured
            return
        self.ownership=captured
        owner = owned[3]
        require(owned[2] == BRANCH and (self.owner is None or self.owner is owner), 'Prefix owner/branch changed')
        self.owner = owner
        stats = self.ownership['prefixStatistics']
        for key, maximum in (('checkpointBytes',16*1024**2),('frameBytes',64*1024**2),('historyBytes',4*1024**2),('immutableInputBytes',128*1024**2)):
            require(stats[key] <= maximum, 'Private prefix retention exceeds ' + key)
            self.retention_peaks[key] = max(self.retention_peaks.get(key, 0), stats[key])
        if self.owner_metadata is None:
            contract = owner._template._protection_contract
            self.owner_metadata = {'sceneId': owner._scene.identity,
                'protectionContractHash': contract['contentHash'] if contract else None,
                'verification': {'originSnapshotHash':owner._origin_hash,'actionLogHash':owner._action_hash,'implementation':owner._implementation,'prefixImplementation':owner._prefix_implementation},
                'expectedEndpointStateId': owner._snapshot['stateId'],
                'sourceArchiveSha256':NATIVE_SHA,'canonicalReferenceSha256':file_sha(REFERENCE)}
            self.spool.owner_metadata = self.owner_metadata
        require(len(owner._history) in (self.spool.actions, self.spool.actions + 1), 'Prefix advanced more than one action')
        if len(owner._history) > self.spool.actions:
            index = self.spool.actions
            from resectionlab.core import thaw_json
            row, expected = thaw_json(owner._receipt(index)), self.reference[index]
            for key in ('action','stateBefore','stateAfter','status','reason','reasonCode','completedFraction','elapsedSeconds','events'):
                require(canonical_json(row.get(key)) == canonical_json(expected.get(key)), f'Unprojected public receipt differs: action {index}, {key}')
            require(row['deltas']['removedCells'] == expected['removedCells'], 'Unprojected removal delta differs')
            frames = owner._frames.get(index)
            require(frames is not None and len(frames) == FRAME_COUNTS[index], 'Expected committed observation count differs')
            require(frames[0]['stateId'] == expected['stateBefore'] and frames[-1]['stateId'] == expected['stateAfter'], 'Committed observation ancestry differs')
            self.spool.append(index, frames, row)
            self.resources()

    def install(self, session_class):
        original_execute, original_close = session_class.execute, session_class.close
        guard = self
        def execute(session, operation, args, request, progress):
            try:
                guard.admit_command(operation, args)
            except BaseException as error:
                guard.failure={'type':type(error).__name__,'message':str(error),'beforeDispatch':True,'operation':operation,'command':args.get('command')}
                raise
            previous = guard.context
            guard.context = {'operation': operation, **{key: args.get(key) for key in ('command','caseHash','branchId','position','frameIndex')}}
            before, began = len(guard.calls), time.perf_counter()
            previous_json, json_started = None, False
            try:
                if guard.json_counter is not None:
                    previous_json = guard.json_counter.begin_command()
                    json_started = True
                result = original_execute(session, operation, args, request, progress)
                if operation == 'loadCase':
                    guard.case_hash = result['caseHash']
                if session not in guard.sessions:
                    guard.sessions.append(session)
                # Inspect only committed ownership, after the production command.
                if operation == 'surgicalRehearsal':
                    guard.capture(session)
                    if args.get('command') == 'seekPrefixFrame':
                        require(result.get('verification',{}).get('status') == 'reconstructed-prefix' and result['verification']['readOnly'] is True and result.get('readOnly') is True,
                                'Observed response lost read-only prefix authority')
                        require(not result.get('budgetStopped') and not result['verification'].get('failure'), 'Prefix stopped before action admission')
                        require(args['frameIndex'] == 0 or len(guard.calls) == before, 'Warm observation executed physics')
                guard.resources()
                return result
            except BaseException as error:
                guard.failure = {'type':type(error).__name__,'message':str(error),'context':guard.context}
                raise
            finally:
                counted = guard.json_counter.end_command(previous_json) if json_started else None
                guard.commands.append({**guard.context, 'steps':len(guard.calls)-before, 'seconds':time.perf_counter()-began,
                                       **({'nativeJson':counted} if counted is not None else {})})
                guard.context = previous
        def close(session):
            guard.capture(session)
            return original_close(session)
        session_class.execute, session_class.close = execute, close

    def __enter__(self):
        try:
            adapter = self._json_adapter
            if adapter is None:
                from resectionlab import surgical_prefix_json_native
                adapter = surgical_prefix_json_native
            self.json_counter = NativeJsonAdmissionCounter(adapter)
            super().__enter__()
            self.json_counter.__enter__()
        except BaseException as error:
            self.failure = {'type':type(error).__name__,'message':str(error),'nativeJsonCounter':True}
            if self.json_counter is not None:self.json_counter.close()
            if self.tool is not None:super().__exit__(*sys.exc_info())
            raise
        def watch():
            while not self._stop.wait(.25):
                try:
                    self.resources()
                except BaseException as error:
                    self.failure = {'type':type(error).__name__,'message':str(error),'watchdog':True}
                    try:
                        atomic_json(self.output/'physics-guard.json', self.receipt())
                    finally:
                        # Receipt/hash/disk failures must never disable the RSS stop.
                        os._exit(70)
        self._watcher = threading.Thread(target=watch, daemon=True, name='vascular-qualification-budget')
        try:self._watcher.start()
        except BaseException:
            self.failure = {'type':'NativeJsonCounterWatchdogSetupFailure','nativeJsonCounter':True}
            try:self.json_counter.close()
            finally:super().__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._watcher:
            self._watcher.join(1)
        try:
            try:self._check_json_counter()
            except BaseException:
                # The failed counter is still recorded, but must not mask the body.
                if args[0] is None:raise
        finally:
            try:
                if self.json_counter is not None:self.json_counter.close()
            finally:super().__exit__(*args)

    def finish(self):
        self.spool.close()
        atomic_json(self.output/'physics-guard.json', self.receipt())

    def receipt(self):
        return {'schema':'resectionlab.vascular-prefix-qualification-guard.v1',
            'permittedSteps':self.calls,'forbiddenSteps':self.forbidden,'publicStepAttempts':dict(self.attempts),
            'nonStepMonitoringCallbacks':self.other_callbacks,'commands':self.commands,'failure':self.failure,
            'ownership':self.ownership,'ownerMetadata':self.owner_metadata,'retentionPeaks':self.retention_peaks,
            'processPeakResidentBytes':self.rss(),'maxProcessResidentBytes':MAX_RSS,
            'wallSeconds':self.clock()-self.began,'maxWallSeconds':MAX_SECONDS,
            'spool':self.spool.statistics,'reference':{'path':str(REFERENCE),'sha256':file_sha(REFERENCE)},
            'nativeJson':self.json_counter.receipt() if self.json_counter is not None else None,
            'residentMeaning':'Whole Python peak RSS; macOS bytes / Linux KiB converted to bytes. Source geometry/runtime/transient allocations included.'}


def checked_field_tokens(state, tokens):
    expected = {}
    if 'flapComponents' in state:
        encoded=json.dumps(state['flapComponents'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        expected['flapComponents']=hashlib.sha256(encoded).hexdigest()
    require(type(tokens) is dict and tokens==expected,'Observed display field token differs from actual decoded state')


def observed_activation(state, action, tools):
    key,op=action['slot'],action['operation']
    slot=state['slots'][key]
    active=(op in ('apply','hold','grasp','extract','activate') or slot['active'] and op=='move'
            or op=='inspect' and tools[slot['instrument']]['effect']=='probe')
    return {hand:bool(active if hand==key else value['active']) for hand,value in state['slots'].items()}


def audit_spool(output, audit_output=None):
    """Separate process after GUI shutdown: no physics, cache, or admission."""
    sys.path.insert(0, str(ROOT/'src'))
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_tools import TOOLS
    from accept_native import vascular_sample, fluid_audit, drilling_debris_audit, wound_audit
    output = Path(output)
    audit_output=Path(audit_output) if audit_output is not None else output
    audit_output.mkdir(parents=True,exist_ok=True)
    guard = json.loads((output/'physics-guard.json').read_text())
    metadata = guard['ownerMetadata']
    require(guard['reference']['sha256'] == file_sha(REFERENCE), 'Canonical reference changed')
    require(metadata['sourceArchiveSha256']==NATIVE_SHA and metadata['canonicalReferenceSha256']==file_sha(REFERENCE) and metadata['expectedEndpointStateId']==END_STATE,'Spool owner/source/reference binding differs')
    require(file_sha(NATIVE) == NATIVE_SHA, 'Original native changed')
    expected = [json.loads(line) for line in REFERENCE.read_text().splitlines()]
    require((output/'observed-states.bin').stat().st_size <= MAX_SPOOL and (output/'observed-actions.jsonl').stat().st_size <= MAX_MANIFEST, 'Audit spool exceeds declared bounds')
    results, tuples = [], []
    seen, generated_debris, previous_state, fluid_offset = {}, 0., None, None
    prior_time = -math.inf
    began = time.monotonic()
    with ReadOnlyStepMonitor(SurgicalEngine) as no_step:
        scene = load_environment(NATIVE)
        require(scene.identity == metadata['sceneId'], 'Independent source scene differs')
        with (output/'observed-states.bin').open('rb') as data, (output/'observed-actions.jsonl').open() as manifest:
            for index,line in enumerate(manifest):
                require(index < 73, 'Extra spooled action')
                record = json.loads(line)
                require(record['owner']==metadata,'Manifest prefix ownership changed')
                require(record['actionIndex'] == index and len(record['frames']) == FRAME_COUNTS[index], 'Manifest sequence differs')
                receipt, final = record['receipt'], None
                for frame_index,frame in enumerate(record['frames']):
                    require(time.monotonic()-began<180 and peak_rss()<=MAX_RSS,'Post-GUI audit wall/RSS budget exceeded')
                    loc = frame['blob']; offset, length, digest = loc['offset'],loc['bytes'],loc['sha256']
                    require(type(offset) is int and type(length) is int and 0 <= offset < MAX_SPOOL and 0 < length <= MAX_RAW_STATE and offset+length <= guard['spool']['binaryBytes'], 'Invalid spooled offset')
                    data.seek(offset); blob = data.read(length)
                    require(len(blob) == length and hashlib.sha256(blob).hexdigest() == digest, 'Compressed spool bytes differ')
                    decoder=zlib.decompressobj(); raw=decoder.decompress(blob, MAX_RAW_STATE+1)
                    require(len(raw)<=MAX_RAW_STATE and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'Decoded state exceeds budget/is incomplete')
                    state=json.loads(raw)
                    hashed={'scene':scene.identity,'state':state}
                    if metadata['protectionContractHash'] is not None: hashed['protectionContract']=metadata['protectionContractHash']
                    identity=semantic_digest(hashed)
                    require(identity==frame['stateId'],'Independent decoded observation state hash differs')
                    checked_field_tokens(state,frame.get('fieldTokens'))
                    require(digest not in seen or seen[digest]==(offset,length,identity),'Deduplicated blob identity differs')
                    seen[digest]=(offset,length,identity)
                    timestamp=frame['observedTimeSeconds']
                    require(math.isfinite(timestamp) and timestamp==state['elapsedSeconds'] and timestamp>=prior_time,'Observation timestamp differs/decreases')
                    prior_time=timestamp
                    activation=frame['effectiveActivation']
                    require(set(activation)==set(state['slots']) and all(type(value) is bool for value in activation.values()),'Observed activation map differs')
                    if frame_index==0:
                        require(frame['kind']=='action_start' and identity==receipt['stateBefore'],'Action-start ancestry differs')
                        require(previous_state is None or state==previous_state,'Adjacent action boundary/start state differs')
                        require(activation=={key:bool(slot['active']) for key,slot in state['slots'].items()},'Start activation differs from recorded slots')
                        if fluid_offset is None:
                            fluid_offset=state['fluidAddedMm3']-math.fsum([math.fsum(state['fluid'].values()),math.fsum(s.get('reservoirMm3',0.) for s in state['sources']),state['fluidRemovedMm3'],state['fluidEscapedMm3']])
                    elif frame_index<len(record['frames'])-1:
                        require(frame['kind']=='microstep','Intermediate observation kind differs')
                    else:
                        require(frame['kind']=='action_boundary' and identity==receipt['stateAfter'],'Finalization ancestry differs')
                    if frame_index>0:
                        require(activation==observed_activation(state,receipt['action'],TOOLS),'Effective activation differs from recorded action and slots')
                    final=(state,frame)
                    tuples.append({'branchId':BRANCH,'position':index,'frameIndex':frame_index,'stateId':identity,'simulatedSeconds':timestamp})
                state=final[0]
                before_removed=set(previous_state['removed']) if previous_state is not None else set()
                # Initial source state has no committed removal; its actual first
                # observation was already verified above. Use receipt deltas too.
                if index==0:
                    first=record['frames'][0]['blob'];data.seek(first['offset']);initial=json.loads(zlib.decompress(data.read(first['bytes'])));before_removed=set(initial['removed'])
                delta=sorted(set(state['removed'])-before_removed)
                require(canonical_json(receipt['action'])==canonical_json(expected[index]['action']) and receipt['deltas']['removedCells']==delta,'Unprojected action/removal ancestry differs')
                for event in receipt.get('events',[]):
                    if event.get('kind')=='drill_work': generated_debris+=event['removedVolumeMm3']
                    elif event.get('kind')=='bone_kerf': generated_debris+=event['cells']*scene.voxel_mm3
                shell=SimpleNamespace(state=state)
                detail={'position':index+1,'action':receipt['action'],'stateBefore':receipt['stateBefore'],'stateAfter':receipt['stateAfter'],
                    'status':receipt['status'],'reason':receipt.get('reason'),'reasonCode':receipt.get('reasonCode'),
                    'completedFraction':receipt.get('completedFraction'),'elapsedSeconds':receipt.get('elapsedSeconds'),
                    'removedCells':delta,'events':receipt.get('events'),'fluid':fluid_audit(state,fluid_offset),
                    'vascularState':vascular_sample(shell,index+1),'drillingAndDebris':drilling_debris_audit(state,scene,generated_debris)}
                wound_audit(state,scene)
                require(canonical_json(detail)==canonical_json(expected[index]),f'Complete canonical trace row {index+1} differs')
                results.append(detail); previous_state=state
                atomic_json(audit_output/'spool-audit-progress.json',{'actionsProcessed':len(results),'observationsProcessed':len(tuples),'allProcessedFullCanonicalRecordsExact':True,'lastCanonicalStateId':identity,'lastSimulatedSeconds':prior_time,'publicStepAttempts':dict(no_step.attempts),'sourceArchiveSha256':NATIVE_SHA})
        require(len(results)==73 and len(tuples)==1149 and tuples[-1]['stateId']==END_STATE,'Full vascular spool incomplete')
    (audit_output/'audited-canonical.actions.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in results))
    atomic_json(audit_output/'audited-observations.json',tuples)
    report={'passed':True,'actions':len(results),'observations':len(tuples),'allFullCanonicalRecordsExact':True,
        'historicalAllFrameIdComparison':False,'observationMeaning':'Fresh committed public observations independently decoded/hashed and checked for ancestry, timestamps, activation and full canonical endpoints. Historical per-frame hashes unavailable.',
        'publicStepAttempts':dict(no_step.attempts),'binarySha256':file_sha(output/'observed-states.bin'),
        'manifestSha256':file_sha(output/'observed-actions.jsonl'),'referenceSha256':file_sha(REFERENCE),
        'sourceSha256':file_sha(NATIVE),'processPeakResidentBytes':peak_rss()}
    atomic_json(audit_output/'spool-audit.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['audit']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,help='Separate diagnostic results directory; input spool remains untouched')
    arguments=parser.parse_args()
    try:
        audit_spool(arguments.output,arguments.audit_output)
    except BaseException as error:
        atomic_json((arguments.audit_output or arguments.output)/'spool-audit.json',{'passed':False,'error':repr(error),'processPeakResidentBytes':peak_rss()})
        raise
