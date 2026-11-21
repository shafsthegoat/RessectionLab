"""Local automation timings for the existing sidecar protocol.

Stage durations are inclusive and can overlap. Only elapsed monotonic times
leave the interpreter. No samples or geometry enter the wire response.
"""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import sys
from time import monotonic_ns


PREFIX = 'RESSECTIONLAB_REHEARSAL_TIMING '
SCHEMA = 'resectionlab.rehearsal-timing.v1'
_MAX_LINE_CHARACTERS = 4096
_BOUNDED_STAGES = frozenset({'simulationStep', 'referenceExecution', 'deformationTick'})
_NATIVE_COUNTERS = ('residentActionCalls', 'displacementSnapshotCalls', 'displacementSnapshotFallbackCalls')
_active_timing = ContextVar('rehearsal_timing', default=None)


def _text(value):
    return value if type(value) is str and len(value) <= 128 else None


def _position(value):
    return value if type(value) is int and 0 <= value < 2**53 else None


class RequestTiming:
    def __init__(self, request_id, command, args):
        action = args.get('action')
        action = action if isinstance(action, dict) else {}
        self.identity = {
            'requestId': request_id, 'command': command,
            'caseHash': _text(args.get('caseHash')),
            'expectedStateId': _text(args.get('expectedStateId')),
            'requestedBranchId': _text(args.get('branchId')),
            'requestedPosition': _position(args.get('position')),
            'requestedFrameIndex': _position(args.get('frameIndex')),
            'actionOperation': _text(action.get('operation')),
            'actionSlot': _text(action.get('slot')),
            'actionInstrument': _text(action.get('instrument')),
        }
        self.durations = {}
        self.calls = {}
        self.starts = {}
        self.bounds = {}
        self.response = {}
        self.deformation = None
        self.native_bindings = {}
        self.queued_ns = 0

    @classmethod
    def for_operation(cls, request_id, operation, args):
        if operation == 'surgicalRehearsal':
            command = _text(args.get('command', 'inspect'))
        elif operation in {'saveCase', 'loadCase', 'anatomySavePackage', 'anatomyOpen'}:
            command = operation
        else:
            return None
        return cls(request_id, command, args) if command is not None else None

    def add(self, name, duration_ns):
        self.durations[name] = self.durations.get(name, 0) + duration_ns
        self.calls[name] = self.calls.get(name, 0) + 1

    def bind_source(self, branch_id, position):
        self.identity.update(sourceBranchId=_text(branch_id), sourcePosition=_position(position))

    def record_bounds(self, stage, started_ns, finished_ns):
        start = started_ns - self.queued_ns
        end = finished_ns - self.queued_ns
        bounds = self.bounds.get(stage)
        if bounds is None:
            self.bounds[stage] = {'firstStart': start, 'lastEnd': end,
                                  'longestStart': start, 'longestEnd': end}
        else:
            bounds['firstStart'] = min(bounds['firstStart'], start)
            bounds['lastEnd'] = max(bounds['lastEnd'], end)
            if end - start > bounds['longestEnd'] - bounds['longestStart']:
                bounds.update(longestStart=start, longestEnd=end)

    def record_response(self, event, byte_count, started_ns, serialized_ns, locked_ns, written_ns):
        encode_ns = serialized_ns - started_ns
        lock_ns = locked_ns - serialized_ns
        flush_ns = written_ns - locked_ns
        self.add('serialize', encode_ns)
        self.add('responseOutputLock', lock_ns)
        self.add('writeFlush', flush_ns)
        self.response.update(terminalEvent=event['event'], responseBytes=byte_count,
                             responseWriteStartSinceQueueNs=locked_ns - self.queued_ns)
        result = event.get('result')
        if not isinstance(result, dict):
            error = event.get('error')
            if isinstance(error, dict):
                self.response['reasonCode'] = _text(error.get('code'))
            return
        self.response.update(responseCaseHash=_text(result.get('caseHash')),
                             branchId=_text(result.get('branchId')),
                             position=_position(result.get('position')))
        view = result.get('view')
        if isinstance(view, dict):
            self.response['responseStateId'] = _text(view.get('stateId'))
        playback = result.get('playback')
        if isinstance(playback, dict):
            self.response.update(playbackPosition=_position(playback.get('position')),
                                 frameIndex=_position(playback.get('frameIndex')))
        outcome = result if self.identity['command'] == 'preview' else result.get('outcome')
        if isinstance(outcome, dict):
            self.response.update(status=_text(outcome.get('status')), reasonCode=_text(outcome.get('reasonCode')))

    def write(self, stream):
        row = {'schema': SCHEMA, 'process': 'python', **self.identity, **self.response,
               'durationsNs': self.durations, 'calls': self.calls, 'stageBoundsNs': self.bounds,
               'nativeBindings': self.native_bindings,
               'deformationKernels': self.deformation if self.deformation is not None else {
                   'legacySweepObstacleCalls': self.calls.get('sweepObstacle', 0),
                   'continuumTicks': self.calls.get('continuumTick', 0),
               }}
        try:
            encoded = json.dumps({key: value for key, value in row.items() if value is not None},
                                 separators=(',', ':'), allow_nan=False)
        except (ValueError, TypeError):
            encoded = json.dumps({'schema': SCHEMA, 'process': 'python',
                                  'requestId': self.identity['requestId'], 'invalidDiagnostics': True})
        if len(encoded) > _MAX_LINE_CHARACTERS:
            encoded = json.dumps({'schema': SCHEMA, 'process': 'python',
                                  'requestId': self.identity['requestId'], 'overflow': True})
        try:
            stream.write(PREFIX + encoded + '\n')
            stream.flush()
        except OSError:
            # The stdout protocol has already delivered the terminal event.
            pass


class RehearsalTimingMonitor:
    """Observe selected function bodies without wrapping native-guarded methods."""

    def __init__(self):
        from .core import semantic_digest
        from .desktop_bridge import _require_json_budget
        from .surgical_engine import SurgicalEngine, _displacement_snapshot
        from .surgical_native_engine import prepare
        from . import native_backend, surgical_continuum, surgical_deformation_v2, surgical_practice

        self.kernel_scope = getattr(surgical_deformation_v2, 'kernel_execution_scope', None)
        self.native_counters = native_backend._counters
        self.displacement_adapter = getattr(native_backend, 'try_displacement_snapshot', None)
        self.backend_fallback = (native_backend._identity['fallbackReason'] or 'reference-selected'
                                 if native_backend._module is None else None)
        self.stages = {
            function.__code__: stage for stage, function in (
                ('simulationStep', SurgicalEngine.step), ('posePreview', SurgicalEngine.preview),
                ('referenceExecution', SurgicalEngine._execute), ('residentPrepare', prepare),
                ('deformationTick', surgical_deformation_v2.tick), ('continuumTick', surgical_continuum.tick),
                ('sweepObstacle', getattr(surgical_deformation_v2, '_obstacle_collision', None)),
                ('displacementSnapshot', _displacement_snapshot),
                ('nativeDisplacementSnapshot', self.displacement_adapter),
                ('snapshot', SurgicalEngine.snapshot), ('view', SurgicalEngine.view),
                ('observe', SurgicalEngine.observe), ('stateIdentity', SurgicalEngine.state_id.fget),
                ('semanticDigest', semantic_digest), ('metrics', SurgicalEngine.metrics),
                ('practiceMetrics', surgical_practice.metrics), ('practiceAdmission', surgical_practice.admission),
                ('jsonBudget', _require_json_budget),
            ) if function is not None
        }
        self.monitor = sys.monitoring
        self.tool = next((tool for tool in (3, self.monitor.PROFILER_ID) if self.monitor.get_tool(tool) is None), None)
        if self.tool is None:
            raise RuntimeError('No monitoring slot for rehearsal timings')
        self.monitor.use_tool_id(self.tool, 'ResectionLab rehearsal timing')
        events = self.monitor.events
        try:
            self.monitor.register_callback(self.tool, events.PY_START, self._start)
            self.monitor.register_callback(self.tool, events.PY_RETURN, self._finish)
            self.monitor.register_callback(self.tool, events.PY_UNWIND, self._finish)
            self.monitor.set_events(self.tool, events.PY_UNWIND)
            for code in self.stages:
                self.monitor.set_local_events(self.tool, code, events.PY_START | events.PY_RETURN)
        except BaseException:
            self.close()
            raise

    def _start(self, code, _offset):
        timing = _active_timing.get()
        if timing is not None:
            timing.starts.setdefault(code, []).append(monotonic_ns())

    def _finish(self, code, _offset, value):
        timing = _active_timing.get()
        if timing is None or not (starts := timing.starts.get(code)):
            return
        finished_ns, started_ns = monotonic_ns(), starts.pop()
        stage = self.stages[code]
        timing.add(stage, finished_ns - started_ns)
        if stage in _BOUNDED_STAGES:
            timing.record_bounds(stage, started_ns, finished_ns)
        if stage == 'residentPrepare' and type(value) is tuple:
            timing.calls['residentPrepareResult'] = timing.calls.get('residentPrepareResult', 0) + 1
        if stage == 'nativeDisplacementSnapshot' and type(value) is tuple:
            timing.calls['nativeDisplacementSnapshotResult'] = timing.calls.get('nativeDisplacementSnapshotResult', 0) + 1

    @contextmanager
    def sample(self, timing):
        before = {key: self.native_counters.get(key, 0) for key in _NATIVE_COUNTERS}
        token = _active_timing.set(timing)
        try:
            with self.kernel_scope() if self.kernel_scope is not None else nullcontext() as kernels:
                timing.deformation = kernels
                yield
        finally:
            _active_timing.reset(token)
            counters = {key: self.native_counters.get(key, 0) - value for key, value in before.items()}
            attempts = counters['displacementSnapshotCalls']
            declines = counters['displacementSnapshotFallbackCalls']
            skipped = timing.calls.get('displacementSnapshot', 0) - attempts
            reasons = {'native-input-declined': declines} if declines else {}
            if skipped > 0:
                reason = ('legacy-reference' if self.displacement_adapter is None
                          else self.backend_fallback or 'native-precondition-not-met')
                reasons[reason] = skipped
            timing.native_bindings = {**counters, 'displacementSnapshotSuccessfulCalls':
                timing.calls.get('nativeDisplacementSnapshotResult', 0), 'displacementSnapshotFallbackReasons': reasons}

    def close(self):
        events = self.monitor.events
        for code in self.stages:
            self.monitor.set_local_events(self.tool, code, 0)
        self.monitor.set_events(self.tool, 0)
        for event in (events.PY_START, events.PY_RETURN, events.PY_UNWIND):
            self.monitor.register_callback(self.tool, event, None)
        self.monitor.free_tool_id(self.tool)
