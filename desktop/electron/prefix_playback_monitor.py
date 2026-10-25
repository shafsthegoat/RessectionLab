"""Benchmark-only CPython monitor for three ordered original prefix actions.

The inherited guard owns and checks the monitoring APIs and engine code
inventory. This lane permits exactly the two requested boundary preparations
and the first action-2 frame preparation; every later frame read is step-free.
No engine method or cache is replaced.
"""
import sys
import resource
import time

from readonly_step_monitor import ForbiddenPhysicsStep, ReadOnlyStepMonitor


class PrefixPlaybackMonitor(ReadOnlyStepMonitor):
    def __init__(self, engine_base):
        super().__init__(engine_base)
        self.context = None
        self.calls = []
        self.forbidden = []
        self.sessions = []
        self.commands = []

    def _callback(self, code, offset, *extra):
        if code.co_name != 'step':
            self.other_callbacks += 1
            return None if extra else self.monitor.DISABLE
        frame = sys._getframe(1)
        if frame.f_code is not code:
            raise RuntimeError('Monitoring callback callee frame differs')
        obj = frame.f_locals.get('self')
        if not isinstance(obj, self.engine_base):
            return
        row = dict(self.context or {})
        row.update(priorActions=len(obj.actions), event='resume' if extra else 'entry')
        expected = [('seekPrefix', 2, None, 0), ('seekPrefix', 2, None, 1),
                    ('seekPrefixFrame', 2, 0, 2)]
        key = (row.get('command'), row.get('position'), row.get('frameIndex'), row['priorActions'])
        if len(self.calls) >= len(expected) or key != expected[len(self.calls)]:
            self.forbidden.append(row)
            self.attempts['unexpectedPrefixStep'] += 1
            raise ForbiddenPhysicsStep('Prefix benchmark attempted an unexpected or fourth physics step')
        self.calls.append(row)

    def install(self, session_class):
        original_execute, original_close = session_class.execute, session_class.close
        guard = self

        def execute(session, operation, args, request, progress):
            previous = guard.context
            guard.context = {'operation': operation, **{key: args.get(key) for key in
                             ('command', 'branchId', 'position', 'frameIndex')}}
            before = len(guard.calls)
            started = time.perf_counter()
            try:
                return original_execute(session, operation, args, request, progress)
            finally:
                guard.commands.append({**guard.context, 'steps': len(guard.calls) - before, 'seconds': time.perf_counter() - started})
                guard.context = previous
                if session not in guard.sessions:
                    guard.sessions.append(session)
                # A warm read must never execute a step, regardless of global count.
                if operation == 'surgicalRehearsal' and args.get('command') == 'seekPrefixFrame' and args.get('frameIndex', 0) >= 1 and len(guard.calls) != before:
                    raise ForbiddenPhysicsStep('Warm prefix frame attempted physics')

        def close(session):
            # Capture ownership before close deletes the private prefix owner.
            guard.capture(session)
            return original_close(session)

        session_class.execute, session_class.close = execute, close

    def capture(self, session):
        owned = getattr(session, 'surgical_prefix_session', None)
        self.ownership = {
            'beforeClose': True,
            'canonicalSessionCreated': hasattr(session, 'surgical_sessions'),
            'replayCacheCreated': hasattr(session, 'surgical_replay_cache'),
            'frameCacheCreated': hasattr(session, 'surgical_frame_cache'),
            'frameAdmissionsCreated': hasattr(session, 'surgical_frame_admissions'),
            'prefixStatistics': owned[3].statistics if owned else None,
        }

    def receipt(self):
        for session in self.sessions:
            if not hasattr(self, 'ownership'):
                self.capture(session)
        return {'permittedSteps': self.calls, 'forbiddenSteps': self.forbidden,
                'publicStepAttempts': dict(self.attempts),
                'nonStepMonitoringCallbacks': self.other_callbacks, 'commands': self.commands,
                'processPeakResidentBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                'residentMeaning': 'Whole Python process peak RSS, including source anatomy; private retained prefix allocations are reported separately by owner statistics.',
                'ownership': getattr(self, 'ownership', None)}
