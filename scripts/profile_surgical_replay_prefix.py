#!/usr/bin/env python3
"""Bounded current-code replay profiling of the preserved three-action prefix.

This replays only the original early legacy drilling actions, never a policy or
the full patient operation. No source workspace is written or duplicated. A
short in-memory archive must pass canonical replay before observed frames are
prepared. Timed display reads then run under the published no-step guard.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time

from check_surgical_rl_readiness import ROOT, Budget, require, write_json
sys.path.insert(0, str(ROOT / 'testcases/outputs/full-case-rehearsal-20261020/review'))
from readonly_step_monitor import ReadOnlyStepMonitor
from resectionlab.core import freeze_json, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache
# Freeze the engine-subclass inventory before the read-only monitoring phase.
from resectionlab import surgical_telemetry, surgical_planning, full_case_rehearsal

NATIVE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab'
NATIVE_SHA = '095659e54f381355e8f5655b6c5907990b2ac4767dc9ce58297e3bc4d6f3268d'
BRANCH = 'current-original-prefix-3'
BASELINE = ROOT / 'testcases/outputs/rl-readiness-20261022/desktop-baseline/playback-authenticated/report.json'
TRACE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020/final/primary-canonical.actions.jsonl'


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PreparationSteps:
    """Count actual step entries without changing engine code or cache identity."""
    def __init__(self):
        self.calls = []
        self.monitor = sys.monitoring
        self.tool = next((i for i in (4, 3, 2, 1, 0, 5) if self.monitor.get_tool(i) is None), None)
        require(self.tool is not None, 'No monitoring slot for bounded preparation')

    def callback(self, code, offset):
        if code.co_name != 'step':
            return self.monitor.DISABLE
        frame = sys._getframe(1)
        obj = frame.f_locals.get('self')
        if isinstance(obj, SurgicalEngine):
            if len(self.calls) >= 16:
                raise RuntimeError('Unexpected extra physical replay; refusing full operation')
            self.calls.append({'class': type(obj).__name__, 'priorActions': len(obj.actions)})

    def __enter__(self):
        self.monitor.use_tool_id(self.tool, 'bounded-current-prefix-preparation')
        self.monitor.register_callback(self.tool, self.monitor.events.PY_START, self.callback)
        self.monitor.set_events(self.tool, self.monitor.events.PY_START)
        return self

    def __exit__(self, *args):
        self.monitor.set_events(self.tool, 0)
        self.monitor.register_callback(self.tool, self.monitor.events.PY_START, None)
        self.monitor.free_tool_id(self.tool)


def run(output, budget):
    require(sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12), 'CPython 3.12 required')
    require(file_sha(NATIVE) == NATIVE_SHA, 'Preserved episode bytes differ')
    expected = {row['frameIndex']: row['stateId'] for row in json.loads(BASELINE.read_text())['playbackIpc']}
    with TRACE.open() as stream:
        original = [json.loads(next(stream)) for _ in range(3)]
    session = BridgeSession(output / 'transfers')
    session.surgical_replay_cache = SurgicalReplayCache(verified_cache=VerifiedSurgicalCache(output / 'verified-cache'))
    session.surgical_frame_cache = SurgicalFrameCache(directory=output / 'observed-cache')
    def call(command, body):
        return budget.timed(lambda: session.execute(command, body, _Request(command), lambda *_: None))
    try:
        loaded, load_seconds = call('loadCase', {'path': str(NATIVE)})
        key = loaded['caseHash']
        entry = session.cases[key]
        artifacts = thaw_json(entry.artifacts)
        old = artifacts['surgical_toolbox']
        snapshot = old['branches']['full-primary']
        bounded = {name: copy.deepcopy(snapshot[name]) for name in ('schema', 'sceneId', 'protectionContract') if name in snapshot}
        bounded.update(actions=copy.deepcopy(snapshot['actions'][:3]), stateId=original[-1]['stateAfter'])
        artifacts['surgical_toolbox'] = {
            'environmentSchema': old['environmentSchema'], 'physicsProfile': old['physicsProfile'],
            'mapping': old['mapping'], 'branches': {BRANCH: bounded}, 'branchId': BRANCH,
            'position': 3, 'parents': {BRANCH: None},
        }
        entry.artifacts = freeze_json(artifacts)
        args = {'caseHash': key, 'command': 'inspect'}
        with PreparationSteps() as preparation:
            final, canonical_seconds = call('surgicalRehearsal', args)
            require(final['view']['stateId'] == bounded['stateId'], 'Current canonical prefix endpoint differs')
            for index, row in enumerate(final['timeline']):
                require(row['stateAfter'] == original[index]['stateAfter'] and row['events'] == original[index]['events']
                        and row['status'] == original[index]['status'], 'Canonical legacy prefix receipt differs')
            body = {'caseHash': key, 'command': 'seekFrame', 'branchId': BRANCH,
                    'position': 2, 'frameIndex': 0, 'expectedStateId': final['view']['stateId']}
            response, prepare_seconds = call('surgicalRehearsal', body)
        require(response['playback']['frameCount'] >= 12, 'Expected observed prefix frames unavailable')
        write_json(output / 'canonical-prefix-snapshot.json', session.surgical_sessions[key][1].snapshot())
        body.update(timelineToken=response['timelineToken'], viewFieldTokens=response['viewFieldTokens'])
        rows = []
        guard = ReadOnlyStepMonitor(SurgicalEngine)
        with guard:
            for index in range(12):
                body['frameIndex'] = index
                response, seconds = call('surgicalRehearsal', body)
                require(response['view']['stateId'] == expected[index], 'Current observed frame differs from original baseline')
                require(response['view']['stateId'] == response['playback']['frameStateId'], 'Frame identity mismatch')
                require(response['timelineOmitted'] and response['viewFieldsOmitted'] == ['flapComponents'], 'Warm field reuse was not authenticated')
                rows.append({'frameIndex': index, 'handlerSeconds': seconds,
                             'stateId': response['view']['stateId'], 'responseBytes': len(json.dumps(response).encode())})
                body['viewFieldTokens'] = response['viewFieldTokens']
        require(file_sha(NATIVE) == NATIVE_SHA, 'Profiling changed the preserved episode')
        values = [row['handlerSeconds'] for row in rows]
        return {'scope': 'Current production Python source, original three-action legacy prefix only; in-memory archive, no duplicated source workspace. Not full-case admission or GUI timing.',
                'source': str(NATIVE), 'sourceSha256': NATIVE_SHA, 'sourceUnchanged': True,
                'no_training': True, 'optimizer_updates': 0, 'policy_executions': 0,
                'canonicalActions': 3, 'preparationPhysicalSteps': preparation.calls,
                'loadSeconds': load_seconds, 'canonicalPrefixSeconds': canonical_seconds,
                'observedActionPreparationSeconds': prepare_seconds,
                'timedReadPhysicalSteps': dict(guard.attempts), 'frames': rows,
                'handlerMedianSeconds': statistics.median(values), 'handlerMeanSeconds': statistics.mean(values),
                'handlerP95Seconds': sorted(values)[math.ceil(.95 * len(values)) - 1],
                'allOriginalFrameIdsExact': True, 'currentImplementation': True,
                'fullCaseQualification': False, 'guiQualification': False}
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=120.)
    args = parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 180:
        parser.error('Wall-time cap must be in (0,180] seconds')
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT / 'testcases/outputs').resolve()):
        parser.error('Use a new output directory below testcases/outputs')
    output.mkdir(parents=True)
    budget = Budget(args.max_wall_seconds)
    def timeout(*_):
        raise TimeoutError('Current prefix profiling deadline exhausted')
    previous = signal.signal(signal.SIGALRM, timeout)
    report = {'scope': 'Current original three-action replay prefix only', 'no_training': True, 'optimizer_updates': 0}
    code = 0
    try:
        signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
        report.update(run(output, budget), status='passed')
    except Exception as error:
        report.update(status='incomplete' if isinstance(error, TimeoutError) else 'failed', error={'type': type(error).__name__, 'message': str(error)})
        code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        report['wallSeconds'] = time.perf_counter() - budget.started
        write_json(output / 'report.json', report)
    print(json.dumps({name: report[name] for name in ('status', 'wallSeconds')}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
