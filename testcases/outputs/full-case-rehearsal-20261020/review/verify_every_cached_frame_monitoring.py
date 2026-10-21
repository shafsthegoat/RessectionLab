#!/usr/bin/env python3
"""Fresh-process, read-only verification of every actual cached patient frame.

Global sys.monitoring blocks the same step-named Engine calls as the legacy
profile guard, including future subclasses and generator/coroutine resumptions.
Non-step START/RESUME locations disable their repeated callbacks. Engine code
is never monkeypatched, and every legacy frame/SHA/state assertion is retained.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache, implementation_fingerprint, _read
from resectionlab.workspace_bundle import load_workspace
# Load the known observer class before inventory sealing; no engine is created.
from resectionlab import surgical_telemetry
from readonly_step_monitor import ReadOnlyStepMonitor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def timings(values):
    ordered = sorted(values)
    return {'count': len(ordered), 'p50Seconds': statistics.median(ordered),
            'p95Seconds': ordered[max(0, math.ceil(len(ordered) * .95) - 1)],
            'maxSeconds': max(ordered), 'totalSeconds': sum(ordered)}


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python disables audit assertions; refusing cache admission')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('branch')
    parser.add_argument('acceptance', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    original = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
    native = args.native.resolve()
    native_sha, source_sha = sha(native), sha(original)
    assert source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
    canonical = json.loads(args.acceptance.read_text())
    assert canonical['auditStatus'] == 'passed' and canonical['branch'] == args.branch
    assert canonical['nativeSha256'] == native_sha and canonical['originalSha256'] == source_sha
    fingerprint = implementation_fingerprint()
    assert canonical['implementationFingerprint'] == fingerprint, 'Canonical receipt is from an obsolete implementation'
    _, artifacts, _ = load_workspace(native)
    snapshot = artifacts['surgical_toolbox']['branches'][args.branch]
    scene = load_environment(native)
    labels_hash = array_digest(scene.labels)
    assert labels_hash == canonical['labelArrayHash']
    step_monitor = ReadOnlyStepMonitor(SurgicalEngine)
    physical_calls = step_monitor.attempts
    verified = VerifiedSurgicalCache()
    frame_cache = SurgicalFrameCache()
    replay_cache = SurgicalReplayCache(verified_cache=verified)
    all_times, summaries, field_hashes = [], [], {}
    frame_total = 0
    with step_monitor:
        loaded = verified.restore(SurgicalEngine, scene, snapshot, include_history=True)
        assert loaded is not None, 'Cache-only canonical admission missed; no replay fallback'
        current, history = loaded
        assert current.state_id == canonical['stateId'] == snapshot['stateId']
        admission = frame_cache.admit(current, snapshot, replay_cache, verified_history=history)
        for position, outcome in enumerate(history):
            item = frame_cache.store.load(admission, position)
            assert item is not None, f'Missing authenticated observed action {position}'
            action_times = []
            for frame in item['frames']:
                for field in frame['fields']['fields'].values():
                    digest = field['sha256']
                    assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
                    assert field['file'] == 'part-' + digest + '.json.gz'
                    if digest not in field_hashes:
                        data = _read(item['entry'] / field['file'], frame_cache.store.max_file_bytes)
                        assert hashlib.sha256(data).hexdigest() == digest
                        field_hashes[digest] = len(data)
            for index, expected in enumerate(item['frames']):
                step_monitor.check()
                frame_started = time.perf_counter()
                observed, ancestry, playback = frame_cache.seek(current, admission.snapshot, position, index,
                    replay_cache, verified_history=admission.history, admission=admission, display_only=True)
                elapsed = time.perf_counter() - frame_started
                action_times.append(elapsed); all_times.append(elapsed); frame_total += 1
                assert observed.scene is scene and observed.state_id == expected['stateId']
                assert playback['readOnly'] and playback['frameIndex'] == index
                assert playback['frameCount'] == len(item['frames'])
                assert playback['observedTimeSeconds'] == expected['observedTimeSeconds']
                if index == 0: assert observed.state_id == outcome['stateBefore']
                if index == len(item['frames']) - 1: assert observed.state_id == outcome['stateAfter']
                assert ancestry is admission.history
                step_monitor.check()
            summaries.append({'actionIndex': position, 'frames': len(item['frames']),
                              'stateBefore': outcome['stateBefore'], 'stateAfter': outcome['stateAfter'],
                              'warmSeekTiming': timings(action_times)})
            if (position + 1) % 10 == 0:
                print(json.dumps({'cachedActionsVerified': position + 1, 'of': len(history),
                                  'actualFramesVerified': frame_total}), flush=True)
    assert not physical_calls
    assert frame_cache.store.misses == 0 and frame_cache.store.publications == 0
    assert frame_cache.store.hits == len(history) + frame_total
    assert verified.hits == 2 and verified.misses == 0 and verified.publications == 0
    assert sha(native) == native_sha and sha(original) == source_sha and array_digest(scene.labels) == labels_hash
    assert implementation_fingerprint() == fingerprint
    receipt = {'schema': 'resectionlab.every-observed-frame-cache-review.v1', 'status': 'passed',
        'native': str(native), 'nativeSha256': native_sha, 'sourceSha256': source_sha,
        'sceneId': scene.identity, 'labelArrayHash': labels_hash,
        'implementationFingerprint': fingerprint, 'canonicalAcceptance': str(args.acceptance.resolve()),
        'actions': len(history), 'actualFramesDecodedAndSought': frame_total,
        'uniqueStateFieldBlobsVerified': len(field_hashes), 'uniqueStateFieldBlobBytes': sum(field_hashes.values()),
        'publicPhysicsStepCalls': dict(physical_calls), 'totalPublicPhysicsStepCalls': sum(physical_calls.values()),
        'sysSetprofileUsed': False, 'sysMonitoringUsed': True,
        'stepGuardEvents': ['PY_START','PY_RESUME','PY_THROW'],
        'nonStepMonitoringCallbacks': step_monitor.other_callbacks,
        'stepGuardScope': "Exact legacy predicate: Python code named step with isinstance(callee.self,SurgicalEngine), globally including future code. Non-step START/RESUME locations disable only themselves; step locations remain armed even after non-engine calls.",
        'stepInventoryAndMonitoringConfigurationStable': True,
        'sourceAndNativeUnchanged': True, 'allActionBoundaryStatesEqualCanonical': True,
        'endpointStateId': snapshot['stateId'], 'verifiedStateCache': verified.statistics,
        'observedFrameCache': frame_cache.store.statistics, 'frameMemoryCache': frame_cache.statistics,
        'warmFrameSeekTiming': timings(all_times), 'actionsVerified': summaries,
        'timingMeaning': 'Globally monitored backend cache decode/seek with one-time non-step START/RESUME callbacks; no per-call sys.setprofile overhead, renderer or GUI claim.',
        'wallSeconds': time.perf_counter() - started, 'guiVerified': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'actions', 'actualFramesDecodedAndSought',
        'uniqueStateFieldBlobsVerified', 'publicPhysicsStepCalls', 'warmFrameSeekTiming', 'wallSeconds')}), flush=True)


if __name__ == '__main__':
    main()
