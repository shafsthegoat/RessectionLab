#!/usr/bin/env python3
"""Export existing descriptive telemetry after a cache-only admitted baseline.

This performs one explicitly requested observed public-action replay, not a
new scenario, new policy, new acceptance run, or expertise evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_telemetry import capture_episode, export_telemetry
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache, implementation_fingerprint
from resectionlab.workspace_bundle import load_workspace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('branch')
    parser.add_argument('output', type=Path)
    parser.add_argument('--prepare-frames', '--collect-frames', dest='collect_frames', action='store_true',
                        help='Populate authenticated actual observed frames in the same replay')
    args = parser.parse_args()
    started = time.perf_counter()
    native = args.native.resolve()
    original = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
    source_sha, native_sha = sha(original), sha(native)
    assert source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
    assert not args.output.exists(), 'Preserve any existing telemetry export; choose a new output directory'
    _, artifacts, _ = load_workspace(native)
    snapshot = artifacts['surgical_toolbox']['branches'][args.branch]
    scene = load_environment(native)
    assert scene.identity == 'sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856'
    fingerprint = implementation_fingerprint()
    labels_hash = array_digest(scene.labels)
    cache = VerifiedSurgicalCache()
    baseline = cache.restore(SurgicalEngine, scene, snapshot)
    assert baseline is not None and cache.hits == 1 and cache.misses == 0, 'Cache-only admission missed; no replay fallback'
    assert baseline.state_id == snapshot['stateId']
    print(json.dumps({'cacheOnlyBaselineAdmitted': True, 'actions': len(baseline.actions),
                      'observedPublicReplayStarting': True}), flush=True)
    frame_cache = admission = None
    options = {}
    if args.collect_frames:
        frame_cache = SurgicalFrameCache()
        replay_cache = SurgicalReplayCache(verified_cache=cache)
        admission = frame_cache.admit(baseline, snapshot, replay_cache, verified_history=baseline.history)
        assert cache.hits == 2 and cache.misses == 0, 'Frame admission must also hit authenticated canonical cache'
        options['frame_collector'] = frame_cache.collector(admission)
    episode = capture_episode(baseline, label='scripted', episode_id=args.branch, interval_seconds=.2, **options)
    assert episode['expectedStateVerified'] and episode['finalStateId'] == baseline.state_id
    assert len(episode['attempts']) == len(baseline.actions)
    assert all(a['status'] in ('accepted', 'partial') for a in episode['attempts'])
    assert sha(original) == source_sha and sha(native) == native_sha
    assert array_digest(scene.labels) == labels_hash
    assert implementation_fingerprint() == fingerprint
    frame_count = 0
    if frame_cache is not None:
        assert frame_cache.store.publications == len(baseline.actions), 'Observed frame publication is incomplete'
        for position in range(len(baseline.actions)):
            item = frame_cache.store.load(admission, position)
            assert item is not None, f'Observed action frame cache is missing {position}'
            frame_count += len(item['frames'])
    paths = export_telemetry([episode], args.output)
    receipt = {'schema': 'resectionlab.observed-telemetry-export.v1', 'status': 'passed',
        'native': str(native), 'nativeSha256': native_sha, 'sourceSha256': source_sha,
        'sceneId': scene.identity, 'labelArrayHash': labels_hash,
        'implementationFingerprint': fingerprint, 'baselineAdmission': 'authenticated-cache-only-hit',
        'cache': cache.statistics, 'additionalObservedPublicReplays': 1,
        'episodeId': args.branch, 'label': episode['label'], 'intervalSeconds': .2,
        'initialStateId': episode['initialStateId'], 'finalStateId': episode['finalStateId'],
        'expectedEndpointVerified': True, 'sourceAndNativeUnchanged': True,
        'observedFrameCollectorEnabled': args.collect_frames, 'actualCachedFrames': frame_count,
        'observedFrameCache': frame_cache.store.statistics if frame_cache is not None else None,
        'reportingRows': len(episode['samples']), 'committedObservations': len(episode['observations']),
        'events': len(episode['events']), 'summary': episode['summary'],
        'paths': paths, 'wallSeconds': time.perf_counter() - started,
        'clinicalInjuryOrExpertiseInferred': False, 'guiVerified': False}
    (args.output / 'export-receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'reportingRows', 'committedObservations',
                      'finalStateId', 'wallSeconds')}), flush=True)


if __name__ == '__main__':
    main()
