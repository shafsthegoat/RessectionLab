"""Profile uncached public-action replay and compare every accepted boundary across backends."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import sys
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab import native_backend
from resectionlab.core import semantic_digest
from resectionlab.imaging import read_case_artifacts
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_sequence_outcomes import original_target_residual_mm3


def sha256_file(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def timings(samples):
    return {'count': len(samples), 'medianMs': float(np.median(samples)),
            'p95Ms': float(np.percentile(samples, 95)), 'maxMs': max(samples)}


def profile_rows(profile):
    rows = []
    for (file, line, name), (unique, total, own, cumulative, _) in pstats.Stats(profile).stats.items():
        if 'resectionlab' in file.lower() or 'resectionlab' in name.lower():
            rows.append({'function': f'{Path(file).name}:{line}:{name}', 'calls': total,
                         'selfMs': 1000 * own, 'inclusiveMs': 1000 * cumulative})
    return {'self': sorted(rows, key=lambda r: -r['selfMs'])[:20],
            'inclusive': sorted(rows, key=lambda r: -r['inclusiveMs'])[:20]}


class CaptureAudit:
    """Compare original compositor receipts with independently restored action endpoints."""

    def __init__(self, path, archive_hash, branch, scene, actions):
        data = json.loads(path.read_text())
        if (data.get('windowEverVisible') is not False or not data.get('passed')
                or data['sourceFileSha256'] != archive_hash or data['branchId'] != branch
                or data['sceneId'] != scene.identity or data['sourceCaseHash'] != scene.source['caseHash']
                or data['steps'] != len(actions) or not data['completeEpisode']):
            raise ValueError('Capture is not a complete hidden replay of this exact archive, scene and branch')
        self.rows = data['actions']
        self.starts = []
        self.ends = []
        self.kinds = Counter()
        previous_time = -float('inf')
        previous = None
        for frame in data['frames']:
            if 'position' not in frame:  # Exterior overview, outside the accepted native timeline.
                continue
            i, j, count = (frame[k] for k in ('position', 'frameIndex', 'frameCount'))
            if (frame['branchId'] != branch or frame['time'] < previous_time - 1e-8 or count < 2
                    or not 0 <= j < count or (i, j) != (previous or (0, 0))):
                raise ValueError(f'Unordered, wrong-branch or missing observation at frame {(i, j)}')
            previous = (i+1, 0) if j == count-1 else (i, j+1)
            previous_time = frame['time']
            self.kinds[frame['kind']] += 1
            if j == 0:
                self.starts.append(frame)
            if j == count-1:
                self.ends.append(frame)
        if (previous != (len(actions), 0)
                or not len(self.starts) == len(self.ends) == len(self.rows) == len(actions)):
            raise ValueError('Missing terminal observations or accepted actions')
        self.report = {'captureSHA256': sha256_file(path), 'renderer': data['rendererAsset'],
                       'matchedActions': len(actions), 'observations': dict(self.kinds)}
        self.body_poses = {}
        self.object_ids = set()
        self.exposure = {'movedBodies': Counter(), 'bodyTransitions': [], 'objectTransitions': []}
        self.object_states = {}

    def exposure_boundary(self, engine, position):
        exposure = engine.state.get('operativeExposureV1')
        if exposure is None:
            return
        actor = engine.history[-1]['action']['slot'] if position else None
        for name, body in exposure['bodies'].items():
            pose = (body['angleRad'], tuple(body['translationMm']), tuple(body['quaternion']))
            current = (body['status'], body['heldBy'], pose)
            old = self.body_poses.get(name)
            if old and pose != old[2]:
                if actor not in (old[1], body['heldBy']):
                    raise ValueError(f'{name} moved without the acting hand holding it at {position}')
                self.exposure['movedBodies'][name] += 1
            if not old or old[:2] != current[:2]:
                self.exposure['bodyTransitions'].append([position, name, *current[:2]])
            self.body_poses[name] = current
        ids = {o['id'] for o in exposure['objects']}
        if not self.object_ids <= ids or len(ids) != len(exposure['objects']):
            raise ValueError(f'Placed object identity lost or repeated at {position}')
        self.object_ids = ids
        for obj in exposure['objects']:
            key = obj['id']
            current = (obj['status'], obj['heldBy'], obj.get('sourceCell'))
            old = self.object_states.get(key)
            if old and old[2] != current[2]:
                raise ValueError(f'{key} lost its source cell at {position}')
            if obj['status'] == 'carried':
                from scipy.spatial.transform import Rotation
                slot = engine.state['slots'][obj['heldBy']]
                tip = np.asarray(slot['position'])
                xyz = tip + Rotation.from_quat(slot['quaternion']).apply(obj['offset'])
                if not np.allclose(xyz, obj['positionMm'], atol=1e-8, rtol=0):
                    raise ValueError(f'{key} left its hand at {position}')
            if old != current:
                self.exposure['objectTransitions'].append([position, key, obj['kind'], *current])
                self.object_states[key] = current
        self.report['exposure'] = self.exposure

    def boundary(self, engine, position):
        self.exposure_boundary(engine, position)
        if position < len(self.starts) and self.starts[position]['stateId'] != engine.state_id:
            raise ValueError(f'Captured action start differs at {position}')
        if not position:
            return
        index = position-1
        frame, row = self.ends[index], self.rows[index]
        history = engine.history[-1]
        if (row['position'] != index or row['native'] != history or row['stateId'] != engine.state_id
                or frame['stateId'] != engine.state_id or frame['kind'] != 'action_boundary'):
            raise ValueError(f'Captured accepted history or end state differs at {position}')
        metrics = engine.metrics()
        expected = {'time': engine.state['elapsedSeconds'], 'removedCells': len(engine.state['removed']),
                    'targetRemoved': metrics['targetRemovedMm3'], 'nonTargetRemoved': metrics['nonTargetRemovedMm3'],
                    'originalResidual': original_target_residual_mm3(engine), 'contactCells': len(engine.state['contact']),
                    'tools': {hand: {key: slot[key] for key in ('instrument', 'position', 'quaternion', 'inserted')}
                              for hand, slot in engine.state['slots'].items()}}
        for key, value in expected.items():
            if frame[key] != value:
                raise ValueError(f'Captured {key} differs at {position}')
        if 'jawMm' in frame:
            jaws = {hand: slot['parameters'].get('jawOpeningMm') for hand, slot in engine.state['slots'].items()}
            if jaws != frame['jawMm']:
                raise ValueError(f'Captured jaw settings differ at {position}')


def collect(episode, output, peer=None, profile=False, check_each_kernel=False, capture=None):
    record = read_case_artifacts(episode)['surgical_toolbox']
    branch = record['branchId']
    saved = record['branches'][branch]
    scene = load_environment(episode, branch_id=branch)
    source_hash = sha256_file(episode)
    source_identity = scene.identity
    audit = CaptureAudit(capture, source_hash, branch, scene, saved['actions']) if capture else None
    boundaries = []
    transition_ms = []
    executors = Counter()
    fallbacks = Counter()
    groups = defaultdict(list)
    previous = None

    def boundary(engine):
        nonlocal previous
        entered = perf_counter_ns()
        position = len(boundaries)
        if position:
            action = saved['actions'][position - 1]
            elapsed = (entered - previous) * 1e-6
            transition_ms.append(elapsed)
            groups[action['operation']].append(elapsed)
            executor = 'cpp-resident' if engine.__dict__.get('_resident') is not None else 'python'
            executors[executor] += 1
            if engine.__dict__.get('_native_disabled'):
                fallbacks[str(engine.__dict__['_native_disabled'])] += 1
        boundaries.append(engine.state_id)
        if audit:
            audit.boundary(engine, position)
        previous = perf_counter_ns()

    native_backend.reset_counters()
    profiler = cProfile.Profile() if profile else None
    if profiler:
        profiler.enable()
    start = perf_counter_ns()
    # Reproduce the original entry path on the same case and same current
    # physics. Every native binding then performs its own full disk check.
    guard = patch.object(native_backend, 'operative_python_substep',
                         lambda advance, *args: advance(*args)) if check_each_kernel else nullcontext()
    with guard:
        engine = SurgicalEngine.restore(scene, saved, use_cache=False, force_verify=True, on_boundary=boundary)
    took_ms = (perf_counter_ns() - start) * 1e-6
    if profiler:
        profiler.disable()
    backend = native_backend.backend_identity()
    if engine.snapshot() != saved:
        raise ValueError('Full restored snapshot differs from the saved episode')
    if source_hash != sha256_file(episode) or source_identity != engine.scene.identity:
        raise ValueError('Replay changed input source identity')
    recorded_history = record.get('timeline', {}).get(branch)
    if recorded_history is not None:
        if len(recorded_history) != len(engine.history):
            raise ValueError('Stored timeline length differs')
        for i, row in enumerate(recorded_history):
            if (row['stateBefore'], row['stateAfter']) != (boundaries[i], boundaries[i + 1]):
                raise ValueError(f'Stored timeline differs at boundary {i + 1}')

    exact = {'archiveSHA256': source_hash, 'branch': branch, 'scene': scene.identity,
             'stateIds': boundaries, 'historyDigest': semantic_digest(engine.history),
             'endpoint': engine.state_id, 'metrics': engine.metrics(),
             'actionCount': len(engine.history), 'snapshotMatches': True}
    result = {'identity': exact, 'backend': backend, 'transitionAuthority': dict(executors),
              'residentDeclineReasons': dict(fallbacks), 'sourcePropertyNames': sorted(scene.properties),
              'hardware': platform.uname()._asdict(), 'profiled': profile,
              'guardControl': 'full-file-check-per-kernel' if check_each_kernel else 'installed',
              'replayWallMs': took_ms, 'actionWallMs': transition_ms,
              'latency': timings(transition_ms) if transition_ms else None,
              'latencyByOperation': {op: timings(v) for op, v in groups.items()},
              'perBoundaryChecks': len(boundaries), 'storedTimelineChecked': recorded_history is not None}
    if profiler:
        result['cpuAttribution'] = profile_rows(profiler)
    if audit:
        result['captureAudit'] = audit.report
    if peer:
        comparison = json.loads(peer.read_text())['identity']
        if exact != comparison:
            raise ValueError('Paired replay diverged: ' + ', '.join(k for k in exact if exact[k] != comparison.get(k)))
        result['exactBackendParity'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--peer', type=Path, help='Require exact state IDs, history, endpoint and metrics from the other backend.')
    parser.add_argument('--profile', action='store_true', help='Use a separate run for cProfile CPU attribution.')
    parser.add_argument('--check-each-kernel', action='store_true', help='Exercise the previous per-binding file validation for a paired timing control.')
    parser.add_argument('--capture', type=Path, help='Audit the matching complete hidden desktop capture against every cold replay boundary.')
    args = parser.parse_args()
    result = collect(args.episode, args.output, args.peer, args.profile, args.check_each_kernel, args.capture)
    print(json.dumps({k: result[k] for k in ('transitionAuthority', 'residentDeclineReasons', 'replayWallMs', 'latency')}))
