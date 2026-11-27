"""Profile uncached public-action replay and compare every accepted boundary across backends."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import cProfile
import hashlib
import gzip
import json
from pathlib import Path
import platform
import pstats
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab import native_backend
from resectionlab.core import semantic_digest
from resectionlab.imaging import read_case_artifacts
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_bridge import display_timeline
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


def transported_digest(value):
    # JSON.stringify on the desktop writes 1 for a Python 1.0. Both denote
    # the same double; preserve boolean/string distinctions and all fractions.
    def normalized(item):
        if type(item) is dict:
            return {key: normalized(x) for key,x in item.items()}
        if type(item) is list:
            return [normalized(x) for x in item]
        if type(item) is float and item.is_integer() and abs(item) < 2**53:
            return int(item)
        return item
    return semantic_digest(normalized(value))


def boundary_witness(engine):
    slots = engine.state['slots']
    metrics = engine.metrics()
    return {'stateId':engine.state_id, 'switches':{s:t['active'] for s,t in slots.items()},
            'jaws':{s:t['parameters'].get('jawOpeningMm') for s,t in slots.items()},
            'displayHistory':transported_digest(display_timeline(engine.history[-1:])[0]) if engine.history else None,
            'fields':{'time':engine.state['elapsedSeconds'], 'removedCells':len(engine.state['removed']),
                      'targetRemoved':metrics['targetRemovedMm3'], 'nonTargetRemoved':metrics['nonTargetRemovedMm3'],
                      'originalResidual':original_target_residual_mm3(engine), 'contactCells':len(engine.state['contact']),
                      'tools':{s:{k:t[k] for k in ('instrument','position','quaternion','inserted')} for s,t in slots.items()}}}


class CaptureAudit:
    """Compare original compositor receipts with independently restored action endpoints."""

    def __init__(self, path, archive_hash, branch, scene, case_hash, action_count):
        data = json.loads(path.read_text())
        if (data.get('windowEverVisible') is not False or not data.get('passed')
                or data['sourceFileSha256'] != archive_hash or data['branchId'] != branch
                or data['sceneId'] != scene or data['sourceCaseHash'] != case_hash
                or data['steps'] != action_count or not data['completeEpisode']):
            raise ValueError('Capture is not a complete hidden replay of this exact archive, scene and branch')
        self.rows = data['actions']
        self.starts = []
        self.ends = []
        self.kinds = Counter()
        compositor_ms = defaultdict(list)
        steady_ms = defaultdict(list)
        previous_time = -float('inf')
        previous = None
        action_frames = 0
        chapter = None
        for frame in data['frames']:
            if 'position' not in frame:  # Exterior overview, outside the accepted native timeline.
                continue
            i, j, count = (frame[k] for k in ('position', 'frameIndex', 'frameCount'))
            if j == 0:
                action_frames = count
            if (frame['branchId'] != branch or frame['time'] < previous_time - 1e-8 or count < 2
                    or count != action_frames or not 0 <= j < count or (i, j) != (previous or (0, 0))):
                raise ValueError(f'Unordered, wrong-branch or missing observation at frame {(i, j)}')
            previous = (i+1, 0) if j == count-1 else (i, j+1)
            previous_time = frame['time']
            self.kinds[frame['kind']] += 1
            for key, value in frame.get('timing', {}).items():
                if value is not None:
                    compositor_ms[key].append(value)
                    if chapter == frame['chapter']:
                        steady_ms[key].append(value)
            chapter = frame['chapter']
            if j == 0:
                self.starts.append(frame)
            if j == count-1:
                self.ends.append(frame)
        if (previous != (action_count, 0)
                or not len(self.starts) == len(self.ends) == len(self.rows) == action_count):
            raise ValueError('Missing terminal observations or accepted actions')
        self.report = {'captureSHA256': sha256_file(path), 'renderer': data['rendererAsset'],
                       'matchedActions': action_count, 'observations': dict(self.kinds)}
        if compositor_ms:
            self.report['desktopObservationMs'] = {k:timings(v) for k,v in compositor_ms.items()}
            self.report['sameChapterObservationMs'] = {k:timings(v) for k,v in steady_ms.items()}

    def boundary(self, witness, position):
        switches = witness['switches']
        if position < len(self.starts):
            start = self.starts[position]
            if (start['stateId'] != witness['stateId'] or start['kind'] != 'action_start'
                    or ('active' in start and start['active'] != switches)):
                raise ValueError(f'Captured action start or latched switch differs at {position}')
        if not position:
            return
        index = position-1
        frame, row = self.ends[index], self.rows[index]
        history = row['native']
        if (row['position'] != index or transported_digest(history) != witness['displayHistory'] or row['stateId'] != witness['stateId']
                or frame['status'] != history['status'] or frame['operation'] != history['action']['operation']
                or frame['stateId'] != witness['stateId'] or frame['kind'] != 'action_boundary'):
            raise ValueError(f'Captured accepted history or end state differs at {position}')
        expected = witness['fields']
        for key, value in expected.items():
            if frame[key] != value:
                raise ValueError(f'Captured {key} differs at {position}')
        if 'jawMm' in frame:
            jaws = witness['jaws']
            if jaws != frame['jawMm']:
                raise ValueError(f'Captured jaw settings differ at {position}')
        if 'active' in frame:
            actor = history['action']['slot']
            other = 'left' if actor == 'right' else 'right'
            if frame['active'][other] != switches[other]:
                raise ValueError(f'Captured persistent hand activation differs at {position}')
            active = frame['operation'] in ('apply', 'hold', 'grasp', 'activate')
            inactive = frame['operation'] in ('withdraw', 'deactivate', 'release')
            if (active or inactive) and frame['active'][actor] != active:
                raise ValueError(f'Captured acting tip activation differs at {position}')


def live_bridge_timings(path, exact, history):
    data = json.loads(path.read_text())
    if data['sha256'] != exact['archiveSHA256'] or data['branchId'] != exact['branch']:
        raise ValueError('Live receipt belongs to a different archive or branch')
    rows = data['actions']
    offset = data['parent']['position']
    times = []
    groups = defaultdict(list)
    for row, accepted in zip(rows, history[offset:], strict=True):
        for key in ('action', 'stateBefore', 'stateAfter', 'events', 'status', 'completedFraction'):
            if row[key] != accepted[key]:
                raise ValueError(f'Live {key} differs at recorded action {row["index"]}')
        times.append(row['wallMs'])
        groups[row['phase']].append(row['wallMs'])
    return {'receiptSHA256': sha256_file(path), 'matchedActions': len(rows),
            'method': 'Wall time around public BridgeSession command in the live Python driver, excluding GPU.',
            'all': timings(times) if times else None,
            'byPhase': {name:timings(values) for name,values in groups.items()}}


def scrub_capture(episode, capture):
    """Open the saved file in a fresh production bridge and seek backwards, forwards and to a paused frame."""
    from resectionlab.desktop_bridge import BridgeSession, _Request
    data = json.loads(capture.read_text())
    native = [f for f in data['frames'] if f.get('kind') in ('microstep','action_boundary')]
    interior = [f for f in native if f['kind']=='microstep']
    if not interior:
        raise ValueError('Capture has no physical microstep for a paused-state comparison')
    middle = interior[len(interior)//2]
    route = [native[-1], interior[0], middle, middle, native[-1]]
    receipts = []
    with TemporaryDirectory() as scratch:
        session = BridgeSession(Path(scratch)/'transfer')
        def request(operation, **payload):
            return session.execute(operation, payload, _Request(operation), lambda *_:None)
        try:
            opened = request('loadCase', path=str(episode.resolve(strict=True)))
            def command(op, **kwargs):
                return request('surgicalRehearsal', caseHash=opened['caseHash'], command=op, **kwargs)
            live = command('inspect')
            editable = live['view']['stateId']
            if live['branchId'] != data['branchId'] or editable != data['actions'][-1]['stateId']:
                raise ValueError('Fresh bridge opened a different branch or editable endpoint')
            for frame in route:
                start = perf_counter_ns()
                reply = command('seekFrame', branchId=data['branchId'], position=frame['position'],
                                frameIndex=frame['frameIndex'], expectedStateId=editable)
                elapsed = (perf_counter_ns()-start)*1e-6
                view = reply['view']
                observed = reply['playback']
                if (not observed['readOnly'] or view['stateId'] != frame['stateId']
                        or observed['observedTimeSeconds'] != frame['time']
                        or view['metrics']['targetRemovedMm3'] != frame['targetRemoved']
                        or any(any(view['slots'][s][k] != value for k,value in fields.items())
                               for s,fields in frame['tools'].items())):
                    raise ValueError('Cold nonsequential desktop seek differs from the recorded raster state')
                receipts.append({'position':frame['position'], 'frameIndex':frame['frameIndex'],
                                 'state':view['stateId'], 'wallMs':elapsed})
            if command('inspect')['view']['stateId'] != editable:
                raise ValueError('Read-only frame seeks mutated the editable endpoint')
        finally:
            session.close()
    return {'editableEndpoint':editable, 'seeks':receipts, 'archiveRemainsSHA256':sha256_file(episode)}


def collect(episode, output, peer=None, profile=False, check_each_kernel=False, capture=None, milestones=None, live=None, scrub=False, witness_path=None):
    record = read_case_artifacts(episode)['surgical_toolbox']
    branch = record['branchId']
    saved = record['branches'][branch]
    scene = load_environment(episode, branch_id=branch)
    source_hash = sha256_file(episode)
    source_identity = scene.identity
    witness_identity = {'archive_hash':source_hash,'branch':branch,'scene':scene.identity,
                        'case_hash':scene.source['caseHash'],'action_count':len(saved['actions'])}
    audit = CaptureAudit(capture, **witness_identity) if capture else None
    witnesses = []
    milestone_audit = None
    if milestones:
        from glioma_episode_milestones import Milestones
        milestone_audit = Milestones(scene)
    boundaries = []
    transition_ms = []
    executors = Counter()
    fallbacks = Counter()
    groups = defaultdict(list)
    previous = None
    observer_ns = 0

    def boundary(engine):
        nonlocal previous, observer_ns
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
        if audit or witness_path:
            frame = boundary_witness(engine)
            if audit:
                audit.boundary(frame, position)
            if witness_path:
                witnesses.append(frame)
        if milestone_audit:
            milestone_audit.boundary(engine, position)
        previous = perf_counter_ns()
        observer_ns += previous-entered

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
              'replayWallMs': took_ms, 'observerWallMs':observer_ns*1e-6,
              'replayExcludingObserverMs':took_ms-observer_ns*1e-6, 'actionWallMs': transition_ms,
              'latency': timings(transition_ms) if transition_ms else None,
              'latencyByOperation': {op: timings(v) for op, v in groups.items()},
              'perBoundaryChecks': len(boundaries), 'storedTimelineChecked': recorded_history is not None}
    if profiler:
        result['cpuAttribution'] = profile_rows(profiler)
    if witness_path:
        data = json.dumps({'identity':witness_identity,'boundaries':witnesses},allow_nan=False,separators=(',',':')).encode()
        witness_path.write_bytes(gzip.compress(data,compresslevel=4,mtime=0))
        result['witness'] = {'file':witness_path.name,'sha256':sha256_file(witness_path),'boundaries':len(witnesses)}
    if audit:
        result['captureAudit'] = audit.report
    if scrub:
        if not capture:
            raise ValueError('--scrub requires --capture')
        result['coldDesktopScrub'] = scrub_capture(episode, capture)
    if live:
        result['liveBridge'] = live_bridge_timings(live, exact, engine.history)
    if milestone_audit:
        evidence = {'archiveSHA256': source_hash, 'branch': branch, 'scene': scene.identity,
                    'origin': boundaries[0], 'endpoint': boundaries[-1],
                    'actionCount': len(boundaries)-1, **milestone_audit.finish(engine)}
        milestones.write_text(json.dumps(evidence, allow_nan=False, indent=2)+'\n')
        result['milestoneAudit'] = {'file': milestones.name, 'sha256': sha256_file(milestones),
                                    'satisfied': evidence['satisfied'], 'missing': evidence['missing']}
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
    parser.add_argument('--milestones', type=Path, help='Write an independent physical milestone and final inventory audit.')
    parser.add_argument('--require-complete', action='store_true', help='Exit unsuccessfully if the saved episode misses any physical workflow gate.')
    parser.add_argument('--live', type=Path, help='Match live public action receipts and summarize bridge round-trip times.')
    parser.add_argument('--witness', type=Path, help='Keep compressed exact boundary evidence to check a film rendered later.')
    parser.add_argument('--scrub', action='store_true', help='Cold-open the episode in a separate bridge and seek captured frames out of order.')
    args = parser.parse_args()
    if args.require_complete and not args.milestones:
        parser.error('--require-complete requires --milestones')
    result = collect(args.episode, args.output, args.peer, args.profile, args.check_each_kernel, args.capture, args.milestones, args.live, args.scrub, args.witness)
    print(json.dumps({k: result[k] for k in ('transitionAuthority', 'residentDeclineReasons', 'replayWallMs', 'latency')}))
    if args.require_complete and not result['milestoneAudit']['satisfied']:
        sys.exit('Unmet physical gates: ' + ', '.join(result['milestoneAudit']['missing']))
