"""Profile the headless bridge against an existing desktop practice receipt."""
import argparse
from contextlib import closing, nullcontext
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import subprocess
from tempfile import TemporaryDirectory
import time

import numpy as np
import scipy

from resectionlab import native_backend
from resectionlab import surgical_deformation_v2 as lattice
from resectionlab.desktop_bridge import BridgeSession, _Request


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def profile_rows(profiler):
    return sorted(({
        'function': f'{Path(file).name}:{line}:{name}', 'calls': total,
        'ownMs': 1000 * own, 'cumulativeMs': 1000 * cumulative,
    } for (file, line, name), (_, total, own, cumulative, _) in pstats.Stats(profiler).stats.items()),
        key=lambda row: row['cumulativeMs'], reverse=True)


def run(args):
    recorded = json.loads(args.receipt.read_text())
    source = (args.source or Path(recorded['source'])).resolve(strict=True)
    source_sha = digest(source)
    if source_sha != recorded['originalSha256']:
        raise ValueError('Source package differs from the frozen desktop receipt')
    if args.variant == 'numpy':
        if not hasattr(native_backend, 'deformation_kernels'):
            raise ValueError('This revision has no batched NumPy comparison')
        native_backend.deformation_kernels = lambda: (None, None, 'benchmark-numpy-selected')
    scope = lattice.kernel_execution_scope if args.kernel_counts else nullcontext
    report = {
        'protocol': 'Replay frozen input actions through a headless BridgeSession; check every outcome and state after timing.',
        'gitHead': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source': str(source), 'sourceSha256': source_sha, 'receiptSha256': digest(args.receipt),
        'actionStreamSha256': hashlib.sha256(json.dumps(
            [row['result']['requested'] for row in recorded['actions']],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'platform': platform.platform(), 'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'latticeSource': str(Path(lattice.__file__).resolve()), 'latticeSourceSha256': digest(Path(lattice.__file__)),
        'nativeBackend': native_backend.strict_live_identity(), 'variant': args.variant, 'actions': [],
    }
    args.scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.scratch) as directory:
        scratch = Path(directory)
        with closing(BridgeSession(scratch / 'transactions', run_dir=scratch / 'runs')) as session:
            def call(operation, **values):
                return session.execute(operation, values, _Request('profile-rehearsal-kernels'), lambda *_: None)

            case_hash = call('loadCase', path=str(source))['caseHash']

            def practice(command, **values):
                return call('surgicalRehearsal', caseHash=case_hash, command=command, **values)

            parent = recorded['parent']
            opened = practice('inspect')
            opened = practice('selectBranch', branchId=parent['branchId'], expectedStateId=opened['view']['stateId'])
            if opened['position'] != parent['position']:
                opened = practice('seek', position=parent['position'], expectedStateId=opened['view']['stateId'])
            assert opened['view']['stateId'] == parent['stateId'] and case_hash == parent['caseHash']
            state = practice('practice', branchId=opened['branchId'], position=opened['position'],
                expectedStateId=opened['view']['stateId'],
                checkpointSeal=opened['practiceAdmission']['checkpointSeal'])
            episode = state['practiceEpisode']['episodeId']
            report['scene'] = {name: state['view']['scene'][name] for name in ('sceneId', 'shape', 'affine')}
            for index, recorded_action in enumerate(recorded['actions'][:args.through], 1):
                expected = recorded_action['result']
                request = expected['requested']
                before = state['view']['stateId']
                assert before == recorded_action['before']
                profile = cProfile.Profile() if index in args.profile else None
                if profile or args.kernel_counts:
                    native_backend.reset_counters()
                with scope() as kernels:
                    if profile: profile.enable()
                    start = time.perf_counter_ns()
                    state = practice('step', action=request, expectedStateId=before, expectedEpisodeId=episode)
                    ms = (time.perf_counter_ns() - start) / 1e6
                    if profile: profile.disable()
                view, outcome = state['view'], state['outcome']
                assert view['stateId'] == expected['witness']['stateId'], index
                assert state['position'] == expected['witness']['position'], index
                for field in ('status', 'reasonCode', 'completedFraction'):
                    assert outcome.get(field) == expected.get(field), (index, field, outcome.get(field), expected.get(field))
                for field, value in expected['witness']['metrics'].items():
                    assert view['metrics'][field] == value, (index, field)
                row = {
                    'step': index, 'action': request, 'sourceStateId': before, 'acceptedStateId': view['stateId'],
                    'status': outcome['status'], 'reasonCode': outcome.get('reasonCode'), 'headlessMs': ms,
                }
                if profile:
                    row['profile'] = profile_rows(profile)
                if args.kernel_counts or profile:
                    identity = native_backend.backend_identity()
                    row['deformationKernels'] = kernels
                    row['nativeCounters'] = identity.get('counters')
                    row['residentExecution'] = identity.get('residentExecution')
                    engine = session.surgical_sessions[case_hash][1]
                    row['residentFallbackReason'] = getattr(engine, '_native_disabled', None)
                report['actions'].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, help='An identical copy of the receipt source package')
    parser.add_argument('--variant', choices=('production', 'numpy'), default='production')
    parser.add_argument('--through', type=int, help='Stop after this many actions')
    parser.add_argument('--profile', type=int, action='append', default=[], metavar='STEP')
    parser.add_argument('--kernel-counts', action='store_true')
    parser.add_argument('--scratch', type=Path, default=Path('testcases/outputs'))
    args = parser.parse_args()
    if args.output.exists() or (args.through is not None and args.through < 1):
        parser.error('Choose a new output and a positive action limit')
    if args.kernel_counts and not hasattr(lattice, 'kernel_execution_scope'):
        parser.error('This revision has no request-local kernel counters')
    result = run(args)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
