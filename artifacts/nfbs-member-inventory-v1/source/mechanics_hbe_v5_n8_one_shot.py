"""Release-gated, one-attempt HBE v5 N8 native supervisor preparation.

There is deliberately no execution release in Git. Import and inspection do not
launch FEBio. A separate reviewed release must bind committed source and the
one fresh attempt directory before ``--execute`` can spend one native call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = 'manifests/experiments/hbe-v5-n8-one-shot-preparation-v1.json'
CANARY = 'manifests/experiments/hbe-v5-n8-canary-source-only-v1.json'
OUTPUT = 'outputs/mechanics/hbe-v5-n8-one-shot-v1/attempt-01'
OLD_RUN = 'outputs/mechanics/hbe-01-03-poc-v1/experiment-accelerate-csc-v1/runs/compression-N8-S60-reference'
OLD_INDEX = 'artifacts/mechanics/hbe-accelerate-experiment-result-v1/raw-output-index.json'
RUN_ID = 'compression:N8:S60:reference'
CAPS = {'native_calls': 1, 'attempts': 1, 'wall_seconds': 90,
        'sampled_process_group_rss_bytes': 3 * 1024**3,
        'active_output_bytes': 64 * 1024**2, 'numerical_threads': 1}
SOURCE_PATHS = (
    'scripts/mechanics_hbe_v5_n8_one_shot.py',
    'scripts/mechanics_hbe_v5_n8_canary.py',
    'scripts/mechanics_hbe_branch_calibration_v5.py',
    'scripts/mechanics_hbe_branch_calibration_v4.py',
    'scripts/mechanics_hbe_v5_source_bindings.py',
    'scripts/mechanics_hbe_v5_frame.py',
    'scripts/mechanics_hbe_v5_stream.py',
    'scripts/mechanics_hbe_backend.py',
    'scripts/mechanics_hbe_access.py',
    'scripts/mechanics_hbe_evaluation.py',
    'scripts/mechanics_hbe_readout.py',
    'scripts/mechanics_hbe_halfheight_readout.py',
    'scripts/mechanics_hbe_outputs.py',
    'scripts/mechanics_hbe_physics.py',
    'scripts/mechanics_febio_verification.py',
    'scripts/mechanics_patient_constraints.py',
    'scripts/febio_runtime.py',
)
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE',
              'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
HEX = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
BACKEND = re.compile(r'\*\s+Selecting linear solver accelerate\s+\*\Z', re.I)
EXPECTED_NATIVE = {'specimen.feb', 'solver.log', 'nodes.log', 'elements.log',
                   'console.txt', 'receipt.json'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local(relative: str, *, root: Path = ROOT) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or '..' in path.parts:
        raise ValueError('Repository-relative path required')
    base = root.resolve()
    current = base
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Symlinked input or output path refused')
    result = (base / path).resolve()
    if not result.is_relative_to(base):
        raise ValueError('Path escapes repository')
    return result


def file_hash(path: Path, *, maximum: int = 128 * 1024**2) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ValueError('Bound file is absent, special or oversized')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Bound file changed while hashing')
    return digest.hexdigest()


def bound(binding: dict, expected: str, *, root: Path = ROOT,
          maximum: int = 1024**2) -> bytes:
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or binding['path'] != expected or not isinstance(binding['sha256'], str)
            or not HEX.fullmatch(binding['sha256'])):
        raise ValueError('Exact path/SHA256 binding required')
    path = local(expected, root=root)
    if file_hash(path, maximum=maximum) != binding['sha256']:
        raise ValueError('Bound file hash differs: ' + expected)
    data = path.read_bytes()
    if len(data) > maximum or sha(data) != binding['sha256']:
        raise ValueError('Bound file changed after hash: ' + expected)
    return data


def git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null',
               GIT_CONFIG_SYSTEM='/dev/null', GIT_OPTIONAL_LOCKS='0')
    return env


def source_hashes(release: dict, *, root: Path = ROOT) -> dict[str, str]:
    commit = release.get('source_commit')
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError('Full source commit required')
    if set(release.get('source_bindings', {})) != set(SOURCE_PATHS):
        raise ValueError('Full executing source closure required')
    head = subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=root,
                          env=git_env(), capture_output=True, check=True,
                          timeout=10).stdout.decode().strip()
    if head != commit:
        raise ValueError('Checkout differs from source release commit')
    result = {}
    for relative in SOURCE_PATHS:
        data = bound(release['source_bindings'][relative], relative, root=root)
        name = f'{commit}:{relative}'
        size = subprocess.run(['/usr/bin/git', 'cat-file', '-s', name], cwd=root,
                              env=git_env(), capture_output=True, check=True,
                              timeout=10).stdout
        if int(size) > 1024**2:
            raise ValueError('Committed source exceeds one MiB')
        original = subprocess.run(['/usr/bin/git', 'cat-file', 'blob', name], cwd=root,
                                  env=git_env(), capture_output=True, check=True,
                                  timeout=10).stdout
        if len(original) > 1024**2 or sha(original) != sha(data):
            raise ValueError('Committed source differs: ' + relative)
        result[relative] = sha(data)
    return result


def audit_imports(*, root: Path = ROOT) -> None:
    allowed = set(SOURCE_PATHS)
    base = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('scripts.'):
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        if not path.is_relative_to(base / 'scripts') or str(path.relative_to(base)) not in allowed:
            raise ValueError('Unbound executing scripts import: ' + name)
    fixture = sys.modules.get('scripts.mechanics_patient_constraints')
    if fixture is not None:
        patch = getattr(fixture, 'patch', None)
        if patch is None or Path(patch.__file__).resolve() != base / 'scripts/mechanics_febio_verification.py':
            raise ValueError('Unbound dynamically loaded checker')


def validate_preparation(value: dict) -> None:
    if (not isinstance(value, dict)
            or value.get('schema') != 'hbe-v5-n8-one-shot-preparation-v1'
            or value.get('status') != 'prepared_not_released_or_executed'
            or value.get('release') is not None or value.get('run_id') != RUN_ID
            or value.get('canary_manifest') != CANARY
            or value.get('output_directory') != OUTPUT or value.get('caps') != CAPS
            or value.get('phase_gates') != {'other_v5_rows': False, 'fit': False,
                                           'held_out_torque': False, 'physical_validation': False}):
        raise ValueError('One-shot N8 preparation or phase gate changed')


def verify_old_control(preparation: dict, *, root: Path = ROOT) -> dict:
    """Authenticate the observed old-endpoint control used only to set caps."""
    control = preparation.get('old_control', {})
    if set(control) != {'solver_execution', 'raw_output_index',
                        'observed_old_endpoint_wall_seconds',
                        'observed_old_endpoint_nodes_bytes',
                        'observed_old_endpoint_elements_bytes',
                        'observed_old_endpoint_solver_bytes',
                        'observed_old_endpoint_console_bytes'}:
        raise ValueError('Old N8 cap evidence changed')
    execution = json.loads(bound(control['solver_execution'], OLD_RUN + '/solver-execution.json',
                                 root=root, maximum=16*1024))
    index = json.loads(bound(control['raw_output_index'], OLD_INDEX,
                             root=root, maximum=1024**2))
    if (execution.get('status') != 'completed' or execution.get('exit_code') != 0
            or execution.get('seconds_cap') != 90
            or execution.get('elapsed_seconds') != control['observed_old_endpoint_wall_seconds']
            or not 0 < execution['elapsed_seconds'] < CAPS['wall_seconds']
            or execution.get('command', [])[-4:] != ['-i', 'specimen.feb', '-o', 'solver.log']):
        raise ValueError('Historical old-endpoint solver evidence differs')
    for name, field in [('nodes.log', 'nodes'), ('elements.log', 'elements'),
                        ('solver.log', 'solver'), ('console.txt', 'console')]:
        key = 'runs/compression-N8-S60-reference/' + name
        record = index.get('files', {}).get(key)
        if (not isinstance(record, dict) or record.get('bytes') !=
                control['observed_old_endpoint_' + field + '_bytes']
                or record.get('sha256') != file_hash(local(OLD_RUN + '/' + name, root=root),
                                                     maximum=32*1024**2)):
            raise ValueError('Historical saved N8 output evidence differs')
    return {'elapsed_seconds': execution['elapsed_seconds'],
            'saved_output_bytes': sum(control['observed_old_endpoint_' + field + '_bytes']
                                      for field in ('nodes', 'elements', 'solver', 'console'))}


def validate_release(release: dict, *, root: Path = ROOT) -> dict:
    """Check all frozen inputs before any attempt directory or native launch."""
    keys = {'schema', 'status', 'run_id', 'source_commit', 'source_bindings',
            'preparation', 'canary', 'old_source_deck', 'native_mesh',
            'adapted_deck_sha256', 'backend_profile', 'runtime_identity',
            'output_directory'}
    if (not isinstance(release, dict) or set(release) != keys
            or release['schema'] != 'hbe-v5-n8-one-call-release-v1'
            or release['status'] != 'root_released_one_native_call'
            or release['run_id'] != RUN_ID or release['output_directory'] != OUTPUT):
        raise ValueError('Missing or altered separate one-call release')
    preparation = json.loads(bound(release['preparation'], PREPARATION, root=root))
    validate_preparation(preparation)
    old_control = verify_old_control(preparation, root=root)
    from scripts import mechanics_hbe_v5_n8_canary as canary
    from scripts import mechanics_hbe_backend as backend
    from scripts import mechanics_hbe_v5_source_bindings as sources
    if bound(release['canary'], CANARY, root=root) != sources._read_bound(
            root, {'path': CANARY, 'sha256': canary.MANIFEST_SHA256}, maximum=128*1024):
        raise ValueError('Canary manifest differs from pinned source preparation')
    canary.verify_preparation(root)
    manifest, decks = canary.expected_preparation(root)
    if (release['old_source_deck'] != manifest['old_source_deck']
            or release['native_mesh'] != manifest['full_native_mesh']
            or release['adapted_deck_sha256'] != manifest['new_endpoint_decks']['new_endpoint_accelerate']['sha256']
            or release['backend_profile'] != manifest['backend_profile']
            or release['runtime_identity'] != manifest['runtime_identity']
            or sha(decks['new_endpoint_accelerate']) != release['adapted_deck_sha256']):
        raise ValueError('Released N8 deck, mesh, runtime or profile differs')
    hashes = source_hashes(release, root=root)
    profile = backend.verify_profile(root, release['backend_profile'])
    audit_imports(root=root)
    if (profile['profile_id'] != 'accelerate_csc_v1'
            or profile['runtime_identity'] != release['runtime_identity']
            or profile['runtime']['executable_sha256'] !=
               profile['runtime']['libraries']['install/bin/febio4']):
        raise ValueError('Repaired runtime/profile identity differs')
    old = bound(release['old_source_deck'], manifest['old_source_deck']['path'],
                root=root, maximum=sources.MAX_SOURCE_BYTES)
    mesh = bound(release['native_mesh'], manifest['full_native_mesh']['path'],
                 root=root, maximum=sources.MAX_MESH_BYTES)
    return {'source_hashes': hashes, 'profile': profile,
            'old_control': old_control,
            'preparation_sha256': release['preparation']['sha256'],
            'canary_sha256': release['canary']['sha256'],
            'old_source': old, 'mesh': mesh,
            'deck': decks['new_endpoint_accelerate']}


def read_release(path: Path) -> tuple[bytes, tuple[int, int]]:
    if '..' in Path(path).parts:
        raise ValueError('Release path traversal refused')
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Symlinked release refused')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise ValueError('Bound regular release required')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise ValueError('Release changed before read')
        data = os.read(fd, 1024**2 + 1)
        after = os.fstat(fd)
        if (len(data) != opened.st_size or len(data) > 1024**2
                or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) !=
                   (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)):
            raise ValueError('Release changed during read')
    finally:
        os.close(fd)
    return data, (opened.st_dev, opened.st_ino)


def durable_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def active_bytes(directory: Path) -> int:
    total = 0
    count = 0
    deadline = time.monotonic() + 1
    stack = [(directory, 0)]
    while stack:
        base, depth = stack.pop()
        if depth > 16:
            raise ValueError('Output depth cap')
        with os.scandir(base) as entries:
            for entry in entries:
                count += 1
                if count > 256 or time.monotonic() > deadline:
                    raise ValueError('Output scan work cap')
                st = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(st.st_mode):
                    total += st.st_size
                elif stat.S_ISDIR(st.st_mode):
                    stack.append((Path(entry.path), depth+1))
                else:
                    raise ValueError('Native output contains link or special file')
                if total > CAPS['active_output_bytes']:
                    return total
    return total


def private_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if (key.startswith(('DYLD_', 'PYTHON', 'OMP_', 'MKL_', 'OPENBLAS_',
                            'VECLIB_', 'NUMEXPR_'))
                or key in {'LD_PRELOAD', 'LD_LIBRARY_PATH', '__PYVENV_LAUNCHER__', 'MKLROOT'}):
            env.pop(key)
    env.update(THREAD_ENV)
    return env


def supervise(executable: str, directory: Path, receipt: dict,
              *, rss_observer=None, popen=None, sleep=time.sleep) -> dict:
    if rss_observer is None:
        from scripts.febio_runtime import process_group_rss
        rss_observer = process_group_rss
        audit_imports()
    if popen is None:
        popen = subprocess.Popen
    command = [executable, '-noconfig', '-no_title', '-i', 'specimen.feb',
               '-o', 'solver.log']
    receipt.update(command=command, native_calls_attempted=1, status='starting',
                   exit_code=None, kill_reason=None,
                   peak_sampled_process_group_rss_bytes=0,
                   peak_sampled_active_output_bytes=0)
    durable_json(directory / 'receipt.json', receipt)
    started = time.monotonic()
    process = None
    try:
        with (directory / 'console.txt').open('xb') as console:
            process = popen(command, cwd=directory, env=private_environment(),
                            stdout=console, stderr=subprocess.STDOUT,
                            start_new_session=True)
            receipt['pid'] = process.pid
            durable_json(directory / 'receipt.json', receipt)
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= CAPS['wall_seconds']:
                    receipt['kill_reason'] = 'wall_cap'
                    break
                code = process.poll()
                rss, members = rss_observer(process.pid, timeout_seconds=min(1., CAPS['wall_seconds']-elapsed))
                active = active_bytes(directory)
                receipt['peak_sampled_process_group_rss_bytes'] = max(
                    receipt['peak_sampled_process_group_rss_bytes'], rss)
                receipt['peak_sampled_active_output_bytes'] = max(
                    receipt['peak_sampled_active_output_bytes'], active)
                if rss > CAPS['sampled_process_group_rss_bytes']:
                    receipt['kill_reason'] = 'process_group_rss_cap'
                elif active > CAPS['active_output_bytes']:
                    receipt['kill_reason'] = 'active_output_cap'
                elif code is not None:
                    receipt['exit_code'] = code
                    if members:
                        receipt['kill_reason'] = 'descendants_outlived_solver'
                    break
                elif not members:
                    if process.poll() is None:
                        raise RuntimeError('Running process group RSS unavailable')
                    continue
                if receipt['kill_reason']:
                    break
                receipt['elapsed_seconds'] = elapsed
                durable_json(directory / 'receipt.json', receipt)
                sleep(.1)
    except BaseException as error:
        receipt['kill_reason'] = receipt['kill_reason'] or 'supervision_exception'
        receipt['supervision_error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if process is not None:
            try:
                if receipt['kill_reason'] or process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                receipt['exit_code'] = process.wait(timeout=2)
            except BaseException as error:
                receipt['kill_reason'] = receipt['kill_reason'] or 'cleanup_exception'
                receipt['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['status'] = ('native_exit_zero' if receipt['exit_code'] == 0
                             and receipt['kill_reason'] is None
                             and receipt['elapsed_seconds'] < CAPS['wall_seconds']
                             else 'failed_or_incomplete')
        durable_json(directory / 'receipt.json', receipt)
    return receipt


def inspect_outputs(directory: Path, context: dict, receipt: dict,
                    *, root: Path = ROOT) -> dict:
    if receipt['status'] != 'native_exit_zero':
        raise ValueError('Native exit or resource guard failed')
    if active_bytes(directory) > CAPS['active_output_bytes']:
        raise ValueError('Final active output cap')
    if {entry.name for entry in directory.iterdir()} != EXPECTED_NATIVE:
        raise ValueError('Unexpected or missing native output files')
    bindings = {}
    for name in EXPECTED_NATIVE - {'receipt.json'}:
        path = directory / name
        maximum = 32*1024**2 if name in {'solver.log', 'console.txt'} else CAPS['active_output_bytes']
        bindings[name] = {'path': str(path.relative_to(root)),
                          'sha256': file_hash(path, maximum=maximum),
                          'bytes': path.stat().st_size}
    if bindings['specimen.feb']['sha256'] != sha(context['deck']):
        raise ValueError('Executed deck changed')
    for name in ('console.txt', 'solver.log'):
        text = (directory / name).read_text(encoding='ascii')
        selections = [line.strip() for line in text.splitlines()
                      if 'selecting linear solver' in line.lower()]
        if (len(selections) != 1 or not BACKEND.fullmatch(selections[0])
                or re.search(r'\b(?:fallback|fall\s+back|switching\s+(?:linear\s+)?solver)\b',
                             text, re.I)):
            raise ValueError('Exactly one actual Accelerate selection required in both logs')
    from scripts import mechanics_hbe_v5_frame as frame
    from scripts import mechanics_hbe_v5_stream as reader
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    from scripts import mechanics_hbe_v5_source_bindings as sources
    audit_imports(root=root)
    study = (root / v5.DECLARATION_PATH).read_bytes()
    prior_path = json.loads(study)['previous_v4']['path']
    prior = (root / prior_path).read_bytes()
    contract = frame.verified_schedule(study, prior, RUN_ID,
                                       context['old_source'], context['deck'],
                                       json.loads(bound({'path': CANARY,
                                                         'sha256': context['canary_sha256']},
                                                        CANARY, root=root))['deck_adapter_receipt'])
    mesh = json.loads(context['mesh'])
    with (directory / 'nodes.log').open(encoding='utf-8', errors='strict') as nodes, \
         (directory / 'elements.log').open(encoding='utf-8', errors='strict') as elements, \
         (directory / 'solver.log').open(encoding='utf-8', errors='strict') as solver:
        readout = reader.evaluate_stream(contract, mesh, nodes, elements, solver)
    if readout['frame_count'] != 61 or readout['numerical_passed'] is not True:
        raise ValueError('Full saved 61-frame N8 numerical readout failed')
    for name, record in bindings.items():
        if file_hash(directory / name) != record['sha256']:
            raise ValueError('Saved native output changed during readout')
    readout['provenance'].update(native_output_observed=True,
                                 generated_fixture_only=False,
                                 output_origin='one_supervised_native_FEBio_attempt',
                                 source_binding_checked=True,
                                 physical_validation_pass=None,
                                 measured_response_accessed=False,
                                 patient_data_accessed=False)
    return {'native_output_bindings': bindings, 'saved_numerical_readout': readout}


def execute(release_path: Path, *, root: Path = ROOT) -> dict:
    raw, identity = read_release(release_path)
    release = json.loads(raw)
    context = validate_release(release, root=root)
    directory = local(OUTPUT, root=root)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'hbe-v5-n8-one-call-receipt-v1', 'status': 'reserved',
               'run_id': RUN_ID, 'release_sha256': sha(raw),
               'release_path': str(release_path.resolve()),
               'source_commit': release['source_commit'],
               'source_hashes_before': context['source_hashes'],
               'preparation_sha256': context['preparation_sha256'],
               'canary_sha256': context['canary_sha256'],
               'old_source_deck_sha256': release['old_source_deck']['sha256'],
               'native_mesh_sha256': release['native_mesh']['sha256'],
               'adapted_deck_sha256': release['adapted_deck_sha256'],
               'runtime_identity_sha256': release['runtime_identity']['sha256'],
               'backend_profile_sha256': release['backend_profile']['sha256'],
               'caps': CAPS, 'no_retry': True,
               'sampling_limit': 'Brief between-sample RSS/output peaks may be missed.',
               'interpretation': 'Numerical specimen software check only; no measured fit or patient validation.'}
    durable_json(directory / 'receipt.json', receipt)
    try:
        with (directory / 'specimen.feb').open('xb') as deck:
            deck.write(context['deck'])
            deck.flush()
            os.fsync(deck.fileno())
        if file_hash(directory / 'specimen.feb') != release['adapted_deck_sha256']:
            raise ValueError('Copied deck differs before native call')
        supervise(context['profile']['runtime']['executable'], directory, receipt)
        if receipt['status'] != 'native_exit_zero':
            raise ValueError('Native attempt failed or exceeded cap')
        receipt.update(inspect_outputs(directory, context, receipt, root=root))
        if (read_release(release_path) != (raw, identity)
                or source_hashes(release, root=root) != context['source_hashes']
                or bound(release['preparation'], PREPARATION, root=root) !=
                   (root / PREPARATION).read_bytes()):
            raise ValueError('Released input changed after native call')
        from scripts import mechanics_hbe_backend as backend
        after = backend.verify_profile(root, release['backend_profile'])
        audit_imports(root=root)
        if (after['runtime_identity'] != release['runtime_identity']
                or after['inputs'] != context['profile']['inputs']):
            raise ValueError('Runtime/profile changed after native call')
        for key, path in [('canary', CANARY),
                          ('old_source_deck', release['old_source_deck']['path']),
                          ('native_mesh', release['native_mesh']['path'])]:
            bound(release[key], path, root=root,
                  maximum=128*1024 if key == 'canary' else 128*1024**2)
        from scripts import mechanics_hbe_v5_n8_canary as canary
        canary.verify_preparation(root)
        receipt['status'] = 'passed_numerical_software_only'
    except BaseException as error:
        receipt['status'] = 'failed_or_incomplete'
        receipt['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            receipt['final_active_output_bytes'] = active_bytes(directory)
        except BaseException as error:
            receipt['status'] = 'failed_or_incomplete'
            receipt['final_output_error'] = {'type': type(error).__name__, 'message': str(error)}
        durable_json(directory / 'receipt.json', receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise ValueError('Explicit --execute and separate release required')
    result = execute(args.release)
    print(json.dumps({'status': result['status'],
                      'receipt': str(local(OUTPUT) / 'receipt.json')}))
    return 0 if result['status'] == 'passed_numerical_software_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
