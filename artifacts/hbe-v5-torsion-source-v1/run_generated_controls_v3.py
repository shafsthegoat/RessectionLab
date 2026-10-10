"""Root-invoked generated controls only; never a torsion-data release.

Reuses the source-pinned OwnedStage cleanup and process-group RSS observer.
Worker file/process audit is accidental-access protection, not a hostile sandbox.
Invoke with .venv/bin/python -I -S -B; this file does not execute on import.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = HERE / 'stage'
OUT = ROOT / 'build/hbe-v5-torsion-generated-controls-v2/attempt-03'
RELEASE_SHA = 'e43c8d1986822400d5861dcd3fc3a9317bfa176fd13f6120605060fec30dc60f'
TEST_SHA = 'cab0bd901321ff6d0f99b714205ef4fc0e0a1484630fdc777f22fb663b340408'
TOTAL_SECONDS, WORK_SECONDS = 20., 16.
RSS_BYTES, LOG_BYTES = 512 * 1024**2, 1024**2


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('Source or fixture declaration changed: ' + str(path))
    return raw


def json_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def sources():
    release = json.loads(checked(HERE / 'false-release.json', RELEASE_SHA))
    if release['execution_released'] is not False or release['upstream'] is not None:
        raise ValueError('Only the false candidate may describe these generated controls')
    paths = {}
    for relative, expected in release['source_bindings'].items():
        path = STAGE / relative if (STAGE / relative).is_file() else ROOT / relative
        checked(path, expected)
        paths[str(path.resolve())] = expected
    test = STAGE / 'tests/test_hbe_v5_torsion_evaluation.py'
    checked(test, TEST_SHA)
    paths[str(test.resolve())] = TEST_SHA
    return release, paths, test


def worker():
    if os.getpid() != os.getpgrp() or os.environ.get('HBE_GENERATED_OWNER') != str(os.getppid()):
        raise ValueError('The outer runner must own the sole worker process group')
    _, allowed_sources, test = sources()
    runner_sha = os.environ.get('HBE_GENERATED_RUNNER_SHA256', '')
    if len(runner_sha) != 64 or any(c not in '0123456789abcdef' for c in runner_sha):
        raise ValueError('Exact parent runner hash required')
    checked(Path(__file__).resolve(), runner_sha)
    allowed_sources[str(Path(__file__).resolve())] = runner_sha
    area = OUT / 'worker'
    site = ROOT / '.venv/lib/python3.12/site-packages'
    runtime_roots = (Path(sys.base_prefix).resolve(), site.resolve())
    expected_denials, unexpected_denials = [], []
    checking_guard = [False]

    def refuse(reason):
        (expected_denials if checking_guard[0] else unexpected_denials).append(reason)
        raise PermissionError('generated-only test boundary: ' + reason)

    def guard(event, args):
        if event in {'subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty',
                     'os.posix_spawn', 'os.exec', 'socket.__new__', 'socket.connect',
                     'socket.bind', 'socket.getaddrinfo'}:
            refuse(event)
        if event != 'open' or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if path.is_relative_to(area):
            return
        if writing:
            refuse('write_outside_generated_worker:' + str(path))
        if str(path) in allowed_sources or path == Path(os.devnull):
            return
        if any(path.is_relative_to(base) for base in runtime_roots):
            return
        refuse('read_outside_source_runtime_generated_fixture:' + str(path))

    sys.addaudithook(guard)
    # Prove refusal before OS lookup; no real data/response body is opened.
    checking_guard[0] = True
    for relative in ('data/forbidden-generated-probe.zip', 'outputs/forbidden-predictions.json',
                     'artifacts/forbidden-response.csv', 'sources/forbidden-source.bin'):
        try:
            (ROOT / relative).open('rb')
        except PermissionError:
            pass
        else:
            raise AssertionError('Repository payload guard did not refuse')
    checking_guard[0] = False
    sys.path[:0] = [str(ROOT), str(site)]
    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HBE_TORSION_SOURCE_ROOT=str(STAGE),
                      TMPDIR=str(area / 'tmp'), TEMP=str(area / 'tmp'), TMP=str(area / 'tmp'))
    import pytest
    started = time.monotonic()
    code = 1
    error = None
    try:
        code = int(pytest.main(['-q', '-c', str(area / 'pytest.ini'),
            '--rootdir=' + str(area), '--confcutdir=' + str(test.parent),
            '--basetemp=' + str(area / 'fixtures'), '-o', 'cache_dir=' + str(area / 'cache'),
            '-o', 'log_file=' + str(area / 'pytest-internal.log'),
            '--junitxml=' + str(area / 'junit.xml'), str(test)]))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:1024]}
    finally:
        json_new(area / 'guard-receipt.json', {'version': 'hbe-torsion-generated-guard-v1',
            'pytest_exit_code': code, 'elapsed_pytest_seconds': time.monotonic() - started,
            'expected_refusals': expected_denials, 'unexpected_refusals': unexpected_denials,
            'error': error, 'actual_archive_response_reads': 0,
            'no_process_or_network_from_tests': True, 'test_sha256': TEST_SHA})
    return code if not unexpected_denials and error is None else 1


def parent():
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Use .venv/bin/python -I -S -B')
    started = time.monotonic()
    release, source_map, test = sources()
    OUT.mkdir(parents=True, exist_ok=False)
    area = OUT / 'worker'
    area.mkdir()
    (area / 'tmp').mkdir()
    (area / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    sys.path.insert(0, str(ROOT))
    from launchers.hbe_v5_ordinal9_continuation_v1 import OwnedStage
    from scripts.febio_runtime import process_group_rss
    record = {'version': 'hbe-torsion-generated-supervision-v1', 'status': 'reserved',
        'candidate_sha256': RELEASE_SHA, 'test_sha256': TEST_SHA,
        'runner_sha256': digest(Path(__file__).read_bytes()), 'source_bindings': source_map,
        'caps': {'total_seconds': TOTAL_SECONDS, 'worker_seconds': WORK_SECONDS,
                 'sampled_group_rss_bytes': RSS_BYTES, 'log_bytes': LOG_BYTES, 'threads': 1},
        'automatic_retry': False, 'exit_code': None, 'peak_sampled_group_rss_bytes': 0,
        'rss_samples': 0, 'sampling_limit': 'Brief peaks between samples may be missed.'}
    json_new(OUT / 'intent.json', record)
    owner = OwnedStage()
    process = None
    end = started + TOTAL_SECONDS
    work_end = started + WORK_SECONDS
    old = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Owned generated test runner interrupted')
    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, interrupted)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'DYLD_', 'LD_'))}
    env.update(HBE_GENERATED_OWNER=str(os.getpid()), HBE_GENERATED_RUNNER_SHA256=record['runner_sha256'],
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    try:
        command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(area / 'absent-cache'),
                   str(Path(__file__).resolve()), '--worker']
        record['command'] = command
        with (OUT / 'pytest.log').open('xb') as log:
            process = owner(command, cwd=area, env=env, stdout=log,
                            stderr=-2, start_new_session=True)
            record['pid'] = process.pid
            while True:
                remaining = work_end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Generated worker wall cap')
                code = process.poll()
                rss, members = process_group_rss(process.pid, timeout_seconds=min(.5, remaining))
                record['peak_sampled_group_rss_bytes'] = max(record['peak_sampled_group_rss_bytes'], rss)
                record['rss_samples'] += 1
                if rss > RSS_BYTES or (OUT / 'pytest.log').stat().st_size > LOG_BYTES:
                    raise RuntimeError('Generated worker RSS/log cap')
                if code is not None:
                    record['exit_code'] = code
                    if members:
                        raise RuntimeError('Worker left process-group members')
                    break
                if not members and process.poll() is None:
                    raise RuntimeError('Live worker RSS unavailable')
                time.sleep(min(.1, max(0., work_end - time.monotonic())))
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)[:1024]}
    finally:
        try:
            record['cleanup'] = owner.cleanup(process_group_rss, end)
            record['exit_code'] = process.poll() if process is not None else None
        except BaseException as exc:
            record['cleanup'] = {'contained': False, 'direct_child_reaped': False,
                                 'errors': [type(exc).__name__ + ':' + str(exc)[:512]]}
        for sig, handler in old.items():
            signal.signal(sig, handler)
        record['elapsed_seconds'] = time.monotonic() - started
        cleanup = record['cleanup']
        record['status'] = 'passed' if (record['exit_code'] == 0 and 'error' not in record
            and cleanup.get('contained') is True and cleanup.get('direct_child_reaped') is True
            and cleanup.get('remaining_members') == [] and not cleanup.get('errors')
            and cleanup.get('fallback_used') is False and record['elapsed_seconds'] < TOTAL_SECONDS) else 'failed'
        try:
            if record['status'] == 'passed':
                guard = json.loads((area / 'guard-receipt.json').read_bytes())
                if guard['pytest_exit_code'] != 0 or guard['unexpected_refusals'] or len(guard['expected_refusals']) != 4:
                    raise ValueError('Worker guard receipt differs')
                sources()  # Source-only postguard, no runtime or scientific outputs.
                checked(Path(__file__).resolve(), record['runner_sha256'])
            record['log_bytes'] = (OUT / 'pytest.log').stat().st_size if (OUT / 'pytest.log').exists() else 0
            record['log_sha256'] = None
            if record['log_bytes'] <= LOG_BYTES and (OUT / 'pytest.log').exists():
                record['log_sha256'] = digest((OUT / 'pytest.log').read_bytes())
            elif record['log_bytes'] > LOG_BYTES:
                raise ValueError('Final log cap exceeded')
            record['elapsed_seconds'] = time.monotonic() - started
            if record['elapsed_seconds'] >= TOTAL_SECONDS:
                raise TimeoutError('Final generated control deadline')
        except BaseException as exc:
            record['status'] = 'failed'
            record['finalization_error'] = {'type': type(exc).__name__, 'message': str(exc)[:1024]}
        json_new(OUT / 'supervisor.json', record)
    print(json.dumps({'status': record['status'], 'output': str(OUT),
                      'elapsed_seconds': record['elapsed_seconds']}))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--worker']):
        raise SystemExit('No options except the private --worker mode')
    raise SystemExit(worker() if sys.argv[1:] == ['--worker'] else parent())
