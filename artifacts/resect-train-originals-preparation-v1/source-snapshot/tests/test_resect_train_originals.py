"""Offline protocol controls with opaque byte fixtures, never patient payloads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
BODY = b'opaque-transfer-control-0123456789'


@pytest.fixture
def intake(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    name = 'resect_originals_controls'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/acquire_resect_train_originals.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    original = subprocess.Popen

    def local_only(command, *args, **kwargs):
        if str(command[0]) == '/usr/bin/curl':
            pytest.fail('real network prohibited')
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', local_only)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', lambda *a, **k: pytest.fail('real network prohibited'))
    return module


@pytest.fixture
def case(intake, tmp_path):
    source = {'id': 'resect-original-control.bin', 'patient_group': 'RESECT:Case3', 'role': 'TRAIN',
              'source_url': intake.PREFIX + 'RESECT/NIFTI/Case3/US/control.bin',
              'bytes': len(BODY), 'expected_md5': hashlib.md5(BODY).hexdigest(), 'sha256': None,
              'destination': 'data/anatomy/resect-v1/Case3/control.bin'}
    manifest = {'sources': [source], 'bounds': {'compressed_cache_bytes': 1024**3,
                'free_disk_reserve_bytes': 0, 'max_attempts_per_object': 3,
                'minimum_cooldown_seconds': 900, 'supervisor_seconds': 300}}
    return intake.Queue(tmp_path, manifest), source


def trial(intake, queue, source, prefix=None):
    number = len(queue.attempts(source)) + 1
    path = queue.object_dir(source) / 'attempts' / f'{number:04d}'
    path.mkdir(parents=True)
    intent = {'source_id': source['id'], 'source_binding': queue.binding(source),
              'manifest_sha256': intake.MANIFEST_SHA, 'attempt': number,
              'prefix': prefix or {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}}
    queue.save(path / 'intent.json', intent)
    return path, intent


def response(path, source, offset, body, *, status=None, content_range=None, extra=''):
    status = status or (206 if offset else 200)
    headers = f'HTTP/1.1 {status} Response\r\nContent-Length: {source["bytes"] - offset}\r\n'
    if offset:
        actual_range = content_range or f'bytes {offset}-{source["bytes"]-1}/{source["bytes"]}'
        headers += f'Content-Range: {actual_range}\r\n'
    (path / 'headers.local').write_text(headers + extra + '\r\n')
    (path / 'body.partial').write_bytes(body)


def first_partial(intake, queue, source, length=9):
    path, intent = trial(intake, queue, source)
    response(path, source, 0, BODY[:length])
    result = intake.close_attempt(queue, source, path, intent, time.monotonic() + 10, curl_code=28)
    assert result['status'] == 'transport_deferred'
    assert queue.partial(source).read_bytes() == BODY[:length]
    assert not (path / 'body.partial').exists()
    return intake.prefix_state(queue, source, time.monotonic() + 10)


def test_actual_portable_inventory_is_complete_train_metadata_only(intake):
    manifest = intake.load_manifest()
    assert len(manifest['sources']) == 104
    assert sum(s['bytes'] for s in manifest['sources']) == 672336857
    assert len({s['patient_group'] for s in manifest['sources']}) == 14
    assert len(manifest['proofs']) == 9
    assert not any(p['path'].startswith('build/') for p in manifest['proofs'])
    assert not any(s['patient_group'] == 'RESECT:Case4' for s in manifest['sources'])
    assert intake.runtime_binding(manifest)
    assert intake.status(intake.Queue(ROOT, manifest))['metadata_only']


@pytest.mark.parametrize('change', ['role', 'url', 'bytes', 'md5', 'destination', 'kind'])
def test_metadata_source_mutations_are_rejected(intake, change):
    manifest = intake.load_manifest()
    proofs = {p['original_path']: (ROOT / p['path']).read_bytes() for p in manifest['proofs']}
    manifest['sources'][0][change if change not in ('url', 'md5') else {'url': 'source_url', 'md5': 'expected_md5'}[change]] = {
        'role': 'SELECT', 'url': intake.PREFIX + 'unlisted', 'bytes': 1, 'md5': '0'*32,
        'destination': 'data/anatomy/resect-v1/Case4/unsafe', 'kind': 'original_mri'}[change]
    with pytest.raises(intake.Refusal):
        intake.validate_manifest(manifest, proofs)


def test_two_attempt_range_resume_and_repeat_verification(intake, case):
    queue, source = case
    prefix = first_partial(intake, queue, source)
    path, intent = trial(intake, queue, source, prefix)
    response(path, source, prefix['bytes'], BODY[prefix['bytes']:])
    result = intake.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=0)
    assert result['status'] == 'byte_verified'
    assert queue.target(source).read_bytes() == BODY
    assert not queue.partial(source).exists()
    assert queue.completion(source)['sha256'] == hashlib.sha256(BODY).hexdigest()
    assert intake.recover(queue, source, time.monotonic()+10)['sha256'] == hashlib.sha256(BODY).hexdigest()
    assert len(queue.attempts(source)) == 2


@pytest.mark.parametrize('wrong', ['200', 'start', 'end', 'total', 'duplicate_range', 'duplicate_length', 'encoding', 'overflow'])
def test_invalid_resume_never_changes_retained_prefix(intake, case, wrong):
    queue, source = case
    prefix = first_partial(intake, queue, source)
    path, intent = trial(intake, queue, source, prefix)
    offset = prefix['bytes']
    ranges = {'start': f'bytes {offset+1}-{len(BODY)-1}/{len(BODY)}',
              'end': f'bytes {offset}-{len(BODY)-2}/{len(BODY)}',
              'total': f'bytes {offset}-{len(BODY)-1}/{len(BODY)+1}'}
    extra = {'duplicate_range': f'Content-Range: bytes {offset}-{len(BODY)-1}/{len(BODY)}\r\n',
             'duplicate_length': f'Content-Length: {len(BODY)-offset}\r\n',
             'encoding': 'Content-Encoding: gzip\r\n'}.get(wrong, '')
    response(path, source, offset, BODY if wrong == '200' else BODY[offset:] + (b'x' if wrong == 'overflow' else b''),
             status=200 if wrong == '200' else 206, content_range=ranges.get(wrong), extra=extra)
    result = intake.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=0)
    assert result['status'] == 'terminal_failure'
    assert queue.partial(source).read_bytes() == BODY[:offset]
    assert (path / 'body.partial').exists() and not queue.target(source).exists()


def test_complete_bad_md5_is_retained_without_tainting_prefix(intake, case):
    queue, source = case
    prefix = first_partial(intake, queue, source)
    path, intent = trial(intake, queue, source, prefix)
    response(path, source, prefix['bytes'], b'!' * (len(BODY)-prefix['bytes']))
    result = intake.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=0)
    assert result['error_code'] == 'source_fixity'
    assert queue.partial(source).read_bytes() == BODY[:prefix['bytes']]
    assert not queue.target(source).exists()


def test_changed_prefix_refused_before_another_request(intake, case):
    queue, source = case
    first_partial(intake, queue, source)
    queue.partial(source).write_bytes(b'x' * 9)
    with pytest.raises(intake.Refusal, match='prefix_changed'):
        intake.run_one(queue, source, {}, launcher=lambda *a, **k: pytest.fail('must not launch'))


@pytest.mark.parametrize('boundary', ['before_append', 'mid_append', 'after_append', 'after_compaction'])
def test_interrupted_merge_recovers_without_duplicate_bytes(intake, case, monkeypatch, boundary):
    queue, source = case
    prefix = first_partial(intake, queue, source)
    path, intent = trial(intake, queue, source, prefix)
    response(path, source, prefix['bytes'], BODY[prefix['bytes']:])
    original_save = queue.save
    original_reserve = queue.reserve
    with monkeypatch.context() as context:
        if boundary in ('before_append', 'mid_append'):
            def interrupted_reserve(n):
                if (path / 'merge.json').exists():
                    if boundary == 'mid_append':
                        with queue.partial(source).open('ab') as handle:
                            handle.write(BODY[prefix['bytes']:prefix['bytes']+4])
                    raise KeyboardInterrupt()
                return original_reserve(n)
            context.setattr(queue, 'reserve', interrupted_reserve)
        else:
            def interrupted_save(where, value):
                if where.name == 'merged.json' and boundary == 'after_append':
                    raise KeyboardInterrupt()
                return original_save(where, value)
            context.setattr(queue, 'save', interrupted_save)
        if boundary == 'after_compaction':
            intake.merge_suffix(queue, source, path, intent, time.monotonic()+10)
            assert not (path / 'body.partial').exists()
        else:
            with pytest.raises(KeyboardInterrupt):
                intake.merge_suffix(queue, source, path, intent, time.monotonic()+10)
    result = intake.recover(queue, source, time.monotonic()+10)
    assert result['sha256'] == hashlib.sha256(BODY).hexdigest()
    assert queue.target(source).read_bytes() == BODY
    assert len(queue.attempts(source)) == 2


def test_partial_after_compaction_recovers_as_resumable(intake, case):
    queue, source = case
    path, intent = trial(intake, queue, source)
    response(path, source, 0, BODY[:7])
    intake.merge_suffix(queue, source, path, intent, time.monotonic()+10)
    assert intake.recover(queue, source, time.monotonic()+10) is None
    assert queue.attempts(source)[0][2]['status'] == 'transport_deferred'
    assert intake.prefix_state(queue, source, time.monotonic()+10)['bytes'] == 7


def test_corrupt_torn_append_is_never_recovered(intake, case, monkeypatch):
    queue, source = case
    prefix = first_partial(intake, queue, source)
    path, intent = trial(intake, queue, source, prefix)
    response(path, source, prefix['bytes'], BODY[prefix['bytes']:])
    with monkeypatch.context() as context:
        context.setattr(queue, 'reserve', lambda n: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            intake.merge_suffix(queue, source, path, intent, time.monotonic()+10)
    with queue.partial(source).open('ab') as handle:
        handle.write(b'bad')
    with pytest.raises(intake.Refusal, match='torn_merge_content'):
        intake.recover(queue, source, time.monotonic()+10)
    assert not queue.target(source).exists()


def test_quota_includes_retained_staging_and_stops_before_launch(intake, case):
    queue, source = case
    queue.cache.mkdir(parents=True)
    (queue.cache / 'failed.partial').write_bytes(b'x' * 300)
    queue.bounds['compressed_cache_bytes'] = 350
    with pytest.raises(intake.Refusal, match='compressed_cache_quota'):
        intake.run_one(queue, source, {}, launcher=lambda *a, **k: pytest.fail('must not launch'))
    assert not (queue.object_dir(source) / 'attempts').exists()


def test_hardlinks_count_once_and_symlinks_refused(intake, case, tmp_path):
    queue, source = case
    queue.cache.mkdir(parents=True)
    first = queue.cache / 'one.partial'
    first.write_bytes(b'abc')
    os.link(first, queue.cache / 'two.partial')
    assert queue.used_bytes() == 3
    (queue.cache / 'bad').symlink_to(tmp_path / 'absent')
    with pytest.raises(intake.Refusal, match='symlink_path'):
        queue.used_bytes()


@pytest.mark.parametrize('http_status', [429, 503, 302])
def test_provider_retry_after_and_attempt_count_survive_restart(intake, case, http_status):
    queue, source = case
    for number in range(1, 4):
        path, intent = trial(intake, queue, source)
        response(path, source, 0, b'', status=http_status, extra='Retry-After: 1800\r\n')
        result = intake.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=22)
        assert result['provider_not_before_epoch'] >= time.time() + 1799
    restarted = intake.Queue(queue.root, queue.manifest)
    assert len(restarted.attempts(source)) == 3
    assert intake.states(restarted)[1] >= time.time() + 1799
    with pytest.raises(intake.Refusal, match='attempt_limit'):
        intake.run_one(restarted, source, {}, launcher=lambda *a, **k: pytest.fail('no fourth attempt'))


@pytest.mark.parametrize('code', [35, 51, 58, 60, 77, 83, 90, 91])
def test_tls_failures_are_terminal_without_publication(intake, case, code):
    queue, source = case
    path, intent = trial(intake, queue, source)
    response(path, source, 0, BODY)
    result = intake.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=code)
    assert result['status'] == 'terminal_failure' and result['error_code'] == 'tls_failure'
    assert not queue.target(source).exists() and not queue.partial(source).exists()


def test_native_resume_command_retains_tls_and_stages_before_merge(intake, case, monkeypatch):
    queue, source = case
    prefix = first_partial(intake, queue, source)
    path, intent = trial(intake, queue, source, prefix)
    monkeypatch.setattr(intake.native, 'system_curl_binding', lambda **k: {'available': True, 'trust': 'existing_macos_system_store'})
    captured = {}

    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            captured['command'] = command
            captured['environment'] = kwargs['env']
            kwargs['stdout'].write(BODY[prefix['bytes']:])
            headerfd = int(command[command.index('--dump-header')+1].split('/')[-1])
            os.write(headerfd, f'HTTP/1.1 206 Partial\r\nContent-Length: {len(BODY)-prefix["bytes"]}\r\nContent-Range: bytes {prefix["bytes"]}-{len(BODY)-1}/{len(BODY)}\r\n\r\n'.encode())
        def poll(self): return self.returncode
        def wait(self): return self.returncode
        def kill(self): self.returncode = -9

    monkeypatch.setattr(intake.subprocess, 'Popen', Process)
    assert intake.native_fetch(queue, source, path, intent, time.monotonic()+10) == 0
    argv = captured['command']
    assert argv[0:2] == ['/usr/bin/curl', '-q']
    assert argv[argv.index('--range')+1] == f'{prefix["bytes"]}-{len(BODY)-1}'
    assert '--no-location' in argv and '--tlsv1.2' in argv and '-k' not in argv
    assert argv[-1] == source['source_url']
    assert argv[argv.index('--max-filesize')+1] == str(len(BODY)-prefix['bytes'])
    assert queue.partial(source).read_bytes() == BODY[:prefix['bytes']]
    assert (path / 'body.partial').read_bytes() == BODY[prefix['bytes']:]


def test_native_timeout_reaps_child_and_retains_attempt(intake, case, monkeypatch):
    queue, source = case
    path, intent = trial(intake, queue, source)
    monkeypatch.setattr(intake.native, 'system_curl_binding', lambda **k: {'available': True})
    state = {'killed': False, 'waited': False}
    class Process:
        def __init__(self, *a, **k): pass
        def poll(self): return -9 if state['killed'] else None
        def kill(self): state['killed'] = True
        def wait(self): state['waited'] = True; return -9
    monkeypatch.setattr(intake.subprocess, 'Popen', Process)
    with pytest.raises(intake.io.IntakeDeadline):
        intake.native_fetch(queue, source, path, intent, time.monotonic()+.01)
    assert state['killed'] and state['waited']
    assert (path / 'native-request.json').exists() and (path / 'body.partial').exists()


def test_expired_merge_budget_never_publishes(intake, case):
    queue, source = case
    path, intent = trial(intake, queue, source)
    response(path, source, 0, BODY)
    result = intake.close_attempt(queue, source, path, intent, time.monotonic()-1, curl_code=0)
    assert result['status'] == 'interrupted'
    assert not queue.target(source).exists() and (path / 'body.partial').exists()
    original_failure = (path / 'result.json').read_bytes()
    assert intake.recover(queue, source, time.monotonic()+10)['sha256'] == hashlib.sha256(BODY).hexdigest()
    assert (path / 'result.json').read_bytes() == original_failure
    assert (path / 'recovery.json').exists()


def local_run_layout(intake, queue, monkeypatch):
    # The real scheduler runs over isolated protocol fixtures. Production CLI
    # still requires the exact immutable 104-file manifest and helper bindings.
    for name in (*intake.CODE, intake.MANIFEST):
        path = queue.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    monkeypatch.setattr(intake, 'runtime_binding', lambda *a, **k: {})


def test_continuous_queue_crosses_first_eight_without_pause(intake, case, monkeypatch):
    queue, source = case
    sources = []
    for index in range(9):
        item = copy.deepcopy(source)
        item.update(id=f'control-{index}', destination=f'data/anatomy/resect-v1/control-{index}.bin')
        sources.append(item)
    queue.manifest['sources'] = sources
    queue.sources = {s['id']: s for s in sources}
    local_run_layout(intake, queue, monkeypatch)
    called = []

    def local_transfer(q, s, runtime):
        called.append(s['id'])
        path, intent = trial(intake, q, s)
        response(path, s, 0, BODY)
        return intake.close_attempt(q, s, path, intent, time.monotonic()+10, curl_code=0)

    monkeypatch.setattr(intake, 'run_one', local_transfer)
    monkeypatch.setattr(intake.time, 'sleep', lambda *a: pytest.fail('no review or cooldown pause expected'))
    assert intake.run(queue) == 0
    assert called == [s['id'] for s in sources]
    assert intake.status(queue)['counts'] == {'byte_verified': 9}
    called.clear()
    assert intake.run(queue) == 0
    assert called == []


def test_recovered_rate_limit_blocks_other_object_before_network(intake, case, monkeypatch):
    queue, source = case
    other = copy.deepcopy(source)
    other.update(id='second-control', destination='data/anatomy/resect-v1/second.bin')
    queue.sources[other['id']] = other
    queue.manifest['sources'].append(other)
    local_run_layout(intake, queue, monkeypatch)
    path, intent = trial(intake, queue, source)
    # Simulates death after receiving 429, before the result receipt.
    response(path, source, 0, b'', status=429, extra='Retry-After: 1800\r\n')
    monkeypatch.setattr(intake, 'run_one', lambda *a, **k: pytest.fail('provider cooldown must block both objects'))
    def leave_wait(seconds):
        assert 0 < seconds <= 30
        raise KeyboardInterrupt()
    monkeypatch.setattr(intake.time, 'sleep', leave_wait)
    with pytest.raises(KeyboardInterrupt):
        intake.run(queue)
    assert queue.attempts(source)[0][2]['status'] == 'rate_deferred'
    assert queue.attempts(other) == []


def test_completed_file_change_fails_independent_reuse_check(intake, case, monkeypatch):
    queue, source = case
    path, intent = trial(intake, queue, source)
    response(path, source, 0, BODY)
    intake.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=0)
    receipt = (queue.object_dir(source)/'completion.json').read_bytes()
    queue.target(source).write_bytes(b'!' * len(BODY))
    with pytest.raises(intake.Refusal, match='source_fixity'):
        intake.recover(queue, source, time.monotonic()+10)
    assert (queue.object_dir(source)/'completion.json').read_bytes() == receipt
    local_run_layout(intake, queue, monkeypatch)
    assert intake.run(queue) == 2
    reports = list((queue.cache/'runs').glob('*/completion.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['all_complete'] is False
