"""Metadata/process/cache controls using retained real bytes; no image decoding."""
import copy
from email.message import Message
from email.utils import formatdate
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_RESECT = ROOT / 'data/annotations/resect-seg-v1'
REAL_LAUSANNE = ROOT / 'data/anatomy/ds003949-v1.0.1'


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('continuous_source_acquisition')
    original = subprocess.Popen
    def local_only(command, *args, **kwargs):
        if any(str(arg).startswith(('http://', 'https://')) for arg in command):
            pytest.fail('real network command forbidden')
        return original(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', local_only)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', lambda *a, **k: pytest.fail('real network forbidden'))
    return module


@pytest.fixture
def isolated(queue, monkeypatch):
    with tempfile.TemporaryDirectory(prefix='continuous-source-controls-', dir=ROOT / 'build') as folder:
        folder = Path(folder)
        monkeypatch.setattr(queue, 'CACHE', folder / 'queue')
        monkeypatch.setattr(queue.resect, 'DATA', folder / 'resect')
        monkeypatch.setattr(queue.lausanne, 'DATA', folder / 'lausanne')
        monkeypatch.setattr(queue.lausanne, 'CACHE', folder / 'lausanne/train-intake-v1')
        yield folder


@pytest.fixture
def sources(queue):
    return queue.catalog()


def sample(sources, key='resect-Case3-during-mask'):
    return next(s for s in sources if s['key'] == key)


def install_actual(queue, source):
    original = (REAL_RESECT if source['dataset'] == 'resect' else REAL_LAUSANNE) / source['entry']['path']
    if not original.exists():
        pytest.skip('retained authentic bytes unavailable')
    target = queue.target_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    return target


def local_run(queue, source, folder, *, offline=True):
    run = folder / 'run'
    run.mkdir(exist_ok=True)
    declaration = {'offline': offline}
    return queue.run_one(source, run, declaration, [], {})


class Response(io.BytesIO):
    def __init__(self, payload, url, headers, status=200):
        super().__init__(payload)
        self.url, self.headers, self.status = url, headers, status
    def geturl(self):
        return self.url


def test_full_exact_catalog_roles_versions_rights_without_qc(queue, sources):
    assert len(sources) == 890
    assert sum(s['entry']['bytes'] for s in sources) == 10578593091
    assert {s['role'] for s in sources} == {'TRAIN'}
    resect = [s for s in sources if s['dataset'] == 'resect']
    assert len(resect) == 50
    assert len({s['person'] for s in resect}) == 13
    assert not any(s['person'] in {'RESECT:Case4', 'RESECT:Case11'} for s in resect)
    assert not any(s.get('pair_id') == 'Case15-during' for s in resect)
    assert sum(bool(s['entry'].get('short_alias')) for s in resect) == 10
    assert {s['release']['license'] for s in resect} == {'CC-BY-4.0', 'CC-BY-NC-SA-4.0'}
    assert all(s['entry'].get('file_revision') == 2 for s in resect if s['provider'] == 'osf')
    assert not queue.CLAIMS['training_admitted'] and queue.CLAIMS['header_qc'] == 'not_run'


def test_frozen_manifest_changed_refuses_before_network(queue, monkeypatch):
    original = queue.read_small
    monkeypatch.setattr(queue, 'read_small', lambda p, **k: original(p, **k)+b' ' if p == queue.lausanne.INDEX else original(p, **k))
    with pytest.raises(queue.Refusal, match='lausanne_index_changed'):
        queue.catalog()


@pytest.mark.parametrize('value,expected', [('120', 1120), ('0', 1000), (' 900 ', 1900),
    (formatdate(1400, usegmt=True), 1400), (formatdate(500, usegmt=True), 1000),
    ('garbage', None), ('-5', None), (None, None), ('9'*129, None)])
def test_retry_after_numeric_date_and_invalid(queue, value, expected):
    assert queue.retry_after(value, 1000) == expected


def test_observer_retains_date_without_signed_query(queue, isolated):
    deadline = time.monotonic()+10
    url = 'https://osf.io/download/control/?secret=DO_NOT_RETAIN'
    headers = Message()
    headers['Retry-After'] = formatdate(time.time()+7200, usegmt=True)
    def denied(request, **kwargs):
        raise HTTPError(request.full_url, 429, 'DO_NOT_RETAIN', headers, io.BytesIO())
    opener = queue.ObservedOpener(denied, isolated, deadline)
    with pytest.raises(HTTPError):
        opener.open(Request(url), timeout=30)
    raw = (isolated/'http-01.json').read_text()
    assert 'DO_NOT_RETAIN' not in raw and 'https://' not in raw
    assert opener.failure['retry_after_utc_epoch'] > time.time()+7100


def test_provider_cooldown_continues_other_sources_and_resumes(queue, sources, isolated):
    mask = sample(sources)
    image = sample(sources, 'resect-Case3-during-image')
    records = {s['key']: [] for s in (mask, image)}
    outcome = {'status': 'rate_deferred', 'provider': 'osf', 'network_attempted': True,
               'http_failure': {'retry_after_utc_epoch': 20000}}
    queue.apply_backoff(outcome, [], {}, 1000)
    providers = queue.provider_state({'mask': [outcome]})
    chosen, wake = queue.choose_source([mask,image], records, providers, 1000, set(), False)
    assert chosen == image and wake is None
    chosen, wake = queue.choose_source([mask], {mask['key']:[outcome]}, providers, 1000, set(), False)
    assert chosen is None and wake == 20000
    assert queue.choose_source([mask], {mask['key']:[outcome]}, providers, 20000, set(), False)[0] == mask


def test_retry_bound_persists_and_does_not_reset_server_delay(queue):
    old = [{'status': 'rate_deferred', 'network_attempted': True}]*2
    outcome = {'status':'rate_deferred','provider':'osf','http_failure':{'retry_after_utc_epoch':999999}}
    queue.apply_backoff(outcome, old, {'osf':{'not_before':5000,'failures':2}}, 1000)
    assert outcome['status'] == 'transport_attempts_exhausted'
    assert outcome['provider_not_before_utc_epoch'] == 999999


def test_tls_is_terminal_never_relaxed(queue):
    assert queue.failure_kind(URLError(ssl.SSLCertVerificationError('secret URL')), None) == 'tls_failed'
    assert queue.failure_kind(queue.Refusal('source_fixity_mismatch'), None) == 'integrity_or_transport_failed'


@pytest.mark.parametrize('key', ['resect-Case3-during-mask', 'resect-Case3-during-image'])
def test_actual_cached_pilot_download_only_receipt_and_resume(queue, sources, isolated, key):
    source = sample(sources, key)
    install_actual(queue, source)
    outcome = local_run(queue, source, isolated)
    assert outcome['status'] == 'existing_byte_verified'
    assert outcome['verified_bytes'] == source['entry']['bytes']
    assert outcome['network_attempted'] is False
    assert outcome['decoded_array_bytes'] == 0 and outcome['header_qc'] == 'not_run'
    assert queue.history(source) == [outcome]
    assert not (isolated/'resect/train-v1').exists()


def test_actual_sidecar_materialized_without_network(queue, sources, isolated):
    source = next(s for s in sources if s['provider'] == 'embedded')
    outcome = local_run(queue, source, isolated)
    assert outcome['status'] == 'byte_verified' and not outcome['network_attempted']
    assert queue.digest(queue.target_path(source).read_bytes()) == source['entry']['sha256']


def test_corrupt_actual_cached_bytes_refused_without_overwrite(queue, sources, isolated):
    source = sample(sources)
    target = install_actual(queue, source)
    raw = target.read_bytes()
    corrupted = bytes([raw[0] ^ 1]) + raw[1:]
    target.write_bytes(corrupted)
    outcome = local_run(queue, source, isolated)
    assert outcome['status'] == 'integrity_or_transport_failed'
    assert target.read_bytes() == corrupted
    assert not outcome['network_attempted']


def test_symlink_source_refused(queue, sources, isolated):
    source = sample(sources)
    target = queue.target_path(source)
    target.parent.mkdir(parents=True)
    target.symlink_to(REAL_RESECT/source['entry']['path'])
    outcome = local_run(queue, source, isolated)
    assert outcome['status'] == 'control_failed'
    assert outcome['error_code'] == 'symlink_path_refused'
    assert target.is_symlink()


def test_offline_missing_stays_missing_and_does_not_call_transport(queue, sources, isolated):
    source = sample(sources)
    outcome = local_run(queue, source, isolated)
    assert outcome['status'] == 'missing_offline' and not outcome['network_attempted']
    assert not queue.target_path(source).exists()


def test_real_annotation_via_optional_opener_is_exact_and_download_only(queue, sources, isolated):
    source = sample(sources)
    raw_path = REAL_RESECT/source['entry']['path']
    if not raw_path.exists():
        pytest.skip('retained authentic pilot unavailable')
    raw = raw_path.read_bytes()
    headers = Message()
    headers['Content-Length'] = str(len(raw))
    class ActualBytes:
        def open(self, request, **kwargs):
            assert request.full_url == source['entry']['source_url']
            return Response(raw, request.full_url, headers)
    sha = queue.resect.transfer_annotation(source['entry']['id'], isolated,
        deadline=time.monotonic()+10, events=[], opener=ActualBytes())
    assert sha == source['entry']['sha256']
    assert queue.target_path(source).read_bytes() == raw


def test_range_refusal_preserves_existing_partial(queue, sources, isolated):
    source = next(s for s in sources if s['provider']=='openneuro_s3')
    target = queue.target_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    original = REAL_LAUSANNE/source['entry']['path']
    if not original.exists():
        pytest.skip('retained authentic Lausanne unavailable')
    with original.open('rb') as handle:
        raw = handle.read(1024)
    partial = target.with_name(target.name+'.partial')
    partial.write_bytes(raw)
    def ignored(request, **kwargs):
        return Response(b'', request.full_url, Message(), status=200)
    observer = queue.ObservedOpener(ignored, isolated, time.monotonic()+10)
    with pytest.raises(queue.Refusal, match='range_resume_not_honored'):
        queue.annex.acquire_file(source['entry'], queue.lausanne.DATA, opener=observer)
    assert partial.read_bytes() == raw and not target.exists()


def test_real_lausanne_range_resume_exact_version_and_md5(queue, sources, isolated):
    source = next(s for s in sources if s['provider']=='openneuro_s3')
    original = REAL_LAUSANNE/source['entry']['path']
    if not original.exists():
        pytest.skip('retained authentic Lausanne unavailable')
    raw = original.read_bytes()
    target = queue.target_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_name(target.name+'.partial').write_bytes(raw[:1024])
    from urllib.parse import parse_qs, urlsplit
    headers = Message()
    headers['Content-Length'] = str(len(raw)-1024)
    headers['Content-Range'] = f'bytes 1024-{len(raw)-1}/{len(raw)}'
    headers['x-amz-version-id'] = parse_qs(urlsplit(source['entry']['source_url']).query)['versionId'][0]
    def remaining(request, **kwargs):
        assert request.get_header('Range') == 'bytes=1024-'
        assert request.full_url == source['entry']['source_url']
        return Response(raw[1024:], request.full_url, headers, 206)
    result = queue.annex.acquire_file(source['entry'], queue.lausanne.DATA,
        opener=queue.ObservedOpener(remaining, isolated, time.monotonic()+10))
    assert result == 'downloaded_and_verified'
    assert queue.verified_bytes(source, deadline=time.monotonic()+10) == queue.digest(raw)


@pytest.mark.parametrize('mutation', ['empty', 'wrong_key', 'wrong_source', 'qc_claim', 'wrong_size', 'wrong_hash', 'bad_status'])
def test_valid_real_result_mutations_refused(queue, sources, isolated, mutation):
    source = sample(sources)
    install_actual(queue, source)
    outcome = local_run(queue, source, isolated)
    trial = ROOT/outcome['attempt']
    intent, result = queue.read_json(trial/'intent.json'), queue.read_json(trial/'result.json')
    if mutation == 'empty':
        result = {}
    elif mutation == 'wrong_key':
        result['source_key'] = 'unlisted'
    elif mutation == 'wrong_source':
        result['source']['entry']['source_url'] += 'unlisted'
    elif mutation == 'qc_claim':
        result['header_qc'] = 'passed'
    elif mutation == 'wrong_size':
        result['verified_bytes'] = 0
    elif mutation == 'wrong_hash':
        result['sha256'] = '0'*64
    else:
        result['status'] = 'acquired_qc_passed'
    with pytest.raises(queue.Refusal):
        queue.validate_result(source, intent, trial, result, time.monotonic()+10)


def test_late_valid_worker_is_not_accepted(queue, sources, isolated, monkeypatch):
    source = sample(sources)
    original_clock = time.monotonic
    offset = [0]
    monkeypatch.setattr(queue.time, 'monotonic', lambda: original_clock()+offset[0])
    run = isolated/'run'
    run.mkdir()
    def completed(command, log, *, deadline, on_start):
        on_start(99999999)
        trial = ROOT/command[-2]
        intent = queue.read_json(trial/'intent.json')
        install_actual(queue, source)
        queue.acquire_one(source, trial, intent, deadline=intent['deadline'])
        offset[0] = 301
        return 'completed', 0
    outcome = queue.run_one(source, run, {'offline':False}, [], {}, launcher=completed)
    assert outcome['status'] == 'transport_deferred' and outcome['deadline_exceeded']
    assert (ROOT/outcome['attempt']/'result.json').exists()


@pytest.mark.parametrize('failure', ['launch', 'interrupt'])
def test_launch_or_interrupt_retains_active_attempt_and_closed_outcome(queue, sources, isolated, failure):
    source = sample(sources)
    run = isolated/'run'
    run.mkdir()
    def failed(command, log, *, deadline, on_start):
        if failure == 'interrupt':
            on_start(99999999)
            raise KeyboardInterrupt()
        raise OSError('private transport diagnostic must not leak')
    if failure == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            queue.run_one(source, run, {'offline':False}, [], {}, launcher=failed)
    else:
        queue.run_one(source, run, {'offline':False}, [], {}, launcher=failed)
    records = queue.history(source)
    assert len(records) == 1
    assert records[0]['status'] in {'control_failed', 'interrupted'}
    trial = ROOT/records[0]['attempt']
    assert (trial/'intent.json').exists() and (trial/'outcome.json').exists()
    assert 'private transport' not in (trial/'outcome.json').read_text()


def test_changed_journal_result_is_detected(queue, sources, isolated):
    source = sample(sources)
    install_actual(queue, source)
    outcome = local_run(queue, source, isolated)
    path = ROOT/outcome['attempt']/'result.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(queue.Refusal, match='history_result_changed'):
        queue.history(source)


def test_unclosed_attempt_keeps_link_and_refuses_silent_retry(queue, sources, isolated):
    source = sample(sources)
    trial = queue.object_directory(source)/'attempts'/'control-interrupted'
    queue.save(trial/'intent.json', {'source_key':source['key'],'source_binding':queue.source_binding(source)})
    with pytest.raises(queue.Refusal, match='unclosed_attempt_preserved'):
        queue.history(source)
    assert (trial/'intent.json').exists()


def test_runtime_closure_does_not_include_or_import_scientific_qc(queue):
    binding = queue.runtime_binding()
    assert all('/data_policy.py' not in p and '/imaging.py' not in p for p in binding['files'])
    assert 'nibabel' not in queue.__dict__ and 'numpy' not in queue.__dict__


def test_frozen_worker_positive_over_actual_cache(queue, sources, isolated, monkeypatch):
    source = sample(sources)
    monkeypatch.setattr(queue, 'catalog', lambda: sources)
    run, declaration = queue.freeze_run(sources, False)
    def launch(command, log, *, deadline, on_start):
        on_start(99999999)
        install_actual(queue, source)
        with monkeypatch.context() as context:
            context.setattr(queue.os, 'getppid', lambda: os.getpid())
            result = queue.worker(command[-2], command[-1])
        assert result['status'] == 'existing_byte_verified'
        return 'completed', 0
    result = queue.run_one(source, run, declaration, [], {}, launcher=launch)
    assert result['status'] == 'existing_byte_verified'
    assert (ROOT/result['attempt']/'worker-claim.json').exists()
    assert queue.history(source) == [result]
    assert queue.read_json(run/'declaration.json')['range_resume']['resect'] is False
    for name, sha in declaration['execution']['files'].items():
        assert queue.digest((run/'snapshot'/name).read_bytes()) == sha


def test_final_parent_accounting_after_valid_result_obeys_deadline(queue, sources, isolated, monkeypatch):
    source = sample(sources)
    install_actual(queue, source)
    original_clock, original_validate = time.monotonic, queue.validate_result
    offset = [0]
    monkeypatch.setattr(queue.time, 'monotonic', lambda: original_clock()+offset[0])
    def valid_then_late(*args):
        sha = original_validate(*args)
        offset[0] = 301
        return sha
    monkeypatch.setattr(queue, 'validate_result', valid_then_late)
    result = local_run(queue, source, isolated)
    assert result['status'] == 'transport_deferred' and result['deadline_exceeded']
    trial = ROOT/result['attempt']
    assert queue.read_json(trial/'result.json')['status'] == 'existing_byte_verified'
    assert result['result_sha256'] == queue.digest((trial/'result.json').read_bytes())


def test_annotation_rate_limit_date_flows_to_persistent_outcome(queue, sources, isolated, monkeypatch):
    source = sample(sources)
    headers = Message()
    headers['Retry-After'] = formatdate(time.time()+7200, usegmt=True)
    def deny(self, request, **kwargs):
        raise HTTPError(request.full_url, 429, 'signed-secret', headers, io.BytesIO())
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', deny)
    run = isolated/'run'
    run.mkdir()
    def one_worker(command, log, *, deadline, on_start):
        on_start(99999999)
        trial = ROOT/command[-2]
        intent = queue.read_json(trial/'intent.json')
        queue.acquire_one(source, trial, intent, deadline=intent['deadline'])
        return 'completed', 0
    outcome = queue.run_one(source, run, {'offline':False}, [], {}, launcher=one_worker)
    assert outcome['status'] == 'rate_deferred'
    assert outcome['provider_not_before_utc_epoch'] > time.time()+7100
    assert queue.provider_state({source['key']:queue.history(source)})['osf']['not_before'] == outcome['provider_not_before_utc_epoch']
    assert not queue.target_path(source).exists()
    assert not any('signed-secret' in p.read_text() for p in (ROOT/outcome['attempt']).glob('*.json'))


def test_process_supervision_timeout_reaps_actual_local_child(queue, isolated):
    process_ids = []
    start = time.monotonic()
    result = queue.supervise([sys.executable, '-c', 'import time; time.sleep(10)'], isolated/'child.log',
        deadline=start+.15, on_start=process_ids.append)
    assert result[0] == 'timeout' and time.monotonic()-start < 3
    with pytest.raises(ProcessLookupError):
        os.kill(process_ids[0], 0)


@pytest.mark.parametrize('complete,expected_exit', [(True, 0), (False, 2)])
def test_continuous_cli_reports_terminal_unresolved_and_continues_other_file(queue, sources, isolated, monkeypatch, complete, expected_exit):
    image, mask = sample(sources, 'resect-Case3-during-image'), sample(sources)
    install_actual(queue, mask)
    if complete:
        install_actual(queue, image)
    # Two authentic declared members suffice to test queue termination; no
    # network or artificial patient/volume fixture enters the control.
    monkeypatch.setattr(queue, 'catalog', lambda: [image, mask])
    monkeypatch.setattr(sys, 'argv', ['continuous_source_acquisition.py', 'run', '--offline'])
    assert queue.main() == expected_exit
    results = list((queue.CACHE/'runs').glob('*/finished.json'))
    assert len(results) == 1
    result = queue.read_json(results[0])
    assert result['status'] == ('all_bytes_verified' if complete else 'completed_with_unresolved_files')
    assert result['unresolved_files'] == (0 if complete else 1)
    assert queue.history(mask)[-1]['status'] == 'existing_byte_verified'
    assert result['header_qc'] == 'not_run' and result['training_admitted'] is False
