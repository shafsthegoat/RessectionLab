"""Actual source metadata/bytes and software refusal controls; no patient generation."""
import copy
from email.message import Message
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / 'data/annotations/resect-seg-v1'


@pytest.fixture
def intake(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('resect_train_intake')
    original_popen = subprocess.Popen

    def local_process_only(command, *args, **kwargs):
        if any(str(arg).startswith(('http://', 'https://')) for arg in command):
            pytest.fail('real network command prohibited in controls')
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', local_process_only)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', lambda *a, **k: pytest.fail('real network prohibited'))
    return module


@pytest.fixture
def isolated(intake, monkeypatch):
    with tempfile.TemporaryDirectory(prefix='resect-train-controls-', dir=ROOT / 'build') as directory:
        monkeypatch.setattr(intake, 'DATA', Path(directory) / 'data')
        intake.DATA.mkdir()
        yield Path(directory)


def install_rights(intake):
    target = intake.DATA / 'rights/README.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'artifacts/resect-component-admission-v1/rights-review-02/README.txt', target)


def pilot_available():
    return all((REAL_DATA / ('originals/' + name)).is_file() for name in
               ['Case3-US-during.nii.gz', 'Case3-US-during-resection.nii.gz'])


def run_worker_locally(intake, monkeypatch):
    # Exercise the real worker over isolated real cached bytes, with launch replaced.
    def launch(command, log, *, deadline, on_start):
        on_start(99999999)
        with monkeypatch.context() as context:
            context.setattr(intake.os, 'getppid', lambda: os.getpid())
            result = intake.worker(command[command.index('--run-id')+1], command[command.index('--intent-sha')+1])
        return ('completed', 0) if result['status'] == 'completed' else ('worker_failed', 1)
    monkeypatch.setattr(intake, 'supervise', launch)


def test_exact_manifest_and_source_metadata_all_25_pairs(intake):
    manifest = intake.require_manifest()
    assert len(manifest['pairs']) == 25 and len(manifest['sources']) == 50
    assert manifest['summary']['remaining_scientific_bytes'] == 548605591
    assert sum(bool(s.get('short_alias')) for s in manifest['sources']) == 10
    assert {p['patient_group'] for p in manifest['pairs']}.isdisjoint(
        {p['patient_group'] for p in manifest['excluded_members']})
    assert {(p['patient_group'], p['phase']) for p in manifest['missing_annotations']} == {
        ('RESECT:Case11', 'during'), ('RESECT:Case11', 'after'), ('RESECT:Case15', 'during')}
    assert not any(e['path'].startswith('build/') for e in manifest['metadata_bindings'])
    assert manifest['rights_and_semantics']['noncommercial_only']
    assert manifest['bounds']['decoded_array_bytes'] == 0
    proposal = json.loads((ROOT / manifest['qualification']['path']).read_text())
    cohort = json.loads((ROOT / manifest['cohort']['path']).read_text())
    for field, value in [('role', 'SELECT'), ('patient_group', 'RESECT:Case4'), ('phase', 'before')]:
        modified = copy.deepcopy(manifest)
        modified['pairs'][0][field] = value
        with pytest.raises(intake.Refusal):
            intake.validate_records(modified, proposal, cohort)


@pytest.mark.parametrize('source_id', ['Case2-during-mask', 'Case12-during-mask', 'Case15-after-mask',
    'Case16-during-mask', 'Case16-after-mask', 'Case17-during-mask', 'Case17-after-mask',
    'Case18-during-mask', 'Case18-after-mask', 'Case21-after-mask', 'Case12-after-mask'])
def test_long_and_all_ten_short_aliases_resolve_only_selected_file(intake, source_id):
    source = intake.source_by_id(source_id)
    intake.checked_route(source, source.source_url, initial=True)
    canonical = f'https://osf.io/download/{source.osf_file_id}/?revision=2'
    intake.checked_route(source, canonical)
    for host in ['files.osf.io', 'files.de-1.osf.io']:
        provider = f'https://{host}/v1/resources/jv8bk/providers/osfstorage/{source.osf_file_id}'
        intake.checked_route(source, provider+'?version=2', body=True)
        intake.checked_route(source, provider)
        with pytest.raises(intake.Refusal):
            intake.checked_route(source, provider, body=True)
    terminal = f'https://storage.googleapis.com/cos-osf-prod-files-de-1/{source.sha256}?X-Goog-Signature=control-private'
    intake.checked_route(source, terminal, body=True)
    assert 'control-private' not in json.dumps(intake.pilot.safe_route(terminal))
    wrong = intake.source_by_id('Case3-after-mask').source_url
    with pytest.raises(intake.Refusal):
        intake.checked_route(source, wrong)
    with pytest.raises(intake.Refusal):
        intake.checked_route(source, terminal, initial=True)


@pytest.mark.parametrize('change', ['version1', 'latest', 'duplicate', 'other_id', 'other_bucket', 'other_hash',
                                    'userinfo', 'http', 'port', 'fragment', 'space', 'unicode', 'invalidport'])
def test_annotation_route_refusals_do_not_expose_query(intake, change):
    source = intake.source_by_id('Case12-during-mask')
    provider = f'https://files.de-1.osf.io/v1/resources/jv8bk/providers/osfstorage/{source.osf_file_id}'
    terminal = f'https://storage.googleapis.com/cos-osf-prod-files-de-1/{source.sha256}'
    candidates = {
        'version1': provider+'?version=1', 'latest': source.source_url.replace('?revision=2',''),
        'duplicate': provider+'?version=2&version=2', 'other_id': provider.replace(source.osf_file_id,'0'*24),
        'other_bucket': terminal.replace('cos-osf-prod-files-de-1','elsewhere'),
        'other_hash': terminal.replace(source.sha256,'0'*64),
        'userinfo': terminal.replace('https://','https://user@'), 'http': terminal.replace('https:','http:'),
        'port': terminal.replace('.com/','.com:444/'), 'fragment': terminal+'#secret',
        'space': terminal+'?secret=private value', 'unicode': terminal+'?secret=é',
        'invalidport': terminal.replace('.com/','.com:wrong/')}
    with pytest.raises(intake.Refusal) as error:
        intake.checked_route(source, candidates[change])
    assert 'secret' not in str(error.value) and 'https://' not in str(error.value)


@pytest.mark.parametrize('url', ['https://s3.nird.sigma2.no/unlisted.nii.gz',
    'https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/RESECT/NIFTI/Case4/US/Case4-US-during.nii.gz'])
def test_native_host_does_not_authorize_other_scientific_files(intake, url):
    source = intake.source_by_id('Case3-during-image')
    with pytest.raises(intake.Refusal):
        intake.checked_route(source, url)


def test_changed_retained_rights_refuses_manifest_before_transport(intake, monkeypatch):
    original = intake.read_small
    rights = ROOT / 'artifacts/resect-component-admission-v1/rights-review-02/README.txt'
    def changed(path, **kwargs):
        value = original(path, **kwargs)
        return value[:-1] if path == rights else value
    monkeypatch.setattr(intake, 'read_small', changed)
    with pytest.raises(intake.Refusal, match='retained_metadata_changed'):
        intake.require_manifest()


def test_manifest_byte_change_and_unknown_pair_refuse(intake, isolated, monkeypatch):
    changed = isolated / 'manifest.json'
    changed.write_bytes(intake.MANIFEST.read_bytes()+b' ')
    with monkeypatch.context() as context:
        context.setattr(intake, 'MANIFEST', changed)
        with pytest.raises(intake.Refusal, match='manifest_changed'):
            intake.require_manifest()
    for pair in ['Case4-during','Case11-during','Case15-during','Case1-after']:
        with pytest.raises(intake.Refusal, match='pair_not_frozen_train'):
            intake.resolve_pair(intake.require_manifest(), pair)


@pytest.mark.skipif(not pilot_available(), reason='authentic local Case3 cache unavailable')
def test_real_cached_pair_verification_and_two_offline_runs(intake, isolated, monkeypatch):
    install_rights(intake)
    for name in ['Case3-US-during.nii.gz','Case3-US-during-resection.nii.gz']:
        target = intake.DATA / 'originals' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(REAL_DATA / 'originals' / name, target)
    run_worker_locally(intake, monkeypatch)
    for run in ['actual-cache-01', 'actual-cache-02']:
        result = intake.run_pair('Case3-during', run, existing_only=True)
        assert result['status'] == result['worker_status'] == 'completed'
        assert [f['status'] for f in result['source_results']] == ['existing_verified','existing_verified']
        assert sum(f['verified_bytes'] for f in result['source_results']) == 9184649
        worker = json.loads((intake.run_path(run) / 'worker-result.json').read_text())
        assert not any(f['events'] for f in worker['files'])
        assert worker['decoded_array_bytes'] == worker['optimizer_updates'] == 0
        assert worker['training_admitted'] is False
    assert json.loads((intake.run_path('actual-cache-02') / 'worker-result.json').read_text())['files'][0]['prior']['run_id'] == 'actual-cache-01'


def test_authentic_rights_file_bounds_fixity_symlink_and_prior_hash(intake, isolated):
    manifest = intake.require_manifest(); entry = manifest['rights_prerequisite']
    install_rights(intake); path = intake.DATA / entry['path']; raw = path.read_bytes()
    assert intake.verify_file(path, entry, deadline=time.monotonic()+2) == entry['sha256']
    with pytest.raises(intake.Refusal, match='source_fixity_mismatch'):
        intake.verify_file(path, entry, deadline=time.monotonic()+2, receipt_sha='0'*64)
    path.write_bytes(raw[:-1])
    with pytest.raises(intake.Refusal, match='source_size_or_type'):
        intake.verify_file(path, entry, deadline=time.monotonic()+2)
    path.write_bytes(bytes([raw[0]^1])+raw[1:])
    with pytest.raises(intake.Refusal, match='source_fixity_mismatch'):
        intake.verify_file(path, entry, deadline=time.monotonic()+2)
    link=isolated/'link';link.symlink_to(path)
    with pytest.raises(intake.Refusal, match='symlink_path_refused'):
        intake.verify_file(link, entry, deadline=time.monotonic()+2)


class Response(io.BytesIO):
    def __init__(self, value, url, size, status=200):
        super().__init__(value); self.status=status; self.url=url; self.headers=Message()
        self.headers['Content-Length']=str(size)
    def geturl(self):return self.url


@pytest.mark.skipif(not pilot_available(), reason='authentic local Case3 cache unavailable')
def test_exact_annotation_redirect_and_authentic_byte_publication(intake, isolated, monkeypatch):
    source=intake.source_by_id('Case3-during-mask')
    raw=(REAL_DATA/source.path).read_bytes()
    terminal=f'https://storage.googleapis.com/cos-osf-prod-files-de-1/{source.sha256}?X-Goog-Signature=control-private'
    provider=f'https://files.de-1.osf.io/v1/resources/jv8bk/providers/osfstorage/{source.osf_file_id}?version=2'
    urls=[source.source_url,provider,terminal];calls=[]
    class Transport:
        def open(self, request, **kwargs):
            i=len(calls);calls.append(request.full_url);assert request.full_url==urls[i]
            if i<2:
                headers=Message();headers['Location']=urls[i+1]
                raise HTTPError(request.full_url,302,'redirect',headers,None)
            return Response(raw,request.full_url,source.bytes)
    monkeypatch.setattr(intake,'build_opener',lambda *a:Transport())
    trial=isolated/'trial';trial.mkdir();events=[]
    assert intake.transfer_annotation(source.id,trial,deadline=time.monotonic()+5,events=events)==source.sha256
    assert (intake.DATA/source.path).read_bytes()==raw and len(calls)==3
    assert 'control-private' not in ''.join(p.read_text() for p in trial.glob('*.json'))


@pytest.mark.parametrize('scenario',['other_file','version1','duplicate_length','wrong_size','overrun','rate_limit','loop'])
def test_annotation_transport_refusals_without_retry_or_publication(intake, isolated, monkeypatch, scenario):
    source=intake.source_by_id('Case3-during-mask');calls=[]
    class Transport:
        def open(self, request, **kwargs):
            calls.append(request.full_url)
            if scenario in {'other_file','version1','rate_limit','loop'}:
                headers=Message()
                headers['Location']=('https://osf.io/download/unlisted/?revision=2' if scenario=='other_file' else
                                     source.source_url.replace('revision=2','revision=1') if scenario=='version1' else source.source_url)
                raise HTTPError(request.full_url,429 if scenario=='rate_limit' else 302,'control-private',headers,None)
            # Empty response tests format refusal; no invented anatomy.
            response=Response(b'',request.full_url,source.bytes)
            if scenario=='duplicate_length':response.headers['Content-Length']=str(source.bytes)
            if scenario=='wrong_size':response.headers.replace_header('Content-Length','1')
            if scenario=='overrun':
                response.read=lambda cap: b'format-control'*(cap+1)
            return response
    monkeypatch.setattr(intake,'build_opener',lambda *a:Transport())
    trial=isolated/'trial';trial.mkdir()
    with pytest.raises(intake.Refusal) as error:
        intake.transfer_annotation(source.id,trial,deadline=time.monotonic()+3,events=[])
    assert 'control-private' not in str(error.value)
    assert len(calls)==(5 if scenario=='loop' else 1)
    assert not (intake.DATA/source.path).exists()


class CurlControl:
    def __init__(self, interruption=None):self.interruption=interruption;self.killed=False;self.calls=0
    def wait(self,timeout=None):
        self.calls+=1
        if self.calls==1 and self.interruption:raise self.interruption
        return 0
    def poll(self):return None if self.interruption and not self.killed else 0
    def kill(self):self.killed=True


@pytest.mark.parametrize('interrupt',[False,True])
def test_native_exact_command_bounds_and_child_cleanup(intake, isolated, monkeypatch, interrupt):
    source=intake.source_by_id('Case2-during-image');captured=[]
    process=CurlControl(subprocess.TimeoutExpired('curl',.1) if interrupt else None)
    monkeypatch.setattr(intake.pilot,'system_curl_binding',lambda **k:{'available':True,'trust':'existing_macos_system_store'})
    def launch(command,**kwargs):
        captured.append(command)
        os.write(kwargs['pass_fds'][0],f'HTTP/2 200\r\nContent-Length: {source.bytes}\r\n\r\n'.encode())
        return process
    monkeypatch.setattr(intake.subprocess,'Popen',launch)
    trial=isolated/'trial';trial.mkdir()
    with pytest.raises((intake.Refusal,subprocess.TimeoutExpired)):
        intake.transfer_original(source.id,trial,deadline=time.monotonic()+2,events=[])
    command=captured[0]
    assert command[:3]==['/usr/bin/curl','-q','--no-location'] and command[-1]==source.source_url
    assert command[command.index('--max-filesize')+1]==str(source.bytes)
    assert command[command.index('--retry')+1]=='0'
    assert not {'-k','--insecure','--cacert','--capath','--location'} & set(command)
    assert process.killed is interrupt and process.calls==2
    assert (trial/'body.partial').exists() and not (intake.DATA/source.path).exists()


def test_missing_rights_and_failed_source_history_require_explicit_retry(intake, isolated, monkeypatch):
    run_worker_locally(intake,monkeypatch)
    first=intake.run_pair('Case2-during','failed-01',existing_only=True)
    assert first['status']=='worker_failed' and len(first['sources'])==2
    assert not (intake.run_path('failed-01')/'files').exists()
    refused=intake.run_pair('Case2-during','refused-implicit',existing_only=True)
    assert refused['status']=='setup_failed' and refused['error_code']=='explicit_retry_of_prior_attempt_required'
    install_rights(intake)
    explicit=intake.run_pair('Case2-during','failed-02',existing_only=True,retry_of='failed-01')
    assert explicit['status']=='worker_failed'
    receipt=json.loads((intake.run_path('failed-02')/'files/Case2-during-image/receipt.json').read_text())
    assert receipt['error_code']=='missing_source_or_changed_cache' and not receipt['events']


@pytest.mark.parametrize('failure',['before_start','after_start','with_receipt','bad_receipt','timeout'])
def test_supervision_failure_preserves_active_attempt_links_and_reaps_child(intake, isolated, monkeypatch, failure):
    import real_intake_io
    pids=[]
    def supervisor(command, log, *, deadline, on_start):
        if failure=='before_start':raise OSError('launch control')
        def started(pid):
            pids.append(pid);on_start(pid)
            if failure in {'with_receipt','bad_receipt'}:
                source_sha=intake.digest(intake.read_small(log.parent/'source.json'))
                intent_sha=intake.digest(intake.read_small(log.parent/'intent.json'))
                value={'run_id':'supervision-control','pair_id':'Case2-during','manifest_sha256':intake.MANIFEST_SHA,
                       'execution_source_sha256':source_sha,'intent_sha256':intent_sha,'status':'failed','files':[],**intake.CLAIMS}
                if failure=='bad_receipt':value['execution_source_sha256']='0'*64
                intake.save(log.parent/'worker-result.json',value)
            if failure!='timeout':raise KeyboardInterrupt('software interruption control')
        return real_intake_io.supervise([sys.executable,'-B','-c','import time; time.sleep(30)'],log,
                                       deadline=time.monotonic()+.15,on_start=started)
    monkeypatch.setattr(intake,'supervise',supervisor)
    result=intake.run_pair('Case2-during','supervision-control',existing_only=True)
    expected={'before_start':'worker_start_failed','after_start':'interrupted_after_start',
              'with_receipt':'interrupted_after_start','bad_receipt':'worker_failed','timeout':'timeout'}[failure]
    assert result['status']==expected and result['attempt'].endswith('supervision-control')
    assert len(result['sources'])==2
    for entry in result['sources']:
        marker=json.loads((ROOT/entry['marker_path']).read_text())
        assert marker['run_id']=='supervision-control'
    for pid in pids:
        with pytest.raises(ProcessLookupError):os.kill(pid,0)
    closure=json.loads((intake.run_path('supervision-control')/'supervision.json').read_text())
    assert closure==result
    if failure in {'with_receipt','bad_receipt'}:assert 'worker_receipt_sha256' in result
    if failure=='bad_receipt':assert result['receipt_error_code']=='worker_receipt_not_admitted'


def test_second_marker_failure_keeps_first_reserved_source_link(intake, isolated, monkeypatch):
    original=intake.save
    def save(path,value):
        if 'source-state/Case2-during-mask' in str(path):raise OSError('marker control')
        original(path,value)
    monkeypatch.setattr(intake,'save',save)
    report=intake.run_pair('Case2-during','marker-control',existing_only=True)
    assert report['status']=='setup_failed' and len(report['sources'])==2
    assert (ROOT/report['sources'][0]['marker_path']).exists()
    assert not (ROOT/report['sources'][1]['marker_path']).exists()


@pytest.mark.parametrize('seconds',[0,-1,301,float('inf'),float('nan'),True])
def test_unbounded_invocation_refuses_before_source_access(intake,seconds):
    with pytest.raises(intake.Refusal,match='invocation_time_bound'):
        intake.run_pair('Case2-during','unused-control',seconds=seconds)


def copy_authentic_pilot(intake):
    install_rights(intake)
    for name in ['Case3-US-during.nii.gz', 'Case3-US-during-resection.nii.gz']:
        target = intake.DATA / 'originals' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(REAL_DATA / 'originals' / name, target)


@pytest.mark.skipif(not pilot_available(), reason='authentic local Case3 cache unavailable')
@pytest.mark.parametrize('mutation', ['empty', 'duplicate', 'failed', 'wrong_pair_file', 'source_binding',
    'execution_binding', 'expected_bytes', 'wrong_path', 'published_hash', 'missing_file_receipt',
    'receipt_disagreement', 'wrong_sha', 'wrong_claim_type'])
def test_completed_worker_requires_exact_pair_and_matching_file_receipts(intake, isolated, monkeypatch, mutation):
    copy_authentic_pilot(intake)
    run_worker_locally(intake, monkeypatch)
    actual_worker = intake.supervise
    mutated_receipt = []

    def corrupt_software_receipt(command, log, *, deadline, on_start):
        status, code = actual_worker(command, log, deadline=deadline, on_start=on_start)
        assert status == 'completed' and code == 0
        path = log.parent / 'worker-result.json'
        value = json.loads(path.read_text())
        file = value['files'][0]
        if mutation == 'empty': value['files'] = []
        elif mutation == 'duplicate': value['files'][1] = copy.deepcopy(file)
        elif mutation == 'failed': file['status'] = 'failed'
        elif mutation == 'wrong_pair_file': file['source_id'] = 'Case3-after-image'
        elif mutation == 'source_binding': file['source_binding'] = '0'*64
        elif mutation == 'execution_binding': file['execution_source_sha256'] = '0'*64
        elif mutation == 'expected_bytes': file['expected_bytes'] -= 1
        elif mutation == 'wrong_path': file['path'] = 'originals/Case4-US-during.nii.gz'
        elif mutation == 'published_hash': file['published_sha256'] = '0'*64
        elif mutation == 'wrong_sha': file['sha256'] = '0'*64
        elif mutation == 'wrong_claim_type': file['training_admitted'] = 0
        file_path = log.parent / 'files/Case3-during-image/receipt.json'
        if mutation == 'missing_file_receipt':
            file_path.unlink()
        elif mutation == 'receipt_disagreement':
            saved = json.loads(file_path.read_text())
            saved['status'] = 'recovered_verified'
            file_path.write_bytes(intake.encode(saved))
        elif mutation not in {'empty', 'duplicate', 'wrong_pair_file'}:
            # Preserve agreement so source/byte binding must reject these mutations.
            file_path.write_bytes(intake.encode(file))
        path.write_bytes(intake.encode(value))
        mutated_receipt.append(path.read_bytes())
        return status, code

    monkeypatch.setattr(intake, 'supervise', corrupt_software_receipt)
    report = intake.run_pair('Case3-during', 'mutated-success', existing_only=True)
    assert report['status'] == 'worker_failed'
    assert report['receipt_error_code'] == 'worker_receipt_not_admitted'
    assert 'worker_status' not in report and 'source_results' not in report
    assert report['worker_receipt_sha256'] == intake.digest(mutated_receipt[0])
    attempt = intake.run_path('mutated-success')
    assert (attempt / 'worker-result.json').read_bytes() == mutated_receipt[0]
    assert json.loads((attempt / 'supervision.json').read_text()) == report
    assert all((ROOT / entry['marker_path']).exists() for entry in report['sources'])


@pytest.mark.skipif(not pilot_available(), reason='authentic local Case3 cache unavailable')
@pytest.mark.parametrize('late_stage', ['worker_return', 'final_bookkeeping'])
def test_parent_deadline_after_valid_actual_cached_worker_is_timeout(intake, isolated, monkeypatch, late_stage):
    copy_authentic_pilot(intake)
    clock = [100.]
    monkeypatch.setattr(intake.time, 'monotonic', lambda: clock[0])
    run_worker_locally(intake, monkeypatch)
    actual_worker = intake.supervise
    actual_retain = intake.retain_worker
    retained = []

    def launch(command, log, *, deadline, on_start):
        status, code = actual_worker(command, log, deadline=deadline, on_start=on_start)
        assert status == 'completed' and code == 0
        if late_stage == 'worker_return': clock[0] = deadline + .125
        return status, code

    def retain(report, attempt):
        actual_retain(report, attempt)
        retained.append(report.get('worker_status'))
        if late_stage == 'final_bookkeeping' and len(retained) == 2:
            clock[0] = 103.125

    monkeypatch.setattr(intake, 'supervise', launch)
    monkeypatch.setattr(intake, 'retain_worker', retain)
    report = intake.run_pair('Case3-during', 'late-valid-pair', seconds=3, existing_only=True)
    assert retained == ['completed', 'completed']
    assert report['worker_status'] == 'completed' and len(report['source_results']) == 2
    assert sum(f['verified_bytes'] for f in report['source_results']) == 9184649
    assert report['elapsed_seconds'] == 3.125 and report['max_seconds'] == 3
    assert report['status'] == 'timeout' and report['deadline_exceeded'] is True
    assert report['status_before_deadline_refusal'] == 'completed'
    closure = intake.run_path('late-valid-pair') / 'supervision.json'
    assert json.loads(closure.read_text()) == report
    before = closure.read_bytes()
    # A closed timeout still permits an explicit continuation using verified cache.
    clock[0] = 200.
    monkeypatch.setattr(intake, 'supervise', actual_worker)
    monkeypatch.setattr(intake, 'retain_worker', actual_retain)
    continued = intake.run_pair('Case3-during', 'after-timeout', retry_of='late-valid-pair', existing_only=True)
    assert continued['status'] == 'completed'
    assert all(f['status'] == 'existing_verified' for f in continued['source_results'])
    assert closure.read_bytes() == before
