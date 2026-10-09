"""Offline archive protocol controls; opaque fixtures, no clinical payloads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
BODY = b'opaque-archive-protocol-fixture-0123456789'


@pytest.fixture
def archive(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('synthrad_controls', ROOT / 'scripts/acquire_synthrad_task1_archive.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real = subprocess.Popen

    def local_only(command, *args, **kwargs):
        if str(command[0]) == '/usr/bin/curl' and command[1:] != ['-q', '--version']:
            pytest.fail('network prohibited by offline controls')
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', local_only)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', lambda *a, **k: pytest.fail('network prohibited'))
    return module


@pytest.fixture
def case(archive, tmp_path):
    manifest = copy.deepcopy(archive.load_manifest())
    source = manifest['sources'][0]
    source.update(bytes=len(BODY), expected_md5=hashlib.md5(BODY).hexdigest(),
                  source_url='https://example.invalid/opaque-test-fixture')
    manifest['bounds']['free_disk_reserve_bytes'] = 0
    return archive.Queue(tmp_path, manifest), source


def trial(archive, queue, source, prefix=None):
    number = len(queue.attempts(source)) + 1
    path = queue.object_dir(source) / 'attempts' / f'{number:04d}'
    path.mkdir(parents=True)
    intent = {'source_id': source['id'], 'source_binding': queue.binding(source),
              'manifest_sha256': archive.MANIFEST_SHA, 'attempt': number,
              'prefix': prefix or {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}}
    queue.save(path / 'intent.json', intent)
    return path, intent


def response(path, source, offset, body, *, status=None, content_range=None, extra=''):
    status = status or (206 if offset else 200)
    headers = f'HTTP/1.1 {status} Response\r\nContent-Length: {source["bytes"] - offset}\r\n'
    if offset:
        headers += 'Content-Range: ' + (content_range or f'bytes {offset}-{source["bytes"]-1}/{source["bytes"]}') + '\r\n'
    (path / 'headers.local').write_text(headers + extra + '\r\n')
    (path / 'body.partial').write_bytes(body)


def partial(archive, queue, source):
    path, intent = trial(archive, queue, source)
    response(path, source, 0, BODY[:9])
    result = archive.transfer.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=28)
    assert result['status'] == 'transport_deferred'
    return archive.transfer.prefix_state(queue, source, time.monotonic()+10)


def test_frozen_archive_roles_authority_helpers_and_exact_bounds(archive):
    manifest = archive.load_manifest()
    assert archive.runtime_binding(manifest)
    source = manifest['sources'][0]
    assert source['source_url'] == 'https://zenodo.org/api/records/7260705/files/Task1.zip/content'
    assert source['bytes'] == 14471900926
    assert source['expected_md5'] == '360bc61e2320d5c5a7145454a81c566a'
    assert manifest['rights']['license_spdx'] == 'CC-BY-NC-4.0'
    roles = json.loads((ROOT / manifest['role_source']['path']).read_text())
    assert roles['split']['counts'] == {'TRAIN':126, 'SELECT':27, 'MEASUREMENT_EVAL':27}
    b = manifest['bounds']
    assert (b['workers'], b['compressed_cache_bytes'], b['free_disk_reserve_bytes']) == (1,32*1024**3,8*1024**3)
    assert (b['network_attempt_seconds'], b['max_attempts_per_object'], b['total_campaign_seconds']) == (21600,3,86400)
    assert (b['hash_or_recovery_phase_seconds'], b['worker_log_bytes'], b['supervisor_termination_grace_seconds']) == (1800,65536,30)
    assert archive.status(archive.Queue(ROOT, manifest))['metadata_only']
    assert manifest['claims']['decoded_array_bytes'] == 0


@pytest.mark.parametrize('which', ['manifest', 'freeze', 'role', 'proof'])
def test_changed_frozen_input_fails_before_any_request(archive, tmp_path, which):
    manifest = archive.load_manifest()
    frozen = json.loads((ROOT / archive.FREEZE).read_text())
    for name in [archive.FREEZE, *frozen['files_sha256']]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    name = {'manifest':archive.MANIFEST, 'freeze':archive.FREEZE,
            'role':manifest['role_source']['path'], 'proof':manifest['proofs'][0]['path']}[which]
    with (tmp_path / name).open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(archive.Refusal):
        archive.load_manifest(tmp_path)


def test_whole_object_200_then_idempotent_verification(archive, case):
    queue, source = case
    path, intent = trial(archive, queue, source)
    response(path, source, 0, BODY)
    result = archive.transfer.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=0)
    assert result['status'] == 'byte_verified'
    assert queue.target(source).read_bytes() == BODY
    assert archive.transfer.recover(queue, source, time.monotonic()+10)['sha256'] == hashlib.sha256(BODY).hexdigest()
    assert not result['training_admitted'] and result['decoded_array_bytes'] == 0


def test_partial_resume_and_history_survive_new_queue(archive, case):
    queue, source = case
    prefix = partial(archive, queue, source)
    restarted = archive.Queue(queue.root, queue.manifest)
    path, intent = trial(archive, restarted, source, prefix)
    response(path, source, prefix['bytes'], BODY[prefix['bytes']:])
    result = archive.transfer.close_attempt(restarted, source, path, intent, time.monotonic()+10, curl_code=0)
    assert result['status'] == 'byte_verified'
    assert restarted.target(source).read_bytes() == BODY
    assert len(restarted.attempts(source)) == 2
    assert not restarted.partial(source).exists()


@pytest.mark.parametrize('wrong', ['200','start','end','total','length','encoding','md5'])
def test_bad_resume_retains_prefix_and_failed_response(archive, case, wrong):
    queue, source = case
    prefix = partial(archive, queue, source)
    path, intent = trial(archive, queue, source, prefix)
    n = prefix['bytes']
    ranges = {'start':f'bytes {n+1}-{len(BODY)-1}/{len(BODY)}',
              'end':f'bytes {n}-{len(BODY)-2}/{len(BODY)}',
              'total':f'bytes {n}-{len(BODY)-1}/{len(BODY)+1}'}
    extra = {'length':'Content-Length: 0\r\n', 'encoding':'Content-Encoding: gzip\r\n'}.get(wrong,'')
    response(path, source, n, b'!'*(len(BODY)-n) if wrong=='md5' else BODY[n:],
             status=200 if wrong=='200' else 206, content_range=ranges.get(wrong), extra=extra)
    result = archive.transfer.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=0)
    assert result['status'] == 'terminal_failure'
    assert queue.partial(source).read_bytes() == BODY[:n]
    assert (path/'body.partial').exists() and not queue.target(source).exists()


def test_torn_append_recovers_with_original_failure_unchanged(archive, case, monkeypatch):
    queue, source = case
    prefix = partial(archive, queue, source)
    path, intent = trial(archive, queue, source, prefix)
    response(path, source, prefix['bytes'], BODY[prefix['bytes']:])
    original = queue.reserve
    def stop(n):
        if (path/'merge.json').exists():
            with queue.partial(source).open('ab') as handle:
                handle.write(BODY[prefix['bytes']:prefix['bytes']+4])
            raise KeyboardInterrupt()
        return original(n)
    with monkeypatch.context() as patch:
        patch.setattr(queue, 'reserve', stop)
        with pytest.raises(KeyboardInterrupt):
            archive.transfer.merge_suffix(queue, source, path, intent, time.monotonic()+10)
    assert archive.transfer.recover(queue, source, time.monotonic()+10)['sha256'] == hashlib.sha256(BODY).hexdigest()
    assert queue.target(source).read_bytes() == BODY


def test_attempt_limit_and_provider_cooldown_are_persistent(archive, case):
    queue, source = case
    for _ in range(3):
        path, intent = trial(archive, queue, source)
        response(path, source, 0, b'', status=429, extra='Retry-After: 1800\r\n')
        archive.transfer.close_attempt(queue, source, path, intent, time.monotonic()+10, curl_code=22)
    restarted = archive.Queue(queue.root, queue.manifest)
    assert archive.transfer.states(restarted)[1] > time.time()+1790
    with pytest.raises(archive.Refusal, match='attempt_limit'):
        archive.run_one(restarted, source, {}, {}, time.monotonic()+10, {},
                        launcher=lambda *a,**k:pytest.fail('fourth request forbidden'))


def test_campaign_is_not_renewed_and_clock_rollback_fails(archive, case, monkeypatch):
    queue, _ = case
    with monkeypatch.context() as patch:
        patch.setattr(archive.time, 'time', lambda:1000.)
        first = archive.campaign(queue, create=True)
        patch.setattr(archive.time, 'time', lambda:2000.)
        assert archive.campaign(queue, create=True) == first
        patch.setattr(archive.time, 'time', lambda:999.)
        with pytest.raises(archive.Refusal, match='campaign_binding_or_clock'):
            archive.campaign(queue)


def test_cache_counts_staged_files_and_hardlinks_and_refuses_overflow(archive, case):
    queue, source = case
    queue.cache.mkdir(parents=True)
    f = queue.cache/'body.partial'
    f.write_bytes(b'x'*300)
    os.link(f,queue.cache/'hardlink')
    assert queue.used_bytes()==300
    queue.bounds['compressed_cache_bytes']=350
    with pytest.raises(archive.Refusal, match='compressed_cache_quota'):
        archive.run_one(queue,source,{}, {},time.monotonic()+10,{},
                        launcher=lambda *a,**k:pytest.fail('quota must precede launch'))
    assert not (queue.object_dir(source)/'attempts').exists()


def local_driver(archive, queue, monkeypatch):
    manifest_path=queue.root/archive.MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes((ROOT/archive.MANIFEST).read_bytes())
    monkeypatch.setattr(archive,'runtime_binding',lambda *a,**k:{})
    monkeypatch.setattr(archive.transfer.native,'system_curl_binding',lambda **k:{'available':True})


def test_driver_runs_continuously_and_reuses_verified_archive(archive,case,monkeypatch):
    queue,source=case
    local_driver(archive,queue,monkeypatch)
    called=[]
    def local_attempt(q,s,*args):
        called.append(s['id'])
        path,intent=trial(archive,q,s)
        response(path,s,0,BODY)
        return archive.transfer.close_attempt(q,s,path,intent,time.monotonic()+10,curl_code=0)
    monkeypatch.setattr(archive,'run_one',local_attempt)
    monkeypatch.setattr(archive.time,'sleep',lambda *a:pytest.fail('no review pause'))
    assert archive.run(queue)==0
    assert called==[source['id']]
    assert archive.run(queue)==0 and len(called)==1
    assert not archive.status(queue)['training_admitted']


def test_expired_campaign_does_not_launch_or_reset(archive,case,monkeypatch):
    queue,source=case
    local_driver(archive,queue,monkeypatch)
    first=archive.campaign(queue,create=True)
    monkeypatch.setattr(archive.time,'time',lambda:first['deadline_epoch']+1)
    monkeypatch.setattr(archive,'run_one',lambda *a,**k:pytest.fail('expired campaign'))
    assert archive.run(queue)==2
    assert archive.campaign(queue)==first
    assert queue.attempts(source)==[]


def test_cooldown_beyond_campaign_retains_failure_without_wait(archive,case,monkeypatch):
    queue,source=case
    local_driver(archive,queue,monkeypatch)
    active=archive.campaign(queue,create=True)
    path,intent=trial(archive,queue,source)
    response(path,source,0,b'',status=429,extra='Retry-After: 999999\r\n')
    result=archive.transfer.close_attempt(queue,source,path,intent,time.monotonic()+10,curl_code=22)
    original=(path/'result.json').read_bytes()
    monkeypatch.setattr(archive,'run_one',lambda *a,**k:pytest.fail('provider cooldown'))
    monkeypatch.setattr(archive.time,'sleep',lambda *a:pytest.fail('out-of-budget wait'))
    assert archive.run(queue)==2
    assert (path/'result.json').read_bytes()==original
    assert archive.campaign(queue)==active


def test_supervisor_caps_log_and_stops_noisy_process(archive,tmp_path):
    result=archive.supervise([sys.executable,'-B','-c','import os,time; os.write(1,b"x"*200000); time.sleep(10)'],
        tmp_path/'worker.log',deadline=time.monotonic()+2,grace_seconds=.15,log_bytes=65536,on_start=lambda pid:None)
    assert result['state']=='log_limit'
    assert (tmp_path/'worker.log').stat().st_size==65536
    assert result['elapsed_seconds']<2


def test_supervisor_gives_cooperative_child_termination_grace(archive,tmp_path):
    marker=tmp_path/'finished'
    script=('import signal,time,pathlib; '
            'signal.signal(signal.SIGTERM,lambda *a:(time.sleep(.1),pathlib.Path('+repr(str(marker))+').write_text("done"),exit(0))); '
            'time.sleep(10)')
    result=archive.supervise([sys.executable,'-B','-c',script],tmp_path/'worker.log',
        deadline=time.monotonic()+.3,grace_seconds=.5,log_bytes=65536,on_start=lambda pid:None)
    assert result['state']=='timeout' and result['sigterm_sent']
    assert marker.read_text()=='done'
    assert result['returncode']==0


def test_supervisor_kills_only_after_grace_for_uncooperative_child(archive,tmp_path):
    result=archive.supervise([sys.executable,'-B','-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)'],
        tmp_path/'worker.log',deadline=time.monotonic()+.3,grace_seconds=.2,log_bytes=65536,on_start=lambda pid:None)
    assert result['state']=='timeout' and result['returncode']==-signal.SIGKILL
    assert result['elapsed_seconds']>=.49
    assert result['elapsed_seconds']<2


def test_native_command_is_ordinary_get_then_exact_resume_with_system_tls(archive,case,monkeypatch):
    queue,source=case
    captured=[]
    monkeypatch.setattr(archive.transfer.native,'system_curl_binding',lambda **k:{'available':True})
    class Process:
        returncode=0
        def __init__(self,command,**kwargs):
            captured.append(command)
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(archive.transfer.subprocess,'Popen',Process)
    path,intent=trial(archive,queue,source)
    archive.transfer.native_fetch(queue,source,path,intent,time.monotonic()+10)
    assert '--range' not in captured[0]
    for name in ['body.partial','headers.local','native-request.json']:
        (path/name).unlink()
    response(path,source,0,BODY[:9])
    archive.transfer.close_attempt(queue,source,path,intent,time.monotonic()+10,curl_code=28)
    prefix=archive.transfer.prefix_state(queue,source,time.monotonic()+10)
    path,intent=trial(archive,queue,source,prefix)
    archive.transfer.native_fetch(queue,source,path,intent,time.monotonic()+10)
    cmd=captured[1]
    assert cmd[cmd.index('--range')+1]==f'9-{len(BODY)-1}'
    assert cmd[:2]==['/usr/bin/curl','-q'] and '--tlsv1.2' in cmd and '--no-location' in cmd
    assert '--insecure' not in cmd and '-k' not in cmd


def test_worker_executes_only_bound_local_fixture_and_publishes(archive,case,monkeypatch):
    queue,source=case
    active=archive.campaign(queue,create=True)
    path=queue.object_dir(source)/'attempts'/'0001'
    path.mkdir(parents=True)
    client={'available':True}
    intent={'source_id':source['id'],'source_binding':queue.binding(source),
            'manifest_sha256':archive.MANIFEST_SHA,'attempt':1,
            'prefix':{'bytes':0,'sha256':hashlib.sha256(b'').hexdigest()},
            'supervisor_pid':os.getppid(),'network_deadline':time.monotonic()+10,
            'campaign_deadline':time.monotonic()+30,'campaign_deadline_epoch':active['deadline_epoch'],
            'runtime':{},'client':client,'created_utc':archive.transfer.utc()}
    queue.save(path/'intent.json',intent)
    monkeypatch.setattr(archive,'ROOT',queue.root)
    monkeypatch.setattr(archive,'load_manifest',lambda:queue.manifest)
    monkeypatch.setattr(archive,'runtime_binding',lambda *a,**k:{})
    monkeypatch.setattr(archive.transfer.native,'system_curl_binding',lambda **k:client)
    called=[]
    def local_fetch(q,s,p,i,d):
        called.append(s['id'])
        response(p,s,0,BODY)
        return 0
    monkeypatch.setattr(archive.transfer,'native_fetch',local_fetch)
    with pytest.raises(archive.Refusal,match='worker_intent_or_runtime'):
        archive.worker(str(path.relative_to(queue.root)),'0'*64)
    assert called==[]
    assert archive.worker(str(path.relative_to(queue.root)),archive.transfer.digest(archive.transfer.encode(intent)))==0
    assert called==[source['id']] and queue.target(source).read_bytes()==BODY
    assert (path/'transport-finished.json').exists()


def test_parent_interruption_closes_and_preserves_received_partial(archive,case):
    queue,source=case
    active=archive.campaign(queue,create=True)
    def interrupted_launcher(command,log,**kwargs):
        assert kwargs['grace_seconds']==30 and kwargs['log_bytes']==65536
        path=log.parent
        response(path,source,0,BODY[:9])
        return {'state':'interrupted','returncode':-15}
    with pytest.raises(KeyboardInterrupt):
        archive.run_one(queue,source,{}, {'available':True},time.monotonic()+100,active,
                        launcher=interrupted_launcher)
    assert queue.partial(source).read_bytes()==BODY[:9]
    records=queue.attempts(source)
    assert len(records)==1 and records[0][2]['status']=='transport_deferred'
    assert (records[0][0]/'supervision.json').exists()
