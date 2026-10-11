"""Filesystem/fake-parent controls only; no native or actual worker process."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exposure_monitor_repair',HERE/'run_owned.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def disappearing_scan(tmp_path,name,live=True):
    (tmp_path/'stable.json').write_bytes(b'1234')
    doomed=tmp_path/name;doomed.write_bytes(b'old temporary')
    original=Path.lstat
    def lstat(path,*args,**kwargs):
        if path==doomed and doomed.exists():doomed.unlink()
        return original(path,*args,**kwargs)
    with patch.object(Path,'lstat',lstat):
        return R.tree_bytes(tmp_path,monitor_owned_temporaries=live)


@pytest.mark.parametrize('name',['report.json.tmp','progress.json.tmp'])
def test_atomic_owned_temporary_disappearance_does_not_fail_live_sample(tmp_path,name):
    assert disappearing_scan(tmp_path,name)==4


@pytest.mark.parametrize('name',['report.json','unrelated.tmp'])
def test_other_disappeared_file_is_not_ignored(tmp_path,name):
    with pytest.raises(FileNotFoundError):disappearing_scan(tmp_path,name)


@pytest.mark.parametrize('scope',['source','final_output'])
def test_source_and_final_scans_are_strict_even_for_owned_temporary(tmp_path,scope):
    directory=tmp_path/scope;directory.mkdir()
    with pytest.raises(FileNotFoundError):disappearing_scan(directory,'report.json.tmp',live=False)


def test_permission_error_is_not_a_transient_disappearance(tmp_path):
    doomed=tmp_path/'report.json.tmp';doomed.write_bytes(b'1')
    original=Path.lstat
    def lstat(path,*args,**kwargs):
        if path==doomed:raise PermissionError('denied')
        return original(path,*args,**kwargs)
    with patch.object(Path,'lstat',lstat),pytest.raises(PermissionError):
        R.tree_bytes(tmp_path,monitor_owned_temporaries=True)


def test_existing_temporary_counts_fully_and_symlinks_remain_refused(tmp_path):
    (tmp_path/'report.json').write_bytes(b'123')
    (tmp_path/'report.json.tmp').write_bytes(b'45678')
    assert R.tree_bytes(tmp_path,monitor_owned_temporaries=True)==8
    assert R.tree_bytes(tmp_path)==8
    (tmp_path/'progress.json.tmp').symlink_to(tmp_path/'report.json')
    with pytest.raises(ValueError,match='Linked'):R.tree_bytes(tmp_path,monitor_owned_temporaries=True)


@pytest.mark.parametrize('mode',['partial','missing_criterion','bad_progress','late_final_scan'])
def test_fake_parent_persists_failure_and_eleven_prior_calls(tmp_path,monkeypatch,mode):
    here=tmp_path/'source';here.mkdir()
    config=json.loads((HERE/'config.json').read_bytes());config['output_directory']='output'
    (here/'config.json').write_text(json.dumps(config))
    raw=json.dumps({'execution_released':True,'files':{}}).encode()
    (here/'source-index.json').write_bytes(raw)
    index=hashlib.sha256(raw).hexdigest()
    report={'status':'running'} if mode=='partial' else {'status':'completed','source_unchanged':True,
        'completed_rows':18,'rows':[{}]*18,'missing_rows':[]}
    if mode in ('bad_progress','late_final_scan'):report['useful_criterion']={'meets_predeclared_useful_criterion':True}
    progress={'native_call_costs_including_incomplete_row':{'native_strokes_started':11,'native_strokes_returned':11}}
    class FakeProcess:
        pid=1234
        returncode=-15 if mode=='partial' else 0
        def __init__(self,*args,**kwargs):
            (tmp_path/'output/report.json').write_text(json.dumps(report))
            (tmp_path/'output/progress.json').write_text('bad json' if mode=='bad_progress' else json.dumps(progress))
        def poll(self):return self.returncode
    def receipt(path,value):path.write_text(json.dumps(value))
    helper=SimpleNamespace(FastDarwinSampler=lambda:object(),cleanup_owned=lambda *args:[],_receipt=receipt)
    monkeypatch.setattr(R,'HERE',here);monkeypatch.setattr(R,'ROOT',tmp_path)
    monkeypatch.setitem(R.sys.modules,'run_contact_owned',helper)
    monkeypatch.setattr(R.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(R.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(R.sys,'argv',['run_owned.py','--source-index-sha256',index,'--execute'])
    if mode=='late_final_scan':
        original_scan=R.tree_bytes;refused={'once':False}
        def scan(root,**kwargs):
            if root==tmp_path/'output' and not kwargs.get('monitor_owned_temporaries',False) and not refused['once']:
                refused['once']=True
                raise FileNotFoundError('strict final scan failed')
            return original_scan(root,**kwargs)
        monkeypatch.setattr(R,'tree_bytes',scan)
    assert R.main()==1
    saved=json.loads((tmp_path/'output/launcher.json').read_bytes())
    assert saved['status']=='incomplete_or_failed' and saved['child_reaped']
    if mode in ('partial','missing_criterion'):
        assert saved['useful_criterion'] is None and 'finalization_error' not in saved
        assert saved['last_saved_progress']==progress
        assert saved['worker_report_status']==report['status']
    else:assert 'finalization_error' in saved
    assert saved['cleanup_errors']==[] and saved['remaining_owned_pids']==[]
