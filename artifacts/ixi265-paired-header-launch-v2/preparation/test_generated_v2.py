"""Generated-only lifecycle controls. Never opens an original source archive."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ixi_launch_test',HERE/'launch_v2.py')
L=importlib.util.module_from_spec(spec);exec(compile((HERE/'launch_v2.py').read_bytes(),str(HERE/'launch_v2.py'),'exec'),L.__dict__)

class FakeChild:
    def __init__(self,alive=True):self.pid=778899;self.returncode=None if alive else 0;self.waits=0;self.kills=0;self.terms=0
    def poll(self):return self.returncode
    def terminate(self):self.terms+=1;self.returncode=-15
    def kill(self):self.kills+=1;self.returncode=-9
    def wait(self,timeout=None):
        self.waits+=1
        if self.returncode is None:raise subprocess.TimeoutExpired('fake',timeout)
        return self.returncode

class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='generated-',dir=HERE);self.root=Path(self.tmp.name)
        self.S=L.module(L.SUPERVISOR,'generated_supervisor');self.io=L.module(L.IO,'generated_atomic')
    def tearDown(self):self.tmp.cleanup()
    def rss(self,value=1,members=None):return types.SimpleNamespace(process_group_rss=lambda pid,timeout_seconds:(value,[{'pid':pid,'rss_bytes':value}] if members is None else members))
    def run_fake(self,child,**kwargs):
        return L.supervise_owned(['fake'],self.root,kwargs.pop('deadline',time.monotonic()+1),kwargs.pop('on_start',lambda pid:None),self.S,kwargs.pop('rss',self.rss()),popen=lambda *a,**k:child)
    def test_01_exited_no_group_signal(self):
        child=FakeChild(False)
        with mock.patch.object(self.S.os,'killpg',side_effect=AssertionError('postexit signal')):r=self.run_fake(child)
        self.assertTrue(r['parent_reaped_worker']);self.assertIsNone(r['sampled_group_peak_RSS_bytes']);self.assertEqual(child.kills,0)
    def test_02_eperm_direct_term(self):
        child=FakeChild()
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,rss=self.rss(L.LIMITS['sampled_group_RSS_bytes']+1))
        self.assertEqual(r['supervision'],'RSS_bound_exceeded');self.assertEqual(child.terms,1);self.assertTrue(r['parent_reaped_worker'])
    def test_03_cleanup_unexpected_error_fallback(self):
        child=FakeChild()
        with mock.patch.object(self.S.os,'killpg',side_effect=OSError()):r=self.run_fake(child,deadline=time.monotonic()-1)
        self.assertEqual(r['supervision'],'launch_or_cleanup_failed');self.assertEqual(child.kills,1);self.assertTrue(r['parent_reaped_worker'])
    def test_04_popen_failure(self):
        with mock.patch.object(L.subprocess,'Popen',side_effect=OSError()):
            r=L.supervise_owned(['fake'],self.root,time.monotonic()+1,lambda pid:None,self.S,self.rss(),popen=L.subprocess.Popen)
        self.assertIsNone(r['worker_pid']);self.assertEqual(r['supervision'],'launch_or_cleanup_failed')
    def test_05_onstart_failure(self):
        child=FakeChild()
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,on_start=lambda pid:(_ for _ in ()).throw(OSError()))
        self.assertEqual(r['supervision'],'supervisor_interrupted_or_failed');self.assertTrue(r['parent_reaped_worker'])
    def test_06_probe_failure(self):
        child=FakeChild();probe=types.SimpleNamespace(process_group_rss=mock.Mock(side_effect=OSError()))
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,rss=probe)
        self.assertEqual(r['supervision'],'supervisor_interrupted_or_failed');self.assertTrue(r['parent_reaped_worker'])
    def test_07_output_cap(self):
        child=FakeChild()
        with mock.patch.object(self.S,'output_bytes',return_value=L.LIMITS['outer_output_bytes']),mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child)
        self.assertEqual(r['supervision'],'output_bound_exceeded')
    def test_08_wall_cap(self):
        child=FakeChild()
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,deadline=time.monotonic()-1)
        self.assertEqual(r['supervision'],'timeout');self.assertTrue(r['parent_reaped_worker'])
    def test_09_descendant_refuses(self):
        child=FakeChild()
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,rss=self.rss(members=[{'pid':1,'rss_bytes':1}]))
        self.assertEqual(r['supervision'],'supervisor_interrupted_or_failed')
    def test_10_worker_guard(self):
        for event in ('os.fork','os.forkpty','os.posix_spawn','os.exec','subprocess.Popen','os.system','socket.connect','socket.getaddrinfo'):
            with self.assertRaises(L.Refusal):L.deny_worker_process_and_network(event,())
        L.deny_worker_process_and_network('open',('generated',))
    def test_11_false_release_no_archives(self):
        p=self.root/'release.json';p.write_bytes(L.encode(L.candidate()))
        with mock.patch.object(L,'archive_prerequisites',side_effect=AssertionError('archive prereq')):
            L.check_release(p,L.sha(p.read_bytes()),False)
            with self.assertRaisesRegex(L.Refusal,'root_release_required'):L.check_release(p,L.sha(p.read_bytes()),True)
    def test_12_protocol_delta_only(self):
        r=L.candidate();r['archive_completion_receipt']={'path':'generated.json','sha256':'0'*64};r['archive_bindings']={'generated':{}}
        a=json.loads(L.source(L.PROTOCOL));b=L.child_protocol(r)
        self.assertEqual({k for k in a if a[k]!=b[k]},{'execution_released','archive_completion_receipt','archive_bindings'})
    def parent_failure(self,mode):
        output=self.root/'attempt';release=L.candidate();release['archive_completion_receipt']={};release['archive_bindings']={}
        modules={L.SUPERVISOR:self.S,L.IO:self.io,L.RSS:self.rss()}
        real_save=L.save
        def save(directory,name,value,io,supervisor):
            if mode=='save' and name=='terminal.json':raise OSError('generated store fault')
            result=real_save(directory,name,value,io,supervisor)
            if mode=='term_publish' and name=='terminal.json':os.kill(os.getpid(),signal.SIGTERM)
            return result
        def supervision(*a,**k):
            if mode=='term_after':os.kill(os.getpid(),signal.SIGTERM)
            return {'supervision':'launch_or_cleanup_failed','worker_exit_status':None,'parent_reaped_worker':True,'worker_pid':None}
        with mock.patch.object(L,'OUTPUT',output),mock.patch.object(L,'runtime_fixity',return_value=1),mock.patch.object(L,'archive_prerequisites'),mock.patch.object(L,'module',side_effect=lambda p,n:modules[p]),mock.patch.object(L,'supervise_owned',side_effect=supervision),mock.patch.object(L,'save',side_effect=save):
            if mode=='save':
                with self.assertRaises(OSError):L.parent(release,self.root/'release.json','a'*64)
            else:self.assertEqual(L.parent(release,self.root/'release.json','a'*64),1)
        self.assertTrue((output/'intent.json').exists())
        return output
    def test_13_parent_launch_failure_has_unknown_receipt(self):
        p=self.parent_failure('launch');r=json.loads((p/'terminal.json').read_bytes());self.assertIsNone(r['other_person_member_body_bytes_read']);self.assertFalse(r['archive_read_accounting_complete'])
    def test_14_final_save_failure_has_fallback(self):self.assertTrue((self.parent_failure('save')/'terminal-publication-failure.json').exists())
    def test_15_sigterm_after_supervision(self):self.assertTrue(json.loads((self.parent_failure('term_after')/'terminal.json').read_bytes())['termination_requested'])
    def test_16_sigterm_during_final_save(self):self.assertTrue((self.parent_failure('term_publish')/'post-publication-refusal.json').exists())
    def test_17_semantic_exit0_not_success(self):
        with self.assertRaisesRegex(L.Refusal,'worker_semantic_refusal'):L.semantic_success({'status':'failed_preserved_no_retry'},{},'',{})
    def test_18_real_generated_child_timeout_reaped(self):
        r=L.supervise_owned([sys.executable,'-I','-S','-B','-c','import time; time.sleep(3)'],self.root,time.monotonic()+.15,lambda pid:None,self.S,self.rss())
        self.assertEqual(r['supervision'],'timeout');self.assertTrue(r['parent_reaped_worker'])
    def test_19_unresolved_cleanup_remains_failure(self):
        child=FakeChild();child.kill=mock.Mock(side_effect=PermissionError());child.terminate=mock.Mock(side_effect=PermissionError())
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,deadline=time.monotonic()-1)
        self.assertFalse(r['parent_reaped_worker']);self.assertEqual(r['direct_cleanup_error'],'PermissionError')
    def test_20_live_empty_group_refuses(self):
        child=FakeChild()
        with mock.patch.object(self.S.os,'killpg',side_effect=PermissionError()):r=self.run_fake(child,rss=self.rss(members=[]))
        self.assertEqual(r['supervision'],'supervisor_interrupted_or_failed');self.assertTrue(r['parent_reaped_worker'])
        self.assertEqual(r['group_RSS_observations'],0);self.assertIsNone(r['sampled_group_peak_RSS_bytes'])
    def test_21_exited_during_probe_does_not_signal_stale_group(self):
        child=FakeChild()
        def probe(pid,timeout_seconds):child.returncode=0;return 0,[]
        with mock.patch.object(self.S.os,'killpg',side_effect=AssertionError('stale group')):
            r=self.run_fake(child,rss=types.SimpleNamespace(process_group_rss=probe))
        self.assertEqual(r['supervision'],'completed');self.assertIsNone(r['sampled_group_peak_RSS_bytes'])
    def test_22_review_must_bind_terminal_candidate(self):
        release=L.candidate();release.update(execution_released=True,release_head='a'*40,released_at='generated')
        audit={'schema':'ixi265-header-launch-review-v1','decision':'GO_FOR_EXACT_IXI265_HEADER_LAUNCH',
            'launcher_sha256':release['launcher_sha256'],'release_candidate_sha256':L.reviewed_candidate_sha(release)}
        proof=self.root/'audit.json';proof.write_bytes(L.encode(audit))
        release['independent_review']={'decision':audit['decision'],'path':str(proof.relative_to(L.ROOT)),'sha256':L.sha(proof.read_bytes())}
        path=self.root/'release.json'
        def check():
            path.write_bytes(L.encode(release));return L.check_release(path,L.sha(path.read_bytes()),True)
        startup={'bootstrap_sha256':L.PINS[L.BOOT],'environment':L.ENV}
        flags=types.SimpleNamespace(isolated=True,no_site=True,dont_write_bytecode=True)
        with mock.patch.object(L.sys,'_remind_header_startup',startup,create=True),mock.patch.object(L.sys,'flags',flags):
            check()
            release['archive_completion_receipt']={'path':'generated-other.json','sha256':'e'*64}
            with self.assertRaisesRegex(L.Refusal,'review_candidate_binding'):check()
            release['archive_completion_receipt']=None;audit['decision']='HOLD';proof.write_bytes(L.encode(audit));release['independent_review']['sha256']=L.sha(proof.read_bytes())
            with self.assertRaisesRegex(L.Refusal,'review_candidate_binding'):check()

if __name__=='__main__':
    import sys
    unittest.main(verbosity=2)
