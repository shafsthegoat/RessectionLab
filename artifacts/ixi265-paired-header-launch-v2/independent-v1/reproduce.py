"""Generated-only launcher gate negatives; no archive/image access."""
from pathlib import Path
import hashlib,json,sys,time,types
from unittest import mock
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'build/ixi265-paired-header-launch-preparation-v1/launch.py'
SOURCE_SHA='a5999d91bd53a7c39c2a37e0650d4d3f30e7027a978457ad84e0f4194ce65615'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)) and any(p in str(args[0]) for p in ['/data/acquisition/','/sources/']):
        raise AssertionError('actual_archive_patient_or_source_open_forbidden')
sys.addaudithook(guard)
raw=(HERE/'snapshots/launch.py').read_bytes();assert sha(raw)==SOURCE_SHA
L=types.ModuleType('launch_review');L.__file__=str(SOURCE);exec(compile(raw,str(SOURCE),'exec',dont_inherit=True),L.__dict__)
generated=HERE/'generated';generated.mkdir()
wrong=generated/'unrelated-hold.json';wrong.write_text('{"decision":"HOLD","release_candidate_sha256":"0000000000000000000000000000000000000000000000000000000000000000"}\n')
r=L.candidate();r.update(execution_released=True,release_head='0'*40,released_at='generated',
    independent_review={'decision':'GO_FOR_EXACT_IXI265_HEADER_LAUNCH','path':str(wrong.relative_to(ROOT)),'sha256':sha(wrong.read_bytes())})
p=generated/'release.json';p.write_bytes(L.encode(r))
with mock.patch.object(L.sys,'flags',types.SimpleNamespace(isolated=True,no_site=True,dont_write_bytecode=True)), \
     mock.patch.object(L.sys,'_remind_header_startup',{'bootstrap_sha256':L.PINS[L.BOOT],'environment':L.ENV},create=True):
    accepted=L.check_release(p,sha(p.read_bytes()),executing=True)
assert accepted['execution_released'] is True

S=L.module(L.SUPERVISOR,'supervisor_for_missing_rss')
class Child:
    pid=776655;returncode=None
    def poll(self):return self.returncode
    def wait(self,timeout=None):assert self.returncode is not None;return self.returncode
    def kill(self):self.returncode=-9
    def terminate(self):self.returncode=-15
child=Child();observations=[]
def missing(pid,timeout_seconds):
    observations.append({'owned_child_live':child.poll() is None,'rss':0,'members':[]});return 0,[]
with mock.patch.object(S.time,'sleep',side_effect=lambda seconds:setattr(child,'returncode',0)), \
     mock.patch.object(S.os,'killpg',side_effect=AssertionError('unexpected post-exit PGID signal')):
    result=L.supervise_owned(['generated-only'],generated,time.monotonic()+2,lambda pid:None,S,
                             types.SimpleNamespace(process_group_rss=missing),popen=lambda *a,**k:child)
assert observations==[{'owned_child_live':True,'rss':0,'members':[]}]
assert result['supervision']=='completed' and result['sampled_group_peak_RSS_bytes'] is None
audit={'decision':'HOLD','launcher_sha256':SOURCE_SHA,
       'release_template_sha256':'43d242b44d6d377ac0a02fa1d283b4163eabe93e32ea89a2a312e4e593680c83',
       'release_candidate_sha256':'7995e002616535c1941bfdf3d680a34df47146a4a919b056cf4bac6583094325',
       'findings':[{'id':'review_file_semantics_and_exact_candidate_not_checked',
                    'actual_referenced_review':json.loads(wrong.read_bytes()),'executing_gate_accepted':True},
                   {'id':'missing_live_group_rss_not_refused','probe_evidence':observations,'result':result}],
       'provided_tests':{'passed':19,'seconds':0.296},
       'scope':'Only source and generated gate/process metadata; no original archives, images, or scientific worker execution',
       'actual_execution_authorized':False}
(HERE/'audit.json').write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
print(json.dumps(audit,sort_keys=True))
