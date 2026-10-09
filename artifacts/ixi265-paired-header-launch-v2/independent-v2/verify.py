"""Independent source/generated-only final launcher controls."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,sys,time,types
from unittest import mock
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PREP=ROOT/'build/ixi265-paired-header-launch-preparation-v1'
PIN='819dffd069ed2cf5cf67c2148d3b8bc71af2f9d222cbfe3a7d7264966eb91c2e'
CAND='c099b5759a0a92624c4d8ba1f1ba587c846173c202b422b731ec11b34d0e9c5c'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)) and any(str(args[0]).startswith(str(ROOT/p)) for p in ('data/','sources/')):
        raise AssertionError('actual_archive_patient_or_synced_source_open_forbidden')
sys.addaudithook(guard)
raw=(HERE/'snapshots/launch_v2.py').read_bytes();assert sha(raw)==PIN
L=types.ModuleType('reviewed_launch_v2');L.__file__=str(PREP/'launch_v2.py')
exec(compile(raw,L.__file__,'exec',dont_inherit=True),L.__dict__)
generated=HERE/'generated';generated.mkdir()
frozen=(HERE/'snapshots/terminal-bound-CANDIDATE-NOT-RELEASED-v2.json').read_bytes();assert sha(frozen)==CAND
candidate=json.loads(frozen);assert L.reviewed_candidate_sha(candidate)==CAND
checks=[]
def check(name,fn,refuses=False):
    try:fn()
    except L.Refusal as e:
        assert refuses,(name,str(e));checks.append({'name':name,'status':'expected_refusal','reason':str(e)})
    else:
        assert not refuses,name;checks.append({'name':name,'status':'pass'})

# Validate review semantics and whole canonical candidate without running parent.
release=copy.deepcopy(candidate);release.update(execution_released=True,release_head='a'*40,released_at='generated-only')
review={'schema':'ixi265-header-launch-review-v1','decision':'GO_FOR_EXACT_IXI265_HEADER_LAUNCH','launcher_sha256':PIN,'release_candidate_sha256':CAND}
proof=generated/'review.json';rp=generated/'release.json'
def executing(doc,review_doc):
    proof.write_bytes(L.encode(review_doc));doc=copy.deepcopy(doc)
    doc['independent_review']={'decision':'GO_FOR_EXACT_IXI265_HEADER_LAUNCH','path':str(proof.relative_to(ROOT)),'sha256':sha(proof.read_bytes())}
    rp.write_bytes(L.encode(doc));return L.check_release(rp,sha(rp.read_bytes()),executing=True)
startup={'bootstrap_sha256':L.PINS[L.BOOT],'environment':L.ENV}
with mock.patch.object(L.sys,'_remind_header_startup',startup,create=True),mock.patch.object(L.sys,'flags',types.SimpleNamespace(isolated=True,no_site=True,dont_write_bytecode=True)):
    check('exact_candidate_review',lambda:executing(release,review))
    for key,bad in [('decision','HOLD'),('schema','wrong'),('launcher_sha256','0'*64),('release_candidate_sha256','0'*64)]:
        changed={**review,key:bad};check('review_'+key,lambda c=changed:executing(release,c),True)
    for key in ('archive_completion_receipt','archive_bindings','archive_stat_proofs'):
        changed=copy.deepcopy(release)
        if key=='archive_completion_receipt':changed[key]['sha256']='0'*64
        else:changed[key].pop(next(iter(changed[key])))
        check('candidate_'+key,lambda c=changed:executing(c,review),True)
    for key,bad in [('person_group','IXI:999'),('role','SELECT'),('output_directory',str(generated/'elsewhere'))]:
        changed={**release,key:bad};check('scope_'+key,lambda c=changed:executing(c,review),True)
    changed=copy.deepcopy(release);changed['limits']['worker_seconds']+=1
    check('scope_limit',lambda:executing(changed,review),True)
    changed={**release,'execution_released':False};check('unreleased_refuses',lambda:executing(changed,review),True)

# Real generated files exercise transport receipt joins and stat continuity only.
fixture=generated/'transport';fixture.mkdir()
scope={'files':[]};completion={'status':'all_bytes_verified','verified_files':3,'unresolved_files':0,'cohort_sha256':'c'*64,'files':[]}
r={'archive_completion_receipt':{'path':L.COMPLETION,'sha256':None},'archive_bindings':{},'archive_stat_proofs':{},
   'source_sha256':{'manifests/experiments/ixi-component-person-cohort-v1.json':'c'*64}}
stat_docs={};terms={}
for i,name in enumerate(('IXI-T1.tar','IXI-MRA.tar','vessel_dataset.zip')):
    file=fixture/L.ARCHIVES/name;file.parent.mkdir(parents=True,exist_ok=True);payload=('generated-%s'%i).encode();file.write_bytes(payload)
    st=file.stat();d=sha(payload);md5=hashlib.md5(payload).hexdigest()
    scope['files'].append({'filename':name,'bytes':len(payload),'expected_md5':md5})
    completion['files'].append({'path':name,'status':'byte_verified','sha256':d,'source_md5':md5,'bytes':len(payload)})
    term_path='receipts/'+name+'.json';stat_path='stats/'+name+'.json'
    terms[name]={'status':'byte_verified','sha256':d,'bytes':len(payload),'source_md5':md5}
    stat_docs[name]={'schema':'ixi-verified-archive-stat-binding-v1','archive_path':L.ARCHIVES+'/'+name,
       'source_payload_bytes_read':0,'fresh_payload_rehash':False,'regular_not_symlink':True,'partial_absent':True,
       'modification_and_metadata_change_times_not_after_verification':True,
       'stat':{'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns},
       'recorded_archive_sha256':d,'expected_bytes':len(payload),'recorded_archive_source_md5':md5,
       'verification_receipt_path':term_path,'verification_receipt_sha256':None}
    r['archive_bindings'][name]={'sha256':d,'stat':{'dev':st.st_dev,'ino':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}}
    r['archive_stat_proofs'][name]={'path':stat_path,'sha256':None}
def write(relative,obj):
    p=fixture/relative;p.parent.mkdir(parents=True,exist_ok=True);raw=L.encode(obj);p.write_bytes(raw);return sha(raw)
write('manifests/experiments/ixi-t1-mra-vessel-byte-intake-v1.json',scope)
def transport(doc=None,comp=None,stats=None,terminals=None):
    doc=copy.deepcopy(r if doc is None else doc);comp=copy.deepcopy(completion if comp is None else comp)
    stats=copy.deepcopy(stat_docs if stats is None else stats);terminals=copy.deepcopy(terms if terminals is None else terminals)
    for name,s in stats.items():
        s['verification_receipt_sha256']=write(s['verification_receipt_path'],terminals[name])
        doc['archive_stat_proofs'][name]['sha256']=write(doc['archive_stat_proofs'][name]['path'],s)
    doc['archive_completion_receipt']['sha256']=write(L.COMPLETION,comp)
    with mock.patch.object(L,'ROOT',fixture):return L.archive_prerequisites(doc)
check('generated_transport_complete',transport)
for key,bad in [('status','partial'),('verified_files',2),('unresolved_files',1),('cohort_sha256','0'*64)]:
    badcomp={**completion,key:bad};check('transport_'+key,lambda c=badcomp:transport(comp=c),True)
for key,bad in [('source_payload_bytes_read',1),('fresh_payload_rehash',True),('regular_not_symlink',False),('partial_absent',False),('modification_and_metadata_change_times_not_after_verification',False),('archive_path','wrong')]:
    docs=copy.deepcopy(stat_docs);docs['IXI-T1.tar'][key]=bad
    check('stat_'+key,lambda c=docs:transport(stats=c),True)
for key,bad in [('status','partial'),('sha256','0'*64),('bytes',0)]:
    docs=copy.deepcopy(terms);docs['IXI-T1.tar'][key]=bad
    check('terminal_'+key,lambda c=docs:transport(terminals=c),True)
target=fixture/L.ARCHIVES/'IXI-T1.tar';os.utime(target,ns=(target.stat().st_atime_ns,target.stat().st_mtime_ns+1000000))
check('publication_stat_changed',transport,True)

# Missing live RSS refuses; a real exit race never sends a stale group signal.
S=L.module(L.SUPERVISOR,'independent_supervisor')
class Child:
    pid=776655
    def __init__(self):self.returncode=None
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:raise subprocess.TimeoutExpired('generated',timeout)
        return self.returncode
    def kill(self):self.returncode=-9
    def terminate(self):self.returncode=-15
child=Child()
with mock.patch.object(S.os,'killpg',side_effect=PermissionError()):
    result=L.supervise_owned(['generated'],generated,time.monotonic()+1,lambda pid:None,S,
        types.SimpleNamespace(process_group_rss=lambda *a,**k:(0,[])),popen=lambda *a,**k:child)
assert result['supervision']=='supervisor_interrupted_or_failed' and result['parent_reaped_worker']
checks.append({'name':'missing_live_group_RSS','status':'expected_refusal','result':result})
child=Child()
def exit_race(*a,**k):child.returncode=0;return 0,[]
with mock.patch.object(S.os,'killpg',side_effect=AssertionError('stale_group_signal')):
    result=L.supervise_owned(['generated'],generated,time.monotonic()+1,lambda pid:None,S,
        types.SimpleNamespace(process_group_rss=exit_race),popen=lambda *a,**k:child)
assert result['supervision']=='completed' and result['sampled_group_peak_RSS_bytes'] is None
checks.append({'name':'confirmed_exit_race','status':'pass','result':result})

# A real generated child writes stdout/stderr; the reused supervisor discards both.
childdir=generated/'stdio';childdir.mkdir();rss=L.module(L.RSS,'generated_group_probe')
result=L.supervise_owned([sys.executable,'-I','-S','-B','-c','import sys; print("x"*100000); print("y"*100000,file=sys.stderr)'],
        childdir,time.monotonic()+3,lambda pid:None,S,rss)
assert result['supervision']=='completed' and result['parent_reaped_worker'] and list(childdir.iterdir())==[]
checks.append({'name':'real_child_stdout_stderr_discarded','status':'pass','result':result})

# Exact cold bootstrap, then guarded generated header/import compatibility.
def cold(release_path,entry,args):
    doc=json.loads(release_path.read_bytes());command=[str(Path(doc['runtime']['prefix'])/'bin/python'),'-I','-S','-B',str(ROOT/L.BOOT),str(release_path),sha(release_path.read_bytes()),str(entry),*args]
    p=subprocess.run(command,cwd=ROOT,env=doc['startup_contract']['environment'],capture_output=True,timeout=10)
    assert p.returncode==0,(p.returncode,p.stderr.decode()[:1000]);return json.loads(p.stdout)
actual_candidate=PREP/'terminal-bound-CANDIDATE-NOT-RELEASED-v2.json'
cold_check=cold(actual_candidate,PREP/'launch_v2.py',['--release',str(actual_candidate),'--release-sha256',CAND,'--check-only'])
assert cold_check['archive_opens']==0 and cold_check['execution_released'] is False and cold_check['archive_prerequisites_checked'] is False
probe=cold(PREP/'generated-probe-release-v2.json',PREP/'generated_header_probe_v2.py',['--generated-only'])
assert probe['status']=='PASS' and probe['actual_archive_opens']==0 and probe['guard_active'] is True
result={'checks':checks,'cold_check_only':cold_check,'guarded_generated_headers':probe,'original_archive_or_patient_opens':0,
        'launcher_sha256':PIN,'release_candidate_sha256':CAND}
(HERE/'independent-controls.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({'controls':len(checks),'cold_check_only':cold_check,'guarded_generated_headers':probe},sort_keys=True))
