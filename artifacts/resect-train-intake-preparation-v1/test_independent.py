"""Offline receipt/process controls; only authentic existing compressed Case3 bytes."""
from pathlib import Path
import hashlib,importlib,json,os,shutil,subprocess,sys,time,urllib.request
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def intake(monkeypatch,tmp_path):
 monkeypatch.syspath_prepend(str(ROOT/'scripts'))
 m=importlib.import_module('resect_train_intake')
 monkeypatch.setattr(m,'DATA',tmp_path/'data');m.DATA.mkdir()
 original=subprocess.Popen
 def local_only(command,*a,**kw):
  if any(str(x).startswith(('http://','https://')) for x in command):raise AssertionError('network forbidden')
  return original(command,*a,**kw)
 monkeypatch.setattr(subprocess,'Popen',local_only)
 monkeypatch.setattr(urllib.request.OpenerDirector,'open',lambda *a,**kw:pytest.fail('network forbidden'))
 return m

def fake_completed_launch(m,monkeypatch,*,files='full',late=None):
 def launch(command,log,*,deadline,on_start):
  on_start(99999999)
  attempt=log.parent;intent=json.loads((attempt/'intent.json').read_text());source=(attempt/'source.json').read_bytes()
  records=[]
  for sid in [e['source_id'] for e in intent['sources']]:
   s=m.source_by_id(sid)
   records.append({'source_id':sid,'status':'existing_verified','sha256':s.sha256 or '1'*64,'verified_bytes':s.bytes,'retained_partial_bytes':0})
  if files=='empty':records=[]
  if files=='duplicate':records=[records[0],records[0]]
  if files=='failed':records[0]['status']='failed'
  record={'run_id':intent['run_id'],'pair_id':intent['pair_id'],'manifest_sha256':m.MANIFEST_SHA,
   'execution_source_sha256':m.digest(source),'intent_sha256':m.digest((attempt/'intent.json').read_bytes()),
   'status':'completed','files':records,**m.CLAIMS}
  m.save(attempt/'worker-result.json',record)
  if late:late[0]=deadline+.125
  return 'completed',0
 monkeypatch.setattr(m,'supervise',launch)

@pytest.mark.parametrize('files',['empty','duplicate','failed'])
def test_parent_refuses_incomplete_or_wrong_worker_success(intake,monkeypatch,files):
 fake_completed_launch(intake,monkeypatch,files=files)
 r=intake.run_pair('Case2-during','wrong-result-'+files,existing_only=True)
 assert r['status']!='completed',{'status':r['status'],'source_results':r.get('source_results')}

def test_final_parent_elapsed_over_limit_cannot_be_completed(intake,monkeypatch):
 rights=intake.DATA/'rights/README.txt';rights.parent.mkdir();shutil.copyfile(ROOT/'artifacts/resect-component-admission-v1/rights-review-02/README.txt',rights)
 (intake.DATA/'originals').mkdir()
 for name in ('Case3-US-during.nii.gz','Case3-US-during-resection.nii.gz'):
  shutil.copyfile(ROOT/'data/annotations/resect-seg-v1/originals'/name,intake.DATA/'originals'/name)
 clock=[100.]
 monkeypatch.setattr(intake.time,'monotonic',lambda:clock[0])
 def launch(command,log,*,deadline,on_start):
  on_start(99999999)
  with monkeypatch.context() as p:
   p.setattr(intake.os,'getppid',lambda:os.getpid())
   worker=intake.worker(command[command.index('--run-id')+1],command[command.index('--intent-sha')+1])
  assert worker['status']=='completed' and len(worker['files'])==2
  clock[0]=deadline+.125
  return 'completed',0
 monkeypatch.setattr(intake,'supervise',launch)
 r=intake.run_pair('Case3-during','late-completion',seconds=3,existing_only=True)
 assert r['elapsed_seconds']>3
 assert r['worker_status']=='completed' and len(r['source_results'])==2
 assert sum(v['verified_bytes'] for v in r['source_results'])==9184649
 assert r['status']=='timeout' and r['status_before_deadline_refusal']=='completed'
 assert r['deadline_exceeded'] is True and r['elapsed_seconds']==3.125

def test_all_bound_metadata_is_git_tracked_and_fresh_checkout_preflight(intake,tmp_path):
 m=intake.require_manifest();checkout=tmp_path/'portable-checkout'
 paths={*intake.CODE,'manifests/resect-train-cavity-acquisition-v1.json',*[e['path'] for e in m['metadata_bindings']]}
 for name in paths:
  if name not in ('scripts/resect_train_intake.py','manifests/resect-train-cavity-acquisition-v1.json'):
   subprocess.run(['git','ls-files','--error-unmatch',name],cwd=ROOT,check=True,capture_output=True)
  target=checkout/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
 run=subprocess.run([sys.executable,'-B',str(checkout/'scripts/resect_train_intake.py'),'preflight'],cwd=checkout,capture_output=True,text=True,timeout=5)
 assert run.returncode==0,run.stdout+run.stderr
 r=json.loads(run.stdout);assert r['status']=='metadata_preflight_passed' and r['decoded_array_bytes']==0
 assert not (checkout/'data').exists()

def test_cache_failure_retains_first_success_and_explicit_retry_links(intake,monkeypatch):
 rights=intake.DATA/'rights/README.txt';rights.parent.mkdir();shutil.copyfile(ROOT/'artifacts/resect-component-admission-v1/rights-review-02/README.txt',rights)
 original=ROOT/'data/annotations/resect-seg-v1/originals/Case3-US-during.nii.gz'
 target=intake.DATA/'originals'/original.name;target.parent.mkdir();shutil.copyfile(original,target)
 def launch(command,log,*,deadline,on_start):
  on_start(99999999)
  with monkeypatch.context() as p:
   p.setattr(intake.os,'getppid',lambda:os.getpid())
   r=intake.worker(command[command.index('--run-id')+1],command[command.index('--intent-sha')+1])
  return ('completed',0) if r['status']=='completed' else ('worker_failed',1)
 monkeypatch.setattr(intake,'supervise',launch)
 first=intake.run_pair('Case3-during','partial-real-cache',existing_only=True)
 assert first['status']=='worker_failed' and first['source_results'][0]['status']=='existing_verified' and first['source_results'][1]['status']=='failed'
 before=(intake.run_path('partial-real-cache')/'supervision.json').read_bytes()
 second=intake.run_pair('Case3-during','implicit-real-cache',existing_only=True)
 assert second['status']=='setup_failed' and second['error_code']=='explicit_retry_of_prior_attempt_required'
 mask=ROOT/'data/annotations/resect-seg-v1/originals/Case3-US-during-resection.nii.gz';shutil.copyfile(mask,intake.DATA/'originals'/mask.name)
 third=intake.run_pair('Case3-during','explicit-real-cache',existing_only=True,retry_of='partial-real-cache')
 assert third['status']=='completed' and all(v['status']=='existing_verified' for v in third['source_results'])
 assert (intake.run_path('partial-real-cache')/'supervision.json').read_bytes()==before
 assert sum(v['verified_bytes'] for v in third['source_results'])==9184649
