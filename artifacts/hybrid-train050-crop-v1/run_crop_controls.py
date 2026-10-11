"""One root-authorized generated-only suite using existing process controls."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
INDEX_SHA='1615cc9a7237f3bb0bbcbf537d14ab895fd60e7a69ecd79184988754caa6f789'
RELEASE_SHA='8ebd22c2ffd63995bc4e7f2b690041a14c3209402812d968d458a03fb4f4d141'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 index=HERE/'crop-source-index.json';release=HERE/'crop-prepared-release.json'
 assert sha(index)==INDEX_SHA and sha(release)==RELEASE_SHA
 assert json.loads(release.read_bytes())['execution_released'] is False
 rows=json.loads(index.read_bytes())['files']
 def pins():return {r['path']:sha(ROOT/r['path']) for r in rows}
 before=pins();assert before=={r['path']:r['sha256'] for r in rows}
 output=HERE/'crop-controls-attempt-01';output.mkdir(exist_ok=False)
 prefix=output/'fresh-pycache';assert not prefix.exists()
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
 for key in ['PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','LD_PRELOAD','DYLD_LIBRARY_PATH']:os.environ.pop(key,None)
 helper_path=ROOT/'scripts/real_intake_io.py';helper_raw=helper_path.read_bytes()
 helper=types.ModuleType('real_intake_io');helper.__file__=str(helper_path)
 exec(compile(helper_raw,str(helper_path),'exec'),helper.__dict__)
 command=[str(ROOT/'.venv/bin/python'),'-I','-B','-X','pycache_prefix='+str(prefix),'-m','unittest','discover','-s',str(HERE),'-p','test_crop*.py','-v']
 started=time.monotonic();status='not_started';code=None;pid=None;error=None
 def on_start(value):
  nonlocal pid
  pid=value;save(output/'start.json',{'pid':pid,'pgid':pid,'command':command,'deadline_seconds':30,'threads':1})
 try:
  with helper.termination_cleanup():status,code=helper.supervise(command,output/'tests.log',deadline=started+30,on_start=on_start)
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)
 elapsed=time.monotonic()-started;after=pins()
 log=output/'tests.log';text=log.read_text() if log.exists() else ''
 passed=status=='completed' and code==0 and error is None and elapsed<30 and before==after and log.stat().st_size<=4*1024**2 and '\nRan 21 tests in ' in text and text.rstrip().endswith('OK')
 receipt={'status':'pass' if passed else 'failed','worker_status':status,'exit_code':code,'error':error,'elapsed_seconds':elapsed,'timeout_seconds':30,'threads':1,'pid':pid,'pgid':pid,'owned_group_cleanup':'existing supervise finally SIGKILL group and wait, including normal completion','peak_child_rss_bytes':int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),'rss_is_observed_not_capped':True,'output_log_bytes':log.stat().st_size if log.exists() else None,'output_log_cap_acceptance_bytes':4*1024**2,'command':command,'source_index_sha256':INDEX_SHA,'false_release_sha256':RELEASE_SHA,'source_hashes_before':before,'source_hashes_after':after,'source_unchanged':before==after,'helper_path':str(helper_path.relative_to(ROOT)),'helper_sha256':hashlib.sha256(helper_raw).hexdigest(),'runner_sha256':sha(Path(__file__)),'tests_log_sha256':sha(log) if log.exists() else None,'expected_tests':21,'acquired_array_or_DICOM_reads':0,'patient_crop_executed':False,'canonical_edits':False}
 save(output/'receipt.json',receipt)
 print(json.dumps({k:receipt[k] for k in ['status','worker_status','exit_code','elapsed_seconds','peak_child_rss_bytes','source_unchanged','expected_tests']}))
 return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
