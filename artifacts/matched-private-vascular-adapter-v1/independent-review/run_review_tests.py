"""Small generated test suite with actual payload/network access refused."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;C=ROOT/'build/matched-private-vascular-adapter-v1'
paths=[C/'matched_private_vascular.py',C/'private_vascular_evaluation.py',C/'test_adapter.py',ROOT/'src/resectionlab/private_vascular_evaluation.py',ROOT/'src/resectionlab/independent_geometry_batch.py',ROOT/'src/resectionlab/research_estimate_planning.py',ROOT/'tests/test_private_vascular_evaluation.py',OUT/'test_independent_adapter.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(ROOT)):sha(p) for p in paths};blocked=[]
def guard(event,args):
 if event in ('socket.connect','socket.getaddrinfo'):
  blocked.append(event);raise AssertionError('actual network forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).endswith(('.nii','.nii.gz','.tar','.zip','.pt','.ckpt','.bin','.mat')):
  blocked.append(str(args[0]));raise AssertionError('model/patient/archive payload forbidden')
sys.addaudithook(guard)
import pytest
start=time.monotonic()
code=pytest.main(['-q',str(C/'test_adapter.py'),str(ROOT/'tests/test_private_vascular_evaluation.py'),str(OUT/'test_independent_adapter.py'),'-o','cache_dir='+str(OUT/'pytest-cache'),'--basetemp='+str(OUT/'tmp')])
after={str(p.relative_to(ROOT)):sha(p) for p in paths}
receipt={'status':'PASS' if code==0 and before==after and not blocked and 'torch' not in sys.modules else 'FAIL','pytest_exit_code':code,'elapsed_seconds':time.monotonic()-start,'source_hashes_before':before,'source_hashes_after':after,'blocked_attempts':blocked,'torch_imported':'torch' in sys.modules,'source_files_unchanged_during_tests':before==after,'scope':'Small generated scripted-policy and geometry fixtures only; no model forwards/training, patient/archive payload or network access.'}
(OUT/'test-review.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':receipt['status'],'test_review_sha256':sha(OUT/'test-review.json')}))
raise SystemExit(0 if receipt['status']=='PASS' else 1)
