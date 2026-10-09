"""Tiny metadata controls only, with real payload/network access forbidden."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;C=ROOT/'build/ixi-vascular-admission-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[C/'ixi_vascular_admission.py',C/'test_admission.py',OUT/'test_independent_admission.py',ROOT/'manifests/experiments/ixi-component-person-cohort-v1.json']
before={str(p.relative_to(ROOT)):sha(p) for p in paths};blocked=[]
stdlib_archive=str(Path(sys.base_prefix)/'lib'/f'python{sys.version_info.major}{sys.version_info.minor}.zip')
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        blocked.append(event);raise AssertionError('network forbidden')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path!=stdlib_archive and path.endswith(('.nii','.nii.gz','.tar','.zip','.pt','.ckpt','.bin','.mat')):
            blocked.append(path);raise AssertionError('patient/model/archive payload forbidden')
sys.addaudithook(guard)
import pytest
started=time.monotonic()
code=pytest.main(['-q',str(C/'test_admission.py'),str(OUT/'test_independent_admission.py'),'-o','cache_dir='+str(OUT/'pytest-cache')])
after={str(p.relative_to(ROOT)):sha(p) for p in paths}
result={'status':'PASS' if code==0 and before==after and not blocked else 'FAIL','pytest_exit_code':code,'elapsed_seconds':time.monotonic()-started,'source_hashes_before':before,'source_hashes_after':after,'files_unchanged':before==after,'blocked_payload_or_network_attempts':blocked,'policy_library_imported':'torch' in sys.modules,'image_library_imported':'nibabel' in sys.modules,'scope':'Already-public cohort JSON and generated metadata/4x4 matrices only; no patient/header/image/archive array or model weights, no model/native execution or training.'}
(OUT/'test-review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'test_review_sha256':sha(OUT/'test-review.json')}))
raise SystemExit(0 if result['status']=='PASS' else 1)
