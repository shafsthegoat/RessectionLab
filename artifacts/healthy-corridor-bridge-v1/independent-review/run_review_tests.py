"""Generated geometry controls with patient/model/archive/network access refused."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;C=ROOT/'build/healthy-corridor-bridge-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[C/'healthy_corridor.py',C/'test_corridor.py',OUT/'test_independent_corridor.py']+[ROOT/'src/resectionlab'/n for n in ['core.py','geometry.py','evaluation.py','functional_events.py','ixi_vascular_admission.py','private_vascular_evaluation.py','vascular_contact_streaming.py','independent_geometry_batch.py']]
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
code=pytest.main(['-q',str(C/'test_corridor.py'),str(OUT/'test_independent_corridor.py'),'-o','cache_dir='+str(OUT/'pytest-cache')])
after={str(p.relative_to(ROOT)):sha(p) for p in paths}
result={'status':'PASS' if code==0 and before==after and not blocked else 'FAIL','pytest_exit_code':code,'elapsed_seconds':time.monotonic()-started,'source_hashes_before':before,'source_hashes_after':after,'files_unchanged':before==after,'blocked_payload_or_network_attempts':blocked,'policy_library_imported':'torch' in sys.modules,'image_library_imported':'nibabel' in sys.modules,'scope':'Generated16-cubed public/reference arrays, scripted callbacks and reused geometric queries only; no patient/model/archive payloads, model forwards, training, network or external native solver.'}
(OUT/'test-review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'test_review_sha256':sha(OUT/'test-review.json')}))
raise SystemExit(0 if result['status']=='PASS' else 1)
