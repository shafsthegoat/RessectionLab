"""Final source bindings and tiny candidate controls; no solver/data replay."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
BASE=ROOT/'build/hbe-v5-n12-nocache-comparison-v1/adoption_candidate'
EXPECTED={'hbe_v5_nocache_v1.py':'efeda27abc3c1a6851e1066789e51a2f05b94728def62e71061d03da7f4629de',
          'hbe_v5_nocache_hash_v1.py':'96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc',
          'hbe_v5_nocache_host_v1.py':'2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f',
          'test_generated_launcher.py':'1421f31a442bbaa6eb154b7bd3f494065b7477290557c5c20d803e34a5173078'}
bindings={}
def bind(p):
    st=p.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_size<=128*1024,p
    raw=p.read_bytes()
    after=p.lstat()
    assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    b={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    bindings[str(p.relative_to(ROOT))]=b
    return b

for name,expected in EXPECTED.items(): assert bind(BASE/name)['sha256']==expected,name
bind(BASE/'README.md')
template_path=ROOT/'build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json'
assert bind(template_path)['sha256']=='32d7566d10d1a671fe1d7fcd2cb3bfc73dc70c3632d286480fd860e2e15d2feb'
template=json.loads(template_path.read_bytes())
assert len(template['source_bindings'])==20
source_commit='d2b540f842ef1074db1e9b9c3ca7a767ea150f89'
for path,b in template['source_bindings'].items():
    assert bind(ROOT/path)['sha256']==b['sha256']
    historic=subprocess.run(['/usr/bin/git','show',source_commit+':'+path],cwd=ROOT,capture_output=True,check=True,timeout=5).stdout
    assert hashlib.sha256(historic).hexdigest()==b['sha256']
assert not (ROOT/template['output_directory']).exists()
assert not (ROOT/'outputs/mechanics/hbe-v5-nocache-policy-v1/08-tension-N12-S60-reference/attempt-01').exists()
for name in EXPECTED:
    if name.startswith('hbe_'):
        p=ROOT/'launchers'/name
        assert not p.exists(), 'Source candidate already integrated; review status must be refreshed'
environment=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
tests=[]
for p in [BASE/'test_generated_launcher.py',OUT/'test_inherited_cleanup.py',OUT/'test_owned_integration.py']:
    bind(p)
    result=subprocess.run(['python3','-B',str(p),'-v'],cwd=ROOT,env=environment,capture_output=True,text=True,timeout=30)
    assert len(result.stdout)+len(result.stderr)<32*1024
    log=OUT/(p.stem+'.txt')
    log.write_text(result.stdout+result.stderr)
    tests.append({'source':str(p.relative_to(ROOT)),'exit_code':result.returncode,
                  'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    assert result.returncode==0,(p,result.stderr)
for path,b in list(bindings.items()):
    now=bind(ROOT/path)
    assert now==b,path
result={'status':'passed_source_and_generated_candidate_audit','candidate_only':True,
        'head':subprocess.check_output(['/usr/bin/git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'frozen_twenty_source_historical_commit':source_commit,'frozen_twenty_source_bytes_unchanged':True,
        'native_output_and_policy_sidecar_absent':True,'extension_tracked_paths_absent':True,
        'bindings':bindings,'generated_test_runs':tests,'native_calls':0,'full_predecessor_reads':0,
        'hbe_readouts':0,'model_calls':0,'measured_or_patient_reads':0,'repository_tracked_writes':0,
        'compute_or_head_freeze_held':False}
(OUT/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['bindings','generated_test_runs']},sort_keys=True))
