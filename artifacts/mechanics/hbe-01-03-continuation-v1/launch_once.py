"""Capture one explicit released invocation; the reviewed runner owns all caps."""
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[3]
ev=Path(__file__).resolve().parent
launch=json.loads((ev/'prospective-launch.json').read_text())
source=root/'outputs/validation/hbe-continuation-a9f2783/frozen-source/scripts/mechanics_hbe_experiment.py'
command=[str(root/'.venv/bin/python'),str(source),'--root',str(root),'--protocol',launch['protocol']['path'],'--protocol-sha256',launch['protocol']['sha256'],'--release',launch['release']['path'],'--release-sha256',launch['release']['sha256'],'--execute','--reuse-verified-coarse']
def save(name,value):
 with (ev/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
started=time.monotonic()
save('execution-started.json',{'command':command,'cwd':str(root),'release':launch['release'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'utc':datetime.now(timezone.utc).isoformat(),'no_retry':True,'supervised_worker_seconds':launch['remaining_supervised_seconds']})
with (ev/'launcher.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 save('launcher-process.json',{'pid':child.pid,'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()})
 code=child.wait()
save('launcher-completion.json',{'exit_code':code,'launcher_elapsed_seconds':time.monotonic()-started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'no_retry':True})
print(json.dumps({'exit_code':code,'launcher_elapsed_seconds':time.monotonic()-started}),flush=True)
raise SystemExit(code)
