"""Single-use bounded stdlib saved-audit supervisor; no scientific execution."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
result=HERE/'supervision.json';log=HERE/'audit.log'
if result.exists() or log.exists():raise SystemExit('Exclusive supervisor output exists')
source=HERE/'audit.py';start=time.monotonic();peak=0;stop=None;checks=0
env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
with log.open('xb') as f:
    p=subprocess.Popen([sys.executable,'-I','-B',str(source)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    while p.poll() is None:
        if time.monotonic()-start>60:stop='wall_time_limit';break
        sample=subprocess.run(['/bin/ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True)
        if sample.returncode==0 and sample.stdout.strip():
            checks+=1;rss=int(sample.stdout.strip())*1024;peak=max(peak,rss)
            if rss>128*1024**2:stop='128MiB_RSS_limit';break
        elif p.poll() is None:stop='RSS_unavailable';break
        time.sleep(.075)
    if stop:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    code=p.wait()
row={'schema':'saved-stdlib-audit-supervision-v1','audit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'command':[sys.executable,'-I','-B',str(source)],'wall_cap_seconds':60,'rss_cap_bytes':128*1024**2,
    'exit_code':code,'stop_reason':stop,'child_reaped':True,'elapsed_seconds':time.monotonic()-start,
    'sampled_peak_rss_bytes':peak,'rss_samples':checks,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
    'scientific_work':False}
receipt=HERE/'receipt.json'
if receipt.exists():row['receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
raw=json.dumps(row,sort_keys=True,indent=2).encode()+b'\n'
with result.open('xb') as f:f.write(raw)
print(json.dumps(row,sort_keys=True))
raise SystemExit(code if not stop else 2)
