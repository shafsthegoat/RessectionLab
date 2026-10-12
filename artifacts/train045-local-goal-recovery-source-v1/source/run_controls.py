"""One <=60s sequential generated check job; no automatic retry."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
OUT=HERE/'controls-01'
OUT.mkdir()
sources=[*HERE.glob('*.py'),HERE/'config.json',HERE.parent/'model/policy.py']
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
started=time.monotonic();deadline=started+60.
env=dict(os.environ)
env.update({k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
receipt={'status':'running','deadline_seconds':60,'per_process_rss_cap_bytes':1024**3,'automatic_retry':False,'runs':[]}
def run(name, script, *args):
    command=[sys.executable,'-I','-B',str(HERE/script),*args]
    row={'name':name,'command':command,'sampled_peak_rss_bytes':0,'rss_samples':0}
    t=time.monotonic();process=None;reason=None
    with (OUT/(name+'.log')).open('x') as log:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                if time.monotonic()>=deadline:reason='60_second_job_cap';break
                sampled=subprocess.run(['/bin/ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True,timeout=1)
                if sampled.returncode==0 and sampled.stdout.strip():
                    rss=int(sampled.stdout.strip())*1024;row['rss_samples']+=1
                    row['sampled_peak_rss_bytes']=max(row['sampled_peak_rss_bytes'],rss)
                    if rss>1024**3:reason='1_GiB_process_cap';break
                time.sleep(.03)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
            row['exit_code']=None if process is None else process.wait(timeout=3)
    row['elapsed_seconds']=time.monotonic()-t;row['stop_reason']=reason
    raw=(OUT/(name+'.log')).read_text();row['log_sha256']=hashlib.sha256(raw.encode()).hexdigest()
    try:row['result']=json.loads(raw.splitlines()[-1])
    except (ValueError,IndexError):row['result']=None
    receipt['runs'].append(row)
    if reason or row['exit_code']!=0:raise RuntimeError(name+' did not pass')
    return row['result']
try:
    result=run('recovery','control_bootstrap.py')
    if result['status']!='pass':raise RuntimeError('Generated control failure')
    receipt['status']='pass'
except BaseException as error:
    receipt['status']='failed';receipt['failure']={'type':type(error).__name__,'message':str(error)}
finally:
    receipt['elapsed_seconds']=time.monotonic()-started
    if receipt['elapsed_seconds']>=60:receipt['status']='failed';receipt['failure']={'type':'Deadline','message':'60s exceeded'}
    receipt['all_children_reaped']=True
    receipt['source_sha256']=pins
    receipt['source_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
    if not receipt['source_unchanged']:receipt['status']='failed'
    (OUT/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps(receipt,sort_keys=True))
sys.exit(0 if receipt['status']=='pass' else 1)
