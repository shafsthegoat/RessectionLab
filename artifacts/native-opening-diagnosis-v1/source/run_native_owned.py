"""Single60s/2GiB generated control, existing owned cleanup; no retry."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
from run_contact_owned import FastDarwinSampler,cleanup_owned,_receipt


def main():
    output=HERE/'native-run-v1';supervision=HERE/'native-run-v1.supervision'
    if output.exists() or supervision.exists():raise FileExistsError('one attempt only')
    supervision.mkdir();sources=[HERE/'native_saturation_worker.py',Path(__file__),HERE/'result.json']
    sources+=list((ROOT/'src/resectionlab').rglob('*.py'))
    _receipt(supervision/'declaration.json',{'scope':'root-authorized generated native monotone opening saturation',
        'caps':{'wall_seconds':60,'worker_seconds':55,'sampled_rss_bytes':2*1024**3,'previews':3000,'commits':256,'threads':1},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'attempts':1,'automatic_retry':False})
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
    for k in ('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','LD_PRELOAD','DYLD_LIBRARY_PATH'):env.pop(k,None)
    read,write=os.pipe();env['RESECTIONLAB_PARENT_LEASE_FD']=str(read)
    sampler=FastDarwinSampler();started=time.monotonic();peak=0;reason=None;actions=[];errors=[];detached={};remaining=[];process=None
    def stop(signum,frame):raise InterruptedError('parent terminated')
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        with (supervision/'worker.log').open('x') as log:
            try:process=subprocess.Popen([sys.executable,'-I','-B','-X','pycache_prefix='+str(supervision/'fresh-pycache'),str(HERE/'native_saturation_worker.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(read,),start_new_session=True)
            finally:os.close(read)
            try:
                while process.poll() is None:
                    if time.monotonic()-started>=60:reason='parent_wall_cap';break
                    measurement=sampler.group(process.pid,process.pid);detached.update(measurement['detached_descendant_start_identities'])
                    peak=max(peak,measurement['process_group_resident_bytes'])
                    if peak>2*1024**3:reason='parent_sampled_memory_cap';break
                    time.sleep(.1)
            finally:remaining=cleanup_owned(process,sampler,detached,actions,errors)
    except BaseException as error:reason=type(error).__name__+':'+str(error)
    finally:
        os.close(write)
        _receipt(supervision/'receipt.json',{'exit_code':None if process is None else process.poll(),'stop_reason':reason,
            'elapsed_seconds':time.monotonic()-started,'sampled_peak_rss_bytes':peak,'cleanup_actions':actions,
            'cleanup_errors':errors,'remaining_owned_pids':remaining,'reaped':process is not None and process.poll() is not None,
            'result_sha256':hashlib.sha256((output/'result.json').read_bytes()).hexdigest() if (output/'result.json').exists() else None,
            'memory_limit':'sampled, not kernel hard','automatic_retry':False})
        signal.signal(signal.SIGTERM,previous)
    return 0 if reason is None and process is not None and process.returncode==0 and not errors and not remaining else 1


if __name__=='__main__':raise SystemExit(main())
