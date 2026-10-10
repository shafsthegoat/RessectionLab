"""One sequential baseline/stage generated benchmark; never an execution release."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
from run_contact_owned import FastDarwinSampler,cleanup_owned,_receipt


def main():
    output=HERE/'generated-benchmark-v1';output.mkdir(exist_ok=False)
    paths=[HERE/'generated_benchmark.py',Path(__file__),*HERE.glob('*/resectionlab/*.py'),*(ROOT/'src/resectionlab').rglob('*.py')]
    _receipt(output/'declaration.json',{'scope':'two sequential generated251x331x250 arms; no patient arrays/models',
        'caps':{'wall_seconds':60,'per_arm_seconds':25,'rss_bytes':2*1024**3,'threads':1,'previews_per_arm':64,'commits_per_arm':16},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'automatic_retry':False})
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
    for k in ('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','LD_PRELOAD','DYLD_LIBRARY_PATH'):env.pop(k,None)
    sampler=FastDarwinSampler();start=time.monotonic();outcomes=[];reason=None
    def stop(*args):raise InterruptedError('generated benchmark parent terminated')
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        for variant in ('baseline','stage'):
            read,write=os.pipe();env['RESECTIONLAB_PARENT_LEASE_FD']=str(read)
            process=None;peak=0;actions=[];errors=[];detached={};remaining=[];arm_start=time.monotonic()
            try:
                with (output/(variant+'.log')).open('x') as log:
                    try:process=subprocess.Popen([sys.executable,'-I','-B','-X','pycache_prefix='+str(output/'fresh-pycache'),str(HERE/'generated_benchmark.py'),'--variant',variant,'--output',str(output/(variant+'.json'))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(read,),start_new_session=True)
                    finally:os.close(read)
                    while process.poll() is None:
                        if time.monotonic()-start>=60:reason='aggregate_wall_cap';break
                        measured=sampler.group(process.pid,process.pid);detached.update(measured['detached_descendant_start_identities'])
                        peak=max(peak,measured['process_group_resident_bytes'])
                        if peak>2*1024**3:reason='sampled_memory_cap';break
                        time.sleep(.1)
            finally:
                if process is not None:remaining=cleanup_owned(process,sampler,detached,actions,errors)
                os.close(write)
                outcomes.append({'variant':variant,'elapsed_seconds':time.monotonic()-arm_start,'sampled_peak_rss_bytes':peak,
                    'exit_code':None if process is None else process.poll(),'cleanup_actions':actions,'cleanup_errors':errors,'remaining_owned_pids':remaining,
                    'reaped':process is not None and process.poll() is not None})
            if reason is not None or outcomes[-1]['exit_code']!=0 or errors or remaining:break
    except BaseException as error:reason=type(error).__name__+':'+str(error)
    finally:
        _receipt(output/'receipt.json',{'arms':outcomes,'elapsed_seconds':time.monotonic()-start,'stop_reason':reason,
            'automatic_retry':False,'memory_limit':'sampled, not kernel hard'})
        signal.signal(signal.SIGTERM,previous)
    if reason is not None or len(outcomes)!=2 or any(a['exit_code']!=0 or a['cleanup_errors'] or a['remaining_owned_pids'] for a in outcomes):return 1
    baseline,stage=[json.loads((output/(v+'.json')).read_text()) for v in ('baseline','stage')]
    parity={k:baseline[k]==stage[k] for k in ('source_hash','model_hash')}
    parity['committed_signatures']=baseline['branch_commits']['signatures']==stage['branch_commits']['signatures']
    _receipt(output/'comparison.json',{'exact_parity':parity,'pass':all(parity.values()),
        'scope':'generated source cardinality only; no patient runtime/memory extrapolation',
        'baseline_result_sha256':hashlib.sha256((output/'baseline.json').read_bytes()).hexdigest(),
        'stage_result_sha256':hashlib.sha256((output/'stage.json').read_bytes()).hexdigest()})
    return 0 if all(parity.values()) else 1


if __name__=='__main__':raise SystemExit(main())
