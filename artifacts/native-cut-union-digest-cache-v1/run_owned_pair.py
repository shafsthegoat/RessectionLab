"""Root-dispatched tests then one fixed-row baseline/candidate profile pair.

Reuses existing FastDarwinSampler/cleanup_owned and durable _receipt. One total
120s deadline, sequential fresh process groups, 2GiB sampled group RSS, one
thread, no retries. Timing includes tests and supervision; only successful
candidate tests permit either profile. No acquired inputs or model execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
WALL_SECONDS=120.
EXECUTION_SECONDS=114.  # Reserve six seconds inside the total cap for owned cleanup.
MEMORY_BYTES=2*1024**3
OUTPUT_BYTES=96*1024
STAGE_BYTES=200*1024
THREAD_ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def tree_bytes(root):
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            path=Path(directory)/name;info=path.lstat()
            if stat.S_ISLNK(info.st_mode):raise ValueError('Output/stage symlink refused')
            if stat.S_ISREG(info.st_mode):total+=info.st_size
            elif not stat.S_ISDIR(info.st_mode):raise ValueError('Nonregular output/stage refused')
    return total


def guard(index):
    for rel,expected in index['files'].items():
        path=ROOT/rel
        if path.is_symlink() or not path.is_file() or sha(path)!=expected:raise ValueError('Bound source changed: '+rel)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index',type=Path,required=True)
    parser.add_argument('--source-index-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();index_path=args.source_index.resolve()
    if sha(index_path)!=args.source_index_sha256:raise ValueError('Wrong dispatch index')
    index=json.loads(index_path.read_text());guard(index)
    output=args.output.resolve()
    if output.parent!=HERE or output.exists() or output.is_symlink():raise ValueError('Fresh direct child output directory required')
    if tree_bytes(HERE)>STAGE_BYTES-OUTPUT_BYTES:raise ValueError('Insufficient reserved output headroom')
    sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
    from run_contact_owned import FastDarwinSampler,cleanup_owned,_receipt
    output.mkdir(exist_ok=False)
    started=time.monotonic();sampler=FastDarwinSampler();phases=[];failure=None;comparison=None
    record={'schema':'owned_one_row_union_digest_cache_pair_v1','status':'running',
        'source_index_sha256':args.source_index_sha256,'phase_order':['tests','baseline','candidate'],
        'caps':{'total_seconds':WALL_SECONDS,'execution_seconds':EXECUTION_SECONDS,'cleanup_reserve_seconds':6.,'sampled_group_rss_bytes':MEMORY_BYTES,'output_bytes':OUTPUT_BYTES,'total_stage_bytes':STAGE_BYTES,'threads':1},
        'comparison_scope':'One instrumented ordered fixed-row pair; not a production speedup estimate',
        'acquired_inputs':False,'model_calls':0,'phases':phases}
    _receipt(output/'declaration.json',record)
    env={**os.environ,**THREAD_ENV,'PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}
    try:
        for phase in record['phase_order']:
            guard(index)
            if time.monotonic()-started>=EXECUTION_SECONDS:raise TimeoutError('Execution deadline before '+phase)
            errors=[];actions=[];detached={};remaining=[];process=None;reason=None;peak=samples=0
            lease_read=lease_write=None;phase_start=time.monotonic()
            command=[sys.executable,'-I','-B',str(HERE/'dispatch_worker.py'),'--phase',phase,'--output',str(output/(phase+'.json'))]
            try:
                lease_read,lease_write=os.pipe();childenv={**env,'RESECTIONLAB_PARENT_LEASE_FD':str(lease_read)}
                with (output/(phase+'.log')).open('x') as log:
                    try:process=subprocess.Popen(command,cwd=ROOT,env=childenv,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease_read,),start_new_session=True)
                    finally:os.close(lease_read);lease_read=None
                    while process.poll() is None:
                        elapsed=time.monotonic()-started
                        if elapsed>=EXECUTION_SECONDS:reason='execution_wall_cap_with_cleanup_reserve';break
                        if phase=='tests' and time.monotonic()-phase_start>=45:reason='generated_test_45s_cap';break
                        measured=sampler.group(process.pid,process.pid)
                        detached.update(measured['detached_descendant_start_identities'])
                        samples+=1;peak=max(peak,measured['process_group_resident_bytes'])
                        if measured['process_group_resident_bytes']>MEMORY_BYTES:reason='sampled_group_rss_cap';break
                        if os.fstat(log.fileno()).st_size>16*1024:reason='phase_log_cap';break
                        if tree_bytes(output)>OUTPUT_BYTES-8192 or tree_bytes(HERE)>STAGE_BYTES-8192:reason='output_or_stage_cap';break
                        time.sleep(.2)
            except BaseException as error:reason=reason or type(error).__name__+':'+str(error)
            finally:
                if process is not None:
                    try:remaining=cleanup_owned(process,sampler,detached,actions,errors)
                    except BaseException as error:
                        errors.append('cleanup:'+str(error))
                        try:
                            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
                            process.wait(timeout=2)
                        except BaseException as final:errors.append('cleanup_fallback:'+str(final))
                        try:remaining=sampler.group(process.pid,process.pid)['pids']
                        except BaseException as final:errors.append('remaining:'+str(final))
                for fd in (lease_read,lease_write):
                    if fd is not None:
                        try:os.close(fd)
                        except OSError as error:errors.append('lease_close:'+str(error))
            phases.append({'phase':phase,'command':command,'elapsed_seconds':time.monotonic()-phase_start,
                'returncode':None if process is None else process.poll(),'stop_reason':reason,'sampled_peak_rss_bytes':peak,
                'samples':samples,'cleanup_actions':actions,'cleanup_errors':errors,'remaining_owned_pids':remaining,
                'child_reaped':process is not None and process.poll() is not None,
                'log_sha256':sha(output/(phase+'.log')) if (output/(phase+'.log')).exists() else None})
            _receipt(output/'progress.json',{'elapsed_seconds':time.monotonic()-started,'phases':phases})
            if process is None or process.returncode!=0 or reason or errors or remaining:raise RuntimeError('Incomplete phase: '+phase)
        b=json.loads((output/'baseline.json').read_text());c=json.loads((output/'candidate.json').read_text())
        if b['variant']!='baseline' or c['variant']!='candidate':raise ValueError('Variant binding')
        for key in ('schema','row_id','retained_report_sha256','projection','projection_sha256','exact_full_history_sha256','independent_certificate','retained_native_and_work_exact'):
            if b[key]!=c[key]:raise AssertionError('Exact paired mismatch: '+key)
        if b['retained_native_and_work_exact'] is not True:raise ValueError('Historical parity not proven')
        comparison={'exact_semantic_parity':True,'row_id':b['row_id'],
            'baseline_native_seconds':b['native_seconds'],'candidate_native_seconds':c['native_seconds'],
            'baseline_digest_json_calls':b['timers']['ledger_canonical_json_freeze_thaw_hash']['calls'],
            'candidate_digest_json_calls':c['timers']['ledger_canonical_json_freeze_thaw_hash']['calls'],
            'baseline_result_sha256':sha(output/'baseline.json'),'candidate_result_sha256':sha(output/'candidate.json'),
            'full_history_sha256':b['exact_full_history_sha256']}
        guard(index)
        if time.monotonic()-started>WALL_SECONDS:raise TimeoutError('Total deadline at finalization')
    except BaseException as error:failure=type(error).__name__+':'+str(error)
    finally:
        record.update(status='complete' if failure is None else 'failed_or_unresolved',failure=failure,
            elapsed_seconds=time.monotonic()-started,comparison=comparison,
            sampled_peak_group_rss_bytes=max((p['sampled_peak_rss_bytes'] for p in phases),default=0),
            cleanup_errors=[e for p in phases for e in p['cleanup_errors']],remaining_owned_pids=[pid for p in phases for pid in p['remaining_owned_pids']],
            output_bytes_before_receipt=tree_bytes(output),stage_bytes_before_receipt=tree_bytes(HERE))
        _receipt(output/'receipt.json',record)
    print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds'],'failure':failure,'receipt_sha256':sha(output/'receipt.json')}))
    return int(failure is not None)

if __name__=='__main__':raise SystemExit(main())
