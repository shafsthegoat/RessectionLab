"""Thin fixed TRAIN rollout adaptation of the existing contact owned-process monitor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).parent.parent))
from run_contact_owned import ROOT, sha, cleanup_owned, FastDarwinSampler, _receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--source-index',type=Path,required=True)
    parser.add_argument('--source-index-sha256',required=True)
    args=parser.parse_args()
    if sha(args.source_index)!=args.source_index_sha256:raise ValueError('Rollout source index changed')
    index=json.loads(args.source_index.read_text())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=args.expected_head:raise ValueError('Canonical commit differs from released HEAD')
    for relative,expected in index['source_files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Rollout source changed: '+relative)
    input_index=ROOT/index['input_index']['path']
    if sha(input_index)!=index['input_index']['sha256']:raise ValueError('Rollout inputs index changed')
    input_record=json.loads(input_index.read_text())
    for row in input_record['data'].values():
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Pinned TRAIN rollout input changed: '+row['path'])
    checkpoint=input_record['checkpoint']
    if sha(ROOT/checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Released final checkpoint bytes changed')
    output=ROOT/'build/goal-conditioned-policy-v1/full-teacher-rollout-run-v1'
    if output.exists():raise FileExistsError('One rollout attempt only; no retry or overwrite')
    supervision=output.with_name(output.name+'.supervision');supervision.mkdir(exist_ok=False)
    caps={'hard_total_wall_seconds':45.,'sampled_owned_tree_rss_bytes':1073741824,
          'geometry_previews':2048,'threads':1,'sample_interval_seconds':.2,'attempts':1,'automatic_retry':False}
    worker=Path(__file__).with_name('rollout_worker.py')
    _receipt(supervision/'declaration.json',{'version':'owned-fixed24-TRAIN-rollout-v1','head':head,
        'source_index_path':str(args.source_index.resolve()),'source_index_sha256':args.source_index_sha256,
        'input_index_sha256':sha(input_index),'caps':caps,'scope':'fixed24_TRAIN_goals48forward_cap0updates',
        'output':str(output),'source_files':index['source_files']})
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    lease_read,lease_write=os.pipe();env['RESECTIONLAB_PARENT_LEASE_FD']=str(lease_read)
    command=[sys.executable,'-I','-B','-X','pycache_prefix='+str(supervision/'fresh-pycache'),str(worker),
        '--output',str(output),'--input-index',str(input_index),'--input-index-sha256',sha(input_index)]
    sampler=FastDarwinSampler();started=time.monotonic();peak=0;samples=0;reason=None;errors=[];process=None
    cleanup_actions=[];detached_seen={};final_owned_pids=[]
    try:
        with (supervision/'worker.log').open('x') as log:
            try:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(lease_read,),start_new_session=True)
            finally:os.close(lease_read)
            try:
                while process.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>=caps['hard_total_wall_seconds']:reason='hard_total_wall_cap';break
                    if os.fstat(log.fileno()).st_size>8*1024*1024:reason='worker_log_cap';break
                    measured=sampler.group(process.pid,process.pid);rss=measured['process_group_resident_bytes']
                    detached_seen.update(measured['detached_descendant_start_identities'])
                    samples+=1;peak=max(peak,rss)
                    if rss>caps['sampled_owned_tree_rss_bytes']:reason='sampled_owned_tree_rss_cap';break
                    _receipt(supervision/'progress.json',{'elapsed_seconds':elapsed,'worker_pid':process.pid,
                        'sampled_peak_rss_bytes':peak,'samples':samples,'owned_process_measurement':measured})
                    time.sleep(.2)
            finally:final_owned_pids=cleanup_owned(process,sampler,detached_seen,cleanup_actions,errors)
    except BaseException as error:
        reason=reason or type(error).__name__+':'+str(error);raise
    finally:
        os.close(lease_write)
        code=None if process is None else process.poll();result_path=output/'result.json'
        final_elapsed=time.monotonic()-started
        if final_elapsed>=caps['hard_total_wall_seconds']:reason=reason or 'hard_total_wall_cap'
        complete=reason is None and code==0 and not errors and result_path.is_file()
        _receipt(supervision/'receipt.json',{'status':'complete' if complete else 'failed',
            'stop_reason':reason,'exit_code':code,'cleanup_errors':errors,'cleanup_actions':cleanup_actions,
            'final_owned_pids':final_owned_pids,'worker_termination_confirmed':code is not None,
            'elapsed_seconds':final_elapsed,'sampled_peak_rss_bytes':peak,'samples':samples,
            'sampling_limit':'transient peaks or detached descendants between samples may be missed',
            'worker_process_scope':'owned process group plus discovered descendants via FastDarwinSampler',
            'result_sha256':sha(result_path) if result_path.is_file() else None,'caps':caps,'automatic_retry':False})
    if not complete:raise RuntimeError('Owned TRAIN rollout did not complete: '+str(reason or code))


if __name__=='__main__':main()
