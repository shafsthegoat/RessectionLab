"""One owned canonical smoke/pilot child, using the existing cleanup/lease pattern.

Source preparation is not an execution release. No automatic retry. The pilot
has cooperative stage caps plus this hard total wall and sampled owned tree RSS.
The existing Darwin sampler includes process-group members and descendants.
"""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from resectionlab.legacy_transfer_supervisor import _receipt
from darwin_fast_sampler import FastDarwinSampler


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cleanup_owned(process, sampler, detached_seen, actions, errors):
    """Reuse the owned-group/identity-matched descendant cleanup pattern."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is None:
            try:
                os.killpg(process.pid, sig)
                actions.append({'target':'owned_process_group','signal':int(sig)})
            except ProcessLookupError: pass
        remaining = sampler.group(process.pid, process.pid)
        identities = dict(detached_seen)
        identities.update(remaining['detached_descendant_start_identities'])
        for pid in remaining['pids']:
            identity = sampler.process_start_identity(pid)
            if identity is not None: identities[str(pid)] = identity
        detached_seen.update(identities)
        for pid, identity in identities.items():
            actions.append(sampler.signal_if_same_process(int(pid), identity, sig))
        try: process.wait(timeout=2)
        except subprocess.TimeoutExpired: pass
        time.sleep(.05)
    remaining = sampler.group(process.pid, process.pid)
    unresolved = set(remaining['pids'])
    for pid, identity in detached_seen.items():
        if sampler.process_start_identity(int(pid)) == tuple(identity): unresolved.add(int(pid))
    if unresolved: errors.append('owned_processes_remain:'+','.join(map(str,sorted(unresolved))))
    return sorted(unresolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke','pilot'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--released-experiment-hash')
    args = parser.parse_args()
    if args.mode == 'pilot' and not args.released_experiment_hash:
        parser.error('Pilot requires its separately released experiment hash')
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True, timeout=5).strip()
    if head != args.expected_head: raise ValueError('Canonical source commit differs from released HEAD')
    index_path = Path(__file__).with_name('learning-promotion.json')
    index = json.loads(index_path.read_text())
    source_bindings = {}
    for row in index['files']:
        path = ROOT/row['destination']
        if sha(path) != row['sha256']: raise ValueError('Canonical promoted source changed: '+row['destination'])
        source_bindings[row['destination']] = row['sha256']
    for relative, expected in index['dependencies'].items():
        if sha(ROOT/relative) != expected: raise ValueError('Canonical dependency changed: '+relative)
        source_bindings[relative] = expected
    output = args.output.resolve()
    if output.exists(): raise FileExistsError('Attempt output already exists; no retry/overwrite')
    if args.mode == 'pilot' and output != ROOT/'outputs/learning/public-contact-v1/attempt-01':
        raise ValueError('Pilot output must be the root-fixed one-attempt destination')
    supervision = output.with_name(output.name+'.supervision')
    supervision.mkdir(parents=True, exist_ok=False)
    caps = {'hard_total_wall_seconds': 45. if args.mode == 'smoke' else 1500.,
        'sampled_owned_tree_rss_bytes': 1073741824, 'sample_interval_seconds': .2,
        'teacher_stage_seconds': 4. if args.mode == 'smoke' else 120.,
        'learning_method_seconds': None if args.mode == 'smoke' else 60.,
        'online_method_episode_seconds': None if args.mode == 'smoke' else 12.,
        'stage_limit_enforcement': 'cooperative task checks; parent hard total wall and sampled owned-tree RSS',
        'threads': 1, 'automatic_retry': False}
    worker = Path(__file__).with_name('contact_owned_worker.py')
    _receipt(supervision/'declaration.json', {'version':'owned-contact-learning-attempt-v1',
        'mode':args.mode,'head':head,'caps':caps,'source_bindings':source_bindings,
        'worker_sha256':sha(worker),'supervisor_sha256':sha(__file__),
        'darwin_sampler_sha256':sha(Path(__file__).with_name('darwin_fast_sampler.py')),
        'source_index_sha256':sha(index_path),'output':str(output),
        'released_experiment_hash':args.released_experiment_hash,
        'scope':'separate_smoke_not_pilot' if args.mode=='smoke' else 'fixed_generated_pilot'})
    env = {**os.environ, 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    lease_read, lease_write = os.pipe(); env['RESECTIONLAB_PARENT_LEASE_FD'] = str(lease_read)
    command = [sys.executable,'-I','-B','-X','pycache_prefix='+str(supervision/'fresh-pycache'),
        str(worker),'--mode',args.mode,'--output',str(output)]
    if args.released_experiment_hash: command += ['--released-experiment-hash',args.released_experiment_hash]
    sampler=FastDarwinSampler()
    started=time.monotonic(); peak=0; samples=0; reason=None; errors=[]; process=None
    cleanup_actions=[]; detached_seen={}; final_owned_pids=[]
    try:
        with (supervision/'worker.log').open('x') as log:
            try:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(lease_read,),start_new_session=True)
            finally:
                os.close(lease_read)
            try:
                while process.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>=caps['hard_total_wall_seconds']:
                        reason='hard_total_wall_cap';break
                    if os.fstat(log.fileno()).st_size>8*1024*1024:
                        reason='worker_log_cap';break
                    measured=sampler.group(process.pid,process.pid)
                    rss=measured['process_group_resident_bytes']
                    detached_seen.update(measured['detached_descendant_start_identities'])
                    samples+=1;peak=max(peak,rss)
                    if rss>caps['sampled_owned_tree_rss_bytes']:
                        reason='sampled_owned_tree_rss_cap';break
                    _receipt(supervision/'progress.json',{'elapsed_seconds':elapsed,
                        'worker_pid':process.pid,'sampled_peak_rss_bytes':peak,'samples':samples,
                        'owned_process_measurement':measured})
                    time.sleep(.2)
            finally:
                final_owned_pids=cleanup_owned(process,sampler,detached_seen,cleanup_actions,errors)
    except BaseException as error:
        reason=reason or type(error).__name__+':'+str(error)
        raise
    finally:
        os.close(lease_write)
        code=None if process is None else process.poll()
        result_path=output/'result.json'
        complete=reason is None and code==0 and not errors and result_path.is_file()
        _receipt(supervision/'receipt.json',{'status':'complete' if complete else 'failed',
            'mode':args.mode,'stop_reason':reason,'exit_code':code,'cleanup_errors':errors,
            'worker_termination_confirmed':code is not None,'elapsed_seconds':time.monotonic()-started,
            'cleanup_actions':cleanup_actions,'final_owned_pids':final_owned_pids,
            'sampled_peak_rss_bytes':peak,'samples':samples,
            'sampling_limit':'transient peaks or detached descendants between samples may be missed',
            'worker_process_scope':'owned process group plus discovered descendants via FastDarwinSampler',
            'completion_scope':'worker return is separate from whether result reached fixed pilot endpoint',
            'result_sha256':sha(result_path) if result_path.is_file() else None,
            'caps':caps,'automatic_retry':False})
    if not complete: raise RuntimeError('Owned contact attempt did not complete: '+str(reason or code))


if __name__=='__main__':main()
