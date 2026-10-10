"""One root-dispatched, owned, four-TRAIN initial inventory. No patient work by default."""
import argparse,json,os,signal,stat,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from batch_contract import CAPS,SUBJECTS,OUTPUT,guard,complete_result,sha,need

def tree_bytes(root):
    if not root.exists():return 0
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            path=Path(directory)/name
            try:info=path.lstat()
            except FileNotFoundError:continue
            if stat.S_ISLNK(info.st_mode):raise ValueError('Output symlink refused')
            if stat.S_ISREG(info.st_mode):total+=info.st_size
            elif not stat.S_ISDIR(info.st_mode):raise ValueError('Nonregular output refused')
    return total

def cleanup_phase(process,sampler,detached,actions,errors,cleanup_owned):
    """Keep the retained Popen cleanup path even if ownership sampling fails."""
    remaining=[]
    try:
        remaining=cleanup_owned(process,sampler,detached,actions,errors)
    except BaseException as error:
        errors.append('owned_cleanup_error:'+type(error).__name__+':'+str(error))
        try:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
                actions.append({'target':'owned_process_group_fallback','signal':int(signal.SIGKILL)})
        except BaseException as failure:
            errors.append('owned_group_fallback_error:'+type(failure).__name__+':'+str(failure))
        try:
            if process.poll() is None:
                process.kill()
                actions.append({'target':'retained_direct_child_fallback','signal':int(signal.SIGKILL)})
        except BaseException as failure:
            errors.append('direct_child_kill_error:'+type(failure).__name__+':'+str(failure))
        try:process.wait(timeout=2)
        except BaseException as failure:
            errors.append('direct_child_wait_error:'+type(failure).__name__+':'+str(failure))
        for pid,identity in detached.items():
            try:actions.append(sampler.signal_if_same_process(int(pid),identity,signal.SIGKILL))
            except BaseException as failure:errors.append('descendant_cleanup_error:'+str(failure))
        try:remaining=list(sampler.group(process.pid,process.pid)['pids'])
        except BaseException as failure:errors.append('final_owned_sample_error:'+str(failure))
        try:
            if process.poll() is None:remaining.append(process.pid)
        except BaseException as failure:
            errors.append('retained_child_poll_error:'+str(failure));remaining.append(process.pid)
    return sorted(set(remaining))

def final_extents(output,supervision):
    total,aux=tree_bytes(output),tree_bytes(supervision)
    need(total<=CAPS['output_bytes'],'final_output_cap')
    need(aux<=CAPS['supervision_bytes']-65536,'final_supervision_cap')
    need((supervision/'worker.log').stat().st_size<=CAPS['log_bytes'],'final_log_cap')
    return total,aux

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    release,index,canonical_ready=guard(a.release.resolve(),a.release_sha256,execute=not a.check_only)
    if a.check_only:
        print(json.dumps({'status':'metadata_contract_valid','execution_released':release['execution_released'],'canonical_promotions_ready':canonical_ready,'public_arrays_opened':0,'workers_started':0}));return 0
    # Source/helper validation and root dispatch precede monitor import and worker creation.
    sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
    from run_contact_owned import FastDarwinSampler,cleanup_owned,_receipt
    supervision=OUTPUT.with_name(OUTPUT.name+'.supervision')
    need(not OUTPUT.exists() and not OUTPUT.is_symlink() and not supervision.exists() and not supervision.is_symlink(),'one_attempt_exclusive_paths')
    supervision.mkdir(exist_ok=False)
    runtime=ROOT/'.venv/bin/python'
    command=[str(runtime),'-I','-B','-X','pycache_prefix='+str(supervision/'fresh-pycache'),str(HERE/'initial_inventory_worker.py'),'--declaration',str(a.release.resolve()),'--declaration-sha256',a.release_sha256]
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    for key in ('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','LD_PRELOAD','DYLD_LIBRARY_PATH'):env.pop(key,None)
    _receipt(supervision/'declaration.json',{'release_sha256':a.release_sha256,'source_index':release['source_index'],'public_index':release['public_index'],'head':release['expected_head'],'caps':CAPS,'command':command,'condition':release['occupancy_condition'],'training_admitted':False,'parent_pid':os.getpid()})
    started=time.monotonic();sampler=None;process=None;lease_read=lease_write=None
    peak=samples=0;reason=None;remaining=[];detached={};actions=[];errors=[];result={};result_sha=None;total=aux=None
    def stop(*_):raise InterruptedError('Owned initial-inventory parent interrupted')
    prior=signal.signal(signal.SIGTERM,stop)
    try:
        sampler=FastDarwinSampler();lease_read,lease_write=os.pipe();env['RESECTIONLAB_PARENT_LEASE_FD']=str(lease_read)
        with (supervision/'worker.log').open('x') as log:
            try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease_read,),start_new_session=True)
            finally:os.close(lease_read);lease_read=None
            while process.poll() is None:
                if time.monotonic()-started>=CAPS['parent_seconds']:reason='parent_wall_cap';break
                if os.fstat(log.fileno()).st_size>CAPS['log_bytes']:reason='log_cap';break
                measured=sampler.group(process.pid,process.pid);detached.update(measured['detached_descendant_start_identities']);samples+=1
                peak=max(peak,measured['process_group_resident_bytes'])
                if peak>CAPS['sampled_rss_bytes']:reason='sampled_memory_cap';break
                total,aux=tree_bytes(OUTPUT),tree_bytes(supervision)
                if total>CAPS['output_bytes'] or aux>CAPS['supervision_bytes']-65536:reason='output_cap';break
                _receipt(supervision/'progress.json',{'worker_pid':process.pid,'elapsed_seconds':time.monotonic()-started,'sampled_peak_rss_bytes':peak,'owned_process_measurement':measured,'output_bytes':total})
                time.sleep(.2)
    except BaseException as error:reason=reason or type(error).__name__+':'+str(error)
    finally:
        if process is not None:remaining=cleanup_phase(process,sampler,detached,actions,errors,cleanup_owned)
        for fd in (lease_read,lease_write):
            if fd is not None:
                try:os.close(fd)
                except OSError as error:errors.append('lease_close_error:'+str(error))
        code=None if process is None else process.poll()
        try:
            total,aux=final_extents(OUTPUT,supervision)
            guard(a.release.resolve(),a.release_sha256,execute=True)
            path=OUTPUT/'result.json';need(path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2,'bounded_terminal_result')
            result=json.loads(path.read_bytes());result_sha=sha(path);complete_result(result,release,a.release_sha256)
        except BaseException as error:reason=reason or 'terminal_guard:'+type(error).__name__+':'+str(error)
        elapsed=time.monotonic()-started
        if elapsed>=CAPS['parent_seconds']:reason=reason or 'parent_wall_cap'
        complete=reason is None and code==0 and samples>0 and not errors and not remaining
        receipt={'status':'complete' if complete else 'failed_or_unresolved','stop_reason':reason,'exit_code':code,'worker_pid':None if process is None else process.pid,'worker_termination_confirmed':code is not None,'elapsed_seconds':elapsed,'sampled_peak_rss_bytes':peak,'samples':samples,'caps':CAPS,'remaining_owned_pids':remaining,'cleanup_actions':actions,'cleanup_errors':errors,'output_bytes':total,'supervision_bytes_before_receipt':aux,'result_sha256':result_sha,'result_status':result.get('status'),'source_index':release['source_index'],'release_sha256':a.release_sha256,'training_admitted':False,'sampling_limit':'transient peaks or descendants between samples may be missed','planned_subjects':SUBJECTS,'automatic_retry':False}
        _receipt(supervision/'receipt.json',receipt)
        if tree_bytes(supervision)>CAPS['supervision_bytes']:
            complete=False;receipt.update(status='failed_or_unresolved',stop_reason='closed_supervision_output_cap');_receipt(supervision/'receipt.json',receipt)
        signal.signal(signal.SIGTERM,prior)
    return 0 if complete else 1
if __name__=='__main__':raise SystemExit(main())
