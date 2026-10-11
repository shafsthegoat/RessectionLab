"""One root-dispatched exposure matrix using the previous owned lifecycle."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_bytes(root, *, monitor_owned_temporaries=False):
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(directory)/name
            try:s=p.lstat()
            except FileNotFoundError as error:
                # The two worker JSON writers replace these exact top-level
                # temporary files atomically. A live sample may list one just
                # before replacement. Other errors/paths remain fatal, and
                # every source/final scan remains strict after worker reap.
                if (error.errno==errno.ENOENT and monitor_owned_temporaries and Path(directory)==Path(root)
                        and name in files and name in {'report.json.tmp','progress.json.tmp'}):
                    continue
                raise
            if stat.S_ISLNK(s.st_mode):raise ValueError('Linked output/source refused')
            if stat.S_ISREG(s.st_mode):total+=s.st_size
            elif not stat.S_ISDIR(s.st_mode):raise ValueError('Nonregular output/source refused')
    return total


def guard(index):
    for name,digest in index['files'].items():
        p=ROOT/name
        if p.is_symlink() or not p.is_file() or sha(p)!=digest:raise ValueError('Bound source/reference changed: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    path=HERE/'source-index.json'
    if sha(path)!=a.source_index_sha256:raise ValueError('Exact source index required')
    index=json.loads(path.read_bytes());guard(index)
    spec=json.loads((HERE/'config.json').read_bytes());caps=spec['caps']
    if not a.execute:print(json.dumps({'status':'source_only_checked','native_calls':0}));return 0
    if index['execution_released'] is not True:raise ValueError('Root source freeze must release execution')
    output=ROOT/spec['output_directory']
    if output.exists() or output.is_symlink():raise FileExistsError('Fresh one-attempt output required')
    if tree_bytes(HERE)>caps['source_bytes']:raise ValueError('Source reservation cap')
    if shutil.disk_usage(ROOT).free<50*1024**3+caps['output_bytes']:raise ValueError('50GiB floor plus output reserve')
    sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
    from run_contact_owned import FastDarwinSampler,cleanup_owned,_receipt
    output.mkdir(parents=True);started=time.monotonic();sampler=FastDarwinSampler()
    record={'schema':'owned_same_capsule_exposure_matrix_v1','status':'running','source_index_sha256':a.source_index_sha256,
        'caps':caps,'planned_rows':18,'planned_action_slots':180,'scope':spec['scope'],'automatic_retry':False,
        'sampled_peak_group_rss_bytes':0,'rss_samples':0}
    _receipt(output/'intent.json',record)
    process=None;reason=None;errors=[];actions=[];detached={};remaining=[];lease_read=lease_write=None;previous={};stop={'requested':False}
    def interrupted(signum,frame):stop['requested']=True;raise KeyboardInterrupt('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,interrupted)
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','DYLD_','LD_'))};env.update(THREADS,PYTHONDONTWRITEBYTECODE='1')
    try:
        cache=output/'absent-worker-cache'
        command=[sys.executable,'-I','-B','-X','pycache_prefix='+str(cache),str(HERE/'probe_exposure.py'),
            '--worker','--output',str(output),'--source-index-sha256',a.source_index_sha256]
        record['command']=command;lease_read,lease_write=os.pipe();env['RESECTIONLAB_PARENT_LEASE_FD']=str(lease_read)
        with (output/'worker.log').open('x') as log:
            try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease_read,),start_new_session=True)
            finally:os.close(lease_read);lease_read=None
            while process.poll() is None:
                if time.monotonic()-started>=caps['cooperative_worker_seconds']:reason='work_deadline_with_cleanup_reserve';break
                measured=sampler.group(process.pid,process.pid);detached.update(measured['detached_descendant_start_identities'])
                record['rss_samples']+=1;record['sampled_peak_group_rss_bytes']=max(record['sampled_peak_group_rss_bytes'],measured['process_group_resident_bytes'])
                if not measured['pids'] and process.poll() is None:reason='live_group_rss_unknown';break
                if measured['process_group_resident_bytes']>caps['worker_peak_rss_bytes']:reason='rss_cap';break
                if os.fstat(log.fileno()).st_size>64*1024:reason='log_cap';break
                if tree_bytes(output,monitor_owned_temporaries=True)>caps['output_bytes']-64*1024:reason='output_cap';break
                if shutil.disk_usage(ROOT).free<50*1024**3:reason='free_disk_floor';break
                time.sleep(.2)
    except BaseException as error:reason=reason or type(error).__name__+':'+str(error)
    finally:
        try:
            if process is not None:
                try:remaining=cleanup_owned(process,sampler,detached,actions,errors)
                except BaseException as error:
                    errors.append('cleanup:'+str(error))
                    try:
                        if process.poll() is None:process.kill()
                        process.wait(timeout=2)
                    except BaseException as final:errors.append('direct_child_cleanup:'+str(final))
                    try:remaining=sampler.group(process.pid,process.pid)['pids']
                    except BaseException as final:errors.append('remaining:'+str(final))
            for fd in (lease_read,lease_write):
                if fd is not None:
                    try:os.close(fd)
                    except OSError as error:errors.append('lease_close:'+str(error))
            record.update(worker_exit_code=None if process is None else process.poll(),stop_reason=reason,cleanup_actions=actions,
                cleanup_errors=errors,remaining_owned_pids=remaining,child_reaped=process is not None and process.poll() is not None,
                elapsed_seconds=time.monotonic()-started)
            accepted=False
            try:
                guard(index);report=json.loads((output/'report.json').read_bytes())
                accepted=(process is not None and process.returncode==0 and reason is None and not errors and not remaining and not stop['requested']
                    and report['status']=='completed' and isinstance(report.get('useful_criterion'),dict)
                    and report['source_unchanged'] and report['completed_rows']==18
                    and len(report['rows'])==18 and not report['missing_rows'] and record['elapsed_seconds']<caps['parent_wall_seconds']
                    and tree_bytes(output)<=caps['output_bytes']-16384)
                record['report_sha256']=sha(output/'report.json');record['useful_criterion']=report.get('useful_criterion')
                record['worker_report_status']=report.get('status')
                progress=output/'progress.json'
                if progress.exists():
                    if progress.stat().st_size>64*1024:raise ValueError('Compact progress receipt cap')
                    record['last_saved_progress']=json.loads(progress.read_bytes())
            except BaseException as error:
                accepted=False
                record['finalization_error']=type(error).__name__+':'+str(error)
            record['status']='completed' if accepted else 'incomplete_or_failed'
            record['output_bytes_before_receipt']=tree_bytes(output);_receipt(output/'launcher.json',record)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds'],'stop_reason':reason}))
    return 0 if record['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
