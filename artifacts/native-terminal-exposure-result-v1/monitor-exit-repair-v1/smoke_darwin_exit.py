"""Root-dispatched one benign Darwin child, <=5s; no model/project imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
SAMPLER=ROOT/'build/goal-conditioned-policy-v1/darwin_fast_sampler.py'
SAMPLER_SHA='906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c'
HELPER_SHA='aff4b197ea033aef8b160f43eabdd5a03394ae4024b6fc2f59f5241f2fcbfe14'
CHILD='import os,sys; os.read(int(sys.argv[1]),1)'


def source(path,expected):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Exact smoke dependency required: '+str(path))
    namespace={'__name__':path.stem,'__file__':str(path)}
    exec(compile(raw,str(path),'exec'),namespace)
    return namespace


def main():
    if sys.platform!='darwin':raise RuntimeError('This smoke must run on the actual Darwin host')
    started=time.monotonic();deadline=started+5.
    output=HERE/'darwin-smoke-01';output.mkdir(exist_ok=False)
    record={'schema':'owned-child-exit-darwin-smoke-v1','status':'running','automatic_retry':False,
        'planned_children':1,'wall_limit_seconds':5,'sampler_sha256':SAMPLER_SHA,'helper_sha256':HELPER_SHA,
        'child_source':CHILD,'child_source_sha256':hashlib.sha256(CHILD.encode()).hexdigest(),
        'scope':'One blocked-on-pipe benign Python child; no native/model/patient imports','phases':[],'errors':[]}
    (output/'intent.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    process=owned=None;read_fd=write_fd=None
    try:
        Sampler=source(SAMPLER,SAMPLER_SHA)['FastDarwinSampler']
        Owned=source(HERE/'owned_child_exit.py',HELPER_SHA)['OwnedChildExit']
        sampler=Sampler();read_fd,write_fd=os.pipe()
        process=subprocess.Popen([sys.executable,'-I','-B','-c',CHILD,str(read_fd)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            pass_fds=(read_fd,),start_new_session=True)
        os.close(read_fd);read_fd=None
        record['child_pid']=process.pid
        owned=Owned(process,sampler)
        observed=owned.sample()
        record['phases'].append({'phase':'first_real_live_sample','elapsed_seconds':time.monotonic()-started,
                                 'observation':observed})
        if observed['state']!='running' or not observed['rss_observed']:
            raise RuntimeError('Gate-held child must have an actual live RSS observation')
        os.write(write_fd,b'x');os.close(write_fd);write_fd=None
        # Deterministically put the child's real exit between two parent samples.
        # The second sample and cleanup must never ask the old sampler for its
        # dead-root RSS; cleanup still performs actual libproc group enumeration.
        code=process.wait(timeout=min(.5,max(0.,deadline-time.monotonic())))
        record['phases'].append({'phase':'direct_child_exit_between_samples','pid':process.pid,
                                 'wait_returncode':code,'elapsed_seconds':time.monotonic()-started})
        observed=owned.sample()
        record['phases'].append({'phase':'second_real_sample_after_exit','observation':observed})
        if code!=0 or observed['state']!='exited' or observed['rss_observed']:
            raise RuntimeError('Exit0 must remain a reap event, not a zero-RSS observation')
    except BaseException as error:
        record['errors'].append({'phase':'smoke','exception':type(error).__name__,
                                 'errno':getattr(error,'errno',None),'message':str(error),
                                 'pid':None if process is None else process.pid})
    finally:
        for fd in (read_fd,write_fd):
            if fd is not None:
                try:os.close(fd)
                except OSError as error:record['errors'].append({'phase':'pipe.close','errno':error.errno,'message':str(error)})
        if owned is not None:
            record['lifecycle']=owned.cleanup(deadline=deadline)
        elif process is not None:
            try:
                if process.poll() is None:process.kill()
                process.wait(timeout=min(.25,max(0.,deadline-time.monotonic())))
            except BaseException as error:record['errors'].append({'phase':'uninitialized.direct_fallback','errno':getattr(error,'errno',None),'message':str(error)})
        record['elapsed_seconds']=time.monotonic()-started
        lifecycle=record.get('lifecycle',{})
        passed=(not record['errors'] and lifecycle.get('verified') is True and lifecycle.get('child_reaped') is True
                and lifecycle.get('exit_code')==0 and lifecycle.get('remaining_owned_pids')==[]
                and record['elapsed_seconds']<=5. and len(record['phases'])==3)
        record['status']='passed' if passed else 'failed'
        record['smoke_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        raw=(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if len(raw)>32768:raise RuntimeError('Small smoke receipt cap')
        with (output/'receipt.json').open('xb') as stream:stream.write(raw)
        print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds'],
                          'receipt_sha256':hashlib.sha256(raw).hexdigest(),'child_pid':record.get('child_pid')}))
    return 0 if record['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
