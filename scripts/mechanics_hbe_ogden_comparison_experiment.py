"""One owned, serial shape comparison. False release performs source preflight only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

CORE='scripts/mechanics_hbe_ogden_comparison.py'
RUNNER='scripts/mechanics_hbe_ogden_comparison_experiment.py'
HELPER='scripts/mechanics_hbe_v5_fitted_confirmation_experiment.py'
FREE_DISK_FLOOR_BYTES=50*1024**3


def check_free_disk(core,directory,*,reserve_outputs=False):
    """Preserve the user's free-space floor, including all reserved new outputs."""
    import shutil
    required=FREE_DISK_FLOOR_BYTES+(core.CAPS['new_output_bytes'] if reserve_outputs else 0)
    core.need(shutil.disk_usage(directory).free>=required,'50 GiB free disk floor')


def bounded(path,maximum,expected):
    """Bootstrap-only same-byte admission, copied from the reviewed bounded reader."""
    p=Path(path)
    if '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):
        raise ValueError('Linked/escaping bootstrap input')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or not 0<a.st_size<=maximum:raise ValueError('Bounded regular bootstrap input')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as f:raw=f.read(maximum+1)
        b=os.fstat(fd);identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        if identity(a)!=identity(b) or identity(b)!=identity(p.lstat()) or len(raw)!=a.st_size:
            raise ValueError('Bootstrap input changed')
    finally:os.close(fd)
    if not isinstance(expected,str) or len(expected)!=64 or hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Bootstrap SHA differs')
    return raw


def bootstrap(root,release_path,digest,*,executing):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Require -I -S -B and fresh absent -X pycache_prefix')
    relative=Path(release_path)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Relative release path required')
    raw=bounded(root/relative,128*1024,digest)
    preliminary=json.loads(raw)
    sources=preliminary.get('source_bindings',{})
    if not all(name in sources for name in (CORE,RUNNER,HELPER)):raise ValueError('Missing source closure')
    for name,expected in sources.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('scripts','launchers') or p.suffix!='.py':
            raise ValueError('Source closure path scope')
        bounded(root/p,1024**2,expected)
    if Path(__file__).resolve()!=root/RUNNER:raise ValueError('Canonical runner required')
    # -B plus the checked fresh cache prefix prevents pre-existing local pyc admission.
    cache=Path(sys.pycache_prefix or '')
    if (not cache.is_absolute() or '..' in cache.parts or not cache.is_relative_to(root/'build')
            or cache.exists() or any(x.is_symlink() for x in (cache,*cache.parents))):
        raise ValueError('Fresh absent cache prefix under build required')
    sys.path.insert(0,str(root))
    from scripts import mechanics_hbe_v5_fitted_confirmation_experiment as helper
    # Reparse with duplicate/nonfinite refusal before admitting any operation.
    release=helper.unique_json(raw)
    from scripts import mechanics_hbe_ogden_comparison as core
    rb={'path':release_path,'sha256':digest}
    checked,declaration=core.check_release(root,rb,executing=executing)
    core.need(checked==release,'exact parsed release')
    core.verify_sources(root,release,declaration,committed=executing)
    runtime=core.runtime(root,release);sites=runtime['site_packages']
    core.need(len(sites)==1 and Path(sites[0]).is_absolute() and Path(sites[0]).is_dir(),'one pinned site root')
    sys.path.insert(0,sites[0])
    return core,helper,rb,release,declaration


def worker(root,core,helper,rb,release,declaration,mode,deadline,request_binding):
    core.need(os.getpid()==os.getpgrp(),'owned worker group')
    cap=core.CAPS['preparation_seconds' if mode=='prepare' else 'profile_seconds' if mode=='profile' else 'readout_seconds']
    core.need(time.monotonic()<deadline<=time.monotonic()+cap,'parent deadline')
    request=core.read_binding(root,request_binding,maximum=128*1024)
    intent=core.read_binding(root,request['intent'],maximum=128*1024)
    core.need(intent['release']==rb and request['mode']==mode and deadline<=intent['work_deadline'],'parent request identity')
    def audit(event,args):
        if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp',
                     'socket.connect','socket.bind','socket.getaddrinfo'}:
            raise RuntimeError('Numerical readout/preparation cannot spawn or use network')
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if name.endswith(('.csv','.mat','.tar','.zip','.tar.gz')) or '/data/mechanics/' in name or '/data/patients/' in name:
                raise RuntimeError('No measured archive/member, other donor or patient access')
    sys.addaudithook(audit)
    out=core.local(root,core.OUTPUT)
    if mode=='prepare':
        from scripts import mechanics_hbe_backend as backend
        profile=backend.verify_profile(root,declaration['backend_profile'])
        core.need(profile['runtime_identity']==declaration['runtime_identity'],'native runtime identity')
        prepared=core.prepare(root,declaration,out)
        core.exclusive_json(out/'prepared.json',{'runs':prepared,'runtime':profile['runtime']},maximum=512*1024)
    elif mode.startswith('read-'):
        index=int(mode.removeprefix('read-'));core.need(index in range(6),'declared run index')
        prepared=core.read_binding(root,request['prepared'],maximum=512*1024)
        execution=core.read_binding(root,request['execution'],maximum=512*1024)
        row=core.RUNS[index];name=core.run_name(row)
        report=core.read_run(root,declaration,row,prepared['runs'][name],execution,deadline=deadline)
        core.exclusive_json(out/name/'readout.json',report,maximum=8*1024**2)
        core.need(report['numerical_passed'],'individual numerical check failed; evidence retained')
    elif mode=='screen':
        reports={name:core.read_binding(root,b,maximum=8*1024**2) for name,b in request['reports'].items()}
        screen=core.screen(reports,temporal=False)
        core.exclusive_json(out/'spatial-screen.json',screen,maximum=4*1024**2)
        core.need(screen['passed'],'two-grid screen failed; no temporal calls or profiling')
    elif mode=='profile':
        core.finish(root,declaration,request['reports'],out)
    else:raise ValueError('Undeclared worker mode')
    core.audit_loaded(root,release)
    core.need(time.monotonic()<deadline,'worker terminal deadline')


def expected_outputs(core):
    expected={'intent.json':128*1024,'prepared.json':512*1024,'spatial-screen.json':4*1024**2,
              'numerical-screen.json':4*1024**2,'comparison.json':512*1024}
    for mode in ('prepare','read-0','read-1','read-2','read-3','screen','read-4','read-5','profile'):
        for name,cap in {'request.json':128*1024,'receipt.json':1024**2,'readout-console.txt':4*1024**2}.items():
            expected['worker-'+mode+'/'+name]=cap
    for row in core.RUNS:
        for name,cap in {'specimen.feb':32*1024**2,'nodes.log':256*1024**2,'elements.log':256*1024**2,
             'solver.log':16*1024**2,'console.txt':16*1024**2,'receipt.json':1024**2,
             'execution.json':512*1024,'readout.json':8*1024**2}.items():
            expected[core.run_name(row)+'/'+name]=cap
    return expected


def execute(root,core,helper,rb,release,declaration,*,started=None):
    core.need(release['execution_released'] is True,'root execution release required')
    start=time.monotonic() if started is None else started;outer=start+core.CAPS['total_seconds']
    work_deadline=outer-core.CAPS['cleanup_reserve_seconds']
    out=core.local(root,core.OUTPUT);out.mkdir(parents=True,exist_ok=False)
    record={'schema':'hbe-ogden-shape-terminal-v1','status':'reserved','release':rb,'caps':core.CAPS,
        'native_calls':0,'native_calls_accounting':'attempted','positive_scale_profiles':0,
        'profile_accounting':'completed worker reports only; on interrupted profile up to two may have occurred',
        'archive_or_member_reads':0,'held_out_reads':0,'physical_validation_pass':None,'stages':{}}
    intent={'release':rb,'work_deadline':work_deadline,'caps':core.CAPS,'source_commit':release['source_commit']}
    intent_binding={'path':core.OUTPUT+'/intent.json','sha256':core.exclusive_json(out/'intent.json',intent)}
    stop={'requested':False};old={};owner=None
    def interrupted(signum,frame):stop['requested']=True;raise KeyboardInterrupt('termination requested')
    for signum in (signal.SIGTERM,signal.SIGINT):old[signum]=signal.signal(signum,interrupted)
    try:
        from scripts import mechanics_hbe_v5_remaining_one_shot as stages
        from scripts import mechanics_hbe_v5_n8_one_shot as io
        from scripts import mechanics_hbe_backend as backend
        from scripts import febio_runtime
        from launchers.hbe_v5_ordinal9_continuation_v1 import OwnedStage
        check_free_disk(core,out,reserve_outputs=True)
        env=io.private_environment();used_caches=[]
        def recheck():
            current,decl=core.check_release(root,rb,executing=True)
            core.need(current==release and decl==declaration,'authority changed')
            core.verify_sources(root,release,declaration,committed=False);core.audit_loaded(root,release)
            helper.check_cache(root,sys.pycache_prefix)
            for cache in used_caches:helper.check_cache(root,cache)
            core.need(not stop['requested'] and time.monotonic()<work_deadline,'remaining operation budget')
        def observed(pgid,*,timeout_seconds):
            check_free_disk(core,out)
            core.need(stages.active_bytes(out,core.CAPS['new_output_bytes'])<=core.CAPS['new_output_bytes'],'aggregate output cap')
            return febio_runtime.process_group_rss(pgid,timeout_seconds=timeout_seconds)
        def run(name,kind,command,directory,seconds,output_cap):
            nonlocal owner
            recheck();owner=OwnedStage();receipt={'phase':name,'release':rb}
            wall=min(seconds,work_deadline-time.monotonic());core.need(wall>0,'remaining stage budget')
            try:
                stage=stages.supervise_stage(kind,command,directory,receipt,
                    cwd=directory if kind=='native' else root,environment=env,wall_cap=wall,
                    rss_cap=core.CAPS['sampled_group_rss_bytes'],output_cap=output_cap,rss_observer=observed,popen=owner)
            finally:
                cleanup=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
                record['stages'][name]={'receipt':receipt,'cleanup':cleanup}
            helper.require_stage(core,stage,cleanup);owner=None
        def pyworker(mode,seconds,**payload):
            directory=out/('worker-'+mode);directory.mkdir(exist_ok=False)
            request={'intent':intent_binding,'mode':mode,**payload}
            rbq={'path':str((directory/'request.json').relative_to(root)),
                 'sha256':core.exclusive_json(directory/'request.json',request,maximum=128*1024)}
            cache=root/'build/hbe-ogden-shape-run-caches-v1'/('unused-'+mode)
            helper.check_cache(root,cache);used_caches.append(cache)
            deadline=min(time.monotonic()+seconds,work_deadline)
            command=[sys.executable,'-I','-S','-B','-X','pycache_prefix='+str(cache),str(root/RUNNER),
                '--root',str(root),'--release',rb['path'],'--release-sha256',rb['sha256'],'--worker',mode,
                '--deadline',repr(deadline),'--request',rbq['path'],'--request-sha256',rbq['sha256']]
            run(mode,'readout',command,directory,min(seconds,deadline-time.monotonic()),16*1024**2)
        pyworker('prepare',core.CAPS['preparation_seconds'])
        prepared_raw=helper.regular_bytes(out/'prepared.json',512*1024);prepared=helper.unique_json(prepared_raw)
        prepared_binding={'path':core.OUTPUT+'/prepared.json','sha256':core.sha(prepared_raw)}
        reports={}
        for index,row in enumerate(core.RUNS):
            if index==4:pyworker('screen',core.CAPS['readout_seconds'],reports=reports)
            profile=backend.verify_runtime_binding(root,declaration['runtime_identity'])
            core.need(profile['runtime']==prepared['runtime'],'prepared runtime identity')
            name=core.run_name(row);directory=out/name;binding=prepared['runs'][name]['deck']
            core.read_binding(root,binding,maximum=32*1024**2,decode=False)
            command=[profile['runtime']['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
            record['native_calls']+=1
            run(name,'native',command,directory,core.CAPS['native_seconds'][index],core.CAPS['run_output_bytes'])
            stages._check_backend(directory)
            bindings={}
            for key,cap in {**core.primitive_caps(row),'solver':16*1024**2}.items():
                path=directory/(key+'.log');bindings[key]={'path':str(path.relative_to(root)),
                                                        'sha256':io.file_hash(path,maximum=cap)}
            execution={'run_id':prepared['runs'][name]['contract']['run_id'],'deck':binding,
                'command':command,'primitive_bindings':bindings,'supervision':record['stages'][name],
                'runtime_identity':declaration['runtime_identity']}
            eb={'path':str((directory/'execution.json').relative_to(root)),
                'sha256':core.exclusive_json(directory/'execution.json',execution,maximum=512*1024)}
            pyworker('read-'+str(index),core.CAPS['readout_seconds'],prepared=prepared_binding,execution=eb)
            reports[name]={'path':str((directory/'readout.json').relative_to(root)),
                          'sha256':io.file_hash(directory/'readout.json',maximum=8*1024**2)}
        pyworker('profile',core.CAPS['profile_seconds'],reports=reports)
        record['positive_scale_profiles']=2
        expected=expected_outputs(core);sealed=helper.snapshot(root,core,out,expected)
        report=core.read_binding(root,{'path':core.OUTPUT+'/comparison.json','sha256':sealed['comparison.json']['sha256']})
        core.need(record['native_calls']==6 and report['alpha_candidates']==list(core.ALPHAS)
             and report['new_shape_screen_passed'] is True and report['physical_validation_pass'] is None,
             'exact numerical/diagnostic completion')
        recheck();core.need(helper.snapshot(root,core,out,expected)==sealed,'late output mutation')
        record.update(status=core.DONE,output_inventory=sealed,elapsed_seconds=time.monotonic()-start)
        digest=core.exclusive_json(out/'terminal.json',record)
        expected['terminal.json']=1024**2;sealed['terminal.json']={'bytes':(out/'terminal.json').stat().st_size,'sha256':digest}
        recheck();core.need(helper.snapshot(root,core,out,expected)==sealed,'persisted terminal changed')
        publication={'schema':'hbe-ogden-shape-publication-v1','accepted':True,'release':rb,'outputs':sealed,
                     'physical_validation_pass':None,'elapsed_seconds':time.monotonic()-start}
        digest=core.exclusive_json(out/'publication.json',publication)
        expected['publication.json']=1024**2;sealed['publication.json']={'bytes':(out/'publication.json').stat().st_size,'sha256':digest}
        recheck();core.need(helper.snapshot(root,core,out,expected)==sealed and time.monotonic()<outer,'final publication seal/deadline')
    except BaseException as error:
        record.update(status='failed_shape_comparison',error_type=type(error).__name__,error=str(error)[:2048])
    finally:
        try:
            if owner is not None:
                from scripts import febio_runtime
                record['fallback_cleanup']=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
            if record['status']!=core.DONE:
                core.exclusive_json(out/'finalization-failure.json',record)
                if not (out/'terminal.json').exists():core.exclusive_json(out/'terminal.json',record)
                if not (out/'publication.json').exists():core.exclusive_json(out/'publication.json',{'accepted':False,'release':rb})
        finally:
            for signum,handler in old.items():signal.signal(signum,handler)
    return record


def main():
    started=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--release',required=True)
    p.add_argument('--release-sha256',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',choices=('prepare','read-0','read-1','read-2','read-3','screen','read-4','read-5','profile'),help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    p.add_argument('--request',help=argparse.SUPPRESS);p.add_argument('--request-sha256',help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.resolve()
    if args.execute and args.worker:raise ValueError('one operation mode required')
    core,helper,rb,release,declaration=bootstrap(root,args.release,args.release_sha256,executing=args.execute or bool(args.worker))
    if args.worker:
        worker(root,core,helper,rb,release,declaration,args.worker,args.deadline,
               {'path':args.request,'sha256':args.request_sha256});return 0
    if not args.execute:
        print(json.dumps({'status':'source_preflight_only','native_calls':0,'measured_json_reads':0}));return 0
    result=execute(root,core,helper,rb,release,declaration,started=started)
    print(json.dumps({'status':result['status'],'native_calls':result['native_calls'],'physical_validation_pass':None}))
    return 0 if result['status']==core.DONE else 1


if __name__=='__main__':raise SystemExit(main())
