"""Owned fixed-fit continuation; no root release means no native or response access."""
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

CORE='scripts/mechanics_hbe_v5_fitted_confirmation.py'
RUNNER='scripts/mechanics_hbe_v5_fitted_confirmation_experiment.py'


def regular_bytes(path,maximum,*,expected=None,allow_empty=False):
    """One bounded stable regular read; FIFO/symlink refusal precedes body access."""
    path=Path(path)
    if '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Linked or escaping input forbidden')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not (0 if allow_empty else 1)<=before.st_size<=maximum:
            raise ValueError('Bounded regular input required')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as stream:raw=stream.read(maximum+1)
        after=os.fstat(fd)
        identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        if identity(before)!=identity(after) or identity(after)!=identity(path.lstat()) or len(raw)!=before.st_size:
            raise ValueError('Input changed while read')
    finally:os.close(fd)
    if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Input digest differs')
    return raw


def unique_json(raw):
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def check_cache(root,path):
    cache=Path(path or '')
    if (not cache.is_absolute() or '..' in cache.parts or not cache.is_relative_to(root/'build')
            or cache.exists() or cache.is_symlink()
            or any(p.is_symlink() for p in cache.parents)):
        raise ValueError('Fresh absent cache prefix under ignored build required')
    return cache


def bootstrap(root,release_path,release_sha,*,worker):
    """Verify exact local source before imports; never consult a cached project pyc."""
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Use -I -S -B and a fresh absent -X pycache_prefix')
    cache=check_cache(root,sys.pycache_prefix)
    relative=Path(release_path)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Relative release required')
    raw=regular_bytes(root/relative,1024**2,expected=release_sha)
    release=unique_json(raw)
    bindings=release.get('source_bindings')
    if (type(bindings) is not dict or any(type(bindings.get(name)) is not str
            or len(bindings[name])!=64 or any(x not in '0123456789abcdef' for x in bindings[name])
            for name in (CORE,RUNNER))):raise ValueError('Exact executing source hashes required')
    sources={name:regular_bytes(root/name,1024**2,expected=bindings[name])
             for name in (CORE,RUNNER)}
    if any(name not in release['source_bindings'] for name in sources):raise ValueError('Missing executing source binding')
    if Path(__file__).resolve()!=root/RUNNER:raise ValueError('Canonical runner required')
    # This core reuses the existing stdlib-only authority module. Authenticate
    # ALL local sources before making the canonical import root available.
    for name,digest in bindings.items():
        relative=Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('scripts','launchers')
                or relative.suffix!='.py' or type(digest) is not str or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Exact local source inventory required')
        regular_bytes(root/relative,1024**2,expected=digest)
    sys.path.insert(0,str(root))
    source=sources[CORE]  # Hash and compile the exact same bounded bytes.
    spec=importlib.util.spec_from_loader('axial_fit_core',loader=None,origin=str(root/CORE))
    core=importlib.util.module_from_spec(spec);core.__file__=str(root/CORE)
    exec(compile(source,str(root/CORE),'exec'),core.__dict__)
    rb={'path':release_path,'sha256':release_sha}
    checked,declaration=core.check_release(root,rb,executing=worker)
    core.verify_sources(root,checked,declaration,committed=worker or checked['execution_released'])
    sites=checked['runtime']['site_packages']
    if not isinstance(sites,list) or len(sites)!=1:raise ValueError('One pinned site-package root required')
    site=Path(sites[0])
    if not site.is_absolute() or not site.is_dir() or site.is_symlink():raise ValueError('Runtime site root')
    # -S excludes .pth/sitecustomize; only the released directory is added explicitly.
    sys.path[:0]=[str(root),str(site)]
    if worker:
        wanted=root/'build/hbe-v5-fixed-fit-confirmation-v1'/('unused-'+os.environ.get('HBE_CONTINUATION_MODE','')+'-pycache')
        if cache!=wanted:raise ValueError('Exact worker cache prefix required')
    return core,rb,checked,declaration


def worker(root,core,rb,release,declaration,mode,deadline):
    if os.getpid()!=os.getpgrp() or mode not in {'prepare','readout'}:
        raise ValueError('Owned declared worker required')
    cap=core.CAPS['preparation_seconds' if mode=='prepare' else 'readout_seconds']
    core.need(time.monotonic()<deadline<=time.monotonic()+cap,'parent-owned deadline')
    out=core.local(root,core.OUTPUT)
    intent=core.read_binding(root,{'path':core.OUTPUT+'/intent.json','sha256':os.environ['HBE_CONTINUATION_INTENT_SHA256']})
    core.need(intent['release']==rb and deadline<=intent['work_deadline'],'exact parent intent')
    def audit(event,args):
        if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp',
                     'socket.connect','socket.bind','socket.getaddrinfo'}:
            raise RuntimeError('Readout/preparation worker cannot spawn or use network')
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if name.endswith(('.csv','.mat')) or '/data/mechanics/' in name or '/data/patients/' in name:
                raise RuntimeError('No measured member or patient access')
    sys.addaudithook(audit)
    if mode=='prepare':
        from scripts import mechanics_hbe_backend as backend
        profile=backend.verify_profile(root,declaration['backend_profile'])
        core.need(profile['runtime_identity']==declaration['runtime_identity'],'fixed native runtime')
        prepared=core.prepare(root,declaration,out)
        core.exclusive_json(out/'prepared.json',{'branches':prepared,'runtime':profile['runtime']})
    else:
        prepared=core.read_binding(root,{'path':core.OUTPUT+'/prepared.json','sha256':os.environ['HBE_PREPARED_SHA256']})
        executions=core.read_binding(root,{'path':core.OUTPUT+'/executions.json','sha256':os.environ['HBE_EXECUTIONS_SHA256']})
        numeric=core.read_pairs(root,declaration,prepared['branches'],executions,deadline=deadline)
        # Preserve unsuccessful numerical reductions before refusing the freeze.
        core.exclusive_json(out/'numerical.json',numeric,maximum=4*1024**2)
        core.need(numeric['passed'],'fitted numerical confirmation failed')
        core.finish(root,declaration,rb,out,numeric,executions)
    core.audit_loaded(root,release)
    core.need(time.monotonic()<deadline,'worker terminal deadline')


def snapshot(root,core,out,expected):
    """Seal a fixed small inventory, streaming hashes of large primitive files."""
    from scripts import mechanics_hbe_v5_remaining_one_shot as stages
    from scripts.mechanics_hbe_v5_n8_one_shot import file_hash
    total=stages.active_bytes(out,core.CAPS['new_output_bytes'])
    core.need(total<=core.CAPS['new_output_bytes'],'aggregate output cap')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    core.need(actual==set(expected),'exact continuation file inventory')
    result={}
    for name,cap in sorted(expected.items()):
        path=core.local(root,str((out/name).relative_to(root)))
        result[name]={'bytes':path.stat().st_size,'sha256':file_hash(path,maximum=cap)}
    return result


def expected_outputs(core):
    expected={name:4*1024**2 for name in ('intent.json','prepared.json','executions.json',
              'numerical.json','predictions.json','freeze.json','continuation-ledger.json')}
    for stage in ('prepare-stage','readout-stage'):
        expected.update({stage+'/receipt.json':1024**2,stage+'/readout-console.txt':4*1024**2})
    for branch in core.AXIAL:
        expected.update({branch+'/'+name:cap for name,cap in {
            'specimen.feb':32*1024**2,'nodes.log':core.primitive_caps(branch)['nodes'],
            'elements.log':core.primitive_caps(branch)['elements'],'solver.log':16*1024**2,
            'console.txt':16*1024**2,'receipt.json':1024**2}.items()})
    return expected


def require_stage(core,stage,cleanup):
    core.need(stage['status']=='completed_within_caps' and stage['exit_code']==0
         and cleanup['contained'] and cleanup['direct_child_reaped']
         and cleanup['remaining_members']==[] and cleanup['errors']==[]
         and cleanup['fallback_used'] is False and cleanup['exit_code']==0,'owned stage failed')


def execute(root,core,rb,release,declaration,*,started=None):
    core.need(release['execution_released'] is True,'separate root release required')
    checked,decl=core.check_release(root,rb,executing=True)
    core.need(checked==release and decl==declaration,'exact authority')
    start=time.monotonic() if started is None else started
    outer=start+core.CAPS['total_seconds'];work_deadline=outer-core.CAPS['cleanup_reserve_seconds']
    out=core.local(root,core.OUTPUT);out.mkdir(parents=True,exist_ok=False)
    record={'schema':'hbe-v5-fixed-fit-terminal-v1','status':'reserved','release':rb,
            'caps':core.CAPS,'native_calls':0,'native_calls_accounting':'attempted','fit_calls':0,'mesher_calls':0,
            'measured_member_reads':0,'held_out_member_reads':0,'physical_validation_pass':None,'stages':{}}
    intent={'release':rb,'work_deadline':work_deadline,'caps':core.CAPS,'source_commit':release['source_commit']}
    intent_sha=core.exclusive_json(out/'intent.json',intent)
    stop={'requested':False};old={};owner=None
    def interrupted(signum,frame):stop['requested']=True;raise KeyboardInterrupt('termination requested')
    for signum in (signal.SIGTERM,signal.SIGINT):old[signum]=signal.signal(signum,interrupted)
    try:
        import shutil
        from scripts import mechanics_hbe_v5_remaining_one_shot as stages
        from scripts import mechanics_hbe_v5_n8_one_shot as io
        from scripts import mechanics_hbe_backend as backend
        from scripts import febio_runtime
        from launchers.hbe_v5_ordinal9_continuation_v1 import OwnedStage
        core.need(shutil.disk_usage(out).free>=core.CAPS['new_output_bytes']+2*1024**3,'disk reserve')
        env=io.private_environment();env['HBE_CONTINUATION_INTENT_SHA256']=intent_sha
        def recheck():
            current,current_decl=core.check_release(root,rb,executing=True)
            core.need(current==release and current_decl==declaration,'authority changed')
            core.verify_sources(root,release,declaration,committed=False);core.audit_loaded(root,release)
            core.fixed_fit(root,declaration)
            check_cache(root,sys.pycache_prefix)
            for mode in ('prepare','readout'):check_cache(root,root/'build/hbe-v5-fixed-fit-confirmation-v1'/('unused-'+mode+'-pycache'))
            core.need(not stop['requested'] and time.monotonic()<work_deadline,'phase work deadline')
        def observed(pgid,*,timeout_seconds):
            core.need(stages.active_bytes(out,core.CAPS['new_output_bytes'])<=core.CAPS['new_output_bytes'],
                      'aggregate output budget')
            return febio_runtime.process_group_rss(pgid,timeout_seconds=timeout_seconds)
        def run(name,kind,command,directory,seconds,environment,output_cap):
            nonlocal owner
            recheck();owner=OwnedStage();receipt={'phase':name,'release':rb}
            wall=min(seconds,work_deadline-time.monotonic());core.need(wall>0,'remaining stage budget')
            try:
                stage=stages.supervise_stage(kind,command,directory,receipt,cwd=(directory if kind=='native' else root),
                    environment=environment,wall_cap=wall,rss_cap=core.CAPS['sampled_group_rss_bytes'],
                    output_cap=output_cap,rss_observer=observed,popen=owner)
            finally:
                cleanup=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
                record['stages'][name]={'receipt':receipt,'cleanup':cleanup}
            require_stage(core,stage,cleanup);owner=None
            return receipt
        def pyworker(mode,seconds):
            directory=out/(mode+'-stage');directory.mkdir(exist_ok=False)
            deadline=min(time.monotonic()+seconds,work_deadline)
            env['HBE_CONTINUATION_MODE']=mode
            command=[sys.executable,'-I','-S','-B','-X','pycache_prefix='+str(root/'build/hbe-v5-fixed-fit-confirmation-v1'/('unused-'+mode+'-pycache')),
                str(root/RUNNER),'--root',str(root),'--release',rb['path'],'--release-sha256',rb['sha256'],
                '--worker',mode,'--deadline',repr(deadline)]
            return run(mode,'readout',command,directory,min(seconds,deadline-time.monotonic()),env,16*1024**2)
        pyworker('prepare',core.CAPS['preparation_seconds'])
        prepared_raw=regular_bytes(out/'prepared.json',4*1024**2);prepared=unique_json(prepared_raw)
        env['HBE_PREPARED_SHA256']=core.sha(prepared_raw)
        executions={}
        for branch in core.AXIAL:
            # Recheck the exact runtime and exact prepared deck before EACH native call.
            profile=backend.verify_runtime_binding(root,declaration['runtime_identity'])
            core.need(profile['runtime']==prepared['runtime'],'prepared native identity')
            directory=out/branch;binding=prepared['branches'][branch]['deck']
            core.read_binding(root,binding,maximum=32*1024**2,decode=False)
            command=[profile['runtime']['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
            record['native_calls']+=1
            receipt=run(branch,'native',command,directory,core.CAPS['native_seconds'][branch],env,
                        core.CAPS['branch_output_bytes'][branch])
            stages._check_backend(directory)
            bindings={}
            for key,cap in {**core.primitive_caps(branch),'solver':16*1024**2}.items():
                path=directory/(key+'.log');bindings[key]={'path':str(path.relative_to(root)),
                    'sha256':io.file_hash(path,maximum=cap)}
            executions[branch]={'run_id':prepared['branches'][branch]['contract']['run_id'],
                'fixed_fit_sha256':core.FIT_SHA,'deck':binding,'command':command,
                'primitive_bindings':bindings,'supervision':record['stages'][branch],
                'runtime_identity':declaration['runtime_identity']}
        env['HBE_EXECUTIONS_SHA256']=core.exclusive_json(out/'executions.json',executions)
        pyworker('readout',core.CAPS['readout_seconds'])
        core.need(record['native_calls']==2,'exactly two completed native confirmations')
        expected=expected_outputs(core);sealed=snapshot(root,core,out,expected)
        freeze=core.read_binding(root,{'path':core.OUTPUT+'/freeze.json','sha256':sealed['freeze.json']['sha256']})
        core.need(freeze['mu_Pa']==core.MU and freeze['fixed_fit']==declaration['fixed_fit']
             and freeze['fitted_executions']==executions and freeze['held_out_access_released'] is False,
             'fixed freeze semantic mismatch')
        core.recheck_fixed_inputs(root,declaration,freeze)
        recheck();core.need(snapshot(root,core,out,expected)==sealed,'output changed before terminal')
        record.update(status=core.DONE,freeze={'path':core.OUTPUT+'/freeze.json','sha256':sealed['freeze.json']['sha256']},
                      output_inventory=sealed,elapsed_before_terminal_seconds=time.monotonic()-start)
        terminal_sha=core.exclusive_json(out/'terminal.json',record)
        expected['terminal.json']=1024**2
        sealed['terminal.json']={'bytes':(out/'terminal.json').stat().st_size,'sha256':terminal_sha}
        core.need(snapshot(root,core,out,expected)==sealed,'persisted terminal changed')
        recheck()
        publication={'schema':'hbe-v5-fixed-fit-publication-v1','accepted':True,'release':rb,
                     'outputs':sealed,'physical_validation_pass':None,'elapsed_seconds':time.monotonic()-start}
        digest=core.exclusive_json(out/'publication.json',publication)
        expected['publication.json']=1024**2;sealed['publication.json']={'bytes':(out/'publication.json').stat().st_size,'sha256':digest}
        core.recheck_fixed_inputs(root,declaration,freeze)
        recheck();core.need(snapshot(root,core,out,expected)==sealed,'publication bytes changed')
        core.need(time.monotonic()<outer and not stop['requested'],'terminal deadline')
    except BaseException as error:
        record.update(status='failed_fixed_fit_continuation',error_type=type(error).__name__,error=str(error)[:2048])
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
    p.add_argument('--worker',choices=('prepare','readout'),help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.resolve()
    if args.execute and args.worker:raise ValueError('one mode required')
    core,rb,release,declaration=bootstrap(root,args.release,args.release_sha256,worker=bool(args.worker))
    if args.worker:worker(root,core,rb,release,declaration,args.worker,args.deadline);return 0
    if args.execute:
        result=execute(root,core,rb,release,declaration,started=started)
        print(json.dumps({'status':result['status'],'physical_validation_pass':None}))
        return 0 if result['status']==core.DONE else 1
    # Deliberately source/declaration-only: no specimen output or native-runtime open.
    print(json.dumps({'status':'source_preflight_only','native_calls':0,'native_calls_accounting':'attempted','fit_calls':0,'measured_member_reads':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
