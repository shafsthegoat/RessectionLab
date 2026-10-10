"""Bounded torsion-only command after accepted fixed-fit freeze; no implicit access."""
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

CORE='scripts/mechanics_hbe_v5_torsion_evaluation.py'
RUNNER='scripts/mechanics_hbe_v5_torsion_evaluation_experiment.py'


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
    # Authenticate reused project sources before making imports available.
    for name,digest in bindings.items():
        path=Path(name)
        if (path.is_absolute() or '..' in path.parts or path.parts[0] not in ('scripts','launchers')
                or path.suffix!='.py' or type(digest) is not str or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Exact local source inventory required')
        regular_bytes(root/path,1024**2,expected=digest)
    sys.path.insert(0,str(root))
    source=sources[CORE]  # Hash and compile the exact same bounded bytes.
    spec=importlib.util.spec_from_loader('torsion_evaluation_core',loader=None,origin=str(root/CORE))
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
        wanted=root/checked['output_directory']/'unused-worker-pycache'
        if cache!=wanted:raise ValueError('Exact worker cache prefix required')
    return core,rb,checked,declaration


def worker(root,core,rb,release,declaration,deadline):
    if os.getpid()!=os.getpgrp():raise ValueError('Owned worker must lead its process group')
    if not isinstance(deadline,float) or not time.monotonic()<deadline<=time.monotonic()+60:
        raise ValueError('Parent-owned remaining deadline required')
    out=core.local(root,core.OUTPUT)
    intent=core.read_binding(root,{'path':core.OUTPUT+'/intent.json',
        'sha256':os.environ['HBE_TORSION_INTENT_SHA256']})
    if intent['release']!=rb or intent['deadline']!=deadline:raise ValueError('Exact parent intent required')
    if (out/'state.json').exists() or (out/'access.jsonl').exists():raise ValueError('Worker attempt consumed')
    # The bounded worker has no reason to spawn, network, or dynamically read other measurements.
    def audit(event,args):
        if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp',
                     'socket.connect','socket.bind','socket.getaddrinfo'}:
            raise RuntimeError('Torsion-only worker cannot spawn or use network')
    sys.addaudithook(audit)
    core.audit_loaded(root,release)
    return core.run_evaluation(root,rb,release,declaration,deadline=deadline)


BASE_OUTPUTS=frozenset({'intent.json','intent-receipt.json','receipt.json','readout-console.txt',
                        'state.json','access.jsonl','held-out-metrics.json','observed-torsion.json'})


def output_snapshot(core,out,names):
    if {p.name for p in out.iterdir()}!=set(names):raise ValueError('Exact output inventory differs')
    snapshot={};total=0
    for name in sorted(names):
        raw=regular_bytes(out/name,core.CAPS['new_output_bytes'],allow_empty=name=='readout-console.txt')
        snapshot[name]={'bytes':len(raw),'sha256':core.sha(raw)};total+=len(raw)
        core.need(total<=core.CAPS['new_output_bytes'],'output byte cap')
    return snapshot


def recheck_authority(root,core,rb,release,declaration,out):
    checked,decl=core.check_release(root,rb,executing=True)
    core.need(checked==release and decl==declaration,'final root authority changed')
    core.verify_sources(root,release,declaration,committed=False)
    core.audit_loaded(root,release)
    core.admit_upstream(root,release,declaration)
    check_cache(root,sys.pycache_prefix);check_cache(root,out/'unused-worker-pycache')


def finalize(root,core,rb,release,declaration,out,record,*,start,outer,stop,sealed):
    """Reuse the prior receipt/publication pattern; failure never overwrites evidence."""
    terminal=out/'terminal.json';publication=out/'publication.json'
    try:
        if record['status']==core.DONE:
            core.need(not stop['requested'] and time.monotonic()<outer,'termination or outer deadline')
            recheck_authority(root,core,rb,release,declaration,out)
            core.need(output_snapshot(core,out,BASE_OUTPUTS)==sealed,'output changed before terminal')
            record['output_inventory']=sealed
        record['elapsed_before_terminal_publication_seconds']=time.monotonic()-start
        record['termination_requested']=stop['requested']
        if stop['requested'] or time.monotonic()>=outer:record['status']='failed_torsion_evaluation_attempt'
        raw=core.canonical(record)
        core.exclusive_json(terminal,record,maximum=32768)
        regular_bytes(terminal,32768,expected=core.sha(raw))
        if record['status']==core.DONE:
            terminal_binding={'bytes':len(raw),'sha256':core.sha(raw)}
            expected={**sealed,'terminal.json':terminal_binding}
            core.need(output_snapshot(core,out,set(expected))==expected,'output changed during terminal publication')
            recheck_authority(root,core,rb,release,declaration,out)
            core.need(not stop['requested'] and time.monotonic()<outer,'termination or publication deadline')
            published={'schema':'hbe-v5-torsion-evaluation-publication-v1','accepted':True,'release':rb,
                'outputs':expected,'elapsed_before_publication_seconds':time.monotonic()-start,
                'physical_validation_pass':None}
            public_raw=core.canonical(published)
            core.need(sum(x['bytes'] for x in expected.values())+len(public_raw)<=core.CAPS['new_output_bytes'],
                      'terminal output cap')
            core.exclusive_json(publication,published,maximum=32768)
            expected['publication.json']={'bytes':len(public_raw),'sha256':core.sha(public_raw)}
            core.need(output_snapshot(core,out,set(expected))==expected,'persisted publication/output changed')
            recheck_authority(root,core,rb,release,declaration,out)
            core.need(output_snapshot(core,out,set(expected))==expected,'output changed during final authority check')
            core.need(not stop['requested'] and time.monotonic()<outer,'terminal publication exceeded envelope')
        else:
            core.exclusive_json(publication,{'schema':'hbe-v5-torsion-evaluation-publication-v1','accepted':False,
                'release':rb,'terminal_sha256':core.sha(raw),'physical_validation_pass':None},maximum=32768)
    except BaseException as error:
        record.update(status='failed_torsion_evaluation_attempt',finalization_error_type=type(error).__name__,
                      finalization_error=str(error)[:2048],termination_requested=stop['requested'])
        failure={'schema':'hbe-v5-torsion-evaluation-publication-failure-v1','accepted':False,
                 'release':rb,'error_type':type(error).__name__,'error':str(error)[:2048],
                 'elapsed_seconds':time.monotonic()-start,'physical_validation_pass':None}
        # Preserve even a mutated/partially written earlier receipt. This marker
        # invalidates publication; a later consumer must require its absence.
        core.exclusive_json(out/'finalization-failure.json',failure,maximum=8192)
        if not terminal.exists():core.exclusive_json(terminal,record,maximum=32768)
        if not publication.exists():core.exclusive_json(publication,failure,maximum=32768)
    return record


def execute(root,core,rb,release,declaration,*,started=None):
    # Gate before reservation; source-bound false templates remain metadata-only.
    core.need(release['execution_released'] is True,'root release required')
    checked,decl=core.check_release(root,rb,executing=True)
    core.need(checked==release and decl==declaration,'exact reviewed root authority required')
    out=core.local(root,core.OUTPUT);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic() if started is None else started
    deadline=start+core.CAPS['worker_seconds'];outer=start+core.CAPS['total_seconds']
    intent={'schema':'hbe-v5-torsion-evaluation-intent-v1','release':rb,'deadline':deadline,
            'source_commit':release['source_commit'],'caps':core.CAPS,'automatic_retry':False}
    intent_sha=core.exclusive_json(out/'intent.json',intent)
    record={'schema':'hbe-v5-torsion-evaluation-supervision-v1','status':'reserved','release':rb,
            'declaration':release['declaration'],'source_commit':release['source_commit'],
            'caps':core.CAPS,'native_calls':0,'mesher_calls':0,'fit_calls':0,'held_out_member_reads':None,
            'measured_access_accounting':'unknown_until_worker_state_and_ledger_checked',
            'physical_validation_pass':None,'parent_cache_prefix':sys.pycache_prefix}
    core.exclusive_json(out/'intent-receipt.json',record)
    owner=None;stage=None;sealed=None;stop={'requested':False};old_handlers={}
    def interrupted(signum,frame):
        stop['requested']=True
        raise KeyboardInterrupt('termination requested')
    for signum in (signal.SIGINT,signal.SIGTERM):
        old_handlers[signum]=signal.signal(signum,interrupted)
    try:
        from scripts import mechanics_hbe_v5_remaining_one_shot as stages
        from scripts import mechanics_hbe_v5_n8_one_shot as io
        from scripts import febio_runtime
        from launchers.hbe_v5_ordinal9_continuation_v1 import OwnedStage
        core.audit_loaded(root,release)
        owner=OwnedStage()
        cache=out/'unused-worker-pycache'
        core.need(not cache.exists() and not cache.is_symlink(),'worker cache already exists')
        command=[sys.executable,'-I','-S','-B','-X','pycache_prefix='+str(cache),str(root/RUNNER),
            '--root',str(root),'--release',rb['path'],'--release-sha256',rb['sha256'],
            '--worker','--deadline',repr(deadline)]
        env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','DYLD_','LD_'))
             and k!='__PYVENV_LAUNCHER__'}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',HBE_TORSION_INTENT_SHA256=intent_sha)
        remaining=deadline-time.monotonic();core.need(remaining>0,'startup exhausted worker window')
        try:
            stage=stages.supervise_stage('readout',command,out,record,cwd=root,environment=env,
                wall_cap=remaining,rss_cap=core.CAPS['sampled_group_rss_bytes'],
                output_cap=core.CAPS['new_output_bytes']-131072,
                rss_observer=febio_runtime.process_group_rss,popen=owner)
        finally:
            record['cleanup']=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
        cleanup=record['cleanup']
        core.need(stage['status']=='completed_within_caps' and stage['exit_code']==0
            and cleanup['contained'] and cleanup['direct_child_reaped'] and not cleanup['remaining_members']
            and cleanup['fallback_used'] is False and cleanup['exit_code']==0
            and not cleanup['errors'] and not stop['requested'],'worker supervision/cleanup failed')
        sealed=output_snapshot(core,out,BASE_OUTPUTS)
        state_binding={'path':core.OUTPUT+'/state.json','sha256':sealed['state.json']['sha256']}
        state=core.read_binding(root,state_binding)
        core.need(state['status']==core.DONE and state['fit_calls']==0
            and state['native_calls']==0 and state['mesher_calls']==0 and state['held_out_member_reads']==2
            and state['held_out_responses_accessed'] is True and state['new_freeze_saved'] is False
            and state['physical_validation_pass'] is None and state['empirical_tolerance'] is None
            and state['release']==rb and state['upstream']==release['upstream'],'worker semantic failure')
        events=[unique_json(x) for x in regular_bytes(out/'access.jsonl',1024**2,expected=sealed['access.jsonl']['sha256']).splitlines()]
        core.validate_terminal_ledger(root,events,rb,release,declaration,state)
        core.need(state['held_out_metrics']=={'path':core.OUTPUT+'/held-out-metrics.json',
                  'sha256':sealed['held-out-metrics.json']['sha256']},'worker output binding changed')
        metrics=core.read_binding(root,state['held_out_metrics'],maximum=4*1024**2)
        core.need(metrics.get('physical_validation_pass') is None and metrics.get('empirical_tolerance') is None
            and metrics.get('independent_donor_validation') is False
            and set(metrics.get('branches',{}))==set(core.TORSION),'descriptive result only')
        core.need(state['observed_torsion']=={'path':core.OUTPUT+'/observed-torsion.json',
                  'sha256':sealed['observed-torsion.json']['sha256']},'saved observed identity')
        observed=core.read_binding(root,state['observed_torsion'],maximum=1024**2)
        core.need(observed.get('schema')=='hbe-held-out-torsion-curves-v1' and observed.get('release')==rb
            and set(observed.get('curves',{}))==set(core.TORSION)
            and {b:observed['curves'][b]['source_sha256'] for b in core.TORSION}==state['member_sha256'],
            'saved observed source identity')
        core.verify_sources(root,release,declaration,committed=False)
        core.audit_loaded(root,release)
        record.update(status=core.DONE,state=state_binding,held_out_metrics=state['held_out_metrics'],
            observed_torsion=state['observed_torsion'],
            access_ledger={'path':core.OUTPUT+'/access.jsonl','sha256':sealed['access.jsonl']['sha256']},
            measured_access_accounting='exact_two_completed_torsion_members_2556_uncompressed_bytes',
            fit_calls=0,held_out_member_reads=2,new_freeze_saved=False,upstream=release['upstream'],
            empirical_tolerance=None,patient_tool_mechanics_admitted=False)
        core.need(output_snapshot(core,out,BASE_OUTPUTS)==sealed,'semantic output changed')
    except BaseException as error:
        record.update(status='failed_torsion_evaluation_attempt',error_type=type(error).__name__,
                      error=str(error)[:2048])
    finally:
        # Keep termination handling active through durable final publication.
        try:
            if owner is not None and 'cleanup' not in record:
                from scripts import febio_runtime
                record['cleanup']=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
            finalize(root,core,rb,release,declaration,out,record,start=start,outer=outer,stop=stop,sealed=sealed)
        finally:
            for signum,handler in old_handlers.items():signal.signal(signum,handler)
    return record


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--release',required=True)
    p.add_argument('--release-sha256',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.resolve()
    if args.execute and args.worker:raise ValueError('one mode required')
    core,rb,release,declaration=bootstrap(root,args.release,args.release_sha256,worker=args.worker)
    if args.worker:
        worker(root,core,rb,release,declaration,args.deadline);return 0
    if args.execute:
        result=execute(root,core,rb,release,declaration,started=started)
        print(json.dumps({'status':result['status'],'physical_validation_pass':None}))
        return 0 if result['status']==core.DONE else 1
    # No acquired archive, selected CSV or fit in check-only mode.
    # A false template has no upstream identities. Check only fixed source/roles.
    core.read_binding(root,declaration['roles']);core.read_binding(root,declaration['protocol'])
    core.audit_loaded(root,release)
    print(json.dumps({'status':'metadata_preflight_only','measured_member_reads':0,'fit_calls':0,'native_calls':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
