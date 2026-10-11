"""One saved-readout finalization recovery; no native or numerical-screen replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
from types import SimpleNamespace

RUNNER='scripts/mechanics_hbe_ogden_recovery.py'
ORIGINAL='scripts/mechanics_hbe_ogden_comparison_experiment.py'
ORIGINAL_SHA='aa974110a5b57a7cd409ffa43feceaf348b4635647ded3ca2e21282c7a32bd03'
OUTPUT='outputs/mechanics/hbe-ogden-shape-comparison-v1/postprocess-01'
PRIOR='outputs/mechanics/hbe-ogden-shape-comparison-v1/attempt-01'
UPSTREAM={
 'terminal':{'path':PRIOR+'/terminal.json','sha256':'ca8127909803be8b64758504fc8a01f3d56040c8153c7df0b39fffcbf978a74e'},
 'request':{'path':PRIOR+'/worker-profile/request.json','sha256':'a3f03ae75f0a7e62eb199c03315500ffc9e432763ad6e5c10c32746ee36297e0'},
 'console':{'path':PRIOR+'/worker-profile/readout-console.txt','sha256':'5ec3ebf9fcd084abec99db63976ca989717e4e8b195334cc94e37be7b5e3d2f7'},
 'screen':{'path':PRIOR+'/numerical-screen.json','sha256':'546722291e7f78ad2c0ca4fb0dae4a8f769ee22188ef1bb4fdaed4db732b49ff'},
 'release':{'path':'build/hbe-ogden-shape-v1/root-release.json','sha256':'674ab5eb044db49289a31bd6c5a78d770a4b7cd8cec6b1a7aff92be5ca40bb28'}}
CAPS={'total_seconds':45,'worker_seconds':35,'cleanup_seconds':10,
      'sampled_group_rss_bytes':512*1024**2,'new_output_bytes':1024**2,
      'threads':1,'attempts':1,'native_calls':0,'numerical_replays':0,
      'positive_scale_profiles':2,'baseline_reproductions':1,'automatic_retry':False}
DONE='completed_saved_readout_shape_comparison'


def bootstrap(root,path,digest,*,executing):
    # Execute the unchanged reviewed bootstrap utility from authenticated source bytes.
    fd=os.open(root/ORIGINAL,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=1024**2:raise ValueError('Bounded supervisor source')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as f:raw=f.read(1024**2+1)
    finally:os.close(fd)
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA:raise ValueError('Original supervisor source differs')
    old={'__name__':'shape_recovery_bootstrap','__file__':str(root/ORIGINAL)}
    exec(compile(raw,str(root/ORIGINAL),'exec'),old)
    bounded=old['bounded']
    if not(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise ValueError('Require -I -S -B and absent -X pycache_prefix')
    p=Path(path)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Relative release')
    release_raw=bounded(root/p,128*1024,digest);pre=json.loads(release_raw)
    for name,expected in pre['source_bindings'].items():
        q=Path(name)
        if q.is_absolute() or '..' in q.parts or q.parts[0] not in ('scripts','launchers') or q.suffix!='.py':
            raise ValueError('Source path scope')
        bounded(root/q,1024**2,expected)
    for name in (RUNNER,old['CORE'],old['HELPER'],ORIGINAL):
        if name not in pre['source_bindings']:raise ValueError('Incomplete source closure')
    if Path(__file__).resolve()!=root/RUNNER:raise ValueError('Canonical recovery runner required')
    cache=Path(sys.pycache_prefix or '')
    if not cache.is_absolute() or '..' in cache.parts or not cache.is_relative_to(root/'build') or cache.exists() or any(x.is_symlink() for x in (cache,*cache.parents)):
        raise ValueError('Fresh absent cache required')
    sys.path.insert(0,str(root))
    from scripts import mechanics_hbe_v5_fitted_confirmation_experiment as helper
    from scripts import mechanics_hbe_ogden_comparison as core
    release=helper.unique_json(release_raw)
    core.need(set(release)=={'schema','execution_released','source_commit','source_bindings','runtime_source',
                            'declaration','output_directory','caps','storage_receipt','upstream'},'release keys')
    core.need(release['schema']=='hbe-ogden-saved-recovery-release-v1' and type(release['execution_released']) is bool
        and (not executing or release['execution_released']) and release['caps']==CAPS
        and release['output_directory']==OUTPUT and release['upstream']==UPSTREAM,'exact recovery release')
    core.need(release['declaration']=={'path':core.DECLARATION,'sha256':core.DECLARATION_SHA256},'same declaration')
    declaration=core.read_binding(root,release['declaration'],maximum=128*1024)
    core.need(declaration['alpha_candidates']==[2.,-8.] and declaration['fit']['sha256']==core.FIT_SHA
              and declaration['specimen_role']=='DEVELOPMENT','unchanged scientific scope')
    core.verify_sources(root,release,{'source_paths':declaration['source_paths']+[RUNNER]},committed=executing)
    sites=core.runtime(root,release)['site_packages'];core.need(len(sites)==1,'one locked site')
    sys.path.insert(0,sites[0])
    if executing:core.read_binding(root,release['storage_receipt'],maximum=1024**2)
    return core,helper,release,declaration


def admitted_inputs(root,core,declaration):
    """Authenticate retained decisions and references without recalculating screens."""
    prior=core.read_binding(root,UPSTREAM['terminal'],maximum=512*1024)
    core.need(prior['status']=='failed_shape_comparison' and prior['native_calls']==6
              and prior['positive_scale_profiles']==0 and prior['release']==UPSTREAM['release'],'original failed attempt')
    console=core.read_binding(root,UPSTREAM['console'],maximum=64*1024,decode=False)
    core.need(b'exclusive_json(out/\'comparison.json\'' in console and b'ValueError: output cap' in console,
              'exact final serialization failure')
    original=core.read_binding(root,UPSTREAM['release'],maximum=128*1024)
    core.need(original['declaration']=={'path':core.DECLARATION,'sha256':core.DECLARATION_SHA256},'original declaration')
    for name in [core.run_name(row) for row in core.RUNS]+['read-'+str(i) for i in range(6)]:
        s=prior['stages'][name];stage=s['receipt']['native_stage' if name.startswith(('compression-','tension-')) else 'readout_stage']
        c=s['cleanup']
        core.need(stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None
            and c['contained'] and c['direct_child_reaped'] and c['exit_code']==0 and not c['fallback_used']
            and c['remaining_members']==[] and c['errors']==[],'completed six solves and readouts')
    screen=core.read_binding(root,UPSTREAM['screen'],maximum=4*1024**2)
    core.need(screen['schema']=='hbe-ogden-two-grid-screen-v1' and screen['passed'] is True
        and set(screen['pairs'])=={'compression/mesh','compression/time','tension/mesh','tension/time'}
        and all(p['pairwise_limit_passed'] is True for p in screen['pairs'].values())
        and screen['physical_validation_pass'] is None,'retained passing numerical screen')
    request=core.read_binding(root,UPSTREAM['request'],maximum=128*1024)
    core.need(request['mode']=='profile' and set(request['reports'])=={core.run_name(row) for row in core.RUNS},'exact six reports')
    numerical={}
    for row in core.RUNS:
        name=core.run_name(row);binding=request['reports'][name]
        core.need(binding['path']==PRIOR+'/'+name+'/readout.json','same retained readout')
        numerical[name]=core.read_binding(root,binding,maximum=8*1024**2)
        core.need(numerical[name]['numerical_passed'] is True,'retained stream numerical pass')
    e=declaration['evidence'];comparison=core.read_binding(root,e['comparison'],maximum=16*1024**2)
    core.need(comparison['numerical_qualification_passed'] is True and comparison['physical_validation_pass'] is None
        and comparison['comparison_manifest']==e['manifest'],'accepted alpha-two comparison')
    manifest=core.read_binding(root,e['manifest'])
    pairs={b:comparison['all_adjacent_signed_differences'][b]['N16-N24'] for b in core.AXIAL}
    core.need(all(p['pairwise_limit_passed'] is True for p in pairs.values()),'retained alpha-two pair pass')
    refs={b:manifest['rows'][f'{b}:N24:S60:reference']['readout'] for b in core.AXIAL}
    old={b:core.read_binding(root,binding,maximum=8*1024**2) for b,binding in refs.items()}
    return numerical,old,pairs,request['reports'],refs


def worker(root,core,release,declaration,deadline):
    core.need(os.getpid()==os.getpgrp() and time.monotonic()<deadline<=time.monotonic()+CAPS['worker_seconds'],'owned deadline')
    def audit(event,args):
        if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp','socket.connect','socket.bind','socket.getaddrinfo'}:
            raise RuntimeError('Saved-only worker cannot spawn/network')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0])
            if p.endswith(('.csv','.mat','.tar','.zip','.tar.gz','.feb','.log')) or '/data/' in p:
                raise RuntimeError('No raw measurement/native primitive/runtime payload access')
    sys.addaudithook(audit)
    from scripts.mechanics_hbe_v5_n8_one_shot import durable_json
    numerical,old,pairs,reports,refs=admitted_inputs(root,core,declaration)
    out=core.local(root,OUTPUT)
    accounting={'prior_attempt':{'positive_scale_profiles':2,'baseline_reproductions':1,'completed_reports':0},
        'recovery_attempt':{'positive_scale_profiles':0,'baseline_reproductions':0},
        'cumulative':{'positive_scale_profiles':2,'baseline_reproductions':1},
        'count_basis':'calls entered; original traceback places failure after all three completed',
        'new_native_calls':0,'numerical_replays':0,'alpha_optimizer_calls':0,'new_measured_member_reads':0}
    durable_json(out/'arithmetic-accounting.json',accounting)
    def mark(kind):
        key='positive_scale_profiles' if kind=='profile' else 'baseline_reproductions'
        accounting['recovery_attempt'][key]+=1;accounting['cumulative'][key]+=1
        durable_json(out/'arithmetic-accounting.json',accounting)
    result=core.comparison_output(root,declaration,numerical,old,pairs,on_arithmetic=mark)
    result['saved_readout_recovery']={'upstream':UPSTREAM,'readouts':reports,'baseline_references':refs,'accounting':accounting,
        'numerical_screen_recomputed':False,'original_attempt_preserved':True}
    core.exclusive_json(out/'comparison.json',result,maximum=512*1024)
    # Recheck every consumed source byte; no second profiling or screen evaluation.
    for binding in [*UPSTREAM.values(),*reports.values(),*refs.values(),declaration['fit'],declaration['evidence']['comparison'],declaration['evidence']['manifest']]:
        core.read_binding(root,binding,maximum=16*1024**2,decode=False)
    core.audit_loaded(root,release);core.need(time.monotonic()<deadline,'worker terminal deadline')


def execute(root,core,helper,rb,release,declaration,*,started=None):
    core.need(release['execution_released'] is True,'separate root release')
    start=time.monotonic() if started is None else started;outer=start+CAPS['total_seconds'];deadline=outer-CAPS['cleanup_seconds']
    out=core.local(root,OUTPUT);out.mkdir(parents=True,exist_ok=False)
    record={'schema':'hbe-ogden-saved-recovery-terminal-v1','status':'reserved','release':rb,'upstream':UPSTREAM,
            'native_calls':0,'numerical_replays':0,'physical_validation_pass':None}
    core.exclusive_json(out/'intent.json',{'release':rb,'deadline':deadline,'caps':CAPS,'upstream':UPSTREAM})
    owner=None;stop={'requested':False};previous={}
    def interrupted(signum,frame):stop['requested']=True;raise KeyboardInterrupt('termination requested')
    for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,interrupted)
    try:
        from scripts import mechanics_hbe_v5_remaining_one_shot as stages
        from scripts import mechanics_hbe_v5_n8_one_shot as io
        from scripts import febio_runtime
        from scripts import mechanics_hbe_ogden_comparison_experiment as original
        from launchers.hbe_v5_ordinal9_continuation_v1 import OwnedStage
        bounded_core=SimpleNamespace(CAPS=CAPS,need=core.need,local=core.local)
        original.check_free_disk(bounded_core,out,reserve_outputs=True)
        cache=root/'build/hbe-ogden-recovery-cache-v1/unused-worker';helper.check_cache(root,cache)
        def recheck():
            core.need(core.read_binding(root,rb,maximum=128*1024)==release,'release unchanged')
            core.verify_sources(root,release,{'source_paths':declaration['source_paths']+[RUNNER]},committed=False)
            core.audit_loaded(root,release);helper.check_cache(root,sys.pycache_prefix);helper.check_cache(root,cache)
            core.need(not stop['requested'] and time.monotonic()<outer,'terminal budget/termination')
        def observed(pgid,*,timeout_seconds):
            original.check_free_disk(bounded_core,out)
            core.need(stages.active_bytes(out,CAPS['new_output_bytes'])<=CAPS['new_output_bytes'],'aggregate cap')
            return febio_runtime.process_group_rss(pgid,timeout_seconds=timeout_seconds)
        recheck();directory=out/'worker';directory.mkdir();receipt={'release':rb};owner=OwnedStage()
        command=[sys.executable,'-I','-S','-B','-X','pycache_prefix='+str(cache),str(root/RUNNER),
            '--root',str(root),'--release',rb['path'],'--release-sha256',rb['sha256'],'--worker','--deadline',repr(deadline)]
        try:
            stage=stages.supervise_stage('readout',command,directory,receipt,cwd=root,environment=io.private_environment(),
                wall_cap=deadline-time.monotonic(),rss_cap=CAPS['sampled_group_rss_bytes'],output_cap=CAPS['new_output_bytes'],
                rss_observer=observed,popen=owner)
        finally:
            cleanup=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
            record['supervision']={'receipt':receipt,'cleanup':cleanup}
        helper.require_stage(core,stage,cleanup);owner=None
        expected={'intent.json':64*1024,'comparison.json':512*1024,'arithmetic-accounting.json':16*1024,
                  'worker/receipt.json':64*1024,'worker/readout-console.txt':64*1024}
        sealed=helper.snapshot(root,bounded_core,out,expected)
        accounting=core.read_binding(root,{'path':OUTPUT+'/arithmetic-accounting.json','sha256':sealed['arithmetic-accounting.json']['sha256']})
        core.need(accounting['cumulative']=={'positive_scale_profiles':4,'baseline_reproductions':2},'exact recovery accounting')
        recheck();core.need(helper.snapshot(root,bounded_core,out,expected)==sealed,'late output mutation')
        record.update(status=DONE,accounting=accounting,output_inventory=sealed,elapsed_seconds=time.monotonic()-start)
        h=core.exclusive_json(out/'terminal.json',record);expected['terminal.json']=128*1024
        sealed['terminal.json']={'bytes':(out/'terminal.json').stat().st_size,'sha256':h}
        recheck();core.need(helper.snapshot(root,bounded_core,out,expected)==sealed,'terminal seal')
        h=core.exclusive_json(out/'publication.json',{'accepted':True,'release':rb,'outputs':sealed,'physical_validation_pass':None})
        expected['publication.json']=128*1024;sealed['publication.json']={'bytes':(out/'publication.json').stat().st_size,'sha256':h}
        recheck();core.need(helper.snapshot(root,bounded_core,out,expected)==sealed and time.monotonic()<outer,'publication seal')
    except BaseException as error:
        record.update(status='failed_saved_readout_recovery',error_type=type(error).__name__,error=str(error)[:2048])
    finally:
        try:
            if owner is not None:
                from scripts import febio_runtime
                record['fallback_cleanup']=owner.cleanup(febio_runtime.process_group_rss,min(outer,time.monotonic()+5))
            if record['status']!=DONE:
                core.exclusive_json(out/'finalization-failure.json',record)
                if not(out/'terminal.json').exists():core.exclusive_json(out/'terminal.json',record)
                if not(out/'publication.json').exists():core.exclusive_json(out/'publication.json',{'accepted':False,'release':rb})
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    return record


def main():
    start=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--release',required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float)
    a=p.parse_args();root=a.root.resolve()
    if a.execute and a.worker:raise ValueError('One operation')
    core,helper,release,decl=bootstrap(root,a.release,a.release_sha256,executing=a.execute or a.worker)
    if a.worker:worker(root,core,release,decl,a.deadline);return 0
    if not a.execute:print(json.dumps({'status':'source_preflight_only','native_calls':0,'measured_json_reads':0}));return 0
    result=execute(root,core,helper,{'path':a.release,'sha256':a.release_sha256},release,decl,started=start)
    print(json.dumps({'status':result['status'],'native_calls':0}));return 0 if result['status']==DONE else 1


if __name__=='__main__':raise SystemExit(main())
