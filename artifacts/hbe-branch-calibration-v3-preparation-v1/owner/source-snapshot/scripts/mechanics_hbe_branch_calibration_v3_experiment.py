#!/usr/bin/env python3
"""Two released fitted axial confirmations and conditional same-specimen torque evaluation."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import mechanics_hbe_branch_calibration_v3 as core

common,old,runtime,access=core.common,core.old,core.runtime,core.access
DONE='completed_descriptive_one_specimen_comparison'


def prepare(context,fit_binding):
    fit=access.verify_binding(context['root'],fit_binding,maximum_bytes=1024**2,read_json=True)
    mu=core.positive(fit['mu_Pa']);prepared={}
    for branch in core.AXIAL:
        target=context['directory']/'runs'/branch;target.mkdir(parents=True,exist_ok=False)
        skyline,deck,loading=core.fitted_contents(context['registry'],context['study'],branch,mu)
        for name,text in [('skyline.feb',skyline),('specimen.feb',deck)]:
            with (target/name).open('x') as stream:stream.write(text)
        old.saved(context['root'],target/'loading.json',loading)
        prepared[branch]={k:old.binding(context['root'],target/v) for k,v in
                          [('source','skyline.feb'),('deck','specimen.feb'),('loading','loading.json')]}
    return core.receipts.durable_json(context['root'],context['directory']/'preparation'/'prepared.json',
        {'schema':'hbe-branch-fitted-preparation-v3','fit':fit_binding,'mu_Pa':mu,'cases':prepared,
         'declaration_sha256':core.DECLARATION_SHA256,'native_calls':0,'mesher_calls':0})


def worker(root,release_binding,deadline):
    # Supervision already covers this preflight, imports, hashing and all later stages.
    study=core.declaration(root);directory=root/study['output_root']/'experiment'
    state={'schema':'hbe-branch-calibration-state-v3','status':'preflight','native_calls':0,'mesher_calls':0,
           'calibration_access_attempted':False,'calibration_responses_accessed':False,
           'held_out_access_attempted':False,'held_out_responses_accessed':False,'runs':{},
           'physical_validation_pass':None,'automatic_retry':False}
    runtime.write_json(directory/'state.json',state)
    try:
        context=core.preflight(root,release_binding,deadline=deadline);reg=context['registry'];caps=study['budgets']
        archive_bytes=access.local_path(root,context['release']['source_archive']['path']).stat().st_size
        observed_caps={'generated_output_bytes':caps['new_total_output_bytes']-archive_bytes,
                       'each_active_run_output_bytes':caps['active_output_bytes_by_branch']['compression']}
        if observed_caps['generated_output_bytes']<=0:raise ValueError('Archive exhausts new-output allowance')
        with old.OutputWatch(root/study['output_root'],directory/'output-watch.json',observed_caps) as watch:
            released=core.BranchReleasedStudy(context,deadline)
            state.update(status='reading_released_axial_calibration',calibration_access_attempted=True,
                         calibration_responses_accessed=None)
            runtime.write_json(directory/'state.json',state)
            calibration=released.read_calibration({b:study['csv_schemas'][b] for b in core.AXIAL})
            state['calibration_responses_accessed']=True
            references={b:core.curve_from_row(b,context['references'][b]) for b in core.AXIAL}
            fit=core.evaluation.fit_scale(calibration,references)
            state['fit']=core.receipts.durable_json(root,directory/'fit.json',fit)
            pred=core.predictions(context['references'],fit['scale'])
            state['predictions']=core.receipts.durable_json(root,directory/'predictions.json',pred)
            state['status']='preparing_two_fitted_decks';runtime.write_json(directory/'state.json',state)
            started=time.monotonic();subdeadline=min(deadline,started+60)
            prepdir=directory/'preparation';prepdir.mkdir(exist_ok=False);watch.active=prepdir
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),
                     '--release',release_binding['path'],'--release-sha256',release_binding['sha256'],
                     '--prepare-fit',state['fit']['path'],'--fit-sha256',state['fit']['sha256']]
            common.pure_child(command,prepdir,subdeadline,60)
            prep_binding=old.binding(root,prepdir/'prepared.json');prep=reg.bound(prep_binding)
            if (prep.get('schema')!='hbe-branch-fitted-preparation-v3' or prep.get('fit')!=state['fit']
                    or prep.get('mu_Pa')!=fit['mu_Pa'] or prep.get('declaration_sha256')!=core.DECLARATION_SHA256
                    or prep.get('native_calls')!=0 or prep.get('mesher_calls')!=0 or set(prep['cases'])!=set(core.AXIAL)):
                raise ValueError('Complete exact fitted deck preparation required')
            # Authenticate both actual decks before any native call, inside the60s allowance.
            for branch in core.AXIAL:
                target=directory/'runs'/branch
                for key,name in [('source','skyline.feb'),('deck','specimen.feb'),('loading','loading.json')]:
                    binding=prep['cases'][branch][key]
                    if access.local_path(root,binding['path'])!=target/name:raise ValueError('Prepared deck origin differs')
                    reg.bound(binding,json_value=key=='loading')
                skyline,deck,loading=core.fitted_contents(reg,study,branch,fit['mu_Pa'])
                if ((target/'skyline.feb').read_text()!=skyline or (target/'specimen.feb').read_text()!=deck
                        or reg.read(prep['cases'][branch]['loading'])!=loading):
                    raise ValueError('Prepared deck physics differs')
            common.remaining_seconds(subdeadline,60)
            state.update(preparation=prep_binding,preparation_seconds=time.monotonic()-started)
            fitted={}
            for branch in core.AXIAL:
                target=directory/'runs'/branch;watch.active=None
                observed_caps['each_active_run_output_bytes']=caps['active_output_bytes_by_branch'][branch];watch.active=target
                reg.verify_all(deadline)
                seconds=common.remaining_seconds(deadline,caps['native_seconds_by_branch'][branch])
                if state['native_calls']>=2:raise ValueError('Exactly two maximum native calls')
                state['native_calls']+=1;state['status']='solving_fitted_'+branch
                runtime.write_json(directory/'state.json',state)
                command=[context['context']['runtime']['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
                execution=old.solve(command,target,seconds)
                core.receipts.require_completed_native(execution,seconds)
                primitives={k:old.binding(root,target/v) for k,v in
                            [('deck','specimen.feb'),('loading','loading.json'),('nodes','nodes.log'),('elements','elements.log'),('solver','solver.log')]}
                q=study['axial_references'][branch];primitives['mesh']=q['native_primitives']['mesh']
                record={'schema':'hbe-branch-fitted-execution-v3','branch':branch,
                        'run_id':f'{branch}:N{q["counts"]["N"]}:S120:fitted','mu_Pa':fit['mu_Pa'],
                        'declaration_sha256':core.DECLARATION_SHA256,'release':release_binding,
                        'runtime_identity':study['runtime_identity'],'backend_profile':study['backend_profile'],
                        'backend_source_deck':prep['cases'][branch]['source'],'reconstruction':q['reconstruction'],
                        'primitive_bindings':primitives,'command':command,'execution':execution}
                fitted[branch]=core.receipts.durable_json(root,target/'execution.json',record)
                state['runs'][branch]=fitted[branch];runtime.write_json(directory/'state.json',state)
            watch.active=None
            state['status']='independently_checking_both_fitted_pairs';runtime.write_json(directory/'state.json',state)
            state['freeze']=released.freeze_predictions(fit_binding=state['fit'],prediction_binding=state['predictions'],
                fitted_runs=fitted,output_path=str((directory/'parameter-prediction-freeze.json').relative_to(root)))
            state.update(status='parameters_and_predictions_frozen_before_holdout',held_out_access_attempted=True,
                         held_out_responses_accessed=None);runtime.write_json(directory/'state.json',state)
            held=released.evaluate_held_out(freeze_binding=state['freeze'],schemas={b:study['csv_schemas'][b] for b in core.TORSION})
            state['held_out_responses_accessed']=True
            state['held_out_metrics']=core.receipts.durable_json(root,directory/'held-out-metrics.json',held)
            fitted_curves={b:core.evaluation.scale_prediction(c,fit['scale']) for b,c in references.items()}
            state['calibration_metrics']=core.receipts.durable_json(root,directory/'calibration-metrics.json',
                core.evaluation.paired_metrics(calibration,fitted_curves,characteristic_response_scale=fit['mu_Pa']*common.R**2))
            state['status']=DONE;reg.verify_all(deadline);common.remaining_seconds(deadline,3600)
            state['charged_source_archive_bytes']=archive_bytes
    except BaseException as error:
        state.update(status='failed_or_incomplete',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        runtime.write_json(directory/'state.json',state)


def finalize(root,directory,raw_root,result,started,archive_bytes):
    """Same accepted durable-publication accounting, with descriptive-study status."""
    cap,output_cap=3600,2*1024**3
    result['seconds_before_publication']=time.monotonic()-started
    if not 0<=result['seconds_before_publication']<cap:result['status']='failed_or_incomplete'
    rb=core.receipts.durable_json(root,directory/'result.json',result)
    retained=old.meshing.tree_bytes(raw_root)+archive_bytes;elapsed=time.monotonic()-started
    close={'schema':'hbe-branch-calibration-publication-v3','result':rb,'accepted':result['status']==DONE and elapsed<cap,
           'elapsed_through_result_fsync_and_scan_seconds':elapsed,'cap_seconds':cap,'output_cap_bytes':output_cap,
           'retained_including_source_archive_before_closeout':retained,'retained_including_closeout':0,
           'closeout_own_publication_inside_clock_claim':False,'hard_real_time_or_power_loss_guarantee':False}
    for _ in range(8):
        total=retained+len(access.canonical_json(close));accepted=close['accepted'] and total<=output_cap
        if total==close['retained_including_closeout'] and accepted==close['accepted']:break
        close.update(retained_including_closeout=total,accepted=accepted)
    else:raise RuntimeError('Publication accounting failed to settle')
    cb=core.receipts.durable_json(root,directory/'publication-check.json',close)
    return dict(result,result_binding=rb,publication=cb,status=result['status'] if close['accepted'] else 'failed_or_incomplete')


def launch(root,release_binding):
    started=time.monotonic();deadline=started+3600
    study=core.declaration(root);release=access.verify_binding(root,release_binding,maximum_bytes=1024**2,read_json=True)
    if (release.get('authorized') is not True or release.get('schema')!='hbe-branch-calibration-release-v3'
            or release.get('study')!={'path':core.DECLARATION_PATH,'sha256':core.DECLARATION_SHA256}):
        raise ValueError('Explicit branch calibration source release required')
    archive=release['source_archive'];access.verify_binding(root,archive)
    archive_bytes=access.local_path(root,archive['path']).stat().st_size
    raw_root=root/study['output_root'];raw_root.mkdir(parents=True,exist_ok=True)
    old.saved(root,raw_root/'.started.json',{'release':release_binding,'no_retry':True})
    directory=raw_root/'experiment';directory.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--release',release_binding['path'],
             '--release-sha256',release_binding['sha256'],'--worker','--deadline',repr(deadline)]
    env=runtime.private_environment({'caps':{'thread_environment':core.receipts.THREADS}})
    for key in list(env):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__','LD_PRELOAD','LD_LIBRARY_PATH'):env.pop(key)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    supervision=runtime.supervise(command,directory/'supervision',cwd=root,environment=env,
        seconds=common.remaining_seconds(deadline,3600),rss_bytes=3*1024**3)
    result={'schema':'hbe-branch-calibration-result-v3','release':release_binding,'supervision':supervision,
            'status':'failed_or_incomplete','physical_validation_pass':None,'charged_source_archive_bytes':archive_bytes}
    try:
        result['state']=old.binding(root,directory/'state.json');state=access.verify_binding(root,result['state'],read_json=True)
        if (supervision['status']=='completed' and state['status']==DONE and state['native_calls']==2
                and state['mesher_calls']==0 and state['preparation_seconds']<60
                and state['calibration_responses_accessed'] is True and state['held_out_responses_accessed'] is True):
            result['status']=DONE
    except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error)}
    return finalize(root,directory,raw_root,result,started,archive_bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--release',required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    p.add_argument('--prepare-fit',help=argparse.SUPPRESS);p.add_argument('--fit-sha256',help=argparse.SUPPRESS)
    args=p.parse_args();root=args.root.resolve();binding={'path':args.release,'sha256':args.release_sha256}
    if args.worker:
        if os.getpid()!=os.getpgrp():raise ValueError('Worker must lead supervised group')
        worker(root,binding,args.deadline)
    elif args.prepare_fit:
        if os.getpid()==os.getpgrp():raise ValueError('Preparation child must remain in supervised group')
        prepare(core.preflight(root,binding),{'path':args.prepare_fit,'sha256':args.fit_sha256})
    elif args.execute:
        if launch(root,binding)['status']!=DONE:raise SystemExit(1)
    else:
        core.preflight(root,binding,deadline=time.monotonic()+60)
        print('Exact metadata/source preflight passed; no curves, fit or solver executed.')


if __name__=='__main__':main()
