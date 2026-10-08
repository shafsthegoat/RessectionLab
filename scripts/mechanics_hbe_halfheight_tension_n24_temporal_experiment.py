#!/usr/bin/env python3
"""Released one-case N24 tension temporal diagnostic; zero remeshing, no retry."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import mechanics_hbe_halfheight_tension_n24_temporal as case
from scripts import mechanics_hbe_halfheight_temporal_common as common

receipts=common.receipts
old, access, backend, runtime=receipts.old,receipts.access,receipts.backend,receipts.runtime
NEW={'halfheight_temporal_common':common,'halfheight_tension_n24_temporal':case}
RUNNER_KEY='halfheight_tension_n24_temporal_runner'


def source_inventory(root,release,study,bound):
    sources=release['source_bindings']
    if set(sources)!=set(study['inherited_source_sha256'])|set(NEW)|{RUNNER_KEY}:
        raise ValueError('Exact24-module tension source closure required')
    wanted={}
    for key,binding in sources.items():
        path=access.local_path(root,binding['path'])
        if key in study['inherited_source_sha256']:
            if binding['sha256']!=study['inherited_source_sha256'][key] or path.name!=study['inherited_source_filenames'][key]:
                raise ValueError('Inherited helper changed')
            module=sys.modules.get('scripts.'+path.stem)
            actual=None if module is None else Path(module.__file__).resolve()
        else:actual=Path(__file__).resolve() if key==RUNNER_KEY else Path(NEW[key].__file__).resolve()
        if path!=actual:raise ValueError('Imported source origin differs')
        bound(binding,json_value=False,maximum_bytes=1024**2)
        wanted['scripts/'+path.name]=binding['sha256']
    for binding in study['inherited_declarations']+[release['study']]:
        bound(binding)
        wanted[binding['path']]=binding['sha256']
    receipts.verify_archive(root,release['source_archive'],release['source_commit'],wanted,32)
    bound(release['source_archive'],json_value=False)


def baseline_metadata(root,study,context,bound):
    """Accept the verified tension case while preserving its study's global failure."""
    b=study['baseline']
    review,result,state,execution,readout,comparison=(bound(b[k]) for k in
        ('independent_review','result','state','execution','baseline_readout','comparison'))
    if (review.get('whole_record_mismatches')!=0 or review.get('readout_records_reproduced')!=8
            or review.get('states_reproduced')!=488 or review.get('comparison_reproduced') is not True
            or review.get('result_sha256')!=b['result']['sha256'] or review.get('comparison_sha256')!=b['comparison']['sha256']
            or review.get('source_commit')!=b['source_commit'] or review.get('accepted_spatial_convergence') is not False
            or review.get('failed_criteria')!=['compression:N12-N16-N24/reaction_trend']
            or result.get('status')!='failed_or_incomplete' or result.get('state')!=b['state']
            or result.get('baseline')!=b['execution_baseline'] or result.get('measured_data_accessed') is not False
            or result['supervision'].get('exit_code')!=1 or result['supervision'].get('kill_reason') is not None
            or result['supervision'].get('cleanup_error') is not None
            or comparison.get('passed') is not False or state.get('solver_invocations')!=3
            or state.get('gmsh_generation_calls')!=0):
        raise ValueError('Exact independently verified prior failure required')
    row=state['runs'][b['run_id']]
    if (row.get('status')!='passed_individual_numerical_checks' or row.get('readout')!=b['baseline_readout']
            or row.get('input_bindings')!=b['primitive_bindings'] or execution.get('execution')!=row['execution']
            or execution.get('schema')!='hbe-halfheight-spatial-run-execution-v1'
            or execution.get('run_id')!=b['run_id'] or execution.get('study')!=b['declaration']
            or execution.get('primitive_bindings')!=b['primitive_bindings']
            or execution.get('reconstruction')!=b['reconstruction']
            or execution.get('backend_profile')!=b['backend_profile']
            or execution.get('runtime_identity')!=context['runtime_identity']
            or readout.get('passed') is not True or readout.get('execution_binding')!=b['execution']
            or readout.get('primitive_bindings')!=b['primitive_bindings']
            or readout.get('reconstruction')!=b['reconstruction']):
        raise ValueError('Accepted individual N24 tension origin differs')
    receipts.require_completed_native(execution['execution'],420)
    baseline,release=bound(b['execution_baseline']),bound(b['release'])
    bound(b['original_release'])
    if (release.get('authorized') is not True or release.get('phase')!='solve'
            or release.get('study')!=b['declaration']
            or any(release.get(k)!=b[k] for k in ('source_bindings','source_archive','source_commit','interpreter','backend_profile'))
            or b['original_release']['sha256']!=b['release']['sha256']
            or baseline.get('release_binding')!=b['original_release']
            or baseline.get('source_bindings')!=b['source_bindings']
            or baseline.get('runtime_identity')!=context['runtime_identity']):
        raise ValueError('Historical source/runtime release differs')
    old_study=bound(b['declaration']);case.spatial.require_study(old_study)
    wanted={}
    for key,binding in b['source_bindings'].items():
        if binding['sha256']!=study['inherited_source_sha256'][key]:raise ValueError('Historical helper changed')
        bound(binding,json_value=False,maximum_bytes=1024**2)
        wanted['scripts/'+Path(binding['path']).name]=binding['sha256']
    for binding in [old_study[k] for k in ('original_protocol','original_resolution_declaration','original_halfheight_declaration')]+[b['declaration']]:
        wanted[binding['path']]=binding['sha256']
    receipts.verify_archive(root,b['source_archive'],b['source_commit'],wanted,18)
    bound(b['source_archive'],json_value=False)
    prep=bound(b['preparation'])
    if prep.get('status')!='prepared_not_solved':raise ValueError('Accepted old preparation required')
    for key,binding in b['primitive_bindings'].items():bound(binding,json_value=False,maximum_bytes=256*1024**2)
    for key in ('full_mesh','reconstruction'):bound(b[key],json_value=False)
    bound(execution['backend_source_deck'],json_value=False)
    backend.verify_deck(access.local_path(root,execution['backend_source_deck']['path']).read_bytes(),
                        access.local_path(root,b['primitive_bindings']['deck']['path']).read_bytes())
    for binding in b['older_tension_readouts_for_sensitivity'].values():bound(binding)


def preflight(root,study_binding,release_binding):
    root,inputs=Path(root).resolve(),{}
    def bound(binding,*,json_value=True,maximum_bytes=16*1024**2):
        value=access.verify_binding(root,binding,maximum_bytes=maximum_bytes,read_json=json_value)
        inputs[str(access.local_path(root,binding['path']))]=binding['sha256'];return value
    study=case.declaration(root,study_binding);bound(study_binding)
    release=bound(release_binding)
    if (release.get('schema')!='hbe-tension-n24-temporal-release-v1' or release.get('authorized') is not True
            or release.get('phase')!='solve' or release.get('study')!=study_binding):
        raise ValueError('Explicit separate tension temporal release required')
    source_inventory(root,release,study,bound)
    b=study['baseline']
    if release.get('interpreter')!=b['interpreter'] or release.get('backend_profile')!=b['backend_profile']:
        raise ValueError('Original interpreter/backend required')
    python=Path(sys.executable).resolve()
    if python!=Path(b['interpreter']['path']).resolve() or runtime.sha(python)!=b['interpreter']['sha256']:
        raise ValueError('Executing interpreter differs')
    inputs[str(python)]=b['interpreter']['sha256']
    context=backend.verify_profile(root,b['backend_profile']);inputs.update(context['inputs'])
    if context['runtime_identity']!=b['runtime_identity']:raise ValueError('Runtime identity differs')
    proposal=bound(study['reviewed_proposal'])
    if proposal.get('authorized') is not False or proposal.get('case')!=study['case']:
        raise ValueError('Exact prospective single case required')
    bound(b['scale_homogeneity_note'],json_value=False)
    for binding in study['preserved_failures'].values():bound(binding)
    baseline_metadata(root,study,context,bound)
    return {'root':str(root),'study':study,'study_binding':study_binding,'release_binding':release_binding,
        'inputs':inputs,'source_bindings':release['source_bindings'],'executable':context['runtime']['executable'],
        'runtime_identity':context['runtime_identity'],'backend_profile':b['backend_profile'],
        'output_root':str(root/study['output_root']),'directory':str(root/study['output_root']/'experiment')}


def prepare(root,plan):
    directory=Path(plan['directory'])/'preparation'
    protocol,extraction,_,_=case.geometry(root,plan['study'])
    skyline,deck,loading=case.contents(root,plan['study'],protocol,extraction)
    for filename,content in (('skyline.feb',skyline),('specimen.feb',deck)):
        with (directory/filename).open('x') as stream:stream.write(content)
    old.saved(root,directory/'loading.json',loading);old.recheck(plan['inputs'])
    prepared={k:old.binding(root,directory/f) for k,f in
              (('deck','specimen.feb'),('loading','loading.json'),('backend_source_deck','skyline.feb'))}
    prepared.update(mesh=plan['study']['baseline']['primitive_bindings']['mesh'],reconstruction=plan['study']['baseline']['reconstruction'])
    receipts.durable_json(root,directory/'prepared.json',{'schema':'hbe-tension-n24-pure-deck-v1',
        'study':plan['study_binding'],'run_id':case.RUN_ID,'case':prepared,'solver_calls':0,'gmsh_generation_calls':0})


def worker(root,plan,baseline_binding,deadline):
    directory=Path(plan['directory']);caps=plan['study']['budgets'];b=plan['study']['baseline']
    state={'schema':'hbe-tension-n24-temporal-state-v1','status':'running','run_id':case.RUN_ID,
           'solver_invocations':0,'gmsh_generation_calls':0,'measured_data_accessed':False}
    runtime.write_json(directory/'state.json',state)
    try:
        with old.OutputWatch(plan['output_root'],directory/'output-watch.json',caps) as watch:
            started=time.monotonic();subdeadline=min(deadline,started+60)
            target=directory/'preparation';target.mkdir(exist_ok=False);watch.active=target
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),
                '--prepare-baseline',baseline_binding['path'],'--baseline-sha256',baseline_binding['sha256']]
            common.pure_child(command,target,subdeadline,60)
            prepared_binding=old.binding(root,target/'prepared.json');record=access.verify_binding(root,prepared_binding,read_json=True)
            if (record.get('schema')!='hbe-tension-n24-pure-deck-v1' or record.get('study')!=plan['study_binding']
                    or record.get('run_id')!=case.RUN_ID or record.get('solver_calls')!=0 or record.get('gmsh_generation_calls')!=0):
                raise ValueError('Complete pure preparation receipt required')
            prepared=record['case']
            if set(prepared)!={'mesh','reconstruction','deck','loading','backend_source_deck'}:
                raise ValueError('Exact prepared case required')
            for key,filename in (('deck','specimen.feb'),('loading','loading.json'),('backend_source_deck','skyline.feb')):
                if access.local_path(root,prepared[key]['path'])!=target/filename:raise ValueError('Prepared input origin differs')
                access.verify_binding(root,prepared[key],maximum_bytes=16*1024**2)
            if prepared['mesh']!=b['primitive_bindings']['mesh'] or prepared['reconstruction']!=b['reconstruction']:
                raise ValueError('Retained geometry differs')
            common.remaining_seconds(subdeadline,60)
            state.update(preparation=prepared_binding,preparation_elapsed_seconds=time.monotonic()-started)
            target=directory/'run';target.mkdir(exist_ok=False);watch.active=target
            for key,filename in (('deck','specimen.feb'),('loading','loading.json')):
                with (target/filename).open('xb') as stream:stream.write(access.local_path(root,prepared[key]['path']).read_bytes())
                if runtime.sha(target/filename)!=prepared[key]['sha256']:raise ValueError('Copied solver input differs')
            primitives={'mesh':prepared['mesh'],'deck':old.binding(root,target/'specimen.feb'),'loading':old.binding(root,target/'loading.json')}
            old.recheck(plan['inputs']);seconds=common.remaining_seconds(deadline,caps['each_solver_seconds'])
            if state['solver_invocations']!=0:raise ValueError('Only one native call permitted')
            state['solver_invocations']=1;runtime.write_json(directory/'state.json',state)
            command=[plan['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
            state['native_execution']=old.solve(command,target,seconds)
            receipts.require_completed_native(state['native_execution'],seconds)
            primitives.update({key:old.binding(root,target/filename) for key,filename in
                               (('nodes','nodes.log'),('elements','elements.log'),('solver','solver.log'))})
            execution=old.saved(root,target/'execution.json',{'schema':'hbe-tension-n24-native-execution-v1',
                'run_id':case.RUN_ID,'study':plan['study_binding'],'runtime_identity':plan['runtime_identity'],
                'backend_profile':plan['backend_profile'],'backend_source_deck':prepared['backend_source_deck'],
                'executable':old.binding(root,plan['executable']),'primitive_bindings':primitives,
                'reconstruction':b['reconstruction'],'command':command,'execution':state['native_execution']})
            report=case.read_run(root,plan['study_binding'],primitives);report['execution_binding']=execution
            state['readout']=old.saved(root,target/'readout.json',report)
            if report['passed'] is not True:raise ValueError('Individual numerical gate failed')
            rows={int(N):access.verify_binding(root,binding,read_json=True) for N,binding in b['older_tension_readouts_for_sensitivity'].items()}
            rows[24]=access.verify_binding(root,b['baseline_readout'],read_json=True)
            comparison=case.compare(rows,report,plan['study'],access.verify_binding(root,b['comparison'],read_json=True))
            state['comparison']=old.saved(root,directory/'comparison.json',comparison)
            if comparison['original_temporal_checks_passed'] is not True:raise ValueError('Original temporal gate failed; evidence retained')
            state['status']='completed_numerical_diagnostic_only'
        old.recheck(plan['inputs']);common.remaining_seconds(deadline,caps['aggregate_specimen_seconds'])
    except BaseException as error:
        state.update(status='failed_or_incomplete',error={'type':type(error).__name__,'message':str(error)});raise
    finally:runtime.write_json(directory/'state.json',state)


def launch(root,study_binding,release_binding):
    started=time.monotonic();plan=preflight(root,study_binding,release_binding)
    caps=plan['study']['budgets'];deadline=started+caps['aggregate_specimen_seconds']
    directory=Path(plan['directory']);raw_root=Path(plan['output_root'])
    common.remaining_seconds(deadline,600);raw_root.mkdir(parents=True,exist_ok=True)
    old.saved(root,raw_root/'.solve-started.json',{'study':study_binding,'release':release_binding,'no_retry':True})
    directory.mkdir(exist_ok=False);baseline=old.saved(root,directory/'baseline.json',plan)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--worker-baseline',baseline['path'],
             '--baseline-sha256',baseline['sha256'],'--deadline',repr(deadline)]
    env=runtime.private_environment({'caps':{'thread_environment':receipts.THREADS}})
    for key in list(env):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__','LD_PRELOAD','LD_LIBRARY_PATH'):env.pop(key)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    supervision=runtime.supervise(command,directory/'supervision',cwd=root,environment=env,
        seconds=common.remaining_seconds(deadline,600),rss_bytes=caps['sampled_process_family_rss_bytes'])
    result={'schema':'hbe-tension-n24-temporal-result-v1','phase':'solve','baseline':baseline,'supervision':supervision,
            'status':'failed_or_incomplete','aggregate_cap_seconds':600,'preparation_nested_in_aggregate':True,'measured_data_accessed':False}
    try:
        old.recheck(plan['inputs']);result['state']=old.binding(root,directory/'state.json')
        state=access.verify_binding(root,result['state'],read_json=True)
        if (supervision['status']=='completed' and state['status']=='completed_numerical_diagnostic_only'
                and state.get('solver_invocations')==1 and state.get('gmsh_generation_calls')==0
                and 0<=state.get('preparation_elapsed_seconds',math.inf)<60):result['status']=state['status']
    except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error)}
    return receipts.finalize_result(root,directory,raw_root,result,started,600,caps['generated_output_bytes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--release');parser.add_argument('--release-sha256');parser.add_argument('--execute',action='store_true')
    for arg in ('worker-baseline','prepare-baseline','baseline-sha256'):parser.add_argument('--'+arg,help=argparse.SUPPRESS)
    parser.add_argument('--deadline',type=float,help=argparse.SUPPRESS);args=parser.parse_args();root=args.root.resolve()
    if args.worker_baseline or args.prepare_baseline:
        if args.worker_baseline and args.prepare_baseline:raise ValueError('Ambiguous private mode')
        binding={'path':args.worker_baseline or args.prepare_baseline,'sha256':args.baseline_sha256}
        plan=access.verify_binding(root,binding,read_json=True)
        if access.canonical_json(plan)!=access.canonical_json(preflight(root,plan['study_binding'],plan['release_binding'])):
            raise ValueError('Source/input plan changed')
        if args.worker_baseline:
            if os.getpgrp()!=os.getpid():raise ValueError('Worker must lead supervised group')
            worker(root,plan,binding,args.deadline)
        else:
            if os.getpgrp()==os.getpid():raise ValueError('Pure child must remain in supervised group')
            prepare(root,plan)
        return
    study={'path':case.DECLARATION_PATH,'sha256':case.DECLARATION_SHA256};release={'path':args.release,'sha256':args.release_sha256}
    if args.execute:
        if launch(root,study,release)['status']=='failed_or_incomplete':raise SystemExit(1)
    else:preflight(root,study,release);print('Metadata verified; no preparation/native solve executed.')


if __name__=='__main__':main()
