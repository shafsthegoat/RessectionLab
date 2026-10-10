"""Disabled by default: SELECT013 native-domain crop and saved-domain review.

Reuses the SELECT parent ownership/cleanup and source-geometry conversion path.
No case is admitted to training by this driver, even on conversion success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
SUBJECTS=('ReMIND-013',)
KINDS={'structural_t1ce','whole_tumor','cerebrum'}
CAPS={'seconds_per_case':300,'total_seconds':330,'sampled_rss_bytes':3*1024**3,
      'output_bytes_per_case':256*1024**2,'log_bytes_per_phase':4*1024**2,
      'threads':1,'automatic_retry':False}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,reason):
    if not value:raise ValueError(reason)
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
def terminal_result(path,phase,subject,case_sha,reader_sha):
    """An exit code alone never proves a complete, correctly bound QC result."""
    result={};result_hash=None
    try:
        need(path.is_file() and not path.is_symlink(),'missing_or_nonregular_result')
        need(0<path.stat().st_size<=16*1024**2,'terminal_result_extent')
        raw=path.read_bytes();result_hash=hashlib.sha256(raw).hexdigest();result=json.loads(raw)
        need(isinstance(result,dict),'terminal_result_object')
        status='saved_public_domain_geometry_checked_anatomy_unreviewed' if phase=='saved-domain-review' else 'public_crop_source_samples_and_label_grids_converted_anatomy_unreviewed'
        need(result.get('status')==status,'phase_success_status')
        need(result.get('case_sha256')==case_sha and result.get('executing_script_sha256')==reader_sha,'phase_source_case_binding')
        need(result.get('patient_id')==subject and result.get('role')=='SELECT','phase_patient_role_binding')
        need(result.get('public_only') is True and result.get('private_reference_loaded') is False and result.get('training_admitted') is False and result.get('optimizer_updates_performed')==0,'phase_public_nontraining_boundary')
        need(result.get('full_coverage_factory_compatible') is False,'full_coverage_factory_refusal')
        if phase=='crop-mr-domains':
            need(result.get('crop_policy')=='public_source_domain_union_no_extrapolation' and result.get('support_source_domain_required') is True,'explicit_domain_crop')
            need(all(result['annotations'][k]['placement']['source_positive_centres_outside_target_grid']==0 for k in ('cerebrum','whole_tumor')),'source_positive_crop_loss')
        else:
            need(result.get('public_input_manifest') is None and result.get('original_source_reopened') is False,'saved_review_boundary')
        return result,result_hash,None
    except BaseException as error:
        return result if isinstance(result,dict) else {},result_hash,'terminal_guard:'+type(error).__name__+':'+str(error)
def final_log_size(path):
    size=path.stat().st_size
    return size,'final_phase_log_cap' if size>CAPS['log_bytes_per_phase'] else None
def preflight(release,expected_sha):
    need(sha(release)==expected_sha,'release_hash')
    value=json.loads(release.read_bytes()); prepared=json.loads((HERE/'prepared-release.json').read_bytes())
    candidate=dict(value);candidate['execution_released']=False
    need(candidate==prepared and type(value['execution_released']) is bool,'only_execution_release_boolean_may_change')
    source_path=ROOT/value['source_index']['path'];need(sha(source_path)==value['source_index']['sha256'],'source_index_hash')
    source=json.loads(source_path.read_bytes())
    for record in source['files']:
        path=ROOT/record['path'];need(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'source_changed:'+record['path'])
    index_path=ROOT/value['public_index']['path'];need(sha(index_path)==value['public_index']['sha256'],'public_index_hash')
    index=json.loads(index_path.read_bytes())
    need(index['subjects']==list(SUBJECTS) and index['planned_denominator']==1 and value['caps']==CAPS,'fixed_batch_or_caps')
    cohort=json.loads((ROOT/'manifests/experiments/remind-component-cohort-v1.json').read_bytes())
    role_map={m['subject']:(m['role'],m['patient_group']) for m in cohort['members']}
    bound=[]
    for row in index['cases']:
        person=row['patient_id'];need(person in SUBJECTS and row['role']=='SELECT' and row['public_input_manifest'] is None and not row['training_admitted'],'unqualified_SELECT013_only')
        path=ROOT/row['case']['path'];need(sha(path)==row['case']['sha256'],'case_hash');case=json.loads(path.read_bytes())
        need((case['role'],case['patient_group'])==role_map[person] and case['patient_id']==person and case['role']=='SELECT','frozen_role')
        need(len(case['series'])==3 and {s['kind'] for s in case['series']}==KINDS,'exact_public_triple')
        parent=case['parent_bindings']['source_binding'];flat_path=Path(parent['path'])
        if not flat_path.is_absolute():flat_path=ROOT/flat_path
        need(sha(flat_path)==parent['sha256'],'flat_hash');flat=json.loads(flat_path.read_bytes())
        need(all(flat[k]==case[k] for k in ['patient_id','patient_group','role']),'flat_identity')
        nested={o['path']:{**o,'series_uuid':s['series_uuid']} for s in case['series'] for o in s['objects']}
        declared={o['path']:o for o in flat['objects']}
        need(nested==declared and len(nested)==len(flat['objects'])==row['objects']==flat['total_objects'],'exact_object_binding')
        need(sum(o['bytes'] for o in flat['objects'])==row['bytes']==flat['total_bytes'],'exact_byte_binding')
        header=ROOT/row['headers']['path'];need(sha(header)==row['headers']['sha256'],'frozen_header_hash')
        saved=json.loads(header.read_bytes());need(saved['case_sha256']==row['case']['sha256'] and saved['patient_id']==person and saved['public_only'] is True and saved['status']=='header_geometry_and_ancestry_projected','frozen_header_identity')
        bound.append((person,path,row['case']['sha256'],header,row['headers']['sha256']))
    need(tuple(row[0] for row in bound)==SUBJECTS,'fixed_order')
    need(sum(r['objects'] for r in index['cases'])==194 and sum(r['bytes'] for r in index['cases'])==33046694,'fixed_totals')
    return value,bound
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    release,bound=preflight(args.release.resolve(),args.release_sha256)
    if args.check_only:
        print(json.dumps({'status':'prepared_metadata_contract_valid','subjects':SUBJECTS,'objects':194,'bytes':33046694,'execution_released':release['execution_released'],'patient_source_files_opened':0,'workers_started':0}));return 0
    need(release['execution_released'] is True,'root_dispatch_required_before_any_patient_source_or_worker')
    # Exact helper sources have been checked above. Import only after dispatch.
    sys.path.insert(0,str(ROOT/'build/goal-conditioned-policy-v1'))
    from run_contact_owned import FastDarwinSampler,cleanup_owned,_receipt
    output=HERE/'actual-domain-qc-v1';output.mkdir(exist_ok=False)
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    for key in ('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','LD_PRELOAD','DYLD_LIBRARY_PATH'):env.pop(key,None)
    runtime=ROOT/'build/idc-acquisition-venv/bin/python'
    reader_sha=sha(HERE/'stage/resectionlab/remind_planning_qc.py'); review_sha=sha(HERE/'audit_saved_domains.py')
    _receipt(output/'declaration.json',{'release_path':str(args.release.resolve()),'release_sha256':args.release_sha256,'source_index':release['source_index'],'public_index':release['public_index'],'parent_sha256':sha(__file__),'caps':CAPS,'runtime':str(runtime),'public_only':True,'policy_or_training':False,'internal_crop_cap':'existing110s/2GiB checkpoints retained','scope':'native domain-preserving crop plus saved domain/count audit only; no factory manifest; root visual/intended-use review separate'})
    sampler=FastDarwinSampler();started=time.monotonic();outcomes=[];fatal=None
    def stop(*_):raise InterruptedError('public SELECT013 QC parent interrupted')
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        for subject,case,expected,header,header_sha in bound:
            case_started=time.monotonic();case_paths=[]
            for phase in ('crop-mr-domains','saved-domain-review'):
                target=output/(subject+'-'+phase)
                if phase=='crop-mr-domains':
                    command=[str(runtime),'-I','-B','-X','pycache_prefix='+str(output/'fresh-pycache'),str(HERE/'prepare_expanded.py'),phase,'--public-only','--case',str(case),'--case-sha256',expected,'--repository-root',str(ROOT),'--output',str(target),'--headers',str(header),'--headers-sha256',header_sha]
                else:
                    conversion=output/(subject+'-crop-mr-domains/conversion-result.json')
                    command=[str(runtime),'-I','-B','-X','pycache_prefix='+str(output/'fresh-pycache'),str(HERE/'audit_saved_domains.py'),'--subject',subject,'--conversion-sha256',sha(conversion),'--output',str(target)]
                process=None;peak=0;reason=None;actions=[];errors=[];detached={};remaining=[]
                phase_started=time.monotonic();case_paths.append(target);log_path=output/(subject+'-'+phase+'.log')
                try:
                    with log_path.open('x') as log:
                        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        while process.poll() is None:
                            if time.monotonic()-case_started>=CAPS['seconds_per_case']:reason='case_wall_cap';break
                            if time.monotonic()-started>=CAPS['total_seconds']:reason='batch_wall_cap';break
                            total=sum(p.stat().st_size for folder in case_paths if folder.exists() for p in folder.rglob('*') if p.is_file())
                            if total>CAPS['output_bytes_per_case'] or os.fstat(log.fileno()).st_size>CAPS['log_bytes_per_phase']:reason='output_cap';break
                            measured=sampler.group(process.pid,process.pid);detached.update(measured['detached_descendant_start_identities']);peak=max(peak,measured['process_group_resident_bytes'])
                            if peak>CAPS['sampled_rss_bytes']:reason='sampled_memory_cap';break
                            time.sleep(.1)
                except BaseException as error:
                    reason=reason or 'phase_parent_error:'+type(error).__name__+':'+str(error)
                finally:
                    if process is not None:remaining=cleanup_phase(process,sampler,detached,actions,errors,cleanup_owned)
                    total=log_bytes=None
                    try:
                        total=sum(p.stat().st_size for folder in case_paths if folder.exists() for p in folder.rglob('*') if p.is_file())
                        if total>CAPS['output_bytes_per_case']:reason=reason or 'output_cap'
                        log_bytes,log_failure=final_log_size(log_path)
                        reason=reason or log_failure
                    except BaseException as error:
                        reason=reason or 'final_size_guard:'+type(error).__name__+':'+str(error)
                    if time.monotonic()-case_started>=CAPS['seconds_per_case']:reason=reason or 'case_wall_cap'
                    if time.monotonic()-started>=CAPS['total_seconds']:reason=reason or 'batch_wall_cap'
                    result_path=target/('saved-domain-review.json' if phase=='saved-domain-review' else 'conversion-result.json')
                    result,result_hash,result_error=terminal_result(result_path,phase,subject,expected,review_sha if phase=='saved-domain-review' else reader_sha)
                    if result_error is None:
                        try:
                            need(result['header_snapshot_sha256']==header_sha,'result_frozen_header_binding')
                            if phase=='saved-domain-review':need(result['conversion_receipt_sha256']==sha(conversion),'saved_review_conversion_binding')
                        except BaseException as error:result_error='terminal_link:'+type(error).__name__+':'+str(error)
                    reason=reason or result_error
                    code=None
                    try:code=None if process is None else process.poll()
                    except BaseException as error:errors.append('final_child_poll_error:'+str(error))
                    phase_complete=reason is None and code==0 and not errors and not remaining
                    record={'subject':subject,'role':'SELECT','phase':phase,'argv':command,'elapsed_seconds':time.monotonic()-phase_started,'case_elapsed_seconds':time.monotonic()-case_started,'sampled_peak_rss_bytes':peak,'case_output_bytes':total,'final_phase_log_bytes':log_bytes,'exit_code':code,'stop_reason':reason,'cleanup_actions':actions,'cleanup_errors':errors,'remaining_owned_pids':remaining,'reaped':process is not None and code is not None,'result_sha256':result_hash,'result_status':result.get('status'),'result_guard_error':result_error,'phase_complete':phase_complete,'intended_use_admitted':False}
                    outcomes.append(record);_receipt(output/(subject+'-'+phase+'-supervisor.json'),record)
                if reason or errors or remaining:fatal=reason or 'cleanup_failure';break
                if record['exit_code']!=0 or result.get('status')=='failed':break
            if fatal:break
    except BaseException as error:fatal=type(error).__name__+':'+str(error)
    finally:
        _receipt(output/'receipt.json',{'outcomes':outcomes,'elapsed_seconds':time.monotonic()-started,'fatal':fatal,'roles_changed':False,'public_only':True,'caps':CAPS,'memory_limit':'sampled, not kernel hard','planned_subjects':SUBJECTS,'failed_cases_replaced':False,'training_admitted':False,'automatic_source_artifact_clearance':False})
        _receipt(output/'pending-qualification-index.json',{'planned_denominator':1,'cases':[{'patient_id':p,'role':'SELECT','public_input_manifest':None,'training_admitted':False,'status':'DOMAIN_SOURCE_RESULT_REQUIRES_ROOT_VISUAL_AND_INTENDED_USE_REVIEW' if any(r['subject']==p for r in outcomes) else 'NOT_RUN_RETAINED_IN_DENOMINATOR'} for p in SUBJECTS],'mismatch_policy':'Preserve native/placed full-target counts, known-positive/known-zero/unknown support partition and source-domain masks. No factory manifest, target clipping, support filling or replacement.'})
        signal.signal(signal.SIGTERM,previous)
    return 0 if fatal is None and len(outcomes)==2 and all(r['phase_complete'] for r in outcomes) else 1
if __name__=='__main__':raise SystemExit(main())
