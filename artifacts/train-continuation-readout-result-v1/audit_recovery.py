"""Terminal-only scalar audit of the cache-only twelve-route recovery.

No numpy/torch, feature archive, head, checkpoint, patient array or native call.
Reuses exact prior independent target/choice/history code rather than inventing
a second mathematical definition. Root must supply final three SHA256 pins.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
OLD=ROOT/'build/train-continuation-value-v1/independent'
DEPENDENCIES={
 'audit_saved.py':'ed744949bffe8f64607242b5e87255397445d0344211349e0501a1b5209cab31',
 'recorded_target_checks.py':'7a7e9b39c0fc9f03b7e92b250e67dbcd3e2d47d44beac4dc20b2eeeba082d977',
 'guarded_choice_checks.py':'41908c2088893c48b5978e42ad1699a95f9685aa8f5e062361edfe92fbd0a59a'}
for name,digest in DEPENDENCIES.items():
    if hashlib.sha256((OLD/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Prior independent code changed: '+name)
spec=importlib.util.spec_from_file_location('prior_independent_saved_audit',OLD/'audit_saved.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
need,close,read_json,semantic=base.need,base.close,base.read_json,base.semantic
SUBJECTS=base.SUBJECTS
METHODS=('GREEDY','Q64','CALIBRATED')
ORIGINAL='build/train-continuation-value-v1/run-01'
ORIGINAL_SHA='9c8c9534151e4287883f33dd530a9b64e02a1e70929a3d12e2bff4ae731e53ee'

def relative_close(a,b,label,atol=1e-10,rtol=1e-10):
    close(a,b,label,tol=atol+rtol*max(abs(a),abs(b)))

def check_corpus(report,index,output,original):
    manifest=read_json(output/'trace-manifest.json')
    need(semantic(manifest)==report['trace_manifest_hash']==original['trace_manifest_hash'],
         'Identical original fixed36 manifest')
    declared={r['key']:r for r in manifest}
    items=report['reconstructed_traces']
    need(len(items)==36 and len({r['key'] for r in items})==36,'36 unique reconstructed traces')
    rows=[]
    for item in items:
        key,subject,family=item['key'],item['subject'],item['kind']
        update=None if family=='teacher' else int(key.rsplit('RL',1)[1])
        folder=base.RL+'/teachers/'+subject if family=='teacher' else base.RL+'/RL/update-%02d/'%update+subject
        evidence=[read_json(ROOT/(folder+'/'+name),index['input_metadata'][folder+'/'+name])
                  for name in ('complete-trace.json','plan.json','native-replay.json')]
        row=base.validate_saved_episode(*evidence,subject=subject,family=family,update=update)
        rows.append(row)
        need(item['states']==row['steps'] and item['historical_trace_seal']==row['trace_seal']
             and item['historical_plan_seal']==row['plan_seal'],'Reconstruction original evidence joins')
        need(item['descriptor_hash']==semantic(declared[key])
             and item['history_hash']==declared[key]['history_hash'],'Reconstructed manifest join')
        need(item['original_trace_sha256']==index['input_metadata'][folder+'/complete-trace.json']
             and item['original_replay_sha256']==index['input_metadata'][folder+'/native-replay.json'],
             'Exact original trace/replay files')
        base.valid_geometry(item['audit'],key)
        need(item['audit']['committed_history_hash']==evidence[2]['independent_geometry']['committed_history_hash'],
             'Exact independently accepted reconstructed history')
    summary=base.validate_corpus(rows)
    logged=read_json(output/'logged-targets.json')
    old_logged=read_json(ROOT/(ORIGINAL+'/logged-targets.json'),index['input_metadata'][ORIGINAL+'/logged-targets.json'])
    for key in ('target_formula','targets','weights','STOP_rows','manifest_hash'):
        need(logged[key]==old_logged[key],'Original logged scalar field exactly retained: '+key)
    need(old_logged['feature_array_not_saved'] is True and logged['feature_array_not_saved'] is False
         and logged['feature_cache']==report['cache']['file']
         and logged['feature_cache_sha256']==report['cache']['sha256'],
         'Only newly materialized authenticated feature-cache metadata differs')
    need(len(logged['targets'])==len(logged['weights'])==len(logged['STOP_rows'])==497,'497 complete rows')
    pos=0; subject_rows={s:[] for s in SUBJECTS}
    for row in rows:
        for label in row['labels']:
            base.check_target_rounding(logged['targets'][pos],label['mandatory_logged_action_return'])
            need(logged['weights'][pos]==base.f32(1./(4*9*row['steps'])),'Original equal subject/trace/state weight')
            need(logged['STOP_rows'][pos] is (label['action_id']=='STOP'),'STOP0 rows retained')
            subject_rows[row['subject']].append(pos);pos+=1
    corpus=report['corpus']
    need(corpus['states']==pos==497 and corpus['STOP_rows']==sum(logged['STOP_rows'])
         and corpus['costs']['policy_forwards_completed']==corpus['costs']['policy_forwards_started']==497,
         'All497 authenticated feature forwards charged')
    return summary,manifest,logged,subject_rows

def check_cache(report,index,output,manifest):
    cache=report['cache'];prior=report['prior_cache_attempt'];folder=prior['directory']
    need(prior['failure_preserved'] and prior['status']=='failed_or_incomplete'
         and report['reconstruction_executed_this_attempt'] is False and report['cache_copied'] is False,
         'Original failed cache attempt retained; no reconstruction/copy in recovery')
    origin_report=read_json(ROOT/(folder+'/report.json'),prior['pins']['report.json'])
    origin_parent=read_json(ROOT/(folder+'/launcher.json'),prior['pins']['launcher.json'])
    prior_audit=read_json(HERE/'failure-audit-01.json','37d3be700f1994d317b46b099c44ef3fb7a2e14f6db8e48f68679e66ddbd014e')
    need(prior_audit['recovery_point']=='PASS_SAVED_CACHE_ORIGIN_AND_BYTES'
         and prior_audit['report_sha256']==prior['pins']['report.json']
         and prior['source_index_sha256']==origin_report['source_index_sha256'],
         'Independent recovery-point audit exact origin')
    need(origin_report['failure']==prior['failure'] and origin_report['head_optimizer_updates']==0
         and origin_report['cases']=={},'Unchanged failed original experiment')
    need(report['corpus']==origin_report['corpus'] and report['corpus_seal']==origin_report['corpus_seal']
         and report['reconstructed_traces']==origin_report['reconstructed_traces'],
         'Reuse exact already independently reconstructed corpus, no invented reconstruction')
    binding=read_json(ROOT/cache['binding_file'],cache['binding_sha256'])
    evidence=read_json(ROOT/cache['evidence_file'],cache['evidence_sha256'])
    receipt=read_json(ROOT/(folder+'/cache-receipt.json'),prior['pins']['cache-receipt.json'])
    need(receipt['binding']==binding and receipt['sha256']==cache['sha256']==prior['pins']['authenticated-features.zip']
         and receipt['bytes']==cache['bytes']==prior_audit['cache']['bytes'],
         'Exact independently checksummed cache and external binding')
    need(cache['file']==folder+'/authenticated-features.zip' and cache['copied_this_attempt'] is False
         and cache['reconstruction_paid_once'] and not cache['pixels_or_base_weights_copied'],
         'Explicit read-only prior cache path')
    metadata={**report['corpus'],'manifest':manifest}
    need(binding['metadata_hash']==semantic(metadata) and binding['corpus_seal']==report['corpus_seal']
         and semantic({'metadata':metadata,**evidence['arrays']})==report['corpus_seal'],
         'Exact original metadata/array digest corpus seal')
    origin=binding['origin'];paid=origin['original_reconstruction_costs']
    need(origin['source_index_sha256']==evidence['source_index_sha256']==prior['source_index_sha256']
         and origin['reconstruction_evidence_sha256']==cache['evidence_sha256'],
         'Old execution source remains cache origin, distinct current execution')
    for module,digest in origin['feature_source_sha256'].items():
        need(index['files']['src/'+module.replace('.','/')+'.py']==digest,'Unchanged frozen feature source')
    loaded=cache['load_costs']
    need(paid==evidence['original_reconstruction_costs']==loaded['original_reconstruction_costs']
         ==prior['reconstruction_costs'],'Paid original reconstruction remains explicit')
    need(loaded['cache_sha256']==cache['sha256'] and loaded['bytes_read']==cache['bytes']
         and loaded['original_costs_replaced_or_amortized'] is False
         and loaded['cache_policy_forwards']==loaded['cache_native_previews']==loaded['cache_observations_constructed']==0,
         'Current bounded cache reload has no new scientific collection')
    cumulative=report['cumulative_iteration_costs']
    need(cumulative['native_preview_entries']==prior['native_previews']+report['native_budget']['native_preview_entries']
         and cumulative['frozen_policy_forwards']==prior['frozen_policy_forwards']+report['total_frozen_policy_forwards']
         and cumulative['head_optimizer_updates']==64 and cumulative['prior_failed_attempt_not_amortized'],
         'Cumulative actual costs include failed origin')
    close(prior['parent_seconds'],origin_parent['elapsed_seconds'],'Prior parent cost exact')
    close(cumulative['failed_cache_attempt_worker_seconds'],origin_report['worker_seconds'],'Prior worker cost exact')
    close(cumulative['recovery_worker_seconds'],report['worker_seconds'],'Current worker cost exact')
    return {'sha256':cache['sha256'],'bytes':cache['bytes'],'prior_report_sha256':prior['pins']['report.json'],
            'prior_failure_audit_sha256':'37d3be700f1994d317b46b099c44ef3fb7a2e14f6db8e48f68679e66ddbd014e',
            'current_reload_seconds':loaded['cache_load_seconds'],'original_costs':paid,
            'prior_parent_seconds':prior['parent_seconds'],'archive_not_decoded':True}


def check_predictions(report,output,logged,subject_rows):
    record=report['logged_predictions'];p=read_json(output/record['file'],record['sha256'])
    need(p['labels_file']=='logged-targets.json' and p['labels_sha256']==base.READS[str((output/'logged-targets.json').relative_to(ROOT))]['sha256']
         and p['manifest_hash']==report['trace_manifest_hash'] and p['corpus_seal']==report['corpus_seal'],
         'Saved scalar predictions exact order/labels/corpus')
    need(p['head_hashes']=={'Q64':report['head_fit']['fitted_head_hash'],'CALIBRATED':report['calibration']['fitted_head_hash']},
         'Predictions bound to exact head identities')
    need(p['head_forwards']==record['head_forwards']==report['logged_prediction_readouts_attempted']
         ==report['logged_prediction_readouts_completed']==2
         and p['encoder_forwards']==record['encoder_forwards']==p['native_calls']==record['native_calls']==0,
         'Two extra head-only diagnostic readouts charged')
    result={}
    for method in ('Q64','CALIBRATED'):
        prediction=p['predictions'][method]
        need(len(prediction)==497 and all(type(v) in (int,float) and math.isfinite(v) and v==base.f32(v) for v in prediction),
             '497 finite deployed float32 predictions')
        need(all(not stop or v==0 for stop,v in zip(logged['STOP_rows'],prediction)),'Fixed STOP0 predictions')
        terms=[w*(v-y)**2 for w,v,y in zip(logged['weights'],prediction,logged['targets'])]
        value=math.fsum(terms)
        expected=report['calibration']['initial_weighted_mse' if method=='Q64' else 'final_weighted_mse']
        relative_close(value,expected,'Independent saved prediction global weighted MSE')
        cases={}
        for subject,ids in subject_rows.items():
            mass=math.fsum(logged['weights'][i] for i in ids)
            mse=math.fsum(terms[i] for i in ids)/mass
            key='weighted_mse_before' if method=='Q64' else 'weighted_mse_after_deployed_float32'
            relative_close(mse,report['calibration']['per_case'][subject][key],
                           'Independent saved prediction case weighted MSE')
            cases[subject]=mse
        result[method]={'weighted_mse':value,'per_case':cases}
    result['prediction_file_sha256']=record['sha256']
    result['limitation']='Verifies scalar residual arithmetic from saved neural predictions, not a repeated forward or features.'
    return result

def check_fit(report,original,logged,subject_rows):
    fit=report['head_fit']; old=original['head_fit']
    keys=('base_parameter_hash','architecture_hash','initial_head_hash','fitted_head_hash',
          'config','updates','final_weighted_mse','target','weighting','initialization')
    for key in keys:need(fit[key]==old[key],'Exact original64 fit '+key)
    need(len(fit['updates'])==64 and [v['update'] for v in fit['updates']]==list(range(1,65))
         and report['head_optimizer_updates']==64,'64 completed updates')
    need(fit['costs']['head_optimizer_steps']==fit['costs']['head_backward_calls']==fit['costs']['head_training_forwards']==64,
         'Exact fit work count')
    need(fit['costs']['base_optimizer_steps']==fit['costs']['base_policy_forwards']==0,'Frozen base during fit')
    need(fit['corpus_seal']==report['corpus_seal'] and fit['base_parameter_hash']==report['corpus']['base_parameter_hash'],
         'Actual fit/corpus/base binding')
    for key in ('initial_case_weighted_MSE','final_case_weighted_MSE'):
        need(report[key]==original[key],'Casewise original64 fit reproduced')
    cal=report['calibration'];solver=cal['solver'];cost=cal['costs']
    need(report['calibration_calls_attempted']==report['calibration_calls_completed']==1,'One calibration')
    need(cal['corpus_seal']==fit['corpus_seal'] and cal['manifest_hash']==report['trace_manifest_hash']
         and cal['fitted64_report_hash']==semantic(fit) and cal['initial_head_hash']==fit['fitted_head_hash'],
         'Calibration bound to exact fitted64/corpus')
    need(cal['base_parameter_hash']==fit['base_parameter_hash'] and cal['architecture_hash']==fit['architecture_hash']
         and cal['target']==fit['target'] and cal['weighting']==fit['weighting'],'No input/target/base change')
    singular=solver['singular_values']
    need(singular and all(math.isfinite(s) and s>=0 for s in singular)
         and all(a>=b for a,b in zip(singular,singular[1:])),'Finite descending reported singular values')
    cutoff=singular[0]*1e-8;kept=[s for s in singular if s>cutoff]
    close(solver['absolute_cutoff'],cutoff,'Fixed absolute cutoff')
    need(solver['relative_cutoff']==cal['config']['relative_svd_cutoff']==1e-8
         and solver['rank']==len(kept)>0,'Fixed retained rank')
    active=497-sum(logged['STOP_rows']);columns=solver['design_columns']
    need(1<=columns<=513 and len(singular)==min(active,columns)
         and solver['active_rows']==active and solver['STOP_rows_excluded']==sum(logged['STOP_rows']),
         'Full active-row/column denominator')
    need(solver['discarded_or_null_directions']==columns-len(kept)
         and solver['truncated_subspace_only'] is (len(kept)<columns),'Discarded/nullspace disclosure')
    relative_close(solver['retained_condition_number'],singular[0]/kept[-1],'Retained condition arithmetic')
    need(solver['retained_condition_number']<=1e8,'Declared retained numerical condition')
    relative_close(solver['weighted_residual_l2_float64']**2,cal['float64_solution_weighted_mse'],
                   'Float64 residual norm squared equals global weighted MSE')
    expected_mass=math.fsum(logged['weights'])
    close(cal['weight_sum'],expected_mass,'Global denominator not renormalized')
    close(cal['active_weight_sum'],math.fsum(w for w,s in zip(logged['weights'],logged['STOP_rows']) if not s),
          'Active mass excludes STOP only for solve')
    sse_before=sse_after=sse_solved=0.
    for subject in SUBJECTS:
        row=cal['per_case'][subject];ids=subject_rows[subject]
        mass=math.fsum(logged['weights'][i] for i in ids)
        need(row['rows']==len(ids),'Case row denominator')
        close(row['weight_mass'],mass,'Case weight denominator')
        relative_close(row['weighted_mse_before'],report['final_case_weighted_MSE'][subject],
                       'Calibration begins at reported exact64 case MSE')
        relative_close(row['weighted_SSE_before'],mass*row['weighted_mse_before'],'Case initial SSE')
        relative_close(row['weighted_SSE_after_deployed_float32'],mass*row['weighted_mse_after_deployed_float32'],
                       'Case deployed SSE')
        sse_before+=row['weighted_SSE_before'];sse_after+=row['weighted_SSE_after_deployed_float32']
        sse_solved+=mass*row['weighted_mse_after_float64']
    for actual,expected,label in ((cal['initial_weighted_mse'],sse_before,'Initial'),
          (cal['final_weighted_mse'],sse_after,'Deployed'),(cal['float64_solution_weighted_mse'],sse_solved,'Float64')):
        relative_close(actual,expected,label+' case/global MSE join')
    relative_close(cal['float32_deployment_mse_minus_float64'],sse_after-sse_solved,'Float32 deployment difference')
    need(sse_after<=sse_before+1e-6*max(1.,sse_before),'Declared deployed objective guard')
    need(cost['hidden_forwards_attempted']==cost['hidden_forwards_completed']==1
         and cost['head_readout_forwards_attempted']==cost['head_readout_forwards_completed']==2
         and cost['svd_calls_attempted']==cost['svd_calls_completed']==1,'Calibration work count')
    need(all(cost[k]==0 for k in ('base_policy_forwards','base_optimizer_steps','head_optimizer_steps','head_backward_calls')),
         'No extra model fitting or policy forwards in calibration')
    return {'Q64_head_hash':fit['fitted_head_hash'],'calibrated_head_hash':cal['fitted_head_hash'],
            'solver':solver,'before_mse':sse_before,'deployed_mse':sse_after,'float64_mse':sse_solved,
            'per_case':cal['per_case'],'calibration_costs':cost,
            'limitation':'Actual residuals are recomputed from saved scalar predictions below; this does not rerun the neural forward. Generated solver oracle passed separately.'}

def run(args):
    stage=HERE.parent;output=stage/'recovery-01'
    report=read_json(output/'report.json',args.report_sha256)
    launcher=read_json(output/'launcher.json',args.launcher_sha256)
    index=read_json(stage/'recovery-runner/source-index.json',args.index_sha256)
    need(index['execution_released'] is True,'Released source index')
    need(report['source_index_sha256']==launcher['source_index_sha256']==args.index_sha256
         and launcher['report_sha256']==args.report_sha256,'Terminal source/report/launcher pins')
    config=read_json(stage/'recovery-runner/config.json',index['files']['build/continuation-next-iteration-v1/recovery-runner/config.json'])
    need(config==report['config'] and config['TRAIN']==list(SUBJECTS) and config['methods']==list(METHODS),
         'Exact four TRAIN by three methods')
    need(config['traces']==36 and config['states']==497 and config['head_updates']==64 and config['max_steps']==24,
         'Unchanged corpus/updates/horizon')
    if report['status']!='completed':
        # A legitimate interrupted experiment is evidence, not a reader failure.
        completed={s:{m:v.get('complete') is True for m,v in rows.items()}
                   for s,rows in report.get('cases',{}).items()}
        return {'scope':'Saved incomplete experiment, no complete criterion asserted',
            'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,
            'source_index_sha256':args.index_sha256,'report_status':report['status'],
            'supervision_status':launcher['status'],'failure':report.get('failure'),
            'completed_arms':completed,'independent_criterion':{'comparison_admissible':False,'useful_gain':False},
            'native_budget':report.get('native_budget'),'fit_failure':report.get('failed_head_fit_accounting'),
            'calibration_failure':report.get('failed_calibration_accounting'),
            'parent_lifecycle':launcher.get('owned_lifecycle'),'input_files':base.READS}
    need(report['source_unchanged'] and not report['closed_roles_opened'] and report['new_training_rollouts']==0
         and not report['original_collection_actor_reexecuted'] and report['base_optimizer_updates']==0,
         'Fixed source/roles/base and no new collection')
    original=read_json(ROOT/(ORIGINAL+'/report.json'),ORIGINAL_SHA)
    need(index['input_metadata'][ORIGINAL+'/report.json']==ORIGINAL_SHA
         and original['useful_criterion']['useful_gain'] is False
         and report['historical_IL_failure_preserved'] and report['original_negative']['report_sha256']==ORIGINAL_SHA,
         'Original negative result and IL failure retained')
    summary,manifest,logged,subject_rows=check_corpus(report,index,output,original)
    cache=check_cache(report,index,output,manifest)
    calibration=check_fit(report,original,logged,subject_rows)
    scalar_predictions=check_predictions(report,output,logged,subject_rows)
    need(set(report['cases'])==set(SUBJECTS),'All four cases retained')
    outcomes={s:{} for s in SUBJECTS}
    for subject in SUBJECTS:
        rows=report['cases'][subject]
        need(set(rows)==set(METHODS),'All three methods per case')
        need(rows['GREEDY']['world']==rows['Q64']['world']==rows['CALIBRATED']['world'],'Matched source/world/public inputs')
        for method in METHODS:
            item=rows[method]
            need(read_json(output/'evaluation'/subject/(method+'.json'))==item,'Exact arm/final-report correspondence')
            normalized=copy.deepcopy(item)
            mode='Q_GUIDED' if method!='GREEDY' else method
            normalized['method']=mode
            outcomes[subject][method]=base.audit_route(normalized,mode,subject)
            if method!='CALIBRATED':
                old=original['cases'][subject][mode]
                for key in ('actions','world','nominal_history_hash','history_hash','decisions','history'):
                    need(item[key]==old[key],'Exact historical baseline '+subject+'/'+method+'/'+key)
                need(item['audit']['outcomes']==old['audit']['outcomes'] and item['original_route_reproduced'],
                     'Exact historical baseline physical outcome')
    checks={};tol=1e-8
    for subject in SUBJECTS:
        a,b=outcomes[subject]['GREEDY'],outcomes[subject]['CALIBRATED']
        checks[subject]={'same_world':True,'target_no_loss':b['target_mm3']+tol>=a['target_mm3'],
            'outside_no_increase':b['outside_mm3']<=a['outside_mm3']+tol,
            'contact_no_increase':b['contact_upper_mm3']<=a['contact_upper_mm3']+tol,
            'return_no_loss':b['return']+tol>=a['return']}
    def total(m,k):return sum(outcomes[s][m][k] for s in SUBJECTS)
    gains={'fewer_planning_previews':total('CALIBRATED','planning_previews')<total('GREEDY','planning_previews'),
           'fewer_planning_transitions':total('CALIBRATED','planning_transitions')<total('GREEDY','planning_transitions'),
           'shorter_complete_tool_path':total('CALIBRATED','path_mm')<total('GREEDY','path_mm')-tol,
           'fewer_executed_nonstop_actions':total('CALIBRATED','motions')<total('GREEDY','motions')}
    criterion={'comparison_admissible':True,'checks':checks,'strict_gains':gains,
               'useful_gain':all(all(v.values()) for v in checks.values()) and any(gains.values())}
    for key,value in criterion.items():need(report['useful_criterion'][key]==value,'Independent useful gate '+key)
    need(report['useful_criterion']['tie_is_negative'] and not report['useful_criterion']['setup_and_head_training_amortized'],
         'Ties negative and setup unamortized')
    for method in ('Q64','CALIBRATED'):
        calls=total(method,'q_batches');cost=report['head_inference_costs'][method]
        need(cost['policy_forwards_started']==cost['policy_forwards_completed']==cost['head_forwards']==calls,
             'Exact per-method full forwards')
    need(report['total_frozen_policy_forwards']==sum(total(m,'q_batches') for m in ('Q64','CALIBRATED')),
         'Current forwards are inference only; original extraction separately charged')
    visits=report['source_visits']
    need(len(visits)==4 and all(v['source_released'] for v in visits)
         and all(sum(v['subject']==s for v in visits)==1 for s in SUBJECTS),'Four closed evaluation-only source visits')
    need(report['checkpoint_loads']==1 and report['baseline_reproduced'] is True,'Single frozen base plus reproduced control')
    caps=config['caps'];budget=report['native_budget']
    need(budget['counting_reliable'] and budget['native_preview_entries']<=caps['native_previews']
         and report['total_frozen_policy_forwards']<=caps['frozen_policy_forwards'],'Global preview/forward caps')
    lifecycle=launcher.get('owned_lifecycle') or {}
    clean=(launcher['status']=='completed' and launcher.get('stop_reason') is None
           and lifecycle.get('verified') and lifecycle.get('child_reaped') and lifecycle.get('exit_code')==0
           and not lifecycle.get('remaining_owned_pids') and not lifecycle.get('errors'))
    if launcher['status']=='completed':
        need(launcher['worker_exit_code']==0 and clean,'Clean terminal supervised completion')
    need(launcher['sampled_peak_group_rss_bytes']<=caps['worker_peak_rss_bytes']
         and report['self_peak_rss_bytes']<=caps['worker_peak_rss_bytes']
         and launcher['elapsed_seconds']<caps['parent_wall_seconds'],'Recorded resource caps')
    return {'scope':'Saved JSON scalar/lineage/certificate audit; no feature/head/native read or replay',
        'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,'source_index_sha256':args.index_sha256,
        'clean_supervision':bool(clean),'supervision_status':launcher['status'],
        'supervision_failure':None if clean else {'stop_reason':launcher.get('stop_reason'),'lifecycle':lifecycle},
        'corpus':summary,'cache':cache,'calibration':calibration,'scalar_predictions':scalar_predictions,
        'cumulative_iteration_costs':report['cumulative_iteration_costs'],
        'outcomes':outcomes,'independent_criterion':criterion,'historical_baselines_exact':True,
        'total_native_previews':budget['native_preview_entries'],'total_frozen_forwards':report['total_frozen_policy_forwards'],
        'parent_seconds':launcher['elapsed_seconds'],'sampled_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],
        'limits':['Same four TRAIN worlds only; no held-out evidence or Q* claim',
                  'Actual-positive guard excludes negative preparation; it is not learned avoidance',
                  'Saved scalar prediction residuals recomputed; no neural-forward replay or independent feature extraction',
                  'Independent native geometry is hash/certificate joined, not repeated',
                  'All original and current setup costs retained; inclusive timings must not be summed'],
        'input_files':base.READS,'reused_independent_code':DEPENDENCIES}

def main():
    parser=argparse.ArgumentParser()
    for arg in ('report-sha256','launcher-sha256','index-sha256','output'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();target=Path(args.output).resolve()
    need(target.parent==HERE and not target.exists(),'Fresh result in own scope')
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Saved independent audit30s cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30.)
    try:result=run(args);result['audit_status']='PASS_SAVED_EVIDENCE';code=0
    except BaseException as error:
        result={'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(error).__name__,'message':str(error)},
                'input_files':base.READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0.)
    result.update(checks=base.CHECKS,seconds=time.monotonic()-started,model_calls=0,native_calls=0,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
    need(len(raw)<=64*1024,'Saved result64KiB cap')
    with target.open('xb') as stream:stream.write(raw)
    print(json.dumps({k:result[k] for k in ('audit_status','checks','seconds','model_calls','native_calls')}))
    return code

if __name__=='__main__':raise SystemExit(main())
