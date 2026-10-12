"""One terminal-only tail contrast audit: stdlib JSON/scalars, no payload decode.

Reuses frozen prior independent trace/route/calibration checks. Tail recurrence,
selection algebra and scalar residuals are independently recomputed here. Exact
terminal hashes are required; failure is retained, never retried by this script.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import struct
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
PRIOR=ROOT/'build/continuation-next-iteration-v1/independent'

def load_module(name,path,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Prior independent code changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

prior=load_module('independent_calibration_recovery',PRIOR/'audit_recovery.py',
    'd8d09aa5990c3aed2346e36bf4e9f4d13e2e782c384aa1bb297d8bff34659161')
oracle=load_module('independent_tail_oracle',HERE/'tail_oracle.py',
    '84c636d68a71ff99124b1925b2c8318178ad5b97d35af5f4cd47833afa924db6')
base=prior.base
need,close,read_json,semantic=base.need,base.close,base.read_json,base.semantic
relative_close=prior.relative_close
SUBJECTS=base.SUBJECTS
METHODS=('GREEDY','DIRECT_CALIBRATED','TAIL')
DIRECT='build/continuation-next-iteration-v1/recovery-01'
DIRECT_SHA='c423ea87fae67fcdc5751ff0c338297429c343b912d4a7baf28dd7cb5dce3d5b'
DIRECT_LAUNCH='6745bc9b70b4087b8adf574862298143ce792180dc1130b194342cf50e419d56'
TARGET='best_recorded_subsequent_prefix_including_STOP0_after_mandatory_action'

def scalar_array_digest(values,dtype):
    """Reproduce only declared numeric digest encoding, never load an array."""
    fmt={'<f8':'d','<f4':'f','|b1':'?'}[dtype]
    raw=json.dumps({'shape':[len(values)],'dtype':dtype},sort_keys=True).encode()
    digest=hashlib.sha256(raw)
    digest.update(b''.join(struct.pack('<'+fmt,value) for value in values))
    return 'sha256:'+digest.hexdigest()

def cache_lineage(report,index,direct,direct_parent,manifest):
    """Use the already checked external cache origin; do not open ZIP bytes."""
    previous=read_json(PRIOR/'recovery-audit-01.json',
        'ced58cfc0ed63b0d7d93095463a31f18af09bd92b21a2903251af0b34698f961')
    origin=read_json(PRIOR/'failure-audit-01.json',
        '37d3be700f1994d317b46b099c44ef3fb7a2e14f6db8e48f68679e66ddbd014e')
    need(previous['audit_status']=='PASS_SAVED_EVIDENCE' and previous['clean_supervision']
         and previous['report_sha256']==DIRECT_SHA and previous['launcher_sha256']==DIRECT_LAUNCH,
         'Exact independently audited direct negative retained')
    need(origin['recovery_point']=='PASS_SAVED_CACHE_ORIGIN_AND_BYTES',
         'Prior failed-cache origin independently checked without decoding')
    need(report['prior_cache_attempt']==direct['prior_cache_attempt']
         and report['reconstruction_executed_this_attempt'] is False and report['cache_copied'] is False,
         'Original failure retained, no new reconstruction or cache copy')
    for field in ('corpus','corpus_seal','reconstructed_traces','trace_manifest_hash'):
        need(report[field]==direct[field],'Exact previously reconstructed '+field)
    cache=report['cache'];old=direct['cache']
    for field in ('file','sha256','bytes','binding_file','binding_sha256','evidence_file',
                  'evidence_sha256','reconstruction_paid_once','pixels_or_base_weights_copied','copied_this_attempt'):
        need(cache[field]==old[field],'Exact existing cache metadata '+field)
    binding=read_json(ROOT/cache['binding_file'],cache['binding_sha256'])
    evidence=read_json(ROOT/cache['evidence_file'],cache['evidence_sha256'])
    metadata={**report['corpus'],'manifest':manifest}
    need(binding['metadata_hash']==semantic(metadata) and binding['corpus_seal']==report['corpus_seal']
         and semantic({'metadata':metadata,**evidence['arrays']})==report['corpus_seal'],
         'External metadata and array digests bind same original corpus')
    for module,digest in binding['origin']['feature_source_sha256'].items():
        need(index['files']['src/'+module.replace('.','/')+'.py']==digest,'Frozen original feature producer')
    loaded=cache['load_costs'];old_loaded=old['load_costs']
    for field in ('cache_sha256','bytes_read','original_reconstruction_costs','original_costs_replaced_or_amortized',
                  'cache_policy_forwards','cache_native_previews','cache_observations_constructed'):
        need(loaded[field]==old_loaded[field],'No changed cache science '+field)
    paid=report['prior_direct_attempt']
    need(paid['directory']==DIRECT and paid['pins']['report.json']==DIRECT_SHA
         and paid['pins']['launcher.json']==DIRECT_LAUNCH and paid['negative_result_preserved'],
         'Exact prior direct negative referenced')
    need(paid['native_previews']==direct['native_budget']['native_preview_entries']
         and paid['frozen_policy_forwards']==direct['total_frozen_policy_forwards']
         and paid['head_optimizer_updates']==direct['head_optimizer_updates']==64,
         'Prior direct costs retained')
    close(paid['worker_seconds'],direct['worker_seconds'],'Prior direct worker time')
    close(paid['parent_seconds'],direct_parent['elapsed_seconds'],'Prior direct parent time')
    cumulative=report['cumulative_iteration_costs'];failed=report['prior_cache_attempt']
    need(cumulative['native_preview_entries']==failed['native_previews']+paid['native_previews']+report['native_budget']['native_preview_entries']
         and cumulative['frozen_policy_forwards']==failed['frozen_policy_forwards']+paid['frozen_policy_forwards']+report['total_frozen_policy_forwards'],
         'All failed origin, direct recovery and current costs retained')
    need(cumulative['current_head_optimizer_updates']==cumulative['prior_direct_head_optimizer_updates']==64
         and cumulative['prior_failed_attempt_not_amortized'],'All actual64-update fits paid separately')
    for field,value in (('failed_cache_attempt_worker_seconds',failed['worker_seconds']),
                        ('prior_direct_worker_seconds',paid['worker_seconds']),
                        ('tail_contrast_worker_seconds',report['worker_seconds'])):
        close(cumulative[field],value,'Actual cumulative time component')
    return {'cache_sha256':cache['sha256'],'cache_bytes':cache['bytes'],
        'cache_load_seconds':loaded['cache_load_seconds'],'feature_payload_not_read':True,
        'direct_negative_report_sha256':DIRECT_SHA,'original_failed_report_sha256':failed['pins']['report.json']}

def tail_labels(report,output,manifest,logged):
    record=report['tail_targets'];labels=read_json(output/record['file'],record['sha256'])
    need(record['file']=='logged-tail-targets.json' and labels['schema']=='same497-recorded-tail-targets-v1',
         'Explicit tail-label file/schema')
    need(labels['manifest_hash']==report['trace_manifest_hash'] and labels['original_Q_file']=='logged-targets.json'
         and labels['original_Q_sha256']==base.READS[str((output/'logged-targets.json').relative_to(ROOT))]['sha256'],
         'Original Q labels/current float64 tail labels join')
    rewards=[];tails=[];stops=[]
    for trace in manifest:
        values=[v['reward'] for v in trace['decisions']]
        expected=oracle.recorded_tail_targets(values)
        rewards.extend(values);tails.extend(expected)
        stops.extend(v['action_id']=='STOP' for v in trace['decisions'])
    need(labels['current_rewards']==rewards and labels['STOP_rows']==stops==logged['STOP_rows']
         and labels['weights']==logged['weights'],'Exact row order/current rewards/weights/STOP')
    need(len(labels['tail_targets'])==len(rewards)==497,'All497 tail rows')
    for i,(expected,actual,reward,q,stop) in enumerate(zip(tails,labels['tail_targets'],rewards,logged['targets'],stops)):
        # Independent fsum enumeration differs only in normal float64 summation.
        relative_close(actual,expected,'Tail equals independently enumerated recorded future prefix',atol=1e-12,rtol=2e-15)
        need(actual>=0 and base.f32(reward+actual)==q,'One exact float32 Q join, no clipped mandatory current cost')
        if stop:need(reward==actual==q==0,'STOP labels exact zero')
    declaration=labels['target_declaration']
    need(declaration==report['tail_fit']['target_declaration'] and semantic(declaration)==labels['target_seal']==report['tail_fit']['target_seal'],
         'Tail target declaration semantic seal')
    need(declaration['target']==TARGET and declaration['source_corpus_seal']==report['corpus_seal']
         and declaration['source_manifest_hash']==report['trace_manifest_hash']
         and declaration['rows']==497 and declaration['STOP_rows']==sum(stops)
         and declaration['tail_precision']=='float64_manifest_recurrence'
         and declaration['original_target_join']=='exact_float32_current_reward_plus_float64_tail',
         'Exact recorded-tail meaning and source denominator')
    for field,values,dtype in (('current_rewards_hash',rewards,'<f8'),
             ('tail_targets_hash',labels['tail_targets'],'<f8'),('weight_hash',logged['weights'],'<f4'),
             ('original_Q_targets_hash',logged['targets'],'<f4'),('STOP_hash',stops,'|b1')):
        need(declaration[field]==scalar_array_digest(values,dtype),'Independent scalar bytes '+field)
    return labels

def tail_fit(report,labels,logged,subject_rows):
    fit=report['tail_fit'];direct=report['calibration'];cost=fit['costs'];solver=fit['solver']
    need(report['tail_calls_attempted']==report['tail_calls_completed']==1,'One tail solve only')
    need(fit['version']=='frozen-recorded-continuation-tail-readout-v1' and fit['target']==TARGET
         and fit['prediction_semantics']=='raw_recorded_continuation_tail_excluding_current_reward'
         and fit['deployment']=='exact_immediate_nominal_reward_plus_max_zero_predicted_tail',
         'Raw tail type meaning distinct from direct Q')
    need(fit['corpus_seal']==report['corpus_seal'] and fit['manifest_hash']==report['trace_manifest_hash']
         and fit['calibrated_report_hash']==semantic(direct) and fit['initial_head_hash']==direct['fitted_head_hash'],
         'Exact direct-calibrated initialization/corpus')
    for field in ('base_parameter_hash','architecture_hash','frozen_hidden_hash','hidden_feature_hash','weighting'):
        need(fit[field]==direct[field],'Unchanged direct-calibrated hidden/base '+field)
    singular=solver['singular_values'];cutoff=singular[0]*1e-8;kept=[v for v in singular if v>cutoff]
    need(all(math.isfinite(v) and v>=0 for v in singular) and all(a>=b for a,b in zip(singular,singular[1:])),
         'Finite descending singular spectrum')
    need(singular==direct['solver']['singular_values'],'Same frozen weighted design spectrum')
    close(solver['absolute_cutoff'],cutoff,'Fixed cutoff arithmetic')
    need(solver['relative_cutoff']==fit['config']['relative_svd_cutoff']==1e-8 and fit['config']['solves']==1
         and solver['rank']==len(kept)>0,'Single declared fixed-rank solution')
    active=497-sum(logged['STOP_rows']);columns=solver['design_columns']
    need(solver['active_rows']==active and solver['STOP_rows_excluded']==sum(logged['STOP_rows'])
         and len(singular)==min(active,columns) and solver['discarded_or_null_directions']==columns-len(kept),
         'Full row denominator/nullspace disclosure')
    relative_close(solver['retained_condition_number'],singular[0]/kept[-1],'Retained condition arithmetic')
    need(solver['retained_condition_number']<=1e8,'Fixed numerical condition cap')
    relative_close(solver['weighted_residual_l2_float64']**2,fit['float64_solution_tail_weighted_mse'],
                   'Reported float64 residual norm and objective agree')
    close(fit['weight_sum'],math.fsum(logged['weights']),'Original full weight mass')
    close(fit['active_weight_sum'],math.fsum(w for w,s in zip(logged['weights'],logged['STOP_rows']) if not s),
          'STOP excluded only from solve, no active mass renormalization')
    need(fit['raw_tail_weighted_mse']<=fit['initial_raw_tail_weighted_mse']+1e-6*max(1.,fit['initial_raw_tail_weighted_mse']),
         'Declared deployed tail objective nonregression')
    need(cost['hidden_forwards_attempted']==cost['hidden_forwards_completed']==1
         and cost['head_readout_forwards_attempted']==cost['head_readout_forwards_completed']==2
         and cost['svd_calls_attempted']==cost['svd_calls_completed']==1,'One hidden batch/SVD/two head readouts')
    need(all(cost[k]==0 for k in ('base_policy_forwards','base_optimizer_steps','head_optimizer_steps','head_backward_calls')),
         'No tail Adam/backward/base optimization')
    return {'head_hash':fit['fitted_head_hash'],'target_seal':fit['target_seal'],'solver':solver,'costs':cost}

def prediction_metrics(report,output,labels,logged,subject_rows,index):
    record=report['logged_predictions'];p=read_json(output/record['file'],record['sha256'])
    need(p['schema']=='same497-direct-and-tail-predictions-v1' and p['labels_file']=='logged-tail-targets.json'
         and p['labels_sha256']==report['tail_targets']['sha256'] and p['corpus_seal']==report['corpus_seal']
         and p['manifest_hash']==report['trace_manifest_hash'],'Predictions bind labels/order/corpus')
    need(p['head_hashes']=={'DIRECT_CALIBRATED':report['calibration']['fitted_head_hash'],'TAIL':report['tail_fit']['fitted_head_hash']}
         and p['prediction_semantics']=={'DIRECT_CALIBRATED':'current_reward_included','TAIL':'raw_tail_excludes_current_reward'},
         'Separate raw-tail and total-Q predictions')
    need(p['head_forwards']==record['head_forwards']==report['logged_prediction_readouts_attempted']
         ==report['logged_prediction_readouts_completed']==2 and p['encoder_forwards']==p['native_calls']==0,
         'Two additional charged head-only readouts')
    for name in ('DIRECT_CALIBRATED','TAIL'):
        need(len(p['predictions'][name])==497 and all(type(v) in (int,float) and math.isfinite(v) and base.f32(v)==v for v in p['predictions'][name]),
             '497 finite deployed float32 scalar predictions')
        need(all(not s or v==0 for s,v in zip(logged['STOP_rows'],p['predictions'][name])),'Fixed STOP0 predictions')
    old=read_json(ROOT/(DIRECT+'/logged-predictions.json'),index['input_metadata'][DIRECT+'/logged-predictions.json'])
    need(p['predictions']['DIRECT_CALIBRATED']==old['predictions']['CALIBRATED'],'Exact original direct predictions reproduced')
    direct,tail=p['predictions']['DIRECT_CALIBRATED'],p['predictions']['TAIL']
    clipped=[max(0.,v) for v in tail]
    pairs={'direct_calibrated_Q_weighted_mse':(direct,logged['targets']),
           'initial_raw_tail_weighted_mse':(direct,labels['tail_targets']),
           'raw_tail_weighted_mse':(tail,labels['tail_targets']),
           'clipped_tail_weighted_mse':(clipped,labels['tail_targets']),
           'reconstructed_Q_weighted_mse':([r+t for r,t in zip(labels['current_rewards'],clipped)],logged['targets'])}
    metrics={};cases={s:{} for s in SUBJECTS}
    for key,(prediction,target) in pairs.items():
        value=oracle.weighted_residual(prediction,target,logged['weights'],logged['STOP_rows'])
        relative_close(value,report['tail_fit'][key],'Independent497 scalar residual '+key)
        metrics[key]=value
        for subject,ids in subject_rows.items():
            mass=math.fsum(logged['weights'][i] for i in ids)
            row=report['tail_fit']['per_case'][subject]
            need(row['rows']==len(ids),'Per-case complete denominator')
            close(row['weight_mass'],mass,'Per-case original weight mass')
            mse=math.fsum(logged['weights'][i]*(prediction[i]-target[i])**2 for i in ids)/mass
            cases[subject][key]=mse
            if key in row:relative_close(mse,row[key],'Independent per-case residual '+key)
    need(sum(v<0 and not s for v,s in zip(tail,logged['STOP_rows']))==report['tail_fit']['negative_raw_tail_rows'],
         'Raw negative tail count retained before deployment clamp')
    relative_close(metrics['direct_calibrated_Q_weighted_mse'],report['calibration']['final_weighted_mse'],
                   'Direct calibration scalar residual reproduced')
    return {'global':metrics,'per_case':cases,'negative_raw_tail_rows':report['tail_fit']['negative_raw_tail_rows'],
        'prediction_file_sha256':record['sha256'],'limitation':'Scalar residual check, no neural-forward or feature reconstruction'}

def audited_route(item,method,subject):
    normalized=copy.deepcopy(item);normalized['method']='GREEDY' if method=='GREEDY' else 'Q_GUIDED'
    if method=='TAIL':
        plan=item['planning']
        need(plan['method']=='actual-positive-exact-reward-plus-recorded-tail-rollout-v1'
             and plan['immediate_reward_added_to_tail'] is True and plan['immediate_reward_added_to_Q'] is False
             and plan['additional_native_previews_for_tail']==plan['additional_observations_for_tail']==0,
             'Tail-only score meaning and no hidden extra query work')
        for row,new in zip(plan['decisions'],normalized['planning']['decisions']):
            scores=row['scores'];ids=[v['action_id'] for v in scores];rewards=[v['reward'] for v in scores]
            productive=any(v>0 for v in rewards[1:])
            predictions=[v['raw_predicted_tail'] for v in scores] if productive else None
            oracle.audit_choice(ids,[True]*len(ids),rewards,predictions,row['selected_action_id'])
            for i,score in enumerate(scores):
                if i==0:
                    need(score['raw_predicted_tail']==score['clipped_predicted_tail']==score['combined_nominal_q']==score['predicted_q']==0,
                         'STOP0 never adds current reward/tail')
                elif productive:
                    value=rewards[i]+max(0.,predictions[i])
                    need(score['clipped_predicted_tail']==max(0.,predictions[i])
                         and score['combined_nominal_q']==score['predicted_q']==value,
                         'Exact current nominal reward plus clipped tail once')
                else:
                    need(all(score[k] is None for k in ('raw_predicted_tail','clipped_predicted_tail','combined_nominal_q','predicted_q')),
                         'No unnecessary tail inference when no actual positive action')
            reason='exact_immediate_reward_plus_nonnegative_recorded_tail' if productive else 'no_actual_positive_action'
            need(row['selection_reason']==reason,'Explicit tail selection reason')
            # Algebra independently verified above. Reuse existing route/state/
            # certificate checker with only its direct-Q reason vocabulary mapped.
            if productive:new['selection_reason']='positive_q_among_actual_positive'
    result=base.audit_route(normalized,normalized['method'],subject)
    if method=='TAIL':result['choice_reasons']=[r['selection_reason'] for r in item['planning']['decisions']]
    return result

def run(args):
    stage=HERE.parent;output=stage/'run-01'
    report=read_json(output/'report.json',args.report_sha256)
    launcher=read_json(output/'launcher.json',args.launcher_sha256)
    index=read_json(stage/'runner/source-index.json',args.index_sha256)
    need(index['execution_released'] and report['source_index_sha256']==launcher['source_index_sha256']==args.index_sha256
         and launcher['report_sha256']==args.report_sha256,'Exact terminal report/release/index joins')
    config=read_json(stage/'runner/config.json',index['files']['build/continuation-tail-v1/runner/config.json'])
    need(config==report['config'] and config['TRAIN']==list(SUBJECTS) and config['methods']==list(METHODS)
         and config['traces']==36 and config['states']==497 and config['head_updates']==64 and config['max_steps']==24,
         'Fixed four TRAIN/36 traces/497 rows/64 original updates/three methods/h24')
    if report['status']!='completed':
        return {'report_status':report['status'],'supervision_status':launcher['status'],
            'failure':report.get('failure'),'partial_costs':report.get('costs'),
            'comparison_admissible':False,'useful_gain':False,'native_budget':report.get('native_budget'),
            'parent_lifecycle':launcher.get('owned_lifecycle'),'input_files':base.READS}
    need(report['source_unchanged'] and not report['closed_roles_opened'] and report['new_training_rollouts']==0
         and not report['original_collection_actor_reexecuted'] and report['base_optimizer_updates']==0,
         'No new collection/closed role/base update')
    original=read_json(ROOT/(prior.ORIGINAL+'/report.json'),prior.ORIGINAL_SHA)
    direct=read_json(ROOT/(DIRECT+'/report.json'),DIRECT_SHA)
    direct_parent=read_json(ROOT/(DIRECT+'/launcher.json'),DIRECT_LAUNCH)
    need(index['input_metadata'][DIRECT+'/report.json']==DIRECT_SHA and index['input_metadata'][DIRECT+'/launcher.json']==DIRECT_LAUNCH
         and original['useful_criterion']['useful_gain'] is False and direct['useful_criterion']['useful_gain'] is False
         and report['historical_IL_failure_preserved'],'All earlier negative endpoints retained')
    summary,manifest,logged,subject_rows=prior.check_corpus(report,index,output,original)
    cache=cache_lineage(report,index,direct,direct_parent,manifest)
    calibration=prior.check_fit(report,original,logged,subject_rows)
    for field in ('solver','fitted_head_hash','initial_head_hash','frozen_hidden_hash','hidden_feature_hash',
                  'initial_weighted_mse','final_weighted_mse','float64_solution_weighted_mse','per_case'):
        need(report['calibration'][field]==direct['calibration'][field],'Exact retained direct calibration '+field)
    labels=tail_labels(report,output,manifest,logged)
    fit=tail_fit(report,labels,logged,subject_rows)
    scalars=prediction_metrics(report,output,labels,logged,subject_rows,index)
    need(set(report['cases'])==set(SUBJECTS),'Complete four-case denominator')
    outcomes={s:{} for s in SUBJECTS}
    for subject in SUBJECTS:
        rows=report['cases'][subject]
        need(set(rows)==set(METHODS),'Complete three-method denominator')
        need(rows['GREEDY']['world']==rows['DIRECT_CALIBRATED']['world']==rows['TAIL']['world'],
             'Matched public/source/world per case')
        for method in METHODS:
            item=rows[method]
            need(read_json(output/'evaluation'/subject/(method+'.json'))==item,'Exact saved route/report join')
            outcomes[subject][method]=audited_route(item,method,subject)
            if method!='TAIL':
                old=direct['cases'][subject]['CALIBRATED' if method=='DIRECT_CALIBRATED' else method]
                for field in ('actions','world','nominal_history_hash','history_hash','decisions','history'):
                    need(item[field]==old[field],'Exact historical baseline '+subject+'/'+method+'/'+field)
                need(item['audit']['outcomes']==old['audit']['outcomes'] and item['original_route_reproduced'],
                     'Exact historical baseline physical outcomes')
            if method!='GREEDY':
                head=report['tail_fit'] if method=='TAIL' else report['calibration']
                need(item['planning']['fitted_head_hash']==head['fitted_head_hash']
                     and item['planning']['fitted_report_hash']==semantic(head),'Actual deployed head/report binding')
    checks={};tol=1e-8
    for subject in SUBJECTS:
        a,b=outcomes[subject]['GREEDY'],outcomes[subject]['TAIL']
        checks[subject]={'same_world':True,'target_no_loss':b['target_mm3']+tol>=a['target_mm3'],
            'outside_no_increase':b['outside_mm3']<=a['outside_mm3']+tol,
            'contact_no_increase':b['contact_upper_mm3']<=a['contact_upper_mm3']+tol,
            'return_no_loss':b['return']+tol>=a['return']}
    def total(m,k):return sum(outcomes[s][m][k] for s in SUBJECTS)
    gains={'fewer_planning_previews':total('TAIL','planning_previews')<total('GREEDY','planning_previews'),
        'fewer_planning_transitions':total('TAIL','planning_transitions')<total('GREEDY','planning_transitions'),
        'shorter_complete_tool_path':total('TAIL','path_mm')<total('GREEDY','path_mm')-tol,
        'fewer_executed_nonstop_actions':total('TAIL','motions')<total('GREEDY','motions')}
    criterion={'comparison_admissible':True,'checks':checks,'strict_gains':gains,
               'useful_gain':all(all(row.values()) for row in checks.values()) and any(gains.values())}
    for key,value in criterion.items():need(report['useful_criterion'][key]==value,'Independent full useful gate '+key)
    need(report['useful_criterion']['tie_is_negative'] and not report['useful_criterion']['setup_and_head_training_amortized'],
         'No tie win or free fit/setup')
    for method in ('DIRECT_CALIBRATED','TAIL'):
        calls=total(method,'q_batches');cost=report['head_inference_costs'][method]
        need(cost['policy_forwards_started']==cost['policy_forwards_completed']==cost['head_forwards']==calls,
             'One real policy/head batch per productive state')
    need(report['total_frozen_policy_forwards']==sum(total(m,'q_batches') for m in ('DIRECT_CALIBRATED','TAIL')),
         'Total inference forwards exact, no original extraction silently repeated')
    visits=report['source_visits']
    need(len(visits)==4 and all(v['source_released'] for v in visits)
         and all(sum(v['subject']==s for v in visits)==1 for s in SUBJECTS)
         and report['checkpoint_loads']==1 and report['baseline_reproduced'],'Four source visits/one frozen base/all controls reproduced')
    caps=config['caps'];budget=report['native_budget'];lifecycle=launcher.get('owned_lifecycle') or {}
    need(budget['counting_reliable'] and budget['native_preview_entries']<=caps['native_previews']
         and report['total_frozen_policy_forwards']<=caps['frozen_policy_forwards'],'Preview/forward cap')
    clean=(launcher['status']=='completed' and launcher.get('stop_reason') is None
        and lifecycle.get('verified') and lifecycle.get('child_reaped') and lifecycle.get('exit_code')==0
        and not lifecycle.get('remaining_owned_pids') and not lifecycle.get('errors'))
    if launcher['status']=='completed':need(launcher['worker_exit_code']==0 and clean,'Clean supervised terminal')
    need(launcher['sampled_peak_group_rss_bytes']<=caps['worker_peak_rss_bytes']
         and report['self_peak_rss_bytes']<=caps['worker_peak_rss_bytes']
         and launcher['elapsed_seconds']<caps['parent_wall_seconds'],'Recorded RSS/time caps')
    return {'scope':'Independent saved scalar/lineage/route audit; no payload/model/native read or replay',
        'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,'source_index_sha256':args.index_sha256,
        'clean_supervision':bool(clean),'supervision_status':launcher['status'],
        'supervision_failure':None if clean else {'stop_reason':launcher.get('stop_reason'),'lifecycle':lifecycle},
        'corpus':summary,'cache':cache,'tail_fit':fit,'scalar_predictions':scalars,
        'outcomes':outcomes,'independent_criterion':criterion,'historical_baselines_exact':True,
        'cumulative_iteration_costs':report['cumulative_iteration_costs'],
        'current_native_previews':budget['native_preview_entries'],'current_frozen_forwards':report['total_frozen_policy_forwards'],
        'parent_seconds':launcher['elapsed_seconds'],'sampled_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],
        'limits':['Same four TRAIN worlds only, no held-out/Q-optimality/clinical claim',
          'Actual-positive guard excludes negative preparation; it is not learned avoidance',
          'Independent prefix labels/scalar residuals/selection checked; no neural forward or cached features read',
          'Independent native geometry certificates joined, not re-executed',
          'All prior negative attempts and actual costs retained; inclusive times are not additive'],
        'input_files':base.READS}

def main():
    parser=argparse.ArgumentParser()
    for arg in ('report-sha256','launcher-sha256','index-sha256','output'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();target=Path(args.output).resolve()
    need(target.parent==HERE and not target.exists(),'Exclusive output in own scope')
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Independent saved audit30s cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30.)
    try:result=run(args);result['audit_status']='PASS_SAVED_EVIDENCE';code=0
    except BaseException as error:
        result={'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(error).__name__,'message':str(error)},
                'input_files':base.READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0.)
    result.update(checks=base.CHECKS,seconds=time.monotonic()-started,model_calls=0,native_calls=0,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
    need(len(raw)<=64*1024,'Saved audit output64KiB cap')
    with target.open('xb') as stream:stream.write(raw)
    print(json.dumps({k:result[k] for k in ('audit_status','checks','seconds','model_calls','native_calls')}))
    return code

if __name__=='__main__':raise SystemExit(main())
