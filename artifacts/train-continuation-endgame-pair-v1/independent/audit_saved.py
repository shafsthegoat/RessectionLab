"""One terminal JSON/scalar audit of the fixed045 decision9 continuation pair.

No numpy, torch, features, weights, cache payload or native imports. Policy
information and pre-replay sequencing are source/record joins, not a repeated
execution. Only root-supplied exact terminal pins authorize an audit invocation.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
OLD=ROOT/'build/continuation-tail-v1/independent/audit_saved.py'
if hashlib.sha256(OLD.read_bytes()).hexdigest()!='431b7f113efede72ec15db4f8ce06757bac2b394f6b9cf8375e58808e2cb1d7c':
    raise ValueError('Frozen prior independent checker differs')
spec=importlib.util.spec_from_file_location('independent_tail_result',OLD)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
base=old.base
need,close,read_json,semantic=base.need,base.close,base.read_json,base.semantic
SUBJECT='ReMIND-045'
ARMS=('RETAINED_TAIL','IMMEDIATE_BEST')
TAIL='build/continuation-tail-v1/run-01'
TAIL_SHA='1bee2b1d4c40a55b666ce3269a39c4a5052e1911297b22c6c8702fb21000af66'
TAIL_LAUNCH='c9cf8269ceffda7ccfed0060f04164fa88e9cf59a597c5d20c851a5bb7e4d3f0'
TAIL_AUDIT_SHA='22d6461208c18a6751e9681530cdb8fc6c572c36eedc79a262f9103976e5cd18'
FORCED={'RETAINED_TAIL':'nominal-cavity-796668398d38557bae703113',
        'IMMEDIATE_BEST':'nominal-cavity-22839a32a529e5e57ad1d0b7'}

def declared_treatment(report,retained):
    actual=report['treatment'];arm=retained['cases'][SUBJECT]['TAIL']
    prefix=[]
    for i,row in enumerate(arm['planning']['decisions'][:8]):
        expected={k:row[k] for k in ('selected_action_id','observation_hash','source_state_hash',
                                    'result_state_hash','actual_reward','terminated')}
        expected['current_tool_before']=None if i==0 else arm['history'][i-1]['tool_id']
        prefix.append(expected)
    row=arm['planning']['decisions'][8]
    common={k:row[k] for k in ('step','source_state_hash','observation_hash','scores')}
    common['current_tool']=arm['history'][7]['tool_id']
    need(actual=={'subject':SUBJECT,'prefix':prefix,'common_state':common,
                 'forced_actions':FORCED,'no_future_suffix_given_to_planner':True},
         'Exact predeclared8-prefix/common-state/two treatments only')
    need(common['step']==8 and all(p['terminated'] is False for p in prefix),
         'Zero-based intervention8 after eight nonterminal decisions')
    positive=[r for r in common['scores'][1:] if r['reward']>0]
    best=max(positive,key=lambda r:r['reward'])
    need(row['selected_action_id']==FORCED['RETAINED_TAIL']
         and best['action_id']==FORCED['IMMEDIATE_BEST'],'Distinct historical tail/current immediate-best choice')
    need(FORCED['RETAINED_TAIL']!=FORCED['IMMEDIATE_BEST'],'One meaningful treatment difference')
    return arm,prefix,common

def source_and_model(report,index,retained,retained_parent):
    checked=read_json(ROOT/'build/continuation-tail-v1/independent/audit-01.json',TAIL_AUDIT_SHA)
    need(checked['audit_status']=='PASS_SAVED_EVIDENCE' and checked['clean_supervision']
         and checked['report_sha256']==TAIL_SHA and checked['launcher_sha256']==TAIL_LAUNCH
         and checked['independent_criterion']['useful_gain'] is False,'Prior negative tail independently accepted')
    need(index['input_metadata'][TAIL+'/report.json']==TAIL_SHA
         and index['input_metadata'][TAIL+'/launcher.json']==TAIL_LAUNCH,
         'Prior tail pins included in released input closure')
    for field in ('corpus','corpus_seal','trace_manifest_hash','reconstructed_traces','prior_cache_attempt','prior_direct_attempt'):
        need(report[field]==retained[field],'Identical saved source/corpus/cost origin '+field)
    need(report['reconstruction_executed_this_attempt'] is False and report['cache_copied'] is False,
         'No new reconstruction/cache copy')
    for field in ('file','sha256','bytes','binding_file','binding_sha256','evidence_file',
                  'evidence_sha256','reconstruction_paid_once','pixels_or_base_weights_copied','copied_this_attempt'):
        need(report['cache'][field]==retained['cache'][field],'Identical already checked cache metadata '+field)
    binding=read_json(ROOT/report['cache']['binding_file'],report['cache']['binding_sha256'])
    for module,digest in binding['origin']['feature_source_sha256'].items():
        need(index['files']['src/'+module.replace('.','/')+'.py']==digest,'Unchanged original feature producer')
    costs=report['cache']['load_costs'];oldcosts=retained['cache']['load_costs']
    for field in ('cache_sha256','bytes_read','original_reconstruction_costs','original_costs_replaced_or_amortized',
                  'cache_policy_forwards','cache_native_previews','cache_observations_constructed'):
        need(costs[field]==oldcosts[field],'No free/repeated cache science '+field)
    for field in ('base_parameter_hash','architecture_hash','initial_head_hash','fitted_head_hash',
                  'config','updates','final_weighted_mse','target','weighting','initialization'):
        need(report['head_fit'][field]==retained['head_fit'][field],'Exact reproduced64 math '+field)
    need(report['head_optimizer_updates']==64 and len(report['head_fit']['updates'])==64,
         'All64 reproduced updates paid, not an inference-only cost claim')
    for field in ('initial_case_weighted_MSE','final_case_weighted_MSE'):
        need(report[field]==retained[field],'Exact original per-case fit scalar '+field)
    for field in ('version','target','weighting','corpus_seal','manifest_hash','base_parameter_hash',
                  'architecture_hash','initial_head_hash','fitted_head_hash','frozen_hidden_hash',
                  'hidden_feature_hash','float64_solution_hash','solver','weight_sum','active_weight_sum',
                  'initial_weighted_mse','final_weighted_mse','float64_solution_weighted_mse',
                  'float32_deployment_mse_minus_float64','per_case','config'):
        need(report['calibration'][field]==retained['calibration'][field],'Exact reproduced direct readout '+field)
    need({k:v for k,v in report['tail_fit'].items() if k not in ('costs','calibrated_report_hash')}
         =={k:v for k,v in retained['tail_fit'].items() if k not in ('costs','calibrated_report_hash')},
         'Exact tail semantics, target, model parameters and solver; only elapsed parent report differs')
    need(report['tail_fit']['calibrated_report_hash']==semantic(report['calibration']),
         'Current exact direct-to-tail report binding')
    for method,key in (('calibration','calibration'),('tail','tail_fit')):
        need(report[method+'_calls_attempted']==report[method+'_calls_completed']==1,'One '+method+' solve')
        cost=report[key]['costs']
        need(cost['svd_calls_attempted']==cost['svd_calls_completed']==1
             and all(cost[k]==0 for k in ('head_optimizer_steps','head_backward_calls','base_optimizer_steps','base_policy_forwards')),
             'Reproduced readout only, no new base learning')
    previous=report['prior_tail_attempt']
    need(previous['directory']==TAIL and previous['pins']=={'report.json':TAIL_SHA,'launcher.json':TAIL_LAUNCH}
         and previous['audit_sha256']==TAIL_AUDIT_SHA and previous['negative_result_preserved'],
         'Prior negative tail immutable')
    need(previous['native_previews']==retained['native_budget']['native_preview_entries']
         and previous['frozen_policy_forwards']==retained['total_frozen_policy_forwards']
         and previous['head_optimizer_updates']==64,'Prior tail actual costs retained')
    close(previous['worker_seconds'],retained['worker_seconds'],'Prior tail worker time')
    close(previous['parent_seconds'],retained_parent['elapsed_seconds'],'Prior tail parent time')

def terminal_catalog(catalog,plan):
    last=plan['decisions'][-1]
    need(catalog['captured']=='after_final_choice_before_terminal_transition'
         and catalog['state_hash']==last['source_state_hash'] and catalog['step']==last['step']
         and catalog['selected_action']==last['selected_action_id'] and catalog['additional_queries']==0,
         'Catalog captured at preterminal current state, no extra discovery')
    need(catalog['certified_count']==last['legal_nonstop_actions'],'Catalog/current certified inventory denominator')
    groups=catalog['groups']
    keys=[tuple(g[k] for k in ('tool_id','proposal_kind','proposal_disposition','native_reason')) for g in groups]
    need(keys==sorted(set(keys)),'Stable unique scalar catalog groups')
    for g in groups:
        need(all(type(g[k]) is int and g[k]>=0 for k in ('slots','proposed','certified'))
             and g['proposed']<=g['slots'] and g['certified']<=g['slots'],'Catalog count ranges')
    need(sum(g['slots'] for g in groups)==catalog['ledger_slots']
         and sum(g['proposed'] for g in groups)==catalog['proposed_count']
         and sum(g['certified'] for g in groups)==catalog['certified_count'],'Catalog scalar sums')
    return catalog

def branch(result,plan,method,prefix,common,report):
    actions,decisions,history=result['actions'],result['decisions'],result['history']
    n=len(actions);accounting=plan['planning']
    need(result['complete'] and result['method']==method and 9<=n<=24
         and n==len(decisions)==len(history)==len(accounting['decisions']),
         'Complete full branch within original absolute24 horizon')
    need(actions==plan['actions']==[d['action_id'] for d in decisions]==[h['action_id'] for h in history],
         'All action/plan/replay/history records agree')
    need((actions[-1]=='STOP' and 'STOP' not in actions[:-1] and result['terminal_reason']=='STOP')
         or ('STOP' not in actions and n==24 and result['terminal_reason']=='HORIZON'),
         'Actual complete STOP or absolute24 endpoint')
    need(all(d['terminated'] is (i==n-1) for i,d in enumerate(decisions)),'No posttermination suffix')
    need(actions[8]==FORCED[method] and actions[:8]==[p['selected_action_id'] for p in prefix],
         'Only declared ninth treatment after fixed prefix')
    world=result['world'];audit=result['audit'];outcomes=audit['outcomes']
    need(world['subject']==SUBJECT and world['max_steps']==24,'Same declared subject/absolute horizon')
    base.valid_geometry(audit,method)
    need(audit['committed_history_hash']==result['history_hash']
         and all(audit[k]==world[k] for k in ('source_hash','decision_model_hash')),
         'Full independent history/source/world certificate')
    need(accounting['complete'] and result['planning_replay_join'] and accounting['initial_steps']==0
         and accounting['max_steps']==accounting['max_calls']==24,
         'Fresh full planning and full replay, never reset horizon at treatment')
    need(accounting['forced_action_id']==FORCED[method] and accounting['intervention_index']==8
         and accounting['diagnostic_only'] and accounting['negative_preparation_allowed'] is False,
         'Only fixed treatment allowed, productive guard unchanged')
    need(accounting['nominal_history_hash']==result['nominal_history_hash']
         and accounting['context_hash']==result['context_hash']
         and all(accounting[k]==world[k] for k in ('source_hash','decision_model_hash')),
         'Exact full nominal/authoritative state context join')
    need(accounting['fitted_head_hash']==report['tail_fit']['fitted_head_hash']
         and accounting['fitted_report_hash']==semantic(report['tail_fit']),
         'One reproduced frozen tail for both continuations')
    need(accounting['immediate_reward_added_to_tail'] and not accounting['immediate_reward_added_to_Q']
         and accounting['additional_native_previews_for_tail']==accounting['additional_observations_for_tail']==0,
         'Current reward added exactly once, no hidden additional scorer queries')
    forwards=0;total=0.;current_tool=None;switches=0
    for i,(d,h,planned) in enumerate(zip(decisions,history,accounting['decisions'])):
        need(d['reward']==h['reward']==planned['actual_reward'],'Committed reward counted once')
        for field in ('source_state_hash','result_state_hash','observation_hash','terminated'):
            need(planned[field]==d[field],'Exact planned/replayed '+field)
        need(planned['step']==i and planned['selected_action_id']==d['action_id'],
             'Absolute step/action selection join')
        if i:need(decisions[i-1]['result_state_hash']==d['source_state_hash'],'Continuous native state chain')
        if d['action_id']=='STOP':
            need(d['reward']==0 and d['source_state_hash']==d['result_state_hash'],'STOP does not change state/reward')
        else:
            need(h['outcome_scope']=='separate_evaluator_reference','Authoritative replay reference evaluation scope')
            switches+=int(current_tool is not None and current_tool!=h['tool_id']);current_tool=h['tool_id']
        scores=planned['scores'];ids=[r['action_id'] for r in scores];rewards=[r['reward'] for r in scores]
        positive=any(r>0 for r in rewards[1:]);predicted=[r['raw_predicted_tail'] for r in scores] if positive else None
        need(planned['all_current_legal_actions_scored'] and planned['legal_nonstop_actions']==len(scores)-1,
             'Full current certified inventory, not a treatment-specific subset')
        for j,row in enumerate(scores):
            need(row['actual_positive_guard'] is bool(j and row['reward']>0),'Exact productive guard')
            if j==0:need(row['reward']==row['raw_predicted_tail']==row['clipped_predicted_tail']==row['combined_nominal_q']==row['predicted_q']==0,'STOP0')
            elif positive:
                need(math.isfinite(row['raw_predicted_tail']) and row['clipped_predicted_tail']==max(0.,row['raw_predicted_tail'])
                     and row['combined_nominal_q']==row['predicted_q']==row['reward']+row['clipped_predicted_tail'],
                     'Exact current reward plus nonnegative raw tail once')
            else:
                need(all(row[k] is None for k in ('raw_predicted_tail','clipped_predicted_tail','combined_nominal_q','predicted_q')),
                     'No tail inference for no-positive STOP')
        unforced=(max((j for j in range(1,len(scores)) if rewards[j]>0),key=lambda j:scores[j]['combined_nominal_q'])
                  if positive else 0)
        if i==8:
            need({'step':i,'source_state_hash':d['source_state_hash'],'observation_hash':d['observation_hash'],
                  'scores':scores,'current_tool':history[7]['tool_id']}==common,'Exact common state/tool/ordered actual and predicted scores')
            need(planned['intervention'] is True and planned['unforced_selected_action_id']==ids[unforced]==FORCED['RETAINED_TAIL']
                 and planned['selection_reason']=='predeclared_single_index8_intervention'
                 and scores[ids.index(d['action_id'])]['reward']>0,'Only fixed treatment bypasses ordinary ranking')
        else:
            old.oracle.audit_choice(ids,[True]*len(ids),rewards,predicted,d['action_id'])
            need('intervention' not in planned and planned['selection_reason']==('exact_immediate_reward_plus_nonnegative_recorded_tail' if positive else 'no_actual_positive_action'),
                 'All other actions follow unchanged tail rule')
        chosen=scores[ids.index(d['action_id'])]
        need(chosen['reward']==d['reward'] and planned['selected_q']==chosen['combined_nominal_q'],
             'Selected forecast/reward equals executed action')
        if i<8:
            p=prefix[i]
            need({k:planned[k] for k in ('selected_action_id','observation_hash','source_state_hash','result_state_hash','actual_reward','terminated')}
                 =={k:p[k] for k in p if k!='current_tool_before'},'Exact ordinary prefix facts')
            need(p['current_tool_before']==(None if i==0 else history[i-1]['tool_id']),'Current tool/reward context preserved')
        total+=d['reward'];forwards+=int(positive)
    close(total,outcomes['total_reward'],'Full realized reward sum')
    need(total==accounting['estimated_incremental_return'],'Full planning/replay return exact')
    for name in ('target_removed_mm3','normal_removed_mm3','complete_tool_path_length_mm'):
        close(math.fsum(h[name] for h in history),outcomes[name],'Physical increment sum '+name)
    need(sum(a!='STOP' for a in actions)==outcomes['nonstop_actions'] and switches==outcomes['tool_changes'],
         'Exact intervention and switching counts')
    need(accounting['q_batches_started']==accounting['q_batches_completed']==forwards
         ==accounting['scorer_costs']['policy_forwards_completed']==accounting['scorer_costs']['head_forwards'],
         'One charged frozen batch per productive state including duplicated prefix')
    need(accounting['model_transition_calls']==accounting['committed_transitions']==n
         and accounting['native_previews_started']==accounting['native_previews_completed'],
         'All planning commits/previews started and returned')
    need(result['planning_costs']==plan['planning_costs']
         and result['planning_costs']['native_transition_calls']==result['replay_costs']['native_transition_calls']==n,
         'Every prefix/treatment/suffix transition paid in both planning and fresh replay')
    catalog=terminal_catalog(accounting['terminal_decision_catalog'],accounting)
    return {'actions':n,'motions':outcomes['nonstop_actions'],'switches':switches,'outcomes':outcomes,
        'prefix_return':math.fsum(d['reward'] for d in decisions[:8]),
        'remaining_return_including_forced_action':math.fsum(d['reward'] for d in decisions[8:]),
        'subsequent_return_after_forced_action':math.fsum(d['reward'] for d in decisions[9:]),
        'planning_previews':result['planning_costs']['native_preview_entries'],
        'replay_previews':result['replay_costs']['native_preview_entries'],
        'planning_seconds':result['planning_costs']['complete_wall_seconds'],
        'replay_seconds':result['replay_costs']['complete_wall_seconds'],
        'frozen_forwards':forwards,'history_hash':result['history_hash'],
        'terminal_decision_catalog':catalog}

def run(args):
    stage=HERE.parent;out=stage/'run-01'
    report=read_json(out/'report.json',args.report_sha256)
    launcher=read_json(out/'launcher.json',args.launcher_sha256)
    index=read_json(stage/'runner/source-index.json',args.index_sha256)
    need(index['execution_released'] and report['source_index_sha256']==launcher['source_index_sha256']==args.index_sha256
         and launcher['report_sha256']==args.report_sha256,'Exact released terminal report/launcher/index')
    config=read_json(stage/'runner/config.json',index['files']['build/continuation-endgame-pair-v1/runner/config.json'])
    need(config==report['config'] and config['subject']==SUBJECT and config['methods']==list(ARMS)
         and config['forced_actions']==FORCED and config['max_steps']==24
         and config['prefix_actions']==config['intervention_index']==8,'Exact prospective pair and absolute horizon')
    if report['status']!='completed':
        return {'scope':'Incomplete saved experiment; no causal effect accepted','report_status':report['status'],
            'supervision_status':launcher['status'],'failure':report.get('failure'),
            'comparison_admissible':False,'partial_costs':report.get('costs'),
            'native_budget':report.get('native_budget'),'input_files':base.READS}
    need(report['source_unchanged'] and not report['closed_roles_opened'] and report['new_training_rollouts']==0
         and not report['original_collection_actor_reexecuted'] and report['base_optimizer_updates']==0,
         'No changed source/base/split or new learning data')
    retained=read_json(ROOT/(TAIL+'/report.json'),TAIL_SHA)
    retained_parent=read_json(ROOT/(TAIL+'/launcher.json'),TAIL_LAUNCH)
    source_and_model(report,index,retained,retained_parent)
    original,prefix,common=declared_treatment(report,retained)
    need(set(report['plans'])==set(report['branches'])==set(ARMS),'Two full planned/executed branches retained')
    sealed=report['plans_sealed_before_replay']
    plans=read_json(out/sealed['file'],sealed['sha256'])
    need(sealed['file']=='plans.json' and plans==report['plans'],'Both full plans separately sealed before replay per reviewed worker')
    outcomes={}
    for method in ARMS:
        plan=plans[method];result=report['branches'][method]
        need(plan['complete'] and plan['plan_seal']==sealed['seals'][method]
             ==semantic({k:v for k,v in plan.items() if k!='plan_seal'})==result['plan_seal'],
             'Plan hash binds all chosen continuation/cost/accounting before outcomes')
        need(read_json(out/'evaluation'/SUBJECT/(method+'.json'))==result,'Exact separate saved branch')
        outcomes[method]=branch(result,plan,method,prefix,common,report)
    a,b=(report['branches'][m] for m in ARMS)
    need(a['world']==b['world']==original['world'] and a['context_hash']==b['context_hash']
         and a['decisions'][:8]==b['decisions'][:8] and a['history'][:8]==b['history'][:8],
         'Common actual physical/state/context prefix, only treatment differs')
    for field in ('actions','decisions','history','world','nominal_history_hash','history_hash'):
        need(a[field]==original[field],'Exact retained-arm full parity '+field)
    need(a['audit']['outcomes']==original['audit']['outcomes'] and a['original_route_reproduced']
         and report['baseline_reproduced'],'Unchanged originalbranch reproduces full outcome')
    selected={row['action_id']:row for row in common['scores']}
    left,right=(selected[FORCED[m]] for m in ARMS)
    delta=outcomes[ARMS[0]]['remaining_return_including_forced_action']-outcomes[ARMS[1]]['remaining_return_including_forced_action']
    predicted=left['combined_nominal_q']-right['combined_nominal_q']
    immediate=left['reward']-right['reward'];forecast_tail=left['clipped_predicted_tail']-right['clipped_predicted_tail']
    diagnosis=report['diagnosis']
    need(diagnosis['comparison_admissible'] and diagnosis['deployable_improvement_claim'] is False
         and diagnosis['difference_orientation']=='RETAINED_TAIL_minus_IMMEDIATE_BEST','Diagnostic scope/orientation retained')
    close(diagnosis['prefix_return'],outcomes[ARMS[0]]['prefix_return'],'Shared full prefix return')
    for method in ARMS:
        close(diagnosis['remaining_return_including_forced_action'][method],outcomes[method]['remaining_return_including_forced_action'],
              'Forced action included exactly once in remaining return')
        close(outcomes[method]['prefix_return']+outcomes[method]['remaining_return_including_forced_action'],
              outcomes[method]['outcomes']['total_reward'],'Full return=prefix+remaining, no omitted setup')
    for field,value in (('realized_remaining_return_difference',delta),('predicted_remaining_Q_difference',predicted),
        ('predicted_tail_difference',forecast_tail),('immediate_reward_difference',immediate),
        ('prediction_difference_error',predicted-delta)):
        close(diagnosis[field],value,'Independent diagnostic '+field)
    close(predicted,immediate+forecast_tail,'Current immediate difference + predicted tail difference once')
    full_delta=a['audit']['outcomes']['total_reward']-b['audit']['outcomes']['total_reward']
    close(full_delta,delta,'Common prefix cancels exactly in total-vs-remaining effect')
    need(diagnosis['realized_order_matches_predicted']==((delta>0)==(predicted>0) if delta and predicted else None),
         'Correct order/tie diagnosis')
    for field,value in diagnosis['full_outcome_difference'].items():
        close(value,a['audit']['outcomes'][field]-b['audit']['outcomes'][field],'Complete outcome difference '+field)
    for method in ARMS:
        need(diagnosis['full_outcomes'][method]=={k:report['branches'][method]['audit']['outcomes'][k] for k in diagnosis['full_outcomes'][method]},
             'Full outcome scalar projection')
    visits=report['source_visits']
    need(len(visits)==1 and visits[0]['subject']==SUBJECT and visits[0]['source_released']
         and report['checkpoint_loads']==1,'One released045 source and one frozen base load')
    calls=sum(r['frozen_forwards'] for r in outcomes.values());counts=report['head_inference_costs']
    need(report['total_frozen_policy_forwards']==counts['TAIL']['policy_forwards_started']
         ==counts['TAIL']['policy_forwards_completed']==counts['TAIL']['head_forwards']==calls,
         'All repeated prefix and continuation forwards charged')
    need(counts['DIRECT_CALIBRATED']['policy_forwards_started']==counts['DIRECT_CALIBRATED']['head_forwards']==0,
         'Direct head is reproduction only, not another unreported route')
    paid=[report[k] for k in ('prior_cache_attempt','prior_direct_attempt','prior_tail_attempt')]
    cumulative=report['cumulative_iteration_costs']
    need(cumulative['native_preview_entries']==sum(v['native_previews'] for v in paid)+report['native_budget']['native_preview_entries']
         and cumulative['frozen_policy_forwards']==sum(v['frozen_policy_forwards'] for v in paid)+calls
         and cumulative['current_head_optimizer_updates']==64 and cumulative['prior_failed_attempt_not_amortized'],
         'Cumulative original failure/direct/tail/pair costs retained')
    caps=config['caps'];lifecycle=launcher.get('owned_lifecycle') or {};budget=report['native_budget']
    clean=(launcher['status']=='completed' and launcher.get('stop_reason') is None and lifecycle.get('verified')
        and lifecycle.get('child_reaped') and lifecycle.get('exit_code')==0
        and not lifecycle.get('remaining_owned_pids') and not lifecycle.get('errors'))
    if launcher['status']=='completed':need(clean and launcher['worker_exit_code']==0,'Verified clean supervised completion')
    need(budget['counting_reliable'] and budget['native_preview_entries']<=caps['native_previews']
         and calls<=caps['frozen_policy_forwards'] and launcher['elapsed_seconds']<caps['parent_wall_seconds']
         and launcher['sampled_peak_group_rss_bytes']<=caps['worker_peak_rss_bytes']
         and report['self_peak_rss_bytes']<=caps['worker_peak_rss_bytes'],'Fixed measured caps')
    return {'scope':'One fixed simulator common-state continuation intervention; no policy/clinical improvement claim',
        'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,'source_index_sha256':args.index_sha256,
        'clean_supervision':bool(clean),'supervision_status':launcher['status'],
        'comparison_admissible':True,'original_full_route_parity':True,'world':a['world'],
        'both_full_plans_sealed':sealed,'common_observation_hash':common['observation_hash'],
        'common_state_hash':common['source_state_hash'],'common_current_tool':common['current_tool'],
        'outcomes':outcomes,'independent_effect':{'difference_orientation':'RETAINED_TAIL_minus_IMMEDIATE_BEST',
            'full_return_difference':full_delta,'remaining_return_difference':delta,
            'immediate_reward_difference':immediate,'predicted_tail_difference':forecast_tail,
            'predicted_Q_difference':predicted,'prediction_difference_error':predicted-delta},
        'current_native_previews':budget['native_preview_entries'],'current_frozen_forwards':calls,
        'current_head_updates':64,'current_direct_solves':1,'current_tail_solves':1,
        'parent_seconds':launcher['elapsed_seconds'],'sampled_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],
        'cumulative_iteration_costs':cumulative,
        'limits':['Only fixed045 state/two actions/unchanged continuation are identified',
          'Past full report may be decoded for authentication; helper receives no future suffix',
          'Catalog is preterminal/current-emitted inventory; not post-STOP exhaustion or all-geometry feasibility',
          'Certificates checked, no repeated native geometry or neural forward; weights/cache not decoded',
          'All prior/current setup and duplicate prefixes paid; inclusive timing phases overlap'],
        'input_files':base.READS}

def main():
    parser=argparse.ArgumentParser()
    for name in ('report-sha256','launcher-sha256','index-sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();target=Path(args.output).resolve()
    need(target.parent==HERE and not target.exists(),'Exclusive independent receipt')
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Saved audit30s cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30.)
    try:result=run(args);result['audit_status']='PASS_SAVED_EVIDENCE';code=0
    except BaseException as exc:
        result={'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(exc).__name__,'message':str(exc)},
                'input_files':base.READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0.)
    result.update(checks=base.CHECKS,seconds=time.monotonic()-started,model_calls=0,native_calls=0,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(result,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
    need(len(raw)<=64*1024,'Audit receipt64KiB cap')
    with target.open('xb') as stream:stream.write(raw)
    print(json.dumps({k:result[k] for k in ('audit_status','checks','seconds','model_calls','native_calls')}))
    return code

if __name__=='__main__':raise SystemExit(main())
