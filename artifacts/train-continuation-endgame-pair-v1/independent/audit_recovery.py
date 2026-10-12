"""One saved-only sealed-plan replay recovery audit; no scientific imports.

The original planned scores are immutable paid inputs. The newly released replay
context is separately authenticated; every public world/state/observation and
physical history join remains exact. Context identity is never overwritten.
Original audit_saved.py and failed attempt remain unchanged. This program reads
JSON/source bytes only; it does not import runner, model, native or array code.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import time
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
original=HERE/'audit_saved.py'
if hashlib.sha256(original.read_bytes()).hexdigest()!='c57ba3391c4e73fad15c851bbff0d51477fc5a5b2bd0282bf0eb5307b880b99b':
    raise ValueError('Immutable original prepared pair auditor differs')
spec=importlib.util.spec_from_file_location('independent_original_pair',original)
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
base,old=pair.base,pair.old
need,close,read_json,semantic=base.need,base.close,base.read_json,base.semantic
SUBJECT,ARMS,FORCED=pair.SUBJECT,pair.ARMS,pair.FORCED
terminal_catalog=pair.terminal_catalog
FAILED='build/continuation-endgame-pair-v1/run-01'
OLD_INDEX='build/continuation-endgame-pair-v1/runner/source-index.json'
OLD_INDEX_SHA='c2a0b2f70fbc8692db425c85132082b26216aa62f0470c641077bb94a5458a16'
PINS={'report.json':'2a50f3a22b9c298681a863785b489c8e09b398974bce82d733464f015ff8337e',
      'launcher.json':'beedf54160707e8f2571a4e9a6d4add7f99ebf4cdbd5f320e7af4f89dece1808',
      'plans.json':'6798288cbb05cce8692f61f64438b68fcad8aef15ebc5ec008b2f3e3a151521b'}
FAILURE_AUDIT='build/continuation-endgame-pair-v1/independent/failure-audit-01.json'
FAILURE_SHA='258ab0dd042bdbde8bbbf42915d0bc6a6e4b970d8cb0fde86f8ca9396b4b1854'
REVIEWED_RUNTIME={
 'worker.py':'9061ab5d3b50b30327eb94f774888fca748d913323a7d5d2595d38249b6097fa',
 'recovery_inputs.py':'726e07d18ee62668fec96305070372dfaabe89b4c8f73fd3b9c9834943de084c',
 'common.py':'2bba1a759e24edceb74f955cc5c38454a2ac1ee37226a00446e88576ab401376',
 'run_owned.py':'e7884ed0fa9e1461058ad20c791784295fb1edb569643f010c7b97207a6b9e9c',
 'freeze_sources.py':'24e8a1c91ab84ef6e84837b58ab25111c52ed979ea22f00f7c67a77cae5a61af',
 'config.json':'dd6eee4ba744ad27a6e9911f7053935aec96dc573c648b7398d9dbfd0a748baf'}

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
         and accounting['context_hash']==report['plan_admission']['original_context_hash']
         and all(accounting[k]==world[k] for k in ('source_hash','decision_model_hash')),
         'Exact full nominal/source join; original planning context separately bound by admission')
    need(accounting['fitted_head_hash']==report['original_fit_for_audit']['fitted_head_hash']
         and accounting['fitted_report_hash']==semantic(report['original_fit_for_audit']),
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
    need(plan['planning_costs']['native_transition_calls']==result['replay_costs']['native_transition_calls']==n,
         'Every prefix/treatment/suffix transition paid in both planning and fresh replay')
    catalog=terminal_catalog(accounting['terminal_decision_catalog'],accounting)
    return {'actions':n,'motions':outcomes['nonstop_actions'],'switches':switches,'outcomes':outcomes,
        'prefix_return':math.fsum(d['reward'] for d in decisions[:8]),
        'remaining_return_including_forced_action':math.fsum(d['reward'] for d in decisions[8:]),
        'subsequent_return_after_forced_action':math.fsum(d['reward'] for d in decisions[9:]),
        'planning_previews':plan['planning_costs']['native_preview_entries'],
        'replay_previews':result['replay_costs']['native_preview_entries'],
        'planning_seconds':plan['planning_costs']['complete_wall_seconds'],
        'replay_seconds':result['replay_costs']['complete_wall_seconds'],
        'original_planning_frozen_forwards':forwards,'history_hash':result['history_hash'],
        'terminal_decision_catalog':catalog}

def run(args):
    stage=HERE.parent;out=stage/'recovery-01';runtime=stage/'replay-recovery'
    r=read_json(out/'report.json',args.report_sha256)
    launcher=read_json(out/'launcher.json',args.launcher_sha256)
    index=read_json(runtime/'source-index.json',args.index_sha256)
    need(index['execution_released'] and r['source_index_sha256']==launcher['source_index_sha256']==args.index_sha256
         and launcher['report_sha256']==args.report_sha256,'Exact released recovery terminal/source pins')
    config=read_json(runtime/'config.json',REVIEWED_RUNTIME['config.json'])
    need(config==r['config'] and config['subject']==SUBJECT and config['methods']==list(ARMS)
         and config['max_steps']==24 and config['caps']['frozen_policy_forwards']==0
         and config['output_directory']=='build/continuation-endgame-pair-v1/recovery-01',
         'Same declared two-plan replay, absolute24 and zero model budget')
    for name,digest in REVIEWED_RUNTIME.items():
        path=runtime/name;relative=str(path.relative_to(ROOT))
        need(index['files'][relative]==digest and hashlib.sha256(path.read_bytes()).hexdigest()==digest,
             'Exact independently reviewed substantive recovery runtime '+name)
    prior_index=read_json(ROOT/OLD_INDEX,OLD_INDEX_SHA)
    canonical=lambda data:{p:d for p,d in data['files'].items() if p.startswith('src/')}
    need(canonical(index)==canonical(prior_index),'Exact original canonical scientific source closure')
    need(index['checkpoints']=={} and index['public_arrays']==prior_index['public_arrays']
         and len(index['public_arrays'])==5,'No checkpoint allowlist, exact original045 five public arrays')
    need(all(not p.endswith(('.zip','.psckpt','.qckpt','.npy')) for p in index['input_metadata']),
         'Metadata admission exposes no cache/checkpoint/array payload')
    for name,digest in PINS.items():
        need(index['input_metadata'][FAILED+'/'+name]==digest,'Original failed input fixed '+name)
    need(index['input_metadata'][OLD_INDEX]==OLD_INDEX_SHA
         and index['input_metadata'][FAILURE_AUDIT]==FAILURE_SHA,'Original authority and independent recovery-point pins')
    failed=read_json(ROOT/FAILED/'report.json',PINS['report.json'])
    failed_parent=read_json(ROOT/FAILED/'launcher.json',PINS['launcher.json'])
    plans=read_json(ROOT/FAILED/'plans.json',PINS['plans.json'])
    reviewed=read_json(ROOT/FAILURE_AUDIT,FAILURE_SHA)
    need(reviewed['audit_status']=='PASS_SAVED_RECOVERY_POINT'
         and reviewed['recovery_point']=='PASS_SEALED_PLANS_FOR_REPLAY_ONLY_NOT_ACCEPTED_EXPERIMENT'
         and reviewed['report_sha256']==PINS['report.json'] and reviewed['plans_sha256']==PINS['plans.json'],
         'Independent origin proves sealed plans, not accepted failed endpoints')
    need(failed['status']=='failed_or_incomplete' and failed['branches']=={}
         and failed_parent['status']=='incomplete_or_failed' and failed_parent['worker_exit_code']==1,
         'Original scientific/publication failure remains immutable negative')
    retained=read_json(ROOT/(pair.TAIL+'/report.json'),pair.TAIL_SHA)
    original,prefix,common=pair.declared_treatment(failed,retained)
    need(r['treatment']==failed['treatment'] and plans==failed['plans']
         and set(plans)==set(ARMS),'Exact previously sealed plans/treatment, no replanning')
    origin=r['failed_attempt']
    expected_origin={'directory':FAILED,'pins':PINS,'source_index_sha256':OLD_INDEX_SHA,
        'status':failed['status'],'failure':failed['failure'],'worker_seconds':failed['worker_seconds'],
        'parent_seconds':failed_parent['elapsed_seconds'],'native_previews':failed['native_budget']['native_preview_entries'],
        'frozen_policy_forwards':failed['total_frozen_policy_forwards'],'head_optimizer_updates':failed['head_optimizer_updates'],
        'calibration_calls':failed['calibration_calls_completed'],'tail_calls':failed['tail_calls_completed'],
        'original_failure_preserved':True,'prior_cumulative_costs':failed['cumulative_iteration_costs']}
    need(origin==expected_origin and origin['native_previews']==3458 and origin['frozen_policy_forwards']==24,
         'Full failed attempt and nested prior costs retained, never zeroed')
    admission=r['plan_admission']
    expected_admission={'accepted':True,'source':'exact_failed_pair_plans_before_any_replay',
        'plans_sha256':PINS['plans.json'],'plan_seals':{m:plans[m]['plan_seal'] for m in ARMS},
        'original_context_hash':plans[ARMS[0]]['planning']['context_hash'],
        'head_hashes':{k:failed[k]['fitted_head_hash'] for k in ('head_fit','calibration','tail_fit')},
        'new_model_cache_checkpoint_loads':0,'new_planning_calls':0}
    need(admission==expected_admission==read_json(out/'plan-admission.json')
         and r['independent_recovery_point']=={'path':FAILURE_AUDIT,'sha256':FAILURE_SHA},
         'Exact original context/head/seal admission, no fitted payload needed')
    need(r['sealed_plans_admitted'] and r['source_unchanged'] and not r['closed_roles_opened']
         and all(r[k]==0 for k in ('checkpoint_loads','cache_loads','head_optimizer_updates',
              'new_planning_calls','new_training_rollouts','total_frozen_policy_forwards')),
         'No role expansion/replanning/training/cache/model/policy-forward work')
    if r['status']!='completed':
        return {'scope':'Replay recovery incomplete; no paired causal effect accepted',
            'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,
            'source_index_sha256':args.index_sha256,'comparison_admissible':False,
            'report_status':r['status'],'supervision_status':launcher['status'],'failure':r.get('failure'),
            'partial_branches':list(r['branches']),'costs':r['costs'],'native_budget':r['native_budget'],
            'original_failure_retained':origin,'input_files':base.READS}
    need(set(r['branches'])==set(ARMS),'Both full authoritative endpoints present')
    # Only the old fit metadata is supplied to the independent scalar checks;
    # it was never supplied to the recovery actor (there is no actor call).
    audit_report={**r,'original_fit_for_audit':failed['tail_fit']}
    summaries={};evidence_pins={}
    for method in ARMS:
        plan=plans[method];result=r['branches'][method]
        need(plan['complete'] and plan['plan_seal']==admission['plan_seals'][method]
             ==failed['plans_sealed_before_replay']['seals'][method]
             ==semantic({k:v for k,v in plan.items() if k!='plan_seal'})==result['plan_seal'],
             'External complete plan/action/model/cost seal unchanged before fresh replay')
        need(plan['planning']['context_hash']==admission['original_context_hash'],
             'Both original plans share original admission identity')
        evidence_path=out/'evaluation'/(method+'-evidence.json')
        evidence=read_json(evidence_path)
        added={'planning_replay_join','replay_costs'}|({'original_route_reproduced'} if method=='RETAINED_TAIL' else set())
        need(set(result)-set(evidence)==added
             and {k:v for k,v in result.items() if k not in added}==evidence,
             'Full independent evidence saved before parity; only joins/costs added afterward')
        evidence_pins[method]=hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        need(result['replay_costs']==r['costs']['replay.'+method],'Exact inclusive replay phase attribution')
        summaries[method]=branch(result,plan,method,prefix,common,audit_report)
    a,b=(r['branches'][m] for m in ARMS)
    context=read_json(out/(SUBJECT+'-context.json'))
    need(semantic(context)==a['context_hash'] and context['role']=='TRAIN'
         and {k:context[k] for k in a['world']}==a['world'],
         'Persisted fresh TRAIN admission seals the exact replay context/public world')
    need(a['world']==b['world']==original['world'] and a['context_hash']==b['context_hash']
         and a['decisions'][:8]==b['decisions'][:8] and a['history'][:8]==b['history'][:8],
         'Exact same new replay context/public world/eight-prefix for both arms')
    for field in ('actions','decisions','history','world','nominal_history_hash','history_hash'):
        need(a[field]==original[field],'Full exact JSON-normalized original route parity '+field)
    need(a['audit']['outcomes']==original['audit']['outcomes'] and a['original_route_reproduced']
         and r['baseline_reproduced'],'Retained endpoint exactly reproduces every outcome')
    selected={v['action_id']:v for v in common['scores']}
    left,right=(selected[FORCED[m]] for m in ARMS)
    delta=summaries[ARMS[0]]['remaining_return_including_forced_action']-summaries[ARMS[1]]['remaining_return_including_forced_action']
    predicted=left['combined_nominal_q']-right['combined_nominal_q']
    immediate=left['reward']-right['reward'];tail=left['clipped_predicted_tail']-right['clipped_predicted_tail']
    diagnosis=r['diagnosis']
    need(diagnosis['comparison_admissible'] and not diagnosis['deployable_improvement_claim']
         and diagnosis['difference_orientation']=='RETAINED_TAIL_minus_IMMEDIATE_BEST','Fixed-pair causal scope only')
    close(diagnosis['prefix_return'],summaries[ARMS[0]]['prefix_return'],'Shared prefix')
    for method in ARMS:
        close(diagnosis['remaining_return_including_forced_action'][method],summaries[method]['remaining_return_including_forced_action'],
              'Full suffix includes forced action once')
        close(summaries[method]['prefix_return']+summaries[method]['remaining_return_including_forced_action'],
              summaries[method]['outcomes']['total_reward'],'All initial preparation paid once per route')
    for key,value in [('realized_remaining_return_difference',delta),('predicted_remaining_Q_difference',predicted),
                      ('predicted_tail_difference',tail),('immediate_reward_difference',immediate),
                      ('prediction_difference_error',predicted-delta)]:
        close(diagnosis[key],value,'Independent diagnosis '+key)
    close(predicted,immediate+tail,'Immediate plus predicted tail difference')
    full_delta=a['audit']['outcomes']['total_reward']-b['audit']['outcomes']['total_reward']
    close(full_delta,delta,'Common prefix cancels, no omitted cost')
    need(diagnosis['realized_order_matches_predicted']==((delta>0)==(predicted>0) if delta and predicted else None),
         'Realized ordering/tie against forecast')
    for field,value in diagnosis['full_outcome_difference'].items():
        close(value,a['audit']['outcomes'][field]-b['audit']['outcomes'][field],'Physical outcome difference '+field)
    for method in ARMS:
        need(diagnosis['full_outcomes'][method]=={k:r['branches'][method]['audit']['outcomes'][k]
             for k in diagnosis['full_outcomes'][method]},'Exact full physical outcome projection')
    visits=r['source_visits'];need(len(visits)==1 and visits[0]['subject']==SUBJECT and visits[0]['source_released'],
         'One released public045 source for both fresh replays')
    costs=r['costs'];budget=r['native_budget'];caps=config['caps']
    reconstruction=costs['same045_source.public_reconstruction'];enclosing=costs['same045_source']
    replay_previews=sum(r['branches'][m]['replay_costs']['native_preview_entries'] for m in ARMS)
    need(reconstruction['native_preview_entries']+replay_previews==budget['native_preview_entries']
         and enclosing['native_preview_entries']==replay_previews
         and costs['saved_plan_metadata_admission']['native_preview_entries']==0,
         'Setup plus two disjoint replay phases join total; enclosing phase not double counted')
    for key,row in costs.items():
        need(row.get('policy_forward_calls',0)==row.get('policy_forward_returned',0)==0,'Zero new forwards in every phase')
    transitions=sum(r['branches'][m]['replay_costs']['native_transition_calls'] for m in ARMS)
    need(transitions==sum(len(plans[m]['actions']) for m in ARMS)
         and all(r['branches'][m]['replay_costs']['native_transition_calls']
                 ==r['branches'][m]['replay_costs']['native_transition_returned'] for m in ARMS),
         'All fresh full prefix/treatment/suffix replay commits paid once')
    phases=budget['native_preview_profile']['phases']
    need(sum(v['started'] for v in phases.values())==sum(v['returned'] for v in phases.values())==budget['native_preview_entries']
         and all(v['raised']==0 for v in phases.values()),'Started/returned native-preview profile accounting')
    cumulative=r['pair_plus_recovery_costs']
    need(cumulative['native_previews']==3458+budget['native_preview_entries']
         and cumulative['frozen_policy_forwards']==24
         and cumulative['original_fitting_and_replay_failure_cost_not_erased']
         and cumulative['earlier_iterations']==failed['cumulative_iteration_costs'],
         'Failed pair plus recovery exact, earlier_iterations explicitly already includes failed pair')
    close(cumulative['worker_seconds'],failed['worker_seconds']+r['worker_seconds'],'Pair+recovery measured worker time')
    cleanup=launcher.get('owned_lifecycle') or {}
    clean=(launcher['status']=='completed' and launcher.get('stop_reason') is None
        and launcher.get('worker_exit_code')==0 and cleanup.get('verified') and cleanup.get('child_reaped')
        and cleanup.get('exit_code')==0 and not cleanup.get('remaining_owned_pids')
        and not cleanup.get('remaining_state_unknown') and not cleanup.get('errors'))
    if launcher['status']=='completed':need(clean,'Verified clean root supervision')
    need(budget['counting_reliable'] and budget['status']=='complete_history_awaiting_independent_audit'
         and budget['blocked_preview_attempts']==0 and budget['native_preview_entries']<=caps['native_previews']
         and launcher['elapsed_seconds']<caps['parent_wall_seconds'] and r['worker_seconds']<caps['work_seconds']
         and launcher['rss_samples']>0 and launcher['sampled_peak_group_rss_bytes']<=caps['worker_peak_rss_bytes']
         and r['self_peak_rss_bytes']<=caps['worker_peak_rss_bytes'],'Measured declared time/counter/RSS caps')
    return {'scope':'One fixed simulator state/treatment continuation; no policy/clinical improvement claim',
        'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,'source_index_sha256':args.index_sha256,
        'comparison_admissible':True,'clean_supervision':bool(clean),'supervision_status':launcher['status'],
        'original_failure_retained':origin,'original_full_route_parity':True,'world':a['world'],
        'original_planning_context_hash':admission['original_context_hash'],'recovery_context_hash':a['context_hash'],
        'both_original_plan_seals':admission['plan_seals'],'preassertion_evidence_sha256':evidence_pins,
        'common_state_hash':common['source_state_hash'],'common_observation_hash':common['observation_hash'],
        'common_current_tool':common['current_tool'],'outcomes':summaries,
        'independent_effect':{'difference_orientation':'RETAINED_TAIL_minus_IMMEDIATE_BEST',
          'full_return_difference':full_delta,'remaining_return_difference':delta,
          'immediate_reward_difference':immediate,'predicted_tail_difference':tail,
          'predicted_Q_difference':predicted,'prediction_difference_error':predicted-delta},
        'current_native_previews':budget['native_preview_entries'],'current_replay_transitions':transitions,
        'current_frozen_forwards':0,'current_head_updates':0,'current_fits':0,'current_policy_plans':0,
        'costs_inclusive':costs,'parent_seconds':launcher['elapsed_seconds'],'worker_seconds':r['worker_seconds'],
        'sampled_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],'pair_plus_recovery_costs':cumulative,
        'independent_all_iterations_costs':{
          'native_previews':failed['cumulative_iteration_costs']['native_preview_entries']+budget['native_preview_entries'],
          'frozen_policy_forwards':failed['cumulative_iteration_costs']['frozen_policy_forwards'],
          'scope':'cache-fill failure + direct recovery + tail + failed pair + fresh replay recovery; older original64 separate'},
        'limits':['Source/record sequencing verified; no native or neural computation repeated by reviewer',
          'Context identity changes only for release authority; every public world/state/observation/geometry record remains joined',
          'Entire retained JSON histories/decisions equal with no omitted values or numerical tolerance',
          'Preterminal catalog is output-only and covers emitted candidates, not all possible geometry',
          'Fixed simulator pair diagnoses decision credit, not general policy improvement or observed surgical outcomes',
          'Failed original remains failed; planning/fitting/first replay paid once plus both new full replays',
          'Inclusive timing/counter phases nest and must not be summed twice'],
        'input_files':base.READS}


def main():
    parser=argparse.ArgumentParser()
    for key in ('report-sha256','launcher-sha256','index-sha256','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();target=Path(args.output).resolve()
    need(target.parent==HERE and not target.exists(),'Exclusive independent receipt within owned scope')
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Saved audit30s cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30.)
    try:result=run(args);result['audit_status']='PASS_SAVED_EVIDENCE';code=0
    except BaseException as error:
        result={'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(error).__name__,'message':str(error)},
            'input_files':base.READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0.)
    result.update(checks=base.CHECKS,seconds=time.monotonic()-started,model_calls=0,native_calls=0,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(result,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    need(len(raw)<=64*1024,'Receipt64KiB cap')
    with target.open('xb') as stream:stream.write(raw)
    print(json.dumps({k:result[k] for k in ('audit_status','checks','seconds','model_calls','native_calls')}))
    return code

if __name__=='__main__':raise SystemExit(main())
