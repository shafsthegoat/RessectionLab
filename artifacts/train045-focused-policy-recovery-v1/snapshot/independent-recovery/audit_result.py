"""Bounded independent saved-evidence audit; no scientific runtime imports.

Source preparation only until root supplies terminal report/launcher hashes.
The only gzip inputs are full saved JSON traces and authenticated sidecars.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
INDEX = 'adb6d91cb2106f957c085fee7c6b438fc095510f63c903dc9c9d0b8f48677b40'
OLD_REPORT = 'c31b88e88c77b8f50d262193ce2424d4abf203eb23fbab14eed63de032229038'
OLD_AUDIT = 'f02af1cd2731162b3bc7c27d1d683b87e6b23904b7e4fdf3c5188dbc648bf4b1'
CONTROLLER = 'initial_none_then_first_current_goal_overlap_active_first_v1'


def load_own(name, filename, digest):
    path = STAGE/'independent'/filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Immutable independent source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_own('_focused_failure_reader', 'audit_failure.py',
    'c289ad8f9292d86cdcd2ad9c07d371226a87c7ef90eb0dfaf30110ada40d93c3')
oracle = load_own('_focused_scalar_oracle', 'scalar_oracles.py',
    'f7c4ce700b44816d9c61708dc7cf0d8b852c6055e8a5f45801578d869d063af6')
semantic = base.semantic


def exact(a, x, y, label):
    a.need(semantic(x) == semantic(y), label)


def sealed(a, row, field='seal'):
    a.need(row[field] == semantic({k:v for k,v in row.items() if k != field}), 'Valid semantic seal')


def point_key(row):
    return row['tool_id'], tuple(row['entry_mm']), tuple(row['tip_mm'])


def audit_arm(a, method, account, branch, plan, prefix, frontier, sidecars, reward):
    packets = plan['selected_action_records']; history = account['explored_history']
    n = len(packets); suffix = n-12
    sealed(a, plan)
    a.need(account['complete'] is True and account['method'] == method
        and account['controller'] == CONTROLLER and account['initial_focus'] is None
        and account['fixed_prefix_length'] == 12 and account['positive_immediate_guard'] is True
        and account['training_updates'] == 0 and account['selected_equals_explored'] is True,
        'Complete declared same-controller positive-guard method')
    a.need(12 < n <= 24 and len(history) == n and len(account['states']) == suffix
        and account['model_transition_calls'] == account['model_transition_completed'] == suffix,
        'Absolute24 horizon and paid suffix transitions')
    exact(a, packets[:12], prefix, 'Authenticated fixed prefix in full plan')
    exact(a, packets, account['explored_action_records'], 'Full plan equals actual planned packets')
    exact(a, semantic(history), plan['planned_history_hash'], 'Full planned history binding')
    exact(a, semantic(history[:12]), frontier['nominal_history_hash'], 'Same actual prefix history')
    stops = [i for i,p in enumerate(packets) if p['action_id']=='STOP']
    a.need(stops == [n-1] if stops else n == 24, 'Complete STOP-or-absolute-horizon endpoint')
    total = sum(row['reward'] for row in history)
    a.close(total, plan['planned_total_return'], 'Full reward sum')
    a.close(total, account['selected_total_return'], 'Full planning reward')
    a.close(sum(row['reward'] for row in history[12:]), account['selected_net_suffix_return'],
        'Suffix return excludes inherited prefix value')
    a.close(account['fixed_prefix_return'], frontier['total_reward'], 'Unchanged scalar incumbent')
    a.need(branch['complete'] is True and branch['planned_physical_history_matches'] is True
        and branch['plan_seal'] == plan['seal'] and branch['actions'] == [p['action_id'] for p in packets],
        'Full authoritative replay binds complete sealed plan')
    cert = branch['audit']; out = cert['outcomes']
    a.need(cert['accepted'] is True and cert['complete_episode'] is True and cert['geometry']['feasible'] is True,
        'Existing complete independent geometry certificate accepted')
    a.need(cert['committed_history_hash'] == branch['history_hash']
        and cert['source_hash'] == branch['source_hash'] == frontier['source_hash']
        and cert['decision_model_hash'] == branch['decision_model_hash'] == frontier['decision_model_hash']
        and branch['native_model_hash'] == frontier['native_model_hash'], 'Endpoint source/model/history join')
    a.need(len(branch['history']) == n, 'Complete replay history denominator')
    reconstructed = []; previous_tool = None; changes=0; removed=set(); contact=set()
    volume_per_cell=None
    for i, (packet,row,compact) in enumerate(zip(packets,history,branch['history'])):
        sealed(a,packet)
        a.need(packet['steps_taken'] == i and packet['current_tool'] == previous_tool
            and packet['source_hash'] == frontier['source_hash']
            and packet['decision_model_hash'] == frontier['decision_model_hash']
            and packet['action_id'] == row['action_id'] == compact['action_id'], 'Packet state/tool/action sequence')
        actual=dict(row)
        if packet['action_id']=='STOP':
            a.need(packet['physical_action'] is None and row['reward']==0
                and not row.get('removed_indices_native') and not row.get('microsteps'), 'STOP has no tissue or reward')
        else:
            a.need(row['outcome_scope']=='permitted_nominal_model'
                and packet['physical_action']=={k:row[k] for k in ('tool_id','entry_mm','tip_mm')},
                'Planning uses public nominal feedback and exact physical packet')
            actual['outcome_scope']='separate_evaluator_reference'
            cells={tuple(c) for c in row['removed_indices_native']}
            a.need(len(cells)==len(row['removed_indices_native']) and not removed.intersection(cells),
                'No duplicate removed cell reward')
            removed.update(cells); contact.update(tuple(c) for c in row.get('contact_indices_native',[]))
            if cells:
                v=(row['target_removed_mm3']+row['normal_removed_mm3'])/len(cells)
                if volume_per_cell is None:volume_per_cell=v
                a.close(v,volume_per_cell,'Consistent stored native cell volume')
            start=row.get('physical_start_mm',row['entry_mm'])
            distance=math.sqrt(sum((x-y)**2 for x,y in zip(row['tip_mm'],start)))
            changed=int(previous_tool is not None and previous_tool!=row['tool_id']);changes+=changed
            expected=(reward['target_per_mm3']*row['target_removed_mm3']
                -reward['normal_per_mm3']*row['normal_removed_mm3']-reward['action_cost']
                -2*reward['motion_per_mm']*distance-reward['tool_change_cost']*changed)
            a.close(expected,row['reward'],'Unchanged declared reward including tool and path costs')
            a.close(2*distance,row['complete_tool_path_length_mm'],'Complete-tool travel')
            previous_tool=row['tool_id']
            if i>=12:a.need(row['reward']>0.,'Strict actual-positive suffix action')
        reconstructed.append(actual)
        a.need(semantic(actual)==compact['history_row_hash'], 'Exact full replay row; scope only normalized')
        a.need(semantic(row.get('removed_indices_native',[]))==compact['removed_indices_hash']
            and semantic(row.get('contact_indices_native',[]))==compact['contact_indices_hash'], 'Replay cell-set digests')
    a.need(semantic(reconstructed)==branch['history_hash'], 'Reconstructed complete authoritative history hash')
    for field in ('target_removed_mm3','normal_removed_mm3','total_reward','complete_tool_path_length_mm'):
        source='reward' if field=='total_reward' else field
        a.close(sum(r[source] for r in history),out[field],'Independent saved scalar outcome sum '+field)
    a.need(out['tool_changes']==changes and out['nonstop_actions']==n-len(stops),'Exact intervention/switch counts')
    if volume_per_cell is not None:
        a.close(len(removed)*volume_per_cell,out['simulated_removed_volume_mm3'],'Unique removed-cell volume')
        a.close(len(contact)*volume_per_cell,out['cumulative_contacted_tissue_upper_bound_mm3'],'Union contact bound')
        a.close(len(contact-removed)*volume_per_cell,out['currently_retained_contacted_tissue_upper_bound_mm3'],'Retained contact bound')
    a.close(out['target_removed_mm3']/out['total_reference_target_mm3'],out['reference_target_fraction_removed'],
        'Full target denominator retained')
    a.need(out['partial_contact_reward_weight']==0. and out['motor_surrogate'] is None
        and out['language_surrogate'] is None and out['clinical_deficit_probability'] is None,
        'No physical/clinical inference added to geometric reward')
    choices=[]; goals_count=0; pages_count=0; preview_lower=0; forward_states=0
    active=None
    for j,state in enumerate(account['states']):
        i=12+j; candidates=state['candidates']; ids=[r['action_id'] for r in candidates]
        pages=state['pages']; goals=state['goals']; selected=state['selected']; packet=packets[i]
        a.need(state['steps']==i and len(ids)==len(set(ids))<=153 and ids[0]=='STOP'
            and state['complete_page_enumeration'] is True, 'Unique complete current full catalog')
        a.need([p['page_index'] for p in pages]==list(range(len(pages))) and 1<=len(pages)<=5,
            'Every bounded page visited in order')
        extra=pages[0]['additional_emitted_count']; base_count=pages[0]['base_emitted_count']
        a.need(base_count<=120 and extra<=32 and len(pages)==1+(extra+7)//8
            and len(goals)<=120,'Same discovery/action limits')
        ordered=['STOP'];owners=[0]
        for owner,page in enumerate(pages):
            a.need(page['source_state_hash']==state['source_state_hash'] and page['steps_taken']==i
                and page['source_hash']==frontier['source_hash'] and page['decision_model_hash']==frontier['decision_model_hash'],
                'All pages from same current world/state')
            ordered.extend(page['page_action_ids']);owners.extend([owner]*len(page['page_action_ids']))
        a.need(ids==ordered and [r['owner_page'] for r in candidates]==owners,
            'Base deduplicated once; exact extra-page ownership and stable ties')
        rewards=[r['immediate_reward'] for r in candidates]
        positive=any(r>0 for r in rewards[1:])
        raw=[r['raw_tail'] for r in candidates] if method=='TAIL' and positive else None
        if raw is None:a.need(all(r['raw_tail'] is None and r['clipped_tail'] is None for r in candidates),
            'No invented tail values without forward')
        else:forward_states+=1
        expected=oracle.choose_positive(ids,rewards,raw)
        a.need(selected['action_id']==expected and selected['packet']==packet,'Independent selected scalar argmax')
        for k,r in enumerate(candidates):
            a.need(r['actual_positive'] is (rewards[k]>0),'Actual reward eligibility')
            value=rewards[k] if raw is None else rewards[k]+max(0.,raw[k])
            a.close(r['ranking_value'],value,'Immediate reward once plus clipped raw tail')
            if raw is not None:a.close(r['clipped_tail'],max(0.,raw[k]),'Tail clipping exact')
        winner=ids.index(expected)
        a.need(packet['query_catalog_digest']==state['query_catalog_digest'] and packet['focus']==state['goal_focus']
            and packet['source_state_hash']==state['source_state_hash'], 'Chosen packet binds exact current query')
        if expected!='STOP':
            matches=[r for r in pages[owners[winner]]['ledger'] if r.get('action_id')==expected
                and r.get('verification')=='NATIVE_VERIFIED']
            a.need(bool(matches) and all(r['feasible'] is True and
                {k:r[k] for k in ('tool_id','entry_mm','tip_mm')}==packet['physical_action'] for r in matches),
                'Chosen emitted native certificate is owned by selected page')
        a.close(selected['immediate_reward'],history[i]['reward'],'Selected certified/committed reward')
        a.close(selected['cumulative_return'],sum(r['reward'] for r in history[:i+1]),'Full cumulative return once')
        expected_active=active
        if state['focus_explicitly_cleared'] is not None:
            exact(a,state['focus_explicitly_cleared'],active,'Only prior active goal explicitly cleared')
            expected_active=None
        exact(a,state['active_goal'],expected_active,'Same deterministic active-goal lifecycle')
        next_active=expected_active;effect=None
        cells={tuple(c) for c in history[i].get('removed_indices_native',[])}
        for goal in goals:
            side=goal['sidecar']
            if side is not None:
                a.need(sidecars['records'][side['fingerprint']]==side
                    and side['source_state_hash']==state['source_state_hash'], 'Current goal sidecar authenticated')
            if expected=='STOP':continue
            direct=point_key(history[i])==point_key(goal['goal']) and history[i]['target_removed_mm3']>0
            overlap=sorted(cells.intersection(tuple(c) for c in goal['retained_blocker_cells']))
            if direct or overlap:
                next_active=None if direct else goal['goal']
                effect={'goal':goal['goal'],'direct_goal':direct,'current_blockers_removed':[list(c) for c in overlap]}
                break
        exact(a,selected['focus_effect'],effect,'Shared controller uses actual public goal overlap, not learned score')
        exact(a,selected['next_active_goal'],next_active,'Next active physical goal')
        active=next_active
        goals_count+=len(goals);pages_count+=len(pages);preview_lower+=base_count+extra+len(goals)
        choices.append({'absolute_action':i+1,'action_id':expected,'physical_action':packet['physical_action'],
            'catalog_actions':len(ids)-1,'owner_page':owners[winner],'reward':history[i]['reward'],
            'tail':None if raw is None else raw[winner],
            'immediate_regret':max(rewards)-rewards[winner],
            'state_hash':state['source_state_hash'],'query_catalog_digest':state['query_catalog_digest']})
    a.need(account['costs']['union_observations']==suffix and account['costs']['pages_completed']==pages_count
        and account['costs']['goal_previews_started']==account['costs']['goal_previews_completed']==goals_count,
        'Discovery and observation counters from saved states')
    if method=='TAIL':
        cost=account['scorer_costs']
        a.need(cost['policy_forwards_started']==cost['policy_forwards_completed']==
            cost['head_forwards_started']==cost['head_forwards_completed']==
            cost['score_calls_started']==cost['score_calls_completed']==forward_states,
            'One unchanged full policy and head forward per productive state')
        a.need(cost['candidate_rows_scored']==sum(len(s['candidates']) for s in account['states']
            if any(r['immediate_reward']>0 for r in s['candidates'][1:])), 'All current rows scored once')
    else:a.need(account['scorer_costs'] is None,'GREEDY needs no model')
    return {'outcomes':out,'suffix_return':sum(r['reward'] for r in history[12:]),
        'suffix_transitions':suffix,'planning_preview_lower':preview_lower,'tail_forwards':forward_states,
        'choices':choices,'history_hash':branch['history_hash'],'plan_seal':plan['seal']}


def inspect(a, report_sha, launcher_sha):
    directory=STAGE/'recovery-01'
    index=a.read(STAGE/'runner-recovery/source-index.json',INDEX)
    report=a.read(directory/'report.json',report_sha);parent=a.read(directory/'launcher.json',launcher_sha)
    a.need(report['source_index_sha256']==parent['source_index_sha256']==INDEX
        and parent['report_sha256']==report_sha and index['execution_released'] is True,
        'External immutable terminal/source pins')
    for path,digest in index['files'].items():a.raw(ROOT/path,digest)
    life=parent['owned_lifecycle']
    a.need(life['verified'] is True and life['child_reaped'] is True and life['remaining_owned_pids']==[]
        and life['errors']==[] and life['remaining_state_unknown'] is False,'Verified final owned-process cleanup')
    a.need(report['source_unchanged'] is True,'Source closure unchanged')
    if report['status']!='completed' or parent['status']!='completed':
        return {'completed_comparison':False,'status':'preserved_terminal_failure','failure':report.get('failure'),
            'worker_status':report['status'],'parent_status':parent['status'],
            'costs':report.get('costs'),'native_budget':report.get('native_budget'),
            'cumulative_costs':report.get('failed_plus_recovery_costs'),'cleanup':life}
    a.need(parent['worker_exit_code']==life['exit_code']==0 and parent['stop_reason'] is None,'Clean completed worker')
    a.need(report['baseline_prefix_reproduced'] is True and report['frozen_models_unchanged'] is True
        and report['closed_roles_opened'] is False and report['cache_loads']==report['head_optimizer_updates']==0,
        'Inference only, unchanged frozen models and fixed TRAIN role')
    a.need(report['checkpoint_loads']==report['checkpoint_loads_attempted']==2
        and len(report['source_visits'])==1 and report['source_visits'][0]['subject']=='ReMIND-045'
        and report['source_visits'][0]['source_released'] is True,'Exact source/model denominator')
    a.need(report['model_provenance']['checkpoints']==index['checkpoints'],'Saved-head external payload pins')
    old=a.read(STAGE/'run-01/report.json',OLD_REPORT)
    old_audit=a.read(STAGE/'independent/failure-audit-01.json',OLD_AUDIT)
    a.need(old_audit['status']=='PASS_failure_evidence' and old['status']=='failed_or_incomplete',
        'Original incomplete attempt remains negative')
    prior=report['preserved_failed_attempt']
    a.need(prior['report_sha256']==OLD_REPORT and prior['native_previews']==old['native_budget']['native_preview_entries']==4733
        and prior['policy_forwards']==0 and prior['native_transition_counts']==old['native_transition_counts'],
        'Preserved exact failed attempt and paid work')
    tail_path='build/continuation-tail-v1/run-01/report.json'
    saved_tail=a.read(ROOT/tail_path,index['input_metadata'][tail_path])
    exact(a,report['model_provenance']['original_tail_fit'],saved_tail['tail_fit'],'Original head fit record unchanged')
    exact(a,report['model_provenance']['original_training_worlds'],saved_tail['corpus']['worlds'],'All original trained worlds unchanged')
    a.need(set(saved_tail['corpus']['worlds'])=={'ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'},
        'No changed learning split')
    old_index=a.read(STAGE/'runner/source-index.json',old['source_index_sha256'])
    exact(a,index['public_arrays'],old_index['public_arrays'],'Same five public payload pins; arrays not decoded')
    a.need(len(index['public_arrays'])==5,'Exact unchanged public source input count')
    prefix_file=a.read(directory/'fixed-prefix.json');prefix=prefix_file['focused_packets'];frontier=report['fixed_frontier']
    sealed(a,frontier);exact(a,frontier,prefix_file['focused_frontier'],'Shared actual frontier')
    a.need(frontier['steps']==len(prefix)==len(prefix_file['joins'])==12,'Fixed twelve actual actions')
    exact(a,frontier,old['fixed_frontier'],'Same physical/model prefix as retained failed experiment')
    deployment=a.read(directory/'focused-deployment.json');a.need(deployment['seal']==semantic(deployment['record']),
        'Explicit separately sealed focused deployment')
    d=deployment['record']
    a.need(d['prefix_length']==12 and d['training_updates']==0 and d['feature_change'] is False
        and d['private_reference_input'] is False and d['controller']==CONTROLLER
        and d['candidate_limit']==153 and d['trained_world']['subject']=='ReMIND-045'
        and d['trained_world']['max_steps']==24,'New action-family admission with unchanged public features')
    exact(a,d['trained_world'],saved_tail['corpus']['worlds']['ReMIND-045'],'Explicit old-world→new-context join')
    seal=report['plans_sealed_before_replay']
    plans=a.read(directory/'plans.json',seal['sha256'])
    a.need(set(plans)==set(report['branches'])==set(report['selections'])=={'GREEDY','TAIL'},'Both complete arms retained')
    accounts={m:a.compressed(directory,report['selections'][m]) for m in ('GREEDY','TAIL')}
    sidecars=a.compressed(directory,report['sidecars'])
    a.need(len(sidecars['records'])==report['sidecars']['unique_count']<=4096,'Bounded unique saved sidecars')
    for key,row in sidecars['records'].items():
        a.need(key==row['fingerprint']==semantic({k:v for k,v in row.items() if k!='fingerprint'}),'Exact sidecar authentication')
    result={}
    for method in ('GREEDY','TAIL'):
        exact(a,accounts[method]['context'],d,'Same explicitly admitted context for both methods')
        a.need(accounts[method]['context_seal']==deployment['seal'] and seal['seals'][method]==plans[method]['seal'],
            'Context and pre-replay plan seals')
        endpoint=a.read(directory/'evaluation'/(method+'.json'))
        exact(a,endpoint,report['branches'][method],'Complete endpoint evidence retained before report acceptance')
        result[method]=audit_arm(a,method,accounts[method],endpoint,plans[method],prefix,frontier,sidecars,d['reward'])
    # Saved continuation is evaluator-only and checked after both new plans seal.
    transport=report['retained_trace_transport']
    a.need(transport['plans_json_sha256_before_decode']==seal['sha256']
        and report['retained_trace_loads_attempted']==report['retained_trace_loads_completed']==1,
        'Prior continuation decoded only after both complete plans were sealed')
    prior_trace=a.compressed(STAGE/'run-01',old['incomplete_selection']['GREEDY'])['policy']
    new=accounts['GREEDY']
    exact(a,prior_trace['explored_action_records'],new['explored_action_records'][:23],'Exact retained23 packet sequence')
    exact(a,prior_trace['explored_history'],new['explored_history'][:23],'Exact retained23 full native histories')
    for key in ('context','context_seal','controller','initial_focus','fixed_prefix_return'):
        exact(a,prior_trace[key],new[key],'Retained unchanged '+key)
    before=[r for r in prior_trace['states'] if 'result_state_hash' in r.get('selected',{})]
    a.need(len(before)==11,'Exact original committed decision denominator')
    exact(a,before,new['states'][:11],'Every retained decision: order, scores, goal, focus and full page ledgers')
    a.need(report['retained_greedy_prefix_parity']['accepted'] is True
        and report['retained_greedy_prefix_parity']['exact_committed_suffix_decisions']==11,'Reported parity independently corroborated')
    g=result['GREEDY']['outcomes'];t=result['TAIL']['outcomes']
    fields=('positive_target_source_cells_removed','target_removed_mm3','total_reward','normal_removed_mm3',
        'complete_tool_path_length_mm','tool_changes','nonstop_actions',
        'cumulative_contacted_tissue_upper_bound_mm3','currently_retained_contacted_tissue_upper_bound_mm3')
    delta={k:t[k]-g[k] for k in fields}
    nonreg=(delta['positive_target_source_cells_removed']>=0 and delta['target_removed_mm3']>=-1e-8
        and delta['total_reward']>=-1e-8 and delta['normal_removed_mm3']<=1e-8)
    strict=(delta['positive_target_source_cells_removed']>0 or delta['target_removed_mm3']>1e-8 or delta['total_reward']>1e-8)
    gate=bool(nonreg and strict)
    diagnosis=report['diagnosis']
    a.need(diagnosis['comparison_admissible'] is True and diagnosis['useful_gain'] is gate
        and diagnosis['quality_nonregression'] is nonreg and diagnosis['strict_quality_improvement'] is strict
        and diagnosis['tie_is_negative'] is True,'Independent exact prospective useful gate')
    exact(a,delta,diagnosis['deltas_tail_minus_greedy'],'All reported outcome deltas')
    a.close(g['total_reference_target_mm3'],t['total_reference_target_mm3'],'Unchanged full target denominator')
    costs=report['costs'];budget=report['native_budget'];caps=report['config']['caps']
    suffix=sum(result[m]['suffix_transitions'] for m in result)
    a.need(report['native_transition_counts']=={'attempted':36+2*suffix,'completed':36+2*suffix,'raised':0}
        and report['native_transition_attempts']==36+2*suffix
        and seal['native_transition_attempts_at_seal']==24+suffix,'Prefix/plan/replay native costs once')
    a.need(report['shared_replay']['prefix_transitions']==12
        and report['shared_replay']['suffix_transitions']=={m:result[m]['suffix_transitions'] for m in result},
        'One fresh shared prefix; two complete authoritative suffixes')
    leaf=[key for key in costs if key!='same045_source']
    count=sum(costs[key].get('native_preview_entries',0) for key in leaf)
    a.need(budget['counting_reliable'] is True and budget['failure'] is None
        and budget['history_complete_caller_attestation'] is True
        and budget['native_preview_entries']==count<=caps['native_previews'],'Reliable full-work preview total; nested scopes not doubled')
    forwards=result['TAIL']['tail_forwards']
    a.need(report['total_frozen_policy_forwards']==forwards<=caps['frozen_policy_forwards'],
        'All actual frozen policy inference paid')
    for method in result:
        phase=costs[method+'_suffix_planning']
        a.need(phase['native_preview_entries']==result[method]['planning_preview_lower'],
            'All base/extra/goal discovery queries charged exactly')
    exact(a,report['focused_tail_scoring_costs'],accounts['TAIL']['scorer_costs'],'Full wrapper costs retained')
    declaration=report['focused_scoring_declaration'];a.need(declaration['seal']==semantic(declaration['record']),
        'Frozen focused inference declaration')
    a.need(declaration['record']['head_hash']==index['checkpoints']['TAIL']['parameters']
        and declaration['record']['context_hash']==deployment['seal'], 'Exact saved head deployed through new capability')
    cumulative=report['failed_plus_recovery_costs']
    a.need(cumulative['native_previews']==4733+count and cumulative['native_transition_attempts']==35+36+2*suffix
        and cumulative['frozen_policy_forwards']==forwards and cumulative['checkpoint_loads']==4,
        'Failed attempt plus recovery counts preserved')
    # Worker accumulates this timestamp just before its own final timestamp;
    # it is a nested partial wall scope, not asserted equal to the later timer.
    a.need(old['worker_seconds']<=cumulative['worker_seconds']<=old['worker_seconds']+report['worker_seconds']
        and cumulative['prior_parent_seconds']==prior['parent_seconds'],'Cumulative worker time has truthful timestamp scope')
    a.need(report['worker_seconds']<caps['work_seconds'] and parent['elapsed_seconds']<caps['parent_wall_seconds']
        and max(report['self_peak_rss_bytes'],parent['sampled_peak_group_rss_bytes'])<=caps['worker_peak_rss_bytes']
        and report['native_transition_attempts']<=caps['native_transitions'],'Owned total resource caps')
    common=[]
    for x,y in zip(result['GREEDY']['choices'],result['TAIL']['choices']):
        if x['state_hash']!=y['state_hash'] or x['query_catalog_digest']!=y['query_catalog_digest']:break
        common.append({'absolute_action':x['absolute_action'],'same_choice':x['action_id']==y['action_id'],
            'greedy':x,'tail':y})
        if x['action_id']!=y['action_id']:break
    return {'completed_comparison':True,'useful_gain':gate,'quality_nonregression':nonreg,
        'strict_quality_improvement':strict,'deltas_tail_minus_greedy':delta,'arms':result,
        'common_state_choices_until_first_divergence':common,
        'retained11_decisions_exact':True,'model_refit':False,
        'costs':{'recovery_previews':count,'recovery_transitions':36+2*suffix,'recovery_forwards':forwards,
            'worker_seconds':report['worker_seconds'],'parent_seconds':parent['elapsed_seconds'],
            'sampled_peak_group_rss_bytes':parent['sampled_peak_group_rss_bytes'],
            'phase_costs':costs,'failed_plus_recovery':cumulative,
            'both_parent_seconds':prior['parent_seconds']+parent['elapsed_seconds'],
            'historical_prefix_costs':report['retained_prefix']['prior_costs_not_erased'],
            'scope':'Inclusive phase timers overlap; per-method timing is fixed-order N=1, not a generalized speed estimate.'},
        'claims':'One annotation-assisted TRAIN045 action-family transfer with a frozen tail head and native positive guard. Full target/outside/reward gate reproduced; travel/switch/contact reported separately. No new learned reward, safety, clinical or held-out claim.',
        'limits':'This auditor independently reconstructs saved scalar choices, rewards and parity. Geometry certificates were already produced by the independent full-history worker checker. No arrays or neural predictions independently recomputed here.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--report-sha256',required=True)
    p.add_argument('--launcher-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output).resolve()
    if output.parent!=HERE or output.exists():raise ValueError('Fresh output in independent-recovery required')
    started=time.monotonic();a=base.Audit();receipt={'schema':'focused-policy-recovery-independent-result-v1',
        'pins':{'report':args.report_sha256,'launcher':args.launcher_sha256,'index':INDEX}}
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved audit')));signal.alarm(30)
    code=0
    try:
        findings=inspect(a,args.report_sha256,args.launcher_sha256)
        receipt.update(status='PASS_saved_evidence',findings=findings)
    except BaseException as error:
        code=1;receipt.update(status='FAIL_reader_or_evidence',failure={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.alarm(0)
        # Source closure checked in full but indexed once; avoid a second copy.
        sources={k:v for k,v in a.files.items() if k.endswith('.py')}
        receipt.update(checks=a.checks,elapsed_seconds=time.monotonic()-started,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            checked_source_count=len(sources),checked_sources_digest=semantic(sources),
            evidence_files={k:v for k,v in a.files.items() if k not in sources},
            native_calls=0,model_calls=0,patient_payload_reads=0)
        raw=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
        if len(raw)>65536:raise ValueError('64KiB saved receipt cap')
        with output.open('xb') as stream:stream.write(raw)
        print(json.dumps({'status':receipt['status'],'checks':a.checks,'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'elapsed_seconds':receipt['elapsed_seconds']}))
    return code


if __name__=='__main__':raise SystemExit(main())
