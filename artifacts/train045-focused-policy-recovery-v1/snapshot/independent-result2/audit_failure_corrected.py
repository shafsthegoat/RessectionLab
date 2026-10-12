"""One saved-only audit of the third incomplete focused-policy attempt.
No scientific imports or patient/model payload reads; saved gzip JSON only.
"""
import hashlib,importlib.util,json,math,signal,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
PINS={'report':'079cfc74ae0499430b9a93438a94d7f3154a7559a972a40eee71f17fb8ab1af1',
'launcher':'abad11ff2dbf4f9655b9ea11d6949017abdccfd7350996286e776bf14928284d',
'index':'9112bf00953b168bf36ba409538dcbb10a0ac018272e4d65af921a8f09c342a6'}

def load(name,filename,expected):
    path=STAGE/'independent'/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Independent source changed')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=load('_saved_failure_base','audit_failure.py','c289ad8f9292d86cdcd2ad9c07d371226a87c7ef90eb0dfaf30110ada40d93c3')
oracle=load('_scalar_oracles','scalar_oracles.py','f7c4ce700b44816d9c61708dc7cf0d8b852c6055e8a5f45801578d869d063af6')
semantic=base.semantic

def eq(a,x,y,label):a.need(semantic(x)==semantic(y),label)
def seal(a,p):a.need(p['seal']==semantic({k:v for k,v in p.items() if k!='seal'}),'Packet semantic seal')

def trace(a,account,method,frontier,prefix,sidecars,complete,expected_commits,expected_states):
    history=account['explored_history'];packets=account['explored_action_records'];states=account['states']
    a.need(account['method']==method and account['complete'] is complete and account['initial_focus'] is None
        and account['controller']=='initial_none_then_first_current_goal_overlap_active_first_v1'
        and account['positive_immediate_guard'] is True and account['training_updates']==0,'Declared fixed method/information boundary')
    a.need(len(history)==len(packets)==12+expected_commits and len(states)==expected_states
        and account['model_transition_calls']==account['model_transition_completed']==expected_commits,'Actual complete/incomplete denominators')
    eq(a,packets[:12],prefix,'Actual fixed12 prefix packets');eq(a,semantic(history[:12]),frontier['nominal_history_hash'],'Actual fixed12 history')
    a.close(sum(r['reward'] for r in history[:12]),frontier['total_reward'],'Fixed prefix reward')
    a.close(account['fixed_prefix_return'],frontier['total_reward'],'Initial incumbent scalar')
    a.need(account['context_seal']==semantic(account['context']) and account['context']['prefix_length']==12
        and account['context']['private_reference_input'] is False and account['context']['feature_change'] is False,
        'Separate focused admission with unchanged public features')
    previous=None
    for i,(packet,row) in enumerate(zip(packets,history)):
        seal(a,packet)
        a.need(packet['steps_taken']==i and packet['current_tool']==previous and packet['action_id']==row['action_id']
            and packet['source_hash']==frontier['source_hash'] and packet['decision_model_hash']==frontier['decision_model_hash'],
            'Actual packet step/tool/source sequence')
        a.need(packet['action_id']!='STOP' and packet['physical_action']=={k:row[k] for k in ('tool_id','entry_mm','tip_mm')}
            and row['outcome_scope']=='permitted_nominal_model','Recorded physical action and nominal-only planning feedback')
        previous=row['tool_id']
        if i>=12:a.need(row['reward']>0,'Each committed suffix motion is actually positive')
    if complete:
        a.need(len(packets)==24 and account['selected_equals_explored'] is True,'GREEDY horizon completion is planning only')
        a.close(account['selected_total_return'],sum(r['reward'] for r in history),'GREEDY planned total')
        a.close(account['selected_net_suffix_return'],sum(r['reward'] for r in history[12:]),'GREEDY planned suffix')
    else:a.need(len(packets)<24 and 'selected_total_return' not in account,'TAIL incomplete; no fabricated terminal return')
    choices=[];query_count=0;goals_count=0;pages_count=0;scored_rows=0;committed=0
    for j,state in enumerate(states):
        i=12+j;rows=state['candidates'];ids=[r['action_id'] for r in rows];pages=state['pages'];selected=state['selected']
        a.need(state['steps']==i and len(ids)==len(set(ids))<=153 and ids[0]=='STOP'
            and state['complete_page_enumeration'] is True,'Full ordered unique bounded actor catalog')
        base_count=pages[0]['base_emitted_count'];extra=pages[0]['additional_emitted_count']
        a.need(base_count<=120 and extra<=32 and len(pages)==1+(extra+7)//8
            and [p['page_index'] for p in pages]==list(range(len(pages))),'Unchanged base/page limits')
        ordered=['STOP'];owners=[0]
        for k,page in enumerate(pages):
            a.need(page['source_state_hash']==state['source_state_hash'] and page['steps_taken']==i
                and page['source_hash']==frontier['source_hash'],'Every page same current world/state')
            ordered.extend(page['page_action_ids']);owners.extend([k]*len(page['page_action_ids']))
        a.need(ids==ordered and [r['owner_page'] for r in rows]==owners,'Duplicate base removed once; exact stable page ownership')
        rewards=[r['immediate_reward'] for r in rows];raw=None if method=='GREEDY' else [r['raw_tail'] for r in rows]
        winner=oracle.choose_positive(ids,rewards,raw);index=ids.index(winner)
        a.need(winner!='STOP' and selected['action_id']==winner,'Independent full-catalog positive-guard scalar argmax')
        for k,row in enumerate(rows):
            a.need(row['actual_positive'] is (rewards[k]>0),'Actual positivity, not predicted positivity')
            a.close(row['ranking_value'],rewards[k] if raw is None else rewards[k]+max(0.,raw[k]),'Actual r plus raw tail clipped exactly once')
            if raw is None:a.need(row['raw_tail'] is None and row['clipped_tail'] is None,'GREEDY no model predictions')
            else:a.close(row['clipped_tail'],max(0.,raw[k]),'Stored tail clipping')
            if k:
                verified=[r for r in pages[owners[k]]['ledger'] if r.get('action_id')==ids[k]
                    and r.get('verification')=='NATIVE_VERIFIED']
                a.need(bool(verified) and all(r['feasible'] is True for r in verified),'Every scored ray owns a legal current native certificate')
        packet=selected['packet'];seal(a,packet)
        a.need(packet['source_state_hash']==state['source_state_hash'] and packet['query_catalog_digest']==state['query_catalog_digest']
            and packet['focus']==state['goal_focus'],'Current selected packet scope/focus/catalog')
        matches=[r for r in pages[owners[index]]['ledger'] if r.get('action_id')==winner and r.get('verification')=='NATIVE_VERIFIED']
        a.need(all({k:r[k] for k in ('tool_id','entry_mm','tip_mm')}==packet['physical_action'] for r in matches),'Chosen physical geometry belongs to owning page')
        done='result_state_hash' in selected
        if done:
            committed+=1;eq(a,packet,packets[i],'Exact committed selected packet')
            a.close(selected['immediate_reward'],history[i]['reward'],'Certified reward paid once')
            a.close(selected['cumulative_return'],sum(r['reward'] for r in history[:i+1]),'Cumulative full reward')
            if j+1<len(states):a.need(selected['result_state_hash']==states[j+1]['source_state_hash'],'Next actual native state link')
        else:a.need(j==len(states)-1 and i==len(packets),'Only final scored state uncommitted; no added action/reward')
        for goal in state['goals']:
            side=goal['sidecar']
            if side is not None:a.need(sidecars['records'][side['fingerprint']]==side and side['source_state_hash']==state['source_state_hash'],
                'Each goal failure sidecar retained authentically')
        query_count+=base_count+extra+len(state['goals']);goals_count+=len(state['goals']);pages_count+=len(pages)
        if raw is not None:scored_rows+=len(rows)
        immediate_order=sorted(range(1,len(rows)),key=lambda k:(-rewards[k],k))
        choices.append({'absolute_action':i+1,'committed':done,'action_id':winner,'reward':rewards[index],
            'raw_tail':None if raw is None else raw[index],'combined':rows[index]['ranking_value'],
            'immediate_rank':immediate_order.index(index)+1,'immediate_regret':max(rewards)-rewards[index],
            'physical_action':packet['physical_action']})
    a.need(committed==expected_commits and account['costs']['union_observations']==expected_states
        and account['costs']['pages_completed']==pages_count
        and account['costs']['goal_previews_started']==account['costs']['goal_previews_completed']==goals_count,
        'Completed/uncommitted work and all observation/discovery counts')
    score=account['scorer_costs']
    if method=='GREEDY':a.need(score is None,'No GREEDY forward')
    else:
        a.need(score['score_calls_started']==score['score_calls_completed']==score['policy_forwards_started']==score['policy_forwards_completed']
            ==score['head_forwards_started']==score['head_forwards_completed']==expected_states
            and score['candidate_rows_scored']==scored_rows,'All completed TAIL catalog scoring charged including uncommitted final state')
    return {'planning_complete':complete,'independent_endpoint_replay':False,'committed_suffix':expected_commits,
        'scored_states':expected_states,'total_steps':len(history),'provisional_total_return':sum(r['reward'] for r in history),
        'provisional_suffix_return':sum(r['reward'] for r in history[12:]),'queries_from_saved_states':query_count,
        'choices':choices,'score_costs':score}

def inspect(a):
    directory=STAGE/'recovery-02';index=a.read(STAGE/'runner-recovery2/source-index.json',PINS['index'])
    report=a.read(directory/'report.json',PINS['report']);parent=a.read(directory/'launcher.json',PINS['launcher'])
    for name,digest in index['files'].items():a.raw(ROOT/name,digest)
    a.need(report['source_index_sha256']==parent['source_index_sha256']==PINS['index'] and index['execution_released'] is True
        and parent['report_sha256']==PINS['report'] and report['source_unchanged'] is True,'Exact root-released terminal source/report pins')
    life=parent['owned_lifecycle']
    a.need(report['status']=='failed_or_incomplete' and parent['status']=='incomplete_or_failed'
        and parent['worker_exit_code']==life['exit_code']==1 and life['verified'] is True and life['child_reaped'] is True
        and life['remaining_owned_pids']==[] and life['errors']==[] and life['remaining_state_unknown'] is False
        and parent['stop_reason'] is None,'Honest failed worker with clean verified cleanup')
    a.need(report['failure']=={'committed':False,'message':'Focused rollout time cap','phase':'TAIL_suffix_planning','type':'TimeoutError'},'Actual TAIL cooperative deadline')
    a.need(report['branches']=={} and report['plans_sealed_before_replay'] is False and not (directory/'plans.json').exists()
        and report['frozen_models_unchanged'] is False,'No sealed pair, independent replay or invented overall success')
    a.need(set(report['selections'])=={'GREEDY'} and report['selections']['GREEDY']['complete'] is True
        and report['selections']['GREEDY']['validation']=='accepted' and set(report['incomplete_selection'])=={'TAIL'},'Complete planning account retained; second arm incomplete')
    a.need(report['closed_roles_opened'] is False and report['cache_loads']==report['head_optimizer_updates']==0
        and report['checkpoint_loads']==report['checkpoint_loads_attempted']==2,'Same single TRAIN inference, no fitting or feature-cache reads')
    eq(a,report['model_provenance']['checkpoints'],index['checkpoints'],'Payload pins metadata only; no payload opened')
    original=a.read(STAGE/'run-01/report.json','c31b88e88c77b8f50d262193ce2424d4abf203eb23fbab14eed63de032229038')
    transport=a.read(STAGE/'recovery-01/report.json','4179a5f960f442ee71e069c060a7e86c471b6b7fd8ca3732ae792ad5c90a2819')
    old_audit=a.read(STAGE/'independent/failure-audit-01.json','f02af1cd2731162b3bc7c27d1d683b87e6b23904b7e4fdf3c5188dbc648bf4b1')
    a.need(old_audit['status']=='PASS_failure_evidence' and original['status']==transport['status']=='failed_or_incomplete','Both earlier failures retained unchanged')
    old_index=a.read(STAGE/'runner/source-index.json',original['source_index_sha256'])
    eq(a,index['public_arrays'],old_index['public_arrays'],'Exact same five public array pins, no array read')
    prefix_file=a.read(directory/'fixed-prefix.json');frontier=report['fixed_frontier'];prefix=prefix_file['focused_packets']
    eq(a,frontier,prefix_file['focused_frontier'],'Actual common frontier file');eq(a,frontier,original['fixed_frontier'],'Old/new physical frontier same')
    a.need(frontier['seal']==semantic({k:v for k,v in frontier.items() if k!='seal'}) and len(prefix)==12,'Same authenticated12-action prefix')
    sidecars=a.compressed(directory,report['sidecars'])
    a.need(len(sidecars['records'])==report['sidecars']['unique_count']==1821,'All retained unique sidecars')
    for k,v in sidecars['records'].items():a.need(k==v['fingerprint']==semantic({x:y for x,y in v.items() if x!='fingerprint'}),'Saved sidecar fingerprint')
    greedy=a.compressed(directory,report['selections']['GREEDY']);tailwrapper=a.compressed(directory,report['incomplete_selection']['TAIL']);tail=tailwrapper['policy']
    a.need(tailwrapper['scoring'] is None,'Exception after completed score; no separate failed forward account')
    eq(a,greedy['context'],tail['context'],'Both actors receive same explicitly admitted task/world/controller')
    findings={'GREEDY':trace(a,greedy,'GREEDY',frontier,prefix,sidecars,True,12,12),
        'TAIL':trace(a,tail,'TAIL',frontier,prefix,sidecars,False,10,11)}
    oldtrace=a.compressed(STAGE/'run-01',original['incomplete_selection']['GREEDY'])['policy']
    admissions=[]
    for folder,trace_row,source_pin in (('run-01',oldtrace,original['source_index_sha256']),
            ('recovery-02',greedy,PINS['index'])):
        own=STAGE/folder
        context=a.read(own/'ReMIND-045-context.json')
        bindings=a.read(own/'evaluation/source/admitted-public-bindings.json')
        release_raw=a.raw(own/'world-admission.json');release=json.loads(release_raw)
        a.need(release['root_source_index_sha256']==source_pin and release['execution_released'] is True,
            'Each release independently binds its actual source index')
        a.need(bindings['protocol']['runtime_release_sha256']==hashlib.sha256(release_raw).hexdigest()
            and context['protocol_sha256']==semantic(bindings['protocol'])
            and trace_row['context']['original_context_hash']==semantic(context)
            and trace_row['context_seal']==semantic(trace_row['context']),
            'Each original context hash and derived seal bind its OWN saved admission/release')
        admissions.append((context,bindings,release))
    # Narrow identity normalization: distinct releases are genuine; all science is exact.
    eq(a,{k:v for k,v in admissions[0][2].items() if k!='root_source_index_sha256'},
         {k:v for k,v in admissions[1][2].items() if k!='root_source_index_sha256'},'Release records differ only execution source index')
    eq(a,{k:v for k,v in admissions[0][1].items() if k!='protocol'},
         {k:v for k,v in admissions[1][1].items() if k!='protocol'},'All public binding/QC/source metadata exact')
    eq(a,{k:v for k,v in admissions[0][1]['protocol'].items() if k!='runtime_release_sha256'},
         {k:v for k,v in admissions[1][1]['protocol'].items() if k!='runtime_release_sha256'},'Admission protocols differ only authenticated runtime release')
    eq(a,{k:v for k,v in admissions[0][0].items() if k!='protocol_sha256'},
         {k:v for k,v in admissions[1][0].items() if k!='protocol_sha256'},'Original contexts differ only derived protocol identity')
    eq(a,{k:v for k,v in oldtrace['context'].items() if k!='original_context_hash'},
         {k:v for k,v in greedy['context'].items() if k!='original_context_hash'},'Focused contexts differ only release-derived original identity')
    for key in ('controller','initial_focus','fixed_prefix_return'):eq(a,oldtrace[key],greedy[key],'Retained original '+key)
    eq(a,oldtrace['explored_history'],greedy['explored_history'][:23],'Exact old23 committed physical histories')
    eq(a,oldtrace['explored_action_records'],greedy['explored_action_records'][:23],'Exact old23 action packets')
    eq(a,oldtrace['states'],greedy['states'][:11],'Exact old11 decisions including full ranks/ledgers/focus')
    a.need(report['retained_trace_loads_attempted']==report['retained_trace_loads_completed']==0,
        'Historical future trace never loaded by policy; only this independent post-terminal audit compared it')
    costs=report['costs'];budget=report['native_budget'];caps=report['config']['caps']
    expected=sum(v.get('native_preview_entries',0) for k,v in costs.items() if k!='same045_source')
    a.need(expected==budget['native_preview_entries']==7120 and budget['counting_reliable'] is True
        and budget['status']=='failed' and budget['history_complete_caller_attestation'] is None,'Reliable final preview total; negative budget status preserved')
    profile=budget['native_preview_profile']['phases']['unclassified']
    a.need(profile['started']==profile['returned']==7120 and profile['raised']==0,'Complete native profile joins counters')
    a.need(report['native_transition_counts']=={'attempted':46,'completed':46,'raised':0}
        and report['native_transition_attempts']==24+12+10,'All actual prefix/planning transitions counted once')
    a.need(report['total_frozen_policy_forwards']==costs['TAIL_suffix_planning']['policy_forward_calls']==11,'Eleven forwards including scored but uncommitted action')
    for method,row in findings.items():a.need(row['queries_from_saved_states']==costs[method+'_suffix_planning']['native_preview_entries'],'All discovery previews visible in saved states')
    cumulative=report['failed_plus_recovery_costs']
    a.need(cumulative['native_previews']==4733+4850+7120 and cumulative['native_transition_attempts']==35+36+46
        and cumulative['frozen_policy_forwards']==11 and cumulative['checkpoint_loads']==6 and cumulative['prior_failed_attempts']==2,'All three attempted experiments remain paid')
    for key,old in (('preserved_failed_attempt',original),('preserved_transport_failure',transport)):
        preserved=report[key]
        a.need(preserved['native_previews']==old['native_budget']['native_preview_entries'] and preserved['policy_forwards']==old['total_frozen_policy_forwards']
            and preserved['native_transition_counts']==old['native_transition_counts'],'Exact historical work joins')
        a.close(preserved['worker_seconds'],old['worker_seconds'],'Historical worker cost')
    prior_seconds=original['worker_seconds']+transport['worker_seconds']
    a.need(prior_seconds<=cumulative['worker_seconds']<=prior_seconds+report['worker_seconds'],'Cumulative worker timestamp scope')
    a.need(report['worker_seconds']<caps['work_seconds'] and parent['elapsed_seconds']<caps['parent_wall_seconds']
        and max(report['self_peak_rss_bytes'],parent['sampled_peak_group_rss_bytes'])<caps['worker_peak_rss_bytes']
        and expected<caps['native_previews'],'Outer work/memory/query budgets respected; inner TAIL deadline failed')
    a.need(costs['TAIL_suffix_planning']['complete_wall_seconds']>=report['config']['planner_seconds'],'Measured inner TAIL time cap')
    network=costs['TAIL_suffix_planning']['policy_forward_inclusive_seconds'];scoring=tail['scorer_costs']['elapsed_seconds']
    a.need(0<network<=scoring<=costs['TAIL_suffix_planning']['complete_wall_seconds'],'Nested scored-catalog/network timing ordering')
    return {'scientific_comparison':'INCOMPLETE; no sealed pair or fresh independent endpoints; no useful-gain verdict',
        'failure':report['failure'],'planning_only':findings,'retained_old11_decisions_exact':True,
        'parity_guard_defect':{'old_original_context_hash':oldtrace['context']['original_context_hash'],
            'new_original_context_hash':greedy['context']['original_context_hash'],
            'old_context_seal':oldtrace['context_seal'],'new_context_seal':greedy['context_seal'],
            'scientific_fields_exact':True,'all23_packets_and_histories_exact':True,
            'cause':'Fresh root source-index changes world-admission file hash, admission protocol hash, original context hash, and derived focused context seal. Each chain validated separately; all other values exactly equal.',
            'scope':'Current worker failed earlier at TAIL deadline. Its post-plan exact-context parity guard would also reject this legitimate identity change; that latent wrapper defect is not a scientific action mismatch.'},
        'performance':{'GREEDY_complete_plan_seconds':costs['GREEDY_suffix_planning']['complete_wall_seconds'],
            'TAIL_incomplete_plan_seconds':costs['TAIL_suffix_planning']['complete_wall_seconds'],
            'TAIL_scored_catalog_seconds':scoring,'TAIL_network_forward_inclusive_seconds':network,
            'scored_catalog_minus_network_seconds':scoring-network,
            'network_share_of_scored_catalog':network/scoring,
            'interpretation':'The scored-catalog interval includes integrity/certificate/feature/head work. Network time alone does not explain the deadline. Subtracting nested timers is descriptive; other guard costs are not separately isolated.'},
        'costs':{'current_previews':7120,'current_transitions':46,'current_forwards':11,'worker_seconds':report['worker_seconds'],
            'parent_seconds':parent['elapsed_seconds'],'peak_rss_bytes':parent['sampled_peak_group_rss_bytes'],
            'phase_costs':costs,'all_three_attempts':cumulative,
            'all_three_parent_seconds':cumulative['prior_parent_seconds']+parent['elapsed_seconds']},
        'cleanup':life,'limits':'No arrays, neural predictions, or geometry replay independently recomputed; planning histories are not accepted physical/patient outcomes. All current evidence is annotation-assisted TRAIN045 simulation.'}

def main():
    output=HERE/'failure-audit-03-corrected.json'
    if output.exists():raise ValueError('Preserve existing receipt')
    started=time.monotonic();a=base.Audit();receipt={'schema':'focused-policy-third-failure-independent-v2','pins':PINS,'retained_failed_reader_receipt_sha256':'9b94e3e40b868118abebe7ba2370ac04481a2ab524d458db048eeffb49f86c72'}
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved audit')));signal.alarm(30);code=0
    try:receipt.update(status='PASS_failure_evidence',findings=inspect(a))
    except BaseException as error:code=1;receipt.update(status='FAIL_reader_or_evidence',failure={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.alarm(0);source={k:v for k,v in a.files.items() if k.endswith('.py')}
        receipt.update(checks=a.checks,elapsed_seconds=time.monotonic()-started,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            checked_source_count=len(source),checked_sources_digest=semantic(source),evidence_files={k:v for k,v in a.files.items() if k not in source},
            native_calls=0,model_calls=0,patient_payload_reads=0)
        raw=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
        if len(raw)>65536:raise ValueError('64KiB receipt cap')
        with output.open('xb') as stream:stream.write(raw)
        print(json.dumps({'status':receipt['status'],'checks':a.checks,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'elapsed_seconds':receipt['elapsed_seconds']}))
    return code
if __name__=='__main__':raise SystemExit(main())
