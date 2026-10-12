"""Single saved-only failed-pair recovery-point audit; no scientific imports."""
import hashlib,importlib.util,json,math
from pathlib import Path
import signal,time
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
original=HERE/'audit_saved.py'
if hashlib.sha256(original.read_bytes()).hexdigest()!='c57ba3391c4e73fad15c851bbff0d51477fc5a5b2bd0282bf0eb5307b880b99b':
    raise ValueError('Unrun original terminal auditor changed')
spec=importlib.util.spec_from_file_location('prepared_pair_auditor',original)
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
base=pair.base
need,read_json,semantic,close=base.need,base.read_json,base.semantic,base.close
PINS={'report':'2a50f3a22b9c298681a863785b489c8e09b398974bce82d733464f015ff8337e',
      'launcher':'beedf54160707e8f2571a4e9a6d4add7f99ebf4cdbd5f320e7af4f89dece1808',
      'index':'c2a0b2f70fbc8692db425c85132082b26216aa62f0470c641077bb94a5458a16'}

def run():
    stage=HERE.parent;out=stage/'run-01'
    r=read_json(out/'report.json',PINS['report']);l=read_json(out/'launcher.json',PINS['launcher'])
    index=read_json(stage/'runner/source-index.json',PINS['index'])
    need(index['execution_released'] and r['source_index_sha256']==l['source_index_sha256']==PINS['index']
         and l['report_sha256']==PINS['report'],'Exact failed terminal/source pins')
    need(r['status']=='failed_or_incomplete' and r['source_unchanged'] and l['status']=='incomplete_or_failed'
         and l['worker_exit_code']==1,'Failed attempt retained as failed')
    error=r['failure']
    need(error['phase']=='replay.RETAINED_TAIL' and error['type']=='ValueError'
         and error['message']=='Retained full native history/state projection changed','Exact late replay parity failure')
    cleanup=l['owned_lifecycle']
    need(cleanup['verified'] and cleanup['child_reaped'] and cleanup['exit_code']==1
         and not cleanup['remaining_owned_pids'] and not cleanup['errors'],'Failed child cleanly contained/reaped')
    need(r['branches']=={} and not r['baseline_reproduced'] and 'diagnosis' not in r,
         'No accepted pair endpoints or scientific treatment effect published')
    need(not r['closed_roles_opened'] and r['new_training_rollouts']==0
         and not r['original_collection_actor_reexecuted'] and r['base_optimizer_updates']==0,
         'No role/base/world learning expansion')
    retained=read_json(ROOT/(pair.TAIL+'/report.json'),pair.TAIL_SHA)
    oldparent=read_json(ROOT/(pair.TAIL+'/launcher.json'),pair.TAIL_LAUNCH)
    pair.source_and_model(r,index,retained,oldparent)
    old,prefix,common=pair.declared_treatment(r,retained)
    sealed=r['plans_sealed_before_replay'];plans=read_json(out/sealed['file'],sealed['sha256'])
    need(sealed['file']=='plans.json' and plans==r['plans'] and set(plans)==set(pair.ARMS),
         'Both separately saved complete plans available without regeneration')
    summaries={};total_forwards=0
    for arm in pair.ARMS:
        plan=plans[arm];p=plan['planning'];actions=plan['actions'];rows=p['decisions']
        need(plan['complete'] and plan['plan_seal']==sealed['seals'][arm]
             ==semantic({k:v for k,v in plan.items() if k!='plan_seal'}),'Exact sealed full planning record')
        need(p['complete'] and p['failure'] is None and p['initial_steps']==0 and p['max_steps']==p['max_calls']==24,
             'Completed original absolute24 planning horizon')
        need(9<=len(actions)<=24 and len(actions)==len(rows)
             and actions==p['selected_actions']==p['explored_actions']==[d['selected_action_id'] for d in rows],
             'Complete aligned full action sequence')
        need((actions[-1]=='STOP' and 'STOP' not in actions[:-1]) or ('STOP' not in actions and len(actions)==24),
             'Actual terminal STOP/horizon in sealed plan')
        need(actions[8]==pair.FORCED[arm] and p['forced_action_id']==pair.FORCED[arm]
             and p['intervention_index']==8 and p['diagnostic_only'],'Only declared treatment')
        need(p['fitted_head_hash']==r['tail_fit']['fitted_head_hash'] and p['fitted_report_hash']==semantic(r['tail_fit'])
             and p['source_hash']==old['world']['source_hash'] and p['decision_model_hash']==old['world']['decision_model_hash'],
             'Bound frozen model/world')
        total=0.;forwards=0
        for i,d in enumerate(rows):
            need(d['step']==i and d['transition_status']=='committed'
                 and d['terminated'] is (i==len(rows)-1),'Complete absolute-step native planning')
            if i:need(rows[i-1]['result_state_hash']==d['source_state_hash'],'Continuous sealed state chain')
            scores=d['scores'];ids=[x['action_id'] for x in scores];rewards=[x['reward'] for x in scores]
            productive=any(v>0 for v in rewards[1:]);tails=[x['raw_predicted_tail'] for x in scores] if productive else None
            need(d['all_current_legal_actions_scored'] and d['legal_nonstop_actions']==len(scores)-1,'Complete current inventory')
            for j,s in enumerate(scores):
                if j==0:need(s['reward']==s['predicted_q']==s['raw_predicted_tail']==s['clipped_predicted_tail']==s['combined_nominal_q']==0,'STOP0')
                elif productive:
                    need(math.isfinite(s['raw_predicted_tail']) and s['clipped_predicted_tail']==max(0.,s['raw_predicted_tail'])
                         and s['combined_nominal_q']==s['predicted_q']==s['reward']+s['clipped_predicted_tail'],
                         'Unchanged exact current reward plus clipped tail')
                else:need(all(s[k] is None for k in ('raw_predicted_tail','clipped_predicted_tail','predicted_q','combined_nominal_q')),'No STOP inference')
            if i==8:
                need({k:d[k] for k in ('step','source_state_hash','observation_hash','scores')}
                     =={k:common[k] for k in common if k!='current_tool'},'Exact declared common current state/scores')
                need(d['unforced_selected_action_id']==pair.FORCED['RETAINED_TAIL'] and d['intervention']
                     and d['selection_reason']=='predeclared_single_index8_intervention','Only index8 ranking exception')
                need(d['actual_reward']>0,'Forced candidate obeys actual-positive guard')
            else:pair.old.oracle.audit_choice(ids,[True]*len(ids),rewards,tails,d['selected_action_id'])
            if i<8:
                need({k:d[k] for k in prefix[i] if k!='current_tool_before'}
                     =={k:prefix[i][k] for k in prefix[i] if k!='current_tool_before'},'Unchanged eight-prefix record')
            selected=scores[ids.index(d['selected_action_id'])]
            need(d['actual_reward']==selected['reward'] and d['selected_q']==selected['combined_nominal_q'],
                 'Chosen reward counted once in saved plan')
            total+=d['actual_reward'];forwards+=int(productive)
        need(total==p['estimated_incremental_return'],'Full planned reward sum')
        need(p['model_transition_calls']==p['committed_transitions']==len(actions)
             and p['native_previews_started']==p['native_previews_completed'], 'All saved work counted')
        need(p['q_batches_started']==p['q_batches_completed']==p['scorer_costs']['policy_forwards_completed']
             ==p['scorer_costs']['head_forwards']==forwards,'All forwards paid')
        pair.terminal_catalog(p['terminal_decision_catalog'],p)
        total_forwards+=forwards
        summaries[arm]={'plan_seal':plan['plan_seal'],'actions':len(actions),'action_ids':actions,
            'history_hash':p['nominal_history_hash'],'planned_return':total,'planning_costs':plan['planning_costs'],
            'forwards':forwards,'authoritative_replay_accepted':False}
    need(plans['RETAINED_TAIL']['actions']==old['actions']
         and plans['RETAINED_TAIL']['planning']['nominal_history_hash']==old['nominal_history_hash'],
         'Retained planned full native history/actions match historical endpoint')
    need(plans['RETAINED_TAIL']['planning']['decisions'][:8]==plans['IMMEDIATE_BEST']['planning']['decisions'][:8],
         'Both complete plans have identical shared ordinary prefix')
    need(r['native_budget']['native_preview_entries']==3458 and r['total_frozen_policy_forwards']==total_forwards==24,
         'Exact failed attempt3458 previews/24 forwards preserved')
    cumulative=r['cumulative_iteration_costs'];paid=[r[k] for k in ('prior_cache_attempt','prior_direct_attempt','prior_tail_attempt')]
    need(cumulative['native_preview_entries']==sum(v['native_previews'] for v in paid)+3458
         and cumulative['frozen_policy_forwards']==sum(v['frozen_policy_forwards'] for v in paid)+24,
         'Prior failed/direct/tail and failed pair all remain paid')
    # Read only immutable source bytes authenticated by the actual source index.
    sourcepins={}
    for name in ('build/continuation-endgame-pair-v1/runner/worker.py',
                 'build/continuation-endgame-pair-v1/runner/common.py','src/resectionlab/native_resection.py'):
        raw=(ROOT/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        need(digest==index['files'][name],'Exact run source for failure diagnosis')
        sourcepins[name]=digest
    historical_tuple_fields=sum('tip_mm' in h and 'entry_mm' in h for h in old['history'])
    need(historical_tuple_fields==12 and all(isinstance(h['tip_mm'],list) and isinstance(h['entry_mm'],list)
         for h in old['history'] if 'tip_mm' in h),'Historical JSON coordinate fields are lists')
    return {'recovery_point':'PASS_SEALED_PLANS_FOR_REPLAY_ONLY_NOT_ACCEPTED_EXPERIMENT',
        'original_attempt_status':r['status'],'original_failure':error,'report_sha256':PINS['report'],
        'launcher_sha256':PINS['launcher'],'source_index_sha256':PINS['index'],
        'plans_file':str((out/sealed['file']).relative_to(ROOT)),'plans_sha256':sealed['sha256'],
        'plans':summaries,'observed_clean_failure_lifecycle':cleanup,
        'attempt_costs':{'native_previews':3458,'frozen_forwards':24,'head_updates':r['head_optimizer_updates'],
            'direct_svd':r['calibration_calls_completed'],'tail_svd':r['tail_calls_completed'],
            'worker_seconds':r['worker_seconds'],'parent_seconds':l['elapsed_seconds'],'phase_costs':r['costs']},
        'cumulative_iteration_costs':cumulative,'source_diagnosis_pins':sourcepins,
        'diagnosis':'Native typed tuple entry/tip survive compact_history; JSON original lists make strict Python history comparison fail. Error occurs after readout, independent audit and action/world/nominal-history/outcome parity checks by authenticated source control flow.',
        'remaining_uncertainty':'The first replay live result was not saved before assertion, so additional decision/projection mismatch cannot be independently excluded from saved output. New replay must save certificate/projection and require full JSON-normalized equality; exclude no values and use no fitted tolerance.',
        'recovery_scope':'Use only these exact two complete sealed action plans for new authoritative replay; no model/head/cache decode, fitting, replanning or treatment change. Original failure remains negative.',
        'input_files':base.READS}

if __name__=='__main__':
    destination=HERE/'failure-audit-01.json'
    if destination.exists():raise FileExistsError('Preserve original audit')
    start=time.monotonic()
    def timeout(*_):raise TimeoutError('30s saved-only cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30.)
    try:record=run();record['audit_status']='PASS_SAVED_RECOVERY_POINT';code=0
    except BaseException as e:record={'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(e).__name__,'message':str(e)},'input_files':base.READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0.)
    record.update(seconds=time.monotonic()-start,checks=base.CHECKS,model_calls=0,native_calls=0,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(record,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    if len(raw)>65536:raise ValueError('64KiB receipt cap')
    with destination.open('xb') as f:f.write(raw)
    print(json.dumps({k:record[k] for k in ('audit_status','seconds','checks','model_calls','native_calls')}))
    raise SystemExit(code)
