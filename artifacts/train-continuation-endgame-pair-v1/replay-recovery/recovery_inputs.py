"""Exact saved-plan admission. Only JSON metadata; never load a policy or cache."""
from pathlib import Path
from common import (ROOT,IL,PUBLIC,PUBLIC_SHA,COHORT,COHORT_SHA,PAIR_SUBJECT,
    TAIL_RUN,TAIL_PINS,TAIL_AUDIT,TAIL_AUDIT_SHA,read_json,canonical,semantic,pair_declaration)

FAILED='build/continuation-endgame-pair-v1/run-01'
OLD_INDEX='build/continuation-endgame-pair-v1/runner/source-index.json'
OLD_INDEX_SHA='c2a0b2f70fbc8692db425c85132082b26216aa62f0470c641077bb94a5458a16'
PINS={'report.json':'2a50f3a22b9c298681a863785b489c8e09b398974bce82d733464f015ff8337e',
 'launcher.json':'beedf54160707e8f2571a4e9a6d4add7f99ebf4cdbd5f320e7af4f89dece1808',
 'plans.json':'6798288cbb05cce8692f61f64438b68fcad8aef15ebc5ec008b2f3e3a151521b'}
FAILURE_AUDIT='build/continuation-endgame-pair-v1/independent/failure-audit-01.json'
FAILURE_AUDIT_SHA='258ab0dd042bdbde8bbbf42915d0bc6a6e4b970d8cb0fde86f8ca9396b4b1854'
WORLD_KEYS=('subject','source_hash','decision_model_hash','public_observation_binding','observation_track','max_steps')


def exact_json_equal(actual,expected,label):
    """Canonical JSON represents tuples/lists identically; no dropped fields/tolerance."""
    if canonical(actual)!=canonical(expected):raise ValueError('Exact canonical JSON mismatch: '+label)
    return True


def metadata_pins():
    old=read_json(ROOT/OLD_INDEX,OLD_INDEX_SHA)
    pins={FAILURE_AUDIT:FAILURE_AUDIT_SHA,OLD_INDEX:OLD_INDEX_SHA,**{FAILED+'/'+k:v for k,v in PINS.items()},
          TAIL_RUN+'/report.json':TAIL_PINS['report.json'],TAIL_AUDIT:TAIL_AUDIT_SHA,
          PUBLIC:PUBLIC_SHA,COHORT:COHORT_SHA}
    pins[IL+'/configuration.json']=old['input_metadata'][IL+'/configuration.json']
    public=read_json(ROOT/PUBLIC,PUBLIC_SHA);arrays={}
    for row in public['cases']:
        path=Path(row['path']);name=str(path.relative_to(ROOT));pins[name]=row['sha256']
        record=read_json(path,row['sha256'])
        if record['patient_id']!=row['patient_id'] or record['private_evaluation_files_included'] is not False:
            raise ValueError('Original public-only manifest changed')
        if row['patient_id']==PAIR_SUBJECT:
            arrays={str(Path(v['path']).relative_to(ROOT)):{k:v[k] for k in ('sha256','bytes','dtype')}
                    for v in record['input_files'].values()}
    if arrays!=old['public_arrays'] or len(arrays)!=5:raise ValueError('Exact original045 public array allowlist required')
    return pins,arrays


def require_saved_plans(failed,parent,plans,retained,old_index):
    cleanup=parent['owned_lifecycle'];failure=failed['failure']
    if (failed['status']!='failed_or_incomplete' or not failed['source_unchanged']
        or failure['phase']!='replay.RETAINED_TAIL' or failure['type']!='ValueError'
        or failure['message']!='Retained full native history/state projection changed'
        or failed['branches'] or parent['worker_exit_code']!=1 or parent['status']!='incomplete_or_failed'
        or not cleanup['verified'] or not cleanup['child_reaped'] or cleanup['exit_code']!=1
        or cleanup['errors'] or cleanup['remaining_owned_pids'] or cleanup['remaining_state_unknown']
        or parent['report_sha256']!=PINS['report.json'] or failed['source_index_sha256']!=OLD_INDEX_SHA
        or parent['source_index_sha256']!=OLD_INDEX_SHA or not old_index['execution_released']):
        raise ValueError('Exact cleanly contained preserved failure required')
    for key in ('baseline_head_reproduced','direct_calibration_reproduced','tail_fit_reproduced'):
        if failed[key] is not True:raise ValueError('Historical fitted-head reproduction incomplete')
    if failed['head_optimizer_updates']!=64 or failed['calibration_calls_completed']!=1 or failed['tail_calls_completed']!=1:
        raise ValueError('Original fixed fitted endpoints incomplete')
    if failed['tail_fit']['fitted_head_hash']!=retained['tail_fit']['fitted_head_hash']:
        raise ValueError('Original retained tail head differs')
    exact_json_equal(plans,failed['plans'],'separate sealed plans/report')
    if set(plans)!={'RETAINED_TAIL','IMMEDIATE_BEST'} or failed['plans_sealed_before_replay']['sha256']!=PINS['plans.json']:
        raise ValueError('Both original complete sealed plans required')
    arm=retained['cases'][PAIR_SUBJECT]['TAIL']
    prefix,common=pair_declaration(arm,failed['config'])
    exact_json_equal(failed['treatment']['prefix'],prefix,'eight prefix checks')
    exact_json_equal(failed['treatment']['common_state'],common,'common state')
    context=None
    for name,plan in plans.items():
        account=plan['planning'];decisions=account['decisions'];actions=plan['actions']
        if (not plan['complete'] or not account['complete'] or not 9<=len(actions)<=24
            or plan['plan_seal']!=semantic({k:v for k,v in plan.items() if k!='plan_seal'})
            or failed['plans_sealed_before_replay']['seals'][name]!=plan['plan_seal']
            or account['selected_actions']!=actions or account['explored_actions']!=actions
            or len(decisions)!=len(actions) or account['max_steps']!=24 or account['initial_steps']!=0
            or account['model_transition_calls']!=len(actions) or account['committed_transitions']!=len(actions)
            or account['call_cap_reached'] or account['time_cap_reached'] or account['cancelled'] or account['failure']
            or account['fitted_report_hash']!=semantic(failed['tail_fit'])
            or account['fitted_head_hash']!=failed['tail_fit']['fitted_head_hash']
            or account['base_parameter_hash']!=failed['tail_fit']['base_parameter_hash']
            or account['source_hash']!=arm['world']['source_hash']
            or account['decision_model_hash']!=arm['world']['decision_model_hash']):
            raise ValueError('Plan seal, source, head, horizon or completion differs')
        if context is not None and context!=account['context_hash']:raise ValueError('Branches differ in original context')
        context=account['context_hash'];total=0.
        for i,(action,row) in enumerate(zip(actions,decisions)):
            if (row['step']!=i or row['selected_action_id']!=action or row['transition_status']!='committed'
                or row['terminated'] is not (i==len(actions)-1) or not row['all_current_legal_actions_scored']):
                raise ValueError('Sealed decision/action/termination differs')
            scores=row['scores'];ids=[r['action_id'] for r in scores]
            if len(set(ids))!=len(ids) or not ids or ids[0]!='STOP' or action not in ids:
                raise ValueError('Full ordered action scores required')
            positive=[]
            for j,score in enumerate(scores):
                if score['actual_positive_guard'] is not bool(j and score['reward']>0):raise ValueError('Productive guard changed')
                raw=score['raw_predicted_tail'];clipped=score['clipped_predicted_tail'];combined=score['combined_nominal_q']
                if j==0:
                    if score['reward']!=0 or raw!=0 or clipped!=0 or combined!=0:raise ValueError('STOP0 changed')
                elif raw is not None:
                    if clipped!=max(0.,raw) or combined!=score['reward']+clipped:raise ValueError('Tail decomposition changed')
                if score['predicted_q']!=combined:raise ValueError('Combined ranking differs')
                if j and score['reward']>0:positive.append(j)
            chosen=0 if not positive else max(positive,key=lambda j:scores[j]['combined_nominal_q'])
            if i==8:
                exact_json_equal({k:row[k] for k in ('step','source_state_hash','observation_hash','scores')},
                    {k:common[k] for k in ('step','source_state_hash','observation_hash','scores')},'index8 common state')
                chosen=ids.index(failed['config']['forced_actions'][name])
                if not row.get('intervention') or scores[chosen]['reward']<=0:raise ValueError('Single treatment invalid')
            elif row.get('intervention'):raise ValueError('Extra intervention')
            if action!=ids[chosen] or row['actual_reward']!=scores[chosen]['reward']:
                raise ValueError('Unchanged frozen-tail selection or realized reward differs')
            if i<8:
                exact_json_equal({k:row[k] for k in prefix[i] if k!='current_tool_before'},
                    {k:v for k,v in prefix[i].items() if k!='current_tool_before'},'prefix decision')
            if i and row['source_state_hash']!=decisions[i-1]['result_state_hash']:raise ValueError('State chain changed')
            total+=row['actual_reward']
            if row['actual_total_nominal_return']!=total:raise ValueError('Cumulative reward changed')
        if total!=account['estimated_incremental_return'] or (actions[-1]!='STOP' and len(actions)!=24):
            raise ValueError('Complete absolute-horizon return differs')
    if plans['RETAINED_TAIL']['actions']!=arm['actions']:raise ValueError('Retained route actions differ')
    return {'accepted':True,'source':'exact_failed_pair_plans_before_any_replay',
        'plans_sha256':PINS['plans.json'],'plan_seals':{k:v['plan_seal'] for k,v in plans.items()},
        'original_context_hash':context,'head_hashes':{k:failed[k]['fitted_head_hash'] for k in ('head_fit','calibration','tail_fit')},
        'new_model_cache_checkpoint_loads':0,'new_planning_calls':0}
