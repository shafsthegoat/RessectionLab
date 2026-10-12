"""Pure scalar protocol, reset schedule and all-generation experience accounting."""
import hashlib,json,math
METHODS=('uniform','reverse')
WINDOWS=((0,128,16),(128,256,15),(256,384,14),(384,512,13),(512,768,12))
EXPECTED_PPO={'n_steps':64,'batch_size':4,'n_epochs':4,'learning_rate':.0003,
    'gamma':1.,'gae_lambda':1.,'clip_range':.2,'ent_coef':.01,'vf_coef':.5,
    'max_grad_norm':.5,'target_kl':.03,'device':'cpu','verbose':0}


def semantic(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def reverse_start(local_decisions):
    if type(local_decisions) is not int or not 0<=local_decisions<=768:
        raise ValueError('Fixed local decision clock0..768 required')
    for lower,upper,start in WINDOWS:
        if lower<=local_decisions<upper:return start
    return 12  # SB3 may automatically reset after the final completed decision.


def require_campaign(report,parent,*,report_sha,index_sha,configuration):
    c=parent['owned_lifecycle']
    if (report['status']!='completed' or not report['source_unchanged'] or not report['qualified']
        or report['patients']!=1 or report['subject']!='ReMIND-045' or report['role']!='TRAIN'
        or parent['status']!='completed' or parent['report_sha256']!=report_sha
        or report['source_index_sha256']!=index_sha or parent['source_index_sha256']!=index_sha
        or not c['verified'] or not c['child_reaped'] or c['exit_code']!=0 or c['errors']
        or c['remaining_owned_pids'] or c['remaining_state_unknown']
        or set(report['arms'])!={'frozen_encoder-seed0','trainable_encoder-seed0'}
        or report['config']['milestones']!=[256] or report['config']['seeds']!=[0]
        or configuration['milestones']!=[512,1024] or configuration['seeds']!=[0]
        or configuration['arms']!=list(METHODS)
        or configuration['reverse_windows']!=[list(x) for x in WINDOWS]):
        raise ValueError('Completed paired256 is required for this conditional comparison')
    for key in ('goal','prefix_actions','absolute_horizon','query','curriculum_starts',
                'evaluation_start','reward_policy','success','torch_threads'):
        if configuration[key]!=report['config'][key]:raise ValueError('Task/PPO input changed: '+key)
    for name,arm in report['arms'].items():
        if (arm['status']!='completed' or arm['seed']!=0 or arm['ppo_config']!=EXPECTED_PPO
            or arm['milestones']!=[256] or len(arm['checkpoints'])!=1
            or arm['experience']['decisions']!=256 or arm['segment_experience']['decisions']!=192
            or not arm['optimizer_resume_join']['all_tensors_and_optimizer_scalars_exact']):
            raise ValueError('Both genuine optimizer-preserving256 arms must be complete')
        row=arm['checkpoints'][0];endpoint=row['evaluation']
        if (row['decisions']!=256 or row['checkpoint']['decisions']!=256
            or not row['checkpoint']['includes_optimizer_state']
            or row['optimizer_attempted']!=row['optimizer_completed']
            or not 0<row['additional_optimizer_completed']<=192
            or row['policy_costs']['optimizer_completed']!=row['optimizer_completed']
            or not endpoint['complete'] or not endpoint['audit']['accepted']
            or not endpoint['local_reward_audit_accepted']
            or semantic(endpoint)!=semantic(report['evaluations'][name+'-eval-256'])):
            raise ValueError('Authenticated256 checkpoint and full evaluation required')
        if endpoint['success'] is not False:
            raise ValueError('Conditional release closed: a256 full-start endpoint succeeded')
        if name=='frozen_encoder-seed0' and (row['encoder_changed'] or
            row['encoder_hash']!=arm['initialization']['encoder_hash']):
            raise ValueError('Shared starting encoder must be the unchanged frozen arm')
    return {'report_sha256':report_sha,'source_index_sha256':index_sha,
        'shared_checkpoint_sha256':report['arms']['frozen_encoder-seed0']['checkpoints'][0]['checkpoint']['sha256'],
        'condition':'Both256 full-start endpoints independently accepted and failed',
        'branch_scope':'Two continuations from one frozen256 endpoint, not two new patients'}


def _costsum(*rows):
    result={}
    for row in rows:
        for key,value in row.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Finite nonnegative paid cost required')
            result[key]=result.get(key,0)+value
    return result


def merge_experience(previous,current):
    """One immutable parent lineage plus new segment; never relabel old unfinished work."""
    for row in (previous,current):
        if row['patients']!=1:raise ValueError('Exactly one retained patient required')
        for key in ('resets','decisions','completed_episodes','successful_episodes','automatic_STOP_transitions',
                    'unfinished_episode_decisions','unfinished_executed_episodes'):
            if type(row[key]) is not int or row[key]<0:raise ValueError('Exact nonnegative counts required')
        if (sum(row['start_counts'].values())!=row['resets']
            or len(row['episode_returns'])!=row['completed_episodes']
            or row['successful_episodes']>row['completed_episodes']):raise ValueError('Episode denominator differs')
        for key,count in (('source_state_hashes','distinct_source_states'),
                          ('state_physical_action_hashes','distinct_state_physical_actions')):
            if len(set(row[key]))!=len(row[key]) or len(row[key])!=row[count]:raise ValueError('Unique identity mismatch')
    if 'retired_segments' in previous:
        retired=json.loads(json.dumps(previous['retired_segments'],allow_nan=False))
    else:
        retired=[]
        if any(previous.get(k,0) for k in ('retired_unfinished_episodes_at_resume',
                'retired_unfinished_decisions_at_resume','retired_open_resets_at_resume')):
            retired.append({'ended_at_decisions':64,'unfinished_episodes':previous['retired_unfinished_episodes_at_resume'],
                'unfinished_decisions':previous['retired_unfinished_decisions_at_resume'],
                'open_resets':previous['retired_open_resets_at_resume'],
                'catalog_costs':dict(previous['retired_live_episode_costs']),
                'scope':'Inherited64 retirement recorded in authenticated256 receipt'})
    already_retired=sum(x['unfinished_decisions'] for x in retired)
    already_open=sum(x['open_resets'] for x in retired)
    prior_complete=previous.get('completed_episode_decisions',
        previous['decisions']-already_retired-previous['unfinished_episode_decisions'])
    if prior_complete+already_retired+previous['unfinished_episode_decisions']!=previous['decisions']:
        raise ValueError('Inherited completed/retired/live decision partition differs')
    prior_open=previous['resets']-previous['completed_episodes']-already_open
    if prior_open<0:raise ValueError('Inherited reset partition differs')
    if prior_open or previous['unfinished_executed_episodes'] or previous['live_episode_costs']:
        retired.append({'ended_at_decisions':previous['decisions'],
            'unfinished_episodes':previous['unfinished_executed_episodes'],
            'unfinished_decisions':previous['unfinished_episode_decisions'],'open_resets':prior_open,
            'catalog_costs':dict(previous['live_episode_costs']),
            'scope':'Retired without terminal/failure relabeling; new physical episode at resume'})
    result={'patients':1}
    for key in ('resets','decisions','completed_episodes','successful_episodes','automatic_STOP_transitions'):
        result[key]=previous[key]+current[key]
    result['start_counts']={k:previous['start_counts'][k]+current['start_counts'][k] for k in previous['start_counts']}
    for key,count in (('source_state_hashes','distinct_source_states'),('state_physical_action_hashes','distinct_state_physical_actions')):
        result[key]=sorted(set(previous[key])|set(current[key]));result[count]=len(result[key])
    result['episode_returns']=[*previous['episode_returns'],*current['episode_returns']]
    result['completed_episode_decisions']=prior_complete+current['decisions']-current['unfinished_episode_decisions']
    result['retired_segments']=retired
    for field,name in (('unfinished_decisions','retired_unfinished_decisions_total'),
                       ('unfinished_episodes','retired_unfinished_episodes_total'),('open_resets','retired_open_resets_total')):
        result[name]=sum(x[field] for x in retired)
    result['unfinished_episode_decisions']=current['unfinished_episode_decisions']
    result['unfinished_executed_episodes']=current['unfinished_executed_episodes']
    result['current_open_resets']=current['resets']-current['completed_episodes']
    result['decision_partition_exact']=(result['completed_episode_decisions']+result['retired_unfinished_decisions_total']
        +result['unfinished_episode_decisions']==result['decisions'])
    result['reset_partition_exact']=(result['completed_episodes']+result['retired_open_resets_total']
        +result['current_open_resets']==result['resets'])
    if not result['decision_partition_exact'] or not result['reset_partition_exact']:raise ValueError('Cumulative partition differs')
    result['completed_catalog_costs']=_costsum(previous['completed_catalog_costs'],current['completed_catalog_costs'])
    result['retired_catalog_costs']=_costsum(*(x['catalog_costs'] for x in retired))
    result['live_episode_costs']=dict(current['live_episode_costs'])
    result['branch_lineage_shared_with_other_method']=True
    result['scope']='Per branch includes shared256 once; cross-branch paid totals must not duplicate shared experience'
    return result
