"""Scalar-only admission/accounting for one64→256 optimizer continuation."""
import hashlib,json,math
ARMS=('frozen_encoder-seed0','trainable_encoder-seed0')
EXPECTED_PPO={'n_steps':64,'batch_size':4,'n_epochs':4,'learning_rate':.0003,
    'gamma':1.,'gae_lambda':1.,'clip_range':.2,'ent_coef':.01,'vf_coef':.5,
    'max_grad_norm':.5,'target_kl':.03,'device':'cpu','verbose':0}


def semantic(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require_pilot(report,parent,*,report_sha,index_sha,configuration):
    c=parent['owned_lifecycle']
    if (report['status']!='completed' or not report['source_unchanged'] or not report['qualified']
        or report['patients']!=1 or report['subject']!='ReMIND-045' or report['role']!='TRAIN'
        or parent['status']!='completed' or parent['report_sha256']!=report_sha
        or report['source_index_sha256']!=index_sha or parent['source_index_sha256']!=index_sha
        or not c['verified'] or not c['child_reaped'] or c['exit_code']!=0 or c['errors']
        or c['remaining_owned_pids'] or c['remaining_state_unknown']
        or set(report['arms'])!=set(ARMS) or report['config']['milestones']!=[64]
        or report['config']['seeds']!=[0] or configuration['milestones']!=[256]
        or configuration['seeds']!=[0] or configuration['arms']!=['frozen_encoder','trainable_encoder']):
        raise ValueError('Clean completed paired64 pilot is mandatory')
    for key in ('goal','prefix_actions','absolute_horizon','query','curriculum_starts',
                'evaluation_start','reward_policy','success','torch_threads'):
        if configuration[key]!=report['config'][key]:raise ValueError('Scientific protocol changed: '+key)
    for name in ARMS:
        arm=report['arms'][name]
        if (arm['status']!='completed' or arm['seed']!=0 or arm['ppo_config']!=EXPECTED_PPO
            or arm['milestones']!=[64] or len(arm['checkpoints'])!=1
            or arm['experience']['decisions']!=64):raise ValueError('Complete fixed pilot arm required')
        row=arm['checkpoints'][0]
        if (row['decisions']!=64 or row['checkpoint']['decisions']!=64
            or not row['checkpoint']['includes_optimizer_state']
            or not 0<row['optimizer_completed']<=64 or row['optimizer_attempted']!=row['optimizer_completed']
            or row['policy_costs']['optimizer_completed']!=row['optimizer_completed']
            or row['policy_costs']['optimizer_attempted']!=row['optimizer_attempted']):
            raise ValueError('Real completed optimizer endpoint required')
        for label,result in (('init',arm['initial_evaluation']),('64',row['evaluation'])):
            if (not result['complete'] or not result['audit']['accepted']
                or not result['local_reward_audit_accepted']
                or semantic(result)!=semantic(report['evaluations'][name+'-eval-'+label])):
                raise ValueError('Exact audited inherited evaluation required')
        validate_experience(arm['experience'])
    return {'report_sha256':report_sha,'source_index_sha256':index_sha,
        'arm_checkpoint_sha256':{n:report['arms'][n]['checkpoints'][0]['checkpoint']['sha256'] for n in ARMS},
        'replay_scope':'Inherited initial and64 audits retained by exact report hash; no repeated inference'}


def validate_experience(value):
    if value['patients']!=1:raise ValueError('One patient only')
    for key in ('resets','decisions','completed_episodes','successful_episodes','automatic_STOP_transitions',
                'unfinished_episode_decisions','unfinished_executed_episodes'):
        if type(value[key]) is not int or value[key]<0:raise ValueError('Nonnegative exact experience count required')
    for key,count in (('source_state_hashes','distinct_source_states'),
                      ('state_physical_action_hashes','distinct_state_physical_actions')):
        rows=value[key]
        if len(rows)!=len(set(rows)) or len(rows)!=value[count] or any(not isinstance(x,str) for x in rows):
            raise ValueError('Actual unique identity list/count mismatch')
    if (set(value['start_counts'])!={'12','13','14','15','16'}
        or any(type(x) is not int or x<0 for x in value['start_counts'].values())
        or sum(value['start_counts'].values())!=value['resets']
        or value['successful_episodes']>value['completed_episodes']
        or value['completed_episodes']>value['resets']
        or len(value['episode_returns'])!=value['completed_episodes']
        or value['unfinished_episode_decisions']>value['decisions']
        or value['unfinished_executed_episodes']>1
        or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in value['episode_returns'])):
        raise ValueError('Episode/reset denominators differ')
    for rows in (value['completed_catalog_costs'],value['live_episode_costs']):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in rows.values()):
            raise ValueError('Finite positive paid costs required')


def merge_experience(previous,current):
    """Counts sum; identities union. Retired live episodes do not become failures."""
    validate_experience(previous);validate_experience(current)
    sums=('resets','decisions','completed_episodes','successful_episodes','automatic_STOP_transitions')
    result={key:previous[key]+current[key] for key in sums};result['patients']=1
    result['start_counts']={k:previous['start_counts'][k]+current['start_counts'][k] for k in previous['start_counts']}
    for key,count in (('source_state_hashes','distinct_source_states'),
                      ('state_physical_action_hashes','distinct_state_physical_actions')):
        result[key]=sorted(set(previous[key])|set(current[key]));result[count]=len(result[key])
    result['episode_returns']=[*previous['episode_returns'],*current['episode_returns']]
    result['unfinished_episode_decisions']=current['unfinished_episode_decisions']
    result['unfinished_executed_episodes']=current['unfinished_executed_episodes']
    result['retired_unfinished_episodes_at_resume']=previous['unfinished_executed_episodes']
    result['retired_unfinished_decisions_at_resume']=previous['unfinished_episode_decisions']
    result['retired_open_resets_at_resume']=previous['resets']-previous['completed_episodes']
    result['completed_episode_decisions']=(previous['decisions']-previous['unfinished_episode_decisions']
        +current['decisions']-current['unfinished_episode_decisions'])
    result['decision_partition_exact']=(result['completed_episode_decisions']
        +result['retired_unfinished_decisions_at_resume']+result['unfinished_episode_decisions']==result['decisions'])
    result['completed_catalog_costs']={k:previous['completed_catalog_costs'].get(k,0)+current['completed_catalog_costs'].get(k,0)
        for k in previous['completed_catalog_costs'].keys()|current['completed_catalog_costs'].keys()}
    result['retired_live_episode_costs']=dict(previous['live_episode_costs'])
    result['live_episode_costs']=dict(current['live_episode_costs'])
    result['restart_scope']='Fresh uniform genuine snapshot reset; no physical/RNG trajectory continuation claim'
    return result
