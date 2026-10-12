"""Scalar admission of an explicitly incomplete ancestor; no relabelled completion."""
import hashlib,json


def semantic(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def recovery_start(decisions):
    if type(decisions) is not int or not 0<=decisions<=512:
        raise ValueError('Recovery clock must be0..512')
    return 14 if decisions<128 else 13 if decisions<256 else 12


def require_failure(report,parent,curve,postmortem,old_index,*,pins,configuration):
    c=parent['owned_lifecycle'];last=parent['last_saved_costs']
    if (parent['status']!='incomplete_or_failed' or parent['stop_reason']!='rss_cap'
        or parent['report_sha256']!=pins['report'] or parent['source_index_sha256']!=pins['index']
        or report['source_index_sha256']!=pins['index'] or report['status']=='completed'
        or not c['verified'] or not c['child_reaped'] or c['exit_code']!=-15 or c['errors']
        or c['remaining_owned_pids'] or c['remaining_state_unknown']
        or parent['sampled_peak_group_rss_bytes']<=parent['caps']['worker_peak_rss_bytes']
        or postmortem['source_index_sha256']!=pins['index'] or postmortem['mismatches']
        or postmortem['source_files']!=old_index['files']
        or postmortem['files_checked']!=len(old_index['files'])):
        raise ValueError('Exact failed RSS lineage and separate postmortem source check required')
    if (configuration['resume_decisions']!=512 or configuration['additional_decisions']!=512
        or configuration['schedule_offset']!=256 or configuration['arms']!=['reverse']
        or configuration['milestones']!=[1024] or configuration['seeds']!=[0]
        or configuration['caps']['worker_peak_rss_bytes']!=4*1024**3):
        raise ValueError('Only reverse512 recovery under unchanged4GiB is declared')
    for key in ('goal','prefix_actions','absolute_horizon','query','curriculum_starts','evaluation_start',
                'reward_policy','success','torch_threads','reverse_windows'):
        if configuration[key]!=report['config'][key]:raise ValueError('Frozen task/schedule changed: '+key)
    uniform=report['arms']['uniform-seed0'];u=uniform['checkpoints'][-1]
    if (uniform['status']!='completed' or u['decisions']!=1024
        or not u['evaluation']['complete'] or not u['evaluation']['audit']['accepted']
        or not u['evaluation']['local_reward_audit_accepted']
        or semantic(u['evaluation'])!=semantic(report['evaluations']['uniform-seed0-eval-1024'])):
        raise ValueError('Retained completed uniform1024 endpoint required')
    if (curve['arm']!='reverse' or curve['seed']!=0 or len(curve['checkpoints'])!=1
        or curve['status']=='completed' or curve['shared_starting_checkpoint']!=uniform['shared_starting_checkpoint']
        or curve['ppo_config']!=uniform['ppo_config']):
        raise ValueError('Original incomplete reverse arm and exact common frozen256 required')
    q=curve['checkpoints'][0]
    if (q['decisions']!=512 or q['optimizer_completed']!=353 or q['optimizer_attempted']!=353
        or q['experience']['decisions']!=512 or q['experience']['unfinished_episode_decisions']!=6
        or q['encoder_changed'] or q['encoder_hash']!=curve['initialization']['encoder_hash']
        or not q['checkpoint']['includes_optimizer_state']
        or q['checkpoint']['sha256']!='18794e285ae2b7d6219fe297bcf5a6e9d060c46710b822f6c3dc2575e5fbac47'
        or not q['evaluation']['complete'] or not q['evaluation']['audit']['accepted']
        or not q['evaluation']['local_reward_audit_accepted']
        or semantic(q['evaluation'])!=semantic(report['evaluations']['reverse-seed0-eval-512'])):
        raise ValueError('Exact audited reverse512+353Adam checkpoint required')
    if (last['decisions']!=832 or last['optimizer_completed']!=393
        or last['optimizer_attempted']!=393 or not last['native_budget']['counting_reliable']):
        raise ValueError('Exact durable interrupted-work boundary changed')
    return {'original_campaign_completed':False,'failed_parent_sha256':pins['launcher'],
        'failed_report_sha256':pins['report'],'postmortem_sha256':pins['postmortem'],
        'last_saved_clock':832,'restored_clock':512,'restored_Adam_steps':353,
        'discarded_postcheckpoint_decisions_lower_bound':320,'discarded_postcheckpoint_Adam_steps_lower_bound':40,
        'discarded_work_is_not_checkpoint_lineage':True,'exact_unobserved_tail_work':None,
        'failed_parent_seconds':parent['elapsed_seconds'],
        'failed_native_preview_lower_bound':last['native_budget']['native_preview_entries'],
        'failed_native_transition_lower_bound':last['transitions']['attempted'],
        'fresh_RNG_and_environment':True,'seamless_trajectory_claim':False,
        'comparison_caveat':'Completed uniform retained; reverse restarted from512 with arithmetic-preserving memory change, not the interrupted trajectory continued.'}


def complete_result(report):
    try:
        arm=report['arms']['reverse-seed0'];q=arm['checkpoints'][0];e=q['evaluation']
        return (report['status']=='completed' and report['source_unchanged'] and report['qualified']
            and set(report['arms'])=={'reverse-seed0'} and len(report['source_visits'])==1
            and arm['status']=='completed' and arm['experience']['decisions']==1024
            and arm['segment_experience']['decisions']==512 and q['decisions']==1024
            and q['optimizer_attempted']==q['optimizer_completed'] and 0<q['additional_optimizer_completed']<=512
            and not q['encoder_changed'] and arm['optimizer_resume_join']['all_tensors_and_optimizer_scalars_exact']
            and e['complete'] and e['audit']['accepted'] and e['local_reward_audit_accepted']
            and q['evaluation_isolation']['unchanged_training_state'] and q['evaluation_isolation']['rng_restored']
            and not report['recovery_admission']['original_campaign_completed'])
    except (KeyError,TypeError,IndexError):return False
