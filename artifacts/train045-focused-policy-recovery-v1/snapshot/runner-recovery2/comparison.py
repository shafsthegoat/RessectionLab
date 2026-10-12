"""Scalar packet/completion checks and predeclared matched-quality gate."""
import math
from common import semantic

METHODS=('GREEDY','TAIL')


def plain(value):
    # Lazy import keeps worker boundary installation ahead of scientific imports.
    from resectionlab.core import freeze_json,thaw_json
    return thaw_json(freeze_json(value))


def validate_plan(packets,account,prefix,frontier,horizon):
    # The actual common-prefix packets are immutable mappings, not ordinary JSON.
    packets,account,prefix,frontier=(plain(v) for v in (packets,account,prefix,frontier))
    if (account.get('complete') is not True or any(account.get(k,False) for k in
            ('call_cap_reached','time_cap_reached','preview_cap_reached','score_cap_reached'))
            or not len(prefix)<len(packets)<=horizon or len(prefix)!=frontier['steps']
            or semantic(packets[:len(prefix)])!=semantic(prefix)
            or account.get('model_transition_calls')!=len(packets)-len(prefix)
            or len(account.get('explored_history',()))!=len(packets)
            or semantic(account.get('explored_action_records'))!=semantic(packets)):
        raise ValueError('Complete genuine fixed-prefix rollout required')
    stops=[i for i,p in enumerate(packets) if p['action_id']=='STOP']
    if (stops and stops!=[len(packets)-1]) or (not stops and len(packets)!=horizon):
        raise ValueError('Complete STOP-or-absolute-horizon packet plan required')
    for i,(packet,row) in enumerate(zip(packets,account['explored_history'])):
        reward=row['reward']
        if (packet['steps_taken']!=i or packet['action_id']!=row['action_id']
            or packet['source_hash']!=frontier['source_hash']
            or packet['decision_model_hash']!=frontier['decision_model_hash']
            or packet['seal']!=semantic({k:v for k,v in packet.items() if k!='seal'})
            or type(reward) not in (int,float) or not math.isfinite(reward)):
            raise ValueError('Full packet/history/state/reward binding required')
        if i>=len(prefix) and (reward!=0 if packet['action_id']=='STOP' else reward<=0):
            raise ValueError('Every selected nonSTOP suffix action must be actually positive')
    return True


def matched_gate(branches):
    admitted=set(branches)==set(METHODS) and all(
        row.get('complete') is True and row.get('audit',{}).get('accepted') is True
        and row['audit'].get('complete_episode') is True
        and row['audit'].get('geometry',{}).get('feasible') is True
        and row.get('planned_physical_history_matches') is True for row in branches.values())
    result={'comparison_admissible':bool(admitted),'useful_gain':False,'tie_is_negative':True,
        'scope':'Frozen learned productive-continuation ranking on the same expanded catalog; simulation only.',
        'tolerance':1e-8}
    if not admitted:return result
    a=branches['GREEDY']['audit']['outcomes'];b=branches['TAIL']['audit']['outcomes']
    fields=('positive_target_source_cells_removed','target_removed_mm3','total_reward','normal_removed_mm3',
        'complete_tool_path_length_mm','tool_changes','nonstop_actions',
        'cumulative_contacted_tissue_upper_bound_mm3','currently_retained_contacted_tissue_upper_bound_mm3')
    if any(type(x.get(k)) not in (int,float) or not math.isfinite(x[k]) for x in (a,b) for k in fields):
        result.update(comparison_admissible=False,reason='Nonfinite or missing complete outcomes');return result
    delta={k:b[k]-a[k] for k in fields}
    no_regression=(delta['positive_target_source_cells_removed']>=0 and delta['target_removed_mm3']>=-1e-8
        and delta['total_reward']>=-1e-8 and delta['normal_removed_mm3']<=1e-8)
    strict=(delta['positive_target_source_cells_removed']>0 or delta['target_removed_mm3']>1e-8
        or delta['total_reward']>1e-8)
    result.update(deltas_tail_minus_greedy=delta,quality_nonregression=no_regression,
        strict_quality_improvement=strict,useful_gain=bool(no_regression and strict),
        outside_removal_nonregression=delta['normal_removed_mm3']<=1e-8,
        full_constraints_accepted=True,
        switching_travel_contact_scope='Reported separately; no claim these improve unless measured deltas do.')
    return result
