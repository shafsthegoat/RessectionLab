"""Post-plan exact deterministic join; never supplies a row to either planner."""
from common import semantic
from comparison import plain


def require_retained_greedy_prefix(retained,recovered):
    retained,recovered=plain(retained),plain(recovered)
    old=retained.get('policy')
    if (not isinstance(old,dict) or retained.get('scoring') is not None or old.get('method')!='GREEDY'
        or old.get('complete') is not False or old.get('fixed_prefix_length')!=12
        or old.get('model_transition_calls')!=11 or old.get('model_transition_completed')!=11
        or len(old.get('explored_action_records',()))!=23 or len(old.get('explored_history',()))!=23
        or recovered.get('method')!='GREEDY' or recovered.get('complete') is not True
        or recovered.get('fixed_prefix_length')!=12 or len(recovered.get('explored_history',()))<23):
        raise ValueError('Retained11 executed decisions and complete recovered GREEDY required')
    # The final uncommitted old state may exist; only committed states authorize parity.
    before=[row for row in old['states'] if 'result_state_hash' in row.get('selected',{})]
    after=[row for row in recovered['states'] if 'result_state_hash' in row.get('selected',{})]
    if len(before)!=11 or len(after)<11:raise ValueError('Exactly11 old committed selection states required')
    def exact(a,b,label):
        if semantic(a)!=semantic(b):raise ValueError('Recovery changed retained GREEDY '+label)
    exact(old['context'],recovered['context'],'public context')
    exact(old['context_seal'],recovered['context_seal'],'context seal')
    exact(old['controller'],recovered['controller'],'controller')
    exact(old['initial_focus'],recovered['initial_focus'],'initial focus')
    exact(old['fixed_prefix_return'],recovered['fixed_prefix_return'],'original cumulative return')
    exact(old['explored_action_records'],recovered['explored_action_records'][:23],'full23 packet prefix')
    exact(old['explored_history'],recovered['explored_history'][:23],'full23 native history prefix')
    rows=[]
    for i,(a,b) in enumerate(zip(before,after[:11]),13):
        # No timing/cost fields occur in states: exact includes candidate order,
        # scores, packet/certificate ownership, focus, goals and every ledger row.
        exact(a,b,'complete decision state '+str(i))
        packet=a['selected']['packet']
        if packet['action_id']=='STOP' or a['selected']['immediate_reward']<=0:
            raise ValueError('Saved nonterminal positive selected motion required')
        rows.append({'absolute_action':i,'action_id':packet['action_id'],
            'physical_action':packet['physical_action'],'packet_seal':packet['seal'],
            'state_and_scores_hash':semantic(a),'history_row_hash':semantic(old['explored_history'][i-1]),
            'focus':packet['focus'],'immediate_reward':a['selected']['immediate_reward']})
    return {'accepted':True,'exact_committed_suffix_decisions':11,'exact_full_prefix_actions':23,
        'retained_complete':False,'recovered_complete':True,'old_trace_used_by_planner':False,
        'matched_scope':'All23 packets and raw native histories; all first11 selected-state records including ordered scores, goals, focus, ownership and full ledger.',
        'packet_prefix_hash':semantic(old['explored_action_records']),
        'history_prefix_hash':semantic(old['explored_history']),'decisions':rows}
