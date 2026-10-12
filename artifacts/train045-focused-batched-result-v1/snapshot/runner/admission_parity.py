"""Evaluator-only exact trace parity, with each release authenticated in its own chain."""
from comparison import plain
from common import semantic


def admission_join(old_account,new_account,old,new):
    old_account,new_account,old,new=map(plain,(old_account,new_account,old,new))
    for account,chain in ((old_account,old),(new_account,new)):
        context=account['context']
        if (semantic(context)!=account['context_seal']
            or context['original_context_hash']!=semantic(chain['context'])
            or chain['context']['protocol_sha256']!=semantic(chain['bindings']['protocol'])
            or chain['bindings']['protocol']['runtime_release_sha256']!=chain['release_sha256']
            or chain['release']['root_source_index_sha256']!=chain['source_index_sha256']):
            raise ValueError('Context must bind to its own complete authenticated admission chain')
    def equal_without(a,b,key):
        if semantic({k:v for k,v in a.items() if k!=key})!=semantic({k:v for k,v in b.items() if k!=key}):
            raise ValueError('Unexpected scientific admission/context difference outside '+key)
    equal_without(old_account['context'],new_account['context'],'original_context_hash')
    equal_without(old['context'],new['context'],'protocol_sha256')
    equal_without(old['bindings'],new['bindings'],'protocol')
    equal_without(old['bindings']['protocol'],new['bindings']['protocol'],'runtime_release_sha256')
    equal_without(old['release'],new['release'],'root_source_index_sha256')
    return {'old_context_seal':old_account['context_seal'],'new_context_seal':new_account['context_seal'],
        'old_original_context_hash':old_account['context']['original_context_hash'],
        'new_original_context_hash':new_account['context']['original_context_hash'],
        'old_release_sha256':old['release_sha256'],'new_release_sha256':new['release_sha256'],
        'only_change':'Each context joins its own runtime release; every other context, binding, protocol and release field exact.'}


def require_saved_trace_parity(old,new,old_admission,new_admission,*,method,committed_suffix,old_complete):
    old,new=plain(old),plain(new);n=12+committed_suffix
    if (old['method']!=method or new['method']!=method or old['complete'] is not old_complete or not new['complete']
        or old['fixed_prefix_length']!=12 or new['fixed_prefix_length']!=12
        or len(old['explored_action_records'])!=n or len(old['explored_history'])!=n
        or len(new['explored_action_records'])<n):raise ValueError('Exact retained committed trace denominator required')
    join=admission_join(old,new,old_admission,new_admission)
    for field in ('controller','initial_focus','fixed_prefix_return'):
        if semantic(old[field])!=semantic(new[field]):raise ValueError('Retained '+field+' differs')
    for field in ('explored_action_records','explored_history'):
        if semantic(old[field])!=semantic(new[field][:n]):raise ValueError('Retained full '+field+' differs')
    a=[r for r in old['states'] if 'result_state_hash' in r.get('selected',{})]
    b=[r for r in new['states'] if 'result_state_hash' in r.get('selected',{})]
    if len(a)!=committed_suffix or len(b)<committed_suffix or semantic(a)!=semantic(b[:committed_suffix]):
        raise ValueError('Retained complete decision states/order/scores/focus differ')
    if old_complete and (len(new['explored_action_records'])!=n or len(b)!=committed_suffix):
        raise ValueError('Complete retained route length changed')
    return {'accepted':True,'method':method,'exact_full_prefix_actions':n,'exact_committed_suffix_decisions':committed_suffix,
        'old_complete':old_complete,'admission_join':join,'packets_hash':semantic(old['explored_action_records']),
        'history_hash':semantic(old['explored_history']),'decision_states_hash':semantic(a),'old_trace_used_by_planner':False}
