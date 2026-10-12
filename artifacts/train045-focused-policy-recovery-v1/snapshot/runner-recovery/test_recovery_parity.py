"""Generated scalar parity transport; no saved scientific payload is read."""
import copy,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import semantic
from recovery_parity import require_retained_greedy_prefix
from compact_output import encode_gzip,decode_gzip


def fixture():
    packets=[];history=[];states=[]
    for i in range(24):
        packet={'action_id':'generated-'+str(i),'physical_action':{'tool_id':'generated','tip_mm':[0,0,i]},
            'focus':None if i<12 else {'physical_goal':'generated-current-goal'},'steps_taken':i}
        packet['seal']=semantic(packet);packets.append(packet)
        history.append({'action_id':packet['action_id'],'reward':1.,'generated_native_record':i})
        if i>=12:states.append({'selected':{'packet':packet,'result_state_hash':semantic(i),
            'immediate_reward':1.,'next_active_goal':'generated-goal'},'candidates':[{'id':'other','score':.5},{'id':packet['action_id'],'score':1.}],
            'goal_focus':packet['focus'],'pages':[{'complete':True}]})
    new={'method':'GREEDY','complete':True,'fixed_prefix_length':12,'model_transition_calls':12,
        'model_transition_completed':12,'explored_action_records':packets,'explored_history':history,
        'states':states,'context':{'generated':True},'context_seal':semantic('generated'),
        'controller':'unchanged','initial_focus':None,'fixed_prefix_return':12.}
    old=copy.deepcopy(new);old.update(complete=False,model_transition_calls=11,model_transition_completed=11,
        explored_action_records=old['explored_action_records'][:23],explored_history=old['explored_history'][:23],
        states=old['states'][:11]+[{'incomplete_uncommitted':True}])
    return {'policy':old,'scoring':None},new


def test_complete_exact_join_and_lossless_bounded_decode():
    retained,new=fixture();raw,binding=encode_gzip(retained)
    result=require_retained_greedy_prefix(decode_gzip(raw,binding),new)
    assert result['accepted'] and result['exact_committed_suffix_decisions']==11
    assert result['exact_full_prefix_actions']==23 and not result['old_trace_used_by_planner']
    assert len(result['decisions'])==11 and result['decisions'][0]['absolute_action']==13


@pytest.mark.parametrize('field',['action','physical','history','score','focus','order','ownership','complete'])
def test_any_scientific_prefix_change_refuses(field):
    retained,new=fixture()
    if field=='action':new['explored_action_records'][12]['action_id']='foreign'
    elif field=='physical':new['explored_action_records'][12]['physical_action']['tip_mm'][0]=1
    elif field=='history':new['explored_history'][22]['reward']=2
    elif field=='score':new['states'][4]['candidates'][0]['score']=.6
    elif field=='focus':new['states'][4]['goal_focus']={'changed':True}
    elif field=='order':new['states'][4]['candidates'].reverse()
    elif field=='ownership':new['states'][4]['selected']['packet']['owner_page']=3
    else:new['complete']=False
    with pytest.raises(ValueError):require_retained_greedy_prefix(retained,new)
