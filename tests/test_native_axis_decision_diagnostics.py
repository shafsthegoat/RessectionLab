"""Constructed saved records only; never load or execute a model/environment."""
import copy
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_native_axis_decision_diagnostics import diagnose_pairs


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def descriptor(values, shape, dtype='<f4'):
    return {'values':values,'shape':shape,'dtype':dtype}


def records():
    result=[]
    for update in (0,1):
        for ordinal,seed in enumerate((11,23)):
            for step in range(3):
                ids=['STOP','first','aliased','other']
                stop=[1.] + [0.]*14
                row=[0., float(10+step)] + [0.]*13
                other=[0., float(5+step)] + [0.]*13
                actions=[stop,row,copy.deepcopy(row),other]
                state=[f32(step/3),0.,0.,0.,0.,0.]
                logits=[0.,3.,3.,1.] if update==0 else [2.,4.,4.,1.]
                inputs={key:descriptor(copy.deepcopy(actions),[4,15]) for key in
                    ('source_action_features','action_features','actor_action_features')}
                inputs.update({key:descriptor(state,[6]) for key in ('state_features','source_state_features')})
                inputs.update(action_ids=ids,action_mask=descriptor([True]*4,[4],'|b1'))
                payload={'version':'learner-decision-observer-v1','role':'selection','seed':seed,
                    'update':update,'panel':update,'episode':ordinal,'step':step,'inputs':inputs,
                    'logits':descriptor(logits,[4]),'value':float(update), 'selected_index':1,
                    'selected_action_id':'first','decision_rule':'deterministic_argmax','forced_reason':None,
                    'forward_evaluated':True}
                result.append({'kind':'decision','status':'step_returned','role':'selection','seed':seed,
                    'decision_id':f'decision-{len(result)}','payload':payload})
    return result


def test_all_six_pairs_group_three_states_and_exact_aliases():
    result=diagnose_pairs(records(),[11,23],'initial','latest')
    assert result['selection_record_pairs']==6 and result['unique_states']==3
    assert result['all_chosen_actions_unchanged']
    for pair in result['pairs']:
        assert not pair['logits_exactly_unchanged']
        assert pair['rank_changed_actions']==2
        assert pair['value_delta']==1
        assert pair['legal_nonstop_actions']==3 and pair['unique_nonstop_feature_rows']==2
        assert pair['alias_groups'][0]['action_ids']==['first','aliased']
        assert pair['initial_top_two_margin']==pair['latest_top_two_margin']==0
        assert pair['initial_policy_hash']=='initial' and pair['latest_policy_hash']=='latest'


@pytest.mark.parametrize('attack',['missing','duplicate','ordering','state','features','mask','phase','chosen','float64'])
def test_any_pairing_or_actual_forward_mismatch_fails_closed(attack):
    events=records()
    event=events[6]
    payload=event['payload']
    if attack=='missing':events.pop()
    elif attack=='duplicate':events[-1]=copy.deepcopy(events[-2])
    elif attack=='ordering':payload['inputs']['action_ids'][1:3]=['aliased','first'];payload['selected_action_id']='aliased'
    elif attack=='state':
        for key in ('state_features','source_state_features'):payload['inputs'][key]['values'][0]=.5
    elif attack=='features':
        for key in ('action_features','actor_action_features','source_action_features'):
            payload['inputs'][key]['values'][3][1]=8.
    elif attack=='mask':payload['inputs']['action_mask']['values'][3]=False
    elif attack=='phase':payload['panel']=0
    elif attack=='chosen':payload['selected_index']=2;payload['selected_action_id']='aliased'
    else:payload['inputs']['action_features']['dtype']='<f8'
    with pytest.raises(ValueError):diagnose_pairs(events,[11,23],'initial','latest')


def test_near_equal_features_are_not_declared_exact_aliases():
    events=records()
    for event in events:
        for key in ('action_features','actor_action_features','source_action_features'):
            event['payload']['inputs'][key]['values'][2][1]=f32(event['payload']['inputs'][key]['values'][2][1]+.00001)
    result=diagnose_pairs(events,[11,23],'initial','latest')
    assert all(pair['unique_nonstop_feature_rows']==3 and pair['alias_groups']==[] for pair in result['pairs'])
