"""Small generated controls; root runs these in its serial compute slot."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(HERE))
import paths
import pytest
from resectionlab.core import semantic_digest
from resectionlab.native_resection import NativeResectionEngine
from local_goal import LocalEpisode,local_reward
from beam import Node,branch,propose,retain

spec=importlib.util.spec_from_file_location('local_goal_search_endpoint',HERE/'worker.py')
endpoint=importlib.util.module_from_spec(spec);spec.loader.exec_module(endpoint)


def test_all_negative_choices_remain_eligible_and_both_ranking_lanes_present():
    rows=[{'index':i,'local_reward':-float(i),'blocker_overlap':overlap}
          for i,overlap in ((1,0),(2,0),(3,8),(4,9),(5,1))]
    assert propose(rows)==(1,2,4,3)
    assert propose([{'index':1,'local_reward':-1.,'blocker_overlap':0}])==(1,)
    with pytest.raises(ValueError):propose([{'index':1,'local_reward':float('nan'),'blocker_overlap':0}])


def test_live_beam_cannot_be_taken_by_STOP_and_streaming_retention_matches_full():
    def node(name,value,cleared,done=False):
        return Node(SimpleNamespace(done=done,total_reward=value), (name,),
            cleared=frozenset(range(cleared)))
    rows=[node('STOP',0,0,True),node('cheap',-.1,0),node('progress',-.3,5),
          node('costly',-.8,3),node('later-best',-.05,1),node('later-progress',-.4,7)]
    expected=[n.path for n in retain(rows)]
    assert expected==[('later-best',),('later-progress',)]
    streamed=[]
    for row in rows:streamed=retain([*streamed,row])
    assert [n.path for n in streamed]==expected


def test_owned_current_certificates_branch_without_repreview_or_parent_mutation(monkeypatch):
    from test_local_goal import fixture
    context,snapshot=fixture();parent=LocalEpisode(context,snapshot)
    catalog=parent.observe()
    before=(parent.work._engine.state_hash,semantic_digest(parent.work._history),
        catalog._seal,tuple(id(p) for p in catalog.pages),tuple(catalog.packets))
    candidates=[i for i in range(1,len(catalog.identifiers)) if
        not local_reward(context.goal,catalog.scores[i],parent.work._current_tool,already_succeeded=False)[1]]
    assert candidates,'Generated control needs one non-goal certified preparation'
    def forbidden(*args,**kwargs):raise AssertionError('Branch commit rebuilt native catalog')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    a=branch(parent,candidates[0]);b=branch(parent,candidates[0]);stop=branch(parent,0)
    assert a.work is not b.work and a.work._engine is not b.work._engine
    assert a.work._engine.state_hash==b.work._engine.state_hash
    assert a.packets==b.packets and a.rows==b.rows
    assert a.catalog is None and b.catalog is None and stop.done and stop.total_reward==0
    assert not catalog._consumed and not parent.done
    assert before==(parent.work._engine.state_hash,semantic_digest(parent.work._history),
        catalog._seal,tuple(id(p) for p in catalog.pages),tuple(catalog.packets))
    catalog.require_batched()
    with pytest.raises(ValueError):branch(parent,len(catalog.identifiers))


def test_independent_reward_reconstruction_keeps_negative_preparation_and_one_bonus():
    prefix=[{'action_id':'prefix','tool_id':'a','removed_indices_native':[]} for _ in range(12)]
    prep={'action_id':'prep','tool_id':'a','removed_indices_native':[],
        'complete_tool_path_length_mm':10.,'normal_removed_mm3':0.}
    goal={**prep,'action_id':'goal','tool_id':'b','removed_indices_native':[[1,2,3]]}
    stop={'action_id':'STOP','tool_id':None,'removed_indices_native':[]}
    result=endpoint.reward_audit((1,2,3),[*prefix,prep,goal,stop])
    assert result['accepted'] and result['success'] and result['bonus_count']==1
    assert result['rows']==pytest.approx([-.04,9.93,0.])
    assert result['total']==pytest.approx(9.89)
    assert endpoint.reward_audit((1,2,3),[*prefix,stop])['total']==0
    with pytest.raises(ValueError):endpoint.reward_audit((1,2,3),[*prefix,goal,prep,stop])


def test_search_reaches_delayed_goal_without_teacher_path_and_preserves_cap_negative(monkeypatch):
    """Scalar state-machine control of search; native ownership is tested above."""
    import copy
    import beam
    from local_goal import GoalSpec
    goal=GoalSpec((9,9,9),'a',(0.,0.,0.),(1.,0.,0.))
    context=SimpleNamespace(goal=goal)
    counters={'previews':0,'transitions':0}
    class ToyEpisode:
        def __init__(self,context,snapshot,guard):
            self.context=context;self.guard=guard;self.catalog=None
            self.work=SimpleNamespace(_steps=12,_current_tool='a',_engine=SimpleNamespace(state_hash='initial'),_history=[])
            self.total_reward=0.;self.success=False;self.done=False;self.packets=[];self.rows=[];self.cleared=0
        def observe(self):
            self.guard()
            if self.catalog is None:
                counters['previews']+=1
                blocker=(self.cleared,0,0)
                records=[{'action_id':'STOP','reward':0.}]
                for name,removed,travel in [('cheap',[],0.),('clearing',[blocker],10.)]:
                    if name=='clearing' and self.cleared==4:removed=[blocker,goal.cell]
                    records.append({'action_id':name,'tool_id':'a','removed_indices_native':removed,
                        'outcome_scope':'permitted_nominal_model','normal_removed_mm3':0.,
                        'complete_tool_path_length_mm':travel})
                self.catalog=SimpleNamespace(identifiers=('STOP','cheap','clearing'),scores=records,
                    _seal='catalog'+str(self.work._steps),observation=SimpleNamespace(fingerprint='obs'),work=self.work)
            return self.catalog
    def toy_branch(parent,index):
        cat=parent.observe();child=copy.deepcopy(parent);child.catalog=None;counters['transitions']+=1
        score=cat.scores[index];packet={'action_id':cat.identifiers[index]};child.packets.append(packet)
        value,success,terms=local_reward(goal,score,'a',already_succeeded=False)
        child.work._steps+=1;child.total_reward+=value;child.success=success
        child.cleared+=int(index==2)
        child.done=bool(index==0 or success or child.work._steps==18)
        child.rows.append({'reward':value});child.work._history.append(packet)
        child.work._engine.state_hash=str((child.work._steps,child.cleared))
        return child
    monkeypatch.setattr(beam,'LocalEpisode',ToyEpisode)
    monkeypatch.setattr(beam,'branch',toy_branch)
    monkeypatch.setattr(beam,'current_blockers',lambda cat:(
        frozenset(tuple(c) for c in cat.scores[2]['removed_indices_native'] if tuple(c)!=goal.cell),{'present':True}))
    limits={'seconds':5,'native_previews':1000,'native_transitions':100}
    winner,account=beam.search(context,None,limits=limits,usage=lambda:dict(counters))
    assert winner.success and winner.total_reward==pytest.approx(9.8)
    assert account['complete'] and account['negative_children']>0 and not account['teacher_path_used']
    assert len(winner.packets)==5 and account['successful_children']>0
    counters.update(previews=0,transitions=0)
    winner,capped=beam.search(context,None,limits={**limits,'native_transitions':1},usage=lambda:dict(counters))
    assert winner.done and winner.total_reward==0 and not winner.success
    assert not capped['complete'] and 'transition allowance' in capped['cap_reason']
