"""Call-local membrane unions preserve the original validator's hook ordering."""
import ast
import copy
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
from resectionlab import surgical_cranial_stages as stages


def _legacy_validate_effects(engine):
    """Frozen pre-memo validator, kept independent of the optimized edge loop."""
    contract=getattr(engine,'_cranial_stages',None)
    if contract is None:return
    sets=source_partition_sets(contract);removed=set(engine.state['removed']);rim=sets['retained']
    if removed-(sets['bone']|sets['preparation']|sets['work']):_fail('REMOVAL_SOURCE_SCOPE')
    if removed&rim:_fail('RETAINED_RIM_REMOVAL')
    if any(int(c) in rim and np.linalg.norm(v)>0 for c,v in engine.state.get('deformation',{}).items()):_fail('RETAINED_RIM_DEFORMATION')
    fragments=[*engine.state['holds'].values(),*engine.state.get('detachedFragments',[])]
    if any(rim&set(h['cells']) for h in fragments):_fail('RETAINED_RIM_HOLD')
    for field in ('cutEdges','transectedEdges'):
        for edge in engine.state.get(field,[]):
            e=tuple(sorted(edge))
            if set(e)&(sets['preparation']|rim) and e not in sets['boundary']:_fail('RETAINED_MEMBRANE_TOPOLOGY')
    allowed=set(contract.attachment_ids)
    allowed.update(a['id'] for a in engine.scene.properties['attachments'] if set(a['cells'])&sets['bone'] and a['kind'] in ('soft','periosteal'))
    if set(engine.state['separatedAttachments'])-allowed:_fail('ATTACHMENT_RELEASE_SOURCE_SCOPE')


def _case(name):
    state={'removed': [], 'holds': {}, 'cutEdges': [(2,3), (2,3)],
           'transectedEdges': [], 'separatedAttachments': []}
    partitions={'bone': frozenset({1}), 'preparation': frozenset({2}),
                'retained': frozenset({3}), 'work': frozenset({4}), 'boundary': frozenset({(2,3)})}
    engine=SimpleNamespace(_cranial_stages=SimpleNamespace(attachment_ids=()), state=state,
        scene=SimpleNamespace(properties={'attachments': []}))
    events=[];bindings={}
    if name=='zero':state['cutEdges']=[]
    elif name=='no-contract':engine._cranial_stages=None
    elif name in ('sort', 'sort-error'):
        class Cell(int):
            def __lt__(self,other):
                events.append('sort')
                if name=='sort-error':raise RuntimeError('original sort failure')
                return int.__lt__(self,other)
            def __hash__(self):events.append('hash');return int.__hash__(self)
        class Preparation(frozenset):
            def __or__(self,other):events.append('union');return frozenset.__or__(self,other)
        state['cutEdges']=[(Cell(3),Cell(2))];partitions['preparation']=Preparation({2})
    elif name in ('hash-preparation', 'replace-transected', 'replace-retained'):
        class Cell(int):
            def __hash__(self):
                events.append(name)
                if name=='hash-preparation':partitions['preparation']=frozenset({8})
                elif name=='replace-transected':state['transectedEdges']=[(2,9)]
                else:partitions['retained']=frozenset({9})
                return int.__hash__(self)
        state['cutEdges']=[(Cell(8),9)] if name=='hash-preparation' else [(Cell(2),3)] if name=='replace-transected' else [(Cell(3),4)]
    elif name=='none-preparation':
        class Partitions(dict):
            reads=0
            def __getitem__(self,key):
                if key=='preparation':
                    self.reads+=1;events.append(('preparation',self.reads))
                    if self.reads>1:return None
                return super().__getitem__(key)
        partitions=Partitions(partitions)
    elif name in ('mutable', 'frozen-subclass', 'plain-set'):
        base=set if name=='mutable' else frozenset
        class Preparation(base):
            def __or__(self,other):events.append('union');return base.__or__(self,other)
        partitions['preparation']={2} if name=='plain-set' else Preparation({2})
    elif name in ('dedup-rim', 'dedup-edge'):
        class Cell(int):
            def __hash__(self):return int.__hash__(self)
            def __eq__(self,other):events.append('dedup-equality');return int.__eq__(self,other)
        if name=='dedup-rim':
            partitions['retained']=frozenset({Cell(2)})
            state['cutEdges']=[(2,2),(2,2)]
        else:state['cutEdges']=[(2,3),(2,Cell(2))]
        partitions['boundary']=frozenset({(2,2),(2,3)})
    elif name=='fresh-custom-intersection':
        retained=[];calls=[]
        class EdgeSet(set):
            def __and__(self,other):
                events.append(('fresh-union',all(other is not previous for previous in retained)))
                retained.append(other);return set.__and__(self,other)
        def custom_set(values=()):
            if isinstance(values,tuple) and len(values)==2:
                calls.append(None)
                if len(calls)>1:return EdgeSet(values)
            return set(values)
        bindings['set']=custom_set;state['cutEdges']=[(2,3)]*3
    elif name=='type-all':
        def changed(*args):events.append('unexpected-callback');raise AssertionError('New type/all lookup')
        bindings.update(type=changed,all=changed)
    elif name.startswith('refusal-'):
        reason=name.removeprefix('refusal-')
        if reason=='REMOVAL_SOURCE_SCOPE':state['removed']=[99,3]
        elif reason=='RETAINED_RIM_REMOVAL':state['removed']=[3];partitions['work']|={3}
        elif reason=='RETAINED_RIM_DEFORMATION':state['deformation']={'3':[0.,0.,1.]}
        elif reason=='RETAINED_RIM_HOLD':state['holds']={'left':{'cells':[3]}}
        elif reason=='RETAINED_MEMBRANE_TOPOLOGY':state['cutEdges']=[(2,9)]
        elif reason=='ATTACHMENT_RELEASE_SOURCE_SCOPE':state['separatedAttachments']=['unbound']
    return engine,partitions,events,bindings


@pytest.mark.parametrize('name', ['zero','no-contract','sort','sort-error','hash-preparation',
    'replace-transected','replace-retained','none-preparation','mutable','frozen-subclass',
    'plain-set','dedup-rim','dedup-edge','fresh-custom-intersection','type-all',
    *('refusal-'+reason for reason in ('REMOVAL_SOURCE_SCOPE','RETAINED_RIM_REMOVAL',
      'RETAINED_RIM_DEFORMATION','RETAINED_RIM_HOLD','RETAINED_MEMBRANE_TOPOLOGY','ATTACHMENT_RELEASE_SOURCE_SCOPE'))])
def test_effects_match_legacy_result_state_and_callback_order(monkeypatch,name):
    outcomes=[]
    for legacy in (True,False):
        engine,partitions,events,bindings=_case(name)
        before=json.dumps(engine.state,sort_keys=True)
        with monkeypatch.context() as patch:
            patch.setattr(stages,'source_partition_sets',lambda contract:partitions)
            for key,value in bindings.items():patch.setattr(stages,key,value,raising=False)
            function=FunctionType(_legacy_validate_effects.__code__,vars(stages)) if legacy else stages.validate_effects
            try:result=('return',function(engine))
            except Exception as error:result=(type(error).__name__,str(error))
        outcomes.append((result,copy.deepcopy(events),json.dumps(engine.state,sort_keys=True)))
    assert outcomes[0]==outcomes[1]
    result,events,state=outcomes[0]
    if name=='sort':assert events.index('sort')<events.index('hash')<events.index('union')
    if name=='none-preparation':assert result[0]=='TypeError'
    if name.startswith('refusal-'):
        assert result==('ValueError','CRANIAL_STAGE_'+name.removeprefix('refusal-'))
        assert state==before
    if name=='fresh-custom-intersection':assert events==[('fresh-union',True)]*2
    if name=='type-all':assert events==[]
    if name.startswith('dedup-'):assert events


def test_immutable_partition_checks_are_once_per_call_and_zero_edges_do_not_scan(monkeypatch):
    engine,partitions,_,_=_case('zero')
    monkeypatch.setattr(stages,'source_partition_sets',lambda contract:partitions)
    original=stages._effects_integer_members;checked=[]
    def record(values):checked.append(values);return original(values)
    monkeypatch.setattr(stages,'_effects_integer_members',record)
    stages.validate_effects(engine)
    assert checked==[]
    engine.state['cutEdges']=[(2,3)]*28
    for _ in range(2):stages.validate_effects(engine)
    assert sum(values is partitions['preparation'] for values in checked)==2
    assert sum(values is partitions['retained'] for values in checked)==2


@pytest.mark.parametrize('kind',['set-alias','heap-fake','metaclass','callable'])
def test_preimport_frozenset_captures_decline_without_callbacks(monkeypatch,kind):
    events=[]
    class Meta(type):
        def __getattribute__(cls,name):events.append(name);raise AssertionError('Metaclass hook invoked')
    class Callable:
        def __getattribute__(self,name):events.append(name);raise AssertionError('Capture hook invoked')
    replacement=(set if kind=='set-alias' else type('frozenset',(object,),{'__module__':'builtins'})
        if kind=='heap-fake' else Meta('frozenset',(object,),{}) if kind=='metaclass' else Callable())
    nodes=[node for node in ast.parse(Path(stages.__file__).read_text()).body
           if isinstance(node,ast.Assign) and all(isinstance(target,ast.Name) and
               target.id.startswith('_effects_') for target in node.targets)]
    namespace={'frozenset':replacement}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated-effects-startup','exec'),namespace)
    assert namespace['_effects_frozenset_genuine'] is False and events==[]
    for key,value in namespace.items():
        if key.startswith('_effects_'):monkeypatch.setattr(stages,key,value)
    def no_scan(values):raise AssertionError('Unsupported startup capture reused')
    monkeypatch.setattr(stages,'_effects_integer_members',no_scan)
    engine,partitions,_,_=_case('zero');engine.state['cutEdges']=[(2,3)]*3
    monkeypatch.setattr(stages,'source_partition_sets',lambda contract:partitions)
    assert stages.validate_effects(engine) is None and events==[]


def test_later_public_frozenset_replacement_adds_no_callback(monkeypatch):
    engine,partitions,_,_=_case('zero');engine.state['cutEdges']=[(2,3)]*3
    monkeypatch.setattr(stages,'source_partition_sets',lambda contract:partitions)
    def changed(*args):raise AssertionError('New public frozenset lookup')
    monkeypatch.setattr(stages,'frozenset',changed,raising=False)
    assert stages.validate_effects(engine) is None
