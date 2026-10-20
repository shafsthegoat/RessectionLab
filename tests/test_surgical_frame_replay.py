import copy
import numpy as np
import pytest
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_frame_replay import SurgicalFrameCache


def episode(duration=1.2):
    labels=np.zeros((24,24,24),np.uint16)
    labels[10:12,10:12,10:13]=1
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'target','material':'tumor','target':True}],{}, {})
    engine=SurgicalEngine(scene)
    for action in [
        {'slot':'left','operation':'exchange','instrument':'cusa','duration':.2},
        {'slot':'left','operation':'place','pose':{'position':[10.,10.,5.],'quaternion':[0.,0.,0.,1.]},'parameters':{'radiusMm':.6},'duration':.2},
        {'slot':'left','operation':'apply','pose':{'position':[10.,10.,12.],'quaternion':[0.,0.,0.,1.]},'duration':duration},
    ]:
        assert engine.step(action)['status']=='accepted'
    return engine


def test_exact_microstep_frames_have_causal_removal_and_final_boundary():
    engine=episode(); original=copy.deepcopy(engine.snapshot()); history=copy.deepcopy(engine.history)
    frames=SurgicalFrameCache(); replay=SurgicalReplayCache()
    initial,timeline,first=frames.seek(engine,original,2,0,replay)
    assert first['readOnly'] and first['isActionBoundary']
    assert not initial.state['removed']
    counts=[]; times=[]
    for i in range(first['frameCount']):
        observed,_,info=frames.seek(engine,original,2,i,replay)
        counts.append(len(observed.state['removed'])); times.append(info['observedTimeSeconds'])
    assert counts==sorted(counts) and counts[0]==0 and counts[-1]>0
    assert times==sorted(times)
    assert observed.snapshot()==engine.snapshot()
    assert observed.history==history
    assert engine.snapshot()==original
    boundary,_=frames.boundary(original,3,verified_history=history)
    assert boundary.snapshot()==original
    assert type(boundary) is SurgicalEngine
    assert not hasattr(boundary,'_frames')


def test_frame_cache_rejects_out_of_range_and_changed_snapshot():
    engine=episode(); saved=engine.snapshot(); frames=SurgicalFrameCache(); replay=SurgicalReplayCache()
    with pytest.raises(ValueError,match='unavailable'): frames.seek(engine,saved,3,0,replay)
    _,_,info=frames.seek(engine,saved,2,0,replay)
    with pytest.raises(ValueError,match='unavailable'): frames.seek(engine,saved,2,info['frameCount'],replay)
    altered=copy.deepcopy(saved);altered['checkpoint']['state']['travelMm']+=1
    assert frames.boundary(altered,3) is None


def test_frame_hits_reject_modified_canonical_history_and_detach_editable_boundaries():
    engine=episode(); saved=engine.snapshot(); canonical=copy.deepcopy(engine.history)
    frames=SurgicalFrameCache(); replay=SurgicalReplayCache()
    _,_,info=frames.seek(engine,saved,2,0,replay,verified_history=canonical)
    modified=copy.deepcopy(canonical)
    modified[0]['events'].append({'kind':'unverified-history'})
    with pytest.raises(ValueError,match='canonical admission'):
        frames.seek(engine,saved,2,0,replay,verified_history=modified)
    with pytest.raises(ValueError,match='canonical admission'):
        frames.boundary(saved,3,verified_history=modified)
    assert frames.boundary(saved,3) is None
    boundary,timeline=frames.boundary(saved,3,verified_history=canonical)
    assert boundary.snapshot()==saved and boundary.history==canonical
    timeline[0]['events'].append({'kind':'caller-owned'})
    assert boundary.history==canonical
    next_action={'slot':'right','operation':'inspect','duration':.2}
    reference=engine.clone()
    assert boundary.step(next_action)==reference.step(next_action)
    assert boundary.snapshot()==reference.snapshot() and boundary.history==reference.history
    observed,_,_=frames.seek(engine,saved,2,info['frameCount']-1,replay,verified_history=canonical)
    assert observed.snapshot()==saved and observed.history==canonical


def test_caller_endpoint_state_alone_cannot_admit_a_frame_archive():
    engine=episode()
    engine.state['travelMm']+=1.
    altered=engine.snapshot()
    # Even an internally matching caller checkpoint is insufficient: the
    # canonical full history still ends at the genuinely executed state.
    with pytest.raises(ValueError,match='archive endpoint.*canonical admission'):
        SurgicalFrameCache().seek(engine,altered,2,0,SurgicalReplayCache(),verified_history=engine.history)


def test_large_legal_archive_does_not_evict_frames_or_reexecute_on_each_frame_hit(monkeypatch):
    from resectionlab.surgical_replay_cache import _owned_bytes
    # The public API accepts finite numeric duration strings. Long insignificant
    # fractional tails make a large legal archive with cheap actual physics;
    # no engine state or canonical history is fabricated or mutated.
    labels=np.zeros((4,4,4),np.uint16)
    source=SurgicalScene(labels,np.eye(4),[],{'kind':'large-history-engineering-fixture'}, {})
    engine=SurgicalEngine(source)
    for index in range(5):
        duration='.1'+'0'*(14*1024**2)+str(index)
        assert engine.step({'slot':'left','operation':'inspect','duration':duration})['status']=='accepted'
    saved=engine.snapshot(); canonical=engine.history
    assert _owned_bytes((engine.actions,canonical)) > 64*1024**2
    frames=SurgicalFrameCache(); replay=SurgicalReplayCache(max_bytes=1024)
    _,_,info=frames.seek(engine,saved,4,0,replay,verified_history=canonical)
    assert frames.statistics['actions']==1 and frames.statistics['ownedBytes'] < 64*1024**2
    item=next(iter(frames._actions.values()))
    assert item['shell'].state=={} and item['shell'].actions==[] and item['shell'].history==[]
    assert not any(name in item for name in ('baseline','endpoint','history'))
    def unexpected_step(*_args,**_kwargs):
        raise AssertionError('A repeated observed frame hit executed physics again')
    monkeypatch.setattr(SurgicalEngine,'step',unexpected_step)
    for position in (0,1,info['frameCount']-1):
        observed,timeline,_=frames.seek(engine,saved,4,position,replay,verified_history=canonical)
        assert timeline==canonical
    assert observed.snapshot()==saved and observed.history==canonical
    boundary,timeline=frames.boundary(saved,5,verified_history=canonical)
    assert boundary.snapshot()==saved and boundary.history==canonical and timeline==canonical
    assert frames.statistics['actions']==1 and frames.statistics['ownedBytes'] < 64*1024**2


@pytest.mark.parametrize('persistent', [False, True])
def test_short_apply_boundary_displays_observed_energy_without_latching_switch(tmp_path, persistent):
    from resectionlab.surgical_telemetry import capture_episode
    engine=episode(duration=.1)
    assert engine.state['removed'] and not engine.state['slots']['left']['active']
    saved=engine.snapshot(); history=copy.deepcopy(engine.history)
    frames=SurgicalFrameCache(directory=tmp_path); replay=SurgicalReplayCache()
    if persistent:
        token=frames.admit(engine,saved,replay,verified_history=history)
        # The invalid apply occurs at the same canonical position, but must
        # neither publish a frame nor leak an energized tip into the next start.
        attempted=copy.deepcopy(engine.actions)
        attempted.insert(2,{'slot':'left','operation':'apply','duration':0.})
        report=capture_episode(engine,attempted,frame_collector=frames.collector(token))
        assert report['summary']['invalidActionsCumulative']==1
        assert frames.store.publications==len(engine.actions)
        def seek(index):
            return frames.seek(engine,token.snapshot,2,index,replay,verified_history=token.history,admission=token)
    else:
        def seek(index): return frames.seek(engine,saved,2,index,replay,verified_history=history)
    initial,_,info=seek(0)
    assert info['frameCount']==2, 'sub-cadence action has only real start and finalized boundary'
    assert not info['effectiveActivation']['left']
    assert not initial.state['removed'] and not initial.state['slots']['left']['active']
    final,_,info=seek(1)
    assert info['effectiveActivation']['left'] and not info['effectiveActivation']['right']
    assert final.snapshot()==saved and not final.state['slots']['left']['active']
    assert engine.snapshot()==saved and engine.history==history


def test_failed_first_microstep_never_observes_or_lights_an_energized_tip(tmp_path):
    from resectionlab.surgical_frame_replay import _ObservedFrames
    from resectionlab.surgical_telemetry import capture_episode
    labels=np.zeros((24,24,24),np.uint16); labels[8,8,8]=1
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'bone','material':'bone'}],{}, {})
    engine=SurgicalEngine(scene)
    pose=lambda z:{'position':[8.,8.,z],'quaternion':[0.,0.,0.,1.]}
    actions=[{'slot':'left','operation':'exchange','instrument':'cusa','duration':.1},
             {'slot':'left','operation':'place','pose':pose(6.5),'parameters':{'radiusMm':.9},'duration':.1}]
    for action in actions: assert engine.step(action)['status']=='accepted'
    failed={'slot':'left','operation':'apply','pose':pose(6.9),'duration':.1}
    legacy=_ObservedFrames(scene); legacy.__dict__.update(engine.clone().__dict__)
    legacy._frames=[]; legacy._next_frame_time=legacy.state['elapsedSeconds']+.2
    legacy._last_effective_activation=None
    outcome=legacy.step(failed)
    assert outcome['status']=='rejected' and outcome['reason']=='UNSUPPORTED_MATERIAL'
    assert legacy._frames==[] and legacy._last_effective_activation is None
    assert legacy.snapshot()==engine.snapshot() and not legacy.state['slots']['left']['active']
    frames=SurgicalFrameCache(directory=tmp_path); replay=SurgicalReplayCache()
    token=frames.admit(engine,engine.snapshot(),replay,verified_history=engine.history)
    collector=frames.collector(token)
    report=capture_episode(engine,[*actions,failed],frame_collector=collector)
    assert report['summary']['invalidActionsCumulative']==1 and collector.last_effective_activation is None
    assert frames.store.publications==2 and frames.store.load(token,2) is None
    for position in range(2):
        item=frames.store.load(token,position)
        assert all(not frame['effectiveActivation']['left'] for frame in item['frames'])
    assert not list(tmp_path.glob('.pending-*'))
