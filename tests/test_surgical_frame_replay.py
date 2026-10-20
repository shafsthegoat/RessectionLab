import copy
import numpy as np
import pytest
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_frame_replay import SurgicalFrameCache


def episode():
    labels=np.zeros((24,24,24),np.uint16)
    labels[10:12,10:12,10:13]=1
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'target','material':'tumor','target':True}],{}, {})
    engine=SurgicalEngine(scene)
    for action in [
        {'slot':'left','operation':'exchange','instrument':'cusa','duration':.2},
        {'slot':'left','operation':'place','pose':{'position':[10.,10.,5.],'quaternion':[0.,0.,0.,1.]},'parameters':{'radiusMm':.6},'duration':.2},
        {'slot':'left','operation':'apply','pose':{'position':[10.,10.,12.],'quaternion':[0.,0.,0.,1.]},'duration':1.2},
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
    boundary,_=frames.boundary(original,3)
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
