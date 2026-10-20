"""Cheap engineering fixtures for persistent real-substep playback; no patient data."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_observed_cache import (observed_implementation_fingerprint,
    ImplementationChanged)
from resectionlab.surgical_telemetry import capture_episode


def episode():
    scene=SurgicalScene(np.zeros((8,8,8),np.uint16),np.eye(4),[],{'kind':'observed-cache-engineering'}, {})
    engine=SurgicalEngine(scene)
    start=engine.state['slots']['left']['position']
    actions=[{'slot':'left','operation':'move','pose':{'position':[start[0]+1,*start[1:]],'quaternion':[0,0,0,1]},'duration':.6},
             {'slot':'right','operation':'inspect','duration':.2}]
    for action in actions: assert engine.step(action)['status']=='accepted'
    return engine


def admitted(engine, directory):
    cache=SurgicalFrameCache(directory=directory); replay=SurgicalReplayCache()
    token=cache.admit(engine,engine.snapshot(),replay,verified_history=engine.history)
    return cache,replay,token


def seek(cache,replay,token,engine,position=0,index=0):
    return cache.seek(engine,token.snapshot,position,index,replay,
        verified_history=token.history,admission=token,display_only=True)


def counted(fn):
    steps=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is SurgicalEngine.step.__code__: steps.append(1)
    sys.setprofile(profile)
    try: result=fn()
    finally: sys.setprofile(None)
    return result,len(steps)


def test_telemetry_warms_actual_frames_once_and_fresh_cache_hits_without_physics(tmp_path):
    engine=episode(); saved=copy.deepcopy(engine.snapshot())
    cache,replay,token=admitted(engine,tmp_path)
    report=capture_episode(engine,frame_collector=cache.collector(token))
    assert report['expectedStateVerified'] and cache.store.publications==2
    fresh,replay2,token2=admitted(engine,tmp_path)
    (view,history,info),steps=counted(lambda:seek(fresh,replay2,token2,engine))
    assert steps==0 and info['frameCount']>=4 and info['frameStateId']==view.state_id
    assert info['frames'][0]['kind']=='action_start'
    assert info['frames'][-1]['kind']=='action_boundary'
    assert history is token2.history
    last,_,last_info=seek(fresh,replay2,token2,engine,index=info['frameCount']-1)
    assert last.state_id==engine.history[0]['stateAfter'] and last_info['isActionBoundary']
    with pytest.raises(TypeError): last.actions[0]['duration']=999
    script='''import runpy, sys
f=runpy.run_path('tests/test_surgical_observed_cache.py')
engine=f['episode']()
cache,replay,token=f['admitted'](engine,sys.argv[1])
(_,_,info),steps=f['counted'](lambda:f['seek'](cache,replay,token,engine))
assert steps==0 and cache.store.publications==0 and info['frameCount']>=4
print('fresh process exact hit')
'''
    process=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    boundary,_=fresh.boundary(token2.snapshot,1,verified_history=token2.history,admission=token2)
    assert isinstance(boundary.actions,list) and isinstance(boundary.history,list)
    boundary.actions[0]['duration']=9
    assert token2.snapshot['actions'][0]['duration']==.6 and engine.snapshot()==saved
    assert not list(tmp_path.glob('.pending-*'))
    # Content-addressed stable topology fields are shared across observations.
    item=fresh.store.load(token2,0)
    assert len({f['fields']['fields']['slots']['file'] for f in item['frames']})>1
    stable=[f['fields']['fields']['removed']['file'] for f in item['frames']]
    assert len(set(stable))==1


def test_fingerprint_is_stable_in_a_fresh_python_process():
    current=observed_implementation_fingerprint()
    command='from resectionlab.surgical_observed_cache import observed_implementation_fingerprint as f; print(f())'
    result=subprocess.run([sys.executable,'-c',command],check=True,capture_output=True,text=True)
    assert current is not None and result.stdout.strip()==current


def test_signed_manifest_or_field_tamper_falls_back_to_observed_public_step(tmp_path):
    engine=episode(); cache,replay,token=admitted(engine,tmp_path)
    capture_episode(engine,frame_collector=cache.collector(token))
    entry=cache.store.entry(token)
    (entry/'action-0.mac').write_bytes(b'0'*64)
    (_,_,info),steps=counted(lambda:seek(cache,replay,token,engine))
    assert steps>0 and info['readOnly']
    item=cache.store.load(token,0)
    field=item['frames'][0]['fields']['fields']['elapsedSeconds']['file']
    (entry/field).write_bytes(b'corrupt')
    (_,_,info),steps=counted(lambda:seek(cache,replay,token,engine))
    assert steps>0 and info['frameStateId']==token.history[0]['stateBefore']


def test_token_refuses_changed_source_or_mutable_snapshot(tmp_path):
    engine=episode(); cache,replay,token=admitted(engine,tmp_path)
    with pytest.raises(TypeError): token.snapshot['actions'][0]['duration']=1
    with pytest.raises(ValueError,match='exact private'):
        cache.seek(engine,engine.snapshot(),0,0,replay,verified_history=token.history,admission=token)
    other=episode()
    with pytest.raises(ValueError,match='exact private'): seek(cache,replay,token,other)


def test_rejected_and_wrong_receipt_never_publish(tmp_path):
    engine=episode(); cache,replay,token=admitted(engine,tmp_path)
    baseline=engine.fresh(); collector=cache.collector(token)
    collector.begin(baseline,{'slot':'missing','operation':'inspect'})
    collector.end(baseline,{}, {'status':'rejected'})
    assert cache.store.publications==0 and not list(tmp_path.glob('.pending-*'))
    action=copy.deepcopy(engine.actions[0]); collector.begin(baseline,action)
    outcome=baseline.step(action); outcome['reason']='different receipt'
    with pytest.raises(ValueError,match='outcome differs'): collector.end(baseline,action,outcome)
    assert cache.store.publications==0 and not list(tmp_path.glob('.pending-*'))


def test_loaded_producer_defaults_and_constants_invalidate_cache_identity(tmp_path,monkeypatch):
    import resectionlab.surgical_telemetry as telemetry
    engine=episode(); cache,replay,token=admitted(engine,tmp_path)
    fingerprint=token.fingerprint
    changed=dict(telemetry.capture_episode.__kwdefaults__); changed['interval_seconds']=.4
    monkeypatch.setattr(telemetry.capture_episode,'__kwdefaults__',changed)
    assert observed_implementation_fingerprint()!=fingerprint
    fresh=cache.admit(engine,engine.snapshot(),replay)
    assert fresh.fingerprint!=fingerprint
    monkeypatch.undo()
    monkeypatch.setattr(telemetry._ObservedEngine._sample,'__defaults__',('initial',None,False,None,.1))
    assert observed_implementation_fingerprint()!=fingerprint
    with pytest.raises(ImplementationChanged): token.check()
    monkeypatch.undo()
    monkeypatch.setattr(telemetry,'LABELS',(*telemetry.LABELS,'changed-producer'))
    assert observed_implementation_fingerprint()!=fingerprint
    with pytest.raises(ImplementationChanged): token.check()


def test_budget_unavailable_is_cache_miss_and_fallback_remains_exact(tmp_path):
    engine=episode(); cache=SurgicalFrameCache(directory=tmp_path,max_disk_bytes=1024)
    replay=SurgicalReplayCache(); token=cache.admit(engine,engine.snapshot(),replay)
    capture_episode(engine,frame_collector=cache.collector(token))
    assert cache.store.publications==0 and cache.store.owner.statistics['diskBytes']<=1024
    view,_,info=seek(cache,replay,token,engine)
    assert info['readOnly'] and view.state_id==token.history[0]['stateBefore']
