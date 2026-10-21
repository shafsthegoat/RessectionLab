"""Tiny generated fixtures compare legacy profile versus monitoring guard."""
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from readonly_step_monitor import ReadOnlyStepMonitor,ForbiddenPhysicsStep

@contextmanager
def legacy(base):
 def callback(frame,event,arg):
  if event=='call' and frame.f_code.co_name=='step' and isinstance(frame.f_locals.get('self'),base):
   raise ForbiddenPhysicsStep('legacy')
 sys.setprofile(callback)
 try:yield
 finally:sys.setprofile(None)

def guard(kind,base):return legacy(base) if kind=='legacy' else ReadOnlyStepMonitor(base)

def fixture(cls=SurgicalEngine):
 labels=np.zeros((12,12,12),np.uint16);labels[5,5,5]=1
 scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'fixture','material':'bone'}],{'kind':'tiny-step-guard-fixture'}, {})
 return cls(scene)

@pytest.mark.parametrize('kind',['legacy','monitor'])
def test_actual_public_step_is_blocked_before_snapshot_or_history_changes(kind):
 engine=fixture();before=engine.snapshot()
 with pytest.raises(ForbiddenPhysicsStep):
  with guard(kind,SurgicalEngine):engine.step({'slot':'left','operation':'inspect','duration':.1})
 assert engine.snapshot()==before and not engine.history

@pytest.mark.parametrize('kind',['legacy','monitor'])
def test_inherited_and_future_override_bodies_never_enter(kind):
 bodies=[]
 class Base:pass
 class Inherited(Base):
  def step(self):bodies.append('inherited')
 class Child(Inherited):pass
 with pytest.raises(ForbiddenPhysicsStep):
  with guard(kind,Base):Child().step()
 with pytest.raises(ForbiddenPhysicsStep):
  with guard(kind,Base):
   class Future(Base):
    def step(self):bodies.append('future')
   Future().step()
 assert not bodies

@pytest.mark.parametrize('kind',['legacy','monitor'])
def test_step_code_used_first_by_nonengine_stays_enabled(kind):
 bodies=[]
 class Base:pass
 def step(self):bodies.append(type(self).__name__)
 class Ordinary:pass
 with pytest.raises(ForbiddenPhysicsStep):
  with guard(kind,Base):
   step(Ordinary())
   class Future(Base):pass
   Future.step=step
   Future().step()
 assert bodies==['Ordinary']

@pytest.mark.parametrize('kind',['legacy','monitor'])
@pytest.mark.parametrize('resume',['start','next','throw','close'])
def test_generator_step_start_resume_throw_close_matches_legacy(kind,resume):
 bodies=[]
 class Signal(Exception):pass
 class Base:
  def step(self):
   bodies.append('start')
   try:
    yield 1
    bodies.append('resumed')
   except Signal:bodies.append('throwbody')
   finally:bodies.append('finally')
 generator=Base().step()
 if resume!='start':assert next(generator)==1
 before=bodies[:]
 with pytest.raises(ForbiddenPhysicsStep):
  with guard(kind,Base):
   if resume in ('start','next'):next(generator)
   elif resume=='throw':generator.throw(Signal)
   else:generator.close()
 # Both legacy and monitoring propagate guard failure through existing finally.
 assert bodies==before+(['finally'] if resume!='start' else [])
 generator.close()

@pytest.mark.parametrize('kind',['legacy','monitor'])
@pytest.mark.parametrize('resume',['start','send','throw','close'])
def test_async_step_start_resume_throw_close_matches_legacy(kind,resume):
 bodies=[]
 class Await:
  def __await__(self):yield 'waiting'
 class Signal(Exception):pass
 class Base:
  async def step(self):
   bodies.append('start')
   try:
    await Await()
    bodies.append('resumed')
   except Signal:bodies.append('throwbody')
   finally:bodies.append('finally')
 coroutine=Base().step()
 if resume!='start':assert coroutine.send(None)=='waiting'
 before=bodies[:]
 with pytest.raises(ForbiddenPhysicsStep):
  with guard(kind,Base):
   if resume in ('start','send'):coroutine.send(None)
   elif resume=='throw':coroutine.throw(Signal)
   else:coroutine.close()
 assert bodies==before+(['finally'] if resume!='start' else [])
 coroutine.close()

def test_nonstep_json_calls_are_disabled_and_step_attempts_remain_armed():
 import json
 class Base:pass
 def noise():return [i for i in range(20)]
 with ReadOnlyStepMonitor(Base) as monitor:
  noise();json.dumps(noise());first=monitor.other_callbacks
  for _ in range(100):noise();json.dumps(noise())
  monitor.check()
  assert monitor.other_callbacks-first<15
 assert not monitor.attempts

def test_disabled_monitoring_configuration_fails_closed():
 class Base:pass
 with pytest.raises(RuntimeError,match='events changed'):
  with ReadOnlyStepMonitor(Base) as monitor:
   sys.monitoring.set_events(monitor.tool,0)
   monitor.check()

def test_overwritten_callback_fails_closed():
 class Base:pass
 with pytest.raises(RuntimeError,match='callback changed'):
  with ReadOnlyStepMonitor(Base) as monitor:
   sys.monitoring.register_callback(monitor.tool,sys.monitoring.events.PY_START,lambda *args:None)
   monitor.check()


def test_new_class_without_step_attempt_still_invalidates_result():
 class Base:pass
 with pytest.raises(RuntimeError,match='inventory changed'):
  with ReadOnlyStepMonitor(Base) as monitor:
   class Future(Base):
    def step(self):pass
   monitor.check()

def test_monitoring_unavailable_fails_closed(monkeypatch):
 monkeypatch.delattr(sys,'monitoring')
 with pytest.raises(RuntimeError,match='unavailable'):ReadOnlyStepMonitor(object)

def test_occupied_tool_slots_fail_closed():
 class Base:pass
 m=sys.monitoring;owned=[]
 try:
  for tool in range(6):
   if m.get_tool(tool) is None:
    m.use_tool_id(tool,'guard-fixture-occupied');owned.append(tool)
  with pytest.raises(RuntimeError,match='No monitoring tool'):
   with ReadOnlyStepMonitor(Base):pass
 finally:
  for tool in owned:m.free_tool_id(tool)


@pytest.mark.parametrize('target',['guard','verifier'])
@pytest.mark.parametrize('optimized',['flag','environment'])
def test_optimized_python_refuses_before_admission(target,optimized):
 import os
 import subprocess
 from pathlib import Path
 folder=Path(__file__).resolve().parent
 env=os.environ.copy();env.pop('PYTHONOPTIMIZE',None)
 command=[sys.executable]
 if optimized=='flag':command.append('-O')
 else:env['PYTHONOPTIMIZE']='1'
 if target=='guard':
  command.extend(['-c',
   'import sys;sys.path.insert(0,'+repr(str(folder))+');'
   'from readonly_step_monitor import ReadOnlyStepMonitor;'
   'ReadOnlyStepMonitor(object)'])
 else:command.append(str(folder/'verify_every_cached_frame_monitoring.py'))
 result=subprocess.run(command,env=env,capture_output=True,text=True)
 assert result.returncode!=0
 assert 'RuntimeError: Optimized Python disables audit assertions' in result.stderr
 assert 'Cache-only canonical admission missed' not in result.stderr
 assert 'arguments are required' not in result.stderr
