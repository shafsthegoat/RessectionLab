"""Tiny monitoring API probe only; no patient input or engine replay."""
import sys,json
m=sys.monitoring;tool=4;assert m.get_tool(tool) is None
m.use_tool_id(tool,'resectionlab-monitoring-frame-probe')
events=[];bodies=[]
class Root:pass
class Child(Root):
 def step(self):bodies.append('entered')
def step(self):bodies.append('standalone')
def generator():
 yield 1
 yield 2
class Blocked(BaseException):pass
def callback(code,offset):
 frame=sys._getframe(1)
 events.append((code.co_name,frame.f_code.co_name,offset, type(frame.f_locals.get('self')).__name__))
 if code.co_name!='step':return m.DISABLE
 if isinstance(frame.f_locals.get('self'),Root):raise Blocked('before body')
m.register_callback(tool,m.events.PY_START,callback);m.register_callback(tool,m.events.PY_RESUME,callback)
m.set_events(tool,m.events.PY_START|m.events.PY_RESUME)
try:
 for _ in range(2):
  try:Child().step()
  except Blocked:pass
 step(None)
 for _ in generator():pass
 for _ in generator():pass
finally:
 m.set_events(tool,0)
 m.register_callback(tool,m.events.PY_START,None);m.register_callback(tool,m.events.PY_RESUME,None);m.free_tool_id(tool)
assert bodies==['standalone'],bodies
print(json.dumps({'bodies':bodies,'events':events}))
