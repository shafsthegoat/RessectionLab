"""Audit-only no-step monitoring; never modifies an engine or its code.

Mirrors the legacy profile predicate: a Python function named 'step' whose
callee-frame self is an instance of the configured engine base. All such code,
including code loaded after activation, remains monitored. Non-step event
locations disable themselves only after their first callback. PY_THROW keeps
throw/close resumptions covered as well as PY_START/PY_RESUME. Unsupported or
changed monitoring configurations fail closed; no unguarded fallback exists.
"""
import sys
import uuid
from collections import Counter

class ForbiddenPhysicsStep(BaseException):
    pass

class ReadOnlyStepMonitor:
    def __init__(self, engine_base):
        if not __debug__:
            raise RuntimeError('Optimized Python disables audit assertions; refusing unguarded audit')
        if sys.implementation.name!='cpython' or sys.version_info[:2]!=(3,12):
            raise RuntimeError('Monitoring guard validated only for CPython3.12; refusing unguarded audit')
        self.engine_base=engine_base;self.attempts=Counter();self.other_callbacks=0
        self.inventory=None
        self.monitor=getattr(sys,'monitoring',None)
        if self.monitor is None:raise RuntimeError('sys.monitoring unavailable; refusing unguarded audit')
        self.flags=self.monitor.events.PY_START|self.monitor.events.PY_RESUME|self.monitor.events.PY_THROW
        self.callback=self._callback;self.callback_code=self._callback.__func__.__code__
        self.name='resectionlab-readonly-step-'+uuid.uuid4().hex
        self.tool=None;self.started=False
        self.api={key:getattr(self.monitor,key) for key in ('get_tool','use_tool_id','free_tool_id',
            'get_events','set_events','register_callback')}

    def _callback(self,code,offset,*extra):
        if code.co_name!='step':
            self.other_callbacks+=1
            # PY_THROW is a global-only event and cannot return DISABLE.
            return None if extra else self.monitor.DISABLE
        frame=sys._getframe(1)
        if frame.f_code is not code:raise RuntimeError('Monitoring callback callee frame differs')
        obj=frame.f_locals.get('self')
        if isinstance(obj,self.engine_base):
            self.attempts[type(obj).__module__+'.'+type(obj).__qualname__+'.step']+=1
            raise ForbiddenPhysicsStep('Read-only audit attempted a public physics step')
        # Never DISABLE step code called first on a non-engine object.

    def _inventory(self):
        # This detects code/class drift; it does NOT filter global event coverage.
        classes=set();pending=[self.engine_base]
        while pending:
            cls=pending.pop()
            if cls in classes:continue
            classes.add(cls);pending.extend(type.__subclasses__(cls))
        rows=[]
        for cls in classes:
            fn=getattr(cls,'step',None)
            rows.append((cls,fn,getattr(fn,'__code__',None)))
        return frozenset(rows)

    def __enter__(self):
        if self.started:raise RuntimeError('Monitoring guard cannot be reused')
        self.started=True
        self.inventory=self._inventory()
        if sys.getprofile() is not None or sys.gettrace() is not None:
            raise RuntimeError('Competing profile/trace; refusing altered audit instrumentation')
        for tool in (4,3,2,1,0,5):
            if self.monitor.get_tool(tool) is None:
                try:self.monitor.use_tool_id(tool,self.name)
                except ValueError:continue
                self.tool=tool;break
        if self.tool is None:raise RuntimeError('No monitoring tool ID; refusing unguarded audit')
        try:
            for event in (self.monitor.events.PY_START,self.monitor.events.PY_RESUME,self.monitor.events.PY_THROW):
                if self.monitor.register_callback(self.tool,event,self.callback) is not None:
                    raise RuntimeError('Unexpected monitoring callback on unused tool ID')
            self.monitor.set_events(self.tool,self.flags)
            self.check()
        except BaseException:
            self._close();raise
        return self

    def check(self):
        if self._inventory()!=self.inventory:raise RuntimeError('Engine step code/class inventory changed; refusing audit result')
        if self.tool is None or self.monitor is not getattr(sys,'monitoring',None):raise RuntimeError('Monitoring guard unavailable')
        if any(getattr(self.monitor,key) is not value for key,value in self.api.items()):raise RuntimeError('Monitoring API changed')
        if self.callback.__func__.__code__ is not self.callback_code:raise RuntimeError('Monitoring callback code changed')
        if self.monitor.get_tool(self.tool)!=self.name or self.monitor.get_events(self.tool)!=self.flags:
            raise RuntimeError('Monitoring ownership/events changed; refusing audit result')
        for event in (self.monitor.events.PY_START,self.monitor.events.PY_RESUME,self.monitor.events.PY_THROW):
            if self.monitor.register_callback(self.tool,event,self.callback) is not self.callback:
                raise RuntimeError('Monitoring callback changed; refusing audit result')
        if sys.getprofile() is not None or sys.gettrace() is not None:raise RuntimeError('Competing profile/trace appeared')
        if self.attempts:raise ForbiddenPhysicsStep('Public step attempt occurred; audit cannot pass')

    def _close(self):
        if self.tool is not None:
            self.monitor.set_events(self.tool,0)
            for event in (self.monitor.events.PY_START,self.monitor.events.PY_RESUME,self.monitor.events.PY_THROW):
                self.monitor.register_callback(self.tool,event,None)
            self.monitor.free_tool_id(self.tool);self.tool=None

    def __exit__(self,kind,value,traceback):
        try:
            if kind is None:self.check()
        finally:self._close()
