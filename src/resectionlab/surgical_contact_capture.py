"""Transactional geometry contexts at selected native continuum fixed steps.

This observer neither changes canonical state nor advances a larger domain.
Intermediate fields are conditional phases of an accepted native microtick,
not separately accepted public states.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from unittest.mock import patch

import numpy as np

from . import surgical_continuum as continuum
from .surgical_continuum_capture import MechanicsCaptureEngine,MechanicsCaptureFailure,capture_mechanics_inputs

_current=ContextVar('native_contact_geometry_capture',default=None)


class ContactCaptureEngine(MechanicsCaptureEngine):
    def __init__(self,*args,selected_phases=(),**kwargs):
        super().__init__(*args,**kwargs)
        phases=tuple(selected_phases)
        if any(len(p)!=2 or any(type(v) is not int or v<0 for v in p) for p in phases):
            raise ValueError('CONTACT_CAPTURE_PHASES')
        self._selected_contact_phases=frozenset(phases);self._contact_frames=[];self._pending_contact_frames=None

    def clone(self):
        other=super().clone();other._contact_frames=self._contact_frames[:];other._pending_contact_frames=None
        return other

    def _substep(self,key,op,p,q,active,dt):
        self._pending_contact_frames=None
        try:events=super()._substep(key,op,p,q,active,dt)
        except BaseException:
            self._pending_contact_frames=None;raise
        if self._pending_contact_frames is not None:
            tick_index=len(self._mechanics_ticks)-1;mechanics=self._mechanics_ticks[-1]
            context={'slots':copy.deepcopy(self.state['slots']),
                'parametersBySlot':{slot:copy.deepcopy(self.parameters(value)) for slot,value in self.state['slots'].items()},
                'holds':copy.deepcopy(self.state['holds']),'removedSourceIds':self.state['removed'][:],
                'cutFaces':copy.deepcopy(self.state.get('cutEdges',[])),
                'detachedFragments':copy.deepcopy(self.state.get('detachedFragments',[])),
                'toolMechanicsRevision':self.state.get('toolMechanicsRevision'),
                'guardReport':self.guard_report(),'requestedPose':{'position':np.asarray(p).tolist(),'quaternion':np.asarray(q).tolist()},
                'protectedSourceIds':sorted(self._protected_ids),
                'actionIndex':len(self.actions),'operation':op,'slot':key,'effectiveActive':bool(active),
                'mechanics':copy.deepcopy(mechanics)}
            for frame in self._pending_contact_frames:
                if frame['nativeTickIndex']!=tick_index:raise MechanicsCaptureFailure('Geometry/microtick join changed')
                end=frame['withinTick']+1==mechanics['nativeFixedSteps']
                if end:
                    canonical=self.state[continuum.PROPERTY]['cornerDisplacementMm']
                    packed=np.asarray([canonical[str(cell)] for cell in frame['sourceCells']])
                    if not np.array_equal(packed.view(np.uint64),frame['cornerDisplacementMm'].view(np.uint64)):
                        raise MechanicsCaptureFailure('Final native geometry field differs from canonical packed state')
                self._contact_frames.append({**frame,**context,'atCommittedTickEnd':end,
                    'scope':'Conditional BE field at resolved native microtick poses; no larger-domain action admission'})
        self._pending_contact_frames=None
        return events

    def contact_frames(self):
        return copy.deepcopy(self._contact_frames)


@contextmanager
def capture_contact_geometry():
    """Observe real native solves and publish only fully committed microticks."""
    system_original=continuum._system;memory_original=continuum.update_memory
    def system(engine,*args,**kwargs):
        value=system_original(engine,*args,**kwargs);frame=_current.get()
        if frame is not None and frame['engine'] is engine:frame['mesh']=value['mesh']
        return value
    def memory(memories,displacement,times,h):
        result=memory_original(memories,displacement,times,h);frame=_current.get()
        if frame is not None:
            within=frame['count'];frame['count']+=1
            if within in frame['wanted']:
                mesh=frame.get('mesh')
                if mesh is None or np.asarray(displacement).shape!=(mesh.nodes.size,):
                    raise MechanicsCaptureFailure('No source mesh for native fixed-step field')
                field=np.asarray(displacement,float).reshape((-1,3))[mesh.element_nodes].copy()
                if not np.isfinite(field).all():raise MechanicsCaptureFailure('Invalid native geometry field')
                field.flags.writeable=False
                frame['fields'].append({'nativeTickIndex':frame['tick'],'withinTick':within,
                    'sourceCells':list(mesh.cells),'cornerDisplacementMm':field,'fixedStepSeconds':float(h)})
        return result
    with capture_mechanics_inputs():
        tick_original=continuum.tick
        def tick(engine,dt,diagnostic=None):
            if not isinstance(engine,ContactCaptureEngine):return tick_original(engine,dt,diagnostic)
            index=len(engine._mechanics_ticks)
            wanted={within for tick_index,within in engine._selected_contact_phases if tick_index==index}
            if not wanted:return tick_original(engine,dt,diagnostic)
            frame={'engine':engine,'tick':index,'wanted':wanted,'count':0,'fields':[]}
            token=_current.set(frame)
            try:
                result=tick_original(engine,dt,diagnostic)
                pending=engine._pending_mechanics
                if (pending is None or frame['count']!=pending[1]['nativeFixedSteps']
                        or {row['withinTick'] for row in frame['fields']}!=wanted):
                    raise MechanicsCaptureFailure('Selected native fixed-step geometry incomplete')
                engine._pending_contact_frames=frame['fields']
                return result
            finally:_current.reset(token)
        with patch.object(continuum,'_system',system),patch.object(continuum,'update_memory',memory),patch.object(continuum,'tick',tick):
            yield
