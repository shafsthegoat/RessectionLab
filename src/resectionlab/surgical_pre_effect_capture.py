"""Transactional native pre-effect CUSA observations; no new model admission."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from unittest.mock import patch

import numpy as np

from . import surgical_cusa_v3 as cusa,surgical_continuum as continuum
from .surgical_continuum_capture import MechanicsCaptureEngine,MechanicsCaptureFailure,capture_mechanics_inputs

_current=ContextVar('native_pre_effect_capture',default=None)


def _before(engine,key,op,p,q,active,dt):
    packed=engine.state.get(continuum.PROPERTY,{}).get('cornerDisplacementMm',{})
    cells=sorted(map(int,packed))
    if not cells:raise MechanicsCaptureFailure('Pre-effect native corner field unavailable')
    u=np.asarray([packed[str(cell)] for cell in cells],float)
    centers=np.asarray([engine.state['deformation'][str(cell)] for cell in cells],float)
    if (u.shape!=(len(cells),8,3) or not np.isfinite(u).all()
            or not np.array_equal(u.mean(axis=1).view(np.uint64),centers.view(np.uint64))):
        raise MechanicsCaptureFailure('Pre-effect native center/corner field differs')
    return {'geometryPhase':'before-physical-substep','sourceCells':cells,'cornerDisplacementMm':u,
        'slots':copy.deepcopy(engine.state['slots']),
        'parametersBySlot':{slot:copy.deepcopy(engine.parameters(value)) for slot,value in engine.state['slots'].items()},
        'holds':copy.deepcopy(engine.state['holds']),'removedSourceIds':engine.state['removed'][:],
        'cutFaces':copy.deepcopy(engine.state.get('cutEdges',[])),
        'detachedFragments':copy.deepcopy(engine.state.get('detachedFragments',[])),
        'toolMechanicsRevision':engine.state.get('toolMechanicsRevision'),'guardReport':engine.guard_report(),
        'protectedSourceIds':sorted(engine._protected_ids),'cusaBefore':copy.deepcopy(engine.state.get('cusaV3')),
        'elapsedSecondsBefore':engine.state['elapsedSeconds'],'nativeRemainderBeforeSeconds':engine.state['deformationModel']['remainderSeconds'],
        'requestedPose':{'position':np.asarray(p).tolist(),'quaternion':np.asarray(q).tolist()},
        'actionIndex':len(engine.actions),'operation':op,'slot':key,'effectiveActive':bool(active),'durationSeconds':float(dt)}


class PreEffectCaptureEngine(MechanicsCaptureEngine):
    def __init__(self,*args,selected_ticks=(),**kwargs):
        super().__init__(*args,**kwargs);ticks=tuple(selected_ticks)
        if any(type(tick) is not int or tick<0 for tick in ticks):raise ValueError('PRE_EFFECT_CAPTURE_TICKS')
        self._selected_pre_effect_ticks=frozenset(ticks);self._pre_effect_frames=[]

    def clone(self):
        other=super().clone();other._pre_effect_frames=self._pre_effect_frames[:];return other

    def _substep(self,key,op,p,q,active,dt):
        index=len(self._mechanics_ticks)
        if index not in self._selected_pre_effect_ticks:return super()._substep(key,op,p,q,active,dt)
        before=_before(self,key,op,p,q,active,dt)
        frame={'engine':self,'phase':None,'queries':[],'limiters':[],'applications':[]}
        token=_current.set(frame)
        try:events=super()._substep(key,op,p,q,active,dt)
        finally:_current.reset(token)
        if len(self._mechanics_ticks)!=index+1:raise MechanicsCaptureFailure('Pre-effect native microtick missing')
        self._pre_effect_frames.append({**before,'nativeTickIndex':index,'mechanics':copy.deepcopy(self._mechanics_ticks[-1]),
            'limiterCalls':frame['limiters'],'applicationCalls':frame['applications'],
            'resolvedSlotsAfter':copy.deepcopy(self.state['slots']),'removedSourceIdsAfter':self.state['removed'][:],
            'cusaAfter':copy.deepcopy(self.state.get('cusaV3')),
            'scope':'Native pre-effect input/effect receipt of a fully committed microtick; no larger-domain action admission'})
        return events

    def pre_effect_frames(self):return copy.deepcopy(self._pre_effect_frames)


@contextmanager
def capture_pre_effects():
    limiter_original=cusa.contact_limited_pose;apply_original=cusa.apply;order_original=cusa._ordered_contacts
    def ordered(engine,cells,position):
        result=order_original(engine,cells,position);frame=_current.get()
        if frame is not None and frame['engine'] is engine:
            if frame['phase'] not in ('limiter','apply') or not isinstance(cells,(list,tuple,np.ndarray)):
                raise MechanicsCaptureFailure('Unbound native CUSA order query')
            frame['queries'].append({'positionMm':np.asarray(position).tolist(),
                'candidateSourceIds':list(map(int,cells)),'orderedSourceIds':list(map(int,result))})
        return result
    def limiter(engine,start,requested,radius=cusa.TIP_RADIUS_MM):
        frame=_current.get()
        if frame is None or frame['engine'] is not engine:return limiter_original(engine,start,requested,radius)
        if frame['phase'] is not None:raise MechanicsCaptureFailure('Nested native CUSA limiter')
        frame['phase']='limiter';begin=len(frame['queries'])
        try:p,ids=limiter_original(engine,start,requested,radius)
        finally:frame['phase']=None
        frame['limiters'].append({'startMm':np.asarray(start).tolist(),'requestedMm':np.asarray(requested).tolist(),
            'radiusMm':float(radius),'resolvedMm':p.tolist(),'retainedSourceIds':list(map(int,ids)),
            'queries':copy.deepcopy(frame['queries'][begin:])})
        return p,ids
    def apply(engine,cells,dt,params,*,position=None,active=True):
        frame=_current.get()
        if frame is None or frame['engine'] is not engine:return apply_original(engine,cells,dt,params,position=position,active=active)
        if frame['phase'] is not None:raise MechanicsCaptureFailure('Nested native CUSA apply')
        frame['phase']='apply';begin=len(frame['queries'])
        before=copy.deepcopy(engine.state.get('cusaV3'));removed=engine.state['removed'][:]
        try:events=apply_original(engine,cells,dt,params,position=position,active=active)
        finally:frame['phase']=None
        frame['applications'].append({'positionMm':np.asarray(position).tolist(),'durationSeconds':float(dt),
            'parameters':copy.deepcopy(params),'effectiveActive':active,'candidateSourceIds':list(map(int,cells)),
            'orderedQueries':copy.deepcopy(frame['queries'][begin:]),'cusaBefore':before,
            'removedSourceIdsBefore':removed,'cusaAfter':copy.deepcopy(engine.state['cusaV3']),
            'removedSourceIdsAfter':engine.state['removed'][:],'events':copy.deepcopy(events)})
        return events
    with capture_mechanics_inputs(),patch.object(cusa,'_ordered_contacts',ordered),patch.object(cusa,'contact_limited_pose',limiter),patch.object(cusa,'apply',apply):
        yield
