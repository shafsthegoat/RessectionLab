"""Committed native mechanics inputs for explicitly offline replay studies.

This observer adds no canonical state. Rejected candidates and post-physics
failed microsteps cannot publish input records. Its inputs describe the source
native model; they do not establish a larger domain's tool/collision admission.
"""
from contextlib import contextmanager
import copy
from unittest.mock import patch

import numpy as np

from .surgical_engine import SurgicalEngine
from . import surgical_continuum as continuum
from . import surgical_deformation_v2 as brain


class MechanicsCaptureFailure(RuntimeError):
    pass


class MechanicsCaptureEngine(SurgicalEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._mechanics_ticks=[];self._mechanics_topologies={};self._pending_mechanics=None

    def clone(self):
        other=super().clone()
        other._mechanics_ticks=self._mechanics_ticks[:]
        other._mechanics_topologies=self._mechanics_topologies.copy()
        other._pending_mechanics=None
        return other

    def _substep(self,key,op,p,q,active,dt):
        self._pending_mechanics=None
        try:
            events=super()._substep(key,op,p,q,active,dt)
        except BaseException:
            self._pending_mechanics=None
            raise
        if self._pending_mechanics is not None:
            topology,payload=self._pending_mechanics
            if topology not in self._mechanics_topologies:
                self._mechanics_topologies[topology]=len(self._mechanics_topologies)
            self._mechanics_ticks.append({**payload,'topologyId':self._mechanics_topologies[topology],
                'actionIndex':len(self.actions),'slot':key,'operation':op})
        self._pending_mechanics=None
        return events

    def mechanics_inputs(self):
        topologies=[None]*len(self._mechanics_topologies)
        for (removed,cuts,rigid,active,anchors),index in self._mechanics_topologies.items():
            topologies[index]={'removedSourceIds':list(removed),'cutFaces':[list(e) for e in cuts],
                'rigidSourceIds':list(rigid),'nativeActiveSourceIds':list(active),'nativeAnchorSourceIds':list(anchors)}
        return {'schema':'resectionlab.committed-native-mechanics-inputs.v1',
                'topologies':topologies,'ticks':copy.deepcopy(self._mechanics_ticks),
                'scope':'Committed source-native inputs only; no larger-domain public-action/collision admission'}


@contextmanager
def capture_mechanics_inputs():
    original=continuum.tick
    def observed(engine,dt,diagnostic=None):
        is_capture=isinstance(engine,MechanicsCaptureEngine)
        before_remainder=(float(engine.state.get('deformationModel',{}).get('remainderSeconds',0.)) if is_capture else None)
        result=original(engine,dt,diagnostic)
        if not is_capture:return result
        try:
            if engine._pending_mechanics is not None:raise MechanicsCaptureFailure('Multiple continuum ticks in one microstep')
            if 'patientPosition' in engine.state:raise MechanicsCaptureFailure('Body-loaded capture needs explicit nodal loads')
            model=engine.state['deformationModel'];active=set(model['active']);contacts=[]
            h=float(engine.scene.properties['tissuePhysicsV2']['brain']['fixedStepSeconds'])
            p=engine.scene.properties[continuum.PROPERTY]
            for key,hold in sorted(engine.state['holds'].items()):
                if 'target' not in hold:continue
                cells=sorted(set(hold['cells'])&active)
                for cell in cells:
                    target=np.subtract(hold['target'],hold['origin'])+np.asarray(hold.get('baseDisplacements',{}).get(str(cell),[0.,0.,0.]))
                    if target.shape!=(3,) or not np.isfinite(target).all():raise MechanicsCaptureFailure('Invalid captured contact target')
                    contacts.append({'slot':key,'sourceCell':int(cell),'stiffnessNPerMm':float(p['handleStiffnessNPerMm']/len(cells)),
                                     'targetDisplacementMm':target.tolist()})
            topology=(tuple(map(int,engine.state['removed'])),tuple(sorted(brain._blocked(engine))),
                      tuple(sorted(map(int,brain.rigid_cells(engine)))),tuple(map(int,model['active'])),tuple(map(int,model['anchors'])))
            steps=int(np.floor((before_remainder+dt+1e-12)/h)) if active else 0
            engine._pending_mechanics=(topology,{'timeSeconds':float(engine.state['elapsedSeconds']),
                'durationSeconds':float(dt),'fixedStepSeconds':h,'nativeFixedSteps':steps,
                'nativeRemainderBeforeSeconds':before_remainder,'nativeRemainderAfterSeconds':float(model['remainderSeconds']),
                'nativeContinuumActive':bool(active),'contacts':contacts,'bodyForce':'zero'})
        except MechanicsCaptureFailure:raise
        except Exception as error:raise MechanicsCaptureFailure('Cannot capture declared mechanics inputs') from error
        return result
    with patch.object(continuum,'tick',observed):yield
