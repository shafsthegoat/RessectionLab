"""Opt-in corridor backend for the source-driven research episode.

No live SurgicalEngine cap or larger-domain public-action admission changes.
Requests remain state-driven; only the declared absolute CUSA pose is bounded.
"""
import copy

from .surgical_closed_loop import CusaContinuumEpisode,CoupledPreparation,hold_contacts
from .surgical_continuum_corridor import corridor_partition
from .surgical_continuum_topology import SourceDomain


def prepare_corridor_domain(scene,root_domain,initial,commands,limits):
    active=[c for c in commands if c['operation']=='apply']
    if not active or any(c['slot']!=active[0]['slot'] or c['pose']!=active[0]['pose'] for c in active):
        raise ValueError('CORRIDOR_RESEARCH_ABSOLUTE_REQUEST')
    slot=active[0]['slot'];parameters=initial['parametersBySlot'][slot]
    if initial['slots'][slot]['instrument']!='cusa' or parameters['radiusMm']!=1.5 or parameters['contactGeometryRevision']!=2:
        raise ValueError('CORRIDOR_RESEARCH_TIP_GEOMETRY')
    selection,partition=corridor_partition(root_domain.geometry,hold_contacts(scene,initial['holds']),
        initial['slots'][slot]['position'],active[0]['pose']['position'],parameters['radiusMm'],limits['maxCenterDisplacementMm'])
    return SourceDomain(root_domain.geometry,partition['coreFreeDofs']),selection,partition


class _CorridorEpisode(CusaContinuumEpisode):
    def __init__(self,*args,planned_slot,planned_pose,**kwargs):
        self._planned_slot=planned_slot;self._planned_pose=copy.deepcopy(planned_pose)
        super().__init__(*args,**kwargs)

    def advance(self,command):
        if command['operation']=='apply' and (command['slot']!=self._planned_slot or command['pose']!=self._planned_pose):
            raise ValueError('CORRIDOR_RESEARCH_UNPLANNED_REQUEST')
        return super().advance(command)


def create_corridor_episode(scene,grid,root_domain,initial,alpha,times,h,measure,limits,commands,*,check=lambda:None):
    """Create a fresh full-field episode with strict retained-update refusal.

    CUSA ranking, capacity, resolved poses and masks are computed from current
    fields on every tick. Release and stationary inspection keep their normal
    semantics. Other absolute tip requests require a separately prepared plan.
    """
    check();domain,selection,partition=prepare_corridor_domain(scene,root_domain,initial,commands,limits)
    active=next(c for c in commands if c['operation']=='apply')
    backend=CoupledPreparation(domain,alpha,times,allow_fresh_exterior_updates=False,check=check)
    episode=_CorridorEpisode(scene,grid,domain,initial,alpha,times,h,measure,limits,backend=backend,check=check,
        planned_slot=active['slot'],planned_pose=active['pose'])
    return episode,selection,partition
