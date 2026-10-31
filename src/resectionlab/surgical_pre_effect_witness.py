"""One-step pre-effect witnesses conditioned on captured native state.

Each counterfactual resets to its own native ledger. Predicted state is never
carried to another event; no larger-domain action or episode is admitted.
"""
import copy
from fractions import Fraction
import math
from types import SimpleNamespace

import numpy as np

from . import surgical_cusa_v3 as cusa
from .surgical_continuum_program import field_hash
from .surgical_motion import swept_envelopes,pair_conflict,envelopes
from .surgical_source_contacts import SourceContactGeometry
from .surgical_tools import TOOLS


def pre_event_fields(current,old,payload,row):
    """Recover the original old-node field, including removed-only nodes."""
    free=np.flatnonzero(current['freeDofs']);n=len(current['nodes'])
    if not np.array_equal(payload['freeIndices'],free):raise ValueError('PRE_EFFECT_FIELD_SUPPORT')
    before=np.zeros((n,3));z=np.zeros((len(payload['beforeMemories']),n,3))
    if payload['beforeDisplacement'].shape!=(len(free),) or payload['beforeMemories'].shape!=(len(z),len(free)):
        raise ValueError('PRE_EFFECT_FIELD_SUPPORT')
    before.ravel()[free]=payload['beforeDisplacement'];z.reshape((len(z),-1))[:,free]=payload['beforeMemories']
    if row['event'] is not None:
        lookup={int(cell):i for i,cell in enumerate(old['cells'])}
        try:rows=np.asarray([lookup[int(c)] for c in current['cells']])
        except KeyError as error:raise ValueError('PRE_EFFECT_RETAINED_OWNER') from error
        gone=sorted(set(map(int,old['cells']))-set(map(int,current['cells'])))
        if gone!=row['event']['removedSourceIds'] or row['withinTick']!=0:raise ValueError('PRE_EFFECT_EVENT_JOIN')
        oldnodes=old['elementNodes'][rows].ravel();newnodes=current['elementNodes'].ravel()
        u=np.full_like(old['nodes'],np.nan,dtype=float);memory=np.full((len(z),len(u),3),np.nan)
        u[oldnodes]=before[newnodes];memory[:,oldnodes]=z[:,newnodes]
        orphan=np.setdiff1d(np.arange(len(u)),np.unique(oldnodes))
        if (not np.array_equal(orphan,payload['orphanOldNodeIndices'])
                or payload['orphanOldDisplacement'].shape!=(len(orphan),3)
                or payload['orphanOldMemories'].shape!=(len(z),len(orphan),3)):
            raise ValueError('PRE_EFFECT_ORPHAN_SUPPORT')
        u[orphan]=payload['orphanOldDisplacement'];memory[:,orphan]=payload['orphanOldMemories']
        if (not np.array_equal(u[oldnodes].view(np.uint64),before[newnodes].view(np.uint64))
                or not np.array_equal(memory[:,oldnodes].view(np.uint64),z[:,newnodes].view(np.uint64))):
            raise ValueError('PRE_EFFECT_SOURCE_CONTINUITY')
    else:
        if not np.array_equal(current['cells'],old['cells']) or len(payload['orphanOldNodeIndices']):raise ValueError('PRE_EFFECT_UNEXPECTED_TOPOLOGY')
        u,memory=before,z
    if (not np.isfinite(u).all() or not np.isfinite(memory).all()
            or field_hash(u.ravel(),memory.reshape((len(memory),-1)))!=row['previousArrayHash']):
        raise ValueError('PRE_EFFECT_PREVIOUS_FIELD_HASH')
    return u[old['elementNodes']],memory[:,old['elementNodes']]


def capacity_prediction(before,removed,ordered,volume,profile,parameters,dt,active):
    """Independent one-step rational allocation of the existing CUSA surrogate."""
    cusa.validate(profile);power,duty=cusa.validate_parameters(parameters)
    if (type(active) is not bool or type(dt) not in (float,int) or not math.isfinite(dt) or dt<0
            or not math.isfinite(volume) or volume<=0):raise ValueError('PRE_EFFECT_CAPACITY_INPUT')
    state=copy.deepcopy(before);removed=set(removed)
    if state['schema']!=cusa.SCHEMA or set(state['progressMm3'])!=set(state['progressExact']) or set(state['progressMm3'])!=set(state['contactSeconds']):
        raise ValueError('PRE_EFFECT_CAPACITY_LEDGER')
    for key,value in state['progressExact'].items():
        if (not isinstance(value,list) or len(value)!=2 or any(type(v) is not int for v in value) or value[1]<=0):
            raise ValueError('PRE_EFFECT_CAPACITY_LEDGER')
        progress=Fraction(*value)
        if not 0<=progress<Fraction.from_float(float(volume)) or float(progress)!=state['progressMm3'][key]:raise ValueError('PRE_EFFECT_CAPACITY_LEDGER')
    stale=[key for key in state['progressMm3'] if int(key) in removed]
    discarded=math.fsum(state['progressMm3'].pop(key) for key in stale)
    for key in stale:state['progressExact'].pop(key);state['contactSeconds'].pop(key)
    state['discardedCapacityMm3']+=discarded
    budget=profile['capacityMm3PerSecond']*power*duty*dt if active else 0.
    if not math.isfinite(budget):raise ValueError('PRE_EFFECT_CAPACITY_OVERFLOW')
    if len(set(ordered))!=len(ordered) or set(ordered)&removed:raise ValueError('PRE_EFFECT_CAPACITY_CONTACTS')
    remaining=Fraction.from_float(budget);unit=Fraction.from_float(float(volume));new=[];spent=[]
    for cell in ordered if budget>0 else ():
        key=str(cell);old=Fraction(*state['progressExact'][key]) if key in state['progressExact'] else Fraction(0)
        required=unit-old;allocated=min(required,remaining)
        if allocated<=0:continue
        state['contactSeconds'][key]=state['contactSeconds'].get(key,0.)+dt
        remaining-=allocated;spent.append(float(allocated))
        if allocated>=required:
            new.append(cell)
            for name in ('progressMm3','progressExact','contactSeconds'):state[name].pop(key,None)
        else:
            value=old+allocated;state['progressMm3'][key]=float(value);state['progressExact'][key]=[value.numerator,value.denominator]
    aspirated=len(new)*volume
    state['inputCapacityMm3']+=budget;state['aspiratedSolidMm3']+=aspirated;state['unusedCapacityMm3']+=float(remaining)
    state['effectiveActivationSeconds']+=power*duty*dt if active else 0.
    events=[{'kind':'cusa_contact','inputCapacityMm3':budget,'allocatedCapacityMm3':math.fsum(spent),
        'unusedCapacityMm3':float(remaining),'aspiratedSolidMm3':aspirated,'discardedCapacityMm3':discarded,
        'removedCells':new,'pendingCells':len(state['progressMm3'])}] if budget or discarded else []
    return {'cusaAfter':state,'removedSourceIdsAfter':sorted(removed|set(new)),'events':events}


def require_pre_effect_context(frame):
    if (frame['geometryPhase']!='before-physical-substep' or frame['operation']!='apply' or not frame['effectiveActive']
            or frame['slots'][frame['slot']]['instrument']!='cusa' or frame['toolMechanicsRevision']!=2
            or len(frame['limiterCalls'])!=1 or len(frame['applicationCalls'])!=1
            or frame['cutFaces'] or frame['detachedFragments']
            or frame['protectedSourceIds'] or frame['guardReport']['configured']
            or any(slot['active'] for key,slot in frame['slots'].items() if key!=frame['slot'])):
        raise ValueError('PRE_EFFECT_CONTEXT_UNADMITTED')
    app=frame['applicationCalls'][0];limiter=frame['limiterCalls'][0];key=frame['slot'];mechanics=frame['mechanics']
    if (app['cusaBefore']!=frame['cusaBefore'] or app['removedSourceIdsBefore']!=frame['removedSourceIds']
            or app['durationSeconds']!=frame['durationSeconds'] or app['parameters']!=frame['parametersBySlot'][key]
            or not app['effectiveActive'] or app['positionMm']!=limiter['resolvedMm']
            or limiter['startMm']!=frame['slots'][key]['position'] or limiter['requestedMm']!=frame['requestedPose']['position']
            or frame['resolvedSlotsAfter'][key]['position']!=limiter['resolvedMm']
            or app['cusaAfter']!=frame['cusaAfter'] or app['removedSourceIdsAfter']!=frame['removedSourceIdsAfter']
            or any(frame[k]!=mechanics[k] for k in ('actionIndex','operation','slot'))
            or mechanics['durationSeconds']!=frame['durationSeconds']
            or mechanics['nativeRemainderBeforeSeconds']!=frame['nativeRemainderBeforeSeconds']):
        raise ValueError('PRE_EFFECT_NATIVE_PHASE_JOIN')


def limiter_prediction(geometry,frame):
    key=frame['slot'];calls=[]
    def touched(a,b,radius):
        ids=geometry.occupancy.material_ids(geometry.grid_cover(a,b,radius))
        calls.append({'positionMm':np.asarray(a).tolist(),'candidateSourceIds':ids})
        return ids
    view=SimpleNamespace(scene=geometry.scene,by_label=geometry.by_label,
        state={'holds':copy.deepcopy(frame['holds']),'removed':sorted(geometry.removed),
            'toolMechanicsRevision':frame['toolMechanicsRevision'],
            'deformation':{str(int(c)):v.tolist() for c,v in zip(geometry.field.cells,geometry.u.mean(axis=1))}},touched=touched)
    start=frame['slots'][key]['position'];requested=frame['requestedPose']['position'];radius=frame['parametersBySlot'][key]['radiusMm']
    p,ids=cusa.contact_limited_pose(view,start,requested,radius)
    if len(calls)%2 or any(calls[i]!=calls[i+1] for i in range(0,len(calls),2)):raise ValueError('PRE_EFFECT_LIMITER_QUERY_STRUCTURE')
    queries=[{**calls[i],'orderedSourceIds':geometry.cusa_order(calls[i]['positionMm'])} for i in range(0,len(calls),2)]
    return {'startMm':start,'requestedMm':requested,'radiusMm':radius,'resolvedMm':p.tolist(),
        'retainedSourceIds':list(map(int,ids)),'queries':queries}


def pre_effect_prediction(scene,grid,frame,cells,u,*,check=lambda:None):
    require_pre_effect_context(frame)
    geometry=SourceContactGeometry(scene,grid,frame['removedSourceIds'],cells,u,
        holds=frame['holds'],detached_fragments=frame['detachedFragments'],check=check)
    limiter=limiter_prediction(geometry,frame);key=frame['slot'];params=frame['parametersBySlot'][key]
    p=limiter['resolvedMm'];q=frame['requestedPose']['quaternion'];order=geometry.cusa_order(p)
    profile=cusa.profile(SimpleNamespace(scene=scene,state={'toolMechanicsRevision':frame['toolMechanicsRevision']}))
    allocated=capacity_prediction(frame['cusaBefore'],frame['removedSourceIds'],order,float(scene.voxel_mm3),profile,params,frame['durationSeconds'],True)
    covers=[];moving=swept_envelopes(TOOLS['cusa'],params,frame['slots'][key],{'position':p,'quaternion':q})
    for slot,value in frame['slots'].items():
        tool=TOOLS[value['instrument']];held=set(frame['holds'].get(slot,{}).get('cells',[]))
        pieces=moving if slot==key else envelopes(tool,frame['parametersBySlot'][slot],value['position'],value['quaternion'])
        for index,(a,b,radius,part) in enumerate(pieces):
            hit=geometry.capsule(a,b,radius);positive=set(hit['witnessedSourceIds']);uncertain=set(hit['indeterminateSourceIds'])
            covers.append({'slot':slot,'part':part,'piece':index,'startMm':a.tolist(),'endMm':b.tolist(),'radiusMm':radius,
                'coverPointWitnessSourceIds':sorted(positive),'indeterminateCoverSourceIds':sorted(uncertain),
                'rasterCoverSourceIds':hit['rasterSourceIds'],'positiveCoverPoints':hit['positivePoints'],
                'shaftForbiddenCoverPointSourceIds':sorted(positive-held) if part=='shaft' else [],
                'shaftForbiddenIndeterminateCoverSourceIds':sorted(uncertain-held) if part=='shaft' else [],
                'shaftForbiddenRasterCoverSourceIds':sorted(set(hit['rasterSourceIds'])-held) if part=='shaft' else [],
                'actualSweptCollisionEstablished':False,'scope':'Enclosing capsule cover; positive cover overlap is not an actual swept contact witness'})
    other_pairs=[{'otherSlot':slot,'conservativeCoverConflict':pair_conflict(moving,envelopes(TOOLS[value['instrument']],
        frame['parametersBySlot'][slot],value['position'],value['quaternion']))} for slot,value in frame['slots'].items() if slot!=key]
    return {'limiter':limiter,'orderedSourceIds':order,'candidateSourceIds':geometry.occupancy.material_ids(geometry.grid_cover(p,p,1.5)),
        **allocated,'covers':covers,'toolPairCovers':other_pairs,'conditionedOnCapturedNativeBeforeLedger':True,
        'predictedStatePropagated':False,'largerDomainPublicActionsAdmitted':False,'physicalValidationPassed':False}


def require_native_prediction_exact(frame,prediction):
    app=frame['applicationCalls'][0]
    expected_query={'positionMm':prediction['limiter']['resolvedMm'],'candidateSourceIds':prediction['candidateSourceIds'],
        'orderedSourceIds':prediction['orderedSourceIds']}
    if (prediction['limiter']!=frame['limiterCalls'][0] or app['orderedQueries']!=[expected_query]
            or prediction['candidateSourceIds']!=app['candidateSourceIds']
            or any(prediction[k]!=app[k] for k in ('cusaAfter','removedSourceIdsAfter','events'))):
        raise ValueError('PRE_EFFECT_NATIVE_PREDICTION_DIFFERS')
