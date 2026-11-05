"""Opt-in finite-face retractor boundary on the authoritative native clock.

The body, material, topology, support and history remain the existing continuum.
Only the retractor boundary changes. All chronological state is JSON-owned by
an engine transition; operator caches contain no poses, multipliers or history.
"""
import copy

import numpy as np

PROPERTY='retractorSurfaceContactV1'
SCHEMA='resectionlab.retractor-surface-contact.v1'
MODES=('nonadhesive-frictionless-face','bonded-centroid-control')


def profile(mode=MODES[0]):
    if mode not in MODES:raise ValueError('RETRACTOR_CONTACT_MODE')
    return {'schema':SCHEMA,'mode':mode,'surfaceQuadratureOrder':2,'halfThicknessMm':.4,
        'scope':'Research fixed-orientation flat-face boundary; frictionless, no adhesion. Loaded edge entry is unqualified. Forces and consequences are not patient calibrated.'}


def validate(scene,continuum):
    p=scene.properties.get(PROPERTY)
    if p is None:return
    if not isinstance(p,dict) or p.get('mode') not in MODES or p!=profile(p['mode']):
        raise ValueError('COMPLETE_RETRACTOR_SURFACE_PROFILE_REQUIRED')
    solver=scene.properties.get('continuumLinearSolveV1',{})
    if continuum is None or solver.get('method')!='proportional-history-compliance':
        raise ValueError('RETRACTOR_SURFACE_REQUIRES_PROPORTIONAL_CONTINUUM')


def enabled(engine):
    return engine.scene.properties.get(PROPERTY,{}).get('mode')==MODES[0]


def is_surface_slot(engine,key):
    return enabled(engine) and engine.state['slots'][key]['instrument']=='brain_retractor'


def _state(engine):
    return engine.state.setdefault(PROPERTY,{'schema':SCHEMA,'nextSession':0,'observed':{},'solved':{},'queue':[],
        'toolWorkNmm':0.,'bodyWorkNmm':0.,'contactActivationDissipationNmm':0.,'substeps':0,
        'initialEnergyNmm':None,'maxCumulativeEnergyResidualNmm':0.,'last':None})


def enqueue(engine,dt):
    state=_state(engine);slots={}
    holds=engine.state['holds']
    if any(not is_surface_slot(engine,key) or 'target' not in hold for key,hold in holds.items()):
        raise ValueError('RETRACTOR_SURFACE_MIXED_HOLDS_UNQUALIFIED')
    if len(holds)>1:raise ValueError('RETRACTOR_SURFACE_MULTIPLE_BLADES_UNQUALIFIED')
    for key,hold in sorted(holds.items()):
        slot=engine.state['slots'][key];params=engine.parameters(slot)
        if 'normalContactSession' not in hold:
            state['nextSession']+=1;hold['normalContactSession']=state['nextSession']
            hold['normalContactQuaternion']=copy.deepcopy(slot['quaternion'])
            hold['normalContactShape']=[params['radiusMm'],min(params['depthMm'],5.)]
        q=hold['normalContactQuaternion'];shape=hold['normalContactShape']
        from scipy.spatial.transform import Rotation
        if (Rotation.from_quat(q).inv()*Rotation.from_quat(slot['quaternion'])).magnitude()>1e-10:
            raise ValueError('RETRACTOR_CONTACT_ROTATION_UNQUALIFIED')
        if shape!=[params['radiusMm'],min(params['depthMm'],5.)]:raise ValueError('RETRACTOR_CONTACT_GEOMETRY_CHANGED')
        old=state['observed'].get(key);session=hold['normalContactSession']
        start=old['position'] if old and old['session']==session else hold['origin']
        slots[key]={'session':session,'start':list(start),'end':list(slot['position']),
            'origin':list(hold['origin']),'quaternion':list(q),'shape':list(shape)}
    state['queue'].append({'seconds':float(dt),'slots':slots})
    state['observed']={key:{'session':v['session'],'position':v['end']} for key,v in slots.items()}


def _consume(state,h):
    remaining=h;endpoint={}
    while remaining>1e-13:
        if not state['queue']:raise ValueError('RETRACTOR_SURFACE_CLOCK_UNDERRUN')
        segment=state['queue'][0];seconds=segment['seconds'];take=min(remaining,seconds);fraction=take/seconds
        endpoint={key:{**v,'position':(np.asarray(v['start'])+fraction*(np.asarray(v['end'])-v['start'])).tolist()}
            for key,v in segment['slots'].items()}
        remaining-=take
        if seconds-take<=1e-13:state['queue'].pop(0)
        else:
            segment['seconds']=seconds-take
            for key,v in segment['slots'].items():v['start']=endpoint[key]['position']
    return endpoint


def advance(engine,system,u,memories,gravity,steps,h,diagnostic=None):
    from . import surgical_continuum as continuum
    from . import surgical_deformation_v2 as old
    from .surgical_retractor_surface import exposed_quadrature,PlanarBlade
    from .surgical_retractor_contact import step
    state=_state(engine);physical=engine.state[continuum.PROPERTY];model=engine.state['deformationModel']
    p=engine.scene.properties[continuum.PROPERTY];settings=old._settings(engine)
    mesh=system['mesh'];matrices=system['matrices'];free=system['free'];removed=set(engine.state['removed'])
    if system['contacts']:raise ValueError('RETRACTOR_SURFACE_BONDED_OPERATOR')
    surface=exposed_quadrature(engine.scene,mesh,removed,old._blocked(engine))
    solver=system['history_solve'];factor=solver._equilibrium_factor()
    inverse=lambda rhs:factor.solve(rhs)/solver.beta
    rest=engine.scene.points(mesh.cells);labels=np.array([label for label,region in engine.by_label.items()
        if region['material'] in ('bone','vessel','unknown')],dtype=engine.scene.labels.dtype)
    removed_array=np.asarray(sorted(removed),dtype=np.int64)
    for _ in range(steps):
        endpoint=_consume(state,h);key=next(iter(endpoint),None);old_blade=blade=None
        if key is not None:
            pose=endpoint[key];last=state['solved'].get(key)
            start=last['position'] if last and last['session']==pose['session'] else pose['origin']
            old_blade=PlanarBlade.create(start,pose['quaternion'],*pose['shape'])
            blade=PlanarBlade.create(pose['position'],pose['quaternion'],*pose['shape'])
        before=u.copy()
        # The parent transition's cancellation callback is checked by each
        # contact solve as well as the native action scheduler.
        check=getattr(engine,'_transition_check',None) or (lambda:None)
        result=step(matrices,free,u,memories,p['relaxationSeconds'],h,gravity,mesh,surface,old_blade,blade,inverse=inverse,check=check)
        u=result['displacement'];memories=result['memories'];local=u.reshape(-1,3)[mesh.element_nodes]
        gradient=continuum._gradient(engine.scene,local,system['gradients']);maximum=float(np.linalg.norm(gradient,axis=(2,3)).max())
        if maximum>p['maxElementGradientNorm']+1e-12:raise ValueError('CONTINUUM_SMALL_GRADIENT_DOMAIN: retractor contact; no strain projection')
        centers=local.mean(axis=1);old_centers=before.reshape(-1,3)[mesh.element_nodes].mean(axis=1)
        if np.max(np.linalg.norm(centers,axis=1))>settings['maxDisplacementMm']:raise ValueError('CONTINUUM_DISPLACEMENT_DOMAIN')
        for alpha in (.25,.5,.75,1.):
            world=rest+old_centers+alpha*(centers-old_centers)
            vox=np.rint(world@engine.scene.inverse[:3,:3].T+engine.scene.inverse[:3,3]).astype(int)
            inside=np.all((vox>=0)&(vox<np.asarray(engine.scene.shape)),axis=1)
            if old._obstacle_collision(engine,vox,inside,removed_array,labels):raise ValueError('CONTINUUM_OBSTACLE_COLLISION: coupled increment rejected')
        if np.linalg.norm(result['forceOnTissueN'])>settings['maxForceN'] or result['summedNormalForceN']>settings['maxForceN']:
            raise ValueError('CONTINUUM_FORCE_DOMAIN: no force clipping')
        if state['initialEnergyNmm'] is None:state['initialEnergyNmm']=result['oldEnergyNmm']+physical['removedStoredEnergyNmm']
        for field in ('toolWorkNmm','bodyWorkNmm','contactActivationDissipationNmm'):state[field]+=result[field]
        for field in ('dissipatedEnergyNmm','numericalDissipationNmm'):physical[field]+=result[field]
        cumulative=(result['energyNmm']-state['initialEnergyNmm']+physical['removedStoredEnergyNmm']
            +physical['dissipatedEnergyNmm']+physical['numericalDissipationNmm']+state['contactActivationDissipationNmm']
            -state['bodyWorkNmm']-state['toolWorkNmm'])
        if abs(cumulative)>1e-10:raise ValueError('RETRACTOR_CONTACT_CUMULATIVE_ENERGY')
        state['maxCumulativeEnergyResidualNmm']=max(state['maxCumulativeEnergyResidualNmm'],abs(cumulative))
        state['substeps']+=1
        state['solved']={k:{'session':v['session'],'position':v['position']} for k,v in endpoint.items()}
        state['last']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()
            if k not in ('displacement','memories','pointIndices','normalForcesN','gapMm')}
        state['last'].update(slot=key,unknownFovFaces=len(surface.unknown_fov_faces),sampledConstraintCount=len(result['pointIndices']),
            minGapMm=float(np.min(result['gapMm'],initial=0.)),cumulativeEnergyResidualNmm=cumulative)
        # These are actual sampled contact owners, not the initial hold's seeds.
        for cell in result['contactSourceIds']:
            cell=str(cell);engine.state['contact'][cell]=engine.state['contact'].get(cell,0.)+h
        for label,increments in continuum.exposure_increment(engine.scene,mesh,system['gradients'],u,memories,p['materials'],h).items():
            ledger=physical['exposureBySourceLabel'].setdefault(label,{k:0. for k in increments})
            for field,value in increments.items():ledger[field]+=value
        physical.update(maxGradientNorm=max(physical.get('maxGradientNorm',0.),maximum),
            maxFreeResidualN=max(physical.get('maxFreeResidualN',0.),result['maxFreeResidualN']),
            maxEnergyBalanceResidualNmm=max(physical.get('maxEnergyBalanceResidualNmm',0.),abs(result['energyBalanceResidualNmm'])))
    if abs(sum(v['seconds'] for v in state['queue'])-model['remainderSeconds'])>1e-10:
        raise ValueError('RETRACTOR_SURFACE_CLOCK_REMAINDER')
    physical['cornerDisplacementMm']=continuum._pack(mesh,u)
    physical['memoryDisplacementMm']=[continuum._pack(mesh,z) for z in memories]
    physical.update(activeElements=len(mesh.cells),activeDofs=len(u),
        handleReactionsN={key:result['reactionOnInstrumentN'].tolist()} if key is not None else {},
        energyBalanceScope='Body/history + moving nonadhesive tool work + BE contact activation loss + removed stored energy; no contact potential.',
        boundary='Clamped retained inactive soft interface; sampled finite exposed-face contact; other faces traction-free; no CSF/ICP support')
    engine.state['deformation']={str(c):centers[i].tolist() for i,c in enumerate(mesh.cells)}
    model['velocities']={str(c):((centers[i]-old_centers[i])/h).tolist() for i,c in enumerate(mesh.cells)}
    model.update(elasticEnergyNmm=result['energyNmm'],activeNodes=len(mesh.cells),model=continuum.SCHEMA,
        maxNeighborDisplacementRatio=max((float(np.linalg.norm(centers[i]-centers[j])/length)
            for i,j,length in system['neighbor_pairs']),default=0.))
    if diagnostic is not None:
        diagnostic['substeps']+=steps
        diagnostic['maxFreeResidualN']=max(diagnostic['maxFreeResidualN'],physical['maxFreeResidualN'])
        diagnostic['maxAbsDiscreteEnergyBalanceNmm']=max(diagnostic['maxAbsDiscreteEnergyBalanceNmm'],physical['maxEnergyBalanceResidualNmm'])
