"""Read-only local CUSA core/fan proposals after a source-bound passage.

Generic research apertures, whole-cell effects and the unchanged 0.5 mm shaft
are explicit assumptions. No calibrated energy or aspiration model is claimed.
Every action needs public engine preview/step, guards and target-only admission.
No source/engine state is modified and no transition executes in this module.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from .surgical_motion import envelopes, swept_envelopes, pair_conflict
from .surgical_tools import TOOLS


def local_debulking_candidates(engine, *, slot='left', limit=6, radius_mm=1.5):
    """Rank checked local target effects and bounded core positioning records.

    radius_mm is a bounded maximum: explicit 6 mm requests fall back through
    4/2.5/1.5 mm where the larger working pocket is unavailable; default 1.5 mm
    never widens its actual proposed aperture. Preserves the existing record
    contract. Added fan paths
    rotate about a source-derived cleared inner aperture rather than assuming a
    parallel translation of a shaft through intact bone. Geometry prediction
    checks every fine swept envelope in engine order and credits only preceding
    tip effects before checking each subsequent shaft. Public preview remains
    mandatory, including for positioning records with zero predicted removal.
    """
    state=engine.state['slots'][slot]
    if state['instrument']!='cusa' or not state['inserted']:return []
    if not .1<=float(radius_mm)<=6.:
        raise ValueError('Local research aperture must be in [0.1, 6] mm')
    requested_max=float(radius_mm)
    allowed_radii=sorted({requested_max,*[r for r in (1.5,2.5,4.,6.) if r<=requested_max]},reverse=True)
    scene=engine.scene; tool=TOOLS['cusa']; params=engine.parameters(state)
    current=np.asarray(state['position'],float); q=list(state['quaternion'])
    rotation=Rotation.from_quat(q); axis=rotation.apply([0.,0.,1.])
    owner=engine.occupancy(); labels=scene.labels.ravel()
    targets={r['id'] for r in scene.regions if r.get('target')}
    bones={r['id'] for r in scene.regions if r['material']=='bone'}
    other=[]
    for key,value in engine.state['slots'].items():
        if key!=slot:
            other.extend(envelopes(TOOLS[value['instrument']],engine.parameters(value),value['position'],value['quaternion']))
    assumptions={'apertureModel':'generic research CUSA aperture; no calibrated energy model',
                 'shaftRadiusMm':.5,'wholeCellEffects':True,'sourceSceneId':scene.identity,
                 'admission':'public preview/step, full instrument collision and matching guards required',
                 'withdrawal':'aperture persists; explicitly check/reduce it before narrow-entry withdrawal',
                 'fanModel':'small-angle rigid rotation about a cleared source-derived inner aperture; not tissue retraction'}

    def identities(parts,removed):
        work=set(); shaft=set()
        for a,b,r,part in parts:
            cells=set(owner.material_ids(engine.cells(a,b,r)))-removed
            (shaft if part=='shaft' else work).update(cells)
        return work,shaft

    def validate(path,working_radius,*,prefix=False):
        """Immutable geometric transition prediction in the engine's order."""
        local_removed=set(); accepted=[]
        previous={'position':current.tolist(),'quaternion':q}
        actual_params={**params,'radiusMm':float(working_radius)}
        for pose in path:
            parts=swept_envelopes(tool,actual_params,previous,pose)
            if other and pair_conflict(parts,other):break
            work,shaft=identities(parts,local_removed)
            tip=set(owner.material_ids(engine.cells(previous['position'],pose['position'],working_radius)))-local_removed
            if shaft or any(int(labels[c]) not in targets for c in work|tip):break
            # Shaft checks occur before this microstep's removal in the engine.
            local_removed.update(tip)
            accepted.append(pose);previous=pose
        if not accepted or not prefix and len(accepted)!=len(path):return None
        return accepted,local_removed

    def stationary_pocket(point,working_radius,removed=frozenset()):
        actual_params={**params,'radiusMm':float(working_radius)}
        work,shaft=identities(envelopes(tool,actual_params,point,q),removed)
        if shaft or any(int(labels[c]) not in targets for c in work):return None
        tip=set(owner.material_ids(engine.cells(point,point,working_radius)))-set(removed)
        return len(tip) if all(int(labels[c]) in targets for c in tip) else None

    def straight_path(points,step=.05):
        path=[];start=current
        for endpoint in points:
            endpoint=np.asarray(endpoint,float)
            count=max(1,int(np.ceil(np.linalg.norm(endpoint-start)/step)))
            for f in np.linspace(0.,1.,count+1)[1:]:
                path.append({'position':(start*(1.-f)+endpoint*f).tolist(),'quaternion':q})
            start=endpoint
        return path if len(path)<=512 else []

    def record(kind,path,working_radius,predicted,**extra):
        travel=sum(float(np.linalg.norm(np.asarray(p['position'])-np.asarray(previous['position'])))
                   for previous,p in zip([{'position':current.tolist()},*path[:-1]],path))
        action={'slot':slot,'operation':'apply','path':path,
                'parameters':{'radiusMm':float(working_radius)},'duration':max(.1,min(30.,travel/10.))}
        return {'kind':kind,'action':action,'predictedTargetCells':len(predicted),
                'workingRadiusMm':float(working_radius),'requiresPublicPreview':True,
                'assumptions':assumptions,**extra}

    records=[]
    active_radius=None;pocket=None
    for trial in allowed_radii:
        trial_count=stationary_pocket(current,trial)
        if trial_count is not None:
            active_radius=trial;pocket=trial_count;break
    if active_radius is not None:
        checked=validate([{'position':current.tolist(),'quaternion':q}],active_radius)
        if checked and checked[1]:
            records.append(record('core-removal',checked[0],active_radius,checked[1],strategy='static-core'))
            # A large verified pure-target pocket needs one meaningful trial,
            # not dozens of expensive fan proposals before that first effect.
            if len(checked[1])>=500:return records[:limit]
        # The source bone nearest the tip along the recorded outward ray marks
        # an inner aperture candidate. Project its cell onto that SAME ray, so
        # the starting orientation matches exactly and no pose jump is hidden.
        ray_end=current-axis*50.
        source_cells=engine.cells(current,ray_end,.1)
        bone_cells=[int(c) for c in source_cells if int(labels[c]) in bones]
        pivot=None
        if bone_cells:
            xyz=scene.points(bone_cells);along=(xyz-current) @ (-axis)
            valid=along>2.
            if np.any(valid):
                depth=float(np.min(along[valid]));pivot=current-axis*depth
        if pivot is not None:
            base_axis=axis.copy()
            def pivot_quaternion(point):
                desired=(point-pivot)/np.linalg.norm(point-pivot)
                cross=np.cross(base_axis,desired);norm=float(np.linalg.norm(cross))
                if norm<1e-12:return q
                turn=Rotation.from_rotvec(cross/norm*np.arctan2(norm,float(np.dot(base_axis,desired))))
                return (turn*rotation).as_quat().tolist()
            x=rotation.apply([1.,0.,0.]);y=rotation.apply([0.,1.,0.])
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                lateral=(x*dx+y*dy)/np.hypot(dx,dy)
                destination=current+lateral*6.
                path=[]
                for f in np.linspace(0.,1.,301)[1:]:
                    point=current*(1.-f)+destination*f
                    path.append({'position':point.tolist(),'quaternion':pivot_quaternion(point)})
                checked=validate(path,active_radius,prefix=True)
                if checked and len(checked[0])>=5 and checked[1]:
                    records.append(record('core-removal',checked[0],active_radius,checked[1],
                                          strategy='aperture-pivot-fan',pivotMm=pivot.tolist(),
                                          lateralReachMm=float(np.linalg.norm(np.asarray(checked[0][-1]['position'])-current))))
        for travel in (-2.,-1.,1.,2.):
            path=straight_path([current+axis*travel])
            checked=validate(path,active_radius,prefix=True) if path else None
            if checked and checked[1]:records.append(record('core-removal',checked[0],active_radius,checked[1],strategy='axial-core'))
    # A small-aperture axial retreat can reach the core from a healthy interface.
    # Quarter-voxel shifts/inward approaches are searched only when no broad
    # pocket exists, and every shaft movement is checked rather than assumed.
    positioning=[]
    moves=[([current-axis*d],f'axial-retreat-{d}') for d in (2.,4.,8.,12.)]
    if active_radius is None:
        x=rotation.apply([1.,0.,0.]);y=rotation.apply([0.,1.,0.])
        for dx,dy in [(0,0),(.25,0),(-.25,0),(0,.25),(0,-.25)]:
            shifted=current+x*dx+y*dy
            moves.append(([shifted,shifted+axis*18.],f'checked-core-approach-{dx}-{dy}'))
    for points,strategy in moves:
        path=straight_path(points)
        checked=validate(path,params['radiusMm']) if path else None
        if not checked:continue
        endpoint=np.asarray(checked[0][-1]['position'])
        next_radius=None;next_count=0
        for trial in allowed_radii:
            trial_count=stationary_pocket(endpoint,trial,checked[1])
            if trial_count:
                next_radius=trial;next_count=trial_count;break
        if next_radius is None:continue
        max_radius=float(next_radius);max_count=next_count
        for trial in (2.5,4.,6.):
            if trial<=max_radius:continue
            trial_count=stationary_pocket(endpoint,trial,checked[1])
            if trial_count:max_radius=trial;max_count=trial_count
        positioning.append(record('core-positioning',checked[0],params['radiusMm'],checked[1],
                                  strategy=strategy,nextPocketRadiusMm=float(next_radius),
                                  nextPocketPredictedTargetCells=next_count,
                                  maxTargetOnlyPocketRadiusMm=max_radius,
                                  maxTargetOnlyPocketPredictedTargetCells=max_count))
    records.sort(key=lambda r:-r['predictedTargetCells'])
    positioning.sort(key=lambda r:-(r['predictedTargetCells']+r['maxTargetOnlyPocketPredictedTargetCells']))
    # Prefer an available actual larger tip effect; positioning is explicitly
    # separate so its possible zero-removal action is never misreported.
    return (records+positioning)[:limit]
