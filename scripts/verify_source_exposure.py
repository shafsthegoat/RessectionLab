"""Independent native/whole-tool/source-cell audit of bounded preparation.

No exposure proposer, compiler or predictor is imported. Literal CUSA geometry,
independent source-box distances, native microsteps and capacity/clock ledgers
check every held attempt. This is not empirical tissue-model qualification.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from verify_anatomical_encounter import (ROOT, IndependentFootprint, literal_capsules, axial_capsules,
    file_hash, semantic_digest, physical_hash, SurgicalEngine, load_environment, load_workspace)
from resectionlab.surgical_task_admission import verify_source
from resectionlab.surgical_task_admission import source_evidence, task_spec
from resectionlab.surgical_planning_observation import project_observation

SCOPE = ('Finite annotation-assisted intracerebral preparation through mapped non-target tissue, '
    'with zero additional target removal. Exterior/skull/dura and complete vascular/functional anatomy '
    'remain unassessed. Native masks and source annotations are unchanged; only executed cell removal '
    'and persistent physical burdens evolve. Engineering bounds and assumed aspiration rate are not clinical limits.')


def require(value, reason):
    if not value: raise ValueError('INDEPENDENT_SOURCE_EXPOSURE: '+reason)


def seal(value, key):
    require(value[key] == semantic_digest({k: v for k, v in value.items() if k != key}), key+' differs')


def state(engine):
    return {'slots': copy.deepcopy(engine.state['slots']), 'removed': engine.state['removed'][:],
            'clock': engine.state['elapsedSeconds'], 'cusa': copy.deepcopy(engine.state['cusaV3'])}


class Capture(SurgicalEngine):
    def clone(self):
        other = super().clone(); other.trace = list(self.trace); return other

    def _substep(self, hand, operation, position, quaternion, active, dt):
        before = state(self); events = super()._substep(hand, operation, position, quaternion, active, dt)
        self.trace.append({'before': before, 'after': state(self), 'dt': dt, 'operation': operation, 'hand': hand, 'active': active})
        return events


def summary(engine, before):
    new = set(engine.state['removed'])-set(before['removed'])
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    target = sum(int(engine.scene.labels.flat[c]) in targets for c in new)*engine.scene.voxel_mm3
    return {'additionalTargetMm3': target, 'additionalNonTargetMm3': len(new)*engine.scene.voxel_mm3-target,
        'elapsedSeconds': engine.state['elapsedSeconds']-before['elapsedSeconds'],
        'withdrawn': all(not s['active'] and not s['inserted'] and not s['grip'] for s in engine.state['slots'].values())}


def stationary_capsules(slot):
    if slot['instrument'] in ('cusa','frazier_suction'): return literal_capsules(slot)
    require(slot['instrument']=='cranial_drill','unsupported parked other instrument')
    params=slot.get('parameters',{})
    require(params.get('radiusMm',2.)==2. and params.get('depthMm',2.)==2.,'parked drill geometry differs')
    from scipy.spatial.transform import Rotation
    tip=np.asarray(slot['position']);direction=Rotation.from_quat(slot['quaternion']).apply([0.,0.,1.])
    return [(tip-40.*direction,tip-2.*direction,.5,'shaft'),(tip-2.*direction,tip,2.,'working')]


def segment_distance(a,b,c,d):
    """Constrained least-squares plus rectangle edges, separate from native kernel."""
    def point_edge(point,start,end):
        edge=end-start;norm=float(edge@edge)
        return float(np.linalg.norm(point-start-edge*(np.clip((point-start)@edge/norm,0.,1.) if norm else 0.)))
    distances=[point_edge(p,s,t) for p,s,t in ((a,c,d),(b,c,d),(c,a,b),(d,a,b))]
    matrix=np.column_stack((b-a,c-d));parameters=np.linalg.lstsq(matrix,c-a,rcond=None)[0]
    if np.all(parameters>=0.) and np.all(parameters<=1.):distances.append(float(np.linalg.norm(a-c+matrix@parameters)))
    return min(distances)


def trace_check(frame, boxes, labels, voxel, rate):
    a, b = frame['before'], frame['after']; require(frame['hand'] == 'right', 'unsupported active hand')
    require(a['slots']['left'] == b['slots']['left'], 'stationary other tool changed')
    require(abs(b['clock']-a['clock']-frame['dt']) <= 1e-9, 'shared physical interval differs')
    require(set(a['removed']) <= set(b['removed']), 'removal was reversed')
    old, new = a['slots']['right'], b['slots']['right']
    covers = axial_capsules(old, new) if frame['operation'] in ('apply', 'withdraw') else literal_capsules(new)
    parked=stationary_capsules(a['slots']['left'])
    for p,q,radius,part in parked:
        require(not (boxes._cells(p,q,radius)-set(a['removed'])),'parked other instrument intersects retained source')
        for start,end,working_radius,_ in covers:
            require(segment_distance(p,q,start,end)>radius+working_radius+1e-9,'complete tools intersect during preparation')
    for p, q, radius, part in covers:
        if part == 'shaft': require(not (boxes._cells(p, q, radius)-set(a['removed'])), 'retained source cell intersects complete shaft')
    if frame['operation'] == 'place':
        require(not (boxes._cells(np.asarray(new['position']), np.asarray(new['position']), 1.5)-set(a['removed'])), 'placement enters source tissue')
    if frame['operation'] == 'withdraw':
        existing = boxes._cells(np.asarray(old['position']), np.asarray(old['position']), 1.5)-set(a['removed'])
        for p,q,radius,part in covers:
            if part == 'working': require((boxes._cells(p,q,radius)-set(a['removed'])) <= existing, 'withdrawal enters new working-tissue contact')
    changed = set(b['removed'])-set(a['removed'])
    if changed:
        require(frame['operation'] == 'apply' and frame['active'], 'inactive operation removed tissue')
        touched = boxes._cells(np.asarray(new['position']), np.asarray(new['position']), 1.5)
        require(changed <= touched and all(labels[int(c)] == 'soft_tissue' for c in changed), 'removed cells exceed current tip/source non-target')
    before, after = a['cusa'], b['cusa']
    delta = lambda name: after[name]-before[name]
    expected = rate*frame['dt'] if frame['active'] and frame['operation'] == 'apply' else 0.
    require(abs(delta('inputCapacityMm3')-expected) <= 1e-9, 'capacity depends on more than the one active interval')
    require(abs(delta('aspiratedSolidMm3')-len(changed)*voxel) <= 1e-9, 'aspirated volume differs from source cells')
    pending_before = sum(before['progressMm3'].values()); pending_after = sum(after['progressMm3'].values())
    require(abs(pending_before+delta('inputCapacityMm3')-pending_after-delta('unusedCapacityMm3')
        -delta('discardedCapacityMm3')-delta('aspiratedSolidMm3')) <= 1e-8, 'capacity conservation differs')


def complete(phases, row, original):
    index = 0
    def take(operation, position, quaternion, duration):
        nonlocal index
        if index == len(phases): return None
        action, result, slot = phases[index]
        good = result['status'] == 'accepted' or (operation == 'apply' and result['status'] == 'partial'
            and result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS' and abs(result['elapsedSeconds']-duration) < 1e-8)
        if not good or semantic_digest(action) != semantic_digest({'slot':'right', 'operation':operation, 'pose':{'position':position,'quaternion':quaternion},
                'parameters':{'radiusMm':1.5,'power':1.,'duty':1.},'duration':duration}): return None
        index += 1; return slot
    for path in row['paths']:
        if take('place', path['outsideMm'], row['quaternion'], .1) is None: return False
        while True:
            slot = take('apply', path['endpointMm'], row['quaternion'], 1.)
            if slot is None: return False
            if np.linalg.norm(np.asarray(slot['position'])-path['endpointMm']) < 1e-5: break
        if take('withdraw', path['outsideMm'], row['quaternion'], .5) is None: return False
    if take('place', original['position'], original['quaternion'], .1) is None: return False
    last = take('withdraw', original['position'], original['quaternion'], .1)
    return last is not None and index == len(phases) and not last['active'] and not last['inserted'] and not last['grip']


def source_ray_points(ray, baseline, exterior):
    scene = baseline.scene; removed = set(baseline.state['removed'])
    require(type(ray) is dict and set(ray)=={'surfaceMm','sourceTargetMm'} and all(type(p) is list and len(p)==3
        and all(type(x) in (int,float) and np.isfinite(x) for x in p) for p in ray.values()),'source ray declaration differs')
    source, goal = np.asarray(ray['surfaceMm']), np.asarray(ray['sourceTargetMm'])
    def owner(point):
        index = (point-scene.affine[:3,3])@scene.inverse[:3,:3].T
        require(np.allclose(index,np.rint(index),atol=1e-8,rtol=0.) and np.all(index>=0) and np.all(index<scene.shape), 'source representative is not a native cell center')
        cell = int(np.ravel_multi_index(np.rint(index).astype(int),scene.shape)); require(cell not in removed,'source representative was already removed'); return cell
    materials = {r['id']:r['material'] for r in scene.regions}; target_ids = {r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor'}
    source_cell,goal_cell=owner(source),owner(goal)
    require(materials.get(int(scene.labels.flat[source_cell]))=='soft_tissue' and int(scene.labels.flat[goal_cell]) in target_ids,'declared source surface/target ownership differs')
    def neighbors(cell):
        ijk=np.asarray(np.unravel_index(cell,scene.shape))
        for axis in range(3):
            for sign in (-1,1):
                other=ijk.copy();other[axis]+=sign
                if np.all(other>=0) and np.all(other<scene.shape):yield int(np.ravel_multi_index(other,scene.shape))
    require(any(exterior.flat[c] for c in neighbors(source_cell)),'source representative lacks exterior-connected boundary')
    require(any(c in removed or int(scene.labels.flat[c]) not in target_ids for c in neighbors(goal_cell)),'target representative is not a retained boundary cell')
    return source,goal


def geometry(row, baseline, controls, occupied, targets, exterior):
    scene=baseline.scene;removed=set(baseline.state['removed'])
    source,goal=source_ray_points({k:row[k] for k in ('surfaceMm','sourceTargetMm')},baseline,exterior)
    materials={r['id']:r['material'] for r in scene.regions}
    distance = float(np.linalg.norm(goal-source)); direction = (goal-source)/distance
    require(3.5<distance<controls['maxPathMm'] and abs(distance-row['surfaceTargetDistanceMm'])<1e-8 and np.allclose(direction,row['axisRasUnit'],atol=1e-9,rtol=0.),'declared source ray differs')
    from scipy.spatial.transform import Rotation
    basis=Rotation.from_quat(row['quaternion']).as_matrix();require(np.allclose(basis[:,2],direction,atol=1e-9,rtol=0.),'tool axis differs from source ray')
    width=controls['lateralSweepMm']; offsets=[np.zeros(3)]+([basis[:,0]*width,-basis[:,0]*width,basis[:,1]*width,-basis[:,1]*width] if width else [])
    require(len(row['paths'])==len(offsets),'lateral path inventory differs'); union=set()
    require(row['outsideMm']==row['paths'][0]['outsideMm'] and row['endpointMm']==row['paths'][0]['endpointMm'],'displayed primary path differs')
    for offset,path in zip(offsets,row['paths'],strict=True):
        start=source-direction*8.+offset;require(np.allclose(path['outsideMm'],start,atol=1e-8,rtol=0.),'path outside point differs')
        require(not (targets._cells(start,start,1.5)-removed) and bool(targets._cells(start,goal+offset,1.5)-removed),'target ray is not bracketed')
        low,high=0.,distance+8.
        for _ in range(26):
            middle=(low+high)/2
            if targets._cells(start,start+direction*middle,1.5)-removed:high=middle
            else:low=middle
        endpoint=start+direction*(low-.25)
        require(np.allclose(endpoint,path['endpointMm'],atol=1e-7,rtol=0.) and np.dot(endpoint-(source+offset),direction)>=.5,'target stand-off or path depth differs')
        cells=occupied._cells(start,endpoint,1.5)-removed
        require(bool(cells) and all(materials[int(scene.labels.flat[c])]=='soft_tissue' for c in cells),'path tube contains non-brain/target source material')
        union.update(cells)
    require(abs(len(union)*scene.voxel_mm3-row['sourceTubeMm3'])<1e-9 and row['sourceTubeMm3']<=controls['maxNonTargetMm3'],'source tube union differs')


def verify(case_path, preparation_path):
    case_path, preparation_path = Path(case_path).resolve(), Path(preparation_path).resolve()
    require(all(p.is_relative_to(ROOT/'testcases') for p in (case_path, preparation_path)), 'inputs must stay in testcases')
    inputs = {str(p): file_hash(p) for p in (case_path, preparation_path)}
    proof = json.loads(preparation_path.read_text()); seal(proof, 'preparationSeal'); public = proof['publicPlan']; seal(public, 'publicSeal')
    fixed=proof['schema']=='resectionlab.source-exposure-preparation.v2'
    require((proof['schema'],public['schema']) == (('resectionlab.source-exposure-preparation.v2','resectionlab.public-source-exposure-preparation.v2') if fixed
        else ('resectionlab.source-exposure-preparation.v1','resectionlab.public-source-exposure-preparation.v1')),'preparation schema differs')
    require(('sourceRay' in public)==fixed,'source ray schema differs')
    require(proof['nativeReplayVerified'] is True,'native replay declaration differs')
    _, artifacts, _ = load_workspace(case_path); record = artifacts['surgical_toolbox']; scene = load_environment(case_path)
    source = [s for s in record['branches'].values() if semantic_digest(s) == proof['sourceSnapshotHash']]
    require(bool(source) and scene.identity == proof['sourceSceneId'], 'preparation source branch differs')
    require(semantic_digest(verify_source(scene, native_collection=proof['sourceVerification']['nativeCollectionPath'])) == semantic_digest(proof['sourceVerification']), 'source qualification differs')
    baseline = SurgicalEngine.restore(scene, source[0], use_cache=False); require(baseline.state_id == proof['sourceStateId'], 'source physical state differs')
    target_ids=[r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor'];center=scene.points(np.flatnonzero(np.isin(scene.labels,target_ids))).mean(axis=0).tolist()
    proposal=task_spec(entry={'centerMm':center,'radiusMm':15.},target={'centerMm':center,'radiusMm':15.},axis=[0.,0.,1.],allow_unreviewed=True)
    observed=project_observation(baseline,proposal,source_evidence(scene)['availableEvidence'])
    body={'schema':'resectionlab.source-exposure-observation.v1','baseline':observed.manifest(),
          'capabilities':{'operations':['place','apply','withdraw'],'instrument':'cusa','radiusMm':1.5,'maxAdditionalTargetMm3':0.,'sourceDomain':'observed mapped labels and physical state'},'scope':SCOPE}
    require(semantic_digest(public['publicObservation'])==semantic_digest({**body,'inputHash':semantic_digest(body),'evaluationReferencesExposed':False}) and public['scope']==proof['scope']==SCOPE,'public source input/contract differs')
    materials = {r['id']: r['material'] for r in scene.regions}; labels = scene.labels.ravel()
    class Materials:
        def __getitem__(self, cell): return materials[int(labels[cell])]
    boxes = IndependentFootprint(scene.labels != 0, scene.affine, [], {'marginMm':0.,'maxNearMm3':1000.})
    target_boxes=IndependentFootprint(np.isin(scene.labels,target_ids),scene.affine,[],{'marginMm':0.,'maxNearMm3':1000.})
    # Independently qualify the declared source surface, rather than trusting
    # the producer's chosen voxel or treating internal holes as exterior.
    from scipy.ndimage import binary_propagation
    free=scene.labels==0;free.flat[baseline.state['removed']]=True;seed=np.zeros(scene.shape,bool)
    for axis in range(3):
        for side in (0,-1):
            slab=[slice(None)]*3;slab[axis]=side;seed[tuple(slab)]=free[tuple(slab)]
    exterior=binary_propagation(seed,mask=free)
    if fixed:source_ray_points(public['sourceRay'],baseline,exterior)
    controls = public['controls']; require(set(controls) == {'maxNonTargetMm3','maxActions','maxSeconds','maxCandidates','maxProposals','maxPathMm','maxWallSeconds','lateralSweepMm'}, 'control schema differs')
    require(all(type(value) is int if key in ('maxActions','maxCandidates','maxProposals')
        else type(value) in (int,float) and np.isfinite(value) for key,value in controls.items()),'control types differ')
    require(0 < controls['maxNonTargetMm3'] <= 200 and 6 <= controls['maxActions'] <= 128 and 2 <= controls['maxSeconds'] <= 180, 'unbounded preparation')
    require(1 <= controls['maxCandidates'] <= 12 and len(public['attempts']) <= controls['maxCandidates'], 'candidate inventory differs')
    require(not fixed or len(public['attempts'])<=1,'fixed source ray permits at most one held attempt')
    require(0<=controls['lateralSweepMm']<=1. and 3.5<controls['maxPathMm']<=40. and 1<=controls['maxProposals']<=1024 and 0<controls['maxWallSeconds']<=90.,'proposal bounds differ')
    require(public['decision']['selectedIndex'] is None or type(public['decision']['selectedIndex']) is int,'selection type differs')
    rate = scene.properties.get('cusaV3', {}).get('capacityMm3PerSecond', 2.); results = []; chosen = None; microsteps = 0
    for index, attempt in enumerate(public['attempts']):
        require(type(attempt['complete']) is bool,'completion type differs')
        if fixed:require(semantic_digest({k:attempt['route'][k] for k in public['sourceRay']})==semantic_digest(public['sourceRay']),'held attempt replaced source ray')
        geometry(attempt['route'],baseline,controls,boxes,target_boxes,exterior)
        engine = baseline.clone(); engine.__class__ = Capture; engine.trace = []; phases = []
        require(len(attempt['actionProgram']) <= controls['maxActions'] and sum(a['duration'] for a in attempt['actionProgram']) <= controls['maxSeconds']+1e-10, 'program exceeds action/clock bound')
        for action, receipt, expected in zip(attempt['actionProgram'], attempt['actionReceipts'], attempt['physicalStateHashes'], strict=True):
            require(action['slot']=='right' and action['operation'] in ('place','apply','withdraw') and semantic_digest(action['parameters'])==semantic_digest({'radiusMm':1.5,'power':1.,'duty':1.})
                and type(action.get('duration')) in (int,float) and 0<action['duration']<=1., 'unsupported preparation action')
            engine.trace = []; before = engine.snapshot(); result = engine.step(copy.deepcopy(action))
            require(semantic_digest({k:result.get(k) for k in ('status','reason','reasonCode','completedFraction','elapsedSeconds')}) == semantic_digest(receipt) and physical_hash(engine) == expected, 'native prefix differs')
            if result['status'] == 'rejected': require(not engine.trace and engine.snapshot() == before, 'rejected action leaked effects')
            for frame in engine.trace: trace_check(frame, boxes, Materials(), scene.voxel_mm3, rate)
            microsteps += len(engine.trace); phases.append((action, result, copy.deepcopy(engine.state['slots']['right'])))
        require(semantic_digest(summary(engine, baseline.state)) == semantic_digest(attempt['summary']), 'preparation summary differs')
        done = complete(phases, attempt['route'], baseline.state['slots']['right'])
        require(done == attempt['complete'] and (attempt['failure'] is None) == done, 'complete/failed program differs')
        if done:
            require(attempt['summary']['additionalTargetMm3'] == 0 and 0 < attempt['summary']['additionalNonTargetMm3'] <= controls['maxNonTargetMm3'], 'complete preparation burden differs')
            if chosen is None: chosen = index
        if index == public['decision']['selectedIndex']:
            require(engine.snapshot() == record['branches'][record['branchId']], 'installed preparation differs')
        results.append({'index':index, 'complete':done, 'summary':attempt['summary']})
    require(public['decision']['selectedIndex'] == chosen and public['decision']['operation'] == ('PREPARE' if chosen is not None else 'STOP'), 'selection differs')
    if chosen is None:require(semantic_digest(record['branches'][record['branchId']])==semantic_digest(baseline.snapshot()),'STOP installed an unselected physical state')
    require(all(file_hash(Path(p)) == h for p,h in inputs.items()), 'native inputs changed')
    require(verify_source(scene,native_collection=proof['sourceVerification']['nativeCollectionPath'])==proof['sourceVerification'],'source bytes changed during audit')
    return {'passed':True,'sourceSceneId':scene.identity,'results':results,'microsteps':microsteps,
        'boxTests':boxes.box_tests,'sourceInputsUnchanged':True,'inputSha256':inputs,
        'scope':'Independent literal complete CUSA/axial-source-box checks, native prefixes, irreversible source removal and clock/capacity conservation. No preparation search optimality, unknown exterior anatomy, clinical risk or empirical mechanics claim.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--case',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require(args.output.resolve().is_relative_to(ROOT/'testcases') and not args.output.exists(),'preserve evidence in testcases')
    result=verify(args.case,args.preparation);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
