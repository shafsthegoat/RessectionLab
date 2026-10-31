"""Frozen generated producer evidence challenged by an independent auditor."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy import ndimage as ndi

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.workspace_bundle import load_workspace
from resectionlab import surgical_exposure as producer
from test_surgical_exposure import intact, controls, completed

PATH=Path(__file__).resolve().parents[1]/'scripts/verify_source_exposure.py'
spec=importlib.util.spec_from_file_location('independent_source_exposure_audit',PATH)
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def reseal(proof):
    public=proof['publicPlan']
    public['publicSeal']=semantic_digest({k:v for k,v in public.items() if k!='publicSeal'})
    proof['preparationSeal']=semantic_digest({k:v for k,v in proof.items() if k!='preparationSeal'})
    return proof


@pytest.fixture(scope='module')
def evidence(intact,controls,completed,tmp_path_factory):
    baseline,native=intact; selected,proof=completed
    root=tmp_path_factory.mktemp('generated-exposure-audit'); cleanup=None
    if not root.resolve().is_relative_to((audit.ROOT/'testcases').resolve()):
        cleanup=tempfile.TemporaryDirectory(prefix='generated-exposure-audit-',dir=audit.ROOT/'testcases')
        root=Path(cleanup.name)
    original=native.root.parent/'scene/patient.ressectionlab'
    case,artifacts,_=load_workspace(original)
    record=copy.deepcopy(artifacts['surgical_toolbox'])
    record.update(branches={'source':baseline.snapshot(),'prepared':selected.snapshot()},branchId='prepared',
        position=len(selected.actions),parents={'source':None,'prepared':{'branchId':'source','position':len(baseline.actions)}},
        timeline={'source':copy.deepcopy(baseline.history),'prepared':copy.deepcopy(selected.history)})
    artifacts=copy.deepcopy(artifacts); artifacts['surgical_toolbox']=record
    case_path=root/'prepared.ressectionlab'; save_case(case,case_path,artifacts=artifacts)
    count=0
    def run(value, *, source=case_path):
        nonlocal count
        count+=1; path=root/f'proof-{count:03d}.json'; path.write_text(json.dumps(value))
        return audit.verify(source,path)
    try: yield {'baseline':baseline,'selected':selected,'proof':proof,'run':run,'root':root,
                'case':case,'artifacts':artifacts,'native':native,'controls':controls,'casePath':case_path}
    finally:
        if cleanup is not None: cleanup.cleanup()


def test_valid_frozen_native_proof_passes_literal_geometry_and_preserves_inputs(evidence):
    result=evidence['run'](evidence['proof'])
    assert result['passed'] and result['sourceInputsUnchanged']
    assert result['microsteps']>0 and result['boxTests']>0
    row=result['results'][0]
    assert row['complete'] and row['summary']['additionalTargetMm3']==0.
    assert row['summary']==evidence['proof']['publicPlan']['attempts'][0]['summary']


@pytest.mark.parametrize('change',['native-hash','receipt','omitted-prefix','summary','complete','failure',
    'choice','operation','public-observation','public-scope','outer-scope','observation-scope',
    'radius','width','path-width','path-endpoint','route-axis','route-quaternion','source-owner',
    'source-tube','source-state','native-verified','route-outside-shadow','route-endpoint-shadow'])
def test_self_resealed_proof_forgeries_are_refused(evidence,change):
    proof=copy.deepcopy(evidence['proof']); public=proof['publicPlan']; attempt=public['attempts'][0]; route=attempt['route']
    if change=='native-hash': attempt['physicalStateHashes'][0]='0'*64
    elif change=='receipt': attempt['actionReceipts'][0]['elapsedSeconds']+=.01
    elif change=='omitted-prefix': attempt['physicalStateHashes'].pop()
    elif change=='summary': attempt['summary']['additionalNonTargetMm3']+=1.
    elif change=='complete': attempt['complete']=False
    elif change=='failure': attempt['failure']='invented-failure'
    elif change=='choice': public['decision']['selectedIndex']=None
    elif change=='operation': public['decision']['operation']='STOP'
    elif change=='public-observation': public['publicObservation']['baseline']['inputHash']='private-canary'
    elif change=='public-scope': public['scope']='clinical-access-certified'
    elif change=='outer-scope': proof['scope']='clinical-access-certified'
    elif change=='observation-scope': public['publicObservation']['scope']='clinical-access-certified'
    elif change=='radius': attempt['actionProgram'][0]['parameters']['radiusMm']=.6
    elif change=='width': public['controls']['lateralSweepMm']=.3
    elif change=='path-width': route['paths'][0]['outsideMm'][1]+=.3
    elif change=='path-endpoint': route['paths'][0]['endpointMm'][1]+=.1
    elif change=='route-axis': route['axisRasUnit']=[0.,0.,0.]
    elif change=='route-quaternion': route['quaternion']=[1.,0.,0.,0.]
    elif change=='source-owner': route['surfaceMm']=route['sourceTargetMm'][:]
    elif change=='source-tube': route['sourceTubeMm3']+=1.
    elif change=='source-state': proof['sourceStateId']='0'*64
    elif change=='native-verified': proof['nativeReplayVerified']=False
    elif change=='route-outside-shadow': route['outsideMm'][0]+=1.
    else: route['endpointMm'][0]+=1.
    reseal(proof)
    with pytest.raises((ValueError,AssertionError)):
        evidence['run'](proof)


@pytest.mark.parametrize('key,value',[('maxActions',64.),('maxCandidates',True),('maxProposals',256.),
    ('maxNonTargetMm3',True),('lateralSweepMm',False),('maxSeconds',float('nan')),
    ('maxPathMm',float('inf')),('maxWallSeconds',float('nan'))])
def test_controls_match_exact_producer_types_and_finite_bounds(evidence,key,value):
    proof=copy.deepcopy(evidence['proof']); proof['publicPlan']['controls'][key]=value
    if value!=value or value in (float('inf'),float('-inf')):
        # Nonfinite JSON cannot obtain a valid semantic seal either.
        with pytest.raises(ValueError): reseal(proof)
        return
    reseal(proof)
    with pytest.raises(ValueError): evidence['run'](proof)


@pytest.mark.parametrize('change',['selected-index-bool','complete-integer','fraction-bool',
                                  'summary-withdrawn-integer','public-private-flag-integer'])
def test_resealed_boolean_numeric_substitutions_are_not_exact_evidence(evidence,change):
    proof=copy.deepcopy(evidence['proof']); public=proof['publicPlan']; attempt=public['attempts'][0]
    if change=='selected-index-bool': public['decision']['selectedIndex']=False
    elif change=='complete-integer': attempt['complete']=1
    elif change=='fraction-bool': attempt['actionReceipts'][0]['completedFraction']=True
    elif change=='summary-withdrawn-integer': attempt['summary']['withdrawn']=1
    else: public['publicObservation']['evaluationReferencesExposed']=0
    with pytest.raises(ValueError): evidence['run'](reseal(proof))


def incomplete_proof(evidence, *, duration=1.):
    proof=copy.deepcopy(evidence['proof']); public=proof['publicPlan']; attempt=public['attempts'][0]
    actions=copy.deepcopy(attempt['actionProgram'][:2]); actions[1]['duration']=duration
    model=evidence['baseline'].clone(); receipts=[]; hashes=[]
    for action in actions:
        result=model.step(action)
        receipts.append({k:result.get(k) for k in ('status','reason','reasonCode','completedFraction','elapsedSeconds')})
        hashes.append(audit.physical_hash(model))
    attempt.update(actionProgram=actions,actionReceipts=receipts,physicalStateHashes=hashes,
        summary=audit.summary(model,evidence['baseline'].state),complete=False,failure='PREPARATION_ACTION_OR_TIME_BUDGET')
    public['decision'].update(operation='STOP',selectedIndex=None)
    return reseal(proof)


def test_real_incomplete_prefix_has_no_selected_installation(evidence):
    proof=incomplete_proof(evidence)
    artifacts=copy.deepcopy(evidence['artifacts']); record=artifacts['surgical_toolbox']
    record.update(branchId='source',position=len(evidence['baseline'].actions))
    path=evidence['root']/'incomplete-source.ressectionlab'; save_case(evidence['case'],path,artifacts=artifacts)
    result=evidence['run'](proof,source=path)
    assert result['passed'] and not result['results'][0]['complete']
    assert result['microsteps']>0


def test_native_two_second_partial_apply_cannot_bypass_one_second_task_cap(evidence):
    proof=incomplete_proof(evidence,duration=2.)
    assert proof['publicPlan']['attempts'][0]['actionReceipts'][1]['elapsedSeconds']==pytest.approx(2.)
    artifacts=copy.deepcopy(evidence['artifacts']); record=artifacts['surgical_toolbox']
    record.update(branchId='source',position=len(evidence['baseline'].actions))
    path=evidence['root']/'duration-source.ressectionlab'; save_case(evidence['case'],path,artifacts=artifacts)
    with pytest.raises(ValueError): evidence['run'](proof,source=path)


def test_capacity_conservation_alone_cannot_hide_double_counted_physical_interval(evidence):
    baseline=evidence['baseline'].clone(); baseline.__class__=audit.Capture; baseline.trace=[]
    actions=evidence['proof']['publicPlan']['attempts'][0]['actionProgram']
    for action in actions[:2]:
        baseline.trace=[]; baseline.step(copy.deepcopy(action))
    frame=copy.deepcopy(next(row for row in baseline.trace if row['active']))
    frame['after']['cusa']['inputCapacityMm3']+=1.
    frame['after']['cusa']['unusedCapacityMm3']+=1.
    scene=baseline.scene
    boxes=audit.IndependentFootprint(scene.labels!=0,scene.affine,[],{'marginMm':0.,'maxNearMm3':1000.})
    materials={r['id']:r['material'] for r in scene.regions}
    class Labels:
        def __getitem__(self,cell): return materials[int(scene.labels.flat[cell])]
    with pytest.raises(ValueError,match='one active interval'):
        audit.trace_check(frame,boxes,Labels(),scene.voxel_mm3,2.)


def test_installed_branch_must_equal_the_selected_actual_native_endpoint(evidence):
    artifacts=copy.deepcopy(evidence['artifacts']); record=artifacts['surgical_toolbox']
    record.update(branchId='source',position=len(evidence['baseline'].actions))
    path=evidence['root']/'wrong-install.ressectionlab'; save_case(evidence['case'],path,artifacts=artifacts)
    with pytest.raises(ValueError,match='installed preparation differs'):
        evidence['run'](evidence['proof'],source=path)


@pytest.mark.parametrize('change',['interior-source','interior-target'])
def test_surface_and_target_boundary_claims_are_checked_against_source_grid(evidence,change):
    baseline=evidence['baseline']; scene=baseline.scene
    row=copy.deepcopy(evidence['proof']['publicPlan']['attempts'][0]['route'])
    retained=scene.labels.copy(); retained.flat[baseline.state['removed']]=0
    free=retained==0; seed=np.zeros(scene.shape,bool)
    for dimension in range(3):
        for side in (0,-1):
            slab=[slice(None)]*3; slab[dimension]=side; seed[tuple(slab)]=free[tuple(slab)]
    exterior=ndi.binary_propagation(seed,mask=free)
    ids=[r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor']
    target=np.isin(scene.labels,ids)
    boxes=audit.IndependentFootprint(scene.labels!=0,scene.affine,[],{'marginMm':0.,'maxNearMm3':1000.})
    targets=audit.IndependentFootprint(target,scene.affine,[],{'marginMm':0.,'maxNearMm3':1000.})
    audit.geometry(row,baseline,evidence['controls'],boxes,targets,exterior)
    if change=='interior-source':
        row['surfaceMm']=(np.asarray(row['surfaceMm'])+row['axisRasUnit']).tolist()
        message='exterior-connected boundary'
    else:
        inner=np.flatnonzero(ndi.binary_erosion(target).ravel()); assert len(inner)
        row['sourceTargetMm']=scene.points(inner[:1])[0].tolist()
        message='retained boundary cell'
    with pytest.raises(ValueError,match=message):
        audit.geometry(row,baseline,evidence['controls'],boxes,targets,exterior)


@pytest.mark.parametrize('a,b,c,d,expected',[
    ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),0.),
    ((-1,0,0),(1,0,0),(0,-1,2),(0,1,2),2.),
    ((0,0,0),(2,0,0),(1,3,0),(3,3,0),3.),
    ((0,0,0),(1,0,0),(2,0,0),(3,0,0),1.),
    ((0,0,0),(1,0,0),(1,0,0),(1,1,0),0.),
    ((0,1,0),(0,1,0),(-1,0,0),(1,0,0),1.),
    ((0,0,0),(0,0,0),(1,2,2),(1,2,2),3.),
    ((0,0,0),(1,0,0),(2,1,0),(2,3,0),2.**.5)])
def test_independent_segment_distance_crossing_parallel_and_degenerate_cases(a,b,c,d,expected):
    a,b,c,d=(np.asarray(p,float) for p in (a,b,c,d))
    for first,last,other,other_last in ((a,b,c,d),(b,a,d,c),(c,d,a,b)):
        assert audit.segment_distance(first,last,other,other_last)==pytest.approx(expected,abs=1e-12)


def synthetic_frame(left,right_before,right_after,removed=()):
    ledger={'inputCapacityMm3':0.,'aspiratedSolidMm3':0.,'unusedCapacityMm3':0.,
            'discardedCapacityMm3':0.,'progressMm3':{}}
    before={'slots':{'left':left,'right':right_before},'removed':list(removed),'clock':0.,'cusa':ledger}
    after={'slots':{'left':copy.deepcopy(left),'right':right_after},'removed':list(removed),
           'clock':.01,'cusa':copy.deepcopy(ledger)}
    return {'before':before,'after':after,'hand':'right','operation':'apply','active':False,'dt':.01}


def tool(instrument,position):
    return {'instrument':instrument,'position':list(position),'quaternion':[0.,0.,0.,1.],
            'parameters':{'radiusMm':2. if instrument=='cranial_drill' else 1.5,'depthMm':2.}}


def source_boxes(cell):
    mask=np.zeros((32,32,64),bool); mask[cell]=True
    return audit.IndependentFootprint(mask,np.eye(4),[],{'marginMm':0.,'maxNearMm3':1000.})


def test_parked_drill_literal_retains_entire_shaft_and_distal_capsule():
    shaft,working=audit.stationary_capsules(tool('cranial_drill',(10,10,40)))
    assert np.array_equal(shaft[0],[10,10,0]) and np.array_equal(shaft[1],[10,10,38]) and shaft[2]==.5
    assert np.array_equal(working[0],[10,10,38]) and np.array_equal(working[1],[10,10,40]) and working[2]==2.
    with pytest.raises(ValueError,match='unsupported parked'):
        audit.stationary_capsules(tool('brain_retractor',(10,10,40)))


def test_trace_checks_pair_intersection_over_complete_axial_union():
    left=tool('cranial_drill',(0,0,40)); start=tool('cusa',(0,0,-10)); end=tool('cusa',(0,0,100))
    parked=audit.stationary_capsules(left)
    for right in (start,end):
        assert all(audit.segment_distance(a,b,c,d)>r+s for a,b,r,_ in parked
                   for c,d,s,_ in audit.literal_capsules(right))
    frame=synthetic_frame(left,start,end)
    with pytest.raises(ValueError,match='complete tools intersect'):
        audit.trace_check(frame,source_boxes((10,10,15)),None,1.,2.)


def test_stationary_hand_cannot_hide_retained_source_contact_but_removed_source_is_clear():
    cell=(10,10,15); left=tool('cranial_drill',(10,10,40)); right=tool('cusa',(25,25,45))
    with pytest.raises(ValueError,match='parked other instrument intersects retained source'):
        audit.trace_check(synthetic_frame(left,right,copy.deepcopy(right)),source_boxes(cell),None,1.,2.)
    removed=int(np.ravel_multi_index(cell,(32,32,64)))
    audit.trace_check(synthetic_frame(left,right,copy.deepcopy(right),[removed]),source_boxes(cell),None,1.,2.)
