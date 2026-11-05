"""Paired-lane software controls, never acquired or biological evidence."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab import surgical_task_admission as admission
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine


@pytest.fixture(scope='module')
def paired_source(tmp_path_factory):
    root=tmp_path_factory.mktemp('generated-paired-lane-software-control');shape=(28,28,28)
    image=nib.Nifti1Image(np.indices(shape).sum(axis=0).astype(np.float32),np.eye(4))
    image.header.set_xyzt_units('mm');nib.save(image,root/'generated-scan.nii.gz')
    brain=np.zeros(shape,np.uint8);brain[5:23,5:23,5:23]=1
    target=np.zeros(shape,np.uint8);target[8:20,8:20,9:19]=1
    image=nib.Nifti1Image(target,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,root/'generated-labels.nii.gz')
    native=PatientImagingCollection.create(root/'native','generated-paired-lane-control')
    native.document['study_timepoint']='preoperative'
    native.add_series(root/'generated-scan.nii.gz',series_id='T1',modality='T1',timepoint='preoperative')
    native.series('T1')['source_mirror_revision']='generated-paired-lane-control-v1'
    native.add_layer(brain,np.eye(4),layer_id='brain',source_series='T1',kind='brain',role='context',color='#aaaaaa')
    native.add_layer(target,np.eye(4),layer_id='target',source_series='T1',kind='tumor',role='target',color='#aa00aa',
        provenance={'kind':'observed_source_annotation','source':native.original(root/'generated-labels.nii.gz')})
    native.save();declarations=root/'declarations';(declarations/'generated').mkdir(parents=True)
    (declarations/'generated/development-declaration.json').write_text(json.dumps({'role':'DEVELOPMENT',
        'cases':['generated-paired-lane-control'],'mirror_revision':'generated-paired-lane-control-v1'}))
    patch=pytest.MonkeyPatch();patch.setattr(admission,'_DECLARATIONS_ROOT',declarations)
    exported=prepare_surgical_environment(native.root,root/'scene',layer_ids={'brain':'brain','tumor':'target'})
    engine=SurgicalEngine(load_environment(exported['casePath']))
    for action in [dict(slot='right',operation='configure',mechanicsRevision=2,clockRevision=1,duration=.1),
        dict(slot='right',operation='place',duration=.1,pose={'position':[45.,45.,-10.],'quaternion':[0.,0.,0.,1.]}),
        dict(slot='right',operation='withdraw',duration=.1)]:assert engine.step(action)['status']=='accepted'
    try:yield engine,native,{'surfaceMm':[13.,13.,5.],'sourceTargetMm':[13.,13.,9.]}
    finally:patch.undo()

import copy
from resectionlab import surgical_exposure as exposure, surgical_two_tool as two
from resectionlab.core import semantic_digest
from resectionlab.surgical_planning_observation import physical_state_hash


@pytest.fixture(scope='module')
def paired_runs(paired_source):
    engine,native,ray=paired_source
    raw={**exposure.DEFAULTS,'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.,'maxCandidates':1,'maxWallSeconds':90.}
    rows=[];before=engine.snapshot()
    for width in (0.,3.42):
        prepared,proof=exposure.prepare(engine,{**raw,'lateralSweepMm':width},native_collection=native.root,
            source_ray=ray,family=exposure.PAIRED_FAMILY)
        assert prepared is not None,proof['publicPlan']
        task=admission.task_spec(entry={'centerMm':ray['surfaceMm'],'radiusMm':6.},
            target={'centerMm':ray['sourceTargetMm'],'radiusMm':6.},axis=[0.,0.,1.],allow_unreviewed=True)
        admission.bind_admission(prepared,admission.prepare_admission(prepared,native_collection=native.root,task=task))
        comparison=two.generate_comparison(prepared,two.defaults(prepared)['defaults'])
        rows.append((prepared,proof,comparison))
    assert engine.snapshot()==before
    return raw,rows


def test_actual_second_lane_changes_complete_sequence_in_generated_control(paired_source,paired_runs):
    engine,_,ray=paired_source;_,rows=paired_runs
    assert [r[2]['publicPlan']['decision']['selectedId'] for r in rows]==['stop','long']
    assert rows[0][1]['publicPlan']['attempts'][0]['summary']['additionalNonTargetMm3'] < rows[1][1]['publicPlan']['attempts'][0]['summary']['additionalNonTargetMm3'] <= 200.
    for index,(prepared,proof,comparison) in enumerate(rows):
        assert proof['schema']==exposure.PAIRED_SCHEMA and proof['publicPlan']['family']==exposure.PAIRED_FAMILY
        attempt=proof['publicPlan']['attempts'][0];assert len(attempt['route']['paths'])==(5 if index==0 else 10)
        assert attempt['summary']['additionalTargetMm3']==0. and attempt['summary']['withdrawn']
        assert len(attempt['actionProgram'])<=128 and attempt['summary']['elapsedSeconds']<=180.
        native=engine.clone()
        for action,expected in zip(attempt['actionProgram'],attempt['physicalStateHashes'],strict=True):
            native.step(copy.deepcopy(action));assert physical_state_hash(native)==expected
        assert native.snapshot()==prepared.snapshot()
    final,_,receipt=two.materialize(rows[1][0],rows[1][2],'long')
    assert receipt['withdrawalComplete'] and receipt['nativeMatchesPublicPrediction']
    assert final.state['removed']!=rows[1][0].state['removed']
    long=next(r for r in rows[1][2]['publicPlan']['alternatives'] if r['id']=='long')
    assert long['summary']['additionalTargetMm3']>=1. and long['summary']['additionalNonTargetMm3']==0.


@pytest.mark.parametrize('family,width,ray',[(True,0.,True),('unknown',0.,True),(exposure.PAIRED_FAMILY,4.01,True),
    (exposure.PAIRED_FAMILY,0.,False),(None,3.42,True)])
def test_paired_family_bounds_and_explicit_fixed_ray(paired_source,paired_runs,family,width,ray):
    engine,native,source=paired_source;raw,_=paired_runs;before=engine.snapshot()
    with pytest.raises(ValueError):exposure.prepare(engine,{**raw,'lateralSweepMm':width},native_collection=native.root,
        source_ray=source if ray else None,family=family)
    assert engine.snapshot()==before


def test_paired_execution_budget_failure_preserves_source(paired_source,paired_runs):
    engine,native,ray=paired_source;raw,_=paired_runs;before=engine.snapshot()
    candidate,proof=exposure.prepare(engine,{**raw,'lateralSweepMm':3.42,'maxActions':6},
        native_collection=native.root,source_ray=ray,family=exposure.PAIRED_FAMILY)
    assert candidate is None and proof['publicPlan']['decision']['operation']=='STOP'
    assert proof['publicPlan']['attempts'] and not proof['publicPlan']['attempts'][0]['complete']
    assert engine.snapshot()==before


@pytest.mark.parametrize('field',['family','route','path-inventory','path-endpoint'])
def test_resealed_producer_cannot_replace_paired_geometry(paired_source,paired_runs,monkeypatch,field):
    engine,native,ray=paired_source;raw,rows=paired_runs;public=copy.deepcopy(rows[1][1]['publicPlan'])
    if field=='family':public['family']='cardinal-v1'
    elif field=='route':public['attempts'][0]['route']['surfaceMm'][0]+=.1
    elif field=='path-inventory':public['attempts'][0]['route']['paths'].pop()
    else:public['attempts'][0]['route']['paths'][-1]['endpointMm'][2]+=.1
    public['publicSeal']=semantic_digest({k:v for k,v in public.items() if k!='publicSeal'})
    monkeypatch.setattr(exposure,'plan_observed',lambda *a,**kw:copy.deepcopy(public))
    with pytest.raises(ValueError):exposure.prepare(engine,{**raw,'lateralSweepMm':3.42},
        native_collection=native.root,source_ray=ray,family=exposure.PAIRED_FAMILY)

from resectionlab import surgical_preparation_planning as planning, surgical_preparation_workspace as workspace
from resectionlab.imaging import load_case


@pytest.fixture(scope='module')
def paired_comparison(paired_source):
    engine,native,ray=paired_source
    controls={**copy.deepcopy(planning.DEFAULT_CONTROLS),'preparationFamily':exposure.PAIRED_FAMILY,
        'sourceRay':ray,'target':{'centerMm':ray['sourceTargetMm'],'radiusMm':6.},'entryRadiusMm':6.,'widthsMm':[0.,3.42]}
    before=engine.snapshot();report=planning.generate(engine,controls,native.root)
    assert engine.snapshot()==before
    return engine,native,controls,report


def test_full_plan_policy_and_installation_use_actual_second_lane(paired_comparison):
    engine,native,_,report=paired_comparison
    assert report['schema']==planning.FAMILY_SCHEMA
    assert report['policies']=={'corridorFirst':{'selectedPreparationId':'width-0','suffixId':'stop'},
        'completePlanAware':{'selectedPreparationId':'width-1','suffixId':'long'}}
    case=load_case(native.root.parent/'scene/patient.ressectionlab')
    record={'branches':{'main':engine.snapshot()},'branchId':'main','position':len(engine.actions),
        'parents':{'main':None},'timeline':{'main':copy.deepcopy(engine.history)}}
    item=workspace.install_report(engine,record,case.semantic_hash,report)
    final,receipt=workspace.install_application(engine,record,case.semantic_hash,item,'width-1','withdraw')
    assert receipt['suffixActions']>0 and receipt['preparationActions']>0 and receipt['withdrawalComplete']
    assert record['branches']['main']==engine.snapshot() and record.get('taskAdmission') is None
    assert workspace.validate_workspace(engine.scene,record,case.semantic_hash)
    view=workspace.planning_view(final,record,case.semantic_hash)
    assert view['applications'][0]['preparationFamily']==exposure.PAIRED_FAMILY
    workspace.bind_for_branch(final,record,case.semantic_hash);assert admission.admission(final) is not None
    record.update(branchId='main',position=len(engine.actions));workspace.bind_for_branch(engine,record,case.semantic_hash)
    assert admission.admission(engine) is None


@pytest.mark.parametrize('change',['family','schema','width-bound'])
def test_paired_comparison_cannot_reseal_a_different_family(paired_comparison,change):
    engine,_,_,original=paired_comparison;report=copy.deepcopy(original)
    if change=='family':report['controls']['preparationFamily']='cardinal-v1'
    elif change=='schema':report['schema']=planning.SCHEMA
    else:report['controls']['widthsMm'][1]=4.01
    report['reportSeal']=semantic_digest({k:v for k,v in report.items() if k!='reportSeal'})
    with pytest.raises(ValueError):planning.validate_report(engine,report)


def test_legacy_fixed_v2_native_proof_is_exact_at_frozen_pre_family_commit(paired_source,paired_runs):
    import subprocess,sys,types
    engine,native,ray=paired_source;raw,_=paired_runs
    path='src/resectionlab/surgical_exposure.py';name='resectionlab._exposure_before_paired_66abd626'
    code=subprocess.check_output(['git','show','66abd626490048c665677ae2ad7aeb0af9eeb8c7:'+path])
    module=types.ModuleType(name);module.__package__='resectionlab';module.__file__=str(Path.cwd()/path);sys.modules[name]=module
    try:
        exec(compile(code,module.__file__,'exec'),module.__dict__)
        for width in (0.,.3,1.):
            old,proof=module.prepare(engine,{**raw,'lateralSweepMm':width},native_collection=native.root,source_ray=ray)
            current,actual=exposure.prepare(engine,{**raw,'lateralSweepMm':width},native_collection=native.root,source_ray=ray)
            assert json.dumps(proof,sort_keys=True,separators=(',',':'))==json.dumps(actual,sort_keys=True,separators=(',',':'))
            assert (old.snapshot() if old else None)==(current.snapshot() if current else None)
    finally:sys.modules.pop(name,None)


def test_paired_public_choice_invariant_to_private_reference_and_callback(paired_source,paired_runs):
    engine,native,ray=paired_source;raw,rows=paired_runs;changed=engine.clone()
    changed._private_evaluation_truth=np.ones(engine.scene.shape,bool)
    changed._private_evaluator=lambda:pytest.fail('private evaluation truth reached paired preparation')
    controls={**raw,'lateralSweepMm':3.42};mutable=copy.deepcopy(ray)
    def check():
        controls['lateralSweepMm']=0.;mutable['sourceTargetMm'][:]=[0.,0.,0.]
    candidate,proof=exposure.prepare(changed,controls,native_collection=native.root,source_ray=mutable,
        family=exposure.PAIRED_FAMILY,check=check)
    assert proof==rows[1][1] and candidate.snapshot()==rows[1][0].snapshot()


def test_independent_full_comparison_audit_covers_installed_positive_suffix(paired_comparison):
    import sys
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    from verify_preparation_comparison import verify
    from resectionlab.imaging import save_case
    from resectionlab.workspace_bundle import load_workspace
    engine,native,_,report=paired_comparison;case,artifacts,_=load_workspace(native.root.parent/'scene/patient.ressectionlab')
    record={**copy.deepcopy(artifacts['surgical_toolbox']),'branches':{'main':engine.snapshot()},'branchId':'main','position':len(engine.actions),
        'parents':{'main':None},'timeline':{'main':copy.deepcopy(engine.history)}}
    item=workspace.install_report(engine,record,case.semantic_hash,report)
    final,_=workspace.install_application(engine,record,case.semantic_hash,item,'width-1','withdraw')
    source=native.root.parent/'paired-positive-study.ressectionlab';save_case(case,source,artifacts={'surgical_toolbox':record})
    result=verify(source,native.root.parent/'independent-positive-study')
    assert result['passed'] and len(result['revisions'])==1 and len(result['revisions'][0]['applications'])==1
    assert result['revisions'][0]['rows'][1]['geometry']['passed']
