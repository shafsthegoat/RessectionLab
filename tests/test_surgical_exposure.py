"""Generated executable preparation controls; no anatomical/clinical claim."""
import copy
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from resectionlab import surgical_exposure as exposure
from resectionlab.core import thaw_json, semantic_digest
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_planning_observation import PlanningObservation, physical_state_hash
from resectionlab import surgical_task_admission as admission


class Cancelled(RuntimeError):
    pass


@pytest.fixture(scope='module')
def intact(tmp_path_factory):
    root=tmp_path_factory.mktemp('generated-exposure-source'); shape=(28,28,28)
    image=nib.Nifti1Image(np.indices(shape).sum(axis=0).astype(np.float32),np.eye(4))
    image.header.set_xyzt_units('mm'); nib.save(image,root/'scan.nii.gz')
    brain=np.zeros(shape,np.uint8); brain[5:23,5:23,5:23]=1
    tumor=np.zeros(shape,np.uint8); tumor[11:15,11:15,11:15]=1
    image=nib.Nifti1Image(tumor,np.eye(4)); image.header.set_xyzt_units('mm'); nib.save(image,root/'labels.nii.gz')
    native=PatientImagingCollection.create(root/'native','generated-exposure-person')
    native.document['study_timepoint']='preoperative'
    native.add_series(root/'scan.nii.gz',series_id='T1',modality='T1',timepoint='preoperative')
    native.series('T1')['source_mirror_revision']='generated-exposure-revision'
    native.add_layer(brain,np.eye(4),layer_id='brain',source_series='T1',kind='brain',role='context',color='#aaaaaa')
    native.add_layer(tumor,np.eye(4),layer_id='target',source_series='T1',kind='tumor',role='target',color='#aa00aa',
        provenance={'kind':'observed_source_annotation','source':native.original(root/'labels.nii.gz')})
    native.save()
    declarations=root/'declarations'; (declarations/'generated').mkdir(parents=True)
    (declarations/'generated/development-declaration.json').write_text(json.dumps({'role':'DEVELOPMENT',
        'cases':['generated-exposure-person'],'mirror_revision':'generated-exposure-revision'}))
    patch=pytest.MonkeyPatch(); patch.setattr(admission,'_DECLARATIONS_ROOT',declarations)
    exported=prepare_surgical_environment(native.root,root/'scene',layer_ids={'brain':'brain','tumor':'target'})
    engine = SurgicalEngine(load_environment(exported['casePath']))
    for action in [dict(slot='right', operation='configure', mechanicsRevision=2, clockRevision=1, duration=.1),
                   dict(slot='right', operation='place', duration=.1,
                        pose={'position':[45.,45.,-10.],'quaternion':[0.,0.,0.,1.]}),
                   dict(slot='right', operation='withdraw', duration=.1)]:
        assert engine.step(action)['status']=='accepted'
    assert not engine.state['removed']
    try: yield engine,native
    finally: patch.undo()


@pytest.fixture(scope='module')
def controls():
    return {**exposure.DEFAULTS,'maxCandidates':1,'maxWallSeconds':90.}


def hashes(collection):
    return {str(p.relative_to(collection.root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in collection.root.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def completed(intact, controls):
    engine,native=intact; before=engine.snapshot(); files=hashes(native)
    selected,report=exposure.prepare(engine,controls,native_collection=native.root)
    assert selected is not None, report['publicPlan']
    assert before==engine.snapshot() and hashes(native)==files
    return selected,report


def test_actual_public_native_prefixes_and_irreversible_burdens_match(intact, controls, completed):
    baseline,native=intact; selected,report=completed
    public=report['publicPlan']; chosen=public['attempts'][public['decision']['selectedIndex']]
    assert public['decision']['operation']=='PREPARE' and chosen['complete'] and chosen['failure'] is None
    assert report['nativeReplayVerified'] is True
    assert chosen['summary']['additionalTargetMm3']==0.
    assert 0.<chosen['summary']['additionalNonTargetMm3']<=controls['maxNonTargetMm3']
    assert chosen['summary']['withdrawn'] and chosen['summary']['elapsedSeconds']<=controls['maxSeconds']
    model=baseline.clone(); observation=exposure.project_exposure(baseline)
    assert physical_state_hash(observation.planning_engine(controls))==physical_state_hash(baseline)
    previous=set(model.state['removed'])
    for action,receipt,expected in zip(chosen['actionProgram'],chosen['actionReceipts'],chosen['physicalStateHashes'],strict=True):
        result=model.step(copy.deepcopy(action))
        assert exposure._receipt(result)==receipt and physical_state_hash(model)==expected
        assert previous<=set(model.state['removed']); previous=set(model.state['removed'])
        assert action['slot']=='right' and action['operation'] in ('place','apply','withdraw')
        assert action['parameters']=={'radiusMm':1.5,'power':1.,'duty':1.} and 0.<action['duration']<=1.
    assert model.snapshot()==selected.snapshot()
    assert selected.actions[:len(baseline.actions)]==baseline.actions
    assert selected.state['elapsedSeconds']>baseline.state['elapsedSeconds']
    assert selected.state['cusaV3']['inputCapacityMm3']>baseline.state['cusaV3']['inputCapacityMm3']
    projected=exposure.project_exposure(selected).planning_engine(controls)
    assert physical_state_hash(projected)==physical_state_hash(selected)
    targets={r['id'] for r in selected.scene.regions if r.get('target') and r['material']=='tumor'}
    assert not any(int(selected.scene.labels.flat[c]) in targets for c in selected.state['removed'])
    assert str(native.root) not in json.dumps(public) and native.document['patient_id'] not in json.dumps(public)


@pytest.mark.parametrize('key,value', [('maxNonTargetMm3',0),('maxNonTargetMm3',201),('maxActions',5),
    ('maxActions',129),('maxActions',64.),('maxActions',True),('maxSeconds',1),('maxSeconds',181),
    ('maxCandidates',0),('maxCandidates',13),('maxProposals',0),('maxProposals',1025),
    ('maxPathMm',3.5),('maxPathMm',41),('maxWallSeconds',0),('maxWallSeconds',91),
    ('lateralSweepMm',-.1),('lateralSweepMm',1.1),('lateralSweepMm',np.nan),('maxSeconds',np.inf)])
def test_invalid_controls_refuse_before_model_work(controls, key, value):
    raw=copy.deepcopy(controls); raw[key]=value
    with pytest.raises(ValueError): exposure.normalize(raw)


@pytest.mark.parametrize('change',['extra-radius','extra-reset','missing'])
def test_exact_control_schema_cannot_hide_setup_or_reset(controls,change):
    raw=copy.deepcopy(controls)
    if change=='extra-radius': raw['radiusMm']=.1
    elif change=='extra-reset': raw['reset']=True
    else: raw.pop('maxActions')
    with pytest.raises(ValueError,match='exact bounded controls'): exposure.normalize(raw)


@pytest.mark.parametrize('method',['fresh','reset','replay'])
def test_public_task_cannot_reset_observed_anatomy_or_capacity(completed,controls,method):
    model=exposure.project_exposure(completed[0]).planning_engine(controls); before=model.snapshot()
    with pytest.raises(ValueError,match='cannot reset'):
        getattr(model,method)(*([[]] if method=='replay' else []))
    assert model.snapshot()==before


@pytest.mark.parametrize('change',['slot','operation','instrument','radius','parameter','activation','duration'])
def test_public_actions_cannot_escape_preparation_capability(intact,controls,change):
    model=exposure.project_exposure(intact[0]).planning_engine(controls); before=model.snapshot()
    action={'slot':'right','operation':'place','duration':.1,
            'pose':{'position':[45.,45.,-10.],'quaternion':[0.,0.,0.,1.]},'parameters':{'radiusMm':1.5,'power':1.,'duty':1.}}
    if change=='slot': action['slot']='left'
    elif change=='operation': action['operation']='configure'
    elif change=='instrument': action['instrument']='cranial_drill'
    elif change=='radius': action['parameters']['radiusMm']=.6
    elif change=='parameter': action['parameters']['flowMm3PerSecond']=1000.
    elif change=='activation': action['parameters']['power']=2.
    else: action['duration']=1.01
    result=model.step(action)
    assert result['status']=='rejected' and model.snapshot()==before


@pytest.mark.parametrize('limitation',['actions','time','remaining-native-actions','no-proposal'])
def test_failed_or_incomplete_execution_never_becomes_selected(intact,controls,limitation):
    observation=exposure.project_exposure(intact[0]); raw=copy.deepcopy(controls)
    if limitation=='actions': raw['maxActions']=6
    elif limitation=='time': raw['maxSeconds']=2.
    elif limitation=='no-proposal': raw['maxNonTargetMm3']=.1
    else:
        fields=thaw_json(observation.baseline.fields); fields['remainingActions']=2
        observation=exposure.ExposureObservation(PlanningObservation(observation.baseline.labels,observation.baseline.affine,fields))
    public=exposure.plan_observed(observation,raw)
    assert public['decision']['operation']=='STOP' and public['decision']['selectedIndex'] is None
    assert not any(a['complete'] for a in public['attempts'])


def test_incomplete_attempt_is_replayed_natively_but_no_state_is_installed(intact,controls):
    engine,native=intact; before=engine.snapshot(); raw={**controls,'maxActions':6}
    selected,report=exposure.prepare(engine,raw,native_collection=native.root)
    assert selected is None and report['nativeReplayVerified']
    assert report['publicPlan']['attempts'] and not report['publicPlan']['attempts'][-1]['complete']
    assert engine.snapshot()==before


def test_unexecuted_baseline_removal_cannot_acquire_forward_replay_authority(intact,controls):
    original,native=intact; engine=original.clone()
    cell=int(np.flatnonzero(engine.scene.labels.ravel())[0])
    engine.state['removed']=[cell]
    with pytest.raises(ValueError):
        exposure.prepare(engine,controls,native_collection=native.root)
    assert original.state['removed']==[]


@pytest.mark.parametrize('change',['schema','scope','inputs','seal','decision','complete','radius','duration','route-endpoint'])
def test_native_boundary_rejects_resealed_producer_claims(intact,controls,completed,monkeypatch,change):
    baseline,native=intact; public=copy.deepcopy(completed[1]['publicPlan'])
    attempt=public['attempts'][public['decision']['selectedIndex']]
    if change=='schema': public['schema']='resectionlab.public-source-exposure-preparation.v99'
    elif change=='scope': public['scope']='Clinical access proven'
    elif change=='inputs': public['controls']['maxNonTargetMm3']-=1.
    elif change=='decision': public['decision'].update(operation='STOP',selectedIndex=None)
    elif change=='complete':
        attempt.update(complete=False,failure='invented-failure'); public['decision'].update(operation='STOP',selectedIndex=None)
    elif change=='radius': attempt['actionProgram'][0]['parameters']['radiusMm']=.6
    elif change=='duration': attempt['actionProgram'][0]['duration']=2.
    elif change=='route-endpoint': attempt['route']['paths'][0]['endpointMm'][0]+=.25
    public['publicSeal']=semantic_digest({k:v for k,v in public.items() if k!='publicSeal'})
    if change=='seal': public['publicSeal']='0'*64
    monkeypatch.setattr(exposure,'plan_observed',lambda *args,**kwargs:copy.deepcopy(public))
    before=baseline.snapshot()
    with pytest.raises(ValueError): exposure.prepare(baseline,controls,native_collection=native.root)
    assert baseline.snapshot()==before


def test_empty_route_with_only_parking_cannot_claim_complete_preparation(intact,controls,monkeypatch):
    baseline,native=intact; public=exposure.plan_observed(exposure.project_exposure(baseline),controls)
    attempt=public['attempts'][public['decision']['selectedIndex']]; model=baseline.clone()
    attempt['route']['paths']=[]
    attempt['actionProgram']=copy.deepcopy(attempt['actionProgram'][-2:])
    attempt['actionReceipts']=[]; attempt['physicalStateHashes']=[]
    for action in attempt['actionProgram']:
        attempt['actionReceipts'].append(exposure._receipt(model.step(action)))
        attempt['physicalStateHashes'].append(physical_state_hash(model))
    attempt['summary']=exposure._summary(model,baseline.state)
    assert attempt['summary']['additionalNonTargetMm3']==0 and attempt['summary']['withdrawn']
    public['publicSeal']=semantic_digest({k:v for k,v in public.items() if k!='publicSeal'})
    monkeypatch.setattr(exposure,'plan_observed',lambda *args,**kwargs:copy.deepcopy(public))
    with pytest.raises(ValueError,match='burden bounds|completeness'):
        exposure.prepare(baseline,controls,native_collection=native.root)
    assert not exposure._complete_program([],{'paths':[]},baseline.state['slots']['right'])


def test_side_sweep_uses_multiple_real_fixed_radius_paths(intact,controls):
    observation=exposure.project_exposure(intact[0])
    raw={**controls,'lateralSweepMm':.3,'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.}
    rows=exposure._shortlist(observation,exposure.normalize(raw),lambda:None)
    assert rows
    route=rows[0]; paths=route['paths']; assert len(paths)==5
    center=np.asarray(paths[0]['outsideMm']); axis=np.asarray(route['axisRasUnit'])
    for path in paths[1:]:
        offset=np.asarray(path['outsideMm'])-center
        assert np.linalg.norm(offset)==pytest.approx(.3) and offset@axis==pytest.approx(0.,abs=1e-10)
    attempt=exposure._compile(observation.planning_engine(raw),route,raw,lambda:None)
    assert attempt['summary']['additionalTargetMm3']==0.
    assert all(a['parameters']['radiusMm']==1.5 for a in attempt['actionProgram'])
    assert sum(a['operation']=='place' for a in attempt['actionProgram'])>=2
    assert attempt['complete'] or attempt['failure'] is not None


@pytest.mark.parametrize('burden',['target','non-target'])
def test_compiler_preserves_failed_burden_evidence_instead_of_selecting_it(intact,controls,burden):
    observation=exposure.project_exposure(intact[0])
    route=exposure._shortlist(observation,controls,lambda:None)[0]
    raw=copy.deepcopy(controls)
    if burden=='target': route['paths'][0]['endpointMm']=route['sourceTargetMm'][:]
    else: raw['maxNonTargetMm3']=1.
    attempt=exposure._compile(observation.planning_engine(raw),route,raw,lambda:None)
    assert not attempt['complete'] and attempt['failure']=='PREPARATION_BURDEN_LIMIT'
    assert attempt['actionReceipts'] and len(attempt['physicalStateHashes'])==len(attempt['actionProgram'])
    if burden=='target': assert attempt['summary']['additionalTargetMm3']>0.
    else: assert attempt['summary']['additionalNonTargetMm3']>raw['maxNonTargetMm3']
    assert intact[0].state['removed']==[]


def test_private_references_and_caller_control_callback_cannot_change_frozen_plan(intact,controls,completed):
    engine,native=intact; changed=engine.clone()
    changed._private_evaluation_truth=np.ones(engine.scene.shape,bool)
    changed._private_evaluator=lambda:pytest.fail('Private evaluator was read')
    raw=copy.deepcopy(controls); calls=0
    def callback():
        nonlocal calls
        calls+=1
        raw['maxNonTargetMm3']=0.; raw['maxActions']=0; raw['radiusMm']=.1
    selected,report=exposure.prepare(changed,raw,native_collection=native.root,check=callback)
    assert calls>0 and selected is not None
    assert report['publicPlan']==completed[1]['publicPlan']
    assert selected.snapshot()==completed[0].snapshot()


def test_cancellation_during_native_attempt_propagates_without_changing_source(intact,controls,monkeypatch):
    engine,native=intact; before=engine.snapshot(); original=SurgicalEngine.step; native_attempts=[]
    def cancel_native(model,action,**kwargs):
        if type(model) is SurgicalEngine and action.get('operation')=='apply':
            native_attempts.append(action); raise Cancelled('native prefix cancelled')
        return original(model,action,**kwargs)
    monkeypatch.setattr(SurgicalEngine,'step',cancel_native)
    with pytest.raises(Cancelled,match='native prefix cancelled'):
        exposure.prepare(engine,controls,native_collection=native.root)
    assert native_attempts and engine.snapshot()==before


def test_wall_budget_cannot_publish_a_partial_study(intact,controls,monkeypatch):
    engine,native=intact; before=engine.snapshot(); ticks=iter([0.,1000.])
    monkeypatch.setattr(exposure.time,'monotonic',lambda:next(ticks))
    with pytest.raises(ValueError,match='computation budget exhausted'):
        exposure.prepare(engine,controls,native_collection=native.root)
    assert engine.snapshot()==before


def test_callback_native_state_mutation_is_refused(intact,controls):
    original,native=intact; engine=original.clone(); changed=False
    def callback():
        nonlocal changed
        if not changed:
            changed=True
            assert engine.step({'slot':'right','operation':'inspect','duration':.1})['status']=='accepted'
    with pytest.raises(ValueError,match='checkpoint changed|inputs changed|replay'):
        exposure.prepare(engine,controls,native_collection=native.root,check=callback)
    assert changed
