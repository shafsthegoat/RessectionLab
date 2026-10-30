"""Small conditional endpoint controls; no source replay or solver calls."""
import copy
import numpy as np
import pytest

from resectionlab.geometry import GeometryScene
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_contact_survey import ConditionalContactSurvey,require_sample_join,require_supported_context


def fixture():
    labels=np.zeros((8,8,8),np.uint16);labels[3,3,3]=1;labels[6,3,3]=2
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'brain','material':'soft_tissue','target':False},
        {'id':2,'name':'vessel','material':'vessel','target':False}],{'kind':'generated contact-survey control'}, {})
    cell=int(np.ravel_multi_index((3,3,3),scene.shape));outside=int(np.ravel_multi_index((6,3,3),scene.shape))
    frame={'holds':{},'detachedFragments':[],'removedSourceIds':[],'withinTick':0,'atCommittedTickEnd':True,
        'mechanics':{'nativeFixedSteps':1},'protectedSourceIds':[],
        'slots':{'left':{'instrument':'cusa','position':[3.,3.,3.],'quaternion':[0.,0.,0.,1.]}},
        'parametersBySlot':{'left':{'radiusMm':1.5,'depthMm':2.,'contactGeometryRevision':2}}}
    return scene,GeometryScene(np.zeros(scene.shape,bool),scene.affine),frame,cell,outside


def test_all_source_coverage_includes_outside_field_material_and_removed_is_excluded():
    scene,grid,frame,cell,outside=fixture();survey=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),np.zeros((1,8,3)))
    contact=survey.capsule([6.,3.,3.],[6.,3.,3.],.1)
    assert contact['witnessedSourceIds']==[outside] and contact['outsideFieldCandidateSourceIds']==[outside]
    assert contact['rasterSourceIds']==[outside] and contact['completeDeclaredSourceCoverage']
    frame['removedSourceIds']=[outside]
    gone=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),np.zeros((1,8,3))).capsule([6.,3.,3.],[6.,3.,3.],.1)
    assert gone['witnessedSourceIds']==gone['indeterminateSourceIds']==gone['rasterSourceIds']==[]


def test_continuous_point_contact_exposes_center_raster_blind_spot():
    scene,grid,frame,cell,_=fixture();u=np.zeros((1,8,3));u[:,:,0]=.1
    survey=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),u)
    contact=survey.capsule([4.04,3.,3.],[4.04,3.,3.],.5)
    assert contact['rasterSourceIds']==[] and contact['witnessedSourceIds']==[cell]


def test_shaft_exempts_only_same_hand_hold_and_protected_contact_remains_visible():
    scene,grid,frame,cell,_=fixture();frame['slots']['left']['position']=[3.,3.,4.]
    frame['holds']={'right':{'target':[0.,0.,0.],'cells':[cell]}};frame['protectedSourceIds']=[cell]
    foreign=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),np.zeros((1,8,3))).run()
    shaft=foreign['capsules'][0];assert shaft['part']=='shaft' and shaft['forbiddenWitnessedSourceIds']==[cell]
    frame['holds']={'left':{'target':[0.,0.,0.],'cells':[cell]}}
    own=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),np.zeros((1,8,3))).run()
    assert own['capsules'][0]['forbiddenWitnessedSourceIds']==[]
    assert own['capsules'][0]['protectedWitnessedSourceIds']==[cell]
    assert not own['largerDomainPublicActionsAdmitted'] and not own['completeActionChecks'] and not own['cusaEffectValidated']
    assert not own['unconfiguredCriticalAnatomyValidated']


@pytest.mark.parametrize('change',[
    {'detachedFragments':[{'cells':[1],'offset':[0,0,0]}]},
    {'holds':{'left':{'cells':[1],'offset':[0,0,0]}}},
    {'withinTick':1},{'atCommittedTickEnd':False}])
def test_unsupported_rigid_states_and_mismatched_phases_refuse(change):
    _,_,frame,_,_=fixture();frame.update(change)
    with pytest.raises(ValueError):require_supported_context(frame)


@pytest.mark.parametrize('key,value',[('withinTick',1),('fixedStepSeconds',.1),('operation','apply'),('slot','right'),
    ('sourceOperatorIdentity','other'),('sampleFieldsSha256','other'),('tissueState',2)])
def test_exact_field_phase_join_refuses_consistent_looking_metadata_changes(key,value):
    _,_,frame,_,_=fixture();frame.update(branch='short',coupledStep=1,sampleFieldsFile='field.npz',sampleFieldsSha256='hash',
        tissueState=0,fixedStepSeconds=1/120,actionIndex=15,operation='move',slot='left',sourceOperatorIdentity='operator')
    sample={'branch':'short','step':1,'file':'field.npz','sha256':'hash','tissueState':0}
    row={'branch':'short','step':1,'withinTick':0,'h':1/120,'actionIndex':15,'operation':'move','slot':'left',
        'tissueState':0,'sourceOperatorIdentity':'operator'}
    require_sample_join(frame,sample,row);frame[key]=value
    with pytest.raises(ValueError):require_sample_join(frame,sample,row)


def test_ambiguous_working_contact_and_negative_obstacle_survey_do_not_admit_any_action():
    scene,grid,frame,cell,_=fixture();frame['slots']['left']['position']=[5.,3.,3.]
    result=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),np.zeros((1,8,3))).run()
    work=next(c for c in result['capsules'] if c['part']=='working')
    assert cell in work['indeterminateSourceIds'] and not work['workingContactLegalityValidated']
    assert result['tissueObstaclePointSurvey']['blockingPointWitnesses']==[]
    assert not result['tissueObstaclePointSurvey']['unknownAnatomyValidated']


def test_survey_is_read_only_and_candidate_order_matches_existing_cusa_rule():
    from types import SimpleNamespace
    from resectionlab.surgical_cusa_v3 import _ordered_contacts
    scene,grid,frame,cell,_=fixture();before=copy.deepcopy(frame);u=np.zeros((1,8,3));u[:,:,0]=.1
    survey=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),u)
    fake=SimpleNamespace(scene=scene,by_label=survey.by_label,state={'holds':{},'removed':[],
        'deformation':{str(cell):u[0].mean(axis=0).tolist()}},touched=lambda a,b,r:survey.occupancy.material_ids(survey.grid_cover(a,b,r)))
    expected=_ordered_contacts(fake,[cell],np.array(frame['slots']['left']['position']))
    assert survey.cusa_order('left')==expected==[cell]
    survey.run();assert frame==before and np.array_equal(u[:,:,0],np.full((1,8),.1))


def test_grid_frame_mismatch_refuses_before_claiming_whole_source_coverage():
    scene,_,frame,cell,_=fixture();affine=np.eye(4);affine[0,3]=100.
    wrong=GeometryScene(np.zeros(scene.shape,bool),affine)
    with pytest.raises(ValueError,match='GRID_FRAME'):
        ConditionalContactSurvey(scene,wrong,frame,np.array([cell]),np.zeros((1,8,3)))


def test_caller_context_mutation_does_not_change_owned_exemptions_or_poses():
    scene,grid,frame,cell,_=fixture();frame['slots']['left']['position']=[3.,3.,4.]
    survey=ConditionalContactSurvey(scene,grid,frame,np.array([cell]),np.zeros((1,8,3)));before=survey.run()
    frame['holds']={'left':{'target':[0.,0.,0.],'cells':[cell]}}
    frame['slots']['left']['position']=[0.,0.,0.]
    assert survey.run()==before
