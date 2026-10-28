"""Native contact, bimanual held-cell and complete retractor cleanup contracts."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_staged_skills import SkillLimits, make_skill_checkpoint, restore_skill_checkpoint
from resectionlab.surgical_transition_meter import TransitionMeter
from resectionlab.surgical_retraction_skills import (
    acquire_retractor, held_move, release_retractor, checked_pair_action, shaft_clearance)


def fixture():
    labels=np.zeros((24,24,20),np.uint16)
    labels[8,8,8]=1
    scene=SurgicalScene(labels,np.eye(4),
        [{'id':1,'name':'brain','material':'soft_tissue','target':False}],
        {'kind':'generated native retraction skill fixture'}, {'tissuePhysicsV2':default_profile()})
    engine=SurgicalEngine(scene)
    pose=lambda z:{'position':[8.,8.,z],'quaternion':[0.,0.,0.,1.]}
    proposal={'sourceSceneId':scene.identity,'outsidePose':pose(2.),'contactPose':pose(8.)}
    return engine,proposal


def test_catalog_retractor_acquires_one_step_contact_moves_and_completely_withdraws():
    engine,proposal=fixture()
    original=engine.snapshot()
    labels=engine.scene.labels.copy()
    meter=TransitionMeter(200)
    limits=SkillLimits(240,200)
    held,receipt=acquire_retractor(meter.wrap(engine),proposal,limits=limits,meter=meter,check=lambda:None)
    assert receipt['status']=='accepted',receipt['reason']
    binding=receipt['binding']
    assert binding['heldCells'] and shaft_clearance(held)['clear']
    last=held.history[-1]
    assert last['action']['operation']=='hold' and last['action']['duration']==.05
    assert held.state['holds']['right']['origin']==held.state['slots']['right']['position']
    assert held.parameters(held.state['slots']['right'])['radiusMm']==3.
    assert any(x['reason']=='RETAINED_TISSUE_COLLISION' for x in receipt['attempts'])
    position=np.asarray(binding['origin'])+[0.,.5,0.]
    moved,move=held_move(held,position,binding=binding,limits=limits,meter=meter,check=lambda:None)
    assert move['status']=='accepted'
    assert moved.metrics()['brainMaxDisplacementMm']>0
    assert moved.metrics()['nonTargetRemovedMm3']==0.
    seals={'source_seal':'sha256:'+'1'*64,'task_seal':'sha256:'+'2'*64,
           'inventory_seal':'sha256:'+'3'*64}
    checkpoint=make_skill_checkpoint(moved,limits=limits,meter=meter,stage='held',**seals)
    resumed,resumed_limits,ledger=restore_skill_checkpoint(moved.scene,checkpoint,meter=meter,**seals)
    assert resumed.state_id==moved.state_id and resumed.state['holds']==moved.state['holds']
    assert ledger['spentPublicCalls']==meter.calls
    clean,release=release_retractor(resumed,proposal,binding=binding,limits=resumed_limits,meter=meter,check=lambda:None)
    assert release['status']=='accepted',release['reason']
    assert not clean.state['holds'] and not clean.state['slots']['right']['inserted']
    assert all(not s['inserted'] and not s['active'] for s in clean.state['slots'].values())
    assert SurgicalEngine.restore(clean.scene,clean.snapshot()).state==clean.state
    assert np.array_equal(engine.scene.labels,labels)
    assert engine.snapshot()==original


def test_cusa_cannot_remove_retractor_held_brain_even_below_volume_cap():
    engine,proposal=fixture()
    meter=TransitionMeter(200)
    limits=SkillLimits(240,200)
    branch,receipt=acquire_retractor(meter.wrap(engine),proposal,limits=limits,meter=meter,check=lambda:None)
    binding=receipt['binding']
    for action in (
        {'slot':'left','operation':'exchange','instrument':'cusa','duration':.1},
        {'slot':'left','operation':'place','pose':{'position':[8.,8.,14.],'quaternion':[1.,0.,0.,0.]},
         'parameters':{'radiusMm':.6},'duration':.1}):
        branch,out=checked_pair_action(branch,action,retractor_hand='right',expected_hold=binding,
            limits=limits,meter=meter,check=lambda:None)
        assert out['status']=='accepted'
    action={'slot':'left','operation':'apply',
        'pose':{'position':[8.,8.,8.],'quaternion':[1.,0.,0.,0.]},'duration':.5}
    before=branch.snapshot()
    # The native removal mechanism allows this; the retraction skill additionally
    # preserves its source-bound hold rather than manufacturing a cleared patch.
    assert branch.preview(action)['metrics']['nonTargetRemovedMm3']==1.
    refused,out=checked_pair_action(branch,action,retractor_hand='right',expected_hold=binding,
        limits=limits,meter=meter,check=lambda:None)
    assert out['status']=='refused' and out['reason']=='FORBIDDEN_SOURCE_CELL_REMOVAL'
    assert refused.snapshot()==before==branch.snapshot()
    assert refused.state['holds']['right']['cells']==binding['heldCells']


def test_retractor_has_no_implicit_profile_or_changed_source_admission():
    engine,proposal=fixture()
    meter=TransitionMeter(20)
    limits=SkillLimits(40,20)
    wrong=copy.deepcopy(proposal);wrong['sourceSceneId']='unrelated'
    with pytest.raises(ValueError,match='another source'):
        acquire_retractor(meter.wrap(engine),wrong,limits=limits,meter=meter,check=lambda:None)
    assert meter.calls==0 and not engine.actions


def test_native_outside_placement_refusal_recovers_outward_without_shrinking_blade():
    from scipy.spatial.transform import Rotation
    labels=np.zeros((24,24,20),np.uint16);labels[10,8,4]=1
    scene=SurgicalScene(labels,np.eye(4),
        [{'id':1,'name':'brain','material':'soft_tissue','target':False}],
        {'kind':'generated placement recovery fixture'},{'tissuePhysicsV2':default_profile()})
    engine=SurgicalEngine(scene)
    q=Rotation.from_euler('z',90.,degrees=True).as_quat().tolist()
    proposal={'sourceSceneId':scene.identity,
        'outsidePose':{'position':[8.,8.,6.],'quaternion':q},
        'contactPose':{'position':[8.,8.,16.],'quaternion':q}}
    meter=TransitionMeter(200)
    branch,receipt=acquire_retractor(meter.wrap(engine),proposal,limits=SkillLimits(240,200),
        meter=meter,check=lambda:None)
    assert receipt['status']=='accepted',receipt['reason']
    assert receipt['proposalUsed']['outsideRecoveryOffsetMm']==8.
    assert receipt['proposalUsed']['outsidePose']['position']==[8.,8.,-2.]
    attempts=[entry for part in receipt['attempts'] for entry in part['attempts']
              if entry['action']['operation']=='place']
    assert len(attempts)==3
    assert all(entry['action']['parameters']['radiusMm']==3. for entry in attempts)
    assert all(entry['reason'].startswith('PLACE_OUTSIDE_MATERIAL:') for entry in attempts[:2])
    assert attempts[-1]['status']=='committed'
    assert branch.state['holds']['right']['cells']
    clean, release = release_retractor(branch,receipt['proposalUsed'],binding=receipt['binding'],
        limits=SkillLimits(240,200),meter=meter,check=lambda:None)
    assert release['status']=='accepted',release['reason']
    assert clean.state['slots']['right']['position']==[8.,8.,-2.]
    assert not clean.state['holds'] and not clean.state['slots']['right']['inserted']
    assert not engine.actions
