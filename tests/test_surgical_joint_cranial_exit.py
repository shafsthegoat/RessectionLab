"""Actual native joint exits; generated stock controls, no patient admission."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_craniotomy import PROPERTY,binding,contract_hash
from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
from resectionlab.surgical_rl_protection import ProtectionAnnotation,ProtectionSet,GuardConfig
from resectionlab.surgical_joint_cranial_skills import (
    freeze_joint_cranial_limits,checked_joint_actions,checked_joint_cusa_motion)
from resectionlab.surgical_joint_cranial_exit import find_checked_joint_exit,joint_exit_complete
from resectionlab.surgical_transition_meter import TransitionMeter
from test_surgical_membrane_transport import joint_source,bone_actions,membrane_cut_actions,membrane_grasp_actions
from test_surgical_bone_transport import pose,request


@pytest.fixture(scope='module')
def joint_work():
    original,c,contour,p,r=joint_source(curved=True)
    labels=original.scene.labels.copy();labels[32,32,66]=4
    regions=original.scene.regions+[{'id':4,'name':'Generated healthy W contact','material':'soft_tissue'}]
    properties=copy.deepcopy(original.scene.properties);raw=properties.pop(PROPERTY)
    base=SurgicalScene(labels,original.scene.affine,regions,original.scene.source,properties)
    raw.update(binding(base));raw['contentHash']=contract_hash(raw);properties[PROPERTY]=raw
    scene=SurgicalScene(labels,base.affine,regions,base.source,properties)
    base=SurgicalEngine(scene)
    contract=freeze_cranial_stage_contract(base,membrane_cells=p,retained_membrane_cells=r,
        membrane_anchor_cells=c.membrane_anchor_cells,
        work_cells=tuple(sorted((*c.work_cells,int(np.ravel_multi_index((32,32,66),labels.shape))))),
        evidence=c.as_dict()['evidence'])
    source=SurgicalEngine(scene,cranial_stages=contract)
    initial=source.snapshot();meter=TransitionMeter(1200)
    limits=freeze_joint_cranial_limits(source,meter=meter,max_bone_mm3=285.,
        max_preparation_nonbone_mm3=113.,max_work_nonbone_mm3=20.,
        max_native_actions=360,max_public_calls=1200,max_wall_seconds=180.)
    def checked(e,aa,name):
        n,receipt=checked_joint_actions(e,aa,limits=limits,meter=meter,phase='unit.exit.setup.'+name)
        assert receipt['status']=='accepted',receipt
        return n
    bone=checked(source,bone_actions(contour),'bone')
    cut=checked(bone,membrane_cut_actions(bone,contract),'cuts')
    held=checked(cut,membrane_grasp_actions(),'held')
    opened=checked(held,[request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))],'extract')
    ready=checked(opened,[request('exchange',instrument='cusa'),request('place',pose=pose())],'work-entry')
    work,receipt=checked_joint_cusa_motion(ready,request('apply',pose=pose(z=66.5),duration=30.),
        limits=limits,meter=meter,phase='unit.exit.setup.work',dwell_seconds=30.)
    assert receipt['status']=='accepted',receipt
    assert receipt['ledger']['wholeNonTargetRemovedMm3']==399. and receipt['ledger']['targetRemovedMm3']==18.
    assert source.snapshot()==initial
    return SimpleNamespace(source=source,initial=initial,work=work,bone=bone,held=held,
        meter=meter,limits=limits,contract=contract)


def test_actual_joint_work_exit_keeps_b285_p113_w1_and_native_replay(joint_work):
    f=joint_work;before=f.work.snapshot();calls=f.meter.calls
    exited,receipt=find_checked_joint_exit(f.work,limits=f.limits,meter=f.meter,phase='unit.exit.positive')
    assert receipt['status']=='accepted' and receipt['admittedExit'] and joint_exit_complete(exited)
    assert receipt['stageReadiness']['wholeExitComplete'] and receipt['stageReadiness']['duraCleared']
    assert [receipt['ledger'][k] for k in ('boneRemovedMm3','preparationNonboneRemovedMm3','workNonTargetRemovedMm3','wholeNonTargetRemovedMm3','targetRemovedMm3')]==[285.,113.,1.,399.,18.]
    assert receipt['sequenceOutcome']['dimensions']['nonTargetRemovedMm3']['cumulative']==399.
    assert receipt['ledger']['spentPublicCalls']==f.meter.calls and receipt['publicCallCost']==f.meter.calls-calls
    assert not receipt['admittedPatientTask'] and f.work.snapshot()==before and f.source.snapshot()==f.initial
    replay=f.meter.wrap(f.source.fresh())
    with f.meter.phase('unit.exit.verification.fresh-replay'):
        for action,event in zip(exited.actions,exited.history):
            result=replay.step(action)
            assert result['status']==event['status'] and result.get('reason')==event.get('reason') and result['stateId']==event['stateAfter']
    assert replay.snapshot()==exited.snapshot()


def test_intermediate_bone_exit_does_not_claim_membrane_clearance(joint_work):
    f=joint_work;calls=f.meter.calls
    result,receipt=find_checked_joint_exit(f.bone,limits=f.limits,meter=f.meter)
    assert receipt['status']=='accepted' and receipt['mode']=='already_exterior' and receipt['admittedExit']
    assert not receipt['stageReadiness']['duraCleared'] and not receipt['stageReadiness']['workEntrySupported']
    assert receipt['ledger']['wholeNonTargetRemovedMm3']==285. and receipt['ledger']['preparationNonboneRemovedMm3']==0.
    assert f.meter.calls==calls and result.state_id==f.bone.state_id


def test_unposed_inactive_slots_are_legitimate_but_no_task_is_admitted(joint_work):
    f=joint_work;calls=f.meter.calls
    result,receipt=find_checked_joint_exit(f.source,limits=f.limits,meter=f.meter)
    assert receipt['status']=='accepted' and joint_exit_complete(result) and receipt['publicCalls']==0
    assert not receipt['stageReadiness']['workEntrySupported'] and not receipt['admittedPatientTask']
    assert f.meter.calls==calls and f.source.snapshot()==f.initial


def test_protected_lateral_withdrawal_refuses_after_charged_deactivation(joint_work):
    f=joint_work;scene=f.source.scene
    # Separate early-stage control: a frozen no-go in extracranial air. The
    # accepted placement/activation is clear; the requested lateral exit shaft
    # crosses it. No later preparation sweeps are asserted safe in this scene.
    mask=np.zeros(scene.shape,bool);mask[34,32,30]=True
    protection=ProtectionSet(scene,[ProtectionAnnotation.from_mask(scene,'generated-exit-shaft-exclusion',mask,
        provenance='Generated extracranial hard-guard control, not anatomical coverage')],clearance_mm=0.,required=())
    guard=GuardConfig(require_critical_coverage=False);base=SurgicalEngine(scene,protection=protection,guard_config=guard)
    c=f.contract;bound=freeze_cranial_stage_contract(base,membrane_cells=c.membrane_cells,
        retained_membrane_cells=c.retained_membrane_cells,membrane_anchor_cells=c.membrane_anchor_cells,
        work_cells=c.work_cells,evidence=c.as_dict()['evidence'])
    source=SurgicalEngine(scene,protection=protection,guard_config=guard,cranial_stages=bound)
    meter=TransitionMeter(100);limits=freeze_joint_cranial_limits(source,meter=meter,max_bone_mm3=285.,
        max_preparation_nonbone_mm3=113.,max_work_nonbone_mm3=20.,max_native_actions=30,max_public_calls=100,max_wall_seconds=30.)
    prefix,setup=checked_joint_actions(source,[request('configure',mechanicsRevision=2,clockRevision=1),
        request('place',pose=pose()),request('activate')],limits=limits,meter=meter)
    assert setup['status']=='accepted',setup
    before=prefix.snapshot();calls=meter.calls
    stopped,receipt=find_checked_joint_exit(prefix,limits=limits,meter=meter,
        outside_poses={'left':pose(35,32,55)},max_candidates=1,phase='unit.exit.protected')
    assert receipt['status']=='refused' and not receipt['admittedExit'] and not receipt['exitComplete']
    checked=receipt['attempts'][0]['checked']
    assert checked['reason'].split(':')[0]=='PROTECTED_STRUCTURE',checked
    assert meter.calls-calls==receipt['publicCallCost']==3
    assert len(stopped.actions)==len(prefix.actions)+1 and stopped.actions[-1]['operation']=='deactivate'
    assert stopped.state['removed']==prefix.state['removed'] and receipt['ledger']['wholeNonTargetRemovedMm3']==0.
    assert prefix.snapshot()==before and source.actions==[]


def test_shared_call_reserve_stops_with_actual_accepted_prefix(joint_work):
    f=joint_work;before=f.work.snapshot();calls=f.meter.calls
    with f.meter.reserve(f.meter.remaining-2):
        stopped,receipt=find_checked_joint_exit(f.work,limits=f.limits,meter=f.meter,phase='unit.exit.reserve')
    assert receipt['status']=='interrupted' and 'TransitionReserveReached' in receipt['reason']
    assert not receipt['admittedExit'] and len(stopped.actions)==len(f.work.actions)+1
    assert stopped.actions[-1]['operation']=='deactivate' and f.meter.calls==calls+2
    assert receipt['ledger']['wholeNonTargetRemovedMm3']==399. and f.work.snapshot()==before


def test_cancellation_after_real_commit_preserves_prefix_and_costs(joint_work):
    f=joint_work;calls=f.meter.calls;before=f.work.snapshot();phase='unit.exit.cancel'
    def cancel():
        prior=f.meter.last_committed_engine(phase=phase+'.candidate.0')
        if prior is not None and len(prior.actions)>len(f.work.actions):
            raise InterruptedError('stop after actual exit deactivation')
    stopped,receipt=find_checked_joint_exit(f.work,limits=f.limits,meter=f.meter,check=cancel,phase=phase)
    assert receipt['status']=='interrupted' and 'stop after actual' in receipt['reason']
    assert not receipt['admittedExit'] and len(stopped.actions)==len(f.work.actions)+1
    assert f.meter.calls==calls+2 and f.work.snapshot()==before
    assert receipt['ledger']['wholeNonTargetRemovedMm3']==399.


def test_late_receipt_serialization_wall_crossing_cannot_admit(joint_work,monkeypatch):
    import resectionlab.surgical_joint_cranial_exit as module
    f=joint_work;calls=f.meter.calls;serialized=False;real=module.json.dumps
    def serialize(value,**kwargs):
        nonlocal serialized
        result=real(value,**kwargs);serialized=True;return result
    monkeypatch.setattr(module,'json',SimpleNamespace(dumps=serialize))
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:f.limits.frozen_at+(f.limits.max_wall_seconds+1 if serialized else .1)))
    result,receipt=find_checked_joint_exit(f.bone,limits=f.limits,meter=f.meter)
    assert serialized and receipt['status']=='interrupted' and receipt['reason']=='WALL_BUDGET'
    assert receipt['exitComplete'] and not receipt['admittedExit'] and not receipt['ledger']['withinLimits']
    assert f.meter.calls==calls and result.state_id==f.bone.state_id


def test_held_or_released_whole_patch_is_not_an_exit(joint_work):
    f=joint_work;calls=f.meter.calls
    held,receipt=find_checked_joint_exit(f.held,limits=f.limits,meter=f.meter)
    assert receipt['status']=='refused' and receipt['reason']=='JOINT_EXIT_FRAGMENT_TRANSPORT_REQUIRED'
    assert not receipt['exitComplete'] and f.meter.calls==calls
    released,r=checked_joint_actions(f.held,[request('move',pose=pose(z=61.5)),request('release')],limits=f.limits,meter=f.meter,phase='unit.exit.release-control')
    assert r['status']=='accepted' and len(released.state['detachedFragments'][0]['cells'])==113
    calls=f.meter.calls;result,receipt=find_checked_joint_exit(released,limits=f.limits,meter=f.meter)
    assert receipt['status']=='refused' and not receipt['exitComplete'] and f.meter.calls==calls
    assert result.state['detachedFragments']==released.state['detachedFragments'] and receipt['ledger']['wholeNonTargetRemovedMm3']==285.


def test_both_posed_tools_are_checked_and_deactivated(joint_work):
    f=joint_work
    both,r=checked_joint_actions(f.work,[request('place',slot='right',pose=pose(32,36,55)),request('activate',slot='right')],limits=f.limits,meter=f.meter,phase='unit.exit.twohands')
    assert r['status']=='accepted'
    result,receipt=find_checked_joint_exit(both,limits=f.limits,meter=f.meter)
    assert receipt['status']=='accepted' and joint_exit_complete(result)
    assert all(not s['active'] and not s['inserted'] and not s['grip'] for s in result.state['slots'].values())
    assert not result.state['holds'] and not result.state.get('detachedFragments')


def test_foreign_meter_cannot_drop_preparation_cost(joint_work):
    f=joint_work;calls=f.meter.calls
    with pytest.raises(ValueError,match='SHARED_METER|LIMIT'):
        find_checked_joint_exit(f.work,limits=f.limits,meter=TransitionMeter(1200))
    assert f.meter.calls==calls


def test_already_exterior_new_stock_scissors_needs_no_incision_depth_or_entry_credit():
    source,c,contour,p,r=joint_source(curved=True);meter=TransitionMeter(100)
    limits=freeze_joint_cranial_limits(source,meter=meter,max_bone_mm3=285.,
        max_preparation_nonbone_mm3=113.,max_work_nonbone_mm3=20.,max_native_actions=60,
        max_public_calls=100,max_wall_seconds=30.)
    prefix,setup=checked_joint_actions(source,bone_actions(contour)+[request('exchange',instrument='dural_scissors')],
        limits=limits,meter=meter)
    assert setup['status']=='accepted' and c.scissors_depth_mm==4.
    assert prefix.parameters(prefix.state['slots']['left'])['depthMm']==2.
    calls=meter.calls;outside,receipt=find_checked_joint_exit(prefix,limits=limits,meter=meter)
    assert receipt['status']=='accepted' and receipt['mode']=='already_exterior'
    assert receipt['admittedExit'] and not receipt['stageReadiness']['workEntrySupported']
    assert meter.calls==calls and outside.state_id==prefix.state_id and source.actions==[]
