"""Independent public-native search controls; generated sources, no patient tasks."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest

from resectionlab.core import CaseData,SourceRef,thaw_json
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_craniotomy import PROPERTY,binding,contract_hash
from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
from resectionlab.surgical_transition_meter import TransitionMeter
from resectionlab.surgical_joint_cranial_skills import (
    checked_joint_actions,checked_joint_cusa_motion,freeze_joint_cranial_limits)
from resectionlab.surgical_joint_cranial_exit import find_checked_joint_exit
from resectionlab.surgical_joint_cranial_search import (
    freeze_joint_resection_goal,checked_joint_sequence_search,joint_goal_assessment)
from test_surgical_joint_cranial_skills import prepared,preparation_program
from test_surgical_membrane_transport import joint_source
from test_surgical_bone_transport import pose,request


def cusa_proposal(engine,name='checked-work',z=66.5):
    return {'id':name,'sourceStateId':engine.state_id,'kind':'cusa',
            'motion':request('apply',pose=pose(z=z),duration=30.)}


@pytest.fixture(scope='module')
def ready():
    source,contract,contour,p,r,meter,limits=prepared(max_public_calls=800,max_wall_seconds=240.)
    goal=freeze_joint_resection_goal(source,limits=limits)
    strict=freeze_joint_resection_goal(source,limits=limits,minimum_target_fraction=1.)
    spatial=freeze_joint_resection_goal(source,limits=limits,minimum_zones=8)
    initial=source.snapshot()
    engine,receipt=checked_joint_actions(source,preparation_program(source,contract,contour)+[
        request('exchange',instrument='cusa'),request('place',pose=pose())],limits=limits,meter=meter,phase='independent.search.preparation')
    assert receipt['status']=='accepted' and receipt['ledger']['wholeNonTargetRemovedMm3']==334.
    assert source.snapshot()==initial
    return SimpleNamespace(source=source,initial=initial,engine=engine,meter=meter,limits=limits,goal=goal,strict=strict,spatial=spatial)


@pytest.fixture(scope='module')
def completed(ready):
    f=ready;before=f.engine.snapshot()
    result=checked_joint_sequence_search(f.engine,propose=lambda e:[cusa_proposal(e)],
        limits=f.limits,goal=f.goal,meter=f.meter,horizon=1,phase='independent.search.completed')
    assert result.selected is not None,result.evidence
    assert f.engine.snapshot()==before and f.source.snapshot()==f.initial
    return result


def test_actual_fraction_and_frozen_spatial_denominator_remain_separate(ready,completed):
    f=ready;node=completed.selected;exit_receipt=node.path[-1]['exit']
    whole=joint_goal_assessment(node.exited,goal=f.strict,limits=f.limits,exit_receipt=exit_receipt)
    assert not whole['completedFrozenTask'] and whole['creditedOriginalTargetCells']==18
    assert whole['originalTargetCells']==27 and whole['originalTargetTreatmentFraction']==pytest.approx(2/3)
    assert 'ORIGINAL_TARGET_FRACTION_BELOW_GOAL' in whole['reasons']
    spatial=joint_goal_assessment(node.exited,goal=f.spatial,limits=f.limits,exit_receipt=exit_receipt)
    assert sum(n>0 for n in spatial['zoneTreatmentCreditCells'])==8
    assert spatial['zonesReachingRequiredOriginalFraction']<8
    assert not spatial['completedFrozenTask'] and 'ORIGINAL_SPATIAL_PROGRESS_BELOW_GOAL' in spatial['reasons']
    assert not whole['admittedPatientTask'] and not spatial['admittedPatientTask']


def test_real_distinct_work_candidates_compete_without_losing_preparation(ready):
    f=ready;before=f.engine.snapshot()
    result=checked_joint_sequence_search(f.engine,propose=lambda e:[
        cusa_proposal(e,'shallower',66.5),cusa_proposal(e,'deeper',67.)],limits=f.limits,
        goal=f.goal,meter=f.meter,horizon=1,phase='independent.search.competing')
    checked=[r for r in result.attempts if r['status']=='work_and_exit_checked']
    assert len(checked)==2,result.attempts
    assert len({r['workStateId'] for r in checked})==2
    assert result.selected is not None and len(result.progress)==2
    credits=[n.goal_assessment['creditedOriginalTargetCells'] for n in result.progress]
    assert result.selected.goal_assessment['creditedOriginalTargetCells']==max(credits)
    assert all(n.ledger['boneRemovedMm3']==285. and n.ledger['preparationNonboneRemovedMm3']==49. for n in result.progress)
    assert all(n.ledger['spentPublicCalls']==f.meter.calls for n in result.progress)
    assert f.engine.snapshot()==before and f.source.snapshot()==f.initial


def test_later_probe_reserve_exhaustion_retains_prior_checked_completion(ready,completed):
    f=ready;cost=completed.evidence['physicalCallsDuringSearch'];calls=f.meter.calls
    with f.meter.reserve(f.meter.remaining-cost):
        result=checked_joint_sequence_search(f.engine,propose=lambda e:[cusa_proposal(e,'first'),cusa_proposal(e,'second')],
            limits=f.limits,goal=f.goal,meter=f.meter,horizon=1,phase='independent.search.call-reserve')
    assert result.selected is not None and result.evidence['returnedFrozenTask']
    assert result.evidence['budgetExhausted'] and not result.evidence['interrupted']
    assert 'TransitionReserveReached' in result.evidence['stopReason']
    assert f.meter.calls-calls==cost and result.selected.ledger['spentPublicCalls']==f.meter.calls
    assert result.selected.goal_assessment['completedFrozenTask'] and not result.selected.goal_assessment['admittedPatientTask']


def test_cancel_after_checked_node_preserves_physical_progress_without_completion(ready):
    f=ready;cancel=False
    def proposals(e):
        nonlocal cancel
        if e.state_id!=f.engine.state_id:
            cancel=True;return []
        return [cusa_proposal(e)]
    def check():
        if cancel:raise InterruptedError('independent cancellation after checked child')
    result=checked_joint_sequence_search(f.engine,propose=proposals,limits=f.limits,goal=f.goal,
        meter=f.meter,horizon=2,check=check,phase='independent.search.cancel')
    assert result.selected is None and not result.alternatives and result.evidence['decision']=='STOP'
    assert result.evidence['interrupted'] and result.progress
    assert all(not n.goal_assessment['completedFrozenTask'] for n in result.progress)
    assert all('SEARCH_CANCELLED_OR_INTERRUPTED_NOT_ADMITTED' in n.goal_assessment['reasons'] for n in result.progress)
    assert all(n.ledger['withinLimits'] and n.ledger['wholeNonTargetRemovedMm3']==334. for n in result.progress)
    assert f.source.snapshot()==f.initial


def test_final_goal_readout_wall_crossing_refuses_previously_checked_node(ready,monkeypatch):
    import resectionlab.surgical_joint_cranial_search as module
    f=ready;late=False;original=module.JointResectionGoal.declaration
    def declaration(goal):
        nonlocal late
        value=original(goal);late=True;return value
    monkeypatch.setattr(module.JointResectionGoal,'declaration',declaration)
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:
        f.limits.frozen_at+f.limits.max_wall_seconds+1 if late else time.monotonic()))
    result=checked_joint_sequence_search(f.engine,propose=lambda e:[cusa_proposal(e)],
        limits=f.limits,goal=f.goal,meter=f.meter,horizon=1,phase='independent.search.final-wall')
    assert late and result.selected is None and not result.alternatives
    assert result.evidence['decision']=='STOP' and result.progress
    assert all(not n.goal_assessment['completedFrozenTask'] and not n.ledger['withinLimits'] for n in result.progress)
    assert all('WALL_BUDGET' in n.goal_assessment['reasons'] for n in result.progress)
    assert f.source.snapshot()==f.initial


@pytest.fixture(scope='module')
def overlap_ready():
    original,c,contour,p,r=joint_source();mapping=copy.deepcopy(original.scene.regions)
    normal={'id':4,'name':'Generated later-owned normal source membership','material':'soft_tissue','target':False};mapping.append(normal)
    original_target=original.scene.labels==2;normal_mask=original_target.copy();normal_mask[:,:,69:]=False
    assert int(original_target.sum())==27 and int(normal_mask.sum())==18
    props=copy.deepcopy(original.scene.properties);bone=props.pop(PROPERTY)
    masks={row['name']:original.scene.labels==row['id'] for row in mapping[:-1]};masks[normal['name']]=normal_mask
    metadata={'surgical_properties':props,'scope':'Generated target-membership/material-ownership overlap accounting control, no clinical evidence',
        'surgical_adapter':{'ownershipPrecedence':[row['name'] for row in mapping]},
        'anatomy_layers':{row['name']:{'kind':{'bone':'skull','tumor':'tumor','membrane':'dura','soft_tissue':'brain'}[row['material']]} for row in mapping}}
    base=CaseData(case_id='generated-joint-credit-original-ownership-overlap',mri=original.scene.labels.astype(np.float32),
        affine=original.scene.affine,compartments=masks,metadata=metadata,
        source_refs=(SourceRef('generated','simulated:joint-credit-overlap',provenance='simulated'),))
    bone.update(binding(from_case(base,mapping)));bone['contentHash']=contract_hash(bone)
    metadata=thaw_json(base.metadata);metadata['surgical_properties'][PROPERTY]=bone;case=replace(base,metadata=metadata)
    scene=from_case(case,mapping);assert int(np.count_nonzero(scene.labels==2))==9
    source=SurgicalEngine(scene)
    contract=freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,
        membrane_anchor_cells=c.membrane_anchor_cells,work_cells=tuple(map(int,np.flatnonzero(np.isin(scene.labels,[2,4])))),
        evidence=c.as_dict()['evidence'])
    source=SurgicalEngine(scene,cranial_stages=contract);initial=source.snapshot();meter=TransitionMeter(800)
    limits=freeze_joint_cranial_limits(source,meter=meter,max_bone_mm3=285.,max_preparation_nonbone_mm3=49.,
        max_work_nonbone_mm3=20.,max_native_actions=300,max_public_calls=800,max_wall_seconds=180.)
    goal=freeze_joint_resection_goal(source,limits=limits)
    ready,receipt=checked_joint_actions(source,preparation_program(source,contract,contour)+[
        request('exchange',instrument='cusa'),request('place',pose=pose())],limits=limits,meter=meter,phase='independent.overlap.preparation')
    assert receipt['status']=='accepted',receipt
    return SimpleNamespace(source=source,initial=initial,engine=ready,meter=meter,limits=limits,goal=goal)


def test_actual_original_annotation_removal_is_not_owned_treatment_credit(overlap_ready):
    f=overlap_ready;before=f.engine.snapshot()
    work,receipt=checked_joint_cusa_motion(f.engine,request('apply',pose=pose(z=66.5),duration=30.),
        limits=f.limits,meter=f.meter,phase='independent.overlap.work')
    assert receipt['status']=='accepted',receipt
    exited,exit_receipt=find_checked_joint_exit(work,limits=f.limits,meter=f.meter,phase='independent.overlap.exit')
    assert exit_receipt['status']=='accepted',exit_receipt
    goal=joint_goal_assessment(exited,goal=f.goal,limits=f.limits,exit_receipt=exit_receipt)
    assert goal['originalTargetCells']==27 and goal['ownedOriginalTargetCells']==9
    assert goal['removedOriginalTargetCells']==18 and goal['originalTargetFractionRemoved']==pytest.approx(2/3)
    assert goal['creditedOriginalTargetCells']==0 and goal['originalTargetTreatmentMm3']==0.
    assert goal['originalTargetTreatmentFraction']==0. and not goal['completedFrozenTask']
    assert sum(goal['zoneTreatmentCreditCells'])==0 and 'ORIGINAL_TARGET_FRACTION_BELOW_GOAL' in goal['reasons']
    assert exit_receipt['ledger']['workNonTargetRemovedMm3']==18. and exit_receipt['ledger']['targetRemovedMm3']==0.
    assert exit_receipt['ledger']['wholeNonTargetRemovedMm3']==352.
    assert f.engine.snapshot()==before and f.source.snapshot()==f.initial and not goal['admittedPatientTask']


def _comparison_record(*,physical=18.,credited=18.,thermal=None):
    """Pure ranking record; not a native state, branch or admission fixture."""
    from resectionlab.surgical_joint_cranial_search import JointSequenceNode
    dimensions={name:{'cumulative':None,'status':'unavailable','aggregation':'cumulative'} for name in (
        'toolExchanges','brainStrainSquaredMm3S','brainStressSquaredPa2Mm3S','thermalPeakC',
        'thermalExposureMm3KS','bloodGeneratedMm3','motorEncounterMm3','languageEncounterMm3')}
    if thermal is not None:dimensions['thermalPeakC']={'cumulative':thermal,'status':'measured','aggregation':'peak'}
    return JointSequenceNode(None,None,(),{'boneRemovedMm3':285.,'preparationNonboneRemovedMm3':49.,
        'workNonTargetRemovedMm3':0.,'wholeNonTargetRemovedMm3':334.,'modeledElapsedSeconds':100.},
        {'originalTargetRemovedMm3':physical,'originalTargetTreatmentMm3':credited}, {'dimensions':dimensions})


def test_comparator_missing_consequence_is_incomparable_with_measured_zero():
    from resectionlab.surgical_joint_cranial_search import _dominates
    unknown=_comparison_record(credited=18.)
    measured_better=_comparison_record(credited=19.,thermal=0.)
    assert not _dominates(measured_better,unknown) and not _dominates(unknown,measured_better)
    measured_same=_comparison_record(credited=18.,thermal=0.)
    assert _dominates(measured_better,measured_same)


def test_comparator_recommendation_and_pareto_use_treatment_not_physical_union():
    from resectionlab.surgical_joint_cranial_search import _dominates,_key
    physical_only=_comparison_record(physical=27.,credited=0.)
    treatment=_comparison_record(physical=18.,credited=18.)
    assert _key(treatment)<_key(physical_only) and _dominates(treatment,physical_only)
    assert not _dominates(physical_only,treatment)
