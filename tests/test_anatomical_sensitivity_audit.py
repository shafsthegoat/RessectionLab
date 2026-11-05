"""Independent destination-label audit controls on generated anatomy only."""
import ast
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resectionlab import surgical_anatomical_sensitivity as producer
from resectionlab import surgical_two_tool as planning
from resectionlab.core import semantic_digest
from resectionlab.surgical_anatomical_encounter import EncounterTracker
from test_surgical_anatomical_encounter import context_case, context_report
from test_surgical_two_tool import qualified, prepared

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/verify_anatomical_sensitivity.py'
spec=importlib.util.spec_from_file_location('independent_anatomical_sensitivity_audit',PATH)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def reseal(value,key):
    value[key]=semantic_digest({k:v for k,v in value.items() if k!=key});return value


def reseal_study(study):
    reseal(study['assessment'],'assessmentSeal');return reseal(study,'studySeal')


@pytest.fixture(scope='module')
def studies(context_case,context_report):
    return {step:producer.generate_sensitivity(context_case[1],context_report,{'contextVoxelShift':step}) for step in (0,1)}


def test_independent_mapping_has_no_wrap_resampling_or_axis_alignment_assumption():
    mask=np.zeros((3,4,2),bool);mask[0,1,0]=True;mask[-1,2,1]=True
    affine=np.array([[0.,-2.,0.,10.],[1.,0.,0.,20.],[0.,0.,.5,30.],[0.,0.,0.,1.]])
    worlds=audit.scenarios(mask,affine,1)
    assert [w['id'] for w in worlds]==['nominal','grid-0-','grid-0+','grid-1-','grid-1+','grid-2-','grid-2+']
    plus=worlds[2];assert plus['mask'][1,1,0] and not plus['known'][0].any()
    assert plus['mask'].sum()==1 and plus['outOfFieldContextCells']==1
    assert plus['translationRasMm']==[0.,1.,0.]
    assert worlds[4]['translationRasMm']==[-2.,0.,0.]
    tiny=audit.scenarios(np.ones((1,1,1),bool),np.eye(4),2)
    assert all(not w['mask'].any() and not w['known'].any() and w['outOfFieldContextCells']==1 for w in tiny[1:])
    for bad in (True,1.,-1,3):
        with pytest.raises(ValueError):audit.scenarios(mask,affine,bad)


def test_no_planner_or_sensitivity_predictor_is_an_audit_oracle():
    names=[node.module for node in ast.walk(ast.parse(PATH.read_text())) if isinstance(node,ast.ImportFrom)]
    assert not any(n and ('surgical_two_tool' in n or 'surgical_anatomical_sensitivity' in n or 'surgical_strategy_sensitivity' in n) for n in names)


def test_zero_and_seven_scenario_native_studies_are_independent_and_private_invariant(context_case,context_report,studies,monkeypatch):
    engine=context_case[1];before=engine.snapshot()
    class PrivateOwner:
        def __deepcopy__(self,memo):pytest.fail('Private evaluator copied')
        def __call__(self):pytest.fail('Private evaluator invoked')
    def forbidden(*args,**kwargs):pytest.fail('Production predictor used as oracle')
    monkeypatch.setattr(producer,'_replay',forbidden);monkeypatch.setattr(planning,'_summary',forbidden)
    monkeypatch.setattr(planning,'_check_burdens',forbidden);monkeypatch.setattr(planning,'_compile',forbidden)
    monkeypatch.setattr(EncounterTracker,'observe',forbidden)
    engine._private_evaluator=PrivateOwner();engine._private_truth=np.ones(engine.scene.shape,bool)
    try:results={step:audit.verify(study,context_report,engine) for step,study in studies.items()}
    finally:del engine._private_evaluator,engine._private_truth
    assert results[0]['passed'] and results[0]['scenarioCount']==1
    assert results[1]['passed'] and results[1]['scenarioCount']==7
    assert all(c['completeExecution'] and c['withdrawalComplete'] for c in results[1]['candidates'])
    assert any(c['summary']['additionalTargetMm3']>0. for c in results[1]['candidates'])
    assert all(o['encounter']['baselineRemovedContextMm3']>0. for c in results[1]['candidates'] for o in c['outcomes'])
    assert engine.snapshot()==before


@pytest.mark.parametrize('change',['family','translation','mask','coverage','lost-context','candidate','scenario-count',
    'physical','receipt','prefix','new-work','unknown','margin','summary','policy','calibration'])
def test_resealed_scenario_outcome_policy_and_metadata_forgeries_refuse(context_case,context_report,studies,change):
    study=copy.deepcopy(studies[1]);assessment=study['assessment'];row=assessment['candidates'][0];outcome=row['outcomes'][1]
    config=assessment['scenarioConfiguration']
    if change=='family':config['family']='measured-registration-errors'
    elif change=='translation':config['scenarios'][1]['translationRasMm'][0]+=.1
    elif change=='mask':config['scenarios'][1]['maskHash']='invented'
    elif change=='coverage':config['scenarios'][1]['knownCoverageHash']='invented'
    elif change=='lost-context':config['scenarios'][1]['outOfFieldContextCells']+=1
    elif change=='candidate':assessment['candidates'].pop()
    elif change=='scenario-count':assessment['scenarioCount']=1
    elif change=='physical':row['physicalStateHashes'][0]='invented'
    elif change=='receipt':row['actionReceipts'][0]['elapsedSeconds']+=.01
    elif change=='prefix':outcome['contextPrefixHashes'][0]='invented'
    elif change=='new-work':outcome['encounter']['newContextWorkCells']=[-1]
    elif change=='unknown':outcome['encounter']['unknownEncounteredCells']=[-1]
    elif change=='margin':row['fixedMarginEncounter']['marginMm']+=.1
    elif change=='summary':row['summary']['additionalTargetMm3']+=1.
    elif change=='policy':assessment['policies']['allDeclaredScenarios']='long' if assessment['policies']['allDeclaredScenarios']!='long' else 'stop'
    else:assessment['calibratedUncertainty']['supported']=True
    with pytest.raises(ValueError):audit.verify(reseal_study(study),context_report,context_case[1])


def test_native_annotation_bytes_remain_required_after_study_sealing(context_case,context_report,studies):
    collection=context_case[4];path=collection.path(context_report['contextBinding']['original']['path']);data=path.read_bytes()
    try:
        path.write_bytes(data+b'changed source annotation')
        with pytest.raises(ValueError,match='original|annotation|bytes'):audit.verify(studies[0],context_report,context_case[1])
    finally:path.write_bytes(data)


def test_saved_workspace_linkage_and_unchanged_native_bytes(context_case,context_report,studies,tmp_path):
    from resectionlab.imaging import save_case
    case,engine,original,_,_=context_case;record=copy.deepcopy(original)
    comparison=reseal({'schema':'resectionlab.two-tool-workspace.v1','revision':1,'caseHash':case.semantic_hash,
        'sourceBranchId':'admitted','sourceSnapshotHash':semantic_digest(engine.snapshot()),'report':context_report},'workspaceSeal')
    study=reseal({'schema':'resectionlab.strategy-sensitivity-workspace.v1','revision':1,'comparisonRevision':1,'study':studies[1]},'workspaceSeal')
    record.update(strategyComparisons=[comparison],strategySensitivityStudies=[study])
    assert tmp_path.resolve().is_relative_to(ROOT/'testcases')
    path=tmp_path/'anatomical-study.ressectionlab';save_case(case,path,artifacts={'surgical_toolbox':record});before=path.read_bytes()
    result=audit.verify_workspace(path)
    assert result['passed'] and result['inputsUnchanged'] and result['sensitivityRevision']==1 and path.read_bytes()==before
    record['strategySensitivityStudies'][0]['comparisonRevision']=2;reseal(record['strategySensitivityStudies'][0],'workspaceSeal')
    save_case(case,path,artifacts={'surgical_toolbox':record})
    with pytest.raises(ValueError,match='linked comparison'):audit.verify_workspace(path)
