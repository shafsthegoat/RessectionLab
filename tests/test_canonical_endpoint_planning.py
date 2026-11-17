"""Actual generated Generate/cold controls with unchanged zero-debris endpoint."""
import copy
from dataclasses import replace

import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_cranial_planning import (
    compare_cranial_strategies, install_cranial_comparison, validate_cranial_report, cranial_summary)


@pytest.fixture(scope='module', params=['clean','debris','fluid','two-roots'])
def generated_plan(request):
    from scripts.generated_cranial_demo_case import build_generated_cranial_case
    variant=request.param
    case,base,record=build_generated_cranial_case(1.,include_cleanup=variant!='debris',
        endpoint_fluid_source=variant=='fluid')
    if variant=='two-roots':
        from scripts.generated_cranial_demo_case import add_lateral_entry_root
        add_lateral_entry_root(base,record)
    result=compare_cranial_strategies(base,record,planning_horizon=3)
    install_cranial_comparison(record,result,base)
    return variant,case,base,record,result


def test_generate_completion_and_stop_are_actual_shared_endpoint(generated_plan):
    variant,_,_,record,_=generated_plan;report=record['cranialPlanning']
    useful=[r for r in report['alternatives'] if r['goalAssessment']['creditedOriginalTargetCells']]
    assert useful,report['failures']
    if variant in ('clean','two-roots'):
        assert report['decision']=='COMPLETED_FROZEN_TASK'
        assert any(r['endpointAssessment']['classification']=='successful-treatment' for r in useful)
        assert any(r['endpointAssessment']['classification']=='valid-stop-with-residual' for r in useful)
        if variant=='two-roots':
            assert len({r['rootId'] for r in useful})==2
    else:
        assert report['decision']=='STOP'
        assert all(not r['completedFrozenTask'] and not r['validStop'] for r in useful)
        assert all(variant in r['endpointAssessment']['pending'] for r in useful)
        assert all(not r['eligible'] for r in report['selectionEvidence']['candidates'] if
            r['values']['targetRemovedMm3']>0)
    for row in report['alternatives']:
        assert row['completedFrozenTask']==row['goalAssessment']['completedFrozenTask']==row['endpointAssessment']['completedTask']
    dto=cranial_summary(report)
    assert dto['selectionEvidence']==report['selectionEvidence'] and 'selectionProofs' not in dto


def test_cold_reopen_remeasures_every_candidate_including_dirty_prefix(generated_plan,tmp_path):
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    from resectionlab.surgical_scene import from_case
    variant,case,_,record,_=generated_plan
    path=tmp_path/(variant+'.ressectionlab');save_case(case,path,artifacts={'surgical_toolbox':record})
    cold=load_case(path);saved=read_case_artifacts(path)['surgical_toolbox'];scene=from_case(cold,saved['mapping'])
    proof=validate_cranial_report(scene,saved,saved['cranialPlanning'])
    assert proof['verified'] and proof['recommendationVerified']
    bad=copy.deepcopy(saved);report=bad['cranialPlanning']
    row=next(r for r in report['alternatives'] if r['goalAssessment']['creditedOriginalTargetCells'])
    row['completedFrozenTask']=row['goalAssessment']['completedFrozenTask']=True
    row['endpointAssessment']['completedTask']=False
    report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
    with pytest.raises(ValueError,match='CRANIAL_PLANNING_REPORT_'):
        validate_cranial_report(scene,bad,report)


def test_truncation_cannot_retain_completed_or_valid_stop(generated_plan):
    from resectionlab.surgical_sequential_task import COMMON_CONFIG,endpoint_assessment
    from resectionlab.surgical_joint_cranial_search import endpoint_eligible
    variant,_,_,_,result=generated_plan
    if variant not in ('clean','two-roots'):
        assert all(not r['completedFrozenTask'] for _,r in result['branches'])
        return
    engine=next(e for e,r in result['branches'] if r['completedFrozenTask'])
    truncated=endpoint_assessment(engine,COMMON_CONFIG,stopped=True,externally_truncated=True)
    assert truncated['truncated'] and not truncated['completedTask'] and not endpoint_eligible(truncated)
    exact=endpoint_assessment(engine,replace(COMMON_CONFIG,max_attempts=len(engine.actions)+1),
        attempts=len(engine.actions)+1,stopped=True)
    assert exact['truncated'] and not exact['completedTask'] and not endpoint_eligible(exact)


def test_cold_selection_cannot_omit_an_unshown_checked_candidate(generated_plan):
    _,_,base,original,_=generated_plan
    record=copy.deepcopy(original);report=record['cranialPlanning']
    shown={r['candidateKey'] for r in report['alternatives']}
    omitted=next(r for r in report['selectionProofs'] if r['candidateKey'] not in shown)
    report['selectionProofs'].remove(omitted)
    cert=report['selectionEvidence'];removed=next(r for r in cert['candidates'] if r['candidateKey']==omitted['candidateKey'])
    cert['candidates'].remove(removed);cert['candidateCount']-=1;cert['eligibleCount']-=int(removed['eligible'])
    report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
    with pytest.raises(ValueError,match='CRANIAL_PLANNING_REPORT_SEARCH_CANDIDATE_CHANGED'):
        validate_cranial_report(base,record,report)


def test_consequence_cleanup_uses_the_same_physical_endpoint(generated_plan):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_consequences import consequence_vector
    from resectionlab.surgical_cranial_planning_roots import FIELD
    variant,_,_,record,result=generated_plan
    roots={r['id']:r for r in record[FIELD]['roots']}
    for engine,row in result['branches']:
        source=SurgicalEngine.restore(engine.scene,roots[row['rootId']]['sourceSnapshot'],use_cache=False)
        vector=consequence_vector(engine,source)
        assert vector['cleanupComplete']==row['endpointAssessment']['cleanupComplete']
        if variant in ('debris','fluid') and row['goalAssessment']['creditedOriginalTargetCells']:
            assert not vector['cleanupComplete']


@pytest.mark.parametrize('field',['maxPhysicalCalls','maxWallSeconds','exportReplayReserveCalls','wallSeconds'])
def test_cold_budget_cannot_be_resealed_past_its_source_limits(generated_plan,field):
    _,_,base,original,_=generated_plan;record=copy.deepcopy(original);report=record['cranialPlanning']
    cost=report['cost'];cost[field]=cost['maxWallSeconds']+1 if field=='wallSeconds' else cost[field]+1
    report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
    with pytest.raises(ValueError,match='CRANIAL_PLANNING_REPORT_SOURCE_GLOBAL_BUDGET_CHANGED'):
        validate_cranial_report(base,record,report)


def test_branch_renaming_and_display_order_do_not_change_cold_preference(generated_plan):
    from resectionlab.surgical_cranial_planning_roots import FIELD,create_registry,validate_registry
    from resectionlab.surgical_cranial_geometry import recorded_cranial_geometry
    _,_,base,original,_=generated_plan;record=copy.deepcopy(original);report=record['cranialPlanning']
    names={r['branchId']:'renamed-'+str(i) for i,r in enumerate(report['alternatives'])}
    before=report['selectionEvidence']['selectedCandidateKey']
    for field in ('branches','parents','timeline'):
        record[field]={names.get(k,k):v for k,v in record.get(field,{}).items()}
    registry=record[FIELD];mapping={names.get(k,k):v for k,v in registry['branchRoots'].items()}
    declarations=[{k:v for k,v in r.items() if k not in ('sceneId','rootSeal')} for r in registry['roots']]
    record[FIELD]=create_registry(base,declarations,mapping)
    scenes=validate_registry(base,record[FIELD]);roots={r['id']:r for r in record[FIELD]['roots']}
    for row in report['alternatives']:
        row['branchId']=names[row['branchId']]
        row['recordedGeometry']=recorded_cranial_geometry(base,scenes[row['rootId']],
            roots[row['rootId']]['sourceSnapshot'],record['branches'][row['branchId']],row,report['sourceStateId'])
    report['selectedBranchId']=names.get(report['selectedBranchId'],report['selectedBranchId'])
    report['alternatives'].reverse()
    report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
    assert validate_cranial_report(base,record,report)['recommendationVerified']
    assert report['selectionEvidence']['selectedCandidateKey']==before
