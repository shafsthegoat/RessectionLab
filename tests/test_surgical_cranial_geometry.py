"""Display geometry must retain accepted physical prefixes and source overlaps."""
from types import SimpleNamespace
import copy
import numpy as np

from resectionlab.surgical_cranial_geometry import recorded_cranial_geometry


def test_partial_requested_suffix_is_never_drawn_and_membership_overlap_survives(monkeypatch):
    from resectionlab import surgical_procedure_policy as procedure
    from resectionlab import surgical_scene
    scene = SimpleNamespace(identity='scene', source={'caseHash':'case'},
        points=lambda ids: np.array([[float(c),0.,0.] for c in ids]).reshape((-1,3)))
    evidence = {'kind':'neural','reviewStatus':'generated-engineering','provenance':'Independent generated declaration'}
    monkeypatch.setattr(procedure, '_validated', lambda _: {'record':{**{k:0. for k in ('maximumNeuralRemovalMm3','maximumNeuralCutSourceMm3','maximumBoneRemovalMm3','maximumNonbonePreparationRemovalMm3','maximumModeledElapsedSeconds','maximumStructureRemovalMm3','maximumStructureCutSourceMm3')},'policyHash':'policy','structures':{'motor-cortex':evidence,'corticospinal-tract':evidence}},'structureCells':{'motor-cortex':{1},'corticospinal-tract':{1}}})
    monkeypatch.setattr(surgical_scene, 'original_target_memberships', lambda _: {'target':(2,3)})
    source={'stateId':'root','checkpoint':{'state':{'slots':{'left':{'instrument':'cusa','position':[0.,0.,0.]}}}}}
    action=lambda end:{'slot':'left','operation':'apply','pose':{'position':end}}
    history=[{'action':action([0.,0.,9.]),'stateBefore':'root','stateAfter':'prefix','status':'partial','completedFraction':.3,'events':[{'kind':'cusa_contact_dwell','position':[0.,0.,3.]}]},
        {'action':action([0.,0.,12.]),'stateBefore':'prefix','stateAfter':'final','status':'partial','completedFraction':.2,'contactPosition':[0.,0.,5.],'events':[]}]
    row={'branchId':'branch','nativeHistory':history,'sequenceOutcome':{'dimensions':{'structureRemovedMm3':{'motor-cortex':{'cumulative':0.},'corticospinal-tract':{'cumulative':0.}},'structureCutSourceMm3':{'motor-cortex':{'cumulative':0.},'corticospinal-tract':{'cumulative':0.}}}}}
    snapshot={'stateId':'final','actions':[e['action'] for e in history],'checkpoint':{'state':{'removed':[2],'debrisMm3':0.,'fluid':{},'sources':[]}}}
    original=copy.deepcopy((row,source,snapshot))
    g=recorded_cranial_geometry(scene,scene,source,snapshot,row,'common')
    assert g['paths'][0]['pointsRasMm']==[[0.,0.,0.],[0.,0.,3.]]
    assert g['omittedPartialActions']==[1] and len(g['paths'])==1
    assert [s['sourceCells'] for s in g['structures']]==[[1],[1]]
    assert g['residualSourceCells']==[3]
    assert (row,source,snapshot)==original


def test_cold_cranial_report_remeasures_paths_cleanup_and_common_endpoint():
    from test_structure_aware_cranial_planning import procedure_case
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies, install_cranial_comparison, validate_cranial_report
    from resectionlab.core import semantic_digest
    import pytest
    _,scene,record=procedure_case(0.)
    result=compare_cranial_strategies(scene,record,planning_horizon=3)
    install_cranial_comparison(record,result,scene)
    assert validate_cranial_report(scene,record,record['cranialPlanning'])['verified']
    for field in ('path','membership','cleanup','policy','endpoint'):
        changed=copy.deepcopy(record);report=changed['cranialPlanning'];row=report['alternatives'][0]
        g=row['recordedGeometry']
        if field=='path':g['paths'][-1]['pointsRasMm'][-1][0]+=.5
        if field=='membership':g['structures'][0]['pointsRasMm'][0][0]+=.5
        if field=='cleanup':g['cleanup']['debrisPresentMm3']=0.
        if field=='policy':g['harmBounds']['maximumNeuralRemovalMm3']+=1.
        if field=='endpoint':row['endpointAssessment']['classification']='successful-treatment'
        g['geometrySeal']=semantic_digest({k:v for k,v in g.items() if k!='geometrySeal'})
        report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
        with pytest.raises(ValueError,match='REPORT_RECORDED_GEOMETRY_CHANGED|REPORT_COMMON_ENDPOINT_CHANGED'):
            validate_cranial_report(scene,changed,report)
