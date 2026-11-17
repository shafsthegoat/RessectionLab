"""Arithmetic checked-node fixtures; none of these are physical operations."""
from types import SimpleNamespace
import pytest

from resectionlab.surgical_joint_cranial_forest import forest_priority


def node(name, *, target=100., bone=1., supply=0., drainage=0., neural=0., structures=None, dirty=False):
    q=lambda v:{'status':'unavailable' if v is None else 'source_count','cumulative':v}
    dims={k:q(v) for k,v in [('neuralRemovedMm3',neural),('neuralCutSourceMm3',0.),
        ('protectedRemovedMm3',0.),('supplyDeficitSeconds',supply),('drainageDeficitSeconds',drainage)]}
    dims['structureRemovedMm3']={k:q(v) for k,v in (structures or {'a':0.,'b':0.}).items()}
    dims['structureCutSourceMm3']={k:q(0.) for k in dims['structureRemovedMm3']}
    endpoint={'terminated':not dirty,'truncated':False,'validWithdrawal':not dirty,
        'classification':'in-progress' if dirty else 'successful-treatment' if target else 'valid-stop-with-residual'}
    engine=SimpleNamespace(scene=SimpleNamespace(identity='fixture-scene'),state_id=name,actions=[])
    return SimpleNamespace(exited=engine,work=engine,path=({'proposalId':name,'preparationOnly':False},),
        goal_assessment={'originalTargetTreatmentMm3':target,'creditedOriginalTargetCells':int(target),
            'completedFrozenTask':bool(target),'endpointAssessment':endpoint},
        ledger={'withinLimits':True,'boneRemovedMm3':bone,'preparationNonboneRemovedMm3':0.,
            'workNonTargetRemovedMm3':neural,'wholeNonTargetRemovedMm3':bone+neural,'modeledElapsedSeconds':1.},
        sequence_outcome={'dimensions':dims})


def test_lower_bone_cannot_purchase_supply_exposure_across_roots():
    low_bone=node('cheap',bone=1.,supply=20.)
    no_deficit=node('costlier',bone=2.)
    selected,proof=forest_priority([('first',low_bone),('second',no_deficit)],procedure=True)
    assert selected[1] is no_deficit and proof['reason']=='max_treatment_within_harm_bounds'
    assert proof['candidateCount']==proof['eligibleCount']==2


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('rename',[False,True])
def test_opposing_supply_drainage_is_unresolved_under_root_and_branch_reordering(reverse,rename):
    a=node('z' if rename else 'first',bone=1.,supply=0.,drainage=20.)
    b=node('a' if rename else 'second',bone=100.,supply=20.,drainage=0.)
    rows=[('root-z' if rename else 'root-a',a),('root-a' if rename else 'root-b',b)]
    if reverse:rows.reverse()
    selected,proof=forest_priority(rows,procedure=True)
    assert selected is None
    assert proof['reason']=='unresolved_exposure_tradeoff_stop_with_residual'


def test_opposing_structure_effects_and_unknown_required_exposure_do_not_become_bone_savings():
    a=node('a',structures={'a':1.,'b':0.},bone=1.)
    b=node('b',structures={'a':0.,'b':1.},bone=100.)
    selected,proof=forest_priority([('a',a),('b',b)],procedure=True)
    assert selected is None and proof['reason'].startswith('unresolved_neural')
    a=node('a',supply=None,bone=0.);b=node('b',supply=1.,bone=100.)
    selected,proof=forest_priority([('a',a),('b',b)],procedure=True)
    assert selected[1] is b
    assert next(r for r in proof['candidates'] if r['rootId']=='a')['missingRequiredMetrics']==['supplyDeficitSeconds']


def test_old_completion_flag_cannot_make_dirty_endpoint_eligible():
    dirty=node('dirty',target=100.,dirty=True);clean=node('clean',target=50.,bone=2.)
    selected,proof=forest_priority([('dirty',dirty),('clean',clean)],procedure=True)
    assert dirty.goal_assessment['completedFrozenTask'] and selected[1] is clean
    assert next(r for r in proof['candidates'] if r['rootId']=='dirty')['eligible'] is False


def test_same_final_state_keeps_distinct_causal_program_and_saved_branch_ids():
    from resectionlab.surgical_joint_cranial_forest import candidate_key,candidate_branch_id
    a=node('same-state');b=node('same-state')
    a.exited.actions=[{'operation':'inspect','slot':'left'}]
    b.exited.actions=[{'operation':'inspect','slot':'right'}]
    assert candidate_key('root',a)!=candidate_key('root',b)
    assert candidate_branch_id('root',a)!=candidate_branch_id('root',b)


def test_actual_forest_compares_more_than_displayed_candidates_before_export(monkeypatch):
    """Inject checked-node fixtures at the search boundary, never fake physics."""
    from resectionlab import surgical_joint_cranial_search as search
    from resectionlab import surgical_joint_cranial_skills as skills
    from resectionlab import surgical_cranial_planning_roots as registry
    from resectionlab import surgical_planning_admission as admission
    from resectionlab import surgical_joint_cranial_forest as forest
    from resectionlab.surgical_procedure_policy import PROPERTY
    monkeypatch.setattr(admission,'require_planning_admission',lambda _:None)
    monkeypatch.setattr(registry,'source_binding',lambda scene:{'baseSceneId':scene.identity,'same':'fixture anatomy'})
    monkeypatch.setattr(skills,'freeze_joint_cranial_limits',lambda *args,**kwargs:None)
    monkeypatch.setattr(search,'freeze_joint_resection_goal',lambda *args,**kwargs:SimpleNamespace(_zones=({1},)))
    rows={'a':[node('a1',bone=1.,drainage=20.),node('a2',bone=2.,drainage=20.),node('a3',bone=3.,drainage=20.)],
          'b':[node('b1',bone=100.,supply=20.)]}
    for key in rows:rows[key].append(node('stop-'+key,target=0.,bone=0.))
    def checked(source,*args,**kwargs):
        candidates=rows[source.name]
        return SimpleNamespace(alternatives=tuple(candidates),progress=tuple(candidates),
            attempts=(),evidence={'interrupted':False})
    monkeypatch.setattr(search,'checked_joint_sequence_search',checked)
    exports=[]
    def export(source,candidate,meter,**kwargs):
        exports.append(candidate.state_id);return candidate
    monkeypatch.setattr(forest,'replay_export',export)
    roots=[]
    for key in rows:
        source=SimpleNamespace(name=key,actions=[],history=[],scene=SimpleNamespace(identity=key,properties={PROPERTY:{}}),
            snapshot=lambda:{})
        roots.append({'id':key,'engine':source,'limits':{'same':True},'propose':lambda _:[]})
    result=forest.search_opening_forest(roots,budget=forest.OpeningComparisonBudget(100,30.,export_reserve=10))
    assert result['selected'] is None and result['decision']=='STOP'
    assert len(result['branches'])==len(exports)==3
    assert result['selectionEvidence']['candidateCount']==6
    assert result['selectionEvidence']['reason']=='unresolved_exposure_tradeoff_stop_with_residual'
    assert result['selectionEvidence']['checkedStopCandidates']
    assert set(exports)!=set(r['finalStateId'] for r in result['selectionEvidence']['candidates'])
