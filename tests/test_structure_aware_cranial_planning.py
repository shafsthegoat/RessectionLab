"""Physical generated controls for the declared procedure, never patient anatomy."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, semantic_digest, thaw_json
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_procedure_policy import PROPERTY, make_procedure_policy


def procedure_case(neural_limit=0., *, cerebellar_shortcut=False):
    """Prospective anatomy/route inventory; no expected outcome enters search."""
    from test_surgical_membrane_transport import joint_source, bone_actions
    from test_surgical_joint_cranial_skills import preparation_program
    from test_surgical_bone_transport import request, pose
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_craniotomy import binding, contract_hash
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
    from resectionlab.surgical_cranial_planning_roots import candidate_scene, create_registry, source_binding, FIELD
    template, stage, contour, _, _ = joint_source()
    labels = template.scene.labels.copy()
    labels[33, 32, 66] = labels[22, 22, 70] = labels[23, 22, 70] = 4
    neural_position=(34,32,67) if cerebellar_shortcut else (33,32,66)
    if cerebellar_shortcut:
        # Keep this generated shortcut below the supplied membrane checkpoint;
        # source preparation itself must remain clear of the excluded anatomy.
        labels[33,32,66]=0
        labels[neural_position]=4
    regions = template.scene.regions + [dict(id=4, name='Declared generated neural tissue', material='soft_tissue')]
    props = copy.deepcopy(template.scene.properties)
    opening = props.pop('cranialAccessV1')
    case = CaseData(case_id='generated-structure-aware-approaches', mri=labels.astype(np.float32),
        compartments={r['name']: labels == r['id'] for r in regions}, affine=template.scene.affine,
        source_refs=(SourceRef('generated', 'synthetic://structure-aware-approaches', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True,
            'surgical_adapter': {'type': 'generated-engineering-control'}, 'surgical_properties': props,
            'anatomy_layers': {r['name']: {'kind': r['material']} for r in regions}})
    base = from_case(case, regions)
    flat = lambda xyz: int(np.ravel_multi_index(xyz, labels.shape))
    intracranial = [flat((x,y,z)) for x in range(21,44) for y in range(21,44) for z in range(61,75)]
    evidence = dict(provenance='Independent generated closed-shell anatomy and superior entry declaration before search; no acquired anatomy',
        evidenceSha256=semantic_digest({'intracranial': intracranial, 'source': base.source}).split(':')[-1],
        origin='independent-source-declaration', reviewStatus='generated-engineering')
    structures = {name: dict(evidence, kind='neural', cells=[flat(xyz)], coverageCells=intracranial)
        for name, xyz in [('brainstem',(22,22,70)), ('cerebellum',(23,22,70)), ('supratentorial-neural',neural_position)]}
    critical = {flat((22,22,70)), flat((23,22,70))}
    if cerebellar_shortcut:
        # Distinct generated source anatomy supplied before any search.
        structures['cerebellum']['cells'].append(flat(neural_position))
        structures['cerebellum']['cells'].sort()
        structures['supratentorial-neural']['cells']=[]
        critical.add(flat(neural_position))
    permitted = sorted(set(intracranial)-critical)
    entry = [flat((x,y,61)) for x in range(24,41) for y in range(24,41)]
    raw = make_procedure_policy(base, policy_id='generated-supratentorial-harm-v1', evidence=evidence,
        structures=structures, intracranial_cells=intracranial, observed_cells=range(labels.size),
        entry_regions={'superior-entry': dict(evidence,cells=entry)},
        permitted_compartments={'supratentorial-work': dict(evidence,cells=permitted),
            'bone-preparation':dict(evidence,cells=list(map(int,np.flatnonzero(labels==1)))),
            'membrane-preparation':dict(evidence,cells=list(map(int,np.flatnonzero(labels==3))))},
        work_domain=dict(evidence, minimumVoxelBoundary=[-.5]*3, maximumVoxelBoundary=(np.asarray(labels.shape)-.5).tolist()),
        tissue_permissions={r['material']: {'remove':True,'cut':True} for r in regions},
        neural_max_mm3=neural_limit, bone_max_mm3=285., preparation_nonbone_max_mm3=49., modeled_time_max_seconds=240.,
        structure_max_mm3={'brainstem':0.,'cerebellum':0.,'supratentorial-neural':neural_limit}, start_stage='post-scalp')
    metadata = thaw_json(case.metadata); metadata['surgical_properties'][PROPERTY] = raw
    case = replace(case, metadata=metadata)
    base = from_case(case, regions)
    opening.update(binding(base)); opening['contentHash'] = contract_hash(opening)
    scene = candidate_scene(base,'cranialAccessV1',opening)
    native = SurgicalEngine(scene)
    contract = freeze_cranial_stage_contract(native, membrane_cells=stage.membrane_cells,
        retained_membrane_cells=stage.retained_membrane_cells, membrane_anchor_cells=stage.membrane_anchor_cells,
        work_cells=sorted([*stage.work_cells,flat(neural_position)]), evidence=stage.as_dict()['evidence'])
    native = SurgicalEngine(scene, protection=native.protection, guard_config=native.guard_config, cranial_stages=contract)
    opening_actions = bone_actions(contour)
    prep = preparation_program(native,contract,contour)[len(opening_actions):] + [request('exchange',instrument='cusa'), request('place',pose=pose())]
    center = base.points(sorted(contract.original_target_cells)).mean(axis=0)
    routes = [{'id':name, 'motion':request('apply',pose=pose(float(center[0]+offset),float(center[1]),66.5),duration=30.)}
        for name,offset in [('central',0.),('lateral',-1.5)]]
    planning = {'limits': {'max_bone_mm3':285.,'max_preparation_nonbone_mm3':49.,'max_work_nonbone_mm3':20.,
        'max_native_actions':300,'max_public_calls':1800,'max_wall_seconds':240.,
        'max_bone_contact_replans':6,'max_tissue_contact_replans':8,'export_replay_reserve_calls':350},
        'openingSkill':{'id':'source-opening','kind':'opening','actions':opening_actions}, 'preparationActions':prep,
        'workRoutes':routes,'outsidePoses':{}, 'qualification':{'status':'reviewed','sourceBinding':source_binding(base),
            'evidenceSha256':evidence['evidenceSha256'],'blockers':[], 'provenance':'Generated engineering declaration only; no patient qualification'}}
    roots = [{'id':'superior-opening','label':'Declared superior approaches','openingProperty':'cranialAccessV1',
        'openingContract':opening,'sourceSnapshot':native.snapshot(),'planning':planning}]
    common = SurgicalEngine(base).snapshot()
    record = {'mapping':regions,'branches':{'main':common},'branchId':'main','position':0,
        'parents':{'main':None},'timeline':{'main':[]},FIELD:create_registry(base,roots)}
    return case, base, record


def test_policy_case_roundtrip_preserves_predeclaration_source_binding(tmp_path):
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    from resectionlab.surgical_cranial_planning_roots import validate_registry,FIELD
    case,base,record = procedure_case()
    path=tmp_path/'procedure.ressectionlab';save_case(case,path,artifacts={'surgical_toolbox':record})
    loaded=load_case(path);saved=read_case_artifacts(path)['surgical_toolbox']
    restored=from_case(loaded,saved['mapping'])
    assert restored.identity==base.identity
    assert validate_registry(restored,saved[FIELD])


def _rank_node(target, neural, bone, structure=None, preparation=False):
    from types import SimpleNamespace
    q=lambda v:{'cumulative':v,'status':'source_count'}
    dims={k:q(None) for k in ('toolExchanges','brainStrainSquaredMm3S','brainStressSquaredPa2Mm3S',
        'thermalPeakC','thermalExposureMm3KS','bloodGeneratedMm3','motorEncounterMm3','languageEncounterMm3')}
    dims.update(neuralRemovedMm3=q(neural),neuralCutSourceMm3=q(0),protectedRemovedMm3=q(0),
        structureRemovedMm3={k:q(v) for k,v in (structure or {}).items()})
    return SimpleNamespace(goal_assessment={'originalTargetTreatmentMm3':target},
        ledger={'boneRemovedMm3':bone,'preparationNonboneRemovedMm3':0.,'workNonTargetRemovedMm3':neural,
            'wholeNonTargetRemovedMm3':bone+neural,'modeledElapsedSeconds':1.},sequence_outcome={'dimensions':dims},
        path=({'proposalId':f'{target}-{neural}-{bone}','preparationOnly':preparation},))


def test_beam_preserves_lower_neural_alternative_and_never_buys_neural_with_bone():
    from resectionlab.surgical_joint_cranial_search import _key,_retain_beam,_dominates
    hundred=_rank_node(100.,0.,100.)
    hundred_one=_rank_node(101.,1.,0.)
    more_harm=_rank_node(100.5,2.,0.)
    retained=_retain_beam([hundred_one,more_harm,hundred],2)
    assert any(n is hundred for n in retained) and any(n is hundred_one for n in retained)
    same_target_neural=_rank_node(100.,1.,0.)
    assert _key(hundred)<_key(same_target_neural)
    assert not _dominates(hundred,hundred_one) and not _dominates(hundred_one,hundred)
    a=_rank_node(100.,1.,0.,{'a':1.,'b':0.})
    b=_rank_node(100.,1.,0.,{'a':0.,'b':1.})
    assert not _dominates(a,b) and not _dominates(b,a)


@pytest.fixture(scope='module')
def planned_procedure():
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies,install_cranial_comparison
    case,base,record=procedure_case(0.)
    result=compare_cranial_strategies(base,record,planning_horizon=3)
    install_cranial_comparison(record,result,base)
    return case,base,record


def test_generate_selects_useful_work_and_checked_stop_with_honest_residual(planned_procedure,tmp_path):
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    from resectionlab.surgical_cranial_planning import validate_cranial_report
    _,base,record=planned_procedure;report=record['cranialPlanning']
    selected=next(r for r in report['alternatives'] if r['branchId']==report['selectedBranchId'])
    assert report['decision']=='STOP_WITH_RESIDUAL'
    assert 0<selected['goalAssessment']['originalTargetTreatmentMm3']<27.
    assert selected['ledger']['neuralRemovedMm3']==0
    assert selected['ledger']['boneRemovedMm3']==285. and selected['ledger']['preparationNonboneRemovedMm3']==49.
    assert selected['wholeExitComplete'] and selected['sequenceOutcome']['withdrawalComplete']
    assert any(f['code']=='DECLARED_ANATOMICAL_HARM_CONSTRAINT' for f in report['failures'])
    path=tmp_path/'selected.ressectionlab';save_case(planned_procedure[0],path,artifacts={'surgical_toolbox':record})
    loaded=load_case(path);saved=read_case_artifacts(path)['surgical_toolbox']
    restored=from_case(loaded,saved['mapping'])
    assert restored.identity==base.identity
    assert validate_cranial_report(restored,saved,saved['cranialPlanning'])['verified']


@pytest.mark.parametrize('field',['stopReason','rowStopReason','unknownMotor','elapsed','neural','policy','selection'])
def test_cold_comparison_remeasures_policy_outcomes_and_stop(planned_procedure,field):
    from resectionlab.surgical_cranial_planning import validate_cranial_report
    _,base,original=planned_procedure;record=copy.deepcopy(original);report=record['cranialPlanning']
    row=report['alternatives'][0]
    if field=='stopReason': report['stopReason']='COMPLETE'
    if field=='rowStopReason': row['stopReason']='COMPLETE'
    if field=='unknownMotor': row['sequenceOutcome']['dimensions']['motorEncounterMm3'].update(status='source_count',cumulative=0.,baseline=0.,incremental=0.)
    if field=='elapsed': row['sequenceOutcome']['dimensions']['elapsedSeconds']['cumulative']=0.
    if field=='neural': row['sequenceOutcome']['dimensions']['neuralRemovedMm3']['cumulative']=1.
    if field=='policy': report['procedurePolicy']['policyHash']='changed'
    if field=='selection': report['selectedBranchId']=None
    report['comparisonSeal']=semantic_digest({k:v for k,v in report.items() if k!='comparisonSeal'})
    with pytest.raises(ValueError,match='CRANIAL_PLANNING_REPORT_'):
        validate_cranial_report(base,record,report)


def test_cerebellar_shortcut_is_excluded_from_actual_generate_with_large_neural_allowance():
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies
    _,base,record=procedure_case(1000.,cerebellar_shortcut=True)
    before=copy.deepcopy(record)
    result=compare_cranial_strategies(base,record,planning_horizon=3)
    assert record==before
    selected=next(row for row in result['report']['alternatives'] if
        row['branchId']==result['report']['selectedBranchId'])
    assert selected['goalAssessment']['originalTargetTreatmentMm3']>0,result['report']['failures']
    assert selected['ledger']['structureRemovedMm3']['cerebellum']==0
    assert selected['ledger']['neuralRemovedMm3']==0
    assert any('PROTECTED_STRUCTURE' in row['message'] for row in result['report']['failures']),result['report']['failures']


def test_supported_exposure_tradeoff_survives_canonical_consequence_selection(monkeypatch):
    """Function-level outcome regression, separate from physical demonstrations."""
    from types import SimpleNamespace
    from test_surgical_procedure_policy import source
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab import surgical_consequences as consequences
    from resectionlab.surgical_joint_cranial_search import _compare_checked_nodes
    baseline=SurgicalEngine(source());template=consequences.consequence_vector(baseline,baseline)
    nodes=[];vectors={}
    for tag,target,peak in [('stop',0.,37.),('more-treatment',101.,50.),('less-exposure',100.,37.)]:
        node=SimpleNamespace(exited=SimpleNamespace(scene=baseline.scene,tag=tag),
            goal_assessment={'originalTargetTreatmentMm3':target})
        vectors[tag]=copy.deepcopy(template)
        vectors[tag]['dimensions']['thermalPeakC'].update(status='modeled_unvalidated',cumulative=peak,incremental=peak-37.,reason=None)
        nodes.append(node)
    monkeypatch.setattr(consequences,'consequence_vector',lambda engine,base:copy.deepcopy(vectors[engine.tag]))
    selected,frontier,decision=_compare_checked_nodes(nodes[1:],nodes[0],SimpleNamespace(_source_engine=baseline))
    assert selected is nodes[1] and len(frontier)==2
    assert 'thermalPeakC' in decision['policy']['requiredMetrics']
    vectors['more-treatment']['dimensions']['thermalPeakC'].update(status='unavailable',cumulative=None,incremental=None)
    selected,_,decision=_compare_checked_nodes(nodes[1:],nodes[0],SimpleNamespace(_source_engine=baseline))
    assert selected is nodes[2]
    assert any(v['reason']=='unknown_required_consequence' for v in decision['violations']['candidate-1'])
