"""Synthetic policy decisions and generated-source runner transaction controls.

Runner tests keep real source verification, preparation, admission and native
checkpoint writes; downstream adapters are stubs, not surgical success evidence.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.workspace_bundle import load_workspace
from test_surgical_exposure import intact, controls, completed
from test_source_exposure_audit import evidence

PATH=Path(__file__).resolve().parents[1]/'scripts/compare_source_preparations.py'
spec=importlib.util.spec_from_file_location('source_preparation_comparison_under_test',PATH)
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def row(identity,nt=1.,prep_seconds=1.,*,access=True,complete=False,suffix_seconds=1.,feasible=True,progress=True):
    suffix={'id':'short','feasible':feasible,'taskProgressMet':progress,'withdrawal':'complete',
            'summary':{'additionalNonTargetMm3':0.,'additionalTargetMm3':1.,'elapsedSeconds':suffix_seconds}}
    return {'id':identity,'preparation':{'attempts':[{'complete':True,'summary':{
        'additionalNonTargetMm3':nt,'additionalTargetMm3':0.,'elapsedSeconds':prep_seconds}}],
        'decision':{'selectedIndex':0,'operation':'PREPARE'}},
        'strategy':{'approaches':{'routes':[{'id':'synthetic-route'}] if access else [],'rejections':[]},
                    'rejections':[],'decision':{'selectedId':'short' if complete else 'stop','reason':None},
                    'alternatives':[suffix]}}


def test_complete_policy_uses_total_cost_and_corridor_policy_keeps_futile_choice():
    rows=[row('no-access',.1,access=False),row('cheap-futile',1.,2.),
          row('fast-prep-slow-suffix',2.,1.,complete=True,suffix_seconds=9.),
          row('slow-prep-fast-suffix',2.,3.,complete=True,suffix_seconds=2.),
          row('more-collateral-fastest',3.,.1,complete=True,suffix_seconds=.1)]
    before=copy.deepcopy(rows); corridor,joint=runner.choose_policies(rows)
    assert corridor['id']=='cheap-futile' and joint['id']=='slow-prep-fast-suffix'
    assert rows==before


def test_policies_break_cost_ties_in_frozen_matrix_order():
    a=row('first',2.,3.,complete=True,suffix_seconds=2.)
    b=row('second',2.,3.,complete=True,suffix_seconds=2.)
    assert tuple(r['id'] for r in runner.choose_policies([a,b]))==('first','first')
    assert tuple(r['id'] for r in runner.choose_policies([b,a]))==('second','second')


@pytest.mark.parametrize('feasible,progress',[(False,True),(True,False),(1,True),(True,1)])
def test_nonstop_label_alone_does_not_qualify_complete_plan(feasible,progress):
    rows=[row('unqualified',complete=True,feasible=feasible,progress=progress)]
    corridor,joint=runner.choose_policies(rows)
    assert corridor is rows[0] and joint is None


def test_all_stop_does_not_become_positive_evidence_or_discard_empty_cells():
    empty={'id':'empty','preparation':{'attempts':[],'decision':{'operation':'STOP','selectedIndex':None}},'strategy':None}
    rows=[empty,row('no-access',access=False),row('checked-but-no-sequence')]
    before=copy.deepcopy(rows); corridor,joint=runner.choose_policies(rows)
    assert corridor['id']=='checked-but-no-sequence' and joint is None
    assert rows==before and len(rows)==3
    assert runner.choose_policies(rows[:2])==(None,None)


@pytest.mark.parametrize('location',['decision','strategy-rejection','geometry-rejection'])
@pytest.mark.parametrize('reason',['COMPUTATION_BUDGET','GEOMETRY_WALL_BUDGET'])
def test_partial_search_budget_is_unknown_in_each_structured_reason_location(location,reason):
    item=row('incomplete'); plan=item['strategy']
    if location=='decision': plan['decision']['reason']=reason
    elif location=='strategy-rejection': plan['rejections']=[{'reason':reason}]
    else: plan['approaches']['rejections']=[{'reason':reason}]
    assert runner.incomplete_search_rows([item])==['incomplete']


def test_budget_words_in_metadata_do_not_fabricate_incomplete_search():
    item=row('BUDGET-label'); item['scope']='COMPUTATION_BUDGET is described, not observed'
    item['strategy']['rejections']=[{'reason':'SECONDARY_TOOL_COLLISION','description':'GEOMETRY_WALL_BUDGET'}]
    assert runner.incomplete_search_rows([item])==[]


@pytest.fixture
def transaction(evidence,monkeypatch,request):
    root=evidence['root']/request.node.name; root.mkdir()
    native=root/'native'; shutil.copytree(evidence['native'].root,native)
    artifacts=copy.deepcopy(evidence['artifacts']); record=artifacts['surgical_toolbox']
    record.update(branchId='source',position=len(evidence['baseline'].actions))
    # Four existing branches deliberately collide with the old count-based names.
    record['branches']['preparation-source-4']=copy.deepcopy(record['branches']['source'])
    record['branches']['prepared-task-4']=copy.deepcopy(record['branches']['prepared'])
    for key in ('preparation-source-4','prepared-task-4'):
        record['parents'][key]=None; record['timeline'][key]=[]
    source=root/'source.ressectionlab'; save_case(evidence['case'],source,artifacts=artifacts)
    route=evidence['proof']['publicPlan']['attempts'][0]['route']
    ray={key:copy.deepcopy(route[key]) for key in ('surfaceMm','sourceTargetMm')}
    design={'schema':'resectionlab.matched-exposure-development-design.v1',
        'source':str(source.relative_to(runner.ROOT)),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rays':[{'id':'generated-ray','sourceRay':ray}],'widthsMm':[0.],
        'controls':{k:v for k,v in evidence['controls'].items() if k!='lateralSweepMm'},
        'task':{'entryRadiusMm':.5,'targetRadiusMm':6.,'targetCenterMm':ray['sourceTargetMm'],
                'requiredEvidence':['brain','target'],'allowUnreviewedEvidence':True},
        'downstream':{'shortSeconds':.25,'longSeconds':1.,'minimumTargetMm3':1.,'contextLabelValue':2,
                      'contextMarginMm':0.,'contextMaxNearMm3':1000.},
        'scope':'Generated runner transaction only; downstream stubs are not surgical evidence.',
        'budget':'One generated cell; no runtime advantage claim.'}
    design_path=root/'design.json'; design_path.write_text(json.dumps(design)); output=root/'result'
    monkeypatch.setattr(sys,'argv',[str(PATH),'--design',str(design_path),'--output',str(output),
        '--native-collection',str(native),'--acknowledge-research'])
    options=[{'layerId':'synthetic-context-adapter','labelValue':2}]
    def defaults(_):
        return {'defaults':{'geometry':{}},'contextOptions':copy.deepcopy(options)}
    monkeypatch.setattr(runner,'defaults',defaults)
    counter={'generate':0,'validate':0}
    def generate(_,raw):
        counter['generate']+=1
        return {'reportSeal':'synthetic-downstream-adapter','publicPlan':row('synthetic')['strategy']}
    monkeypatch.setattr(runner,'generate_comparison',generate)
    def validate(_,comparison):
        counter['validate']+=1
        sealed=json.loads((output/'sealed-policies.json').read_text())
        assert sealed['policySeal']==semantic_digest({k:v for k,v in sealed.items() if k!='policySeal'})
        assert comparison['reportSeal']=='synthetic-downstream-adapter'
    monkeypatch.setattr(runner,'validate_report',validate)
    return {'root':root,'source':source,'original':record,'design':design,'designPath':design_path,'output':output,
            'options':options,'counter':counter,'generate':generate,'native':native}


def test_runner_preserves_colliding_historical_branches_and_seals_before_native_validation(transaction):
    value=transaction; before=value['source'].read_bytes(); runner.main()
    result=json.loads((value['output']/'result.json').read_text())
    assert result['passed'] and result['sourceUnchanged'] and result['designUnchanged'] and result['allDeclaredCellsRetained']
    assert result['designSha256']==hashlib.sha256(value['designPath'].read_bytes()).hexdigest()
    assert value['counter']=={'generate':1,'validate':1} and value['source'].read_bytes()==before
    checkpoint=Path(result['rows'][0]['case']); _,artifacts,_=load_workspace(checkpoint)
    saved=artifacts['surgical_toolbox']
    assert saved['branchId'] not in value['original']['branches']
    for key,snapshot in value['original']['branches'].items():
        assert semantic_digest(saved['branches'][key])==semantic_digest(snapshot)


def test_runner_cannot_attribute_earlier_computation_to_mutated_design(transaction,monkeypatch):
    value=transaction
    def generate(*args,**kwargs):
        result=value['generate'](*args,**kwargs)
        value['designPath'].write_text(value['designPath'].read_text()+'\n')
        return result
    monkeypatch.setattr(runner,'generate_comparison',generate)
    with pytest.raises(ValueError,match='Design bytes changed'):
        runner.main()
    assert not (value['output']/'result.json').exists()


@pytest.mark.parametrize('matches',[0,2])
def test_runner_refuses_missing_or_ambiguous_observed_context(transaction,matches):
    value=transaction
    value['options'][:]=[{'labelValue':2,'layerId':'source-'+str(i)} for i in range(matches)]
    with pytest.raises(ValueError,match='Exactly one eligible supplied context'):
        runner.main()
    assert value['counter']['generate']==0 and not (value['output']/'result.json').exists()


def test_runner_marks_partial_downstream_search_inconclusive(transaction,monkeypatch):
    value=transaction
    def generate(*args,**kwargs):
        result=value['generate'](*args,**kwargs)
        result['publicPlan']['approaches']['rejections']=[{'reason':'GEOMETRY_WALL_BUDGET'}]
        return result
    monkeypatch.setattr(runner,'generate_comparison',generate)
    runner.main(); result=json.loads((value['output']/'result.json').read_text())
    assert result['passed'] and result['allDeclaredCellsRetained']
    assert result['comparisonConclusive'] is False and result['completePlanFeasibilityChanged'] is None
    assert result['incompleteRows']==['generated-ray-w0p0']


def test_unavailable_functional_evidence_refuses_before_any_preparation_even_for_empty_matrix_cells(transaction,monkeypatch):
    item=transaction; item['design']['task']['requiredEvidence']=['brain','target','functional']
    item['design']['controls']['maxNonTargetMm3']=.1
    item['designPath'].write_text(json.dumps(item['design']))
    def forbidden(*args,**kwargs): pytest.fail('Unavailable functional evidence reached preparation')
    monkeypatch.setattr(runner,'prepare',forbidden)
    with pytest.raises(ValueError,match='functional'): runner.main()
    assert not (item['output']/'result.json').exists() and item['counter']['generate']==0


@pytest.mark.parametrize('key,value',[('targetRadiusMm',0.),('entryRadiusMm',16.),('requiredEvidence',[False]),
    ('shortSeconds',0.),('longSeconds',3.),('minimumTargetMm3',9.),('contextLabelValue',True),
    ('contextMarginMm',6.),('contextMaxNearMm3',1001.),('duration-order',None)])
def test_invalid_operating_declarations_refuse_even_when_no_preparation_can_complete(transaction,monkeypatch,key,value):
    item=transaction; design=item['design']
    design['controls']['maxNonTargetMm3']=.1
    if key in design['task']: design['task'][key]=value
    elif key=='duration-order': design['downstream']['shortSeconds']=design['downstream']['longSeconds']
    else: design['downstream'][key]=value
    item['designPath'].write_text(json.dumps(design))
    def forbidden(*args,**kwargs): pytest.fail('Invalid operating design reached preparation')
    monkeypatch.setattr(runner,'prepare',forbidden)
    with pytest.raises(ValueError): runner.main()
    assert not item['output'].exists()
