"""Generated arrays and sealed geometry only; no patient/model payload reads."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PAYLOAD_OPENS=[]
AUDIT_ACTIVE=False
def forbid(event,args):
    if AUDIT_ACTIVE and event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0])
        if p.endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin')):
            PAYLOAD_OPENS.append(p);raise AssertionError('Patient/model payload forbidden')
sys.addaudithook(forbid)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

from resectionlab import private_vascular_evaluation as v
helpers=load('generated_research_fixture',ROOT/'tests/test_research_estimate_planning.py')
from resectionlab.core import array_digest,semantic_digest
from resectionlab.research_estimate_planning import research_planning_from_estimates


@pytest.fixture(autouse=True)
def scoped_payload_guard():
    """Keep the permanent audit hook inert outside this module's test cases."""
    global AUDIT_ACTIVE
    AUDIT_ACTIVE = True
    try:
        yield
    finally:
        AUDIT_ACTIVE = False


@pytest.fixture
def tmp_path():
    """Exercise the real build-only boundary without requiring pytest flags."""
    (ROOT / 'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vascular-test-', dir=ROOT / 'build') as directory:
        yield Path(directory)


def identity(spec):
    return {'source_domain':'generated_fixture','dataset':'generated-vascular-control','person_id':'generated:unit-fixture',
        'role':'GENERATED_DEVELOPMENT','structural_source_sha256':spec.sources[0].source_file_sha256,
        'structural_frame_sha256':spec.sources[0].frame_sha256}


def fixed_moves(task):
    ids=[]
    for tool,point in [('short-wide-opener',[4.,4.,1.]),('long-narrow-cutter',[4.,4.,5.])]:
        action=next(r['action_id'] for r in task.candidate_inventory()['ledger']
                    if r['feasible'] and r['tool_id']==tool and np.allclose(r['tip_mm'],point))
        ids.append(action);task.advance_planning(action)
    return tuple(ids),{'model_transition_calls':2,'actor_forward_calls':0}


@pytest.fixture
def setup(tmp_path):
    spec=helpers.fixture_spec()
    plan=research_planning_from_estimates(spec,method='SEARCH',configuration_hash=semantic_digest({'rule':'fixed-generated-two-moves'}),planner=fixed_moves)
    public=identity(spec);seal=tmp_path/'strategy.json'
    sha=v.write_strategy_seal(seal,plan=plan,spec=spec,planning_identity=public)
    return spec,plan,public,seal,sha


def reference(setup,*,positive=True,partial=False,angle=0.,noncongruent=False,small_fov=False):
    spec,plan,public,seal,sha=setup
    transform=np.eye(4)
    transform[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    transform[:3,3]=[5.,-3.,2.] if angle else [0.,0.,0.]
    base=np.eye(4);base[:3,3]=[-8.,-8.,-14.]
    shape=(24,24,24)
    if small_fov:
        shape=(9,9,7);base=np.eye(4)
    affine=transform@base
    if noncongruent:affine[:3,3]+=[.25,0,0]
    mask=np.zeros(shape,bool);coverage=np.ones(shape,bool)
    if positive:
        if small_fov:mask[4,4,5]=True
        else:
            mask[12,12,19]=True  # One completely removed planning cell.
            mask[12,12,5]=True   # Proximal shaft only, outside nominal support.
    if partial:coverage[12,12,10]=False
    binding=v.VascularReferenceBinding(sha,semantic_digest(public),public['person_id'],public['role'],
        public['structural_source_sha256'],public['structural_frame_sha256'],'b'*64,'c'*64,
        v._grid(shape,affine),array_digest(mask),array_digest(coverage),tuple(map(tuple,transform)),'d'*64,
        {'source_domain':'generated_fixture','kind':'generated_control','method_record_sha256':'e'*64,
         'initializer_model_sha256':[],'review_status':'generated_not_human_reviewed',
         'coverage_meaning':'annotation_domain_not_vessel_completeness','transform_direction':'planning_RAS_mm_to_reference_RAS_mm'})
    return v.VascularReference(binding,mask,coverage,affine)


def evaluate(setup,ref,out,loader=None,**kwargs):
    spec,plan,public,seal,sha=setup
    return v.evaluate_private_vessels(seal_path=seal,seal_sha256=sha,spec=spec,planning_identity=public,
        reference_binding=ref.binding,load_reference=loader or (lambda:ref),output_directory=out,**kwargs)


def test_complete_seal_before_private_load_and_no_planning_after(setup,tmp_path,monkeypatch):
    ref=reference(setup);opened=[];out=tmp_path/'evaluation'
    for name in ('research_strategy_from_record','_replay_strategy_nominal','independent_check_native_history'):
        old=getattr(v,name)
        def guarded(*args,_old=old,**kwargs):
            assert not opened;return _old(*args,**kwargs)
        monkeypatch.setattr(v,name,guarded)
    def loader():
        assert setup[3].is_file() and (out/'attempt.json').is_file()
        assert len(json.loads(setup[3].read_bytes())['strategy']['physical_history'])==2
        opened.append(True);return ref
    result=evaluate(setup,ref,out,loader)
    assert result['status']=='evaluated_generated_vascular_reference' and opened==[True]
    assert result['microstep_count']>2 and result['action_count']==result['nonstop_sweeps']==2
    assert result['nominal_geometry']['feasible']
    assert json.loads((out/'report.json').read_bytes())==result
    assert result['shaft']['positive_reference_cells']==1
    assert result['tip']['positive_reference_cells']==1
    assert result['whole_tool']['positive_reference_cells']==2
    assert result['removed_overlap']['positive_overlap_cells']==1
    assert result['removed_overlap']['positive_overlap_mm3']==pytest.approx(1.)
    assert result['whole_tool']['biological_vessel_free'] is result['clinical_injury_probability'] is None


def test_rotated_frame_transform_preserves_full_tool_and_removed_counts(setup,tmp_path):
    original=evaluate(setup,reference(setup),tmp_path/'original')
    rotated=evaluate(setup,reference(setup,angle=np.pi/6),tmp_path/'rotated')
    assert rotated['status']==original['status']=='evaluated_generated_vascular_reference'
    for part in ('shaft','tip','whole_tool','removed_overlap'):
        assert rotated[part]==original[part]


def test_partial_coverage_and_no_hit_are_unknown(setup,tmp_path):
    result=evaluate(setup,reference(setup,positive=False,partial=True),tmp_path/'partial')
    assert result['status']=='evaluated_generated_vascular_reference'
    assert result['whole_tool']['positive_reference_cells']==0
    assert result['whole_tool']['unknown_reference_cells']>0
    assert result['whole_tool']['annotated_positive_encounter'] is None


def test_known_hit_remains_positive_with_partial_coverage(setup,tmp_path):
    result=evaluate(setup,reference(setup,partial=True),tmp_path/'partial_hit')
    assert result['whole_tool']['annotated_positive_encounter'] is True
    assert result['whole_tool']['annotation_coverage_complete_for_sweep'] is False


def test_outside_fov_no_hit_remains_unknown(setup,tmp_path):
    result=evaluate(setup,reference(setup,positive=False,small_fov=True),tmp_path/'outside')
    assert result['whole_tool']['outside_reference_fov'] is True
    assert result['whole_tool']['annotated_positive_encounter'] is None


@pytest.mark.parametrize('field,value',[('person_id','generated:another'),('structural_source_sha256','f'*64),
    ('structural_frame_sha256','f'*64),('strategy_file_sha256','f'*64),('planning_identity_hash','f'*64)])
def test_same_person_source_and_seal_mismatch_refuses_before_loader(setup,tmp_path,field,value):
    ref=reference(setup);changed=replace(ref.binding,**{field:value});calls=[]
    spec,plan,public,seal,sha=setup
    with pytest.raises(ValueError,match='reference_person_or_planning_source_mismatch'):
        v.evaluate_private_vessels(seal_path=seal,seal_sha256=sha,spec=spec,planning_identity=public,
            reference_binding=changed,load_reference=lambda:calls.append(True),output_directory=tmp_path/'bad')
    assert calls==[] and not (tmp_path/'bad').exists()


def test_reference_swap_changes_outcomes_only_for_same_durable_strategy(setup,tmp_path):
    before=setup[3].read_bytes();fingerprint=setup[0].fingerprint
    hit=evaluate(setup,reference(setup),tmp_path/'hit')
    empty=evaluate(setup,reference(setup,positive=False),tmp_path/'empty')
    assert hit['strategy_file_sha256']==empty['strategy_file_sha256']==setup[4]
    assert hit['plan_seal_hash']==empty['plan_seal_hash']==setup[1].seal_hash
    assert hit['full_history_hash']==empty['full_history_hash']
    assert hit['whole_tool']['annotated_positive_encounter'] is True
    assert empty['whole_tool']['annotated_positive_encounter'] is False
    assert setup[3].read_bytes()==before and setup[0].fingerprint==fingerprint


@pytest.mark.parametrize('resign',[False,True])
def test_tampered_microsteps_refuse_before_loader_even_with_new_file_digest(setup,tmp_path,resign):
    spec,plan,public,seal,sha=setup;ref=reference(setup);calls=[]
    value=json.loads(seal.read_bytes());value['strategy']['physical_history'][0]['microsteps'][0]['tip_end_mm'][0]+=.125
    if resign:value['content_hash']=semantic_digest({k:v for k,v in value.items() if k!='content_hash'})
    seal.write_bytes(v._json(value));newsha=v.digest(seal.read_bytes()) if resign else sha
    binding=replace(ref.binding,strategy_file_sha256=newsha)
    with pytest.raises(ValueError):
        v.evaluate_private_vessels(seal_path=seal,seal_sha256=newsha,spec=spec,planning_identity=public,
            reference_binding=binding,load_reference=lambda:calls.append(True),output_directory=tmp_path/'tampered')
    assert calls==[]


def test_loaded_source_lineage_mismatch_persists_failure_without_retry(setup,tmp_path):
    ref=reference(setup);lineage=dict(ref.binding.source_lineage);lineage['method_record_sha256']='f'*64
    foreign=v.VascularReference(replace(ref.binding,source_lineage=lineage),ref.mask,ref.coverage,ref.affine_ras_mm)
    calls=[]
    result=evaluate(setup,ref,tmp_path/'mismatch',lambda:(calls.append(True),foreign)[1])
    assert calls==[True] and result['status']=='evaluation_failed' and result['outcomes'] is None
    assert json.loads((tmp_path/'mismatch/report.json').read_bytes())==result


def test_noncongruent_grid_removal_is_explicitly_unsupported(setup,tmp_path):
    result=evaluate(setup,reference(setup,noncongruent=True),tmp_path/'noncongruent')
    assert result['status']=='evaluated_generated_vascular_reference'
    assert result['whole_tool']['positive_reference_cells']>0
    assert result['removed_overlap']['status']=='unsupported_noncongruent_grids'
    assert result['removed_overlap']['positive_overlap_mm3'] is None


def test_roi_local_removed_indices_use_shifted_planning_affine(tmp_path):
    spec=helpers.fixture_spec(roi_start=(2,2,0),roi_stop=(8,8,7))
    plan=research_planning_from_estimates(spec,method='SEARCH',configuration_hash=semantic_digest({'roi':True}),planner=fixed_moves)
    public=identity(spec);seal=tmp_path/'roi-seal.json';sha=v.write_strategy_seal(seal,plan=plan,spec=spec,planning_identity=public)
    setup=(spec,plan,public,seal,sha)
    result=evaluate(setup,reference(setup),tmp_path/'roi')
    assert result['status']=='evaluated_generated_vascular_reference'
    assert result['removed_overlap']['positive_overlap_cells']==1


def test_loader_failure_is_one_saved_negative_no_private_error_text(setup,tmp_path):
    ref=reference(setup);calls=[]
    def failing():calls.append(True);raise OSError('private-values-that-must-not-be-emitted')
    result=evaluate(setup,ref,tmp_path/'failed',failing)
    assert calls==[True] and result['status']=='evaluation_failed' and result['outcomes'] is None
    assert 'private-values' not in (tmp_path/'failed/report.json').read_text()


def test_existing_attempt_never_reopens_reference(setup,tmp_path):
    ref=reference(setup);out=tmp_path/'once';evaluate(setup,ref,out);calls=[]
    with pytest.raises(FileExistsError):evaluate(setup,ref,out,lambda:calls.append(True))
    assert calls==[]


def test_no_patient_payloads_opened():
    assert PAYLOAD_OPENS==[]


def test_path_traversal_cannot_escape_ignored_directory(tmp_path):
    with pytest.raises(ValueError,match='path_traversal_forbidden'):
        v._safe_path(ROOT/'build/../escaped-strategy.json')


def test_oversized_seal_refuses_without_json_decode(tmp_path,monkeypatch):
    path=tmp_path/'oversized.json';path.write_bytes(b' '* (v.MAX_DOCUMENT_BYTES+1))
    monkeypatch.setattr(v.json,'loads',lambda *a,**k:pytest.fail('oversized data must not parse'))
    with pytest.raises(ValueError,match='seal_missing_or_oversized'):
        v._read_bound(path,'a'*64)


def test_fifo_seal_refuses_without_blocking(tmp_path):
    path=tmp_path/'not-a-regular-seal';os.mkfifo(path)
    with pytest.raises(ValueError,match='seal_missing_or_oversized'):
        v._read_bound(path,'a'*64)


def test_person_identity_is_bound_to_existing_planning_declaration(setup,tmp_path):
    spec,plan,public,seal,sha=setup;public={**public,'person_id':'generated:unrelated'}
    with pytest.raises(ValueError,match='person_declaration_mismatch'):
        v.write_strategy_seal(tmp_path/'foreign.json',plan=plan,spec=spec,planning_identity=public)


@pytest.mark.parametrize('kind',['scale','reflection','wrong_direction','human_lineage','learned_initializer'])
def test_unsupported_transform_and_unadmitted_lineage_refuse(setup,kind):
    ref=reference(setup)
    if kind in ('scale','reflection'):
        transform=np.eye(4);transform[0,0]=2. if kind=='scale' else -1.
        with pytest.raises(ValueError,match='unsupported_transform'):
            replace(ref.binding,planning_to_reference_ras_mm=tuple(map(tuple,transform)))
    else:
        lineage=dict(ref.binding.source_lineage)
        if kind=='wrong_direction':lineage['transform_direction']='reference_to_planning'
        elif kind=='human_lineage':lineage['source_domain']='patient'
        else:lineage['initializer_model_sha256']=['e'*64]
        with pytest.raises(ValueError,match='lineage_scope'):
            replace(ref.binding,source_lineage=lineage)


def test_reference_mutation_is_rejected_and_saved(setup,tmp_path):
    ref=reference(setup)
    object.__setattr__(ref,'mask',np.zeros_like(ref.mask))
    result=evaluate(setup,ref,tmp_path/'mutated')
    assert result['status']=='evaluation_failed' and result['outcomes'] is None


def test_stop_only_plan_has_no_tool_or_removed_encounters(tmp_path):
    spec=helpers.fixture_spec();plan=helpers.plan(spec);public=identity(spec);seal=tmp_path/'stop.json'
    sha=v.write_strategy_seal(seal,plan=plan,spec=spec,planning_identity=public)
    setup=(spec,plan,public,seal,sha)
    result=evaluate(setup,reference(setup),tmp_path/'stop')
    assert result['status']=='evaluated_generated_vascular_reference'
    assert result['nonstop_sweeps']==0 and result['microstep_count']==0
    assert result['whole_tool']['touched_reference_cells']==0
    assert result['removed_overlap']['positive_overlap_cells']==0
    assert result['clinical_injury_probability'] is None
